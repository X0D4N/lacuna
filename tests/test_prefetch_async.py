"""True async prefetch: returns without blocking, wait gates residency,
stats stay consistent under concurrency, and generation ids are identical
with prefetch on/off.

The 1-layer tiny model in test_paged_model.py cannot exercise the
forward-loop integration (there is no next layer), so this file builds
its own 3-layer tiny Qwen2 GGUF with the same GGUF-writing helpers.
"""

import struct
import threading

import numpy as np
import pytest

from lacuna.pager import PagedWeightStore, LacWriter
from lacuna.paged_model import PagedQwen2
from lacuna.quant import Q2, Q4


def _s(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _kv_str(key: str, val: str) -> bytes:
    return _s(key) + struct.pack("<I", 8) + _s(val)


def _kv_u32(key: str, val: int) -> bytes:
    return _s(key) + struct.pack("<I", 4) + struct.pack("<I", val)


def _kv_f32(key: str, val: float) -> bytes:
    return _s(key) + struct.pack("<I", 6) + struct.pack("<f", val)


def _kv_str_array(key: str, vals) -> bytes:
    out = _s(key) + struct.pack("<I", 9) + struct.pack("<I", 8)
    out += struct.pack("<Q", len(vals))
    for v in vals:
        out += _s(v)
    return out


def build_tiny_gguf_3layer(path: str) -> str:
    """3-layer tiny Qwen2 GGUF: hidden 16, 2 Q heads, 1 KV head, F32,
    4-token toy vocab. Same shapes per layer as the 1-layer builder."""
    rng = np.random.RandomState(0)

    def w(shape):
        return (rng.standard_normal(shape) * 0.05).astype("<f4")

    tensors = [
        ("token_embd.weight", [16, 4], w((4, 16))),
        ("output.weight", [16, 4], w((4, 16))),
        ("output_norm.weight", [16], w((16,))),
    ]
    for li in range(3):
        b = f"blk.{li}."
        tensors += [
            (b + "attn_norm.weight", [16], w((16,))),
            (b + "ffn_norm.weight", [16], w((16,))),
            (b + "attn_q.weight", [16, 16], w((16, 16))),
            (b + "attn_q.bias", [16], w((16,))),
            (b + "attn_k.weight", [16, 8], w((8, 16))),
            (b + "attn_k.bias", [8], w((8,))),
            (b + "attn_v.weight", [16, 8], w((8, 16))),
            (b + "attn_v.bias", [8], w((8,))),
            (b + "attn_output.weight", [16, 16], w((16, 16))),
            (b + "ffn_gate.weight", [16, 32], w((32, 16))),
            (b + "ffn_up.weight", [16, 32], w((32, 16))),
            (b + "ffn_down.weight", [32, 16], w((16, 32))),
        ]
    kvs = [
        _kv_str("general.architecture", "qwen2"),
        _kv_u32("qwen2.block_count", 3),
        _kv_u32("qwen2.embedding_length", 16),
        _kv_u32("qwen2.feed_forward_length", 32),
        _kv_u32("qwen2.attention.head_count", 2),
        _kv_u32("qwen2.attention.head_count_kv", 1),
        _kv_f32("qwen2.attention.layer_norm_rms_epsilon", 1e-6),
        _kv_f32("qwen2.rope.freq_base", 10000.0),
        _kv_str("tokenizer.ggml.model", "gpt2"),
        _kv_str_array("tokenizer.ggml.tokens", ["a", "b", "Ġ", "ab"]),
        _kv_str_array("tokenizer.ggml.merges", []),
    ]
    header = b"GGUF" + struct.pack("<I", 3)
    header += struct.pack("<Q", len(tensors)) + struct.pack("<Q", len(kvs))
    header += b"".join(kvs)
    infos = b""
    offset = 0
    blobs = []
    for name, dims, arr in tensors:
        infos += _s(name) + struct.pack("<I", len(dims))
        for d in dims:
            infos += struct.pack("<Q", d)
        infos += struct.pack("<I", 0)  # F32
        infos += struct.pack("<Q", offset)
        blob = arr.tobytes()
        blobs.append(blob)
        offset += len(blob)
    head = header + infos
    pad = (-len(head)) % 32
    with open(path, "wb") as f:
        f.write(head + b"\x00" * pad + b"".join(blobs))
    return path


@pytest.fixture()
def spp_path(tmp_path):
    p = str(tmp_path / "w.lac")
    rng = np.random.default_rng(42)
    with LacWriter(p, page_size=4096) as w:
        w.add_tensor("blk.0.attn.wq", rng.standard_normal((32, 16)).astype(np.float32),
                     layer=0, precision=Q4)
        w.add_tensor("blk.0.ffn.w1", rng.standard_normal((48, 16)).astype(np.float32),
                     layer=0, precision=Q2)
        w.add_tensor("blk.1.ffn.w2", rng.standard_normal((40, 16)).astype(np.float32),
                     layer=1, precision=Q2)
    return p


def test_prefetch_returns_without_blocking(spp_path, monkeypatch):
    """prefetch() must return while a payload load is still blocked."""
    started = threading.Event()
    release = threading.Event()
    orig = PagedWeightStore._load_page

    def blocked(self, idx):
        started.set()
        release.wait(timeout=30)
        return orig(self, idx)

    monkeypatch.setattr(PagedWeightStore, "_load_page", blocked)
    try:
        with PagedWeightStore(spp_path, hot_budget_bytes=10**9) as s:
            result = {}

            def call():
                result["handle"] = s.prefetch([0])

            t = threading.Thread(target=call)
            t.start()
            assert started.wait(timeout=10), "payload load never started"
            # The load is blocked on `release`, which is still unset: if
            # prefetch blocked, this thread would still be alive.
            t.join(timeout=5)
            assert not t.is_alive(), "prefetch blocked on the payload load"
            assert not s.is_resident(0)
            release.set()
            assert s.wait_prefetch(result["handle"]) == 1
            assert s.is_resident(0)
    finally:
        release.set()


def test_prefetch_errors_recorded_not_raised(spp_path, monkeypatch):
    """A failed payload load lands on handle.errors; prefetch never raises,
    and the demand fault stays the loud path for that page."""
    orig = PagedWeightStore._load_page

    def flaky(self, idx):
        if idx == 1:
            raise OSError("simulated read failure")
        return orig(self, idx)

    monkeypatch.setattr(PagedWeightStore, "_load_page", flaky)
    with PagedWeightStore(spp_path, hot_budget_bytes=10**9) as s:
        handle = s.prefetch([0, 1, 2])  # must not raise
        # Wait first: completion callbacks (which record errors) run
        # before waiters wake, so errors are populated by this point.
        assert s.wait_prefetch(handle) == 2
        assert len(handle.errors) == 1
        assert handle.errors[0][0] == 1
        assert isinstance(handle.errors[0][1], OSError)
        assert s.is_resident(0) and s.is_resident(2)
        assert not s.is_resident(1)
        with pytest.raises(OSError):
            s.get_page(1)  # demand fault is loud


def test_stats_consistent_under_concurrent_prefetch_and_reads(spp_path):
    """Concurrent get_page + prefetch: no crash, and hits+faults equals
    exactly the number of get_page calls (locked stats, no lost counts)."""
    errors = []
    n_threads = 8
    calls_per_thread = 25

    def reader(s, barrier):
        try:
            barrier.wait(timeout=10)
            for _ in range(calls_per_thread):
                for i in range(s.n_pages):
                    s.get_page(i)
        except Exception as e:  # noqa: BLE001 — collected, asserted below
            errors.append(e)

    def prefetcher(s, barrier):
        try:
            barrier.wait(timeout=10)
            for _ in range(calls_per_thread):
                h = s.prefetch(list(range(s.n_pages)))
                s.wait_prefetch(h)
        except Exception as e:  # noqa: BLE001 — collected, asserted below
            errors.append(e)

    with PagedWeightStore(spp_path, hot_budget_bytes=4096) as s:
        n_pages = s.n_pages
        barrier = threading.Barrier(n_threads)
        threads = [
            threading.Thread(target=reader if i % 2 == 0 else prefetcher,
                             args=(s, barrier))
            for i in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        assert not any(t.is_alive() for t in threads), "thread hung"
        assert errors == [], f"concurrent errors: {errors[:3]}"
        n_gets = (n_threads // 2) * calls_per_thread * n_pages
        assert s.stats["hits"] + s.stats["faults"] == n_gets
        assert s._cached_bytes <= 4096
        assert s.stats["prefetches"] > 0


def test_ids_identical_prefetch_on_off(tmp_path):
    """Generation ids with prefetch ON equal ids with prefetch OFF on a
    3-layer tiny model (prefetch genuinely fires: layers 1 and 2 exist).
    The ON run must show prefetch submissions; the OFF run none."""
    gguf = build_tiny_gguf_3layer(str(tmp_path / "tiny3.gguf"))
    m_on = PagedQwen2(gguf, hot_budget_bytes=64 * 1024, enable_prefetch=True)
    _, ids_on = m_on.generate("ab a", max_tokens=8)
    n_pref = m_on.store.stats["prefetches"]
    m_on.close()
    m_off = PagedQwen2(gguf, hot_budget_bytes=64 * 1024, enable_prefetch=False)
    _, ids_off = m_off.generate("ab a", max_tokens=8)
    m_off.close()
    assert n_pref > 0, "prefetch never fired in the ON run"
    assert m_off.store.stats["prefetches"] == 0
    assert ids_on == ids_off


def test_prefetch_corrupt_header_raises_synchronously(spp_path):
    """Fail-fast contract (mirrors the fault-injection test's method, but
    this file owns its own fixture): mutating a page-0 header precision
    byte makes prefetch raise ValueError synchronously, not via a handle."""
    with PagedWeightStore(spp_path, hot_budget_bytes=10**9) as s:
        off = s.page_table()[0]["offset"]
    with open(spp_path, "r+b") as f:
        f.seek(off)
        f.write(bytes([Q2]))  # precision mismatch vs the table
    with PagedWeightStore(spp_path, hot_budget_bytes=10**9) as s:
        with pytest.raises(ValueError, match="precision mismatch"):
            s.prefetch([0])
