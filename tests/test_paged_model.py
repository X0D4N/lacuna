"""Paged executor tests on a tiny synthetic Qwen2 GGUF (no model file).

Builds a minimal valid GGUF in a tmp dir: 1 layer, hidden 16, 2 Q heads,
1 KV head, F32 tensors, 4-token toy vocab. This exercises the real
paged forward pass (parse -> pager -> dequant -> transformer -> greedy)
without needing a downloaded model or llama.cpp.
"""

import os
import struct

import numpy as np
import pytest

from lacuna.paged_model import PagedQwen2


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


def build_tiny_gguf(path: str) -> str:
    rng = np.random.RandomState(0)

    def w(shape):
        return (rng.standard_normal(shape) * 0.05).astype("<f4")

    # (name, dims [ne0, ne1], logical array) — dims are fastest-first.
    tensors = [
        ("token_embd.weight", [16, 4], w((4, 16))),
        ("output.weight", [16, 4], w((4, 16))),
        ("output_norm.weight", [16], w((16,))),
        ("blk.0.attn_norm.weight", [16], w((16,))),
        ("blk.0.ffn_norm.weight", [16], w((16,))),
        ("blk.0.attn_q.weight", [16, 16], w((16, 16))),
        ("blk.0.attn_q.bias", [16], w((16,))),
        ("blk.0.attn_k.weight", [16, 8], w((8, 16))),
        ("blk.0.attn_k.bias", [8], w((8,))),
        ("blk.0.attn_v.weight", [16, 8], w((8, 16))),
        ("blk.0.attn_v.bias", [8], w((8,))),
        ("blk.0.attn_output.weight", [16, 16], w((16, 16))),
        ("blk.0.ffn_gate.weight", [16, 32], w((32, 16))),
        ("blk.0.ffn_up.weight", [16, 32], w((32, 16))),
        ("blk.0.ffn_down.weight", [32, 16], w((16, 32))),
    ]
    kvs = [
        _kv_str("general.architecture", "qwen2"),
        _kv_u32("qwen2.block_count", 1),
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


def test_tiny_forward_finite(tmp_path):
    gguf = build_tiny_gguf(os.path.join(tmp_path, "tiny.gguf"))
    m = PagedQwen2(gguf, hot_budget_bytes=1024 * 1024)
    logits = m.forward([0, 1, 2], 0)
    assert logits.shape == (4,)
    assert np.isfinite(logits).all()
    assert m.store.stats["faults"] > 0  # weights really came via the pager
    m.close()


def test_tiny_generate_deterministic(tmp_path):
    gguf = build_tiny_gguf(os.path.join(tmp_path, "tiny.gguf"))
    m = PagedQwen2(gguf, hot_budget_bytes=1024 * 1024)
    text1, ids1 = m.generate("ab a", max_tokens=4)
    text2, ids2 = m.generate("ab a", max_tokens=4)
    assert ids1 == ids2 and len(ids1) == 4  # no EOS in metadata: full run
    assert text1 == text2 and text1
    m.close()


def test_engine_uses_paged_for_supported_arch(tmp_path, monkeypatch):
    import lacuna.engine as engine_mod

    monkeypatch.setattr(engine_mod, "_find_llama_server", lambda *a: None)
    gguf = build_tiny_gguf(os.path.join(tmp_path, "tiny.gguf"))
    eng = engine_mod.Engine(gguf)
    try:
        assert eng.mode == "lacuna"
        assert eng.paged_error is None
        text = eng.generate("ab a", max_tokens=4)
        assert text and not text.startswith("[simulated]")
    finally:
        eng.close()


def test_engine_uses_paged_for_lac_path(tmp_path, monkeypatch):
    """Regression: Engine(<model>.lac) must find the sibling GGUF.

    Two .lac naming conventions exist: pull/convert use splitext
    (tiny.gguf -> tiny.lac) while PagedQwen2's auto-convert appends
    (tiny.gguf -> tiny.gguf.lac). Engine used to strip ".lac" and use
    the bare stem as the sibling, so for pull-style .lac files
    _try_paged opened a nonexistent path, failed with
    FileNotFoundError, and Engine silently degraded to the
    [simulated] mock path while mode still read "lacuna" — the CLI
    smoke on a pulled model printed mock gibberish instead of the
    model's own output.
    """
    import lacuna.engine as engine_mod
    from lacuna.convert import convert as _convert

    monkeypatch.setattr(engine_mod, "_find_llama_server", lambda *a: None)
    gguf = build_tiny_gguf(os.path.join(tmp_path, "tiny.gguf"))
    # Pull-style: tiny.lac next to tiny.gguf.
    lac_pull = os.path.join(tmp_path, "tiny.lac")
    _convert(gguf, lac_pull, quiet=True)
    eng = engine_mod.Engine(lac_pull)
    try:
        assert eng._paged is not None
        assert eng.paged_error is None
        assert eng.mode == "lacuna"
        text = eng.generate("ab a", max_tokens=4)
        assert text and not text.startswith("[simulated]")
    finally:
        eng.close()
    # Append-style: tiny.gguf.lac next to tiny.gguf.
    lac_append = gguf + ".lac"
    _convert(gguf, lac_append, quiet=True)
    eng = engine_mod.Engine(lac_append)
    try:
        assert eng._paged is not None
        assert eng.paged_error is None
        assert eng.mode == "lacuna"
        text = eng.generate("ab a", max_tokens=4)
        assert text and not text.startswith("[simulated]")
    finally:
        eng.close()


def test_head_blocks_and_embed_rows_cached(tmp_path):
    # Sliced head/embed must go through the bounded _dq cache (the
    # pre-fix regression was re-dequantizing the head every token):
    # after a forward, the head block and the used embed rows are in
    # _dq, but no whole embed/head matrix is ever held.
    gguf = build_tiny_gguf(os.path.join(tmp_path, "tiny.gguf"))
    m = PagedQwen2(gguf, hot_budget_bytes=1024 * 1024)
    m.forward([0, 1, 2], 0)
    assert any(k.startswith("output.weight#rows:") for k in m._dq)
    assert "token_embd.weight#row:0" in m._dq
    assert "token_embd.weight#row:1" in m._dq
    assert not hasattr(m, "embed") and not hasattr(m, "lm_head")
    m.close()


def test_read_tensor_rows_matches_slice(tmp_path):
    from lacuna.gguf import GGUFError, read_tensor, read_tensor_rows

    gguf = build_tiny_gguf(os.path.join(tmp_path, "tiny.gguf"))
    m = PagedQwen2(gguf, hot_budget_bytes=1024 * 1024)
    try:
        for name in ("token_embd.weight", "output.weight"):
            info = m.by_name[name]
            full = read_tensor(m.store, info)
            assert np.array_equal(read_tensor_rows(m.store, info, 0, 4), full)
            assert np.array_equal(read_tensor_rows(m.store, info, 1, 3), full[1:3])
            assert np.array_equal(read_tensor_rows(m.store, info, 2, 3), full[2:3])
        # 1-D tensors (norms) cannot be row-sliced.
        with pytest.raises(GGUFError):
            read_tensor_rows(m.store, m.by_name["output_norm.weight"], 0, 1)
        # Out-of-range rows fail loud.
        with pytest.raises(GGUFError):
            read_tensor_rows(m.store, m.by_name["output.weight"], 0, 5)
    finally:
        m.close()


def test_no_held_full_embed_or_head(tmp_path):
    gguf = build_tiny_gguf(os.path.join(tmp_path, "tiny.gguf"))
    m = PagedQwen2(gguf, hot_budget_bytes=1024 * 1024)
    try:
        assert not hasattr(m, "embed") and not hasattr(m, "lm_head")
        assert m.out_norm.shape == (16,)  # only the tiny final norm is held
        text, ids = m.generate("ab a", max_tokens=4)
        assert len(ids) == 4 and text
    finally:
        m.close()
