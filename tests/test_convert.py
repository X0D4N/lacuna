"""Converter test: tiny fake GGUF (F32) -> .lac -> pages readable."""

import hashlib
import os
import struct
from unittest import mock

import numpy as np
import pytest

from lacuna.convert import convert, parse_gguf_header
from lacuna.pager import MAGIC, PagedWeightStore


def _write_fake_gguf(path):
    # minimal GGUF v3: header, 0 KV, 2 F32 tensors, 32-aligned data
    tensors = [
        ("blk.0.attn.wq", (8, 4), np.float32),
        ("blk.0.ffn.w1", (6, 4), np.float32),
    ]
    rng = np.random.default_rng(3)
    blobs = [(n, rng.standard_normal(s).astype(np.float32)) for n, s, _ in tensors]
    with open(path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))   # version
        f.write(struct.pack("<Q", 2))    # n_tensors
        f.write(struct.pack("<Q", 0))    # n_kv
        infos = []
        off = 0
        for name, arr in blobs:
            b = name.encode()
            f.write(struct.pack("<Q", len(b)) + b)
            f.write(struct.pack("<I", 2))
            f.write(struct.pack("<Q", arr.shape[1]))
            f.write(struct.pack("<Q", arr.shape[0]))
            f.write(struct.pack("<I", 0))  # F32
            f.write(struct.pack("<Q", off))
            infos.append(arr)
            off += arr.nbytes
        pos = f.tell()
        f.write(b"\x00" * ((-pos) % 32))
        for arr in infos:
            f.write(arr.tobytes())


def test_convert_roundtrip(tmp_path):
    gguf = str(tmp_path / "tiny.gguf")
    spp = str(tmp_path / "tiny.lac")
    _write_fake_gguf(gguf)
    infos, data_start = parse_gguf_header(gguf)
    assert [i["name"] for i in infos] == ["blk.0.attn.wq", "blk.0.ffn.w1"]

    stats = convert(gguf, spp, page_mb=1, quiet=True, precision_policy="extreme")
    assert stats["tensors"] == 2
    assert stats["precision_policy"] == "extreme"
    assert stats["tiers"]["q4"] == 1  # attn -> hot tier
    assert stats["tiers"]["q2"] == 1  # ffn  -> warm tier

    with open(spp, "rb") as f:
        assert f.read(4) == MAGIC
    with PagedWeightStore(spp, hot_budget_bytes=10**9) as s:
        assert s.precision_policy == "extreme"
        assert s.names == ["blk.0.attn.wq", "blk.0.ffn.w1"]
        assert s.n_pages >= 2
        p0 = s.get_page(0)
        assert p0.shape[1] == 4 and np.all(np.isfinite(p0))


def test_convert_exact_default_is_bit_identical(tmp_path):
    """Default policy: every tensor carried as RAW bytes, bit-identical."""
    gguf = str(tmp_path / "tiny.gguf")
    spp = str(tmp_path / "tiny.lac")
    _write_fake_gguf(gguf)
    infos, _ = parse_gguf_header(gguf)

    stats = convert(gguf, spp, page_mb=1, quiet=True)  # default policy
    assert stats["precision_policy"] == "exact"
    assert stats["tiers"]["raw"] == 2

    with PagedWeightStore(spp, hot_budget_bytes=10**9) as s:
        assert s.precision_policy == "exact"
        # tensor 0's pages concatenated == the exact source bytes
        pages = [i for i, e in enumerate(s._entries) if e["tensor_id"] == 0]
        blob = b"".join(s.get_page(i).tobytes() for i in pages)
    # rebuild the source bytes for tensor 0 from the fake GGUF
    rng = np.random.default_rng(3)
    src = rng.standard_normal((8, 4)).astype(np.float32)
    assert blob == src.tobytes()


def test_convert_survives_short_preads(tmp_path):
    """Regression for the >2GiB truncation: a single os.pread syscall is
    capped by the OS (Linux MAX_RW_COUNT = 0x7ffff000 bytes) and returns
    short WITHOUT error. Simulate a miniature cap (37 bytes per call):
    convert must loop and still produce bit-identical output on both the
    RAW (exact) and dense (extreme) read paths."""
    gguf = str(tmp_path / "tiny.gguf")
    spp = str(tmp_path / "tiny.lac")
    _write_fake_gguf(gguf)
    real_pread = os.pread

    def capped_pread(fd, n, offset):
        return real_pread(fd, min(n, 37), offset)

    for policy in ("exact", "extreme"):
        out = spp + "." + policy
        with mock.patch.object(os, "pread", capped_pread):
            stats = convert(gguf, out, page_mb=1, quiet=True,
                            precision_policy=policy)
        assert stats["tensors"] == 2
        if policy == "exact":
            assert stats["bytes_in"] == (8 * 4 + 6 * 4) * 4
            with PagedWeightStore(out, hot_budget_bytes=10**9) as s:
                pages = [i for i, e in enumerate(s._entries)
                         if e["tensor_id"] == 0]
                blob = b"".join(s.get_page(i).tobytes() for i in pages)
            rng = np.random.default_rng(3)
            src = rng.standard_normal((8, 4)).astype(np.float32)
            assert blob == src.tobytes()


def _write_gguf_declared(path, dims, data_bytes):
    """GGUF whose header DECLARES an F32 tensor of `dims` but whose data
    section physically holds only `data_bytes` (a truncated source)."""
    with open(path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))
        f.write(struct.pack("<Q", 1))    # n_tensors
        f.write(struct.pack("<Q", 0))    # n_kv
        name = b"blk.0.big"
        f.write(struct.pack("<Q", len(name)) + name)
        f.write(struct.pack("<I", len(dims)))
        for d in dims:
            f.write(struct.pack("<Q", d))
        f.write(struct.pack("<I", 0))    # F32
        f.write(struct.pack("<Q", 0))    # offset
        pos = f.tell()
        f.write(b"\x00" * ((-pos) % 32))
        f.write(b"\x00" * data_bytes)


def test_convert_raises_on_truncated_source(tmp_path):
    """Fail-loud guard: when the source file ends before the declared
    tensor size, convert must raise — never write a short tensor."""
    gguf = str(tmp_path / "trunc.gguf")
    spp = str(tmp_path / "trunc.lac")
    # declares 4096x4096 F32 = 64 MiB, physically holds 1 MiB
    _write_gguf_declared(gguf, (4096, 4096), 1 << 20)
    with pytest.raises(OSError):
        convert(gguf, spp, quiet=True)


def _write_sparse_big_gguf(path, nbytes):
    """One opaque (Q4_0-typed) tensor of `nbytes`, data section sparse
    (reads back as zeros) with sentinel patterns at the start, straddling
    the old 0x7ffff000 truncation point, and at the tail."""
    name = b"token_embd.weight"
    cap = 2_147_479_552  # 0x7ffff000, the old silent-truncation length
    with open(path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<Q", 0))
        f.write(struct.pack("<Q", len(name)) + name)
        f.write(struct.pack("<I", 2))
        f.write(struct.pack("<Q", nbytes))
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<I", 2))    # Q4_0 -> opaque sizing path
        f.write(struct.pack("<Q", 0))
        pos = f.tell()
        data_start = pos + ((-pos) % 32)
        f.write(b"\x00" * (data_start - pos))
        f.truncate(data_start + nbytes)
    fd = os.open(path, os.O_RDWR)
    try:
        os.pwrite(fd, b"A" * 64, data_start)
        os.pwrite(fd, b"B" * 64, data_start + cap - 32)
        os.pwrite(fd, b"C" * 64, data_start + nbytes - 64)
    finally:
        os.close(fd)
    return data_start


@pytest.mark.skipif(
    os.environ.get("LACUNA_BIG_TENSOR_TEST") != "1",
    reason="writes a >2GiB .lac; set LACUNA_BIG_TENSOR_TEST=1 (and "
           "LACUNA_BIG_TENSOR_DIR to a dir with ~5GB free) to run",
)
def test_convert_tensor_over_2gib_bit_exact(tmp_path):
    """>2GiB single-tensor round-trip: the exact failure the Kaggle
    ladder measured on Command R+ 104B's token_embd.weight
    (2,580,480,000 source bytes landed as 2,147,479,552)."""
    workdir = os.environ.get("LACUNA_BIG_TENSOR_DIR") or str(tmp_path)
    os.makedirs(workdir, exist_ok=True)
    gguf = os.path.join(workdir, "big.gguf")
    spp = os.path.join(workdir, "big.lac")
    nbytes = (1 << 31) + 1_000_000
    try:
        data_start = _write_sparse_big_gguf(gguf, nbytes)
        stats = convert(gguf, spp, quiet=True)  # exact policy
        assert stats["bytes_in"] == nbytes

        fd = os.open(gguf, os.O_RDONLY)
        hs = hashlib.sha256()
        off, left = data_start, nbytes
        while left > 0:
            step = min(left, 1 << 23)
            hs.update(os.pread(fd, step, off))
            off += step
            left -= step
        os.close(fd)

        with PagedWeightStore(spp, hot_budget_bytes=1 << 30) as s:
            tid = s.tensor_id("token_embd.weight")
            pages = sorted(s.tensor_pages(tid), key=lambda d: d["offset"])
            hl = hashlib.sha256()
            plen = 0
            for d in pages:
                arr = s.get_page(d["page_id"])
                hl.update(arr.tobytes())
                plen += arr.nbytes
        assert plen == nbytes
        assert hl.digest() == hs.digest()
    finally:
        for p in (gguf, spp):
            if os.path.exists(p):
                os.unlink(p)
