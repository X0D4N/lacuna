"""Converter test: tiny fake GGUF (F32) -> .lac -> pages readable."""

import struct

import numpy as np

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

    stats = convert(gguf, spp, page_mb=1, quiet=True)
    assert stats["tensors"] == 2
    assert stats["tiers"]["q4"] == 1  # attn -> hot tier
    assert stats["tiers"]["q2"] == 1  # ffn  -> warm tier

    with open(spp, "rb") as f:
        assert f.read(4) == MAGIC
    with PagedWeightStore(spp, hot_budget_bytes=10**9) as s:
        assert s.names == ["blk.0.attn.wq", "blk.0.ffn.w1"]
        assert s.n_pages >= 2
        p0 = s.get_page(0)
        assert p0.shape[1] == 4 and np.all(np.isfinite(p0))
