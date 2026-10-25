"""Paged executor tests on a tiny synthetic Qwen2 GGUF (no model file).

Builds a minimal valid GGUF in a tmp dir: 1 layer, hidden 16, 2 Q heads,
1 KV head, F32 tensors, 4-token toy vocab. This exercises the real
paged forward pass (parse -> pager -> dequant -> transformer -> greedy)
without needing a downloaded model or llama.cpp.
"""

import os
import struct

import numpy as np

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
