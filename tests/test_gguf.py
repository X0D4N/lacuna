"""GGUF parser + dequantizer tests: hand-computed blocks, no model file.

Every expected vector below is computed by hand from the ggml block
formats (llama.cpp ggml/src/ggml-quants.c), not from the implementation
under test — a mapping error in the implementation fails these tests.
The dequantizers are additionally verified bit-exact against ggml's own
exported functions on real model bytes (manual check, 2026-10-25, not part
of this hermetic suite since it needs libggml + a model file).
"""

import struct

import numpy as np
import pytest

from lacuna.gguf import (
    dequant_f16,
    dequant_f32,
    dequant_q4_k,
    dequant_q5_0,
    dequant_q6_k,
    dequant_q8_0,
    parse_gguf,
    tensor_nbytes,
    tensor_shape,
)


def _f16_bytes(v: float) -> bytes:
    return struct.pack("<e", v)


# ------------------------------------------------------------- q5_0 ---


def test_q5_0_all_minus_16():
    # d=1, qh=0, qs=0 -> every code is 0, value = (0-16)*1 = -16.
    blk = _f16_bytes(1.0) + b"\x00" * 4 + b"\x00" * 16
    out = dequant_q5_0(blk, 32)
    assert out.shape == (32,)
    assert np.all(out == -16.0)


def test_q5_0_high_bit_sets_zero():
    # d=1, qh all ones, qs=0 -> low part 0|0x10=16 -> 0; high part 0|0x10=16 -> 0.
    blk = _f16_bytes(1.0) + b"\xff" * 4 + b"\x00" * 16
    out = dequant_q5_0(blk, 32)
    assert np.all(out == 0.0)


def test_q5_0_mixed_nibbles():
    # qs[0] = 0x21: low nibble 1 -> 1-16 = -15 at y[0]; high nibble 2 -> -14 at y[16].
    blk = _f16_bytes(2.0) + b"\x00" * 4 + bytes([0x21] + [0] * 15)
    out = dequant_q5_0(blk, 32)
    assert out[0] == -30.0  # (1-16)*2
    assert out[16] == -28.0  # (2-16)*2
    assert out[1] == -32.0


# ------------------------------------------------------------- q8_0 ---


def test_q8_0_values():
    blk = _f16_bytes(0.5) + bytes([1, 255, 2, 0] + [0] * 28)
    out = dequant_q8_0(blk, 32)
    assert out[0] == 0.5
    assert out[1] == -0.5  # int8 255 == -1
    assert out[2] == 1.0


# ------------------------------------------------------------- q4_k ---


def test_q4_k_first_group():
    # d=1, dmin=1, scales[0]=1 (pair0 scale), scales[4]=0 (pair0 min),
    # all other scale bytes 0. qs[0:32]=0x11 -> low nibbles 1 -> 1*1-0 = 1
    # at y[0:32]; high nibbles 1 with pair1 (scale 0, min 0) -> 0 at y[32:64].
    scales = bytes([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    qs = bytes([0x11] * 32 + [0] * 96)
    blk = _f16_bytes(1.0) + _f16_bytes(1.0) + scales + qs
    out = dequant_q4_k(blk, 256)
    assert out.shape == (256,)
    assert np.all(out[0:32] == 1.0)
    assert np.all(out[32:] == 0.0)


def test_q4_k_min_subtraction():
    # pair0: scale 2 (scales[0]=2), min 3 (scales[4]=3); d=1, dmin=2.
    # qs low nibble 5 -> 1*2*5 - 2*3 = 4.
    scales = bytes([2, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0])
    qs = bytes([0x55] * 32 + [0] * 96)
    blk = _f16_bytes(1.0) + _f16_bytes(2.0) + scales + qs
    out = dequant_q4_k(blk, 256)
    assert np.all(out[0:32] == 4.0)


# ------------------------------------------------------------- q6_k ---


def test_q6_k_layout_d_last():
    # Layout is ql[128] | qh[64] | scales[16] | d[2] (d LAST).
    # scales all 1, ql/qh all 0 -> every code (0|0)-32 = -32, value -32*1*1.
    blk = b"\x00" * 128 + b"\x00" * 64 + b"\x01" * 16 + _f16_bytes(1.0)
    out = dequant_q6_k(blk, 256)
    assert out.shape == (256,)
    assert np.all(out == -32.0)


def test_q6_k_full_codes():
    # ql=0xFF, qh=0xFF -> 4-bit low 15 | 2-bit high 3<<4 = 63 -> 63-32 = 31.
    blk = b"\xff" * 128 + b"\xff" * 64 + b"\x01" * 16 + _f16_bytes(1.0)
    out = dequant_q6_k(blk, 256)
    assert np.all(out == 31.0)


# ------------------------------------------------------------- f32/f16 ---


def test_f32_f16_roundtrip_values():
    vals = np.array([1.5, -2.25, 0.0, 3.75], dtype=np.float32)
    assert np.array_equal(dequant_f32(vals.astype("<f4").tobytes(), 4), vals)
    assert np.allclose(
        dequant_f16(vals.astype("<f2").tobytes(), 4), vals, atol=1e-3
    )


# ------------------------------------------------------------- parser ---


def _build_minimal_gguf(tmp_path):
    """A 1-tensor F32 GGUF file: 1 KV entry, tensor 'w' dims [2, 3]."""
    import os

    path = os.path.join(tmp_path, "mini.gguf")
    data = b""
    data += b"GGUF" + struct.pack("<I", 3)
    data += struct.pack("<Q", 1)  # 1 tensor
    data += struct.pack("<Q", 1)  # 1 kv
    key = b"general.architecture"
    data += struct.pack("<Q", len(key)) + key
    data += struct.pack("<I", 8)  # string
    val = b"qwen2"
    data += struct.pack("<Q", len(val)) + val
    name = b"w"
    data += struct.pack("<Q", len(name)) + name
    data += struct.pack("<I", 2)
    data += struct.pack("<Q", 2) + struct.pack("<Q", 3)  # dims [2, 3]
    data += struct.pack("<I", 0)  # F32
    data += struct.pack("<Q", 0)  # offset 0
    pad = (-len(data)) % 32
    data += b"\x00" * pad
    assert len(data) % 32 == 0
    data += np.arange(6, dtype="<f4").tobytes()
    with open(path, "wb") as f:
        f.write(data)
    return path


def test_parse_minimal_gguf(tmp_path):
    path = _build_minimal_gguf(tmp_path)
    md, infos, ds = parse_gguf(path)
    assert md["general.architecture"] == "qwen2"
    assert len(infos) == 1
    assert infos[0]["name"] == "w"
    assert infos[0]["dims"] == [2, 3]
    assert infos[0]["type"] == 0
    assert tensor_nbytes(infos[0]) == 24
    assert tensor_shape(infos[0]["dims"]) == (3, 2)


def test_unsupported_type_raises(tmp_path):
    path = _build_minimal_gguf(tmp_path)
    md, infos, ds = parse_gguf(path)
    infos[0]["type"] = 99
    with pytest.raises(ValueError):
        tensor_nbytes(infos[0])
