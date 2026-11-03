"""GGUF parser + dequantizer tests: hand-computed blocks, no model file.

Every expected vector below is computed by hand from the ggml block
formats (llama.cpp ggml/src/ggml-quants.c), not from the implementation
under test — a mapping error in the implementation fails these tests.
The dequantizers are additionally verified bit-exact against ggml's own
exported functions: hand-computed vectors below, plus a ctypes test
against libggml-base.so on synthetic random blocks (skipped when the
library is absent, so the suite stays hermetic elsewhere).
"""

import ctypes
import os
import struct

import numpy as np
import pytest

from lacuna.gguf import (
    dequant_bf16,
    dequant_f16,
    dequant_f32,
    dequant_q2_k,
    dequant_q3_k,
    dequant_q4_k,
    dequant_q5_0,
    dequant_q5_k,
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


# ------------------------------------------------------------- q2_k ---


def test_q2_k_first_and_second_half():
    # d=1, dmin=0.5. scales[0]=0x11 -> dl=1, ml=0.5 (group 0);
    # scales[2]=0x11 -> same for group 2 (shift 2); scales[8]=0x21 ->
    # dl=1, ml=1.0 (second half, group 0).
    # qs[0]=0x05: bits 0-1 = 1 -> y[0] = 1*1-0.5 = 0.5; bits 2-3 = 1
    # -> y[32] = 0.5. qs[1]=0 -> y[1] = 1*0-0.5 = -0.5.
    # qs[32]=0x02 -> y[128] = 1*2-1.0 = 1.0.
    scales = bytearray(16)
    scales[0] = 0x11
    scales[2] = 0x11
    scales[8] = 0x21
    qs = bytearray(64)
    qs[0] = 0x05
    qs[32] = 0x02
    blk = bytes(scales) + bytes(qs) + _f16_bytes(1.0) + _f16_bytes(0.5)
    out = dequant_q2_k(blk, 256)
    assert out.shape == (256,)
    assert out[0] == 0.5
    assert out[1] == -0.5
    assert out[32] == 0.5
    assert out[128] == 1.0


# ------------------------------------------------------------- q3_k ---


def _q3_k_block(hmask, qs, scales_raw, d=1.0):
    return bytes(hmask) + bytes(qs) + bytes(scales_raw) + _f16_bytes(d)


def test_q3_k_scale_unpack_and_high_bit():
    # scales raw: raw[0]=0x01, raw[8]=0x02 -> unpacked scale[0] =
    # 1 | (2<<4) = 33 -> dl = 1*(33-32) = 1. scale[1] = 0 -> dl = -32.
    # qs[0]=1, hm[0]=1 -> code 1-0 = 1 -> y[0] = 1.
    # qs[16]=0, hm[16]=0 -> code 0-4 = -4 -> y[16] = -32 * -4 = 128.
    scales_raw = bytearray(12)
    scales_raw[0] = 0x01
    scales_raw[8] = 0x02
    qs = bytearray(64)
    qs[0] = 1
    hm = bytearray(32)
    hm[0] = 1
    out = dequant_q3_k(_q3_k_block(hm, qs, scales_raw), 256)
    assert out.shape == (256,)
    assert out[0] == 1.0
    assert out[16] == 128.0


def test_q3_k_high_bit_clear_and_second_half():
    # Same scales as above (dl=1 for group 0). qs[0]=1, hm[0]=0 ->
    # code 1-4 = -3 -> y[0] = -3.
    # Second half: qs[32]=1, hm[0]=0x10 (bit 4) -> code 1-0 = 1;
    # scale[8] = 0 -> dl = -32 -> y[128] = -32.
    scales_raw = bytearray(12)
    scales_raw[0] = 0x01
    scales_raw[8] = 0x02
    qs = bytearray(64)
    qs[0] = 1
    qs[32] = 1
    hm = bytearray(32)
    hm[0] = 0x10
    out = dequant_q3_k(_q3_k_block(hm, qs, scales_raw), 256)
    assert out[0] == -3.0
    assert out[128] == -32.0


# ------------------------------------------------------------- q5_k ---


def _q5_k_block(scales, qh, qs, d=1.0, dmin=1.0):
    return _f16_bytes(d) + _f16_bytes(dmin) + bytes(scales) + bytes(qh) + bytes(qs)


def test_q5_k_low_nibble_and_high_bit():
    # pair0: scale 1 (scales[0]=1), min 0 (scales[4]=0) -> d1=1, m1=0.
    # qs[0]=0x11 -> low nibble 1; qh[0]=0 -> code 1 -> y[0] = 1.
    scales = bytearray(12)
    scales[0] = 1
    qs = bytearray(128)
    qs[0] = 0x11
    out = dequant_q5_k(_q5_k_block(scales, bytearray(32), qs), 256)
    assert out.shape == (256,)
    assert out[0] == 1.0
    # Same block with qh[0]=1 (high bit for group 0, first 32):
    # code 1+16 = 17 -> y[0] = 17.
    qh = bytearray(32)
    qh[0] = 1
    out = dequant_q5_k(_q5_k_block(scales, qh, qs), 256)
    assert out[0] == 17.0


def test_q5_k_second_group_and_high_nibble():
    # Group 1 pair (pair2): scale 2 (scales[2]=2), min 1 (scales[6]=1)
    # -> d1=2, m1=1. qs[32]=0x21 low nibble 1; qh[0] bit 2 (0x04) ->
    # code 1+16 = 17 -> y[64] = 2*17-1 = 33.
    # Group 0 pair1: scale 1 (scales[1]=1), min 0 (scales[5]=0);
    # qs[0]=0x21 high nibble 2; qh[0] bit 1 (0x02) -> code 2+16 = 18
    # -> y[32] = 18.
    scales = bytearray(12)
    scales[1] = 1
    scales[2] = 2
    scales[6] = 1
    qs = bytearray(128)
    qs[0] = 0x21
    qs[32] = 0x21
    qh = bytearray(32)
    qh[0] = 0x06
    out = dequant_q5_k(_q5_k_block(scales, qh, qs), 256)
    assert out[32] == 18.0
    assert out[64] == 33.0


# ------------------------------------------- ctypes bit-exact vs ggml ---


_LIBGGML = os.environ.get(
    "LACUNA_LIBGGML",
    os.path.expanduser("~/workspace/tools/llama.cpp/build/bin/libggml-base.so"),
)


@pytest.mark.parametrize(
    "fn_name, py_fn, block_bytes, d_offs",
    [
        ("dequantize_row_q2_K", dequant_q2_k, 84, (80, 82)),
        ("dequantize_row_q3_K", dequant_q3_k, 110, (108,)),
        ("dequantize_row_q5_K", dequant_q5_k, 176, (0, 2)),
    ],
)
@pytest.mark.skipif(
    not os.path.exists(_LIBGGML), reason="libggml-base.so not present"
)
def test_k_quant_bit_exact_vs_ggml(fn_name, py_fn, block_bytes, d_offs):
    # Synthetic random blocks (d/dmin pinned to sane fp16 so no inf/nan
    # from random exponent bits can poison the comparison); the C
    # reference and the numpy mirror must agree bit-for-bit.
    lib = ctypes.CDLL(_LIBGGML)
    fn = getattr(lib, fn_name)
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
    fn.restype = None
    rng = np.random.RandomState(1234)
    nb = 7
    raw = bytearray(rng.randint(0, 256, size=nb * block_bytes, dtype=np.uint8))
    for i in range(nb):
        for j, off in enumerate(d_offs):
            v = 1.0 if j == 0 else 0.5
            raw[i * block_bytes + off:i * block_bytes + off + 2] = struct.pack("<e", v)
    mine = py_fn(bytes(raw), nb * 256)
    ref = np.zeros(nb * 256, dtype=np.float32)
    in_buf = ctypes.create_string_buffer(bytes(raw), len(raw))
    out_buf = (ctypes.c_float * (nb * 256)).from_buffer(ref)
    fn(ctypes.addressof(in_buf), ctypes.addressof(out_buf), nb * 256)
    assert mine.shape == ref.shape
    assert np.array_equal(mine, ref), (
        f"{fn_name}: max abs diff "
        f"{np.max(np.abs(mine.astype(np.float64) - ref.astype(np.float64)))}"
    )


@pytest.mark.skipif(
    not os.path.exists(_LIBGGML), reason="libggml-base.so not present"
)
def test_bf16_bit_exact_vs_ggml():
    """BF16 dequantizer vs ggml's own ``ggml_bf16_to_fp32_row`` via
    ctypes. Compared as raw uint32 bit patterns, so NaN payloads
    (which never compare equal as floats) are pinned too. Random
    bytes plus pinned specials: +0, -0, 1.0, -1.0, +inf, -inf, a
    quiet NaN, and a signaling-NaN payload."""
    lib = ctypes.CDLL(_LIBGGML)
    fn = lib.ggml_bf16_to_fp32_row
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
    fn.restype = None
    rng = np.random.RandomState(99)
    n = 64
    raw = bytearray(rng.randint(0, 256, size=n * 2, dtype=np.uint8))
    specials = [0x0000, 0x8000, 0x3F80, 0xBF80, 0x7F80, 0xFF80,
                0x7FC0, 0x7F81]
    for i, bits in enumerate(specials):
        raw[2 * i:2 * i + 2] = struct.pack("<H", bits)
    mine = dequant_bf16(bytes(raw), n)
    ref = np.zeros(n, dtype=np.float32)
    in_buf = ctypes.create_string_buffer(bytes(raw), len(raw))
    out_buf = (ctypes.c_float * n).from_buffer(ref)
    fn(ctypes.addressof(in_buf), ctypes.addressof(out_buf), n)
    assert mine.shape == ref.shape
    assert np.array_equal(mine.view("<u4"), ref.view("<u4")), (
        "bf16 vs ggml bit mismatch at "
        f"{np.where(mine.view('<u4') != ref.view('<u4'))[0][:8]}"
    )


def test_convert_passthrough_q2_k(tmp_path):
    # convert.py must NOT requantize or reject Q2_K: exact policy copies
    # raw bytes. Build a 1-tensor Q2_K GGUF, convert, read back through
    # the pager, and compare against direct dequantization.
    import os

    from lacuna.convert import convert as _convert
    from lacuna.gguf import read_tensor
    from lacuna.pager import PagedWeightStore

    rng = np.random.RandomState(7)
    raw = rng.randint(0, 256, size=2 * 84, dtype=np.uint8).tobytes()
    path = os.path.join(tmp_path, "k.gguf")
    data = b"GGUF" + struct.pack("<I", 3)
    data += struct.pack("<Q", 1) + struct.pack("<Q", 0)  # 1 tensor, 0 kv
    name = b"w2"
    data += struct.pack("<Q", len(name)) + name
    data += struct.pack("<I", 2)
    data += struct.pack("<Q", 256) + struct.pack("<Q", 2)  # dims [256, 2]
    data += struct.pack("<I", 10)  # Q2_K
    data += struct.pack("<Q", 0)
    pad = (-len(data)) % 32
    data += b"\x00" * pad + raw
    with open(path, "wb") as f:
        f.write(data)
    lac = path + ".lac"
    _convert(path, lac, quiet=True)
    store = PagedWeightStore(lac, hot_budget_bytes=4 * 1024 * 1024)
    got = read_tensor(store, {"name": "w2", "dims": [256, 2], "type": 10, "offset": 0})
    assert got.shape == (2, 256)
    assert np.array_equal(got.reshape(-1), dequant_q2_k(raw, 512))
    store.close()


# ------------------------------------------------------------- bf16 ---


def _bf16_bits(f32_val: float) -> int:
    """Top 16 bits of a float32 value — the BF16 encoding. Exact
    values (1.0, -2.5, ...) round-trip exactly; the test pins the bit
    pattern, not a decimal approximation."""
    return struct.unpack("<I", struct.pack("<f", f32_val))[0] >> 16


def test_bf16_hand_computed_values():
    # Values exactly representable in BF16: widening is a shift, so
    # the decoded float32 must equal the original bit-for-bit.
    vals = [1.0, -2.5, 0.5, 3.0, 0.0, -0.0, 128.0, 0.25]
    raw = b"".join(struct.pack("<H", _bf16_bits(v)) for v in vals)
    got = dequant_bf16(raw, len(vals))
    want = np.array(vals, dtype=np.float32)
    assert np.array_equal(got.view("<u4"), want.view("<u4"))
    # tensor_nbytes via _TYPE_INFO: 1 element, 2 bytes.
    assert tensor_nbytes({"name": "w", "dims": [4, 2], "type": 30,
                          "offset": 0}) == 16


def test_bf16_gguf_roundtrip_through_pager(tmp_path):
    """Synthetic BF16 GGUF -> .lac conversion (exact policy, bytes
    copied verbatim) -> read_tensor through the pager. The decoded
    tensor must equal the BF16 widening of the written values."""
    import os

    from lacuna.convert import convert as _convert
    from lacuna.gguf import read_tensor
    from lacuna.pager import PagedWeightStore

    vals = np.array([[1.0, -2.5, 0.5, 3.0],
                     [0.0, -0.0, 128.0, 0.25]], dtype=np.float32)
    raw = b"".join(struct.pack("<H", _bf16_bits(float(v)))
                    for v in vals.reshape(-1))
    path = os.path.join(tmp_path, "b.gguf")
    data = b"GGUF" + struct.pack("<I", 3)
    data += struct.pack("<Q", 1) + struct.pack("<Q", 0)  # 1 tensor, 0 kv
    name = b"w2"
    data += struct.pack("<Q", len(name)) + name
    data += struct.pack("<I", 2)
    data += struct.pack("<Q", 4) + struct.pack("<Q", 2)  # dims [4, 2]
    data += struct.pack("<I", 30)  # BF16
    data += struct.pack("<Q", 0)
    pad = (-len(data)) % 32
    data += b"\x00" * pad + raw
    with open(path, "wb") as f:
        f.write(data)
    lac = path + ".lac"
    _convert(path, lac, quiet=True)
    store = PagedWeightStore(lac, hot_budget_bytes=4 * 1024 * 1024)
    got = read_tensor(store, {"name": "w2", "dims": [4, 2], "type": 30,
                              "offset": 0})
    assert got.shape == (2, 4)
    assert np.array_equal(got.view("<u4"), vals.view("<u4"))
    store.close()


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
