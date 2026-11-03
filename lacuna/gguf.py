"""GGUF reading for Lacuna's paged executor: metadata, tensors, tokenizer.

This module is the bridge between a real GGUF model file and Lacuna's
:class:`~lacuna.pager.PagedWeightStore`. It does three things, nothing more:

1. **Parse** a GGUF v3 file: full metadata KV section (architecture dims,
   attention config, RoPE, norms, tokenizer vocab/merges/bos/eos) plus the
   tensor index (name, dims, type, data offset). ``convert.parse_gguf_header``
   only *skips* the KV section; this parser *reads* it.
2. **Dequantize** GGUF block formats (Q5_0, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K,
   Q6_K, F16, F32, BF16) in float32,
   mirroring the reference ``dequantize_row_*`` implementations in
   llama.cpp's ``ggml/src/ggml-quants.c`` operation for operation (same
   scale/min packing, same nibble order, same float32 arithmetic). A wrong
   nibble order here silently produces garbage weights, so the mapping is
   checked against that source, not from memory.
3. **Read tensors through the pager**: :func:`read_tensor` pulls a tensor's
   bytes page by page from a ``PagedWeightStore`` (exact policy only — RAW
   pages are the source bytes) and dequantizes them. No weight byte ever
   bypasses the store: the executor's memory discipline *is* the pager's
   LRU budget, and ``store.stats`` reports exactly what was faulted.

GGUF layout facts used here (v3, little-endian):
- Header: magic ``GGUF`` | u32 version | u64 tensor_count | u64 kv_count.
- KV entry: string key | u32 value_type | value. Value types: 0 uint8,
  1 int8, 2 uint16, 3 int16, 4 uint32, 5 int32, 6 float32, 7 bool,
  8 string, 9 array, 10 uint64, 11 int64, 12 float64. Array: u32 elem type
  | u64 count | elements (strings carry their own u64 length each).
- Tensor info: string name | u32 n_dims | n_dims x u64 dims | u32 type |
  u64 offset (relative to the data section, which starts at the next
  32-byte boundary after the header).
- Dims are stored fastest-first: for a 2-D weight, ``dims == [n_in, n_out]``
  and the data is row-major with ``n_in`` contiguous. The logical array is
  therefore reshaped ``(n_out, n_in)``.
- GGML type ids (this llama.cpp checkout's ggml.h — NOT the older numbering
  some references still show): F32=0, F16=1, Q4_0=2, Q4_1=3, Q5_0=6,
  Q5_1=7, Q8_0=8, Q8_1=9, Q2_K=10, Q3_K=11, Q4_K=12, Q5_K=13, Q6_K=14,
  Q8_K=15, BF16=30. Decoded here: F32, F16, BF16, Q5_0, Q8_0, Q2_K, Q3_K,
  Q4_K, Q5_K, Q6_K; anything else raises loudly.
"""

from __future__ import annotations

import struct

import numpy as np

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3

# GGML type id -> (values per block, bytes per block, dequantizer name)
_TYPE_INFO = {
    0: (1, 4, "f32"),       # F32
    1: (1, 2, "f16"),       # F16
    6: (32, 22, "q5_0"),    # Q5_0
    8: (32, 34, "q8_0"),    # Q8_0
    10: (256, 84, "q2_k"),   # Q2_K
    11: (256, 110, "q3_k"),  # Q3_K
    12: (256, 144, "q4_k"),  # Q4_K
    13: (256, 176, "q5_k"),  # Q5_K
    14: (256, 210, "q6_k"),  # Q6_K
    30: (1, 2, "bf16"),      # BF16 (ggml.h: GGML_TYPE_BF16 = 30)
}

_KV_FMT = {
    0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2),
    4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1),
    10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8),
}


class GGUFError(ValueError):
    """Raised for malformed GGUF input or unsupported types. Fails loud."""


# ---------------------------------------------------------------- parsing ---


def _read_str(f) -> str:
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8", errors="replace")


def _read_value(f, vtype: int):
    if vtype == 8:  # string
        return _read_str(f)
    if vtype == 9:  # array
        (etype,) = struct.unpack("<I", f.read(4))
        (n,) = struct.unpack("<Q", f.read(8))
        return [_read_value(f, etype) for _ in range(n)]
    fmt, _ = _KV_FMT[vtype]
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]


def parse_gguf(path: str):
    """Return ``(metadata, tensor_infos, data_start)`` for a GGUF v3 file.

    ``metadata`` maps key -> value (str/int/float/bool/list). ``tensor_infos``
    is a list of dicts with ``name``, ``dims`` (fastest-first), ``type`` and
    ``offset`` (bytes from ``data_start``).
    """
    with open(path, "rb") as f:
        if f.read(4) != GGUF_MAGIC:
            raise GGUFError(f"not a GGUF file: {path}")
        (version,) = struct.unpack("<I", f.read(4))
        if version != GGUF_VERSION:
            raise GGUFError(f"GGUF v{version} not supported (need v3): {path}")
        (n_tensors,) = struct.unpack("<Q", f.read(8))
        (n_kv,) = struct.unpack("<Q", f.read(8))
        metadata = {}
        for _ in range(n_kv):
            key = _read_str(f)
            (vtype,) = struct.unpack("<I", f.read(4))
            metadata[key] = _read_value(f, vtype)
        infos = []
        for _ in range(n_tensors):
            name = _read_str(f)
            (n_dims,) = struct.unpack("<I", f.read(4))
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
            (ttype,) = struct.unpack("<I", f.read(4))
            (offset,) = struct.unpack("<Q", f.read(8))
            infos.append(
                {"name": name, "dims": dims, "type": ttype, "offset": offset}
            )
        pos = f.tell()
        data_start = pos + ((-pos) % 32)
    return metadata, infos, data_start


def tensor_numel(dims) -> int:
    n = 1
    for d in dims:
        n *= int(d)
    return n


def tensor_nbytes(info) -> int:
    """Exact byte length of a tensor's data in the GGUF file."""
    if info["type"] not in _TYPE_INFO:
        raise GGUFError(
            f"tensor {info['name']}: unsupported GGML type {info['type']}"
        )
    per_block, bytes_per_block, _ = _TYPE_INFO[info["type"]]
    n = tensor_numel(info["dims"])
    if n % per_block:
        raise GGUFError(
            f"tensor {info['name']}: {n} elements not a multiple of "
            f"block size {per_block}"
        )
    return (n // per_block) * bytes_per_block


def tensor_shape(dims):
    """Logical numpy shape for GGUF dims (fastest-first). 1-D and 2-D only."""
    if len(dims) == 1:
        return (int(dims[0]),)
    if len(dims) == 2:
        return (int(dims[1]), int(dims[0]))  # (n_out, n_in)
    raise GGUFError(f"unsupported tensor rank {len(dims)} (dims {dims})")


# -------------------------------------------------------------- dequant ---


def _f16(b: np.ndarray) -> np.ndarray:
    # Slices of the block matrix are strided views; a dtype-changing view
    # needs a contiguous buffer, so copy first (2 bytes per block: cheap).
    return np.ascontiguousarray(b).view("<f2").astype(np.float32)


def dequant_f32(data: bytes, n: int) -> np.ndarray:
    arr = np.frombuffer(data, dtype="<f4", count=n)
    return np.array(arr, dtype=np.float32, copy=True)


def dequant_f16(data: bytes, n: int) -> np.ndarray:
    return _f16(np.frombuffer(data, dtype=np.uint8, count=n * 2)).copy()


def dequant_bf16(data: bytes, n: int) -> np.ndarray:
    """BF16 -> float32: exact by construction, not an approximation.

    BF16 is float32 truncated to its top 16 bits (1 sign, 8 exponent,
    7 mantissa bits), so widening is a left shift by 16 into the
    float32 bit pattern — no rounding, no table, no arithmetic. This
    mirrors ggml's ``ggml_bf16_to_fp32_row`` (verified bit-exact via
    ctypes in tests/test_gguf.py, including NaN payloads, which survive
    the shift intact). Never downcasts: the output is float32.
    """
    u32 = np.frombuffer(data, dtype="<u2", count=n).astype(np.uint32)
    u32 <<= np.uint32(16)
    return u32.view(np.float32).copy()


def dequant_q5_0(data: bytes, n: int) -> np.ndarray:
    """Mirror of ggml dequantize_row_q5_0: 32 values per 22-byte block.

    Block: fp16 d | 4 qh bytes (one uint32, little-endian) | 16 qs bytes.
    For j in 0..15: the low nibble of qs[j] plus bit j of qh gives value j
    (minus 16, times d); the high nibble plus bit j+12 of qh gives value
    j+16 (minus 16, times d).
    """
    nb = n // 32
    blocks = np.frombuffer(data, dtype=np.uint8, count=nb * 22).reshape(nb, 22)
    d = _f16(blocks[:, 0:2]).reshape(nb)
    qh = np.ascontiguousarray(blocks[:, 2:6]).view("<u4").reshape(nb)
    qs = blocks[:, 6:22]
    j = np.arange(16, dtype=np.uint32)
    xh_0 = (((qh[:, None] >> j) << 4) & 0x10).astype(np.int32)
    xh_1 = ((qh[:, None] >> (j + 12)) & 0x10).astype(np.int32)
    x0 = ((qs & 0x0F).astype(np.int32) | xh_0) - 16
    x1 = ((qs >> 4).astype(np.int32) | xh_1) - 16
    out = np.empty((nb, 32), dtype=np.float32)
    out[:, :16] = x0.astype(np.float32) * d[:, None]
    out[:, 16:] = x1.astype(np.float32) * d[:, None]
    return out.reshape(-1)


def dequant_q8_0(data: bytes, n: int) -> np.ndarray:
    """Mirror of ggml dequantize_row_q8_0: 32 values per 34-byte block,
    y = qs[i] * d with d a float16 scale."""
    nb = n // 32
    blocks = np.frombuffer(data, dtype=np.uint8, count=nb * 34).reshape(nb, 34)
    d = _f16(blocks[:, 0:2]).reshape(nb)
    qs = blocks[:, 2:34].astype(np.int8).astype(np.float32)
    return (qs * d[:, None]).reshape(-1)


def _q4_k_scales_min(scales: np.ndarray):
    """Unpack the 8 (scale, min) pairs from a Q4_K block's 12 scale bytes.

    Mirror of ggml get_scale_min_k4: for pair index j < 4, scale and min
    are the low 6 bits of scales[j] and scales[j+4]; for j >= 4 they are
    reassembled from the low/high nibbles of scales[j+4] plus the top
    2 bits stashed in scales[j-4] / scales[j].
    Returns (sc, mn) as float32 arrays of shape (nb, 8).
    """
    nb = scales.shape[0]
    sc = np.empty((nb, 8), dtype=np.float32)
    mn = np.empty((nb, 8), dtype=np.float32)
    for j in range(8):
        if j < 4:
            sc[:, j] = scales[:, j] & 63
            mn[:, j] = scales[:, j + 4] & 63
        else:
            sc[:, j] = (scales[:, j + 4] & 0x0F) | ((scales[:, j - 4] >> 6) << 4)
            mn[:, j] = (scales[:, j + 4] >> 4) | ((scales[:, j] >> 6) << 4)
    return sc, mn


def dequant_q4_k(data: bytes, n: int) -> np.ndarray:
    """Mirror of ggml dequantize_row_q4_K: 256 values per 144-byte block.

    Block: fp16 d | fp16 dmin | 12 scale bytes | 128 qs bytes. Per 64-value
    group, two (scale, min) pairs apply: the low nibbles of 32 qs bytes give
    the first 32 values (d1 * code - m1), the high nibbles the next 32
    (d2 * code - m2). All arithmetic in float32, as in the C code.
    """
    nb = n // 256
    blocks = np.frombuffer(data, dtype=np.uint8, count=nb * 144).reshape(nb, 144)
    d = _f16(blocks[:, 0:2]).reshape(nb)
    dmin = _f16(blocks[:, 2:4]).reshape(nb)
    sc, mn = _q4_k_scales_min(blocks[:, 4:16])
    qs = blocks[:, 16:144]
    out = np.empty((nb, 256), dtype=np.float32)
    for g in range(4):
        d1 = (d * sc[:, 2 * g]).astype(np.float32)
        m1 = (dmin * mn[:, 2 * g]).astype(np.float32)
        d2 = (d * sc[:, 2 * g + 1]).astype(np.float32)
        m2 = (dmin * mn[:, 2 * g + 1]).astype(np.float32)
        qb = qs[:, g * 32 : (g + 1) * 32]
        lo = (qb & 0x0F).astype(np.float32)
        hi = (qb >> 4).astype(np.float32)
        out[:, g * 64 : g * 64 + 32] = d1[:, None] * lo - m1[:, None]
        out[:, g * 64 + 32 : g * 64 + 64] = d2[:, None] * hi - m2[:, None]
    return out.reshape(-1)


def dequant_q2_k(data: bytes, n: int) -> np.ndarray:
    """Mirror of ggml dequantize_row_q2_K: 256 values per 84-byte block.

    Block: 16 scale bytes | 64 qs bytes | fp16 d | fp16 dmin. Two
    128-value halves; each half consumes 8 scale bytes, one per 16-value
    group: value = d * (scale & 0xF) * code - dmin * (scale >> 4), where
    code is the 2-bit field of qs at bit position 2*j within the half's
    32 qs bytes (j = group index inside the half). Groups run j=0..3
    with shift 0,2,4,6, each covering two 16-value subgroups (qs[0:16]
    then qs[16:32] of the half).
    """
    nb = n // 256
    blocks = np.frombuffer(data, dtype=np.uint8, count=nb * 84).reshape(nb, 84)
    scales = blocks[:, 0:16]
    qs = blocks[:, 16:80]
    d = _f16(blocks[:, 80:82]).reshape(nb)
    dmin = _f16(blocks[:, 82:84]).reshape(nb)
    out = np.empty((nb, 256), dtype=np.float32)
    for h in range(2):
        qh = qs[:, h * 32:(h + 1) * 32]
        for j in range(4):
            shift = 2 * j
            sc1 = scales[:, h * 8 + 2 * j]
            sc2 = scales[:, h * 8 + 2 * j + 1]
            dl1 = d * (sc1 & 0x0F).astype(np.float32)
            ml1 = dmin * (sc1 >> 4).astype(np.float32)
            dl2 = d * (sc2 & 0x0F).astype(np.float32)
            ml2 = dmin * (sc2 >> 4).astype(np.float32)
            base = h * 128 + j * 32
            out[:, base:base + 16] = (
                dl1[:, None]
                * ((qh[:, 0:16] >> shift) & 3).astype(np.float32)
                - ml1[:, None]
            )
            out[:, base + 16:base + 32] = (
                dl2[:, None]
                * ((qh[:, 16:32] >> shift) & 3).astype(np.float32)
                - ml2[:, None]
            )
    return out.reshape(-1)


def dequant_q3_k(data: bytes, n: int) -> np.ndarray:
    """Mirror of ggml dequantize_row_q3_K: 256 values per 110-byte block.

    Block: 32 hmask bytes | 64 qs bytes | 12 scale bytes | fp16 d.
    The 12 scale bytes unpack into 16 6-bit scales (see below); each
    scale covers one 16-value group with dl = d * (scale - 32). Each
    value's code is its 2 low bits from qs plus its high bit from
    hmask: code = low2 - (high_bit ? 0 : 4), so code ranges -4..3.
    The hmask bytes are NOT advanced between the two 128-value halves:
    the first half uses hmask bits 0..3 (m = 1,2,4,8) and the second
    half reuses the same bytes with bits 4..7 (m = 16,32,64,128).

    Scale unpack (ggml's aux-word shuffle, little-endian words a0, a1
    from scales[0:8], tmp from scales[8:12]):
    w0 = (a0 & 0x0f0f0f0f) | ((tmp & 0x03030303) << 4)
    w1 = (a1 & 0x0f0f0f0f) | (((tmp >> 2) & 0x03030303) << 4)
    w2 = ((a0 >> 4) & 0x0f0f0f0f) | (((tmp >> 4) & 0x03030303) << 4)
    w3 = ((a1 >> 4) & 0x0f0f0f0f) | (((tmp >> 6) & 0x03030303) << 4)
    and the 16 scales are w0..w3's bytes in order.
    """
    nb = n // 256
    blocks = np.frombuffer(data, dtype=np.uint8, count=nb * 110).reshape(nb, 110)
    hm = blocks[:, 0:32]
    qs = blocks[:, 32:96]
    raw = blocks[:, 96:108]
    d = _f16(blocks[:, 108:110]).reshape(nb)
    a = np.ascontiguousarray(raw).view("<u4").reshape(nb, 3)
    a0, a1, tmp = a[:, 0], a[:, 1], a[:, 2]
    k1 = np.uint32(0x03030303)
    k2 = np.uint32(0x0F0F0F0F)
    w0 = (a0 & k2) | (((tmp >> np.uint32(0)) & k1) << np.uint32(4))
    w1 = (a1 & k2) | (((tmp >> np.uint32(2)) & k1) << np.uint32(4))
    w2 = ((a0 >> np.uint32(4)) & k2) | (((tmp >> np.uint32(4)) & k1) << np.uint32(4))
    w3 = ((a1 >> np.uint32(4)) & k2) | (((tmp >> np.uint32(6)) & k1) << np.uint32(4))
    scales = (
        np.stack([w0, w1, w2, w3], axis=1)
        .view(np.uint8)
        .astype(np.int8)
    )
    out = np.empty((nb, 256), dtype=np.float32)
    for h in range(2):
        qh = qs[:, h * 32:(h + 1) * 32]
        for j in range(4):
            bit = h * 4 + j
            for s in range(2):
                g = h * 8 + j * 2 + s
                dl = d * (scales[:, g].astype(np.float32) - np.float32(32))
                low = ((qh[:, s * 16:(s + 1) * 16] >> (2 * j)) & 3).astype(np.float32)
                high = (hm[:, s * 16:(s + 1) * 16] >> bit) & 1
                code = low - np.where(high != 0, np.float32(0), np.float32(4))
                base = h * 128 + j * 32 + s * 16
                out[:, base:base + 16] = dl[:, None] * code
    return out.reshape(-1)


def dequant_q5_k(data: bytes, n: int) -> np.ndarray:
    """Mirror of ggml dequantize_row_q5_K: 256 values per 176-byte block.

    Block: fp16 d | fp16 dmin | 12 scale bytes | 32 qh bytes | 128 qs
    bytes. Four 64-value groups; group g uses scale/min pair 2g (first
    32 values) and 2g+1 (next 32) from the Q4_K-style 12-byte packing,
    low nibbles of its 32 qs bytes for the first 32 values and high
    nibbles for the next 32, plus the high bit from qh byte bit 2g
    (first 32) / 2g+1 (next 32): value = d*scale*code - dmin*min with
    code in 0..31.
    """
    nb = n // 256
    blocks = np.frombuffer(data, dtype=np.uint8, count=nb * 176).reshape(nb, 176)
    d = _f16(blocks[:, 0:2]).reshape(nb)
    dmin = _f16(blocks[:, 2:4]).reshape(nb)
    sc, mn = _q4_k_scales_min(blocks[:, 4:16])
    qh = blocks[:, 16:48]
    qs = blocks[:, 48:176]
    out = np.empty((nb, 256), dtype=np.float32)
    for g in range(4):
        d1 = (d * sc[:, 2 * g]).astype(np.float32)
        m1 = (dmin * mn[:, 2 * g]).astype(np.float32)
        d2 = (d * sc[:, 2 * g + 1]).astype(np.float32)
        m2 = (dmin * mn[:, 2 * g + 1]).astype(np.float32)
        ql = qs[:, g * 32:(g + 1) * 32]
        c1 = (ql & 0x0F).astype(np.float32) + np.where(
            ((qh >> (2 * g)) & 1) != 0, np.float32(16), np.float32(0)
        )
        c2 = (ql >> 4).astype(np.float32) + np.where(
            ((qh >> (2 * g + 1)) & 1) != 0, np.float32(16), np.float32(0)
        )
        base = g * 64
        out[:, base:base + 32] = d1[:, None] * c1 - m1[:, None]
        out[:, base + 32:base + 64] = d2[:, None] * c2 - m2[:, None]
    return out.reshape(-1)


def dequant_q6_k(data: bytes, n: int) -> np.ndarray:
    """Mirror of ggml dequantize_row_q6_K: 256 values per 210-byte block.

    Block layout (ggml-common.h — note d is LAST, unlike Q4_K/Q5_0/Q8_0):
    128 ql bytes | 64 qh bytes | 16 int8 scales | fp16 d. Per 128-value
    half, 32 steps of 4 values each: each value combines a 4-bit low part
    from ql with a 2-bit high part from qh, minus 32, times d and the
    sub-block scale (scales are consumed 8 per half, in pairs per step).
    """
    nb = n // 256
    blocks = np.frombuffer(data, dtype=np.uint8, count=nb * 210).reshape(nb, 210)
    d = _f16(blocks[:, 208:210]).reshape(nb)
    sc_all = blocks[:, 192:208].astype(np.int8).astype(np.float32)
    ql_all = blocks[:, 0:128]
    qh_all = blocks[:, 128:192]
    out = np.empty((nb, 256), dtype=np.float32)
    for h in range(2):
        ql = ql_all[:, h * 64 : (h + 1) * 64]
        qh = qh_all[:, h * 32 : (h + 1) * 32]
        sc = sc_all[:, h * 8 : (h + 1) * 8]
        for l in range(32):
            is_ = l // 16
            q1 = (((ql[:, l] & 0x0F) | ((qh[:, l] & 3) << 4)).astype(np.int16) - 32).astype(np.float32)
            q2 = (((ql[:, l + 32] & 0x0F) | (((qh[:, l] >> 2) & 3) << 4)).astype(np.int16) - 32).astype(np.float32)
            q3 = (((ql[:, l] >> 4) | (((qh[:, l] >> 4) & 3) << 4)).astype(np.int16) - 32).astype(np.float32)
            q4 = (((ql[:, l + 32] >> 4) | (((qh[:, l] >> 6) & 3) << 4)).astype(np.int16) - 32).astype(np.float32)
            base = h * 128
            out[:, base + l] = d * sc[:, is_] * q1
            out[:, base + 32 + l] = d * sc[:, is_ + 2] * q2
            out[:, base + 64 + l] = d * sc[:, is_ + 4] * q3
            out[:, base + 96 + l] = d * sc[:, is_ + 6] * q4
    return out.reshape(-1)


_DEQUANT = {0: dequant_f32, 1: dequant_f16, 6: dequant_q5_0, 8: dequant_q8_0, 10: dequant_q2_k, 11: dequant_q3_k, 12: dequant_q4_k, 13: dequant_q5_k, 14: dequant_q6_k, 30: dequant_bf16}


def dequantize(data: bytes, ttype: int, n: int) -> np.ndarray:
    """Dequantize ``n`` values of GGML type ``ttype`` from raw bytes."""
    fn = _DEQUANT.get(ttype)
    if fn is None:
        raise GGUFError(f"unsupported GGML type {ttype} for dequantization")
    return fn(data, n)


# ----------------------------------------------------------- store read ---


def read_tensor(store, info) -> np.ndarray:
    """Read one tensor through a PagedWeightStore and return its f32 array.

    Exact policy only: RAW pages carry the source bytes, so the concatenated
    page payloads (in page-table order) are the tensor's GGUF bytes. The
    byte length is checked against the tensor's declared size — a short
    tensor is corruption and fails loud. Every byte arrives via
    ``store.get_page`` (fault + LRU accounting); nothing reads the .lac
    file directly.
    """
    if getattr(store, "precision_policy", None) != "exact":
        raise GGUFError(
            f"paged executor needs an exact-policy store, got "
            f"{getattr(store, 'precision_policy', None)!r}"
        )
    tid = store.tensor_id(info["name"])
    if tid is None:
        raise GGUFError(f"tensor {info['name']} not present in the store")
    pages = store.tensor_pages(tid)
    blob = b"".join(store.get_page(e["page_id"]).tobytes() for e in pages)
    want = tensor_nbytes(info)
    if len(blob) != want:
        raise GGUFError(
            f"tensor {info['name']}: paged bytes {len(blob)} != "
            f"declared {want}"
        )
    flat = dequantize(blob, info["type"], tensor_numel(info["dims"]))
    return flat.reshape(tensor_shape(info["dims"]))


def read_tensor_rows(store, info, r0: int, r1: int) -> np.ndarray:
    """Read rows ``[r0, r1)`` of a 2-D tensor through a PagedWeightStore.

    Same pager discipline as :func:`read_tensor` (exact policy, every
    byte via ``store.get_page``), but only the pages overlapping the
    requested byte range are faulted. Row ``i`` of a 2-D tensor occupies
    ``(dims[0] // values_per_block) * bytes_per_block`` contiguous bytes
    — rows are block-aligned exactly when ``dims[0]`` is a multiple of
    the block size, which is enforced here (otherwise a row would split
    a quantization block and could not be dequantized alone).

    This is what makes large vocabularies runnable on small machines:
    the embedding matrix and lm_head are never materialized whole —
    the executor gathers only the rows it needs (input token ids, one
    vocab block of logits at a time). A 151936-row head at hidden 896
    is ~545 MB as float32; at 100B-class hidden sizes it would be
    gigabytes. The logits *vector* (one float per vocab row) is small
    and is the only thing ever assembled in full.
    """
    if getattr(store, "precision_policy", None) != "exact":
        raise GGUFError(
            f"paged executor needs an exact-policy store, got "
            f"{getattr(store, 'precision_policy', None)!r}"
        )
    if len(info["dims"]) != 2:
        raise GGUFError(
            f"tensor {info['name']}: read_tensor_rows needs 2-D dims, "
            f"got {info['dims']}"
        )
    if info["type"] not in _TYPE_INFO:
        raise GGUFError(
            f"tensor {info['name']}: unsupported GGML type {info['type']}"
        )
    n_in = int(info["dims"][0])
    n_out = int(info["dims"][1])
    if not (0 <= r0 <= r1 <= n_out):
        raise GGUFError(
            f"tensor {info['name']}: row range [{r0}, {r1}) outside "
            f"[0, {n_out})"
        )
    per_block, bytes_per_block, _ = _TYPE_INFO[info["type"]]
    if n_in % per_block:
        raise GGUFError(
            f"tensor {info['name']}: row length {n_in} not a multiple of "
            f"block size {per_block}; rows cannot be dequantized alone"
        )
    row_bytes = (n_in // per_block) * bytes_per_block
    start = r0 * row_bytes
    end = r1 * row_bytes
    tid = store.tensor_id(info["name"])
    if tid is None:
        raise GGUFError(f"tensor {info['name']} not present in the store")
    # Walk pages in order. page_shape reads only the page header, so
    # boundaries are known without faulting payloads; only overlapping
    # pages are pulled through get_page.
    pieces: list[bytes] = []
    off = 0
    for e in store.tensor_pages(tid):
        _, plen = store.page_shape(e["page_id"])  # RAW: (0, payload bytes)
        p_start, p_end = off, off + plen
        if p_end > start and p_start < end:
            arr = store.get_page(e["page_id"])
            lo = max(start, p_start) - p_start
            hi = min(end, p_end) - p_start
            pieces.append(arr[lo:hi].tobytes())
        off = p_end
        if off >= end:
            break
    blob = b"".join(pieces)
    if len(blob) != end - start:
        raise GGUFError(
            f"tensor {info['name']}: paged row bytes {len(blob)} != "
            f"wanted {end - start}"
        )
    flat = dequantize(blob, info["type"], (r1 - r0) * n_in)
    return flat.reshape((r1 - r0, n_in))
