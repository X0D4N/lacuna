"""GGUF reading for Lacuna's paged executor: metadata, tensors, tokenizer.

This module is the bridge between a real GGUF model file and Lacuna's
:class:`~lacuna.pager.PagedWeightStore`. It does three things, nothing more:

1. **Parse** a GGUF v3 file: full metadata KV section (architecture dims,
   attention config, RoPE, norms, tokenizer vocab/merges/bos/eos) plus the
   tensor index (name, dims, type, data offset). ``convert.parse_gguf_header``
   only *skips* the KV section; this parser *reads* it.
2. **Dequantize** GGUF block formats (Q5_0, Q8_0, Q4_K, Q6_K, F16, F32) in float32,
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
  Q8_K=15. Only F32, F16, Q5_0, Q8_0, Q4_K, Q6_K are decoded here;
  anything else raises loudly.
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
    12: (256, 144, "q4_k"),  # Q4_K
    14: (256, 210, "q6_k"),  # Q6_K
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


_DEQUANT = {0: dequant_f32, 1: dequant_f16, 6: dequant_q5_0, 8: dequant_q8_0, 12: dequant_q4_k, 14: dequant_q6_k}


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
