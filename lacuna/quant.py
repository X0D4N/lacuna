"""Adaptive per-page precision (M6) — numpy only.

Three codecs, chosen per page by measured activation frequency:

- Q4  (4 bit): hot pages — symmetric 16-level, max error <= scale/14.
- Q2  (2 bit): warm pages — uniform 4-level {-1,-1/3,1/3,1}*scale, error <= scale/3.
- TQ  (~1.58 bit, ternary): cold pages — values in {-scale, 0, +scale} stored as
  2-bit codes (4 per byte). Cold neurons contribute little by definition, so the
  coarser grid costs little quality.

Precision policies (see also ``lacuna/convert.py``):

- ``"exact"`` (default): pages keep SOURCE precision bit-for-bit — paging and
  sparsity only, zero requantization. In the ``.lac`` file every tensor is
  stored as RAW byte pages, byte-identical to the source GGUF.
- ``"extreme"``: the adaptive Q4/Q2/ternary tiering above — fastest, with a
  documented quality/speed tradeoff.

Honest note on quality: the open-source ecosystem's standard for full
intelligence is Q4_K_M-class quantization. Lacuna's exact mode preserves
whatever precision the source GGUF had, bit-for-bit — there is never
*additional* degradation beyond what the source file already chose.
Post-training 1-2 bit quantization of an existing dense model does degrade
quality (see docs/PRIOR_ART.md: GPTQ-2bit collapses); Lacuna applies the
coarse tiers only under the ``extreme`` policy, which the user opts into
explicitly.
"""

from __future__ import annotations

import numpy as np

Q4 = 4
Q2 = 2
TERNARY = 1  # ~1.58-bit tier
RAW = 0  # opaque bytes (e.g. pre-quantized GGUF blocks), no codec

POLICY_EXACT = "exact"
POLICY_EXTREME = "extreme"
PRECISION_POLICIES = (POLICY_EXACT, POLICY_EXTREME)


def validate_policy(policy: str) -> str:
    """Return the policy if known, else raise. Keeps typos loud, not silent."""
    if policy not in PRECISION_POLICIES:
        raise ValueError(
            f"unknown precision policy {policy!r} (expected one of {PRECISION_POLICIES})"
        )
    return policy

_PRECISION_NAMES = {Q4: "q4", Q2: "q2", TERNARY: "ternary", RAW: "raw"}


def precision_name(p: int) -> str:
    return _PRECISION_NAMES.get(p, f"q{p}")


# ---------------------------------------------------------------- Q4 ---


def pack_q4(x: np.ndarray) -> tuple[np.ndarray, np.float32, tuple[int, ...]]:
    x = np.asarray(x, dtype=np.float32)
    scale = np.float32(np.max(np.abs(x)))  # stored as-is; 0.0 decodes to exact zeros
    enc = scale if scale != 0 else np.float32(1.0)
    # Codes are exactly 0..15: go straight to uint8 (the old int64 intermediate
    # was a wasted full pass over the array; values are unaffected).
    q = np.clip(np.round(x / enc * 7.0), -8, 7) + 8.0
    flat = q.astype(np.uint8).reshape(-1)
    if flat.size % 2:
        flat = np.concatenate([flat, np.array([8], dtype=np.uint8)])  # pad = zero
    packed = flat[0::2] | (flat[1::2] << 4)  # uint8 | uint8 -> uint8
    return packed, scale, x.shape


def unpack_q4(packed: np.ndarray, scale: np.float32, shape: tuple[int, ...]) -> np.ndarray:
    packed = np.asarray(packed, dtype=np.uint8)
    # Stacked interleave instead of two strided writes into an empty buffer:
    # same codes in the same order (low nibble first), one contiguous pass.
    # Measured (round-3 sweep): ~1.6x faster, decoded values bit-identical.
    codes = np.stack([packed & 0x0F, (packed >> 4) & 0x0F], axis=1).reshape(-1)
    vals = (codes.astype(np.float32) - 8.0) / 7.0 * float(scale)
    return vals[: int(np.prod(shape))].reshape(shape)


# ---------------------------------------------------------------- Q2 ---


_Q2_LEVELS = np.array([-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0], dtype=np.float32)

# Decision midpoints between consecutive levels, computed from the float32
# level values exactly as pack_q2's fast path uses them.
_Q2_MIDS = tuple(
    float((_Q2_LEVELS[i] + _Q2_LEVELS[i + 1]) / np.float32(2.0)) for i in range(3)
)
# Half-width of the band around each midpoint where pack_q2 falls back to the
# exact argmin: float32 distance rounding is bounded by ~2.4e-7, so a 1e-5
# band carries >40x margin. Verified by an exhaustive ULP sweep in tests.
_Q2_FIXUP_BAND = 1e-5

# The midpoints as float32. Every _Q2_MIDS value is a float32 widened to
# float64, and xn is float32 widened the same way, so comparing in float32
# gives exactly the float64 comparison result (both operands are float32
# values; widening preserves order). Same for the band check: the float64
# difference of two float32 values is their exact real difference, and its
# float32 rounding error (~6e-13 at band scale) is far inside the margin
# between _Q2_FIXUP_BAND and the widened detection band below — detection
# stays a strict superset of the true band, and elements detected outside
# the true band get the exact argmin, which provably agrees with the
# midpoint code there (disagreement is only possible within ~2.4e-7 of a
# midpoint). Output is therefore bit-identical to the float64 version.
_Q2_MIDS32 = tuple(np.float32(m) for m in _Q2_MIDS)
_Q2_DETECT_BAND = np.float32(2e-5)


def pack_q2(x: np.ndarray) -> tuple[np.ndarray, np.float32, tuple[int, ...]]:
    x = np.asarray(x, dtype=np.float32)
    scale = np.float32(np.max(np.abs(x)))  # stored as-is; 0.0 decodes to exact zeros
    enc = scale if scale != 0 else np.float32(1.0)
    xn = x / enc
    # Fast path: nearest level via the three midpoints, compared in float32
    # (exact — see _Q2_MIDS32 note). The old np.argmin over an (..., 4)
    # broadcast cost ~9x more; the packed codes are bit-identical because
    # elements within _Q2_FIXUP_BAND of a midpoint — the only ones float32
    # distance rounding could flip — are recomputed with the exact argmin
    # below (same subtractions, same first-minimum tie rule).
    # Measured (round-3 sweep): ~1.8x faster than the float64 version.
    codes = np.zeros(xn.shape, dtype=np.uint8)
    for m in _Q2_MIDS32:
        codes += (xn > m).astype(np.uint8)
    near = np.zeros(xn.shape, dtype=bool)
    for m in _Q2_MIDS32:
        near |= np.abs(xn - m) <= _Q2_DETECT_BAND
    if near.any():
        sub = xn[near]
        dist = np.abs(sub[..., None] - _Q2_LEVELS)
        codes[near] = np.argmin(dist, axis=-1).astype(np.uint8)
    flat = codes.reshape(-1)
    pad = (-flat.size) % 4
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.uint8)])
    packed = (
        flat[0::4] | (flat[1::4] << 2) | (flat[2::4] << 4) | (flat[3::4] << 6)
    ).astype(np.uint8)
    return packed, scale, x.shape


def unpack_q2(packed: np.ndarray, scale: np.float32, shape: tuple[int, ...]) -> np.ndarray:
    packed = np.asarray(packed, dtype=np.uint8)
    n = int(np.prod(shape))
    codes = np.empty(packed.size * 4, dtype=np.uint8)
    codes[0::4] = packed & 0x03
    codes[1::4] = (packed >> 2) & 0x03
    codes[2::4] = (packed >> 4) & 0x03
    codes[3::4] = (packed >> 6) & 0x03
    return (_Q2_LEVELS[codes[:n]] * float(scale)).reshape(shape)


# ------------------------------------------------------------ ternary ---


def pack_ternary(x: np.ndarray) -> tuple[np.ndarray, np.float32, tuple[int, ...]]:
    """BitNet-style ternary: values in {-scale, 0, +scale}.

    scale = mean(|x|); entries with |x| <= 0.5*scale snap to zero.
    Codes: 0 -> 0, 1 -> +scale, 2 -> -scale (4 codes per byte).
    """
    x = np.asarray(x, dtype=np.float32)
    scale = np.float32(np.mean(np.abs(x)))  # stored as-is; 0.0 decodes to exact zeros
    enc = scale if scale != 0 else np.float32(1.0)
    # Arithmetic codes instead of two masked assignments over a zeros array:
    # the conditions are disjoint (x > t, x < -t), so the sum is exactly the
    # masked-assignment result — bit-identical, ~10x faster (measured).
    t = 0.5 * enc
    codes = (x > t).astype(np.uint8) + 2 * (x < -t).astype(np.uint8)
    flat = codes.reshape(-1)
    pad = (-flat.size) % 4
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.uint8)])
    packed = (
        flat[0::4] | (flat[1::4] << 2) | (flat[2::4] << 4) | (flat[3::4] << 6)
    ).astype(np.uint8)
    return packed, scale, x.shape


_TERNARY_LEVELS = np.array([0.0, 1.0, -1.0, 0.0], dtype=np.float32)


def unpack_ternary(
    packed: np.ndarray, scale: np.float32, shape: tuple[int, ...]
) -> np.ndarray:
    packed = np.asarray(packed, dtype=np.uint8)
    n = int(np.prod(shape))
    codes = np.empty(packed.size * 4, dtype=np.uint8)
    codes[0::4] = packed & 0x03
    codes[1::4] = (packed >> 2) & 0x03
    codes[2::4] = (packed >> 4) & 0x03
    codes[3::4] = (packed >> 6) & 0x03
    # LUT decode (same shape as unpack_q2): code 3 is never produced by
    # pack_ternary and maps to 0.0, exactly like the old masked assignments.
    return (_TERNARY_LEVELS[codes[:n]] * float(scale)).reshape(shape)


# ------------------------------------------------------------- helpers ---


def quantize(x: np.ndarray, precision: int):
    if precision == Q4:
        return pack_q4(x)
    if precision == Q2:
        return pack_q2(x)
    if precision == TERNARY:
        return pack_ternary(x)
    raise ValueError(f"unknown precision tier {precision}")


def dequantize(packed: np.ndarray, scale: np.float32, shape, precision: int):
    if precision == Q4:
        return unpack_q4(packed, scale, shape)
    if precision == Q2:
        return unpack_q2(packed, scale, shape)
    if precision == TERNARY:
        return unpack_ternary(packed, scale, shape)
    raise ValueError(f"unknown precision tier {precision}")


def roundtrip_max_error(x: np.ndarray, precision: int) -> float:
    """Max |x - dequantize(quantize(x))|. Finite for every tier by construction."""
    packed, scale, shape = quantize(np.asarray(x, dtype=np.float32), precision)
    rt = dequantize(packed, scale, shape, precision)
    return float(np.max(np.abs(np.asarray(x, dtype=np.float32) - rt)))
