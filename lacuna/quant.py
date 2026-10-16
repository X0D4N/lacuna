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
    q = np.clip(np.round(x / enc * 7.0), -8, 7).astype(np.int64) + 8  # 0..15
    flat = q.reshape(-1)
    if flat.size % 2:
        flat = np.concatenate([flat, np.array([8], dtype=np.int64)])  # pad = zero
    packed = (flat[0::2] | (flat[1::2] << 4)).astype(np.uint8)
    return packed, scale, x.shape


def unpack_q4(packed: np.ndarray, scale: np.float32, shape: tuple[int, ...]) -> np.ndarray:
    packed = np.asarray(packed, dtype=np.uint8)
    flat = np.empty(packed.size * 2, dtype=np.float32)
    flat[0::2] = (packed & 0x0F).astype(np.float32)
    flat[1::2] = ((packed >> 4) & 0x0F).astype(np.float32)
    vals = (flat - 8.0) / 7.0 * float(scale)
    return vals[: int(np.prod(shape))].reshape(shape)


# ---------------------------------------------------------------- Q2 ---


_Q2_LEVELS = np.array([-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0], dtype=np.float32)


def pack_q2(x: np.ndarray) -> tuple[np.ndarray, np.float32, tuple[int, ...]]:
    x = np.asarray(x, dtype=np.float32)
    scale = np.float32(np.max(np.abs(x)))  # stored as-is; 0.0 decodes to exact zeros
    enc = scale if scale != 0 else np.float32(1.0)
    xn = x / enc
    codes = np.argmin(np.abs(xn[..., None] - _Q2_LEVELS), axis=-1).astype(np.uint8)
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
    codes = np.zeros(x.shape, dtype=np.uint8)
    codes[x > 0.5 * enc] = 1
    codes[x < -0.5 * enc] = 2
    flat = codes.reshape(-1)
    pad = (-flat.size) % 4
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.uint8)])
    packed = (
        flat[0::4] | (flat[1::4] << 2) | (flat[2::4] << 4) | (flat[3::4] << 6)
    ).astype(np.uint8)
    return packed, scale, x.shape


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
    codes = codes[:n]
    out = np.zeros(n, dtype=np.float32)
    out[codes == 1] = float(scale)
    out[codes == 2] = -float(scale)
    return out.reshape(shape)


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
