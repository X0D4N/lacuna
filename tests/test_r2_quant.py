"""Round-2 codec optimizations: bit-identity vs the original algorithms.

pack_q2 (midpoint fast path + exact fixup band), pack_ternary (arithmetic
codes) and unpack_ternary (LUT) were rewritten for speed in the round-2
sweep. These tests pin them to slow reference implementations of the
ORIGINAL algorithms — any divergence, including at float32 tie points,
fails here.
"""

import numpy as np

from lacuna.quant import (
    _Q2_LEVELS,
    pack_q2,
    pack_ternary,
    unpack_ternary,
)

RNG = np.random.default_rng(2026)


def _pack_q2_reference(x):
    x = np.asarray(x, dtype=np.float32)
    scale = np.float32(np.max(np.abs(x)))
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


def _pack_ternary_reference(x):
    x = np.asarray(x, dtype=np.float32)
    scale = np.float32(np.mean(np.abs(x)))
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


def _unpack_ternary_reference(packed, scale, shape):
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


def _assert_same_pack(a, b):
    assert np.array_equal(a[0], b[0])
    # NaN scale arises when an input contains NaN bit patterns; both
    # implementations produce it identically, so NaN == NaN counts as same.
    assert a[1] == b[1] or (np.isnan(a[1]) and np.isnan(b[1]))
    assert a[2] == b[2]


def test_pack_q2_matches_reference_random():
    for shape in [(1, 1), (3, 5), (64, 64), (257, 131)]:
        x = (RNG.standard_normal(shape) * 7.0).astype(np.float32)
        _assert_same_pack(pack_q2(x), _pack_q2_reference(x))


def test_pack_q2_matches_reference_at_ties_and_edges():
    # exact level values, exact midpoints, zeros, denormals, huge values
    vals = np.array(
        [[-1.0, -2.0 / 3.0, -1.0 / 3.0, 0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0,
          -1e-30, 1e-30, 0.5, -0.5, 123.0, -123.0]],
        dtype=np.float32,
    )
    big = np.concatenate([vals.ravel(), RNG.standard_normal(4093).astype(np.float32)])
    _assert_same_pack(pack_q2(big.reshape(1, -1)), _pack_q2_reference(big.reshape(1, -1)))
    zeros = np.zeros((4, 4), dtype=np.float32)
    _assert_same_pack(pack_q2(zeros), _pack_q2_reference(zeros))


def test_pack_q2_ulp_sweep_around_midpoints():
    # every float32 within +/-200k ULP of each decision midpoint: the fixup
    # band must make the fast path agree with true argmin everywhere here.
    # Bit patterns are built per sign: small NEGATIVE int32 patterns are
    # NaN encodings, not negative floats, so the negative side of the 0.0
    # sweep is constructed via the sign bit instead.
    for mid in (np.float32(-2.0 / 3.0), np.float32(0.0), np.float32(2.0 / 3.0)):
        base = int(mid.view(np.int32))
        offs = np.arange(-200_000, 200_001, dtype=np.int64)
        if mid == 0:
            pats = np.concatenate(
                [offs[offs >= 0], (np.int64(-2**31) + offs[offs > 0])]
            ).astype(np.int32)
        else:
            pats = (base + offs).astype(np.int32)
        vals = pats.view(np.float32).reshape(1, -1)
        assert not np.isnan(vals).any()
        _assert_same_pack(pack_q2(vals), _pack_q2_reference(vals))


def test_pack_ternary_matches_reference():
    for shape in [(1, 1), (3, 5), (64, 64), (257, 131)]:
        x = (RNG.standard_normal(shape) * 2.0).astype(np.float32)
        _assert_same_pack(pack_ternary(x), _pack_ternary_reference(x))
    # values exactly at the +/-0.5*scale snap thresholds
    x = np.array([[1.0, -1.0, 2.0, -2.0, 0.0, 4.0]], dtype=np.float32)
    _assert_same_pack(pack_ternary(x), _pack_ternary_reference(x))
    zeros = np.zeros((4, 4), dtype=np.float32)
    _assert_same_pack(pack_ternary(zeros), _pack_ternary_reference(zeros))


def test_unpack_ternary_matches_reference_including_code3():
    x = (RNG.standard_normal((128, 64)) * 2.0).astype(np.float32)
    packed, scale, shape = pack_ternary(x)
    assert np.array_equal(
        unpack_ternary(packed, scale, shape),
        _unpack_ternary_reference(packed, scale, shape),
    )
    # 0xFF bytes carry code 3 in every slot — never produced by pack, must
    # still decode to 0.0 exactly like the reference
    raw = np.full((16,), 0xFF, dtype=np.uint8)
    assert np.array_equal(
        unpack_ternary(raw, np.float32(2.0), (64,)),
        _unpack_ternary_reference(raw, np.float32(2.0), (64,)),
    )
