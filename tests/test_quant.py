"""Quant codec tests: roundtrip error ordering is strict and finite."""

import numpy as np

from lacuna.quant import Q2, Q4, TERNARY, roundtrip_max_error


def test_error_ordering():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((64, 64)).astype(np.float32)
    e4 = roundtrip_max_error(x, Q4)
    e2 = roundtrip_max_error(x, Q2)
    et = roundtrip_max_error(x, TERNARY)
    assert np.isfinite(e4) and np.isfinite(e2) and np.isfinite(et)
    assert e4 < e2 < et, (e4, e2, et)
    # Q4 bound: <= scale/14
    assert e4 <= np.max(np.abs(x)) / 14 + 1e-6


def test_zero_tensor():
    x = np.zeros((8, 8), dtype=np.float32)
    for p in (Q4, Q2, TERNARY):
        assert roundtrip_max_error(x, p) == 0.0


def test_shapes_preserved():
    from lacuna.quant import dequantize, quantize

    x = np.random.default_rng(1).standard_normal((7, 13)).astype(np.float32)
    for p in (Q4, Q2, TERNARY):
        packed, scale, shape = quantize(x, p)
        assert dequantize(packed, scale, shape, p).shape == x.shape
