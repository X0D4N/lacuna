"""Torch K-quant dequant kernels vs lacuna.gguf (numpy reference).

Covers the kernels added 2026-10-26: Q2_K, Q3_K, Q5_K, Q6_K. Random
block bytes (not valid model weights) are enough: parity is a pure
function of the block layout, and the numpy path is the ggml mirror
already verified elsewhere. Q5_0 deliberately stays a CPU fallback
and is not covered here.

Torch is CPU here (no CUDA on this box); the CUDA-only check is
the parent-launched Kaggle job's work, not this file's.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lacuna.gguf import dequantize as gguf_dequantize
from lacuna.gpu_executor import (
    GGML_Q2_K,
    GGML_Q3_K,
    GGML_Q5_K,
    GGML_Q6_K,
    GPU_DEQUANT_TYPES,
    TorchBackend,
)

_CASES = [
    (GGML_Q2_K, 84, "Q2_K"),
    (GGML_Q3_K, 110, "Q3_K"),
    (GGML_Q5_K, 176, "Q5_K"),
    (GGML_Q6_K, 210, "Q6_K"),
]


def test_kquant_types_are_native():
    for ttype, _, _ in _CASES:
        assert ttype in GPU_DEQUANT_TYPES


@pytest.mark.parametrize("ttype,block_bytes,name", _CASES)
def test_torch_kquant_matches_gguf_exactly(ttype, block_bytes, name):
    rng = np.random.RandomState(1234)
    for nblocks in (1, 3):
        raw = rng.randint(0, 256, size=nblocks * block_bytes,
                          dtype=np.uint8).tobytes()
        ref = gguf_dequantize(raw, ttype, nblocks * 256)
        backend = TorchBackend("cpu")
        got = backend.to_numpy(backend.dequantize(raw, ttype, nblocks * 256))
        assert got.shape == ref.shape
        # Same float32 arithmetic order as gguf: require exact equality,
        # not a loose tolerance — a tolerance here would hide a layout bug.
        # equal_nan=True only: random bytes can make the f16 scale an
        # Inf/NaN, in which case both sides must produce NaN in the
        # same positions; that is still a layout match, not a value.
        assert np.array_equal(got, ref, equal_nan=True), (
            f"{name}: max abs diff "
            f"{np.max(np.abs(got - ref)) if got.size else 0}"
        )


@pytest.mark.parametrize("ttype,block_bytes,name", _CASES)
def test_torch_kquant_rejects_non_multiple(ttype, block_bytes, name):
    backend = TorchBackend("cpu")
    with pytest.raises(ValueError):
        backend.dequantize(b"\x00" * block_bytes, ttype, 255)
