"""GPU executor core tests. Everything here passes on a GPU-less box:
the numpy backend is fully exercised, and every torch test either runs
on torch-CPU (when torch imports) or skips cleanly. No test fakes a GPU
result — a skip is reported as a skip, and the Kaggle script (not this
file) is what validates CUDA execution.

The layer reference below is written independently from
``lacuna.gpu_executor.GPULayer`` (plain numpy, explicit einsum-style
steps from the Qwen2 math description), so the comparison is a real
cross-check, not the same code run twice.
"""

import numpy as np
import pytest

from lacuna import gpu_executor as ge

try:
    import torch  # noqa: F401  (presence probe only; tests use ge APIs)
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

needs_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
torch_cuda = pytest.mark.skipif(
    not ge.gpu_available(), reason="no CUDA torch on this box")


def tiny_config():
    return ge.LayerConfig(hidden=16, ffn=32, n_head=2, n_kv=1,
                          eps=1e-5, rope_theta=10000.0)


def tiny_weights(rng):
    """Synthetic float weights for one tiny layer, as WeightRefs."""
    def w(shape):
        return (rng.standard_normal(shape) * 0.1).astype(np.float32)

    def ref(arr):
        return ge.WeightRef(shape=tuple(arr.shape), array=arr)

    return {
        "attn_norm.weight": ref(w((16,))),
        "attn_q.weight": ref(w((16, 16))),
        "attn_q.bias": ref(w((16,))),
        "attn_k.weight": ref(w((8, 16))),
        "attn_k.bias": ref(w((8,))),
        "attn_v.weight": ref(w((8, 16))),
        "attn_v.bias": ref(w((8,))),
        "attn_output.weight": ref(w((16, 16))),
        "ffn_norm.weight": ref(w((16,))),
        "ffn_gate.weight": ref(w((32, 16))),
        "ffn_up.weight": ref(w((32, 16))),
        "ffn_down.weight": ref(w((16, 32))),
    }


def reference_layer(x, mats, cfg, positions, cache=None):
    """Independent numpy Qwen2 layer (same math, separately written)."""
    eps = cfg.eps
    hd = cfg.head_dim
    group = cfg.group

    def rmsnorm(v, w):
        return v / np.sqrt(np.mean(v * v, axis=-1, keepdims=True) + eps) * w

    def rope(t, pos):
        inv = cfg.rope_theta ** (-np.arange(0, hd, 2, dtype=np.float32) / hd)
        ang = pos[:, None].astype(np.float32) * inv[None, :]
        cos = np.cos(ang)[:, None, :]
        sin = np.sin(ang)[:, None, :]
        h = t.shape[-1] // 2
        x0, x1 = t[..., :h], t[..., h:]
        out = np.empty_like(t)
        out[..., :h] = x0 * cos - x1 * sin
        out[..., h:] = x0 * sin + x1 * cos
        return out

    T = x.shape[0]
    h = rmsnorm(x, mats["attn_norm.weight"])
    q = h @ mats["attn_q.weight"].T + mats["attn_q.bias"]
    k = h @ mats["attn_k.weight"].T + mats["attn_k.bias"]
    v = h @ mats["attn_v.weight"].T + mats["attn_v.bias"]
    q = rope(q.reshape(T, cfg.n_head, hd), positions)
    k = rope(k.reshape(T, cfg.n_kv, hd), positions)
    v = v.reshape(T, cfg.n_kv, hd)
    k = np.repeat(k, group, axis=1)
    v = np.repeat(v, group, axis=1)
    if cache is not None and "k" in cache:
        k = np.concatenate([cache["k"], k], axis=0)
        v = np.concatenate([cache["v"], v], axis=0)
    if cache is not None:
        cache["k"], cache["v"] = k, v
    scores = np.einsum("thd,shd->ths", q, k) / np.sqrt(hd)
    S = k.shape[0]
    col = np.arange(S)[None, None, :]
    row = (np.arange(T))[:, None, None]
    scores = np.where(col <= row, scores, np.float32(-1e30))
    scores = scores - scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    att = np.einsum("ths,shd->thd", probs, v)
    x = x + att.reshape(T, cfg.hidden) @ mats["attn_output.weight"].T
    h2 = rmsnorm(x, mats["ffn_norm.weight"])
    gate = h2 @ mats["ffn_gate.weight"].T
    up = h2 @ mats["ffn_up.weight"].T
    down = (gate / (1.0 + np.exp(-gate)) * up) @ mats["ffn_down.weight"].T
    return x + down


# ------------------------------------------------------- environment


def test_no_gpu_here_reports_honestly():
    """On this box there is no GPU: detection must say so, not crash
    and not claim one."""
    assert ge.gpu_available() is False
    env = ge.describe_environment()
    assert env["has_gpu"] is False
    assert env["torch_cuda"] is False
    assert env["numpy_version"]


def test_detect_backend_auto_falls_back_to_numpy():
    backend, info = ge.detect_backend("auto")
    assert isinstance(backend, ge.NumpyBackend)
    assert backend.is_gpu is False
    assert info["backend"] == "numpy"
    assert "reason" in info["backend_reason"] or info["backend_reason"]


def test_describe_environment_keys():
    env = ge.describe_environment()
    for key in ("has_gpu", "torch_version", "torch_cuda",
                "cuda_device_name", "cupy_version", "numpy_version",
                "nvidia_smi"):
        assert key in env


# ------------------------------------------------------- numpy layer


def test_numpy_layer_matches_reference():
    rng = np.random.RandomState(7)
    cfg = tiny_config()
    refs = tiny_weights(rng)
    mats = {k: np.array(v.array, copy=True) for k, v in refs.items()}
    x = rng.standard_normal((3, 16)).astype(np.float32)
    pos = np.arange(3, dtype=np.float32)
    layer = ge.GPULayer(ge.NumpyBackend(), cfg)
    got = layer.forward(x, refs, pos)
    want = reference_layer(x, mats, cfg, pos)
    assert got.shape == want.shape
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_numpy_layer_deterministic_and_cached():
    rng = np.random.RandomState(11)
    cfg = tiny_config()
    refs = tiny_weights(rng)
    layer = ge.GPULayer(ge.NumpyBackend(), cfg)
    x = rng.standard_normal((1, 16)).astype(np.float32)
    cache: dict = {}
    out1 = layer.forward(x, refs, np.array([0], dtype=np.float32), cache)
    # Same input, fresh cache: identical (determinism).
    out1b = layer.forward(x, refs, np.array([0], dtype=np.float32), {})
    np.testing.assert_array_equal(out1, out1b)
    # Cache influence needs a DIFFERENT second token: with the same
    # token, V is identical at both positions, so attention over [v, v]
    # is v whatever the probs — outputs would be correctly identical.
    x2 = rng.standard_normal((1, 16)).astype(np.float32)
    out_cached = layer.forward(x2, refs, np.array([1], dtype=np.float32),
                               cache)
    out_fresh = layer.forward(x2, refs, np.array([1], dtype=np.float32), {})
    assert np.all(np.isfinite(out_cached))
    assert not np.array_equal(out_cached, out_fresh)
    assert cache["k"].shape[0] == 2  # two cached positions


def test_numpy_dequant_matches_gguf():
    """NumpyBackend dequant == lacuna.gguf dequant, exactly, for the
    block formats the executor streams (random block bytes are valid
    test input: dequantization imposes no validity constraint)."""
    from lacuna.gguf import dequantize as gguf_dq

    rng = np.random.RandomState(3)
    backend = ge.NumpyBackend()
    for ttype, block_bytes, per in ((ge.GGML_Q8_0, 34, 32),
                                    (ge.GGML_Q4_K, 144, 256)):
        raw = rng.randint(0, 256, size=(4, block_bytes), dtype=np.uint8)
        # Pin the float16 scale bytes to 1.0 (0x3C00): random scale bits
        # can be NaN, which only adds warning noise to the comparison.
        raw[:, 0:2] = np.array([0x00, 0x3C], dtype=np.uint8)
        if ttype == ge.GGML_Q4_K:
            raw[:, 2:4] = np.array([0x00, 0x3C], dtype=np.uint8)
        raw = raw.tobytes()
        got = backend.dequantize(raw, ttype, 4 * per)
        want = gguf_dq(raw, ttype, 4 * per)
        np.testing.assert_array_equal(np.asarray(got), want)


def test_missing_weight_fails_loud():
    cfg = tiny_config()
    refs = tiny_weights(np.random.RandomState(1))
    del refs["ffn_down.weight"]
    layer = ge.GPULayer(ge.NumpyBackend(), cfg)
    with pytest.raises(ge.GPUExecutorError):
        layer.forward(np.zeros((1, 16), dtype=np.float32), refs,
                      np.array([0], dtype=np.float32))


# ------------------------------------------------------- executor/budget


def test_executor_budget_and_streaming_free():
    rng = np.random.RandomState(5)
    cfg = tiny_config()
    refs = tiny_weights(rng)
    ex = ge.GPUExecutor(ge.NumpyBackend(), cfg,
                        vram_budget_bytes=1024 * 1024)
    x = rng.standard_normal((2, 16)).astype(np.float32)
    out, stats = ex.run_layer(x, refs, np.arange(2, dtype=np.float32))
    assert np.all(np.isfinite(out))
    assert stats["backend"] == "numpy"
    # After the layer, only the KV cache survives: weights were freed.
    # Cache holds K+V for 2 tokens, post-GQA-repeat (2 heads x 8 dim):
    # 2 tokens * 2 heads * 8 dim * 2 arrays * 4 bytes = 256.
    assert stats["live_bytes_after"] == 2 * 2 * 8 * 2 * 4
    assert stats["live_bytes_after"] < stats["matrices_f32_bytes"]


def test_executor_budget_breach_fails_loud():
    cfg = tiny_config()
    refs = tiny_weights(np.random.RandomState(2))
    ex = ge.GPUExecutor(ge.NumpyBackend(), cfg, vram_budget_bytes=64)
    with pytest.raises(ge.GPUMemoryError):
        ex.run_layer(np.zeros((1, 16), dtype=np.float32), refs,
                     np.array([0], dtype=np.float32))


def test_estimate_layer_vram_110b_class():
    cfg = ge.LayerConfig(hidden=8192, ffn=49152, n_head=64, n_kv=8)
    specs = [
        ((8192, 8192), ge.GGML_Q4_K),   # q
        ((1024, 8192), ge.GGML_Q4_K),   # k
        ((1024, 8192), ge.GGML_Q4_K),   # v
        ((8192, 8192), ge.GGML_Q4_K),   # out
        ((49152, 8192), ge.GGML_Q4_K),  # gate
        ((49152, 8192), ge.GGML_Q4_K),  # up
        ((8192, 49152), ge.GGML_Q4_K),  # down
    ]
    est = ge.GPUExecutor.estimate_layer_vram(specs, cfg)
    # Stored ~0.75 GB; largest single matrix is an MLP matrix
    # (49152x8192 f32 = 1.61 GB), not an attention matrix; whole layer
    # in f32 ~5.4 GB (does NOT fit a 4 GB budget — streaming required).
    assert est["stored_bytes_per_layer"] < 1024**3
    assert est["largest_matrix_f32_bytes"] == 49152 * 8192 * 4
    assert est["f32_bytes_all_matrices"] > 4 * 1024**3
    assert est["kv_bytes_per_token_per_layer"] == 2 * 8 * 128 * 4


def test_read_raw_bytes_needs_exact_store():
    class FakeStore:
        precision_policy = "lossy"

        def tensor_id(self, name):
            return 0

    with pytest.raises(ge.GPUExecutorError):
        ge.read_raw_tensor_bytes(FakeStore(), {"name": "x"})


# ------------------------------------------------------- torch backend
# torch-CPU validates the torch kernels' MATH here. CUDA execution is
# validated only by the Kaggle script; the CUDA test below skips here.


@needs_torch
def test_torch_backend_cpu_layer_matches_reference():
    backend = ge.TorchBackend("cpu")
    assert backend.is_gpu is False
    rng = np.random.RandomState(7)
    cfg = tiny_config()
    refs = tiny_weights(rng)
    mats = {k: np.array(v.array, copy=True) for k, v in refs.items()}
    x = rng.standard_normal((3, 16)).astype(np.float32)
    pos = np.arange(3, dtype=np.float32)
    layer = ge.GPULayer(backend, cfg)
    got = layer.forward(x, refs, pos)
    want = reference_layer(x, mats, cfg, pos)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


@needs_torch
def test_torch_dequant_matches_gguf():
    """Torch Q8_0/Q4_K kernels == gguf dequant (elementwise ops, so
    near-exact; tolerance 1e-6 absorbs only float32 reassociation)."""
    from lacuna.gguf import dequantize as gguf_dq

    backend = ge.TorchBackend("cpu")
    rng = np.random.RandomState(3)
    for ttype, block_bytes, per in ((ge.GGML_Q8_0, 34, 32),
                                    (ge.GGML_Q4_K, 144, 256)):
        raw = rng.randint(0, 256, size=(4, block_bytes), dtype=np.uint8)
        raw[:, 0:2] = np.array([0x00, 0x3C], dtype=np.uint8)
        if ttype == ge.GGML_Q4_K:
            raw[:, 2:4] = np.array([0x00, 0x3C], dtype=np.uint8)
        raw = raw.tobytes()
        got = backend.to_numpy(backend.dequantize(raw, ttype, 4 * per))
        want = gguf_dq(raw, ttype, 4 * per)
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-6)


@needs_torch
def test_torch_backend_cpu_matches_numpy_backend():
    """Same layer, two backends, CPU: outputs must agree tightly.
    This is the parity check the Kaggle script repeats on CUDA."""
    rng = np.random.RandomState(9)
    cfg = tiny_config()
    refs = tiny_weights(rng)
    x = rng.standard_normal((2, 16)).astype(np.float32)
    pos = np.arange(2, dtype=np.float32)
    out_np = ge.GPULayer(ge.NumpyBackend(), cfg).forward(x, refs, pos)
    out_torch = ge.GPULayer(ge.TorchBackend("cpu"), cfg).forward(x, refs, pos)
    np.testing.assert_allclose(out_torch, out_np, rtol=1e-5, atol=1e-5)


@needs_torch
@torch_cuda
def test_torch_cuda_layer_runs():
    """Runs ONLY on a CUDA box (skipped here). Kept in the suite so the
    Kaggle run executes the same test file and this test goes green
    there instead of being skipped."""
    backend = ge.TorchBackend("cuda")
    assert backend.is_gpu is True
    rng = np.random.RandomState(7)
    cfg = tiny_config()
    refs = tiny_weights(rng)
    x = rng.standard_normal((2, 16)).astype(np.float32)
    pos = np.arange(2, dtype=np.float32)
    out = ge.GPULayer(backend, cfg).forward(x, refs, pos)
    assert np.all(np.isfinite(out))
