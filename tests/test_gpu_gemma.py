"""GPU Gemma-4 path (``GPUGemmaModel`` / ``GemmaGPULayer``) vs the CPU
reference (``PagedGemma4``).

Synthetic weights only — no model file is downloaded. The numpy
backend runs on CPU here; the torch-CUDA path is a Kaggle check
(parent launches it), not something these tests can exercise.

What these tests pin (each against the CPU path on the SAME tiny
GGUF, so a divergence is a GPU-path bug, not a weights difference):
  * Sliding-window mask parity: long prompts that force repeated
    cache trims still give matching logits and the same greedy
    choice as the CPU path.
  * Global V = K parity: a global layer without ``attn_v.weight``
    (V = K) matches the CPU path, and so does a global layer that
    carries an explicit V.
  * Global-RoPE parity under stored factors: global RoPE is the
    ggml formula (full head dim, NEOX pairs, angles divided by
    the file's rope_freqs tensor) on both paths — there is no
    partial-rotary knob anymore.
  * GELU-tanh parity: the backend-generic tanh (via exp) matches
    the CPU path's ``np.tanh`` formula.
  * Generate ids == CPU Gemma ids, and Engine dispatch: a Gemma
    file served with ``gpu="numpy"`` lands in mode ``lacuna-gpu``
    with text identical to the GPU-direct and CPU paged paths.
  * Per-layer KV isolation: sliding caches trim to the window
    while global caches grow unbounded, and no layer ever sees
    another layer's cache.
  * VRAM budget breaches raise ``GPUMemoryError`` (never silent
    spill, downcast, or dropped weights).

Tolerance note: GPU and CPU attention run in different layouts
(batched-head matmul vs the CPU path's einsum), so logits are
compared at 2e-3 absolute with an EXACT greedy-choice requirement —
the same near-tie discipline as the dense GPU path. A wide-margin
greedy flip would fail the argmax check, not hide in the tolerance.
"""

import os
import struct

import numpy as np
import pytest

from lacuna.engine import Engine
from lacuna.gemma import PagedGemma4
from lacuna.gpu_executor import (
    GemmaConfig,
    GemmaGPULayer,
    GPUExecutorError,
    GPUGemmaModel,
    GPUMemoryError,
    NumpyBackend,
    WeightRef,
)

HIDDEN, FFN = 16, 24
N_HEAD, N_KV, N_KV_GLOBAL = 4, 2, 1
HEAD_DIM, HEAD_DIM_GLOBAL = 4, 8
WINDOW = 3
EPS = 1e-6


# ------------------------------------------------------------ tiny GGUF

def _s(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _kv_str(key: str, val: str) -> bytes:
    return _s(key) + struct.pack("<I", 8) + _s(val)


def _kv_u32(key: str, val: int) -> bytes:
    return _s(key) + struct.pack("<I", 4) + struct.pack("<I", val)


def _kv_f32(key: str, val: float) -> bytes:
    return _s(key) + struct.pack("<I", 6) + struct.pack("<f", val)


def _kv_str_array(key: str, vals) -> bytes:
    out = _s(key) + struct.pack("<I", 9) + struct.pack("<I", 8)
    out += struct.pack("<Q", len(vals))
    for v in vals:
        out += _s(v)
    return out


def _kv_f32_array(key: str, vals) -> bytes:
    out = _s(key) + struct.pack("<I", 9) + struct.pack("<I", 6)
    out += struct.pack("<Q", len(vals))
    for v in vals:
        out += struct.pack("<f", v)
    return out


def _kv_i32_array(key: str, vals) -> bytes:
    out = _s(key) + struct.pack("<I", 9) + struct.pack("<I", 5)
    out += struct.pack("<Q", len(vals))
    for v in vals:
        out += struct.pack("<i", v)
    return out


def _kv_u32_array(key: str, vals) -> bytes:
    out = _s(key) + struct.pack("<I", 9) + struct.pack("<I", 4)
    out += struct.pack("<Q", len(vals))
    for v in vals:
        out += struct.pack("<I", v)
    return out


SPM_TOKENS = ["<pad>", "<eos>", "<bos>", "a", "b", "▁", "ab", "<0x41>"]
SPM_SCORES = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0]
SPM_TYPES = [3, 3, 3, 1, 1, 1, 1, 6]


def build_tiny_gemma_gguf(path: str, n_layer: int = 6,
                          global_v: bool = False, kv_array_vals=None,
                          with_global_kv: bool = True,
                          tokenizer_model: str = "llama",
                          freq_base_swa: float | None = None,
                          with_gemma4_block_tensors: bool = True,
                          rope_freqs_vals=None,
                          with_rope_freqs: bool = True) -> str:
    """Minimal valid gemma4 GGUF: hidden 16, sliding 4 Q / 2 KV x
    dim 4, global 4 Q / 1 KV x dim 8, FFN 24, window 3, 8-token SPM
    toy vocab, tied embeddings. Layer 5 is global (5:1 rule)."""
    rng = np.random.RandomState(1)

    def w(shape):
        return (rng.standard_normal(shape) * 0.05).astype("<f4")

    tensors = [
        ("token_embd.weight", [16, 8], w((8, 16))),
        ("output_norm.weight", [16], w((16,))),
    ]
    if with_rope_freqs:
        # Global head_dim is 8 in this fixture, so the stored
        # rope_freqs tensor has 8 // 2 = 4 entries.
        ff = ([1.0, 1.0, 1.0, 1.0] if rope_freqs_vals is None
              else list(rope_freqs_vals))
        tensors.append(("rope_freqs.weight", [4],
                        np.array(ff, dtype="<f4")))
    for li in range(n_layer):
        is_global = (li % 6 == 5)
        base = f"blk.{li}."
        tensors.append((base + "attn_norm.weight", [16], w((16,))))
        tensors.append((base + "ffn_norm.weight", [16], w((16,))))
        if with_gemma4_block_tensors:
            # Gemma-4 Peri-LN block tensors (see test_gemma.py).
            tensors.append((base + "post_attention_norm.weight",
                            [16], w((16,))))
            tensors.append((base + "post_ffw_norm.weight",
                            [16], w((16,))))
            tensors.append((base + "layer_output_scale.weight", [1],
                            np.array([0.9], dtype="<f4")))
            hd = HEAD_DIM_GLOBAL if is_global else HEAD_DIM
            tensors.append((base + "attn_q_norm.weight", [hd], w((hd,))))
            tensors.append((base + "attn_k_norm.weight", [hd], w((hd,))))
        if is_global:
            tensors.append((base + "attn_q.weight", [16, 32], w((32, 16))))
            tensors.append((base + "attn_k.weight", [16, 8], w((8, 16))))
            if global_v:
                tensors.append((base + "attn_v.weight", [16, 8], w((8, 16))))
            tensors.append((base + "attn_output.weight", [32, 16], w((16, 32))))
        else:
            tensors.append((base + "attn_q.weight", [16, 16], w((16, 16))))
            tensors.append((base + "attn_k.weight", [16, 8], w((8, 16))))
            tensors.append((base + "attn_v.weight", [16, 8], w((8, 16))))
            tensors.append((base + "attn_output.weight", [16, 16], w((16, 16))))
        tensors.append((base + "ffn_gate.weight", [16, 24], w((24, 16))))
        tensors.append((base + "ffn_up.weight", [16, 24], w((24, 16))))
        tensors.append((base + "ffn_down.weight", [24, 16], w((16, 24))))
    kvs = [
        _kv_str("general.architecture", "gemma4"),
        _kv_u32("gemma4.block_count", n_layer),
        _kv_u32("gemma4.embedding_length", 16),
        _kv_u32("gemma4.feed_forward_length", 24),
        _kv_u32("gemma4.attention.head_count", 4),
        _kv_f32("gemma4.attention.layer_norm_rms_epsilon", 1e-6),
        _kv_u32("gemma4.attention.sliding_window", WINDOW),
        _kv_u32("gemma4.attention.head_dim", 4),
        _kv_f32("gemma4.rope.freq_base", 10000.0),
        _kv_u32("gemma4.attention.global_head_dim", 8),
        _kv_f32("gemma4.rope.freq_base_global", 1000000.0),
        _kv_str("tokenizer.ggml.model", tokenizer_model),
        _kv_str_array("tokenizer.ggml.tokens", SPM_TOKENS),
        _kv_f32_array("tokenizer.ggml.scores", SPM_SCORES),
        _kv_i32_array("tokenizer.ggml.token_type", SPM_TYPES),
        _kv_u32("tokenizer.ggml.bos_token_id", 2),
        _kv_u32("tokenizer.ggml.eos_token_id", 1),
    ]
    if kv_array_vals is not None:
        kvs.append(_kv_u32_array("gemma4.attention.head_count_kv",
                                 kv_array_vals))
    else:
        kvs.append(_kv_u32("gemma4.attention.head_count_kv", 2))
    if with_global_kv:
        kvs.append(_kv_u32("gemma4.attention.head_count_kv_global", 1))
    if freq_base_swa is not None:
        kvs.append(_kv_f32("gemma4.rope.freq_base_swa", freq_base_swa))
    header = b"GGUF" + struct.pack("<I", 3)
    header += struct.pack("<Q", len(tensors)) + struct.pack("<Q", len(kvs))
    header += b"".join(kvs)
    infos = b""
    offset = 0
    blobs = []
    for name, dims, arr in tensors:
        infos += _s(name) + struct.pack("<I", len(dims))
        for d in dims:
            infos += struct.pack("<Q", d)
        infos += struct.pack("<I", 0)  # F32
        infos += struct.pack("<Q", offset)
        blobs.append(arr.tobytes())
        offset += len(arr.tobytes())
    head = header + infos
    pad = (-len(head)) % 32
    with open(path, "wb") as f:
        f.write(head + b"\x00" * pad + b"".join(blobs))
    return path


def _cpu_gpu(tmp_path, global_v=False, **kw):
    gguf = build_tiny_gemma_gguf(
        os.path.join(tmp_path, "tiny.gguf"), global_v=global_v, **kw)
    cpu = PagedGemma4(gguf, hot_budget_bytes=1024 * 1024)
    gpu = GPUGemmaModel.from_gguf(
        gguf, backend=NumpyBackend(), hot_budget_bytes=1024 * 1024)
    return cpu, gpu


def _assert_parity(cpu_logits, gpu_logits):
    assert gpu_logits.shape == cpu_logits.shape
    diff = float(np.max(np.abs(gpu_logits - cpu_logits)))
    assert diff < 2e-3, f"GPU vs CPU logits max diff {diff}"
    assert int(np.argmax(gpu_logits)) == int(np.argmax(cpu_logits))


# ------------------------------------------------------------ the tests

def test_gelu_tanh_parity():
    backend = NumpyBackend()
    layer = GemmaGPULayer(backend, GemmaConfig(
        hidden=HIDDEN, ffn=FFN, n_head=N_HEAD, n_kv=N_KV,
        n_kv_global=N_KV_GLOBAL, layer_types=["sliding_attention"]))
    rng = np.random.RandomState(0)
    x = rng.standard_normal(257).astype(np.float32) * 3.0
    got = backend.to_numpy(layer._gelu_tanh(backend.asarray(x)))
    want = PagedGemma4._gelu_tanh(x)
    assert float(np.max(np.abs(got - want))) < 1e-6


def test_forward_matches_cpu_gemma():
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        cpu, gpu = _cpu_gpu(td)
        try:
            for ids in ([3, 4, 6], [3, 4, 3, 4, 6, 3]):
                cpu.cache = {}
                gpu.reset_cache()
                _assert_parity(cpu.forward(ids, 0), gpu.forward(ids, 0))
        finally:
            cpu.close()
            gpu.close()


def test_from_gguf_per_layer_kv_array_and_gemma4_tag():
    """GPU from_gguf on the real-file shapes: per-layer head_count_kv
    array, no global-KV scalar key, tokenizer model 'gemma4'. The
    resolved config must match the CPU loader's, and forward parity
    must hold against a CPU model built from the same file."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        gguf = build_tiny_gemma_gguf(
            os.path.join(td, "tiny.gguf"),
            kv_array_vals=[2, 2, 2, 2, 2, 1], with_global_kv=False,
            tokenizer_model="gemma4")
        cpu = PagedGemma4(gguf, hot_budget_bytes=1024 * 1024)
        gpu = GPUGemmaModel.from_gguf(
            gguf, backend=NumpyBackend(), hot_budget_bytes=1024 * 1024)
        try:
            assert gpu.config.n_kv == 2
            assert gpu.config.n_kv_global == 1
            assert gpu.config.head_dim == 4
            assert gpu.config.head_dim_global == 8
            for ids in ([3, 4, 6], [3, 4, 3, 4, 6, 3]):
                cpu.cache = {}
                gpu.reset_cache()
                _assert_parity(cpu.forward(ids, 0), gpu.forward(ids, 0))
        finally:
            cpu.close()
            gpu.close()


def test_from_gguf_rope_freq_base_swa():
    """GPU from_gguf: freq_base_swa wins for sliding RoPE only."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        gguf = build_tiny_gemma_gguf(
            os.path.join(td, "tiny.gguf"), freq_base_swa=12345.0)
        gpu = GPUGemmaModel.from_gguf(
            gguf, backend=NumpyBackend(), hot_budget_bytes=1024 * 1024)
        try:
            assert gpu.config.rope_theta == 12345.0
            assert gpu.config.rope_theta_global == 1000000.0
        finally:
            gpu.close()


def test_sliding_mask_parity_vs_cpu_long_prompt():
    """Window 3 with an 8-token prompt: the sliding cache trims
    repeatedly, and every trim must keep GPU == CPU."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        cpu, gpu = _cpu_gpu(td)
        try:
            ids = [3, 4, 6, 3, 4, 6, 3, 4]
            cpu.cache = {}
            gpu.reset_cache()
            _assert_parity(cpu.forward(ids, 0), gpu.forward(ids, 0))
            # Decode parity after a trimmed prefill: step both models
            # with the same next token and compare the following logits.
            cpu.cache = {}
            gpu.reset_cache()
            l_cpu = cpu.forward(ids, 0)
            l_gpu = gpu.forward(ids, 0)
            nxt = int(np.argmax(l_cpu))
            assert nxt == int(np.argmax(l_gpu))
            _assert_parity(cpu.forward([nxt], len(ids)),
                           gpu.forward([nxt], len(ids)))
        finally:
            cpu.close()
            gpu.close()


def test_global_v_equals_k_and_explicit_v():
    import tempfile

    for global_v in (False, True):
        with tempfile.TemporaryDirectory() as td:
            cpu, gpu = _cpu_gpu(td, global_v=global_v)
            try:
                cpu.cache = {}
                gpu.reset_cache()
                _assert_parity(cpu.forward([3, 4, 6, 3], 0),
                               gpu.forward([3, 4, 6, 3], 0))
            finally:
                cpu.close()
                gpu.close()


def test_global_rope_factors_gpu_matches_formula():
    # GPU global _rope under non-trivial stored factors matches the
    # ggml formula (full head dim, NEOX pairs (i, i+4), angles
    # divided by rope_freqs). The old partial rotation FAILS this:
    # it rotated only the first 2 dims and ignored the factors.
    # CPU/GPU forward parity on the same factors model is pinned
    # too, so both paths stay identical.
    import tempfile

    ff = [1.0, 2.0, 0.5, 4.0]
    backend = NumpyBackend()
    layer = GemmaGPULayer(backend, GemmaConfig(
        hidden=HIDDEN, ffn=FFN, n_head=N_HEAD, n_kv=N_KV,
        n_kv_global=N_KV_GLOBAL, head_dim=HEAD_DIM,
        head_dim_global=HEAD_DIM_GLOBAL,
        rope_freqs=np.array(ff, dtype=np.float32),
        layer_types=["full_attention"]))
    t = np.arange(8, dtype=np.float32).reshape(1, 1, 8) + 1.0
    got = backend.to_numpy(layer._rope(
        backend.asarray(t),
        backend.asarray(np.array([5], dtype=np.float32)), True))
    inv = 1000000.0 ** (-np.arange(0, 8, 2, dtype=np.float32) / 8)
    ang = 5 * inv / np.array(ff, dtype=np.float32)
    c, s = np.cos(ang), np.sin(ang)
    x0, x1 = t[0, 0, :4], t[0, 0, 4:]
    want = np.concatenate([x0 * c - x1 * s, x0 * s + x1 * c])
    assert np.allclose(got[0, 0], want, atol=1e-6)
    with tempfile.TemporaryDirectory() as td:
        cpu, gpu = _cpu_gpu(td, rope_freqs_vals=ff)
        try:
            cpu.cache = {}
            gpu.reset_cache()
            _assert_parity(cpu.forward([3, 4, 6], 0),
                           gpu.forward([3, 4, 6], 0))
        finally:
            cpu.close()
            gpu.close()


def test_gpu_global_model_without_rope_freqs_fails_loud():
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        gguf = build_tiny_gemma_gguf(
            os.path.join(td, "tiny.gguf"), with_rope_freqs=False)
        with pytest.raises(GPUExecutorError, match="rope_freqs"):
            GPUGemmaModel.from_gguf(
                gguf, backend=NumpyBackend(),
                hot_budget_bytes=1024 * 1024)


def test_generate_ids_match_cpu_gemma():
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        cpu, gpu = _cpu_gpu(td)
        try:
            cpu_text, cpu_ids = cpu.generate("ab", max_tokens=6)
            gpu_text, gpu_ids, stats = gpu.generate("ab", max_tokens=6)
            assert gpu_ids == cpu_ids
            assert gpu_text == cpu_text
            assert stats["within_300s_bound"] is True
            assert stats["ttft_seconds"] <= stats["total_seconds"]
        finally:
            cpu.close()
            gpu.close()


def test_per_layer_kv_isolation_and_trim():
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        cpu, gpu = _cpu_gpu(td)
        try:
            # Prompt tokens + generated tokens = total positions
            # ("ababab" SPM-encodes to 3 "ab" pieces, not 6 chars).
            _, _, stats = gpu.generate("ababab", max_tokens=6)
            total = stats["prompt_tokens"] + stats["tokens_generated"]
            sliding = gpu._caches[0]
            glob = gpu._caches[5]
            # Sliding cache trims to the window; global grows unbounded.
            assert sliding["k"].shape[0] == WINDOW
            assert sliding["start"] > 0
            assert glob["k"].shape[0] == total
            assert glob["start"] == 0
            # Caches are distinct objects with distinct shapes where
            # the layer geometry differs (sliding KV width 2*4=8 vs
            # global KV width 1*8=8 after repeat both are 4 heads...
            # so assert object identity + independent mutation).
            assert sliding["k"] is not glob["k"]
            before = glob["k"].shape[0]
            gpu.forward([3], total)
            assert glob["k"].shape[0] == before + 1
            assert sliding["k"].shape[0] == WINDOW  # still trimmed
        finally:
            cpu.close()
            gpu.close()


def test_vram_budget_breach_raises():
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        cpu, gpu = _cpu_gpu(td)
        try:
            gpu.vram_budget = 1  # 1 byte: nothing can run
            with pytest.raises(GPUMemoryError):
                gpu.forward([3, 4], 0)
        finally:
            cpu.close()
            gpu.close()


def test_engine_dispatch_gemma(tmp_path, monkeypatch):
    # Keep the test hermetic: no llama-server binary, so the GPU
    # path is the only real-inference route Engine can take.
    monkeypatch.setattr("lacuna.engine._find_llama_server",
                        lambda *a, **k: None)
    gguf = build_tiny_gemma_gguf(os.path.join(tmp_path, "tiny.gguf"))
    cpu = PagedGemma4(gguf, hot_budget_bytes=1024 * 1024)
    try:
        cpu_text, _ = cpu.generate("ab", max_tokens=6)
    finally:
        cpu.close()
    eng = Engine(gguf, gpu="numpy")
    try:
        assert eng.mode == "lacuna-gpu"
        assert eng.gpu_backend == "numpy:cpu"
        assert eng.generate("ab", max_tokens=6) == cpu_text
        assert eng.gpu_stats is not None
        assert eng.gpu_stats["within_300s_bound"] is True
    finally:
        eng.close()


# ------------------------------------------------- gemma-4 block correction ---


def test_missing_gemma4_block_tensors_fail_loud(tmp_path):
    """GPU mirror of the CPU fail-loud test: a Gemma-4 file without
    the Peri-LN block tensors must raise in forward(), never run
    with silently-wrong block math."""
    gguf = build_tiny_gemma_gguf(
        os.path.join(tmp_path, "tiny.gguf"),
        with_gemma4_block_tensors=False)
    gpu = GPUGemmaModel.from_gguf(
        gguf, backend=NumpyBackend(), hot_budget_bytes=1024 * 1024)
    try:
        with pytest.raises(GPUExecutorError) as exc:
            gpu.forward([3, 4, 6], 0)
        assert "attn_q_norm" in str(exc.value)
    finally:
        gpu.close()


def test_layer0_magnitude_stays_sane_at_realistic_norms():
    """Layer-0 magnitude regression test for the 2026-10-26 blowup.

    Direct GemmaGPULayer call (numpy backend) with REAL-FILE
    magnitudes: pre-norms RMS ~8 / ~3.6 (blk.0/1 attn/ffn norms),
    sparse post-norms (most dims ~0.02, a few large, RMS ~6.5 /
    ~2.2), layer_output_scale 0.0894 (blk.0), QK-norm constants
    1.0234375 / 0.12207031, input RMS 1.28 (scaled embed). Matrix
    gains are fan-in compensated: the tiny layer's hidden is 16,
    so matrices use RMS 0.375 to reproduce the real layer's gain
    (sqrt(5376) x 0.02 ~= 1.47 ~= sqrt(16) x 0.375). With the OLD
    math (post-norms + scale + QK-norm + V-norm ignored) this
    layer outputs RMS ~58; with the corrected block math
    (layer-wise scale per llama.cpp gemma4.cpp) it outputs
    RMS 0.44 (measured). The bound 10 separates the two regimes
    by 10x on each side."""
    backend = NumpyBackend()
    layer = GemmaGPULayer(backend, GemmaConfig(
        hidden=HIDDEN, ffn=FFN, n_head=N_HEAD, n_kv=N_KV,
        n_kv_global=N_KV_GLOBAL, head_dim=HEAD_DIM,
        head_dim_global=HEAD_DIM_GLOBAL, window=16,
        layer_types=["sliding_attention"]))
    layer._dims = (N_HEAD, N_KV, HEAD_DIM, False)
    rng = np.random.RandomState(5)

    def mat(shape):
        return (rng.standard_normal(shape) * 0.375).astype(np.float32)

    def sparse_post(rms_big, n_big):
        arr = np.full(HIDDEN, 0.02, dtype=np.float32)
        arr[:n_big] = rng.normal(0, rms_big, n_big).astype(np.float32)
        return arr

    refs = {
        "attn_norm.weight": WeightRef(
            shape=(HIDDEN,),
            array=rng.normal(5, 6, HIDDEN).astype(np.float32)),
        "attn_q.weight": WeightRef(shape=(16, 16), array=mat((16, 16))),
        "attn_k.weight": WeightRef(shape=(8, 16), array=mat((8, 16))),
        "attn_v.weight": WeightRef(shape=(8, 16), array=mat((8, 16))),
        "attn_q_norm.weight": WeightRef(
            shape=(HEAD_DIM,),
            array=np.full(HEAD_DIM, 1.0234375, dtype=np.float32)),
        "attn_k_norm.weight": WeightRef(
            shape=(HEAD_DIM,),
            array=np.full(HEAD_DIM, 0.12207031, dtype=np.float32)),
        "attn_output.weight": WeightRef(shape=(16, 16), array=mat((16, 16))),
        "post_attention_norm.weight": WeightRef(
            shape=(HIDDEN,), array=sparse_post(15.0, 3)),
        "layer_output_scale.weight": WeightRef(
            shape=(1,), array=np.array([0.0894], dtype=np.float32)),
        "ffn_norm.weight": WeightRef(
            shape=(HIDDEN,),
            array=rng.normal(3, 2, HIDDEN).astype(np.float32)),
        "ffn_gate.weight": WeightRef(shape=(24, 16), array=mat((24, 16))),
        "ffn_up.weight": WeightRef(shape=(24, 16), array=mat((24, 16))),
        "ffn_down.weight": WeightRef(shape=(16, 24), array=mat((16, 24))),
        "post_ffw_norm.weight": WeightRef(
            shape=(HIDDEN,), array=sparse_post(5.0, 3)),
    }
    x = (rng.standard_normal((5, HIDDEN)) * 1.28).astype(np.float32)
    out = layer.forward(x, refs, np.arange(5, dtype=np.float32), {})
    rms = float(np.sqrt(np.mean(out.astype(np.float64) ** 2)))
    assert rms < 10.0, f"layer-0 output RMS {rms} (blowup regime)"
