"""Full-model GPU generation (``GPUModel``) and ``Engine`` GPU mode.

Synthetic weights only — the local 0.5B model files were deleted, so
no test here touches a real model file. The numpy backend runs on CPU
here; the torch-CUDA path is exercised by the Kaggle script section
added to ``kaggle_gpu/gpu_executor_test.py`` (the parent launches it).

What these tests pin:
  * GPUModel forward == an independently written numpy reference
    (einsum attention, separate code path from ``GPULayer``'s batched
    matmul) to 1e-4, and greedy generation ids identical to a
    from-scratch reference rollout.
  * End-response timing: ``generate`` reports time-to-first-token
    (prefill) and total time for the full response, plus
    ``within_300s_bound``. On synthetic weights these complete in
    milliseconds, so the bound assertion here is STRUCTURAL — it
    proves the fields exist and the check runs. It does NOT prove a
    real model meets 300 s; that is the headline run's job, using
    these same stats fields.
  * Speculative hook: after a verify forward over draft tokens,
    ``truncate_cache`` rolls KV back to the accepted prefix, and the
    next token's logits match a fresh prefill over the accepted
    sequence (so rejected drafts cannot leak into later tokens).
  * Per-layer KV isolation: one executor per layer, caches differ.
  * Paged source == direct source: the same weights served through a
    fake exact-policy store (raw bytes + row slices, never whole
    matrices) generate identical ids.
  * BF16 decodes natively (numpy + torch when present) — the old
    "BF16 raises" expectation is retired now that gguf.py decodes
    type 30; VRAM budget breaches still raise loud.
  * Engine end-to-end: a synthetic GGUF file written by this test,
    converted to .lac, served by ``Engine(gguf, gpu="numpy")`` in mode
    "lacuna-gpu", with text identical to GPUModel-direct and to the
    CPU paged executor.
"""

import struct

import numpy as np
import pytest

from lacuna.gpu_executor import (
    GPUModel,
    GPUMemoryError,
    LayerConfig,
    NumpyBackend,
    WeightRef,
)
from lacuna.gguf import GGUFError

HIDDEN, FFN, N_HEAD, N_KV, N_LAYER = 32, 64, 4, 2, 2
HEAD_DIM = HIDDEN // N_HEAD
VOCAB = list("abcdefgh")  # 8 single-char tokens; eos id = 8 (unused)
EPS, THETA = 1e-5, 10000.0


# ------------------------------------------------------------ helpers

class _StubTokenizer:
    """encode maps each char to its vocab index; decode reverses it."""

    def __init__(self, vocab):
        self._v = list(vocab)

    def encode(self, text):
        return [self._v.index(c) for c in text]

    def decode(self, ids):
        return "".join(self._v[i] for i in ids)


def _config():
    return LayerConfig(hidden=HIDDEN, ffn=FFN, n_head=N_HEAD, n_kv=N_KV,
                       eps=EPS, rope_theta=THETA)


def _weights(rng):
    """Synthetic F32 weights: (embed, head, out_norm, layers)."""
    def mat(n_out, n_in):
        return rng.normal(0.0, 0.1, size=(n_out, n_in)).astype(np.float32)

    def w(arr):
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        return WeightRef(shape=arr.shape, array=arr)

    def bias(n):
        return w(rng.normal(0, 0.01, n).astype(np.float32))

    embed = mat(len(VOCAB), HIDDEN)
    head = mat(len(VOCAB), HIDDEN)
    out_norm = np.ones(HIDDEN, dtype=np.float32)
    layers = []
    for _ in range(N_LAYER):
        layers.append({
            "attn_norm.weight": w(np.ones(HIDDEN, dtype=np.float32)),
            "attn_q.weight": w(mat(HIDDEN, HIDDEN)),
            "attn_q.bias": bias(HIDDEN),
            "attn_k.weight": w(mat(N_KV * HEAD_DIM, HIDDEN)),
            "attn_k.bias": bias(N_KV * HEAD_DIM),
            "attn_v.weight": w(mat(N_KV * HEAD_DIM, HIDDEN)),
            "attn_v.bias": bias(N_KV * HEAD_DIM),
            "attn_output.weight": w(mat(HIDDEN, HIDDEN)),
            "ffn_norm.weight": w(np.ones(HIDDEN, dtype=np.float32)),
            "ffn_gate.weight": w(mat(FFN, HIDDEN)),
            "ffn_up.weight": w(mat(FFN, HIDDEN)),
            "ffn_down.weight": w(mat(HIDDEN, FFN)),
        })
    return embed, head, out_norm, layers


def _arr(ref):
    return ref.array if ref.array is not None else None


def _layer_arrays(layer):
    return {k: _arr(v) for k, v in layer.items()}


def _model(rng_seed=0, **kw):
    rng = np.random.default_rng(rng_seed)
    embed, head, out_norm, layers = _weights(rng)
    model = GPUModel.from_arrays(
        NumpyBackend(), _config(), embed, head, out_norm, layers,
        _StubTokenizer(VOCAB), eos=None, **kw)
    return model, (embed, head, out_norm, layers)


# ------------------------------------------- independent numpy reference

def _rmsnorm(x, w, eps):
    return (x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True)
                        + np.float32(eps))) * w


def _rope(x, positions, theta):
    T, H, D = x.shape
    h = D // 2
    inv = 1.0 / (theta ** (np.arange(0, D, 2, dtype=np.float32) / D))
    ang = positions[:, None] * inv[None, :]
    cos, sin = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    x0, x1 = x[..., :h], x[..., h:]
    return np.concatenate([x0 * cos - x1 * sin,
                           x0 * sin + x1 * cos], axis=-1)


def _ref_layer(x, mats, cfg, positions):
    T = x.shape[0]
    h = _rmsnorm(x, mats["attn_norm.weight"], cfg.eps)
    q = h @ mats["attn_q.weight"].T + mats["attn_q.bias"]
    k = h @ mats["attn_k.weight"].T + mats["attn_k.bias"]
    v = h @ mats["attn_v.weight"].T + mats["attn_v.bias"]
    q = _rope(q.reshape(T, cfg.n_head, cfg.head_dim), positions, cfg.rope_theta)
    k = _rope(k.reshape(T, cfg.n_kv, cfg.head_dim), positions, cfg.rope_theta)
    v = v.reshape(T, cfg.n_kv, cfg.head_dim)
    k_rep = np.repeat(k, cfg.group, axis=1)
    v_rep = np.repeat(v, cfg.group, axis=1)
    scores = (np.einsum("htd,hsd->hts", q.transpose(1, 0, 2),
                        k_rep.transpose(1, 0, 2))
              / np.float32(np.sqrt(cfg.head_dim)))
    row = positions.astype(np.int64)[:, None]
    col = np.arange(k_rep.shape[0])[None, :]
    scores = np.where(col <= row, scores, np.float32(-1e30))
    scores = scores - scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    o = np.einsum("hts,hsd->htd", probs, v_rep.transpose(1, 0, 2))
    x = x + o.transpose(1, 0, 2).reshape(T, cfg.hidden) @ mats["attn_output.weight"].T
    h2 = _rmsnorm(x, mats["ffn_norm.weight"], cfg.eps)
    gate = h2 @ mats["ffn_gate.weight"].T
    up = h2 @ mats["ffn_up.weight"].T
    silu = gate / (np.float32(1.0) + np.exp(-gate))
    return x + (silu * up) @ mats["ffn_down.weight"].T


def _ref_forward(weights, ids):
    embed, head, out_norm, layers = weights
    cfg = _config()
    x = embed[np.asarray(ids, dtype=int)]
    positions = np.arange(len(ids), dtype=np.float32)
    for layer in layers:
        x = _ref_layer(x, _layer_arrays(layer), cfg, positions)
    last = _rmsnorm(x[-1:], out_norm, cfg.eps)
    return (last @ head.T)[0]


def _ref_generate(weights, prompt_ids, max_tokens):
    ids = list(prompt_ids)
    out = []
    for _ in range(max_tokens):
        nxt = int(np.argmax(_ref_forward(weights, ids)))
        out.append(nxt)
        ids.append(nxt)
    return out


# ------------------------------------------------------------ the tests

def test_forward_matches_reference():
    model, weights = _model()
    ids = [0, 1, 2, 3]
    got = model.forward(ids, 0)
    want = _ref_forward(weights, ids)
    assert got.shape == (len(VOCAB),)
    assert np.max(np.abs(got - want)) < 1e-4


def test_generate_matches_reference_and_reports_timing():
    model, weights = _model()
    text, ids, stats = model.generate("ab", max_tokens=32)
    assert ids == _ref_generate(weights, [0, 1], 32)
    assert text == "".join(VOCAB[i] for i in ids)
    assert stats["prompt_tokens"] == 2
    assert stats["tokens_generated"] == 32
    # End-response timing: TTFT (prefill) must be positive, no larger
    # than the total, and the 300 s bound check must be present and
    # True here. On synthetic weights this is structural only — it
    # proves the fields and the check exist; a real model meeting
    # 300 s is the headline run's measurement, using these fields.
    assert stats["ttft_seconds"] > 0
    assert stats["ttft_seconds"] <= stats["total_seconds"]
    assert stats["total_seconds"] < 300.0
    assert stats["within_300s_bound"] is True
    assert stats["tok_per_s_overall"] > 0
    assert abs(stats["tok_per_s_overall"]
               - 32 / stats["total_seconds"]) < 1e-6
    assert stats["backend"] == "numpy"


def test_eos_stops_generation():
    model, _ = _model()
    logits = model.forward([0, 1], 0)
    model.eos = int(np.argmax(logits))  # first greedy token IS eos
    text, ids, stats = model.generate("ab", max_tokens=32)
    assert ids == [] and text == ""
    assert stats["tokens_generated"] == 0


def test_truncate_cache_matches_fresh_prefill():
    """Speculative-verify rollback: prefix P, drafts D = accepted A +
    rejected R1, R2; truncate to P+A; the next token's logits must
    equal a fresh prefill over P+A+[x] (rejected drafts must not
    leak into later tokens, and positions must stay absolute)."""
    model, _ = _model()
    p, a = [0, 1], [2, 3]
    rejected = [4, 5]
    probe = 6
    model.reset_cache()
    model.forward(p, 0)
    model.forward(a + rejected, len(p))
    model.truncate_cache(len(p) + len(a))
    logits_inc = model.forward([probe], len(p) + len(a))
    # The cache really did roll back: exactly p+a+1 rows per layer
    # (accepted prefix + probe), not p+|D|+1 (drafts included).
    for ex in model._executors:
        assert ex.cache["k"].shape[0] == len(p) + len(a) + 1
    model.reset_cache()
    logits_fresh = model.forward(p + a + [probe], 0)
    assert np.max(np.abs(logits_inc - logits_fresh)) < 1e-4


def test_per_layer_kv_caches_isolated():
    model, _ = _model()
    model.forward([0, 1, 2], 0)
    assert len(model._executors) == N_LAYER
    caches = [ex.cache for ex in model._executors]
    for c in caches:
        assert c["k"].shape == (3, N_HEAD, HEAD_DIM)
        assert c["v"].shape == (3, N_HEAD, HEAD_DIM)
    # Different layers, different weights -> different K/V. (This is
    # the bug class a shared cache would cause: layer 1 concatenating
    # layer 0's keys.)
    assert not np.array_equal(caches[0]["k"], caches[1]["k"])
    assert not np.array_equal(caches[0]["v"], caches[1]["v"])


class _FakeStore:
    """Minimal exact-policy store: F32 blobs split into 64-byte
    pages. Serves tensor_id / tensor_pages / page_shape / get_page
    exactly as the real PagedWeightStore's readers expect."""

    precision_policy = "exact"

    def __init__(self, blobs):
        self._infos = {}
        self._pages = []
        self._tensor_pages = {}
        self._name_to_id = {}
        for idx, (name, dims, blob) in enumerate(blobs):
            self._infos[name] = {"name": name, "dims": dims, "type": 0}
            pids = []
            for off in range(0, len(blob), 64):
                self._pages.append(blob[off:off + 64])
                pids.append(len(self._pages) - 1)
            self._tensor_pages[idx] = pids
            self._name_to_id[name] = idx
        self.stats = {"hits": 0, "faults": 0, "bytes_read": 0}

    def tensor_id(self, name):
        return self._name_to_id.get(name)

    def tensor_pages(self, tid):
        return [{"page_id": p} for p in self._tensor_pages[tid]]

    def page_shape(self, pid):
        return (0, len(self._pages[pid]))

    def get_page(self, pid):
        return np.frombuffer(self._pages[pid], dtype=np.uint8)

    def close(self):
        pass


def _paged_model(rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    embed, head, out_norm, layers = _weights(rng)
    blobs = []
    by_name = {}

    def add(name, arr):
        dims = [arr.shape[1], arr.shape[0]] if arr.ndim == 2 else [arr.shape[0]]
        blobs.append((name, dims, np.ascontiguousarray(arr, dtype=np.float32).tobytes()))
        by_name[name] = {"name": name, "dims": dims, "type": 0}

    add("token_embd.weight", embed)
    add("output.weight", head)
    add("output_norm.weight", out_norm)
    for li, layer in enumerate(layers):
        for key, ref in layer.items():
            add(f"blk.{li}.{key}", ref.array)
    store = _FakeStore(blobs)
    model = GPUModel.from_store(
        NumpyBackend(), _config(), N_LAYER, store, by_name,
        _StubTokenizer(VOCAB), eos=None)
    return model, (embed, head, out_norm, layers)


def test_paged_source_matches_direct():
    paged, _ = _paged_model()
    direct, _ = _model()
    # Same weights (same seed): paged path (raw bytes + row slices)
    # must generate identical ids to the in-memory direct path, and
    # embed rows with gaps must slice correctly (not just runs).
    _, ids_paged, _ = paged.generate("ab", max_tokens=16)
    _, ids_direct, _ = direct.generate("ab", max_tokens=16)
    assert ids_paged == ids_direct
    assert np.max(np.abs(paged.forward([5, 3, 5, 1], 0)
                          - direct.forward([5, 3, 5, 1], 0))) < 1e-4


def test_bf16_decodes_native():
    """BF16 (GGML type 30) decodes natively on both backends — the
    Qwen3-235B native file. numpy path uses gguf.dequant_bf16 (a
    widening shift); the torch path views bfloat16 and casts. Both
    must equal the float32 widening exactly, including -0.0. An
    unsupported type must still fail loud (type 2/Q4_0 has no
    dequantizer), never silently quantize."""
    # BF16 bit patterns: 1.0, -2.5, 0.0, -0.0.
    raw = struct.pack("<HHHH", 0x3F80, 0xC020, 0x0000, 0x8000)
    want = np.array([1.0, -2.5, 0.0, -0.0], dtype=np.float32)
    ref = WeightRef(shape=(4,), raw=raw, ttype=30)
    got_np = ref.materialize(NumpyBackend())
    assert np.array_equal(got_np.view("<u4"), want.view("<u4"))
    try:
        from lacuna.gpu_executor import GPUUnavailableError, TorchBackend

        backend_t = TorchBackend("cpu")
    except GPUUnavailableError:
        pytest.skip("torch not installed")
    got_torch = backend_t.to_numpy(ref.materialize(backend_t))
    assert np.array_equal(got_torch.view("<u4"), want.view("<u4"))
    with pytest.raises(GGUFError):
        WeightRef(shape=(2,), raw=b"\x00\x00\x00\x00",
                  ttype=2).materialize(NumpyBackend())


def test_vram_budget_enforced_across_layers():
    model, _ = _model(vram_budget_bytes=64)
    with pytest.raises(GPUMemoryError):
        model.forward([0, 1], 0)


# ------------------------------------------------------- synthetic GGUF

def _u32(n):
    return struct.pack("<I", n)


def _u64(n):
    return struct.pack("<Q", n)


def _gstr(s):
    b = s.encode("utf-8")
    return _u64(len(b)) + b


def _kv_str(key, value):
    return _gstr(key) + _u32(8) + _gstr(value)


def _kv_u32(key, value):
    return _gstr(key) + _u32(4) + _u32(value)


def _kv_f32(key, value):
    return _gstr(key) + _u32(6) + struct.pack("<f", value)


def _kv_str_array(key, values):
    body = _u32(8) + _u64(len(values)) + b"".join(_gstr(v) for v in values)
    return _gstr(key) + _u32(9) + body


def _write_tiny_gguf(path, weights):
    """Minimal GGUF v3 file for the tiny synthetic model, so Engine
    can be tested end-to-end without a real model file."""
    embed, head, out_norm, layers = weights
    # No eos entry: eos is None, so generation runs the full
    # max_tokens (the eos-stop path is covered by test_eos_stops).
    kv = b"".join([
        _kv_str("general.architecture", "qwen2"),
        _kv_u32("qwen2.block_count", N_LAYER),
        _kv_u32("qwen2.embedding_length", HIDDEN),
        _kv_u32("qwen2.feed_forward_length", FFN),
        _kv_u32("qwen2.attention.head_count", N_HEAD),
        _kv_u32("qwen2.attention.head_count_kv", N_KV),
        _kv_f32("qwen2.attention.layer_norm_rms_epsilon", EPS),
        _kv_f32("qwen2.rope.freq_base", THETA),
        _kv_str_array("tokenizer.ggml.tokens", list(VOCAB)),
        _kv_str_array("tokenizer.ggml.merges", []),
    ])
    tensors = []

    def add(name, arr):
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        dims = [arr.shape[1], arr.shape[0]] if arr.ndim == 2 else [arr.shape[0]]
        tensors.append((name, dims, arr.tobytes()))

    add("token_embd.weight", embed)
    add("output.weight", head)
    add("output_norm.weight", out_norm)
    for li, layer in enumerate(layers):
        for key, ref in layer.items():
            add(f"blk.{li}.{key}", ref.array)
    infos = b""
    offset = 0
    for name, dims, blob in tensors:
        infos += _gstr(name) + _u32(len(dims))
        for d in dims:
            infos += _u64(d)
        infos += _u32(0) + _u64(offset)  # F32
        offset += len(blob)
    header = b"GGUF" + _u32(3) + _u64(len(tensors)) + _u64(10) + kv + infos
    data_start = len(header) + ((-len(header)) % 32)
    with open(path, "wb") as f:
        f.write(header)
        f.write(b"\x00" * (data_start - len(header)))
        for _, _, blob in tensors:
            f.write(blob)


def test_engine_gpu_mode_end_to_end(tmp_path):
    from lacuna.convert import convert
    from lacuna.engine import Engine
    from lacuna.paged_model import PagedQwen2

    rng = np.random.default_rng(0)
    weights = _weights(rng)
    gguf_path = str(tmp_path / "tiny.gguf")
    _write_tiny_gguf(gguf_path, weights)
    convert(gguf_path, gguf_path + ".lac", quiet=True)

    eng = Engine(gguf_path, gpu="numpy")
    try:
        assert eng.mode == "lacuna-gpu"
        assert eng.gpu_backend == "numpy:cpu"
        text_eng = eng.generate("ab", max_tokens=8)
        assert eng.gpu_stats is not None
        assert eng.gpu_stats["tokens_generated"] == 8
        assert eng.gpu_stats["within_300s_bound"] is True
    finally:
        eng.close()

    gpu = GPUModel.from_gguf(gguf_path, backend=NumpyBackend())
    try:
        text_gpu, ids_gpu, _ = gpu.generate("ab", max_tokens=8)
    finally:
        gpu.close()
    assert text_eng == text_gpu

    # And the CPU paged executor agrees on the same weights (greedy
    # ids identical; near-ties aside, this seed is clean — verified
    # by the test passing, and the reference test above pins the
    # math independently).
    cpu = PagedQwen2(gguf_path)
    try:
        text_cpu, ids_cpu = cpu.generate("ab", max_tokens=8)
    finally:
        cpu.close()
    assert ids_cpu == ids_gpu
    assert text_cpu == text_gpu
