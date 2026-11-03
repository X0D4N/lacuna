"""GPU MoE execution (``GPUModel`` in MoE mode) vs the CPU paged MoE
reference (``PagedMoE``).

The CPU paged MoE path is the reference: GPU vs CPU must give the
same greedy choice except at near-ties; a wide-margin flip is a bug,
not a tolerance. So the file-based tests below compare GPU-MoE
logits (numpy backend, CPU here) against CPU ``PagedMoE`` logits on
the SAME synthetic qwen3moe GGUF (built by ``tests.test_moe``'s
builder: 1 layer, hidden 16, ffn 32, 4 experts, top-k 2, F32 tensors,
QK-norm, attention biases), and greedy generate ids must match
exactly. The direct-source tests pin the branches the file path does
not: per-layer KV isolation, VRAM budget, speculative truncate, and
the no-QK-norm (olmoe-style) branch against an independent numpy
reference.

No model file is downloaded and no CUDA is needed: the numpy backend
runs on CPU here; the torch-CUDA MoE check is the parent-launched
Kaggle job's work, not this file's.
"""

import numpy as np
import pytest

from lacuna.gpu_executor import (
    GPUExecutorError,
    GPUModel,
    GPUMemoryError,
    LayerConfig,
    NumpyBackend,
    WeightRef,
)
from lacuna.moe import PagedMoE
from tests.test_moe import build_tiny_moe_gguf


# ------------------------------------------------------------ helpers

def _moe_config(qk_norm=True):
    return LayerConfig(hidden=16, ffn=32, n_head=2, n_kv=1, eps=1e-6,
                       rope_theta=10000.0, n_experts=4, top_k=2,
                       qk_norm=qk_norm)


def _w(arr):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    return WeightRef(shape=arr.shape, array=arr)


def _direct_moe_weights(rng, n_layer=1, shared=False, qk_norm=True,
                        biases=True):
    """Synthetic direct-source MoE weights (in-memory WeightRefs)."""
    def mat(n_out, n_in):
        return rng.normal(0.0, 0.1, size=(n_out, n_in)).astype(np.float32)

    def bias(n):
        return rng.normal(0, 0.01, n).astype(np.float32)

    vocab = list("abcdefgh")
    embed = mat(len(vocab), 16)
    head = mat(len(vocab), 16)
    out_norm = np.ones(16, dtype=np.float32)
    layers = []
    for _ in range(n_layer):
        layer = {
            "attn_norm.weight": _w(np.ones(16, dtype=np.float32)),
            "attn_q.weight": _w(mat(16, 16)),
            "attn_k.weight": _w(mat(8, 16)),
            "attn_v.weight": _w(mat(8, 16)),
            "attn_output.weight": _w(mat(16, 16)),
            "ffn_norm.weight": _w(np.ones(16, dtype=np.float32)),
            "ffn_gate_inp.weight": _w(mat(4, 16)),
            "experts": {},
        }
        if biases:
            layer["attn_q.bias"] = _w(bias(16))
            layer["attn_k.bias"] = _w(bias(8))
            layer["attn_v.bias"] = _w(bias(8))
        if qk_norm:
            layer["attn_q_norm.weight"] = _w(np.ones(8, dtype=np.float32))
            layer["attn_k_norm.weight"] = _w(np.ones(8, dtype=np.float32))
        for e in range(4):
            layer["experts"][("gate", e)] = _w(mat(32, 16))
            layer["experts"][("up", e)] = _w(mat(32, 16))
            layer["experts"][("down", e)] = _w(mat(16, 32))
        if shared:
            layer["ffn_gate_shexp.weight"] = _w(mat(32, 16))
            layer["ffn_up_shexp.weight"] = _w(mat(32, 16))
            layer["ffn_down_shexp.weight"] = _w(mat(16, 32))
        layers.append(layer)
    return vocab, embed, head, out_norm, layers


class _StubTokenizer:
    def __init__(self, vocab):
        self._v = list(vocab)

    def encode(self, text):
        return [self._v.index(c) for c in text]

    def decode(self, ids):
        return "".join(self._v[i] for i in ids)


def _direct_moe_model(rng_seed=0, n_layer=1, shared=False, qk_norm=True,
                      biases=True, **kw):
    rng = np.random.default_rng(rng_seed)
    vocab, embed, head, out_norm, layers = _direct_moe_weights(
        rng, n_layer=n_layer, shared=shared, qk_norm=qk_norm,
        biases=biases)
    model = GPUModel.from_arrays_moe(
        NumpyBackend(), _moe_config(qk_norm=qk_norm), embed, head,
        out_norm, layers, _StubTokenizer(vocab), eos=None, **kw)
    return model, (vocab, embed, head, out_norm, layers)


def _ref_rmsnorm(x, w, eps):
    return (x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True)
                        + np.float32(eps))) * w


def _ref_rope(x, positions, theta):
    T, H, D = x.shape
    h = D // 2
    inv = 1.0 / (theta ** (np.arange(0, D, 2, dtype=np.float32) / D))
    ang = positions[:, None] * inv[None, :]
    cos, sin = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    x0, x1 = x[..., :h], x[..., h:]
    return np.concatenate([x0 * cos - x1 * sin,
                           x0 * sin + x1 * cos], axis=-1)


def _ref_moe_forward(weights, ids):
    """Independent numpy MoE reference (olmoe-style: no QK-norm).
    Separate code path from GPULayer: einsum attention, grouped
    experts via the same v1 router rule re-derived here."""
    vocab, embed, head, out_norm, layers = weights
    cfg = _moe_config(qk_norm=False)
    x = embed[np.asarray(ids, dtype=int)]
    positions = np.arange(len(ids), dtype=np.float32)
    T = len(ids)
    for layer in layers:
        a = {k: v.array for k, v in layer.items() if k != "experts"}
        ex = layer["experts"]
        h = _ref_rmsnorm(x, a["attn_norm.weight"], cfg.eps)
        q = h @ a["attn_q.weight"].T + a["attn_q.bias"]
        k = h @ a["attn_k.weight"].T + a["attn_k.bias"]
        v = h @ a["attn_v.weight"].T + a["attn_v.bias"]
        q = _ref_rope(q.reshape(T, 2, 8), positions, cfg.rope_theta)
        k = _ref_rope(k.reshape(T, 1, 8), positions, cfg.rope_theta)
        v = v.reshape(T, 1, 8)
        k_rep = np.repeat(k, 2, axis=1)
        v_rep = np.repeat(v, 2, axis=1)
        scores = (np.einsum("htd,hsd->hts", q.transpose(1, 0, 2),
                            k_rep.transpose(1, 0, 2))
                  / np.float32(np.sqrt(8)))
        row = positions.astype(np.int64)[:, None]
        col = np.arange(k_rep.shape[0])[None, :]
        scores = np.where(col <= row, scores, np.float32(-1e30))
        scores = scores - scores.max(axis=-1, keepdims=True)
        probs = np.exp(scores)
        probs = probs / probs.sum(axis=-1, keepdims=True)
        o = np.einsum("hts,hsd->htd", probs, v_rep.transpose(1, 0, 2))
        x = x + o.transpose(1, 0, 2).reshape(T, 16) @ a[
            "attn_output.weight"].T
        h2 = _ref_rmsnorm(x, a["ffn_norm.weight"], cfg.eps)
        # v1 router rule, re-derived: softmax -> top-k -> renormalize.
        rlog = h2 @ a["ffn_gate_inp.weight"].T
        p = np.exp(rlog - rlog.max(axis=-1, keepdims=True))
        p = p / p.sum(axis=-1, keepdims=True)
        order = np.argsort(-p, axis=-1, kind="stable")
        sel = order[:, :2]
        wsel = np.take_along_axis(p, sel, axis=-1)
        wsel = wsel / wsel.sum(axis=-1, keepdims=True)
        out = np.zeros_like(h2)
        for e in range(4):
            pos_idx, slot_idx = np.where(sel == e)
            if len(pos_idx) == 0:
                continue
            gu = h2[pos_idx] @ np.concatenate(
                [ex[("gate", e)].array, ex[("up", e)].array]) .T
            gate, up = gu[:, :32], gu[:, 32:]
            act = (gate / (np.float32(1.0) + np.exp(-gate))) * up
            contrib = act @ ex[("down", e)].array.T
            np.add.at(out, pos_idx,
                      contrib * wsel[pos_idx, slot_idx][:, None])
        x = x + out
    last = _ref_rmsnorm(x[-1:], out_norm, cfg.eps)
    return (last @ head.T)[0]


# ------------------------------------------------------------ file-based
# GPU-MoE (paged) vs CPU PagedMoE on the same synthetic GGUF.

def _file_models(tmp_path, shared=False):
    gguf_path = str(tmp_path / "tiny_moe.gguf")
    build_tiny_moe_gguf(gguf_path, shared=shared)
    gpu = GPUModel.from_gguf(gguf_path, backend=NumpyBackend())
    cpu = PagedMoE(gguf_path)
    return gpu, cpu


def test_moe_forward_matches_cpu_paged(tmp_path):
    gpu, cpu = _file_models(tmp_path)
    try:
        assert gpu.moe is True
        got = gpu.forward([0, 1], 0)
        want = cpu.forward([0, 1], 0)
        assert got.shape == want.shape
        assert np.max(np.abs(got - want)) < 1e-4
    finally:
        gpu.close()
        cpu.close()


def test_moe_generate_ids_match_cpu_paged(tmp_path):
    gpu, cpu = _file_models(tmp_path)
    try:
        _, gpu_ids, _ = gpu.generate("a", max_tokens=4)
        _, cpu_ids = cpu.generate("a", max_tokens=4)
        assert gpu_ids == cpu_ids
    finally:
        gpu.close()
        cpu.close()


def test_moe_shared_expert_matches_cpu_paged(tmp_path):
    gpu, cpu = _file_models(tmp_path, shared=True)
    try:
        got = gpu.forward([0, 1, 2], 0)
        want = cpu.forward([0, 1, 2], 0)
        assert np.max(np.abs(got - want)) < 1e-4
    finally:
        gpu.close()
        cpu.close()


def test_moe_demand_only_selected_experts(tmp_path):
    """Per-expert paging audit: the GPU path must fault ONLY the
    experts the router selected. A recording wrapper around the
    paged source's expert_ref logs every (layer, expert) demanded;
    the set must equal the CPU reference's selected set exactly —
    no unselected expert touched, no selected expert missed."""
    gpu, cpu = _file_models(tmp_path)
    try:
        calls: list[tuple[int, int]] = []
        orig = gpu.source.expert_ref

        def rec(li, role, e):
            calls.append((li, e))
            return orig(li, role, e)

        gpu.source.expert_ref = rec
        ids = [0, 1, 2]
        gpu.forward(ids, 0)
        cpu.forward(ids, 0)
        for li, (sel_ids, _w) in cpu.last_routed.items():
            selected = set(int(e) for e in np.unique(sel_ids))
            demanded = set(e for (l, e) in calls if l == li)
            assert demanded == selected, (
                f"layer {li}: demanded {demanded} != selected {selected}")
    finally:
        gpu.close()
        cpu.close()


def test_moe_unsupported_arch_fails_loud(tmp_path):
    gguf_path = str(tmp_path / "other.gguf")
    build_tiny_moe_gguf(gguf_path, arch="deepseek")
    # An architecture outside (qwen2, qwen3moe, olmoe) must fail loud
    # at the from_gguf arch gate — DeepSeek sigmoid+bias routing and
    # MLA are rung 3, never silently run as v1 MoE.
    with pytest.raises(GPUExecutorError):
        GPUModel.from_gguf(gguf_path, backend=NumpyBackend())


# ------------------------------------------------------------ direct-source

def test_moe_kv_isolation_across_layers():
    model, _ = _direct_moe_model(n_layer=2)
    model.forward([0, 1, 2], 0)
    caches = [ex.cache for ex in model._executors]
    assert len(caches) == 2
    for c in caches:
        assert c["k"].shape == (3, 2, 8)
        assert c["v"].shape == (3, 2, 8)
    assert not np.array_equal(caches[0]["k"], caches[1]["k"])
    assert not np.array_equal(caches[0]["v"], caches[1]["v"])


def test_moe_vram_budget_breach_raises():
    model, _ = _direct_moe_model(vram_budget_bytes=64)
    with pytest.raises(GPUMemoryError):
        model.forward([0, 1], 0)


def test_moe_truncate_cache_matches_fresh_prefill():
    model, _ = _direct_moe_model(n_layer=2)
    p, a = [0, 1], [2, 3]
    rejected = [4, 5]
    probe = 6
    model.reset_cache()
    model.forward(p, 0)
    model.forward(a + rejected, len(p))
    model.truncate_cache(len(p) + len(a))
    logits_inc = model.forward([probe], len(p) + len(a))
    for ex in model._executors:
        assert ex.cache["k"].shape[0] == len(p) + len(a) + 1
    model.reset_cache()
    logits_fresh = model.forward(p + a + [probe], 0)
    assert np.max(np.abs(logits_inc - logits_fresh)) < 1e-4


def test_moe_no_qknorm_matches_reference():
    """Olmoe-style branch (qk_norm=False, biases present): GPU-MoE
    forward must match the independent numpy reference."""
    model, weights = _direct_moe_model(qk_norm=False)
    got = model.forward([0, 1, 2, 3], 0)
    want = _ref_moe_forward(weights, [0, 1, 2, 3])
    assert np.max(np.abs(got - want)) < 1e-4


def test_moe_generate_smoke_matches_cpu_reference():
    """GPU-MoE generate smoke on a tiny synthetic model: ids match
    the CPU-side reference rollout (reference forward per token)."""
    model, weights = _direct_moe_model(qk_norm=False)
    _, ids, stats = model.generate("ab", max_tokens=8)
    assert len(ids) == 8
    assert stats["tokens_generated"] == 8
    # Reference rollout: greedy argmax of the reference forward.
    seq = [0, 1]
    want = []
    for _ in range(8):
        nxt = int(np.argmax(_ref_moe_forward(weights, seq)))
        want.append(nxt)
        seq.append(nxt)
    assert ids == want


# ------------------------------------------------------------ Engine dispatch

def test_engine_moe_dispatch(tmp_path):
    """Engine(gguf, gpu='numpy') on a qwen3moe file must land in
    mode 'lacuna-gpu' (the GPU-MoE path), not fall through to the
    CPU chain — and its text must match the CPU MoE reference."""
    from lacuna.engine import Engine

    gguf_path = str(tmp_path / "tiny_moe.gguf")
    build_tiny_moe_gguf(gguf_path)
    eng = Engine(gguf_path, gpu="numpy")
    try:
        assert eng.mode == "lacuna-gpu"
        assert eng.gpu_backend == "numpy:cpu"
        text_eng = eng.generate("a", max_tokens=4)
        assert eng.gpu_stats is not None
        assert eng.gpu_stats["tokens_generated"] == 4
    finally:
        if eng._gpu is not None:
            eng._gpu.close()
    cpu = PagedMoE(gguf_path)
    try:
        text_cpu, _ = cpu.generate("a", max_tokens=4)
        assert text_eng == text_cpu
    finally:
        cpu.close()
