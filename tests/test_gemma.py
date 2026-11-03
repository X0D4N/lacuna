"""Gemma-4 paged executor tests on tiny synthetic GGUFs (no model file).

Builds minimal valid gemma4 GGUFs in a tmp dir: hidden 16, sliding
4 Q / 2 KV heads x dim 4, global 4 Q / 1 KV head x dim 8, FFN 24,
8-token SPM toy vocab. Exercises the real paged forward pass
(parse -> pager -> dequant -> hybrid-attention transformer -> greedy)
plus the SentencePiece tokenizer, without a downloaded model.
"""

import math
import os
import struct

import numpy as np
import pytest

from lacuna.gemma import GemmaModelError, PagedGemma4
from lacuna.spm import SPMTokenizer, SPMTokenizerError


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


def build_tiny_gemma_gguf(path: str, n_layer: int = 6, window: int = 3,
                          global_v: bool = False, tied: bool = True,
                          layer_types=None, arch: str = "gemma4",
                          with_global_kv: bool = True,
                          meta_layer_types=None, kv_array_vals=None,
                          tokenizer_model: str = "llama",
                          with_global_head_dim: bool = True,
                          freq_base_swa: float | None = None,
                          with_gemma4_block_tensors: bool = True,
                          rope_freqs_vals=None,
                          with_rope_freqs: bool = True) -> str:
    rng = np.random.RandomState(1)

    def w(shape):
        return (rng.standard_normal(shape) * 0.05).astype("<f4")

    tensors = [
        ("token_embd.weight", [16, 8], w((8, 16))),
        ("output_norm.weight", [16], w((16,))),
    ]
    if with_rope_freqs:
        # Global head_dim is 8 in this fixture, so the stored
        # rope_freqs tensor has 8 // 2 = 4 entries (ggml-style
        # global RoPE divides angles by these factors).
        ff = ([1.0, 1.0, 1.0, 1.0] if rope_freqs_vals is None
              else list(rope_freqs_vals))
        tensors.append(("rope_freqs.weight", [4],
                        np.array(ff, dtype="<f4")))
    if not tied:
        tensors.append(("output.weight", [16, 8], w((8, 16))))
    for li in range(n_layer):
        is_global = (li % 6 == 5)
        if li == n_layer - 1:
            is_global = True  # loader forces the last layer global;
            # the synthetic file must carry global-shaped tensors there
        if layer_types is not None:
            is_global = layer_types[li] == "full_attention"
        base = f"blk.{li}."
        tensors.append((base + "attn_norm.weight", [16], w((16,))))
        tensors.append((base + "ffn_norm.weight", [16], w((16,))))
        if with_gemma4_block_tensors:
            # Gemma-4 Peri-LN block tensors (real-file inventory,
            # measured 2026-10-26): post-sublayer norms, the per-layer
            # output scale, and the per-head-dim QK-norms. A file
            # without them must fail loud in forward(), never run
            # with silently-wrong block math.
            tensors.append((base + "post_attention_norm.weight",
                            [16], w((16,))))
            tensors.append((base + "post_ffw_norm.weight",
                            [16], w((16,))))
            tensors.append((base + "layer_output_scale.weight", [1],
                            np.array([0.9], dtype="<f4")))
            hd = 8 if is_global else 4
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
        _kv_str("general.architecture", arch),
        _kv_u32("gemma4.block_count", n_layer),
        _kv_u32("gemma4.embedding_length", 16),
        _kv_u32("gemma4.feed_forward_length", 24),
        _kv_u32("gemma4.attention.head_count", 4),
        _kv_f32("gemma4.attention.layer_norm_rms_epsilon", 1e-6),
        _kv_u32("gemma4.attention.sliding_window", window),
        _kv_u32("gemma4.attention.head_dim", 4),
        _kv_f32("gemma4.rope.freq_base", 10000.0),
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
    if with_global_head_dim:
        kvs.append(_kv_u32("gemma4.attention.global_head_dim", 8))
    if freq_base_swa is not None:
        kvs.append(_kv_f32("gemma4.rope.freq_base_swa", freq_base_swa))
    if with_global_kv:
        kvs.append(_kv_u32("gemma4.attention.head_count_kv_global", 1))
    if meta_layer_types is not None:
        kvs.append(_kv_str_array("gemma4.layer_types", meta_layer_types))
    elif layer_types is not None:
        kvs.append(_kv_str_array("gemma4.layer_types", layer_types))
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
        blob = arr.tobytes()
        blobs.append(blob)
        offset += len(blob)
    head = header + infos
    pad = (-len(head)) % 32
    with open(path, "wb") as f:
        f.write(head + b"\x00" * pad + b"".join(blobs))
    return path


def _load(tmp_path, **kw):
    gguf = build_tiny_gemma_gguf(os.path.join(tmp_path, "tiny.gguf"), **kw)
    return PagedGemma4(gguf, hot_budget_bytes=1024 * 1024)


# ---------------------------------------------------------------- layers ---


def test_layer_pattern_5_to_1(tmp_path):
    m = _load(tmp_path)
    try:
        assert m.layer_types == ["sliding_attention"] * 5 + ["full_attention"]
        assert m._layer_dims(0) == (4, 2, 4)   # sliding
        assert m._layer_dims(5) == (4, 1, 8)   # global: 4 KV, dim 512->8
        assert m.window == 3
    finally:
        m.close()


def test_last_layer_forced_global(tmp_path):
    # 7 layers: the 5:1 pattern would leave layer 6 sliding; the rule
    # forces the last layer global.
    m = _load(tmp_path, n_layer=7)
    try:
        assert m.layer_types[-1] == "full_attention"
        assert m.layer_types[5] == "full_attention"
    finally:
        m.close()


def test_layer_types_metadata_override(tmp_path):
    types = ["full_attention"] + ["sliding_attention"] * 5
    m = _load(tmp_path, layer_types=types)
    try:
        assert m.layer_types == types
        assert m._is_global(0) and not m._is_global(1)
    finally:
        m.close()


def test_layer_types_length_mismatch_fails(tmp_path):
    with pytest.raises(GemmaModelError):
        _load(tmp_path, meta_layer_types=["sliding_attention"] * 3)


# ------------------------------------------------------------- attention ---


def test_sliding_mask_hand_computed(tmp_path):
    m = _load(tmp_path)
    try:
        row = np.arange(5)
        col = np.arange(5)
        got = PagedGemma4._attn_mask(row, col, True, 3)
        # Hand-computed: causal AND within last 3 positions (incl. self).
        want = np.array([
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1],
        ], dtype=bool)
        assert np.array_equal(got, want)
        got_full = PagedGemma4._attn_mask(row, col, False, 3)
        want_full = np.tril(np.ones((5, 5), dtype=bool))
        assert np.array_equal(got_full, want_full)
    finally:
        m.close()


def test_attention_matches_hand_loop_reference(tmp_path):
    m = _load(tmp_path)
    try:
        rng = np.random.RandomState(7)
        T, S, H, D = 3, 5, 2, 4
        q = rng.standard_normal((T, H, D)).astype(np.float32)
        k = rng.standard_normal((S, H, D)).astype(np.float32)
        v = rng.standard_normal((S, H, D)).astype(np.float32)
        row_pos = np.array([2, 3, 4])
        col_pos = np.array([0, 1, 2, 3, 4])
        got = m._attention(q, k, v, row_pos, col_pos, True)
        # Hand-computed reference: explicit Python loops, no einsum.
        # Gemma-4 attention scale is 1.0, NOT 1/sqrt(head_dim): QK-norm
        # provides the stability (Keras Gemma4Backbone; the real 31B
        # file's q_norm x k_norm product is exactly 1/8 per layer,
        # which only makes sense with scale 1.0 — with 1/sqrt(d) the
        # scores would collapse to std 0.125, near-uniform attention).
        want = np.zeros((T, H, D), dtype=np.float64)
        for t in range(T):
            for h in range(H):
                scores = []
                for s in range(S):
                    if col_pos[s] > row_pos[t] or row_pos[t] - col_pos[s] >= 3:
                        scores.append(-np.inf)
                    else:
                        scores.append(float(np.dot(q[t, h], k[s, h])))
                mx = max(scores)
                exps = [math.exp(s - mx) for s in scores]
                tot = sum(exps)
                for d in range(D):
                    want[t, h, d] = sum(
                        exps[s] / tot * v[s, h, d] for s in range(S)
                    )
        assert np.allclose(got, want, atol=1e-5)
    finally:
        m.close()


def test_sliding_rope_full_rotation(tmp_path):
    m = _load(tmp_path)
    try:
        # Sliding: full head dim rotates, half-rotation form.
        t = np.array([[[1.0, 0.0, 0.0, 1.0]]], dtype=np.float32)
        out = m._rope(t, np.array([3]), is_global=False)
        theta = 10000.0
        inv = theta ** (-np.arange(0, 4, 2, dtype=np.float32) / 4)
        ang = 3 * inv
        c, s = np.cos(ang), np.sin(ang)
        # halves: x0 = [1, 0], x1 = [0, 1]; out0 = x0*c - x1*s,
        # out1 = x0*s + x1*c.
        want = np.array([[[c[0], -s[1], s[0], c[1]]]], dtype=np.float32)
        assert np.allclose(out, want, atol=1e-6)
    finally:
        m.close()


def test_global_rope_uses_stored_factors(tmp_path):
    # Global RoPE is the ggml formula: FULL head dim (8 here),
    # NEOX pairs (i, i+4), angle = position * theta**(-2i/8) /
    # rope_freqs[i]. The old partial rotation (first 2 dims only,
    # factors ignored) FAILS this test: dims 2-7 would pass through
    # untouched and the pair structure would be (0, 1), not (i, i+4).
    ff = [1.0, 2.0, 0.5, 4.0]
    gguf = build_tiny_gemma_gguf(os.path.join(tmp_path, "tiny.gguf"),
                                 rope_freqs_vals=ff)
    m = PagedGemma4(gguf, hot_budget_bytes=1024 * 1024)
    try:
        assert m.rope_freqs is not None
        assert list(m.rope_freqs) == ff
        t = np.arange(8, dtype=np.float32).reshape(1, 1, 8) + 1.0
        out = m._rope(t, np.array([5]), is_global=True)
        inv = 1000000.0 ** (-np.arange(0, 8, 2, dtype=np.float32) / 8)
        ang = 5 * inv / np.array(ff, dtype=np.float32)
        c, s = np.cos(ang), np.sin(ang)
        x0, x1 = t[0, 0, :4], t[0, 0, 4:]
        want = np.concatenate([x0 * c - x1 * s, x0 * s + x1 * c])
        assert np.allclose(out[0, 0], want, atol=1e-6)
    finally:
        m.close()


def test_global_model_without_rope_freqs_fails_loud(tmp_path):
    gguf = build_tiny_gemma_gguf(os.path.join(tmp_path, "tiny.gguf"),
                                 with_rope_freqs=False)
    with pytest.raises(GemmaModelError, match="rope_freqs"):
        PagedGemma4(gguf, hot_budget_bytes=1024 * 1024)


def test_gelu_tanh_values(tmp_path):
    m = _load(tmp_path)
    try:
        xs = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        got = PagedGemma4._gelu_tanh(xs)
        want = np.array([
            0.5 * float(x) * (1.0 + math.tanh(
                math.sqrt(2.0 / math.pi) * (float(x) + 0.044715 * float(x) ** 3)
            )) for x in xs
        ], dtype=np.float32)
        assert np.allclose(got, want, atol=1e-6)
        assert got[2] == 0.0
    finally:
        m.close()


# -------------------------------------------------------------- forward ---


def test_global_v_equals_k(tmp_path):
    m = _load(tmp_path)  # global layer 5 has no attn_v tensor
    try:
        # V=K semantics per llama.cpp gemma4.cpp: V is the RAW k
        # projection passed through a WEIGHTLESS per-head RMSNorm,
        # while the cached K is the same raw projection passed
        # through the k_norm-WEIGHTED RMSNorm and then RoPE. At
        # position 0 RoPE is the identity, so cached K must equal
        # cached V times the k_norm weight, exactly — this pins V
        # = weightless-normed raw K (not raw K, not k_normed K).
        m.forward([6], 0)
        kc, vc, _start = m.cache[5]
        k_norm = m._w("blk.5.attn_k_norm.weight")
        assert np.allclose(kc, vc * k_norm, atol=1e-6)
    finally:
        m.close()


def test_global_with_v_uses_it(tmp_path):
    m = _load(tmp_path, global_v=True)
    try:
        m.forward([6, 5, 3], 0)
        kc, vc, _start = m.cache[5]
        assert not np.array_equal(kc, vc)
    finally:
        m.close()


def test_sliding_cache_trims_to_window(tmp_path):
    m = _load(tmp_path, window=3)
    try:
        m.forward([6, 5, 3, 6, 5, 3, 6, 5], 0)
        kc, _vc, start = m.cache[0]  # sliding layer
        assert kc.shape[0] == 3 and start == 5
        kc5, _vc5, start5 = m.cache[5]  # global layer keeps all
        assert kc5.shape[0] == 8 and start5 == 0
    finally:
        m.close()


def test_tiny_forward_finite_and_pager_faults(tmp_path):
    m = _load(tmp_path)
    try:
        logits = m.forward([6, 5, 3], 0)
        assert logits.shape == (8,)
        assert np.isfinite(logits).all()
        assert m.store.stats["faults"] > 0  # weights via the pager
    finally:
        m.close()


def test_generate_deterministic(tmp_path):
    m = _load(tmp_path)
    try:
        t1, ids1 = m.generate("ab a", max_tokens=4)
        t2, ids2 = m.generate("ab a", max_tokens=4)
        assert ids1 == ids2 and t1 == t2
    finally:
        m.close()


def test_tied_head(tmp_path):
    m = _load(tmp_path, tied=True)
    try:
        assert m.tied is True
        logits = m.forward([6, 5, 3], 0)
        assert np.isfinite(logits).all()
    finally:
        m.close()


def test_untied_head(tmp_path):
    m = _load(tmp_path, tied=False)
    try:
        assert m.tied is False
        logits = m.forward([6, 5, 3], 0)
        assert np.isfinite(logits).all()
    finally:
        m.close()


def test_embed_scale_default_and_override(tmp_path):
    m = _load(tmp_path)
    try:
        assert m.embed_scale == pytest.approx(4.0)  # sqrt(hidden 16)
    finally:
        m.close()
    gguf = build_tiny_gemma_gguf(os.path.join(tmp_path, "tiny2.gguf"))
    m2 = PagedGemma4(gguf, hot_budget_bytes=1024 * 1024, embed_scale=2.0)
    try:
        assert m2.embed_scale == 2.0
    finally:
        m2.close()


def test_fail_loud(tmp_path):
    with pytest.raises(GemmaModelError):
        _load(tmp_path, arch="qwen2")
    # Scalar KV + no global key + no global layer shapes -> still loud.
    with pytest.raises(GemmaModelError):
        _load(tmp_path, with_global_kv=False,
              layer_types=["sliding_attention"] * 6)


# ------------------------------------------- real-file KV geometry (31B) ---


def test_per_layer_kv_array_no_global_key(tmp_path):
    # Mirrors the real 31B file: per-layer head_count_kv array, no
    # global-KV scalar key. Sliding 2, global 1 (tiny-scale values).
    m = _load(tmp_path, kv_array_vals=[2, 2, 2, 2, 2, 1],
              with_global_kv=False)
    try:
        assert m.n_kv == 2
        assert m.n_kv_global == 1
        assert m._layer_dims(0) == (4, 2, 4)
        assert m._layer_dims(5) == (4, 1, 8)
        assert np.all(np.isfinite(m.forward([3, 4], 0)))
    finally:
        m.close()


def test_per_layer_kv_array_length_mismatch_fails(tmp_path):
    with pytest.raises(GemmaModelError):
        _load(tmp_path, kv_array_vals=[2, 2], with_global_kv=False)


def test_per_layer_kv_array_nonuniform_fails(tmp_path):
    with pytest.raises(GemmaModelError):
        _load(tmp_path, kv_array_vals=[2, 2, 2, 2, 2, 1],
              with_global_kv=False,
              layer_types=["sliding_attention"] * 6)


def test_scalar_no_global_key_shape_fallback(tmp_path):
    # No global key AND no global_head_dim metadata: both must come
    # from the global layer's q/k row counts (32 rows / 4 heads = 8;
    # 8 rows / 8 = 1 KV head).
    m = _load(tmp_path, with_global_kv=False, with_global_head_dim=False)
    try:
        assert m.n_kv_global == 1
        assert m.head_dim_global == 8
    finally:
        m.close()


def test_rope_freq_base_swa_wins_for_sliding(tmp_path):
    # The real file carries freq_base (global) + freq_base_swa
    # (sliding); the _swa key must win for sliding layers only.
    m = _load(tmp_path, freq_base_swa=12345.0)
    try:
        assert m.rope_theta == 12345.0
        assert m.rope_theta_global == 1000000.0
    finally:
        m.close()


def test_spm_gemma4_model_tag_accepted():
    tok = SPMTokenizer.from_metadata({
        "tokenizer.ggml.model": "gemma4",
        "tokenizer.ggml.tokens": SPM_TOKENS,
        "tokenizer.ggml.scores": SPM_SCORES,
        "tokenizer.ggml.token_type": SPM_TYPES,
        "tokenizer.ggml.bos_token_id": 2,
        "tokenizer.ggml.eos_token_id": 1,
    })
    assert tok.bos == 2 and tok.eos == 1
    assert tok.encode("ab") == [6]


def test_spm_unknown_model_tag_rejected():
    with pytest.raises(SPMTokenizerError):
        SPMTokenizer.from_metadata({
            "tokenizer.ggml.model": "gpt2",
            "tokenizer.ggml.tokens": SPM_TOKENS,
            "tokenizer.ggml.scores": SPM_SCORES,
        })


# ------------------------------------------------------------- tokenizer ---


def _spm_tokens(include_abc: bool = True):
    tokens = ["<pad>", "<eos>", "<bos>", "<unk>", "a", "b", "c", "▁",
              "ab", "bc", "<0x41>", "<start>"]
    scores = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 5.0, 0.0, 0.0]
    types = [3, 3, 3, 2, 1, 1, 1, 1, 1, 1, 6, 4]
    if include_abc:
        tokens.insert(10, "abc")
        scores.insert(10, 9.0)
        types.insert(10, 1)
    return SPMTokenizer(tokens, scores, types, bos=2, eos=1)


def test_spm_merge_order_is_score_order():
    # Without an "abc" piece, "abc" must segment [a, bc] because bc
    # (5.0) outranks ab (4.0) — score order, not left-to-right.
    tok = _spm_tokens(include_abc=False)
    assert tok.encode("abc") == [tok.vocab["a"], tok.vocab["bc"]]
    # With "abc" present, the final merge to "abc" wins.
    assert _spm_tokens().encode("abc") == [_spm_tokens().vocab["abc"]]


def test_spm_round_trip_and_space():
    tok = _spm_tokens()
    ids = tok.encode("a b")
    assert tok.decode(ids) == "a b"
    ids = tok.encode("ab")
    assert ids == [tok.vocab["ab"]] and tok.decode(ids) == "ab"


def test_spm_byte_fallback():
    tok = _spm_tokens()
    ids = tok.encode("A")  # 'A' not in vocab; UTF-8 byte 0x41 is.
    assert ids == [tok.vocab["<0x41>"]]
    assert tok.decode(ids) == "A"


def test_spm_special_literal_and_decode_skips():
    tok = _spm_tokens()
    ids = tok.encode("<start>ab")
    assert ids[0] == tok.vocab["<start>"]
    assert tok.decode(ids) == "ab"  # specials are structural, not text


def test_spm_scores_required():
    with pytest.raises(SPMTokenizerError):
        SPMTokenizer.from_metadata({
            "tokenizer.ggml.model": "llama",
            "tokenizer.ggml.tokens": ["a", "b"],
        })


def test_spm_no_dummy_prefix():
    tok = _spm_tokens()
    # add_dummy_prefix=False: "a" encodes without a leading ▁ piece.
    assert tok.encode("a") == [tok.vocab["a"]]


# ---------------------------------------------------------------- engine ---


def test_engine_dispatch_gemma(tmp_path, monkeypatch):
    import lacuna.engine as engine_mod

    monkeypatch.setattr(engine_mod, "_find_llama_server", lambda *a: None)
    gguf = build_tiny_gemma_gguf(os.path.join(tmp_path, "tiny.gguf"))
    eng = engine_mod.Engine(gguf)
    try:
        assert eng.mode == "lacuna"
        assert eng.paged_error is None
        text = eng.generate("ab a", max_tokens=4)
        # Dispatch, not content, is what's pinned here: the tiny
        # random model may greedily emit only specials (decoding to
        # an empty string), which is a valid generation result, not
        # a simulated fallback.
        assert isinstance(text, str) and not text.startswith("[simulated]")
    finally:
        eng.close()


# ------------------------------------------------- gemma-4 block correction ---


def test_missing_gemma4_block_tensors_fail_loud(tmp_path):
    """A Gemma-4 file without the Peri-LN block tensors must FAIL
    LOUD, never run with silently-wrong block math. This is the
    2026-10-26 layer-0 blowup bug class: the old code ignored
    post_attention_norm / post_ffw_norm / layer_output_scale /
    attn_q_norm / attn_k_norm entirely, so a file lacking them ran
    fine and produced garbage (layer-0 out RMS 92 vs input 1.28 on
    real 31B weights)."""
    m = _load(tmp_path, with_gemma4_block_tensors=False)
    try:
        with pytest.raises(GemmaModelError) as exc:
            m.forward([3, 4, 6], 0)
        assert "attn_q_norm" in str(exc.value)
    finally:
        m.close()


def _build_one_layer_sliding_gguf(path: str, wts: dict) -> str:
    """1-layer sliding-only gemma4 GGUF from explicit numpy weights.

    Hidden 16, 4 Q / 2 KV heads x dim 4, FFN 24, window 16, tied
    head, 8-token SPM toy vocab. Metadata carries the global-KV
    scalar keys so the loader need not derive them from a global
    layer (there is none); layer_types metadata pins the single
    layer sliding (the loader's last-layer-global forcing applies
    only to the derived pattern, not to a metadata array)."""
    tensors = [
        ("token_embd.weight", [16, 8], wts["emb"]),
        ("output_norm.weight", [16], wts["out_norm"]),
        ("blk.0.attn_norm.weight", [16], wts["attn_norm"]),
        ("blk.0.post_attention_norm.weight", [16], wts["post_attn"]),
        ("blk.0.layer_output_scale.weight", [1], wts["scale"]),
        ("blk.0.attn_q_norm.weight", [4], wts["q_norm"]),
        ("blk.0.attn_k_norm.weight", [4], wts["k_norm"]),
        ("blk.0.attn_q.weight", [16, 16], wts["attn_q"]),
        ("blk.0.attn_k.weight", [16, 8], wts["attn_k"]),
        ("blk.0.attn_v.weight", [16, 8], wts["attn_v"]),
        ("blk.0.attn_output.weight", [16, 16], wts["attn_output"]),
        ("blk.0.ffn_norm.weight", [16], wts["ffn_norm"]),
        ("blk.0.post_ffw_norm.weight", [16], wts["post_ffw"]),
        ("blk.0.ffn_gate.weight", [16, 24], wts["ffn_gate"]),
        ("blk.0.ffn_up.weight", [16, 24], wts["ffn_up"]),
        ("blk.0.ffn_down.weight", [24, 16], wts["ffn_down"]),
    ]
    kvs = [
        _kv_str("general.architecture", "gemma4"),
        _kv_u32("gemma4.block_count", 1),
        _kv_u32("gemma4.embedding_length", 16),
        _kv_u32("gemma4.feed_forward_length", 24),
        _kv_u32("gemma4.attention.head_count", 4),
        _kv_u32("gemma4.attention.head_count_kv", 2),
        _kv_u32("gemma4.attention.head_count_kv_global", 1),
        _kv_u32("gemma4.attention.global_head_dim", 8),
        _kv_f32("gemma4.attention.layer_norm_rms_epsilon", 1e-6),
        _kv_u32("gemma4.attention.sliding_window", 16),
        _kv_u32("gemma4.attention.head_dim", 4),
        _kv_f32("gemma4.rope.freq_base", 10000.0),
        _kv_f32("gemma4.rope.freq_base_global", 1000000.0),
        _kv_f32("gemma4.rope.partial_rotary_factor", 0.25),
        _kv_str("tokenizer.ggml.model", "llama"),
        _kv_str_array("tokenizer.ggml.tokens", SPM_TOKENS),
        _kv_f32_array("tokenizer.ggml.scores", SPM_SCORES),
        _kv_i32_array("tokenizer.ggml.token_type", SPM_TYPES),
        _kv_u32("tokenizer.ggml.bos_token_id", 2),
        _kv_u32("tokenizer.ggml.eos_token_id", 1),
        _kv_str_array("gemma4.layer_types", ["sliding_attention"]),
    ]
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
        blob = np.asarray(arr, dtype="<f4").tobytes()
        blobs.append(blob)
        offset += len(blob)
    head = header + infos
    pad = (-len(head)) % 32
    with open(path, "wb") as f:
        f.write(head + b"\x00" * pad + b"".join(blobs))
    return path


def test_gemma4_block_math_matches_hand_reference(tmp_path):
    """Independent numpy reference for the CORRECTED Gemma-4 block
    (llama.cpp gemma4.cpp): pre-norm, QK-norm per head, weightless
    V-norm, attention scale 1.0, post-norms, and
    layer_output_scale on the WHOLE layer output. Weights are
    chosen to make every corrected term sharp: post-norms RMS 2
    and scale 0.3 (so omitting either moves the hidden state by
    2-3x), and the real blk.0 QK-norm constants (1.0234375 /
    0.12207031). Any revert of one of the corrections diverges far
    beyond the 1e-4 tolerance."""
    rng = np.random.RandomState(3)

    def w(shape):
        return (rng.standard_normal(shape) * 0.05).astype(np.float32)

    wts = {
        "emb": w((8, 16)),
        "out_norm": rng.normal(1, 0.2, 16).astype(np.float32),
        "attn_norm": rng.normal(1, 0.2, 16).astype(np.float32),
        "post_attn": rng.normal(0, 2, 16).astype(np.float32),
        "scale": np.array([0.3], dtype=np.float32),
        "q_norm": np.full(4, 1.0234375, dtype=np.float32),
        "k_norm": np.full(4, 0.12207031, dtype=np.float32),
        "attn_q": w((16, 16)),
        "attn_k": w((8, 16)),
        "attn_v": w((8, 16)),
        "attn_output": w((16, 16)),
        "ffn_norm": rng.normal(1, 0.2, 16).astype(np.float32),
        "post_ffw": rng.normal(0, 2, 16).astype(np.float32),
        "ffn_gate": w((24, 16)),
        "ffn_up": w((24, 16)),
        "ffn_down": w((16, 24)),
    }
    gguf = _build_one_layer_sliding_gguf(
        os.path.join(tmp_path, "one.gguf"), wts)
    m = PagedGemma4(gguf, hot_budget_bytes=1024 * 1024)
    try:
        ids = [3, 4, 6]
        got = m.forward(ids, 0)
    finally:
        m.close()

    # Independent reference, float64, explicit loops for attention.
    def rmsnorm(x, wgt):
        ms = np.mean(x * x, axis=-1, keepdims=True)
        return (x / np.sqrt(ms + 1e-6)) * wgt

    def gelu_tanh(x):
        return 0.5 * x * (1.0 + np.tanh(
            0.7978845608028654 * (x + 0.044715 * x * x * x)))

    T = len(ids)
    x = wts["emb"][np.asarray(ids)].astype(np.float64) * 4.0  # sqrt(16)
    h = rmsnorm(x, wts["attn_norm"].astype(np.float64))
    q = h @ wts["attn_q"].astype(np.float64).T
    k = h @ wts["attn_k"].astype(np.float64).T
    v = h @ wts["attn_v"].astype(np.float64).T
    q = q.reshape(T, 4, 4)
    k = k.reshape(T, 2, 4)
    q = rmsnorm(q, wts["q_norm"].astype(np.float64))
    k = rmsnorm(k, wts["k_norm"].astype(np.float64))
    # V-norm: weightless per-head RMSNorm (llama.cpp gemma4.cpp).
    v = rmsnorm(v.reshape(T, 2, 4), np.ones(4, dtype=np.float64))
    # RoPE half-rotation, full dim 4, theta 10000.
    inv_freq = 10000.0 ** (-np.arange(0, 4, 2) / 4)
    for name in ("q", "k"):
        t = q if name == "q" else k
        pos = np.arange(T, dtype=np.float64)
        ang = pos[:, None] * inv_freq[None, :]
        cos, sin = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
        x0, x1 = t[..., :2], t[..., 2:4]
        out = np.empty_like(t)
        out[..., :2] = x0 * cos - x1 * sin
        out[..., 2:4] = x0 * sin + x1 * cos
        if name == "q":
            q = out
        else:
            k = out
    k = np.repeat(k, 2, axis=1)
    v = np.repeat(v, 2, axis=1)
    att = np.zeros((T, 4, 4), dtype=np.float64)
    for t in range(T):
        for hh in range(4):
            scores = np.array([float(np.dot(q[t, hh], k[s, hh]))
                               if s <= t else -np.inf
                               for s in range(T)])
            mx = scores.max()
            ex = np.exp(scores - mx)
            probs = ex / ex.sum()
            att[t, hh] = sum(probs[s] * v[s, hh] for s in range(T))
    a = att.reshape(T, 16) @ wts["attn_output"].astype(np.float64).T
    a = rmsnorm(a, wts["post_attn"].astype(np.float64))
    x1 = x + a
    h2 = rmsnorm(x1, wts["ffn_norm"].astype(np.float64))
    gate = h2 @ wts["ffn_gate"].astype(np.float64).T
    up = h2 @ wts["ffn_up"].astype(np.float64).T
    mm = (gelu_tanh(gate) * up) @ wts["ffn_down"].astype(np.float64).T
    mm = rmsnorm(mm, wts["post_ffw"].astype(np.float64))
    # layer_output_scale multiplies the WHOLE layer output
    # (llama.cpp gemma4.cpp), residual included — not each branch.
    x = (x1 + mm) * float(wts["scale"][0])
    last = rmsnorm(x[-1:], wts["out_norm"].astype(np.float64))
    want = (last @ wts["emb"].astype(np.float64).T)[0]
    assert np.allclose(got, want, atol=1e-4, rtol=1e-4), (
        f"max logit diff {np.max(np.abs(got - want))}")
