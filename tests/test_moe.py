"""MoE executor Stage-0 tests: synthetic-GGUF router unit tests.

No model file is downloaded. Builds tiny synthetic MoE GGUFs in a tmp
dir (1 layer, hidden 16, ffn 32, 4 experts, top-k 2, F32 tensors) and
pins, per docs/MOE_EXECUTOR_DESIGN.md section 10 Stage 0:
- router logits for a known hidden state (hand-computed values);
- top-k selection including an exact tie (the section 3 tie rule);
- grouped vs ungrouped expert outputs identical (tight tolerance:
  BLAS row-blocking can differ in last bits across GEMM shapes, so
  the pin is 1e-6 absolute, not bit equality — the grouping math,
  not the BLAS kernel, is what is under test);
- shared-expert term (synthetic only — proves the code path exists);
- expert audit: demanded experts' pages resident, no unselected
  expert's rows ever dequantized (see the page-sharing note in
  test_no_unselected_expert_touched);
- fail loud on unknown architecture, missing expert tensors, and
  wrong stacked dims.

All weights are F32 (native precision, no quantization in the loop):
Stage 0 proves the routing/paging mechanism, not any quantizer.
"""

import os
import struct

import numpy as np
import pytest

from lacuna.moe import PagedMoE, PagedMoEError


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


def _kv_bool(key: str, val: bool) -> bytes:
    return _s(key) + struct.pack("<I", 7) + struct.pack("<?", bool(val))


def build_tiny_moe_gguf(path: str, shared: bool = False,
                        router_weight=None, router_bias=None,
                        arch: str = "qwen3moe", drop=(),
                        gate_exps_dims=None, attn_bias_meta=None,
                        qknorm_value=None) -> str:
    """Tiny synthetic MoE GGUF. Tensor order is load-bearing for the
    shared-expert test: shared tensors are appended LAST, so the
    shared/non-shared pair draws identical weights for every other
    tensor from the same seed."""
    rng = np.random.RandomState(0)

    def w(shape):
        return (rng.standard_normal(shape) * 0.05).astype("<f4")

    # (name, dims [fastest-first], logical array). Stacked expert
    # tensors are stored as (n_experts, out_len, row_len) row-major,
    # so the flattened (n_experts * out_len, row_len) view gives
    # expert e exactly rows [e * out_len, (e + 1) * out_len).
    router_w = (
        np.asarray(router_weight, dtype="<f4")
        if router_weight is not None
        else w((4, 16))
    )
    tensors = [
        ("token_embd.weight", [16, 4], w((4, 16))),
        ("output.weight", [16, 4], w((4, 16))),
        ("output_norm.weight", [16], w((16,))),
        ("blk.0.attn_norm.weight", [16], w((16,))),
        ("blk.0.ffn_norm.weight", [16], w((16,))),
        ("blk.0.attn_q.weight", [16, 16], w((16, 16))),
        ("blk.0.attn_q.bias", [16], w((16,))),
        ("blk.0.attn_k.weight", [16, 8], w((8, 16))),
        ("blk.0.attn_k.bias", [8], w((8,))),
        ("blk.0.attn_v.weight", [16, 8], w((8, 16))),
        ("blk.0.attn_v.bias", [8], w((8,))),
        ("blk.0.attn_output.weight", [16, 16], w((16, 16))),
        # QK-norm tensors (qwen3moe): head_dim = 16 / 2 heads = 8.
        # Appended at the end of the base list so every earlier draw
        # (and every pinned value in the older tests) is unchanged.
        ("blk.0.attn_q_norm.weight", [8],
         np.full((8,), qknorm_value, dtype="<f4")
         if qknorm_value is not None else w((8,))),
        ("blk.0.attn_k_norm.weight", [8],
         np.full((8,), qknorm_value, dtype="<f4")
         if qknorm_value is not None else w((8,))),
        ("blk.0.ffn_gate_inp.weight", [16, 4], router_w),
        ("blk.0.ffn_gate_exps.weight",
         gate_exps_dims or [16, 32, 4], w((4, 32, 16))),
        ("blk.0.ffn_up_exps.weight", [16, 32, 4], w((4, 32, 16))),
        ("blk.0.ffn_down_exps.weight", [32, 16, 4], w((4, 16, 32))),
    ]
    if router_bias is not None:
        tensors.append(("blk.0.ffn_gate_inp.bias", [4],
                        np.asarray(router_bias, dtype="<f4")))
    if shared:
        tensors += [
            ("blk.0.ffn_gate_shexp.weight", [16, 32], w((32, 16))),
            ("blk.0.ffn_up_shexp.weight", [16, 32], w((32, 16))),
            ("blk.0.ffn_down_shexp.weight", [32, 16], w((16, 32))),
        ]
    tensors = [t for t in tensors if t[0] not in drop]
    kvs = [
        _kv_str("general.architecture", arch),
        _kv_u32("qwen3moe.block_count", 1),
        _kv_u32("qwen3moe.embedding_length", 16),
        _kv_u32("qwen3moe.feed_forward_length", 32),
        _kv_u32("qwen3moe.attention.head_count", 2),
        _kv_u32("qwen3moe.attention.head_count_kv", 1),
        _kv_f32("qwen3moe.attention.layer_norm_rms_epsilon", 1e-6),
        _kv_f32("qwen3moe.rope.freq_base", 10000.0),
        _kv_u32("qwen3moe.expert_count", 4),
        _kv_u32("qwen3moe.expert_used_count", 2),
        _kv_str("tokenizer.ggml.model", "gpt2"),
        _kv_str_array("tokenizer.ggml.tokens", ["a", "b", "Ġ", "ab"]),
        _kv_str_array("tokenizer.ggml.merges", []),
    ]
    if attn_bias_meta is not None:
        # Written with the arch prefix, so the executor's
        # "{arch}.attention.bias" lookup finds it.
        kvs.append(_kv_bool(f"{arch}.attention.bias", attn_bias_meta))
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
        blob = np.ascontiguousarray(arr).tobytes()
        blobs.append(blob)
        offset += len(blob)
    head = header + infos
    pad = (-len(head)) % 32
    with open(path, "wb") as f:
        f.write(head + b"\x00" * pad + b"".join(blobs))
    return path


def _load(tmp_path, filename="tiny_moe.gguf", **kw):
    gguf = build_tiny_moe_gguf(os.path.join(tmp_path, filename), **kw)
    return PagedMoE(gguf, hot_budget_bytes=64 * 1024 * 1024)


def test_router_logits_hand_computed(tmp_path):
    # Router rows pick out hidden coordinates 0..3 with scales 1..4;
    # h = [1, 1, 1, 1, 0, ...] gives logits exactly [1, 2, 3, 4].
    router_w = np.zeros((4, 16), dtype="<f4")
    for i in range(4):
        router_w[i, i] = float(i + 1)
    m = _load(tmp_path, router_weight=router_w)
    try:
        h = np.zeros((1, 16), dtype=np.float32)
        h[0, :4] = 1.0
        logits = m.router_logits(h, 0)
        assert logits.shape == (1, 4)
        assert np.allclose(logits[0], [1.0, 2.0, 3.0, 4.0])
    finally:
        m.close()


def test_router_bias_applied(tmp_path):
    router_w = np.zeros((4, 16), dtype="<f4")
    for i in range(4):
        router_w[i, i] = float(i + 1)
    m = _load(tmp_path, router_weight=router_w,
              router_bias=[0.5, -0.5, 0.0, 0.0])
    try:
        h = np.zeros((1, 16), dtype=np.float32)
        h[0, :4] = 1.0
        logits = m.router_logits(h, 0)
        assert np.allclose(logits[0], [1.5, 1.5, 3.0, 4.0])
    finally:
        m.close()


def test_select_experts_tie_and_renorm():
    # Logits [3, 1, 1, -2]: experts 1 and 2 tie on score; the tie
    # breaks to the lower index, so top-2 is {0, 1}, never {0, 2}.
    ids, weights = PagedMoE.select_experts(
        np.array([[3.0, 1.0, 1.0, -2.0]], dtype=np.float32), 2
    )
    assert ids.tolist() == [[0, 1]]
    e3, e1 = np.exp(3.0), np.exp(1.0)
    assert np.allclose(weights[0], [e3 / (e3 + e1), e1 / (e3 + e1)], atol=1e-6)
    assert np.allclose(weights.sum(axis=-1), [1.0])
    # Exact tie on all four: lowest indices win, weights split evenly.
    ids, weights = PagedMoE.select_experts(
        np.array([[2.0, 2.0, 0.0, 0.0]], dtype=np.float32), 2
    )
    assert ids.tolist() == [[0, 1]]
    assert np.allclose(weights[0], [0.5, 0.5], atol=1e-6)
    # Multi-position: each row selects independently.
    ids, _ = PagedMoE.select_experts(
        np.array([[0.0, 5.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 5.0]], dtype=np.float32), 2
    )
    assert ids.tolist() == [[1, 0], [3, 0]]
    with pytest.raises(PagedMoEError):
        PagedMoE.select_experts(np.zeros((1, 4), dtype=np.float32), 5)


def _silu(x):
    return x / (1.0 + np.exp(-x))


def test_grouped_equals_ungrouped(tmp_path):
    m = _load(tmp_path)
    try:
        rng = np.random.RandomState(7)
        h2 = (rng.standard_normal((5, 16)) * 0.5).astype(np.float32)
        grouped = m._moe_ffn(h2, 0)
        # Ungrouped reference: per position, per selected expert, one
        # single-row GEMM at a time, using the same pager reads.
        ids, w = PagedMoE.select_experts(m.router_logits(h2, 0), 2)
        ref = np.zeros_like(h2)
        for t in range(h2.shape[0]):
            for s in range(2):
                e = int(ids[t, s])
                gu = h2[t] @ np.concatenate([
                    m._expert_rows(0, "gate", e),
                    m._expert_rows(0, "up", e),
                ]).T
                gate, up = gu[:32], gu[32:]
                ref[t] += w[t, s] * ((_silu(gate) * up) @ m._expert_rows(0, "down", e).T)
        assert grouped.shape == ref.shape
        assert np.isfinite(grouped).all()
        assert np.allclose(grouped, ref, atol=1e-6, rtol=1e-5)
        assert np.max(np.abs(grouped - ref)) < 1e-5
    finally:
        m.close()


def test_no_unselected_expert_touched(tmp_path):
    # Generous budget: everything faulted stays resident, so the
    # page audit is exact. PAGE-SHARING NOTE, stated honestly: at
    # this tiny scale each stacked tensor fits in ONE page, so all
    # experts share their role's page — page residency alone cannot
    # distinguish experts here (it can at real scale, where experts
    # span many exclusive pages). The precise check at every scale
    # is the dequant cache: only demanded experts' rows are ever
    # dequantized, so only their fused entries may appear in _dq.
    m = _load(tmp_path)
    try:
        m.forward([0, 1, 2], 0)
        ids, _ = m.last_routed[0]
        demanded = set(int(i) for i in ids.ravel())
        assert demanded, "forward routed to no experts"
        # 1. Demanded experts' pages are resident.
        resident = set(m.store.hot_pages())
        for e in demanded:
            assert set(m._expert_page_ids(0, [e])) & resident
        # 2. Resident expert pages are a subset of demanded experts'
        #    pages (nothing outside the demanded set was faulted).
        all_expert_pages = set(m._expert_page_ids(0, list(range(4))))
        demanded_pages = set(m._expert_page_ids(0, sorted(demanded)))
        assert (resident & all_expert_pages) <= demanded_pages
        # 3. The precise check: _dq holds fused entries ONLY for
        #    demanded experts — unselected experts were never read.
        dq_experts = set()
        for key in m._dq:
            if ".expert:" in key:
                dq_experts.add(int(key.split(".expert:")[1].split(":")[0]))
        assert dq_experts == demanded
        assert m.store.stats["faults"] > 0  # weights really came via pager
    finally:
        m.close()
    # Part B: the negative case, forced. In Part A every expert
    # happened to be demanded, so the "unselected never appears"
    # half was vacuous. Here the router can only select experts
    # {0, 1}: h2 = [5, 4, 0, ...] against coordinate-picking rows
    # gives logits [5, 8, 0, 0]. Experts 2 and 3 must never appear
    # in _dq — the page-level negative is still uncheckable at this
    # scale (shared role pages), but the row/dequant level is exact.
    router_w = np.zeros((4, 16), dtype="<f4")
    for i in range(4):
        router_w[i, i] = float(i + 1)
    m2 = _load(tmp_path, filename="tiny_moe_crafted.gguf",
               router_weight=router_w)
    try:
        h2 = np.zeros((3, 16), dtype=np.float32)
        h2[:, 0] = 5.0
        h2[:, 1] = 4.0
        m2._moe_ffn(h2, 0)
        ids2, _ = m2.last_routed[0]
        assert set(int(i) for i in ids2.ravel()) == {0, 1}
        dq2 = set()
        for key in m2._dq:
            if ".expert:" in key:
                dq2.add(int(key.split(".expert:")[1].split(":")[0]))
        assert dq2 == {0, 1}  # experts 2, 3 never read, never dequantized
    finally:
        m2.close()


def test_shared_expert_path(tmp_path):
    # Shared tensors are appended last by the builder, so both models
    # draw identical weights for every other tensor from the seed.
    m_plain = _load(tmp_path)
    gguf_s = build_tiny_moe_gguf(
        os.path.join(tmp_path, "tiny_moe_shared.gguf"), shared=True
    )
    m_shared = PagedMoE(gguf_s, hot_budget_bytes=64 * 1024 * 1024)
    try:
        assert "blk.0.ffn_gate_shexp.weight" not in m_plain.by_name
        assert "blk.0.ffn_gate_shexp.weight" in m_shared.by_name
        out_plain = m_plain.forward([0, 1, 2], 0)
        out_shared = m_shared.forward([0, 1, 2], 0)
        assert np.isfinite(out_plain).all() and np.isfinite(out_shared).all()
        # The shared term contributes: identical models except the
        # shared term must not produce identical logits.
        assert not np.allclose(out_plain, out_shared, atol=1e-6)
        # Shared tensors were read through the pager (pages resident).
        tid = m_shared.store.tensor_id("blk.0.ffn_gate_shexp.weight")
        assert tid is not None
        shexp_pages = {e["page_id"] for e in m_shared.store.tensor_pages(tid)}
        assert shexp_pages & set(m_shared.store.hot_pages())
    finally:
        m_plain.close()
        m_shared.close()


def test_fails_loud(tmp_path):
    # Unknown architecture (DeepSeek-style arch name) fails at load,
    # before any tensor is touched.
    gguf = build_tiny_moe_gguf(
        os.path.join(tmp_path, "bad_arch.gguf"), arch="deepseek2"
    )
    with pytest.raises(PagedMoEError):
        PagedMoE(gguf, hot_budget_bytes=64 * 1024 * 1024)
    # Missing expert tensor fails at load.
    gguf = build_tiny_moe_gguf(
        os.path.join(tmp_path, "missing.gguf"),
        drop=("blk.0.ffn_gate_exps.weight",),
    )
    with pytest.raises(PagedMoEError):
        PagedMoE(gguf, hot_budget_bytes=64 * 1024 * 1024)
    # Wrong stacked dims (3 experts claimed in dims, 4 in metadata).
    gguf = build_tiny_moe_gguf(
        os.path.join(tmp_path, "dims.gguf"), gate_exps_dims=[16, 32, 3]
    )
    with pytest.raises(PagedMoEError):
        PagedMoE(gguf, hot_budget_bytes=64 * 1024 * 1024)


def test_tiny_forward_finite_and_generate(tmp_path):
    m = _load(tmp_path)
    try:
        logits = m.forward([0, 1, 2], 0)
        assert logits.shape == (4,)
        assert np.isfinite(logits).all()
        text1, ids1 = m.generate("ab a", max_tokens=4)
        text2, ids2 = m.generate("ab a", max_tokens=4)
        assert ids1 == ids2 and len(ids1) == 4  # no EOS: full run
        assert text1 == text2 and text1
    finally:
        m.close()


# ------------------------------------------------------- qwen3moe: QK-norm


def test_qk_norm_hand_computed(tmp_path):
    """QK-norm primitive, hand-computed in float64.

    RMSNorm over the last axis: x / sqrt(mean(x^2) + eps) * w, with
    the model's eps (1e-6, from the builder's metadata). x = [3, 4,
    0, ...] over head_dim 8 gives mean-square 25/8; the weights then
    pin the multiply, not just the divide. This is the op the
    qwen3moe forward applies to Q (per Q head) and K (per KV head)
    BEFORE RoPE — llama.cpp applies q_norm/k_norm to the projections
    before the rotary embedding, and the forward here does the same.
    """
    from lacuna.paged_model import PagedQwen2

    m = _load(tmp_path)
    try:
        assert m.qk_norm is True
        x = np.zeros((1, 1, 8), dtype=np.float32)
        x[0, 0, 0] = 3.0
        x[0, 0, 1] = 4.0
        got = PagedQwen2._rmsnorm(m, x, np.ones((8,), dtype=np.float32))
        rms = np.sqrt(25.0 / 8.0 + 1e-6)
        want = np.zeros((1, 1, 8), dtype=np.float64)
        want[0, 0, 0] = 3.0 / rms
        want[0, 0, 1] = 4.0 / rms
        assert np.allclose(got, want, atol=1e-6)
        # Non-trivial weights: the multiply is pinned too.
        w2 = np.arange(1, 9, dtype=np.float32)  # [1..8]
        got2 = PagedQwen2._rmsnorm(m, x, w2)
        assert np.allclose(got2[0, 0, :2], want[0, 0, :2] * [1.0, 2.0],
                           atol=1e-6)
        assert np.allclose(got2[0, 0, 2:], 0.0, atol=1e-6)
    finally:
        m.close()


def test_qk_norm_live_in_forward_and_deterministic(tmp_path):
    """The QK-norm path is live, and the qwen3moe forward is
    deterministic. Two files identical except q_norm/k_norm weights
    (all ones vs all twos) must give different logits — if the norm
    were dead code, they would be identical. Two fresh models from
    the same file must give bit-identical logits."""
    m1 = _load(tmp_path, filename="qn1.gguf", qknorm_value=1.0)
    m2 = _load(tmp_path, filename="qn2.gguf", qknorm_value=2.0)
    try:
        l1 = m1.forward([0, 1, 2], 0)
        l2 = m2.forward([0, 1, 2], 0)
        assert np.isfinite(l1).all() and np.isfinite(l2).all()
        assert not np.allclose(l1, l2, atol=1e-6)  # norm scales q.k by 4x
    finally:
        m1.close()
        m2.close()
    ma = _load(tmp_path, filename="qn1b.gguf", qknorm_value=1.0)
    mb = _load(tmp_path, filename="qn1c.gguf", qknorm_value=1.0)
    try:
        assert np.array_equal(ma.forward([0, 1, 2], 0),
                              mb.forward([0, 1, 2], 0))
    finally:
        ma.close()
        mb.close()


# ------------------------------------------------------- attention bias


_BIASES = ("blk.0.attn_q.bias", "blk.0.attn_k.bias", "blk.0.attn_v.bias")


def test_attention_bias_metadata_matrix(tmp_path):
    """attention_bias is read, never assumed. Metadata wins when
    present and must agree with the tensors; absent metadata infers
    from uniform, complete tensor presence."""
    # (a) Metadata True + biases present -> True.
    m = _load(tmp_path, filename="b1.gguf", attn_bias_meta=True)
    try:
        assert m.attention_bias is True
    finally:
        m.close()
    # (b) Metadata False + biases dropped -> False (Qwen3-235B case:
    # attention_bias=False, no bias tensors). Forward must run.
    m = _load(tmp_path, filename="b2.gguf", attn_bias_meta=False,
              drop=_BIASES)
    try:
        assert m.attention_bias is False
        logits = m.forward([0, 1, 2], 0)
        assert np.isfinite(logits).all()
    finally:
        m.close()
    # (c) Metadata True + biases dropped -> contradiction, fail loud.
    with pytest.raises(PagedMoEError):
        _load(tmp_path, filename="b3.gguf", attn_bias_meta=True,
              drop=_BIASES)
    # (d) Metadata False + biases present -> contradiction, fail loud.
    with pytest.raises(PagedMoEError):
        _load(tmp_path, filename="b4.gguf", attn_bias_meta=False)
    # (e) No metadata + biases present -> inferred True.
    m = _load(tmp_path, filename="b5.gguf")
    try:
        assert m.attention_bias is True
    finally:
        m.close()
    # (f) No metadata + biases dropped -> inferred False.
    m = _load(tmp_path, filename="b6.gguf", drop=_BIASES)
    try:
        assert m.attention_bias is False
    finally:
        m.close()
    # (g) Partial bias set (k dropped only), no metadata -> fail loud.
    with pytest.raises(PagedMoEError):
        _load(tmp_path, filename="b7.gguf", drop=("blk.0.attn_k.bias",))
    # (h) qwen3moe without q_norm -> fail loud at load.
    with pytest.raises(PagedMoEError):
        _load(tmp_path, filename="b8.gguf",
              drop=("blk.0.attn_q_norm.weight",))
    with pytest.raises(PagedMoEError):
        _load(tmp_path, filename="b9.gguf",
              drop=("blk.0.attn_k_norm.weight",))
