"""Paged MoE executor (v1): routed experts through the pager.

Implements ``docs/MOE_EXECUTOR_DESIGN.md`` Stage 0 + the core
router/expert-paging path. This is the CPU, demand-driven, exact
executor: router runs, top-k experts fault on demand, compute never
observes a missing page. No predictor ships in v1 (rung 2/3 are
shadow-gated designs, not code here).

Design rules, carried over from the dense executor (non-negotiable):
- No direct file reads of weights anywhere. Every weight byte arrives
  through :class:`~lacuna.pager.PagedWeightStore` via
  ``lacuna.gguf.read_tensor`` / ``read_tensor_rows``; ``store.stats``
  is the audit trail.
- Exact policy only (``read_tensor`` enforces it): the .lac carries
  source bytes bit-for-bit, so dequantization is the model's own
  quantization, not a Lacuna re-quantization.
- Greedy decoding only (argmax), directly comparable to
  ``llama-cli --temp 0 --top-k 1``.
- Hard budgets, reported separately (page cache vs dequant cache).
- Fail loud, never silent: unknown architecture, unknown router rule,
  missing expert tensors, or wrong stacked dims raise
  :class:`PagedMoEError`.

Architecture implemented (v1):
- Attention, norms, embed/head slicing: the SAME code as the dense
  executor, not a copy. ``PagedMoE`` provides the attribute contract
  ``PagedQwen2``'s primitives need (``_w``, ``_dq``, ``_embed_info``,
  ``_head_info``, ``eps``, ``_inv_freq``, ``hidden``, ``store``) and
  calls those primitives unbound. If ``PagedQwen2``'s attribute
  contract changes, the MoE tests fail loud instead of silently
  diverging. This keeps attention bit-behavior identical to the
  verified dense path by construction.
- Router (v1): softmax over router logits -> top-k -> renormalize
  over the selected k (Mixtral / Qwen-MoE / OLMoE family). Tie rule:
  higher score first, lower expert index first (stable sort). Any
  other router rule (e.g. DeepSeek sigmoid + selection bias) raises
  until implemented + verified — rung 3, not v1.
- Experts: GGUF stacked 3-D tensors, dims fastest-first
  ``[row_length, out_length, n_experts]``. Expert ``e`` is one
  contiguous byte range, i.e. rows ``[e * out_length,
  (e+1) * out_length)`` of the flattened 2-D view
  ``(n_experts * out_length, row_length)`` — read through the
  existing verified ``read_tensor_rows`` on that flattened view.
  No converter change is needed: conversion already copies GGUF
  bytes verbatim (exact policy).
- QK-norm (qwen3moe only): per-layer ``attn_q_norm`` /
  ``attn_k_norm`` RMSNorm weights applied to Q (per Q head) and K
  (per KV head) BEFORE RoPE, with the model's RMS epsilon — the
  same epsilon as the attention norm (llama.cpp uses
  ``f_norm_rms_eps`` for both). Without it attention scores are
  wrong, not near-tie noise. A qwen3moe file missing either tensor
  on any layer fails at load. OLMoE has no QK-norm and skips it.
- Attention bias: read, never assumed. The metadata flag
  ``{arch}.attention.bias`` (the HF config's ``attention_bias``,
  when a converter writes it) wins when present, and must agree
  with the bias tensors' presence or load fails. When the flag is
  absent — the llama.cpp converter writes no such key — bias is
  inferred from tensor presence, which must be uniform across
  layers and complete across q/k/v, else load fails. Qwen3-235B's
  case (``attention_bias=False``, no bias tensors) runs with zero
  biases; a Qwen2-style file with all three biases runs with them.
- Grouped expert GEMM: positions are grouped by expert within each
  layer; one fused gate+up GEMM + one down GEMM per expert per
  group, weighted scatter-add. Unselected experts cost zero faults
  and zero FLOPs — they are never touched.
- Shared experts (optional): dense tensors ``ffn_gate_shexp`` /
  ``ffn_up_shexp`` / ``ffn_down_shexp`` run as one dense term over
  all positions. v1 requires their intermediate size to equal the
  routed experts' ``ffn`` size, else it fails loud (real DeepSeek
  shared experts have their own size — rung 2/3).

What v1 does NOT do (scope fence, per the design): no MLA attention
(DeepSeek is rung 3), no sigmoid+bias routing, no MXFP4/FP8-native
quantizers in the exact tier, no GPU execution, no predictor.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from .bpe import BPETokenizer
from .convert import convert as _convert
from .gguf import (
    _TYPE_INFO,
    parse_gguf,
    read_tensor,
    read_tensor_rows,
)
from .paged_model import PagedQwen2
from .pager import PagedWeightStore


class PagedMoEError(ValueError):
    """Raised for unsupported architectures, router rules, or bad MoE
    tensors. Fails loud."""


# v1: the softmax -> top-k -> renormalize family only. The arch prefix
# doubles as the metadata prefix (e.g. "qwen3moe.block_count").
_SUPPORTED_ARCHS = ("qwen3moe", "olmoe")


class PagedMoE:
    """MoE decoder whose weights live in a PagedWeightStore."""

    def __init__(self, gguf_path: str, lac_path: str | None = None,
                 hot_budget_bytes: int = 256 * 1024 * 1024,
                 dequant_budget_bytes: int | None = None):
        md, infos, _ = parse_gguf(gguf_path)
        arch = md.get("general.architecture")
        if arch not in _SUPPORTED_ARCHS:
            raise PagedMoEError(
                f"paged MoE executor supports {_SUPPORTED_ARCHS}, got "
                f"{arch!r} (DeepSeek sigmoid+bias routing and MLA are "
                f"rung 3, not v1)"
            )
        p = arch + "."
        self.arch = arch
        self.n_layer = int(md[p + "block_count"])
        self.hidden = int(md[p + "embedding_length"])
        self.ffn = int(md[p + "feed_forward_length"])
        self.n_head = int(md[p + "attention.head_count"])
        self.n_kv = int(md[p + "attention.head_count_kv"])
        self.eps = float(md[p + "attention.layer_norm_rms_epsilon"])
        self.rope_theta = float(md.get(p + "rope.freq_base", 1000000.0))
        self.n_experts = int(md[p + "expert_count"])
        self.top_k = int(md[p + "expert_used_count"])
        if not 1 <= self.top_k <= self.n_experts:
            raise PagedMoEError(
                f"top_k {self.top_k} outside [1, {self.n_experts}]"
            )
        self.head_dim = self.hidden // self.n_head
        if self.head_dim * self.n_head != self.hidden:
            raise PagedMoEError("hidden size not divisible by head count")
        if self.n_head % self.n_kv:
            raise PagedMoEError("head count not divisible by kv head count")
        self.group = self.n_head // self.n_kv
        self.tok = BPETokenizer.from_metadata(md)
        self.eos = md.get("tokenizer.ggml.eos_token_id")
        self.by_name = {i["name"]: i for i in infos}
        for name in ("token_embd.weight", "output.weight", "output_norm.weight"):
            if name not in self.by_name:
                raise PagedMoEError(f"required tensor {name} missing")
        # Attention bias + QK-norm are resolved here, at load, so a
        # bad file fails before any forward — never mid-generation.
        self.attention_bias = self._resolve_attention_bias(
            md, p, self.by_name, self.n_layer
        )
        self.qk_norm = arch == "qwen3moe"
        if self.qk_norm:
            for li in range(self.n_layer):
                base = f"blk.{li}."
                for role in ("attn_q_norm", "attn_k_norm"):
                    info = self.by_name.get(base + role + ".weight")
                    if info is None:
                        raise PagedMoEError(
                            f"layer {li}: qwen3moe requires {base + role}"
                            f".weight (QK-norm), tensor missing"
                        )
                    dims = [int(d) for d in info["dims"]]
                    if len(dims) != 1 or dims[0] != self.head_dim:
                        raise PagedMoEError(
                            f"layer {li}: {base + role}.weight has dims "
                            f"{dims}, expected [{self.head_dim}]"
                        )
        # Per-layer MoE tensor validation at load (fail loud, not at
        # first forward): router + stacked experts must exist with the
        # stacked dims the reader view assumes.
        for li in range(self.n_layer):
            base = f"blk.{li}."
            if base + "ffn_gate_inp.weight" not in self.by_name:
                raise PagedMoEError(f"layer {li}: router weight missing")
            for role, row_len, out_len in (
                ("gate", self.hidden, self.ffn),
                ("up", self.hidden, self.ffn),
                ("down", self.ffn, self.hidden),
            ):
                name = base + f"ffn_{role}_exps.weight"
                info = self.by_name.get(name)
                if info is None:
                    raise PagedMoEError(f"layer {li}: {name} missing")
                dims = [int(d) for d in info["dims"]]
                if len(dims) != 3 or dims != [row_len, out_len, self.n_experts]:
                    raise PagedMoEError(
                        f"layer {li}: {name} has dims {dims}, expected "
                        f"[{row_len}, {out_len}, {self.n_experts}]"
                    )
            for role, row_len, out_len in (
                ("gate_shexp", self.hidden, self.ffn),
                ("up_shexp", self.hidden, self.ffn),
                ("down_shexp", self.ffn, self.hidden),
            ):
                name = base + f"ffn_{role}.weight"
                info = self.by_name.get(name)
                if info is None:
                    continue
                dims = [int(d) for d in info["dims"]]
                if len(dims) != 2 or dims != [row_len, out_len]:
                    raise PagedMoEError(
                        f"layer {li}: shared expert {name} has dims "
                        f"{dims}, expected [{row_len}, {out_len}] "
                        f"(v1 requires shared ffn == routed ffn)"
                    )

        lac = lac_path or (gguf_path + ".lac")
        import os as _os

        if not _os.path.exists(lac):
            _convert(gguf_path, lac, quiet=True)
        self.store = PagedWeightStore(lac, hot_budget_bytes=int(hot_budget_bytes))
        # Same slicing contract as the dense executor: only the final
        # norm is held; embed and head are sliced per call.
        self._embed_info = self.by_name["token_embd.weight"]
        self._head_info = self.by_name["output.weight"]
        self.out_norm = read_tensor(self.store, self.by_name["output_norm.weight"])
        self.cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        # Dequantized-weight cache: same bounded-LRU, separately
        # reported pattern as the dense executor. Expert entries are
        # cached per expert, per role (fused gate+up + down).
        self._dq: OrderedDict[str, np.ndarray] = OrderedDict()
        self._dq_bytes = 0
        self._dq_budget = (
            int(dequant_budget_bytes)
            if dequant_budget_bytes is not None
            else int(hot_budget_bytes)
        )
        self._inv_freq = self.rope_theta ** (
            -np.arange(0, self.head_dim, 2, dtype=np.float32) / self.head_dim
        )
        # Diagnostic state, updated per forward: layer -> (ids, weights)
        # arrays of shape (T, top_k). Used by the Stage-0 fault audit
        # (demanded vs faulted expert sets); never a correctness input.
        self.last_routed: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    # -- weight access (pager only, same pattern as dense) -------------
    def _dq_put(self, name: str, arr: np.ndarray) -> None:
        nbytes = int(arr.nbytes)
        if nbytes > self._dq_budget:
            return  # too big to ever cache; caller still uses it
        while self._dq_bytes + nbytes > self._dq_budget and self._dq:
            _, old = self._dq.popitem(last=False)
            self._dq_bytes -= int(old.nbytes)
        self._dq[name] = arr
        self._dq_bytes += nbytes

    def _dq_drop(self, name: str) -> None:
        arr = self._dq.pop(name, None)
        if arr is not None:
            self._dq_bytes -= int(arr.nbytes)

    def _w(self, name: str) -> np.ndarray:
        hit = self._dq.get(name)
        if hit is not None:
            self._dq.move_to_end(name)
            return hit
        info = self.by_name.get(name)
        if info is None:
            raise PagedMoEError(f"required tensor {name} missing")
        arr = read_tensor(self.store, info)
        self._dq_put(name, arr)
        return arr

    # -- attention config (bias + QK-norm) -----------------------------
    @staticmethod
    def _resolve_attention_bias(md, p: str, by_name: dict,
                                n_layer: int) -> bool:
        """Whether Q/K/V biases exist, read — never assumed.

        Layered rule, in order:
        1. Metadata flag ``{arch}.attention.bias`` (bool) wins when
           present, but must agree with the tensors: flag True with
           any bias tensor missing, or flag False with bias tensors
           present, is a contradiction and fails loud.
        2. Flag absent (what the llama.cpp converter produces — it
           writes no such key): infer from tensor presence. Presence
           must be uniform across layers and complete across q/k/v;
           a partial set (some layers, or q without k/v) fails loud.
        """
        per_layer = []
        for li in range(n_layer):
            base = f"blk.{li}."
            per_layer.append(tuple(
                (base + s) in by_name
                for s in ("attn_q.bias", "attn_k.bias", "attn_v.bias")
            ))
        first = per_layer[0]
        if any(flags != first for flags in per_layer):
            raise PagedMoEError(
                "attention bias presence differs across layers "
                f"(layer 0: {first}) — corrupt or mixed file"
            )
        all_present = all(first)
        none_present = not any(first)
        if not (all_present or none_present):
            raise PagedMoEError(
                f"partial attention bias set on layer 0: q/k/v "
                f"present = {first} — a layer needs all three or none"
            )
        key = p + "attention.bias"
        if key in md:
            meta = bool(md[key])
            if meta and not all_present:
                raise PagedMoEError(
                    f"metadata {key}=true but attention bias tensors "
                    f"are missing"
                )
            if not meta and all_present:
                raise PagedMoEError(
                    f"metadata {key}=false but attention bias tensors "
                    f"are present — contradictory file"
                )
            return meta
        return all_present

    def _moe_qkv(self, base: str) -> tuple[np.ndarray, np.ndarray]:
        """One stacked QKV weight+bias for the MoE path, bias-aware.

        The dense executor's ``_fused_qkv`` unconditionally demands
        all three bias tensors, which is wrong for qwen3moe files
        with ``attention_bias=False`` (Qwen3-235B's case). This mirror
        keeps the same fused-cache pattern (components dropped from
        ``_dq`` once fused, so memory stays ~one copy), but the bias
        is the concatenated bias tensors when ``attention_bias`` is
        true and zeros otherwise.
        """
        key = base + "qkv"
        hit = self._dq.get(key)
        if hit is not None:
            self._dq.move_to_end(key)
            self._dq.move_to_end(key + ".bias")
            return hit, self._dq[key + ".bias"]
        w = np.concatenate([
            self._w(base + "attn_q.weight"),
            self._w(base + "attn_k.weight"),
            self._w(base + "attn_v.weight"),
        ])
        if self.attention_bias:
            b = np.concatenate([
                self._w(base + "attn_q.bias"),
                self._w(base + "attn_k.bias"),
                self._w(base + "attn_v.bias"),
            ])
        else:
            b = np.zeros((w.shape[0],), dtype=np.float32)
        for suffix in ("attn_q.weight", "attn_k.weight", "attn_v.weight",
                       "attn_q.bias", "attn_k.bias", "attn_v.bias"):
            self._dq_drop(base + suffix)
        self._dq_put(key, w)
        self._dq_put(key + ".bias", b)
        return w, b

    # -- router ---------------------------------------------------------
    @staticmethod
    def select_experts(router_logits: np.ndarray, top_k: int):
        """Softmax -> top-k -> renormalize (v1 rule).

        ``router_logits``: (T, n_experts) float32. Returns (ids, weights),
        both (T, top_k): ids are expert indices, weights are the
        renormalized softmax probabilities over the selected k.
        Tie rule: higher score first, lower expert index first — a
        stable sort of the negated scores keeps index order on ties.
        """
        logits = np.asarray(router_logits, dtype=np.float32)
        if logits.ndim != 2:
            raise PagedMoEError(
                f"select_experts needs (T, n_experts) logits, got {logits.shape}"
            )
        if not 1 <= top_k <= logits.shape[1]:
            raise PagedMoEError(
                f"top_k {top_k} outside [1, {logits.shape[1]}]"
            )
        x = logits - logits.max(axis=-1, keepdims=True)
        p = np.exp(x)
        p = p / p.sum(axis=-1, keepdims=True)
        order = np.argsort(-p, axis=-1, kind="stable")
        ids = order[:, :top_k]
        w = np.take_along_axis(p, ids, axis=-1)
        w = w / w.sum(axis=-1, keepdims=True)
        return ids, w.astype(np.float32)

    def router_logits(self, h: np.ndarray, li: int) -> np.ndarray:
        """Router logits for hidden states ``h`` (T, hidden) at layer ``li``."""
        base = f"blk.{li}."
        out = h @ self._w(base + "ffn_gate_inp.weight").T
        bias_name = base + "ffn_gate_inp.bias"
        if bias_name in self.by_name:
            out = out + self._w(bias_name)
        return out

    # -- expert reads (flattened-row view over stacked tensors) --------
    def _expert_flat_info(self, li: int, role: str):
        """2-D view of a stacked expert tensor for ``read_tensor_rows``.

        Stacked dims fastest-first are [row_len, out_len, n_experts];
        the flattened view is (n_experts * out_len, row_len), so expert
        ``e`` is exactly rows [e * out_len, (e+1) * out_len). Returns
        (flat_info, rows_per_expert).
        """
        base = f"blk.{li}."
        if role in ("gate", "up"):
            info = self.by_name[base + f"ffn_{role}_exps.weight"]
            rows_per_expert = self.ffn
        elif role == "down":
            info = self.by_name[base + "ffn_down_exps.weight"]
            rows_per_expert = self.hidden
        else:
            raise PagedMoEError(f"unknown expert role {role!r}")
        flat = {
            "name": info["name"],
            "dims": [info["dims"][0], info["dims"][1] * info["dims"][2]],
            "type": info["type"],
            "offset": info["offset"],
        }
        return flat, rows_per_expert

    def _expert_rows(self, li: int, role: str, e: int) -> np.ndarray:
        """Dequantized matrix of expert ``e`` for ``role`` at layer ``li``.

        Only the pages overlapping this expert's byte range are faulted.
        """
        if not 0 <= e < self.n_experts:
            raise PagedMoEError(f"expert {e} outside [0, {self.n_experts})")
        flat, rpe = self._expert_flat_info(li, role)
        return read_tensor_rows(self.store, flat, e * rpe, (e + 1) * rpe)

    def _expert_gateup(self, li: int, e: int) -> np.ndarray:
        """Fused gate+up matrix for expert ``e`` (2*ffn, hidden), cached."""
        key = f"blk.{li}.expert:{e}:gateup"
        hit = self._dq.get(key)
        if hit is not None:
            self._dq.move_to_end(key)
            return hit
        w = np.concatenate([
            self._expert_rows(li, "gate", e),
            self._expert_rows(li, "up", e),
        ])
        self._dq_put(key, w)
        return w

    def _expert_down(self, li: int, e: int) -> np.ndarray:
        """Down matrix for expert ``e`` (hidden, ffn), cached."""
        key = f"blk.{li}.expert:{e}:down"
        hit = self._dq.get(key)
        if hit is not None:
            self._dq.move_to_end(key)
            return hit
        w = self._expert_rows(li, "down", e)
        self._dq_put(key, w)
        return w

    def _expert_page_ids(self, li: int, expert_ids) -> list[int]:
        """Raw page ids overlapping ``expert_ids``' byte ranges, all roles.

        Pure page-table arithmetic: page spans are computed from
        ``page_shape`` (13-byte header reads, no payload faults), so
        this never faults a page. Used by the Stage-0 fault audit and
        by future prefetch scheduling.
        """
        pages: list[int] = []
        seen: set[int] = set()
        for role in ("gate", "up", "down"):
            flat, rpe = self._expert_flat_info(li, role)
            info = self.by_name[flat["name"]]
            per_block, bytes_per_block, _ = _TYPE_INFO[info["type"]]
            row_bytes = (int(info["dims"][0]) // per_block) * bytes_per_block
            tid = self.store.tensor_id(info["name"])
            if tid is None:
                raise PagedMoEError(f"tensor {info['name']} not in store")
            spans: list[tuple[int, int, int]] = []
            off = 0
            for pg in self.store.tensor_pages(tid):
                _, plen = self.store.page_shape(pg["page_id"])
                spans.append((pg["page_id"], off, off + plen))
                off += plen
            for e in expert_ids:
                start, end = e * rpe * row_bytes, (e + 1) * rpe * row_bytes
                for pid, ps, pe in spans:
                    if pe > start and ps < end and pid not in seen:
                        seen.add(pid)
                        pages.append(pid)
        return pages

    # -- MoE FFN ---------------------------------------------------------
    def _moe_ffn(self, h2: np.ndarray, li: int) -> np.ndarray:
        """Routed + shared expert term for hidden states ``h2`` (T, hidden)."""
        ids, weights = self.select_experts(self.router_logits(h2, li), self.top_k)
        self.last_routed[li] = (ids, weights)
        out = np.zeros_like(h2)
        for e in range(self.n_experts):
            pos_idx, slot_idx = np.where(ids == e)
            if len(pos_idx) == 0:
                continue  # unselected expert: zero faults, zero FLOPs
            gateup = self._expert_gateup(li, e)
            gu = h2[pos_idx] @ gateup.T
            gate, up = gu[:, : self.ffn], gu[:, self.ffn :]
            act = PagedQwen2._silu(gate) * up
            contrib = act @ self._expert_down(li, e).T
            w = weights[pos_idx, slot_idx][:, None]
            np.add.at(out, pos_idx, contrib * w)
        base = f"blk.{li}."
        if base + "ffn_gate_shexp.weight" in self.by_name:
            gateup = np.concatenate([
                self._w(base + "ffn_gate_shexp.weight"),
                self._w(base + "ffn_up_shexp.weight"),
            ])
            gu = h2 @ gateup.T
            gate, up = gu[:, : self.ffn], gu[:, self.ffn :]
            out = out + (PagedQwen2._silu(gate) * up) @ self._w(
                base + "ffn_down_shexp.weight"
            ).T
        return out

    # -- forward ---------------------------------------------------------
    def forward(self, ids: list[int], pos_offset: int = 0) -> np.ndarray:
        """Run ``ids`` through the model; return last-position logits (f32).

        Attention/embed/head primitives are the dense executor's own
        methods, called unbound (see module docstring) — the MoE layer
        differs from dense only in the FFN, which is ``_moe_ffn``.
        """
        ids = [int(i) for i in ids]
        n_tok = len(ids)
        positions = np.arange(pos_offset, pos_offset + n_tok)
        x = PagedQwen2._embed_rows(self, ids)
        for li in range(self.n_layer):
            base = f"blk.{li}."
            h = PagedQwen2._rmsnorm(self, x, self._w(base + "attn_norm.weight"))
            qkv_w, qkv_b = self._moe_qkv(base)
            qkv = h @ qkv_w.T + qkv_b
            q, k, v = np.split(
                qkv,
                [self.hidden, self.hidden + self.n_kv * self.head_dim],
                axis=-1,
            )
            q = q.reshape(n_tok, self.n_head, self.head_dim)
            k = k.reshape(n_tok, self.n_kv, self.head_dim)
            if self.qk_norm:
                # QK-norm BEFORE RoPE, per head, with the model eps
                # (PagedQwen2._rmsnorm reads self.eps — PagedMoE
                # carries the same attribute, so the unbound call
                # is exact). Skipping this makes attention scores
                # wrong, not near-tie noise.
                q = PagedQwen2._rmsnorm(
                    self, q, self._w(base + "attn_q_norm.weight"))
                k = PagedQwen2._rmsnorm(
                    self, k, self._w(base + "attn_k_norm.weight"))
            q = PagedQwen2._rope(self, q, positions)
            k = PagedQwen2._rope(self, k, positions)
            v = v.reshape(n_tok, self.n_kv, self.head_dim)
            k = np.repeat(k, self.group, axis=1)
            v = np.repeat(v, self.group, axis=1)
            if li in self.cache:
                kc, vc = self.cache[li]
                k_all = np.concatenate([kc, k], axis=0)
                v_all = np.concatenate([vc, v], axis=0)
            else:
                k_all, v_all = k, v
            self.cache[li] = (k_all, v_all)
            scores = np.einsum("thd,shd->ths", q, k_all) / np.float32(
                np.sqrt(self.head_dim)
            )
            n_keys = k_all.shape[0]
            col = np.arange(n_keys)[None, None, :]
            row = (pos_offset + np.arange(n_tok))[:, None, None]
            scores = np.where(col <= row, scores, np.float32("-inf"))
            scores = scores - scores.max(axis=-1, keepdims=True)
            probs = np.exp(scores)
            probs = probs / probs.sum(axis=-1, keepdims=True)
            att = np.einsum("ths,shd->thd", probs, v_all)
            x = x + att.reshape(n_tok, self.hidden) @ self._w(
                base + "attn_output.weight"
            ).T
            h2 = PagedQwen2._rmsnorm(self, x, self._w(base + "ffn_norm.weight"))
            x = x + self._moe_ffn(h2, li)
        last = PagedQwen2._rmsnorm(self, x[-1:], self.out_norm)
        vocab = int(self._head_info["dims"][1])
        logits = np.empty((vocab,), dtype=np.float32)
        block = 8192
        for b0 in range(0, vocab, block):
            b1 = min(b0 + block, vocab)
            w = PagedQwen2._head_block(self, b0, b1)
            logits[b0:b1] = (last @ w.T)[0]
        return logits

    def generate(self, prompt: str, max_tokens: int = 32) -> tuple[str, list[int]]:
        """Greedy-generate from ``prompt``. Returns (text, token ids).

        Stops after emitting the EOS id (not included in the text), or at
        ``max_tokens``. The cache is reset per call.
        """
        self.cache = {}
        ids = self.tok.encode(prompt)
        logits = self.forward(ids, 0)
        out: list[int] = []
        for _ in range(int(max_tokens)):
            nxt = int(np.argmax(logits))
            if self.eos is not None and nxt == self.eos:
                break
            out.append(nxt)
            logits = self.forward([nxt], len(ids) + len(out) - 1)
        return self.tok.decode(out), out

    def close(self) -> None:
        self.store.close()
