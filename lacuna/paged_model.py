"""Paged transformer executor: real generation with weights via the pager.

This is the module that closes Lacuna's core gap. Before it, ``Engine`` in
``"lacuna"`` mode ran mock orchestration over real pages and labeled the
output ``[simulated]`` — the transformer math itself was never executed.
This executor runs a real decoder (Qwen2 architecture, float32 numpy) where
**every weight byte arrives through** :class:`~lacuna.pager.PagedWeightStore`
— ``read_tensor`` / ``read_tensor_rows`` fault pages under the store's
hard LRU budget, and ``store.stats`` is the audit trail (faults / hits
/ bytes_read).

Design rules, all load-bearing for the honesty claims:
- No direct file reads of weights anywhere in this module. Metadata comes
  from the GGUF header (that is configuration, not weights); every weight
  tensor goes through ``lacuna.gguf.read_tensor`` (whole) or
  ``read_tensor_rows`` (embed/head slices).
- Greedy decoding only (argmax). Deterministic, directly comparable to
  ``llama-cli --temp 0 --top-k 1`` and to ``LlamaServerBackend`` on the
  same prompt. If a token ever differs from the dense baseline, that is
  reported as a mismatch, never smoothed over.
- Exact policy only (``read_tensor`` enforces it): the .lac must carry
  source bytes bit-for-bit, so dequantization here is the model's own
  quantization, not a Lacuna re-quantization.
- Sliced embed + head (2026-10-25): the embedding matrix and the
  lm_head are NEVER materialized whole. Embed rows are gathered only
  for the input token ids of the current call; lm_head logits are
  computed in 8192-row blocks and only the logits vector (one float per
  vocab row) is ever assembled. On the 0.5B each held matrix would be
  ~545 MB float32; on a 100B-class model each would be ~8 GB — slicing
  is what makes large vocabularies runnable on the 6 GB floor at all.
  Peak RSS is measured and reported with the demo, not estimated.

Architecture implemented (Qwen2, verified against the model's own GGUF
metadata and against llama.cpp's dense output token-for-token): RMSNorm,
GQA attention with Q/K/V biases, RoPE in the half-rotation form (rotate
the first half of each head against the second half — this is what
llama.cpp's Qwen2 path computes for this file; the pair-rotation form
was tried first and measurably produced the wrong token, so the form
here is the empirically verified one, not the assumed one), SwiGLU MLP,
KV cache, final norm, separate lm_head (this file carries both
``token_embd.weight`` and ``output.weight`` — the executor trusts the
file, not the assumption that embeddings are tied).
"""

from __future__ import annotations

import numpy as np

from .bpe import BPETokenizer
from .convert import convert as _convert
from .gguf import parse_gguf, read_tensor, read_tensor_rows
from .pager import PagedWeightStore


class PagedModelError(ValueError):
    """Raised for unsupported architectures or missing tensors. Fails loud."""


class PagedQwen2:
    """Qwen2 decoder whose weights live in a PagedWeightStore."""

    def __init__(self, gguf_path: str, lac_path: str | None = None,
                 hot_budget_bytes: int = 256 * 1024 * 1024,
                 dequant_budget_bytes: int | None = None,
                 enable_prefetch: bool = True):
        md, infos, _ = parse_gguf(gguf_path)
        if md.get("general.architecture") != "qwen2":
            raise PagedModelError(
                f"paged executor supports qwen2 only, got "
                f"{md.get('general.architecture')!r}"
            )
        p = "qwen2."
        self.n_layer = int(md[p + "block_count"])
        self.hidden = int(md[p + "embedding_length"])
        self.ffn = int(md[p + "feed_forward_length"])
        self.n_head = int(md[p + "attention.head_count"])
        self.n_kv = int(md[p + "attention.head_count_kv"])
        self.eps = float(md[p + "attention.layer_norm_rms_epsilon"])
        self.rope_theta = float(md.get(p + "rope.freq_base", 1000000.0))
        self.head_dim = self.hidden // self.n_head
        if self.head_dim * self.n_head != self.hidden:
            raise PagedModelError("hidden size not divisible by head count")
        if self.n_head % self.n_kv:
            raise PagedModelError("head count not divisible by kv head count")
        self.group = self.n_head // self.n_kv
        self.tok = BPETokenizer.from_metadata(md)
        self.eos = md.get("tokenizer.ggml.eos_token_id")
        self.by_name = {i["name"]: i for i in infos}
        for name in ("token_embd.weight", "output.weight", "output_norm.weight"):
            if name not in self.by_name:
                raise PagedModelError(f"required tensor {name} missing")

        lac = lac_path or (gguf_path + ".lac")
        import os as _os

        if not _os.path.exists(lac):
            _convert(gguf_path, lac, quiet=True)
        self.store = PagedWeightStore(lac, hot_budget_bytes=int(hot_budget_bytes))
        # Only the final norm is held (hidden floats — tiny). Embed and
        # head are sliced per call (see _embed_rows and forward); holding
        # either whole would cost ~545 MB on the 0.5B and gigabytes on
        # 100B-class models.
        self._embed_info = self.by_name["token_embd.weight"]
        self._head_info = self.by_name["output.weight"]
        self.out_norm = read_tensor(self.store, self.by_name["output_norm.weight"])
        self.cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        # Dequantized-weight cache (speed work, 2026-10-25): v1
        # re-dequantized every tensor on every token, which is most of
        # the 0.14 tok/s cost. This LRU holds dequantized tensors up to
        # dequant_budget_bytes (default: the pager budget). It does NOT
        # bypass the pager — every tensor still arrives via read_tensor
        # on a cache miss — and it is reported separately from the raw
        # page cache (_dq_bytes vs store._cached_bytes) so the memory
        # accounting stays inspectable. At budgets below the model's
        # dequantized size it thrashes like the page cache does; at
        # budgets that fit, later tokens skip disk reads AND dequant.
        from collections import OrderedDict

        self._dq: OrderedDict[str, np.ndarray] = OrderedDict()
        self._dq_bytes = 0
        self._dq_budget = (
            int(dequant_budget_bytes)
            if dequant_budget_bytes is not None
            else int(hot_budget_bytes)
        )
        # Next-layer prefetch (see _layer_page_ids): while layer li
        # computes, the pager loads layer li+1's pages on its pool, so
        # disk latency overlaps compute. Prefetch moves the same bytes
        # earlier — it can never change generated ids; a wrong or
        # missing prediction costs a demand fault, never correctness.
        # Pass enable_prefetch=False for a bit-identical plain run.
        self._enable_prefetch = bool(enable_prefetch)
        # RoPE frequencies: theta^(-2i/dim) for i in 0..dim/2-1.
        self._inv_freq = self.rope_theta ** (
            -np.arange(0, self.head_dim, 2, dtype=np.float32) / self.head_dim
        )

    # -- weight access (pager only) -----------------------------------
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
            raise PagedModelError(f"required tensor {name} missing")
        arr = read_tensor(self.store, info)
        self._dq_put(name, arr)
        return arr

    def _fused_qkv(self, base: str) -> tuple[np.ndarray, np.ndarray]:
        """One stacked QKV weight+bias, so attention projections are a
        single GEMM instead of three. The stacked copy replaces its
        components in the cache, so memory stays ~one copy of the
        weights, not two."""
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
        b = np.concatenate([
            self._w(base + "attn_q.bias"),
            self._w(base + "attn_k.bias"),
            self._w(base + "attn_v.bias"),
        ])
        for suffix in ("attn_q.weight", "attn_k.weight", "attn_v.weight",
                       "attn_q.bias", "attn_k.bias", "attn_v.bias"):
            self._dq_drop(base + suffix)
        self._dq_put(key, w)
        self._dq_put(key + ".bias", b)
        return w, b

    def _fused_gateup(self, base: str) -> np.ndarray:
        """One stacked gate+up weight, so the MLP is two GEMMs (fused,
        down) instead of three."""
        key = base + "gateup"
        hit = self._dq.get(key)
        if hit is not None:
            self._dq.move_to_end(key)
            return hit
        w = np.concatenate([
            self._w(base + "ffn_gate.weight"),
            self._w(base + "ffn_up.weight"),
        ])
        for suffix in ("ffn_gate.weight", "ffn_up.weight"):
            self._dq_drop(base + suffix)
        self._dq_put(key, w)
        return w

    def _head_block(self, b0: int, b1: int) -> np.ndarray:
        """One dequantized lm_head row block, LRU-cached in ``_dq``.

        The sliced head (one ``read_tensor_rows`` per block per token)
        was a speed regression: every token re-dequantized the full
        136M-value head on the 0.5B. Caching blocks in the existing
        bounded, evictable ``_dq`` cache restores the cached speed
        WITHOUT holding the whole matrix — a block is 8192 rows, and
        blocks still evict under budget pressure, so 100B-class heads
        stay runnable. The cache key namespace (``#rows:``) cannot
        collide with tensor names.
        """
        key = f"output.weight#rows:{b0}:{b1}"
        hit = self._dq.get(key)
        if hit is not None:
            self._dq.move_to_end(key)
            return hit
        w = read_tensor_rows(self.store, self._head_info, b0, b1)
        self._dq_put(key, w)
        return w

    def _embed_rows(self, ids: list[int]) -> np.ndarray:
        """Gather embedding rows for ``ids`` without holding the matrix.

        Unique ids are grouped into contiguous runs; each run is one
        ``read_tensor_rows`` call (one set of page faults), then rows are
        reordered back to ``ids`` order. Individual rows are also cached
        in ``_dq`` (bounded, evictable) so repeated tokens across a long
        generation skip re-dequantization; a decode step (one id) faults
        a single row's bytes (~616 B for the 0.5B's Q5_0 rows) on a miss.
        """
        if not ids:
            return np.zeros((0, self.hidden), dtype=np.float32)
        uniq = sorted(set(int(i) for i in ids))
        rows: dict[int, np.ndarray] = {}
        missing: list[int] = []
        for u in uniq:
            key = f"token_embd.weight#row:{u}"
            hit = self._dq.get(key)
            if hit is not None:
                self._dq.move_to_end(key)
                rows[u] = hit
            else:
                missing.append(u)
        if missing:
            runs: list[tuple[int, int]] = []
            start = last = missing[0]
            for u in missing[1:]:
                if u == last + 1:
                    last = u
                else:
                    runs.append((start, last + 1))
                    start = last = u
            runs.append((start, last + 1))
            for a, b in runs:
                arr = read_tensor_rows(self.store, self._embed_info, a, b)
                for k, u in enumerate(range(a, b)):
                    row = np.array(arr[k], dtype=np.float32, copy=True)
                    self._dq_put(f"token_embd.weight#row:{u}", row)
                    rows[u] = row
        return np.stack([rows[i] for i in ids]).astype(np.float32)

    # -- primitives ----------------------------------------------------
    def _rmsnorm(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        ms = np.mean(x * x, axis=-1, keepdims=True)
        return (x / np.sqrt(ms + np.float32(self.eps))) * w

    def _rope(self, t: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """RoPE, half-rotation form: rotate the first half of each head
        against the second half. Verified empirically: this form reproduces
        llama.cpp's dense output token-for-token on this model, while the
        pair-rotation form produced the wrong token (' ' instead of
        ' Paris' after 'The capital of France is')."""
        angles = positions[:, None].astype(np.float32) * self._inv_freq[None, :]
        cos = np.cos(angles)[:, None, :]
        sin = np.sin(angles)[:, None, :]
        h = t.shape[-1] // 2
        x0 = t[..., :h]
        x1 = t[..., h:]
        out = np.empty_like(t)
        out[..., :h] = x0 * cos - x1 * sin
        out[..., h:] = x0 * sin + x1 * cos
        return out

    @staticmethod
    def _silu(x: np.ndarray) -> np.ndarray:
        return x / (np.float32(1.0) + np.exp(-x))

    # -- prefetch ---------------------------------------------------------
    _LAYER_TENSORS = (
        "attn_norm.weight",
        "attn_q.weight", "attn_k.weight", "attn_v.weight",
        "attn_q.bias", "attn_k.bias", "attn_v.bias",
        "attn_output.weight",
        "ffn_norm.weight",
        "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
    )

    def _layer_page_ids(self, li: int) -> list[int]:
        """Raw page ids for layer ``li``'s tensors, in demand order.

        Skips tensors already in the dequantized cache (a demand for them
        would not fault) and, with fused-key awareness, skips q/k/v
        components when the fused ``qkv`` entry is cached (likewise
        gate/up when ``gateup`` is cached) — the components are dropped
        from ``_dq`` on fusion, so this matches exactly what ``_w`` will
        demand. Names with no tensor id are skipped (some files lack
        biases). Only the single forward thread touches ``_dq``, so the
        unsynchronized check here is safe; prefetch threads touch only
        the store's locked cache.
        """
        base = f"blk.{li}."
        dq = self._dq
        skip: set[str] = set()
        if base + "qkv" in dq:
            skip.update(("attn_q.weight", "attn_k.weight", "attn_v.weight",
                         "attn_q.bias", "attn_k.bias", "attn_v.bias"))
        if base + "gateup" in dq:
            skip.update(("ffn_gate.weight", "ffn_up.weight"))
        pages: list[int] = []
        for suffix in self._LAYER_TENSORS:
            if suffix in skip or base + suffix in dq:
                continue
            tid = self.store.tensor_id(base + suffix)
            if tid is None:
                continue
            pages.extend(e["page_id"] for e in self.store.tensor_pages(tid))
        return pages

    # -- forward --------------------------------------------------------
    def forward(self, ids: list[int], pos_offset: int = 0) -> np.ndarray:
        """Run ``ids`` through the model; return last-position logits (f32)."""
        ids = [int(i) for i in ids]
        n_tok = len(ids)
        positions = np.arange(pos_offset, pos_offset + n_tok)
        x = self._embed_rows(ids)  # (T, hidden), sliced, never held whole
        for li in range(self.n_layer):
            base = f"blk.{li}."
            if self._enable_prefetch and li + 1 < self.n_layer:
                # Fire-and-forget, one layer ahead only: the handle is
                # intentionally not retained — the pool holds the
                # futures, and close() waits for them. The next layer's
                # demand faults (via _w) are the fence: if prefetch
                # missed or lost a page, it faults on demand, exactly
                # as in a plain run. Head blocks are NOT prefetched
                # (simplest correct integration; the head is demanded
                # once per forward, after all layers).
                self.store.prefetch(self._layer_page_ids(li + 1))
            h = self._rmsnorm(x, self._w(base + "attn_norm.weight"))
            qkv_w, qkv_b = self._fused_qkv(base)
            qkv = h @ qkv_w.T + qkv_b
            q, k, v = np.split(
                qkv,
                [self.hidden, self.hidden + self.n_kv * self.head_dim],
                axis=-1,
            )
            q = self._rope(q.reshape(n_tok, self.n_head, self.head_dim), positions)
            k = self._rope(k.reshape(n_tok, self.n_kv, self.head_dim), positions)
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
            # Causal attention over everything cached so far.
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
            h2 = self._rmsnorm(x, self._w(base + "ffn_norm.weight"))
            gateup = h2 @ self._fused_gateup(base).T
            gate, up = gateup[:, : self.ffn], gateup[:, self.ffn :]
            down = (self._silu(gate) * up) @ self._w(base + "ffn_down.weight").T
            x = x + down
        last = self._rmsnorm(x[-1:], self.out_norm)
        # Blocked lm_head: only the logits vector is ever assembled whole.
        # Greedy needs argmax, but forward() keeps returning the full
        # vector so generate() is unchanged.
        vocab = int(self._head_info["dims"][1])
        logits = np.empty((vocab,), dtype=np.float32)
        block = 8192
        for b0 in range(0, vocab, block):
            b1 = min(b0 + block, vocab)
            w = self._head_block(b0, b1)
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
