"""Paged transformer executor: real generation with weights via the pager.

This is the module that closes Lacuna's core gap. Before it, ``Engine`` in
``"lacuna"`` mode ran mock orchestration over real pages and labeled the
output ``[simulated]`` — the transformer math itself was never executed.
This executor runs a real decoder (Qwen2 architecture, float32 numpy) where
**every weight byte arrives through** :class:`~lacuna.pager.PagedWeightStore`
— ``read_tensor`` faults pages under the store's hard LRU budget, and
``store.stats`` is the audit trail (faults / hits / bytes_read).

Design rules, all load-bearing for the honesty claims:
- No direct file reads of weights anywhere in this module. Metadata comes
  from the GGUF header (that is configuration, not weights); every weight
  tensor goes through ``lacuna.gguf.read_tensor``.
- Greedy decoding only (argmax). Deterministic, directly comparable to
  ``llama-cli --temp 0 --top-k 1`` and to ``LlamaServerBackend`` on the
  same prompt. If a token ever differs from the dense baseline, that is
  reported as a mismatch, never smoothed over.
- Exact policy only (``read_tensor`` enforces it): the .lac must carry
  source bytes bit-for-bit, so dequantization here is the model's own
  quantization, not a Lacuna re-quantization.
- v1 scope, stated plainly: the embedding matrix and the lm_head are
  dequantized once and held (each is 151936x896 float32, ~545 MB) because
  the lm_head is needed in full at every step; every other tensor streams
  per forward pass through the pager. A later version can slice those two
  per row/block; v1 does not claim otherwise. Peak RSS is measured and
  reported with the demo, not estimated.

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
from .gguf import parse_gguf, read_tensor
from .pager import PagedWeightStore


class PagedModelError(ValueError):
    """Raised for unsupported architectures or missing tensors. Fails loud."""


class PagedQwen2:
    """Qwen2 decoder whose weights live in a PagedWeightStore."""

    def __init__(self, gguf_path: str, lac_path: str | None = None,
                 hot_budget_bytes: int = 256 * 1024 * 1024):
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
        # v1 held tensors (see module docstring): embed + lm_head + out norm.
        self.embed = read_tensor(self.store, self.by_name["token_embd.weight"])
        self.lm_head = read_tensor(self.store, self.by_name["output.weight"])
        self.out_norm = read_tensor(self.store, self.by_name["output_norm.weight"])
        self.cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        # RoPE frequencies: theta^(-2i/dim) for i in 0..dim/2-1.
        self._inv_freq = self.rope_theta ** (
            -np.arange(0, self.head_dim, 2, dtype=np.float32) / self.head_dim
        )

    # -- weight access (pager only) -----------------------------------
    def _w(self, name: str) -> np.ndarray:
        info = self.by_name.get(name)
        if info is None:
            raise PagedModelError(f"required tensor {name} missing")
        return read_tensor(self.store, info)

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

    # -- forward --------------------------------------------------------
    def forward(self, ids: list[int], pos_offset: int = 0) -> np.ndarray:
        """Run ``ids`` through the model; return last-position logits (f32)."""
        ids = [int(i) for i in ids]
        n_tok = len(ids)
        positions = np.arange(pos_offset, pos_offset + n_tok)
        x = self.embed[ids].astype(np.float32)  # (T, hidden)
        for li in range(self.n_layer):
            base = f"blk.{li}."
            h = self._rmsnorm(x, self._w(base + "attn_norm.weight"))
            q = h @ self._w(base + "attn_q.weight").T + self._w(base + "attn_q.bias")
            k = h @ self._w(base + "attn_k.weight").T + self._w(base + "attn_k.bias")
            v = h @ self._w(base + "attn_v.weight").T + self._w(base + "attn_v.bias")
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
            gate = h2 @ self._w(base + "ffn_gate.weight").T
            up = h2 @ self._w(base + "ffn_up.weight").T
            down = (self._silu(gate) * up) @ self._w(base + "ffn_down.weight").T
            x = x + down
        last = self._rmsnorm(x[-1:], self.out_norm)
        return (last @ self.lm_head.T)[0]

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
