"""Speculative paged inference: real draft-verify decoding through the pager.

This is the REAL version of the idea sketched at mock level in
:mod:`lacuna.speculate` (pluggable callables, no model). Here the target is a
real :class:`~lacuna.paged_model.PagedQwen2` whose weights arrive through the
pager, and the draft is a real cheap model (token-id n-gram). Nothing here is
simulated: every emitted token is either a draft token the target's own
argmax agreed with, or the target's argmax itself.

Mechanism (greedy speculative decoding, one batched target pass per block):

- The draft proposes K tokens conditioned on everything emitted so far.
- The target runs the whole block through ``forward_all`` in ONE paged pass:
  every layer's weights are read (and dequantized) once for all K positions,
  not once per token. In a weight-bound regime (model >> budget — the 110B
  case) that divides weight traffic per emitted token by the tokens yielded
  per block; in a compute-bound regime (model fits the cache) it does not,
  because batched compute still scales with positions. Both regimes are
  measured and reported, not assumed.
- The longest prefix the target greedily agrees with is accepted. At the
  first mismatch the target's own token is emitted instead (resample); on
  full acceptance a bonus token (the target's argmax after the block) is
  emitted. Every block therefore yields >= 1 token and costs exactly one
  batched target pass — never a single-token forward per token.

The pending-token trick (why there is no extra single-token pass): an
emitted resample/bonus token has not been processed through the layers yet,
so it is NOT in the KV cache. Instead of paying a single-token forward to
put it there, it becomes the first input of the NEXT verification block:
the next block's input is ``[pending] + new draft``. The cache invariant is
``cache covers emitted-minus-pending``; after each block the cache is
truncated to exactly that prefix, so positions and attention always match
plain greedy decoding.

Exactness (why output is identical to plain greedy by construction): an
accepted draft token equals the target argmax at its position (that is the
acceptance check); a resample/bonus token IS the target argmax at its
position. The emitted sequence is therefore the greedy sequence, and the
cache state after every block is exactly the greedy state. This is verified
empirically, not just argued: tests and the 0.5B runs compare speculative
ids against plain greedy paged ids token-for-token.

``BatchedQwen2.forward_all`` is a deliberate copy of
``PagedQwen2.forward`` with only the head section changed (per-position
logits instead of last-position logits). It lives here, not in
``paged_model.py``, because this task's constraint is new-files-only; the
proper home for it is a ``forward_all`` method on ``PagedQwen2`` itself,
and any upstream change to ``forward`` must be mirrored here until that
move happens. The layer math is byte-for-byte the same code path (same
shapes, same ops), so ``forward_all(...)[-1]`` equals ``forward(...)``
exactly — pinned by test.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from .paged_model import PagedQwen2


class BatchedQwen2(PagedQwen2):
    """PagedQwen2 with a multi-position-logits forward for verification.

    ``forward`` returns only last-position logits, which cannot verify a
    draft block (verification needs the target's prediction at EVERY draft
    position). This method runs the identical layer math and returns logits
    for every input position, so one call verifies a whole draft block.
    """

    def forward_all(self, ids: list[int], pos_offset: int = 0) -> np.ndarray:
        """Run ``ids`` through the model; return per-position logits (T, vocab).

        Cache append behaviour is identical to :meth:`PagedQwen2.forward`:
        K/V for ``ids`` are appended at positions ``pos_offset..``. Callers
        that need the cache to cover a shorter prefix must truncate with
        :meth:`truncate_cache` afterwards (the speculative decoder does).
        """
        ids = [int(i) for i in ids]
        n_tok = len(ids)
        positions = np.arange(pos_offset, pos_offset + n_tok)
        x = self._embed_rows(ids)  # (T, hidden), sliced, never held whole
        for li in range(self.n_layer):
            base = f"blk.{li}."
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
        # Per-position logits: same blocked-head computation as forward(),
        # applied to every position's final hidden state instead of only the
        # last. Only the (T, vocab) logits are assembled, never a weight.
        vocab = int(self._head_info["dims"][1])
        logits = np.empty((n_tok, vocab), dtype=np.float32)
        block = 8192
        for i in range(n_tok):
            last = self._rmsnorm(x[i : i + 1], self.out_norm)
            for b0 in range(0, vocab, block):
                b1 = min(b0 + block, vocab)
                w = self._head_block(b0, b1)
                logits[i, b0:b1] = (last @ w.T)[0]
        return logits

    def truncate_cache(self, n_tokens: int) -> None:
        """Keep only the first ``n_tokens`` positions of every layer's KV.

        Slices are views, which is safe: ``forward``/``forward_all`` never
        mutate cached arrays in place (they concatenate into new arrays).
        """
        for li in self.cache:
            kc, vc = self.cache[li]
            self.cache[li] = (kc[:n_tokens], vc[:n_tokens])

    def cache_len(self) -> int:
        """Positions currently covered by the KV cache (0 when empty)."""
        if not self.cache:
            return 0
        return int(self.cache[0][0].shape[0])


class NgramDraft:
    """Token-id n-gram draft: cheap (dict lookups), no neural cost.

    Trained on token-id sequences (callers encode their corpus with the
    target's own tokenizer, so draft ids live in the target's vocab).
    Proposes autoregressively: longest matching suffix (order 4 down to 1),
    most frequent continuation; falls back to the most frequent unigram.
    Deterministic for a fixed training order (Counter insertion order).
    """

    def __init__(self, max_order: int = 4):
        self.max_order = int(max_order)
        self.tables: dict[int, dict[tuple[int, ...], Counter]] = {
            n: {} for n in range(1, self.max_order + 1)
        }
        self.unigram: Counter = Counter()

    def train(self, sequences) -> None:
        """Train on an iterable of token-id sequences."""
        for seq in sequences:
            seq = [int(t) for t in seq]
            for i, tok in enumerate(seq):
                self.unigram[tok] += 1
                for n in range(1, self.max_order + 1):
                    if i >= n:
                        ctx = tuple(seq[i - n : i])
                        self.tables[n].setdefault(ctx, Counter())[tok] += 1

    def _next(self, ctx: list[int]) -> int:
        for n in range(self.max_order, 0, -1):
            if len(ctx) >= n:
                hit = self.tables[n].get(tuple(ctx[-n:]))
                if hit:
                    return hit.most_common(1)[0][0]
        if self.unigram:
            return self.unigram.most_common(1)[0][0]
        return 0

    def propose(self, prefix: list[int], k: int) -> list[int]:
        """Propose ``k`` draft ids conditioned on ``prefix`` (greedy n-gram)."""
        ctx = [int(t) for t in prefix]
        out: list[int] = []
        for _ in range(int(k)):
            nxt = self._next(ctx)
            out.append(nxt)
            ctx.append(nxt)
        return out


class NeuralDraft:
    """Neural draft backend: a small paged model proposes greedily.

    Wraps a :class:`BatchedQwen2` draft model (needed for
    :meth:`truncate_cache`). The draft's KV cache is kept across
    ``propose`` calls and always covers exactly the emitted prefix: after
    each call the draft tokens (accepted or not) are truncated away, so
    the next call only forwards newly emitted tokens plus ``k``
    single-token steps — never a full prefill. Draft ids must live in the
    target's vocab; callers verify this (same model family) before use,
    because a vocab mismatch would silently propose nonsense ids.

    State: ``_prefix`` is the token list the cache currently covers, and
    ``_next_logits`` is the draft's prediction for the token AFTER that
    prefix (saved before the greedy loop so it survives truncation).
    """

    def __init__(self, model: BatchedQwen2):
        self.model = model
        self.eos = model.eos
        self._prefix: list[int] = []
        self._next_logits = None

    def reset(self) -> None:
        """Drop all draft cache state (next propose re-prefills)."""
        self.model.cache = {}
        self._prefix = []
        self._next_logits = None

    def propose(self, prefix: list[int], k: int) -> list[int]:
        """Propose ``k`` draft ids conditioned on ``prefix`` (greedy).

        Stops early (inclusive) if the draft predicts EOS: tokens after
        EOS would never be emitted by the decoder anyway.
        """
        prefix = [int(t) for t in prefix]
        k = int(k)
        m = self.model
        if len(prefix) < len(self._prefix) or prefix[: len(self._prefix)] != self._prefix:
            # Prefix shrank or diverged (never happens in the decoder
            # loop, which only appends): reset rather than trust a stale
            # cache, which would propose from wrong positions.
            self.reset()
        if len(self._prefix) < len(prefix):
            suffix = prefix[len(self._prefix) :]
            logits = m.forward_all(suffix, len(self._prefix))
            self._next_logits = logits[-1]
            self._prefix = list(prefix)
        if self._next_logits is None or k <= 0:
            return []
        first_logits = self._next_logits
        cur = first_logits
        out: list[int] = []
        for _ in range(k):
            tok = int(np.argmax(cur))
            out.append(tok)
            if self.eos is not None and tok == self.eos:
                break
            cur = m.forward([tok], len(prefix) + len(out) - 1)
        # Draft tokens were never emitted: truncate them away so the
        # cache again covers exactly the emitted prefix, and restore
        # the prediction-after-prefix for the next call.
        m.truncate_cache(len(prefix))
        self._prefix = list(prefix)
        self._next_logits = first_logits
        return out


class SpeculativeDecoder:
    """Greedy speculative decoding over a :class:`BatchedQwen2` target.

    Invariants (see module docstring): ``model`` cache covers exactly
    ``prefix[:cached]`` where ``prefix`` is prompt + emitted tokens; if
    ``cached < len(prefix)`` the last emitted token is pending (emitted but
    not yet processed) and opens the next verification block. Every block
    is exactly one ``forward_all`` call and yields >= 1 token.
    """

    def __init__(self, model: BatchedQwen2, draft, k: int = 4):
        self.model = model
        self.draft = draft
        self.k = int(k)

    def generate(
        self, prompt: str, max_tokens: int = 32, progress_cb=None
    ) -> tuple[str, list[int], dict]:
        """Greedy-generate; returns (text, token ids, stats).

        Ids are identical to plain greedy paged generation by construction
        (see module docstring); callers verify that empirically.
        ``progress_cb``, when given, is called after each round with a
        dict of running counters (rounds/proposed/accepted/emitted) so
        long runs can log incrementally instead of only at the end.
        """
        model = self.model
        model.cache = {}
        prompt_ids = model.tok.encode(prompt)
        prefill = model.forward_all(prompt_ids, 0)
        p_prev = int(np.argmax(prefill[-1]))
        eos = model.eos
        prefix = list(prompt_ids)
        cached = len(prompt_ids)
        out: list[int] = []
        rounds = 0
        proposed = 0
        accepted = 0
        while len(out) < int(max_tokens):
            draft = self.draft.propose(prefix, self.k)
            pending = len(prefix) > cached
            if not draft and not pending:
                # Degenerate draft on the first round: emit the prefill
                # prediction as pending and continue (next block verifies
                # it as its first input). No forward call spent.
                if eos is not None and p_prev == eos:
                    break
                out.append(p_prev)
                prefix.append(p_prev)
                continue
            block = ([prefix[-1]] if pending else []) + list(draft)
            logits = model.forward_all(block, cached)
            # Target prediction before each draft token.
            preds: list[int] = []
            for i in range(len(draft)):
                if pending:
                    # Draft token i sits at block[i + 1]; its prediction is
                    # logits[i] (after block[:i + 1], i.e. pending + prior).
                    preds.append(int(np.argmax(logits[i])))
                else:
                    preds.append(
                        p_prev if i == 0 else int(np.argmax(logits[i - 1]))
                    )
            a = 0
            while a < len(draft) and draft[a] == preds[a]:
                a += 1
            emit: list[int] = list(draft[:a])
            if a < len(draft):
                emit.append(preds[a])  # resample: target's own token
            else:
                emit.append(int(np.argmax(logits[-1])))  # bonus token
            # Emit with plain-greedy EOS semantics: stop at the first EOS,
            # never including it.
            stopped = False
            emitted_now: list[int] = []
            for tok in emit:
                if eos is not None and tok == eos:
                    stopped = True
                    break
                emitted_now.append(tok)
                out.append(tok)
                if len(out) >= int(max_tokens):
                    break
            # Cache now covers old prefix + old pending + accepted draft
            # that was actually emitted. (When a block is cut short by
            # max_tokens or EOS, only the emitted part counts; the loop
            # ends there anyway, and the next generate() resets the cache.)
            new_cached = cached + (1 if pending else 0) + min(a, len(emitted_now))
            model.truncate_cache(new_cached)
            cached = new_cached
            prefix = prefix + emitted_now
            rounds += 1
            proposed += len(draft)
            accepted += a
            if progress_cb is not None:
                progress_cb({
                    "round": rounds, "emitted": len(out),
                    "proposed": proposed, "accepted": accepted,
                    "accepted_this_round": a,
                })
            if stopped or len(out) >= int(max_tokens):
                break
        stats = {
            "rounds": rounds,
            "proposed_tokens": proposed,
            "accepted_tokens": accepted,
            "acceptance_rate": (accepted / proposed) if proposed else 0.0,
            "mean_tokens_per_block": (len(out) / rounds) if rounds else 0.0,
            "tokens": len(out),
        }
        return model.tok.decode(out), out, stats
