"""Speculative paged inference for Gemma-4 (CPU path).

Mirrors :mod:`lacuna.speculative` (the Qwen2 speculative decoder) for
:class:`~lacuna.gemma.PagedGemma4`, with one structural difference that
dominates the design: Gemma's KV cache is NOT a plain prefix.

Each Gemma layer's cache is ``(K, V, start_pos)`` where ``start_pos`` is
the absolute position of the first cached row. Global layers keep
everything (``start_pos`` stays 0); sliding layers trim from the front
to the window. A naive port of the Qwen2 decoder would break here in
two ways, both of which this module avoids by construction:

1. Trimming during verification loses rollback keys. ``PagedGemma4``
   trims a sliding cache at the end of every ``forward`` call. If a
   verification block did that, the window head would move forward by
   the block length, and rolling back to the accepted prefix would
   leave the next query missing keys it still attends to (up to
   ``block_length - 1`` keys). So :meth:`BatchedGemma4.forward_all`
   NEVER trims: the cache grows transiently by the block length, and
   trimming is deferred to :meth:`BatchedGemma4.truncate_cache`.

2. Truncation must re-trim to the accepted prefix's window. Slicing a
   sliding cache to the accepted length is not enough on its own: the
   sliced cache must then be re-trimmed to the last window of the
   accepted prefix, which is exactly what a fresh model would hold
   after that prefix — and exactly the set of keys any future query
   can still attend to (future queries at position ``p`` need keys
   ``>= p - window + 1 >= accepted - window + 1``). ``truncate_cache``
   therefore slices AND re-trims, and the postcondition after every
   block is: sliding caches hold the last window of the emitted
   prefix, global caches hold the whole emitted prefix.

Exactness (why output is identical to plain greedy by construction):
same argument as the Qwen2 decoder — an accepted draft token equals
the target argmax at its position, a resample/bonus token IS the
target argmax — plus the cache argument above: after every block the
cache state is exactly the state plain greedy decoding would hold
after the same emitted prefix (pinned by test, including a long
prompt that forces repeated sliding trims). The one operation that
cannot be repaired from cache is a rollback to a position at or
before a sliding layer's current window head (those keys were already
trimmed by an earlier block); ``truncate_cache`` fails loud there
instead of silently corrupting attention, and the decoder never
issues such a rollback (its rollback distance is at most one block,
far inside the window).

GPU status (stated plainly, not papered over): this module is CPU
only. A GPU Gemma speculative decoder would need three things that
do not exist today, all of which require new GPU code and are out of
this task's scope (which forbids modifying the GPU forward code):
(a) a per-position-logits forward on ``GPUGemmaModel`` (its
``forward`` returns last-position logits only, so a verification
block cannot be run through it); (b) a rollback that addresses
absolute positions uniformly across layer types (its
``truncate_cache`` takes one row count applied to all layers, but
that count means rows-from-cache-front for sliding layers and
absolute positions for global layers — once any sliding layer has
trimmed, no single count is correct for both); (c) no-trim-during-
verification semantics (its layer forward trims after attention on
every call, which is the key-loss failure of point 1 above).
"""

from __future__ import annotations

import numpy as np

from .gemma import GemmaModelError, PagedGemma4
from .speculative import NgramDraft  # noqa: F401  (re-exported for callers)

__all__ = ["BatchedGemma4", "GemmaSpeculativeDecoder", "NgramDraft"]


class BatchedGemma4(PagedGemma4):
    """PagedGemma4 with a multi-position-logits forward for verification.

    ``forward`` returns only last-position logits, which cannot verify
    a draft block (verification needs the target's prediction at EVERY
    draft position). This method runs the identical layer math and
    returns logits for every input position, so one call verifies a
    whole draft block. It lives here, not in ``gemma.py``, because this
    task's constraint is new-files-only; the proper home for it is a
    ``forward_all`` method on ``PagedGemma4`` itself, and any upstream
    change to ``forward`` must be mirrored here until that move
    happens. The layer math is byte-for-byte the same code path (same
    shapes, same ops), so ``forward_all(...)[-1]`` equals ``forward``
    exactly — pinned by test.

    Deliberate difference from ``forward``: ``forward_all`` NEVER
    trims sliding caches (see module docstring). Trimming is deferred
    to :meth:`truncate_cache`, which slices to the accepted prefix
    and re-trims to that prefix's window. Callers that run
    ``forward_all`` without a following ``truncate_cache`` hold the
    full (untrimmed) cache, exactly as a single ``forward`` call does
    within its own call.
    """

    def forward_all(self, ids: list[int], pos_offset: int = 0) -> np.ndarray:
        """Run ``ids`` through the model; return per-position logits.

        Cache append behaviour matches :meth:`PagedGemma4.forward`
        except that sliding caches are NOT trimmed (see class
        docstring); call :meth:`truncate_cache` afterwards to restore
        the window bound. ``pos_offset`` is the absolute position of
        ``ids[0]``; K/V for ``ids`` are appended at those positions.
        """
        ids = [int(i) for i in ids]
        n_tok = len(ids)
        positions = np.arange(pos_offset, pos_offset + n_tok)
        x = self._embed_rows(ids)
        for li in range(self.n_layer):
            base = f"blk.{li}."
            if self._enable_prefetch and li + 1 < self.n_layer:
                self.store.prefetch(self._layer_page_ids(li + 1))
            is_global = self._is_global(li)
            n_q, n_kv, d = self._layer_dims(li)
            group = n_q // n_kv
            if n_q % n_kv:
                raise GemmaModelError(
                    f"layer {li}: Q heads {n_q} not divisible by KV heads {n_kv}"
                )
            h = self._rmsnorm(x, self._w(base + "attn_norm.weight"))
            q = h @ self._w(base + "attn_q.weight").T
            k = h @ self._w(base + "attn_k.weight").T
            qb = self._w_optional(base + "attn_q.bias")
            kb = self._w_optional(base + "attn_k.bias")
            if qb is not None:
                q = q + qb
            if kb is not None:
                k = k + kb
            v_w = self._w_optional(base + "attn_v.weight")
            if v_w is None:
                if not is_global:
                    raise GemmaModelError(
                        f"layer {li}: sliding layer missing attn_v.weight"
                    )
                # Global V = K: V starts as the RAW k projection.
                v_raw = k
            else:
                v_raw = h @ v_w.T
                vb = self._w_optional(base + "attn_v.bias")
                if vb is not None:
                    v_raw = v_raw + vb
            # Mirror of PagedGemma4.forward's corrected block math
            # (QK-norm, weightless V-norm, post-norms, layer-wise
            # layer_output_scale): this method duplicates that block
            # by design (see class docstring), so every correction
            # there must land here too, or verification would check
            # drafts against wrong math.
            q = self._rmsnorm(
                q.reshape(n_tok, n_q, d),
                self._w(base + "attn_q_norm.weight"))
            k = self._rmsnorm(
                k.reshape(n_tok, n_kv, d),
                self._w(base + "attn_k_norm.weight"))
            v = self._rmsnorm(
                v_raw.reshape(n_tok, n_kv, d),
                np.ones(d, dtype=np.float32))
            q = self._rope(q, positions, is_global)
            k = self._rope(k, positions, is_global)
            k = np.repeat(k, group, axis=1)
            v = np.repeat(v, group, axis=1)
            if li in self.cache:
                kc, vc, start = self.cache[li]
                k_all = np.concatenate([kc, k], axis=0)
                v_all = np.concatenate([vc, v], axis=0)
            else:
                k_all, v_all = k, v
                start = int(pos_offset)
            # Attention sees the FULL cache: the sliding-window mask
            # already restricts each query to its window. No trim
            # here (unlike forward): trimming now would move the
            # window head past keys a later rollback still needs.
            col_pos = start + np.arange(k_all.shape[0])
            att = self._attention(q, k_all, v_all, positions, col_pos,
                                  not is_global)
            self.cache[li] = (k_all, v_all, start)
            att_out = att.reshape(n_tok, n_q * d) @ self._w(
                base + "attn_output.weight"
            ).T
            att_out = self._rmsnorm(
                att_out, self._w(base + "post_attention_norm.weight"))
            x1 = x + att_out
            h2 = self._rmsnorm(x1, self._w(base + "ffn_norm.weight"))
            gate = h2 @ self._w(base + "ffn_gate.weight").T
            up = h2 @ self._w(base + "ffn_up.weight").T
            down = (self._gelu_tanh(gate) * up) @ self._w(
                base + "ffn_down.weight"
            ).T
            down = self._rmsnorm(
                down, self._w(base + "post_ffw_norm.weight"))
            # layer_output_scale multiplies the WHOLE layer output.
            x = (x1 + down) * self._w(base + "layer_output_scale.weight")
        # Per-position logits: same blocked-head computation as
        # forward(), applied to every position's final hidden state
        # instead of only the last. Only the (T, vocab) logits are
        # assembled, never a weight.
        vocab = int(self._head_info["dims"][1])
        logits = np.empty((n_tok, vocab), dtype=np.float32)
        block = 8192
        for i in range(n_tok):
            last = self._rmsnorm(x[i : i + 1], self.out_norm)
            for b0 in range(0, vocab, block):
                b1 = min(b0 + block, vocab)
                w = self._head_block(b0, b1)
                logits[i, b0:b1] = (last @ w.T)[0]
        if self.final_logit_softcapping is not None:
            cap = np.float32(self.final_logit_softcapping)
            logits = cap * np.tanh(logits / cap)
        return logits

    def truncate_cache(self, n_kept: int) -> None:
        """Roll every layer's KV cache back to absolute position ``n_kept``.

        After the call, global caches hold exactly positions
        ``[0, n_kept)`` and sliding caches hold the last window of
        ``[0, n_kept)`` that was still cached — i.e. exactly what a
        fresh model would hold after that prefix, and exactly the keys
        any future query can still attend to. Slices are views, which
        is safe: ``forward``/``forward_all`` never mutate cached
        arrays in place (they concatenate into new arrays).

        Fails loud when ``n_kept`` is at or before a layer's current
        window head (and nonzero): those keys were already trimmed by
        an earlier block and cannot be recovered from cache — the
        caller must re-prefill instead. ``n_kept == 0`` always
        succeeds (empty cache, head reset to 0).
        """
        n_kept = int(n_kept)
        if n_kept < 0:
            raise GemmaModelError(f"truncate_cache: negative n_kept {n_kept}")
        for li in list(self.cache):
            k, v, start = self.cache[li]
            end = start + k.shape[0]
            if n_kept >= end:
                k2, v2, start2 = k, v, start
            elif n_kept == 0:
                k2, v2, start2 = k[:0], v[:0], 0
            elif n_kept <= start:
                raise GemmaModelError(
                    f"layer {li}: cannot roll back to {n_kept}: cache "
                    f"starts at {start} (earlier keys were trimmed); "
                    f"re-prefill required"
                )
            else:
                k2, v2, start2 = k[: n_kept - start], v[: n_kept - start], start
            if not self._is_global(li) and k2.shape[0] > self.window:
                drop = k2.shape[0] - self.window
                k2, v2, start2 = k2[drop:], v2[drop:], start2 + drop
            self.cache[li] = (k2, v2, start2)

    def cache_len(self) -> int:
        """Absolute position one past the last cached row (0 when empty).

        Sliding layers report their window head + rows, global layers
        their full length; the max is the emitted-prefix length the
        cache as a whole covers.
        """
        if not self.cache:
            return 0
        return max(int(start + k.shape[0])
                   for k, _v, start in self.cache.values())


class GemmaSpeculativeDecoder:
    """Greedy speculative decoding over a :class:`BatchedGemma4` target.

    Invariants (see module docstring): ``model`` cache covers the
    emitted prefix — global layers fully, sliding layers as the last
    window of it; if ``cached < len(prefix)`` the last emitted token
    is pending (emitted but not yet processed) and opens the next
    verification block. Every block is exactly one ``forward_all``
    call and yields >= 1 token.
    """

    def __init__(self, model: BatchedGemma4, draft, k: int = 4):
        self.model = model
        self.draft = draft
        self.k = int(k)

    def generate(
        self, prompt: str, max_tokens: int = 32, progress_cb=None
    ) -> tuple[str, list[int], dict]:
        """Greedy-generate; returns (text, token ids, stats).

        Ids are identical to plain greedy paged generation by
        construction (see module docstring); callers verify that
        empirically. ``progress_cb``, when given, is called after each
        round with a dict of running counters (rounds/proposed/
        accepted/emitted) so long runs can log incrementally instead
        of only at the end.
        """
        model = self.model
        model.cache = {}
        prompt_ids = model.tok.encode(prompt)
        prefill = model.forward_all(prompt_ids, 0)
        p_prev = int(np.argmax(prefill[-1]))
        # forward_all never trims: restore the sliding bound for the
        # prompt before the first verification block.
        model.truncate_cache(len(prompt_ids))
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
