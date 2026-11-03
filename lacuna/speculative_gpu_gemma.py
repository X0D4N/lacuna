"""Speculative paged inference for Gemma-4 (GPU path).

Mirrors :mod:`lacuna.speculative_gemma` (the CPU Gemma speculative
decoder) for :class:`~lacuna.gpu_executor.GPUGemmaModel`, reusing the
same :class:`~lacuna.speculative.NgramDraft`, the same pending-token
trick, and the same stats. The decoder logic is identical; only the
target model differs.

The three gaps that used to block a GPU Gemma speculative decoder
are closed in :mod:`lacuna.gpu_executor` (see its
``GPUGemmaModel.forward_all`` / ``truncate_cache`` and the
``trim`` flag on ``GemmaGPULayer.forward``):

1. Per-position forward: ``GPUGemmaModel.forward_all`` returns
   logits for every input position, so one call verifies a whole
   draft block (``forward`` returns last-position logits only).
2. Uniform absolute-position rollback: ``truncate_cache`` takes one
   absolute position and is correct for both layer types — global
   caches hold exactly ``[0, n_kept)``, sliding caches hold the last
   window of ``[0, n_kept)`` — so a single call restores exactly the
   accepted prefix on every layer.
3. No-trim-during-verification: ``forward_all`` runs layers with
   ``trim=False``, so the cache grows transiently by the block
   length and trimming is deferred to ``truncate_cache``. Trimming
   during verification would move the window head past keys a
   rollback to the accepted prefix still needs.

Exactness (why output is identical to plain greedy by construction):
same argument as the CPU decoder — an accepted draft token equals
the target argmax at its position, a resample/bonus token IS the
target argmax — plus the cache argument: after every block the
cache state is exactly the state plain greedy decoding would hold
after the same emitted prefix (pinned by test, including a long
prompt that forces repeated sliding trims). The one operation that
cannot be repaired from cache is a rollback to a position at or
before a sliding layer's current window head (those keys were
already trimmed by an earlier block); ``truncate_cache`` fails loud
there instead of silently corrupting attention, and the decoder
never issues such a rollback (its rollback distance is at most one
block, far inside the window).

Tolerance note: GPU and CPU attention run in different layouts
(batched-head matmul vs the CPU path's einsum), so GPU-vs-CPU
logits are compared at 2e-3 absolute with an EXACT greedy-choice
requirement — the same near-tie discipline as the dense GPU path.
GPU spec-vs-GPU plain, however, runs the same backend code path
through the same layer math, so those ids must match exactly (and
do, pinned by test). A wide-margin flip anywhere is a bug, not a
tolerance.
"""

from __future__ import annotations

import numpy as np

from .speculative import NgramDraft  # noqa: F401  (re-exported for callers)

__all__ = ["GPUGemmaSpeculativeDecoder", "NgramDraft"]


class GPUGemmaSpeculativeDecoder:
    """Greedy speculative decoding over a ``GPUGemmaModel`` target.

    Invariants (see module docstring): ``model`` cache covers the
    emitted prefix — global layers fully, sliding layers as the last
    window of it; if ``cached < len(prefix)`` the last emitted token
    is pending (emitted but not yet processed) and opens the next
    verification block. Every block is exactly one ``forward_all``
    call and yields >= 1 token.
    """

    def __init__(self, model, draft, k: int = 4):
        self.model = model
        self.draft = draft
        self.k = int(k)

    def generate(
        self, prompt: str, max_tokens: int = 32, progress_cb=None
    ) -> tuple[str, list[int], dict]:
        """Greedy-generate; returns (text, token ids, stats).

        Ids are identical to plain greedy GPU generation by
        construction (see module docstring); callers verify that
        empirically. ``progress_cb``, when given, is called after each
        round with a dict of running counters (rounds/proposed/
        accepted/emitted) so long runs can log incrementally instead
        of only at the end.
        """
        model = self.model
        model.reset_cache()
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
