"""Speculative draft-verify loop (M5) — the amortization engine.

A tiny draft model (fully resident) proposes K tokens; the big model verifies
all K in ONE sparse pass over the oracle's union mask. Weight-streaming cost is
paid once per K tokens instead of once per token.

Pluggable callables so the loop is testable without a real model:

- ``draft_fn(prefix_tokens, k) -> list[int]``: propose k draft token ids.
- ``verify_fn(prefix_tokens, draft_tokens) -> int``: how many leading drafts the
  target accepts (0..len(drafts)). The target is assumed to have run one pass.

K adapts: halve on a full rejection streak, grow by one on full acceptance
(clamped to [k_min, k_max]).
"""

from __future__ import annotations


def speculative_generate(
    draft_fn,
    verify_fn,
    max_tokens: int,
    k: int = 4,
    k_min: int = 1,
    k_max: int = 8,
    prefix: list[int] | None = None,
) -> dict:
    accepted: list[int] = list(prefix or [])
    out: list[int] = []
    proposed = 0
    verify_passes = 0

    while len(out) < max_tokens:
        drafts = list(draft_fn(accepted, k))
        if not drafts:
            break
        proposed += len(drafts)
        verify_passes += 1
        n = int(verify_fn(accepted, drafts))
        n = max(0, min(n, len(drafts)))
        accepted.extend(drafts[:n])
        out.extend(drafts[:n])

        if n == len(drafts) and n > 0:
            k = min(k_max, k + 1)  # full acceptance: be bolder
        elif n == 0:
            k = max(k_min, k // 2)  # full rejection: back off
        # else: partial acceptance, keep k

        if len(out) >= max_tokens:
            break
        if n == 0 and k == k_min:
            # degenerate draft loop guard: fall back to single-token stepping
            # is the caller's verify_fn returning 0 forever; stop to avoid spin.
            break

    out = out[:max_tokens]
    return {
        "tokens": out,
        "accepted_tokens": len(out),
        "proposed_tokens": proposed,
        "verify_passes": verify_passes,
        "acceptance_rate": (len(out) / proposed) if proposed else 0.0,
        "final_k": k,
    }
