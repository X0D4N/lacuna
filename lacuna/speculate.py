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

``speculative_tree_generate`` is the tree variant (SpecInfer / EAGLE-3 style):
the draft proposes a tree of candidate paths per round and the target verifies
it in one pass, accepting the longest greedily-agreed prefix. Exact by
construction — accepted tokens equal plain target-only greedy decoding.
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


def speculative_tree_generate(
    draft_tree_fn,
    verify_tree_fn,
    max_tokens: int,
    max_depth: int = 4,
    branching: int = 3,
    prefix: list[int] | tuple = (),
) -> dict:
    """Tree speculation (SpecInfer / EAGLE-3 style) — exact by construction.

    Each round the draft proposes a *tree* of candidate token paths; the target
    verifies the whole tree in ONE pass and the longest prefix path the target
    greedily agrees with is accepted. Because acceptance is defined as
    "the target's own greedy choice at every position", the accepted tokens
    are IDENTICAL to plain target-only greedy decoding — speculation can only
    change speed, never output.

    Callables:

    - ``draft_tree_fn(prefix) -> dict``: maps each candidate path (a tuple of
      token ids) to its prior. May optionally accept ``(prefix, depth,
      branching)`` — when it does, the adapted values are passed so the draft
      honors the adaptation; a 1-arg draft works too (adaptation is then
      advisory and reported in the stats).
    - ``verify_tree_fn(prefix, tree) -> tuple``: ONE target pass over the
      tree; returns the longest path the target accepts greedily (possibly
      empty).

    Depth/branching adapt: grow when the target keeps up with the full depth
    (acceptance >= depth), shrink on empty rounds. Two consecutive empty
    rounds end generation rather than spinning.
    """
    accepted: list[int] = list(prefix or [])
    out: list[int] = []
    total_acc = 0
    rounds = 0
    empty_rounds = 0
    depth, branch = int(max_depth), int(branching)
    depth_history = [depth]

    while len(out) < max_tokens:
        try:
            tree = draft_tree_fn(accepted, depth, branch)
        except TypeError:
            tree = draft_tree_fn(accepted)  # 1-arg draft: adaptation advisory
        if not tree:
            break
        rounds += 1
        path = tuple(verify_tree_fn(accepted, tree))
        acc = len(path)
        accepted.extend(path)
        out.extend(path)
        total_acc += acc

        if acc == 0:
            empty_rounds += 1
            depth = max(1, depth - 1)
            branch = max(1, branch - 1)
            if empty_rounds >= 2:
                break  # draft cannot cover the target; stop, don't spin
        elif acc >= depth:
            depth = min(8, depth + 1)  # target keeps up: be bolder
            branch = min(8, branch + 1)
        # partial acceptance: hold steady
        depth_history.append(depth)
        if len(out) >= max_tokens:
            break

    tokens = out[:max_tokens]
    return {
        "tokens": tokens,
        "accepted_tokens": len(tokens),
        "rounds": rounds,
        "mean_acceptance_length": (total_acc / rounds) if rounds else 0.0,
        "final_depth": depth,
        "final_branching": branch,
        "depth_history": depth_history,
        "empty_acceptances": empty_rounds,
    }
