"""Zero-training prefetch primitives: the measurement-first half of the 9/10 program.

The 9/10 program moves Lacuna's confidence from "the simulator says" to
"measured on real weights" with three inventions that need no ML training:

1. :class:`NgramPrefetcher` -- deterministic prefetch for n-gram embedding
   tables. Qwen3.8-Flash-Next carries 51B n-gram embedding params (20M rows,
   bigrams/trigrams); which rows an input needs is a pure function of the
   input text, so this needs zero ML -- hash the n-grams, get the exact row
   set. The oracle never has to learn it.

2. :class:`CacheExpertPredictor` -- CPU-cache-style online predictor for MoE
   expert prefetch. Per-expert 2-bit saturating counters (recency/frequency,
   like a cache replacement policy) plus a bigram Markov table over expert
   activation sequences, updated online per token. Zero training, adapts when
   the workload shifts.

3. :class:`ShadowScorer` -- the instrument. Feed it (predicted page set,
   actual required page set) per decode step; it reports hit rate, recall,
   precision, and fault counts. It never touches control flow or output --
   pure measurement -- so it can ride along on a real inference run and tell
   us exactly how good the prefetch contract is on real weights.

stdlib + numpy only. No network, no torch, no transformers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set

import numpy as np

__all__ = [
    "NgramPrefetcher",
    "NgramSpec",
    "CacheExpertPredictor",
    "ShadowScorer",
]


# ---------------------------------------------------------------------------
# 1. Deterministic n-gram embedding prefetch
# ---------------------------------------------------------------------------

_FNV_OFFSET_64 = 0xCBF29CE484222325
_FNV_PRIME_64 = 0x100000001B3
_MASK64 = 0xFFFFFFFFFFFFFFFF


def _fnv1a_64(values: Sequence[int], salt: int) -> int:
    """FNV-1a over a sequence of ints. Stable across processes and machines
    (unlike builtin hash(), which is salted per-process for str/bytes)."""
    h = _FNV_OFFSET_64 ^ (salt & _MASK64)
    for v in values:
        if v < 0:
            raise ValueError(f"token id must be non-negative, got {v}")
        h ^= v & _MASK64
        h = (h * _FNV_PRIME_64) & _MASK64
    h ^= len(values)  # separate bigram space from trigram space
    h = (h * _FNV_PRIME_64) & _MASK64
    return h


@dataclass(frozen=True)
class NgramSpec:
    """Shape of an n-gram embedding table (cf. Qwen3.8-Flash-Next: 20M rows,
    bigrams/trigrams looked up at layer 2)."""

    table_size: int
    sizes: tuple = (2, 3)
    salt: int = 0

    def __post_init__(self) -> None:
        if self.table_size <= 0:
            raise ValueError("table_size must be positive")
        if not self.sizes or any(n < 1 for n in self.sizes):
            raise ValueError("sizes must be a non-empty tuple of positive ints")


class NgramPrefetcher:
    """Compute the EXACT set of embedding rows an input needs. Pure function,
    zero ML: row index = FNV-1a(n-gram, salt) % table_size.

    The returned row indices map to weight pages through the pager's
    rows-per-page (see ``lacuna/pager.py``); prefetch those pages and the
    51B n-gram table never faults on a hot path. Deterministic: same input
    tokens always yield the same row set, on any machine.
    """

    def __init__(self, table_size: int, sizes: tuple = (2, 3), salt: int = 0) -> None:
        self.spec = NgramSpec(table_size=table_size, sizes=tuple(sizes), salt=salt)

    def rows_for(self, token_ids: Sequence[int]) -> Set[int]:
        """Return the set of embedding row indices in [0, table_size) needed
        for ``token_ids``. Empty input (or no complete n-gram) -> empty set."""
        tokens = [int(t) for t in token_ids]
        rows: Set[int] = set()
        for n in self.spec.sizes:
            if len(tokens) < n:
                continue
            for i in range(len(tokens) - n + 1):
                idx = _fnv1a_64(tokens[i : i + n], self.spec.salt) % self.spec.table_size
                rows.add(idx)
        return rows


# ---------------------------------------------------------------------------
# 2. Cache-style online expert predictor (MoE prefetch, zero training)
# ---------------------------------------------------------------------------

class CacheExpertPredictor:
    """Predict the next token's active MoE experts with no training.

    Two CPU-cache ideas combined:

    * per-expert 2-bit saturating counters (0..3): active -> +1, inactive
      -> -1. Captures recency/frequency like a cache replacement policy.
    * bigram Markov table ``P(next | prev)`` over expert activation sets:
      on ``observe(prev_set -> cur_set)`` every ``(p, c)`` pair with
      ``p in prev_set, c in cur_set`` gets +1, and the whole table decays
      multiplicatively each step so the predictor adapts when the workload
      shifts (no retraining, ever).

    ``predict(k)`` scores each expert as
    ``counter_weight * counter + markov_weight * sum_{p in last} P(. | p)``
    and returns the top-k ids (ties broken by expert id for determinism).
    """

    def __init__(
        self,
        n_experts: int,
        counter_weight: float = 1.0,
        markov_weight: float = 1.0,
        decay: float = 0.99,
    ) -> None:
        if n_experts <= 0:
            raise ValueError("n_experts must be positive")
        if not 0.0 < decay <= 1.0:
            raise ValueError("decay must be in (0, 1]")
        self.n_experts = n_experts
        self.counter_weight = float(counter_weight)
        self.markov_weight = float(markov_weight)
        self.decay = float(decay)
        self._counters = np.zeros(n_experts, dtype=np.int8)  # 2-bit saturating
        self._trans = np.zeros((n_experts, n_experts), dtype=np.float32)
        self._last: List[int] | None = None

    def observe(self, active_expert_ids: Iterable[int]) -> None:
        """Record the expert set activated by one token. Call once per
        decode step, in order."""
        active = [int(e) for e in active_expert_ids]
        for e in active:
            if not 0 <= e < self.n_experts:
                raise ValueError(f"expert id {e} out of range [0, {self.n_experts})")
        active_set = set(active)

        # 2-bit saturating counters: reward active, decay inactive.
        self._counters[list(active_set)] = np.minimum(
            3, self._counters[list(active_set)] + 1
        )
        inactive_mask = np.ones(self.n_experts, dtype=bool)
        inactive_mask[list(active_set)] = False
        self._counters[inactive_mask] = np.maximum(
            0, self._counters[inactive_mask] - 1
        )

        # Bigram Markov update with global decay (adapts to pattern shifts).
        self._trans *= self.decay
        if self._last is not None:
            for p in self._last:
                for c in active_set:
                    self._trans[p, c] += 1.0
        self._last = sorted(active_set)

    def predict(self, k: int) -> List[int]:
        """Return the top-k predicted expert ids for the next token."""
        if k <= 0:
            raise ValueError("k must be positive")
        k = min(k, self.n_experts)
        scores = self.counter_weight * self._counters.astype(np.float32)
        if self._last is not None:
            scores = scores + self.markov_weight * self._trans[self._last, :].sum(axis=0)
        # Deterministic: sort by (-score, expert id).
        order = sorted(range(self.n_experts), key=lambda e: (-scores[e], e))
        return order[:k]

    def reset(self) -> None:
        """Clear all learned state (counters, Markov table, history)."""
        self._counters[:] = 0
        self._trans[:, :] = 0.0
        self._last = None


# ---------------------------------------------------------------------------
# 3. Shadow scorer: measure prediction quality with zero output effect
# ---------------------------------------------------------------------------

class ShadowScorer:
    """Score page-prediction quality without affecting anything.

    Feed :meth:`observe_step` the (predicted page set, actually-required page
    set) for each decode step. It accumulates hit rate, micro recall /
    precision, and fault counts, and :meth:`report` returns them. It never
    influences prefetch decisions or model output -- it only reads the two
    sets, so it can ride along on a live run as pure instrumentation.

    Attach point for real weights (future work, not built here): in
    ``integrations/llama.cpp`` the per-step ``required_pages_for_step`` hook
    (see ``integrations/llama.cpp/INTEGRATION.md`` and the ``sp_buffer``
    required-pages callback) knows both the oracle's predicted union mask and
    the pages the step actually touched. Calling
    ``shadow.observe_step(set(predicted_union), set(touched_pages))`` there,
    once per decode step, turns "the simulator says 90% hit rate" into a
    measured number on real weights -- that is the step that takes Lacuna's
    confidence from 6/10 to 9/10.
    """

    def __init__(self) -> None:
        self._steps = 0
        self._hits = 0          # steps with predicted ⊇ actual (zero faults)
        self._fault_steps = 0   # steps with at least one missed page
        self._overlap = 0       # Σ |predicted ∩ actual|
        self._actual = 0        # Σ |actual|
        self._predicted = 0     # Σ |predicted|
        self._missed = 0        # Σ |actual - predicted|  (fault pages)

    def observe_step(
        self, predicted: Iterable[int], actual: Iterable[int]
    ) -> Dict[str, float]:
        """Record one decode step. Returns this step's stats. Inputs are not
        mutated."""
        p = set(predicted)
        a = set(actual)
        overlap = len(p & a)
        missed = len(a - p)
        hit = missed == 0

        self._steps += 1
        self._hits += 1 if hit else 0
        self._fault_steps += 0 if hit else 1
        self._overlap += overlap
        self._actual += len(a)
        self._predicted += len(p)
        self._missed += missed

        return {
            "hit": float(hit),
            "recall": overlap / len(a) if a else 1.0,
            "precision": overlap / len(p) if p else (1.0 if not a else 0.0),
            "missed_pages": float(missed),
        }

    def report(self) -> Dict[str, float]:
        """Aggregate metrics over all observed steps."""
        if self._steps == 0:
            raise ValueError("no steps observed")
        return {
            "steps": float(self._steps),
            # Fraction of steps with zero page faults.
            "hit_rate": self._hits / self._steps,
            # Micro-averaged over pages: of all required pages, how many were prefetched.
            "recall": self._overlap / self._actual if self._actual else 1.0,
            # Micro-averaged over pages: of all prefetched pages, how many were needed.
            "precision": self._overlap / self._predicted if self._predicted else 1.0,
            "fault_step_rate": self._fault_steps / self._steps,
            "total_fault_pages": float(self._missed),
            # >1 means over-prefetching (bandwidth waste); <1 is impossible
            # when recall is 1.0.
            "overfetch_ratio": self._predicted / self._actual if self._actual else 0.0,
        }
