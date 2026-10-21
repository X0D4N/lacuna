"""Ordo — the routing automaton. A supportive technology for Lacuna.

Invention (2026-10-21). Lacuna's oracle predicts *which* pages a token
needs. Ordo learns something different: the *sequence structure* of
routing itself. A decode session is not a bag of popular experts — it
is a walk through phases (plan, write, debug, explain), and each phase
walks a characteristic path through the model's layers. Routing has
control flow, so Ordo treats the routing trace as a program:

- record the true routing trace (the pages each token actually used —
  the same ground truth the ShadowScorer already observes),
- compile it into a finite-state automaton whose states are *learned
  phase-clusters* of routing paths: training paths are clustered by
  overlap, each cluster is one location in the automaton's control
  flow, and transitions carry the pages the next token actually
  touched when the walk was at that location,
- compile each frequent state into a *segment*: the ranked union of
  pages expected before the state changes, sized to the hot-set
  budget, laid out as one contiguous extent on disk,
- at runtime, the previous token's path locates the automaton; its
  segment is prefetched into the cold end of the hot set — speculation
  may never displace proven work — and segment pages the next token
  actually uses are promoted by the touch itself.

State design note (measured, not assumed): an earlier state definition
— a coarse sketch of anchor-page touches — covered only 14% of next-
token pages at budget on phased synthetic data: too diffuse to be a
control-flow location. Learned path-clusters are the mechanism that
survived measurement; the sketch is recorded in docs/ORDO.md as a
falsified first design.

Why this is not a popularity manifest (research docs §5) or a
co-activation packing (§3): both are *unconditional* statistics — what
fires overall, what fires together. Ordo is *conditional on the path
actually taken*: the same expert lives in different segments under
different states, and the unit of layout is a control-flow segment,
not a frequency rank or a co-firing pair. And Ordo knows when it
doesn't know: if the traces show no sequential structure (many
shallow states instead of few deep ones), the structure gate closes
and the pager runs plain LRU — the simulator's memoryless falsifier
measures exactly that case.

Exactness contract (unchanged from Lacuna): Ordo only moves memory.
A wrong state, a stale segment, a missing trace — each costs faults
and speed, never output. ``ensure_resident`` still blocking-faults the
true page set before compute.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .prefetch import _fnv1a_64

# A page id is (layer, expert). A token path is the tuple of pages one
# token touched, in canonical (sorted) order.
Page = tuple[int, int]
TokenPath = tuple[Page, ...]

_OVERLAP_TAU = 0.15      # min path/centroid overlap to join a cluster
_MAX_CLUSTERS = 48       # control-flow locations are few by design
_CENTROID_SET = 256      # pages that define a cluster for overlap tests
UNKNOWN_STATE = -1


def path_signature(path: TokenPath) -> int:
    """Exact identity of one token's routing path (FNV-1a, layer-mixed)."""
    flat: list[int] = []
    for layer, expert in path:
        flat.append(layer * 1_000_003 + expert)
    return _fnv1a_64(flat, salt=0x04D0)


@dataclass
class RoutingTrace:
    """Ground-truth routing: one TokenPath per decoded token."""

    tokens: list[TokenPath] = field(default_factory=list)

    def record(self, pages: Iterable[Page]) -> None:
        self.tokens.append(tuple(sorted(pages)))

    def save(self, path: str) -> None:
        data = [[[layer, expert] for layer, expert in tok] for tok in self.tokens]
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)

    @classmethod
    def load(cls, path: str) -> "RoutingTrace":
        """Corruption-safe: any bad content yields an empty trace."""
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            tokens = [
                tuple(sorted((int(layer), int(expert)) for layer, expert in tok))
                for tok in data
            ]
            return cls(tokens=tokens)
        except Exception:
            return cls()


@dataclass
class _Cluster:
    counts: Counter = field(default_factory=Counter)
    members: int = 0
    _top: set = field(default_factory=set, repr=False)
    _dirty: bool = field(default=True, repr=False)

    def top_set(self) -> set:
        """The cluster's signature: pages present in at least a
        quarter of its member tokens (min count 1 while young).
        One-off background pages never become identity — without this,
        noise pages make every path 'match' every cluster."""
        if self._dirty:
            floor = max(1, int(0.25 * self.members))
            sig = [(p, c) for p, c in self.counts.items() if c >= floor]
            sig.sort(key=lambda kv: (-kv[1], kv[0]))
            self._top = {p for p, _ in sig[:_CENTROID_SET]}
            self._dirty = False
        return self._top

    def add(self, path: TokenPath) -> None:
        for page in path:
            self.counts[page] += 1
        self.members += 1
        # Refresh the overlap set periodically, not per token — a
        # slightly stale centroid is harmless during training.
        if self.members % 20 == 0:
            self._dirty = True

    def finalize(self) -> None:
        self._dirty = True
        self.top_set()


@dataclass
class OrdoAutomaton:
    """Finite-state automaton over learned routing-path clusters.

    transitions[state][page] = how often `page` was touched by the
    token following a token located at `state`.
    popularity[page] = global touch count — the degenerate fallback:
    with no sequential structure, prediction IS the popularity list.
    """

    transitions: dict[int, Counter] = field(default_factory=lambda: defaultdict(Counter))
    popularity: Counter = field(default_factory=Counter)
    state_support: Counter = field(default_factory=Counter)
    clusters: list[_Cluster] = field(default_factory=list)
    has_structure: bool = field(default=False)
    _pop_ranked: list[Page] | None = field(default=None, repr=False)
    _pred_cache: dict = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------- states

    def _locate(self, path: TokenPath, create: bool) -> int:
        """Cluster id whose centroid overlaps `path` most, or
        UNKNOWN_STATE. With create=True (training), a path that matches
        nothing starts a new location (up to the cluster cap)."""
        path_set = set(path)
        best, best_score = UNKNOWN_STATE, 0.0
        for idx, cluster in enumerate(self.clusters):
            score = len(path_set & cluster.top_set()) / max(1, len(path_set))
            if score > best_score:
                best, best_score = idx, score
        if best_score >= _OVERLAP_TAU:
            return best
        if create and len(self.clusters) < _MAX_CLUSTERS:
            self.clusters.append(_Cluster())
            return len(self.clusters) - 1
        return best if best_score >= _OVERLAP_TAU / 2 else UNKNOWN_STATE

    def state_of(self, path: TokenPath) -> int:
        return self._locate(path, create=False)

    # ------------------------------------------------------------ training

    @classmethod
    def train(cls, traces: Sequence[RoutingTrace]) -> "OrdoAutomaton":
        auto = cls()
        for trace in traces:
            for path in trace.tokens:
                for page in path:
                    auto.popularity[page] += 1
        for trace in traces:
            prev_state: int | None = None
            for path in trace.tokens:
                state = auto._locate(path, create=True)
                if state != UNKNOWN_STATE:
                    auto.clusters[state].add(path)
                if prev_state is not None and prev_state != UNKNOWN_STATE:
                    auto.state_support[prev_state] += 1
                    for page in path:
                        auto.transitions[prev_state][page] += 1
                prev_state = state
        for cluster in auto.clusters:
            cluster.finalize()
        # Structure gate (heuristic, measured on the training traces):
        # real control flow looks like FEW states with DEEP support.
        # Structureless routing explodes into many shallow states —
        # then speculation cannot pay for itself and the automaton
        # refuses to speculate (the pager falls back to plain LRU).
        # A production gate would use held-out coverage instead.
        if auto.state_support:
            mean_support = sum(auto.state_support.values()) / len(auto.state_support)
            auto.has_structure = len(auto.clusters) <= 16 and mean_support >= 20
        return auto

    def _popularity_ranked(self) -> list[Page]:
        if self._pop_ranked is None:
            self._pop_ranked = [
                page for page, _ in sorted(
                    self.popularity.items(), key=lambda kv: (-kv[1], kv[0]))
            ]
        return self._pop_ranked

    # ----------------------------------------------------------- prediction

    def predict_pages(self, state: int | None, budget_pages: int) -> list[Page]:
        """Ranked pages expected next from `state`, capped at budget.

        Conditional first: pages ranked by transition counts from this
        state. Remaining budget fills from global popularity (a new
        phase's early layers still fault less). Deterministic: ties
        break on page id. Unknown state -> pure popularity. Results
        are cached — the automaton is immutable after training.
        """
        key = (state, budget_pages)
        cached = self._pred_cache.get(key)
        if cached is not None:
            return list(cached)
        ranked: list[Page] = []
        seen: set[Page] = set()
        counter = self.transitions.get(state) if state is not None else None
        if counter:
            for page, _ in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
                ranked.append(page)
                seen.add(page)
                if len(ranked) >= budget_pages:
                    self._pred_cache[key] = tuple(ranked)
                    return ranked
        for page in self._popularity_ranked():
            if page not in seen:
                ranked.append(page)
                if len(ranked) >= budget_pages:
                    break
        self._pred_cache[key] = tuple(ranked)
        return ranked

    def compile_segments(
        self, budget_pages: int, min_support: int = 3
    ) -> dict[int, tuple[Page, ...]]:
        """Compile every sufficiently-observed state into a segment:
        the ordered pages of one contiguous disk extent."""
        segments: dict[int, tuple[Page, ...]] = {}
        for state, support in sorted(self.state_support.items()):
            if support >= min_support:
                segments[state] = tuple(self.predict_pages(state, budget_pages))
        return segments

    def coverage(self, trace: RoutingTrace, budget_pages: int) -> float:
        """Fraction of a held-out trace's pages the automaton predicts
        from the previous token's state. MEASURED on the trace."""
        hits = 0
        total = 0
        prev_state: int | None = None
        for path in trace.tokens:
            predicted = set(self.predict_pages(prev_state, budget_pages))
            hits += sum(1 for page in path if page in predicted)
            total += len(path)
            prev_state = self.state_of(path)
        return hits / total if total else 0.0
