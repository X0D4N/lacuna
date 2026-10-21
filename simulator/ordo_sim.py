"""Ordo simulator — SIMULATED numbers only.

Measures steady-state disk bytes/token at the FLOOR hot-set cap
(4.5 GB) for two reference model classes, under three paging policies:

  lru         plain LRU hot set (Lacuna without a learned prior)
  popularity  session-start preload of the globally hottest pages
              (the research §5 manifest, unconditional)
  ordo        Ordo routing automaton: state-conditional segment
              prefetch + segment pinning (lacuna/ordo.py)

Workloads:
  phased      sessions walk topic phases (~150 tokens each); routing
              has sequential structure for Ordo to learn.
  memoryless  FALSIFIER: tokens drawn iid from one global popularity.
              Ordo must NOT beat popularity here; if it does, the sim
              is wrong, not the physics.

Every disk read counts — prefetch and faults alike. A prefetch of a
page that would have faulted anyway saves zero bytes; Ordo can only
win by keeping the RIGHT pages resident (state-conditional residency),
never by reading less than the model needs.
"""

from __future__ import annotations

import bisect
import random
import sys
from collections import OrderedDict
from dataclasses import dataclass

sys.path.insert(0, __file__.rsplit("/", 2)[0])

from lacuna.ordo import UNKNOWN_STATE, OrdoAutomaton, RoutingTrace  # noqa: E402

GB = 1e9
FLOOR_CAP_BYTES = 4.5 * GB


@dataclass(frozen=True)
class ModelClass:
    name: str
    layers: int
    experts: int
    top_k: int
    active_bytes: float  # cold disk bytes per token

    @property
    def pages_per_token(self) -> int:
        return self.layers * self.top_k

    @property
    def page_bytes(self) -> float:
        return self.active_bytes / self.pages_per_token

    @property
    def cap_pages(self) -> int:
        return int(FLOOR_CAP_BYTES / self.page_bytes)


FLASH = ModelClass("flash-6B-active", layers=48, experts=256, top_k=10, active_bytes=3.0 * GB)
TRILLION = ModelClass("trillion-32B-active", layers=61, experts=384, top_k=8, active_bytes=16.0 * GB)


# --------------------------------------------------------------- workloads

class Workload:
    """Weighted per-(topic, layer) expert sampler."""

    def __init__(self, model: ModelClass, topics: int, seed: int, phased: bool,
                 phase_len: float = 150.0):
        self.model = model
        self.phased = phased
        self.phase_len = phase_len
        rng = random.Random(seed)
        self.topics = topics if phased else 1
        # cum_weights[topic][layer] = (experts_by_rank, cumulative weights)
        self.cum: list[list[tuple[list[int], list[float]]]] = []
        for _ in range(self.topics):
            per_layer = []
            for _layer in range(model.layers):
                order = list(range(model.experts))
                rng.shuffle(order)
                if phased:
                    pref = max(model.top_k * 3, model.experts // 8)
                    weights = []
                    for rank, _e in enumerate(order):
                        if rank < pref:
                            weights.append(1.0 / (rank + 1) ** 1.1)
                        else:
                            weights.append(0.002 / (rank + 1) ** 0.5)
                else:  # one global Zipf popularity, iid across tokens
                    weights = [1.0 / (rank + 1) ** 1.1 for rank in range(model.experts)]
                cum = []
                total = 0.0
                for w in weights:
                    total += w
                    cum.append(total)
                per_layer.append((order, cum))
            self.cum.append(per_layer)

    def _sample_layer(self, rng: random.Random, topic: int, layer: int) -> list[int]:
        order, cum = self.cum[topic][layer]
        total = cum[-1]
        picked: list[int] = []
        seen: set[int] = set()
        while len(picked) < self.model.top_k:
            idx = bisect.bisect_left(cum, rng.random() * total)
            expert = order[min(idx, self.model.experts - 1)]
            if expert not in seen:
                seen.add(expert)
                picked.append(expert)
        return picked

    def session(self, seed: int, tokens: int = 600) -> RoutingTrace:
        rng = random.Random(seed)
        topic = rng.randrange(self.topics)
        trace = RoutingTrace()
        for t in range(tokens):
            if self.phased and t > 0 and rng.random() < 1.0 / self.phase_len:
                topic = rng.randrange(self.topics)
            pages = []
            for layer in range(self.model.layers):
                for expert in self._sample_layer(rng, topic, layer):
                    pages.append((layer, expert))
            trace.record(pages)
        return trace


# ----------------------------------------------------------------- pagers

class Pager:
    """Resident-set pager. During a token's serve, its own path is
    pinned — a page is never evicted mid-use (the real engine's
    ensure_resident semantic). Policies differ only in what they do
    BETWEEN tokens, which is the honest contest."""

    def __init__(self, model: ModelClass):
        self.model = model
        self.cap = model.cap_pages
        self.resident: OrderedDict = OrderedDict()
        self.bytes_fault = 0.0   # demand reads: scattered, random-rate
        self.bytes_spec = 0.0    # speculative bulk reads: sequential-rate

    @property
    def bytes_read(self) -> float:
        return self.bytes_fault + self.bytes_spec

    def _evict(self, pinned: frozenset) -> None:
        excess = len(self.resident) - self.cap
        if excess <= 0:
            return
        victims = []
        for page in list(self.resident):  # LRU order, oldest first
            if len(victims) >= excess:
                break
            if page not in pinned:
                victims.append(page)
        for page in victims:
            del self.resident[page]
        while len(self.resident) > self.cap:  # all-pinned safety net
            self.resident.popitem(last=False)

    def load(self, page, pinned: frozenset = frozenset(), front: bool = False) -> None:
        """front=True inserts a speculative page at the COLD end: it is
        evicted before any proven page, and a prefetch hit is left
        untouched (speculation never promotes). Speculative loads are
        bulk sequential reads; demand loads are scattered faults."""
        if page in self.resident:
            if not front:
                self.resident.move_to_end(page)
            return
        if front:
            self.bytes_spec += self.model.page_bytes
        else:
            self.bytes_fault += self.model.page_bytes
        self.resident[page] = True
        if front:
            self.resident.move_to_end(page, last=False)
        self._evict(pinned)

    def serve(self, path, pinned: frozenset = frozenset()) -> None:
        """Pages are used layer by layer (the path is layer-sorted);
        only the current layer's pages are pinned. When one token's
        path exceeds the cap — the trillion class on the floor — this
        is the honest streaming model: the token's own stream evicts
        its early layers, and cross-token hits mostly vanish."""
        group: list = []
        cur_layer = None
        for page in path:
            if page[0] != cur_layer:
                if group:
                    keep = pinned | frozenset(group)
                    for p in group:
                        self.load(p, keep)
                group = []
                cur_layer = page[0]
            group.append(page)
        if group:
            keep = pinned | frozenset(group)
            for p in group:
                self.load(p, keep)


class PopularityPager(Pager):
    def __init__(self, model: ModelClass, automaton: OrdoAutomaton):
        super().__init__(model)
        self.preload = automaton.predict_pages(None, self.cap)

    def begin_session(self) -> None:
        for page in self.preload:
            self.load(page, front=True)  # one bulk sequential preload


class OrdoPager(Pager):
    """Ordo policy: LRU within a state; at a state TRANSITION, demote
    the old state's segment (dead pages leave first) and prefetch the
    new state's segment into the cold end. Speculation fires once per
    phase change, never per token — its cost is amortized over the
    phase, and it can only evict pages outside (current path ∪ new
    segment)."""

    def __init__(self, model: ModelClass, automaton: OrdoAutomaton):
        super().__init__(model)
        self.automaton = automaton
        self._pred_cache: dict = {}
        self._cur_state = None
        self._cur_segment: frozenset = frozenset()

    def _predicted(self, state):
        if state not in self._pred_cache:
            self._pred_cache[state] = self.automaton.predict_pages(state, self.cap)
        return self._pred_cache[state]

    def run_session(self, trace: RoutingTrace, warm_from: int) -> tuple[float, float]:
        """Returns (fault bytes/token, speculative bytes/token) over
        the steady-state window."""
        start_fault = 0.0
        start_spec = 0.0
        gated = not self.automaton.has_structure
        for t, path in enumerate(trace.tokens):
            if t == warm_from:
                start_fault = self.bytes_fault
                start_spec = self.bytes_spec
            self.serve(path)
            if gated:
                continue
            state = self.automaton.state_of(path)
            if state == UNKNOWN_STATE or state == self._cur_state:
                continue
            segment = frozenset(self._predicted(state))
            dead = [p for p in self.resident
                    if p in self._cur_segment and p not in segment]
            for page in dead:  # demote: evict-first, not deleted
                self.resident.move_to_end(page, last=False)
            keep = frozenset(path)
            for page in self._predicted(state):
                self.load(page, keep, front=True)
            self._cur_state = state
            self._cur_segment = segment
        n = len(trace.tokens) - warm_from
        return ((self.bytes_fault - start_fault) / n,
                (self.bytes_spec - start_spec) / n)


SEQ_RATE = 3.5 * GB    # floor NVMe sequential (the drive's spec class)
RAND_RATE = 2.0 * GB   # EST: scattered 2 MB reads on a budget NVMe


def _time(fault_b: float, spec_b: float, rand_rate: float = RAND_RATE) -> float:
    return fault_b / rand_rate + spec_b / SEQ_RATE


def run_policy(model: ModelClass, kind: str, automaton: OrdoAutomaton,
               sessions: list[RoutingTrace], warm_from: int = 100) -> dict[str, float]:
    """Mean steady-state bytes/token split by read kind. SIMULATED."""
    faults, specs = [], []
    for trace in sessions:
        if kind == "ordo":
            pager = OrdoPager(model, automaton)
            f, s = pager.run_session(trace, warm_from)
            faults.append(f)
            specs.append(s)
            continue
        pager = PopularityPager(model, automaton) if kind == "popularity" else Pager(model)
        if kind == "popularity":
            pager.begin_session()
        start_f = 0.0
        start_s = 0.0
        for t, path in enumerate(trace.tokens):
            if t == warm_from:
                start_f = pager.bytes_fault
                start_s = pager.bytes_spec
            pager.serve(path)
        n = len(trace.tokens) - warm_from
        faults.append((pager.bytes_fault - start_f) / n)
        specs.append((pager.bytes_spec - start_s) / n)
    f = sum(faults) / len(faults)
    s = sum(specs) / len(specs)
    return {"gb": (f + s) / GB, "sec": _time(f, s), "fault_gb": f / GB, "spec_gb": s / GB}


def evaluate(model: ModelClass, phased: bool, seed: int = 7) -> dict:
    wl = Workload(model, topics=5, seed=seed, phased=phased)
    train = [wl.session(seed=1000 + i) for i in range(6)]
    test = [wl.session(seed=2000 + i) for i in range(4)]
    automaton = OrdoAutomaton.train(train)
    pop_set = set(automaton.predict_pages(None, model.cap_pages))
    pop_hits = sum(1 for t in test for tok in t.tokens for pg in tok if pg in pop_set)
    pop_total = sum(len(tok) for t in test for tok in t.tokens)
    return {
        "lru": run_policy(model, "lru", automaton, test),
        "popularity": run_policy(model, "popularity", automaton, test),
        "ordo": run_policy(model, "ordo", automaton, test),
        "ordo_coverage": sum(automaton.coverage(t, model.cap_pages) for t in test) / len(test),
        "pop_coverage": pop_hits / pop_total if pop_total else 0.0,
        "clusters": len(automaton.clusters),
    }


def main() -> int:
    print("SIMULATED — Ordo vs baselines, floor cap 4.5 GB, steady state")
    print(f"time model: faults @ {RAND_RATE / GB:.1f} GB/s (EST scattered), "
          f"speculative bulk @ {SEQ_RATE / GB:.1f} GB/s sequential")
    for model in (FLASH, TRILLION):
        for phased in (True, False):
            r = evaluate(model, phased)
            label = "phased    " if phased else "memoryless"
            print(f"\n{model.name}  cold={model.active_bytes / GB:.1f} GB/token  "
                  f"cap={model.cap_pages} pages  workload={label}  "
                  f"clusters={r['clusters']}")
            for kind in ("lru", "popularity", "ordo"):
                d = r[kind]
                print(f"  {kind:<11} {d['gb']:6.2f} GB/token   {d['sec']:5.2f} s/token")
            print(f"  page coverage at cap budget: ordo {r['ordo_coverage']:.1%} "
                  f"vs popularity {r['pop_coverage']:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
