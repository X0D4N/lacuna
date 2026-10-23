# Ordo — the routing automaton (a supportive technology for Lacuna)

Design background: [SUPPORT_TECH_RESEARCH.md](SUPPORT_TECH_RESEARCH.md).
Status: **prototype** — `lacuna/ordo.py` + `simulator/ordo_sim.py`;
simulated numbers only, real-trace measurement still open.

## The idea in one paragraph

Lacuna's oracle predicts *which* pages a token needs. Ordo learns the
*sequence structure* of routing: a decode session walks through phases
(plan, write, debug, explain), and each phase walks a characteristic
path through the model's layers. Ordo records the true routing trace,
compiles it into a finite-state automaton whose states are **learned
phase-clusters of routing paths**, and compiles each frequent state
into a **segment** — the ranked union of pages expected before the
state changes, laid out as one contiguous extent. At a state
transition, the old segment is demoted and the new segment is read in
one bulk sequential pass. Routing has control flow; Ordo is the pager
that knows it.

## Why it is not the crowded ideas

- **Not a popularity manifest** (research §5): popularity is
  unconditional — one global ranking. Ordo's prediction is conditional
  on the state the walk is in; the same expert sits in different
  segments under different states. Popularity survives inside Ordo as
  the fallback for unknown states, and when traces show no sequential
  structure at all, the structure gate (few deep states vs many
  shallow ones) stops Ordo from speculating — it runs as plain LRU
  rather than paying a speculation tax.
- **Not co-activation packing** (§3): co-activation is a pairwise,
  atemporal statistic. A segment is a control-flow artifact: an
  ordered extent compiled from a state's *outgoing transitions*.
- **Not a twin/draft router predictor** (§1, MoE-SpeQ et al.): no
  second model, no neural predictor. The automaton is a few counters
  per state, trained on the engine's own observed traces, inspectable,
  deterministic.

## Mechanism

1. **Trace.** Every run already observes ground-truth routing (the
   ShadowScorer's actual set). Ordo persists it (`RoutingTrace`,
   corruption-safe) — KBs per session.
2. **States.** Training paths are clustered online by overlap with a
   cluster's *signature* (pages present in ≥25% of its member tokens;
   background noise never becomes identity). A path matching nothing
   starts a new state, up to 48. The previous token's cluster is the
   automaton's location.
3. **Segments.** `compile_segments` turns each state's transition
   counts into a ranked page list, truncated at the hot-set budget —
   the physical layout unit: one state = one contiguous extent.
4. **Runtime.** Serve is plain pinned LRU within a state. On a state
   transition only: pages of the old segment outside the new one are
   demoted to evict-first, and the new segment is bulk-loaded at the
   cold end. Speculation fires once per phase change, never per
   token, and may never evict the current token's path.

## Exactness

Unchanged from Lacuna, by construction: Ordo only moves memory. A
wrong state, a stale segment, a missing trace — each costs faults or
speed, never output. `ensure_resident` still blocking-faults the true
page set before compute. Test:
`test_wrong_state_costs_speed_only`.

## What measurement killed (kept honest, on purpose)

Two designs were built and measured before this one survived:

- **Anchor-sketch states** (hash of coarse anchor-group touches):
  14% next-token page coverage at budget on phased synthetic data —
  too diffuse to be a control-flow location. Replaced by learned
  path-clusters (56% on the same class of data).
- **Per-token segment prefetch** (load the whole predicted segment
  every token): 3–6× *worse* than LRU on bytes/token. At floor
  proportions the cache turns over every ~1.5 tokens; a broad
  speculative load mostly evicts proven pages and pays bytes for
  pages never used. This is why the shipped policy speculates only
  at state transitions.

## Simulated results

`simulator/ordo_sim.py` — synthetic workloads, seeded; every disk
read counted (speculative and fault alike); floor cap 4.5 GB;
time model: demand faults at 2.0 GB/s (EST, scattered 2 MB reads on
a budget NVMe), speculative bulk at 3.5 GB/s sequential.

| Workload (SIMULATED) | LRU | Popularity | Ordo | Ordo coverage vs popularity |
|---|---|---|---|---|
| 6B-active, phased | 1.48 GB/tok · 0.74 s | 1.48 · 0.74 s | 1.50 · 0.75 s | **72.5% vs 37.4%** |
| 6B-active, memoryless (falsifier) | 2.19 · 1.09 s | 2.19 · 1.09 s | **2.19 · 1.09 s (gate closed — exactly LRU)** | 50.1% vs 50.6% |
| 32B-active, phased | 16.00 · 8.00 s | 16.00 · 8.00 s | 16.01 · 8.00 s | **23.2% vs 7.8%** |
| 32B-active, memoryless (falsifier) | 16.00 · 8.00 s | 16.00 · 8.00 s | **16.00 · 8.00 s (gate closed)** | 19.5% vs 19.6% |

Reading the table straight:

- **As a predictor, Ordo is 2–3× better than popularity** at the
  same budget (coverage rows). State-conditioning is real signal.
- **As a pager, Ordo is at parity with LRU on bytes** (+1.4% on
  phased 6B, +0.06% on phased 32B) and *exactly* LRU when the
  structure gate closes. The gate exists because the first ungated
  run measured Ordo 26% *worse* on memoryless routing — speculation
  without structure is a tax, so Ordo refuses to speculate there.
- **The 32B-active wall is residency physics, not prediction:**
  the floor cap (137 pages) is smaller than one token's demand
  (488 pages), so every policy streams ~the full 16 GB/token. The
  only lever left in that regime is read *rate* — sequential
  segment extents at 3.5 GB/s would put the same bytes at
  ~4.6 s/token instead of 8.0 — which is what the compiled-segment
  layout is for. Claimed here as arithmetic, not as a measurement.

## Honest boundaries

- **Steady-state bytes are a residency problem, not a prediction
  problem.** At floor proportions (cap ≈ 1.5× one token's active
  set), LRU already tracks the local distribution; the automaton's
  measured byte effect is at transitions and in prediction quality,
  not in magically shrinking the working set. Where the session
  working set is irreducibly ≫ cap, no memory-side policy wins —
  the research report's falsifier stands.
- **Popularity concentration at 384-expert scale is assumed, not
  proven** — the same open empirical bet as the rest of the program.
  Real traces (llama.cpp attach point) decide it.
- All numbers above are SIMULATED on synthetic routing. Nothing here
  has touched a real model's router yet.

## Appends to Lacuna

- `lacuna/prefetch.py` — `ShadowScorer` is the trace tap;
  `RoutingTrace` persists beside it.
- `lacuna/ordo.py` — the automaton (`OrdoAutomaton.train`,
  `predict_pages`, `compile_segments`).
- `lacuna/pager.py` (future) — `.lac` gains a compiled-segment
  section: state extents in traversal order, checksum-verified,
  canonical layout as fallback.
- `lacuna/scheduler.py` (future) — state transitions become
  scheduled bulk reads; the warm-start cache upgrades from a flat
  snapshot to traces + segments.
