# Support technology research — a companion for Lacuna

Sweep date: 2026-10-20. Question: what *supportive* technology, appended to
Lacuna (lookahead sparse-paged inference; core rule — prediction guides
MEMORY, never math, so output stays bit-identical to the full model), can cut
**steady-state disk bytes per token** far below the cold number for
near-trillion MoE models on the **FLOOR** machine?

## Target (corrected 2026-10-20 — FLOOR, not MID)

- RTX 2050, 4 GB VRAM (Lacuna cap 3.2 GB), **6 GB system RAM (Lacuna hot-set
  cap 4.5 GB)**, 500 GB NVMe at ~3.5 GB/s.
- Reference model class: **32B active/token** (Kimi K2.7 Code: 1T total, 384
  experts, 8+1 routed, native INT4). Cold math: 32B × 0.5 B = **16 GB/token**
  → 0.22 tok/s at 3.5 GB/s. That is the number to beat.
- The binding constraint on the floor is the **4.5 GB hot-set budget**, not
  bandwidth alone. Favored mechanisms are therefore ones that (a) shrink the
  resident working set that actually matters, or (b) multiply *effective*
  bandwidth (fewer, larger, sequential reads; reads that fetch future needs
  as a side effect).
- Measurement target: **steady-state bytes/token ≤ 3.5 GB (≥ 1 tok/s)** for a
  32B-active model under a 4.5 GB hot set. This is a target to be measured,
  not a promise — see §8 for what could falsify it.

Honesty conventions: every performance figure below is as reported by its
source (index-grade, not independently verified). Estimates are marked EST.
Speculation is marked as such.

## 0. Landscape shock: the space moved while we were building

Two results define the current frontier and must be named before the six
directions, because they overlap Lacuna itself:

- **SSD-LLaMA** (arXiv 2609.18110, 2026) — "SSD-Native Inference for
  Trillion-Parameter MoE at 1+ Token/s on a Consumer PC." Its *expert-pack*
  stores per-layer expert weights as contiguously indexed blocks with fixed
  offsets (O(1) byte-offset reads), plus an async I/O pipeline; it reports up
  to 77.7% of peak SSD sequential bandwidth (9 GiB/s testbed) and up to 1.8×
  less per-token I/O traffic than llama.cpp. **SGLang adopted the idea** as
  "SSD Expert Pack" (LMSYS blog, 2026-08-29) and runs DeepSeek-V4-Flash and
  Kimi-K3 with it: O_DIRECT reads into pinned buffers, bounded GPU expert
  cache, full expert pool on SSD. This is the closest *system* competitor to
  Lacuna. Its layout is static and per-layer; its cache is runtime-only
  (dies with the process); it has no cross-session memory and no exactness
  contract of the Lacuna kind.
- **Edge0** (open source, Oct 2026) — SSD-streamed MoE on Apple Silicon with
  a trained **Prerouter** that predicts routing a step ahead; 35B model at
  ~2.9 GiB peak active memory, up to 59% decode speedup claimed vs native
  routing. Proof that "predict routing → stream from SSD" is now a shipped
  pattern, not a research curiosity.

Implication: a companion technology that is *only* "predict experts better"
or *only* "pack experts contiguously" is already published/shipped. The
ownable ground is narrower and more specific — see the verdicts.

---

## 1. Draft/twin-model router-schedule prediction

**Idea.** A small local model (draft/twin) predicts the big MoE's expert
routing schedule several tokens ahead — the *schedule*, not the tokens — so
the pager can stream pages before the router ever runs.

**Prior art (dense, and accelerating).**

- **MoE-SpeQ** (arXiv 2511.14102, Nov 2025) is the idea, nearly verbatim: a
  small on-device draft model predicts "the sequence of required experts for
  future tokens"; an orchestrator prefetches from host memory; an adaptive
  governor tunes speculation depth. Up to 2.34× over SOTA offloading on
  Phi-MoE. Target tier: host RAM → GPU over PCIe, not disk.
- **Edge0 Prerouter** (Oct 2026): trained head forecasting next-step routing
  for SSD streaming (above).
- **MiniCPM-V 4.7 Lookahead Expert Scheduling** (ModelBest/OpenBMB,
  2026-10-15): predicts next *layer's* experts during current-layer compute,
  ">80% of required experts prepared in advance, without additional
  training," original router still decides → outputs identical. That last
  property is Lacuna's exactness contract, already shipped by someone else
  for the layer-ahead case.
- **Pre-gated MoE** (arXiv 2308.12066, ISCA 2024): a pre-gate function
  computes layer N+1's expert selection *during* layer N, decoupling
  selection from execution; +23% overhead vs oracle GPU-only.
- **HOBBIT** (arXiv 2411.01433, Nov 2024): layer-level "adaptive expert
  predictor" for next layers + mixed-precision replacement of missed experts
  (the replacement part violates Lacuna's exactness rule; the prediction
  part does not).
- **PROBE** (arXiv 2602.00509, Feb 2026): Gate-Initialized Lookahead
  Predictor that *distills the target router* to forecast next-layer
  activation in distributed serving.
- **DoMoE** (IJCAI 2026): domain-aware semantic expert prediction; builds
  routing tables from profiled token embeddings; 1.9–2.1× over MoE-Infinity
  on Mixtral-8×7B.
- **ST-MoE** (arXiv 2606.15453, 2026), **DuoServe-MoE** (arXiv 2509.07379),
  **MoE-MPMC** (arXiv 2605.11537, SRU sequence predictor over routing
  history), **CoMoE** (arXiv 2508.09208) — all predict experts from spatial
  (cross-layer) and/or temporal (cross-token) correlations.

**What's claimed.** Next-layer prediction accuracy 85–97.6% (ST-MoE 85%;
Pre-Attention, §6, up to 97.62%); end-to-end gains 1.26–2.5× in serving/offload
settings. Almost all evaluation is host-RAM → GPU or datacenter; none targets
an NVMe → 4.5 GB RAM tier.

**Open gap.** Thin. What remains unclaimed: (a) emitting the prediction as a
**K-token union page contract** into a paged store with blocking-fault
exactness (prior art either hides latency or *substitutes* approximately);
(b) distilling router *traces* into a schedule model sized for a 6 GB-RAM
floor machine; (c) the disk tier at all. These are integration gaps, not a
new mechanism.

**Verdict: CROWDED.** Build on it (Lacuna's oracle already lives here), do
not try to own it.

## 2. Co-activation graphs

**Idea.** Experts that fire together — within a layer (top-k partners) and
across layers/tokens — form a graph; use it for prefetch and cache placement.

**Prior art.**

- **MoE-Infinity** (arXiv 2401.14361, Jan 2024; ASPLOS 2025): request-level
  activation tracing; its Expert Activation Matrix explicitly captures
  *group activation* — e.g. E[0,81] co-firing with E[1,116], E[1,72] — and
  uses it for prefetching/caching. Reports fetch-traffic reductions up to
  98.5% (low-activation models, decode) from fetching only active experts.
- **hypura** (open-source NVMe MoE engine): a `CoActivationMatrix` maintains
  same-layer and cross-layer co-firing counts driving speculative prefetch
  from an NVMe backend; its docs claim ~96.5% expected cache hit rate on
  Mixtral-class locality (project's own figure, unverified). This is
  uncomfortably close to Lacuna's niche: NVMe + co-activation + prefetch,
  already in the wild.
- **ST-MoE** (2026): cross-layer correlation tables (CCT) + history tables
  (HT) in its prediction unit — co-activation as lookup tables.
- **SlimCaching** (arXiv 2507.06567, Jul 2025): expert *placement* across
  edge servers where the optimization explicitly models that co-selected
  experts are coupled (non-submodular when K > 1).
- Lacuna's own `CacheExpertPredictor` (bigram Markov over expert sets +
  saturating counters) is a first-order co-activation model already.

**What's claimed.** Co-activation is real, strong, and exploitable; everyone
in the offloading literature now uses some form of it.

**Open gap.** As a *signal*, none — it is the field's common fuel. As a
*standalone companion technology*, nothing to own: a graph that only advises
an LRU cache is MoE-Infinity (2024). The graph becomes ownable only when it
is used to change something nobody else changes — the physical store layout
(§3) and cross-session state (§5).

**Verdict: CROWDED as a mechanism; essential as an input.** Treat the
co-activation graph as fuel, not as the product.

## 3. Self-organizing weight-store layout

**Idea.** Physically reorder the pages of the weight store by predicted
traversal — pages the model touches consecutively sit consecutively on disk —
so a token's page faults collapse into a few long sequential reads instead of
hundreds of scattered 2 MB reads. Database physical-design (clustering,
learned layouts) applied to a model file.

**Prior art.**

- **LLM in a Flash** (Apple, arXiv 2312.11514, Dec 2023): *row-column
  bundling* — a neuron's up-projection row and down-projection column are
  stored adjacently so one flash read fetches both. This is layout
  optimization, but *within one layer, for dense FFN neurons*, and static
  (derived from the architecture, not from observed traversal).
- **SSD-LLaMA expert-pack** (2026) / SGLang SSD Expert Pack: experts packed
  as contiguous per-layer blocks with fixed offsets. Layout for O(1)
  addressing and clean async I/O — again static, per-layer, architecture-
  derived, not trace-derived, and not ordered by traversal across layers.
- **flash-moe** (danveloper, Mar 2026): applies LLM-in-a-Flash bundling to
  run Qwen3.5-397B-A17B at 5.5+ tok/s on a 48 GB MacBook — bundling again,
  within-layer.
- **hypura** ships an explicit `expert_layout` module in its NVMe backend
  (per its indexed source tree) — evidence that layout is an active design
  axis in open-source engines too; details not verified (docs fetch failed).
- Databases: **Flood** (SIGMOD 2020, arXiv 1912.01668) co-optimizes physical
  layout + index for a data/query distribution — the canonical "learned
  layout" result. Classic clustered indexes / usage-map-driven placement say
  the same: records used together, stored together. **No database work
  applies this to neural weights, and no LLM-serving work applies the
  trace-driven version to weights.**

**What's claimed.** Within-layer bundling/packing: real, worth up to ~78% of
peak sequential bandwidth in SSD-LLaMA's setting. Nothing published claims
*cross-layer, traversal-ordered* layout, *rebuilt from live routing traces*
(self-organizing), or *path packing* — physically packing the experts of
consecutive layers that co-fire into shared pages so one fault imports the
token's whole predicted path segment.

**Open gap — and its honest bound.** The gap is real: every existing layout
is static and architecture-derived; nobody closes the loop trace → relayout.
But the prize is bounded, and the report must say so: modern NVMe random
reads are far better than HDD-era intuition, and SSD-LLaMA already reaches
77.7% of *its* testbed's sequential peak with mere per-layer packing. On the
floor's 3.5 GB/s drive the remaining layout wins are (EST) 1.2–2× effective
bandwidth from (i) fewer, larger I/Os (less per-request overhead, deeper
readahead), (ii) path packing turning *future* faults into bytes already in
flight, and (iii) eviction granularity matching traversal. Worth having —
not magic. Relayout also costs SSD writes (endurance/TBW is a real budget;
SSD-LLaMA itself tracks prompts-per-TBW), so relayout must be idle-time,
incremental, and drift-gated.

**Verdict: OWNABLE** — specifically the trace-driven, self-organizing,
cross-layer version. The static versions are taken; the learning loop is not.

## 4. Lossless compression for streaming

**Idea.** Shrink bytes on disk losslessly (dictionary/delta across pages,
entropy coding of quantized values) so the same bandwidth moves more model.

**Prior art and realistic ratios.**

- **ZipNN** (IBM, arXiv 2411.05239, Nov 2024): byte-grouping + Huffman/ZSTD
  over float fields; ~33% saved on BF16, >50% on "clean" models; decode up
  to 80 GB/s multi-threaded — on server CPUs, for distribution/storage.
- **Heilper & Singer** (Intel, arXiv 2508.19263, Aug 2025) extend ZipNN-style
  coding to FP8/FP4: compressed size as low as ~83% of original for FP8
  (≈17% saved); BF16 to ~62–67%.
- **ECF8** (ICLR 2026): exponent-concentration theory + lossless FP8 coding,
  up to **26.9% memory saved** on models up to 671B, bit-exact outputs.
- **Entropy coding of quantized weights** (arXiv 2403.01384, Mar 2024):
  INT8 + entropy coding gained an extra ~2× on OPT-era weights because INT8
  symbol distributions are concentrated. That concentration largely
  *disappears at INT4*: GPTQ/AWQ-style 4-bit codes are near-uniform over 16
  symbols, so the entropy bound leaves only ~(EST) 2–10% on the table, and no
  published streaming system demonstrates more.
- **DFloat11 / Unweight** (Cloudflare tech report, Apr 2026): lossless
  exponent remapping/bitmap schemes (~30% on BF16) aimed at HBM bandwidth
  with decompression fused into GEMM — a GPU-resident technique, not a disk
  streaming one. **ZipLLM** (arXiv 2505.06252) deduplicates across *model
  versions* on a hub; within one model, cross-expert redundancy is
  negligible — experts are near-independent by construction, so
  dictionary/delta across pages has nothing to bite on (no source found
  claiming otherwise).

**The floor problem.** Even where ratios exist (FP8 stores: save ~17–27%),
the decoder runs on the critical path of a weak floor CPU that is already
feeding a GPU; Huffman/ZSTD decode of a 2 MB page is affordable, but the
*win* is a 1.1–1.35× bandwidth multiplier at best — it does not change the
regime, and for INT4 stores (the floor's native case, e.g. Kimi K2.7's
MXFP4/INT4) it shrinks to single digits.

**Verdict: DEAD as the companion technology.** Keep as an optional store
toggle for FP8 models later; it cannot carry the floor target.

## 5. Prefix/session expert-trace caching

**Idea.** Agent workloads reuse huge fixed prefixes (system prompt, tool
definitions, scaffold). Cache not the KV — the **routing trace**: which
experts each layer touched for that prefix and for the session's decode so
far — persist it across sessions, keyed by prefix hash, and replay it to
pre-warm the pager's hot set before the first token and to seed eviction.

**Prior art.**

- **KV/prefix caching is crowded**: PromptCache (MLSys 2024), LMCache
  (KV offload/reuse layer, 2025), SGLang RadixAttention, vLLM automatic
  prefix caching, and every hosted API's prompt caching (2024–). All of it
  caches *activations* to skip *compute*.
- The reuse premise is measured, not hypothetical: an agentic-workload
  characterization (arXiv 2605.26297, 2026) finds most input tokens reused
  across turns, empirical cache-hit ratios **84.6–99.5%**, decode =
  **91.0–98.6%** of LLM time. ProjectDiscovery's production agent (Neo,
  20k-token system prompt) went from 7% → 84% cache hit rate by fixing
  prefix stability (Apr 2026 writeup) — agent scaffolds are *designed* to be
  byte-stable across thousands of calls.
- **MoE-Infinity** traces activations at request level — but in-memory, for
  the live request only; nothing persists across processes or sessions, and
  nothing is keyed by prefix. **DoMoE** builds routing tables from an offline
  profiling corpus per *domain*, not per-prefix online. **Lacuna's own
  warm-start cache** (Round 2) persists the hot set across runs of the same
  model — but it is a flat snapshot, not prefix-keyed, and carries no
  per-layer working-set structure or decode-phase profile.

**What's claimed.** Nobody — in any source found — persists a *routing
trace* keyed by prompt prefix and replays it into a disk-streaming engine's
pager. The KV world proved the workload property (prefixes repeat; decode
dominates) and then cached the wrong artifact for Lacuna's problem: on the
floor, KV for a 32B-active model is small next to the 16 GB/token weight
stream; the scarce resource is the 4.5 GB hot set, and what it should
contain at token 0 is exactly what a trace manifest knows.

**Open gap.** Wide and specific: (a) prefix-hash → per-layer expert working
set manifest, persisted on disk, replayed as a *preload schedule* (bulk
sequential reads before prefill, not faults during decode); (b) session
decode profiles (the working set *drifts* predictably within a session —
manifest + online adaptation); (c) manifests as eviction prior: pages in the
active manifest are pinned against eviction by stray faults. All three are
memory-side only — a stale or wrong manifest costs speed, never output
(`ensure_resident` still blocking-faults the true set).

**Verdict: OWNABLE** — the strongest single gap found, and it is aimed
exactly at the floor's binding constraint: with 4.5 GB, *which* pages are
resident matters more than any predictor's marginal accuracy.

## 6. Residual-stream forecasting for router inputs

**Idea.** Forecast the hidden state h several layers ahead, run the *real*
routers on the forecast, and prefetch from true-router outputs early.

**Prior art — the field routes around this idea.**

- **Deja Vu** (ICML 2023, arXiv 2310.17157): per-layer MLP predictors map a
  layer's *actual* input to that/next layer's sparsity pattern. Lacuna's
  oracle is this lineage. No forecasting of h itself.
- **Pre-gated MoE** (2024): computes an *approximate early gate* from the
  previous block instead of forecasting h.
- **Pre-Attention Expert Prediction** (ETH, Nov 2025): two *linear* probes on
  pre-attention activations predict the same layer's experts at 93.03%
  (DeepSeek-V2-Lite), 94.69% (Qwen3-30B), 97.62% (Phi-mini-MoE) — the
  insight being that ranking is preserved through cheap functions of
  *existing* activations. Again: surrogate router on real activations, not a
  forecast of future activations.
- **PROBE** (2026): distills the router itself for next-layer forecasts.
- **SiDA-MoE** (MLSys 2024, arXiv 2310.18859): an LSTM "hash function"
  trained by truncated distillation *replaces* the router for offloading
  decisions, claiming up to 99% activation-prediction accuracy and <1%
  performance drop — note the drop: SiDA spends exactness, which Lacuna's
  rule forbids for math (though not for memory guidance).
- True future-hidden-state prediction exists in speculative decoding
  (EAGLE-style feature prediction) — for token drafting, at 1-step horizons,
  with verification. Nothing found runs multi-layer h-forecasting to drive
  routers for prefetch.

**Why it collapses.** For page-set purposes, a surrogate router evaluated on
*real* current activations (Pre-Attention/PROBE/DejaVu-style) is cheaper and
strictly more accurate than the true router evaluated on a *forecast* h —
forecast error compounds per layer, while the surrogate's error is bounded
and measured. And under Lacuna's rule the forecast's only consumer is the
pager, which cannot tell the difference. The direction is not wrong; it is
*dominated*.

**Verdict: DEAD as a standalone direction** — subsumed by §1/§6's surrogate
routers, which Lacuna's oracle already implements in DejaVu form.

---

## 7. Ranking

| Rank | Direction | Verdict | Floor leverage (EST) |
|---|---|---|---|
| 1 | §5 Prefix/session expert-trace caching | **OWNABLE** | High — decides what the 4.5 GB holds from token 0; zero-risk to exactness |
| 2 | §3 Self-organizing traversal layout | **OWNABLE** | Medium — 1.2–2× effective bandwidth + path-packed co-fetch (EST, bounded by NVMe physics) |
| 3 | §2 Co-activation graphs | CROWDED (fuel) | Medium — as the signal feeding ranks 1–2 |
| 4 | §1 Twin-model schedule prediction | CROWDED | Medium — real but published repeatedly, incl. shipped products |
| 5 | §4 Lossless streaming compression | DEAD (headline) | Low — 1.1–1.35× on FP8, single digits on INT4, CPU cost on critical path |
| 6 | §6 Residual-stream forecasting | DEAD (subsumed) | Low — dominated by surrogate routers on real activations |

## 8. Recommendation: **Scout — the trace-compiled weight store**

The strongest companion is the fusion of ranks 1+2, driven by rank 3's
signal: a persistent **routing-trace memory** that compiles what the model
*did* into how the store is *arranged* and *pre-loaded*. It runs ahead of the
model twice — ahead in **time** (yesterday's traces warm today's session)
and ahead in **space** (the disk itself is ordered the way the model thinks).

**Mechanism sketch.** Every Lacuna run already observes the ground truth of
routing: the pages each decode step actually required (the ShadowScorer's
"actual" set). Scout records these traces, keyed by a hash of the session's
prompt prefix, into a small persistent trace store beside the `.lac` file.
From accumulated traces it compiles two artifacts. (1) A **session manifest**:
for this prefix (and, as decode proceeds, for this session's own early
tokens), the per-layer expert working set, ranked — replayed at session start
as a bulk sequential preload that fills the 4.5 GB hot set with the pages
history says will matter, before the first token faults anything. (2) A
**compiled layout**: an idle-time relayout pass that rewrites the store's
page order along the co-activation graph — experts that fire consecutively
across layers are packed into shared "path pages" and laid out in traversal
order — so the misses that remain are served by a handful of long sequential
reads that simultaneously import the next layers' likely experts. Online,
the existing oracle/CacheExpertPredictor keeps adapting within the session;
the manifest is its prior, the layout is its accelerator. Every Scout
decision touches memory placement only: a wrong manifest or a stale layout
costs faults and speed, and `ensure_resident` still guarantees the true page
set before compute — output remains bit-identical, by the same construction
Lacuna already proves.

**How it appends to existing modules.**

- `lacuna/prefetch.py` — `ShadowScorer` is already the trace tap (it sees
  predicted vs actual per step); Scout adds a `TraceStore` beside it
  (append-only trace log + prefix hashing, reusing the FNV-1a machinery of
  `NgramPrefetcher`) and a `ManifestBuilder` that turns traces into ranked
  per-layer working sets.
- `lacuna/oracle.py` — the Oracle's union prediction gains the manifest as a
  prior input: predicted set = manifest set ∪ oracle union, with the
  manifest dominating early tokens and decaying as live evidence accrues.
- `lacuna/pager.py` — `.lac` v3 gains an optional *compiled-layout* section:
  a page permutation table + path-packed pages, written copy-on-write by the
  relayout pass with per-page checksums (zero-silent-error standard);
  readers fall back to the canonical layout if the compiled section is
  absent or fails validation. The pager also gains a `preload(manifest)`
  bulk-read path (large sequential extents, io_uring where available).
- `lacuna/scheduler.py` — session start becomes a scheduled bulk preload
  (manifest order = disk order, by construction of the compiled layout);
  decode keeps the existing union-prefetch loop, now faulting mostly on
  sequential extents.
- Warm-start cache (Round 2) is subsumed and upgraded: instead of a flat
  hot-set snapshot, what persists across runs is the *structured* trace +
  manifest, which survives model-store relayout and transfers across
  sessions with the same scaffold.

**Floor-specific design rules.** Manifest preload is capped by the 4.5 GB
budget with ranked truncation (never evict KV/attention-critical pages);
relayout runs only on idle, in bounded increments, gated by measured trace
drift (SSD endurance budget); trace store is small (working sets, not raw
streams — KBs per session, EST).

**What would prove it (measurement plan).**

1. **Traces.** Capture real routing traces from an open MoE at the
   llama.cpp attach point documented in `integrations/llama.cpp` (or a
   HF-side hook on a small MoE, e.g. Qwen3-30B-A3B class, scaled analysis to
   the 32B-active class), on agent-like workloads: fixed scaffold +
   multi-turn tasks, ≥20 sessions sharing a prefix.
2. **Primary metric.** Steady-state **disk bytes/token** (ShadowScorer
   extended to byte accounting) at a hard 4.5 GB hot-set cap, tokens
   100–1000 of each session. Baselines on identical traces: (a) LRU hot set,
   (b) MoE-Infinity-style activation-matrix caching, (c) Lacuna today
   (oracle union + LRU). **Target: Scout ≤ 3.5 GB/token (≥1 tok/s at floor
   NVMe) where the best baseline is expected (EST) at 6–10 GB/token.**
3. **Secondary metrics.** Time-to-first-token cold vs manifest-warm;
   mean/median sequential run length (bytes per I/O) before vs after
   relayout; ShadowScorer recall/precision of manifest-only prediction at
   token 0 vs oracle-only; relayout write volume per session-hour (TBW
   budget check).
4. **Falsifiers (pre-registered).** If real 384-expert traces show the
   session working set is irreducibly ≫ 4.5 GB with flat popularity (no
   heavy-hitter concentration), no memory-side policy reaches 3.5 GB/token
   and Scout's honest ceiling is the baseline-beating number, reported as
   measured. If relayout's run-length gain is < 1.15× on the floor drive,
   drop the compiled layout and ship manifests only.

## 9. Honest caveats

- All third-party numbers are as-reported; several key systems (Edge0,
  hypura, MiniCPM LES) are vendor/project claims without independent
  replication as of this sweep.
- The entire recommendation rests on one empirical property — expert
  popularity concentration + prefix-stable working sets at 384-expert scale
  — which is documented at 8–64-expert scale (MoE-Infinity, ST-MoE) and
  *assumed*, not proven, at Kimi-K2.7 scale. The measurement plan exists
  precisely because this could fail.
- "1+ tok/s for trillion-parameter MoE on a consumer PC" is already claimed
  by SSD-LLaMA on a 9 GiB/s testbed; the floor's 3.5 GB/s drive and 6 GB RAM
  are a harsher regime than any published result found. Scout is a bet that
  trace memory + compiled layout closes that gap; it is a bet, labeled as
  one.

## Sources

MoE-SpeQ — arxiv.org/pdf/2511.14102 · Pre-gated MoE —
ar5iv.labs.arxiv.org/html/2308.12066 · HOBBIT — arxiv.org/html/2411.01433v1 ·
PROBE — scirate.com/arxiv/2602.00509 · DoMoE —
ijcai.org/proceedings/2026/0657.pdf · ST-MoE — arxiv.org/html/2606.15453 ·
DuoServe-MoE — arxiv.org/pdf/2509.07379v2 · MoE-MPMC —
arxiv.org/html/2605.11537 · CoMoE — ar5iv.labs.arxiv.org/html/2508.09208 ·
MoE-CORE — arxiv.org/abs/2610.01950v1 · MoE-Infinity —
arxiv.org/html/2401.14361v2 · hypura co-activation docs —
deepwiki.com/t8/hypura/4.2-neuron-cache-and-co-activation-prediction
(indexed; page fetch unavailable during sweep) · SlimCaching —
arxiv.org/html/2507.06567v2 · LLM in a Flash — arxiv.org/html/2312.11514v2 ·
SSD-LLaMA — arxiv.org/html/2609.18110 · SGLang SSD Expert Pack —
lmsys.org/blog/2026-08-29-sglang-ssd-expert-pack/ · Edge0 —
themenonlab.blog/blog/edge0-streaming-moe-ssd-expert-offload/ · MiniCPM-V 4.7
LES — manilatimes.net, 2026-10-15 (PR Newswire) · Flood —
arxiv.org/pdf/1912.01668 · ZipNN — arxiv.org/html/2411.05239v2 ·
Heilper & Singer — arxiv.org/pdf/2508.19263 · ECF8 —
liner.com/review/to-compress-or-not-pushing-the-frontier-of-lossless-genai ·
Entropy coding of quantized weights — arxiv.org/pdf/2403.01384 · ZipLLM —
arxiv.org/html/2505.06252v2 · Unweight — Cloudflare Cf-TR-2026.04 ·
lmzip — pypi.org/project/lmzip/1.3.0/ · Agentic workload characterization —
arxiv.org/html/2605.26297v1 · ProjectDiscovery Neo caching writeup —
getmegabrain.com/blog/prompt-cache-hit-rate-myth-2026 · Deja Vu —
arxiv.org/pdf/2310.17157 · Pre-Attention Expert Prediction —
research-collection.ethz.ch (Nov 2025) · SiDA-MoE —
arxiv.org/html/2310.18859v2, proceedings.mlsys.org (MLSys 2024) ·
SparseInfer (survey context) — arxiv.org/pdf/2411.12692.
