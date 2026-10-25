# MoE executor — DESIGN ONLY, UNTESTED

Status: design document, 2026-10-25. No code exists for this design. No
number in this document is a measurement — every figure cited from
elsewhere is labeled with its source, and every claim this design makes
is a target until the proof plan in §10 measures it.

Why this design exists: Lacuna's dense executor runs Qwen2 paged from
disk with exact output, proven at 110B scale
(`PAGED_GENERATION_PROOF.md` Appendix B: 8 real tokens, budgets held,
no OOM — but ~35 min/token on free CPU, unusable as-is). The
beat-target runs MoE models up to 2.8T by paging routed experts, and
Lacuna's trillion-tier target list (`docs/TARGET_MODELS.md`) is
dominated by MoE models. A dense-only executor can never run them:
dense 110B moves 41 GB per token; a MoE model moves only its active
set (Kimi K3: 104B active of 2.8T total; Qwen3.8-Flash-Next: 6B active
of 180B). The speed story for Lacuna lives on MoE targets plus the GPU
path (`docs/SPECULATIVE_PAGED_DESIGN.md`). This design extends the
dense executor to MoE without changing its honesty rules.

## 0. Invariants carried over from the dense executor (non-negotiable)

- **Pager-only weight access.** Every weight byte arrives through
  `PagedWeightStore` (`read_tensor` / `read_tensor_rows` in
  `lacuna/gguf.py`). No direct file reads of weights anywhere. The
  store's `stats` (hits / faults / bytes_read) is the audit trail.
- **Exact policy only.** The `.lac` must carry source bytes
  bit-for-bit, so dequantization is the model's own quantization, not
  a Lacuna re-quantization. The read path enforces this and fails
  loud on a non-exact store.
- **Greedy decoding only** (argmax). Deterministic, directly
  comparable to `llama-cli --temp 0 --top-k 1`.
- **Hard budgets, reported separately.** Page-cache bytes and
  dequantized-cache bytes each stay under their budget and are
  reported separately, so memory accounting stays inspectable.
- **Fail loud, never silent.** An unsupported architecture, a missing
  tensor, a short tensor, or an unknown router rule raises. A
  downgrade (e.g. to the `[simulated]` path) is labeled with its
  reason, never silent.
- **Prediction costs time, never correctness.** Any prefetcher or
  predictor may only move bytes earlier. The demand fault is always
  the loud path for a misprediction. This is the same contract the
  dense executor's next-layer prefetch already keeps.

## 1. Model view: what an MoE layer is

A dense transformer layer is: attention + norms + one SwiGLU MLP.
An MoE layer replaces the MLP with three pieces:

1. **Dense slice** — identical machinery to the dense executor:
   attention (GQA + RoPE + KV cache), attention norms, FFN norms.
   Unchanged code paths.
2. **Router (gate)** — one small dense matrix, `hidden → n_experts`
   (plus an optional bias, family-dependent). For 128 experts at
   hidden 8192 this is ~1M parameters — small enough to hold in the
   dequant cache permanently. It is a dense tensor and is treated as
   part of the dense slice for residency.
3. **Routed experts** — `n_experts` SwiGLU MLPs (gate, up, down
   matrices each). Only the top-k experts selected by the router run
   for a given token. In GGUF these are stored as **stacked 3-D
   tensors** (one tensor per matrix role, e.g. gate weights for all
   experts in one tensor).
4. **Shared experts (optional, family-dependent)** — one or more
   SwiGLU MLPs computed for *every* token, added to the routed sum.
   Present in DeepSeek-style models, absent in Qwen/OLMoE-style
   models. Treated as dense tensors (same residency as the dense
   slice) when present.

Tensor inventory is read from the file's own metadata and tensor
index at load time, exactly as `PagedQwen2` does for Qwen2 (it trusts
the file, not assumptions, and raises on missing tensors). The
executor must read, per layer: attention tensors, norms, router
weight (+ bias if the file carries one), stacked expert tensors,
shared-expert tensors if present, and the architecture's routing
parameters (expert count, top-k, shared count) from metadata keys.
Key names vary by architecture (`qwen3moe`, `deepseek2`, `olmoe`,
…); the implementation reads them from the file and fails loud on
an architecture it does not know — it never guesses a routing rule.

## 2. Expert paging via PagedWeightStore

This is the core mechanism, and it reuses the dense executor's
verified code paths rather than inventing an expert tier.

**Layout fact the design rests on.** GGUF stores tensor dims
fastest-first. A stacked expert tensor's dims are
`[row_length, ffn_width, n_experts]`, so the expert index is the
*slowest* dimension — which means expert `e`'s whole matrix is one
contiguous byte range of the tensor: bytes
`[e · ffn · row_length · bytes_per_value, (e+1) · …)`.
Flattened, the stacked tensor is a 2-D tensor of shape
`(n_experts · ffn, row_length)`, and expert `e` is exactly rows
`[e · ffn, (e+1) · ffn)`. **No converter change is needed:**
conversion already copies GGUF bytes verbatim (exact policy, RAW
pages), preserving byte order, and `read_tensor_rows` already
faults only the pages overlapping a requested byte range (it walks
pages by header shape and slices exact byte ranges, ignoring
row bookkeeping). An expert read is therefore a `read_tensor_rows`
call on the flattened row range — the same function the sliced
embed/head path already uses, with the same block-alignment
constraint (row length must be a multiple of the quantization
block size, enforced by that function today).

**Page-boundary sharing.** The store pages at 2 MiB. An expert
spans many pages, and a page may straddle two adjacent experts'
byte ranges. That is harmless: the reader slices exact byte ranges
out of faulted pages, so a shared boundary page just caches a few
bytes of a neighboring expert. Correctness never depends on page
alignment; only the fault count does, and the fault count is
reported, not assumed.

**Per-layer expert page sets.** The dense executor's
`_layer_page_ids(li)` enumerates a layer's dense tensors' pages
for next-layer prefetch. The MoE executor adds
`_expert_page_ids(li, expert_ids)`: the union of page ids
overlapping the requested experts' flattened row ranges, computed
from `store.tensor_pages` + `store.page_shape` without faulting
payloads. This is the only new pager-facing primitive, and it is
pure page-table arithmetic.

**What is deliberately NOT built:** a separate expert cache, a
separate expert tier, or a pinned expert set. There is one store,
one LRU, one budget, one audit trail. Experts compete for residency
with dense tensors under the same rules (see §4).

## 3. Router execution

Order of operations per layer, per forward call:

1. Attention + norms run exactly as in the dense executor. The
   router input is the post-attention-norm hidden state — the same
   hidden state the dense FFN would consume.
2. **Router first, experts second.** The router weight is dense and
   resident, so router logits cost no faults:
   `logits = h @ W_router.T (+ bias if present)`, float32.
3. **Selection.** Top-k experts by score. The scoring and
   normalization rule is **per-architecture, declared in code, and
   verified against the reference at proof time** — never silently
   assumed, because families differ:
   - Softmax over router logits → top-k → renormalize weights over
     the selected k (Mixtral / Qwen-MoE / OLMoE family).
   - Sigmoid scoring, top-k, no renormalization (DeepSeek family).
     DeepSeek additionally adds a per-expert *selection bias* to
     scores for top-k selection only (auxiliary-loss-free load
     balancing) — the bias must NOT enter the output weights.
     This is a rung-3 requirement (§11), not v1.
   - Any other family rule: fail loud until implemented + verified.
4. **Tie-breaking.** Top-k ties are broken deterministically:
   higher score first, lower expert index first. Whether this
   matches the reference implementation's tie behavior is a
   **verification item at proof time**, not an assumption: the
   synthetic-GGUF router tests (§10) pin the rule, and any tie
   divergence against the dense reference is reported in the
   near-tie class (§8), same as the dense executor's step-19 /
   token-13 near-ties (margins 0.0022 / 0.002362).
5. Experts run (grouped, §5), outputs are weighted by the router
   weights and summed, shared experts (if any) are added as a dense
   term, and the residual + norm proceed as in the dense executor.

**Serialization note, stated honestly.** Exact demand-driven MoE
serializes each layer: attention + router, then expert fault, then
expert compute. The router cannot know the next layer's experts
before the current layer's output exists. Prefetch (§7) hides the
fault latency; it does not remove the dependency. Any design that
claims otherwise must show its prediction measured, not asserted.

## 4. Expert buffer / cache policy

One unified policy, reusing the store and the dequant cache:

- **Page cache (`PagedWeightStore`, LRU, hard byte budget).**
  Expert pages, dense pages, and router pages all live here. The
  store's EWMA `_freq` counters already measure per-page reuse, so
  recurring experts are naturally protected by LRU recency +
  frequency — no separate popularity file, no pinned set. Warm
  start (`save_hotset` / `restore_hotset`, CRC-validated, stale
  pages degrade to a cold start) may pre-load a previous session's
  hot set, but those pages still evict under budget pressure: a
  warm start is a cache, not a pin.
- **Dequant cache (`_dq`, bounded LRU, reported separately).**
  Expert entries are cached per expert, per matrix role, in the
  same fused form as the dense executor: one fused gate+up entry
  per expert (the `_fused_gateup` pattern, applied per expert
  instead of per layer) plus a down entry. An expert's fused entry
  is evictable like any other entry; a budget too small for one
  expert's matrices simply re-dequantizes that expert every use
  (thrash, reported — the same honesty as the dense executor's
  sub-model-size budgets).
- **Dense slice residency.** Attention, norms, router, and shared
  experts are faulted first in each layer's schedule (they are
  needed by every token, so they are the highest-value residents).
  This is a scheduling order, not a separate budget: the dense
  slice still evicts under pressure, and if it does, it re-faults
  on demand like everything else.
- **Budget split (recommendation).** Keep the dense executor's two
  budgets — one page budget, one dequant budget, each hard — and do
  not carve a third expert budget. A third budget would split the
  accounting, duplicate the store's job, and hide the real
  tradeoff (dense slice vs experts) behind a configuration knob.
  The proof plan measures whether one budget pair serves both; if
  a split ever wins on measured traces, that measurement — not this
  document — justifies it.

**What this is not (explicitly, so it is not built by accident).**
The beat-target's design is per-layer LRU + a learned pinned hot
set file + a router-lookahead thread. This design uses none of
those three: no per-layer cache (one store LRU), no learned pinned
set (EWMA frequency inside the existing store, still evictable),
no router-lookahead thread (prediction comes from Lacuna's own
primitives in §7, and v1 ships with no prediction at all beyond
the speculative block's exact demand).

## 5. Grouped / batched expert GEMM strategy

The dense executor's FFN is two GEMMs (fused gate+up, down). The
MoE executor generalizes this by **grouping positions by expert**:

- **Single-token decode (the common case).** The token selects k
  experts. Each selected expert runs one fused gate+up GEMM over
  the single hidden vector, then SiLU × up, then one down GEMM.
  Unselected experts cost zero faults and zero FLOPs — they are
  never touched. Per-layer expert cost is k expert-GEMMs, not
  n_experts.
- **Verification blocks (speculative decoding).** A block of K
  draft positions is verified in one batched pass
  (`forward_all`, same structure as `speculative.py`'s
  `BatchedQwen2`). Positions are grouped by expert *within each
  layer*: for each expert with at least one position routed to it,
  one fused gate+up GEMM runs over that expert's position group,
  then one down GEMM, then a weighted scatter-add into the output
  rows. The expert's pages are faulted once per block and shared
  across all positions in its group — this is where the block
  amortizes expert-fetch cost, and it is why the speculative
  design doc says speculation wins big on MoE: the per-token
  working set is small, and the block divides the fetch cost
  across positions.
- **Shared experts** run as one dense GEMM over all positions
  (they are dense tensors; no grouping needed).
- **Numerics.** All math is float32 NumPy, same as the dense
  executor. Grouped GEMMs change only the grouping of rows, not
  the arithmetic per row: a position's output must equal the
  output it would get from a per-position expert GEMM. The
  synthetic tests (§10) pin this: grouped vs ungrouped outputs
  compared exactly.
- **Head note (from the speculative measurements).** In
  verification blocks, per-position head reads thrash at small
  budgets (the 0.5B speculative run read ~2.2× bytes of plain for
  this reason). The MoE design carries the speculative doc's
  follow-up: within one `forward_all` call, hold head blocks in a
  transient per-call cache released after the call, so block
  positions share one head read. At MoE scale the head is a small
  fraction of the file (the 110B head is ~1–3%), so this matters
  less than at 0.5B scale — but it is still specified here so the
  implementation does not re-learn the penalty.

## 6. KV handling

**Unchanged from the dense executor.** Attention in MoE models is
dense attention: every layer attends over the full KV cache, and
expert routing touches only the FFN. So the KV cache is the same
per-layer dict of `(K, V)` arrays, the same causal masking, the
same pending-token truncation rule from speculative decoding (the
cache covers emitted-minus-pending; after each block it is
truncated to exactly that prefix).

Memory formula (from the file's own dims, never assumed):
`layers × kv_heads × head_dim × 2 (K+V) × tokens × 4 bytes`.
For scale: the speculative design doc's worked example for the
110B dense model is ~1.6 GB at 1024 tokens in 2-byte form; MoE
models have their own layer/head counts, and the proof plan
reports the formula's value for the proof model. Long-context
fallback is the same as the dense executor's GPU-path plan (KV on
GPU while it fits, CPU KV with per-layer transfer when it does
not). Paged/offloaded KV is out of v1 scope.

**Excluded from v1, with reason: MLA attention (DeepSeek).**
MLA compresses KV into a latent vector with a different attention
math (absorbed query/key projections, decoupled RoPE). That is a
new attention implementation, not an FFN change, and it belongs
in its own design after the MoE FFN is proven. Any DeepSeek-family
model is therefore rung 3 (§11), not v1, regardless of its expert
count.

## 7. Prefetch and prediction — Lacuna's own primitives only

v1 ships **demand-driven exact execution**: router runs, experts
fault on demand, compute never observes a missing page (the
`ensure_resident` fence is available for any caller that wants
the full page set faulted before compute). No predictor ships in
v1. Prediction is rung 2 and rung 3, each gated by measurement:

- **Rung 2: `CacheExpertPredictor` (`lacuna/prefetch.py`, already
  built).** Zero-training, per-expert 2-bit saturating counters
  (active +1, inactive −1) plus a bigram Markov table over expert
  activation sets with multiplicative decay (0.99), deterministic
  tie-break by expert id. At each decode step it predicts the next
  token's expert set; the executor prefetches those experts'
  pages (async `prefetch()`, fire-and-forget) while the current
  token computes. Mispredicted pages cost faults and possibly an
  eviction — never correctness, by the §0 contract. **Gating:**
  the predictor rides in *shadow mode first* (`ShadowScorer`:
  per-step predicted vs actual page sets, reporting hit rate,
  recall, precision, fault-step rate, overfetch ratio). It ships
  as a live prefetcher only if, on real traces from the proof
  model, it beats demand-only on bytes/token at the same budget.
  A predictor that over-fetches (overfetch ratio > 1 with no
  fault-step reduction) stays shadow-only.
- **Rung 3: Ordo routing automaton (`lacuna/ordo.py`,
  `docs/ORDO.md`).** The sequence of expert sets across tokens is
  a routing trace; Ordo clusters traces online into phase-states
  (signature = pages in ≥25% of member tokens, ≤48 states) and
  compiles each state's outgoing transitions into a segment. For
  MoE, a state transition predicts the next phase's expert set,
  and speculation fires **only at state transitions** — never per
  token (the per-token variant was measured 3–6× worse than LRU
  and killed). The structure gate closes speculation to plain LRU
  when traces show no sequential structure (the ungated variant
  measured 26% worse on memoryless routing). Ordo's simulated
  coverage is 2–3× popularity at equal budget with bytes at LRU
  parity — but those are SIMULATED numbers, and Ordo for expert
  sets is unmeasured until real MoE traces exist. Same gating as
  rung 2: shadow first, ship only on measured wins.
- **Speculative-block demand (already exact, not prediction).**
  Within a verification block, each layer's routers for all K
  positions are computed before that layer's expert compute, so
  the block's expert demand per layer is known exactly at layer
  time. The executor faults exactly that union — no prediction,
  no misprediction, and the union is shared across positions in
  each expert's group (§5). This is v1's main traffic win and it
  needs no predictor at all.

**Prefetch priority rule (carried from the speculative doc).**
Prefetch must stay below demand faults in priority: a mispredicted
prefetch may evict a useful page under LRU, so the prefetcher only
loads pages the layer schedule or a gated predictor names — never
open-ended speculative guesses.

## 8. Exactness rule — the contract

"Exact" for the MoE executor means all of the following, and the
proof plan (§10) checks each one:

1. **Same router decisions as the dense reference.** For the same
   prompt and context, the executor selects the same top-k experts
   per layer per token as the reference implementation (llama.cpp
   dense) would. Router decisions are computed from the same
   weights, dequantized bit-exact from the same GGUF bytes by the
   same dequantizers (`gguf.py`, verified bit-exact vs ggml's own
   functions on the dense side). A router divergence is a
   correctness failure, not a tuning difference.
2. **Same precision as the dense reference.** No silent
   re-quantization, no precision change on the weight path. The
   exact policy enforced by the read path is the mechanism; the
   `.lac` header records it. Models whose native quantization has
   no verified dequantizer in `gguf.py` (MXFP4 for Kimi K3,
   FP8-native DeepSeek/Qwen variants) are **out of the exact tier
   until their dequantizers are added and verified bit-exact vs
   ggml** — see §12. Until then, no exactness claim is made for
   them, and the executor fails loud rather than silently
   approximating.
3. **Token-exact until a near-tie, same as dense.** Bit-identical
   logits vs llama.cpp's kernels are NOT claimed (different
   summation orders and FMA use, same as dense). The claim is the
   dense executor's claim: greedy token ids match the dense
   baseline until a near-tie, and a split is reported with its
   margin (dense step-19 margin 0.0022, token-13 margin 0.002362)
   rather than smoothed over. A systematic math bug breaks the
   first token (as the dense RoPE bug did); a near-tie breaks one
   token at a small margin. The proof distinguishes them the same
   way.
4. **No silent changes, ever.** If the executor cannot meet 1–3
   for a model (unknown architecture, unknown router rule,
   unverified quantizer, MLA attention), it fails loud with the
   reason inspectable — it never silently substitutes an
   approximation and still calls the output exact.

## 9. What this design does not do (scope fence)

- No GPU execution in v1 (CPU executor first; the GPU executor's
  layer-streaming design reuses expert page sets per layer when
  that path is built — GPU is a separate design, not a paragraph
  here).
- No MLA attention, no DeepSeek sigmoid+bias routing, no
  MXFP4/FP8-native quantizers in v1 (§11 rungs).
- No new file format. The `.lac` layout, page size, and exact
  policy are unchanged; expert slicing is a reader view over
  existing pages.
- No claim about any trillion-tier model. The cold-bound
  arithmetic in `TARGET_MODELS.md` (K3: 52 GB/token cold,
  Qwen3.8-Max: 47.5, DeepSeek-V4-Pro: 24.5, K2.7 Code: 16, all at
  ~0.5 bytes/active-param) stands as the pessimistic bound until
  §10-style measurements exist for MoE. The floor claim ("all 10
  runnable on RTX 2050 + 6 GB RAM, exact, caps respected") is
  decided by ShadowScorer-style measurement of expert working-set
  reuse on real traces under the 4.5 GB hot-set budget — not by
  this document, and not by analogy to Colibri's numbers.

## 10. First proof plan — the smallest MoE that proves it

**Proof model (recommendation): a 7B-class MoE with GQA attention,
no MLA, no shared experts — OLMoE-1B-7B is the leading candidate.**
Why this shape: 7B total fits the same measurement regime as the
7B speculative proof (budgets in the hundreds of MB, Kaggle CPU
kernel viable); GQA reuses the dense executor's verified
attention; no MLA means no new attention math; no shared experts
means the routed-expert path is proven in isolation before the
shared-expert term is added (rung 2). The proof must record the
file's repo, filename, byte size, and SHA-256 — the 7B speculative
proof's missing file identity is a named lesson, not to be
repeated. If OLMoE-1B-7B's GGUF proves awkward (unknown arch keys,
unusual router rule), the fallback is Qwen1.5-MoE-A2.7B, and the
fallback must be recorded as a change of plan, not silently
substituted. All dims, expert counts, top-k, and tensor names are
read from the proof file's own metadata at proof time and reported
— this document does not pin them, because they are the file's
facts, not the design's.

**Stage 0 — synthetic-GGUF router unit tests (before any real
model).** Same pattern as `tests/test_paged_model.py`'s tiny
synthetic Qwen2: build tiny synthetic MoE GGUF files with
hand-computed router weights and expert weights, and pin:
router logits for a known hidden state; top-k selection
including an exact tie (verifies the §3 tie rule); grouped vs
ungrouped expert outputs identical; shared-expert term (synthetic
only — proves the code path exists, not a real model's rule);
expert page sets equal the demanded experts' pages and nothing
else (no unselected expert faulted — checked via `store.stats`).
Stage 0 must be green before any real-model run.

**Stage 1 — correctness at cache-fit budgets.** Run the proof
model with budgets large enough to hold its dequantized active
set, greedy, on the two dense-proof prompts ("The capital of
France is", "Name three primary colors", 32 tokens each) plus a
third prompt chosen for longer generation (expert-reuse
measurement needs more than 32 tokens; 128 tokens). Baseline:
llama.cpp `llama-cli --temp 0 --top-k 1` on the same file.
Report: token ids side by side, first-split position and margin
if any split (near-tie rule, §8.3), router decisions per layer
per token if the reference exposes them (if not, state that the
check is end-to-end token identity plus Stage-0 unit proof —
do not claim a router comparison that was not performed).

**Stage 2 — thrash-regime speed at floor-relevant budgets.**
Same prompts at 256 MB and 512 MB page + dequant budgets (the
7B speculative proof's regime). Report, per run: prefill seconds,
per-token seconds, overall tok/s, faults, bytes_read, hit rate,
peak RSS, cached bytes vs budget, dequant bytes vs budget,
distinct experts touched per token, expert reuse rate across
tokens, and a ShadowScorer report for the rung-2 predictor in
shadow mode (predicted vs actual expert page sets per step).
Budgets held is a pass/fail gate, not a footnote.

**Stage 3 — speculative on MoE.** The same 7B proof's
speculative protocol (small dense draft, K=4) applied to the MoE
target: acceptance rate, tokens/block, tok/s vs plain at the same
budgets, ids identical to plain (the speculative exactness rule:
accepted = target argmax, resample/bonus = target argmax, so the
sequence equals greedy by construction — verified by comparison,
not assumed).

**Kill gates (all must pass; any failure stops the claim).**
1. Token identity: 32/32 on both proof prompts, or a split only
   in the near-tie class (margin < 0.01, documented with both
   probabilities, same top-set) — a first-token or large-margin
   split is a bug, not a near-tie.
2. Budgets held on every run (page cache ≤ budget, dequant
   cache ≤ budget, peak RSS reported).
3. No weight byte bypasses the store (audited by `store.stats`
   + code inspection: no direct file reads of weights).
4. No unselected expert faulted in a demand-only run (Stage-0
   check + Stage-2 fault audit: faulted expert set ⊆ demanded
   expert set ∪ prefetched set, with prefetched bytes reported
   separately).
5. No silent precision change (exact policy enforced; quantizer
   coverage verified for the proof file's types).
6. Predictor gating: `CacheExpertPredictor` ships as a live
   prefetcher only if its shadow report beats demand-only on
   bytes/token at the same budget on the proof model's traces.
   Otherwise it stays shadow-only and the design says so.

**Environment.** Kaggle CPU kernel, same pattern as the 110B run
(tree pinned, budgets stated, /tmp for the model file — never
this box's ~20 GB free disk for a multi-GB model). The 110B run's
baseline-harness lesson applies: the dense baseline step gets a
timeout larger than any plausible dense runtime, and its timeout
handler is tested before the run — a harness crash must never
again masquerade as a model result.

## 11. Roadmap rungs (after proof 1)

- **Rung 2:** shared-expert models (Qwen3-30B-A3B class: 30B
  total, small active set — the natural second rung, still GQA,
  still no MLA); `CacheExpertPredictor` live if gated (§7);
  Ordo expert segments if gated (§7); per-call head pinning
  (§5) if the verification-block measurements show the head
  penalty at MoE scale.
- **Rung 3:** DeepSeek family — sigmoid+bias routing (§3) plus
  MLA attention as its own design (not a paragraph in this one);
  shared experts at DeepSeek scale; MXFP4/FP8-native dequantizers
  verified bit-exact vs ggml before any exact claim on those
  models (Kimi K3 native MXFP4 is explicitly "extreme territory,
  not exact" in `TARGET_MODELS.md`).
- **Rung 4:** trillion tier (`TARGET_MODELS.md` list) — each model
  gets its own ShadowScorer working-set measurement under the
  4.5 GB floor budget before any floor claim; disk-fit is checked
  per model (only Inkling 1-bit GGUF and Muse Spark NVFP4 fit a
  500 GB floor disk; K3 at ~1.56 TB needs a bigger NVMe — no
  software fixes a disk that's too small).
- **GPU path:** layer-streaming with per-layer expert page sets
  (§2) streamed to VRAM per layer, same interface surface as the
  CPU executor so `SpeculativeDecoder` runs unchanged — a
  separate design document when the CPU MoE executor is proven,
  tested on a Kaggle GPU kernel like the 104B dense GPU check.

## 12. Open questions, with recommended answers

| # | Question | Recommended answer | What would change it |
|---|---|---|---|
| 1 | One tensor id per stacked expert matrix, or per-expert tensor ids in the `.lac`? | Keep one id per stacked matrix; expert slicing is a flattened-row-range reader view (§2). Zero converter change, reuses verified `read_tensor_rows`. | If profiling shows page-table walks dominating (unlikely: page sets are computed once per layer per block), revisit with measured numbers. |
| 2 | Separate expert cache / budget? | No — one store, one LRU, one page budget + one dequant budget (§4). | A measured trace where a split budget beats unified on bytes/token at the same total budget. |
| 3 | Pinned hot set of frequent experts? | No — EWMA `_freq` inside the existing store, still evictable; warm start is a cache, not a pin (§4). | Nothing short of a measured, reproducible win that survives the memoryless falsifier (Ordo's gate exists because ungated speculation measured 26% worse there). |
| 4 | Router normalization per family? | Arch-driven switch in code, verified by token comparison per family, fail loud on unknown (§3). | A family whose rule is measured to differ from the switch's entry — then the entry is corrected, with the measurement cited. |
| 5 | Top-k tie-breaking? | Higher score, then lower expert index; verified by Stage-0 tie tests and reported as near-tie class on divergence (§3). | A reference comparison showing a different tie order — then the rule is changed to match, with the comparison cited. |
| 6 | MXFP4 / FP8-native models in the exact tier? | Out until their dequantizers are added and verified bit-exact vs ggml (§8.2). K3 stays out of exact claims until then. | A verified dequantizer (bit-exact on real model bytes, same bar as the dense K-quant dequantizers). |
| 7 | MLA attention in v1? | Out — separate design after MoE FFN is proven (§6). | Nothing in v1; MLA is a new attention implementation, not a flag. |
| 8 | Which predictor ships first? | None in v1 (demand-driven exact). `CacheExpertPredictor` rung 2, Ordo rung 3, each gated by shadow measurement beating demand-only (§7). | Shadow reports on real proof-model traces — the gate, not this document, decides. |
| 9 | Page size for expert pages? | Keep the store's 2 MiB pages; expert fault granularity is pages, and experts span many pages (§2). | A measured case where smaller pages cut fault bytes enough to matter (boundary sharing is already negligible at expert sizes). |
| 10 | First proof model? | 7B-class, GQA, no MLA, no shared experts — OLMoE-1B-7B leading candidate, Qwen1.5-MoE-A2.7B fallback, file identity recorded (§10). | The proof file's own metadata contradicting the candidate's assumed shape — then the candidate changes, recorded as a plan change. |
| 11 | Draft for MoE speculative decoding? | A small dense draft (the 7B proof's pattern: 0.5B-class draft, K=4). The draft proposes tokens only; it never predicts experts — verification computes routers exactly (§5, §7). | A measured draft with materially higher acceptance on the proof model (e.g. a distilled draft, the speculative doc's named follow-up). |
| 12 | Does this design copy the beat-target? | No — §4 and §7 name the three mechanisms not used (per-layer LRU, learned pinned hot set, router-lookahead thread) and the Lacuna-native replacements (unified store LRU + EWMA, `CacheExpertPredictor`, Ordo automaton, speculative-block exact demand). | — (a design-compliance statement, not a measurement). |

## 13. What remains unproven (carried forward, not buried)

1. Everything in this document — it is a design, and designs do
   not run models. The proof plan (§10) is the only route from
   this document to a claim.
2. Router-rule correctness for any real MoE family (Stage-0
   synthetic tests + Stage-1 token comparison are the check).
3. Expert working-set reuse on real traces (the floor claim's
   decider, per `TARGET_MODELS.md`) — ShadowScorer on the proof
   model's traces, under the 4.5 GB budget.
4. Predictor value (`CacheExpertPredictor`, Ordo) on real MoE
   traces — shadow-gated, both unmeasured today.
5. Trillion-tier feasibility per model — cold bounds stand until
   rung-4 measurements exist; disk-fit per model stands as a
   physical constraint.
6. GPU MoE execution — separate design, Kaggle GPU kernel test
   route, same as the dense GPU path.
