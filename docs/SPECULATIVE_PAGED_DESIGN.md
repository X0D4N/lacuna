# Speculative paged inference — design + measured results

Status: prototype implemented (`lacuna/speculative.py`), exactness proven on
the tiny model and on the 0.5B. Speed results are regime-dependent and
reported honestly below — the mechanism divides weight traffic, but with a
weak draft it does NOT make the 0.5B faster. Read the numbers before
claiming a speedup.

## Mechanism (what it is)

Greedy speculative decoding wired to the real paged executor:

1. A cheap draft proposes K tokens (default K=4) conditioned on all emitted
   tokens. Shipped draft: token-id n-gram (order 4, backoff, deterministic).
2. The target verifies the whole block in ONE batched pass
   (`BatchedQwen2.forward_all`, per-position logits): every layer's weights
   are read and dequantized once for all K positions, not once per token.
3. The longest prefix the target greedily agrees with is accepted. At the
   first mismatch the target's own token is emitted (resample); on full
   acceptance a bonus token (target argmax after the block) is emitted.
   Every block yields >= 1 token and costs exactly one batched pass.
4. Pending-token trick: a resample/bonus token has not been through the
   layers yet, so it is not in the KV cache. Instead of paying a
   single-token forward to put it there, it opens the NEXT verification
   block (`[pending] + new draft`). Cache invariant: the cache covers
   emitted-minus-pending; after each block it is truncated to exactly that
   prefix, so positions and attention always match plain greedy decoding.

Exactness (by construction, verified empirically): an accepted draft token
equals the target argmax at its position (that is the acceptance check);
a resample/bonus token IS the target argmax at its position. The emitted
sequence is therefore the greedy sequence. Tests pin this for an oracle
draft (full acceptance), a constant wrong draft (all resample), and a real
n-gram draft; the 0.5B runs below compare ids token-for-token.

`forward_all` is a deliberate copy of `PagedQwen2.forward` with only the
head section changed (per-position logits instead of last-position logits).
It lives in `speculative.py`, not `paged_model.py`, because this work's
constraint was new-files-only; the proper home is a `forward_all` method on
`PagedQwen2`, and any upstream change to `forward` must be mirrored here
until that move happens. `forward_all(...)[-1]` equals `forward(...)`
bit-for-bit — pinned by test.

## Measured results — 0.5B, Qwen2.5-0.5B-Instruct Q4_K_M, K=4, n-gram draft

Corpus for the draft: 20 generic sentences (weather, cooking, travel,
work). It contains neither test prompt's continuation — no "capital of
France", no color list. Acceptance below is what a generic n-gram draft
actually achieves, not a tuned number.

### 2GB budget (cache fits — compute-bound regime)

| prompt | plain tok/s | speculative tok/s | speedup | acceptance | tokens/block | ids identical |
|---|---|---|---|---|---|---|
| The capital of France is | 12.80 | 3.45 | 0.27x | 4/112 (3.6%) | 1.14 | YES (32/32) |
| Name three primary colors | 11.99 | 2.66 | 0.22x | 2/120 (1.7%) | 1.07 | YES (32/32) |

Reading: exactness holds (32/32 on both prompts), but speculative is
3.7–4.5x SLOWER here. That is the predicted compute-bound result: a batched
block costs ~K positions of compute and yields ~1 token with a weak draft,
so tok/s scales as (tokens/block)/K. Store traffic is nil in both cases
(2–12 MB read; everything served from the dequant cache after warm-up),
so there is no traffic left to divide. **In a cache-fit regime,
speculative decoding only wins with a draft whose acceptance approaches
K — i.e. a neural draft, not an n-gram.**

### 128MB budget (thrash regime — the regime that matters for 110B)

| prompt | plain tok/s | speculative tok/s | speedup | acceptance | tokens/block | ids identical |
|---|---|---|---|---|---|---|
| The capital of France is | 0.249 | 0.192 | 0.77x | 4/112 (3.6%) | 1.14 | YES (32/32) |
| Name three primary colors | 0.230 | 0.182 | 0.79x | 2/120 (1.7%) | 1.07 | YES (32/32) |

Reading: exactness holds in the thrash regime too (4/4 prompts identical
across both budgets), but speculative is still 21–23% SLOWER with this
draft — and the byte counts say why. Plain reads 13.0 GB per 32 tokens
(406 MB/token stored); speculative reads 28.1–29.9 GB (879–934 MB/token),
~2.2x more. The extra cost is the head: `forward_all` computes logits per
position, and at a 128 MB budget the 19 head blocks thrash across the
block's 5 positions (each position re-faults blocks the previous position
evicted), while plain generation pays the head once per token. So a block
costs ~1000 MB of reads, not ~406 MB.

Breakevens from these numbers (not predictions): thrash-regime speedup
~ (tokens/block) / 2.46 — break-even needs mean accepted >= ~0.5
tokens/block (current n-gram draft: ~0.1). Cache-fit break-even needs
mean accepted ~= 3.2 of K=4. The draft, not the mechanism, is the gap.

Why the 0.5B understates the 110B win — arithmetic, not a measurement:
the 0.5B's head is 136 MB stored of a 485 MB model (28%), so head
re-reads dominate the penalty. The 110B's head is vocab x hidden =
151936 x 8192 ~= 1.24B params, ~1–3% of the 41 GB file. At 110B scale the
per-position head cost is a rounding error next to the 41 GB body pass,
so block bytes ~= plain bytes and break-even acceptance falls to
~0.05 tokens/block. The 110B kernel measures whether that arithmetic
survives contact with reality; it is not claimed here.

## Scaling to dense 110B (what this does and does not buy)

Weight traffic per emitted token divides by tokens-per-block in the
weight-bound regime: with a draft averaging `a` accepted tokens, traffic
is ~1/(a+1) of plain paged decoding. Concretely, if plain 110B paged
generation costs ~60 s/token on free CPU (a full 41 GB weight pass per
token), a draft with a=2 brings it to ~20 s/token; a=3 to ~15 s/token.

What it does NOT buy: the compute floor. A dense 110B token costs
~220 GFLOP of body compute; on a 2-core CPU at ~50 GFLOPS that is ~4–5
s/token even with every weight cached, before dequant and head costs.
Speculative decoding divides traffic, not FLOPs. "Fast like heavy
hardware" for dense 110B therefore needs the GPU path below IN ADDITION
to speculation — speculation alone gets 110B from unusable to slow, not
to fast. Anyone claiming otherwise from these numbers is misreading them.

Where speculation wins big: MoE models. Only the active experts are read
per token, so the per-token working set is small and the draft's block
amortizes the expert-fetch cost. The 110B dense run proves exactness; the
speed story for Lacuna lives on MoE targets (see TARGET_MODELS.md) plus
the GPU path.

Draft quality is the whole game: the breakevens measured on the 0.5B are
mean accepted >= ~0.5 tokens/block in the thrash regime and ~= 3.2 of K=4
in the cache-fit regime. The real follow-up is a distilled neural draft (a
small model trained on the target's greedy outputs, cost ~5–10% of the
target per token, acceptance 2–4 tokens/block in published
speculative-decoding results). An n-gram draft is the honest baseline,
not the destination.

Two further follow-ups the measurements identify (neither implemented
here): (1) per-call head pinning — within one `forward_all` call, hold
head blocks in a transient per-call cache released after the call, so the
block's positions share one head read instead of re-faulting per position
(this is what costs the 2.2x bytes above; at 110B scale the head is too
small for this to matter, but at 0.5B scale it is the whole penalty);
(2) faster dequantization — the 128 MB runs are dequant-bound (plain:
~1.66 GB re-dequantized per token in ~4 s, ~425 MB/s through the NumPy
K-quant kernels), so a faster dequantizer speeds plain and speculative
alike. Extrapolated, and labeled as extrapolation: plain 110B paged
generation on this class of CPU would cost ~370 GB of re-dequantization
per token, i.e. ~14 min/token — speculative with a=2 would cut that to
~5 min/token, still not "fast". The Kaggle kernel measures the real
110B number; if it lands near this extrapolation, the GPU path below
stops being optional for the "fast" claim.

## Companion piece (a): prefetch pipelining — design + thread-safety verdict

Goal: overlap next-layer page-in with current-layer compute, so disk
latency hides behind math instead of stalling it.

What exists today (read from `pager.py`, not assumed):

- `PagedWeightStore` already owns a `ThreadPoolExecutor` (4 workers) and
  `prefetch(indices)` — but it is SYNCHRONOUS despite the name: it submits
  loads and then blocks on every future (`fut.result()`) before returning.
  It batches faults; it does not pipeline them.
- `ensure_resident(page_ids)` / `are_resident(page_ids)` already provide
  the exactness primitive: call with the full required page set before
  compute and compute provably never observes a missing page. A wrong
  prediction costs faults (time), never correctness.
- Thread-safety verdict: cache access is locked (`_lock` guards `_cache`
  in `get_page`/`_insert`/`is_resident`/`are_resident`/`ensure_resident`),
  and page reads are thread-safe (`os.pread` on Linux; serialized file
  handle on Windows). Two races exist but touch only statistics, not
  correctness: `stats` dict increments and `_freq` EWMA updates are
  unlocked, so concurrent readers can lose increments. The page data path
  itself is safe for concurrent faulting.

Design (not yet implemented): a true async prefetch that submits
`_load_page` jobs without waiting and inserts on completion via callback,
driven by the layer schedule (while layer L computes, prefetch layer
L+1's pages; on a draft-verification block, prefetch the whole block's
layer set). Before landing it, the stats/`_freq` races must be put under
the existing lock (a `pager.py` change — out of scope for this
new-files-only task, flagged here so it is not forgotten). Prefetch
priority must stay below demand faults: a mispredicted prefetch may evict
a useful page under LRU, so the prefetcher should only use pages the
layer schedule is about to need, never speculative guesses beyond one
layer ahead.

## Companion piece (b): GPU executor path — DESIGN ONLY, UNTESTED

No GPU exists on this box, so everything in this section is interface and
arithmetic, not a measured result. Do not present it as working.

Floor spec: RTX 2050 (4 GB VRAM) + 6 GB RAM. The design is paged layer
streaming: weights live on disk/NVMe, stream one layer at a time to the
GPU, compute that layer on the GPU, keep the KV cache on the GPU while it
fits (110B KV for moderate context: 80 layers x 8 KV heads x 128 dim x 2
bytes x 2 (K+V) x 1024 tokens ~ 1.6 GB — fits alongside one layer's
weights), and fall back to CPU KV with per-layer transfer when it does
not. One 110B layer at Q2_K is ~0.5 GB; a 4 GB card holds the layer +
activations + a KV slice with room to spare, and CUDA streams overlap
next-layer H2D copy with current-layer compute (the GPU form of the
prefetch above).

Interface (mirrors the CPU executor so callers do not change):

- `PagedGPUQwen2(gguf_path, lac_path, vram_budget_bytes)` with the same
  `forward_all(ids, pos_offset)` / `generate(prompt, max_tokens)` surface
  as `BatchedQwen2`, so `SpeculativeDecoder` runs unchanged on top of it.
- Weight path: layer weights fault from the same `PagedWeightStore` on
  the CPU side, then H2D copy per layer; the GPU never reads disk
  directly. Dequantization happens on the GPU (quant kernels) or on the
  CPU before copy — an open decision, not a settled one.
- Open decisions (do not silently pick): GPU backend library (the repo is
  stdlib+numpy only, so cuBLAS-via-ctypes, cupy, or torch each break that
  rule differently and need a maintainer call); dequant on GPU vs CPU;
  KV placement policy under long contexts.

Test route: a Kaggle GPU kernel (2xT4 on the free tier stood up CUDA
before for the 104B dense check, including a manual driver install).
That kernel is the first real test of this design — nothing here is
claimed until it runs there.

## What remains unproven (with reasons)

1. 110B-scale tok/s and acceptance — needs the real 110B run (Kaggle
   kernel in flight). The 0.5B numbers do not transfer: the head-fraction
   arithmetic says the thrash penalty shrinks at 110B scale, but only the
   kernel measures it. The dequant-bound extrapolation (~14 min/token
   plain) is likewise unconfirmed until the kernel reports.
2. Acceptance at 110B scale with a neural draft — the n-gram draft's 2–4%
   acceptance on the 0.5B does not transfer to any claim about 110B; a
   distilled draft still needs to be built.
3. Prefetch pipelining — designed above, not implemented (needs the
   `pager.py` stats-lock fix plus the async submit path).
4. GPU executor — design only, no GPU on this box; Kaggle GPU kernel is
   the test route.
5. `forward_all` living in `speculative.py` as a copy — drift risk vs
   `PagedQwen2.forward`; the proper home is a method on `PagedQwen2`
   (needs a decision to lift the new-files-only constraint).
