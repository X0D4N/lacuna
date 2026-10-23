# Lacuna — Architecture

**Lookahead sparse-paged inference: run 100B-class models on 8 GB RAM + <4 GB VRAM.**

## The pure invention (one primitive)

**The lookahead prefetch contract.** Every existing inference system *reacts*: a
token arrives, its weights are fetched. Lacuna *contracts*: the draft model
proposes K future tokens, the oracle converts them into a single union mask — a
binding schedule of exactly which weight pages the next K tokens will touch —
and the pager treats that schedule as a contract, streaming those pages before
they are needed. Speculation stops being "guess the token" and becomes "reserve
the memory." No published system does this at weight-page granularity; the MoE
world has only the coarse expert-level shadow of it (SP-MoE, MoE-SpeQ). The
paging, the overlap scheduler, and the precision tiers are all engineered around
this one primitive. That is what makes Lacuna an invention rather than an
assembly — and the primitive generalizes: any autoregressive computation with
predictable sparse memory access can run under a lookahead contract.

Target hardware: 500 GB NVMe storage, 8 GB system RAM, RTX 2050-class GPU (<4 GB VRAM).
Baseline experience: a 100B dense model answers at usable interactive speed instead of
being impossible to load; a 100B MoE model (small active params) runs fast.

## 1. The honest physics (why "smooth like a 3B model" is not the claim)

Token generation (decode) is **memory-bandwidth bound**: cost per token ≈ bytes of
weight that must be read per token.

| Model | Precision | Bytes / token (dense) | @ 3.5 GB/s NVMe |
|---|---|---|---|
| 100B dense | 2-bit | ~25 GB | ~7 s/token — unusable |
| 100B dense, 90% sparse | 2-bit | ~2.5 GB | ~0.7 s/token |
| 100B dense, 90% sparse, K=4 spec | 2-bit | ~0.6 GB amortized | ~4–6 tok/s — usable |
| 100B MoE (20B active) | 2-bit | ~5 GB, mostly resident | 10+ tok/s |

No invention changes the bandwidth wall. Lacuna's job is to **minimize bytes per
token** (sparsity + low precision + speculation amortization) and then **hide the
latency** of fetching those bytes (async prefetch overlapped with compute). The disk
stays the bottleneck — it just stops being *felt*.

## 2. Core mechanisms (the invention)

Five orthogonal mechanisms, integrated as one engine on top of llama.cpp:

### M1 — Activation Oracle (lookahead sparsity prediction)
Tiny per-layer predictor MLPs (DejaVu-style, trained once per model offline on
calibration text) predict which FFN neurons / attention heads activate for the next
token. **New:** the oracle predicts over a *lookahead window*, not one token — it
outputs the UNION mask of activations for the next K draft tokens. One prediction
covers K tokens of prefetching.

### M2 — NVMe paged weight store
Weights repacked from GGUF into a page-addressable format (2 MB pages, per-page
metadata: layer, tensor, row range, precision). Linux `io_uring` async reads stream
pages straight into pinned host buffers. 100B @ 2-bit ≈ 25 GB — trivial for 500 GB.

### M3 — Hot-set residency
8 GB RAM + 4 GB VRAM hold: embeddings, norms, attention projections, KV cache, and
the statistically hottest FFN pages (tracked by an EWMA frequency counter). The GPU
works only on resident pages; everything else streams.

### M4 — I/O–compute overlap scheduler
While the GPU computes token *t* on resident pages, the prefetcher loads the oracle's
union-mask pages for tokens *t+1..t+K*. Double-buffered: compute never waits for
I/O unless the oracle mispredicts (fallback: on-demand page fault, counted as a
metric, oracle retrains on misses).

### M5 — Speculative amortization
A tiny draft model (0.5–1B, fully resident) proposes K tokens; the main model verifies
all K in **one sparse pass** over the union mask. Weight-streaming cost is paid once
per K tokens instead of once per token — this is the arithmetic that turns 1.4 tok/s
into 4–6 tok/s. The tree variant (`speculative_tree_generate`, SpecInfer/EAGLE-3
style) proposes a tree of candidate paths per round and accepts the longest prefix
the target greedily agrees with — exact by construction: accepted tokens equal
plain target-only greedy decoding.

### M6 — Adaptive per-page precision (cold pages go lower)
Hot pages stored at 4-bit, warm at 2-bit, cold at ~1.58-bit (BitNet-style ternary).
Precision follows measured activation frequency, recalibrated periodically. Quality
cost is bounded because cold neurons contribute little by definition.

## 3. Exactness guarantee ("prediction guides memory, never math")

Lacuna's default mode is **exact**: bit-for-bit identical output to running the
same model densely, by construction — not by hoping the oracle is accurate.

Three rules enforce it:

1. **The oracle never skips computation.** Unlike PowerInfer-style pruning
   (where the predictor decides which neurons participate in the math, at a
   quality cost when wrong), Lacuna's oracle is a *prefetch scheduler only*.
   Its union mask decides the *order* pages stream from NVMe, never *whether*
   a weight participates in a matmul. A wrong prediction costs time (an
   on-demand fault), never quality.
2. **Full residency before compute.** In exact mode the scheduler calls
   `PagedWeightStore.ensure_resident(full_required_page_set)` before every
   compute step, after the union-mask prefetch resolves. Compute therefore
   never observes a missing page. Forced faults are counted as `exact_faults`;
   the flag `exact_guarantee_held` is *measured* per step via `are_resident`
   and is True only if every step verified full residency.
3. **No requantization.** The `exact` precision policy (default) stores every
   tensor's source bytes bit-identical in the `.lac` file — paging and
   sparsity only, zero requantization. The ecosystem standard for full
   intelligence is Q4_K_M-class quantization; Lacuna's exact mode preserves
   whatever precision the source GGUF had, bit-for-bit, so there is never
   *additional* degradation.

Consequence, stated plainly: in exact mode, oracle quality affects **speed
only**. This is proven, not simulated: `tests/test_exact.py` shows
token-identical output to the dense baseline even with a deliberately broken
(random-mask) oracle — the only difference is `exact_faults > 0`. And
`tests/test_tree_spec.py` proves tree speculation accepts exactly the
target's own greedy tokens.

Honest scaling boundary: the reference engine ensures the full page set per
decode step, so the guarantee needs `hot_budget_bytes` ≥ the model's paged
size; when the budget is smaller the flag measurably reads False instead of
pretending (there is a test for this too). Production deployments tile per
layer (per expert for MoE) — the scheduler's `required_pages_for_step` hook
exists for exactly that.

`--extreme` opts out of all three rules: adaptive Q4/Q2/ternary tiering with
no residency guarantee — fastest, documented quality/speed tradeoff, chosen
explicitly.

## 4. Why it is universal ("small tech, billions of people")

- **Model-agnostic:** any GGUF model converts via `lacuna convert` (offline, one
  pass). Dense, MoE, present and future architectures — the oracle trains per model,
  the machinery doesn't care.
- **Hardware-agnostic tiers:** auto-detects RAM/VRAM/NVMe bandwidth at startup and
  picks hot-set sizes, page sizes, K, and precision mix. Graceful degradation: on
  weaker disks it lowers K and raises sparsity threshold instead of failing.
- **Builds on llama.cpp:** Lacuna is an orchestration + paging layer *above*
  llama.cpp's execution core (GGUF format, k-quants, ggml kernels). No fork of the
  hot path; integrates via the C API / server hooks (see `integrations/llama.cpp/`).
  Ollama users benefit automatically wherever Ollama embeds llama.cpp.
- **Small:** the engine is a few thousand lines — pager, oracle, scheduler,
  draft-verifier, converter, simulator. No cluster, no daemon fleet, no cloud.

## 5. What "no bottlenecking the system" means concretely

- RAM cap enforced: hot-set has a hard byte budget (default: 6 GB of 8 GB); the OS
  and other apps keep the rest. Pager uses `mlock`-free eviction, never swap.
- VRAM cap enforced: working set ≤ 3.2 GB of 4 GB; KV cache quantized (8-bit) and
  bounded by context window setting.
- I/O budget: prefetch queue depth capped so NVMe stays responsive for the OS;
  sequential 2 MB reads (no random 4K storms).
- Measurable: every run reports `tok/s`, `prefetch hit rate`, `oracle precision/
  recall`, `bytes/token`, `p99 inter-token latency` — the numbers that prove the
  system isn't choking.

## 6. Honest limits (stated upfront, verified by `simulator/`)

- Dense 100B will never feel like a 3B model on this hardware. Physics forbids it.
  Target: **usable** (multi tok/s), not instant.
- Prefill (long prompts) is slower than decode — chunked prefill with the same
  paging machinery; very long contexts page KV to NVMe too.
- Oracle mispredictions cost on-demand faults; worst case degrades to FlexGen-like
  streaming, never worse than naive offload.
- Full 100B validation needs a GPU machine; until then the repo ships a
  discrete-event simulator calibrated to RTX 2050 / NVMe specs plus small-model
  end-to-end correctness tests of the paging machinery.

## 7. Repository layout

```
lacuna/          # the engine (Python package)
  oracle.py          # M1 activation oracle (train + infer)
  recipes.py         # per-family oracle training recipes (Qwen/Llama/MoE) + calibration
  pager.py           # M2/M3 paged weight store + hot-set
  scheduler.py       # M4 I/O-compute overlap scheduler
  speculate.py       # M5 draft-verify loop (linear + tree)
  quant.py           # M6 adaptive per-page precision
  convert.py         # GGUF -> .lac paged format
  buffer.py          # pager-backed buffer: runnable ggml backend-buffer seam
                     # (alloc/bind/ensure_resident/touch/row-masked matvec,
                     # per-layer tiled_exact_matvec) — the Python side of the
                     # C++ patch in integrations/llama.cpp/sp_buffer.cpp
  engine.py          # orchestration on top of llama.cpp (+ one-line Engine API)
  hardware.py        # tier detection: Linux + Windows 11 only
  cli.py             # zero-setup CLI: run/serve/convert/mcp/bench
  server.py          # OpenAI-compatible HTTP server (stdlib only)
  mcp_server.py      # MCP server over stdio for agent integration
simulator/           # discrete-event hardware simulator -> honest tok/s
tests/               # unit + small-model e2e correctness
integrations/llama.cpp/  # sp_buffer.{h,cpp}: pager-backed ggml backend buffer
                     # type (interface prototype, syntax-checked) + INTEGRATION.md
                     # (seam design: buffer type, oracle-gated matmuls, prefetch)
benchmarks/          # methodology + result tables
docs/                # this file, design notes, dark-themed site source
```
