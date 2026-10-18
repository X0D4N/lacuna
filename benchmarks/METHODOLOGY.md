# Benchmark methodology

## The honesty rule

Every performance number in this project is labeled:

- **SIMULATED** — produced by `simulator/simulate.py`, a discrete-event model of
  the target hardware. Not measured on any machine.
- **MEASURED** — produced by `tests/` (correctness/latency of the machinery on
  tiny synthetic inputs) or by a real hardware run documented in RESULTS.md.

No number without a label is a claim.

## What the simulator models

Target hardware: RTX 2050-class GPU (4 GB VRAM), 8 GB system RAM, NVMe with
3.5 GB/s sequential reads.

| Assumption | Value | Source / rationale |
|---|---|---|
| NVMe sequential bandwidth | 3.5 GB/s | typical PCIe 3.0 x4 NVMe |
| Effective sparse-matmul throughput | 8 TFLOPS | RTX 2050 dense FP16 peak is ~20+ TFLOPS; sparse gather/scatter is far lower — 8 is a conservative working figure |
| FFN sparsity | 90% inactive/token | PowerInfer-class, ReLU/ReGLU models (see docs/PRIOR_ART.md) |
| Oracle recall | 0.95 | 5% of active pages fault on demand |
| Draft acceptance | 0.85, K=4 | EAGLE-class speculative decoding |
| Union mask growth | 1.2x single-token mask | sublinear growth over K drafts |
| MoE active set residency | 95% in 6 GB RAM hot-set | EWMA hot pages + 5% expert-miss faults |
| p99 tail factor | 1.15x mean | prefetch stalls / OS jitter |

Per-token time = `compute + max(0, fetch - compute)` (fetch hides behind the
previous token's compute; the uncovered remainder is paid). Compute =
`2 * active_params / 8 TFLOPS`.

## Exact-mode methodology

Exactness is **proven, not simulated**. The discrete-event simulator above does
not model exactness at all — it models speed. The exactness argument is
structural, and it is verified by tests:

1. **Guarantee mechanism** (`lacuna/scheduler.py`, `lacuna/engine.py`): in
   exact mode, after the oracle's union-mask prefetch resolves, the scheduler
   calls `PagedWeightStore.ensure_resident(full_required_page_set)` — a
   blocking fault of every page the step's compute may touch — *before* the
   compute callable runs. The oracle's output only orders the prefetch; it
   never gates what compute reads. `exact_guarantee_held` is measured per
   step via `are_resident`, not assumed.
2. **Token identity** (`tests/test_exact.py`): exact-mode output is
   token-identical to a dense baseline (all pages pinned resident) across
   several prompts — including with a deliberately broken random-mask oracle,
   where the only difference is `exact_faults > 0` (slower, never different).
   A small-budget run measurably reports `exact_guarantee_held = False`,
   proving the flag is a measurement, and its output is still identical.
3. **No requantization** (`tests/test_precision_policy.py`, `tests/test_convert.py`):
   the `exact` precision policy stores source bytes bit-identical; the policy
   id is recorded in the `.lac` header (v2) and exposed by the reader.
4. **Tree speculation** (`tests/test_tree_spec.py`): accepted tokens equal
   plain target-only greedy decoding for the same seed, by construction
   (acceptance is defined as the target's own greedy choice at each position).

What "exact" does NOT claim: it does not make a quantized source model
smarter — it preserves *whatever the source GGUF was*, bit-for-bit, with no
additional degradation. The ecosystem standard for full intelligence is
Q4_K_M-class quantization; Lacuna exact mode keeps you at exactly your
source's level, never below.

## Perplexity harness (Day 3 — MEASURED on a tiny model)

The simulator's "does NOT model" line about Q2/ternary quality cost is now
answered at small scale by `benchmarks/perplexity/`:

- A ~260K-param numpy transformer (RMSNorm + MHA + SwiGLU), trained from
  scratch on 60K tokens of deterministic synthetic English
  (`benchmarks/perplexity/corpus.txt`), held-out eval on 12K tokens.
- Four weight sets on the same eval text: dense FP32 baseline, `exact`
  (RAW byte roundtrip — must equal dense bit-for-bit), `extreme` with the
  PRODUCTION tier rule (`lacuna/convert.py::_tier_for`, 200 KiB threshold →
  Q4 attn/norm/embed, Q2 FFN), and a 50 KiB "ternary stress" variant that
  forces the cold tier on to isolate its cost.
- Perplexity = exp(mean token NLL), teacher-forced, full eval set.
- Run: `PYTHONPATH=. python -m benchmarks.perplexity.measure`
  (or `lacuna perplexity`); writes `benchmarks/perplexity.md`.

Label semantics: every number in `benchmarks/perplexity.md` is MEASURED
(real forward passes through the real codecs) — but on a synthetic
260K-param model, so the *deltas* are the finding and the absolute values
do not transfer to real LLMs. Production pages use per-page scales (finer
than the harness's per-tensor scales), so the measured extreme deltas are a
conservative upper bound.

## What the simulator does NOT model

- Prefill (long-prompt) performance — decode only.
- KV-cache growth with context length.
- PCIe contention, thermal throttling, OS background I/O.

(Quality/perplexity impact of the Q2/ternary cold tiers was on this list
until Day 3: it is now measured at tiny-model scale — see the perplexity
harness section above and `benchmarks/perplexity.md`. A full-scale eval on
a real LLM still needs a GPU box.)

## Reproducing

```bash
python -m simulator.simulate
```

Deterministic: no randomness, no wall-clock dependence.
