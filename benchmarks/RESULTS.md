# Results

## Simulator output (ALL NUMBERS SIMULATED)

Target: RTX 2050 4GB VRAM / 8GB RAM / NVMe 3.5 GB/s. Assumptions:
`benchmarks/METHODOLOGY.md`. Generated 2026-10-16 with
`python -m simulator.simulate` (deterministic).

| Scenario | tok/s | GB/token | p99 latency |
|---|---|---|---|
| (a) Dense 100B @2-bit, naive streaming | 0.14 | 25.00 | 8.21 s |
| (b) + 90% oracle sparsity | 1.33 | 2.62 | 0.86 s |
| (c) + speculative K=4 over union mask (Lacuna) | 3.54 | 0.99 | 0.32 s |
| (d) MoE 100B, 20B active @2-bit, hot-set resident | 14.00 | 0.25 | 0.08 s |

Reading: the full Lacuna stack (c) turns an unusable 0.14 tok/s dense
stream into a usable ~3.5 tok/s on the target box; MoE-100B-class models (d)
with small active parameter counts run fast. Dense 100B will never feel like a
3B model on this hardware — physics forbids it; "usable" is the honest target.

## Simulator sweeps (ALL NUMBERS SIMULATED)

2026-10-20, `python -m simulator.sweeps` (module `simulator/sweeps.py`,
parity-tested against `simulator/simulate.py` scenario_c in
`tests/test_sweeps.py`). Same stated assumptions as the table above;
union-mask growth vs K is a stated guess (5%/depth, = 1.2x at K=4) until
shadow scoring measures it on real weights (blocked: needs a GPU box).

| Oracle sparsity | 80% | 85% | 90% | 95% | 97% |
|---|---|---|---|---|---|
| tok/s | 1.77 | 2.36 | 3.54 | 7.08 | 11.80 |

| Spec depth K | 1 | 2 | 3 | 4 | 5 | 6 | 8 |
|---|---|---|---|---|---|---|---|
| tok/s | 1.27 | 2.24 | 2.98 | 3.54 | 3.96 | 4.26 | 4.62 |

| NVMe GB/s | 1.75 | 3.50 | 7.00 |
|---|---|---|---|
| tok/s | 1.77 | 3.54 | 7.08 |

Reading: sparsity is the dominant lever (90% → 97% doubles tok/s twice);
K has diminishing returns past ~4 at the stated 0.85 acceptance; the
fetch-bound regime scales linearly with NVMe bandwidth.

### Trillion-tier target list: per-token weight movement (SIMULATED)

Specs from `docs/TARGET_MODELS.md` (sourced there). Cold = active × 0.5
bytes/token; MID steady state = 95% hot-set residency (Fahad's 2026-10-19
target — an aspiration for the shadow-scorer to earn, not a claim).

| Model | Active/token | Cold GB/token | Cold tok/s | MID steady GB/token | MID steady tok/s | Cold verdict |
|---|---|---|---|---|---|---|
| Kimi K3 | 104B | 52.0 | 0.07 | 2.60 | 1.3 | DESKTOP only |
| Qwen3.8-Max | 95B | 47.5 | 0.07 | 2.38 | 1.5 | DESKTOP only |
| DeepSeek-V4-Pro | 49B | 24.5 | 0.14 | 1.23 | 2.9 | DESKTOP only |
| LongCat-2.0 | 48B | 24.0 | 0.15 | 1.20 | 2.9 | DESKTOP only |
| MiMo-V2.5-Pro | 42B | 21.0 | 0.17 | 1.05 | 3.3 | DESKTOP only |
| Yuan3.0 Ultra | 68.8B | 34.4 | 0.10 | 1.72 | 2.0 | DESKTOP only |
| Kimi K2.7 Code | 32B | 16.0 | 0.22 | 0.80 | 4.4 | DESKTOP only |
| Ring-1T | 50B | 25.0 | 0.14 | 1.25 | 2.8 | DESKTOP only |
| Inkling | 41B | 20.5 | 0.17 | 1.03 | 3.4 | DESKTOP only |
| Muse Spark Open | 42B | 21.0 | 0.17 | 1.05 | 3.3 | DESKTOP only |

Reading: every model moves > 4 GB/token cold → DESKTOP-only cold, no
exceptions, reproducing the doc's verdicts. Kimi K2.7 Code (32B active,
native INT4) is closest to MID at ~4.4 tok/s steady — the decider is expert
working-set reuse on real traces, measured by the ShadowScorer.

## Perplexity: exact vs extreme precision (MEASURED, tiny model)

2026-10-18, `benchmarks/perplexity/measure.py` (also `lacuna perplexity`).
Real trained forward passes through the real codecs — but on a 258,528-param
synthetic model, so only the *deltas* are the finding. Full methodology and
caveats in `benchmarks/perplexity.md`.

| policy | tier mix | eval ppl | Δppl vs dense | stored bits/param |
|---|---|---|---|---|
| dense FP32 (baseline) | f32×21 | 6.0111 | — | 32.00 |
| exact (RAW pages) | raw[exact]×21 | 6.0111 | 0.000 (bit-identical) | 32.00 |
| extreme · Q4/Q2 | q2×6, q4×15 | 13.1641 | +7.1530 | 3.14 |
| extreme · ternary stress | q4×13, ternary×8 | 41.4250 | +35.4140 | 2.57 |

Reading: `exact` re-verified bit-identical at the perplexity level. The
production `extreme` mix (Q4 attn/norm/embed + Q2 FFN) costs +7.15 ppl on
this tiny model at 3.14 bits/param; forcing the ternary cold tier on costs
+35.41. These are codec stress-test deltas, not LLM quality numbers.

## Measured hot-path optimizations (MEASURED, dev box)

2026-10-20, optimization sweep round 2. All A/B timings measured on the dev
box (7 GB RAM Linux VM, Python layer only); every speedup is **bit-identical
in output** — pinned by exhaustive tests (e.g. the pack_q2 ±200k-ULP sweep).
These are per-op wins; real inference time stays I/O-dominated by design, so
none of these claims multi-x tok/s.

| Hot path | Before | After | Speedup |
|---|---|---|---|
| pack_q2 (midpoint fast path + exact-argmin fixup band) | baseline | — | 8.5x |
| pack_ternary | baseline | — | 10.9x |
| unpack_ternary (LUT decode) | baseline | — | 5.5x |
| pager exact-length reads, RAW pages | baseline | — | 30.8x |
| pager exact-length reads, full 2MiB pages | baseline | — | 1.06x (neutral) |
| pager exact-length reads, small pages | baseline | — | neutral |
| scheduler oracle calls/step (non-exact, per 200 steps) | 600 | 200 | 3x |
| scheduler oracle calls/step (exact, per 200 steps) | 400 | 200 | 2x |
| buffer arena copy primitive | baseline | — | 3.7x |
| page checksum (crc32 chosen over alternatives) | — | 3.19 GB/s | — |

Reading: the RAW-page read path (30.8x) and ternary codec (10.9x pack /
5.5x unpack) were the worst Python-layer offenders; the scheduler now makes
1 oracle call per step instead of 3. Small pages and full 2MiB pages were
already at the I/O floor — reported as neutral, not dropped.

## Measured (tests/)

- `pytest tests/` — **177 passed** (2026-10-21): pager roundtrip + budget enforcement, oracle
  recall ≥ 0.85 at the recall-first operating point, quant error ordering
  (Q4 < Q2 < ternary, all finite), scheduler overlap speedup > 1, converter
  roundtrip of a synthetic GGUF, one-line Engine API, HTTP endpoints, MCP
  stdio smoke test, oracle recipes (13), perplexity harness (7: training
  learns, exact == dense bitwise, extreme degrades gracefully, production
  tier rule, ternary stress, determinism), GGML buffer contract (11:
  masked rows bit-identical vs op-identical reference, unmasked pages never
  fault, eviction loud), prefetch primitives + 9/10 program, optimization
  sweep (behavior-preserving), ease-of-use (pull/Ollama shim/auto-tune),
  simulator sweeps (11: parity with scenario_c, sparsity/K/bandwidth
  monotonicity, trillion specs reproduce docs/TARGET_MODELS.md, all cold
  verdicts DESKTOP-only), warm-start hot-set cache (12: fingerprint-gated,
  per-page CRC32 re-validated, corrupt/stale degrades to cold start),
  production hardening (pull path-traversal guard, stale-prefix 416 restart,
  typed HTTP/MCP/CLI errors), fault-injection + fuzz suites (pager
  OverflowError fixed — file-size bounds enforced as ValueError).

## To measure on real hardware

Checklist for the first real validation run (needs a 4 GB VRAM box + a 100B
model — not available on the dev box):

- [ ] Convert a real 100B GGUF → .lac; record size on disk and convert time.
- [ ] Train oracle predictors on calibration text; record per-layer
      precision/recall on held-out text.
- [ ] Decode 100 tokens, prompts fixed: record tok/s, prefetch hit rate,
      oracle precision/recall, bytes/token, p99 inter-token latency.
- [ ] Same prompts through dense `llama-server` baseline (naive streaming).
- [x] Perplexity on a standard eval set: dense FP16 vs Lacuna tiers
      (quality cost of the Q2/ternary cold pages) — DONE 2026-10-18 at
      tiny-model scale (`benchmarks/perplexity.md`): exact == dense
      bit-identical; extreme Q4/Q2 +7.15 ppl at 3.14 bits/param; ternary
      stress +35.41. Full-scale real-LLM eval still needs a GPU box.
- [ ] 30-minute soak: RSS stays under the RAM hot budget; no swap growth.
