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

## Measured (tests/)

- `pytest tests/` — 59 passed: pager roundtrip + budget enforcement, oracle
  recall ≥ 0.85 at the recall-first operating point, quant error ordering
  (Q4 < Q2 < ternary, all finite), scheduler overlap speedup > 1, converter
  roundtrip of a synthetic GGUF, one-line Engine API, HTTP endpoints, MCP
  stdio smoke test, oracle recipes (13), perplexity harness (7: training
  learns, exact == dense bitwise, extreme degrades gracefully, production
  tier rule, ternary stress, determinism).

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
