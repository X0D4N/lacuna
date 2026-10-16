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

## Measured (tests/)

- `pytest tests/` — 24 passed: pager roundtrip + budget enforcement, oracle
  recall ≥ 0.85 at the recall-first operating point, quant error ordering
  (Q4 < Q2 < ternary, all finite), scheduler overlap speedup > 1, converter
  roundtrip of a synthetic GGUF, one-line Engine API, HTTP endpoints, MCP
  stdio smoke test.

## To measure on real hardware

Checklist for the first real validation run (needs a 4 GB VRAM box + a 100B
model — not available on the dev box):

- [ ] Convert a real 100B GGUF → .lac; record size on disk and convert time.
- [ ] Train oracle predictors on calibration text; record per-layer
      precision/recall on held-out text.
- [ ] Decode 100 tokens, prompts fixed: record tok/s, prefetch hit rate,
      oracle precision/recall, bytes/token, p99 inter-token latency.
- [ ] Same prompts through dense `llama-server` baseline (naive streaming).
- [ ] Perplexity on a standard eval set: dense FP16 vs Lacuna tiers
      (quality cost of the Q2/ternary cold pages).
- [ ] 30-minute soak: RSS stays under the RAM hot budget; no swap growth.
