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

## What the simulator does NOT model

- Prefill (long-prompt) performance — decode only.
- KV-cache growth with context length.
- Quality/perplexity impact of the Q2/ternary cold tiers (needs real evals).
- PCIe contention, thermal throttling, OS background I/O.

## Reproducing

```bash
python -m simulator.simulate
```

Deterministic: no randomness, no wall-clock dependence.
