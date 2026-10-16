"""Discrete-event simulator of the target hardware.

Target: RTX 2050-class GPU (4 GB VRAM), 8 GB system RAM, NVMe 3.5 GB/s seq.

Model (stated assumptions — every number below is SIMULATED, not measured):
- Decode is memory-bandwidth bound: time/token ~= bytes_fetched / bandwidth.
- Compute time/token = 2 * active_params FLOPs / effective_flops, with
  effective sparse-matmul throughput = 8 TFLOPS on the RTX 2050 (stated; the
  2050's dense FP16 peak is ~20+ TFLOPS, sparse gather/scatter is far lower).
- Prefetch overlap: with the union-mask lookahead, fetch for token t+1 hides
  behind compute of token t; uncovered remainder = max(0, fetch - compute).
- Sparsity: 90% of FFN neurons inactive per token (PowerInfer-class, ReLU/ReGLU
  models; see docs/PRIOR_ART.md). Oracle recall 0.95 -> 5% of active pages
  still fault on demand.
- Speculation: K=4, draft acceptance 0.85 (EAGLE-class), union mask = 1.2x the
  single-token mask (sublinear growth). Expected accepted tokens per verify:
  sum(acc^i for i in 0..K-1).
- MoE 100B: 20B active params @2-bit = 5 GB; 95% of the active set is resident
  in the 6 GB RAM hot-set, 5% faults from NVMe per token.
- p99 latency = mean * 1.15 (stated tail assumption: prefetch stalls / OS jitter).

Scenarios:
  (a) dense 100B @2-bit, naive streaming (no sparsity, no speculation)
  (b) + 90% oracle sparsity
  (c) + speculative K=4 verify over the union mask   <- Lacuna full stack
  (d) MoE 100B (20B active) @2-bit with hot-set residency

CLI: ``python -m simulator.simulate`` prints a markdown table.
"""

from __future__ import annotations

import sys

NVME_BW_GBS = 3.5
EFFECTIVE_TFLOPS = 8.0
TAIL_FACTOR = 1.15


def _tok_s(params: float, bytes_per_token: float, active_params: float | None = None) -> dict:
    active = active_params if active_params is not None else params
    fetch_s = bytes_per_token / (NVME_BW_GBS * 1e9)
    compute_s = (2.0 * active) / (EFFECTIVE_TFLOPS * 1e12)
    # overlapped: fetch hides behind compute of the previous token
    token_s = compute_s + max(0.0, fetch_s - compute_s)
    mean_lat = token_s
    return {
        "tok_per_s": 1.0 / token_s if token_s > 0 else 0.0,
        "bytes_per_token_gb": bytes_per_token / 1e9,
        "p99_latency_s": mean_lat * TAIL_FACTOR,
        "fetch_s": fetch_s,
        "compute_s": compute_s,
    }


def scenario_a() -> dict:
    """Dense 100B @2-bit, naive per-token streaming."""
    return _tok_s(params=100e9, bytes_per_token=100e9 * 0.25)


def scenario_b() -> dict:
    """(a) + 90% oracle sparsity; 5% of active pages fault on demand."""
    active_bytes = 100e9 * 0.25 * 0.10
    fault_extra = active_bytes * 0.05  # oracle misses re-fetched
    return _tok_s(params=100e9, bytes_per_token=active_bytes + fault_extra,
                  active_params=100e9 * 0.10)


def scenario_c() -> dict:
    """(b) + speculative K=4 verification over the union mask (full Lacuna)."""
    acc, k = 0.85, 4
    expected_accepted = sum(acc**i for i in range(k))
    single = 100e9 * 0.25 * 0.10 * 1.05  # incl. fault extra
    union = single * 1.2  # union mask grows sublinearly with K
    per_token = union / expected_accepted
    return _tok_s(params=100e9, bytes_per_token=per_token,
                  active_params=100e9 * 0.10)


def scenario_d() -> dict:
    """MoE 100B (20B active) @2-bit, 95% of active set resident in RAM hot-set."""
    active_bytes = 20e9 * 0.25
    return _tok_s(params=100e9, bytes_per_token=active_bytes * 0.05,
                  active_params=20e9)


SCENARIOS = {
    "a": ("Dense 100B @2-bit, naive streaming", scenario_a),
    "b": ("+ 90% oracle sparsity", scenario_b),
    "c": ("+ speculative K=4 over union mask (Lacuna)", scenario_c),
    "d": ("MoE 100B, 20B active @2-bit, hot-set resident", scenario_d),
}


def run_all() -> dict:
    return {key: (label, fn()) for key, (label, fn) in SCENARIOS.items()}


def markdown_table() -> str:
    rows = ["| Scenario | tok/s | GB/token | p99 latency |",
            "|---|---|---|---|"]
    for key, (label, fn) in SCENARIOS.items():
        r = fn()
        rows.append(
            f"| ({key}) {label} | {r['tok_per_s']:.2f} "
            f"| {r['bytes_per_token_gb']:.2f} | {r['p99_latency_s']:.2f} s |"
        )
    return "\n".join(rows)


def main(argv=None) -> int:
    print("# Lacuna simulator results — ALL NUMBERS SIMULATED")
    print("# Target: RTX 2050 4GB VRAM / 8GB RAM / NVMe 3.5 GB/s.")
    print("# Assumptions: see simulator/simulate.py docstring.\n")
    print(markdown_table())
    return 0


if __name__ == "__main__":
    sys.exit(main())
