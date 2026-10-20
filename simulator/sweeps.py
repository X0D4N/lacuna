"""Parameter sweeps over Lacuna's bandwidth-bound decode model.

ALL NUMBERS IN THIS MODULE ARE SIMULATED — same stated assumptions as
simulator/simulate.py (decode is memory-bandwidth bound; overlap hides fetch
behind compute; p99 = mean x 1.15), swept over:

  - oracle sparsity (active fraction of params fetched per token)
  - speculation depth K (draft acceptance 0.85, EAGLE-class)
  - NVMe bandwidth variants (1.75 / 3.5 / 7.0 GB/s)
  - the 10-model trillion-tier target list from docs/TARGET_MODELS.md

Model specs (total/active params) are sourced from docs/TARGET_MODELS.md,
not invented. Cold per-token weight movement = active_params x 0.5 bytes
(Q4), exactly the honest rule in that doc. Tier verdicts reproduce the
doc's cold-bound verdicts; the MID steady-state column shows the 95%
hot-set-residency target Fahad set 2026-10-19 (an aspiration the
shadow-scorer must earn on real weights, not a claim).

CLI: ``python -m simulator.sweeps`` prints markdown tables.
"""

from __future__ import annotations

import math
import sys

from simulator.simulate import EFFECTIVE_TFLOPS, NVME_BW_GBS, TAIL_FACTOR, _tok_s

# ---------------------------------------------------------------------------
# Core parameterized model (scenario_c generalized; parity-tested in
# tests/test_sweeps.py against simulator.simulate.scenario_c).
# ---------------------------------------------------------------------------

SPARSITY_POINTS = (0.80, 0.85, 0.90, 0.95, 0.97)  # oracle sparsity levels
K_POINTS = (1, 2, 3, 4, 5, 6, 8)  # speculation depths
BANDWIDTH_POINTS = (1.75, 3.5, 7.0)  # NVMe GB/s: slow / target / fast Gen4-5
ACCEPTANCE = 0.85  # draft acceptance, EAGLE-class (stated, same as scenario_c)
ORACLE_MISS = 0.05  # 5% of active pages still fault on demand (stated)


def union_factor(k: int) -> float:
    """Union-mask growth vs speculation depth K (stated assumption).

    Sublinear in accepted tokens: each extra draft depth adds 5% to the
    union mask. At K=4 this is exactly 1.2x, matching scenario_c. The true
    growth curve must be measured by shadow scoring on real weights
    (Day 5a; blocked on a GPU box) — until then this is a stated guess.
    """
    return 1.0 + 0.05 * k


def expected_accepted(k: int, acceptance: float = ACCEPTANCE) -> float:
    """Expected accepted tokens per verify: sum(acc^i for i in 0..K-1)."""
    return sum(acceptance**i for i in range(k))


def lacuna_stack(
    params: float,
    active_frac: float,
    k: int = 4,
    acceptance: float = ACCEPTANCE,
    oracle_miss: float = ORACLE_MISS,
) -> dict:
    """Full Lacuna stack (oracle sparsity + speculative K over union mask).

    Bytes per accepted token = (params x 0.25 bytes/param [Q2-class] x
    active_frac x (1 + oracle_miss)) x union_factor(k) / expected_accepted(k).
    Compute assumes the active fraction participates in the matmul.
    """
    single = params * 0.25 * active_frac * (1.0 + oracle_miss)
    per_token_bytes = single * union_factor(k) / expected_accepted(k)
    return _tok_s(
        params=params,
        bytes_per_token=per_token_bytes,
        active_params=params * active_frac,
    )


def _tok_s_bw(
    params: float,
    bytes_per_token: float,
    active_params: float | None,
    bw_gbs: float,
) -> dict:
    """Same model as _tok_s but with an explicit bandwidth (GB/s)."""
    active = active_params if active_params is not None else params
    fetch_s = bytes_per_token / (bw_gbs * 1e9)
    compute_s = (2.0 * active) / (EFFECTIVE_TFLOPS * 1e12)
    token_s = compute_s + max(0.0, fetch_s - compute_s)
    return {
        "tok_per_s": 1.0 / token_s if token_s > 0 else 0.0,
        "bytes_per_token_gb": bytes_per_token / 1e9,
        "p99_latency_s": token_s * TAIL_FACTOR,
    }


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------


def sparsity_sweep(params: float = 100e9, k: int = 4) -> list[tuple[float, dict]]:
    """tok/s vs oracle sparsity for a 100B dense model, full Lacuna stack."""
    return [(s, lacuna_stack(params, 1.0 - s, k=k)) for s in SPARSITY_POINTS]


def k_sweep(params: float = 100e9, active_frac: float = 0.10) -> list[tuple[int, dict]]:
    """tok/s vs speculation depth K at 90% sparsity."""
    return [(k, lacuna_stack(params, active_frac, k=k)) for k in K_POINTS]


def bandwidth_sweep(
    params: float = 100e9, active_frac: float = 0.10, k: int = 4
) -> list[tuple[float, dict]]:
    """tok/s vs NVMe bandwidth for the full Lacuna stack on 100B."""
    single = params * 0.25 * active_frac * (1.0 + ORACLE_MISS)
    per_token_bytes = single * union_factor(k) / expected_accepted(k)
    return [
        (
            bw,
            _tok_s_bw(params, per_token_bytes, params * active_frac, bw),
        )
        for bw in BANDWIDTH_POINTS
    ]


# ---------------------------------------------------------------------------
# Trillion-tier target list — specs from docs/TARGET_MODELS.md (sourced there).
# (name, total_B, active_B)
# ---------------------------------------------------------------------------

TRILLION_MODELS: tuple[tuple[str, float, float], ...] = (
    ("Kimi K3", 2800, 104),
    ("Qwen3.8-Max", 2400, 95),
    ("DeepSeek-V4-Pro", 1600, 49),
    ("LongCat-2.0", 1600, 48),
    ("MiMo-V2.5-Pro", 1020, 42),
    ("Yuan3.0 Ultra", 1010, 68.8),
    ("Kimi K2.7 Code", 1000, 32),
    ("Ring-1T", 1000, 50),
    ("Inkling", 975, 41),
    ("Muse Spark Open", 816, 42),
)

Q4_BYTES_PER_PARAM = 0.5  # the honest rule: active x 0.5 bytes
HOT_SET_RESIDENCY = 0.95  # MID steady-state target: 95% of active set resident
COLD_TIER_LINE_GB = 4.0  # per-token movement above this => DESKTOP-only cold


def trillion_tier(bw_gbs: float = NVME_BW_GBS) -> list[dict]:
    """Per-model cold vs MID-steady-state per-token weight movement.

    Cold = active x 0.5 bytes fetched every token (no hot-set). MID steady
    state = 95% of the active set resident in RAM/VRAM hot-set (Fahad's
    2026-10-19 target: all 10 runnable on MID via hot-set + prefetch).
    Tier verdicts below reproduce docs/TARGET_MODELS.md's cold-bound
    verdicts — no exceptions, no new claims.
    """
    rows = []
    for name, total_b, active_b in TRILLION_MODELS:
        cold_gb = active_b * Q4_BYTES_PER_PARAM
        cold = _tok_s_bw(total_b * 1e9, cold_gb * 1e9, active_b * 1e9, bw_gbs)
        steady_gb = cold_gb * (1.0 - HOT_SET_RESIDENCY)
        steady = _tok_s_bw(total_b * 1e9, steady_gb * 1e9, active_b * 1e9, bw_gbs)
        verdict = "DESKTOP only" if cold_gb > COLD_TIER_LINE_GB else "MID possible"
        rows.append(
            {
                "model": name,
                "total_b": total_b,
                "active_b": active_b,
                "cold_gb_per_token": cold_gb,
                "cold_tok_per_s": cold["tok_per_s"],
                "mid_steady_gb_per_token": steady_gb,
                "mid_steady_tok_per_s": steady["tok_per_s"],
                "cold_verdict": verdict,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Markdown rendering + CLI
# ---------------------------------------------------------------------------


def _row(label: str, r: dict) -> str:
    return (
        f"| {label} | {r['tok_per_s']:.2f} "
        f"| {r['bytes_per_token_gb']:.2f} | {r['p99_latency_s']:.2f} s |"
    )


def markdown_sweeps() -> str:
    out = ["| Oracle sparsity | tok/s | GB/token | p99 latency |",
           "|---|---|---|---|"]
    for s, r in sparsity_sweep():
        out.append(_row(f"{s:.0%}", r))
    out += ["", "| Spec depth K | tok/s | GB/token | p99 latency |",
            "|---|---|---|---|"]
    for k, r in k_sweep():
        out.append(_row(f"K={k}", r))
    out += ["", "| NVMe bandwidth | tok/s | GB/token | p99 latency |",
            "|---|---|---|---|"]
    for bw, r in bandwidth_sweep():
        out.append(_row(f"{bw:.2f} GB/s", r))
    return "\n".join(out)


def markdown_trillion() -> str:
    rows = ["| Model | Active/token | Cold GB/token | Cold tok/s | "
            "MID steady GB/token | MID steady tok/s | Cold verdict |",
            "|---|---|---|---|---|---|---|"]
    for r in trillion_tier():
        rows.append(
            f"| {r['model']} | {r['active_b']:g}B | {r['cold_gb_per_token']:.1f} "
            f"| {r['cold_tok_per_s']:.2f} | {r['mid_steady_gb_per_token']:.2f} "
            f"| {r['mid_steady_tok_per_s']:.1f} | {r['cold_verdict']} |"
        )
    return "\n".join(rows)


def main(argv=None) -> int:
    print("# Lacuna simulator sweeps — ALL NUMBERS SIMULATED")
    print("# Same stated assumptions as simulator/simulate.py; "
          "see module docstring.\n")
    print("## Sparsity / K / bandwidth sweeps (100B, full Lacuna stack)\n")
    print(markdown_sweeps())
    print("\n## Trillion-tier target list: per-token weight movement\n")
    print(markdown_trillion())
    return 0


if __name__ == "__main__":
    sys.exit(main())
