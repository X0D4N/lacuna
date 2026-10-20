"""Sweep tests: model parity with scenario_c, monotonicity, trillion specs.

All assertions are on the deterministic simulator (MEASURED test outcomes
about SIMULATED model behavior — the numbers themselves stay SIMULATED).
"""

import math

from simulator import simulate, sweeps


def _finite_positive(r: dict):
    for key in ("tok_per_s", "bytes_per_token_gb", "p99_latency_s"):
        assert math.isfinite(r[key]) and r[key] > 0, (key, r)


def test_parity_with_scenario_c():
    # lacuna_stack at the scenario_c operating point must reproduce it exactly.
    ref = simulate.scenario_c()
    got = sweeps.lacuna_stack(100e9, 0.10, k=4)
    assert got["tok_per_s"] == ref["tok_per_s"]
    assert got["bytes_per_token_gb"] == ref["bytes_per_token_gb"]
    assert got["p99_latency_s"] == ref["p99_latency_s"]


def test_union_factor_matches_scenario_c_at_k4():
    assert sweeps.union_factor(4) == 1.2


def test_sweeps_finite_positive():
    for _, r in sweeps.sparsity_sweep():
        _finite_positive(r)
    for _, r in sweeps.k_sweep():
        _finite_positive(r)
    for _, r in sweeps.bandwidth_sweep():
        _finite_positive(r)
    for r in sweeps.trillion_tier():
        for key in ("cold_gb_per_token", "cold_tok_per_s",
                    "mid_steady_gb_per_token", "mid_steady_tok_per_s"):
            assert math.isfinite(r[key]) and r[key] > 0, (key, r)


def test_sparsity_monotonic():
    pts = sweeps.sparsity_sweep()
    toks = [r["tok_per_s"] for _, r in pts]
    assert toks == sorted(toks), toks  # more sparsity -> faster
    assert [s for s, _ in pts] == list(sweeps.SPARSITY_POINTS)


def test_k_sweep_improves_with_diminishing_returns():
    pts = dict(sweeps.k_sweep())
    assert pts[8]["tok_per_s"] > pts[1]["tok_per_s"]  # speculation helps
    gain_1_2 = pts[2]["tok_per_s"] - pts[1]["tok_per_s"]
    gain_6_8 = pts[8]["tok_per_s"] - pts[6]["tok_per_s"]
    assert gain_6_8 < gain_1_2  # diminishing returns at stated acceptance


def test_bandwidth_monotonic():
    pts = sweeps.bandwidth_sweep()
    toks = [r["tok_per_s"] for _, r in pts]
    assert toks == sorted(toks), toks
    assert [bw for bw, _ in pts] == list(sweeps.BANDWIDTH_POINTS)


def test_trillion_specs_match_docs_target_models():
    # Cold GB/token must reproduce docs/TARGET_MODELS.md's EST. column
    # (active x 0.5 bytes) — sourced there, not invented here.
    expected = {
        "Kimi K3": 52.0, "Qwen3.8-Max": 47.5, "DeepSeek-V4-Pro": 24.5,
        "LongCat-2.0": 24.0, "MiMo-V2.5-Pro": 21.0, "Yuan3.0 Ultra": 34.4,
        "Kimi K2.7 Code": 16.0, "Ring-1T": 25.0, "Inkling": 20.5,
        "Muse Spark Open": 21.0,
    }
    rows = {r["model"]: r for r in sweeps.trillion_tier()}
    assert set(rows) == set(expected), set(rows) ^ set(expected)
    for name, gb in expected.items():
        assert rows[name]["cold_gb_per_token"] == gb, name


def test_trillion_cold_verdicts_all_desktop_only():
    # The honest rule: every model moves > 4 GB/token cold -> DESKTOP only,
    # reproducing docs/TARGET_MODELS.md. No exceptions.
    for r in sweeps.trillion_tier():
        assert r["cold_verdict"] == "DESKTOP only", r["model"]


def test_trillion_ordering_kimi_k3_slowest_cold():
    rows = sweeps.trillion_tier()
    slowest = min(rows, key=lambda r: r["cold_tok_per_s"])
    assert slowest["model"] == "Kimi K3"  # largest active set: 104B


def test_trillion_mid_steady_beats_cold():
    for r in sweeps.trillion_tier():
        assert r["mid_steady_tok_per_s"] > r["cold_tok_per_s"], r["model"]
        assert math.isclose(r["mid_steady_gb_per_token"],
                            r["cold_gb_per_token"] * (1.0 - sweeps.HOT_SET_RESIDENCY),
                            rel_tol=1e-12)


def test_markdown_shapes():
    t = sweeps.markdown_sweeps()
    assert "Oracle sparsity" in t and "Spec depth K" in t and "NVMe bandwidth" in t
    assert t.count("K=4") == 1
    t2 = sweeps.markdown_trillion()
    assert t2.count("| Kimi K3 |") == 1
    assert "DESKTOP only" in t2
    assert "SIMULATED" not in t2  # labels live in main()/docs, not the table
