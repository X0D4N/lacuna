"""Simulator tests: scenarios ordered, all values finite and positive."""

import math

from simulator.simulate import SCENARIOS, markdown_table, run_all


def test_all_scenarios_finite_positive():
    for key, (_label, fn) in SCENARIOS.items():
        r = fn()
        assert math.isfinite(r["tok_per_s"]) and r["tok_per_s"] > 0, key
        assert math.isfinite(r["p99_latency_s"]) and r["p99_latency_s"] > 0, key
        assert math.isfinite(r["bytes_per_token_gb"]) and r["bytes_per_token_gb"] > 0, key


def test_lacuna_stack_beats_naive():
    res = run_all()
    a = res["a"][1]["tok_per_s"]
    c = res["c"][1]["tok_per_s"]
    assert c > a, (a, c)


def test_moe_fastest_dense_usable():
    res = run_all()
    assert res["d"][1]["tok_per_s"] > res["c"][1]["tok_per_s"]


def test_markdown_table_shape():
    t = markdown_table()
    assert t.count("| (a)") == 1 and "(d)" in t
