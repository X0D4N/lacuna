"""Scheduler tests: overlap beats sequential; live mode runs and counts."""

import time

from lacuna.scheduler import OverlapScheduler, simulate_overlap


def test_sim_overlap_beats_sequential():
    fetch = [0.7] * 8
    compute = [0.05] * 8
    r = simulate_overlap(fetch, compute, [2.5e9] * 8, prefetch_hit_rate=1.0)
    assert r["overlapped_s"] < r["sequential_s"]
    assert r["speedup"] > 1.0
    assert r["tok_per_s"] > 0 and r["p99_latency_s"] > 0
    assert r["bytes_per_token"] == 2.5e9


def test_sim_miss_penalty():
    fetch = [0.7] * 4
    compute = [0.05] * 4
    good = simulate_overlap(fetch, compute, [1e9] * 4, prefetch_hit_rate=1.0)
    bad = simulate_overlap(fetch, compute, [1e9] * 4, prefetch_hit_rate=0.5)
    assert bad["overlapped_s"] > good["overlapped_s"]


def test_live_scheduler_runs():
    pages = {s: [s, s + 100] for s in range(6)}
    resident = set()

    def fetch_fn(pids):
        resident.update(pids)

    def compute_fn(step):
        time.sleep(0.001)
        return step * 2

    sch = OverlapScheduler(lookahead_k=2, workers=2)
    try:
        out = sch.run(6, lambda s: pages[s], fetch_fn, compute_fn,
                      is_resident=lambda p: p in resident)
        assert out == [0, 2, 4, 6, 8, 10]
        assert sch.stats["steps"] == 6
        assert sch.tok_per_s() > 0
    finally:
        sch.close()
