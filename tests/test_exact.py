"""Exactness tests: prediction guides memory, never math.

The guarantee under test: in exact mode, oracle quality affects SPEED ONLY.
Even a deliberately broken (random-mask) oracle must produce token-identical
output to the dense baseline — the only observable difference is faults.
"""

import numpy as np

from lacuna.engine import Engine, LacunaEngine
from lacuna.hardware import detect_hardware
from lacuna.pager import LacWriter, PagedWeightStore
from lacuna.scheduler import OverlapScheduler

# 4 tensors x 64KB, 4KB pages -> 16 pages each = 64 pages total.
_N_TENSORS, _ROWS, _COLS = 4, 256, 64


def _build_file(path, seed=7):
    rng = np.random.default_rng(seed)
    with LacWriter(path, page_size=4096, precision_policy="exact") as w:
        for t in range(_N_TENSORS):
            arr = rng.standard_normal((_ROWS, _COLS)).astype(np.float32)
            w.add_raw_bytes(f"blk.0.t{t}", arr.tobytes())


def test_exact_matches_dense_baseline(tmp_path):
    # Both stores can hold all 64 pages; the exact engine starts cold.
    p = str(tmp_path / "m.lac")
    _build_file(p)
    hw = detect_hardware()
    with PagedWeightStore(p, hot_budget_bytes=10**9) as dense_store:
        dense_store.ensure_resident(list(range(dense_store.n_pages)))
        with PagedWeightStore(p, hot_budget_bytes=10**9) as exact_store:
            dense_eng = LacunaEngine(hw, store=dense_store, exact=True)
            exact_eng = LacunaEngine(hw, store=exact_store, exact=True)
            for prompt in ["hello world", "sparse paging", "exactness guarantee"]:
                a = dense_eng.generate(prompt, max_tokens=12)
                b = exact_eng.generate(prompt, max_tokens=12)
                assert a == b, f"diverged on {prompt!r}"
            assert exact_eng.stats()["exact_guarantee_held"] is True
            assert dense_eng.stats()["exact_guarantee_held"] is True


def test_bad_oracle_only_costs_speed(tmp_path):
    """Deliberately broken oracle (random masks): output must STILL be
    identical — only slower, i.e. strictly more exact_faults than a good
    oracle, with the guarantee held in both cases."""
    p = str(tmp_path / "m.lac")
    _build_file(p)
    hw = detect_hardware()
    rng = np.random.default_rng(0)
    with PagedWeightStore(p, hot_budget_bytes=10**9) as s_good, \
         PagedWeightStore(p, hot_budget_bytes=10**9) as s_bad:
        n_pages = s_good.n_pages
        full = list(range(n_pages))

        def bad_oracle(step):
            # terrible prefetch schedule: a random 10% of pages
            return rng.choice(
                n_pages, size=max(1, n_pages // 10), replace=False
            ).tolist()

        good_eng = LacunaEngine(hw, store=s_good, exact=True,
                                pages_for_step_fn=lambda step: full)
        bad_eng = LacunaEngine(hw, store=s_bad, exact=True,
                               pages_for_step_fn=bad_oracle)
        for prompt in ["hello world", "sparse paging"]:
            assert bad_eng.generate(prompt, max_tokens=12) == good_eng.generate(
                prompt, max_tokens=12
            )
        g, b = good_eng.stats(), bad_eng.stats()
        assert g["exact_guarantee_held"] is True
        assert b["exact_guarantee_held"] is True
        assert b["exact_faults"] > g["exact_faults"], (
            "bad oracle must cost faults (time), never correctness"
        )


def test_guarantee_flag_is_measured_not_assumed(tmp_path):
    """When the hot budget cannot hold the required set, the flag must read
    False — it is measured per step, not a tautology. (A real deployment
    tiles per layer in this regime; see engine docstring.)"""
    p = str(tmp_path / "m.lac")
    _build_file(p)
    hw = detect_hardware()
    with PagedWeightStore(p, hot_budget_bytes=8 * 4096) as s:  # fits ~8/64
        eng = LacunaEngine(hw, store=s, exact=True)
        out = eng.generate("hello", max_tokens=6)
        assert eng.stats()["exact_guarantee_held"] is False
        # ...but the output is still identical to the dense baseline:
        # the guarantee flag reports residency, the math never changed.
        with PagedWeightStore(p, hot_budget_bytes=10**9) as dense:
            dense.ensure_resident(list(range(dense.n_pages)))
            ref = LacunaEngine(hw, store=dense, exact=True).generate(
                "hello", max_tokens=6
            )
        assert out == ref


def test_ensure_resident_is_blocking_and_complete(tmp_path):
    p = str(tmp_path / "m.lac")
    _build_file(p)
    with PagedWeightStore(p, hot_budget_bytes=8 * 4096) as s:
        assert not s.are_resident(range(s.n_pages))
        faults = s.ensure_resident(range(s.n_pages))
        assert faults > 0
        # budget too small to hold everything -> measurably not all resident
        assert s.are_resident(range(s.n_pages)) is False
    with PagedWeightStore(p, hot_budget_bytes=10**9) as s2:
        faults = s2.ensure_resident(range(s2.n_pages))
        assert faults == s2.n_pages  # cold start: every page faulted once
        assert s2.are_resident(range(s2.n_pages)) is True
        # second call: everything already resident -> zero faults
        assert s2.ensure_resident(range(s2.n_pages)) == 0


def test_scheduler_exact_mode(tmp_path):
    """Scheduler-level: compute_fn asserts full residency at call time."""
    p = str(tmp_path / "m.lac")
    _build_file(p)
    with PagedWeightStore(p, hot_budget_bytes=10**9) as s:
        required = list(range(s.n_pages))
        seen = []

        def compute(step):
            assert s.are_resident(required), f"step {step}: page missing at compute"
            seen.append(step)
            return step * 2

        sched = OverlapScheduler(lookahead_k=2)
        try:
            res = sched.run(
                4,
                pages_for_step=lambda step: [0],  # bad prediction: only page 0
                fetch_fn=s.prefetch,
                compute_fn=compute,
                is_resident=s.is_resident,
                exact=True,
                ensure_fn=s.ensure_resident,
                required_pages_for_step=lambda step: required,
            )
        finally:
            sched.close()
        assert res == [0, 2, 4, 6]
        assert seen == [0, 1, 2, 3]
        assert sched.stats["exact_guarantee_held"] is True
        assert sched.stats["exact_faults"] > 0  # bad prefetch -> real faults


def test_scheduler_exact_requires_ensure_fn():
    sched = OverlapScheduler()
    try:
        sched.run(1, lambda s: [], lambda p: None, lambda s: s, exact=True)
    except ValueError:
        pass
    else:
        raise AssertionError("exact=True without ensure_fn must raise")
    finally:
        sched.close()


def test_mock_path_still_works_with_exact():
    """Exact mode with no store (mock-only): must still generate."""
    e = Engine("mock", exact=True)
    try:
        out = e.generate("hello", max_tokens=8)
        assert out.startswith("[mock]")
        # vacuously true: with no pager there is nothing that could be missing
        assert e._engine.exact_guarantee_held is True
    finally:
        e.close()
