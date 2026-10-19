"""Tests for lacuna/prefetch.py -- the 9/10 program's zero-training primitives.

ALL data in this file is SYNTHETIC (hand-built expert cycles, seeded RNG).
Nothing here touches real model weights; the point is to prove the
mechanisms (determinism, beating random, online adaptation, scorer math)
before they meet real weights via ShadowScorer.
"""

import random

import pytest

from lacuna.prefetch import CacheExpertPredictor, NgramPrefetcher, ShadowScorer


# ---------------------------------------------------------------------------
# (a) N-gram prefetch: deterministic and bounded (synthetic token ids)
# ---------------------------------------------------------------------------

def test_ngram_prefetch_deterministic_and_bounded_synthetic():
    table_size = 1_000_003
    token_ids = [11, 22, 33, 44, 55]  # synthetic
    a = NgramPrefetcher(table_size=table_size, sizes=(2, 3), salt=7)
    b = NgramPrefetcher(table_size=table_size, sizes=(2, 3), salt=7)

    rows_a = a.rows_for(token_ids)
    rows_b = b.rows_for(token_ids)

    # Deterministic across instances and calls.
    assert rows_a == rows_b
    assert a.rows_for(token_ids) == rows_a
    # Bounded: every row is a valid table index.
    assert rows_a and all(0 <= r < table_size for r in rows_a)
    # 4 bigrams + 3 trigrams, modulo possible hash collisions.
    assert len(rows_a) <= 7
    # Salt changes the mapping.
    assert NgramPrefetcher(table_size, salt=8).rows_for(token_ids) != rows_a


def test_ngram_prefetch_edge_cases_synthetic():
    p = NgramPrefetcher(table_size=64, sizes=(2, 3))
    assert p.rows_for([]) == set()          # empty input
    assert p.rows_for([5]) == set()         # no complete n-gram
    assert p.rows_for([5, 6]) == {r for r in p.rows_for([5, 6]) if 0 <= r < 64}
    with pytest.raises(ValueError):
        p.rows_for([-1, 2, 3])              # negative token id rejected


# ---------------------------------------------------------------------------
# (b) Predictor beats random on a synthetic repeating expert pattern
# ---------------------------------------------------------------------------

def _synthetic_cycle(step, sets):
    return sets[step % len(sets)]


def test_expert_predictor_beats_random_on_repeating_pattern_synthetic():
    n_experts, k = 64, 4
    # Synthetic repeating workload: 3 rotating expert sets of 4.
    pattern = [{0, 1, 2, 3}, {10, 11, 12, 13}, {20, 21, 22, 23}]
    pred = CacheExpertPredictor(n_experts=n_experts)

    warmup, measured = 60, 120
    recalls = []
    for step in range(warmup + measured):
        actual = _synthetic_cycle(step, pattern)
        if step >= warmup:
            guessed = set(pred.predict(k))
            recalls.append(len(guessed & actual) / len(actual))
        pred.observe(actual)

    measured_recall = sum(recalls) / len(recalls)

    # Random baseline: analytic (k/n) and empirical with a fixed seed.
    analytic_random_recall = k / n_experts  # 0.0625
    rng = random.Random(1234)
    rand_recalls = []
    for step in range(measured):
        actual = _synthetic_cycle(warmup + step, pattern)
        guessed = set(rng.sample(range(n_experts), k))
        rand_recalls.append(len(guessed & actual) / len(actual))
    empirical_random_recall = sum(rand_recalls) / len(rand_recalls)

    baseline = max(analytic_random_recall, empirical_random_recall)
    assert measured_recall >= 2 * baseline, (
        f"predictor recall {measured_recall:.3f} not >= 2x random {baseline:.3f}"
    )
    # On a perfectly repeating pattern the predictor should be near-perfect.
    assert measured_recall >= 0.9


# ---------------------------------------------------------------------------
# (c) Shadow scorer math on a hand-computed example
# ---------------------------------------------------------------------------

def test_shadow_scorer_hand_computed_example():
    s = ShadowScorer()

    # Step 1: pred {1,2,3}, actual {2,3,4} -> overlap 2, miss {4}.
    st1 = s.observe_step({1, 2, 3}, {2, 3, 4})
    assert st1["hit"] == 0.0
    assert st1["recall"] == pytest.approx(2 / 3)
    assert st1["precision"] == pytest.approx(2 / 3)
    assert st1["missed_pages"] == 1.0

    # Step 2: pred {5,6}, actual {5} -> overlap 1, no miss.
    st2 = s.observe_step({5, 6}, {5})
    assert st2["hit"] == 1.0
    assert st2["recall"] == pytest.approx(1.0)
    assert st2["precision"] == pytest.approx(0.5)
    assert st2["missed_pages"] == 0.0

    r = s.report()
    # Hand-computed aggregates:
    #   hit_rate      = 1/2 = 0.5
    #   recall        = (2+1)/(3+1) = 0.75   (micro over pages)
    #   precision     = (2+1)/(3+2) = 0.6
    #   fault steps   = 1/2 = 0.5, fault pages = 1
    #   overfetch     = 5/4 = 1.25
    assert r["steps"] == 2.0
    assert r["hit_rate"] == pytest.approx(0.5)
    assert r["recall"] == pytest.approx(0.75)
    assert r["precision"] == pytest.approx(0.6)
    assert r["fault_step_rate"] == pytest.approx(0.5)
    assert r["total_fault_pages"] == 1.0
    assert r["overfetch_ratio"] == pytest.approx(1.25)


def test_shadow_scorer_does_not_mutate_inputs():
    s = ShadowScorer()
    p, a = {1, 2}, {2, 3}
    s.observe_step(p, a)
    assert p == {1, 2} and a == {2, 3}  # zero effect on caller data


# ---------------------------------------------------------------------------
# (d) Predictor adapts when the synthetic pattern shifts mid-sequence
# ---------------------------------------------------------------------------

def test_predictor_adapts_to_pattern_shift_synthetic():
    n_experts, k = 64, 4
    # Synthetic workload: pattern P1 for 150 steps, then abrupt shift to P2.
    p1 = [{0, 1}, {8, 9}, {16, 17}]
    p2 = [{40, 41}, {48, 49}, {56, 57}]
    pred = CacheExpertPredictor(n_experts=n_experts)

    early_recalls, late_recalls = [], []
    for step in range(300):
        pattern = p1 if step < 150 else p2
        actual = _synthetic_cycle(step, pattern)
        guessed = set(pred.predict(k))
        r = len(guessed & actual) / len(actual)
        if 150 <= step < 165:
            early_recalls.append(r)   # just after the shift
        elif step >= 250:
            late_recalls.append(r)    # long after the shift
        pred.observe(actual)

    early = sum(early_recalls) / len(early_recalls)
    late = sum(late_recalls) / len(late_recalls)
    random_baseline = k / n_experts  # 0.0625

    # Online adaptation: recovers to near-perfect on the new pattern with no
    # retraining, far above random, and better than right after the shift.
    assert late >= 0.6, f"late recall {late:.3f} too low after pattern shift"
    assert late >= 2 * random_baseline
    assert late > early, "predictor did not improve after the shift"
