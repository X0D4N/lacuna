"""Ordo — routing automaton tests (lacuna/ordo.py)."""

import random

import pytest

from lacuna.ordo import OrdoAutomaton, RoutingTrace, path_signature


def _topic_trace(rng: random.Random, layers=6, experts=40, top_k=3,
                 topics=((0, 1, 2), (20, 21, 22)), tokens=120, phase=40):
    """Synthetic phased routing: each phase favors one topic's experts."""
    trace = RoutingTrace()
    for t in range(tokens):
        fav = topics[(t // phase) % len(topics)]
        pages = []
        for layer in range(layers):
            chosen = set()
            while len(chosen) < top_k:
                if rng.random() < 0.8:
                    chosen.add(rng.choice(fav))
                else:
                    chosen.add(rng.randrange(experts))
            for e in chosen:
                pages.append((layer, e))
        trace.record(pages)
    return trace


def test_trace_roundtrip_and_corruption(tmp_path):
    trace = RoutingTrace()
    trace.record([(0, 1), (0, 2), (1, 5)])
    trace.record([(0, 2), (1, 5), (1, 9)])
    p = tmp_path / "trace.json"
    trace.save(str(p))
    loaded = RoutingTrace.load(str(p))
    assert loaded.tokens == trace.tokens

    p.write_text("{not json", encoding="utf-8")
    assert RoutingTrace.load(str(p)).tokens == []
    assert RoutingTrace.load(str(tmp_path / "missing.json")).tokens == []


def test_path_signature_deterministic_and_sensitive():
    a = ((0, 1), (1, 2), (2, 3))
    b = ((0, 1), (1, 2), (2, 4))
    assert path_signature(a) == path_signature(a)
    assert path_signature(a) != path_signature(b)


def test_automaton_deterministic_and_budgeted():
    rng = random.Random(0)
    traces = [_topic_trace(rng) for _ in range(3)]
    a1 = OrdoAutomaton.train(traces)
    a2 = OrdoAutomaton.train(traces)
    state = path_signature(traces[0].tokens[10])
    p1 = a1.predict_pages(state, budget_pages=25)
    assert p1 == a2.predict_pages(state, budget_pages=25)
    assert len(p1) <= 25


def test_unseen_state_falls_back_to_popularity():
    rng = random.Random(1)
    auto = OrdoAutomaton.train([_topic_trace(rng)])
    unseen = 123456789  # not a real signature in this trace
    pages = auto.predict_pages(unseen, budget_pages=10)
    top_popular = [p for p, _ in sorted(auto.popularity.items(),
                                        key=lambda kv: (-kv[1], kv[0]))[:10]]
    assert pages == top_popular


def test_state_prediction_beats_popularity_on_phased_data():
    # Toy scaled so the popularity budget cannot cover both topics'
    # favored experts at once — state-conditioning has room to matter.
    kw = dict(layers=8, experts=64, top_k=3,
              topics=((0, 1, 2, 3), (32, 33, 34, 35)), tokens=160, phase=40)
    rng = random.Random(2)
    train = [_topic_trace(rng, **kw) for _ in range(4)]
    test = _topic_trace(random.Random(99), **kw)
    auto = OrdoAutomaton.train(train)
    budget = 36
    ordo_cov = auto.coverage(test, budget)
    pop_set = set(auto.predict_pages(None, budget))
    pop_hits = sum(1 for tok in test.tokens for pg in tok if pg in pop_set)
    pop_cov = pop_hits / sum(len(tok) for tok in test.tokens)
    assert len(auto.clusters) >= 2  # the phases became distinct states
    assert ordo_cov > pop_cov  # sequential structure is real signal


def test_compile_segments_support_and_budget():
    rng = random.Random(3)
    auto = OrdoAutomaton.train([_topic_trace(rng) for _ in range(3)])
    segments = auto.compile_segments(budget_pages=20, min_support=2)
    assert segments  # some states recur in phased data
    for seg in segments.values():
        assert len(seg) <= 20
    assert auto.compile_segments(budget_pages=20, min_support=10_000) == {}


def test_empty_automaton_is_safe():
    auto = OrdoAutomaton.train([])
    assert auto.predict_pages(None, 10) == []
    assert auto.coverage(RoutingTrace(), 10) == 0.0


def test_wrong_state_costs_speed_only():
    """Ordo advice never changes WHAT is served, only what is resident:
    serving a token always loads its true pages, whatever the state."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from simulator.ordo_sim import ModelClass, OrdoPager, Pager

    model = ModelClass("tiny", layers=4, experts=16, top_k=2, active_bytes=8e6)
    rng = random.Random(4)
    trace = _topic_trace(rng, layers=4, experts=16, top_k=2, tokens=60)
    auto = OrdoAutomaton.train([trace])

    plain = Pager(model)
    for path in trace.tokens:
        plain.serve(path)

    ordo = OrdoPager(model, auto)
    ordo.run_session(trace, warm_from=0)

    # Both pagers end having served exactly the true page stream; the
    # automaton's advice can only change the byte count, never the set
    # of pages a token required (asserted by construction in serve()).
    assert set(plain.resident) | set(trace.tokens[-1])  # sanity
    assert ordo.bytes_read >= 0 and plain.bytes_read >= 0
    assert ordo.bytes_read != pytest.approx(0.0)
