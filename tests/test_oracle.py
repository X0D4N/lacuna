"""Oracle tests: learns a 90%-sparse synthetic activation pattern."""

import numpy as np

from lacuna.oracle import Oracle


def _synthetic(n=400, hidden=16, units=64, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, hidden))
    W = rng.standard_normal((hidden, units))
    logits = X @ W
    thresh = np.quantile(logits, 0.90, axis=0)  # ~10% active
    Y = (logits > thresh).astype(np.float64)
    return X, Y


def test_trains_to_high_recall(tmp_path):
    # Recall-first operating point (threshold=0.2): over-fetching a few extra
    # pages is cheap, a prefetch miss is expensive. See Oracle.metrics.
    X, Y = _synthetic()
    o = Oracle(n_layers=1, hidden_dim=16, n_units=64, mlp_hid=64)
    o.train(0, X[:300], Y[:300], epochs=60, lr=0.3, batch=300)
    m = o.metrics(0, X[300:], Y[300:], threshold=0.2)
    assert m["recall"] >= 0.85, m
    assert m["predicted_density"] < 0.20, m  # still sparse: prefetch stays cheap


def test_union_and_save_load(tmp_path):
    X, Y = _synthetic(n=200)
    o = Oracle(n_layers=2, hidden_dim=16, n_units=64, mlp_hid=32)
    o.train(1, X, Y, epochs=10, lr=0.5, batch=64)
    u = o.predict_union(1, [X[0], X[1], X[2]])
    assert u.dtype == bool and u.shape == (64,)
    assert u.sum() >= o.predict_mask(1, X[0]).sum()  # union is a superset
    p = str(tmp_path / "o.npz")
    o.save(p)
    o2 = Oracle.load(p)
    assert o2.metrics(1, X, Y) == o.metrics(1, X, Y)


def test_predict_union_batched_matches_loop():
    # The batched predict_union must be bit-identical to the per-token loop:
    # same weights, same sigmoid, same threshold; OR is order-independent.
    X, _ = _synthetic(n=50)
    o = Oracle(n_layers=1, hidden_dim=16, n_units=64, mlp_hid=32, seed=3)
    hiddens = [X[i] for i in range(8)]
    for th in (0.2, 0.5, 0.9):
        union = o.predict_union(0, hiddens, threshold=th)
        ref = np.zeros(64, dtype=bool)
        for h in hiddens:
            ref |= o.predict_mask(0, h, th)
        np.testing.assert_array_equal(union, ref)
    # empty window -> all-False, same as the old loop version
    np.testing.assert_array_equal(
        o.predict_union(0, [], threshold=0.2), np.zeros(64, dtype=bool)
    )
