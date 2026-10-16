"""Activation oracle (M1) — tiny per-layer MLP predictors, pure numpy.

DejaVu-style: for each transformer layer a small MLP maps the layer's input
hidden state to a boolean mask over that layer's FFN neurons / attention heads
(1 = will activate for this token). Trained once per model offline on
calibration text; at inference time the oracle predicts the UNION of masks over
a K-token lookahead window, which is what the scheduler prefetches.

numpy-only: forward + manual SGD backprop with binary cross-entropy and a
positive-class weight (activations are ~90% sparse, so the loss must not let
the predictor collapse to all-zeros).
"""

from __future__ import annotations

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class _MLP:
    def __init__(self, in_dim: int, hid: int, out_dim: int, rng: np.random.Generator):
        self.W1 = (rng.standard_normal((in_dim, hid)) * np.sqrt(2.0 / in_dim)).astype(np.float64)
        self.b1 = np.zeros(hid, dtype=np.float64)
        self.W2 = (rng.standard_normal((hid, out_dim)) * np.sqrt(1.0 / hid)).astype(np.float64)
        self.b2 = np.zeros(out_dim, dtype=np.float64)

    def forward(self, X: np.ndarray):
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(z1, 0.0)
        z2 = a1 @ self.W2 + self.b2
        return z1, a1, _sigmoid(z2)

    def step(self, X: np.ndarray, Y: np.ndarray, lr: float, pos_weight: float) -> float:
        n = X.shape[0]
        z1, a1, p = self.forward(X)
        eps = 1e-9
        loss = -(pos_weight * Y * np.log(p + eps) + (1 - Y) * np.log(1 - p + eps)).mean()
        dz2 = (p - Y) / n
        dz2 = dz2 * np.where(Y > 0.5, pos_weight, 1.0)
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (z1 > 0)
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0)
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        return float(loss)


class Oracle:
    """Per-layer activation predictors with train / predict / union / metrics."""

    def __init__(self, n_layers: int, hidden_dim: int, n_units: int, mlp_hid: int = 64, seed: int = 0):
        self.n_layers = int(n_layers)
        self.hidden_dim = int(hidden_dim)
        self.n_units = int(n_units)
        rng = np.random.default_rng(seed)
        self._mlps = [_MLP(hidden_dim, mlp_hid, n_units, rng) for _ in range(n_layers)]

    def train(
        self,
        layer: int,
        X: np.ndarray,
        Y: np.ndarray,
        epochs: int = 20,
        lr: float = 0.3,
        batch: int = 64,
        verbose: bool = False,
    ) -> list[float]:
        """X: (n, hidden_dim) hidden states; Y: (n, n_units) binary masks."""
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        mlp = self._mlps[layer]
        pos = Y.sum()
        pos_weight = float((Y.size - pos) / max(pos, 1.0))
        losses = []
        n = X.shape[0]
        rng = np.random.default_rng(1234 + layer)
        for ep in range(epochs):
            perm = rng.permutation(n)
            tot = 0.0
            for s in range(0, n, batch):
                b = perm[s : s + batch]
                tot += mlp.step(X[b], Y[b], lr, pos_weight)
            losses.append(tot / max(1, (n + batch - 1) // batch))
            if verbose:
                print(f"layer {layer} epoch {ep}: loss {losses[-1]:.4f}")
        return losses

    def predict_proba(self, layer: int, h: np.ndarray) -> np.ndarray:
        h = np.asarray(h, dtype=np.float64).reshape(1, -1)
        _, _, p = self._mlps[layer].forward(h)
        return p[0]

    def predict_mask(self, layer: int, h: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return self.predict_proba(layer, h) > threshold

    def predict_union(self, layer: int, hidden_list, threshold: float = 0.5) -> np.ndarray:
        """OR of predicted masks over a K-token lookahead window of hidden states."""
        union = np.zeros(self.n_units, dtype=bool)
        for h in hidden_list:
            union |= self.predict_mask(layer, h, threshold)
        return union

    def metrics(self, layer: int, X: np.ndarray, Y: np.ndarray, threshold: float = 0.5) -> dict:
        # Prefetch oracles operate recall-first: a lower threshold over-fetches
        # slightly (cheap) instead of missing pages (expensive). Pass
        # threshold=0.2 for the recall-optimized operating point.
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=bool)
        _, _, p = self._mlps[layer].forward(X)
        pred = p > threshold
        tp = np.logical_and(pred, Y).sum()
        fp = np.logical_and(pred, ~Y).sum()
        fn = np.logical_and(~pred, Y).sum()
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        return {
            "precision": float(precision),
            "recall": float(recall),
            "predicted_density": float(pred.mean()),
            "true_density": float(Y.mean()),
        }

    def save(self, path: str) -> None:
        d: dict = {
            "n_layers": self.n_layers,
            "hidden_dim": self.hidden_dim,
            "n_units": self.n_units,
        }
        for i, m in enumerate(self._mlps):
            d[f"W1_{i}"] = m.W1
            d[f"b1_{i}"] = m.b1
            d[f"W2_{i}"] = m.W2
            d[f"b2_{i}"] = m.b2
        np.savez(path, **d)

    @classmethod
    def load(cls, path: str) -> "Oracle":
        z = np.load(path, allow_pickle=False)
        n_layers = int(z["n_layers"])
        o = cls(n_layers, int(z["hidden_dim"]), int(z["n_units"]))
        for i in range(n_layers):
            m = o._mlps[i]
            m.W1 = z[f"W1_{i}"]
            m.b1 = z[f"b1_{i}"]
            m.W2 = z[f"W2_{i}"]
            m.b2 = z[f"b2_{i}"]
        return o
