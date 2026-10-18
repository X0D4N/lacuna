"""Tiny numpy transformer LM for the Day-3 perplexity harness.

Purpose: give the "extreme vs exact" quality question a REAL, measured
answer instead of a hand-waved one. This is a ~260K-parameter decoder-only
transformer (RMSNorm + MHA + SwiGLU), trained from scratch on a deterministic
synthetic English corpus. It is NOT a real LLM and its absolute perplexity
does not transfer to 100B models — what it measures honestly is the
*relative* quality cost of Lacuna's precision tiers (Q4/Q2/ternary codecs
from ``lacuna/quant.py``, assigned by the production rule in
``lacuna/convert.py::_tier_for``) on a real trained forward pass.

Stdlib + numpy only, matching the repo rule.
"""

from __future__ import annotations

import json
import math

import numpy as np

CONFIG = {
    "d_model": 96,
    "n_head": 4,
    "n_layer": 2,
    "d_ff": 192,
    "vocab": 384,
    "seq": 64,
    "eps": 1e-5,
}

EPS = 1e-5


# ------------------------------------------------------------- forward ---


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def _rmsnorm_fwd(x: np.ndarray, g: np.ndarray):
    var = (x * x).mean(axis=-1, keepdims=True)
    r = 1.0 / np.sqrt(var + EPS)
    xn = x * r
    return xn * g, (xn, r)


def _rmsnorm_bwd(dy: np.ndarray, cache, g: np.ndarray):
    xn, r = cache
    dxn = dy * g
    # project out the radial component, then undo the 1/sqrt(var) scaling
    proj = (dxn * xn).mean(axis=-1, keepdims=True)
    dx = (dxn - xn * proj) * r
    dg = (dy * xn).sum(axis=(0, 1))
    return dx, dg


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def _silu_bwd(dy: np.ndarray, x: np.ndarray) -> np.ndarray:
    s = 1.0 / (1.0 + np.exp(-x))
    return dy * (s + x * s * (1.0 - s))


class TinyLM:
    """Decoder-only transformer. Parameter names mirror production tensor
    names (blk.{i}.attn.wq etc.) so ``convert._tier_for`` tiers them the way
    a real conversion would."""

    def __init__(self, cfg: dict | None = None, seed: int = 7):
        self.cfg = dict(CONFIG if cfg is None else cfg)
        d, nh, nl, dff, V = (
            self.cfg["d_model"],
            self.cfg["n_head"],
            self.cfg["n_layer"],
            self.cfg["d_ff"],
            self.cfg["vocab"],
        )
        rng = np.random.default_rng(seed)
        s = 1.0 / math.sqrt(d)
        P = self.params = {}
        P["tok_embeddings.weight"] = (rng.standard_normal((V, d)) * s).astype(np.float32)
        for i in range(nl):
            P[f"blk.{i}.attn.wq"] = (rng.standard_normal((d, d)) * s).astype(np.float32)
            P[f"blk.{i}.attn.wk"] = (rng.standard_normal((d, d)) * s).astype(np.float32)
            P[f"blk.{i}.attn.wv"] = (rng.standard_normal((d, d)) * s).astype(np.float32)
            P[f"blk.{i}.attn.wo"] = (rng.standard_normal((d, d)) * s).astype(np.float32)
            P[f"blk.{i}.attn_norm.weight"] = np.ones(d, dtype=np.float32)
            P[f"blk.{i}.ffn.w1"] = (rng.standard_normal((d, dff)) * s).astype(np.float32)
            P[f"blk.{i}.ffn.w3"] = (rng.standard_normal((d, dff)) * s).astype(np.float32)
            P[f"blk.{i}.ffn.w2"] = (rng.standard_normal((dff, d)) * s).astype(np.float32)
            P[f"blk.{i}.ffn_norm.weight"] = np.ones(d, dtype=np.float32)
        P["output_norm.weight"] = np.ones(d, dtype=np.float32)
        P["output.weight"] = (rng.standard_normal((V, d)) * s).astype(np.float32)
        self.hd = d // nh

    # -- forward ----------------------------------------------------

    def forward(self, ids: np.ndarray, cache: dict | None = None):
        """ids: (B, T) int. Returns (loss, nll_sum, n_tokens, probs_cache)."""
        P, d, nh, nl, hd = self.params, self.cfg["d_model"], self.cfg["n_head"], self.cfg["n_layer"], self.hd
        B, T = ids.shape
        x = P["tok_embeddings.weight"][ids]  # (B, T, d)
        cs = cache  # write into the caller's dict when provided
        if cs is not None:
            cs["ids"] = ids
        for i in range(nl):
            h, c = _rmsnorm_fwd(x, P[f"blk.{i}.attn_norm.weight"])
            if cs is not None:
                cs[f"n1.{i}"] = (c, x.shape)
            q = (h @ P[f"blk.{i}.attn.wq"]).reshape(B, T, nh, hd)
            k = (h @ P[f"blk.{i}.attn.wk"]).reshape(B, T, nh, hd)
            v = (h @ P[f"blk.{i}.attn.wv"]).reshape(B, T, nh, hd)
            scores = np.einsum("bthd,bshd->bhts", q, k) / math.sqrt(hd)
            mask = np.triu(np.ones((T, T), dtype=bool), 1)
            scores = np.where(mask[None, None], -1e9, scores)
            if cs is not None:
                cs[f"attn.{i}"] = (q, k, v, scores)
            attn = _softmax(scores)
            ctx = np.einsum("bhts,bshd->bthd", attn, v).reshape(B, T, d)
            a = ctx @ P[f"blk.{i}.attn.wo"]
            x = x + a
            if cs is not None:
                cs[f"attn_out.{i}"] = (attn, ctx, a, h)

            h2, c2 = _rmsnorm_fwd(x, P[f"blk.{i}.ffn_norm.weight"])
            if cs is not None:
                cs[f"n2.{i}"] = (c2, x.shape)
            g = h2 @ P[f"blk.{i}.ffn.w1"]
            u = h2 @ P[f"blk.{i}.ffn.w3"]
            gu = _silu(g) * u
            f = gu @ P[f"blk.{i}.ffn.w2"]
            x = x + f
            if cs is not None:
                cs[f"ffn.{i}"] = (h2, g, u, gu, f)

        h3, c3 = _rmsnorm_fwd(x, P["output_norm.weight"])
        if cs is not None:
            cs["n3"] = (c3, x.shape)
        logits = h3 @ P["output.weight"].T  # (B, T, V)
        # causal LM loss: predict ids[:, 1:] from logits[:, :-1]
        lg = logits[:, :-1, :].reshape(-1, self.cfg["vocab"])
        tgt = ids[:, 1:].reshape(-1)
        lg = lg - lg.max(axis=-1, keepdims=True)
        lse = np.log(np.exp(lg).sum(axis=-1))
        nll = lse - lg[np.arange(lg.shape[0]), tgt]
        loss = float(nll.mean())
        if cs is not None:
            probs = np.exp(lg - lse[:, None])
            cs["ce"] = (probs, tgt, lg.shape)
        return loss, float(nll.sum()), int(nll.size)

    # -- backward ---------------------------------------------------

    def backward(self, cache: dict) -> dict:
        P, d, nh, nl, hd = self.params, self.cfg["d_model"], self.cfg["n_head"], self.cfg["n_layer"], self.hd
        G = {k: np.zeros_like(v) for k, v in P.items()}
        probs, tgt, lgs = cache["ce"]
        N = probs.shape[0]
        dlg = probs.copy()
        dlg[np.arange(N), tgt] -= 1.0
        dlg /= N
        # recover (B, T) from the attention cache; the CE loss trims the
        # last position, so gradient tensors are (B, T-1, ...)
        Tm1 = None
        for key in cache:
            if key.startswith("attn."):
                q = cache[key][0]
                B, Tm1 = q.shape[0], q.shape[1] - 1
                break
        dlg = dlg.reshape(B, Tm1, -1)
        # d(logits) for positions 0..T-2; pad a zero frame at T-1
        dh3_full = np.zeros((B, Tm1 + 1, d), dtype=np.float32)
        dh3 = dlg @ P["output.weight"]  # (B, T-1, d)
        G["output.weight"] += np.einsum("btv,btd->vd", dlg, self._h3(cache, Tm1))
        dh3_full[:, :-1, :] = dh3
        dx, dgn = _rmsnorm_bwd(dh3_full, cache["n3"][0], P["output_norm.weight"])
        G["output_norm.weight"] += dgn

        for i in reversed(range(nl)):
            # FFN block
            h2, g, u, gu, f = cache[f"ffn.{i}"]
            # f = gu @ W2 ; dx here is dL/df
            G[f"blk.{i}.ffn.w2"] += gu.reshape(-1, gu.shape[-1]).T @ dx.reshape(-1, d)
            dgu = dx @ P[f"blk.{i}.ffn.w2"].T
            dg = _silu_bwd(dgu * u, g)
            du = dgu * _silu(g)
            G[f"blk.{i}.ffn.w1"] += h2.reshape(-1, d).T @ dg.reshape(-1, dg.shape[-1])
            G[f"blk.{i}.ffn.w3"] += h2.reshape(-1, d).T @ du.reshape(-1, du.shape[-1])
            dh2 = dg @ P[f"blk.{i}.ffn.w1"].T + du @ P[f"blk.{i}.ffn.w3"].T
            dx_res = dx
            dx, dg2 = _rmsnorm_bwd(dh2 + dx_res, cache[f"n2.{i}"][0], P[f"blk.{i}.ffn_norm.weight"])
            G[f"blk.{i}.ffn_norm.weight"] += dg2

            # attention block
            attn, ctx, a, h = cache[f"attn_out.{i}"]
            dctx = (dx @ P[f"blk.{i}.attn.wo"].T).reshape(B, Tm1 + 1, nh, hd)
            G[f"blk.{i}.attn.wo"] += ctx.reshape(-1, d).T @ dx.reshape(-1, d)
            q, k, v, scores = cache[f"attn.{i}"]
            dattn = np.einsum("bthd,bshd->bhts", dctx, v)
            # softmax backward
            ds = attn * (dattn - (dattn * attn).sum(axis=-1, keepdims=True))
            T = Tm1 + 1
            mask = np.triu(np.ones((T, T), dtype=bool), 1)
            ds = np.where(mask[None, None], 0.0, ds)
            dq = np.einsum("bhts,bshd->bthd", ds, k) / math.sqrt(hd)
            dk = np.einsum("bhts,bthd->bshd", ds, q) / math.sqrt(hd)
            dv = np.einsum("bhts,bthd->bshd", attn, dctx)
            dq = dq.reshape(B, T, d)
            dk = dk.reshape(B, T, d)
            dv = dv.reshape(B, T, d)
            G[f"blk.{i}.attn.wq"] += h.reshape(-1, d).T @ dq.reshape(-1, d)
            G[f"blk.{i}.attn.wk"] += h.reshape(-1, d).T @ dk.reshape(-1, d)
            G[f"blk.{i}.attn.wv"] += h.reshape(-1, d).T @ dv.reshape(-1, d)
            dh = dq @ P[f"blk.{i}.attn.wq"].T + dk @ P[f"blk.{i}.attn.wk"].T + dv @ P[f"blk.{i}.attn.wv"].T
            dx_res2 = dx
            dx, dg1 = _rmsnorm_bwd(dh + dx_res2, cache[f"n1.{i}"][0], P[f"blk.{i}.attn_norm.weight"])
            G[f"blk.{i}.attn_norm.weight"] += dg1

        # token embedding: scatter-add the residual-stream gradient
        np.add.at(G["tok_embeddings.weight"], cache["ids"], dx)

        # embedding grad
        return G

    def _h3(self, cache: dict, Tm1: int):
        # recover h3 (post-norm hidden) for the output-weight grad: recompute
        # cheaply from n3 cache (xn stored there)
        c3, _shape = cache["n3"]
        xn, _r = c3
        g = self.params["output_norm.weight"]
        return (xn * g)[:, :-1, :]

    # -- training ---------------------------------------------------

    def train_step(self, ids: np.ndarray, opt: dict, lr: float):
        cache: dict = {}
        loss, _, _ = self.forward(ids, cache)
        grads = self.backward(cache)
        # Adam
        m, v = opt["m"], opt["v"]
        b1, b2, eps, t = 0.9, 0.999, 1e-8, opt["t"] + 1
        opt["t"] = t
        for k in self.params:
            g = grads[k]
            m[k] = b1 * m[k] + (1 - b1) * g
            v[k] = b2 * v[k] + (1 - b2) * g * g
            mh = m[k] / (1 - b1**t)
            vh = v[k] / (1 - b2**t)
            self.params[k] -= lr * mh / (np.sqrt(vh) + eps)
        return loss

    def new_optimizer(self):
        return {
            "m": {k: np.zeros_like(v) for k, v in self.params.items()},
            "v": {k: np.zeros_like(v) for k, v in self.params.items()},
            "t": 0,
        }

    # -- io ----------------------------------------------------------

    def save(self, path: str):
        np.savez(
            path,
            **{k: v for k, v in self.params.items()},
            _config_json=np.array(json.dumps(self.cfg)),
        )

    @classmethod
    def load(cls, path: str, cfg: dict | None = None):
        z = np.load(path, allow_pickle=True)
        cfg = json.loads(str(z["_config_json"])) if cfg is None else cfg
        m = cls(cfg, seed=0)
        for k in m.params:
            m.params[k] = z[k].astype(np.float32)
        return m

    def n_params(self) -> int:
        return int(sum(v.size for v in self.params.values()))
