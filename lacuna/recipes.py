"""Oracle training recipes per model family (DejaVu-style calibration).

Day 2 track of the week-hardening plan. This module turns the M1 activation
oracle (``lacuna/oracle.py``) from a class into a repeatable procedure:

  1. pick a model family preset (``qwen`` / ``llama`` / ``moe``) or build a
     ``FamilySpec`` from your model's own config.json,
  2. capture (X, Y) calibration pairs -- per-layer input hidden states plus
     binary activation masks -- from a real model (procedure specified in
     ``docs/ORACLE_RECIPES.md``; the capture hook itself is not built yet),
     or generate synthetic pairs with :func:`synthetic_calibration` to
     validate the pipeline end to end (labeled SYNTHETIC, never real),
  3. train one oracle per family with :func:`train_family_oracle`,
  4. evaluate at the recall-first operating point with
     :func:`evaluate_dataset` and pass the quality gate with
     :func:`gate_report`.

What the oracle predicts, per family:

  - dense SwiGLU (Qwen2/2.5, Llama-3.x): for each layer, which of the
    ``intermediate`` FFN neurons carry signal -- the binarized
    ``silu(gate) * up`` product. Attention projections stay hot-resident
    (ARCHITECTURE.md M3) and are not predicted.
  - MoE (Mixtral-style): two oracles per layer -- a router oracle predicting
    the top-k expert mask (``n_experts`` units) and an FFN oracle predicting
    per-expert neuron masks (``n_experts * intermediate`` units). The
    prefetch union is the union over the lookahead of
    ``expert_mask (x) neuron_masks``.

stdlib + numpy only. No network, no torch, no transformers.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .oracle import Oracle

# Recall-first operating point: over-fetching a few extra pages is cheap, a
# prefetch miss costs an on-demand fault. See Oracle.metrics.
OPERATING_THRESHOLD = 0.2
# Quality gate: a recipe passes only if every layer clears these.
GATE_MIN_RECALL = 0.85
GATE_MAX_PREDICTED_DENSITY = 0.20
GATE_MIN_UNION_RECALL = 0.90
GATE_MIN_ROUTER_RECALL = 0.85


@dataclass(frozen=True)
class FamilySpec:
    """Architecture facts a recipe needs. Build from your model's config.json."""

    name: str
    kind: str  # "dense" | "moe"
    ffn: str  # activation family; only "swiglu" is supported
    n_layers: int
    hidden_dim: int
    intermediate: int  # FFN width; per-expert width for MoE
    n_heads: int
    n_kv_heads: int
    n_experts: int = 0
    top_k: int = 0
    norm: str = "rmsnorm"
    reference_model: str = ""
    capture_note: str = ""


FAMILIES: dict[str, FamilySpec] = {
    "qwen": FamilySpec(
        name="qwen",
        kind="dense",
        ffn="swiglu",
        n_layers=28,
        hidden_dim=3584,
        intermediate=18944,
        n_heads=28,
        n_kv_heads=4,
        reference_model="Qwen2.5-7B (public config.json; substitute your own)",
        capture_note=(
            "GGUF tensors blk.{i}.ffn_gate / ffn_up / ffn_down. X = FFN-block "
            "input (post-attention RMSNorm output); Y = binarized "
            "silu(gate(x)) * up(x) at per-neuron quantile q=0.90. Qwen2/2.5 "
            "uses SwiGLU + GQA + RMSNorm throughout."
        ),
    ),
    "llama": FamilySpec(
        name="llama",
        kind="dense",
        ffn="swiglu",
        n_layers=32,
        hidden_dim=4096,
        intermediate=14336,
        n_heads=32,
        n_kv_heads=8,
        reference_model="Llama-3.1-8B (public config.json; substitute your own)",
        capture_note=(
            "GGUF tensors blk.{i}.ffn_gate / ffn_up / ffn_down. Same capture "
            "procedure as Qwen (SwiGLU + GQA + RMSNorm); only the dims and "
            "the calibration text (chat template for instruct models) differ."
        ),
    ),
    "moe": FamilySpec(
        name="moe",
        kind="moe",
        ffn="swiglu",
        n_layers=32,
        hidden_dim=4096,
        intermediate=14336,
        n_heads=32,
        n_kv_heads=8,
        n_experts=8,
        top_k=2,
        reference_model="Mixtral-8x7B (public config.json; substitute your own)",
        capture_note=(
            "GGUF tensors blk.{i}.ffn_gate_inp (router), ffn_gate_exps / "
            "ffn_up_exps / ffn_down_exps (experts). X = MoE-block input; "
            "R = top-2 router mask from gate_inp logits; per active expert e, "
            "Y_e = binarized silu(gate_e(x)) * up_e(x) at per-expert-neuron "
            "quantile q=0.90 over routed tokens. Experts are SwiGLU."
        ),
    ),
}


def list_families() -> list[str]:
    return sorted(FAMILIES)


def get_family(name: str) -> FamilySpec:
    try:
        return FAMILIES[name]
    except KeyError:
        raise KeyError(
            f"unknown family {name!r}; choose from {list_families()} or build "
            "one with spec_from_dims()"
        ) from None


def spec_from_dims(
    name: str,
    kind: str,
    n_layers: int,
    hidden_dim: int,
    intermediate: int,
    n_heads: int = 0,
    n_kv_heads: int = 0,
    n_experts: int = 0,
    top_k: int = 0,
) -> FamilySpec:
    """Build a FamilySpec from a real model's config.json values."""
    if kind not in ("dense", "moe"):
        raise ValueError(f"kind must be 'dense' or 'moe', got {kind!r}")
    if kind == "moe" and (n_experts <= 0 or top_k <= 0):
        raise ValueError("MoE specs need n_experts > 0 and top_k > 0")
    if kind == "dense" and (n_experts or top_k):
        raise ValueError("dense specs must not set n_experts/top_k")
    for label, v in (("n_layers", n_layers), ("hidden_dim", hidden_dim),
                     ("intermediate", intermediate)):
        if v <= 0:
            raise ValueError(f"{label} must be > 0, got {v}")
    return FamilySpec(
        name=name, kind=kind, ffn="swiglu", n_layers=n_layers,
        hidden_dim=hidden_dim, intermediate=intermediate, n_heads=n_heads,
        n_kv_heads=n_kv_heads, n_experts=n_experts, top_k=top_k,
        reference_model="custom (from your config.json)",
    )


def oracle_units(spec: FamilySpec) -> int:
    """Number of FFN-oracle output units per layer for this family."""
    if spec.kind == "moe":
        return spec.n_experts * spec.intermediate
    return spec.intermediate


def _silu(z: np.ndarray) -> np.ndarray:
    return z / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _rmsnorm_rows(X: np.ndarray) -> np.ndarray:
    return X / np.sqrt((X ** 2).mean(axis=1, keepdims=True) + 1e-12)


def _walk_hidden_states(rng: np.random.Generator, n_tokens: int,
                        seq_len: int, hidden_dim: int) -> np.ndarray:
    """Autoregressive-drift hidden states: low-rank mean-reverting walk + noise.

    Mimics two properties of real transformer hidden states: (1) token to
    token they evolve smoothly (local correlation, global stationarity), and
    (2) they are highly anisotropic -- most variance lives in a low-dimensional
    subspace (the "outlier dimensions" phenomenon), with the rest near-isotropic
    noise. An isotropic walk would make the synthetic prediction target
    artificially harder than the real one.
    """
    rank = max(8, hidden_dim // 8)
    n_seq = max(1, n_tokens // seq_len)
    Z = np.zeros((n_seq, seq_len, rank))
    Z[:, 0, :] = rng.standard_normal((n_seq, rank))
    for t in range(1, seq_len):
        Z[:, t, :] = 0.9 * Z[:, t - 1, :] + rng.standard_normal((n_seq, rank)) * 0.45
    A = rng.standard_normal((rank, hidden_dim)) / np.sqrt(rank)
    X = Z.reshape(-1, rank) @ A
    X = X + rng.standard_normal(X.shape) * 0.3
    return _rmsnorm_rows(X[: n_seq * seq_len])


def _swiglu_masks(X: np.ndarray, rng: np.random.Generator,
                  hidden_dim: int, intermediate: int,
                  density: float) -> np.ndarray:
    """Binarized silu(gate(x)) * up(x) at per-neuron quantile -> (n, intermediate)."""
    scale = 1.0 / np.sqrt(hidden_dim)
    Wg = rng.standard_normal((intermediate, hidden_dim)) * scale
    Wu = rng.standard_normal((intermediate, hidden_dim)) * scale
    act = _silu(X @ Wg.T) * (X @ Wu.T)
    thr = np.quantile(act, 1.0 - density, axis=0)
    return (act > thr).astype(np.float64)


def synthetic_calibration(spec: FamilySpec, n_tokens: int = 4096,
                           seq_len: int = 128, density: float = 0.10,
                           seed: int = 0) -> dict:
    """Generate a synthetic (X, Y) calibration dataset for pipeline validation.

    Returns {"family", "kind", "synthetic": True, "density", "layers":
    [{"X": (n, hidden), "Y": (n, units)}], "router": [{"R": (n, n_experts)}]
    or None}. Y is a deterministic function of X (SwiGLU product), so a
    well-trained oracle reaches the quality gate -- this validates the
    training/eval/gate code path, not real-model quality. Always labeled
    SYNTHETIC; never reported as a real measurement.
    """
    if not 0.0 < density < 1.0:
        raise ValueError(f"density must be in (0, 1), got {density}")
    layers, router = [], []
    for li in range(spec.n_layers):
        rng = np.random.default_rng(seed * 1_000_003 + li * 7919 + 13)
        X = _walk_hidden_states(rng, n_tokens, seq_len, spec.hidden_dim)
        n = X.shape[0]
        if spec.kind == "dense":
            Y = _swiglu_masks(X, rng, spec.hidden_dim, spec.intermediate, density)
        else:
            Wr = rng.standard_normal((spec.n_experts, spec.hidden_dim)) / np.sqrt(spec.hidden_dim)
            topk = np.argpartition(-(X @ Wr.T), spec.top_k, axis=1)[:, : spec.top_k]
            R = np.zeros((n, spec.n_experts))
            R[np.arange(n)[:, None], topk] = 1.0
            Y = np.zeros((n, spec.n_experts * spec.intermediate))
            for e in range(spec.n_experts):
                idx = np.flatnonzero(R[:, e] > 0)
                if idx.size == 0:
                    continue
                Ye = _swiglu_masks(X[idx], rng, spec.hidden_dim, spec.intermediate, density)
                s = e * spec.intermediate
                Y[np.ix_(idx, np.arange(s, s + spec.intermediate))] = Ye
            router.append({"R": R})
        layers.append({"X": X, "Y": Y})
    return {
        "family": spec.name,
        "kind": spec.kind,
        "synthetic": True,
        "density": float(density),
        "layers": layers,
        "router": router or None,
    }


# --- dataset persistence: the .npz schema is the contract between the
# (not yet built) real-capture hook and this training code. ---

def save_dataset(path: str, dataset: dict) -> None:
    d: dict = {
        "family": np.array(dataset["family"]),
        "kind": np.array(dataset["kind"]),
        "synthetic": np.array(bool(dataset["synthetic"])),
        "density": np.array(float(dataset["density"])),
        "n_layers": np.array(len(dataset["layers"])),
    }
    for i, L in enumerate(dataset["layers"]):
        d[f"X_{i}"] = np.asarray(L["X"], dtype=np.float64)
        d[f"Y_{i}"] = np.asarray(L["Y"], dtype=np.float64)
    if dataset.get("router"):
        for i, r in enumerate(dataset["router"]):
            d[f"R_{i}"] = np.asarray(r["R"], dtype=np.float64)
    np.savez(path, **d)


def load_dataset(path: str) -> dict:
    z = np.load(path, allow_pickle=False)
    n_layers = int(z["n_layers"])
    layers = [{"X": z[f"X_{i}"], "Y": z[f"Y_{i}"]} for i in range(n_layers)]
    router = None
    if "R_0" in z:
        router = [{"R": z[f"R_{i}"]} for i in range(n_layers)]
    return {
        "family": str(z["family"]),
        "kind": str(z["kind"]),
        "synthetic": bool(z["synthetic"]),
        "density": float(z["density"]),
        "layers": layers,
        "router": router,
    }


def _check_dataset(spec: FamilySpec, dataset: dict) -> None:
    if dataset["kind"] != spec.kind:
        raise ValueError(
            f"dataset kind {dataset['kind']!r} != spec kind {spec.kind!r}")
    if len(dataset["layers"]) != spec.n_layers:
        raise ValueError(
            f"dataset has {len(dataset['layers'])} layers, spec has {spec.n_layers}")
    want_units = oracle_units(spec)
    for i, L in enumerate(dataset["layers"]):
        X, Y = L["X"], L["Y"]
        if X.shape[1] != spec.hidden_dim:
            raise ValueError(
                f"layer {i}: X dim {X.shape[1]} != spec hidden_dim {spec.hidden_dim}")
        if Y.shape[1] != want_units:
            raise ValueError(
                f"layer {i}: Y dim {Y.shape[1]} != {want_units} oracle units")
        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"layer {i}: X/Y row count mismatch")
    if spec.kind == "moe":
        if not dataset.get("router"):
            raise ValueError("MoE dataset needs per-layer router masks ('R_*')")
        for i, r in enumerate(dataset["router"]):
            if r["R"].shape != (dataset["layers"][i]["X"].shape[0], spec.n_experts):
                raise ValueError(f"layer {i}: router mask shape mismatch")


def _split_indices(n: int, seed: int, test_frac: float = 0.2):
    """Deterministic shuffled train/test split (immune to row order)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    cut = int((1.0 - test_frac) * n)
    return perm[:cut], perm[cut:]


# Validated training regimes per architecture kind. The MoE FFN oracle sees
# much sparser targets (density ~ top_k/n_experts * 0.10) so its pos_weight is
# larger and it needs a gentler lr to stay stable -- measured on synthetic
# calibration (see docs/ORACLE_RECIPES.md), not guessed.
RECIPE_HYPERPARAMS = {
    "dense": {"epochs": 60, "lr": 0.1},
    "moe": {"epochs": 60, "lr": 0.02},
}

THRESHOLD_GRID = (0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60)


def tune_threshold(oracle: Oracle, layer: int, X: np.ndarray, Y: np.ndarray,
                   max_density: float = GATE_MAX_PREDICTED_DENSITY) -> float:
    """Pick the operating threshold maximizing recall under a density cap.

    Tuned on a validation split (see evaluate_dataset). Recall-first: among
    eligible thresholds (predicted mask sparse enough to keep prefetch cheap)
    the tuner takes the highest recall; ties break toward the sparser
    schedule. Falls back to OPERATING_THRESHOLD if nothing on the grid clears
    the cap.
    """
    X = np.asarray(X, dtype=np.float64)
    Yb = np.asarray(Y, dtype=bool)
    _, _, p = oracle._mlps[layer].forward(X)
    best_th, best_rec, best_dens = OPERATING_THRESHOLD, -1.0, 2.0
    for th in THRESHOLD_GRID:
        pred = p > th
        dens = float(pred.mean())
        if dens > max_density:
            continue
        tp = np.logical_and(pred, Yb).sum()
        fn = np.logical_and(~pred, Yb).sum()
        rec = float(tp / (tp + fn)) if (tp + fn) else 0.0
        if rec > best_rec + 1e-9 or (abs(rec - best_rec) <= 1e-9 and dens < best_dens):
            best_rec, best_th, best_dens = rec, th, dens
    return best_th


def union_recall(oracle: Oracle, layer: int, X: np.ndarray, Y: np.ndarray,
                 k: int = 4, threshold: float = OPERATING_THRESHOLD) -> dict:
    """Recall of the K-token lookahead union mask (the actual prefetch schedule).

    Micro-averaged over windows: predicted union vs true union per window.
    This is the metric that matters for M1 -- per-token recall is reported
    too, but prefetching runs on unions.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=bool)
    n = X.shape[0]
    tp = fn = 0
    pred_dens = 0.0
    nwin = 0
    for s in range(0, n, k):
        Xw, Yw = X[s: s + k], Y[s: s + k]
        pu = oracle.predict_union(layer, [x for x in Xw], threshold)
        tu = Yw.any(axis=0)
        tp += int(np.logical_and(pu, tu).sum())
        fn += int(np.logical_and(~pu, tu).sum())
        pred_dens += float(pu.mean())
        nwin += 1
    return {
        "union_recall": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "union_predicted_density": float(pred_dens / max(nwin, 1)),
    }


def evaluate_dataset(spec: FamilySpec, dataset: dict, ffn: Oracle,
                     router: Oracle | None = None,
                     threshold: float = OPERATING_THRESHOLD,
                     k: int = 4, test_frac: float = 0.2,
                     tune: bool = True) -> dict:
    """Per-layer metrics on held-out splits: recall, density, union recall.

    The test split is halved deterministically: the operating threshold is
    tuned on the validation half (density-capped, maximizing recall) and all
    reported metrics are measured on the untouched test half. Each report row
    carries the threshold actually used, so the gate checks the real
    operating point instead of an assumed one.
    """
    rows = []
    for i, L in enumerate(dataset["layers"]):
        X, Y = L["X"], L["Y"]
        _, te = _split_indices(X.shape[0], seed=1000 + i, test_frac=test_frac)
        va, tt = _val_test_halves(len(te), seed=2000 + i)
        th = tune_threshold(ffn, i, X[te[va]], Y[te[va]]) if tune else threshold
        m = ffn.metrics(i, X[te[tt]], Y[te[tt]], threshold=th)
        u = union_recall(ffn, i, X[te[tt]], Y[te[tt]], k=k, threshold=th)
        rows.append({"layer": i, "threshold": th, **m, **u})
    router_rows = None
    if router is not None:
        router_rows = []
        cap = 2.0 * spec.top_k / spec.n_experts
        for i, r in enumerate(dataset["router"]):
            X = dataset["layers"][i]["X"]
            R = r["R"]
            _, te = _split_indices(X.shape[0], seed=1000 + i, test_frac=test_frac)
            va, tt = _val_test_halves(len(te), seed=2000 + i)
            th = tune_threshold(router, i, X[te[va]], R[te[va]],
                                max_density=cap) if tune else threshold
            m = router.metrics(i, X[te[tt]], R[te[tt]], threshold=th)
            router_rows.append({"layer": i, "threshold": th, **m})
    return {
        "family": spec.name,
        "kind": spec.kind,
        "synthetic": bool(dataset["synthetic"]),
        "default_threshold": threshold,
        "lookahead_k": k,
        "n_experts": spec.n_experts,
        "top_k": spec.top_k,
        "layers": rows,
        "router": router_rows,
    }


def _val_test_halves(n: int, seed: int):
    """Deterministic 50/50 split of an index array of length n."""
    rng = np.random.default_rng(seed)
    p = rng.permutation(n)
    h = n // 2
    return p[:h], p[h:]


def train_family_oracle(spec: FamilySpec, dataset: dict, epochs: int | None = None,
                        lr: float | None = None, batch: int = 256,
                        mlp_hid: int = 128, seed: int = 0, verbose: bool = False,
                        eval_k: int = 4) -> dict:
    """Train the per-layer oracle(s) for a family; return oracles + report.

    epochs/lr default to the per-kind validated regime in RECIPE_HYPERPARAMS
    (measured on synthetic calibration, docs/ORACLE_RECIPES.md). For real
    models keep the code path identical and raise epochs until the quality
    gate passes; the gate -- not a fixed epoch count -- is the stop criterion.
    """
    _check_dataset(spec, dataset)
    hyp = RECIPE_HYPERPARAMS[spec.kind]
    epochs = hyp["epochs"] if epochs is None else epochs
    lr = hyp["lr"] if lr is None else lr
    ffn = Oracle(spec.n_layers, spec.hidden_dim, oracle_units(spec),
                 mlp_hid=mlp_hid, seed=seed)
    router = None
    if spec.kind == "moe":
        router = Oracle(spec.n_layers, spec.hidden_dim, spec.n_experts,
                        mlp_hid=max(16, mlp_hid // 2), seed=seed + 1)
    for i, L in enumerate(dataset["layers"]):
        X, Y = L["X"], L["Y"]
        tr, _ = _split_indices(X.shape[0], seed=1000 + i)
        ffn.train(i, X[tr], Y[tr], epochs=epochs, lr=lr, batch=batch,
                  verbose=verbose)
        if router is not None:
            # Router targets are k-hot (not ultra-sparse), so the router
            # always trains in the dense regime regardless of family.
            R = dataset["router"][i]["R"]
            dhyp = RECIPE_HYPERPARAMS["dense"]
            router.train(i, X[tr], R[tr], epochs=dhyp["epochs"],
                         lr=dhyp["lr"], batch=batch, verbose=verbose)
    report = evaluate_dataset(spec, dataset, ffn, router, k=eval_k)
    return {"ffn": ffn, "router": router, "report": report}


def gate_report(report: dict,
                min_recall: float = GATE_MIN_RECALL,
                max_predicted_density: float = GATE_MAX_PREDICTED_DENSITY,
                min_union_recall: float = GATE_MIN_UNION_RECALL,
                min_router_recall: float = GATE_MIN_ROUTER_RECALL
                ) -> tuple[bool, list[str]]:
    """Quality gate: every layer must clear recall/density/union bars.

    The router oracle gets its own density bar: at most 2x the true expert
    density (over-predicting experts means prefetching whole expert shards,
    so the cap is tighter in spirit than the neuron one).
    """
    failures = []
    for row in report["layers"]:
        i = row["layer"]
        if row["recall"] < min_recall:
            failures.append(
                f"layer {i}: recall {row['recall']:.3f} < {min_recall}")
        if row["predicted_density"] > max_predicted_density:
            failures.append(
                f"layer {i}: predicted density {row['predicted_density']:.3f} "
                f"> {max_predicted_density} (prefetch would over-fetch)")
        if row["union_recall"] < min_union_recall:
            failures.append(
                f"layer {i}: union recall {row['union_recall']:.3f} < "
                f"{min_union_recall}")
    router_rows = report.get("router") or []
    if router_rows:
        cap = 2.0 * report["top_k"] / report["n_experts"]
        for row in router_rows:
            i = row["layer"]
            if row["recall"] < min_router_recall:
                failures.append(
                    f"layer {i}: router recall {row['recall']:.3f} < "
                    f"{min_router_recall}")
            if row["predicted_density"] > cap:
                failures.append(
                    f"layer {i}: router predicted density "
                    f"{row['predicted_density']:.3f} > {cap:.2f}")
    return (len(failures) == 0, failures)


def format_report(report: dict) -> str:
    """Human-readable per-layer table for the CLI and docs."""
    src = "SYNTHETIC (pipeline validation only)" if report["synthetic"] \
        else "REAL captured data (measured)"
    lines = [
        f"family={report['family']} kind={report['kind']} source={src}",
        f"default threshold={report['default_threshold']} lookahead_k={report['lookahead_k']}",
        f"{'layer':>5} {'th':>5} {'recall':>7} {'prec':>7} {'pred_dens':>9} "
        f"{'true_dens':>9} {'union_rec':>9}",
    ]
    for row in report["layers"]:
        lines.append(
            f"{row['layer']:>5} {row['threshold']:>5.2f} {row['recall']:>7.3f} "
            f"{row['precision']:>7.3f} {row['predicted_density']:>9.3f} "
            f"{row['true_density']:>9.3f} {row['union_recall']:>9.3f}")
    if report.get("router"):
        lines.append("router oracle (top-k expert masks):")
        lines.append(f"{'layer':>5} {'th':>5} {'recall':>7} {'prec':>7} {'pred_dens':>9}")
        for row in report["router"]:
            lines.append(
                f"{row['layer']:>5} {row['threshold']:>5.2f} {row['recall']:>7.3f} "
                f"{row['precision']:>7.3f} {row['predicted_density']:>9.3f}")
    rec = [r["recall"] for r in report["layers"]]
    lines.append(
        f"summary: min_recall={min(rec):.3f} mean_recall={sum(rec)/len(rec):.3f}")
    return "\n".join(lines)


def quick_spec(spec: FamilySpec) -> FamilySpec:
    """Scaled-down copy of a spec for fast pipeline validation (labeled QUICK)."""
    kw = dict(n_layers=2, hidden_dim=64, intermediate=256)
    if spec.kind == "moe":
        kw.update(n_experts=4, top_k=2)
    return replace(spec, name=spec.name + "-quick", **kw)
