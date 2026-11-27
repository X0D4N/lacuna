"""Day 2 track: oracle recipes -- presets, synthetic data, training, gates.

All training here runs on SYNTHETIC calibration (labeled as such); it
validates the recipe pipeline end to end, not real-model quality.
"""

import numpy as np
import pytest

from lacuna.recipes import (
    GATE_MAX_PREDICTED_DENSITY,
    format_report,
    gate_report,
    get_family,
    list_families,
    load_dataset,
    oracle_units,
    quick_spec,
    save_dataset,
    spec_from_dims,
    synthetic_calibration,
    train_family_oracle,
    tune_threshold,
    union_recall,
)


def test_family_presets():
    assert set(list_families()) == {"qwen", "llama", "moe"}
    qwen = get_family("qwen")
    assert (qwen.n_layers, qwen.hidden_dim, qwen.intermediate) == (28, 3584, 18944)
    assert qwen.kind == "dense" and qwen.ffn == "swiglu"
    llama = get_family("llama")
    assert (llama.n_layers, llama.hidden_dim, llama.intermediate) == (32, 4096, 14336)
    moe = get_family("moe")
    assert (moe.n_experts, moe.top_k) == (8, 2)
    assert oracle_units(qwen) == 18944
    assert oracle_units(moe) == 8 * 14336
    with pytest.raises(KeyError):
        get_family("gpt")


def test_spec_from_dims_validation():
    s = spec_from_dims("tiny", "dense", n_layers=2, hidden_dim=16, intermediate=64)
    assert s.kind == "dense" and oracle_units(s) == 64
    with pytest.raises(ValueError):
        spec_from_dims("x", "moe", 2, 16, 64)  # experts required
    with pytest.raises(ValueError):
        spec_from_dims("x", "dense", 2, 16, 64, n_experts=4)  # not for dense
    with pytest.raises(ValueError):
        spec_from_dims("x", "dense", 2, 16, 0)  # dims must be positive


def test_synthetic_dense_shapes_and_density():
    spec = quick_spec(get_family("qwen"))
    ds = synthetic_calibration(spec, n_tokens=512, seed=3)
    assert ds["synthetic"] is True and ds["router"] is None
    assert len(ds["layers"]) == 2
    X, Y = ds["layers"][0]["X"], ds["layers"][0]["Y"]
    assert X.shape == (512, 64) and Y.shape == (512, 256)
    assert abs(Y.mean() - 0.10) < 0.03  # ~10% active per the density param


def test_synthetic_moe_shapes():
    spec = spec_from_dims("t", "moe", n_layers=2, hidden_dim=32,
                          intermediate=64, n_experts=4, top_k=2)
    ds = synthetic_calibration(spec, n_tokens=256, seed=5)
    assert ds["router"] is not None and len(ds["router"]) == 2
    R = ds["router"][0]["R"]
    assert R.shape == (256, 4)
    assert (R.sum(axis=1) == 2).all()  # top-2 routing, every token
    Y = ds["layers"][0]["Y"]
    assert Y.shape == (256, 4 * 64)


def test_dataset_roundtrip(tmp_path):
    spec = quick_spec(get_family("llama"))
    ds = synthetic_calibration(spec, n_tokens=256, seed=9)
    p = str(tmp_path / "calib.npz")
    save_dataset(p, ds)
    ds2 = load_dataset(p)
    assert ds2["family"] == "llama-quick" and ds2["synthetic"] is True
    assert ds2["layers"][0]["X"].shape == ds["layers"][0]["X"].shape
    np.testing.assert_array_equal(ds2["layers"][1]["Y"], ds["layers"][1]["Y"])


def test_train_dense_passes_gate():
    spec = quick_spec(get_family("qwen"))
    ds = synthetic_calibration(spec, n_tokens=4096, seed=1)
    out = train_family_oracle(spec, ds)
    ok, failures = gate_report(out["report"])
    assert ok, failures
    assert out["router"] is None
    rep = format_report(out["report"])
    assert "SYNTHETIC" in rep and "min_recall" in rep


def test_train_moe_passes_gate():
    spec = spec_from_dims("t", "moe", n_layers=2, hidden_dim=64,
                          intermediate=128, n_experts=4, top_k=2)
    ds = synthetic_calibration(spec, n_tokens=2048, seed=1)
    out = train_family_oracle(spec, ds)
    ok, failures = gate_report(out["report"])
    assert ok, failures
    assert out["router"] is not None
    rrec = [r["recall"] for r in out["report"]["router"]]
    assert min(rrec) >= 0.85


def test_gate_logic():
    good = {
        "top_k": 2, "n_experts": 8,
        "layers": [{"layer": 0, "recall": 0.9, "predicted_density": 0.15,
                    "union_recall": 0.95}],
        "router": [{"layer": 0, "recall": 0.95, "predicted_density": 0.4}],
    }
    ok, failures = gate_report(good)
    assert ok and failures == []
    bad = {
        "top_k": 2, "n_experts": 8,
        "layers": [{"layer": 3, "recall": 0.5, "predicted_density": 0.15,
                    "union_recall": 0.95}],
        "router": None,
    }
    ok, failures = gate_report(bad)
    assert not ok and any("layer 3" in f and "recall" in f for f in failures)


def test_tune_threshold_respects_cap():
    spec = quick_spec(get_family("qwen"))
    ds = synthetic_calibration(spec, n_tokens=512, seed=11)
    out = train_family_oracle(spec, ds, epochs=10)
    X, Y = ds["layers"][0]["X"], ds["layers"][0]["Y"]
    th = tune_threshold(out["ffn"], 0, X, Y)
    assert th in (0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60)
    m = out["ffn"].metrics(0, X, Y, threshold=th)
    assert m["predicted_density"] <= GATE_MAX_PREDICTED_DENSITY + 1e-9


def test_union_recall_sane():
    spec = quick_spec(get_family("llama"))
    ds = synthetic_calibration(spec, n_tokens=512, seed=13)
    out = train_family_oracle(spec, ds, epochs=10)
    X, Y = ds["layers"][0]["X"], ds["layers"][0]["Y"]
    u = union_recall(out["ffn"], 0, X, Y, k=4, threshold=0.2)
    assert 0.0 <= u["union_recall"] <= 1.0
    assert 0.0 <= u["union_predicted_density"] <= 1.0


def test_quick_spec_scales_down():
    full = get_family("moe")
    q = quick_spec(full)
    assert q.n_layers == 2 and q.hidden_dim == 64 and q.n_experts == 4
    assert q.name == "moe-quick" and full.n_layers == 32  # original untouched


def test_cli_oracle_recipe_list(capsys):
    from lacuna.cli import _cmd_oracle_recipe, build_parser
    a = build_parser().parse_args(["oracle-recipe", "--list-families"])
    assert _cmd_oracle_recipe(a) == 0
    out = capsys.readouterr().out
    assert "qwen" in out and "llama" in out and "moe" in out


def test_cli_oracle_recipe_quick_runs():
    from lacuna.cli import _cmd_oracle_recipe, build_parser
    a = build_parser().parse_args(
        ["oracle-recipe", "--family", "llama", "--quick",
         "--tokens", "256", "--epochs", "2"])
    rc = _cmd_oracle_recipe(a)  # underfit on purpose; must not crash
    assert rc in (0, 1)
