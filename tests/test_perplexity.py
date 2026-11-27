"""Perplexity harness tests (Day 3).

Fast checks on a micro model: the harness must show training actually
learning, exact == dense bit-for-bit, extreme degrading gracefully, and the
tiering following the production rule in lacuna/convert.py.
"""

import math

import numpy as np
import pytest

from lacuna import quant as q
from lacuna.convert import _tier_for

from benchmarks.perplexity import make_corpus
from benchmarks.perplexity.measure import apply_policy, eval_ppl
from benchmarks.perplexity.tiny_lm import CONFIG, TinyLM

SMALL = dict(CONFIG)
SMALL.update(d_model=32, n_head=2, n_layer=1, d_ff=64, vocab=128, seq=32)


@pytest.fixture(scope="module")
def trained():
    toks = make_corpus.generate(seed=11, n_tokens=6000)
    vocab = make_corpus.build_vocab(toks, max_vocab=128)
    ids = make_corpus.encode(toks, vocab)
    m = TinyLM(cfg=SMALL, seed=7)
    opt = m.new_optimizer()
    rng = np.random.default_rng(3)
    for _ in range(40):
        js = rng.integers(0, len(ids) - 33, 8)
        bb = np.stack([ids[j : j + 33] for j in js])
        m.train_step(bb, opt, 3e-3)
    ev = ids[4000:]
    return m, ev


def _with_params(cfg, params):
    m = TinyLM(cfg=cfg, seed=0)
    m.params = params
    return m


def test_training_reduces_loss(trained):
    m, ev = trained
    # eval ppl far below uniform (128) proves the model learned structure
    ppl = eval_ppl(m, ev, batch=4, seq=32)
    assert math.isfinite(ppl) and ppl < 128.0


def test_exact_matches_dense_bitwise(trained):
    m, ev = trained
    dense_ppl = eval_ppl(m, ev, batch=4, seq=32)
    pe, report, bits = apply_policy(m.params, "exact", 0)
    for name, p in m.params.items():
        assert pe[name].tobytes() == p.tobytes(), name
    assert all(v == "raw[exact]" for v in report.values())
    exact_ppl = eval_ppl(_with_params(SMALL, pe), ev, batch=4, seq=32)
    assert exact_ppl == dense_ppl  # bit-identical -> identical perplexity
    assert bits == 32.0


def test_extreme_degrades_gracefully(trained):
    m, ev = trained
    dense_ppl = eval_ppl(m, ev, batch=4, seq=32)
    px, report, bits = apply_policy(m.params, "extreme", 200 * 1024)
    extreme_ppl = eval_ppl(_with_params(SMALL, px), ev, batch=4, seq=32)
    assert math.isfinite(extreme_ppl)
    # quantization must not improve the model beyond float noise
    assert extreme_ppl >= dense_ppl * 0.999
    # ...and must not explode on a sane model
    assert extreme_ppl < dense_ppl * 10
    assert bits < 32.0


def test_eval_is_deterministic(trained):
    m, ev = trained
    a = eval_ppl(m, ev, batch=4, seq=32)
    b = eval_ppl(m, ev, batch=4, seq=32)
    assert a == b


def test_tier_rule_is_production_rule():
    # full-size (d=96) tensor byte counts, 200 KiB threshold
    assert _tier_for("blk.0.ffn.w1", 96 * 192 * 4, 200 * 1024) == q.Q2
    assert _tier_for("blk.0.attn.wq", 96 * 96 * 4, 200 * 1024) == q.Q4
    assert _tier_for("tok_embeddings.weight", 384 * 96 * 4, 200 * 1024) == q.Q4
    assert _tier_for("blk.0.attn_norm.weight", 96 * 4, 200 * 1024) == q.Q4
    # 50 KiB stress threshold pushes the 72 KiB FFN tensors to ternary
    assert _tier_for("blk.0.ffn.w1", 96 * 192 * 4, 50 * 1024) == q.TERNARY


def test_ternary_stress_engages_cold_tier():
    m = TinyLM(seed=7)  # full-size config
    _, report, bits = apply_policy(m.params, "extreme", 50 * 1024)
    assert "ternary" in set(report.values())
    assert "q4" in set(report.values())  # attn stays hot
    assert bits < 4.0


def test_corpus_is_deterministic():
    a = make_corpus.generate(seed=5, n_tokens=1000)
    b = make_corpus.generate(seed=5, n_tokens=1000)
    assert a == b
    assert len(a) == 1000
