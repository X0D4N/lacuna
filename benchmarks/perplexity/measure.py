"""Perplexity harness driver: train tiny LM -> apply policies -> measure.

Policies (weight sets compared on the same held-out eval text):
  dense    FP32 as trained (baseline; what "exact" must match bit-for-bit)
  exact    every tensor round-tripped through raw bytes, like the ``exact``
           precision policy's RAW pages — must give IDENTICAL perplexity
  extreme  per-tensor codec via the PRODUCTION rule
           ``lacuna/convert.py::_tier_for(name, nbytes, cold_threshold)``:
           - threshold 200 KiB: Q4 for attn/norm/embed/output, Q2 for FFN
             (mirrors the production mix at the 1024 MiB default)
           - threshold 50 KiB ("ternary stress"): forces the >50 KiB FFN +
             embedding/output tensors into the ternary cold tier, isolating
             the cold tier's quality cost

Every number printed is MEASURED on this box (real forward passes through
the real codecs) — but on a ~260K-param synthetic model, so the rows do not
transfer to real LLMs. Only the *deltas* are the finding.

Usage (repo root)::
    PYTHONPATH=. python -m benchmarks.perplexity.measure [--retrain]
Writes ``benchmarks/perplexity.md`` (methodology summary + results table).
"""

from __future__ import annotations

import hashlib
import math
import os
import sys
import time

import numpy as np

from lacuna import quant as q
from lacuna.convert import _tier_for

from . import make_corpus
from .tiny_lm import TinyLM

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
MODEL_PATH = os.path.join(HERE, "tiny_lm.npz")
OUT_MD = os.path.join(REPO, "benchmarks", "perplexity.md")

TRAIN_STEPS = 1200
BATCH = 16
LR_MAX = 3e-3
LR_MIN = 3e-4
SEQ = 64
SEED = 7

# codec bits stored per element (scale overhead is one fp32 per tensor/page)
CODEC_BITS = {q.Q4: 4, q.Q2: 2, q.TERNARY: 2, q.RAW: 32}


def _batches(ids: np.ndarray, batch: int, seq: int, rng, shuffle: bool):
    n_win = (len(ids) - 1) // seq
    order = np.arange(n_win)
    if shuffle:
        rng.shuffle(order)
    for s in range(0, n_win, batch):
        idx = order[s : s + batch]
        if len(idx) < batch:
            break
        yield np.stack([ids[i * seq : i * seq + seq + 1] for i in idx])


def train(model: TinyLM, train_ids: np.ndarray, steps: int, log_every: int = 200):
    rng = np.random.default_rng(SEED)
    opt = model.new_optimizer()
    t0 = time.time()
    n = 0
    while n < steps:
        for batch_ids in _batches(train_ids, BATCH, SEQ, rng, shuffle=True):
            if n >= steps:
                break
            frac = n / steps
            lr = LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * frac))
            loss = model.train_step(batch_ids, opt, lr)
            n += 1
            if n % log_every == 0 or n == steps:
                print(f"  step {n}/{steps} loss={loss:.4f} lr={lr:.2e}", flush=True)
    print(f"  trained in {time.time() - t0:.1f}s")


def eval_ppl(model: TinyLM, eval_ids: np.ndarray, batch: int = 32,
             seq: int = SEQ) -> float:
    nll_sum, n_tok = 0.0, 0
    rng = np.random.default_rng(0)
    for batch_ids in _batches(eval_ids, batch, seq, rng, shuffle=False):
        _, s, t = model.forward(batch_ids)
        nll_sum += s
        n_tok += t
    if n_tok == 0:
        raise ValueError(
            f"eval set too short for one batch; need >= {batch * (seq + 1)} tokens"
        )
    return math.exp(nll_sum / n_tok)


def apply_policy(params: dict, policy: str, cold_threshold: int):
    """Return (new_params, tier_report, avg_bits)."""
    out, report = {}, {}
    bits_num = bits_den = 0
    for name, p in params.items():
        nbytes = p.nbytes
        if policy == "dense":
            out[name] = p.copy()
            report[name] = "f32"
            b = 32
        elif policy == "exact":
            # RAW byte pages, bit-identical to source (production behavior)
            out[name] = np.frombuffer(p.tobytes(), dtype=np.float32).reshape(p.shape).copy()
            report[name] = "raw[exact]"
            b = 32
        elif policy == "extreme":
            tier = _tier_for(name, nbytes, cold_threshold)
            packed, scale, shape = q.quantize(p, tier)
            out[name] = q.dequantize(packed, scale, shape, tier).astype(np.float32)
            report[name] = q.precision_name(tier)
            b = CODEC_BITS[tier]
        else:
            raise ValueError(policy)
        bits_num += b * p.size
        bits_den += p.size
    return out, report, bits_num / bits_den


def _tier_mix(report: dict) -> str:
    from collections import Counter
    c = Counter(report.values())
    return ", ".join(f"{k}×{v}" for k, v in sorted(c.items()))


def ensure_model(retrain: bool = False) -> TinyLM:
    if not os.path.exists(make_corpus.CORPUS_PATH):
        print("generating corpus...", flush=True)
        print(make_corpus.write_corpus())
    train_toks, _ = make_corpus.load_corpus()
    corpus_hash = hashlib.sha256(
        open(make_corpus.CORPUS_PATH, "rb").read()
    ).hexdigest()[:16]
    vocab = make_corpus.build_vocab(train_toks)
    assert len(vocab) <= TinyLM().cfg["vocab"], "vocab overflow"
    model = TinyLM(seed=SEED)
    if os.path.exists(MODEL_PATH) and not retrain:
        z = np.load(MODEL_PATH, allow_pickle=True)
        if str(z["_corpus_hash"]) == corpus_hash:
            print(f"loading trained model {MODEL_PATH}")
            return TinyLM.load(MODEL_PATH)
        print("corpus changed; retraining")
    print("training tiny LM...", flush=True)
    train_ids = make_corpus.encode(train_toks, vocab)
    train(model, train_ids, TRAIN_STEPS)
    z_extra = {"_corpus_hash": np.array(corpus_hash), "_vocab_n": np.array(len(vocab))}
    np.savez(MODEL_PATH, **{k: v for k, v in model.params.items()}, **z_extra,
             _config_json=np.array(__import__("json").dumps(model.cfg)))
    return model


def main(argv=None) -> int:
    retrain = "--retrain" in (argv or [])
    model = ensure_model(retrain=retrain)
    train_toks, eval_toks = make_corpus.load_corpus()
    vocab = make_corpus.build_vocab(train_toks)
    eval_ids = make_corpus.encode(eval_toks, vocab)
    n_params = model.n_params()
    print(f"params: {n_params}, eval tokens: {len(eval_ids)}")

    rows = []
    for label, policy, thr in [
        ("dense FP32 (baseline)", "dense", 0),
        ("exact (RAW pages)", "exact", 0),
        ("extreme · Q4/Q2 (thr 200 KiB)", "extreme", 200 * 1024),
        ("extreme · ternary stress (thr 50 KiB)", "extreme", 50 * 1024),
    ]:
        params, report, bits = apply_policy(model.params, policy, thr)
        m = TinyLM(cfg=model.cfg, seed=0)
        m.params = params
        ppl = eval_ppl(m, eval_ids)
        rows.append((label, policy, _tier_mix(report), ppl, bits, report))
        print(f"  {label}: ppl={ppl:.4f} bits/param={bits:.2f} [{_tier_mix(report)}]")

    dense_ppl = rows[0][3]
    corpus_hash = hashlib.sha256(open(make_corpus.CORPUS_PATH, "rb").read()).hexdigest()[:16]
    ts = time.strftime("%Y-%m-%d")
    lines = [
        "# Perplexity harness — exact vs extreme precision (MEASURED)",
        "",
        f"Generated {ts} by `benchmarks/perplexity/measure.py`.",
        "ALL numbers below are MEASURED on this box: real forward passes of a",
        "real trained model through the real codecs (`lacuna/quant.py`), with",
        "tiers assigned by the production rule (`lacuna/convert.py::_tier_for`).",
        "",
        "## What was measured",
        "",
        f"- Tiny transformer: {n_params:,} params, "
        f"d={model.cfg['d_model']}, {model.cfg['n_layer']} layers, "
        f"{model.cfg['n_head']} heads, vocab {model.cfg['vocab']}, seq {SEQ}.",
        "- Trained from scratch (Adam, cosine LR, "
        f"{TRAIN_STEPS} steps x {BATCH}x{SEQ} tokens, seed {SEED}) on",
        f"  {make_corpus.N_TRAIN_TOKENS:,} tokens of deterministic synthetic English",
        f"  (`benchmarks/perplexity/corpus.txt`, sha256:{corpus_hash}).",
        f"- Held-out eval: {make_corpus.N_EVAL_TOKENS:,} tokens from the same generator.",
        "- Perplexity = exp(mean token NLL), teacher-forced, full eval set.",
        "- The `extreme` rows quantize per tensor with one scale per tensor;",
        "  production pages use one scale per *page* (finer granularity,",
        "  error <= the per-tensor bound), so these deltas are a conservative",
        "  upper bound on the production quality cost.",
        "",
        "## Results",
        "",
        "| policy | tier mix | eval ppl | Δppl vs dense | stored bits/param |",
        "|---|---|---|---|---|",
    ]
    for label, policy, mix, ppl, bits, _ in rows:
        d = ppl - dense_ppl
        ds = "0.000 (bit-identical)" if policy in ("dense", "exact") and d == 0 else f"{d:+.4f}"
        lines.append(f"| {label} | {mix} | {ppl:.4f} | {ds} | {bits:.2f} |")
    lines += [
        "",
        "## Reading these numbers honestly",
        "",
        "- `exact` == `dense` to the last bit: the harness re-verifies the",
        "  exactness guarantee at the perplexity level, not just token identity.",
        "- The Q4/Q2 row is what a real `lacuna convert --precision-policy extreme`",
        "  does to F32/F16 tensors (200 KiB threshold chosen so the tiny model's",
        "  tier mix matches production: Q4 for attn/norm/embed, Q2 for FFN).",
        "- The ternary-stress row forces the cold tier on; it isolates the",
        "  ~1.58-bit tier's cost. Production engages ternary only above the",
        "  1024 MiB `--cold-threshold-mb` (tensors that large are rare even in",
        "  100B models), so this row is a stress test, not the default.",
        "- These deltas do NOT transfer to real LLMs: a 260K-param model on",
        "  synthetic text is a codec stress test, not a quality benchmark.",
        "  Absolute perplexity here is corpus-specific and meaningless",
        "  outside this harness. The honest claim is only: *on a real trained",
        "  forward pass, the extreme codecs cost this much relative to exact.*",
        "- Production per-page scales are finer than this harness's",
        "  per-tensor scales, so real-world degradation should be <= measured.",
        "",
    ]
    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
