# Perplexity harness — exact vs extreme precision (MEASURED)

Generated 2026-10-18 by `benchmarks/perplexity/measure.py`.
ALL numbers below are MEASURED on this box: real forward passes of a
real trained model through the real codecs (`lacuna/quant.py`), with
tiers assigned by the production rule (`lacuna/convert.py::_tier_for`).

## What was measured

- Tiny transformer: 258,528 params, d=96, 2 layers, 4 heads, vocab 384, seq 64.
- Trained from scratch (Adam, cosine LR, 1200 steps x 16x64 tokens, seed 7) on
  60,000 tokens of deterministic synthetic English
  (`benchmarks/perplexity/corpus.txt`, sha256:e2150e9c865b0226).
- Held-out eval: 12,000 tokens from the same generator.
- Perplexity = exp(mean token NLL), teacher-forced, full eval set.
- The `extreme` rows quantize per tensor with one scale per tensor;
  production pages use one scale per *page* (finer granularity,
  error <= the per-tensor bound), so these deltas are a conservative
  upper bound on the production quality cost.

## Results

| policy | tier mix | eval ppl | Δppl vs dense | stored bits/param |
|---|---|---|---|---|
| dense FP32 (baseline) | f32×21 | 6.0111 | 0.000 (bit-identical) | 32.00 |
| exact (RAW pages) | raw[exact]×21 | 6.0111 | 0.000 (bit-identical) | 32.00 |
| extreme · Q4/Q2 (thr 200 KiB) | q2×6, q4×15 | 13.1641 | +7.1530 | 3.14 |
| extreme · ternary stress (thr 50 KiB) | q4×13, ternary×8 | 41.4250 | +35.4140 | 2.57 |

## Reading these numbers honestly

- `exact` == `dense` to the last bit: the harness re-verifies the
  exactness guarantee at the perplexity level, not just token identity.
- The Q4/Q2 row is what a real `lacuna convert --precision-policy extreme`
  does to F32/F16 tensors (200 KiB threshold chosen so the tiny model's
  tier mix matches production: Q4 for attn/norm/embed, Q2 for FFN).
- The ternary-stress row forces the cold tier on; it isolates the
  ~1.58-bit tier's cost. Production engages ternary only above the
  1024 MiB `--cold-threshold-mb` (tensors that large are rare even in
  100B models), so this row is a stress test, not the default.
- These deltas do NOT transfer to real LLMs: a 260K-param model on
  synthetic text is a codec stress test, not a quality benchmark.
  Absolute perplexity here is corpus-specific and meaningless
  outside this harness. The honest claim is only: *on a real trained
  forward pass, the extreme codecs cost this much relative to exact.*
- Production per-page scales are finer than this harness's
  per-tensor scales, so real-world degradation should be <= measured.
