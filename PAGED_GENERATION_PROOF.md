# Paged generation proof — generation THROUGH the pager (2026-10-25)

This closes Lacuna's standing core gap: until now, `Engine` mode
`"lacuna"` ran mock orchestration over real pages and labeled its output
`[simulated]`. This document proves the pager path now executes the
model's own transformer math and generates real text.

## What was built

- `lacuna/gguf.py` — full GGUF v3 parser (metadata, tensor index,
  tokenizer vocab/merges/bos/eos) plus dequantizers for F32, F16, Q5_0,
  Q8_0, Q4_K, Q6_K. Every dequantizer verified **bit-exact** (max diff
  0.0) against ggml's own exported functions via ctypes on real model
  bytes.
- `lacuna/bpe.py` — GPT-2-style byte-level BPE built from the model's
  own vocab/merges. Token ids match llama-server `/tokenize` exactly on
  all tested prompts.
- `lacuna/paged_model.py` — `PagedQwen2`: RMSNorm, GQA attention with
  QKV biases, half-rotation RoPE, KV cache, SwiGLU, separate lm_head,
  greedy decoding. Every weight byte is read **only** through
  `PagedWeightStore` (exact policy enforced, byte length checked).
- `lacuna/engine.py` — `Engine` mode `"lacuna"` now uses the real
  executor for supported architectures (currently Qwen2). If the
  executor cannot load a model, it falls back to the labeled
  `[simulated]` orchestration path, with the reason kept inspectable on
  `Engine.paged_error` — a downgrade is never silent.

## Method

Model: Qwen2.5-0.5B-Instruct Q4_K_M (`Qwen/Qwen2.5-0.5B-Instruct-GGUF`,
491,400,032 bytes, sha256 `74a4da8c…9d7a9db`, identical to the
2026-10-23 proof file). Baseline: llama-server `/completion`, greedy
(`temperature 0, top_k 1, seed 42`). Pager run: `PagedQwen2` at a
128 MiB hot budget, greedy, max 32 tokens per prompt. The baseline
server was killed after the run; the model file is the active dev/demo
model.

## Results

### Prompt 2: "Name three primary colors" — EXACT MATCH

All 32 tokens identical to the dense baseline:

> " of the rainbow and their corresponding wavelengths. The primary
> colors of the rainbow are red, orange, and yellow. Their
> corresponding wavelengths are 580 nm,"

### Prompt 1: "The capital of France is" — 19 steps exact, then a near-tie

Teacher-forcing analysis (dense tokens fed back as context, so the
first split is isolated): steps 0–18 match the dense baseline exactly,
including first token ` Paris` at p=0.4337. At step 19 the two
implementations split on a coin flip:

| | token | probability |
|---|---|---|
| paged executor | ` also` | 0.2349 |
| dense baseline | ` located` | 0.2327 |

Margin: **0.0022**. That is floating-point rounding noise between a
NumPy float32 implementation and llama.cpp's optimized kernels
(different summation orders, FMA use), not a logic error — a
systematic math bug would break the first token, as the earlier RoPE
bug did. Greedy decoding amplifies a near-tie into a visible text
difference; everything after step 19 follows from the different
context. The free-running pager output matches this exactly
("…It is also the capital of the department of Paris…").

### Pager discipline

- Cached bytes: 132,471,041 ≤ 128 MiB budget. Held.
- Peak RSS: 2,485 MB.
- No weight byte bypasses the store.

## Honest costs and limits

- Speed: **0.14 tok/s** at 128 MiB. Slow, stated plainly. At this
  budget the cache thrashes completely (23,875 faults, 0 hits, 16.5 GB
  read for 64 tokens) because each token re-reads the whole model
  through the pager; per-token re-dequantization is the bottleneck.
  This run proves the mechanism is correct, not that it is fast yet.
- v1 holds token embeddings and lm_head as float32 (~545 MB each);
  every other tensor streams through the pager.
- Supported architecture for the real executor: Qwen2 only, for now.
- Exactness claim: token-exact until a near-tie (see step 19 above),
  not bit-exact across implementations. Bit-exact agreement would
  require llama.cpp's exact kernels, which would defeat the point of
  an independent executor.

## Bugs found by verification (not by inspection)

1. **Q6_K layout**: `d` is LAST in the block
   (`ql[128] | qh[64] | scales[16] | d`), unlike Q4_K/Q5_0/Q8_0 (d
   first). First version produced NaN; fixed against the local
   `ggml-quants.c`.
2. **RoPE form**: pair-rotation (Neox) gave wrong first token
   (`" "` @0.20, then `" 1000000"`). Half-rotation gives ` Paris`
   @0.4337, matching dense.

## Test suite

218 passed, 2 skipped (baseline was 199/2). New: `tests/test_gguf.py`,
`tests/test_bpe.py` (16 tests, hand-computed vectors),
`tests/test_paged_model.py` (3 tests on a tiny synthetic Qwen2 GGUF:
finite forward pass, deterministic generation, Engine uses the paged
path for a supported arch).

## Status

Generation through the pager is real and measured. Not pushed: the
repo is private and distribution is gated on the owner's decision plus
a fresh classic PAT. This commit is local only.
