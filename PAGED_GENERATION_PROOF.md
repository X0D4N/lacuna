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

---

## Appendix A — Speed work, 0.5B (2026-10-25, MEASURED)

### Sliced embed/head

`read_tensor_rows` in `gguf.py` plus per-call embed gathering plus
8192-row lm_head blocks in `paged_model.py`. No held embed/head
float32 matrices anymore (only `out_norm` is held). Verified at the
time: suite 220 passed / 2 skipped, 0.5B generated ids identical on
both proof prompts (32/32), peak RSS 2001 MB versus 3838 MB
pre-change at a 2 GB budget.

### Regression found by measurement, then fixed

Steady-state on the 0.5B at 2 GB fell 16.37 → 2.18 tok/s after
slicing, because sliced head blocks bypassed the dequant cache: every
token re-dequantized the full 136M-value head (0.373 s warm per token
on the 0.5B). Fix: a bounded, evictable LRU cache of dequantized head
blocks inside the existing `_dq` cache — never the whole matrix, so
100B-class heads stay runnable. Result: steady-state back to
14.55 tok/s on the 0.5B at 2 GB, peak RSS 2471 MB, identity still
PASS. Committed as `beb9ddc` together with the Q2_K/Q3_K/Q5_K
dequantizers (bit-exact vs ggml's exported functions on synthetic
blocks via ctypes; no 110B file existed on this box, so synthetic
blocks were the verification — stated plainly).

Note on plain-speed figures: the 14.55 tok/s above is the speed-work
run's post-fix steady-state measurement. The speculative experiment's
own plain baselines on the same model and budget measured 12.80 and
11.99 tok/s on its two prompts (Appendix C). Different runs, different
prompts — both are reported with their context and are not merged
into one canonical number.

## Appendix B — 110B run, Kaggle (2026-10-25, MEASURED)

Kernel `x0d4n7/lacuna-110b-pager`, tree `beb9ddc` (zip sha256
`eec2c098…8b16ef`). Environment as recorded: 4 vCPU Intel Xeon @
2.20 GHz, 32098 MB RAM, /tmp 1126 GB free, Python 3.13.15, numpy
2.1.3. Budgets: 8 GB hot + 8 GB dequant (the "8 GB if host RAM ≥16 GB"
rule).

### Download and convert

- File: `Qwen1.5-110B-Chat-Q2_K.gguf` from
  `second-state/Qwen1.5-110B-Chat-GGUF` (public, no token).
  41,175,701,984 bytes, SHA-256
  `922e4ce5af5e98e5202078969ebaa2f2d8abdf9f80734a4e3f1529093ac16062`,
  downloaded in 1428.8 s at 28.8 MB/s via urllib-range-resume.
- Header check: qwen2, 80 layers, hidden 8192, FFN 49152, 64 Q heads,
  8 KV heads, eps 1e-6, rope theta 1,000,000, 963 tensors, types F32
  401 / Q2_K 321 / Q3_K 160 / Q5_K 80 / Q6_K 1, `output.weight`
  present (not tied).
- Convert (exact policy, 963/963 raw tensors): 324.2 s, `.lac`
  41,170,492,318 bytes, RSS 310 MB.

### Paged generation — prompt 1: "The capital of France is"

Real tokens through the pager at 110B scale:

> " Paris. It is located in the north"

8 tokens `[12095, 13, 1084, 374, 7407, 304, 279, 10200]`,
time-cap truncated (`truncated_at_time_cap: true`). Prefill
2148.3 s; per-token 2093.74–2191.95 s (≈0.0005 tok/s derived from
8 tokens / 17,216 s generation time; the record's overall field
rounds to 0.0). Store: 163 hits, 181,525 faults, 366,875,967,861
bytes read (≈366.9 GB). Raw cached 8,589,391,969 bytes ≤ 8 GB budget;
dequant cached 6,593,445,888 bytes ≤ 8 GB budget; peak RSS
19,864.2 MB on the 32 GB host — budgets held, no OOM.

### Paged generation — prompt 2: "Name three primary colors"

Prefilled in 2249.8 s, emitted 0 tokens. The record's
`truncated_at_time_cap` flag is false and no stop reason is recorded
— missing, not guessed.

### Baseline: NOT COMPLETED, and why the kernel status is ERROR

The llama.cpp CPU build finished and the dense `llama-cli` run on the
same prompt timed out at the 600 s cap with no tokens recorded. The
script's own timeout handler then crashed the run:
`TypeError: can't concat str to bytes` (script.py line 39, `out` str
+ `err` bytes), which set kernel status ERROR. Paged results were
saved before the baseline step, so the ERROR is a baseline-harness
failure, not a pager failure — but it means **no dense
token-identity comparison exists at 110B**.

### Verdict

- Mechanism PROVEN at 110B scale: real, coherent, correct tokens
  through the pager on a 110B model, budgets held, no OOM.
- Token-identity vs dense NOT proven: there is no baseline to compare
  against. (Consistency note, not a proof: the 110B's first 7 token
  ids match the 7B paged run's first 7 ids.)
- Speed NOT proven: ~35 min/token on this free-CPU class is unusable
  as-is. The speculative + GPU path (Appendix C and
  `docs/SPECULATIVE_PAGED_DESIGN.md`) is the route to usable speed
  and is unmeasured at this scale.

## Appendix C — 7B speculative proof + token-13 near-tie (2026-10-25, MEASURED)

Honesty note first: the 7B model file's repo/name/SHA is not recorded
in the surviving run files re-read here — missing. The 7B files were
deleted after the verdict stage; what follows comes from the surviving
run artifacts (`m7b_*`, `dense7b_out.log`, `token13_out.jsonl`,
`dense13_out.json`). Draft: 0.5B class, K=4. Budgets: target 256 MB
page + 256 MB dequant; draft 128 MB page + 128 MB dequant.

### Speculative vs plain, prompt "The capital of France is"

Speculative (K=4): 32 tokens in 8 rounds, 24/32 accepted (75%),
4.0 tokens/block, wall 2214.65 s, 0.0144 tok/s, peak RSS 2167 MB.
Text:

> " Paris. It is located in the northern central part of the
> country, on the banks of the Seine River. Paris is the largest city
> in France and one"

Target store: 28,596 faults, 55,039,296,197 bytes read; draft store:
18,865 faults, 19,246,590,349 bytes read.

Plain: 31 tokens logged (prefill 105.74 s; generation 5490.12 s,
≈0.0056 tok/s) plus token 31 recovered separately (825; recovery run
self-consistency clean, peak RSS 1740 MB). Full 32 ids identical to
speculative 32/32. Speedup ≈2.6x (0.0144 vs ≈0.0055 tok/s). This is
the neural-draft result: speculation with a real draft model, not the
n-gram baseline below.

### 0.5B n-gram draft results (from `docs/SPECULATIVE_PAGED_DESIGN.md`)

Exactness holds — speculative ids identical 32/32 on both prompts at
both budgets — but the n-gram draft is too weak to win on speed:
cache-fit regime 3.7–4.5x SLOWER than plain (plain 12.80/11.99 vs
speculative 3.45/2.66 tok/s); thrash regime 21–23% slower (plain
0.249/0.230 vs speculative 0.192/0.182 tok/s); acceptance 3.6%/1.7%.
The design doc's reading, adopted here: the mechanism is exact; the
draft is the gap. Break-evens measured on the 0.5B: mean accepted
≥ ~0.5 tokens/block in the thrash regime, ~= 3.2 of K=4 in the
cache-fit regime.

### Token-13 near-tie: dense vs paged divergence, verdict NEAR-TIE

Dense 16-token text: "...country. The city". Paged text:
"...country, on the b...". `text_match: false` — the split is at
token 13.

| | token 13 | probability |
|---|---|---|
| paged executor | `,` (id 11) | 0.368005 |
| dense baseline | `.` (id 13) | ≈0.3616 |

Dense probabilities from the recorded logprobs (`.` −1.01726,
`,` −1.03508) give dense margin ≈0.0064; paged margin 0.002362.
Same top-5 id set {11, 13, 304, 323, 389}, same 3rd–5th order.
Positions 0–12: 13/13 self-consistent (paged token == argmax at every
position; margins 0.014–0.996). Float64 softmax check: argmax agrees
(11) in f64 and f32 — no flip; top-1/top-2 logit gap 0.006439.
Supporting check: token 14's margin is comfortable (0.226, argmax
` on` at 0.378), consistent with an isolated near-tie — but that is
not the divergent position and does not resolve it.

Verdict: NEAR-TIE, not a bug — same order as the 0.5B step-19 case
(margin 0.0022). Greedy decoding amplifies a coin flip into a visible
text difference.

### Residuals, stated plainly

1. No full-float64 forward shadow: the f64 check covered the softmax
   only, not the whole forward pass.
2. Byte-identity with stage-2's deleted 7B file is unprovable: no
   recorded hash for that file.
3. Colors prompt incomplete: speculative 12 tokens (5 rounds, 7/20
   accepted), plain 2 tokens. Both runs end without a recorded stop
   reason.
4. 7B model file identity missing from the surviving run files (see
   honesty note above).

### What remains unproven (with reasons)

1. Neural-draft acceptance at 110B scale — the 7B result does not
   transfer to a 110B claim.
2. Prefetch pipelining — designed in the design doc, not implemented
   (needs the `pager.py` stats-lock fix plus the async submit path).
3. GPU executor — design only, no GPU on this box; a Kaggle GPU
   kernel is the test route.
4. `forward_all`'s proper home on `PagedQwen2` — currently a copy in
   `speculative.py`, which is a drift risk against `forward`.

## Suite (2026-10-25, MEASURED)

239 passed, 2 skipped in 55.68 s
(`PYTHONPATH=~/workspace/lacuna <venv python> -m pytest
~/workspace/lacuna/tests/ -q`). The task brief expected 237; the
measured count is 239 (230 at `beb9ddc` + 9 new speculative tests) —
the brief's expectation was stale, the suite itself is green.
