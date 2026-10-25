# GPU executor core — design

**Status (2026-10-25): core built, CPU-validated, CUDA unvalidated.**
`lacuna/gpu_executor.py` implements a single Qwen2 layer forward with
pager-driven weight streaming and a hard VRAM budget. Full-model
generation and `Engine` wiring are NOT in this file — they come later
via the parent. Nothing here claims a GPU result: every GPU number in
this repo is either measured on a stated machine or labeled as what it
is.

## Why a GPU path at all

Plain paged dense 110B on CPU measured ~36 min/token (see
`PAGED_GENERATION_PROOF.md` App. B) and is dequant-bound: every token
re-dequantizes every tensor in numpy. Prefetch overlaps disk reads with
compute, but it cannot remove the CPU dequant cost. The GPU path is the
structural fix: upload raw quantized bytes per matrix, dequantize on
the device, run the GEMMs there, free the matrix. Dequant and GEMM —
the two costs that dominate the CPU number — both move to the device.

For comparison, llama.cpp's dense GPU split on 2x T4 measured 0.7
tok/s on 104B (`REAL_WEIGHTS_104B_GPU.md`) — and hard-failed whenever
weights did not fit VRAM, because it has no spill path. Lacuna's GPU
executor is the spill path: it never requires residency, so a 4 GB
floor GPU is a supported target, not a configuration that fails.

## Backend decision — with evidence

Checked 2026-10-25 on this box (commands and raw results):

| Check | Result |
|---|---|
| `nvidia-smi` | not installed; no `/dev/nvidia*` devices |
| system python3.12 | no torch, no cupy (both `ModuleNotFoundError`) |
| `~/workspace/ai-tools/corsac/.venv` python3.12 | **torch 2.14.1+cpu**, `torch.cuda.is_available()` False, CPU tensors work |
| `~/workspace/kaggle_ladder/.venv` | no torch / cupy / nvidia packages found in site-packages |
| Kaggle GPU kernel (prior run, `REAL_WEIGHTS_104B_GPU.md`) | 2x Tesla T4, 15 GiB each, driver 580.178.04, CUDA 12.8.93 toolkit + nvcc preinstalled, but **no `libcuda.so`** anywhere until the matching driver runfile was extracted (40.8 s fix, documented there) |

Decision:

- **Primary backend: PyTorch CUDA.** It is the only GPU tensor library
  with a realistic chance of already being present on a Kaggle GPU
  kernel, and its CPU build lets the kernels' *math* be validated here
  before any GPU run. Torch is not verified present on Kaggle images
  by this author — the Kaggle script probes for it and reports, and
  only installs it when explicitly asked (`--install-torch`).
- **Fallback backend: numpy** (`NumpyBackend`). Same layer math
  through the same code path, CPU execution. This is what runs on this
  box and what the test suite exercises.
- **NOT implemented: CuPy.** A backend that cannot be executed
  anywhere available would be fake breadth. CuPy stays a documented
  fallback candidate. If a future kernel has cupy but not torch, the
  `Backend` interface is the porting surface (one class, same layer
  code).
- **NOT used: raw CUDA/C++.** No nvcc exists on this box to compile
  or test it. A Kaggle kernel does have nvcc, so a raw-CUDA dequant
  kernel is a possible later optimization — after the torch path is
  measured, not before.

## Data flow (one layer, one token)

```
Pager (.lac pages) ──read_raw_tensor_bytes──> raw GGUF bytes (CPU)
        │  exact policy, length-checked, no direct file reads
        ▼  one matrix at a time, in phase order
Device: raw bytes ──torch dequant kernel──> float32 matrix
        │  F32/F16/Q8_0/Q4_K on device; other types CPU-dequant + upload
        ▼
Device: GEMM (+ bias, norm, RoPE, attention, SwiGLU)
        │  matrix freed before the next phase materializes anything
        ▼
Next phase / next layer. Only the KV cache survives across calls.
```

Phase order inside a layer (this is the streaming discipline, and the
budget enforces it):

1. **Attention phase** — attn_norm, Q/K/V (+ biases), attn_output.
   Live weights: the attention matrices only.
2. **MLP phase, one matrix at a time** — ffn_norm (tiny), then gate
   (activation kept: T x ffn floats, not weights), then up, then
   down. Live weights: one MLP matrix at a time.

Gate and up are deliberately NOT fused on this path, although the CPU
path fuses them: a fused gate+up matrix at 110B scale is 3.22 GB
float32 and would leave almost no headroom under a 4 GB budget.

## Floor arithmetic (110B-class layer: hidden 8192, ffn 49152, 64 Q / 8 KV heads, Q4_K)

| Quantity | Value | How |
|---|---|---|
| Stored bytes / layer | ~0.75 GB | 1358M params x 144/256 |
| Whole layer, float32 | ~5.43 GB | 1358M params x 4 B — does NOT fit 4 GB |
| Largest single matrix, float32 | 1.61 GB | 49152x8192 MLP matrix |
| Fused gate+up, float32 | 3.22 GB | why fusion is refused on this path |
| Attention phase live weights | ~0.60 GB | Q+out 67M x2, K/V 8.4M x2, norms/biases |
| MLP phase live weights | 1.61 GB | one matrix at a time |
| KV cache, all 80 layers | ~0.63 MiB / token | 2 x 8 heads x 128 dim x 4 B x 80 |
| Headroom on a 4 GB budget | ~2 GB | after the largest matrix + a 1k-token cache (~0.63 GB) |

These are arithmetic from the shapes, not measurements — the Kaggle
test measures the real per-layer latency that turns them into tok/s.

## Exactness argument — and its honest limit

The layer math is the same operations in float32 as the CPU path, in
the same order: RMSNorm, fused-QKV-equivalent projections, RoPE in the
verified half-rotation form, GQA repeat, causal attention with a
float32 softmax, SwiGLU. Dequant kernels mirror `lacuna/gguf.py`
block-for-block (same layouts, same float32 arithmetic order) for
F32/F16/Q8_0/Q4_K; other types take the CPU dequantizer's exact
output and upload it, so no type is ever run through a lossier path
than the CPU executor uses.

The limit, stated plainly: a GPU GEMM (cuBLAS) need not reduce in the
same order as a CPU BLAS, so **bit-identical output is not promised**.
What is promised is the same greedy choice except at near-ties — the
exact class of divergence the CPU paged-vs-dense comparison already
shows (a 0.0022 probability margin flipping one token at 0.5B, and a
0.0024 margin at 7B). The Kaggle test therefore measures layer-output
parity (max abs diff) AND, once full generation is wired, token parity
with margins reported for any flip. A flip at a near-tie is a
floating-point fact to report, not a bug to hide; a flip at a wide
margin is a bug.

A float16 compute mode would be faster and is explicitly NOT an
exactness mode: it is not built here, and if it is ever added it must
be opt-in and labeled non-exact. Nothing in this file downcasts.

## What the Kaggle GPU test measures

`kaggle_gpu/gpu_executor_test.py` (run on a GPU kernel; see its header
for exact launch commands). It does NOT claim full-model generation —
the executor is single-layer, so the test measures what a single layer
can honestly show:

1. **Environment**: torch/CUDA presence, device name, VRAM, and the
   libcuda diagnostic (with the documented fix available behind a
   flag, never applied silently).
2. **Kernel parity**: torch dequant (Q8_0, Q4_K) vs `lacuna.gguf`
   dequant on identical random blocks — max abs diff, must be ~0.
3. **Layer parity**: torch-CUDA layer output vs numpy layer output on
   identical synthetic weights — max abs diff + allclose verdict.
4. **Layer latency**: upload + dequant + forward seconds per layer
   for tiny, 0.5B-scale, and 110B-scale synthetic layers, with the
   VRAM peak. The 110B-scale figure extrapolates to tok/s ONLY as
   `80 layers x per-layer seconds` with the extrapolation labeled as
   such — it is not a generation measurement.
5. **Real-weight layer parity** (only with `--gguf <file>`): one real
   layer's weights through the pager, GPU vs CPU layer output, max abs
   diff. Token-vs-CPU-paged-ids comparison is defined but NOT runnable
   until full-model GPU generation is wired: the criterion will be
   N generated tokens identical to the CPU paged run, with any flip
   reported with its top-2 probability margin.

Pass criteria for the Kaggle run: (a) environment reported, (b) dequant
parity max diff <= 1e-6, (c) layer parity max diff <= 1e-4 on synthetic
weights (cuBLAS reassociation allowance; tighter is better and the
measured number is what gets recorded), (d) 110B-scale layer latency
and VRAM peak recorded, (e) no silent fallback: if CUDA is absent the
script exits nonzero saying so.

## Kaggle GPU test results (2026-10-25, measured)

Kernel `x0d4n7/lacuna-gpu-executor-test` (private, GPU-enabled),
status COMPLETE. Raw output kept out of this repo; numbers below are
copied from its `gpu_results.json`.

Environment: 2x Tesla T4, torch 2.11.0+cu128, CUDA available, no
install/fix steps needed (torch and CUDA were preinstalled and
working). CuPy 14.0.1 was also present on the image — noted for the
gap-4 decision, not used here.

- Dequant parity (GPU vs `lacuna/gguf.py`): Q8_0 max diff 0.0, Q4_K
  max diff 0.0 — both pass (criterion <= 1e-6).
- Layer parity vs numpy: tiny layer diff 0.0 (pass); 0.5B-scale layer
  diff 2.384e-07 (pass, criterion <= 1e-4); real-weight 0.5B layer-0
  diff 2.384e-07 (pass).
- Layer latency (synchronized, mean after warmup): tiny 0.001917
  s/layer, peak VRAM ~8.1 MiB; 0.5B-scale 0.032668 s/layer, peak
  ~24.8 MiB; 110B-scale synthetic (Q4_K) 0.979421 s/layer, peak VRAM
  2,789,671,936 B (~2.60 GiB).
- NOT a generation measurement: the 110B-scale 80-layer extrapolation
  (0.0128 tok/s) is arithmetic on a synthetic single-layer timing, not
  tokens generated. It sits below Colibri's 0.05-0.1 tok/s cold floor
  and is not comparable to it. 110B-scale GPU-vs-numpy parity was
  skipped by design this pass. Full-model generation is still
  unwired, so no GPU generation speed is claimed.

## Gaps (not hidden, in priority order)

1. **Torch dequant kernels for Q2_K/Q3_K/Q5_K/Q6_K/Q5_0.** CPU
   fallback keeps them exact but not fast. The 110B file is Q2_K /
   Q3_K / Q5_K heavy, so these kernels are the difference between a
   fast demo layer and a fast full model. Port order: Q6_K (simplest
   of the K family after Q4_K), then Q2_K/Q3_K/Q5_K.
2. **Full-model generation + Engine wiring.** Single-layer forward
   does not generate tokens. Wiring must reuse the pager's
   `read_raw_tensor_bytes` per layer, the sliced embed/head pattern
   from `paged_model` (never materialize either whole), and the KV
   cache across layers — then the token-parity criterion above
   becomes runnable.
3. **Page-by-page upload.** Currently a matrix's raw bytes are
   concatenated on CPU, then uploaded in one H2D copy. Uploading page
   by page (2 MB chunks) would overlap H2D transfer with dequant and
   cut the CPU-side staging buffer.
4. **CuPy backend** (see decision above) and, later, a raw-CUDA
   dequant kernel — only after the torch path is measured, so there is
   a baseline to beat.
5. **Speculative + GPU combination.** The proven 2.6x speculative
   win (7B, CPU) and the GPU path multiply: fewer target passes, each
   faster. That combination is the route to the >= 0.1 tok/s
   110B-class target from the Colibri beat-table — it is a plan, not a
   result, until measured.

## Files

- `lacuna/gpu_executor.py` — backends, dequant kernels, WeightRef,
  raw-bytes pager read, phased layer, budget-enforcing executor.
- `tests/test_gpu_executor.py` — 14 passed, 1 skipped (CUDA-only test)
  on this box (corsac venv: pytest 9.1.1, torch 2.14.1+cpu).
- `kaggle_gpu/gpu_executor_test.py` — the GPU-kernel test script.
