# Lacuna omega-beat audit — 2026-10-26 (subagent run, Lacuna only)

Scope: beat Colibri AND AirLLM by measured runs, not claims.
XIIV was not touched. Nothing in this audit was pushed.

## Correction to the starting premise

The task brief said GPU Engine wiring was MISSING and MoE was
DESIGN ONLY. That was true of committed HEAD (8dc1f5c) and false
of the working tree on 2026-10-26: the working tree already
contains, uncommitted, GPU Engine wiring, a CPU PagedMoE executor,
a GPU MoE path, Gemma-4 executors, and their tests. This audit
is against the working tree, not HEAD. `docs/GPU_EXECUTOR_DESIGN.md`
is stale in the same way (it still says full-model generation is
unwired); do not quote it over the code.

Serena: project `lacuna` is registered at
`~/workspace/lacuna/.serena/project.yml` (python LSP). This
subagent shell has no MCP transport for Serena's symbol tools,
so the audit used the registered project structure plus direct
code reading (AST symbol lists, grep, test runs) — not faked
Serena calls.

## Audit findings (verified, not assumed)

1. GPU Engine wiring — PRESENT in worktree, ABSENT in HEAD.
   `lacuna/engine.py: _try_gpu`, `Engine(path, gpu="cuda"|"auto"|"numpy")`
   -> mode `lacuna-gpu`, Gemma-4 dispatch to `GPUGemmaModel`,
   `gpu="cuda"` raises instead of silently degrading.
   Kill gate: `tests/test_gpu_generation.py::test_engine_gpu_mode_end_to_end`
   and `tests/test_gpu_moe.py::test_engine_moe_dispatch` — both pass
   on CPU with the numpy backend in this run (see below).

2. K-quant GPU kernels — PARTIAL before this run, CLOSED for
   Q2_K/Q3_K/Q5_K/Q6_K in this run.
   Before: torch kernels only for F32/F16/BF16/Q8_0/Q4_K; all other
   K types fell back to CPU dequant + upload (exact, slow). The
   110B proof file is Q2_K, so that fallback directly blocked a
   fast GPU 110B run.
   Now: `TorchBackend.dequantize` has native torch kernels for
   Q2_K, Q3_K, Q5_K, Q6_K, mirroring `lacuna/gguf.py` block
   layouts. Only Q5_0 remains a CPU fallback.
   Kill gate: new `tests/test_gpu_kquant.py` — parity vs gguf
   with exact equality (equal_nan only for random-byte f16 scales
   that are themselves Inf/NaN), 1 and 3 blocks each, plus a
   non-multiple rejection test. 9 passed in this run.

3. MoE executor — PRESENT in worktree (CPU + GPU), synthetic
   proof only.
   `lacuna/moe.py: PagedMoE` (qwen3moe/olmoe, softmax->top-k->
   renormalize, grouped expert GEMM, QK-norm, shared experts,
   demand-only expert paging, fail-loud on DeepSeek/MLA).
   `lacuna/gpu_executor.py: GPUModel` MoE mode reuses the same
   router rule; CPU MoE is the reference.
   Kill gate (design doc §10): Stage 0 synthetic tests pass;
   Stage 1 real-model token identity is NOT done — no 7B-class
   MoE GGUF exists on this box (only synthetic tiny files and a
   vocab file were found). Real proof needs a downloaded
   OLMoE-1B-7B (lead) or Qwen1.5-MoE-A2.7B (fallback) file and a
   Kaggle run. Do not claim it from synthetic tests.

4. Prefetch at 110B — SPLIT, not one gap.
   CPU paged path: next-layer prefetch implemented
   (`lacuna/paged_model.py`, `lacuna/pager.py` async prefetch).
   GPU path: `GPUModel.generate` reads each layer on demand;
   there is NO cross-layer prefetch in `lacuna/gpu_executor.py`
   (no `prefetch` call in that file). Disk latency on the GPU
   path is therefore not overlapped. This was NOT fixed in this
   run: prefetching a whole next layer onto the device would
   break the VRAM budget the GPU path is built around. The
   safe form is CPU-side page prefetch for the next layer
   while the current layer computes — design first, then code,
   with a budget test. Kill gate for that future work: same
   token ids with prefetch on/off, and measured (not claimed)
   time improvement on a real model.
   Separately, `lacuna/prefetch.py` (NgramPrefetcher,
   CacheExpertPredictor, ShadowScorer) is measurement/prediction
   primitives, not disk prefetch — do not conflate them.

5. 6GB floor — UNMEASURED. 110B dense token-identity — UNPROVEN.
   Both remain as stated in the brief. No new measurement in
   this run changes either; the 110B run is a Kaggle job, not a
   this-box job (see harness guard).

6. Head-to-head harness — BUILT in this run.
   `kaggle_gpu/head_to_head.py`: same model, same hardware,
   same prompt, same max_tokens for Lacuna / AirLLM / Colibri.
   Rules enforced in code: an absent engine is `not_available`,
   never 0 tok/s; per-layer latency is never converted to tok/s;
   Colibri numbers are only merged from a supplied measured
   result JSON (`--colibri-result`), never hand-typed; a
   >20 GB model on a non-CUDA box is refused (exit 2) unless
   `--allow-local-110b` is passed. Verified: `--dry-run`
   writes the full schema with all engines not_available;
   a 21 GB sparse file is refused with no output file.

## Test results (this run, CPU box, no CUDA)

- With torch-CPU venv (workspace temp venv, outside the repo):
  `tests/test_gpu_executor.py tests/test_gpu_generation.py
  tests/test_moe.py tests/test_gpu_moe.py tests/test_gpu_gemma.py
  tests/test_gguf.py tests/test_gpu_kquant.py`
  -> **89 passed, 1 skipped** (CUDA-only test, correctly marked).
  Without torch: 76 passed, 5 skipped (torch tests skipped, not failed).
- MoE subset alone: `tests/test_moe.py tests/test_gpu_moe.py`
  -> **22 passed**.
- No test was edited to make a failing result pass, except the
  Q6_K parity test's NaN handling (see §2): random f16 scales
  can be Inf/NaN, so equality is checked with equal_nan=True;
  value equality elsewhere is still exact.

## What remains blocked, with evidence

- CUDA execution: no GPU on this box (`torch.cuda.is_available()
  == False`, no nvidia-smi). All torch-CUDA claims need a Kaggle
  GPU kernel run of `kaggle_gpu/gpu_executor_test.py` and of
  `kaggle_gpu/head_to_head.py`. The Kaggle script already exists
  and is not modified by this run.
- Real-model MoE proof: no 7B-class MoE file on this box.
  Next command (Kaggle, after download): run the proof model
  through `Engine(gguf, gpu="cuda")` and through `PagedMoE`,
  compare greedy token ids, and audit that no unselected expert
  was faulted — the design doc §10 gates, not a new invention.
- Real-model beat vs Colibri/AirLLM: no measured beat exists
  yet. The harness above is the instrument; the beat is not
  claimed until it produces a measured JSON with all three
  engines on the same hardware.
- Working-tree vs HEAD: the GPU/MoE/Gemma stack above is
  uncommitted (8 modified files + 15 untracked files at audit
  time, plus the 2 files added here). Nothing was pushed, and
  no commit was made by this run, per the hard rule. The parent
  must decide commit/push — a force-push would resurrect the
  20 agent-tooling files remote's Oct 24 cleanup removed, so
  merge, not force-push, is the safe route (see the separate
  push-reconciliation evidence already in main chat).
- Q5_0 torch kernel: still a CPU fallback. It is not on the
  110B Q2_K path, so it is a lower priority than the
  GPU-prefetch design in §4.
