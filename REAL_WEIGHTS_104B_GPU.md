# 104B GPU generation speed — measured on free Kaggle GPUs

**Kernel:** `x0d4n7/lacuna-104b-gpu-speed`, version 6, COMPLETE 2026-10-24 20:22:46 UTC.
Total wall: 2585.0 s (43.1 min). Raw results:
`~/workspace/kaggle_ladder/final_out_104b_gpu/results_104b_gpu.jsonl`
(generation transcripts: `gen_ngl999.txt`, `gen_ngl60.txt`, `gen_ngl40.txt`).

**What this is and is not:** this is **llama.cpp dense GPU generation**
(llama-server, CUDA build), measured on the same 104B file as
`REAL_WEIGHTS_LADDER.md` / `REAL_WEIGHTS_LADDER_R2.md`. It is **not**
Lacuna output: Lacuna's pager-in-the-loop generation remains unbuilt,
and nothing here was generated through the pager.

## Environment truth (this run)

- 2 GPUs — confirmed by this run's own VRAM poll (two devices reported
  for every config). The image family matches the earlier runs: 2x Tesla
  T4, 15 GiB each, driver 580.178.04 (v6's libcuda step extracted the
  matching 580.178.04 runfile against the loaded module), CUDA 12.8.93
  toolkit preinstalled (nvcc present — unlike the run 1 / R2 images,
  where nvcc was absent). Host RAM ~32 GB (run 1 / R2 measurement on
  this image family).
- Model file: `mradermacher/c4ai-command-r-plus-GGUF`,
  `c4ai-command-r-plus.Q2_K.gguf`, **39,497,390,944 bytes**,
  SHA-256 `da2b4e5cd52bdb07e10c9b11af0a945361fbac0b27eb03bc829dd4028c5b8852`
  — identical to the hash computed in run 1. Public repo, no HF token.
- Download: `snapshot_download` succeeded on attempt 1 in 337.0 s
  (~117 MB/s) to `/tmp/model104b`. The v4 Xet writer error
  (`File reconstruction error: Internal Writer Error`) did not recur —
  it was transient. Note: the model went to `/tmp`, not
  `/kaggle/working`, because the working directory is only a 20 GB loop
  device and cannot hold a 39.5 GB file.
- Fixed measurement: prompt `The capital of France is` (same as run 1),
  greedy decoding, 32 tokens requested.

## CUDA install route that worked

No `apt-get` toolkit install, no pip CUDA packages. Two steps:

1. **libcuda (the missing piece).** The image ships nvcc and libcudart
   but no `libcuda.so` anywhere (the stubs directory is empty), so no
   CUDA program can link or run. Fix: download the exact matching
   driver runfile
   `https://us.download.nvidia.com/XFree86/Linux-x86_64/580.178.04/NVIDIA-Linux-x86_64-580.178.04.run`,
   `--extract-only`, copy `libcuda.so.580.178.04` to
   `/usr/lib/x86_64-linux-gnu/`, symlink `libcuda.so.1` and `libcuda.so`,
   run `ldconfig`. Measured: 40.8 s, `libcuda.so.1` present afterwards.
2. **llama.cpp CUDA build.** `cmake -DGGML_CUDA=ON
   -DCMAKE_CUDA_ARCHITECTURES=75` (T4 = sm_75). Build succeeded in
   1627.4 s (~27 min); `llama-cli` / `llama-server` build `b1-78a8ae1`,
   CUDA backend included.

## Measured results

| Config | Result | Wall (load+gen+shutdown) | Max VRAM GPU0 / GPU1 | Verbatim output |
|---|---|---|---|---|
| `-ngl 999` (full offload) | **FAILED, rc=1** | 165.1 s | 105 / 105 MiB (nothing loaded) | none — server never became ready |
| `-ngl 60` | **FAILED, rc=1** | 163.0 s | 105 / 105 MiB (nothing loaded) | none — server never became ready |
| `-ngl 40` | **WORKED, rc=0** | 287.5 s | 12,591 / 14,299 MiB | `Paris<|END_OF_TURN_TOKEN|>` |

### Full offload does NOT work

- `-ngl 999` failed during load, verbatim from the log:
  `ggml_backend_cuda_buffer_type_alloc_buffer: allocating 18148.07 MiB
  on device 0: cudaMalloc failed: out of memory`,
  `failed to allocate CUDA0 buffer of size 19029626880`,
  `llama_model_load: error loading model: unable to allocate CUDA0 buffer`,
  `llama_server exited with code 1`.
- `-ngl 60` failed the same way:
  `allocating 16498.24 MiB on device 0: cudaMalloc failed: out of memory`,
  `failed to allocate CUDA0 buffer of size 17299660800`.
- There is **no fallback or CPU spill**: when the device-0 buffer for the
  offloaded layers exceeds one T4's 15 GiB, the server hard-fails
  instead of degrading. 39.5 GB of weights cannot fit in 30 GB of VRAM,
  and this server configuration does not bridge the gap.

### The only working config: `-ngl 40`

- Server started, model loaded from `/tmp/model104b/...gguf`
  (ftype Q2_K), both T4s nearly full (12.6 / 14.3 GiB).
- Exchange, verbatim from `gen_ngl40.txt`:
  - Prompt: `The capital of France is`
  - Output: `Paris<|END_OF_TURN_TOKEN|>`
  - Speeds printed by llama-server itself:
    `[ Prompt: 0.6 t/s | Generation: 0.7 t/s ]`
- The model emitted its end-of-turn token after a single content token
  (`Paris`), so the 32 requested tokens were never produced — the run
  ended early on the model's own stop token, not on a timeout or error.
  The 0.7 tok/s figure therefore covers ~2 generated tokens and is
  noisy, but it is the only generation-speed figure this run produced,
  and it is llama.cpp's own timing, not an estimate.
- Wall 287.5 s covers server start + 39.5 GB model load + generation +
  shutdown; it is not a pure generation time.

## Verdict: can a 104B model generate at usable speed on free hardware?

**No — not on 2x free T4s with this setup.** The only configuration
that loads at all (`-ngl 40`, 40 layers on GPU, the rest on CPU) runs
at **0.7 tok/s** by llama.cpp's own display, with prompt processing at
0.6 tok/s — slower than interactive use, and measured on a
single-token completion. Full offload is impossible (39.5 GB > 30 GB
VRAM) and fails hard rather than spilling. For comparison, the same
model's CPU-only dense baseline in run 1 was 0.0144 tok/s, so the GPU
split is ~49x faster than CPU dense — and still not usable.

This is precisely the gap Lacuna's pager exists to break (stream
weight pages from NVMe ahead of compute instead of demanding full
residency), but that generation path is still unbuilt: every token in
this report came from llama.cpp dense GPU/CPU execution, not from
Lacuna. The honest status is: 104B loads and generates on free GPUs,
at unusable speed, with full offload impossible.

## Version history of this kernel (for the record)

- v1: CUDA 12.8 preinstalled, but the CUDA build failed at link time.
- v2: configure failed — `CUDA::cuda_driver` target missing.
- v3: proved no `libcuda.so` exists anywhere on the image.
- v4: libcuda extraction + CUDA build both succeeded (1627–1702 s),
  but the run errored in the download stage on a transient HuggingFace
  Xet writer error.
- v5: accidental re-push of the v4 notebook (script-edit error on my
  side); errored at 19:37 UTC as predicted (same download-stage fault).
- v6: download moved to `/tmp` with retries + a plain-HTTPS curl
  fallback; snapshot_download succeeded on attempt 1 and the run
  completed. The curl fallback was not needed.
