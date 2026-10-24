# Lacuna Real-Weights Ladder — Measured Report

**Source:** Kaggle kernel `x0d4n7/lacuna-real-weights-ladder` (private, GPU, internet on), v2 source `kernel_src2.py`.
**Run:** started ~19:00 UTC 2026-10-23, status **COMPLETE** (verified 20:31 UTC). Total kernel wall time **5,874.8 s (≈ 1 h 38 min)**.
**Raw evidence:** `~/workspace/kaggle_ladder/final_out/results.jsonl`, `results.log`, `lacuna-real-weights-ladder.log`.
Every number below is copied from those files. Nothing is estimated or reconstructed from memory.

---

## Environment truth (measured in-kernel)

| Item | Measured value |
|---|---|
| GPUs present | 2× Tesla T4, 15,360 MiB each, 0 MiB used (nvidia-smi) |
| Driver / CUDA runtime | 580.178.04 / CUDA 13.0 |
| **nvcc** | **absent from the kernel image** → CUDA build skipped entirely |
| llama.cpp build used | **CPU source build** (`build-cpu/bin`), build time 336.7 s |
| Host RAM | 32,098 MB total |
| CPU | 4× Intel Xeon @ 2.00 GHz |
| Disk free at start | /tmp: 1,126.14 GB · /kaggle/working: 20.94 GB |
| huggingface_hub | 1.29.0, available |

**Consequence:** both T4 GPUs sat idle for the entire ladder. All generation below is **llama.cpp on CPU**. This is an environment fact of this run, not a property of Lacuna.

---

## Per-rung summary

| Rung | Model | Baseline (llama.cpp CPU dense) | Convert GGUF→.lac | Bit-exact reconstruction | Pager probe @ 512 MiB cap | Paged matvec vs dense |
|---|---|---|---|---|---|---|
| 1 | Qwen3-30B-A3B Q4_K_M | ✅ 5.52 tok/s | ✅ 115.5 s | ✅ 579/579 tensors, **0 mismatches** | ✅ 13/13 verified, budget respected | ❌ probe errored — **not measured** |
| 2 | Llama-3.3-70B Q4_K_M | ❌ server killed (rc=-9) | — never ran | — never ran | — never ran | — never ran |
| 3 | Command R+ 104B Q2_K | ✅ 0.0144 tok/s | ✅ 281.3 s | ⚠️ 641/642 tensors, **1 mismatch** | ⚠️ 11/13 verified, budget respected | ❌ probe errored — **not measured** |

End-to-end generation **through Lacuna's pager**: **NOT IMPLEMENTED IN LACUNA** (the kernel recorded this explicitly on every rung). Lacuna ships a pager + paged-buffer contract, not a transformer forward pass. All generated text in this report is llama.cpp dense output and is **not** Lacuna output.

---

## Rung 1 — Qwen3-30B-A3B-Instruct-2507, Q4_K_M (~30B MoE)

- **Repo/file:** `unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF` / `Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf`
- **Size:** 18,556,686,752 bytes (expected size match: ✅)
- **SHA-256:** `6c997b8af17debdfb01d890214400ccbab00db6acc0ba8da5de1cc906c4774d0` (matches expected: ✅)
- **Download:** 179.9 s @ 103.2 MB/s (first attempt short-downloaded at 12,589,930,569 bytes; range-resume retry completed it)
- **Baseline (llama.cpp CPU dense, llama-server, temp 0, seed 42):**
  - Prompt: `The capital of France is`
  - Verbatim output: `" Paris. The capital of the United States is Washington, D.C. The capital of the United Kingdom is London. The capital of Germany is Berlin. The capital"`
  - 32 tokens predicted, generation **5.52 tok/s** (5,797 ms), prompt eval 1,994 ms, wall 7.8 s
- **Convert:** 115.5 s → `.lac` 18,551,077,469 bytes; 579 tensors, precision policy `exact`, tiers: all 579 RAW (no q4/q2/ternary requantization)
- **Reconstruction (bit-exact check of .lac pages vs source GGUF bytes):** 579/579 tensors, 9,327 pages, 18,550,716,416 bytes checked, **0 mismatches**, 135.1 s
- **Pager probe (hot budget 512 MiB = 536,870,912 B):** 13/13 probe tensors (incl. `output.weight`, `token_embd.weight`, 11 `ffn_down_exps` tensors) re-read through the pager and hash-verified byte-identical; 1,080 page faults; max resident 536,716,921 B ≤ budget ✅
- **Tiled matvec probe:** **ERRORED before any comparison** — `BufferCapacityError: tensor 'blk.0.attn_q.weight' needs 18874368 bytes > capacity 4194304`. The probe sized the paged buffer at `W.nbytes // 8` (4,194,304 B), smaller than a single tensor it tried to allocate whole. **No paged-vs-dense diff exists for this rung.**

## Rung 2 — Llama-3.3-70B-Instruct, Q4_K_M (70B dense)

- **Repo/file:** `bartowski/Llama-3.3-70B-Instruct-GGUF` / `Llama-3.3-70B-Instruct-Q4_K_M.gguf`
- **Size:** 42,520,398,816 bytes (expected size match: ✅)
- **SHA-256:** `32df3baccb556f9840059b2528b2dee4d3d516b24afdfb9d0c56ff5f63e3a664` (matches expected: ✅)
- **Download:** 404.5 s @ 105.1 MB/s, no errors
- **Baseline: FAILED.** `llama-server` exited with **rc = −9 (SIGKILL)** ~347 s after launch while loading the 42.5 GB model on a 32 GB-RAM host — the signature of an OOM kill. The server log itself lived in the kernel's /tmp and was not captured, so "OOM" is an inference from the signal + sizes, not a logged fact.
- **Everything after the baseline never ran** (convert, reconstruction, probes). The rung then cleaned up its download.
- **Measured ceiling for this rung: host RAM** (32 GB) vs a 42.5 GB model under llama-server's loading path. Disk was never a constraint (1,125.78 GB free at precheck).

## Rung 3 — c4ai-command-r-plus (104B), Q2_K

- **Resolution:** the ladder's first two candidate repos no longer exist (`bartowski/CohereForAI_c4ai-command-r-plus-GGUF` and `MaziyarPanahi/c4ai-command-r-plus-GGUF` → RepositoryNotFoundError). Resolved at runtime to:
- **Repo/file:** `mradermacher/c4ai-command-r-plus-GGUF` / `c4ai-command-r-plus.Q2_K.gguf` — note the quant is **Q2_K**, because that repo exposed no single-file Q4_K_M/Q3_K_M
- **Size:** 39,497,390,944 bytes (size match vs repo metadata: ✅)
- **SHA-256:** `da2b4e5cd52bdb07e10c9b11af0a945361fbac0b27eb03bc829dd4028c5b8852` (computed in-kernel; no publisher hash existed to compare against)
- **Download:** 411.5 s @ 96.0 MB/s, no errors
- **Baseline (llama.cpp CPU dense):**
  - Prompt: `The capital of France is`
  - Verbatim output: `" Paris. Paris is located in northern France, in the middle of the Île-de-France region. Paris is France's largest city and its economic, political"`
  - 32 tokens predicted at **0.0144 tok/s** (predicted 2,223,887 ms), prompt eval 73,487 ms, wall 2,297.4 s (≈ 38.3 min)
- **Convert:** 281.3 s → `.lac` 39,054,189,487 bytes; 642 tensors, policy `exact`, all RAW
- **Reconstruction:** 642 tensors attempted, 18,946 pages, 39,053,471,744 bytes checked, 314.9 s — **1 mismatch**:
  - `token_embd.weight`: source 2,580,480,000 bytes vs .lac 2,147,479,552 bytes (short by 433,000,448 bytes). All other 641 tensors matched bit-exactly. (Observation only: 2,147,479,552 = 2³¹ − 4,096; the kernel output does not establish the cause.)
- **Pager probe (512 MiB cap):** 11 of 13 probe entries hash-verified (list: `token_embd.weight` ×2, 11 `ffn_down.weight` tensors); 2,996 faults in the probe loop; max resident 535,433,592 B ≤ 536,870,912 budget ✅. The output does not name which entries failed; the reconstruction result above makes `token_embd.weight` the consistent candidate.
- **Tiled matvec probe:** **ERRORED before any comparison** — `BufferCapacityError: tensor 'blk.0.ffn_gate.weight' needs 544997376 bytes > capacity 68124672` (same probe-sizing bug as rung 1). **No paged-vs-dense diff exists for this rung.**

---

## Limits — what this run does and does not establish

1. **All generated text is llama.cpp (CPU, dense).** No text was generated by Lacuna. End-to-end generation through Lacuna's pager is not implemented in Lacuna and was recorded as `NOT_IMPLEMENTED_IN_LACUNA` on every rung.
2. **What Lacuna's own machinery demonstrably did on real weights:**
   - Converted two real GGUFs (30B MoE, 104B) to `.lac` under the `exact` policy (all tensors stored RAW).
   - Reconstructed the 30B model's weights **bit-exactly** (579/579 tensors, 18.55 GB, 0 mismatches).
   - On the 104B model, reconstructed 641/642 tensors bit-exactly and exposed **one real defect/shortfall**: the 2.58 GB `token_embd.weight` came back 433,000,448 bytes short.
   - Served multi-GB tensors through the pager under a hard 512 MiB hot budget without exceeding it (max resident within 0.03% of the cap on both probed rungs).
3. **Paged-vs-dense matvec equality was NOT measured on any rung.** The probe crashed on a buffer-capacity sizing error (probe set capacity below one tensor's size) on rungs 1 and 3; rung 2 died earlier. Any max-abs-diff number would be fabricated — there is none.
4. **The 70B rung's ceiling was host RAM, not disk and not Lacuna:** llama-server was SIGKILLed loading a 42.5 GB model into a 32 GB-RAM host before Lacuna code ran at all.
5. **GPUs were never used:** the Kaggle image has no `nvcc`, so llama.cpp built CPU-only. GPU tok/s numbers do not exist in this run.
6. **Quantization note:** under the `exact` policy the `.lac` files store tensors RAW, so `.lac` size ≈ GGUF size (18.55 GB and 39.05 GB). This run says nothing about Lacuna's q4/q2/ternary tiers.

*Report written 2026-10-23 from the kernel's own output files; raw data in `~/workspace/kaggle_ladder/final_out/`. Recovery note: the original monitoring agent died mid-run; this report was produced by a recovery agent that took over polling, downloaded the outputs from the COMPLETE kernel, and extracted the results above.*
