# Lacuna Real-Weights Ladder — Run 2 (Re-verification) Report

**Source:** Kaggle kernel `x0d4n7/lacuna-real-weights-ladder` (private, GPU, internet on), **version 3**, source `~/workspace/kaggle_ladder/kernel_src_r2.py`.
**Run:** pushed 2026-10-23 ~20:46 UTC, status **COMPLETE**. Total kernel wall time **3,897.1 s (≈ 65 min)**.
**Raw evidence:** `~/workspace/kaggle_ladder/final_out_r2/results_r2.jsonl`, `results_r2.log`, `lacuna-real-weights-ladder.log`.
**Purpose:** close the three open items left by run 1 (`REAL_WEIGHTS_LADDER.md`) — the 70B rung's Lacuna-side proof, the 104B embedding re-check after the truncation fix, and the paged-vs-dense matvec comparison that run 1's probe never measured. Run-1 numbers are cited only as run-1; everything below was measured in this run. Nothing is carried over as assumed.

---

## Code under test (verified in-kernel)

| Item | Value |
|---|---|
| Lacuna commit | `4a8dd6813ca98db106680b8fd88fd7123e595aef` (contains the `_pread_full` truncation fix) |
| Package zip SHA-256 | `1f5acb0e37f708e2320f0b8341c5c36eae6e10016d0d4dfb776103c844a60e83` (in-kernel hash of the decoded zip; matches the locally built zip) |
| Fix present in the code that ran | ✅ kernel asserted `"_pread_full" in lacuna/convert.py` → `true` |
| Python | 3.13.15 |

## Environment truth (measured in-kernel)

| Item | Measured value |
|---|---|
| GPUs present | 2× Tesla T4, 15,360 MiB each, 0 MiB used (nvidia-smi) |
| **nvcc** | **absent from the kernel image** (same as run 1) → no CUDA build |
| llama.cpp build | CPU source build, 351.5 s — built **only** for rung A's optional llama-cli bonus |
| Host RAM | 32,098 MB total |
| Disk free at start | /tmp: 1,126.14 GB · /kaggle/working: 20.94 GB |
| huggingface_hub | 1.29.0 |

**Consequence, stated plainly:** this run is about **correctness closure, not GPU speed**. Both T4s sat idle again; every number below is CPU-side. No GPU tok/s exists in this run.

---

## Per-rung summary

| Rung | Model | Convert GGUF→.lac | Reconstruction | Pager probe @ 512 MiB cap | Paged matvec vs dense (identical reduction partition) |
|---|---|---|---|---|---|
| A | Llama-3.3-70B Q4_K_M | ✅ 319.5 s | ✅ **724/724 tensors, 0 mismatches** | ✅ 13/13 verified, budget respected | ✅ 3 tensors, **max abs diff 0.0, bit-identical** |
| B | Command R+ 104B Q2_K | ✅ 305.0 s | ✅ **642/642 tensors, 0 mismatches** — `token_embd.weight` full-length & bit-exact | ✅ 13/13 verified, budget respected | ✅ 3 tensors, **max abs diff 0.0, bit-identical** (vs both dense references) |
| C | Qwen3-30B-A3B Q4_K_M | ✅ 131.6 s | ✅ **SPOT re-check: 25/25 tensors, 0 mismatches** (full proof is run 1's, not re-claimed) | — not in this rung's scope | ✅ 3 tensors, **max abs diff 0.0, bit-identical** |

End-to-end generation **through Lacuna's pager**: still **NOT IMPLEMENTED IN LACUNA** (recorded explicitly in-kernel again). Nothing in this report is end-to-end Lacuna generation.

---

## Rung A — Llama-3.3-70B-Instruct, Q4_K_M (70B dense) — Lacuna-side only

- **Repo/file:** `bartowski/Llama-3.3-70B-Instruct-GGUF` / `Llama-3.3-70B-Instruct-Q4_K_M.gguf`
- **Size:** 42,520,398,816 bytes (match ✅) · **SHA-256:** `32df3baccb556f9840059b2528b2dee4d3d516b24afdfb9d0c56ff5f63e3a664` (**re-verified this run** against the run-1 value ✅)
- **Download:** 312.4 s @ 136.1 MB/s, no errors
- **Convert (fixed code):** 319.5 s → `.lac` 42,513,323,248 bytes; 724 tensors, precision policy `exact`, tiers: all 724 RAW
- **Reconstruction (FULL, every tensor bit-exact vs source GGUF bytes):** **724/724 tensors, 20,895 pages, 42,512,531,712 bytes checked, 0 mismatches**, 347.1 s
  - `token_embd.weight`: source 591,003,648 B → .lac 591,003,648 B, **match** ✅
  - **Largest tensor in the model:** `output.weight`, 861,880,320 B — checked, match ✅
  - **Any single tensor over 2 GiB? No.** The model's largest tensor is 0.86 GB; the >2 GiB case does not occur in this model (it occurs in rung B, below).
- **Pager probe (hot budget 512 MiB = 536,870,912 B):** 13/13 probe tensors (12 largest + `token_embd.weight`) re-read through the pager and hash-verified byte-identical, **0 failed**; 1,895 page faults; max resident 536,827,103 B ≤ budget ✅
- **Matvec probe (corrected):** see the consolidated matvec table below. All 3 tensors: paged output **bit-identical** to the dense reference with the identical reduction partition (diff exactly 0.0); residency guarantee held on every tensor.
- **OPTIONAL BONUS (not a Lacuna measurement): llama-cli with mmap defaults, 8-token limit, 1,500 s time-box.** Result: **rc = −9 (SIGKILL) after 656.1 s** — killed before producing any token (captured stdout is only llama-cli's loading spinner; stderr empty). Interpretation, labeled as such: even the RAM-conscious mmap CLI path could not hold this 42.5 GB model on the 32 GB-RAM free-tier host. Run 1's measured ceiling for dense *generation* (host RAM) stands, now reproduced with the CLI as well as the server. This does not affect the Lacuna-side results above, which are streaming and never held the model in RAM.

## Rung B — c4ai-command-r-plus (104B), Q2_K — the fix, re-verified on the real file

- **Repo/file:** `mradermacher/c4ai-command-r-plus-GGUF` / `c4ai-command-r-plus.Q2_K.gguf` (same resolution as run 1; quant is Q2_K because that repo exposes no single-file Q4/Q3)
- **Size:** 39,497,390,944 bytes (match ✅) · **SHA-256:** `da2b4e5cd52bdb07e10c9b11af0a945361fbac0b27eb03bc829dd4028c5b8852` (matches the hash computed in run 1 ✅)
- **Download:** 385.3 s @ 102.5 MB/s, no errors
- **Convert (fixed code):** 305.0 s → `.lac` 39,487,197,557 bytes; 642 tensors, policy `exact`, all RAW
- **Reconstruction (FULL):** **642/642 tensors, 19,152 pages, 39,486,472,192 bytes checked, 0 mismatches**, 319.4 s
  - **`token_embd.weight`: source 2,580,480,000 bytes → .lac reconstruction 2,580,480,000 bytes, hash match ✅ — FULL LENGTH, BIT-EXACT.**
    Run 1 (pre-fix code) reconstructed this tensor at 2,147,479,552 bytes — short by 433,000,448 bytes. The truncation is **gone** under the fixed code, on the same file, verified byte-for-byte. This tensor is also the model's largest and its only tensor over 2 GiB.
- **Pager probe (512 MiB cap):** 13/13 verified, **0 failed** (run 1: 11/13, with `token_embd.weight` the consistent suspect — it now verifies through the pager as well); 3,408 faults; max resident 536,267,677 B ≤ budget ✅
- **Matvec probe (corrected):** all 3 tensors **bit-identical (diff exactly 0.0) against both** dense references — see table.

## Rung C — Qwen3-30B-A3B-Instruct-2507, Q4_K_M (~30B MoE) — matvec + spot re-check

- **Repo/file:** `unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF` / `Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf`
- **Size:** 18,556,686,752 bytes (match ✅) · **SHA-256:** `6c997b8a…
...[truncated 7020 chars]