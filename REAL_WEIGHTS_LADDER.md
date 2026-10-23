# Lacuna — Real-Weights Ladder (Kaggle GPU, measured)

Status: IN PROGRESS — kernel `x0d4n7/lacuna-real-weights-ladder` running.
This file is rewritten with measured results when the kernel completes.
Nothing below the environment section is a result yet.

## Environment (measured, env-truth kernel `x0d4n7/lacuna-env-truth`, 2026-10-23)

- GPU: 2× Tesla T4, 15,360 MiB each (driver 580.178.04, CUDA 13.0)
- RAM: 32,098 MB total
- CPU: 4× Intel Xeon @ 2.00 GHz
- Disk at session start: overlay with ~1.1 TB avail shown by `df -h`
  (per-step free space is recorded in the ladder results as it changes)
- HF download probe: 10 MiB in 0.45 s = 23.3 MB/s
- Python 3.13.15, numpy 2.1.3 preinstalled

## Ladder

- Rung 1: Qwen3-30B-A3B-Instruct-2507 Q4_K_M (unsloth) — 18,556,686,752 B
- Rung 2: Llama-3.3-70B-Instruct Q4_K_M (bartowski) — 42,520,398,816 B
- Rung 3: Command R+ 104B (repo/quant resolved at runtime by disk + availability)
