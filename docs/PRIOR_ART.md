# Prior art — 100B-class inference on tiny hardware

Sweep completed 2026-10-16. All figures are as reported by their sources (index-grade;
independent verification not done — see caveats). This file is the novelty evidence
for Lacuna: what exists, and the exact gap it fills.

## What exists

**llama.cpp** (ggml-org/llama.cpp) — mmap'd GGUF, k-quants + i-quants (IQ1_S = 1.5
bits/weight is the most aggressive; IQ3_S = 3.4375 bpw ≈ Q4_K quality), partial GPU
offload (`-ngl`, `--override-tensor`, `--cpu-moe`), Flash Attention, speculative
decoding. v0.5.0 (~Sep 2026) added Metal MoE fusion, Qwen4Exp sparse flash attention,
`--fuse-qkv` — but NO new quantization formats and NO orchestrated disk/NVMe weight
offload. Disk KV-cache checkpointing (`--cache-disk`) is still open issue #20697.
Every NVMe-tier system is a fork or a separate engine.

**Ollama** — embeds a llama.cpp fork and delegates all inference to it. Auto partial
GPU offload, GGUF only. Hard limit: the model must fit in VRAM+RAM; no disk offload.
Their own hardware table puts 70B at 40 GB VRAM (80 GB RAM fallback).

**FlexGen** (UC Berkeley) — linear-programming placement across GPU/CPU/disk plus
4-bit weight/KV compression. OPT-175B: 0.69 tok/s (disk offload, batch 256) /
1.12 tok/s (compressed, batch 144) on a 16 GB T4. Throughput-oriented (big batches),
not interactive latency.

**PowerInfer** — power-law activation locality: hot neurons pinned on GPU, cold on
CPU, per-layer adaptive MLP predictors (hidden dim tuned by sparsity/skewness,
≥95% prediction accuracy, predictors ≈10% of params). ReLU/ReGLU models only;
~80–95% MLP sparsity. Measured 13.20 tok/s average, 29.08 peak on RTX 4090 including
OPT-175B; 11.69x vs llama.cpp. **PowerInfer-2** (smartphone SoCs): neuron clusters +
next-layer prefetch fully overlapped with compute; 11.7 tok/s for Llama-7B on
Snapdragon 8 Gen 3.

**EXL2 / GPTQ / AWQ** — EXL2 supports 2–8 bits/weight fractional with per-layer mixed
precision and the fastest single-GPU tok/s (13B at 4.0 bpw: PPL 4.376 @ 56.78 tok/s
vs Q4_K_M 4.333 @ 30.83). Practical post-training floor ≈ 3–3.5 bpw; GPTQ at 2-bit
collapses (7B PPL 9.72 vs 5.68 FP16).

**MoE offloading** — Qwen3-235B-A22B = 235B total / 22B active; Q4_K_M ≈ 135 GB.
BELLS (llama.cpp fork, 24 GB card): 235B Q2_K at 3.10 → 5.61 tok/s (11.3 with pinned
host memory). ktransformers: 671B DeepSeek-R1 FP8 on 4090D + 382 GB DRAM: 286 tok/s
prefill / 14 tok/s decode. MoE-Lightning: 10.3x vs SOTA offloading on 16 GB T4.
**HOBBIT** (built on llama.cpp, +8k lines): mixed-precision experts, token-level
importance-driven FP16→INT4 / INT8→INT2 replacement, 2.3–18.9x over baselines with
≤1% accuracy loss — the closest single prior art to Lacuna.

**BitNet b1.58** — ternary 1.58-bit weights, trained from scratch; matches FP16
perplexity from 3B (9.91 vs 10.04). The 2B4T variant is 0.4 GB at 29 ms/token on CPU.
Post-training quantization of existing models to 1–2 bits collapses — 1-bit is only
viable when trained.

**Speculative decoding** — EAGLE-3 up to 6.5x speedup (acceptance 0.6–0.8 vs ~0.4 for
vanilla draft models); real-world 1.66x (Qwen3-4B) to 5.6x (LLaMA3.1-8B EAGLE-3).

**Honest ceiling on ≤4 GB VRAM** — AirLLM (70B streamed in 4 GB): seconds *per
token*. oLLM (80B BF16 → 8 GB GPU): ≈0.5 tok/s. SSD-LLaMA (trillion-param MoE on a
consumer PC): 1+ tok/s. Cambricon-LLM (in-flash compute): 3.44 tok/s @ 70B. **No
measured system exceeds ~5 tok/s for 100B+ models on ≤4 GB VRAM with quality held.**

**NVMe paging lineage** — FlexGen, PowerInfer-2, AirLLM, Apple "LLM in a flash"
(windowing + row-column bundling), oLLM (GPUDirect/KvikIO), HOBBIT, SSD-LLaMA,
NVIDIA ICMSP (CES 2026, NVMe KV cache). No published LLM engine was found using
`io_uring`.

## The gap (what Lacuna claims as new)

- **(a) Draft tokens → union of activated weights over a multi-token lookahead.**
  Partial prior art at *MoE-expert granularity only*: SP-MoE (draft-attention →
  cross-model expert predictor, prefetch overlapped with drafting), MoE-SpeQ
  (>96% speculative prefetch hit; notes the expert set grows linearly with
  lookahead k), DraftExpert (expansion-aware, 86–88% prefetch hits), plus an
  N-gram speculative table on NVMe in a llama.cpp fork. **Gap: no dense
  neuron/weight-page-granularity union prediction from draft tokens, and none of
  it on llama.cpp.**
- **(b) Async NVMe prefetch fully overlapped with compute.** Not a gap — published
  by FlexGen, PowerInfer-2 (explicitly), AirLLM, Apple, HOBBIT.
- **(d) Adaptive per-page precision (hot pages high bits, cold pages 1–2 bit).**
  Partial: HOBBIT does token-level importance-driven expert replacement on
  llama.cpp; EdgeMoE does it statically. **Gap: nothing at page granularity,
  nothing driven by measured activation frequency, nothing with a 1–2-bit cold
  tier as a runtime policy.**
- **(e) All of it integrated on llama.cpp.** **Gap confirmed.** HOBBIT covers
  (b) + (d-at-expert-granularity); BELLS and other forks cover subsets;
  SP-MoE / MoE-SpeQ / DraftExpert are not llama.cpp-based. No single engine
  combines (a) + (b) + (d) on llama.cpp. That combination is Lacuna.

## Caveats

SP-MoE figures are snippet-grade (direct PDF fetch timed out). The "2.8T Kimi K3
in 3.72 GB VRAM" AirLLM claim is an unverified social-media claim. PowerInfer's
13.20 vs 8.32 tok/s figures differ by model mix. oLLM / SSD-LLaMA speeds lack
independent verification. Treat all of the above as reported, not confirmed.
