# Latest 70B+ models for Lacuna — ranked research (2026-10-25)

Read-only research. No downloads, pushes, or commits. No invented repo IDs,
sizes, or precisions — anything not found in a public source is marked
"not found" with where it was looked.

## The bar (user's words, as clarified 2026-10-25)

- Any latest 70B+ model. Trillion-parameter models are NOT required and NOT
  the priority — they sit at the far end of the table only.
- Famous names first (his examples: Qwen 3.8, Gemma 4, and similar).
- Native precision (BF16/F16) is the target — heavily quantized files
  (Q2_K, Q4_K class) must never be presented as running the heavy model.
- Fits a ~1TB disk. Closest to the current Qwen2-only executor wins ties.
- Pass/fail for the headline run: 32 tokens within 300 seconds
  (≥0.107 tok/s sustained) with time-to-first-token reported.

## Two honest corrections to assumptions in the plan

1. **There is no latest dense Qwen3-family 70B+ model.** Qwen3 dense tops out
   at 32B (2025); Qwen3.5 dense at 27B; Qwen3.6 dense at 27B; Qwen3.8 dense
   at 27B. Any ≥70B Qwen-family target is MoE, full stop.
2. **Gemma 4 is real, famous, latest, Apache 2.0 — but its largest model is
   31B, below the 70B bar.** It cannot be the 70B+ headline. It is listed
   below the bar, not ranked as a 70B+ target.

## Ranked table (first target first)

| Rank | Model | Total / active | Release | Native precision + size | Fits ~1TB? | License / access | GGUF? | Executor distance from Qwen2 |
|---|---|---|---|---|---|---|---|---|
| **1 — FIRST TARGET** | `Qwen/Qwen3-235B-A22B-Instruct-2507` | 235B / 22B (8 of 128 experts) | Jul 2025 | **BF16 437.91 GB** (official); FP8 220.20 GB | Yes | Apache 2.0, public | Yes — `ggml-org/Qwen3-235B-A22B-GGUF` (incl. BF16 shards) | **Closest**: plain GQA + QK-norm + MoE. No DeltaNet, no MLA, no n-gram table, no Engram. MoE Stage 0 already accepts `qwen3moe`. |
| 2 | `Qwen/Qwen3.5-122B-A10B` | 122B / 10B (8 of 256 experts + 1 shared) | Feb 2026 | **BF16 234 GB** (secondary source, NVIDIA forum) | Yes | Apache 2.0, public | Yes — community (e.g. PRISM-LITE, ROCmFP4) | Farther: hybrid DeltaNet — 36 linear + 12 full-attention layers (3:1). Needs a linear-attention implementation. Multimodal (vision + MTP head to skip for text). |
| 3 | `Qwen/Qwen3.8-Flash-Next` | 180B total = 125B core (6B active) + 51B n-gram table + 4B MTP | Aug 2026 | FP8 172.78 GiB (official smallest); community BF16 ~355 GB (Unsloth) | Yes | `qwen-community-1.0` (listed as "other"), public | Yes — `unsloth/Qwen3.8-Flash-Next-GGUF`, `AtomicChat/Qwen3.8-Flash-Next-GGUF` | Farthest: Gated DeltaNet + Qwen Sparse Attention + n-gram embedding + Gated Residual + MTP. |
| 4 | `deepseek-ai/DeepSeek-V4-Flash` (0731 revision) | 284B / 13B | Apr 2026 (GA Jul 2026) | **No BF16 found.** Shipped precision is FP4 (MoE experts) + FP8 (rest) mixed, ~160–167 GB | Yes (but quantized) | MIT, public | Yes — community only, fork-required (`teamblobfish/DeepSeek-V4-Flash-GGUF`; V4-aware llama.cpp fork, 5 custom ops) | Far: hybrid CSA + HCA attention, Manifold-Constrained Hyper-Connections, sigmoid routing. MoE rung 3, not Stage 0. |
| 5 | `openai/gpt-oss-120b` | 117B / 5.1B (top-4 of 128 experts, 36 layers) | Aug 2025 | **Native precision IS MXFP4 (4-bit)** — ~61–63 GB; BF16 exists only as an upcast (~240 GB, not native weights) | Yes | Apache 2.0, public | Yes — MXFP4 GGUF | Medium-far: MoE + MXFP4 dequantizer needed + Harmony chat format required. Attention details not verified in this pass. |
| 6 | `meta-llama/Llama-4-Scout-17B-16E-Instruct` | 109B / 17B (16 experts) | Apr 2025 | Shipped for on-the-fly 4/8-bit; **BF16 file size not found** in retrieved sources | Unconfirmed | Llama 4 Community License, gated acceptance | Not confirmed in this pass | Far: iRoPE (interleaved rotary), early-fusion multimodal, MoE. Maverick (400B, BF16 exists, ~800 GB computed) is same arch distance, older, bigger — below Scout. |
| 7 | `mistralai/Mistral-Large-3-675B-Instruct-2512-BF16` | 675B / 41B (673B LM + 2.5B vision) | Dec 2025 | BF16 exists but **~1.35 TB computed (675B × 2 bytes) — EXCEEDS ~1TB disk**; FP8 variant fits but is quantized | No (BF16) | Apache 2.0, public | Not found in this pass | Far: granular MoE, arch details vs executor not verified. |
| 8 — far end | `moonshotai/Kimi-K3` | 2.8T / 104B (16 of 896 routed + 2 shared) | Jul 2026 (weights) | Native MXFP4 experts + BF16 rest; **~1.56 TB (96 shards) — EXCEEDS disk**; no BF16 repo found | No | Custom Kimi K3 License, public | Community only (`unsloth/Kimi-K3-GGUF` 1-bit ~553 GiB; `GrEarl/Kimi-K3-GGUF` Q2_K) | Far: Kimi Delta Attention + Gated MLA layers, MXFP4. |
| 9 — far end | `Qwen/Qwen3.8-2.4T-A95B` | 2.4T / 95B (10 of 512 routed + 1 shared) | Aug 2026 | BF16 **~4.89 TB / 224 files — EXCEEDS disk**; FP8 sibling exists, size not verified from a primary source | No | Custom `qwen3.8-max` (revenue/MaaS conditions), public | Not found | Far: hybrid Gated DeltaNet + Gated Attention, MTP trained. |

## Below the 70B bar — famous, latest, but NOT 70B+ targets

| Model | Total | Release | Native precision + size | License | GGUF? | Why it is not the target |
|---|---|---|---|---|---|---|
| `google/gemma-4-31b-it` (Gemma 4 family max) | 31B dense (60 layers, 32 Q / 16 KV heads, head_dim 256; hybrid sliding-window-1024 + global attention; 550M vision encoder) | Apr 2026 | BF16 62.5 GB | Apache 2.0, public | Yes — community BF16/Q8_0/Q6_K/Q5_K/Q4_K/Q3_K | 31B < 70B bar. Whole Gemma 4 family: E2B, E4B, 12B Unified, 26B-A4B MoE, 31B dense — none reaches 70B. Cleanest dense fallback if the bar ever drops (dense, Apache, no MoE needed) but needs a new Gemma arch implementation (sliding-window hybrid, 256 head_dim, Gemma tokenizer) — NOT Qwen2 code. |
| `Qwen/Qwen3.8-27B` | 27.78B dense multimodal (64 layers, 16×[3 linear + 1 full] hybrid, 4 KV heads, head_dim 256) | Aug 2026 | BF16 ~54 GB | Apache 2.0, public | Yes — `unsloth/Qwen3.8-27B-GGUF`, `ggml-org/Qwen3.8-27B-GGUF` | 27B < 70B bar. Hybrid DeltaNet anyway. |
| `Qwen/Qwen3.8-27B` context note | 262,144 native, extensible to 1M via YaRN; thinking on by default but disableable | | | | | |

## Qwen3.8 family, complete (no 70B+ dense member exists)

- `Qwen/Qwen3.8-2.4T-A95B` — 2.4T MoE, custom license, BF16 ~4.89 TB (see rank 9).
- `Qwen/Qwen3.8-27B` — 27B dense, Apache 2.0 (see below-bar table).
- `Qwen/Qwen3.8-Flash-Next` — 180B MoE preview of Qwen4 arch (see rank 3).
- `Qwen3.8-Flash` (hosted production version of Flash-Next) and `Qwen3.8-Max`
  (hosted flagship) are API-only as products: the hosted Max adds vision,
  non-thinking mode, 1M context by default, and built-in tools that are NOT
  in the downloadable weights (text-only, thinking forced on).
- `Qwen3.7-Max` (May 2026) and `Qwen3.7-Plus` (Jun 2026) shipped API-only
  with no downloadable weights — unobtainable as weights.
- Kimi K3.1 Pro weights were not confirmed in this pass — do not assume.

## DeepSeek V4 family (for the record)

- `deepseek-ai/DeepSeek-V4-Flash` — 284B/13B, FP4+FP8 mixed ~167 GB, MIT.
- `deepseek-ai/DeepSeek-V4-Pro` — 1.6T/49B, FP4+FP8 mixed ~865 GB, MIT
  (exceeds a ~1TB disk only marginally on paper; in practice not a disk fit
  once a working copy is added — treat as far end).
- `deepseek-ai/DeepSeek-V4.1-Flash` (1017 revision) — 552B backbone + 196B
  Engram conditional-memory module, 8B active prefill / 16B decode,
  Causal Encoder-Decoder (20-layer encoder + 20-layer decoder), FP8
  safetensors ~510 GB, MIT. Community GGUFs exist (ds4 Q2 340.6 GiB,
  VQ ~113.6 GB, IQ2_XXS) but all are quantized and most need non-stock
  runtimes (ds4, Unsloth pin, or forks). Native precision is FP8, not BF16.
- No BF16 safetensors found for any V4 variant (searched HF pages, guides,
  and community conversions).

## Qwen3 vs Qwen2 — what the executor must handle (for the first target)

Applies to Qwen3-235B-A22B (Qwen3, 2025). Qwen3.5/3.8 differ further
(248,320 vocab, DeltaNet hybrids, partial rotary) — do not conflate.

| Area | Qwen2 (current executor) | Qwen3-235B-A22B (target) | Executor work |
|---|---|---|---|
| QK-norm | None | **NEW: per-layer `q_norm` + `k_norm` RMSNorm weights on Q and K, applied per head BEFORE RoPE** | New tensors + new op. Critical: without it attention scores are wrong, not near-tie noise. |
| Attention | GQA, full attention | GQA, full attention (64 Q / 4 KV heads, head_dim 128). **No MLA** in Qwen3-MoE. No DeltaNet in Qwen3 (2025). | Reuse GQA + half-rotation RoPE as-is. |
| RoPE | Half-rotation form, theta 1,000,000 (Qwen2.5 family) | Same form; theta 1,000,000. **Not a difference** for this model. (Qwen3.5/3.8 use partial_rotary_factor 0.25 — different models, not this one.) | Verify theta + rotary fields in `config.json` at build time; do not hardcode from memory. |
| Norms | RMSNorm, eps 1e-6 | RMSNorm, eps 1e-6, plus q_norm/k_norm above | Same norm code + two extra norm applications. |
| MLP → MoE | Dense SwiGLU | Router: softmax → top-8 of 128 → renormalize (`norm_topk_prob=true`); per-expert SwiGLU, `moe_intermediate_size` 1536 | MoE Stage 0 already implements this routing and accepts `qwen3moe`. Shared expert: none surfaced for 235B in retrieved sources (unlike Qwen3.5-122B's 1 shared expert/layer) — verify in config; if absent, Stage 0's shared path stays unused. |
| Biases | QKV biases present | Default `attention_bias=False` in Qwen3Config; per-checkpoint value not surfaced for 235B in this pass | Read the flag from config; do not assume. |
| Embeddings / head | `output.weight` present (untied) on Qwen1.5-110B | Untied (separate `lm_head`) per architecture tables | Executor already handles present `output.weight`. Verify in config. |
| Tokenizer / vocab | Qwen2.5: 151,936 | **Same vocab size 151,936**, same BPE family | Tokenizer carries over at vocab level. Verify added special tokens (thinking tokens) from `tokenizer_config.json` at build time. |
| Sliding window | No | Qwen3 dense family has sliding-window layers; 235B-MoE uses full attention (Dual Chunk Attention claimed for 262K long context) | Verify layer attention types in config at build time. |
| Context | 128K (Qwen2.5) | 128K base config; 262K native claimed via DCA (secondary source) | Executor context handling unchanged for short prompts. |
| GGUF / dequant | F32/F16/Q8_0/Q4_K/Q6_K/Q2_K/Q3_K/Q5_K | BF16 GGUF path needs **no new dequantizer** (F16/BF16 handling exists); K-quant variants already covered | Native BF16 GGUF is the clean path. |

## Disk logistics for the first target (flag, not a model fact)

- BF16 GGUF (~438 GB) → `.lac` conversion (~438 GB) → ~876 GB held
  simultaneously during conversion. That fits a ~1TB disk but tight.
  Sequence it: download GGUF → convert → delete GGUF → run from `.lac`.
  Do NOT hold safetensors BF16 + GGUF BF16 + `.lac` at once (~1.3 TB).
- The free lanes used so far held ~39.5 GB-class files. A 438 GB holding
  disk is an open logistics question for the build day, separate from
  model selection. The model passes the "~1TB disk" criterion; the lane
  that provides that disk is not yet identified in this research.

## Unverified leads (not ranked — check before use)

- Mistral Small 4 (119B MoE, 6.5B active, Apache 2.0 per an Oct 2026
  shortlist table): repo ID and precision not verified in this pass.
- Muse Spark Open 816B MoE (42B active, Muse Model License, Oct 2026
  shortlist): repo ID, precision, and disk fit not verified in this pass.
- GLM-5.3 / 5.3-Flash (753B MoE class, custom GLM licence, per the same
  shortlist): repo ID and precision not verified in this pass.

## Sources consulted (public web, 2026-10-25)

HF repo pages and model cards (Qwen, moonshotai, deepseek-ai, google,
meta-llama, mistralai, openai, ggml-org, unsloth, community GGUF repos),
QwenLM/Qwen3.8 GitHub news list, Qwen3/Qwen3-MoE config references
(DeepWiki mirrors of public config code), vendor/secondary coverage
(simonwillison.net, NVIDIA developer forum, Galileo, llm-stats,
intuitionlabs, wavespeed, marktechpost, dailyaibrief, spheron, atomic.chat,
dev.to, forasoft Oct 2026 shortlist). Where a figure comes from a secondary
source rather than the repo page itself, it is labeled as such above.
