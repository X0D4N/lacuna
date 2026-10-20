# Target Models — Heavy Latest Local Models for Lacuna

Survey of frontier open-weight models (Oct 2026) that cannot run on normal
systems today, ranked by Lacuna fit. Rule of thumb: **active params per token**
decide per-token weight movement (what Lacuna streams); **total params** decide
NVMe footprint. All figures vendor-reported unless noted.

| Model | Total | Active / token | Experts | License | Disk (native) | Lacuna fit |
|---|---|---|---|---|---|---|
| Qwen3.8-Flash-Next | 180B (125B + 51B n-gram + 4B MTP) | 6B | 512 (10+1) | qwen-community-1.0 | ~185 GB (FP8) | **Best** — 6B active; 51B n-gram table is deterministic lookups, pure function of input text (exact prefetch, zero ML) |
| DeepSeek V4.1 Flash | 552B (+~196B Engram) | 8B prefill / 16B decode | 384 (top-6) + 1 shared | MIT | ~510 GB | **Great** — tiny active set, best license; needs 500GB+ NVMe |

## Trillion tier — Fahad's target list (2026-10-19: "runnable at any cost")

Honest verdict rule, no exceptions: per-token weight movement =
active_params × ~0.5 bytes (Q4). At 3.5 GB/s NVMe, anything over ~4 GB/token
cannot run at usable speed on the FLOOR laptop.

- **FLOOR** = RTX 2050 + 6 GB RAM + 500 GB NVMe (the minimum Lacuna promises)
- **MID** = RTX 4050 6 GB + 16 GB RAM
- **DESKTOP** = 64 GB RAM + fast 2 TB+ NVMe

| # | Model | Total | Active / token | Experts (routing) | License | Disk (native) | Cold GB/token (EST.) | Verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | Kimi K3 (Moonshot) | 2.8T | 104B | 896 (16/token) | Kimi K3 License (MIT-derived, revenue-tiered) | ~1.56 TB (MXFP4 native) | 52 | **DESKTOP only** — ~15 s/token cold; FLOOR/MID cannot. Doesn't fit 500 GB floor disk. |
| 2 | Qwen3.8-Max (Qwen/Qwen3.8-2.4T-A95B) | 2.4T | 95B | 512 (10+1) | custom qwen3.8-max (revenue thresholds) | ~4.89 TB (BF16) | 47.5 | **DESKTOP only** — ~14 s/token cold; FLOOR/MID cannot. |
| 3 | DeepSeek-V4-Pro-0813 | 1.6T | 49B | unknown | MIT | ~893 GB (FP4+FP8 mixed) | 24.5 | **DESKTOP only** — ~7 s/token cold; FLOOR/MID cannot. Best license in tier. |
| 4 | LongCat-2.0 (Meituan) | 1.6T | ~48B (33–56B dynamic) | unknown | MIT (per coverage; verify on HF card) | ~3.2 TB BF16 / ~1.6 TB FP8 (EST.) | 24 | **DESKTOP only** — ~7 s/token cold. N-gram embedding = deterministic prefetch win (like Qwen3.8-Flash-Next). |
| 5 | MiMo-V2.5-Pro (Xiaomi) | 1.02T | 42B | 384 (8/token) | MIT | ~1 TB FP8 (EST.) | 21 | **DESKTOP only** — ~6 s/token cold. MTP module aids throughput. |
| 6 | Yuan3.0 Ultra (YuanLab) | 1010B | 68.8B | unknown | unknown — verify on HF card | ~2 TB BF16 (EST.) | 34.4 | **DESKTOP only** — ~10 s/token cold; heaviest mid-tier active set. |
| 7 | Kimi K2.7 Code (Moonshot) | 1T | 32B | 384 (8+1 shared) | Modified MIT | ~595 GB (native INT4) | 16 | **DESKTOP** — ~4.6 s/token cold; closest to MID but still over the 4 GB line. Best trillion-tier coding fit. |
| 8 | Ring-1T (inclusionAI) | 1T | 50B | unknown | unknown — not stated on card | ~1 TB FP8 / ~2 TB BF16 (EST.) | 25 | **DESKTOP only** — ~7 s/token cold. |
| 9 | Inkling (Thinking Machines) | 975B | 41B | 256 routed + 2 shared (6/token) | Apache 2.0 | ~2 TB BF16 (EST.) / 270–285 GB (1-bit GGUF) | 20.5 | **DESKTOP only** — ~6 s/token cold. 1-bit GGUF fits a 500 GB floor disk; effort dial saves tokens. |
| 10 | Muse Spark Open (Meta) | 816B | 42B | 256 (8/token) | Muse Model License (custom, limited commercial) | ~1.6 TB BF16 / ~484–530 GB NVFP4 | 21 | **DESKTOP only** — ~6 s/token cold. NVFP4 fits 500 GB floor disk. |

### Trillion-tier honesty notes

- **Cold-start bound vs steady-state target.** The per-row verdicts above use
  cold-start math (every page faulted from disk, nothing cached) — the
  pessimistic bound. Lacuna's hot-set caching + lookahead prefetch exist
  precisely to beat it: experts recur across tokens, so after warmup the
  steady-state disk reads per word collapse toward the hot working set.
  **The engineering target (Fahad's directive 2026-10-19, corrected
  2026-10-20): all 10 runnable on the FLOOR (RTX 2050 + 6GB RAM) — exact
  output, hard caps respected, system never stressed.** (Not MID — the floor
  is the min spec for the trillion tier too.) What decides it per model is
  the expert working-set reuse rate on real traces under a 4.5GB hot-set
  budget — measurable with `lacuna/prefetch.py: ShadowScorer`.
  Cold numbers say the floor cannot; the measurement program decides how far
  past them the hot-set design gets. Until measured, the DESKTOP verdicts
  stand as the conservative bound. One physical caveat: models larger than
  the disk itself (e.g. Kimi K3 at ~1.56TB) need a bigger NVMe — no software
  fixes a disk that's too small.
- **Disk is a second wall.** On the 500 GB floor machine, only Inkling (1-bit
  GGUF, ~270 GB) and Muse Spark Open (NVFP4, ~484 GB) physically fit. Everything
  else needs 1–5 TB NVMe.
- **Unknowns that block exact per-model planning:** DeepSeek-V4-Pro expert/routing
  counts; LongCat-2.0 expert/routing; Yuan3.0 Ultra license + routing; Ring-1T
  license + routing. Verify on the HuggingFace model cards before building
  per-model prefetch paths.
- **Sources:** Kimi K3 — moonshotai/Kimi-K3 HF card, digitalapplied.com,
  k3-kimi.com; Qwen3.8-Max — Qwen/Qwen3.8-2.4T-A95B HF, digitalapplied.com,
  llmcheck.net; DeepSeek-V4-Pro — vdf.ai, tokenhot.ai; LongCat-2.0 —
  huggingface.co/meituan-longcat/LongCat-2.0, explainx.ai; MiMo-V2.5-Pro —
  huggingface.co/XiaomiMiMo/MiMo-V2.5-Pro, mimo.xiaomi.com; Yuan3.0 Ultra —
  arxiv.org/abs/2601.14327, huggingface.co/YuanLabAI/Yuan3.0-Ultra;
  Kimi K2.7 Code — huggingface.co/moonshotai/Kimi-K2.7-Code, saipien.org;
  Ring-1T — huggingface.co/inclusionAI/Ring-1T-FP8, arxiv.org/abs/2510.18855;
  Inkling — kie.ai, ai-primer.com, datacamp.com; Muse Spark Open —
  marktechpost.com, alphasignal.ai, techzei.com. All accessed 2026-10-19.
  Figures marked EST. are arithmetic estimates (active × 0.5 bytes), not vendor claims.

## Why the ranking

Lacuna's trick is predicting which weight pages each token needs and streaming
them from NVMe ahead of time. That works when the **active set is small**
(Qwen3.8-Flash-Next: 6B; DeepSeek V4.1 Flash: 8–16B). When 95–104B params are
active per token, the NVMe itself becomes the wall: at 3.5 GB/s, ~50 GB of
weights per token is ~15 s/token before compute even starts. No scheduler fixes
physics — those models need datacenter iron.

## Notes

- Qwen3.8-Flash-Next's n-gram embedding (20M bigrams/trigrams, layer 2) is the
  single most Lacuna-friendly structure in any shipping model: needed rows are
  computable from the input, so prefetch is exact, not predicted.
  (See `lacuna/prefetch.py: NgramPrefetcher`.)
- DeepSeek V4.1 Flash's "Engram" memory component and 890 bytes/token KV cache
  are offload-shaped structures worth a dedicated prefetch path (future work).
- Kimi K3 ships natively MXFP4 (quantization-aware trained) — precision policy
  `extreme` territory, not `exact`.
- All speed claims about these models under Lacuna are TARGETS until the
  shadow-mode scorer (Day 5, branch `prefetch-9of10`) measures real hit rates.

*Surveyed 2026-10-19. Re-check quarterly — this space moves fast.*
