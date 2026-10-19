# Target Models — Heavy Latest Local Models for Lacuna

Survey of frontier open-weight models (Oct 2026) that cannot run on normal
systems today, ranked by Lacuna fit. Rule of thumb: **active params per token**
decide per-token weight movement (what Lacuna streams); **total params** decide
NVMe footprint. All figures vendor-reported unless noted.

| Model | Total | Active / token | Experts | License | Disk (native) | Lacuna fit |
|---|---|---|---|---|---|---|
| Qwen3.8-Flash-Next | 180B (125B + 51B n-gram + 4B MTP) | 6B | 512 (10+1) | qwen-community-1.0 | ~185 GB (FP8) | **Best** — 6B active; 51B n-gram table is deterministic lookups, pure function of input text (exact prefetch, zero ML) |
| DeepSeek V4.1 Flash | 552B (+~196B Engram) | 8B prefill / 16B decode | 384 (top-6) + 1 shared | MIT | ~510 GB | **Great** — tiny active set, best license; needs 500GB+ NVMe |
| Kimi K3 | 2.8T | 104B | 896 (16 active) | modified MIT | ~1.56 TB | Poor on normal systems — 104B active/token ≈ 50GB+ streamed per token; datacenter territory even with paging |
| Qwen3.8-2.4T-A95B | 2.4T | 95B | 512 (10+1) | custom | ~4.89 TB (BF16) | Poor on normal systems — same per-token math as K3 |

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
