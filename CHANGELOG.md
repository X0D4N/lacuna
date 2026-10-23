# Changelog

All notable changes to Lacuna. Numbers are honest: MEASURED = this repo's
tests/benchmarks on the development machine; SIMULATED = `simulator/` output, labeled
everywhere it appears.

## [0.3.0] — 2026-10-23 (showcase release; hardening week Oct 17–23, 2026)

Release prep final gate (2026-10-22): full review pass, CI trigger fixed to
the real `master` branch, sdist builds + installs cleanly in isolation,
README/install scripts reviewed, zero TODOs, no macOS paths, mock backend
explicitly labeled everywhere it appears. Full suite 192/192 green.

### Round 3 — showcase optimization + hardening (2026-10-22)
- pack_q2 float32 fast path (provably exact — see `lacuna/quant.py`):
  1.8x, bit-identical, pinned against a float64 reference on adversarial
  midpoint-clustered inputs. unpack_q4 stacked-nibble interleave: 1.6x,
  bit-identical. (MEASURED; `benchmarks/RESULTS.md`.)
- Server, load-verified (240 requests @ 32 concurrent clients, 240/240;
  12-payload chaos suite clean): listen backlog 5 → 128 (stdlib default
  reset connections under burst), unsupported methods → clean JSON 405,
  framework errors → JSON instead of HTML, oversized bodies drained
  before the 413 reply (no more broken pipes), `content: null` no longer
  becomes the literal prompt `"None"`.

### Added
- `lacuna pull <hf-id>` / `lacuna run <hf-id>`: one command from zero to
  chat — Hub download (stdlib urllib, resume via Range) + auto-convert
  GGUF→`.lac` + run. Pull-if-missing, traversal-proof cache paths
  (`lacuna/pull.py`).
- Ollama-compatible API shim (`lacuna/ollama_shim.py`): `POST /api/generate`,
  `POST /api/chat`, `GET /api/tags`, NDJSON streaming — any Ollama-built
  agent/tool works unchanged. Engine-swap protocol keeps the shim
  inference-agnostic; the default backend is the clearly-labeled mock.
- Hardware auto-tune (`lacuna/hardware.py`): RAM from `/proc/meminfo`
  (Linux) / `GlobalMemoryStatusEx` (Windows 11), VRAM via nvidia-smi with
  safe fallback; RAM hot budget = min(6 GB, 75% of RAM), VRAM working set
  3.2 GB on ≥4 GB cards, auto page size + spec K. Linux + Windows 11 only.
- Oracle training recipes per model family (Qwen/Llama/MoE):
  `lacuna/recipes.py`, `docs/ORACLE_RECIPES.md`, `lacuna oracle-recipe` CLI,
  with `.npz` schema, capture procedure, and gate bars.
- Perplexity harness (`benchmarks/perplexity/`, `lacuna perplexity` CLI):
  real trained forward passes through the real codecs on a 258k-param
  synthetic model. MEASURED: exact == dense bit-identical (ppl 6.0111);
  extreme Q4/Q2 +7.15 ppl at 3.14 bits/param; ternary stress +35.41
  (deltas only — do not transfer to real LLMs).
- GGML integration prototype: `lacuna/buffer.py` (`PagerBackedBuffer` —
  alloc/bind/ensure_resident fence/touch/LRU eviction/row-masked
  matvec/per-layer `tiled_exact_matvec`) + `integrations/llama.cpp/sp_buffer.{h,cpp}`
  (C++ buffer-type translation, `g++ -fsyntax-only -Wall -Wextra` clean;
  real-tree compile pending a GPU box) + `page_table()`/`page_shape()`
  pager accessors.
- 9/10 program: zero-training prefetch primitives — n-gram + cache-expert
  prefetchers, shadow-mode oracle scorer (`lacuna/prefetch.py`). Prototype
  measured 16x over random on synthetic patterns only; real-weight scoring
  needs a GPU box.
- Simulator sweeps (`simulator/sweeps.py`): all 10 trillion-tier targets
  modeled — cold GB/token reproduces `docs/TARGET_MODELS.md`, all cold
  verdicts DESKTOP-only; sparsity/K/bandwidth sweep tables
  (`benchmarks/RESULTS.md`), all SIMULATED-labeled.
- Warm-start persistent hot-set cache (`lacuna/warmstart.py`):
  fingerprint-gated, per-page CRC32 re-validated, corrupt/stale cache
  degrades to cold start (never wrong data); Engine restores on open,
  saves on close.
- Production hardening: shared `http_util` (1 MiB body cap, max_tokens
  clamp 4096, typed-field validation, clean JSON errors — no stack traces
  to clients); MCP errors one clean line, tracebacks to stderr only;
  CLI failures one sentence + next step with proper exit codes;
  `SECURITY.md`; runnable `examples/` (Python API, MCP agent client).
- `docs/TARGET_MODELS.md` (10 trillion-tier targets, honest verdicts) and
  `docs/SUPPORT_TECH_RESEARCH.md` (supportive-tech sweep, Scout selected).

### Fixed
- Fuzz-found `OverflowError` in the pager: file-size bounds now enforced
  as `ValueError`.
- `pack_q2` exhaustive ±200k-ULP sweep pinned in tests (midpoint fast path +
  exact-argmin fixup band).
- `lacuna pull`: path-traversal guard on hub filenames; stale-prefix 416
  restarts cleanly instead of trusting a longer local file.
- One pre-existing unused import in `lacuna/prefetch.py` (untouched file);
  `pyflakes` clean in all touched files.

### Changed
- Default path wires 9/10 prefetch + tree speculation: `Engine.generate`
  prefetches the oracle's predicted set per pass and runs
  `speculative_generate` with the auto-tuned K — honoring the exactness
  guarantee (oracle schedules prefetches only; compute reads the full set).
- Measured hot-path wins (all bit-identical, see `benchmarks/RESULTS.md`):
  pack_q2 8.5x, pack_ternary 10.9x, unpack_ternary 5.5x, RAW pager reads
  30.8x, buffer arena copy 3.7x, scheduler oracle calls/step 3→1 (non-exact),
  crc32 3.19 GB/s chosen. Per-op Python-layer wins; inference budget
  stays I/O-dominated by design — no multi-x tok/s claims.

## [0.2.0] — 2026-10-16

- Exact mode: bit-identical output to dense inference even with a
  deliberately broken (random-mask) oracle — proven by `tests/test_exact.py`.
- Tree speculation (`lacuna/speculate.py`): accepts exactly the target's own
  greedy tokens — proven by `tests/test_tree_spec.py`.
- Precision policies: exact (default) / extreme (adaptive Q4/Q2/ternary,
  documented tradeoff).
- Suite 39/39 green.

## [0.1.0] — 2026-10-16

- Lookahead sparse-paged inference: activation oracle → union mask over
  K-token lookahead → NVMe pager streams ahead of compute → speculative
  verification. `Engine` one-line API, OpenAI-compatible server, MCP server,
  GGUF→`.lac` converter. Suite 24/24 green.
