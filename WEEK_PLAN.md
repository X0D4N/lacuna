# Lacuna — Week Hardening Plan (Oct 17–23, 2026)

Owner: Fahad. Deadline: Saturday Oct 24, 2026 — professional, future-invention level,
trending-ready. **UPDATE 2026-10-18: repo is LIVE at https://github.com/CYPHERLYNX/lacuna**
(repo created + full history pushed via his classic PAT; branch `master`).
His PAT stays ACTIVE (7-day expiry; he re-pastes the value when a push is needed —
I never store the value). Remaining day-tracks commit locally; push each day's commit
when he supplies the token, otherwise leave queued.

Daily cron (`lacuna-week-hardening`) implements the next unchecked track below.
Serena (oraios/serena, MCP coding toolkit) is installed (`~/.local/bin/serena`)
and MANDATORY for all code work: project `lacuna` registered + indexed at
`.serena/project.yml`, health-checked 2026-10-16. Use its semantic tools
(find_symbol, find_referencing_symbols, replace_symbol_body) instead of blind
text edits; re-index after large refactors.
Rules for every run:
- Read this file, pick the first unchecked track, implement it fully.
- Test: `PYTHONPATH=~/workspace/lacuna <venv-python> -m pytest ~/workspace/lacuna/tests/ -q`
  (any venv with pytest+numpy, e.g. ~/workspace/ai-tools/corsac/.venv/bin/python).
- Commit ONLY if all tests pass:
  `git -c user.name="CYPHERLYNX" -c user.email="cypherlynx@users.noreply.github.com" commit`
- NEVER invent numbers: every figure from `simulator/` (label SIMULATED) or tests
  (label MEASURED). No TODOs, no fake integrations, no macOS paths.
- Check off the track below with date + one-line result. If blocked, note why and
  continue to the next track.
- Stay silent in chat; record everything here. Log observations to ~/memory/YYYY-MM-DD.md.
- Stdlib + numpy only. Linux (deb/arch) + Windows 11 only.

## Tracks

- [x] Day 1 — Sat Oct 17: EXACTNESS MERGE. DONE 2026-10-16 (a day early): v0.2.0
      committed (3c056f5 "Lacuna v0.2.0: exact mode + tree speculation"), full
      suite 39/39 green re-verified 2026-10-16T04:50Z. No issues found.
- [x] Day 2 — Sun Oct 18: ORACLE RECIPES. DONE 2026-10-17 (a day early):
      `lacuna/recipes.py` (family presets qwen/llama/moe, synthetic calibration,
      train/evaluate/gate, .npz schema) + `docs/ORACLE_RECIPES.md` (capture
      procedure, storage arithmetic, hyperparams, gate bars) + `lacuna
      oracle-recipe` CLI + 13 new tests. Full suite 52/52 green. Real-model
      capture hook specified but not built (lands with Day 4 GGML track).
- [x] Day 3 — Mon Oct 19: PERPLEXITY HARNESS. DONE 2026-10-18 (a day early):
      `benchmarks/perplexity/` (numpy tiny transformer, 258,528 params, trained
      on 60K deterministic synthetic tokens) + `lacuna perplexity` CLI +
      `benchmarks/perplexity.md` (methodology + results table, all MEASURED) +
      7 new tests. Results: exact == dense bit-identical (ppl 6.0111);
      extreme Q4/Q2 +7.15 ppl at 3.14 bits/param; ternary stress +35.41.
      Honest caveat: deltas only, do not transfer to real LLMs. Full suite
      59/59 green.
- [x] Day 4 — Tue Oct 20: GGML INTEGRATION. DONE 2026-10-19 (a day early):
      `lacuna/buffer.py` (`PagerBackedBuffer`: alloc/bind/ensure_resident fence/
      touch/LRU eviction/row-masked matvec/per-layer `tiled_exact_matvec`) +
      `tests/test_buffer.py` (11 tests, all MEASURED — masked rows bit-identical
      vs op-identical reference, unmasked pages never fault, eviction loud) +
      `integrations/llama.cpp/sp_buffer.{h,cpp}` (C++ buffer-type translation,
      `g++ -fsyntax-only -Wall -Wextra` clean against local shim; real-tree
      compile pending a GPU box) + `page_table()`/`page_shape()` pager accessors.
      Full suite 70/70 green. INTEGRATION.md §1 sketch superseded by the files.
- [ ] Day 5 — Wed Oct 21: BENCHMARK EXPANSION + THE 9/10 PROGRAM (Fahad's directive
      2026-10-19: push confidence from 6/10 to 9/10 by measurement, not claims).
      (a) Shadow-mode oracle scoring: run Lacuna's page predictor alongside real
      llama.cpp inference on a real MoE GGUF, measure page-hit rate on real weights
      with zero effect on output — converts "simulator says" into measured hit rates.
      (b) Deterministic n-gram prefetch: for n-gram-embedding models (Qwen3.8-Flash-Next),
      needed embedding rows are a pure function of the input text — computable with zero
      ML; implement exact prefetch, no oracle needed for those 51B params.
      (c) Cache-style expert predictor: per-expert 2-bit saturating counters + bigram
      Markov table on expert activation sequences, updated online per token — zero
      training, adapts to any MoE in minutes. Prototype on branch `prefetch-9of10`,
      full pytest green before merge. (d) FULL OPTIMIZATION SWEEP (Fahad's directive
      2026-10-19: "optimise, make it whole 100x better"): profile hot paths with
      cProfile, vectorize numpy bottlenecks, kill redundant copies (views/mmap),
      streamline scheduler + pager + oracle; measure before/after on the simulator,
      no behavior changes, suite stays green. Prototype on branch `optimize`.
      Plus: more simulator scenarios (sparsity sweeps,
      K sweeps, bandwidth variants); updated honest tables in README + benchmarks/RESULTS.md.
      (e) TRILLION-TIER BENCHMARK TARGET SET (Fahad's directive 2026-10-19): model the
      10-model target list in the simulator — Kimi K3 (2.8T/104B), Qwen3.8-Max
      (2.4T/95B), DeepSeek-V4-Pro (1.6T/49B), LongCat-2.0 (1.6T/~48B), MiMo-V2.5-Pro
      (1.02T/42B), Yuan3.0 Ultra (1.01T/68.8B), Kimi K2.7 Code (1T/32B), Ring-1T
      (1T/50B), Inkling (975B/41B), Muse Spark Open (816B/42B) — per-token weight
      movement vs. tier verdicts (FLOOR/MID/DESKTOP) per docs/TARGET_MODELS.md.
      Figures verified 2026-10-19; unknowns marked there.
- [ ] Day 6 — Thu Oct 22: EASE OF USE (Fahad's directive 2026-10-19: "make it 100x
      better, faster, easier to use for anyone, with models and agents out there").
      (a) `lacuna pull <hf-id>` / `lacuna run <hf-id>`: one command from zero to chat —
      HuggingFace download (stdlib urllib, no new deps) + auto-convert to .lac + run.
      (b) Ollama-compatible API shim: speak Ollama's /api/generate, /api/chat, /api/tags
      (incl. streaming NDJSON) so every existing agent/tool built for Ollama works with
      Lacuna unchanged — biggest ecosystem unlock. (c) Hardware auto-tune: detect
      RAM/VRAM/disk and pick caps, page size, K automatically — zero config for anyone.
      (d) Speed: wire 9/10 prefetch + tree speculation as the default path; document
      HONEST speedups only (no 100x tok/s claims — physics). Prototype on branch
      `ease-of-use`, full pytest green before merge. Plus: README/docs final pass,
      CHANGELOG.md, verify sdist builds, review install scripts.
- [ ] Day 7 — Fri Oct 23: RELEASE PREP + ZERO-SILENT-ERROR HARDENING (Fahad's directive
      2026-10-19: "zero latency lag, faster response... with zero error or mistakes").
      Honest standard: not literal zero latency (physics) but *feels instant* — first
      token <1s via warm-start persistent hot-set cache (keep frequent pages resident
      between runs), then smooth streaming; and *zero silent errors* — page checksums
      on every .lac page, fault-injection fuzz tests on the pager, deterministic mode
      (fixed seeds, reproducible runs), fail-loud-never-wrong on any inconsistency.
      Full review pass, version bump, final green test run, write the Saturday push
      checklist (repo create + push commands) at the top of this file.
      FINAL GATE (Fahad's directive 2026-10-19): nothing pushes until the repo is
      100% done AND the suite is expanded to 100–200 tests, ALL green, AND the
      professional checklist passes (README renders, CHANGELOG current, sdist builds
      + installs cleanly, install.sh/install.ps1 reviewed, CI green on
      ubuntu+windows × py3.10/11/12, zero TODOs, no mock paths in user-facing flows).
      Only then: ask Fahad to re-paste his PAT, push everything (all branches merged
      to master first), verify the live repo, and report completion.
