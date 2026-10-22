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
- [x] Day 5 — Wed Oct 21: BENCHMARK EXPANSION + THE 9/10 PROGRAM. DONE
      2026-10-20 (a day early): (b)(c) n-gram + cache-expert prefetch and
      (d) optimization sweep already merged 2026-10-19 (2926bfe); (e) new
      `simulator/sweeps.py` models all 10 trillion-tier targets — cold
      GB/token reproduces docs/TARGET_MODELS.md exactly, all cold verdicts
      DESKTOP-only, MID steady-state shows Kimi K2.7 Code closest (~4.4
      tok/s); sparsity/K/bandwidth sweeps (K diminishing past ~4, linear in
      NVMe) in README + benchmarks/RESULTS.md, all SIMULATED-labeled;
      `tests/test_sweeps.py` (11 tests). Full suite 111/111 green.
      BLOCKED: (a) shadow-mode scoring on REAL weights — prototype measured
      16x over random on synthetic patterns only; needs a GPU box with a
      real MoE GGUF + llama.cpp (not available on this box).
- [x] Day 6 — Thu Oct 22: EASE OF USE. DONE 2026-10-21 (a day early, commits
      969395a + d9c27a1): (a)(b)(c) verified merged — `lacuna pull/run <hf-id>`,
      Ollama shim (/api/generate, /api/chat, /api/tags + NDJSON streaming),
      hardware auto-tune; (d) prefetch + tree speculation confirmed as the
      Engine default path; NEW this run: CHANGELOG.md, MANIFEST.in, sdist
      verified (builds + installs cleanly in isolation, API + `lacuna simulate`
      smoke-tested), benchmarks/RESULTS.md measured hot-path table + 111→177
      test inventory, README roadmap final pass, pyflakes clean, stale merged
      branches deleted, .serena/logs gitignored. Full suite 177/177 green.
      (Fahad's directive 2026-10-19: "make it 100x better, faster, easier to
      use for anyone, with models and agents out there").
      ROUND 2 (Fahad's directive 2026-10-20: "100x even better, even optimised" +
      "Anthropic-built level") — DONE 2026-10-20 on master:
      (1) Optimization sweep round 2, all measured A/B + bit-identical:
      pack_q2 8.5x (midpoint fast path + exact-argmin fixup band; exhaustive
      ±200k-ULP sweep pinned in tests), pack_ternary 10.9x, unpack_ternary
      5.5x (LUT); pager exact-length reads — RAW pages 30.8x, full 2MiB
      pages 1.06x, small pages neutral; scheduler oracle calls/step 3->1
      (non-exact 600->200 per 200 steps, exact 400->200); buffer arena
      copy primitive 3.7x; checksum A/B: crc32 3.19 GB/s chosen.
      (2) Warm-start persistent hot-set cache (Day-7 item pulled forward):
      lacuna/warmstart.py — fingerprint-gated, per-page CRC32 re-validated,
      corrupt/stale cache degrades to cold start, never wrong data;
      Engine restores on open / saves on close. 12 tests.
      (3) Test expansion 111 -> 177: +codec identity, +fault injection,
      +hardening, +fuzz suites (all real assertions, no filler).
      (4) Production hardening: shared http_util (1MiB body cap, max_tokens
      clamp 4096, typed-field validation, clean JSON errors — no stack
      traces to clients); MCP errors are one clean line (tracebacks to
      stderr only) + line cap + JSON-RPC parse/invalid errors; pull
      path-traversal guard + stale-prefix 416 restart; CLI failures are
      one sentence + next step with proper exit codes; fuzz found and
      fixed an OverflowError path in the pager (file-size bounds now
      enforced as ValueError); SECURITY.md + examples/ (python API +
      MCP agent client, both verified runnable verbatim); pyflakes clean
      in all touched files (one pre-existing unused import remains in
      untouched lacuna/prefetch.py). Full suite 177/177 green.
- [ ] Day 7 — Fri Oct 23: RELEASE PREP + ZERO-SILENT-ERROR HARDENING
      ROUND 3 (Fahad's directive 2026-10-22: showcase release moved UP to
      Fri Oct 23, 1000s of users, no delay/no error) — DONE 2026-10-22
      (commit 3963c42): pack_q2 float32 fast path 1.8x + unpack_q4 1.6x
      (bit-identical, pinned vs float64 reference); server load hardening
      (backlog 5->128, JSON errors/405, 413 body drain, content:null fix);
      load test 240/240 @32 clients, 12-payload chaos clean; suite
      192/192 green. Remaining for Day 7: final gate pass + push (his PAT). (Fahad's directive
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
