# Lacuna — Week Hardening Plan (Oct 17–23, 2026)

Owner: Fahad. Deadline: Saturday Oct 24, 2026 — professional, future-invention level,
trending-ready. Push to CYPHERLYNX/lacuna happens Saturday AFTER Fahad creates the
empty public repo in the GitHub web UI (connector got 403 on creation) and provides
a PAT or installs the Muse GitHub app. Do NOT push before that.

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
- [ ] Day 4 — Tue Oct 20: GGML INTEGRATION. Advance integrations/llama.cpp/:
      concrete minimal patch or pager-backed buffer prototype moving the design
      toward real C++ integration. Honest about what runs vs what's design.
- [ ] Day 5 — Wed Oct 21: BENCHMARK EXPANSION. More simulator scenarios
      (sparsity sweeps, K sweeps, bandwidth variants); updated honest tables in
      README + benchmarks/RESULTS.md.
- [ ] Day 6 — Thu Oct 22: TRENDING POLISH. README/docs final pass, CHANGELOG.md,
      release-notes draft, verify `pip` sdist builds and installs cleanly,
      review install.sh / install.ps1.
- [ ] Day 7 — Fri Oct 23: RELEASE PREP. Full review pass, version bump, final
      green test run, write the Saturday push checklist (repo create + push
      commands) at the top of this file.
