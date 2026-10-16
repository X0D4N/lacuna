# Lacuna — Week Hardening Plan (Oct 17–23, 2026)

Owner: Fahad. Deadline: Saturday Oct 24, 2026 — professional, future-invention level,
trending-ready. Push to CYPHERLYNX/lacuna happens Saturday AFTER Fahad creates the
empty public repo in the GitHub web UI (connector got 403 on creation) and provides
a PAT or installs the Muse GitHub app. Do NOT push before that.

Daily cron (`lacuna-week-hardening`) implements the next unchecked track below.
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

- [ ] Day 1 — Sat Oct 17: EXACTNESS MERGE. A subagent built the Exactness upgrade
      (exact mode, tree speculation, precision policies). Review the diff for
      correctness and honesty, run the full suite, fix issues, commit as v0.2.0.
- [ ] Day 2 — Sun Oct 18: ORACLE RECIPES. Calibration/training scripts per model
      family (Qwen, Llama, MoE) + docs: how to capture activations, train the
      per-layer predictors, evaluate recall. Synthetic-data recipe included.
- [ ] Day 3 — Mon Oct 19: PERPLEXITY HARNESS. Honest quality measurement:
      exact vs extreme precision policy on a small model; methodology + results
      table in benchmarks/ (SIMULATED or MEASURED, labeled).
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
