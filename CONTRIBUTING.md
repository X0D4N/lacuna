# Contributing to Lacuna

## Ground rules

1. **Stdlib + numpy only.** No new dependencies without a design discussion.
2. **Honest numbers.** Every performance claim is labeled SIMULATED (from
   `simulator/`) or MEASURED (from `tests/` or real hardware runs). Never
   invent benchmark numbers.
3. **Linux and Windows 11 only.** No macOS code paths, docs, or claims.
4. **Tests are mandatory.** `pytest tests/` must pass; keep the suite under
   2 minutes on a small box.
5. **Small and sharp.** This project stays a few thousand lines. A PR that
   doubles the codebase needs a design note first.

## Workflow

```bash
pip install .            # or ./install.sh / install.ps1
python -m pytest tests/ -q
python -m simulator.simulate
```

## What needs help

- Real-hardware validation on an RTX 2050 / 4 GB VRAM box (see
  `benchmarks/RESULTS.md` checklist).
- The ggml backend hook design in `integrations/llama.cpp/INTEGRATION.md`.
- Oracle training recipes per model family.
