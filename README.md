# Lacuna

[![CI](https://github.com/CYPHERLYNX/lacuna/actions/workflows/ci.yml/badge.svg)](https://github.com/CYPHERLYNX/lacuna/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

**Sparse-paged LLM inference for very large models on small hardware.**

Lacuna runs models that do not fit in RAM or VRAM by paging their weights
from NVMe on demand — and makes that fast enough to be usable by predicting
which weights are needed *before* they are needed. A small activation
oracle looks K tokens ahead and produces the union of weight pages those
tokens will touch; the pager streams exactly those pages while the current
token computes. Prediction only schedules memory, never math: in the
default exact mode the output is bit-identical to running the model dense,
which the test suite proves (including against a deliberately broken
oracle — a bad prediction costs time, never correctness).

Minimum target hardware: RTX 2050 (4 GB VRAM), 6 GB RAM, NVMe SSD.
Linux (Debian/Arch) and Windows 11. Python, stdlib + numpy only.

## Install

```bash
git clone https://github.com/CYPHERLYNX/lacuna.git && cd lacuna
./install.sh        # Linux
# .\install.ps1     # Windows 11 (installs Python via winget if missing)
```

A PyPI package (`lacuna-llm`) is in preparation; until it is published,
install from source as above.

## Usage

```bash
lacuna pull <huggingface-repo>     # download a GGUF model + convert to .lac
lacuna run model.gguf "Explain sparse attention"
lacuna run model.gguf "..." --extreme   # adaptive precision, faster, lossy
lacuna simulate                    # estimated tok/s for your hardware class
```

`--exact` is the default: per-page adaptive precision (Q4/Q2/ternary) is
available only under `--extreme`, as an explicit opt-in.

**OpenAI-compatible server** — works with any OpenAI client library:

```bash
lacuna serve model.gguf --port 8080
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="not-needed")
r = client.chat.completions.create(
    model="lacuna-mock",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=64,
)
```

Endpoints: `GET /health`, `GET /v1/models`, `POST /v1/completions`,
`POST /v1/chat/completions` (SSE streaming). An Ollama-compatible shim is
available via `lacuna serve --ollama`.

**MCP server** — for agent frameworks:

```bash
lacuna mcp
```

```json
{ "mcpServers": { "lacuna": { "command": "lacuna", "args": ["mcp"] } } }
```

**Python:**

```python
from lacuna import Engine
print(Engine("model.lac").generate("hello"))
```

Runnable examples are in [`examples/`](examples/). The security model
(local-first, no telemetry, localhost by default) is in
[`SECURITY.md`](SECURITY.md).

## How it works

Five mechanisms, one contract — *reserve the memory before it is needed*:

1. **Activation oracle** — per-layer predictors estimate which units fire
   for the next K draft tokens; the union of their pages is the prefetch set.
2. **Paged weight store** — weights live in a `.lac` file as 2 MB pages,
   streamed from NVMe with exact-length reads.
3. **Hot-set residency** — frequently used pages stay resident under hard
   RAM/VRAM caps (LRU + frequency), persisted across runs with checksums,
   so repeat runs start warm.
4. **Overlap scheduler** — prefetch for tokens t+1..t+K runs concurrently
   with compute for token t; misses fault on demand and are counted.
5. **Tree speculation** — draft tokens are verified in one sparse pass;
   acceptance is exact by construction.

**Ordo** ([`docs/ORDO.md`](docs/ORDO.md)) compiles routing traces into an
automaton of phase states, each with a pre-assembled page segment, so
MoE expert pages can be staged as one sequential read instead of scattered
faults. Full design: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Performance

All figures below are **simulator estimates**, not measurements — the
simulator is calibrated to published RTX 2050 / NVMe 3.5 GB/s figures and
its methodology is documented in
[`benchmarks/METHODOLOGY.md`](benchmarks/METHODOLOGY.md). Measured results
(codec and pager hot paths, perplexity at small scale) are in
[`benchmarks/RESULTS.md`](benchmarks/RESULTS.md).

| Scenario (100B-class, floor hardware) | tok/s | GB/token |
|---|---|---|
| Dense, naive streaming | 0.14 | 25.00 |
| + oracle sparsity (90%) | 1.33 | 2.62 |
| + speculative K=4 over union mask (Lacuna, exact) | 3.54 | 0.99 |
| MoE 100B (20B active), hot-set resident | 14.00 | 0.25 |

For context: no measured system currently exceeds ~5 tok/s for 100B+
models on ≤4 GB VRAM with output quality held
([`docs/PRIOR_ART.md`](docs/PRIOR_ART.md)). Treat the Lacuna figures as
targets for the real-hardware validation below, not as results.

## Status

Current release: **0.3.0**. What is implemented and tested (192 tests):
the paging engine, exactness guarantees, codecs, oracle and Ordo,
warm-start cache, the CLI, the OpenAI/Ollama/MCP servers, and the
llama.cpp backend-buffer contract (`integrations/llama.cpp/`).

What is not done yet, plainly:

- The generation backend wired to the servers today is a deterministic
  reference implementation (responses are labeled `[mock]`). Real-weight
  GGUF execution through the pager is the next milestone; the engine-swap
  interface it will plug into already exists.
- Real-hardware validation on a 4 GB VRAM machine is pending; until then
  all speed figures are simulator estimates, as labeled above.
- PyPI publishing is in preparation.

Roadmap and per-model targets (including the trillion-parameter tier):
[`docs/TARGET_MODELS.md`](docs/TARGET_MODELS.md).

## Related work

PowerInfer, FlexGen, HOBBIT (closest prior art), EAGLE, BitNet — and what
differs here — are surveyed in [`docs/PRIOR_ART.md`](docs/PRIOR_ART.md).

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). House rules: stdlib + numpy
only, every published number measured or labeled simulated, Linux +
Windows 11 only.

## License

Apache-2.0 — see [LICENSE](LICENSE).
