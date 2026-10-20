# Lacuna

[![CI](https://github.com/CYPHERLYNX/lacuna/actions/workflows/ci.yml/badge.svg)](https://github.com/CYPHERLYNX/lacuna/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/lacuna-llm.svg)](https://pypi.org/project/lacuna-llm/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

**Lookahead sparse-paged LLM inference — run 100B-class models on 8 GB RAM + <4 GB VRAM.**

One invention, five mechanisms, one small engine: an activation oracle predicts
the *union* of weights needed over a K-token lookahead window, an NVMe pager
streams those pages ahead of compute, and speculative verification amortizes
the streaming cost across K tokens. Built on llama.cpp. Stdlib + numpy only.

## Honest capability table (ALL NUMBERS SIMULATED)

Target: RTX 2050 4GB VRAM / 8GB RAM / NVMe 3.5 GB/s. Methodology:
[`benchmarks/METHODOLOGY.md`](benchmarks/METHODOLOGY.md). Dense 100B will never
*feel* like a 3B model on this hardware — physics forbids it. The honest target
is **usable**, not instant.

| Scenario | tok/s | GB/token | p99 latency |
|---|---|---|---|
| Dense 100B @2-bit, naive streaming | 0.14 | 25.00 | 8.21 s |
| + 90% oracle sparsity | 1.33 | 2.62 | 0.86 s |
| **+ speculative K=4 over union mask (Lacuna)** | **3.54** | 0.99 | 0.32 s |
| **Lacuna exact mode (default)** | **≈3.54¹** | 0.99 | 0.32 s |
| MoE 100B, 20B active @2-bit, hot-set resident | 14.00 | 0.25 | 0.08 s |

¹ Exact-mode speed is SIMULATED under the same assumptions as the row above
(oracle recall 0.95 — a bad oracle costs faults, i.e. time, never quality).
Exactness itself is not simulated: it is **proven by `tests/test_exact.py`**
— token-identical output to the dense baseline even with a deliberately
broken (random-mask) oracle — and `tests/test_tree_spec.py` proves tree
speculation accepts exactly the target's own greedy tokens. In exact mode the
oracle only schedules prefetches; it never decides what gets computed.
`--extreme` opts out for maximum tok/s at a documented quality tradeoff.

No measured system exceeds ~5 tok/s for 100B+ models on ≤4 GB VRAM with
quality held (see [`docs/PRIOR_ART.md`](docs/PRIOR_ART.md)) — Lacuna's
simulated 3.5 tok/s sits at that ceiling's edge, which is exactly where the
real-hardware validation in [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md)
will confirm or correct it.

### Sweeps (ALL NUMBERS SIMULATED)

100B, full Lacuna stack — `python -m simulator.sweeps`. Same stated
assumptions as the table above.

| Oracle sparsity | 80% | 85% | 90% | 95% | 97% |
|---|---|---|---|---|---|
| tok/s | 1.77 | 2.36 | 3.54 | 7.08 | 11.80 |

| Spec depth K | 1 | 2 | 3 | 4 | 5 | 6 | 8 |
|---|---|---|---|---|---|---|---|
| tok/s | 1.27 | 2.24 | 2.98 | 3.54 | 3.96 | 4.26 | 4.62 |

| NVMe GB/s | 1.75 | 3.50 | 7.00 |
|---|---|---|---|
| tok/s | 1.77 | 3.54 | 7.08 |

Trillion tier: all 10 target models in [`docs/TARGET_MODELS.md`](docs/TARGET_MODELS.md)
move >4 GB/token cold (active params × 0.5 bytes at Q4) → **DESKTOP-only
cold**, no exceptions. At the 95%-hot-set MID steady-state target, Kimi K2.7
Code reaches ~4.4 tok/s — closest to MID, exactly as the doc predicts.
Full per-model table in [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md).

## Install

Linux and Windows 11 only. No macOS support, by design.

```bash
# Debian / Ubuntu
sudo apt install python3 python3-pip && pip install lacuna-llm
# Arch
sudo pacman -Sy python python-pip && pip install lacuna-llm
# Windows 11 (PowerShell)
pip install lacuna-llm
# From source (either OS)
./install.sh        # Linux: detects apt vs pacman
# .\install.ps1     # Windows 11: installs Python via winget if missing
```

## Quickstart

One command does everything — hardware auto-detect, GGUF→`.lac` auto-convert,
sensible defaults, no config files:

```bash
lacuna run model.gguf "Explain sparse attention"
# --exact (default): output identical to the dense baseline, guaranteed
# --extreme:        adaptive precision, fastest, documented quality tradeoff
lacuna simulate        # print the honest tok/s table above
```

**OpenAI-compatible server** (any OpenAI client library works):

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
print(r.choices[0].message.content)
```

Endpoints: `GET /health`, `GET /v1/models`, `POST /v1/completions`,
`POST /v1/chat/completions` (SSE streaming supported).

**MCP server** (agents integrate with zero glue — Hermes and others):

```bash
lacuna mcp
```

```json
{
  "mcpServers": {
    "lacuna": { "command": "lacuna", "args": ["mcp"] }
  }
}
```

Tools: `lacuna_generate {prompt, max_tokens}`, `lacuna_convert
{input_gguf}`, `lacuna_benchmark {scenario}`.

**Python API** (one line):

```python
from lacuna import Engine
print(Engine("model.lac").generate("hello"))
```

Repeat runs start warm: the hot-set is persisted between runs
(checksum-validated; a stale cache just means a cold start), so the first
token doesn't re-pay the paging discovery cost.

**Runnable examples** (offline, mock backend): [`examples/python_api.py`](examples/python_api.py)
(API + auto-tune + warm start) and [`examples/agent_mcp_client.py`](examples/agent_mcp_client.py)
(driving the MCP server over stdio, the way agent frameworks integrate).
Security model — local-first, no telemetry, localhost-only by default:
[`SECURITY.md`](SECURITY.md).

## Architecture

```
                    ┌─────────────────────────┐
  draft model (tiny│  M5 speculative verify   │  K tokens, ONE sparse pass
  , fully resident)└────────────┬────────────┘
                                │ draft tokens t+1..t+K
                    ┌───────────▼────────────┐
  per-layer MLPs    │  M1 activation oracle  │  UNION mask over K tokens
                    └───────────┬────────────┘
                                │ page ids to prefetch
   ┌────────────────────────────▼────────────────────────────┐
   │  M4 scheduler: prefetch t+1..t+K while computing t      │
   │  (double-buffered; misses fault on demand, counted)    │
   └──────────────┬────────────────────────────┬─────────────┘
                  │                            │
      ┌───────────▼───────────┐    ┌───────────▼───────────┐
      │ M2 NVMe paged store   │    │ M3 hot-set residency  │
      │ .lac, 2MB pages,      │    │ 6GB RAM + 3.2GB VRAM  │
      │ io_uring (linux)      │    │ LRU + EWMA frequency  │
      └───────────────────────┘    └───────────────────────┘
                  M6 adaptive precision: hot Q4 / warm Q2 / cold ternary
```

Full design: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Related work

PowerInfer (activation sparsity + predictors), FlexGen (disk-offload LP
placement), HOBBIT (llama.cpp fork, expert-granularity mixed precision —
closest prior art), EAGLE (speculative decoding), BitNet (ternary training).
What Lacuna adds — draft tokens → union activation mask over a multi-token
lookahead, async NVMe prefetch overlapped with compute, per-page adaptive
precision driven by measured frequency, all integrated on llama.cpp — is the
documented gap in [`docs/PRIOR_ART.md`](docs/PRIOR_ART.md).

## Roadmap

- [ ] Real-hardware validation on a 4 GB VRAM box ([checklist](benchmarks/RESULTS.md))
- [x] ggml backend buffer hook — contract prototype + 11 MEASURED tests
  (`lacuna/buffer.py`); C++ patch (`integrations/llama.cpp/sp_buffer.cpp`)
  syntax-checked, real-tree compile pending a GPU box
  ([design](integrations/llama.cpp/INTEGRATION.md))
- [x] Oracle training recipes per model family (Qwen, Llama, MoE) — [`docs/ORACLE_RECIPES.md`](docs/ORACLE_RECIPES.md), `lacuna oracle-recipe`
- [ ] Perplexity evals: quality cost of the Q2/ternary cold tiers
- [ ] Windows NVMe: measured prefetch throughput tuning

## Contributing

Short and real: [`CONTRIBUTING.md`](CONTRIBUTING.md). Stdlib + numpy only,
honest numbers only, Linux + Windows 11 only.

## License

Apache-2.0 — see [LICENSE](LICENSE).
