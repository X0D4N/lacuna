# Lacuna

[![CI](https://github.com/X0D4N/lacuna/actions/workflows/ci.yml/badge.svg)](https://github.com/X0D4N/lacuna/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

Lacuna runs large language models that do not fit in RAM by paging their
weights from disk, with output exact by construction in the default mode.
The headline claim is the big-model one, and it is stated below with
measured numbers only. Everything else in this file is secondary to that.

## What is proven (measured)

All figures in this section are **measured** and come from
[`PAGED_GENERATION_PROOF.md`](PAGED_GENERATION_PROOF.md) and
[`REAL_WEIGHTS_LADDER_R2.md`](REAL_WEIGHTS_LADDER_R2.md). Anything not
measured is labeled as such. No simulator numbers appear here.

### 110B: real tokens through the pager

On a Kaggle CPU host (4 vCPU, 32 GB RAM), the paged executor loaded
Qwen1.5-110B-Chat Q2_K (41,175,701,984 bytes, SHA-256
`922e4ce5…6062`, public file, no token) and generated real, coherent
tokens:

> " Paris. It is located in the north"

- Download: 1428.8 s at 28.8 MB/s. Conversion to `.lac` (exact policy,
  963/963 tensors raw): 324.2 s.
- Generation, prompt "The capital of France is": 8 tokens before a time
  cap cut it off. Prefill 2148.3 s; per-token 2093.74–2191.95 s
  (about 35 minutes per token). Store: 181,525 faults, 366.9 GB read.
  Peak RSS 19,864.2 MB; the 8 GB hot and 8 GB dequant budgets held, no
  out-of-memory.
- Second prompt ("Name three primary colors"): prefilled in 2249.8 s,
  emitted 0 tokens. The stop reason is not recorded — missing, not
  guessed.
- Dense baseline: **not completed**. The dense run timed out at the
  600 s cap, and the harness's own timeout handler then crashed the
  script (`TypeError`), which is why the kernel status reads ERROR.
  That failure is in the baseline harness, not the pager — but it means
  **no dense token-identity comparison exists at 110B**. The mechanism
  is proven at this scale; token-identity against dense is not, and
  speed at this scale is not: ~35 min/token on free-CPU hardware is
  unusable as-is.

### 70B / 104B: conversion ladder, bit-exact

Run 2 of the real-weights ladder (corrected run; run 1 found a real
`>2 GiB` read bug that truncated one tensor, and the fix is what run 2
verifies):

| Model | Convert | Reconstruction vs source bytes | Pager probe @ 512 MiB | Paged matvec vs dense |
|---|---|---|---|---|
| Llama-3.3-70B Q4_K_M | 319.5 s, `.lac` 42,513,323,248 bytes | 724/724 tensors, 42,512,531,712 bytes, 0 mismatches | 13/13 verified, budget held | 3 tensors, max abs diff 0.0 |
| Command R+ 104B Q2_K | 305.0 s, `.lac` 39,487,197,557 bytes | 642/642 tensors, 39,486,472,192 bytes, 0 mismatches | 13/13 verified, budget held | 3 tensors, max abs diff 0.0 |
| Qwen3-30B-A3B Q4_K_M | 115.5 s (run 1) | 579/579 tensors, 0 mismatches (run 1 full; run 2 spot re-check 25/25) | 13/13 verified, budget held | 3 tensors, max abs diff 0.0 |

One measured contrast, from the dense side: a 42.5 GB dense model could
not even load on the 32 GB host (llama-server SIGKILLed during load).
Lacuna's path never holds the model in RAM, which is the whole point
of paging it. Full ladder reports:
[`REAL_WEIGHTS_LADDER.md`](REAL_WEIGHTS_LADDER.md),
[`REAL_WEIGHTS_LADDER_R2.md`](REAL_WEIGHTS_LADDER_R2.md).

### 7B: speculative decoding is faster, and exact

With a 0.5B-class draft model verifying against a 7B target (K=4,
target budgets 256 MB page + 256 MB dequant, draft 128 MB + 128 MB):

- Speculative: 32 tokens in 8 rounds, 24/32 accepted (75%),
  4.0 tokens per target pass, 0.0144 tok/s, peak RSS 2167 MB.
- Plain paged on the same prompt: ≈0.0055 tok/s. Speedup ≈2.6x.
- Token ids identical 32/32 between speculative and plain. The
  mechanism is exact by construction; the draft is what makes it fast,
  and this is the result that proves a real draft does so. (A weaker
  n-gram draft, by contrast, was exact but slower than plain — the
  draft quality is the gap, not the mechanism.)
- Dense comparison at 7B diverged at one token (dense `.` vs paged
  `,` at token 13). Verdict, measured: **near-tie, not a bug** — paged
  margin 0.002362, dense margin ≈0.0064, same top-5 candidate set,
  13/13 self-consistent earlier positions. Greedy decoding amplifies a
  coin flip into a visible text difference. See the proof doc for the
  full table and the stated residuals.

## Install

Python 3.10 or newer. The only runtime dependency is numpy.

```bash
git clone https://github.com/X0D4N/lacuna.git && cd lacuna
pip install .            # or: pip install -e .  for an editable install
```

`./install.sh` (Linux, Debian/Arch) and `./install.ps1` (Windows 11)
do the same via the system package manager. If your venv has no
setuptools, pip's build isolation downloads it automatically when
network is available; otherwise install setuptools first. The package
is not published on PyPI yet — install from source as above.

## Quickstart: run a big model first

This is the route the proof above used. One command resolves a
HuggingFace repo id, downloads the GGUF, converts it to Lacuna's paged
`.lac` format, and runs it through the pager:

```bash
lacuna run second-state/Qwen1.5-110B-Chat-GGUF "The capital of France is" --max-tokens 8
```

What happens, step by step:

1. `lacuna run <owner>/<repo>` pulls if missing: it picks the GGUF in
   the repo (Q4_K_M first, else the smallest file), downloads it with
   resume to `~/.cache/lacuna/models/<owner>__<repo>/`, and converts
   it to `.lac` beside it. The conversion is exact-policy by default:
   source bytes are stored raw, never re-quantized. A second run
   reuses the existing `.lac` and downloads nothing.
2. Generation runs through the pager in exact mode (the default).
   The CLI prints its mode on startup — `mode: lacuna` means the real
   paged executor; anything else is labeled, never silent.

Honest time and disk expectations, from the measured 110B run above:

- Disk: about 85 GB free for a 110B-class model (the GGUF plus its
  `.lac`, roughly the same size under exact policy).
- Download of 41.2 GB took 1428.8 s on the measured connection; yours
  depends on your network. Conversion took 324.2 s.
- Generation on CPU-only weak hardware is slow: about 35 minutes per
  token at 110B scale on a 4-vCPU host. The 8-token proof above took
  hours. Do not expect interactive speed from a 110B on a CPU-only
  machine — usable speed at that scale is unmeasured (see "Not yet").
- The 110B proof ran on a 32 GB host with 8 GB budgets. **110B on the
  6 GB floor below is unmeasured.** What is measured at floor-class
  budgets is the 7B run above (256 MB budgets, 2167 MB peak RSS).

Hardware floor (design target): RTX 2050 (4 GB VRAM), 6 GB RAM, NVMe
SSD. Linux (Debian/Arch) and Windows 11.

## Smoke test: small model, labeled as such

This only proves your install works. It is **not** evidence for the
big-model claim above — never read it as such.

```bash
lacuna pull Qwen/Qwen2.5-0.5B-Instruct-GGUF
lacuna run Qwen/Qwen2.5-0.5B-Instruct-GGUF "The capital of France is" --max-tokens 32
```

Expect `mode: lacuna` and coherent text beginning with ` Paris`.
Wall time depends on machine load and budget; the 0.5B demo is small
and is not the result this repo is selling. The same call works from
Python:

```python
from lacuna import Engine
print(Engine("model.gguf.lac").generate("hello"))
```

## How it works, in plain words

1. **Convert.** `lacuna convert model.gguf model.lac` (or `pull`, which
   converts automatically) copies the model's tensors into 2 MB pages
   in a `.lac` file. Under the default exact policy the bytes are
   stored raw — the `.lac` is the model, paged, not a re-quantization.
2. **Page.** At run time the pager faults pages from disk under a hard
   budget (LRU, exact-length reads) and never holds the whole model.
   A dequantized-weight cache, also bounded and evictable, keeps
   repeat tokens from re-decoding the same weights.
3. **Execute.** The real executor (currently Qwen2 only) runs the
   model's own transformer math — RMSNorm, grouped-query attention,
   half-rotation RoPE, SwiGLU, KV cache, greedy decoding — with every
   weight byte arriving through the pager. If the executor cannot load
   a model, the engine falls back to a labeled `[simulated]` path and
   keeps the reason inspectable; a downgrade is never silent.
4. **Speculate (optional, exact).** A small draft model proposes K
   tokens; the target verifies the whole block in one paged pass and
   keeps only what it agrees with. Output is identical to plain paged
   generation by construction — the 7B proof above is the measurement.
   A bad draft costs time, never correctness.

Related components live in [`docs/`](docs/): the full architecture
([`ARCHITECTURE.md`](docs/ARCHITECTURE.md)), the speculative design
([`SPECULATIVE_PAGED_DESIGN.md`](docs/SPECULATIVE_PAGED_DESIGN.md)),
prior art ([`PRIOR_ART.md`](docs/PRIOR_ART.md)), and per-model targets
([`TARGET_MODELS.md`](docs/TARGET_MODELS.md)). `lacuna serve` exposes
an OpenAI-compatible server (with an Ollama-compatible mode), and
`lacuna mcp` exposes an MCP server; run either with `--help` for its
options. `lacuna bench` / `lacuna simulate` print simulator estimates
— those are labeled simulated wherever they appear and are not results.

## Not yet

Stated plainly, so nobody has to discover these by tripping over them:

- **Prefetch pipelining is implemented, not yet measured at 110B.**
  Async submit with a pager stats-lock fix is in
  (`lacuna/pager.py`, `lacuna/paged_model.py`), with correctness
  tests green on a tiny model and a smoke check on the 0.5B.
  The 110B measurement kernel is launched and running; its
  numbers land when it finishes. Until then, no 110B prefetch
  speed is claimed.
- **GPU executor core is built, GPU measurements pending.** A
  single Qwen2 layer forward (PyTorch CUDA primary, numpy
  fallback) is in [`lacuna/gpu_executor.py`](lacuna/gpu_executor.py),
  CPU-validated (14 passed, 1 CUDA-only test skipped here).
  The Kaggle GPU test is launched; full-model wiring waits
  on it passing. No GPU speed is claimed anywhere in this repo.
- **110B dense token-identity is unproven** (no baseline completed, as
  above). **110B on the 6 GB floor is unmeasured.**
- **Real executor supports Qwen2 only, for now.** Other architectures
  fall back to the labeled `[simulated]` path.
- **Exactness is token-exact until near-ties**, not bit-exact across
  implementations: at genuinely ambiguous positions (0.5B step 19,
  7B token 13, both measured), float rounding flips the greedy argmax
  between implementations. That is noise, not a logic error, and it is
  the reason near-tie margins are reported rather than hidden.
- PyPI publishing is in preparation; the 0.3.0 package is source-only
  for now.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). House rules: stdlib + numpy
only, every published number measured or labeled simulated, Linux +
Windows 11 only. The security model (local-first, no telemetry,
localhost by default) is in [`SECURITY.md`](SECURITY.md).

## License

Apache-2.0 — see [LICENSE](LICENSE).
