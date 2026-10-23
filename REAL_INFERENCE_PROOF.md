# Real Inference Proof — Lacuna generating real text from a real model

Date: 2026-10-23 (Asia/Kolkata). Machine: this dev box — 2 CPU cores,
~8 GB RAM, no GPU. Everything below was actually run here; every quoted
output is copied verbatim from the command output files. Nothing in this
document is a simulator number.

## What changed

`lacuna/engine.py` (commit `d30cf1d`): a new `LlamaServerBackend`
implements the existing `InferenceBackend` protocol by driving a local
llama.cpp `llama-server` subprocess. `Engine("model.gguf")` now uses it
whenever a `llama-server` binary is available (PATH,
`$LACUNA_LLAMA_SERVER`, or the `llama_server=` argument), and reports
`mode == "libllama-dense"`. Before this change, a `.gguf` input silently
produced mock-vocabulary text through the orchestration path. Decoding is
pinned greedy (temperature 0, top_k 1, seed 42) so results are
reproducible and comparable to `llama-cli`.

## Setup (exact commands)

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp ~/workspace/tools/llama.cpp
# llama.cpp commit: 06230cb (build string: b1-06230cb)
cd ~/workspace/tools/llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build build -j2 --target llama-cli llama-server

curl -L -o ~/workspace/lacuna-models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

Model: `Qwen/Qwen2.5-0.5B-Instruct-GGUF`, file
`qwen2.5-0.5b-instruct-q4_k_m.gguf` (public, ungated)

- Size: **491,400,032 bytes**
- SHA-256: `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`
- First 4 bytes: `GGUF` (magic verified)
- llama.cpp reports: `ftype: Q4_K - Medium`

Test prompt (fixed for every run below):

```
What is the capital of France? Answer in one short sentence.
```

## Baseline — llama.cpp's own CLI

```bash
cd ~/workspace/tools/llama.cpp/build/bin
./llama-cli -m ~/workspace/lacuna-models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  -p "What is the capital of France? Answer in one short sentence." \
  -n 64 --temp 0 --top-k 1 --seed 42 --threads 2 -st --no-display-prompt < /dev/null
```

Exit code 0. The generation portion of stdout, verbatim (the full stdout
also contains the llama.cpp banner and command help, omitted here):

```
> What is the capital of France? Answer in one short sentence.
The capital of France is Paris.

[ Prompt: 79.5 t/s | Generation: 26.0 t/s ]
```

Note: with no conversation flag either way, current llama-cli enters
conversation mode for this model and applies its ChatML chat template to
the `-p` string. A first probe without `-st` generated the same answer
at 83.7 / 25.8 t/s and then waited for interactive input.

**Baseline generated text: `The capital of France is Paris.`**

## Lacuna path 1 — Engine API

```python
import os
os.environ["LACUNA_LLAMA_SERVER"] = "/home/hatch/workspace/tools/llama.cpp/build/bin/llama-server"
from lacuna.engine import Engine

eng = Engine("/home/hatch/workspace/lacuna-models/qwen2.5-0.5b-instruct-q4_k_m.gguf")
eng.generate("<|im_start|>user\nWhat is the capital of France? Answer in one short sentence.<|im_end|>\n<|im_start|>assistant\n", max_tokens=64)
```

The backend sends the prompt string verbatim to llama-server's
`/completion` endpoint (it applies no template of its own), so the
caller supplies the same ChatML rendering llama-cli constructed.
Program output, verbatim:

```
mode: libllama-dense | engine init (incl. server load): 1.32s
[templated] wall: 0.60s
[templated] output repr: 'The capital of France is Paris.'
[templated] server timings: {"cache_n": 0, "prompt_n": 21, "prompt_ms": 259.026, "prompt_per_token_ms": 12.33457142857143, "prompt_per_second": 81.07294248453823, "predicted_n": 8, "predicted_ms": 334.361, "predicted_per_token_ms": 47.76585714285714, "predicted_per_second": 20.935455989185343}
[raw] wall: 0.67s
[raw] output repr: ' The capital of France is Paris.'
closed ok
```

The `[raw]` run passed the bare question with no template: same sentence,
plus a leading space (a raw continuation artifact — the first generated
token is ` The`). The `[templated]` output is the one to compare against
the baseline.

## Lacuna path 2 — `lacuna run` CLI

```bash
printf '<|im_start|>user\nWhat is the capital of France? Answer in one short sentence.<|im_end|>\n<|im_start|>assistant\n' \
  | LACUNA_LLAMA_SERVER=/home/hatch/workspace/tools/llama.cpp/build/bin/llama-server \
    PYTHONPATH=~/workspace/lacuna \
    /home/hatch/workspace/ai-tools/corsac/.venv/bin/python -m lacuna.cli run \
    ~/workspace/lacuna-models/qwen2.5-0.5b-instruct-q4_k_m.gguf --max-tokens 64
```

Full stdout, verbatim:

```
lacuna v0.3.0 | RAM 7.8 GB (hot budget 5.8 GB) | VRAM 0.0 GB (working 0.0 GB) | NVMe 3.5 GB/s | page 2 MB | K=4
mode: libllama-dense | exact: True | precision: exact
 The capital of France is Paris.
```

One whitespace difference vs the API run: the CLI reads the prompt from
stdin with `.strip()` (`lacuna/cli.py`), which removes the trailing
newline after `<|im_start|>assistant`. That changes the final prompt
token, and the model's first generated token becomes ` The` instead of
`The`. The sentence content is identical; the leading space is exactly
the `[raw]`-style artifact described above.

## Comparison

| Path | Generated text | Match |
|---|---|---|
| llama-cli (baseline) | `The capital of France is Paris.` | — |
| Lacuna Engine API (templated prompt) | `The capital of France is Paris.` | **exact, character-for-character** |
| Lacuna Engine API (raw prompt) | ` The capital of France is Paris.` | differs by one leading space (no template) |
| Lacuna `lacuna run` CLI (prompt via stdin) | ` The capital of France is Paris.` | differs by one leading space (CLI strips the trailing newline) |

On identical context with greedy decoding, the Lacuna path and llama.cpp's
own CLI produce the same tokens — the server reported 8 predicted tokens
for the templated run, ending at the model's EOS, same as the CLI's
single-sentence stop. Speeds measured on this 2-core box are in the same
range on both paths (prompt ~80 t/s, generation ~21–26 t/s); they are
single runs on a shared VM, not benchmarks.

## Tests

```
PYTHONPATH=~/workspace/lacuna /home/hatch/workspace/ai-tools/corsac/.venv/bin/python -m pytest ~/workspace/lacuna/tests/ -q
195 passed, 1 skipped in 35.26s
```

The one skip is the env-gated real-inference test in
`tests/test_llama_backend.py`; with `LACUNA_LLAMA_SERVER` and
`LACUNA_TEST_GGUF` set it runs for real: **4 passed**, including a live
end-to-end `Engine` generation against the model above.

## What this proves — and what it does not

Proves:

- Lacuna's public API (`Engine`) and CLI (`lacuna run`) can now produce
  **real text from a real model** on an ordinary CPU-only machine,
  through the engine-swap seam the codebase always intended.
- Given the same context and greedy decoding, output through Lacuna is
  **identical to llama.cpp's own CLI** — the integration adds no drift.
- The mode label is truthful: `libllama-dense`, and the mock is no longer
  what a `.gguf` user gets when llama.cpp is installed.

Does **not** prove:

- **The pager was not in the loop.** Generation ran dense inside the
  llama-server process; Lacuna's sparse paging, oracle, and exactness
  machinery did not touch these weights. Sparse/paged real-weight
  execution (mode `"lacuna"`) is still unimplemented — this change wires
  the dense fallback, honestly labeled, nothing more.
- **No large-model claim.** This is a 0.5B model on CPU. Nothing here
  measures or validates 100B-class behavior, the RTX 2050 floor target,
  or any tok/s figure in the simulator tables.
- **Chat templating is the caller's job** on this path today: the
  backend forwards prompt strings verbatim. Product chat behavior (the
  servers in `lacuna/server.py` / `ollama_shim.py` flatten messages to a
  plain prompt) will need template rendering added before instruct
  models answer in chat format through those endpoints.
- The ctypes `libllama` route (`_try_load_libllama`) remains a loader
  stub only; the subprocess server route is the one implemented here.
