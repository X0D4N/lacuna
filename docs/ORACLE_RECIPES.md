# Oracle Recipes — per-family calibration, training, and evaluation

How to produce a Lacuna M1 activation oracle for a real model, per model
family. The code lives in `lacuna/recipes.py`; the CLI entry point is
`lacuna oracle-recipe`. Every number below is either a public architecture
fact (cited), a measured synthetic result (labeled SYNTHETIC), or explicit
arithmetic -- never a claim about a real model we have not trained on.

## The 4-step procedure

1. **Pick a family preset** (`qwen` / `llama` / `moe`) or build a
   `FamilySpec` from your model's own `config.json` with `spec_from_dims()`.
   The preset dims are starting points -- always substitute your model's
   real values; training validates them against the dataset.
2. **Capture (X, Y) calibration pairs** from the real model (procedure
   below), or generate synthetic pairs with `synthetic_calibration()` to
   validate the pipeline (labeled SYNTHETIC).
3. **Train** with `train_family_oracle()` -- one small MLP per layer
   (plus a router oracle per layer for MoE).
4. **Evaluate and gate** with `evaluate_dataset()` + `gate_report()`. The
   gate -- not a fixed epoch count -- is the stop criterion: raise epochs
   (or calibration size) until every layer passes.

## Family reference configs

Public `config.json` values for the 7B-class reference model of each family.
The recipe is size-agnostic; these are the defaults, not the only supported
sizes.

| family | reference | layers | hidden | intermediate | heads | KV heads | experts | top-k | FFN | norm |
|---|---|---|---|---|---|---|---|---|---|---|
| `qwen` | Qwen2.5-7B | 28 | 3584 | 18944 | 28 | 4 | -- | -- | SwiGLU | RMSNorm |
| `llama` | Llama-3.1-8B | 32 | 4096 | 14336 | 32 | 8 | -- | -- | SwiGLU | RMSNorm |
| `moe` | Mixtral-8x7B | 32 | 4096 | 14336 | 32 | 8 | 8 | 2 | SwiGLU | RMSNorm |

What the oracle predicts per layer:

- **Dense (qwen/llama):** which of the `intermediate` FFN neurons carry
  signal -- the binarized `silu(gate(x)) * up(x)` product. Attention
  projections stay hot-resident (ARCHITECTURE.md M3) and are not predicted.
- **MoE (moe):** two oracles per layer. A *router oracle* predicts the
  top-k expert mask (`n_experts` units); an *FFN oracle* predicts per-expert
  neuron masks (`n_experts * intermediate` units). The prefetch schedule is
  the union over the lookahead of `expert_mask x neuron_masks`.

## Capturing real activations

**Status: specified, not built.** Dumping hidden states and FFN
intermediates during a real forward pass needs a hook into the execution
core (llama.cpp tensor access) -- that hook lands with the Day 4 GGML
integration track. What is fixed today is the **contract** the hook must
satisfy: it writes the `.npz` schema defined by
`save_dataset()`/`load_dataset()` (`X_{i}`, `Y_{i}`, plus `R_{i}` for MoE),
and everything downstream trains from that file unchanged.

Per family, for each layer `i`:

- **qwen / llama** (GGUF `blk.{i}.ffn_gate`, `ffn_up`, `ffn_down`):
  `X` = FFN-block input (post-attention RMSNorm output), shape
  `(n_tokens, hidden_dim)`. Run the gate/up projections, compute
  `silu(gate) * up`, and binarize per neuron at quantile `q = 0.90` over the
  calibration corpus -> `Y`, shape `(n_tokens, intermediate)`.
- **moe** (GGUF `blk.{i}.ffn_gate_inp`, `ffn_gate_exps`, `ffn_up_exps`,
  `ffn_down_exps`): `X` = MoE-block input. `R` = top-k mask from the
  `ffn_gate_inp` router logits, shape `(n_tokens, n_experts)`. For each
  active expert `e`, `Y` over its `intermediate` neurons = binarized
  `silu(gate_e) * up_e` at per-expert-neuron quantile `q = 0.90` computed
  over routed tokens only.

Calibration text guidance: 16k-65k tokens (e.g. 128-512 sequences x 128-512
tokens) drawn from the model's deployment domain, deduplicated. More tokens
help monotonically; the gate tells you when you have enough.

Storage arithmetic (honest, fp32 X, bitmask Y), 16k tokens:

| model | X total | Y total | peak (streamed per layer) |
|---|---|---|---|
| Llama-3.1-8B | 32 x 268 MB = 8.6 GB | 32 x 29 MB = 0.94 GB | ~300 MB |
| Mixtral-8x7B | 32 x 268 MB = 8.6 GB | 32 x 235 MB = 7.5 GB | ~500 MB |

The recipe supports per-layer streaming (capture -> train -> discard, never
holding all layers), so calibration fits in RAM on the 8 GB target box.

## Training

`train_family_oracle(spec, dataset, ...)` trains each layer's MLP with the
same numpy SGD as `lacuna/oracle.py` (binary cross-entropy, automatic
positive-class weighting). Validated regimes (measured on synthetic
calibration, not guessed):

| kind | epochs | lr | batch | mlp_hid |
|---|---|---|---|---|
| dense | 60 | 0.1 | 256 | 128 |
| moe (FFN oracle) | 60 | 0.02 | 256 | 128 |
| moe (router oracle) | 60 | 0.1 | 256 | 64 |

The MoE FFN oracle sees much sparser targets
(~`top_k/n_experts * 0.10`), hence the larger positive weight and the
gentler lr (lr=0.1 diverges there -- measured). The router's targets are
k-hot and train in the dense regime. For real models: keep the code path
identical, raise epochs until the gate passes.

Expected cost: numpy-only by design (same constraint as the engine). A
7B-class dense model is a CPU-overnight job; the per-layer oracles are
independent and parallelize trivially across cores.

## Evaluation and the quality gate

Held-out split per layer (deterministic): 80% train, 10% validation, 10%
test. The **operating threshold is tuned, not assumed**: `tune_threshold()`
scans a grid on the validation split and picks the threshold maximizing
recall subject to predicted density <= 0.20 (ties break toward the sparser
schedule). All reported metrics are measured on the untouched test split,
and each report row carries the threshold actually used.

Metrics per layer:

- **recall** at the tuned threshold -- fraction of truly-active units
  predicted. Misses cost on-demand faults.
- **predicted_density** -- fraction of units predicted active. This is the
  prefetch cost knob; the 0.20 cap keeps prefetch cheap.
- **union_recall** over K=4 lookahead windows -- recall of the union mask,
  which is the actual prefetch schedule (M1 predicts unions, not tokens).
- **router recall / density** (MoE): density capped at 2x the true expert
  density.

`gate_report()` passes a recipe only if **every** layer clears:

| check | bar |
|---|---|
| recall | >= 0.85 |
| predicted_density | <= 0.20 |
| union_recall (K=4) | >= 0.90 |
| router recall (MoE) | >= 0.85 |
| router predicted_density (MoE) | <= 2 x top_k/n_experts |

If a layer fails: add calibration tokens from the failing domain, check the
density parameter matches the model's true sparsity, raise epochs, or widen
`mlp_hid`. Do not lower the bars to make a run pass -- the bars are the
recipe.

## Synthetic calibration

`synthetic_calibration(spec, n_tokens=4096, density=0.10, seed=...)`
generates (X, Y) pairs where Y is the binarized SwiGLU product of X through
random projections -- a deterministic function of X, so a well-trained
oracle reaches the gate. The hidden states are a low-rank mean-reverting
walk (smooth token-to-token, stationary, anisotropic like real hidden
states), and MoE routing is top-k of random logits. This validates the
**pipeline** (generate -> train -> evaluate -> gate -> save/load);
it says nothing about real-model quality. All synthetic output is labeled
SYNTHETIC wherever it appears.

Measured on this box (SYNTHETIC, quick specs, seeds fixed):

| family | min recall | max density | min union recall | gate |
|---|---|---|---|---|
| qwen-quick (2 layers) | 0.877 | 0.196 | 0.925 | PASS |
| moe-quick (2 layers) | 0.929 | 0.197 | 0.961 | PASS |

Router oracle (moe-quick): recall 0.99 at density ~0.65-0.75 -- routing is
inherently less sharp than neuron prediction (top-k ranking under
uncertainty); the recipe reports the achieved expert-prefetch reduction
honestly instead of claiming top-k exactness.

## CLI

```bash
# validate the pipeline for a family (synthetic, scaled down -- seconds)
lacuna oracle-recipe --family qwen --quick

# full synthetic recipe for a family (minutes; exercises the real code path)
lacuna oracle-recipe --family moe --tokens 4096 --epochs 60

# train from real captured data (see "Capturing real activations")
lacuna oracle-recipe --family llama --real-data calib_llama.npz --save oracles/llama_ffn.npz

# save the calibration set for inspection / reuse
lacuna oracle-recipe --family qwen --tokens 8192 --save-data calib_qwen.npz

# list family presets
lacuna oracle-recipe --list-families
```

`--save` writes the trained FFN oracle (`.npz`, loadable with
`Oracle.load`); for MoE the router oracle is saved alongside as
`<save>.router.npz`. The command exits 0 iff the quality gate passes, 1
otherwise -- wire it into CI as the recipe's acceptance test.

## What is NOT claimed

- No perplexity or downstream quality numbers: in exact mode the oracle
  affects speed only (proven by `tests/test_exact.py`), so oracle quality
  is reported as recall/density, never as model quality.
- No real-model measurements exist yet: the capture hook is specified
  above but not built. The first MEASURED (non-synthetic) recipe run will
  be recorded here with its full report table when the hook lands.
