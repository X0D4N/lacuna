# Gemma-4 31B in Lacuna — confirmed facts, loader contract, parity checklist

CPU paged path: `lacuna/gemma.py` (`PagedGemma4`) + `lacuna/spm.py`
(`SPMTokenizer`); GPU path: `lacuna/gpu_executor.py` (`GPUGemmaModel`,
Engine mode `lacuna-gpu`). Engine dispatch: `Engine("model.gguf")` tries Qwen2,
then Gemma-4, before the labeled simulated fallback. Real-weight parity
v5 passed 8/8 on BF16 2026-10-26 (see the parity note below); no pushes
or commits were made for this build.

## Target

`google/gemma-4-31b-it` — 30.7B dense, April 2026, Apache 2.0, public.
Text decoder only; the ~550M vision encoder is never read (vision
tensors, if present, are ignored by design).

## Confirmed facts (public sources, 2026-10-25)

| Fact | Value | Sources (converging) |
|---|---|---|
| Layers / hidden / vocab | 60 / 5376 / 262,144 | Kaitchup arch analysis; solidsf bench table; borowiecki paper setup; HF Gemma4TextConfig vocab |
| Embeddings | Tied (`tie_word_embeddings=True`) | Gemma4TextConfig defaults; solidsf table |
| Context | 262,144 tokens | Model cards, Kaitchup, solidsf |
| Layer pattern | 5 sliding : 1 full, ×10; global layers 0-based 5, 11, 17, 23, 29, 35, 41, 47, 53, 59; last layer forced global | aicassindra `layer_types` list; Kaitchup 50 sliding + 10 full; borowiecki partial global list; g4.si5.pl last-layer rule |
| Sliding layers | Window 1024, 32 Q / 16 KV heads, head_dim 256, RoPE default, theta 10,000, full rotary | Kaitchup config values; aicassindra; EAGLE3 blog (50×16×256, window 1024) |
| Global layers | 32 Q / 4 KV heads, head_dim 512, V = K (`attention_k_eq_v`), proportional RoPE, theta 1,000,000, full-dim NEOX rotation with stored `rope_freqs` factors (CORRECTED 2026-10-26: the 0.25 partial factor was a transformers-docs concept, not the GGUF runtime's behavior) | Kaitchup (`num_global_key_value_heads=4`, `global_head_dim=512`); aicassindra; solidsf (32/4 V=K, 256/512); g4.si5.pl `rope_parameters` block; llama.cpp `gemma4.cpp` (proportional rope via `rope_freqs.weight`) |
| MLP / norms | GELU-tanh gated MLP, RMSNorm eps 1e-6, no attention bias (`attention_bias=False`) | Gemma4TextConfig (`hidden_activation='gelu_pytorch_tanh'`, `rms_norm_eps=1e-6`, `attention_bias=False`) |
| PLE / KV sharing | None on 31B (PLE is E2B/E4B only; `num_kv_shared_layers=0`) | HF Gemma4 docs (PLE section); solidsf table |
| Tokenizer | SentencePiece, vocab 262,144, BOS 2 / EOS 1 / PAD 0, `add_dummy_prefix=False` | Gemma4TextConfig; Gemma family tokenizer docs (SPM; Gemma sets dummy prefix False unlike Llama-2) |
| BF16 GGUF (native, real-run file) | `bartowski/google_gemma-4-31B-it-GGUF`, `gemma-4-31B-it-bf16.gguf`, 61.41 GB, **split (sharded)** | bartowski repo file table |
| Unsloth GGUF | `unsloth/gemma-4-31B-it-GGUF` confirmed for UD-Q4_K_XL / Q8_0; a BF16 file there was NOT confirmed | avenchat llama.cpp guide (shows Q8_0 / UD-Q4_K_XL only) |

## Correction to the parent brief

The brief's "32 Q / 16 KV heads, head_dim 256" is true ONLY for
sliding layers. Global layers are 32 Q / 4 KV, head_dim 512, V = K.
Using sliding dims on global layers would read wrong tensor shapes.

## Rejected claim

One secondary source (borowiecki) lists "head_dim 168 (sliding) /
256 (global)". Rejected: 168 = 5376 / 32 (hidden per Q head), not a
head dim. The convergent values 256 / 512 come from config field
names (`head_dim`, `global_head_dim`) in three independent sources.

## NOT FOUND — loader-driven, never defaulted or guessed

- `intermediate_size` (FFN width): no public source in this pass
  (the 10,752 figure in search results is a different draft model).
  Loader reads `gemma4.feed_forward_length` from the file.
- `final_logit_softcapping` / `attn_logit_softcapping` for 31B:
  not found (30.0 belongs to a draft model). Both default to None
  and apply only when the GGUF metadata key is present.
- QK-norm on the text decoder: no evidence (QK-norms reported for
  the vision encoder only). Not implemented on the text path.

## Implemented as explicit knobs (parity items, not hidden guesses)

- **Embedding scale**: defaults to sqrt(hidden) (Gemma-family
  convention, NOT separately confirmed for Gemma-4 31B). Override
  via `gemma4.embedding_scale` metadata or the `embed_scale`
  constructor argument. A wrong scale moves every logit, so parity
  catches it immediately.
- **RoPE form**: half-rotation (HF `rotate_half`), same as
  PagedQwen2. The llama.cpp form for Gemma was not confirmed.
- **Global RoPE** (CORRECTED 2026-10-26): global layers rotate the
  FULL 512-dim head as NEOX pairs `(i, i+256)`, angle = position ×
  theta^(-2i/512) / `rope_freqs[i]`, with `rope_freqs.weight` read
  from the file (fail loud if absent). The old partial-128-dim
  knobs (`rope_partial_mode`, `partial_rotary_factor`) were removed;
  they encoded a transformers-docs concept, not the GGUF runtime's
  behavior, and no knob choice could reconcile them with parity.
- **BOS in generate()**: none prepended (same as the Qwen2 path).
  Whether the real run needs BOS is a parity item.
- **GGUF metadata key names**: the local llama.cpp checkout
  predates gemma4, so keys follow the `gemma4.<field>` convention
  with HF-field aliases for global attention. The loader fails
  loud naming any missing architecture-defining key. Real-file
  load confirms or corrects the key map.
- **Sliding-window edge**: a query attends to the current token
  plus the previous `window - 1` (llama.cpp convention). Cache
  trimming happens AFTER attention within a call (trimming before
  broke early prefill positions with all-`-inf` rows — fixed and
  tested).

## Loader key map (GGUF metadata)

Required (fail loud if absent): `general.architecture == "gemma4"`,
`gemma4.block_count`, `gemma4.embedding_length`,
`gemma4.feed_forward_length`, `gemma4.attention.head_count`,
`gemma4.attention.head_count_kv`,
`gemma4.attention.layer_norm_rms_epsilon`, plus one global-KV key
(`gemma4.attention.head_count_kv_global` preferred, else
`...global_head_count_kv`, `...num_global_key_value_heads`).

Optional with defaults: `gemma4.attention.sliding_window` (1024),
`gemma4.attention.head_dim` (256), `gemma4.rope.freq_base`
(10,000), `gemma4.attention.global_head_dim` (512),
`gemma4.rope.freq_base_global` (1,000,000),
`gemma4.layer_types` (else the 5:1 rule, last forced global),
`gemma4.embedding_scale`, `gemma4.final_logit_softcapping`,
`gemma4.attn_logit_softcapping`.

Tensor names (llama.cpp standard names): `token_embd.weight`,
`output.weight` (absent = tied), `output_norm.weight`,
`blk.{i}.attn_norm.weight`, `attn_q/k/v.weight` (+ optional
`.bias`), `attn_output.weight`, `ffn_norm.weight`,
`ffn_gate/up/down.weight`. Global layers omit `attn_v.weight`
(V = K); a sliding layer missing it fails loud.

## Real-weight parity checklist (update 2026-10-26: items 1–4 DONE — parity v5 passed 8/8; item 6 partially remains)

1. Download the bartowski BF16 GGUF (61.41 GB, split — shard
   handling at convert time is a real-run logistics item; the
   `.lac` converter currently takes a single file).
2. Confirm the GGUF key map above against the real header; correct
   any key name the file actually uses.
3. Confirm `feed_forward_length` from the header (the not-found
   value) — no code change needed, it is loader-driven.
4. Token parity vs `llama-cli --temp 0 --top-k 1` on the dense-proof
   prompts, CPU paged path, short prompts first.
5. DONE 2026-10-26 — token parity vs the reference passed 8/8
   (parity v5, BOS leg, BF16 real weights) after two fixes: the five
   omitted block terms (layer-0 blowup) and full-dim global RoPE
   with stored `rope_freqs` (step-2 near-tie). The old knob-flip
   order is superseded — do not flip knobs against a passing leg.
6. Then: GPU Gemma port (GPUModel is Qwen2-dense only), 128-token
   reuse run, thrash-regime speed at 256/512 MB, speculative on
   Gemma, and the 32-token / 300 s headline gate.
