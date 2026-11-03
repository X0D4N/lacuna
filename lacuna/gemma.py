"""Paged Gemma-4 31B executor: real generation with weights via the pager.

Text-only path for ``google/gemma-4-31b-it`` (Apache 2.0, public,
30.7B dense, April 2026). The ~550M vision encoder is out of scope:
vision tensors, if present in a file, are simply never read — this
class executes the text decoder only. Every weight byte arrives
through :class:`~lacuna.pager.PagedWeightStore`, exactly like
:class:`~lacuna.paged_model.PagedQwen2` (sliced embed/head, bounded
dequantized cache, ``store.stats`` as the audit trail). Greedy
decoding only, deterministic, comparable to
``llama-cli --temp 0 --top-k 1`` once real-weight parity is run.

CONFIRMED FACTS (public sources, 2026-10-25; see docs/GEMMA4_31B.md
for the full table and the not-found list):

- 60 layers, hidden 5376, vocab 262,144, tied embeddings, 262,144
  max positions, RMSNorm eps 1e-6, GELU-tanh gated MLP.
- Hybrid attention, 5 sliding : 1 full repeated 10 times, so the
  global layers are 0-based 5, 11, 17, 23, 29, 35, 41, 47, 53, 59 and
  the last layer is global. The layer pattern is read from GGUF
  metadata when a ``layer_types`` array is present, else derived
  from the 5:1 rule with the last layer forced global.
- Sliding layers: window 1024, 32 Q heads, 16 KV heads, head_dim 256,
  standard RoPE, theta 10,000, full rotary factor.
- Global layers: 32 Q heads, 4 KV heads, head_dim 512, V = K
  (``attention_k_eq_v``; no separate v projection), proportional
  RoPE, theta 1,000,000, full-dim NEOX rotation with the file's
  stored rope_freqs factors (CORRECTED 2026-10-26: the "partial
  rotary factor 0.25" in earlier notes came from transformers
  docs, not the GGUF runtime — the file carries no such key and
  llama.cpp has no such concept).
- Tokenizer: SentencePiece, vocab 262,144, BOS 2 / EOS 1 / PAD 0,
  ``add_dummy_prefix == False`` (see :mod:`lacuna.spm`).
- BF16 GGUF (native precision, the real-run file):
  ``bartowski/google_gemma-4-31B-it-GGUF``,
  ``gemma-4-31B-it-bf16.gguf``, 61.41 GB, split (sharded). Unsloth's
  ``unsloth/gemma-4-31B-it-GGUF`` is confirmed for quants
  (UD-Q4_K_XL, Q8_0); a BF16 file there was NOT confirmed.
- 31B has NO per-layer embeddings (PLE is E2B/E4B only) and NO
  KV-shared layers (``num_kv_shared_layers == 0``).

CORRECTION TO THE PARENT BRIEF: the brief's "32 Q / 16 KV heads,
head_dim 256" is true ONLY for sliding layers. Global layers are
32 Q / 4 KV, head_dim 512, V = K. An executor that used the sliding
dims on global layers would read the wrong tensor shapes and produce
garbage, not near-tie noise.

NOT CONFIRMED — implemented as explicit, overridable knobs, never as
silent guesses (each is a real-weight parity check item):

- ``intermediate_size`` (FFN width): not found in any public source
  in this pass (the 10,752 figure in search results belongs to a
  different draft model). The loader reads
  ``gemma4.feed_forward_length`` from the file; it is never defaulted.
- Embedding scale: Gemma-family convention scales token embeddings
  by sqrt(hidden). That scaling is NOT separately confirmed for the
  Gemma-4 31B text decoder, so it defaults ON (family-consistent)
  and is overridable via ``gemma4.embedding_scale`` metadata or the
  ``embed_scale`` constructor argument. Parity will catch a wrong
  scale immediately (every logit moves).
- RoPE form: half-rotation (HF ``rotate_half``), same as PagedQwen2.
  The llama.cpp form for Gemma was not confirmed; parity decides.
- Global RoPE: CONFIRMED 2026-10-26 from llama.cpp src/models/
  gemma4.cpp + ggml/src/ggml-cpu/ops.cpp (and a ctypes run of the
  compiled kernel against this file's _rope): NEOX half-rotation
  over the FULL global head dim (512), angle = position *
  theta**(-2i/512) / rope_freqs[i], where rope_freqs is the file's
  stored factor tensor (256 fp32 entries). There is NO partial
  rotary in the GGUF runtime — llama.cpp's source contains no
  partial_rotary_factor anywhere, and the old partial-128-dim
  rotation differs from the reference kernel by O(1) on generic
  vectors for ANY factor choice (the pairings themselves differ:
  ggml pairs (i, i+256), partial pairs (i, i+64)). The loader
  reads rope_freqs.weight and fails loud when a model with global
  layers lacks it.
- Softcapping: ``final_logit_softcapping`` / ``attn_logit_softcapping``
  for 31B were not found (the 30.0 figure belongs to a draft model).
  Both default to None (no softcapping) and are applied only when
  the corresponding GGUF metadata key is present.
- QK-norm on the text decoder: CONFIRMED 2026-10-26 from the real
  file (per-layer attn_q_norm / attn_k_norm, exact constants, e.g.
  blk.0 1.0234375 / 0.12207031, product exactly 1/8 every layer
  measured). Implemented as per-head RMSNorm before RoPE, with
  attention scale 1.0 (not 1/sqrt(head_dim)).
- Peri-LN post-norms + layer_output_scale: CONFIRMED 2026-10-26
  from the real file (per-layer post_attention_norm,
  post_ffw_norm, layer_output_scale, e.g. blk.0 scale 0.08935547)
  and from llama.cpp src/models/gemma4.cpp: each branch output is
  post-normed before its residual add, and layer_output_scale
  multiplies the WHOLE layer output (residual included) at the
  end of the layer — not each branch. CORRECTED 2026-10-26: the
  first build omitted all three plus QK-norm and V-norm, which
  blew layer 0 up x72 (out RMS 92.3 vs input 1.28) and saturated
  the final softcap; the omission is now a fail-loud missing
  tensor, never silent.
- V-norm: llama.cpp gemma4.cpp weightless per-head RMSNorms V on
  every layer (explicit v projection, or the raw k projection on
  V=K global layers). Implemented with a ones weight.
- BOS handling in generate(): no BOS is prepended (same as the Qwen2
  path). Whether the real run needs BOS is a parity item.
- GGUF metadata key names for gemma4: the local llama.cpp checkout
  predates gemma4, so key names follow the llama.cpp naming
  convention (``gemma4.<field>``) with HF-field aliases for the
  global-attention fields. The loader fails loud listing any missing
  architecture-defining key instead of defaulting it; real-file load
  will confirm or correct the key map.

Design notes:
- Sliding-window KV cache: each sliding layer keeps only its window
  (cache rows are dropped from the front, with ``start_pos`` tracking
  the absolute position of the first cached row, so RoPE positions
  and the mask stay correct). Global layers keep everything.
- Attention is a per-layer method (``_attention``) so the
  sliding-window mask and the global mask are unit-testable against
  hand-computed references.
- No QKV fusion in v1 (Qwen2 fuses; Gemma v1 uses separate GEMMs for
  clarity across the two layer types). Known inefficiency, not a bug.
"""

from __future__ import annotations

import numpy as np

from .convert import convert as _convert
from .gguf import parse_gguf, read_tensor, read_tensor_rows
from .pager import PagedWeightStore
from .spm import SPMTokenizer

# GGUF layer-type spellings (HF config layer_types values).
_SLIDING = "sliding_attention"
_FULL = "full_attention"


class GemmaModelError(ValueError):
    """Raised for unsupported architectures or missing tensors. Fails loud."""


def _layer_types_from_metadata(md: dict, p: str, n_layer: int) -> list[str]:
    """Layer types: metadata array wins; else the 5:1 rule, last forced
    global (matches every public source for the 31B pattern)."""
    for key in (p + "layer_types", p + "attention.layer_types"):
        raw = md.get(key)
        if raw is not None:
            types = [str(t) for t in raw]
            if len(types) != n_layer:
                raise GemmaModelError(
                    f"layer_types length {len(types)} != block_count {n_layer}"
                )
            for t in types:
                if t not in (_SLIDING, _FULL):
                    raise GemmaModelError(f"unknown layer type {t!r}")
            return types
    types = [_FULL if (li % 6 == 5) else _SLIDING for li in range(n_layer)]
    if types and types[-1] != _FULL:
        types[-1] = _FULL  # last layer is always global
    return types


_GLOBAL_KV_KEYS = (
    "attention.head_count_kv_global",
    "attention.global_head_count_kv",
    "attention.num_global_key_value_heads",
    "num_global_key_value_heads",
)


def resolve_gemma4_kv_geometry(md: dict, p: str, n_layer: int,
                               layer_types: list, by_name: dict
                               ) -> tuple[int, int, int, int]:
    """Resolve (n_kv_sliding, n_kv_global, head_dim, head_dim_global).

    The real Gemma-4 file (bartowski BF16, measured 2026) carries
    ``attention.head_count_kv`` as a PER-LAYER array (60 entries: 16 on
    sliding layers, 4 on global layers) and NO global-KV scalar key, so
    a loader that only accepts a scalar global key fails on the real
    file. Resolution order, all fail-loud, never defaulted silently:

    1. Per-layer array: sliding value = the uniform value at sliding
       positions, global value = the uniform value at global positions.
       Non-uniform values within one layer type, or a length other than
       ``n_layer``, raise.
    2. Scalar + a global-KV metadata key (the four spellings above).
    3. Scalar + shape fallback: head_dim_global = q rows / n_head and
       n_kv_global = k rows / head_dim_global from a present global
       layer, required consistent across all present global layers.
       No present global layer shapes -> raise naming the missing key.

    Finally every present layer's q/k row counts are cross-checked
    against the resolved geometry (GGUF dims are [cols, rows], so rows
    are dims[1]). A present layer that disagrees raises instead of
    running with wrong attention geometry.
    """
    n_head = int(md[p + "attention.head_count"])
    head_dim = int(md.get(p + "attention.head_dim",
                          md.get(p + "head_dim", 256)))
    head_dim_global = int(md.get(p + "attention.global_head_dim",
                                 md.get(p + "global_head_dim", 512)))
    raw_kv = md.get(p + "attention.head_count_kv")
    if isinstance(raw_kv, (list, tuple)):
        vals = [int(v) for v in raw_kv]
        if len(vals) != n_layer:
            raise GemmaModelError(
                f"per-layer head_count_kv length {len(vals)} != "
                f"block_count {n_layer}"
            )
        sliding = {vals[li] for li, t in enumerate(layer_types)
                   if t == _SLIDING}
        glob = {vals[li] for li, t in enumerate(layer_types)
                if t == _FULL}
        if len(sliding) != 1 or len(glob) != 1:
            raise GemmaModelError(
                f"per-layer head_count_kv not uniform per layer type: "
                f"sliding={sorted(sliding)} global={sorted(glob)}"
            )
        n_kv = sliding.pop()
        n_kv_global = glob.pop()
    else:
        n_kv = int(raw_kv)
        n_kv_global = None
        for key in _GLOBAL_KV_KEYS:
            if p + key in md:
                n_kv_global = int(md[p + key])
                break
        if n_kv_global is None:
            q_rows, k_rows = [], []
            for li, t in enumerate(layer_types):
                if t != _FULL:
                    continue
                q = by_name.get(f"blk.{li}.attn_q.weight")
                k = by_name.get(f"blk.{li}.attn_k.weight")
                if q is None or k is None:
                    continue
                q_rows.append(int(q["dims"][1]))
                k_rows.append(int(k["dims"][1]))
            if not q_rows:
                raise GemmaModelError(
                    "global KV head count missing: none of "
                    + " / ".join(p + k for k in _GLOBAL_KV_KEYS)
                    + " present and no global-layer q/k shapes to derive "
                    "it from"
                )
            if any(r % n_head for r in q_rows):
                raise GemmaModelError(
                    f"global q rows {sorted(set(q_rows))} not divisible "
                    f"by head count {n_head}"
                )
            hd = {r // n_head for r in q_rows}
            if len(hd) != 1:
                raise GemmaModelError(
                    f"global q rows imply inconsistent head dims "
                    f"{sorted(hd)}"
                )
            head_dim_global = hd.pop()
            if any(r % head_dim_global for r in k_rows):
                raise GemmaModelError(
                    f"global k rows {sorted(set(k_rows))} not divisible "
                    f"by head dim {head_dim_global}"
                )
            nkv = {r // head_dim_global for r in k_rows}
            if len(nkv) != 1:
                raise GemmaModelError(
                    f"global k rows imply inconsistent KV counts "
                    f"{sorted(nkv)}"
                )
            n_kv_global = nkv.pop()
    for li, t in enumerate(layer_types):
        q = by_name.get(f"blk.{li}.attn_q.weight")
        k = by_name.get(f"blk.{li}.attn_k.weight")
        if q is None or k is None:
            continue
        hd = head_dim_global if t == _FULL else head_dim
        nkv = n_kv_global if t == _FULL else n_kv
        if int(q["dims"][1]) != n_head * hd:
            raise GemmaModelError(
                f"layer {li} q rows {q['dims'][1]} != head count "
                f"{n_head} x head dim {hd}"
            )
        if int(k["dims"][1]) != nkv * hd:
            raise GemmaModelError(
                f"layer {li} k rows {k['dims'][1]} != KV count "
                f"{nkv} x head dim {hd}"
            )
    return n_kv, n_kv_global, head_dim, head_dim_global


class PagedGemma4:
    """Gemma-4 text decoder whose weights live in a PagedWeightStore."""

    def __init__(self, gguf_path: str, lac_path: str | None = None,
                 hot_budget_bytes: int = 256 * 1024 * 1024,
                 dequant_budget_bytes: int | None = None,
                 enable_prefetch: bool = True,
                 embed_scale: float | None = None):
        md, infos, _ = parse_gguf(gguf_path)
        if md.get("general.architecture") != "gemma4":
            raise GemmaModelError(
                f"paged executor supports gemma4 only, got "
                f"{md.get('general.architecture')!r}"
            )
        p = "gemma4."
        # Architecture-defining keys: fail loud, never defaulted.
        for key in ("block_count", "embedding_length", "feed_forward_length",
                    "attention.head_count", "attention.head_count_kv",
                    "attention.layer_norm_rms_epsilon"):
            if p + key not in md:
                raise GemmaModelError(f"required GGUF key {p + key} missing")
        self.n_layer = int(md[p + "block_count"])
        self.hidden = int(md[p + "embedding_length"])
        self.ffn = int(md[p + "feed_forward_length"])
        self.n_head = int(md[p + "attention.head_count"])
        self.eps = float(md[p + "attention.layer_norm_rms_epsilon"])
        # Sliding-window config (window + RoPE theta). The real 31B
        # file carries BOTH rope bases: freq_base (1e6, global) and
        # freq_base_swa (sliding) — the _swa key wins for sliding
        # layers when present, else the plain base (older files).
        self.window = int(md.get(p + "attention.sliding_window",
                                 md.get(p + "sliding_window", 1024)))
        self.rope_theta = float(md.get(
            p + "rope.freq_base_swa",
            md.get(p + "rope.freq_base",
                   md.get(p + "rope.theta", 10000.0))))
        # KV geometry + head dims: per-layer array, metadata scalar, or
        # shape fallback (see resolve_gemma4_kv_geometry). by_name is
        # built here because the shape path needs it.
        self.layer_types = _layer_types_from_metadata(md, p, self.n_layer)
        self.by_name = {i["name"]: i for i in infos}
        (self.n_kv, self.n_kv_global, self.head_dim,
         self.head_dim_global) = resolve_gemma4_kv_geometry(
             md, p, self.n_layer, self.layer_types, self.by_name)
        # Global RoPE base: the global-specific key wins, else the
        # plain base (the real file's 1e6), else the 1e6 default.
        self.rope_theta_global = float(md.get(
            p + "rope.freq_base_global",
            md.get(p + "rope.theta_global",
                   md.get(p + "rope.freq_base", 1000000.0))))
        # No partial-rotary knobs: global RoPE is fully determined
        # by the file (full head dim + stored rope_freqs factors,
        # see module docstring). Any rope.partial_rotary_factor
        # metadata key is ignored — the GGUF runtime never reads it.
        # Softcapping: metadata-driven only, default None. The real
        # 31B file carries final_logit_softcapping 30.0 (measured
        # 2026); attn_logit_softcapping is absent there.
        self.final_logit_softcapping = md.get(p + "final_logit_softcapping")
        if self.final_logit_softcapping is not None:
            self.final_logit_softcapping = float(self.final_logit_softcapping)
        self.attn_logit_softcapping = md.get(p + "attn_logit_softcapping")
        if self.attn_logit_softcapping is not None:
            self.attn_logit_softcapping = float(self.attn_logit_softcapping)
        # Embedding scale: Gemma-family sqrt(hidden) default, overridable.
        if embed_scale is not None:
            self.embed_scale = float(embed_scale)
        elif p + "embedding_scale" in md:
            self.embed_scale = float(md[p + "embedding_scale"])
        else:
            self.embed_scale = float(np.sqrt(self.hidden))
        self.tok = SPMTokenizer.from_metadata(md)
        self.eos = md.get("tokenizer.ggml.eos_token_id")
        if "token_embd.weight" not in self.by_name:
            raise GemmaModelError("required tensor token_embd.weight missing")
        if "output_norm.weight" not in self.by_name:
            raise GemmaModelError("required tensor output_norm.weight missing")
        # Tied embeddings (Gemma4TextConfig tie_word_embeddings=True):
        # a file without output.weight uses the embedding rows as head.
        self.tied = "output.weight" not in self.by_name

        lac = lac_path or (gguf_path + ".lac")
        import os as _os

        if not _os.path.exists(lac):
            _convert(gguf_path, lac, quiet=True)
        self.store = PagedWeightStore(lac, hot_budget_bytes=int(hot_budget_bytes))
        self._embed_info = self.by_name["token_embd.weight"]
        self._head_info = (
            self.by_name["output.weight"] if not self.tied
            else self._embed_info
        )
        self.out_norm = read_tensor(self.store, self.by_name["output_norm.weight"])
        # Global RoPE factors (ggml-style, see module docstring):
        # the file's stored rope_freqs.weight, required whenever the
        # model has global layers (llama.cpp requires it for the
        # first global layer and fails load without it). Sliding-only
        # models never touch it, so they are not required to carry it.
        self.rope_freqs = None
        if any(t == _FULL for t in self.layer_types):
            if "rope_freqs.weight" not in self.by_name:
                raise GemmaModelError(
                    "required tensor rope_freqs.weight missing "
                    "(global RoPE needs the file's stored factors)")
            arr = np.asarray(
                read_tensor(self.store, self.by_name["rope_freqs.weight"]),
                dtype=np.float32).reshape(-1)
            if arr.shape[0] != self.head_dim_global // 2:
                raise GemmaModelError(
                    f"rope_freqs.weight has {arr.shape[0]} entries, "
                    f"expected {self.head_dim_global // 2}")
            if not np.all(np.isfinite(arr)) or np.any(arr == 0):
                raise GemmaModelError(
                    "rope_freqs.weight must be finite and nonzero "
                    "(global RoPE divides by it)")
            self.rope_freqs = arr
        # Cache: layer -> (K, V, start_pos); start_pos is the absolute
        # position of the first cached row (sliding layers trim).
        self.cache: dict[int, tuple[np.ndarray, np.ndarray, int]] = {}
        from collections import OrderedDict

        self._dq: OrderedDict[str, np.ndarray] = OrderedDict()
        self._dq_bytes = 0
        self._dq_budget = (
            int(dequant_budget_bytes)
            if dequant_budget_bytes is not None
            else int(hot_budget_bytes)
        )
        self._enable_prefetch = bool(enable_prefetch)
        # RoPE frequencies. Sliding: full head dim, theta 10,000.
        self._inv_freq = self.rope_theta ** (
            -np.arange(0, self.head_dim, 2, dtype=np.float32) / self.head_dim
        )
        # Global: full head dim, ggml-style (see module docstring).
        # The stored rope_freqs factors divide the angles in _rope;
        # they are kept separate from inv_freq so the formula stays
        # legible against ggml_rope_cache_init.
        self._inv_freq_global = self.rope_theta_global ** (
            -np.arange(0, self.head_dim_global, 2, dtype=np.float32)
            / self.head_dim_global
        )

    # -- layer geometry ------------------------------------------------
    def _layer_dims(self, li: int) -> tuple[int, int, int]:
        """(n_q, n_kv, head_dim) for layer li. Q head count is shared;
        KV count and head dim differ between sliding and global."""
        if self.layer_types[li] == _FULL:
            return self.n_head, self.n_kv_global, self.head_dim_global
        return self.n_head, self.n_kv, self.head_dim

    def _is_global(self, li: int) -> bool:
        return self.layer_types[li] == _FULL

    # -- weight access (pager only) ------------------------------------
    def _dq_put(self, name: str, arr: np.ndarray) -> None:
        nbytes = int(arr.nbytes)
        if nbytes > self._dq_budget:
            return
        while self._dq_bytes + nbytes > self._dq_budget and self._dq:
            _, old = self._dq.popitem(last=False)
            self._dq_bytes -= int(old.nbytes)
        self._dq[name] = arr
        self._dq_bytes += nbytes

    def _dq_drop(self, name: str) -> None:
        arr = self._dq.pop(name, None)
        if arr is not None:
            self._dq_bytes -= int(arr.nbytes)

    def _w(self, name: str) -> np.ndarray:
        hit = self._dq.get(name)
        if hit is not None:
            self._dq.move_to_end(name)
            return hit
        info = self.by_name.get(name)
        if info is None:
            raise GemmaModelError(f"required tensor {name} missing")
        arr = read_tensor(self.store, info)
        self._dq_put(name, arr)
        return arr

    def _w_optional(self, name: str) -> np.ndarray | None:
        if name not in self.by_name:
            return None
        return self._w(name)

    def _head_block(self, b0: int, b1: int) -> np.ndarray:
        """One dequantized lm_head row block (or embedding rows when
        tied), LRU-cached in ``_dq``. The head is never held whole."""
        key = f"output.weight#rows:{b0}:{b1}"
        hit = self._dq.get(key)
        if hit is not None:
            self._dq.move_to_end(key)
            return hit
        w = read_tensor_rows(self.store, self._head_info, b0, b1)
        self._dq_put(key, w)
        return w

    def _embed_rows(self, ids: list[int]) -> np.ndarray:
        """Gather embedding rows for ``ids`` (sliced, never held whole),
        scaled by the embedding scale (Gemma-family sqrt(hidden)
        default; see module docstring for the parity caveat)."""
        if not ids:
            return np.zeros((0, self.hidden), dtype=np.float32)
        uniq = sorted(set(int(i) for i in ids))
        rows: dict[int, np.ndarray] = {}
        missing: list[int] = []
        for u in uniq:
            key = f"token_embd.weight#row:{u}"
            hit = self._dq.get(key)
            if hit is not None:
                self._dq.move_to_end(key)
                rows[u] = hit
            else:
                missing.append(u)
        if missing:
            runs: list[tuple[int, int]] = []
            start = last = missing[0]
            for u in missing[1:]:
                if u == last + 1:
                    last = u
                else:
                    runs.append((start, last + 1))
                    start = last = u
            runs.append((start, last + 1))
            for a, b in runs:
                arr = read_tensor_rows(self.store, self._embed_info, a, b)
                for k, u in enumerate(range(a, b)):
                    row = np.array(arr[k], dtype=np.float32, copy=True)
                    self._dq_put(f"token_embd.weight#row:{u}", row)
                    rows[u] = row
        out = np.stack([rows[i] for i in ids]).astype(np.float32)
        return out * np.float32(self.embed_scale)

    # -- primitives ------------------------------------------------------
    def _rmsnorm(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        ms = np.mean(x * x, axis=-1, keepdims=True)
        return (x / np.sqrt(ms + np.float32(self.eps))) * w

    @staticmethod
    def _gelu_tanh(x: np.ndarray) -> np.ndarray:
        """GELU with the tanh approximation (Gemma MLP activation)."""
        return np.float32(0.5) * x * (
            np.float32(1.0)
            + np.tanh(
                np.float32(0.7978845608028654)
                * (x + np.float32(0.044715) * x * x * x)
            )
        )

    def _rope(self, t: np.ndarray, positions: np.ndarray,
              is_global: bool) -> np.ndarray:
        """RoPE, half-rotation form (HF ``rotate_half``). Sliding layers
        rotate the full head dim. Global layers rotate the FULL head
        dim too, ggml-style: angle = position * inv_freq / rope_freqs
        (the file's stored factors; see module docstring). There is
        no partial rotary and no passthrough on global layers."""
        if is_global:
            rotary = self.head_dim_global
            inv_freq = self._inv_freq_global
            if self.rope_freqs is None:
                raise GemmaModelError(
                    "global RoPE needs rope_freqs.weight (file tensor)")
            angles = (positions[:, None].astype(np.float32)
                      * inv_freq[None, :]
                      / self.rope_freqs[None, :].astype(np.float32))
        else:
            rotary = t.shape[-1]
            inv_freq = self._inv_freq
            angles = positions[:, None].astype(np.float32) * inv_freq[None, :]
        cos = np.cos(angles)[:, None, :]
        sin = np.sin(angles)[:, None, :]
        h = rotary // 2
        x0 = t[..., :h]
        x1 = t[..., h:rotary]
        out = np.empty_like(t)
        out[..., :h] = x0 * cos - x1 * sin
        out[..., h:rotary] = x0 * sin + x1 * cos
        if rotary < t.shape[-1]:
            out[..., rotary:] = t[..., rotary:]
        return out

    @staticmethod
    def _attn_mask(row_pos: np.ndarray, col_pos: np.ndarray,
                   is_sliding: bool, window: int) -> np.ndarray:
        """Boolean allowed-mask (rows = queries, cols = keys). Causal
        always; sliding layers additionally allow only the last
        ``window`` positions (current token included)."""
        causal = col_pos[None, :] <= row_pos[:, None]
        if not is_sliding:
            return causal
        return causal & ((row_pos[:, None] - col_pos[None, :]) < window)

    def _attention(self, q: np.ndarray, k_all: np.ndarray,
                   v_all: np.ndarray, row_pos: np.ndarray,
                   col_pos: np.ndarray, is_sliding: bool) -> np.ndarray:
        """Dot-product attention over cached K/V. ``q`` is (T, n_q, d);
        ``k_all``/``v_all`` are (S, n_q, d) after GQA repeat. Returns
        (T, n_q, d). Unit-testable against a hand-computed loop
        reference (see tests/test_gemma.py).

        Gemma-4 attention scale is 1.0, NOT 1/sqrt(head_dim): QK-norm
        provides the stability (Keras Gemma4Backbone; the real 31B
        file's per-layer q_norm x k_norm product is exactly 1/8,
        which only makes sense with scale 1.0)."""
        scores = np.einsum("thd,shd->ths", q, k_all)
        allowed = self._attn_mask(row_pos, col_pos, is_sliding, self.window)
        scores = np.where(allowed[:, None, :], scores, np.float32("-inf"))
        if self.attn_logit_softcapping is not None:
            cap = np.float32(self.attn_logit_softcapping)
            scores = cap * np.tanh(scores / cap)
        scores = scores - scores.max(axis=-1, keepdims=True)
        probs = np.exp(scores)
        probs = probs / probs.sum(axis=-1, keepdims=True)
        return np.einsum("ths,shd->thd", probs, v_all)

    # -- prefetch ---------------------------------------------------------
    _LAYER_TENSORS = (
        "attn_norm.weight",
        "attn_q.weight", "attn_k.weight", "attn_v.weight",
        "attn_q_norm.weight", "attn_k_norm.weight",
        "attn_output.weight", "post_attention_norm.weight",
        "layer_output_scale.weight",
        "ffn_norm.weight",
        "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
        "post_ffw_norm.weight",
    )

    def _layer_page_ids(self, li: int) -> list[int]:
        base = f"blk.{li}."
        pages: list[int] = []
        for suffix in self._LAYER_TENSORS:
            if base + suffix in self._dq:
                continue
            tid = self.store.tensor_id(base + suffix)
            if tid is None:
                continue  # e.g. attn_v on a V=K global layer
            pages.extend(e["page_id"] for e in self.store.tensor_pages(tid))
        return pages

    # -- forward --------------------------------------------------------
    def forward(self, ids: list[int], pos_offset: int = 0) -> np.ndarray:
        """Run ``ids`` through the model; return last-position logits (f32)."""
        ids = [int(i) for i in ids]
        n_tok = len(ids)
        positions = np.arange(pos_offset, pos_offset + n_tok)
        x = self._embed_rows(ids)
        for li in range(self.n_layer):
            base = f"blk.{li}."
            if self._enable_prefetch and li + 1 < self.n_layer:
                self.store.prefetch(self._layer_page_ids(li + 1))
            is_global = self._is_global(li)
            n_q, n_kv, d = self._layer_dims(li)
            group = n_q // n_kv
            if n_q % n_kv:
                raise GemmaModelError(
                    f"layer {li}: Q heads {n_q} not divisible by KV heads {n_kv}"
                )
            h = self._rmsnorm(x, self._w(base + "attn_norm.weight"))
            q = h @ self._w(base + "attn_q.weight").T
            k = h @ self._w(base + "attn_k.weight").T
            # Optional per-layer biases (Gemma4TextConfig says
            # attention_bias=False; if a file carries them, use them).
            qb = self._w_optional(base + "attn_q.bias")
            kb = self._w_optional(base + "attn_k.bias")
            if qb is not None:
                q = q + qb
            if kb is not None:
                k = k + kb
            v_w = self._w_optional(base + "attn_v.weight")
            if v_w is None:
                if not is_global:
                    raise GemmaModelError(
                        f"layer {li}: sliding layer missing attn_v.weight"
                    )
                # Global V = K (attention_k_eq_v): V starts as the
                # RAW k projection (llama.cpp gemma4.cpp assigns
                # Vcur = Kcur before any norm is applied to either).
                v_raw = k
            else:
                v_raw = h @ v_w.T
                vb = self._w_optional(base + "attn_v.bias")
                if vb is not None:
                    v_raw = v_raw + vb
            # QK-norm (Gemma-4 QKNorm, llama.cpp gemma4.cpp):
            # per-head RMSNorm on Q and K with the file's
            # attn_q_norm / attn_k_norm weights, applied BEFORE RoPE
            # (RoPE preserves per-head RMS, so the order is
            # exactness-neutral, but norm-then-RoPE is the canonical
            # Gemma order). Required, never optional: _w() fails
            # loud on a file that lacks these tensors.
            q = self._rmsnorm(
                q.reshape(n_tok, n_q, d),
                self._w(base + "attn_q_norm.weight"))
            k = self._rmsnorm(
                k.reshape(n_tok, n_kv, d),
                self._w(base + "attn_k_norm.weight"))
            # V-norm (llama.cpp gemma4.cpp): V is ALWAYS weightless
            # per-head RMSNormed — the explicit v projection on
            # sliding layers, and the raw k projection on global
            # layers. It is NOT the k_norm-weighted K (that weight
            # belongs to the scores branch only). V is never RoPE'd.
            v = self._rmsnorm(
                v_raw.reshape(n_tok, n_kv, d),
                np.ones(d, dtype=np.float32))
            q = self._rope(q, positions, is_global)
            k = self._rope(k, positions, is_global)
            k = np.repeat(k, group, axis=1)
            v = np.repeat(v, group, axis=1)
            if li in self.cache:
                kc, vc, start = self.cache[li]
                k_all = np.concatenate([kc, k], axis=0)
                v_all = np.concatenate([vc, v], axis=0)
            else:
                k_all, v_all = k, v
                start = int(pos_offset)
            # Attention sees the FULL cache: the sliding-window mask
            # already restricts each query to its window, and trimming
            # before attention would leave early prefill positions with
            # no allowed keys at all (all -inf rows). Trim only after
            # attention, so future steps stay bounded.
            col_pos = start + np.arange(k_all.shape[0])
            att = self._attention(q, k_all, v_all, positions, col_pos,
                                  not is_global)
            if not is_global and k_all.shape[0] > self.window:
                drop = k_all.shape[0] - self.window
                k_all = k_all[drop:]
                v_all = v_all[drop:]
                start += drop
            self.cache[li] = (k_all, v_all, start)
            # Peri-LN (Gemma-4 pre-norm AND post-norm, tech report
            # sec. 2) + layer_output_scale, exactly as llama.cpp
            # gemma4.cpp builds it: attn_out = inpL +
            # post_attention_norm(attn); the FFN pre-norm reads
            # attn_out; layer output = (attn_out +
            # post_ffw_norm(mlp)) * layer_output_scale — the scale
            # multiplies the WHOLE layer output (residual included),
            # NOT each branch. Omitting the post-norms and the scale
            # was the 2026-10-26 layer-0 blowup: pre-norm weights
            # are large (real blk.0 attn_norm RMS 8.15, blk.1 19.3)
            # and the MLP branch grows quadratically with the
            # pre-norm scale, so un-tamed branches gave layer-0 out
            # RMS 92 vs input 1.28 and ~320k by layer 11.
            att_out = att.reshape(n_tok, n_q * d) @ self._w(
                base + "attn_output.weight"
            ).T
            att_out = self._rmsnorm(
                att_out, self._w(base + "post_attention_norm.weight"))
            x1 = x + att_out
            h2 = self._rmsnorm(x1, self._w(base + "ffn_norm.weight"))
            gate = h2 @ self._w(base + "ffn_gate.weight").T
            up = h2 @ self._w(base + "ffn_up.weight").T
            down = (self._gelu_tanh(gate) * up) @ self._w(
                base + "ffn_down.weight"
            ).T
            down = self._rmsnorm(
                down, self._w(base + "post_ffw_norm.weight"))
            x = (x1 + down) * self._w(base + "layer_output_scale.weight")
        last = self._rmsnorm(x[-1:], self.out_norm)
        vocab = int(self._head_info["dims"][1])
        logits = np.empty((vocab,), dtype=np.float32)
        block = 8192
        for b0 in range(0, vocab, block):
            b1 = min(b0 + block, vocab)
            w = self._head_block(b0, b1)
            logits[b0:b1] = (last @ w.T)[0]
        if self.final_logit_softcapping is not None:
            cap = np.float32(self.final_logit_softcapping)
            logits = cap * np.tanh(logits / cap)
        return logits

    def generate(self, prompt: str, max_tokens: int = 32) -> tuple[str, list[int]]:
        """Greedy-generate from ``prompt``. Returns (text, token ids).
        Stops after emitting the EOS id (not included in the text), or at
        ``max_tokens``. No BOS is prepended (parity item — see module
        docstring). The cache is reset per call."""
        self.cache = {}
        ids = self.tok.encode(prompt)
        logits = self.forward(ids, 0)
        out: list[int] = []
        for _ in range(int(max_tokens)):
            nxt = int(np.argmax(logits))
            if self.eos is not None and nxt == self.eos:
                break
            out.append(nxt)
            logits = self.forward([nxt], len(ids) + len(out) - 1)
        return self.tok.decode(out), out

    def close(self) -> None:
        self.store.close()
