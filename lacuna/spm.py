"""SentencePiece-BPE tokenizer built from GGUF metadata (Gemma family).

Gemma models do NOT use the GPT-2-style BPE in :mod:`lacuna.bpe`:
their tokenizer is SentencePiece (GGUF ``tokenizer.ggml.model ==
"llama"``), with a 262,144-piece vocab, BOS id 2, EOS id 1, PAD id 0
(Gemma4TextConfig defaults), and ``add_dummy_prefix == False`` (Gemma
differs from Llama-2 here: no ``▁`` is prepended to the text).

There is no ``sentencepiece`` package on this box (checked: system
python and the corsac venv both fail to import it), so this module is
Lacuna-native pure Python: it reads ``tokenizer.ggml.tokens``,
``tokenizer.ggml.scores`` and ``tokenizer.ggml.token_type`` from the
GGUF header and implements SentencePiece BPE exactly as far as GGUF
metadata allows:

- Text is NOT pre-tokenized by regex. Whitespace becomes ``▁``
  (U+2581) in place; no dummy prefix is added.
- Initial symbols are characters, except characters missing from the
  vocab, which fall back to their UTF-8 bytes as ``<0xXX>`` pieces
  (token type 6) when those pieces exist.
- BPE merges repeatedly join the adjacent pair whose merged piece has
  the HIGHEST score (SentencePiece BPE merge order is score order).
  A merged piece must itself be in the vocab; otherwise the pair is
  not mergeable.
- Special pieces (token types 3 = control, 4 = user-defined) present
  literally in the input text are matched first, longest first, and
  are never split or merged.
- Decode joins pieces, maps ``▁`` back to a space, expands
  ``<0xXX>`` byte pieces back to bytes, and skips control /
  user-defined specials (they are structural, not text). Generated
  text therefore never contains a literal ``<eos>``.

What this is NOT: it is not the ``sentencepiece`` library, and its
merge order has not been verified against a real Gemma GGUF yet
(no real file has been downloaded). Real-weight parity against
``llama-cli`` on the real tokenizer is a remaining verification item;
until then this tokenizer is verified only on synthetic vocabs
(see tests/test_gemma.py). Scores are REQUIRED: without them BPE
merge order is undefined, so ``from_metadata`` fails loud instead of
silently tokenizing in vocab order.
"""

from __future__ import annotations

import re

# GGUF tokenizer.ggml.token_type values (llama.cpp convention).
_T_NORMAL = 1
_T_UNKNOWN = 2
_T_CONTROL = 3
_T_USER_DEFINED = 4
_T_UNUSED = 5
_T_BYTE = 6

_BYTE_RE = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")


class SPMTokenizerError(ValueError):
    """Raised for malformed SPM metadata or unencodable text. Fails loud."""


class SPMTokenizer:
    """SentencePiece-BPE tokenizer from GGUF metadata."""

    def __init__(self, tokens, scores, token_types,
                 bos: int | None = None, eos: int | None = None):
        self.tokens = list(tokens)
        self.scores = list(scores)
        self.types = list(token_types)
        if not (len(self.tokens) == len(self.scores) == len(self.types)):
            raise SPMTokenizerError(
                f"SPM metadata length mismatch: tokens={len(self.tokens)} "
                f"scores={len(self.scores)} types={len(self.types)}"
            )
        self.vocab = {t: i for i, t in enumerate(self.tokens)}
        self.bos = bos
        self.eos = eos
        # Special pieces (control / user-defined) matched literally.
        self._specials = sorted(
            (t for t, ty in zip(self.tokens, self.types, strict=False)
             if ty in (_T_CONTROL, _T_USER_DEFINED)),
            key=len, reverse=True,
        )
        self._special_re = None
        if self._specials:
            self._special_re = re.compile(
                "(" + "|".join(re.escape(s) for s in self._specials) + ")"
            )

    @classmethod
    def from_metadata(cls, metadata: dict) -> "SPMTokenizer":
        """Build from a GGUF metadata dict. Requires SPM model metadata
        with tokens AND scores; fails loud otherwise. The model tag is
        ``llama`` for most SPM files but ``gemma4`` for real Gemma-4
        files (measured on bartowski/google_gemma-4-31B-it-GGUF, 2026):
        the piece/score tables are the same format either way, so both
        tags are accepted and anything else still fails loud."""
        model = metadata.get("tokenizer.ggml.model")
        if model is not None and model not in ("llama", "gemma4"):
            raise SPMTokenizerError(
                f"SPM tokenizer needs tokenizer.ggml.model 'llama' or "
                f"'gemma4', got {model!r}"
            )
        tokens = metadata.get("tokenizer.ggml.tokens")
        scores = metadata.get("tokenizer.ggml.scores")
        if tokens is None:
            raise SPMTokenizerError("tokenizer.ggml.tokens missing")
        if scores is None:
            raise SPMTokenizerError(
                "tokenizer.ggml.scores missing — BPE merge order is "
                "undefined without scores; refusing to guess"
            )
        types = metadata.get(
            "tokenizer.ggml.token_type", [_T_NORMAL] * len(tokens)
        )
        return cls(
            tokens, scores, types,
            bos=metadata.get("tokenizer.ggml.bos_token_id"),
            eos=metadata.get("tokenizer.ggml.eos_token_id"),
        )

    # -- encode -------------------------------------------------------
    def _initial_symbols(self, text: str) -> list[str]:
        """Split text into initial BPE symbols (chars, byte-fallback)."""
        symbols: list[str] = []
        for ch in text:
            if ch in self.vocab:
                symbols.append(ch)
            else:
                # Byte fallback: one <0xXX> symbol per UTF-8 byte.
                for b in ch.encode("utf-8"):
                    piece = f"<0x{b:02X}>"
                    if piece not in self.vocab:
                        raise SPMTokenizerError(
                            f"character {ch!r} (byte 0x{b:02X}) has no vocab "
                            f"piece and no byte-fallback piece"
                        )
                    symbols.append(piece)
        return symbols

    def _bpe(self, symbols: list[str]) -> list[str]:
        """Merge adjacent pairs by highest merged-piece score."""
        symbols = list(symbols)
        while len(symbols) >= 2:
            best_idx = None
            best_score = None
            for i in range(len(symbols) - 1):
                merged = symbols[i] + symbols[i + 1]
                j = self.vocab.get(merged)
                if j is None:
                    continue
                s = self.scores[j]
                if best_score is None or s > best_score:
                    best_score = s
                    best_idx = i
            if best_idx is None:
                break
            symbols[best_idx:best_idx + 2] = [
                symbols[best_idx] + symbols[best_idx + 1]
            ]
        return symbols

    def encode(self, text: str, add_bos: bool = False) -> list[int]:
        """Encode text. Gemma uses add_dummy_prefix=False: no leading ▁
        is added; spaces inside the text become ▁ in place."""
        ids: list[int] = []
        if add_bos and self.bos is not None:
            ids.append(self.bos)
        # Split on literal special pieces first (never split or merged).
        segments = (
            self._special_re.split(text) if self._special_re else [text]
        )
        for seg in segments:
            if seg in self.vocab and seg in self._specials:
                ids.append(self.vocab[seg])
                continue
            if seg == "":
                continue
            # SPM whitespace handling: space -> ▁, in place.
            norm = seg.replace(" ", "\u2581")
            for piece in self._bpe(self._initial_symbols(norm)):
                j = self.vocab.get(piece)
                if j is None:
                    raise SPMTokenizerError(
                        f"BPE piece {piece!r} is not in the vocab"
                    )
                ids.append(j)
        return ids

    # -- decode -------------------------------------------------------
    def decode(self, ids) -> str:
        """Decode ids to text. Control/user-defined specials are skipped;
        ▁ maps back to a space; <0xXX> pieces expand to bytes."""
        out_bytes = bytearray()
        text_parts: list[str] = []
        for i in ids:
            piece = self.tokens[int(i)]
            ty = self.types[int(i)]
            if ty in (_T_CONTROL, _T_USER_DEFINED, _T_UNUSED):
                continue
            m = _BYTE_RE.match(piece)
            if m is not None:
                out_bytes.append(int(m.group(1), 16))
                continue
            if out_bytes:
                text_parts.append(out_bytes.decode("utf-8", errors="replace"))
                out_bytes = bytearray()
            text_parts.append(piece)
        if out_bytes:
            text_parts.append(out_bytes.decode("utf-8", errors="replace"))
        return "".join(text_parts).replace("\u2581", " ")
