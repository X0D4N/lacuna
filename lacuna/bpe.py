"""GPT-2-style byte-level BPE tokenizer, built from GGUF metadata.

The tokenizer is constructed from a GGUF file's own ``tokenizer.ggml.tokens``
and ``tokenizer.ggml.merges`` — no external tokenizer files, no
``transformers`` dependency. The merge table is the model's own, so encode
agrees with the model by construction (verified against llama-server's
``/tokenize`` endpoint, not assumed).

Pre-tokenization approximates the Qwen2/cl100k pattern with Python ``re``
(``\\p{L}``/``\\p{N}`` are approximated by Unicode-aware ``[^\\W\\d_]`` and
``\\d``). This is exact for plain-ASCII prompts; exotic scripts may split
differently from llama.cpp's regex engine, which is why the /tokenize
check is part of verification for every demo prompt.

Byte mapping is the standard GPT-2 bytes-to-unicode table: printable bytes
map to themselves, the rest to code points 256 and up, so every byte
sequence is representable and decoding is a pure inverse map.
"""

from __future__ import annotations

import re

# Pre-tokenizer: contractions | optional-prefix + letters | 1-3 digits |
# punctuation runs | newlines | trailing-space runs | whitespace runs.
_PRE = re.compile(
    r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\w]?[^\W\d_]+|\d{1,3}| ?[^\s\w]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2 byte encoder: byte value -> single unicode char."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


class BPETokenizer:
    """Byte-level BPE over a GGUF vocab + merges table."""

    def __init__(self, tokens: list[str], merges: list[str], bos: int | None = None, eos: int | None = None):
        self.tokens = list(tokens)
        self.vocab = {t: i for i, t in enumerate(self.tokens)}
        self.ranks: dict[tuple[str, str], int] = {}
        for rank, m in enumerate(merges):
            parts = m.split(" ")
            if len(parts) != 2:
                raise ValueError(f"unparsable merge entry {m!r} (expected two tokens)")
            self.ranks[(parts[0], parts[1])] = rank
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {c: b for b, c in self.byte_encoder.items()}
        self.bos = bos
        self.eos = eos

    @classmethod
    def from_metadata(cls, metadata: dict) -> "BPETokenizer":
        return cls(
            metadata["tokenizer.ggml.tokens"],
            metadata.get("tokenizer.ggml.merges", []),
            bos=metadata.get("tokenizer.ggml.bos_token_id"),
            eos=metadata.get("tokenizer.ggml.eos_token_id"),
        )

    def _bpe(self, piece: str) -> tuple[str, ...]:
        """Merge one pre-token (a byte-level string) into vocab pieces."""
        word = tuple(piece)
        if len(word) == 1:
            return word
        while len(word) >= 2:
            best = None
            best_rank = None
            for i in range(len(word) - 1):
                r = self.ranks.get((word[i], word[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best = (word[i], word[i + 1])
            if best is None:
                break
            merged = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == best:
                    merged.append(word[i] + word[i + 1])
                    i += 2
                else:
                    merged.append(word[i])
                    i += 1
            word = tuple(merged)
            if len(word) == 1:
                break
        return word

    def encode(self, text: str, add_bos: bool = False) -> list[int]:
        ids: list[int] = []
        if add_bos and self.bos is not None:
            ids.append(self.bos)
        for pre in _PRE.findall(text):
            byte_str = "".join(self.byte_encoder[b] for b in pre.encode("utf-8"))
            for piece in self._bpe(byte_str):
                try:
                    ids.append(self.vocab[piece])
                except KeyError:
                    raise ValueError(
                        f"BPE piece {piece!r} from pre-token {pre!r} is not "
                        f"in the vocab — tokenizer/vocab mismatch"
                    ) from None
        return ids

    def decode(self, ids) -> str:
        text = "".join(self.tokens[i] for i in ids)
        raw = bytes(self.byte_decoder[c] for c in text)
        return raw.decode("utf-8", errors="replace")
