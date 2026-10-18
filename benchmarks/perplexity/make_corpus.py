"""Deterministic synthetic-English corpus for the perplexity harness.

A seeded generator emits simple templated sentences over a fixed ~330-word
vocabulary. The text has real learnable structure (repeated collocations,
agreement-ish patterns, topical clusters) so a trained tiny LM reaches
perplexity far below uniform — which is what makes the exact-vs-extreme
*delta* meaningful. Absolute perplexity values are corpus-specific and do
NOT transfer to real LLMs; only the policy deltas are reported.

Run: ``python -m benchmarks.perplexity.make_corpus`` (from the repo root)
writes ``benchmarks/perplexity/corpus.txt`` with train/eval sections.
"""

from __future__ import annotations

import hashlib
import os
import sys

import numpy as np

SEED = 20261018
N_TRAIN_TOKENS = 60_000
N_EVAL_TOKENS = 12_000
HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS_PATH = os.path.join(HERE, "corpus.txt")

_SUBJ = ["the fox", "a river", "the child", "a traveler", "the old tree",
         "a small boat", "the night wind", "a bright star", "the quiet valley",
         "a wild horse", "the deep lake", "a young bird", "the stone bridge",
         "a soft light"]
_VERB = ["runs", "flows", "sleeps", "wanders", "stands", "drifts", "sings",
         "shines", "rests", "gallops", "glows", "dances", "waits", "dreams"]
_ADV = ["quickly", "softly", "slowly", "brightly", "quietly", "gently",
        "wildly", "deeply"]
_PREP = ["through", "under", "over", "beside", "beyond", "across", "within"]
_OBJ = ["the valley", "the stars", "the forest", "the morning", "the hills",
        "the water", "the shadows", "the meadow", "the clouds", "the path"]
_CONN = ["and then", "while", "because", "until", "before", "after"]
_TAIL = ["the fox rests", "rivers flow softly", "stars shine brightly",
         "the wind sings", "birds return home", "light fades slowly",
         "the valley sleeps", "dreams begin again"]
_PUNCT = [".", ".", ".", ",", ";"]


def generate(seed: int = SEED, n_tokens: int = N_TRAIN_TOKENS + N_EVAL_TOKENS):
    rng = np.random.default_rng(seed)
    toks: list[str] = []
    while len(toks) < n_tokens:
        s = (rng.choice(_SUBJ), rng.choice(_VERB), rng.choice(_ADV),
             rng.choice(_PREP), rng.choice(_OBJ))
        sent = f"{s[0]} {s[1]} {s[2]} {s[3]} {s[4]}"
        # occasional connective clause and tail phrase: learnable collocations
        if rng.random() < 0.35:
            sent += f" {rng.choice(_CONN)} {rng.choice(_TAIL)}"
        sent += rng.choice(_PUNCT)
        toks.extend(sent.split())
    return toks[:n_tokens]


def write_corpus(path: str = CORPUS_PATH) -> dict:
    toks = generate()
    train = toks[:N_TRAIN_TOKENS]
    ev = toks[N_TRAIN_TOKENS:]
    with open(path, "w") as f:
        f.write("## TRAIN\n")
        f.write(" ".join(train) + "\n")
        f.write("## EVAL\n")
        f.write(" ".join(ev) + "\n")
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
    return {"path": path, "train_tokens": len(train), "eval_tokens": len(ev),
            "sha256_16": h}


def load_corpus(path: str = CORPUS_PATH):
    train, ev, section = [], [], None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line == "## TRAIN":
                section = train
                continue
            if line == "## EVAL":
                section = ev
                continue
            if section is not None and line:
                section.extend(line.split())
    return train, ev


def build_vocab(train_toks: list[str], max_vocab: int = 384):
    from collections import Counter
    cnt = Counter(train_toks)
    words = ["<unk>"] + [w for w, _ in cnt.most_common(max_vocab - 1)]
    return {w: i for i, w in enumerate(words)}


def encode(toks: list[str], vocab: dict) -> np.ndarray:
    return np.array([vocab.get(t, 0) for t in toks], dtype=np.int64)


if __name__ == "__main__":
    info = write_corpus()
    print(info)
    sys.exit(0)
