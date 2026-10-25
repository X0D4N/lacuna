"""Speculative paged inference tests: exactness against plain greedy.

Runs on the tiny synthetic Qwen2 GGUF (same builder as test_paged_model)
so no model file is needed. The decisive property: speculative ids must
equal plain greedy paged ids for ANY draft — an oracle draft (full
acceptance + bonus path), a constant wrong draft (all-resample path), and
a real n-gram draft. If any of these diverges, the mechanism is wrong.
"""

import os

import numpy as np
import pytest

try:
    from tests.test_paged_model import build_tiny_gguf
except ImportError:  # pytest run with tests/ itself on sys.path
    from test_paged_model import build_tiny_gguf

from lacuna.paged_model import PagedQwen2
from lacuna.speculative import (
    BatchedQwen2,
    NeuralDraft,
    NgramDraft,
    SpeculativeDecoder,
)

MODEL_05B = os.path.expanduser(
    "~/workspace/lacuna-models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
)


class OracleDraft:
    """Proposes a precomputed continuation (full-acceptance path)."""

    def __init__(self, continuation, prompt_len):
        self.continuation = list(continuation)
        self.prompt_len = int(prompt_len)

    def propose(self, prefix, k):
        pos = len(prefix) - self.prompt_len
        return self.continuation[pos : pos + int(k)]


class ConstantDraft:
    """Proposes the same id forever (all-resample path)."""

    def __init__(self, tok=3):
        self.tok = int(tok)

    def propose(self, prefix, k):
        return [self.tok] * int(k)


def _tiny(tmp_path):
    return build_tiny_gguf(os.path.join(tmp_path, "tiny.gguf"))


def test_forward_all_last_matches_forward(tmp_path):
    gguf = _tiny(tmp_path)
    m = BatchedQwen2(gguf, hot_budget_bytes=1024 * 1024)
    try:
        logits_all = m.forward_all([0, 1, 2], 0)
        assert logits_all.shape == (3, 4)
        assert np.isfinite(logits_all).all()
        # The last row must be bit-identical to plain forward() (same ops).
        m.cache = {}
        logits_last = m.forward([0, 1, 2], 0)
        assert np.array_equal(logits_all[-1], logits_last)
    finally:
        m.close()


def test_speculative_matches_plain_oracle_draft(tmp_path):
    gguf = _tiny(tmp_path)
    plain = PagedQwen2(gguf, hot_budget_bytes=1024 * 1024)
    try:
        _, plain_ids = plain.generate("ab a", max_tokens=8)
    finally:
        plain.close()
    spec = BatchedQwen2(gguf, hot_budget_bytes=1024 * 1024)
    try:
        prompt_ids = spec.tok.encode("ab a")
        draft = OracleDraft(plain_ids, len(prompt_ids))
        dec = SpeculativeDecoder(spec, draft, k=4)
        _, spec_ids, stats = dec.generate("ab a", max_tokens=8)
        assert spec_ids == plain_ids
        assert stats["acceptance_rate"] == 1.0  # oracle: everything accepted
        assert stats["rounds"] >= 1
    finally:
        spec.close()


def test_speculative_matches_plain_wrong_draft(tmp_path):
    gguf = _tiny(tmp_path)
    plain = PagedQwen2(gguf, hot_budget_bytes=1024 * 1024)
    try:
        _, plain_ids = plain.generate("ab a", max_tokens=8)
    finally:
        plain.close()
    spec = BatchedQwen2(gguf, hot_budget_bytes=1024 * 1024)
    try:
        dec = SpeculativeDecoder(spec, ConstantDraft(tok=3), k=4)
        _, spec_ids, stats = dec.generate("ab a", max_tokens=8)
        assert spec_ids == plain_ids  # resample path still exact
        assert stats["proposed_tokens"] > 0
    finally:
        spec.close()


def test_speculative_matches_plain_ngram_draft(tmp_path):
    gguf = _tiny(tmp_path)
    plain = PagedQwen2(gguf, hot_budget_bytes=1024 * 1024)
    try:
        _, plain_ids = plain.generate("ab a", max_tokens=8)
    finally:
        plain.close()
    spec = BatchedQwen2(gguf, hot_budget_bytes=1024 * 1024)
    try:
        draft = NgramDraft()
        draft.train([[0, 1, 2, 3, 0, 1], [3, 2, 1, 0, 3, 2]])
        dec = SpeculativeDecoder(spec, draft, k=4)
        _, spec_ids, _ = dec.generate("ab a", max_tokens=8)
        assert spec_ids == plain_ids
    finally:
        spec.close()


def test_cache_invariant_after_generate(tmp_path):
    # No EOS in the tiny model's metadata, so generation runs full. The
    # cache covers prompt + emitted, minus 1 when the last emitted token
    # is a resample/bonus (pending, not yet processed) — or all of it
    # when generation stopped mid-block on accepted tokens (which were
    # block inputs, hence already processed). Both are correct states.
    gguf = _tiny(tmp_path)
    spec = BatchedQwen2(gguf, hot_budget_bytes=1024 * 1024)
    try:
        prompt_ids = spec.tok.encode("ab a")
        dec = SpeculativeDecoder(spec, ConstantDraft(tok=0), k=4)
        _, ids, _ = dec.generate("ab a", max_tokens=8)
        total = len(prompt_ids) + len(ids)
        assert spec.cache_len() in (total - 1, total)
    finally:
        spec.close()


def test_neural_draft_proposes_valid_ids_and_keeps_cache_invariant(tmp_path):
    gguf = _tiny(tmp_path)
    draft_model = BatchedQwen2(gguf, hot_budget_bytes=1024 * 1024)
    try:
        draft = NeuralDraft(draft_model)
        prefix = draft_model.tok.encode("ab a")
        out = draft.propose(prefix, 4)
        assert len(out) == 4  # tiny model has no EOS, so no early stop
        assert all(isinstance(t, int) and 0 <= t <= 3 for t in out)
        # Cache covers exactly the emitted prefix again (draft tokens
        # truncated away), so the next call reuses the cache cleanly.
        assert draft_model.cache_len() == len(prefix)
        out2 = draft.propose(prefix, 4)
        assert out2 == out  # deterministic; cache-reuse path agrees
        # Advancing the prefix reuses the cache for the shared part.
        prefix2 = prefix + out[:2]
        out3 = draft.propose(prefix2, 4)
        assert len(out3) == 4
        assert draft_model.cache_len() == len(prefix2)
    finally:
        draft_model.close()


def test_neural_draft_end_to_end_matches_plain(tmp_path):
    # Draft and target are the same tiny model here (two instances), so
    # the draft's greedy proposals must be fully accepted — this pins
    # the NeuralDraft backend wiring, not a speedup claim.
    gguf = _tiny(tmp_path)
    plain = PagedQwen2(gguf, hot_budget_bytes=1024 * 1024)
    try:
        _, plain_ids = plain.generate("ab a", max_tokens=8)
    finally:
        plain.close()
    target = BatchedQwen2(gguf, hot_budget_bytes=1024 * 1024)
    draft_model = BatchedQwen2(gguf, hot_budget_bytes=1024 * 1024)
    try:
        dec = SpeculativeDecoder(target, NeuralDraft(draft_model), k=4)
        _, spec_ids, stats = dec.generate("ab a", max_tokens=8)
        assert spec_ids == plain_ids
        assert stats["acceptance_rate"] == 1.0
    finally:
        target.close()
        draft_model.close()


def test_ngram_draft_proposes_valid_ids():
    draft = NgramDraft()
    draft.train([[0, 1, 2, 3], [3, 2, 1, 0]])
    out = draft.propose([0, 1], 4)
    assert len(out) == 4
    assert all(isinstance(t, int) and 0 <= t <= 3 for t in out)


@pytest.mark.skipif(
    not os.path.exists(MODEL_05B), reason="local 0.5B model not present"
)
def test_speculative_matches_plain_on_05b():
    # Real-model identity check (16 tokens, 2GB budget — seconds on this
    # box). Skipped wherever the model file is absent.
    budget = 2 * 1024 * 1024 * 1024
    plain = PagedQwen2(MODEL_05B, hot_budget_bytes=budget)
    try:
        _, plain_ids = plain.generate("Name three primary colors", max_tokens=16)
    finally:
        plain.close()
    spec = BatchedQwen2(MODEL_05B, hot_budget_bytes=budget)
    try:
        draft = NgramDraft()
        draft.train([spec.tok.encode(s) for s in _GENERIC_CORPUS])
        dec = SpeculativeDecoder(spec, draft, k=4)
        _, spec_ids, _ = dec.generate("Name three primary colors", max_tokens=16)
        assert spec_ids == plain_ids
    finally:
        spec.close()


_GENERIC_CORPUS = [
    "The weather today is warm and sunny with a light breeze.",
    "She walked to the market in the morning to buy fresh bread.",
    "The train arrived late because of heavy rain on the tracks.",
    "He opened the book and started reading the first chapter.",
    "We cooked rice and vegetables for dinner last night.",
    "The children played in the garden until sunset.",
    "Please close the door before you leave the house.",
    "The river flows quietly through the valley in spring.",
]
