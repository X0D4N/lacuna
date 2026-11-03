"""Speculative decoding for Gemma-4 (GPU path) vs plain greedy.

Synthetic weights only — no model file is downloaded. Reuses the tiny
gemma4 GGUF builder from test_gpu_gemma.py (hidden 16, sliding 4Q/2KV
x dim 4, global 4Q/1KV x dim 8, FFN 24, window 3, 8-token SPM toy
vocab). The numpy backend runs on CPU here; the torch-CUDA path is a
Kaggle check (parent launches it), not something these tests can
exercise.

What these tests pin:

* ``GPUGemmaModel.forward_all`` is the same layer math as
  ``forward`` (last-position logits identical; every row matches
  incremental single-token forwards).
* ``GPUGemmaModel.truncate_cache`` uses uniform absolute positions
  across layer types: rollback equals a fresh prefill (bit-identical
  logits), and rolling back past a trimmed window head fails loud
  instead of silently corrupting attention. ``truncate_cache(0)``
  clears to fresh state.
* Plain ``forward`` still trims (the ``trim=False`` mode is only
  for verification; generation behaviour is unchanged).
* GPU speculative ids are identical to plain greedy GPU ids on
  several prompts, including a long prompt that forces repeated
  sliding trims — with the sliding cache bounded by the window
  throughout — and to plain greedy CPU ids on the same prompts.
* Stats accounting is sane, and generation is deterministic across
  fresh instances.
"""

import os

import numpy as np
import pytest

from lacuna.gemma import PagedGemma4
from lacuna.gpu_executor import (
    GPUExecutorError,
    GPUGemmaModel,
    NumpyBackend,
)
from lacuna.speculative import NgramDraft
from lacuna.speculative_gpu_gemma import GPUGemmaSpeculativeDecoder

try:
    from tests.test_gpu_gemma import build_tiny_gemma_gguf
except ImportError:
    from test_gpu_gemma import build_tiny_gemma_gguf


def _load(tmp_path, **kw):
    gguf = build_tiny_gemma_gguf(os.path.join(tmp_path, "tiny.gguf"), **kw)
    return GPUGemmaModel.from_gguf(
        gguf, backend=NumpyBackend(), hot_budget_bytes=1024 * 1024)


def _draft():
    d = NgramDraft(max_order=4)
    d.train([
        [6, 5, 3, 6, 5, 3, 6, 5, 3, 6, 5, 3],
        [3, 5, 4, 5, 6, 5, 3],
        [6, 5, 6, 5, 6, 5, 3],
        [3, 4, 3, 4, 3, 4],
    ])
    return d


# ------------------------------------------------------------ forward_all ---


def test_forward_all_last_matches_forward(tmp_path):
    # Same layer math, but a (T,H) head matmul need not reduce in
    # the same order as a (1,H) one (measured 9.3e-10 on the tiny
    # model), so compare at 2e-3 with an exact argmax — the GPU
    # test file's discipline — not bitwise.
    m = _load(tmp_path)
    try:
        ids = [6, 5, 3, 6, 5]
        all_logits = m.forward_all(ids, 0)
        assert all_logits.shape == (len(ids), 8)
        m.reset_cache()
        last = m.forward(ids, 0)
        assert np.argmax(all_logits[-1]) == np.argmax(last)
        assert np.allclose(all_logits[-1], last, atol=2e-3)
    finally:
        m.close()


def test_forward_all_rows_match_incremental_forwards(tmp_path):
    # Row i of forward_all must match the last-position logits of
    # single-token forwards run incrementally over the same prefix.
    # Different cache lengths change the attention reduction shape,
    # so compare at 2e-3 with an exact argmax requirement (the GPU
    # test file's discipline), not bitwise.
    m = _load(tmp_path)
    try:
        ids = [6, 5, 3, 6, 5, 3]
        all_logits = m.forward_all(ids, 0)
        m.reset_cache()
        logits = m.forward(ids[:1], 0)
        assert np.argmax(logits) == np.argmax(all_logits[0])
        assert np.allclose(logits, all_logits[0], atol=2e-3)
        for i in range(1, len(ids)):
            logits = m.forward([ids[i]], i)
            assert np.argmax(logits) == np.argmax(all_logits[i])
            assert np.allclose(logits, all_logits[i], atol=2e-3)
    finally:
        m.close()


# ---------------------------------------------------------- truncate_cache ---


def test_truncate_rollback_equals_fresh_prefill(tmp_path):
    ids = [6, 5, 3, 6, 5, 3, 6, 5]
    nxt = 3
    m = _load(tmp_path)
    try:
        m.forward_all(ids, 0)
        m.truncate_cache(5)
        got = m.forward_all([nxt], 5)
    finally:
        m.close()
    m2 = _load(tmp_path)
    try:
        m2.forward_all(ids[:5], 0)
        m2.truncate_cache(5)
        want = m2.forward_all([nxt], 5)
    finally:
        m2.close()
    # allclose, not array_equal: the rollback path's cache was built
    # in a T=8 batch and the fresh path's in a T=5 batch, and a
    # batched-head matmul need not reduce in the same order across
    # batch shapes (same reason the repo's GPU/CPU parity tolerance
    # is 2e-3). Measured divergence here is 1 ULP (~5e-10); 1e-6 is
    # 2000x tighter than the parity tolerance and still pins the
    # rollback semantics exactly.
    assert np.allclose(got, want, atol=1e-6)


def test_truncate_beyond_window_head_fails_loud(tmp_path):
    m = _load(tmp_path)  # window 3
    try:
        m.forward_all([6, 5, 3, 6, 5, 3, 6, 5], 0)
        m.truncate_cache(8)  # re-trims sliding caches to [5, 8)
        sliding = m._caches[0]
        assert sliding["k"].shape[0] == 3 and sliding["start"] == 5
        with pytest.raises(GPUExecutorError):
            m.truncate_cache(3)  # 3 <= head 5: keys already trimmed
        with pytest.raises(GPUExecutorError):
            m.truncate_cache(-1)
        m.truncate_cache(0)  # always allowed: caches cleared
        assert m.cache_len() == 0
        assert all("k" not in c for c in m._caches)
    finally:
        m.close()


def test_cache_len_tracks_emitted_prefix(tmp_path):
    m = _load(tmp_path)
    try:
        assert m.cache_len() == 0
        m.forward_all([6, 5, 3, 6, 5], 0)
        m.truncate_cache(5)
        # Global layers hold all 5; sliding layers hold the last 3.
        assert m.cache_len() == 5
        assert m._caches[0]["k"].shape[0] == 3  # sliding, window-bound
        assert m._caches[5]["k"].shape[0] == 5  # global, unbounded
    finally:
        m.close()


def test_plain_forward_still_trims(tmp_path):
    # The trim=False mode is verification-only: plain forward() must
    # keep trimming sliding caches to the window exactly as before.
    m = _load(tmp_path)
    try:
        m.forward([6, 5, 3, 6, 5, 3, 6, 5], 0)
        assert m._caches[0]["k"].shape[0] == 3
        assert m._caches[0]["start"] == 5
        assert m._caches[5]["k"].shape[0] == 8
        assert m._caches[5]["start"] == 0
    finally:
        m.close()


# ------------------------------------------------------------- spec vs plain ---


def _check_spec_matches_plain(m, prompt, max_tokens, draft, k=4):
    _, plain_ids, _ = m.generate(prompt, max_tokens=max_tokens)
    dec = GPUGemmaSpeculativeDecoder(m, draft, k=k)
    text, spec_ids, stats = dec.generate(prompt, max_tokens=max_tokens)
    assert spec_ids == plain_ids
    assert stats["tokens"] == len(spec_ids)
    assert stats["proposed_tokens"] == stats["rounds"] * k
    assert 0.0 <= stats["acceptance_rate"] <= 1.0
    assert 0 <= stats["accepted_tokens"] <= stats["proposed_tokens"]
    assert text == m.tok.decode(spec_ids)
    return stats


def test_spec_matches_plain_on_prompts(tmp_path):
    m = _load(tmp_path)
    try:
        for prompt in ("ab a", "a b ab", "ab ab a"):
            stats = _check_spec_matches_plain(
                m, prompt, max_tokens=8, draft=_draft())
            assert stats["rounds"] >= 1
    finally:
        m.close()


def test_spec_matches_plain_long_prompt_forcing_trims(tmp_path):
    # 13-token prompt, window 3: sliding caches trim on nearly every
    # step of both plain and speculative generation. Ids must still
    # match exactly, and the sliding cache must stay window-bound.
    m = _load(tmp_path)
    try:
        stats = _check_spec_matches_plain(
            m, "ab ab ab ab ab ab a", max_tokens=8, draft=_draft())
        assert stats["rounds"] >= 1
        assert m._caches[0]["k"].shape[0] <= 3  # sliding window-bound
        assert m._caches[0]["start"] > 0  # head actually moved
    finally:
        m.close()


def test_spec_matches_cpu_plain(tmp_path):
    # Cross-path check: GPU speculative ids must equal plain greedy
    # CPU ids on the same prompts (GPU plain already matches CPU
    # plain, and spec is exact by construction, so this must hold
    # unless a near-tie flips between backends — a flip here is a
    # signal to investigate margins, not to loosen the test).
    gguf = build_tiny_gemma_gguf(os.path.join(tmp_path, "tiny.gguf"))
    cpu = PagedGemma4(gguf, hot_budget_bytes=1024 * 1024)
    gpu = GPUGemmaModel.from_gguf(
        gguf, backend=NumpyBackend(), hot_budget_bytes=1024 * 1024)
    try:
        for prompt in ("ab a", "ab ab a"):
            _, cpu_ids = cpu.generate(prompt, max_tokens=8)
            dec = GPUGemmaSpeculativeDecoder(gpu, _draft(), k=4)
            _, spec_ids, _ = dec.generate(prompt, max_tokens=8)
            assert spec_ids == cpu_ids
    finally:
        cpu.close()
        gpu.close()


def test_spec_deterministic_across_fresh_instances(tmp_path):
    gguf = build_tiny_gemma_gguf(os.path.join(tmp_path, "tiny.gguf"))
    m1 = GPUGemmaModel.from_gguf(
        gguf, backend=NumpyBackend(), hot_budget_bytes=1024 * 1024)
    m2 = GPUGemmaModel.from_gguf(
        gguf, backend=NumpyBackend(), hot_budget_bytes=1024 * 1024)
    try:
        dec1 = GPUGemmaSpeculativeDecoder(m1, _draft(), k=4)
        dec2 = GPUGemmaSpeculativeDecoder(m2, _draft(), k=4)
        t1, ids1, s1 = dec1.generate("ab ab a", max_tokens=8)
        t2, ids2, s2 = dec2.generate("ab ab a", max_tokens=8)
        assert ids1 == ids2 and t1 == t2 and s1 == s2
    finally:
        m1.close()
        m2.close()


def test_spec_accepts_when_draft_knows_pattern(tmp_path):
    # Draft trained on this prompt's own plain output: the n-gram
    # contexts all hit, so acceptance must be nonzero (and ids still
    # identical — acceptance never changes the emitted sequence).
    m = _load(tmp_path)
    try:
        prompt = "ab ab a"
        _, plain_ids, _ = m.generate(prompt, max_tokens=12)
        prompt_ids = m.tok.encode(prompt)
        d = NgramDraft(max_order=4)
        d.train([list(prompt_ids) + list(plain_ids)])
        dec = GPUGemmaSpeculativeDecoder(m, d, k=4)
        _text, spec_ids, stats = dec.generate(prompt, max_tokens=12)
        assert spec_ids == plain_ids
        assert stats["accepted_tokens"] > 0
        assert stats["mean_tokens_per_block"] > 1.0
    finally:
        m.close()
