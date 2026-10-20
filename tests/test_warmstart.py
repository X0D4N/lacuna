"""Warm-start persistent hot-set cache (lacuna/warmstart.py).

The cache must only ever cost a cold start when it is wrong — never wrong
data. These tests pin: restore works, restored bytes are the store's bytes,
corrupt/truncated/stale caches fall back cleanly, a checksum-mismatched
page is skipped at restore and re-faulted from the store on demand, the
hard RAM budget is respected (hottest pages win), and the Engine wires
save/restore automatically.
"""

import os

import numpy as np
import pytest

from lacuna import Engine
from lacuna.pager import PagedWeightStore, LacWriter
from lacuna.quant import Q4
from lacuna.warmstart import default_hotset_path

N_TENSORS = 4


@pytest.fixture()
def lac_path(tmp_path):
    p = str(tmp_path / "w.lac")
    rng = np.random.default_rng(11)
    with LacWriter(p, page_size=4096) as w:
        for i in range(N_TENSORS):
            w.add_tensor(f"t{i}", rng.standard_normal((32, 16)).astype(np.float32),
                         layer=0, precision=Q4)
    return p


def _hot_store(path, budget=10**9):
    s = PagedWeightStore(path, hot_budget_bytes=budget)
    for i in range(s.n_pages):
        s.get_page(i)
    return s


def test_warm_start_restores_residency_without_faults(lac_path, tmp_path):
    cache = str(tmp_path / "w.hotset")
    with _hot_store(lac_path) as s:
        rep = s.save_hotset(cache)
        assert rep["saved"] == s.n_pages
    with PagedWeightStore(lac_path, hot_budget_bytes=10**9) as s2:
        rep = s2.restore_hotset(cache)
        assert rep["restored"] == s2.n_pages
        assert rep["invalidated"] == 0
        assert s2.stats["warm_restored"] == s2.n_pages
        for i in range(s2.n_pages):
            assert s2.is_resident(i)
            s2.get_page(i)
        assert s2.stats["faults"] == 0  # every access was a hit
        assert s2.hit_rate == 1.0


def test_restored_pages_are_byte_identical_to_cold(lac_path, tmp_path):
    cache = str(tmp_path / "w.hotset")
    with _hot_store(lac_path) as s:
        cold = [s.get_page(i).copy() for i in range(s.n_pages)]
        s.save_hotset(cache)
    with PagedWeightStore(lac_path, hot_budget_bytes=10**9) as s2:
        s2.restore_hotset(cache)
        for i in range(s2.n_pages):
            assert np.array_equal(s2.get_page(i), cold[i])


def test_missing_cache_is_clean_noop(lac_path, tmp_path):
    with PagedWeightStore(lac_path, hot_budget_bytes=10**9) as s:
        rep = s.restore_hotset(str(tmp_path / "nope.hotset"))
        assert rep["restored"] == 0 and rep["invalidated"] == 0
        s.get_page(0)  # store still fully functional
        assert s.stats["faults"] == 1


def test_corrupt_cache_falls_back_to_cold_start(lac_path, tmp_path):
    cache = tmp_path / "w.hotset"
    cache.write_bytes(b"\xde\xad\xbe\xef" * 64)
    with PagedWeightStore(lac_path, hot_budget_bytes=10**9) as s:
        rep = s.restore_hotset(str(cache))
        assert rep["restored"] == 0
        # cold path unaffected: pages fault in with correct data
        page = s.get_page(0)
        assert page.shape[1] == 16
        assert s.stats["faults"] == 1


def test_truncated_cache_falls_back_to_cold_start(lac_path, tmp_path):
    cache = str(tmp_path / "w.hotset")
    with _hot_store(lac_path) as s:
        s.save_hotset(cache)
    with open(cache, "r+b") as f:
        f.truncate(os.path.getsize(cache) // 2)
    with PagedWeightStore(lac_path, hot_budget_bytes=10**9) as s:
        rep = s.restore_hotset(cache)
        assert rep["restored"] == 0
        assert s.get_page(0).shape[1] == 16


def test_checksum_mismatch_page_is_skipped_and_refetched(lac_path, tmp_path):
    cache = str(tmp_path / "w.hotset")
    st = os.stat(lac_path)
    with _hot_store(lac_path) as s:
        victim = 1
        victim_off = s.page_table()[victim]["offset"] + 20  # inside payload
        s.save_hotset(cache)
    # flip one payload byte, then restore size+mtime so ONLY the per-page
    # checksum can catch the change (fingerprint still matches)
    with open(lac_path, "r+b") as f:
        f.seek(victim_off)
        b = f.read(1)
        f.seek(victim_off)
        f.write(bytes([b[0] ^ 0xFF]))
    os.utime(lac_path, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert os.stat(lac_path).st_size == st.st_size

    with PagedWeightStore(lac_path, hot_budget_bytes=10**9) as s2:
        rep = s2.restore_hotset(cache)
        assert rep["invalidated"] == 1
        assert rep["restored"] == s2.n_pages - 1
        assert not s2.is_resident(victim)
        # demand fault returns the store's CURRENT bytes, never cached data
        got = s2.get_page(victim)
        assert s2.stats["faults"] == 1
        fresh = s2._load_page(victim)[0]
        assert np.array_equal(got, fresh)


def test_fingerprint_mismatch_ignores_snapshot(lac_path, tmp_path):
    cache = str(tmp_path / "w.hotset")
    with _hot_store(lac_path) as s:
        s.save_hotset(cache)
    # rewrite the model file (different tensors -> different size/mtime)
    rng = np.random.default_rng(99)
    with LacWriter(lac_path, page_size=4096) as w:
        w.add_tensor("other", rng.standard_normal((64, 16)).astype(np.float32),
                     layer=0, precision=Q4)
    with PagedWeightStore(lac_path, hot_budget_bytes=10**9) as s:
        rep = s.restore_hotset(cache)
        assert rep["restored"] == 0
        assert "fingerprint" in rep["reason"]


def test_restore_respects_budget_and_keeps_hottest(lac_path, tmp_path):
    cache = str(tmp_path / "w.hotset")
    with PagedWeightStore(lac_path, hot_budget_bytes=10**9) as s:
        n = s.n_pages
        assert n >= 4
        # distinct frequencies: page i touched i+1 times
        for i in range(n):
            for _ in range(i + 1):
                s.get_page(i)
        freqs = {i: s.frequency(i) for i in range(n)}
        s.save_hotset(cache)
    page_bytes = 13 + (32 * 16 // 2 + 3)  # header + Q4 payload of a (32,16) page
    with PagedWeightStore(lac_path, hot_budget_bytes=2 * page_bytes) as s2:
        s2.restore_hotset(cache)
        # every page validated, but the hard budget holds only the 2 hottest
        assert len(s2.hot_pages()) == 2
        assert s2._cached_bytes <= 2 * page_bytes
        hottest = sorted(freqs, key=freqs.get)[-2:]
        assert set(s2.hot_pages()) == set(hottest)


def test_ewma_frequency_survives_roundtrip(lac_path, tmp_path):
    cache = str(tmp_path / "w.hotset")
    with PagedWeightStore(lac_path, hot_budget_bytes=10**9) as s:
        for _ in range(5):
            s.get_page(0)
        f0 = s.frequency(0)
        s.save_hotset(cache)
    with PagedWeightStore(lac_path, hot_budget_bytes=10**9) as s2:
        s2.restore_hotset(cache)
        assert s2.frequency(0) == pytest.approx(f0)


def test_save_is_atomic_and_leaves_no_tmp(lac_path, tmp_path):
    cache = str(tmp_path / "w.hotset")
    with _hot_store(lac_path) as s:
        s.save_hotset(cache)
    assert os.path.exists(cache)
    assert not os.path.exists(cache + ".tmp")


def test_engine_warm_start_roundtrip(lac_path):
    eng = Engine(lac_path)
    try:
        assert eng.mode == "lacuna"
        eng.generate("hello", max_tokens=4)
        n_pages = eng.store.n_pages
    finally:
        eng.close()
    cache = default_hotset_path(lac_path)
    assert os.path.exists(cache)

    eng2 = Engine(lac_path)
    try:
        assert eng2.store.stats["warm_restored"] == n_pages
        eng2.generate("hello again", max_tokens=4)
        # every page came from the warm cache: zero NVMe faults on run 2
        assert eng2.store.stats["faults"] == 0
    finally:
        eng2.close()


def test_engine_warm_start_disabled_writes_nothing(lac_path):
    eng = Engine(lac_path, warm_start=False)
    try:
        eng.generate("hello", max_tokens=2)
    finally:
        eng.close()
    assert not os.path.exists(default_hotset_path(lac_path))
