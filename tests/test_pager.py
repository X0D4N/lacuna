"""Pager tests: roundtrip, eviction budget, prefetch, stats."""

import numpy as np
import pytest

from lacuna.pager import MAGIC, PagedWeightStore, LacWriter
from lacuna.quant import Q2, Q4, TERNARY


@pytest.fixture()
def spp_path(tmp_path):
    p = str(tmp_path / "w.lac")
    rng = np.random.default_rng(42)
    with LacWriter(p, page_size=4096) as w:
        w.add_tensor("blk.0.attn.wq", rng.standard_normal((32, 16)).astype(np.float32),
                     layer=0, precision=Q4)
        w.add_tensor("blk.0.ffn.w1", rng.standard_normal((48, 16)).astype(np.float32),
                     layer=0, precision=Q2)
        w.add_tensor("blk.1.ffn.w2", rng.standard_normal((40, 16)).astype(np.float32),
                     layer=1, precision=TERNARY)
    return p


def test_magic_and_names(spp_path):
    with open(spp_path, "rb") as f:
        assert f.read(4) == MAGIC
    with PagedWeightStore(spp_path, hot_budget_bytes=10**9) as s:
        assert s.names == ["blk.0.attn.wq", "blk.0.ffn.w1", "blk.1.ffn.w2"]
        assert s.n_pages >= 3


def test_roundtrip_approx(spp_path):
    with PagedWeightStore(spp_path, hot_budget_bytes=10**9) as s:
        page = s.get_page(0)  # first page of the Q4 tensor
        assert page.dtype == np.float32
        assert page.shape[1] == 16
        assert np.all(np.isfinite(page))


def test_eviction_respects_budget(spp_path):
    with PagedWeightStore(spp_path, hot_budget_bytes=2048) as s:
        for i in range(s.n_pages):
            s.get_page(i)
        # hard budget: cached bytes never exceed it
        assert s._cached_bytes <= 2048
        assert s.stats["faults"] > 0


def test_prefetch_and_hit_rate(spp_path):
    with PagedWeightStore(spp_path, hot_budget_bytes=10**9) as s:
        s.prefetch(list(range(s.n_pages)))
        assert set(s.hot_pages()) == set(range(s.n_pages))
        for i in range(s.n_pages):
            s.get_page(i)
        assert s.hit_rate == 1.0
        assert s.stats["prefetches"] == s.n_pages
