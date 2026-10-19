"""Buffer tests: the ggml backend-buffer contract prototype.

All numbers MEASURED on synthetic .lac files (Q4 pages, 2 pages per tensor).
Covered: alloc/bind geometry, the blocking residency fence, row-masked matvec
exactness + page selectivity, loud LRU eviction, capacity errors, RAW
rejection, and the per-layer tiling loop (bit-identical to dense, oracle
quality affecting faults only).
"""

import numpy as np
import pytest

from lacuna.buffer import (
    BufferCapacityError,
    BufferError,
    PagerBackedBuffer,
    tiled_exact_matvec,
)
from lacuna.pager import LacWriter, PagedWeightStore
from lacuna.quant import Q4

_ROWS, _COLS, _LAYERS = 80, 16, 3


def _names():
    return [f"layer.{i}.w" for i in range(_LAYERS)]


@pytest.fixture()
def spp_path(tmp_path):
    p = str(tmp_path / "buf.lac")
    rng = np.random.default_rng(11)
    with LacWriter(p, page_size=512, precision_policy="exact") as w:
        for name in _names():
            w.add_tensor(
                name, rng.standard_normal((_ROWS, _COLS)).astype(np.float32),
                layer=int(name.split(".")[1]), precision=Q4,
            )
    return p


@pytest.fixture()
def store(spp_path):
    with PagedWeightStore(spp_path, hot_budget_bytes=10**9) as s:
        assert s.n_pages == 2 * _LAYERS  # 2 pages per tensor
        yield s


def _dense(buf, name):
    t = buf.alloc(name)
    buf.bind(t)
    d = buf.tensor_data(t)
    buf.release(t)
    return d


def _rowwise_ref(dense, x, mask, pages):
    """Reference using the IDENTICAL op sequence to masked_matvec (per-page
    fancy-index + matvec): identical bytes through identical ops must give
    identical bits. A single full-matrix BLAS call may differ by ~1 ULP
    (reduction order), which is a kernel artifact, not a paging artifact."""
    out = np.zeros(len(dense), dtype=np.float32)
    for _, rs, re in pages:
        m = mask[rs:re]
        if m.any():
            out[rs:re][m] = dense[rs:re][m] @ x
    return out


def test_page_table_and_shape_accessors(store):
    table = store.page_table()
    assert len(table) == store.n_pages
    assert table[0]["page_id"] == 0
    assert set(table[0]) >= {
        "page_id", "tensor_id", "layer", "row_start", "row_end", "precision",
    }
    rows, cols = store.page_shape(0)
    assert (rows, cols) == (62, _COLS)  # (512-13) // 8 bytes/row, Q4


def test_alloc_bind_roundtrip(store):
    buf = PagerBackedBuffer(store, capacity_bytes=1 << 20)
    t = buf.alloc("layer.0.w")
    assert t.shape == (_ROWS, _COLS)
    assert len(t.page_ids) == 2
    with pytest.raises(BufferError):
        buf.tensor_data(t)  # not bound yet: loud, not garbage
    faults = buf.bind(t)
    assert faults == 2  # cold bind faults every page: MEASURED
    assert buf.is_resident(t)
    assert buf.bind(t) == 0  # warm bind faults nothing: MEASURED
    d = buf.tensor_data(t)
    assert d.shape == (_ROWS, _COLS) and np.all(np.isfinite(d))
    buf.release(t)
    assert not buf.is_resident(t)


def test_ensure_resident_is_the_fence(store):
    buf = PagerBackedBuffer(store, capacity_bytes=1 << 20)
    t = buf.alloc("layer.1.w")
    assert buf.ensure_resident(t) == 2
    assert buf.is_resident(t)
    assert store.are_resident(t.page_ids)


def test_masked_matvec_exact_and_selective(store):
    buf = PagerBackedBuffer(store, capacity_bytes=1 << 20)
    big = PagerBackedBuffer(store, capacity_bytes=1 << 20)
    dense = _dense(big, "layer.0.w")
    t = buf.alloc("layer.0.w")
    rng = np.random.default_rng(3)
    x = rng.standard_normal((_COLS,)).astype(np.float32)
    # mask selects rows ONLY from the second page (rows 62..80)
    mask = np.zeros(_ROWS, dtype=bool)
    mask[62:] = True
    out = buf.masked_matvec(t, x, mask)
    assert np.array_equal(out[mask], _rowwise_ref(dense, x, mask, t.pages)[mask])
    assert np.all(out[~mask] == 0.0)
    assert np.allclose(out[mask], (dense[mask] @ x), rtol=1e-5, atol=1e-7)  # right answer
    # selectivity: the first page was never faulted into the arena — MEASURED
    first_page = t.page_ids[0]
    assert not buf.is_page_resident(first_page), "unmasked page must not fault"
    assert buf.is_page_resident(t.page_ids[1])
    buf.release(t)


def test_masked_matvec_full_mask_equals_dense(store):
    buf = PagerBackedBuffer(store, capacity_bytes=1 << 20)
    big = PagerBackedBuffer(store, capacity_bytes=1 << 20)
    dense = _dense(big, "layer.2.w")
    t = buf.alloc("layer.2.w")
    rng = np.random.default_rng(5)
    x = rng.standard_normal((_COLS,)).astype(np.float32)
    out = buf.masked_matvec(t, x, np.ones(_ROWS, dtype=bool))
    assert np.array_equal(out, _rowwise_ref(dense, x, np.ones(_ROWS, bool), t.pages))
    assert np.allclose(out, dense @ x, rtol=1e-5, atol=1e-7)
    buf.release(t)


def test_eviction_is_loud_and_refaults(store):
    # capacity fits one tensor comfortably but not two
    one = _align_up(_ROWS * _COLS * 4)
    buf = PagerBackedBuffer(store, capacity_bytes=one + 4096)
    t0, t1 = buf.alloc("layer.0.w"), buf.alloc("layer.1.w")
    assert buf.bind(t0) == 2
    assert buf.bind(t1) == 2
    assert buf.stats["evictions"] > 0  # MEASURED
    assert not buf.is_resident(t0)  # t0's pages evicted: marked not-resident
    assert buf.is_resident(t1)
    assert buf.bind(t0) == 2  # re-fault on demand: loud, counted
    buf.release(t0)
    buf.release(t1)


def _align_up(n, a=64):
    return (n + a - 1) // a * a


def test_touch_protects_from_eviction(store):
    # capacity holds two tensors: bind a, b; touch a (MRU); bind c ->
    # LRU victim must be b, and a must survive.
    two = 2 * _align_up(_ROWS * _COLS * 4)
    buf = PagerBackedBuffer(store, capacity_bytes=two + 4096)
    a, b, c = (buf.alloc(n) for n in _names())
    assert buf.bind(a) == 2
    assert buf.bind(b) == 2
    buf.touch(a)  # a is now MRU
    assert buf.bind(c) == 2
    assert buf.stats["evictions"] > 0  # MEASURED
    assert buf.is_resident(a), "touched tensor must survive eviction"
    assert not buf.is_resident(b), "LRU tensor must be the victim"
    assert buf.is_resident(c)
    for t in (a, b, c):
        buf.release(t)


def test_capacity_error_is_loud(store):
    buf = PagerBackedBuffer(store, capacity_bytes=1024)
    with pytest.raises(BufferCapacityError):
        buf.alloc("layer.0.w")


def test_raw_pages_rejected(tmp_path):
    p = str(tmp_path / "raw.lac")
    with LacWriter(p, page_size=512, precision_policy="exact") as w:
        w.add_raw_bytes("blob", b"\x00" * 1000)
    with PagedWeightStore(p, hot_budget_bytes=10**9) as s:
        buf = PagerBackedBuffer(s, capacity_bytes=1 << 20)
        with pytest.raises(BufferError):
            buf.alloc("blob")


def _pages_of(store, name):
    tid = store.names.index(name)
    return [
        (e["page_id"], e["row_start"], e["row_end"])
        for e in store.page_table()
        if e["tensor_id"] == tid
    ]


def test_tiled_exact_matches_dense(store):
    buf = PagerBackedBuffer(store, capacity_bytes=_align_up(_ROWS * _COLS * 4) + 4096)
    big = PagerBackedBuffer(store, capacity_bytes=1 << 20)
    rng = np.random.default_rng(9)
    x = rng.standard_normal((_COLS,)).astype(np.float32)
    dense = [_dense(big, n) for n in _names()]
    outs, report = tiled_exact_matvec(buf, _names(), x)
    assert report["guarantee_held"] is True
    assert report["faults"] == 2 * _LAYERS  # MEASURED: every tile faulted once
    for out, d, n in zip(outs, dense, _names()):
        ref = _rowwise_ref(d, x, np.ones(_ROWS, bool), _pages_of(store, n))
        assert np.array_equal(out, ref)  # bit-identical per tile
        assert np.allclose(out, d @ x, rtol=1e-5, atol=1e-7)


def test_tiled_sparse_mask_saves_faults_keeps_exactness(store):
    # Oracle union mask selects 10 rows, all inside each layer's SECOND page:
    # only 3 pages fault (vs 6 for the full mask) — MEASURED — while the
    # selected rows stay bit-identical to dense. This is the row-masked
    # matmul's whole point.
    one = _align_up(_ROWS * _COLS * 4)
    rng = np.random.default_rng(13)
    x = rng.standard_normal((_COLS,)).astype(np.float32)
    masks = []
    for _ in range(_LAYERS):
        m = np.zeros(_ROWS, dtype=bool)
        m[62:72] = True  # rows 62..71 live in page 2 (page 1 covers 0..61)
        masks.append(m)

    buf = PagerBackedBuffer(store, capacity_bytes=one + 4096)
    big = PagerBackedBuffer(store, capacity_bytes=1 << 20)
    dense = [_dense(big, n) for n in _names()]
    outs, report = tiled_exact_matvec(
        buf, _names(), x, row_mask_fn=lambda i: masks[i]
    )
    assert report["guarantee_held"] is True
    assert report["faults"] == _LAYERS  # MEASURED: one page per tile, not two
    for out, d, m, n in zip(outs, dense, masks, _names()):
        ref = _rowwise_ref(d, x, m, _pages_of(store, n))
        assert np.array_equal(out, ref)
        assert np.all(out[~m] == 0.0)
        assert np.allclose(out[m], d[m] @ x, rtol=1e-5, atol=1e-7)
