"""Precision policy tests.

- "exact": bit-identical roundtrip to source bytes; policy recorded in the
  file header and exposed by the reader; v1 (pre-policy) files read as
  "extreme".
- "extreme": documented error bounds per tier (ordering + finiteness).
"""

import struct

import numpy as np
import pytest

from lacuna import quant as q
from lacuna.pager import (
    ENTRY_FMT,
    ENTRY_SIZE,
    HEADER_SIZE_V1,
    MAGIC,
    LacWriter,
    PagedWeightStore,
)


def test_exact_policy_bit_identical(tmp_path):
    rng = np.random.default_rng(5)
    arr = rng.standard_normal((32, 16)).astype(np.float32)
    p = str(tmp_path / "m.lac")
    with LacWriter(p, page_size=4096, precision_policy="exact") as w:
        w.add_raw_bytes("t", arr.tobytes())
    with PagedWeightStore(p, hot_budget_bytes=10**9) as s:
        assert s.precision_policy == "exact"
        back = b"".join(s.get_page(i).tobytes() for i in range(s.n_pages))
    assert back == arr.tobytes()


def test_invalid_policy_rejected(tmp_path):
    with pytest.raises(ValueError):
        LacWriter(str(tmp_path / "x.lac"), precision_policy="lossy-ish")
    assert q.validate_policy("exact") == "exact"
    with pytest.raises(ValueError):
        q.validate_policy("nope")


def test_extreme_policy_documented_bounds():
    rng = np.random.default_rng(6)
    x = rng.standard_normal((16, 16)).astype(np.float32)
    e4 = q.roundtrip_max_error(x, q.Q4)
    e2 = q.roundtrip_max_error(x, q.Q2)
    et = q.roundtrip_max_error(x, q.TERNARY)
    # documented ordering: finer tiers err less; every tier finite.
    assert np.isfinite(e4) and np.isfinite(e2) and np.isfinite(et)
    assert e4 <= e2 <= et
    # zero tensor is exact in every tier (no division-by-zero artifacts)
    z = np.zeros((8, 8), dtype=np.float32)
    for prec in (q.Q4, q.Q2, q.TERNARY):
        assert q.roundtrip_max_error(z, prec) == 0.0


def _write_v1_file(path):
    """Hand-roll a v1 .lac file (pre-policy header) to prove back-compat."""
    page = np.arange(16, dtype=np.float32).tobytes()
    with open(path, "wb") as f:
        f.write(b"\x00" * HEADER_SIZE_V1)  # placeholder
        off = f.tell()
        f.write(struct.pack("<BIIf", q.RAW, 0, len(page), 1.0))
        f.write(page)
        table_off = f.tell()
        f.write(struct.pack(ENTRY_FMT, 0, 0, 0, 0, q.RAW, 0, off))
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<I", 1) + b"t")
        f.seek(0)
        f.write(struct.pack("<4sHIIQ", MAGIC, 1, 4096, 1, table_off))


def test_v1_file_reads_as_extreme(tmp_path):
    p = str(tmp_path / "old.lac")
    _write_v1_file(p)
    with PagedWeightStore(p, hot_budget_bytes=10**9) as s:
        # v1 predates policies and always used adaptive tiering
        assert s.precision_policy == "extreme"
        assert s.n_pages == 1
        assert s.get_page(0).tobytes() == np.arange(16, dtype=np.float32).tobytes()
