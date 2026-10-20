"""Pager fault injection: corrupt/missing/truncated store content must
fail LOUD (a clear exception at open or at fault time) — never a hang,
never a silent short or wrong-shaped page.

Note the one honest exception, not tested as a failure because the .lac
format carries no per-page checksum (that is Day-7 format work): flipping
a payload *data* byte changes decoded values without detection. Header,
table, length, and precision corruption — the structural faults — are
all detected, and the warm-start cache (tests/test_warmstart.py) detects
payload changes across runs via CRC32.
"""

import struct

import numpy as np
import pytest

from lacuna.pager import (
    ENTRY_SIZE,
    HEADER_SIZE,
    PAGE_HDR_SIZE,
    LacWriter,
    PagedWeightStore,
)
from lacuna.quant import Q2, Q4

LOUD = (ValueError, KeyError, IndexError, struct.error, OSError)


@pytest.fixture()
def lac_path(tmp_path):
    p = str(tmp_path / "w.lac")
    rng = np.random.default_rng(5)
    with LacWriter(p, page_size=4096) as w:
        w.add_tensor("a", rng.standard_normal((32, 16)).astype(np.float32),
                     layer=0, precision=Q4)
        w.add_tensor("b", rng.standard_normal((48, 16)).astype(np.float32),
                     layer=0, precision=Q2)
    return p


def _mutate(path, offset, data: bytes):
    with open(path, "r+b") as f:
        f.seek(offset)
        f.write(data)


def test_bad_magic_rejected(lac_path):
    _mutate(lac_path, 0, b"XXXX")
    with pytest.raises(ValueError, match="bad magic"):
        PagedWeightStore(lac_path)


def test_unsupported_version_rejected(lac_path):
    _mutate(lac_path, 4, struct.pack("<H", 99))
    with pytest.raises(ValueError, match="version"):
        PagedWeightStore(lac_path)


def test_unknown_policy_id_rejected(lac_path):
    # v2 header layout: magic(4) ver(2) page_size(4) n_pages(4) table_off(8)
    # then the policy byte at offset 22
    _mutate(lac_path, 22, b"\x07")
    with pytest.raises(ValueError, match="precision policy"):
        PagedWeightStore(lac_path)


def test_empty_file_rejected(tmp_path):
    p = tmp_path / "empty.lac"
    p.write_bytes(b"")
    with pytest.raises(LOUD):
        PagedWeightStore(str(p))


def test_random_garbage_rejected(tmp_path):
    p = tmp_path / "garbage.lac"
    p.write_bytes(np.random.default_rng(1).bytes(4096))
    with pytest.raises(LOUD):
        PagedWeightStore(str(p))


def test_truncated_page_table_rejected(lac_path, tmp_path):
    data = open(lac_path, "rb").read()
    # table lives at the end region; cutting the file in half removes it
    p = tmp_path / "cut.lac"
    p.write_bytes(data[: len(data) // 2])
    with pytest.raises(LOUD):
        PagedWeightStore(str(p))


def test_truncated_page_payload_fails_loud(lac_path, tmp_path):
    # keep header + table valid but cut the file inside the first page by
    # pointing the second page's offset beyond EOF: simplest structural
    # equivalent is shrinking the file to just past page 0's header
    data = open(lac_path, "rb").read()
    p = tmp_path / "short.lac"
    p.write_bytes(data[: HEADER_SIZE + PAGE_HDR_SIZE + 4])
    try:
        s = PagedWeightStore(str(p))
    except LOUD:
        return  # rejected at open: loud, correct
    with pytest.raises(LOUD):
        s.get_page(0)
    s.close()


def test_page_index_out_of_range_fails_loud(lac_path):
    with PagedWeightStore(lac_path, hot_budget_bytes=10**9) as s:
        with pytest.raises(LOUD):
            s.get_page(s.n_pages + 10)


def test_page_header_precision_flip_fails_loud(lac_path):
    with PagedWeightStore(lac_path, hot_budget_bytes=10**9) as s:
        off = s.page_table()[0]["offset"]
    # first header byte is the precision: Q4(4) -> Q2(2) disagrees with table
    _mutate(lac_path, off, bytes([Q2]))
    with PagedWeightStore(lac_path, hot_budget_bytes=10**9) as s:
        with pytest.raises(ValueError, match="precision mismatch"):
            s.get_page(0)
        with pytest.raises(ValueError, match="precision mismatch"):
            s.page_shape(0)


def test_page_offset_beyond_eof_fails_loud(lac_path):
    with PagedWeightStore(lac_path, hot_budget_bytes=10**9) as s:
        n = s.n_pages
    # table offset lives in the header at bytes 14..22; the file_offset
    # field is the last 8 bytes of each table entry
    with open(lac_path, "rb") as f:
        hdr = f.read(HEADER_SIZE)
    (tbl,) = struct.unpack("<Q", hdr[14:22])
    _mutate(lac_path, tbl + ENTRY_SIZE - 8, struct.pack("<Q", 10**12))
    with PagedWeightStore(lac_path, hot_budget_bytes=10**9) as s:
        assert s.n_pages == n
        with pytest.raises(LOUD):
            s.get_page(0)


def test_prefetch_of_corrupt_page_fails_loud(lac_path):
    with PagedWeightStore(lac_path, hot_budget_bytes=10**9) as s:
        off = s.page_table()[0]["offset"]
    _mutate(lac_path, off, bytes([Q2]))  # precision mismatch (see above)
    with PagedWeightStore(lac_path, hot_budget_bytes=10**9) as s:
        with pytest.raises(ValueError, match="precision mismatch"):
            s.prefetch([0])
