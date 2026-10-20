"""NVMe paged weight store + hot-set residency (M2/M3) — numpy + stdlib.

``.lac`` file layout (little-endian)::

    header:  magic b'LAC1' | u16 version | u32 page_size | u32 n_pages |
             u64 table_offset | u8 precision_policy | 3 reserved bytes   (v2)
    pages:   page_header + packed payload, repeated
    table:   n_pages entries of
             u32 tensor_id | u16 layer | u32 row_start | u32 row_end |
             u8 precision | u8 reserved | u64 file_offset
    names:   u32 n_names, then for each: u32 len + bytes (tensor id -> name)

precision_policy ids: 0 = legacy/unspecified, 1 = exact, 2 = extreme.
v1 files (written before policies existed) have no policy byte and are read as
"extreme" — v1 always used adaptive tiering.

Each page payload starts with its own header so pages are self-describing::

    page_header: u8 precision | u32 rows | u32 cols | f32 scale

then the packed bytes for the codec (quant.py). RAW pages (precision 0) store
opaque bytes with rows/cols/scale ignored by the reader (returned as uint8).

Disk I/O paths by platform (Linux and Windows 11 only):
- Linux: ``io_uring`` fast path when a python binding is importable (guarded
  import, silent fallback); otherwise ThreadPoolExecutor + ``os.pread``.
- Windows 11: ThreadPoolExecutor + regular reads (``os.pread`` is unavailable
  on Windows; the reader uses plain ``read`` on a per-thread file handle).
No io_uring references exist on the Windows path.
"""

from __future__ import annotations

import os
import struct
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import quant as _q

MAGIC = b"LAC1"
VERSION = 2
# v1 header (no policy byte). Kept for reading files written before policies.
HEADER_FMT_V1 = "<4sHIIQ"
HEADER_SIZE_V1 = struct.calcsize(HEADER_FMT_V1)
# v2 header: magic | version | page_size | n_pages | table_offset | policy | pad.
HEADER_FMT = "<4sHIIQB3x"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
_POLICY_IDS = {"exact": 1, "extreme": 2}
_POLICY_NAMES = {1: "exact", 2: "extreme"}
ENTRY_FMT = "<IHI IB B Q".replace(" ", "")
ENTRY_SIZE = struct.calcsize(ENTRY_FMT)
PAGE_HDR_FMT = "<BIIf"
PAGE_HDR_SIZE = struct.calcsize(PAGE_HDR_FMT)
# Packed bytes per value for each quantized codec (RAW pages are opaque).
_PAGE_BPR = {_q.Q4: 0.5, _q.Q2: 0.25, _q.TERNARY: 0.25}

import sys as _sys

# io_uring is Linux-only. Guarded import; silent fallback everywhere else.
if _sys.platform.startswith("linux"):
    try:  # optional: faster async disk reads when a binding exists
        import uring  # type: ignore

        _HAS_URING = True
    except ImportError:
        uring = None  # type: ignore
        _HAS_URING = False
else:
    uring = None  # type: ignore
    _HAS_URING = False

_ON_WINDOWS = _sys.platform.startswith("win32")


# ------------------------------------------------------------- writer ---


class LacWriter:
    """Build a .lac file from 2-D float32 tensors, split into pages.

    ``precision_policy`` is recorded in the file header ("exact" or "extreme");
    it is a label describing how the caller chose per-tensor precision — the
    writer itself just stores what it is given.
    """

    def __init__(self, path: str, page_size: int = 2 * 1024 * 1024,
                 precision_policy: str = "exact"):
        self.path = path
        self.page_size = int(page_size)
        self.precision_policy = _q.validate_policy(precision_policy)
        self._f = open(path, "wb")
        self._f.write(b"\x00" * HEADER_SIZE)  # placeholder, rewritten on close
        self._entries: list[tuple] = []
        self._names: list[str] = []

    def add_tensor(
        self, name: str, arr: np.ndarray, layer: int = 0, precision: int = _q.Q2
    ) -> int:
        """Add one 2-D tensor; returns tensor_id. Rows are packed into pages."""
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError("LacWriter.add_tensor needs a 2-D array")
        tensor_id = len(self._names)
        self._names.append(name)
        rows, cols = arr.shape

        if precision == _q.RAW:
            payload = np.ascontiguousarray(arr).view(np.uint8)
            packed_all, scale, _ = payload, np.float32(1.0), arr.shape
            per_row = payload.nbytes // max(rows, 1)
            pack_fn = None
        else:
            pack_fn = _q.quantize
            # bytes per row estimate from a small probe
            probe = arr[: min(rows, 8)]
            packed_probe, _, _ = pack_fn(probe, precision)
            per_row = max(1, packed_probe.nbytes // max(probe.shape[0], 1))

        rows_per_page = max(1, (self.page_size - PAGE_HDR_SIZE) // per_row)
        for start in range(0, rows, rows_per_page):
            chunk = arr[start : start + rows_per_page]
            if pack_fn is None:
                packed = np.ascontiguousarray(chunk).view(np.uint8).reshape(-1)
                scale = np.float32(1.0)
            else:
                packed, scale, _ = pack_fn(chunk, precision)
            off = self._f.tell()
            self._f.write(struct.pack(PAGE_HDR_FMT, precision, chunk.shape[0], cols, float(scale)))
            self._f.write(packed.tobytes())
            self._entries.append(
                (tensor_id, int(layer), start, start + chunk.shape[0], precision, 0, off)
            )
        return tensor_id

    def add_raw_bytes(self, name: str, blob: bytes, layer: int = 0) -> int:
        """Store an opaque byte blob as RAW pages (e.g. pre-quantized GGUF blocks)."""
        tensor_id = len(self._names)
        self._names.append(name)
        per_page = self.page_size - PAGE_HDR_SIZE
        for start in range(0, len(blob), per_page):
            chunk = blob[start : start + per_page]
            off = self._f.tell()
            self._f.write(struct.pack(PAGE_HDR_FMT, _q.RAW, 0, len(chunk), 1.0))
            self._f.write(chunk)
            self._entries.append((tensor_id, int(layer), 0, 0, _q.RAW, 0, off))
        return tensor_id

    def close(self) -> None:
        table_offset = self._f.tell()
        for e in self._entries:
            self._f.write(struct.pack(ENTRY_FMT, *e))
        # name table
        self._f.write(struct.pack("<I", len(self._names)))
        for n in self._names:
            b = n.encode("utf-8")
            self._f.write(struct.pack("<I", len(b)) + b)
        self._f.seek(0)
        self._f.write(
            struct.pack(
                HEADER_FMT,
                MAGIC,
                VERSION,
                self.page_size,
                len(self._entries),
                table_offset,
                _POLICY_IDS[self.precision_policy],
            )
        )
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


# ------------------------------------------------------------- reader ---


class PagedWeightStore:
    """Reader with LRU hot-set, EWMA frequency counters, async prefetch."""

    def __init__(self, path: str, hot_budget_bytes: int = 6 * 1024**3, prefetch_workers: int = 4):
        self.path = path
        self.hot_budget = int(hot_budget_bytes)
        self._fd = os.open(path, os.O_RDONLY)
        with open(path, "rb") as f:
            magic, ver = struct.unpack("<4sH", f.read(6))
            if magic != MAGIC:
                raise ValueError(f"not a .lac file (bad magic): {path}")
            if ver == 1:
                # v1 predates precision policies and always used adaptive
                # tiering -> read as "extreme".
                self.page_size, n_pages, table_off = struct.unpack(
                    "<IIQ", f.read(HEADER_SIZE_V1 - 6)
                )
                self.precision_policy = "extreme"
            elif ver == VERSION:
                self.page_size, n_pages, table_off, pid = struct.unpack(
                    "<IIQB3x", f.read(HEADER_SIZE - 6)
                )
                if pid not in _POLICY_NAMES:
                    raise ValueError(f"unknown precision policy id {pid} in {path}")
                self.precision_policy = _POLICY_NAMES[pid]
            else:
                raise ValueError(f"unsupported .lac version {ver}")
        self.n_pages = n_pages
        self._entries: list[dict] = []
        with open(path, "rb") as f:
            f.seek(table_off)
            # One bulk read for the whole table instead of one syscall per
            # page; struct.iter_unpack parses it with identical results.
            # A short read fails loud here (the old loop raised struct.error).
            table_blob = f.read(n_pages * ENTRY_SIZE)
            if len(table_blob) != n_pages * ENTRY_SIZE:
                raise ValueError(
                    f"truncated page table in {path}: expected "
                    f"{n_pages * ENTRY_SIZE} bytes, got {len(table_blob)}"
                )
            for tid, layer, rs, re_, prec, _res, off in struct.iter_unpack(
                ENTRY_FMT, table_blob
            ):
                self._entries.append(
                    {
                        "tensor_id": tid,
                        "layer": layer,
                        "row_start": rs,
                        "row_end": re_,
                        "precision": prec,
                        "offset": off,
                    }
                )
            (n_names,) = struct.unpack("<I", f.read(4))
            self.names = []
            for _ in range(n_names):
                (ln,) = struct.unpack("<I", f.read(4))
                self.names.append(f.read(ln).decode("utf-8"))

        # Indexes, built once: entries are immutable after init, so the public
        # page table and per-tensor lookups are O(1)/O(tensor pages) instead
        # of O(n_pages) scans per call (buffer allocs once per layer).
        for i, e in enumerate(self._entries):
            e["page_id"] = i
        self._page_table: list[dict] = self._entries  # shared, read-only
        self._name_to_id = {n: i for i, n in enumerate(self.names)}
        self._tensor_index: dict[int, list[dict]] = {}
        for e in self._page_table:
            self._tensor_index.setdefault(e["tensor_id"], []).append(e)

        if _ON_WINDOWS:
            self._fd = -1  # os.pread unavailable; reads go through _win_f
            self._win_f = open(path, "rb")
            self._win_lock = threading.Lock()
        self._lock = threading.Lock()
        self._cache: "OrderedDict[int, tuple[np.ndarray, int]]" = OrderedDict()
        self._cached_bytes = 0
        self._freq: dict[int, float] = {}
        self._pool = ThreadPoolExecutor(max_workers=prefetch_workers)
        self.stats = {"hits": 0, "faults": 0, "bytes_read": 0, "prefetches": 0}

    # -- low-level ------------------------------------------------

    def _read_at(self, offset: int, n: int) -> bytes:
        # Linux: io_uring when a binding is present, else os.pread (thread-safe).
        # Windows 11: os.pread does not exist; use a per-thread file handle
        # with seek+read serialized by a lock.
        if _ON_WINDOWS:
            with self._win_lock:
                self._win_f.seek(offset)
                return self._win_f.read(n)
        return os.pread(self._fd, n, offset)

    @staticmethod
    def _payload_len(prec: int, rows: int, cols: int) -> int:
        """Payload byte length declared by a page header."""
        if prec == _q.RAW:
            return cols
        return int(rows * cols * _PAGE_BPR[prec] + 3)

    def page_stored_bytes(self, idx: int) -> bytes:
        """The exact on-disk bytes of page ``idx`` (header + payload).

        Two exact-length reads — header first, then precisely the payload it
        declares — instead of the old single generous ``page_size`` read.
        Measured (round-2 sweep): RAW pages fault ~30x faster (a 256 KiB
        payload no longer pulls 2 MiB through the page cache), full 2 MiB
        pages ~1.06x, small pages neutral; decoded output bit-identical.

        Raises ValueError on a truncated header or payload: a short page is
        corruption and fails loud here, never silently decodes short.
        """
        e = self._entries[idx]
        hdr = self._read_at(e["offset"], PAGE_HDR_SIZE)
        if len(hdr) < PAGE_HDR_SIZE:
            raise ValueError(
                f"page {idx}: truncated page header in {self.path} "
                f"({len(hdr)} of {PAGE_HDR_SIZE} bytes)"
            )
        prec, rows, cols, _scale = struct.unpack(PAGE_HDR_FMT, hdr)
        n = self._payload_len(prec, rows, cols)
        payload = self._read_at(e["offset"] + PAGE_HDR_SIZE, n) if n else b""
        if len(payload) < n:
            raise ValueError(
                f"page {idx}: truncated page payload in {self.path} "
                f"({len(payload)} of {n} bytes)"
            )
        return hdr + payload

    def _decode_page(self, raw: bytes) -> tuple[np.ndarray, int]:
        """Decode stored page bytes -> (array, total stored length)."""
        prec, rows, cols, scale = struct.unpack(PAGE_HDR_FMT, raw[:PAGE_HDR_SIZE])
        if prec == _q.RAW:
            arr = np.frombuffer(
                raw[PAGE_HDR_SIZE : PAGE_HDR_SIZE + cols], dtype=np.uint8
            ).copy()
            return arr, PAGE_HDR_SIZE + cols
        nbytes = self._payload_len(prec, rows, cols)
        # No .copy(): dequantize only reads its input and returns a fresh
        # array, so the transient view is safe (and cheaper) here. The RAW
        # path above keeps its copy — that array IS the cached page.
        packed = np.frombuffer(
            raw[PAGE_HDR_SIZE : PAGE_HDR_SIZE + nbytes], dtype=np.uint8
        )
        arr = _q.dequantize(packed, np.float32(scale), (rows, cols), prec)
        return arr, PAGE_HDR_SIZE + nbytes

    def _load_page(self, idx: int) -> tuple[np.ndarray, int]:
        return self._decode_page(self.page_stored_bytes(idx))

    def _insert(self, idx: int, arr: np.ndarray, nbytes: int) -> None:
        with self._lock:
            if idx in self._cache:
                self._cache.move_to_end(idx)
                return
            # evict LRU until the newcomer fits (hard budget — never exceeds)
            while self._cache and self._cached_bytes + nbytes > self.hot_budget:
                _, (_, sz) = self._cache.popitem(last=False)
                self._cached_bytes -= sz
            if nbytes <= self.hot_budget:
                self._cache[idx] = (arr, nbytes)
                self._cached_bytes += nbytes

    # -- public ---------------------------------------------------

    def get_page(self, idx: int) -> np.ndarray:
        """Return page idx, faulting from NVMe on miss. Updates EWMA frequency."""
        with self._lock:
            hit = idx in self._cache
            if hit:
                arr, _ = self._cache[idx]
                self._cache.move_to_end(idx)
        if hit:
            self.stats["hits"] += 1
        else:
            arr, nbytes = self._load_page(idx)
            self.stats["faults"] += 1
            self.stats["bytes_read"] += nbytes
            self._insert(idx, arr, nbytes)
        freq = self._freq  # local bind: one attribute lookup per call
        freq[idx] = 0.9 * freq.get(idx, 0.0) + 0.1
        return arr

    def is_resident(self, idx: int) -> bool:
        """True if page idx is currently in the hot-set (no I/O performed)."""
        with self._lock:
            return idx in self._cache

    def are_resident(self, indices) -> bool:
        """True if every listed page is in the hot-set. One lock acquisition."""
        with self._lock:
            cache = self._cache
            return all(i in cache for i in indices)

    def ensure_resident(self, page_ids) -> int:
        """Blocking: fault every missing page into the hot-set (LRU eviction
        per the hard byte budget). Returns the number of pages faulted.

        This is the exactness primitive: call it with the FULL required page
        set before compute runs, and compute provably never observes a missing
        page — regardless of what the oracle predicted. A bad prediction costs
        faults (time), never correctness.

        Works identically on the live path (real NVMe reads) and in tests.
        Callers must size ``hot_budget_bytes >= page_size``; a page larger
        than the whole budget cannot be held, in which case it is re-faulted
        on every call and ``are_resident`` stays False (loud, not silent).
        """
        # De-dupe once, preserving first-seen (fault) order; then detect the
        # miss set under a single lock acquisition instead of one per page.
        # get_page re-checks under its own lock, so the fault semantics are
        # identical to the per-page version.
        seen = set()
        uniq: list[int] = []
        for i in page_ids:
            if i not in seen:
                seen.add(i)
                uniq.append(i)
        with self._lock:
            cache = self._cache
            missing = [i for i in uniq if i not in cache]
        faults = 0
        for i in missing:
            self.get_page(i)  # synchronous fault; updates stats
            faults += 1
        return faults

    def prefetch(self, indices) -> None:
        """Async prefetch: submit reads now, insert into hot-set on completion."""
        futs = []
        with self._lock:
            wanted = [i for i in indices if i not in self._cache]
        for i in wanted:
            futs.append(self._pool.submit(self._load_page, i))
        self.stats["prefetches"] += len(futs)
        for i, fut in zip(wanted, futs):
            try:
                arr, nbytes = fut.result()
            except OSError:
                continue
            self.stats["bytes_read"] += nbytes
            self._insert(i, arr, nbytes)

    def hot_pages(self) -> list[int]:
        with self._lock:
            return list(self._cache.keys())

    def page_table(self) -> list[dict]:
        """Public read of the .lac page table: one dict per page with
        ``page_id``, ``tensor_id``, ``layer``, ``row_start``, ``row_end``,
        ``precision`` and file ``offset``. Used by ``lacuna.buffer`` to map
        tensors to their pages without touching pager internals.

        The returned list is shared and read-only (built once at init) — do
        not mutate it. Previously this rebuilt every dict on every call.
        """
        return self._page_table

    def tensor_id(self, name: str) -> int | None:
        """Tensor id for a tensor name, or None if unknown. O(1)."""
        return self._name_to_id.get(name)

    def tensor_pages(self, tensor_id: int) -> list[dict]:
        """Page-table rows for one tensor, in row order. Shared, read-only."""
        return self._tensor_index.get(tensor_id, [])

    def page_shape(self, page_id: int) -> tuple[int, int]:
        """(rows, cols) of a page from its on-disk header. No payload read."""
        e = self._entries[page_id]
        prec, rows, cols, _scale = struct.unpack(
            PAGE_HDR_FMT, self._read_at(e["offset"], PAGE_HDR_SIZE)
        )
        if prec != e["precision"]:
            raise ValueError(f"page {page_id}: header/table precision mismatch")
        return rows, cols

    def frequency(self, idx: int) -> float:
        return self._freq.get(idx, 0.0)

    @property
    def hit_rate(self) -> float:
        h, f = self.stats["hits"], self.stats["faults"]
        return h / (h + f) if (h + f) else 0.0

    def close(self) -> None:
        self._pool.shutdown(wait=True)
        if _ON_WINDOWS:
            self._win_f.close()
        else:
            os.close(self._fd)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
