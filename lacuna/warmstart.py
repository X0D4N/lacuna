"""Warm-start persistent hot-set cache — the "feels instant" mechanism.

A :class:`~lacuna.pager.PagedWeightStore` learns which pages are hot while
it runs (LRU + EWMA frequency). Without persistence that knowledge dies
with the process: every cold start re-faults the working set page by page.
This module snapshots the hot-set (page ids + EWMA frequencies + a CRC32 of
each page's stored bytes) to a small sidecar file on shutdown and restores
it on the next open, so repeat runs start with the working set already
resident — the first token no longer pays the discovery faults.

Safety (zero-silent-error standard)
-----------------------------------
The cache is an optimization layer only; it can degrade to a cold start,
never to wrong data:

- **Fingerprint**: the snapshot records the store file's size, mtime (ns),
  page count and page size. Any mismatch -> the whole snapshot is ignored.
- **Per-page CRC32**: at restore, every page's stored bytes are re-read
  from the ``.lac`` file and re-checksummed (CRC32 at ~3.2 GB/s, measured —
  on par with NVMe read speed, so validation costs no extra pass over the
  data beyond the read itself). A mismatch invalidates just that page; it
  faults from the store on demand, exactly like a cold start. Cached
  *bytes* are never trusted — pages are always decoded from the store's
  current bytes after the checksum proves they are the bytes the snapshot
  described.
- **Structural cross-check**: the page header must agree with the store's
  page table (``page_shape`` raises otherwise) before a page is inserted.
- **Total functions**: corrupt, truncated, or foreign cache files return a
  zero report; ``save_hotset``/``restore_hotset`` never raise. A broken
  cache costs a cold start, nothing else.

CRC32 is an integrity check against corruption (bit rot, torn writes,
stale snapshots), not a security MAC — the threat model is a local file
owned by the same user (see SECURITY.md).

File layout (little-endian)::

    header:  magic b'LACW' | u16 version | u32 n_entries |
             u64 store_size | u64 store_mtime_ns | u32 n_pages | u32 page_size
    entries: n_entries * (u32 page_id | u32 crc32 | f32 ewma_freq)

Stdlib + numpy only (via the store). Linux + Windows 11.
"""

from __future__ import annotations

import os
import struct
import zlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .pager import PagedWeightStore

MAGIC = b"LACW"
VERSION = 1
_HDR_FMT = "<4sHIQQII"
_HDR_SIZE = struct.calcsize(_HDR_FMT)
_ENTRY_FMT = "<IIf"
_ENTRY_SIZE = struct.calcsize(_ENTRY_FMT)


def default_hotset_path(store_path: str) -> str:
    """Sidecar path for a store: ``model.lac`` -> ``model.lac.hotset``."""
    return store_path + ".hotset"


def _fingerprint(store: "PagedWeightStore") -> tuple[int, int, int, int]:
    st = os.stat(store.path)
    return (st.st_size, st.st_mtime_ns, store.n_pages, store.page_size)


def save_hotset(store: "PagedWeightStore", cache_path: str) -> dict:
    """Snapshot the store's current hot-set to ``cache_path`` (atomic).

    Returns ``{"saved": n, "path": cache_path}`` on success, or
    ``{"saved": 0, "path": cache_path, "reason": ...}`` on any failure —
    never raises (a cache that cannot be written just means the next run
    starts cold).
    """
    try:
        size, mtime_ns, n_pages, page_size = _fingerprint(store)
        entries = []
        for idx in sorted(store.hot_pages()):
            raw = store.page_stored_bytes(idx)
            entries.append((idx, zlib.crc32(raw), float(store.frequency(idx))))
        blob = bytearray()
        blob += struct.pack(
            _HDR_FMT, MAGIC, VERSION, len(entries), size, mtime_ns,
            n_pages, page_size,
        )
        for idx, crc, freq in entries:
            blob += struct.pack(_ENTRY_FMT, idx, crc, freq)
        tmp = cache_path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(blob)
        os.replace(tmp, cache_path)  # atomic: readers see all or nothing
        return {"saved": len(entries), "path": cache_path}
    except (OSError, ValueError, struct.error) as e:
        return {"saved": 0, "path": cache_path, "reason": f"{type(e).__name__}: {e}"}


def restore_hotset(store: "PagedWeightStore", cache_path: str) -> dict:
    """Restore a saved hot-set into ``store``, validating every page.

    Returns ``{"restored": n, "invalidated": m, "reason": ...}``. Never
    raises: any problem — missing file, bad magic, fingerprint mismatch,
    truncated entries, a page whose bytes no longer match its checksum —
    degrades that page (or the whole snapshot) to a normal cold fault.
    """
    report = {"restored": 0, "invalidated": 0, "reason": "ok"}
    try:
        with open(cache_path, "rb") as f:
            blob = f.read()
    except OSError:
        report["reason"] = "no cache file"
        return report
    try:
        if len(blob) < _HDR_SIZE:
            report["reason"] = "cache file truncated (header)"
            return report
        magic, ver, n_entries, size, mtime_ns, n_pages, page_size = struct.unpack(
            _HDR_FMT, blob[:_HDR_SIZE]
        )
        if magic != MAGIC:
            report["reason"] = "not a hot-set cache (bad magic)"
            return report
        if ver != VERSION:
            report["reason"] = f"unsupported hot-set cache version {ver}"
            return report
        if len(blob) != _HDR_SIZE + n_entries * _ENTRY_SIZE:
            report["reason"] = "cache file truncated (entries)"
            return report
        if (size, mtime_ns, n_pages, page_size) != _fingerprint(store):
            report["reason"] = "store changed since snapshot (fingerprint mismatch)"
            return report
        entries = [
            struct.unpack_from(_ENTRY_FMT, blob, _HDR_SIZE + i * _ENTRY_SIZE)
            for i in range(n_entries)
        ]
    except (struct.error, OSError, ValueError) as e:
        report["reason"] = f"unreadable cache: {type(e).__name__}: {e}"
        return report

    # Coldest first: the pager's LRU eviction drops the earliest-inserted
    # pages when the budget fills, so inserting in ascending frequency
    # order leaves the HOTTEST pages resident at the end.
    for idx, crc, freq in sorted(entries, key=lambda e: e[2]):
        try:
            if idx >= store.n_pages:
                report["invalidated"] += 1
                continue
            raw = store.page_stored_bytes(idx)
            if zlib.crc32(raw) != crc:
                # The store's bytes changed under a preserved fingerprint
                # (or the page rotted): never trust the snapshot — the
                # page faults from the store on demand instead.
                report["invalidated"] += 1
                continue
            store.page_shape(idx)  # header must agree with the page table
            arr, nbytes = store._decode_page(raw)
        except (ValueError, OSError, KeyError, struct.error):
            report["invalidated"] += 1
            continue
        store._insert(idx, arr, nbytes)
        store._freq[idx] = float(freq)
        report["restored"] += 1
    store.stats["warm_restored"] = report["restored"]
    store.stats["warm_invalidated"] = report["invalidated"]
    return report
