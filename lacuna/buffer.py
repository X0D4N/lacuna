"""Pager-backed buffer: the runnable contract prototype of Lacuna's ggml
backend-buffer seam (M2/M3 <-> compute).

What this is
------------
``integrations/llama.cpp/INTEGRATION.md`` section 1 designs a custom
``ggml_backend_buffer_type`` whose allocations are owned by the pager's
hot-set and whose tensors are backed by page ids. This module implements
THAT CONTRACT in Python + numpy so it runs on this box and is covered by
tests. The C++ translation is ``integrations/llama.cpp/sp_buffer.cpp``;
both implement the same operations:

- ``alloc``            register a tensor's page -> row geometry (models ggml
                       ``alloc_buffer`` + ``init_tensor``; no bytes move yet)
- ``bind`` / ``ensure_resident``
                       blocking fault of the tensor's pages into the arena
                       BEFORE compute (the exactness fence); returns the
                       fault count. LRU-evicts other pages when the arena is
                       full — eviction marks pages not-resident, loud, counted.
- ``touch``            LRU + EWMA update, no I/O (keeps pager coherent)
- ``is_resident``      measured, never assumed
- ``release``          drop a tensor's residency eagerly
- ``masked_matvec``    row-masked multiply: only the pages of SELECTED rows
                       are faulted; the result is bit-identical to the dense
                       result on those rows. This is the Python model of the
                       row-masked ``ggml_mul_mat`` from the integration doc.
- ``tiled_exact_matvec``
                       the production per-layer tiling loop from the
                       integration doc, runnable here: one layer resident at
                       a time, next-layer prefetch overlipped, exactness
                       measured per tile.

What is real vs what the C++ port changes
-----------------------------------------
REAL here (``tests/test_buffer.py``, all MEASURED): page -> byte mapping,
the blocking residency fence, LRU eviction semantics, row-masked exactness
(identical bytes through identical ops give identical bits — see the ~1 ULP
BLAS reduction-order caveat in ``masked_matvec``), and the tiling loop's
fault accounting.

NOT real here: the arena is a plain ``bytearray``, not pinned host memory
(the C++ port uses pinned memory; pinning affects transfer speed only, never
this contract). There is no ggml on this box — "compute" is numpy.

Stdlib + numpy only. Linux + Windows 11.
"""

from __future__ import annotations

import bisect
from collections import OrderedDict

import numpy as np

from . import quant as _q
from .pager import PagedWeightStore

_ALIGNMENT = 64  # same alignment the C++ buffer type advertises


class BufferCapacityError(MemoryError):
    """Raised when a tensor can never fit the buffer's arena (loud, not silent)."""


class BufferError(RuntimeError):
    """Raised for contract violations (e.g. binding opaque RAW pages)."""


def _align(n: int, a: int = _ALIGNMENT) -> int:
    return (n + a - 1) // a * a


class BufferTensor:
    """A tensor allocated from a :class:`PagerBackedBuffer`.

    ``pages`` is the page table slice for this tensor:
    ``[(page_id, row_start, row_end), ...]`` in row order. Arena offsets are
    assigned per page at bind time (pages are the unit of residency, exactly
    like the C++ side where a tensor is "backed by page ids"). Read the
    assembled bytes via ``PagerBackedBuffer.tensor_data`` once resident.
    """

    def __init__(self, name: str, shape: tuple[int, int], pages):
        self.name = name
        self.shape = shape
        self.dtype = np.float32
        self.pages = list(pages)  # [(page_id, row_start, row_end)]
        # Precomputed for bind_pages: rebuilding this set per call was pure
        # overhead on the hot path.
        self.page_id_set = frozenset(pid for pid, _, _ in self.pages)

    @property
    def page_ids(self) -> list[int]:
        return [pid for pid, _, _ in self.pages]

    @property
    def nbytes(self) -> int:
        rows, cols = self.shape
        return _align(rows * cols * 4)


class PagerBackedBuffer:
    """Python prototype of the ``sp_buffer`` ggml backend buffer type.

    The pager owns weight bytes on NVMe; the buffer owns the *resident* copy
    in its fixed-size arena and the page -> byte mapping ggml needs. The
    arena is the residency budget: binding a page whose bytes don't fit
    LRU-evicts other pages first (marked not-resident, counted). ``store``
    stays the I/O owner (faults, prefetch, its own hot-set); the buffer is
    the residency owner for bound pages — mirroring the C++ split where the
    buffer type's context tracks residency and the pager callbacks do I/O.
    """

    def __init__(self, store: PagedWeightStore, capacity_bytes: int = 256 * 1024**2):
        self.store = store
        self.capacity = int(capacity_bytes)
        if self.capacity <= 0:
            raise ValueError("capacity_bytes must be positive")
        self._arena = bytearray(self.capacity)
        self._free: list[tuple[int, int]] = [(0, self.capacity)]  # (offset, size)
        # Running total of free bytes: _evict_to_fit used to recompute
        # sum(size for _, size in self._free) on every eviction iteration
        # (O(F) per iteration); the counter makes it O(1).
        self._free_bytes = self.capacity
        self._resident: dict[int, tuple[int, int]] = {}  # page_id -> (offset, nbytes)
        self._lru: "OrderedDict[int, None]" = OrderedDict()  # eviction order
        self._tensors: dict[str, BufferTensor] = {}
        self.stats = {"allocs": 0, "binds": 0, "faults": 0, "evictions": 0}

    # -- arena ------------------------------------------------------

    def _reserve(self, nbytes: int) -> int:
        nbytes = _align(nbytes)
        self._evict_to_fit(nbytes)
        for i, (off, size) in enumerate(self._free):
            if size >= nbytes:
                del self._free[i]
                self._free_bytes -= size  # whole region leaves the free pool
                if size > nbytes:
                    self._free.insert(i, (off + nbytes, size - nbytes))
                    self._free_bytes += size - nbytes  # remainder returns
                return off
        # _evict_to_fit freed everything evictable; what remains is genuinely full
        raise BufferCapacityError(
            f"cannot reserve {nbytes} bytes in a {self.capacity}-byte arena"
        )

    def _release_region(self, off: int, nbytes: int) -> None:
        nbytes = _align(nbytes)
        self._free_bytes += nbytes
        # The free list is kept sorted; insort is O(F) vs O(F log F) for
        # append + full sort, with an identical resulting list.
        idx = bisect.bisect_left(self._free, (off, nbytes))
        self._free.insert(idx, (off, nbytes))
        merged = []
        for o, s in self._free:
            if merged and merged[-1][0] + merged[-1][1] == o:
                merged[-1] = (merged[-1][0], merged[-1][1] + s)
            else:
                merged.append((o, s))
        self._free = merged

    def _evict_to_fit(self, nbytes: int) -> None:
        """LRU-evict resident pages until nbytes are free. Eviction marks the
        page not-resident buffer-side (the C++ "buffer marked not-resident on
        eviction"); the next bind re-faults it — loud and counted."""
        while self._free_bytes < nbytes and self._lru:
            pid, _ = self._lru.popitem(last=False)
            off, nb = self._resident.pop(pid)
            self._release_region(off, nb)
            self.stats["evictions"] += 1

    # -- tensor geometry --------------------------------------------

    def _tensor_pages(self, name: str):
        """[(page_id, row_start, row_end), ...] for a tensor, in row order.

        O(tensor pages) via the store's index — the old version rescanned the
        whole page table on every alloc (O(n_pages) per layer).
        """
        tid = self.store.tensor_id(name)
        if tid is None:
            raise BufferError(f"tensor {name!r} not in {self.store.path}")
        rows = []
        for e in self.store.tensor_pages(tid):
            rs, re = e["row_start"], e["row_end"]
            if e["precision"] == _q.RAW or rs == re == 0:
                raise BufferError(
                    f"tensor {name!r} page {e['page_id']} is opaque RAW bytes (no row "
                    f"geometry) — the prototype binds decoded float32 pages; "
                    f"the C++ port handles pre-quantized blocks natively"
                )
            rows.append((e["page_id"], rs, re))
        if not rows:
            raise BufferError(f"tensor {name!r} has no pages")
        return rows

    # -- public contract --------------------------------------------

    def alloc(self, name: str) -> BufferTensor:
        """Register a tensor's page geometry (models ggml ``alloc_buffer`` +
        ``init_tensor``). No bytes move; call ``bind``/``ensure_resident``.

        Raises :class:`BufferCapacityError` when the tensor can never be
        fully resident — the tiling requirement is capacity >= one tile.
        """
        if name in self._tensors:
            raise BufferError(f"tensor {name!r} already allocated (release first)")
        pages = self._tensor_pages(name)
        _r0, cols = self.store.page_shape(pages[0][0])
        shape = (pages[-1][2], cols)  # (total_rows, cols)
        t = BufferTensor(name, shape, pages)
        if t.nbytes > self.capacity:
            raise BufferCapacityError(
                f"tensor {name!r} needs {t.nbytes} bytes > capacity {self.capacity}"
            )
        self._tensors[name] = t
        self.stats["allocs"] += 1
        return t

    def _page_nbytes(self, tensor: BufferTensor, rs: int, re: int) -> int:
        return _align((re - rs) * tensor.shape[1] * 4)

    def bind(self, tensor: BufferTensor) -> int:
        """Fault every non-resident page of the tensor into the arena,
        LRU-evicting other pages when the arena is full.

        Returns the number of pages newly bound into the arena this call
        (buffer-side misses; 0 when everything was already resident).
        This is the blocking fence: after ``bind`` returns, compute may
        read the tensor's bytes.
        """
        return self.bind_pages(tensor, tensor.page_ids)

    def bind_pages(self, tensor: BufferTensor, page_ids) -> int:
        """Bind exactly ``page_ids`` (subset of the tensor's pages).

        The sparse-compute fence: when the oracle's union mask selects rows,
        only their pages are required — this is what ``masked_matvec`` and
        the tiled loop call. Returns the number of pages newly bound into
        the arena this call (buffer-side misses; 0 when everything was
        already resident). Whether the pager then hits its own hot-set or
        faults NVMe is the store's own accounting (``store.stats``).
        """
        wanted = set(page_ids)
        unknown = wanted - tensor.page_id_set
        if unknown:
            raise BufferError(
                f"tensor {tensor.name!r}: pages {sorted(unknown)} not in tensor"
            )
        faults = 0
        cols = tensor.shape[1]
        for pid, rs, re in tensor.pages:
            if pid not in wanted:
                continue
            if pid in self._resident:
                self._lru.move_to_end(pid)
                continue
            nbytes = self._page_nbytes(tensor, rs, re)
            off = self._reserve(nbytes)
            arr = self.store.get_page(pid)  # pager hit or NVMe fault; updates LRU/EWMA
            if arr.dtype != np.float32 or arr.ndim != 2:
                self._release_region(off, nbytes)
                raise BufferError(
                    f"tensor {tensor.name!r} page {pid}: expected decoded "
                    f"float32 2-D page, got {arr.dtype} {arr.shape}"
                )
            arr = np.ascontiguousarray(arr, dtype=np.float32)
            if arr.nbytes > nbytes:
                self._release_region(off, nbytes)
                raise BufferError(
                    f"tensor {tensor.name!r} page {pid}: {arr.nbytes} payload "
                    f"bytes exceed reserved {nbytes}"
                )
            # Write straight into an arena view: the old tobytes() + slice
            # assignment copied the page twice (~3.7x slower per page,
            # measured round 2). Identical bytes land in the arena.
            view = np.frombuffer(
                self._arena, dtype=np.float32, count=arr.size, offset=off
            )
            view[:] = arr.reshape(-1)
            self._resident[pid] = (off, nbytes)
            self._lru[pid] = None
            faults += 1
        self.stats["binds"] += 1
        self.stats["faults"] += faults
        return faults

    def ensure_resident(self, tensor: BufferTensor) -> int:
        """Blocking residency fence before compute. Returns faults this call."""
        return self.bind(tensor)

    def is_resident(self, tensor: BufferTensor) -> bool:
        """Measured residency: True only if every page is in the arena."""
        return all(pid in self._resident for pid in tensor.page_ids)

    def is_page_resident(self, page_id: int) -> bool:
        """Measured per-page residency (no I/O performed)."""
        return page_id in self._resident

    def touch(self, tensor: BufferTensor) -> None:
        """LRU + EWMA update for resident pages, no I/O — the ``pager_touch``
        half of the seam. Keeps the pager's hot-set coherent with buffer use."""
        # Iterate tensor.pages (row order) rather than rebuilding the page_ids
        # list: identical LRU order, no per-call list construction.
        for pid, _, _ in tensor.pages:
            if pid in self._resident:
                self._lru.move_to_end(pid)
                self.store.get_page(pid)  # hit path: LRU + EWMA, never faults

    def touch_pages(self, page_ids) -> None:
        for pid in page_ids:
            if pid in self._resident:
                self._lru.move_to_end(pid)
                self.store.get_page(pid)  # hit path only

    def release(self, tensor: BufferTensor) -> None:
        """Drop a tensor's residency eagerly and forget its geometry."""
        t = self._tensors.pop(tensor.name, None)
        if t is None:
            raise BufferError(f"tensor {tensor.name!r} not allocated")
        for pid in t.page_ids:
            if pid in self._resident:
                off, nb = self._resident.pop(pid)
                self._release_region(off, nb)
            self._lru.pop(pid, None)

    def tensor_data(self, tensor: BufferTensor) -> np.ndarray:
        """Assemble the full tensor from arena pages in row order.

        Raises :class:`BufferError` unless every page is resident — the
        Python equivalent of dereferencing the buffer's data pointer only
        after the residency fence.
        """
        if not self.is_resident(tensor):
            raise BufferError(
                f"tensor {tensor.name!r} not fully resident "
                f"(bind/ensure_resident first)"
            )
        rows, cols = tensor.shape
        out = np.empty((rows, cols), dtype=np.float32)
        for pid, rs, re in tensor.pages:
            off, _nb = self._resident[pid]
            page = np.frombuffer(
                self._arena, dtype=np.float32, count=(re - rs) * cols, offset=off
            ).reshape(re - rs, cols)
            out[rs:re] = page
        return out

    # -- masked compute ----------------------------------------------

    def masked_matvec(
        self, tensor: BufferTensor, x: np.ndarray, row_mask: np.ndarray
    ) -> np.ndarray:
        """Row-masked ``W @ x``: only pages intersecting SELECTED rows are
        faulted; unselected rows' pages are never touched. The result is
        bit-identical to running the same row-wise dot products on the dense
        assembly (identical bytes through identical numpy ops give identical
        bits — this is what the buffer contract guarantees: paging never
        changes the bytes the math sees). It is zero on unselected rows.

        Precision note: a single full-matrix BLAS call (``dense @ x``) may
        differ from row-wise dots by ~1 ULP because of reduction order inside
        the kernel. That is a kernel-scheduling artifact, not a paging
        artifact — the bytes are identical either way.

        This is the Python model of the row-masked ``ggml_mul_mat``: rows
        outside the mask are skipped, so their pages are never faulted.
        """
        x = np.ascontiguousarray(x, dtype=np.float32)
        mask = np.ascontiguousarray(row_mask, dtype=bool)
        rows, cols = tensor.shape
        if x.shape != (cols,) or mask.shape != (rows,):
            raise ValueError(
                f"shape mismatch: tensor {tensor.shape}, x {x.shape}, mask {mask.shape}"
            )
        wanted = [(pid, rs, re) for pid, rs, re in tensor.pages if mask[rs:re].any()]
        self.bind_pages(tensor, [pid for pid, _, _ in wanted])

        out = np.zeros(rows, dtype=np.float32)
        touched: list[int] = []
        for pid, rs, re in wanted:
            off, _nb = self._resident[pid]
            page = np.frombuffer(
                self._arena, dtype=np.float32, count=(re - rs) * cols, offset=off
            ).reshape(re - rs, cols)
            m = mask[rs:re]
            if m.all():
                out[rs:re] = page @ x
            else:
                out[rs:re][m] = page[m] @ x
            touched.append(pid)
        # One batched touch instead of one store.get_page (lock acquisition)
        # per page: LRU end-order and per-page EWMA updates are identical.
        self.touch_pages(touched)
        return out


def tiled_exact_matvec(
    buffer: PagerBackedBuffer,
    tensor_names: list[str],
    x: np.ndarray,
    row_mask_fn=None,
) -> tuple[list[np.ndarray], dict]:
    """Production per-layer tiling loop, runnable here.

    From ``integrations/llama.cpp/INTEGRATION.md``: on real hardware the exact
    mode keeps ONE layer (or expert) resident at a time::

        for layer in layers:
            ensure_resident(pages_of(layer))            # blocking; counted
            compute(layer)                              # never sees a miss
            prefetch(union_mask_pages_of(layer + 1))    # overlapped

    ``tensor_names`` are the per-layer tensors in order; ``row_mask_fn(layer)``
    is the oracle's union mask for that layer (None = full mask). Returns
    ``(per_layer_outputs, report)`` where the report carries MEASURED
    ``faults``, ``prefetch_pages`` and ``guarantee_held`` (True only if every
    tile measured full residency before its compute).
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    outputs: list[np.ndarray] = []
    report = {
        "layers": len(tensor_names),
        "faults": 0,
        "prefetch_pages": 0,
        "guarantee_held": True,
    }
    page_lists = {name: buffer._tensor_pages(name) for name in tensor_names}

    for i, name in enumerate(tensor_names):
        t = buffer.alloc(name)
        # prefetch the NEXT tile's pages while this tile computes (async,
        # exactly like OverlapScheduler's lookahead edge)
        if i + 1 < len(tensor_names):
            nxt = [pid for pid, _, _ in page_lists[tensor_names[i + 1]]]
            buffer.store.prefetch(nxt)
            report["prefetch_pages"] += len(nxt)
        mask = row_mask_fn(i) if row_mask_fn is not None else None
        if mask is None:
            mask = np.ones(t.shape[0], dtype=bool)
        # the fence covers exactly the required set: all pages in exact
        # (full-mask) mode, only the union-mask pages in sparse mode.
        wanted = [pid for pid, rs, re in t.pages if mask[rs:re].any()]
        faults = buffer.bind_pages(t, wanted)
        report["faults"] += faults
        if not all(buffer.is_page_resident(pid) for pid in wanted):
            report["guarantee_held"] = False
        outputs.append(buffer.masked_matvec(t, x, mask))
        buffer.release(t)
    return outputs, report
