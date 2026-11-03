"""GGUF -> .lac converter (offline, one pass, streaming).

Parses the GGUF v3 header with ``struct`` (tensor infos only) and streams each
tensor's bytes via ``os.pread`` — the full weight blob is NEVER loaded into RAM
at once, so converting a 25 GB file on an 8 GB box is safe.

Precision tiers (initial; the pager re-tiers by measured frequency at runtime):
- attention / norms / embeddings -> Q4 (hot tier)
- FFN matrices                     -> Q2 (warm tier)
- tensors larger than --cold-threshold-mb -> ternary (cold tier)
- non-F32/F16 GGUF types           -> RAW opaque byte pages (no requantization)

Precision policies (``--precision-policy``):
- ``exact`` (default): EVERY tensor is carried as RAW byte pages, bit-identical
  to the source GGUF. Paging and sparsity only — zero requantization, so there
  is never additional quality loss beyond what the source file already chose.
- ``extreme``: the adaptive tiering above (plus RAW carry-through for
  pre-quantized GGUF types). Fastest; documented quality/speed tradeoff.

Only F32 (type 0) and F16 (type 1) tensors are requantized, and only under the
extreme policy; anything else (Q4_0, Q8_0, ...) is carried through byte-identical
as RAW pages. This is honest: Lacuna does not claim to improve pre-quantized
GGUF blocks.

CLI: ``python -m lacuna.convert model.gguf model.lac [--page-mb 2] [--precision-policy exact|extreme]``

Sharded GGUF (``python -m lacuna.convert shard1.gguf shard2.gguf model.lac``):
each shard is a standalone valid GGUF v3 file holding a disjoint subset of
the model's tensors, in original order across shards. The convention follows
llama.cpp's ``gguf-split`` tool (``tools/gguf-split/gguf-split.cpp`` in the
local checkout):

- ``split.no`` (uint16): 0-based index of this shard.
- ``split.count`` (uint16): total number of shards.
- ``split.tensors.count`` (int32): TOTAL tensor count across all shards.
- Tensors are never split mid-tensor: each tensor lives wholly in one shard.
- Only shard 0 carries the full KV metadata; later shards carry just the
  split keys. Shard files are conventionally named
  ``model-00001-of-00005.gguf`` (1-based index in the filename).

``convert_sharded`` merges the shards into ONE .lac that is byte-identical
to converting the equivalent single file (the .lac writer's output depends
only on the sequence of tensors added, never on the source filename or
shard boundaries), and fails loud on any inconsistency: a missing shard,
a duplicated tensor, or metadata that disagrees across shards.

Streaming conversion (``StreamingShardConverter``): for lanes where the
writable budget is smaller than shards + .lac combined. Kaggle kernels
can write only ~96.3 GiB to /tmp per kernel (measured twice:
103,448,313,856 bytes, then ``Errno 30 Read-only file system``, while df
still showed ~953 GiB free — df-free is the shared overlay's free space,
not the writable budget). The download-all-then-convert sequence needs
61.4 GB (shards) + 61.4 GB (.lac) = 122.8 GiB for the Gemma 31B BF16 split
and cannot succeed there. The streaming converter converts one shard at
a time into a growing .lac, so the caller deletes each shard file
between conversions: shard 1 (40 GB) + partial .lac (~40 GB) = ~80 GiB,
then shard 2 (21.4 GB) + .lac (61.4 GB) = ~83 GiB peak, under the cap.
Shards must arrive in merged tensor order (split.no 0, 1, 2, ...; or
filename index 1, 2, 3, ...); output is byte-identical to
``convert_sharded`` of the same shards. See the class docstring for the
parity-v3 sequence this enables (llama.cpp loads split GGUF natively, so
no merged GGUF is ever needed).
"""

from __future__ import annotations

import argparse
import math
import os
import re
import struct
import sys

import numpy as np

from . import quant as _q
from .pager import LacWriter

GGUF_MAGIC = b"GGUF"
SUPPORTED_VERSION = 3

# GGUF tensor types we requantize: type_id -> (numpy dtype, bytes/elem)
_DENSE_TYPES = {0: (np.float32, 4), 1: (np.float16, 2)}

# GGUF metadata value types (for skipping the KV section)
_KV_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

# GGUF metadata value types (for reading the KV section)
_KV_FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
           6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}

# Split metadata keys (llama.cpp gguf-split convention; see module docstring).
_SPLIT_NO = "split.no"
_SPLIT_COUNT = "split.count"
_SPLIT_TENSORS_COUNT = "split.tensors.count"
_SPLIT_KEYS = frozenset({_SPLIT_NO, _SPLIT_COUNT, _SPLIT_TENSORS_COUNT})

# ``model-00001-of-00005.gguf`` — 1-based index in the filename.
_FILENAME_SHARD_RE = re.compile(r"-(\d+)-of-(\d+)\.gguf$", re.IGNORECASE)


def _read_u64(f) -> int:
    return struct.unpack("<Q", f.read(8))[0]


def _read_str(f) -> str:
    n = _read_u64(f)
    return f.read(n).decode("utf-8", errors="replace")


def _skip_value(f, vtype: int) -> None:
    if vtype == 8:  # string
        f.read(_read_u64(f))
    elif vtype == 9:  # array
        atype = struct.unpack("<I", f.read(4))[0]
        n = _read_u64(f)
        for _ in range(n):
            _skip_value(f, atype)
    else:
        f.seek(_KV_SIZES[vtype], os.SEEK_CUR)


def _read_kv_value(f, vtype: int):
    """Read one GGUF metadata value (arrays recurse). Raises ValueError
    on an unknown value type instead of silently mis-parsing the rest
    of the header."""
    if vtype == 8:  # string
        return _read_str(f)
    if vtype == 9:  # array
        atype = struct.unpack("<I", f.read(4))[0]
        n = _read_u64(f)
        return [_read_kv_value(f, atype) for _ in range(n)]
    fmt = _KV_FMT.get(vtype)
    if fmt is None:
        raise ValueError(f"unknown GGUF metadata value type {vtype}")
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]


def _kv_equal(a, b) -> bool:
    """Metadata equality that treats NaN == NaN (a float32 NaN payload
    is a legitimate metadata value, and naive ``!=`` would flag every
    shard carrying one as 'mismatched')."""
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return a == b
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_kv_equal(x, y) for x, y in zip(a, b))
    return a == b


def _tier_for(name: str, nbytes: int, cold_threshold: int) -> int:
    lname = name.lower()
    if nbytes >= cold_threshold:
        return _q.TERNARY
    hot_keys = ("embed", "norm", "attn", "wq", "wk", "wv", "wo", "qkv", "output")
    if any(k in lname for k in hot_keys):
        return _q.Q4
    return _q.Q2


def parse_gguf_header(path: str, with_kv: bool = False, kv_keys=None):
    """Return (tensor_infos, data_start), or (tensor_infos, data_start, kv)
    when ``with_kv`` is set. tensor_infos: list of dicts. When ``kv_keys``
    is given, only those metadata keys are materialized (the rest are
    skipped) — so a single-file convert can cheaply check ``split.count``
    without parsing a 262k-entry tokenizer array."""
    f = open(path, "rb")
    magic = f.read(4)
    if magic != GGUF_MAGIC:
        f.close()
        raise ValueError(f"not a GGUF file: {path}")
    (version,) = struct.unpack("<I", f.read(4))
    if version != SUPPORTED_VERSION:
        f.close()
        raise ValueError(f"GGUF v{version} not supported (need v{SUPPORTED_VERSION})")
    n_tensors = _read_u64(f)
    n_kv = _read_u64(f)
    kv = {}
    for _ in range(n_kv):
        key = _read_str(f)
        (vtype,) = struct.unpack("<I", f.read(4))
        if with_kv and (kv_keys is None or key in kv_keys):
            kv[key] = _read_kv_value(f, vtype)
        else:
            _skip_value(f, vtype)
    infos = []
    for _ in range(n_tensors):
        name = _read_str(f)
        (n_dims,) = struct.unpack("<I", f.read(4))
        dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
        (ttype,) = struct.unpack("<I", f.read(4))
        (offset,) = struct.unpack("<Q", f.read(8))
        infos.append({"name": name, "dims": dims, "type": ttype, "offset": offset})
    pos = f.tell()
    data_start = pos + ((-pos) % 32)  # GGUF aligns the data section to 32
    f.close()
    if with_kv:
        return infos, data_start, kv
    return infos, data_start


def _dense_nbytes(info) -> int | None:
    dt, es = _DENSE_TYPES.get(info["type"], (None, None))
    if dt is None:
        return None
    n_elem = 1
    for d in info["dims"]:
        n_elem *= d
    return n_elem * es


def _opaque_nbytes(info, infos, data_start: int, fsize: int) -> int:
    off = info["offset"]
    nbytes = fsize - (data_start + off)  # over-estimate; trimmed below
    later = [i["offset"] for i in infos if i["offset"] > off]
    if later:
        nbytes = min(nbytes, min(later) - off)
    # No size cap here: the span (next tensor's offset, or EOF for the last
    # tensor) is the true bound. An earlier `min(nbytes, 1 << 32)` silently
    # truncated larger opaque tensors; reads now fail loud on early EOF
    # instead (see _pread_full), so a bogus span raises rather than clips.
    return nbytes


# Per-call read size: far below the Linux per-syscall cap (see _pread_full),
# large enough that syscall overhead is negligible.
_READ_CHUNK = 1 << 26  # 64 MiB


def _pread_full(fd: int, nbytes: int, offset: int) -> bytes:
    """Read exactly ``nbytes`` at ``offset``, looping over short reads.

    A single ``os.pread`` syscall transfers at most ``MAX_RW_COUNT``
    (0x7ffff000 = 2,147,479,552) bytes on Linux, so the one-shot pread
    this replaces silently truncated every tensor larger than that to
    exactly 0x7ffff000 bytes (measured: a 2,580,480,000-byte GGUF tensor
    landed in the .lac as 2,147,479,552 bytes). Raises OSError if the
    file ends early — a short tensor is corruption, never a short write.
    """
    parts = []
    got = 0
    while got < nbytes:
        chunk = os.pread(fd, min(nbytes - got, _READ_CHUNK), offset + got)
        if not chunk:
            raise OSError(
                f"short read at offset {offset + got}: wanted {nbytes} "
                f"bytes, got {got} (source file truncated?)"
            )
        parts.append(chunk)
        got += len(chunk)
    if len(parts) == 1:
        return parts[0]
    return b"".join(parts)


def _tensor_chunks(fd: int, nbytes: int, offset: int):
    """Yield a tensor's bytes in <=64 MiB chunks, exactly ``nbytes`` total.

    Streams through :func:`_pread_full`, so tensors larger than RAM (or
    larger than the per-syscall read cap) convert without ever being
    held whole in memory. Raises OSError on early EOF.
    """
    done = 0
    while done < nbytes:
        step = min(nbytes - done, _READ_CHUNK)
        yield _pread_full(fd, step, offset + done)
        done += step


def _convert_one_part(w, stats, infos, data_start: int, fd: int, fsize: int,
                      layer_fn, log, precision_policy: str,
                      cold_threshold: int) -> None:
    """Write one part's tensors through an already-open ``LacWriter``.

    One part is a single file, or one shard of a split model. Tensor bytes
    are always read from the part that owns the tensor (offsets are
    per-file), so the output depends only on the merged tensor sequence,
    never on how the tensors were distributed across files. Mutates
    ``stats`` in place. Shared by :func:`_convert_parts` (one-shot) and
    :class:`StreamingShardConverter` (one shard at a time) — the two paths
    therefore produce byte-identical output for the same tensor sequence.
    """
    for info in infos:
        name, dims, ttype, off = info["name"], info["dims"], info["type"], info["offset"]
        layer = layer_fn(name) if layer_fn else 0
        if precision_policy == _q.POLICY_EXACT:
            # Bit-identical: carry the source bytes untouched, whatever
            # the GGUF type (F32, F16, Q4_K_M, ...). Zero requantization.
            nbytes = _dense_nbytes(info)
            if nbytes is None:
                nbytes = _opaque_nbytes(info, infos, data_start, fsize)
            _, written = w.add_raw_stream(
                name, _tensor_chunks(fd, nbytes, data_start + off), layer=layer
            )
            if written != nbytes:
                raise OSError(
                    f"convert: tensor {name}: wrote {written} of "
                    f"{nbytes} source bytes"
                )
            stats["tiers"]["raw"] += 1
            stats["bytes_in"] += written
            tier_label = "raw[exact]"
        elif ttype in _DENSE_TYPES:
            dt, es = _DENSE_TYPES[ttype]
            nbytes = _dense_nbytes(info)
            raw = _pread_full(fd, nbytes, data_start + off)
            if len(raw) != nbytes:
                raise OSError(
                    f"convert: tensor {name}: read {len(raw)} of "
                    f"{nbytes} source bytes"
                )
            arr = np.frombuffer(raw, dtype=dt).astype(np.float32)
            # GGUF stores dims reversed vs row-major expectation; treat as 2-D
            # (rows = last dim, cols = product of the rest).
            rows = int(dims[-1]) if dims else 1
            cols = int(nbytes // es // max(rows, 1))
            arr = arr.reshape(rows, cols)
            prec = _tier_for(name, nbytes, cold_threshold)
            w.add_tensor(name, arr, layer=layer, precision=prec)
            stats["tiers"][_q.precision_name(prec)] += 1
            stats["bytes_in"] += nbytes
            tier_label = _q.precision_name(prec)
        else:
            # Opaque: carry through as RAW byte pages (no requantization).
            nbytes = _opaque_nbytes(info, infos, data_start, fsize)
            _, written = w.add_raw_stream(
                name, _tensor_chunks(fd, nbytes, data_start + off), layer=layer
            )
            if written != nbytes:
                raise OSError(
                    f"convert: tensor {name}: wrote {written} of "
                    f"{nbytes} source bytes"
                )
            stats["tiers"]["raw"] += 1
            stats["bytes_in"] += written
            tier_label = "raw"
        stats["tensors"] += 1
        log(f"  {name}: {tier_label}")


def _convert_parts(parts, spp_path: str, page_mb: int, cold_threshold_mb: int,
                   layer_fn, quiet: bool, precision_policy: str) -> dict:
    """Shared one-shot conversion core. ``parts`` is a list of (infos,
    data_start, fd, fsize) tuples in merged tensor order — one part for a
    single file, one part per shard for a sharded model."""
    precision_policy = _q.validate_policy(precision_policy)
    cold_threshold = cold_threshold_mb * 1024 * 1024
    stats = {"tensors": 0, "pages": 0, "bytes_in": 0, "tiers": {"q4": 0, "q2": 0, "ternary": 0, "raw": 0},
             "precision_policy": precision_policy}

    def log(msg):
        if not quiet:
            print(msg, flush=True)

    with LacWriter(spp_path, page_size=page_mb * 1024 * 1024,
                   precision_policy=precision_policy) as w:
        for infos, data_start, fd, fsize in parts:
            _convert_one_part(w, stats, infos, data_start, fd, fsize,
                              layer_fn, log, precision_policy,
                              cold_threshold)
    stats["pages"] = None  # available from the written file; kept simple
    return stats


def convert(
    gguf_path: str,
    spp_path: str,
    page_mb: int = 2,
    cold_threshold_mb: int = 1024,
    layer_fn=None,
    quiet: bool = False,
    precision_policy: str = "exact",
) -> dict:
    """Convert GGUF -> .lac. Returns stats dict."""
    infos, data_start, kv = parse_gguf_header(
        gguf_path, with_kv=True, kv_keys={_SPLIT_COUNT})
    declared = kv.get(_SPLIT_COUNT)
    if declared is not None and declared > 1:
        # A lone shard is not a model: converting it alone would silently
        # produce a partial .lac. Fail loud and point at the sharded path.
        raise ValueError(
            f"{gguf_path} is one shard of a {declared}-shard model "
            f"(split.count={declared}); convert all shards together with "
            f"convert_sharded(), not this file alone"
        )
    fd = os.open(gguf_path, os.O_RDONLY)
    try:
        return _convert_parts(
            [(infos, data_start, fd, os.fstat(fd).st_size)],
            spp_path, page_mb, cold_threshold_mb, layer_fn, quiet,
            precision_policy,
        )
    finally:
        os.close(fd)


def _filename_shard_index(path: str):
    """(1-based index, total) from a ``-NNNNN-of-MMMMM.gguf`` filename,
    or None when the filename carries no shard numbering."""
    m = _FILENAME_SHARD_RE.search(os.path.basename(path))
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _load_shards(paths):
    """Parse shard headers, validate the split, and return shard records
    in merged tensor order. Each record: {path, infos, data_start, kv}.

    Order resolution: ``split.no`` metadata wins when present on every
    shard; else ``-NNNNN-of-MMMMM`` filenames when present on every
    shard; else the order the paths were given. Any inconsistency fails
    loud — a missing shard, a duplicated tensor, or metadata that
    disagrees across shards is corruption, never a partial model.
    """
    paths = list(paths)
    if not paths:
        raise ValueError("convert_sharded: no shard paths given")
    records = []
    for p in paths:
        infos, data_start, kv = parse_gguf_header(p, with_kv=True)
        seen = {}
        for info in infos:
            prev = seen.get(info["name"])
            if prev is not None:
                raise ValueError(
                    f"tensor {info['name']!r} appears twice in shard {p}"
                )
            seen[info["name"]] = info
        records.append({"path": p, "infos": infos,
                        "data_start": data_start, "kv": kv})

    # split.count: must agree everywhere it appears, and must equal the
    # number of shards given — anything else means a shard is missing.
    counts = [r["kv"].get(_SPLIT_COUNT) for r in records]
    declared_counts = [c for c in counts if c is not None]
    if declared_counts:
        if len(set(declared_counts)) != 1:
            raise ValueError(
                f"split.count disagrees across shards: {declared_counts}"
            )
        if declared_counts[0] != len(records):
            raise ValueError(
                f"split.count={declared_counts[0]} but {len(records)} "
                f"shard(s) given (missing shard?)"
            )
        if declared_counts[0] < 1:
            raise ValueError(f"invalid split.count={declared_counts[0]}")

    # Filename numbering must agree with split.count when both are present.
    fn_idxs = [_filename_shard_index(r["path"]) for r in records]
    fn_totals = {total for idx in fn_idxs if idx is not None
                 for _, total in [idx]}
    fn_total = None
    if fn_totals:
        if len(fn_totals) != 1:
            raise ValueError(
                f"shard filename totals disagree: {sorted(fn_totals)}"
            )
        fn_total = next(iter(fn_totals))
        if declared_counts and fn_total != declared_counts[0]:
            raise ValueError(
                f"split.count={declared_counts[0]} disagrees with shard "
                f"filename total {fn_total}"
            )

    # Order: split.no wins; else filenames; else given order.
    nos = [r["kv"].get(_SPLIT_NO) for r in records]
    if any(n is not None for n in nos):
        if not all(n is not None for n in nos):
            raise ValueError(
                "some shards carry split.no but others do not — shard "
                "order is ambiguous"
            )
        if len(set(nos)) != len(nos):
            raise ValueError(f"duplicate split.no values: {sorted(nos)}")
        if sorted(nos) != list(range(len(nos))):
            raise ValueError(
                f"split.no values {sorted(nos)} do not cover "
                f"0..{len(nos) - 1} (missing shard?)"
            )
        records = [r for _, r in sorted(zip(nos, records), key=lambda t: t[0])]
    elif all(x is not None for x in fn_idxs):
        idxs = [i for i, _ in fn_idxs]
        if len(set(idxs)) != len(idxs):
            raise ValueError(f"duplicate shard filename indices: {sorted(idxs)}")
        if sorted(idxs) != list(range(1, len(idxs) + 1)):
            raise ValueError(
                f"shard filename indices {sorted(idxs)} do not cover "
                f"1..{len(idxs)} (missing shard?)"
            )
        if fn_total is not None and len(records) != fn_total:
            raise ValueError(
                f"shard filename total {fn_total} disagrees with the "
                f"{len(records)} shard file(s) given"
            )
        records = [r for _, r in sorted(zip(idxs, records), key=lambda t: t[0])]
    elif any(x is not None for x in fn_idxs):
        raise ValueError(
            "some shard filenames carry -NNNNN-of-MMMMM numbering but "
            "others do not — shard order is ambiguous"
        )
    # else: given order stands.

    # split.tensors.count: must agree everywhere, and the merged tensor
    # count must equal it — otherwise a tensor was dropped or invented.
    tcounts = [r["kv"].get(_SPLIT_TENSORS_COUNT) for r in records]
    declared_tcounts = [t for t in tcounts if t is not None]
    if declared_tcounts:
        if len(set(declared_tcounts)) != 1:
            raise ValueError(
                f"split.tensors.count disagrees across shards: "
                f"{declared_tcounts}"
            )
        merged = sum(len(r["infos"]) for r in records)
        if merged != declared_tcounts[0]:
            raise ValueError(
                f"split.tensors.count={declared_tcounts[0]} but shards "
                f"hold {merged} tensors"
            )

    # No tensor duplicated or dropped across shards. A repeated name with
    # different dims/type is reported as such; an identical repeat is
    # still an error — a .lac with two tensors of one name is ambiguous.
    merged_names = {}  # name -> (dims, type, first shard path)
    for r in records:
        for info in r["infos"]:
            prev = merged_names.get(info["name"])
            if prev is not None:
                if prev[0] != info["dims"] or prev[1] != info["type"]:
                    raise ValueError(
                        f"tensor {info['name']!r} appears in multiple "
                        f"shards with different metadata: dims "
                        f"{prev[0]} vs {info['dims']}, type "
                        f"{prev[1]} vs {info['type']}"
                    )
                raise ValueError(
                    f"tensor {info['name']!r} duplicated across shards "
                    f"({prev[2]} and {r['path']})"
                )
            merged_names[info["name"]] = (
                info["dims"], info["type"], r["path"])

    # Non-split KV metadata must agree wherever it appears in more than
    # one shard (split.no differs by design and is excluded; split.count
    # and split.tensors.count were checked above). Shards from gguf-split
    # carry full metadata only on shard 0, so keys absent elsewhere are
    # fine — only outright disagreement fails.
    kv_seen = {}
    for r in records:
        for key, value in r["kv"].items():
            if key in _SPLIT_KEYS:
                continue
            if key in kv_seen and not _kv_equal(kv_seen[key], value):
                raise ValueError(
                    f"metadata key {key!r} disagrees across shards"
                )
            kv_seen.setdefault(key, value)
    return records


def convert_sharded(
    shard_paths,
    spp_path: str,
    page_mb: int = 2,
    cold_threshold_mb: int = 1024,
    layer_fn=None,
    quiet: bool = False,
    precision_policy: str = "exact",
) -> dict:
    """Convert multiple GGUF shards -> ONE .lac, byte-identical to
    converting the equivalent single file. Shards may be given in any
    order when they carry ``split.no`` metadata or ``-NNNNN-of-MMMMM``
    filenames; otherwise they must be given in tensor order. Returns the
    same stats dict as :func:`convert`, plus ``shards``. A failed
    conversion removes the partial output — never a half-written .lac
    pretending to be a model."""
    records = _load_shards(shard_paths)
    fds = []
    try:
        parts = []
        for r in records:
            fd = os.open(r["path"], os.O_RDONLY)
            fds.append(fd)
            parts.append((r["infos"], r["data_start"], fd,
                          os.fstat(fd).st_size))
        stats = _convert_parts(
            parts, spp_path, page_mb, cold_threshold_mb, layer_fn, quiet,
            precision_policy,
        )
    except Exception:
        for fd in fds:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            os.remove(spp_path)
        except OSError:
            pass
        raise
    for fd in fds:
        os.close(fd)
    stats["shards"] = len(records)
    return stats


class StreamingShardConverter:
    """Convert GGUF shards -> ONE .lac one shard at a time, so the caller
    can delete each shard file between conversions.

    Kaggle kernels can write only ~96.3 GiB to /tmp per kernel (measured
    twice: 103,448,313,856 bytes, then ``Errno 30 Read-only file
    system``, while df still showed ~953 GiB free — df-free is the
    shared overlay's free space, not the writable budget, and once
    tripped the kernel's /tmp is read-only for everything, including
    cleanup). The download-all-then-convert sequence needs 61.4 GB
    (shards) + 61.4 GB (.lac) = 122.8 GiB for the Gemma 31B BF16 split
    and therefore cannot succeed there. This class converts one shard
    at a time into a growing .lac:

    Peak-write arithmetic (Gemma 31B BF16: shard 1 = 40 GB, shard 2 =
    21.4 GB):
      shard 1 (40 GB) + partial .lac (~40 GB)             = ~80 GiB
      -> delete shard 1
      shard 2 (21.4 GB) + .lac (61.4 GB)                  = ~83 GiB peak
      83 GiB < 96.3 GiB cap. The .lac alone (61.4 GB) also fits.

    Parity-v3 sequence this enables (llama.cpp loads split GGUF files
    natively from the first shard, so no merged GGUF is ever needed):
      1. download shard 1 -> add_shard (keep shard 1 for the llama leg)
      2. download shard 2 -> add_shard -> finish() -> ONE .lac
      3. run the llama leg from the split shards (native split loading)
      4. delete shards; run the GPU leg from the .lac
    (If the llama leg is not needed, delete each shard right after its
    add_shard call; peak then stays ~83 GiB as computed above.)

    Ordering constraint — unlike :func:`convert_sharded`, which sorts,
    shards MUST arrive in merged tensor order, because tensors are
    written as they arrive and cannot be reordered later. When shards
    carry ``split.no`` they must arrive 0, 1, 2, ...; when they carry
    only ``-NNNNN-of-MMMMM`` filenames, 1, 2, 3, ...; otherwise in the
    caller's given order. Any violation fails loud BEFORE the
    offending shard's tensors are written.

    Validation is the same as :func:`convert_sharded`, checked
    incrementally: split.count agreement + completeness (at finish),
    split.no consecutive coverage, split.tensors.count agreement +
    merged count (at finish), no duplicated tensors, non-split KV
    agreement (NaN-aware). A failed add_shard or finish aborts the
    session and removes the partial .lac — never a half-written file
    pretending to be a model.

    The output is byte-identical to :func:`convert_sharded` of the same
    shards: both write the same tensor sequence through the same
    :func:`_convert_one_part` code path; only the interleaving of
    shard-file deletion differs, which never affects .lac bytes.

    Cross-process resume is deliberately NOT implemented: a Kaggle
    kernel restart does not give back the previous run's /tmp state, so
    a half-converted .lac could not be trusted or resumed there anyway.
    One session = one kernel process, from first shard to finish().
    """

    def __init__(self, spp_path: str, page_mb: int = 2,
                 cold_threshold_mb: int = 1024, layer_fn=None,
                 quiet: bool = False, precision_policy: str = "exact"):
        self.spp_path = spp_path
        self._layer_fn = layer_fn
        self._quiet = quiet
        self._precision_policy = _q.validate_policy(precision_policy)
        self._cold_threshold = cold_threshold_mb * 1024 * 1024
        self._stats = {"tensors": 0, "pages": 0, "bytes_in": 0,
                       "tiers": {"q4": 0, "q2": 0, "ternary": 0, "raw": 0},
                       "precision_policy": self._precision_policy}

        def log(msg):
            if not quiet:
                print(msg, flush=True)

        self._log = log
        # One LacWriter held open for the whole session: pages append to
        # the growing .lac as shards arrive; the entry table + header are
        # written only by finish(). Creating the session creates (and
        # truncates) the output file — call abort() if it must not stay.
        self._writer = LacWriter(
            spp_path, page_size=page_mb * 1024 * 1024,
            precision_policy=self._precision_policy)
        # Validation state, committed only after a shard converts cleanly.
        self._seen_names = {}  # name -> (dims, type, first shard path)
        self._kv_seen = {}
        self._mode = None  # "split_no" | "filename" | "given"
        self._expected_no = 0
        self._expected_fn = 1
        self._fn_total = None
        self._declared_count = None
        self._declared_tcount = None
        self._shard_count = 0
        self._finished = False
        self._aborted = False

    def _validate_shard(self, path: str, infos, kv) -> None:
        """All checks that can run BEFORE writing this shard's tensors.
        Raises ValueError on any inconsistency. Mutates no state — the
        caller commits state only after a clean conversion."""
        # Within-shard duplicates.
        seen = set()
        for info in infos:
            if info["name"] in seen:
                raise ValueError(
                    f"tensor {info['name']!r} appears twice in shard {path}"
                )
            seen.add(info["name"])

        no = kv.get(_SPLIT_NO)
        fn = _filename_shard_index(path)
        if self._shard_count == 0:
            # First shard locks the ordering mode.
            if no is not None:
                if no != 0:
                    raise ValueError(
                        f"streaming conversion must start at split.no=0, "
                        f"got split.no={no} in {path} (add shards in "
                        f"merged tensor order)"
                    )
            elif fn is not None:
                if fn[0] != 1:
                    raise ValueError(
                        f"streaming conversion must start at shard "
                        f"filename index 1, got {fn[0]} in {path} (add "
                        f"shards in merged tensor order)"
                    )
        else:
            mode = self._mode
            if mode == "split_no":
                if no is None:
                    raise ValueError(
                        f"shard {path} carries no split.no but the stream "
                        f"is split.no-ordered (add shards in merged "
                        f"tensor order from split.no=0)"
                    )
                if no != self._expected_no:
                    raise ValueError(
                        f"shard out of order: expected split.no="
                        f"{self._expected_no}, got split.no={no} in "
                        f"{path}"
                    )
            elif mode == "filename":
                if fn is None:
                    raise ValueError(
                        f"shard {path} filename carries no "
                        f"-NNNNN-of-MMMMM numbering but the stream is "
                        f"filename-ordered"
                    )
                if fn[0] != self._expected_fn:
                    raise ValueError(
                        f"shard out of order: expected shard filename "
                        f"index {self._expected_fn}, got {fn[0]} in "
                        f"{path}"
                    )
                if fn[1] != self._fn_total:
                    raise ValueError(
                        f"shard filename total {fn[1]} disagrees with "
                        f"the stream's total {self._fn_total}"
                    )
                if no is not None and no != fn[0] - 1:
                    raise ValueError(
                        f"split.no={no} disagrees with shard filename "
                        f"index {fn[0]} in {path}"
                    )
            else:  # given
                if no is not None or fn is not None:
                    raise ValueError(
                        f"shard {path} carries ordering metadata "
                        f"(split.no={no}, filename index={fn}) but the "
                        f"stream started without any — restart the "
                        f"stream in split.no order from the start"
                    )

        # split.count: must agree everywhere, and the stream must not
        # already exceed it.
        count = kv.get(_SPLIT_COUNT)
        if count is not None:
            if count < 1:
                raise ValueError(f"invalid split.count={count}")
            if (self._declared_count is not None
                    and count != self._declared_count):
                raise ValueError(
                    f"split.count disagrees across shards: "
                    f"{self._declared_count} vs {count}"
                )
            if self._shard_count + 1 > count:
                raise ValueError(
                    f"split.count={count} but a "
                    f"{self._shard_count + 1}th shard arrived"
                )

        # split.tensors.count: must agree everywhere, and the merged
        # count must not overshoot it.
        tcount = kv.get(_SPLIT_TENSORS_COUNT)
        if tcount is not None:
            if (self._declared_tcount is not None
                    and tcount != self._declared_tcount):
                raise ValueError(
                    f"split.tensors.count disagrees across shards: "
                    f"{self._declared_tcount} vs {tcount}"
                )
            if self._stats["tensors"] + len(infos) > tcount:
                raise ValueError(
                    f"split.tensors.count={tcount} but shards would "
                    f"hold {self._stats['tensors'] + len(infos)} tensors"
                )

        # Cross-shard duplicates (dims/type disagreement reported
        # explicitly, as in convert_sharded).
        for info in infos:
            prev = self._seen_names.get(info["name"])
            if prev is not None:
                if prev[0] != info["dims"] or prev[1] != info["type"]:
                    raise ValueError(
                        f"tensor {info['name']!r} appears in multiple "
                        f"shards with different metadata: dims "
                        f"{prev[0]} vs {info['dims']}, type "
                        f"{prev[1]} vs {info['type']}"
                    )
                raise ValueError(
                    f"tensor {info['name']!r} duplicated across shards "
                    f"({prev[2]} and {path})"
                )

        # Non-split KV agreement (split keys excluded by design).
        for key, value in kv.items():
            if key in _SPLIT_KEYS:
                continue
            if key in self._kv_seen and not _kv_equal(
                    self._kv_seen[key], value):
                raise ValueError(
                    f"metadata key {key!r} disagrees across shards"
                )

    def _commit_shard(self, path: str, infos, kv) -> None:
        """Record a cleanly converted shard in the validation state."""
        no = kv.get(_SPLIT_NO)
        fn = _filename_shard_index(path)
        if self._shard_count == 0:
            if no is not None:
                self._mode = "split_no"
            elif fn is not None:
                self._mode = "filename"
                self._fn_total = fn[1]
            else:
                self._mode = "given"
        if self._mode == "split_no":
            self._expected_no += 1
        elif self._mode == "filename":
            self._expected_fn += 1
        count = kv.get(_SPLIT_COUNT)
        if count is not None:
            self._declared_count = count
        tcount = kv.get(_SPLIT_TENSORS_COUNT)
        if tcount is not None:
            self._declared_tcount = tcount
        for info in infos:
            self._seen_names[info["name"]] = (
                info["dims"], info["type"], path)
        for key, value in kv.items():
            if key not in _SPLIT_KEYS:
                self._kv_seen.setdefault(key, value)
        self._shard_count += 1

    def add_shard(self, path: str) -> int:
        """Validate and convert one shard's tensors into the growing
        .lac. Returns the number of tensors added. The caller may delete
        the shard file as soon as this returns — its bytes have been
        consumed (the shard fd is closed here). Any failure aborts the
        session and removes the partial .lac."""
        if self._finished:
            raise ValueError("streaming conversion already finished")
        if self._aborted:
            raise ValueError("streaming conversion was aborted")
        try:
            infos, data_start, kv = parse_gguf_header(path, with_kv=True)
            self._validate_shard(path, infos, kv)
            fd = os.open(path, os.O_RDONLY)
            try:
                _convert_one_part(
                    self._writer, self._stats, infos, data_start, fd,
                    os.fstat(fd).st_size, self._layer_fn, self._log,
                    self._precision_policy, self._cold_threshold)
            finally:
                os.close(fd)
            self._commit_shard(path, infos, kv)
            return len(infos)
        except Exception:
            self.abort()
            raise

    def finish(self) -> dict:
        """Validate completeness, finalize the .lac (entry table +
        header), and return the stats dict (same shape as
        :func:`convert_sharded`, plus ``shards``). A failed finish
        aborts the session and removes the partial .lac."""
        if self._finished:
            raise ValueError("streaming conversion already finished")
        if self._aborted:
            raise ValueError("streaming conversion was aborted")
        try:
            if self._shard_count == 0:
                raise ValueError(
                    "streaming conversion finished with no shards added"
                )
            if (self._declared_count is not None
                    and self._shard_count != self._declared_count):
                raise ValueError(
                    f"split.count={self._declared_count} but "
                    f"{self._shard_count} shard(s) added (missing shard?)"
                )
            if (self._declared_tcount is not None
                    and self._stats["tensors"] != self._declared_tcount):
                raise ValueError(
                    f"split.tensors.count={self._declared_tcount} but "
                    f"shards hold {self._stats['tensors']} tensors"
                )
            self._writer.close()
            self._finished = True
        except Exception:
            self.abort()
            raise
        self._stats["pages"] = None
        self._stats["shards"] = self._shard_count
        return self._stats

    def abort(self) -> None:
        """Abandon the session: release the writer WITHOUT finalizing
        (no entry table, no header) and remove the partial .lac.
        Idempotent; never raises."""
        if self._aborted or self._finished:
            return
        self._aborted = True
        try:
            self._writer.discard()
        except Exception:
            pass
        try:
            os.remove(self.spp_path)
        except OSError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            if not self._finished:
                self.finish()
        else:
            self.abort()
        return False


def _default_layer(name: str) -> int:
    import re

    m = re.search(r"blk\.(\d+)", name)
    return int(m.group(1)) if m else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Convert GGUF -> Lacuna .lac")
    ap.add_argument("input", nargs="+",
                    help="input .gguf file, or several shard files "
                         "(all but the last positional are shards)")
    ap.add_argument("output", help="output .lac file")
    ap.add_argument("--page-mb", type=int, default=2)
    ap.add_argument("--cold-threshold-mb", type=int, default=1024)
    ap.add_argument(
        "--precision-policy",
        choices=list(_q.PRECISION_POLICIES),
        default=_q.POLICY_EXACT,
        help="exact: bit-identical source bytes (default); extreme: adaptive "
        "Q4/Q2/ternary tiering (fastest, documented quality tradeoff)",
    )
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    if len(a.input) == 1:
        stats = convert(
            a.input[0], a.output, page_mb=a.page_mb,
            cold_threshold_mb=a.cold_threshold_mb, layer_fn=_default_layer,
            precision_policy=a.precision_policy,
        )
    else:
        stats = convert_sharded(
            a.input, a.output, page_mb=a.page_mb,
            cold_threshold_mb=a.cold_threshold_mb, layer_fn=_default_layer,
            precision_policy=a.precision_policy,
        )
    if not a.quiet:
        print(f"done: {stats['tensors']} tensors, tiers={stats['tiers']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
