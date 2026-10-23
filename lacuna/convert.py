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
"""

from __future__ import annotations

import argparse
import os
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


def _tier_for(name: str, nbytes: int, cold_threshold: int) -> int:
    lname = name.lower()
    if nbytes >= cold_threshold:
        return _q.TERNARY
    hot_keys = ("embed", "norm", "attn", "wq", "wk", "wv", "wo", "qkv", "output")
    if any(k in lname for k in hot_keys):
        return _q.Q4
    return _q.Q2


def parse_gguf_header(path: str):
    """Return (tensor_infos, data_start). tensor_infos: list of dicts."""
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
    for _ in range(n_kv):
        _read_str(f)  # key
        (vtype,) = struct.unpack("<I", f.read(4))
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
    precision_policy = _q.validate_policy(precision_policy)
    infos, data_start = parse_gguf_header(gguf_path)
    fd = os.open(gguf_path, os.O_RDONLY)
    fsize = os.fstat(fd).st_size
    cold_threshold = cold_threshold_mb * 1024 * 1024
    stats = {"tensors": 0, "pages": 0, "bytes_in": 0, "tiers": {"q4": 0, "q2": 0, "ternary": 0, "raw": 0},
             "precision_policy": precision_policy}

    def log(msg):
        if not quiet:
            print(msg, flush=True)

    with LacWriter(spp_path, page_size=page_mb * 1024 * 1024,
                   precision_policy=precision_policy) as w:
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
    os.close(fd)
    stats["pages"] = None  # available from the written file; kept simple
    return stats


def _default_layer(name: str) -> int:
    import re

    m = re.search(r"blk\.(\d+)", name)
    return int(m.group(1)) if m else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Convert GGUF -> Lacuna .lac")
    ap.add_argument("input", help="input .gguf file")
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
    stats = convert(
        a.input, a.output, page_mb=a.page_mb,
        cold_threshold_mb=a.cold_threshold_mb, layer_fn=_default_layer,
        precision_policy=a.precision_policy,
    )
    if not a.quiet:
        print(f"done: {stats['tensors']} tensors, tiers={stats['tiers']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
