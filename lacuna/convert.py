"""GGUF -> .lac converter (offline, one pass, streaming).

Parses the GGUF v3 header with ``struct`` (tensor infos only) and streams each
tensor's bytes via ``os.pread`` — the full weight blob is NEVER loaded into RAM
at once, so converting a 25 GB file on an 8 GB box is safe.

Precision tiers (initial; the pager re-tiers by measured frequency at runtime):
- attention / norms / embeddings -> Q4 (hot tier)
- FFN matrices                     -> Q2 (warm tier)
- tensors larger than --cold-threshold-mb -> ternary (cold tier)
- non-F32/F16 GGUF types           -> RAW opaque byte pages (no requantization)

Only F32 (type 0) and F16 (type 1) tensors are requantized; anything else
(Q4_0, Q8_0, ...) is carried through byte-identical as RAW pages. This is
honest: Lacuna does not claim to improve pre-quantized GGUF blocks.

CLI: ``python -m lacuna.convert model.gguf model.lac [--page-mb 2]``
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


def convert(
    gguf_path: str,
    spp_path: str,
    page_mb: int = 2,
    cold_threshold_mb: int = 1024,
    layer_fn=None,
    quiet: bool = False,
) -> dict:
    """Convert GGUF -> .lac. Returns stats dict."""
    infos, data_start = parse_gguf_header(gguf_path)
    fd = os.open(gguf_path, os.O_RDONLY)
    cold_threshold = cold_threshold_mb * 1024 * 1024
    stats = {"tensors": 0, "pages": 0, "bytes_in": 0, "tiers": {"q4": 0, "q2": 0, "ternary": 0, "raw": 0}}

    def log(msg):
        if not quiet:
            print(msg, flush=True)

    with LacWriter(spp_path, page_size=page_mb * 1024 * 1024) as w:
        for info in infos:
            name, dims, ttype, off = info["name"], info["dims"], info["type"], info["offset"]
            layer = layer_fn(name) if layer_fn else 0
            if ttype in _DENSE_TYPES:
                dt, es = _DENSE_TYPES[ttype]
                n_elem = 1
                for d in dims:
                    n_elem *= d
                nbytes = n_elem * es
                raw = os.pread(fd, nbytes, data_start + off)
                arr = np.frombuffer(raw, dtype=dt).astype(np.float32)
                # GGUF stores dims reversed vs row-major expectation; treat as 2-D
                # (rows = last dim, cols = product of the rest).
                rows = int(dims[-1]) if dims else 1
                cols = int(n_elem // max(rows, 1))
                arr = arr.reshape(rows, cols)
                prec = _tier_for(name, nbytes, cold_threshold)
                w.add_tensor(name, arr, layer=layer, precision=prec)
                stats["tiers"][_q.precision_name(prec)] += 1
                stats["bytes_in"] += nbytes
                tier_label = _q.precision_name(prec)
            else:
                # Opaque: carry through as RAW byte pages (no requantization).
                # Size unknown without the type table; bound by next tensor or EOF.
                fsize = os.fstat(fd).st_size
                nbytes = fsize - (data_start + off)  # over-estimate; trimmed below
                # find a tighter bound: distance to the next tensor's offset
                later = [i["offset"] for i in infos if i["offset"] > off]
                if later:
                    nbytes = min(nbytes, min(later) - off)
                nbytes = min(nbytes, 1 << 32)
                blob = os.pread(fd, nbytes, data_start + off)
                w.add_raw_bytes(name, blob, layer=layer)
                stats["tiers"]["raw"] += 1
                stats["bytes_in"] += len(blob)
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
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    stats = convert(
        a.input, a.output, page_mb=a.page_mb,
        cold_threshold_mb=a.cold_threshold_mb, layer_fn=_default_layer,
    )
    if not a.quiet:
        print(f"done: {stats['tensors']} tensors, tiers={stats['tiers']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
