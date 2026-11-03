"""Sharded-GGUF converter tests: N shards -> ONE .lac, byte-identical to
converting the equivalent single file, and fail-loud on any split
inconsistency. Synthetic shards only — no model downloads.

Shard convention under test (llama.cpp gguf-split):
- split.no (uint16): 0-based shard index.
- split.count (uint16): total shard count.
- split.tensors.count (int32): TOTAL tensor count across all shards.
- Tensors are whole per shard, in original order across shards.
- Shard 0 carries the full KV; later shards carry just the split keys.
"""

import os
import struct

import numpy as np
import pytest

from lacuna.convert import convert, convert_sharded, main as convert_main
from lacuna.pager import PagedWeightStore

# GGUF metadata value types used by the test writers.
_T_STRING = 8
_T_U16 = 2
_T_I32 = 5
_T_U32 = 4

# (name, rows, cols, gguf_type) for the F32 model tensors; dims are stored
# GGUF-order (cols first, rows last) so convert's rows=dims[-1] holds.
_F32_TENSORS = [
    ("token_embd.weight", 5, 7),
    ("blk.0.attn.wq", 8, 4),
    ("blk.0.ffn.w1", 6, 4),
    ("blk.1.attn.wq", 8, 4),
    ("blk.1.ffn.w1", 6, 4),
]


def _model_blobs():
    """Deterministic (name, dims, type, data_bytes) for the whole model:
    5 F32 tensors + 1 opaque Q4_0-typed tensor (opaque sizing path)."""
    rng = np.random.default_rng(11)
    blobs = []
    for name, rows, cols in _F32_TENSORS:
        arr = rng.standard_normal((rows, cols)).astype(np.float32)
        blobs.append((name, [cols, rows], 0, arr.tobytes()))
    opaque = rng.integers(0, 256, size=100, dtype=np.uint8).tobytes()
    blobs.append(("blk.1.ffn.wq", [100, 1], 2, opaque))
    return blobs


def _kv_str(f, key, value):
    b = key.encode()
    f.write(struct.pack("<Q", len(b)) + b)
    f.write(struct.pack("<I", _T_STRING))
    vb = value.encode()
    f.write(struct.pack("<Q", len(vb)) + vb)


def _kv_int(f, key, vtype, fmt, value):
    b = key.encode()
    f.write(struct.pack("<Q", len(b)) + b)
    f.write(struct.pack("<I", vtype))
    f.write(struct.pack(fmt, value))


def _write_gguf(path, blobs, kv_entries):
    """blobs: (name, dims, type, data_bytes). kv_entries: list of callables
    taking the open file and writing one KV entry each."""
    with open(path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))  # version
        f.write(struct.pack("<Q", len(blobs)))
        f.write(struct.pack("<Q", len(kv_entries)))
        for entry in kv_entries:
            entry(f)
        off = 0
        for name, dims, ttype, data in blobs:
            b = name.encode()
            f.write(struct.pack("<Q", len(b)) + b)
            f.write(struct.pack("<I", len(dims)))
            for d in dims:
                f.write(struct.pack("<Q", d))
            f.write(struct.pack("<I", ttype))
            f.write(struct.pack("<Q", off))
            off += len(data)
        pos = f.tell()
        f.write(b"\x00" * ((-pos) % 32))
        for _, _, _, data in blobs:
            f.write(data)


def _general_kv(arch="testarch"):
    return [
        lambda f: _kv_str(f, "general.architecture", arch),
        lambda f: _kv_int(f, "general.file_type", _T_U32, "<I", 0),
    ]


def _split_kv(shard_no, n_shards, n_tensors_total):
    return [
        lambda f: _kv_int(f, "split.no", _T_U16, "<H", shard_no),
        lambda f: _kv_int(f, "split.count", _T_U16, "<H", n_shards),
        lambda f: _kv_int(f, "split.tensors.count", _T_I32, "<i",
                          n_tensors_total),
    ]


def _write_sharded(tmp_path, plan, style="gguf-split"):
    """Write one single file + shards for the synthetic model.

    plan: list of index-lists, one per shard (indices into _model_blobs()).
    style "gguf-split": shard 0 carries general + split KV, later shards
    split KV only, filenames model-0000N-of-0000M.gguf.
    style "plain": no split metadata anywhere, filenames shardN.gguf.
    style "full-kv": every shard carries general + split KV.
    Returns (single_path, [shard_paths in plan order])."""
    blobs = _model_blobs()
    single = str(tmp_path / "model.gguf")
    _write_gguf(single, blobs, _general_kv())
    n_shards = len(plan)
    shard_paths = []
    for i, idxs in enumerate(plan):
        sub = [blobs[j] for j in idxs]
        if style == "gguf-split":
            kv = _general_kv() + _split_kv(i, n_shards, len(blobs)) \
                if i == 0 else _split_kv(i, n_shards, len(blobs))
            name = f"model-{i + 1:05d}-of-{n_shards:05d}.gguf"
        elif style == "full-kv":
            kv = _general_kv() + _split_kv(i, n_shards, len(blobs))
            name = f"shard{i}.gguf"
        else:  # plain
            kv = []
            name = f"shard{i}.gguf"
        path = str(tmp_path / name)
        _write_gguf(path, sub, kv)
        shard_paths.append(path)
    return single, shard_paths


def _read_back_exact(spp):
    """{tensor_name: concatenated raw page bytes} for an exact-policy .lac."""
    with PagedWeightStore(spp, hot_budget_bytes=10**9) as s:
        names = list(s.names)
        out = {}
        for tid, name in enumerate(names):
            pages = [i for i, e in enumerate(s._entries)
                     if e["tensor_id"] == tid]
            out[name] = b"".join(s.get_page(i).tobytes() for i in pages)
    return names, out


def _assert_same_lac(a_path, b_path):
    with open(a_path, "rb") as f:
        a = f.read()
    with open(b_path, "rb") as f:
        b = f.read()
    assert a == b


def test_sharded_exact_byte_identical_to_single(tmp_path):
    single, shards = _write_sharded(tmp_path, [[0, 1, 2], [3, 4, 5]])
    out_single = str(tmp_path / "single.lac")
    out_shards = str(tmp_path / "shards.lac")
    st1 = convert(single, out_single, page_mb=1, quiet=True)  # exact default
    st2 = convert_sharded(shards, out_shards, page_mb=1, quiet=True)
    assert st1["tensors"] == st2["tensors"] == 6
    assert st2["shards"] == 2
    _assert_same_lac(out_single, out_shards)

    blobs = _model_blobs()
    names, back = _read_back_exact(out_shards)
    assert names == [b[0] for b in blobs]
    for name, _, _, data in blobs:
        assert back[name] == data


def test_sharded_extreme_byte_identical_to_single(tmp_path):
    single, shards = _write_sharded(tmp_path, [[0, 1], [2, 3, 4, 5]])
    out_single = str(tmp_path / "single.lac")
    out_shards = str(tmp_path / "shards.lac")
    st1 = convert(single, out_single, page_mb=1, quiet=True,
                  precision_policy="extreme")
    st2 = convert_sharded(shards, out_shards, page_mb=1, quiet=True,
                          precision_policy="extreme")
    assert st1["tiers"] == st2["tiers"]
    _assert_same_lac(out_single, out_shards)
    with PagedWeightStore(out_shards, hot_budget_bytes=10**9) as s:
        assert s.names == [b[0] for b in _model_blobs()]
        assert np.all(np.isfinite(s.get_page(0)))


def test_shard_order_from_split_no(tmp_path):
    single, shards = _write_sharded(tmp_path, [[0, 1, 2], [3, 4, 5]])
    out_single = str(tmp_path / "single.lac")
    out_rev = str(tmp_path / "rev.lac")
    convert(single, out_single, page_mb=1, quiet=True)
    # Shards passed backwards: split.no metadata must restore order.
    convert_sharded([shards[1], shards[0]], out_rev, page_mb=1, quiet=True)
    _assert_same_lac(out_single, out_rev)


def test_shard_order_from_filename(tmp_path):
    # No split metadata at all: -NNNNN-of-MMMMM filenames order the shards.
    _, shards = _write_sharded(tmp_path, [[0, 1, 2], [3, 4, 5]], style="plain")
    plain0 = str(tmp_path / "model-00001-of-00002.gguf")
    plain1 = str(tmp_path / "model-00002-of-00002.gguf")
    os.rename(shards[0], plain0)
    os.rename(shards[1], plain1)
    # Build the equivalent single file directly (no split metadata).
    blobs = _model_blobs()
    single_path = str(tmp_path / "model.gguf")
    _write_gguf(single_path, blobs, _general_kv())
    out_single = str(tmp_path / "single.lac")
    out_rev = str(tmp_path / "rev.lac")
    convert(single_path, out_single, page_mb=1, quiet=True)
    convert_sharded([plain1, plain0], out_rev, page_mb=1, quiet=True)
    _assert_same_lac(out_single, out_rev)


def test_three_shards_uneven_and_opaque(tmp_path):
    single, shards = _write_sharded(tmp_path, [[0], [1, 2, 3], [4, 5]])
    out_single = str(tmp_path / "single.lac")
    out_shards = str(tmp_path / "shards.lac")
    convert(single, out_single, page_mb=1, quiet=True)
    st = convert_sharded(shards, out_shards, page_mb=1, quiet=True)
    assert st["shards"] == 3 and st["tensors"] == 6
    _assert_same_lac(out_single, out_shards)


def test_missing_shard_split_count(tmp_path):
    # split.count=3 declared, only 2 shard files given.
    blobs = _model_blobs()
    paths = []
    for i, idxs in enumerate([[0, 1, 2], [3, 4, 5]]):
        p = str(tmp_path / f"shard{i}.gguf")
        _write_gguf(p, [blobs[j] for j in idxs],
                    _split_kv(i, 3, len(blobs)))
        paths.append(p)
    with pytest.raises(ValueError, match="split.count=3"):
        convert_sharded(paths, str(tmp_path / "out.lac"), quiet=True)


def test_duplicate_tensor_across_shards(tmp_path):
    blobs = _model_blobs()
    p0 = str(tmp_path / "shard0.gguf")
    p1 = str(tmp_path / "shard1.gguf")
    # Declare the true merged count (7, duplicate included) so the count
    # gate passes and the duplicate gate is what fires.
    _write_gguf(p0, blobs[:3], _split_kv(0, 2, 7))
    _write_gguf(p1, blobs[2:], _split_kv(1, 2, 7))  # blobs[2] twice
    with pytest.raises(ValueError, match="duplicated"):
        convert_sharded([p0, p1], str(tmp_path / "out.lac"), quiet=True)


def test_duplicate_tensor_dims_disagree(tmp_path):
    blobs = _model_blobs()
    p0 = str(tmp_path / "shard0.gguf")
    p1 = str(tmp_path / "shard1.gguf")
    _write_gguf(p0, blobs[:3], _split_kv(0, 2, 7))
    # Same name as blobs[2] but different dims (and matching byte count).
    name, _, _, data = blobs[2]
    twin = (name, [4, 6], 0, data)
    _write_gguf(p1, [twin] + blobs[3:], _split_kv(1, 2, 7))
    with pytest.raises(ValueError, match="dims"):
        convert_sharded([p0, p1], str(tmp_path / "out.lac"), quiet=True)


def test_split_count_disagrees(tmp_path):
    blobs = _model_blobs()
    p0 = str(tmp_path / "shard0.gguf")
    p1 = str(tmp_path / "shard1.gguf")
    _write_gguf(p0, blobs[:3], _split_kv(0, 2, len(blobs)))
    _write_gguf(p1, blobs[3:], _split_kv(1, 3, len(blobs)))  # says 3, other 2
    with pytest.raises(ValueError, match="split.count disagrees"):
        convert_sharded([p0, p1], str(tmp_path / "out.lac"), quiet=True)


def test_split_tensors_count_mismatch(tmp_path):
    blobs = _model_blobs()
    p0 = str(tmp_path / "shard0.gguf")
    p1 = str(tmp_path / "shard1.gguf")
    _write_gguf(p0, blobs[:3], _split_kv(0, 2, 99))  # declares 99, holds 6
    _write_gguf(p1, blobs[3:], _split_kv(1, 2, 99))
    with pytest.raises(ValueError, match="split.tensors.count=99"):
        convert_sharded([p0, p1], str(tmp_path / "out.lac"), quiet=True)


def test_split_no_gap(tmp_path):
    blobs = _model_blobs()
    p0 = str(tmp_path / "shardA.gguf")
    p1 = str(tmp_path / "shardB.gguf")
    # nos 0 and 2, no count: shard 1 is missing and nothing says otherwise.
    _write_gguf(p0, blobs[:3],
                [lambda f: _kv_int(f, "split.no", _T_U16, "<H", 0)])
    _write_gguf(p1, blobs[3:],
                [lambda f: _kv_int(f, "split.no", _T_U16, "<H", 2)])
    with pytest.raises(ValueError, match="do not cover"):
        convert_sharded([p0, p1], str(tmp_path / "out.lac"), quiet=True)


def test_metadata_disagrees_across_shards(tmp_path):
    # Same non-split key, different value, on two shards that both carry it.
    blobs = _model_blobs()
    p0 = str(tmp_path / "shard0.gguf")
    p1 = str(tmp_path / "shard1.gguf")
    _write_gguf(p0, blobs[:3],
                _general_kv("testarch") + _split_kv(0, 2, len(blobs)))
    _write_gguf(p1, blobs[3:],
                _general_kv("otherarch") + _split_kv(1, 2, len(blobs)))
    with pytest.raises(ValueError, match="disagrees across shards"):
        convert_sharded([p0, p1], str(tmp_path / "out.lac"), quiet=True)


def test_single_convert_rejects_lone_shard(tmp_path):
    _, shards = _write_sharded(tmp_path, [[0, 1, 2], [3, 4, 5]])
    with pytest.raises(ValueError, match="convert_sharded"):
        convert(shards[0], str(tmp_path / "out.lac"), quiet=True)


def test_empty_shard_list(tmp_path):
    with pytest.raises(ValueError, match="no shard paths"):
        convert_sharded([], str(tmp_path / "out.lac"), quiet=True)


def test_sharded_single_element_list(tmp_path):
    # A one-shard "split" (split.count=1) converts fine via the sharded path.
    blobs = _model_blobs()
    p = str(tmp_path / "only.gguf")
    _write_gguf(p, blobs, _general_kv() + _split_kv(0, 1, len(blobs)))
    out = str(tmp_path / "out.lac")
    st = convert_sharded([p], out, page_mb=1, quiet=True)
    assert st["shards"] == 1 and st["tensors"] == 6
    names, back = _read_back_exact(out)
    assert names == [b[0] for b in blobs]
    for name, _, _, data in blobs:
        assert back[name] == data


def test_cli_sharded_and_single(tmp_path):
    single, shards = _write_sharded(tmp_path, [[0, 1, 2], [3, 4, 5]])
    out_single = str(tmp_path / "cli-single.lac")
    out_shards = str(tmp_path / "cli-shards.lac")
    assert convert_main([single, out_single, "--quiet"]) == 0
    assert convert_main(shards + [out_shards, "--quiet"]) == 0
    _assert_same_lac(out_single, out_shards)
