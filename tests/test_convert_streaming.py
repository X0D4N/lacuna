"""Streaming shard conversion: one GGUF shard at a time into a growing
.lac, so the caller can delete each shard file between conversions and
stay under a per-kernel writable budget (Kaggle: ~96.3 GiB to /tmp per
kernel, measured twice, while df still showed ~953 GiB free — the
download-all-then-convert sequence needs 122.8 GiB for the Gemma 31B
BF16 split and cannot succeed there).

Also covers the two bundled fixes: LacWriter.__exit__ must not let a
failing close() mask the primary exception, and pull.py's native-
precision selection option (prefer_native).

Synthetic shards only — no model downloads.
"""

import importlib
import os
import shutil

import numpy as np
import pytest

from lacuna.convert import StreamingShardConverter, convert_sharded
from lacuna.pager import LacWriter
from lacuna.pull import (pick_gguf, pull, resolve_shard_set,
                         _is_native)

try:
    from tests.test_convert_sharded import (
        _general_kv, _model_blobs, _read_back_exact,
        _split_kv, _write_gguf, _write_sharded)
    from tests.test_pull_sharded import (
        _api_opener, _fake_download, _siblings_for)
except ImportError:  # run from inside tests/
    from test_convert_sharded import (
        _general_kv, _model_blobs, _read_back_exact,
        _split_kv, _write_gguf, _write_sharded)
    from test_pull_sharded import (
        _api_opener, _fake_download, _siblings_for)

# lacuna/__init__ rebinds the `lacuna.pull` package attribute to the pull()
# function; import_module returns the real module, which monkeypatch needs.
pull_mod = importlib.import_module("lacuna.pull")


def _assert_same_lac(a_path, b_path):
    with open(a_path, "rb") as f:
        a = f.read()
    with open(b_path, "rb") as f:
        b = f.read()
    assert a == b


# ---------------------------------------------------------------------------
# Streaming converter: byte-identical output
# ---------------------------------------------------------------------------

def test_streaming_exact_byte_identical_to_oneshot(tmp_path):
    _, shards = _write_sharded(tmp_path, [[0, 1, 2], [3, 4, 5]])
    out_one = str(tmp_path / "one.lac")
    out_stream = str(tmp_path / "stream.lac")
    convert_sharded(shards, out_one, page_mb=1, quiet=True)
    with StreamingShardConverter(out_stream, page_mb=1,
                                 quiet=True) as s:
        assert s.add_shard(shards[0]) == 3
        assert s.add_shard(shards[1]) == 3
    _assert_same_lac(out_one, out_stream)
    blobs = _model_blobs()
    names, back = _read_back_exact(out_stream)
    assert names == [b[0] for b in blobs]
    for name, _, _, data in blobs:
        assert back[name] == data


def test_streaming_extreme_byte_identical_to_oneshot(tmp_path):
    _, shards = _write_sharded(tmp_path, [[0], [1, 2, 3], [4, 5]])
    out_one = str(tmp_path / "one.lac")
    out_stream = str(tmp_path / "stream.lac")
    st1 = convert_sharded(shards, out_one, page_mb=1, quiet=True,
                          precision_policy="extreme")
    with StreamingShardConverter(out_stream, page_mb=1, quiet=True,
                                 precision_policy="extreme") as s:
        s.add_shard(shards[0])
        s.add_shard(shards[1])
        s.add_shard(shards[2])
        st2 = s.finish()
    assert st1["tiers"] == st2["tiers"]
    assert st2["shards"] == 3 and st2["tensors"] == 6
    _assert_same_lac(out_one, out_stream)


def test_streaming_delete_between_shards(tmp_path):
    """The Kaggle flow: shard 1's file is gone before shard 2 converts,
    yet the output is still byte-identical to the one-shot conversion."""
    _, shards = _write_sharded(tmp_path, [[0, 1, 2], [3, 4, 5]])
    out_one = str(tmp_path / "one.lac")
    out_stream = str(tmp_path / "stream.lac")
    convert_sharded(shards, out_one, page_mb=1, quiet=True)
    s = StreamingShardConverter(out_stream, page_mb=1, quiet=True)
    s.add_shard(shards[0])
    os.remove(shards[0])  # deleted before shard 2 is even touched
    assert not os.path.exists(shards[0])
    s.add_shard(shards[1])
    s.finish()
    _assert_same_lac(out_one, out_stream)


def test_streaming_filename_order(tmp_path):
    _, shards = _write_sharded(tmp_path, [[0, 1, 2], [3, 4, 5]],
                               style="plain")
    plain0 = str(tmp_path / "model-00001-of-00002.gguf")
    plain1 = str(tmp_path / "model-00002-of-00002.gguf")
    os.rename(shards[0], plain0)
    os.rename(shards[1], plain1)
    out_one = str(tmp_path / "one.lac")
    out_stream = str(tmp_path / "stream.lac")
    convert_sharded([plain1, plain0], out_one, page_mb=1, quiet=True)
    with StreamingShardConverter(out_stream, page_mb=1,
                                 quiet=True) as s:
        s.add_shard(plain0)
        s.add_shard(plain1)
    _assert_same_lac(out_one, out_stream)


# ---------------------------------------------------------------------------
# Streaming converter: fail-loud, partial .lac removed
# ---------------------------------------------------------------------------

def test_streaming_out_of_order_split_no_fails(tmp_path):
    _, shards = _write_sharded(tmp_path, [[0, 1, 2], [3, 4, 5]])
    out = str(tmp_path / "out.lac")
    s = StreamingShardConverter(out, page_mb=1, quiet=True)
    with pytest.raises(ValueError, match="must start at split.no=0"):
        s.add_shard(shards[1])  # split.no=1 first
    assert not os.path.exists(out)


def test_streaming_bad_split_no_on_second_shard(tmp_path):
    blobs = _model_blobs()
    p0 = str(tmp_path / "shard0.gguf")
    p1 = str(tmp_path / "shard1.gguf")
    _write_gguf(p0, blobs[:3], _split_kv(0, 2, len(blobs)))
    # Gap: second shard claims split.no=5, not 1.
    _write_gguf(p1, blobs[3:], _split_kv(5, 2, len(blobs)))
    out = str(tmp_path / "out.lac")
    s = StreamingShardConverter(out, page_mb=1, quiet=True)
    s.add_shard(p0)
    with pytest.raises(ValueError, match="out of order"):
        s.add_shard(p1)
    assert not os.path.exists(out)


def test_streaming_missing_last_shard_finish_fails(tmp_path):
    _, shards = _write_sharded(tmp_path, [[0, 1, 2], [3, 4, 5]])
    out = str(tmp_path / "out.lac")
    s = StreamingShardConverter(out, page_mb=1, quiet=True)
    s.add_shard(shards[0])
    with pytest.raises(ValueError, match="split.count=2"):
        s.finish()
    assert not os.path.exists(out)


def test_streaming_duplicate_tensor_fails(tmp_path):
    blobs = _model_blobs()
    p0 = str(tmp_path / "shard0.gguf")
    p1 = str(tmp_path / "shard1.gguf")
    # Declare the true merged count (7, duplicate included) so the count
    # gate passes and the duplicate gate is what fires.
    _write_gguf(p0, blobs[:3], _split_kv(0, 2, 7))
    _write_gguf(p1, blobs[2:], _split_kv(1, 2, 7))  # blobs[2] twice
    out = str(tmp_path / "out.lac")
    s = StreamingShardConverter(out, page_mb=1, quiet=True)
    s.add_shard(p0)
    with pytest.raises(ValueError, match="duplicated"):
        s.add_shard(p1)
    assert not os.path.exists(out)


def test_streaming_metadata_disagrees_fails(tmp_path):
    blobs = _model_blobs()
    p0 = str(tmp_path / "shard0.gguf")
    p1 = str(tmp_path / "shard1.gguf")
    _write_gguf(p0, blobs[:3],
                _general_kv("testarch") + _split_kv(0, 2, len(blobs)))
    _write_gguf(p1, blobs[3:],
                _general_kv("otherarch") + _split_kv(1, 2, len(blobs)))
    out = str(tmp_path / "out.lac")
    s = StreamingShardConverter(out, page_mb=1, quiet=True)
    s.add_shard(p0)
    with pytest.raises(ValueError, match="disagrees across shards"):
        s.add_shard(p1)
    assert not os.path.exists(out)


def test_streaming_truncated_shard_data_fails(tmp_path):
    # Shard 2's tensors are dense-only; cutting its data tail must raise
    # a short-read OSError (not silently convert a clipped tensor).
    _, shards = _write_sharded(tmp_path, [[0, 1, 2], [3, 4]])
    trunc = str(tmp_path / "shard1-trunc.gguf")
    shutil.copyfile(shards[1], trunc)
    with open(trunc, "r+b") as f:
        f.truncate(os.path.getsize(trunc) - 5)
    out = str(tmp_path / "out.lac")
    s = StreamingShardConverter(out, page_mb=1, quiet=True)
    s.add_shard(shards[0])
    with pytest.raises(OSError, match="short read"):
        s.add_shard(trunc)
    assert not os.path.exists(out)


def test_streaming_tensors_count_mismatch_at_finish(tmp_path):
    blobs = _model_blobs()
    p0 = str(tmp_path / "shard0.gguf")
    p1 = str(tmp_path / "shard1.gguf")
    _write_gguf(p0, blobs[:3], _split_kv(0, 2, 99))
    _write_gguf(p1, blobs[3:], _split_kv(1, 2, 99))
    out = str(tmp_path / "out.lac")
    s = StreamingShardConverter(out, page_mb=1, quiet=True)
    s.add_shard(p0)
    s.add_shard(p1)
    with pytest.raises(ValueError, match="split.tensors.count=99"):
        s.finish()
    assert not os.path.exists(out)


def test_streaming_reversed_filename_order_fails(tmp_path):
    _, shards = _write_sharded(tmp_path, [[0, 1, 2], [3, 4, 5]],
                               style="plain")
    plain0 = str(tmp_path / "model-00001-of-00002.gguf")
    plain1 = str(tmp_path / "model-00002-of-00002.gguf")
    os.rename(shards[0], plain0)
    os.rename(shards[1], plain1)
    out = str(tmp_path / "out.lac")
    s = StreamingShardConverter(out, page_mb=1, quiet=True)
    with pytest.raises(ValueError, match="must start at shard filename"):
        s.add_shard(plain1)
    assert not os.path.exists(out)


def test_streaming_context_manager(tmp_path):
    _, shards = _write_sharded(tmp_path, [[0, 1, 2], [3, 4, 5]])
    out_one = str(tmp_path / "one.lac")
    out_stream = str(tmp_path / "stream.lac")
    convert_sharded(shards, out_one, page_mb=1, quiet=True)
    # Clean exit auto-finishes.
    with StreamingShardConverter(out_stream, page_mb=1, quiet=True) as s:
        s.add_shard(shards[0])
        s.add_shard(shards[1])
    _assert_same_lac(out_one, out_stream)
    # Exception inside the block aborts and removes the partial file.
    out_bad = str(tmp_path / "bad.lac")
    with pytest.raises(RuntimeError, match="boom"):
        with StreamingShardConverter(out_bad, page_mb=1, quiet=True) as s:
            s.add_shard(shards[0])
            raise RuntimeError("boom")
    assert not os.path.exists(out_bad)


def test_streaming_lifecycle_guards(tmp_path):
    _, shards = _write_sharded(tmp_path, [[0, 1, 2], [3, 4, 5]])
    out = str(tmp_path / "out.lac")
    s = StreamingShardConverter(out, page_mb=1, quiet=True)
    s.add_shard(shards[0])
    s.add_shard(shards[1])
    s.finish()
    with pytest.raises(ValueError, match="already finished"):
        s.add_shard(shards[0])
    with pytest.raises(ValueError, match="already finished"):
        s.finish()
    # Abort is idempotent and blocks further use.
    out2 = str(tmp_path / "out2.lac")
    s2 = StreamingShardConverter(out2, page_mb=1, quiet=True)
    s2.abort()
    s2.abort()  # no raise
    assert not os.path.exists(out2)
    with pytest.raises(ValueError, match="aborted"):
        s2.add_shard(shards[0])


# ---------------------------------------------------------------------------
# LacWriter: __exit__ must not mask the primary exception
# ---------------------------------------------------------------------------

def test_lacwriter_exit_does_not_mask_primary(tmp_path):
    p = str(tmp_path / "x.lac")
    # Under the old __exit__ (close() unconditionally), the OSError from
    # the failing table write would replace the RuntimeError below.
    with pytest.raises(RuntimeError, match="primary"):
        with LacWriter(p, page_size=512, precision_policy="exact") as w:
            w.add_tensor("t", np.zeros((2, 2), dtype=np.float32))

            def boom(*a, **k):
                raise OSError("disk gone")

            w._f.write = boom
            raise RuntimeError("primary")
    # Discard path: file left unfinalized (zero header), never a table.
    with open(p, "rb") as f:
        assert f.read(4) == b"\x00\x00\x00\x00"


def test_lacwriter_close_idempotent_and_discard(tmp_path):
    p = str(tmp_path / "x.lac")
    w = LacWriter(p, page_size=512, precision_policy="exact")
    w.add_tensor("t", np.zeros((2, 2), dtype=np.float32))
    w.close()
    w.close()  # no raise
    with open(p, "rb") as f:
        assert f.read(4) == b"LAC1"
    p2 = str(tmp_path / "y.lac")
    w2 = LacWriter(p2, page_size=512, precision_policy="exact")
    w2.add_tensor("t", np.zeros((2, 2), dtype=np.float32))
    w2.discard()
    w2.discard()  # idempotent, no raise
    with open(p2, "rb") as f:
        assert f.read(4) == b"\x00\x00\x00\x00"


# ---------------------------------------------------------------------------
# pull.py: prefer_native selection
# ---------------------------------------------------------------------------

def test_is_native_tokens():
    assert _is_native("model-bf16.gguf")
    assert _is_native("gemma-4-31B-it-bf16-00001-of-00002.gguf")
    assert _is_native("model-f16.gguf")
    assert _is_native("model-fp16.gguf")
    assert not _is_native("model-Q4_K_M.gguf")
    assert not _is_native("model-Q8_0.gguf")
    assert not _is_native("model-UD-Q4_K_XL.gguf")
    assert not _is_native("model.gguf")


def test_pick_prefer_native_mixed_listing():
    files = [
        {"rfilename": "model-Q4_K_M.gguf", "size": 10},
        {"rfilename": "model-bf16-00001-of-00002.gguf", "size": 40},
        {"rfilename": "model-bf16-00002-of-00002.gguf", "size": 21},
    ]
    # Default: Q4_K_M first, as before.
    assert pick_gguf(files)["rfilename"] == "model-Q4_K_M.gguf"
    # Native: smallest BF16 file (a shard); resolve_shard_set expands it.
    chosen = pick_gguf(files, prefer_native=True)
    assert chosen["rfilename"] == "model-bf16-00002-of-00002.gguf"
    shard_set = resolve_shard_set(chosen, files)
    assert [f["rfilename"] for f in shard_set] == [
        "model-bf16-00001-of-00002.gguf",
        "model-bf16-00002-of-00002.gguf"]


def test_pick_prefer_native_falls_back_without_native():
    files = [
        {"rfilename": "m-Q8_0.gguf", "size": 100},
        {"rfilename": "m-Q4_K_M.gguf", "size": 60},
    ]
    assert pick_gguf(files, prefer_native=True)["rfilename"] == \
        "m-Q4_K_M.gguf"
    # And a native single file beats a smaller quant file.
    files2 = [
        {"rfilename": "m-Q4_K_M.gguf", "size": 10},
        {"rfilename": "m-bf16.gguf", "size": 90},
    ]
    assert pick_gguf(files2, prefer_native=True)["rfilename"] == \
        "m-bf16.gguf"
    assert pick_gguf(files2)["rfilename"] == "m-Q4_K_M.gguf"


def test_pull_prefer_native_end_to_end(tmp_path, monkeypatch):
    """Fake hub: one Q4_K_M single + one BF16 shard set. prefer_native
    must download the BF16 set; default must download the quant file."""
    src = tmp_path / "src"
    src.mkdir()
    quant = str(src / "model-Q4_K_M.gguf")
    _write_gguf(quant, _model_blobs(), _general_kv())
    _, shards = _write_sharded(src, [[0, 1, 2], [3, 4, 5]])
    # Rename the shards to BF16 names: the synthetic writer's default
    # names carry no native-precision token, so prefer_native would
    # (correctly) fall back without this. Content is name-independent.
    bf16 = []
    for i, p in enumerate(shards):
        q = str(src / f"model-bf16-{i + 1:05d}-of-00002.gguf")
        os.rename(p, q)
        bf16.append(q)
    siblings = _siblings_for([quant] + bf16)
    src_by_name = {os.path.basename(p): p for p in [quant] + bf16}

    calls = []
    monkeypatch.setattr(pull_mod, "download",
                        _fake_download(src_by_name, calls))
    cache = str(tmp_path / "cache-native")
    lac = pull("owner/repo", cache_dir=cache,
               opener=_api_opener(siblings), prefer_native=True)
    assert os.path.basename(lac) == "model-bf16.lac"  # sharded stem
    assert sorted(calls) == ["model-bf16-00001-of-00002.gguf",
                             "model-bf16-00002-of-00002.gguf"]
    names, _ = _read_back_exact(lac)
    assert names == [b[0] for b in _model_blobs()]

    calls.clear()
    monkeypatch.setattr(pull_mod, "download",
                        _fake_download(src_by_name, calls))
    cache2 = str(tmp_path / "cache-default")
    lac2 = pull("owner/repo", cache_dir=cache2,
                opener=_api_opener(siblings))
    assert os.path.basename(lac2) == "model-Q4_K_M.lac"
    assert calls == ["model-Q4_K_M.gguf"]
