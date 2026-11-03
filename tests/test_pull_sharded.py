"""Sharded-pull wiring tests: ``lacuna pull <hf-id>`` on a split repo
downloads EVERY shard, verifies sizes, and converts via convert_sharded
— in shard order — to ONE .lac. Single-file pulls are unchanged.

Synthetic shards only (built by test_convert_sharded's writer); the
download layer is mocked, so no network and no real models. The real
``download()`` resume behavior is covered by test_hardening.py; here the
mock stands in for completed/resumed shard bytes.
"""

import importlib
import io
import json
import os
import shutil

import pytest

import lacuna.convert
from lacuna.convert import convert, convert_sharded
from lacuna.pull import (PullError, pull, pull_sharded, resolve_shard_set,
                         shard_info)
from tests.test_convert_sharded import _write_sharded

# lacuna/__init__ rebinds the `lacuna.pull` package attribute to the pull()
# function, so `import lacuna.pull as pull_mod` would yield the function,
# not the module — import_module returns the real module from sys.modules,
# which is what monkeypatch.setattr(pull_mod, "download", ...) needs.
pull_mod = importlib.import_module("lacuna.pull")


class _Resp:
    """Minimal fake HTTP response for the Hub API call."""

    def __init__(self, body: bytes, status: int = 200):
        self._buf = io.BytesIO(body)
        self.status = status

    def getheader(self, name):
        return None

    def read(self, n=-1):
        return self._buf.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _api_opener(siblings):
    """Fake opener serving the siblings JSON for any API URL."""

    def _open(url):
        body = json.dumps({"siblings": siblings}).encode("utf-8")
        return _Resp(body)

    return _open


def _siblings_for(paths):
    return [{"rfilename": os.path.basename(p),
             "size": os.path.getsize(p)} for p in paths]


def _fake_download(src_by_name, calls, truncate=None):
    """Mock download layer: copy a pre-made local file to dest.

    ``truncate``: {basename: bytes_to_keep} to simulate a short download.
    Records basenames in call order. Mirrors download()'s signature.
    """

    def fake_download(url, dest, progress=None, opener=None,
                      download_base=None):
        name = os.path.basename(dest)
        calls.append(name)
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        if truncate and name in truncate:
            with open(src_by_name[name], "rb") as f:
                data = f.read(truncate[name])
            with open(dest, "wb") as f:
                f.write(data)
        else:
            shutil.copyfile(src_by_name[name], dest)
        if progress is not None:
            size = os.path.getsize(dest)
            progress(size, size)
        return dest

    return fake_download


def _build_src(tmp_path, plan=((0, 1, 2), (3, 4, 5))):
    """Synthetic gguf-split shards in tmp_path/src; returns shard paths."""
    src = tmp_path / "src"
    src.mkdir()
    _, shards = _write_sharded(src, [list(p) for p in plan])
    return shards


def _cache_lac_paths(cache_dir):
    out = []
    for root, _, names in os.walk(cache_dir):
        for n in names:
            if n.endswith(".lac"):
                out.append(os.path.join(root, n))
    return out


def test_pull_sharded_downloads_all_and_converts(tmp_path, monkeypatch):
    shards = _build_src(tmp_path)
    # Listing deliberately REVERSED: pull must still download in order.
    siblings = list(reversed(_siblings_for(shards)))
    src_by_name = {os.path.basename(p): p for p in shards}
    calls = []
    monkeypatch.setattr(pull_mod, "download",
                        _fake_download(src_by_name, calls))
    cache = str(tmp_path / "cache")
    lac = pull("owner/repo", cache_dir=cache,
               opener=_api_opener(siblings))
    assert lac.endswith("model.lac")
    assert os.path.exists(lac)
    # Both shards downloaded, in index order, despite reversed listing.
    assert calls == ["model-00001-of-00002.gguf",
                     "model-00002-of-00002.gguf"]
    for p in shards:
        cached = os.path.join(cache, "models", "owner__repo",
                              os.path.basename(p))
        assert os.path.exists(cached)
    # Byte-identical to converting the same shards directly.
    ref = str(tmp_path / "ref.lac")
    convert_sharded(shards, ref, quiet=True)
    with open(lac, "rb") as f, open(ref, "rb") as g:
        assert f.read() == g.read()


def test_pull_sharded_missing_shard_fails_before_download(tmp_path,
                                                          monkeypatch):
    shards = _build_src(tmp_path)
    # Listing declares only shard 1 of 2 — the set is incomplete.
    siblings = _siblings_for(shards[:1])
    calls = []
    monkeypatch.setattr(
        pull_mod, "download",
        _fake_download({os.path.basename(p): p for p in shards}, calls))
    with pytest.raises(PullError, match="incomplete"):
        pull("owner/repo", cache_dir=str(tmp_path / "cache"),
             opener=_api_opener(siblings))
    assert calls == []  # failed before any download started
    assert _cache_lac_paths(str(tmp_path / "cache")) == []


def test_pull_sharded_size_mismatch_fails_before_conversion(tmp_path,
                                                            monkeypatch):
    shards = _build_src(tmp_path)
    siblings = _siblings_for(shards)
    src_by_name = {os.path.basename(p): p for p in shards}
    calls = []
    monkeypatch.setattr(
        pull_mod, "download",
        _fake_download(src_by_name, calls,
                       truncate={"model-00002-of-00002.gguf": 10}))

    def _must_not_convert(*a, **k):
        raise AssertionError("conversion attempted before verification")

    monkeypatch.setattr(lacuna.convert, "convert_sharded", _must_not_convert)
    with pytest.raises(PullError, match="size mismatch"):
        pull("owner/repo", cache_dir=str(tmp_path / "cache"),
             opener=_api_opener(siblings))
    assert _cache_lac_paths(str(tmp_path / "cache")) == []


def test_pull_sharded_calls_convert_sharded_in_order(tmp_path, monkeypatch):
    shards = _build_src(tmp_path)
    siblings = list(reversed(_siblings_for(shards)))
    src_by_name = {os.path.basename(p): p for p in shards}
    monkeypatch.setattr(pull_mod, "download",
                        _fake_download(src_by_name, []))
    recorded = {}

    def _recorder(shard_paths, spp_path, **kwargs):
        recorded["shards"] = [os.path.basename(p) for p in shard_paths]
        with open(spp_path, "wb") as f:
            f.write(b"dummy-lac")
        return {"shards": len(shard_paths)}

    monkeypatch.setattr(lacuna.convert, "convert_sharded", _recorder)
    lac = pull("owner/repo", cache_dir=str(tmp_path / "cache"),
               opener=_api_opener(siblings))
    assert os.path.exists(lac)
    assert recorded["shards"] == ["model-00001-of-00002.gguf",
                                  "model-00002-of-00002.gguf"]


def test_pull_sharded_progress_is_cumulative(tmp_path, monkeypatch):
    shards = _build_src(tmp_path)
    siblings = _siblings_for(shards)
    src_by_name = {os.path.basename(p): p for p in shards}
    monkeypatch.setattr(pull_mod, "download",
                        _fake_download(src_by_name, []))
    seen = []
    pull("owner/repo", cache_dir=str(tmp_path / "cache"),
         opener=_api_opener(siblings),
         progress=lambda d, t: seen.append((d, t)))
    total = sum(os.path.getsize(p) for p in shards)
    assert seen[-1] == (total, total)
    done_vals = [d for d, _ in seen]
    assert done_vals == sorted(done_vals)  # cumulative, never resets


def test_pull_single_file_path_unchanged(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    single, _ = _write_sharded(src, [[0, 1, 2], [3, 4, 5]])
    siblings = _siblings_for([single])
    monkeypatch.setattr(
        pull_mod, "download",
        _fake_download({os.path.basename(single): single}, []))

    def _must_not_convert_sharded(*a, **k):
        raise AssertionError("sharded path taken for a single file")

    monkeypatch.setattr(lacuna.convert, "convert_sharded",
                        _must_not_convert_sharded)
    cache = str(tmp_path / "cache")
    lac = pull("owner/repo", cache_dir=cache, opener=_api_opener(siblings))
    assert os.path.exists(lac)
    ref = str(tmp_path / "ref.lac")
    convert(single, ref, quiet=True)
    with open(lac, "rb") as f, open(ref, "rb") as g:
        assert f.read() == g.read()


def test_pull_sharded_idempotent_repull(tmp_path, monkeypatch):
    shards = _build_src(tmp_path)
    siblings = _siblings_for(shards)
    src_by_name = {os.path.basename(p): p for p in shards}
    calls = []
    monkeypatch.setattr(pull_mod, "download",
                        _fake_download(src_by_name, calls))
    cache = str(tmp_path / "cache")
    opener = _api_opener(siblings)
    lac1 = pull("owner/repo", cache_dir=cache, opener=opener)
    assert len(calls) == 2
    lac2 = pull("owner/repo", cache_dir=cache, opener=opener)
    assert lac2 == lac1
    assert len(calls) == 2  # .lac present: no re-download
    os.remove(lac1)
    lac3 = pull("owner/repo", cache_dir=cache, opener=opener)
    assert lac3 == lac1
    assert os.path.exists(lac3)


def test_pull_sharded_direct_unsorted_input(tmp_path, monkeypatch):
    shards = _build_src(tmp_path)
    entries = list(reversed(_siblings_for(shards)))
    src_by_name = {os.path.basename(p): p for p in shards}
    monkeypatch.setattr(pull_mod, "download",
                        _fake_download(src_by_name, []))
    recorded = {}

    def _recorder(shard_paths, spp_path, **kwargs):
        recorded["shards"] = [os.path.basename(p) for p in shard_paths]
        with open(spp_path, "wb") as f:
            f.write(b"dummy-lac")
        return {"shards": len(shard_paths)}

    monkeypatch.setattr(lacuna.convert, "convert_sharded", _recorder)
    lac = pull_sharded("owner/repo", entries,
                       cache_dir=str(tmp_path / "cache"))
    assert os.path.exists(lac)
    assert recorded["shards"] == ["model-00001-of-00002.gguf",
                                  "model-00002-of-00002.gguf"]


def test_resolve_shard_set_units():
    files = [
        {"rfilename": "m-00002-of-00003.gguf", "size": 20},
        {"rfilename": "m-00001-of-00003.gguf", "size": 10},
        {"rfilename": "m-00003-of-00003.gguf", "size": 30},
    ]
    out = resolve_shard_set(files[0], files)
    assert [f["rfilename"] for f in out] == [
        "m-00001-of-00003.gguf", "m-00002-of-00003.gguf",
        "m-00003-of-00003.gguf"]
    # Non-shard file -> single-file path (None).
    assert resolve_shard_set({"rfilename": "model.gguf", "size": 5},
                             files) is None
    # Degenerate single shard -> single-file path (None).
    assert resolve_shard_set({"rfilename": "m-00001-of-00001.gguf",
                              "size": 5}, files) is None
    # Totals disagree within one stem -> fail loud.
    bad = files + [{"rfilename": "m-00001-of-00004.gguf", "size": 10}]
    with pytest.raises(PullError, match="inconsistent"):
        resolve_shard_set(files[0], bad)
    # Gap in indices -> fail loud.
    gap = [f for f in files if "00002" not in f["rfilename"]]
    with pytest.raises(PullError, match="incomplete"):
        resolve_shard_set(gap[0], gap)
    # Duplicate index -> fail loud (indices cannot cover 1..total).
    dup = files + [{"rfilename": "m-00002-of-00003.gguf", "size": 20}]
    with pytest.raises(PullError, match="incomplete"):
        resolve_shard_set(files[0], dup)


def test_shard_info_units():
    assert shard_info("model-00001-of-00002.gguf") == (1, 2)
    assert shard_info("sub/model-00012-of-00100.gguf") == (12, 100)
    assert shard_info("model.gguf") is None
    assert shard_info("shard1.gguf") is None
