"""Tests for the 100x ease-of-use program: pull, Ollama shim, auto_tune.

No real network: a local fake HuggingFace Hub server (threading HTTP on
127.0.0.1:0) serves the API + file endpoints, and the Ollama shim is tested
against ``build_ollama_server`` on a free port. The one end-to-end pull test
converts a tiny but VALID GGUF (real v3 header, real F32 tensor) through the
real ``lacuna.convert`` — nothing is faked in the conversion path.
"""

import json
import os
import struct
import threading
import urllib.error
import urllib.request
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from lacuna import (
    InferenceBackend,
    PullError,
    auto_tune,
    build_ollama_server,
    download,
    list_gguf_files,
    pick_gguf,
    pull,
)
from lacuna.hardware import HardwareInfo
from lacuna.cli import build_parser


# ---------------------------------------------------------------------------
# Fake HuggingFace Hub (local HTTP server, no real network)
# ---------------------------------------------------------------------------

class _FakeHubHandler(BaseHTTPRequestHandler):
    # per-test state, reset by _make_hub()
    repos: dict = {}        # hf_id -> {"siblings": [...], "files": {name: bytes}}
    received: list = []     # [(path, {header: value})]
    hits: Counter = Counter()
    ignore_range: bool = False

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _bytes(self, data: bytes, code=200, extra=()):
        self.send_response(code)
        for k, v in extra:
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        type(self).received.append((self.path, dict(self.headers)))
        type(self).hits[self.path] += 1
        if self.path.startswith("/api/models/"):
            hf_id = self.path[len("/api/models/"):]
            repo = type(self).repos.get(hf_id)
            if repo is None:
                self._json({"error": "not found"}, 404)
            else:
                self._json({"siblings": repo["siblings"]})
            return
        if "/resolve/main/" in self.path:
            # /{hf_id}/resolve/main/{filename}
            rest = self.path[1:]
            hf_id, _, filename = rest.partition("/resolve/main/")
            repo = type(self).repos.get(hf_id)
            data = repo["files"].get(filename) if repo else None
            if data is None:
                self._json({"error": "not found"}, 404)
                return
            rng = self.headers.get("Range", "")
            if rng.startswith("bytes=") and not type(self).ignore_range:
                try:
                    start = int(rng[len("bytes="):].rstrip("-"))
                except ValueError:
                    start = 0
                if 0 < start < len(data):
                    part = data[start:]
                    self._bytes(part, 206, [
                        ("Content-Range",
                         f"bytes {start}-{len(data)-1}/{len(data)}"),
                        ("Accept-Ranges", "bytes"),
                    ])
                    return
            self._bytes(data)
            return
        self._json({"error": "not found"}, 404)


def _make_hub(repos, ignore_range=False):
    _FakeHubHandler.repos = repos
    _FakeHubHandler.received = []
    _FakeHubHandler.hits = Counter()
    _FakeHubHandler.ignore_range = ignore_range
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _FakeHubHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    return srv, base


def _repo(files: dict) -> dict:
    return {
        "siblings": [{"rfilename": n, "size": len(b)} for n, b in files.items()],
        "files": files,
    }


# ---------------------------------------------------------------------------
# Tiny but VALID GGUF (v3, 0 KV, 1x F32 tensor) for the honest end-to-end test
# ---------------------------------------------------------------------------

def _tiny_gguf() -> bytes:
    name = b"blk.0.attn.wq"
    data = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
    out = bytearray()
    out += b"GGUF"
    out += struct.pack("<I", 3)     # version
    out += struct.pack("<Q", 1)     # n_tensors
    out += struct.pack("<Q", 0)     # n_kv
    out += struct.pack("<Q", len(name)) + name
    out += struct.pack("<I", 1)     # n_dims
    out += struct.pack("<Q", 4)     # dims[0]
    out += struct.pack("<I", 0)     # type: F32
    out += struct.pack("<Q", 0)     # offset
    out += b"\x00" * ((-len(out)) % 32)  # align data section to 32
    out += data
    return bytes(out)


# ---------------------------------------------------------------------------
# pick_gguf: pure file-pick logic
# ---------------------------------------------------------------------------

def test_pick_prefers_q4_k_m():
    files = [
        {"rfilename": "m-Q8_0.gguf", "size": 100},
        {"rfilename": "m-Q4_K_M.gguf", "size": 60},
        {"rfilename": "m-Q4_K_S.gguf", "size": 50},
    ]
    assert pick_gguf(files)["rfilename"] == "m-Q4_K_M.gguf"


def test_pick_smallest_fallback_without_q4_k_m():
    files = [
        {"rfilename": "m-Q8_0.gguf", "size": 100},
        {"rfilename": "m-Q5_K_M.gguf", "size": 70},
    ]
    assert pick_gguf(files)["rfilename"] == "m-Q5_K_M.gguf"


def test_pick_empty_raises():
    with pytest.raises(PullError):
        pick_gguf([])


# ---------------------------------------------------------------------------
# list_gguf_files against the fake hub
# ---------------------------------------------------------------------------

def test_list_gguf_filters_and_404_is_clean():
    srv, base = _make_hub({"acme/tiny": _repo({"tiny.gguf": b"xx",
                                               "notes.txt": b"hi"})})
    try:
        files = list_gguf_files("acme/tiny", api_base=base)
        assert files == [{"rfilename": "tiny.gguf", "size": 2}]
        with pytest.raises(PullError) as e:
            list_gguf_files("acme/missing", api_base=base)
        assert "not found" in str(e.value)
    finally:
        srv.shutdown()


def test_list_gguf_no_gguf_is_clean():
    srv, base = _make_hub({"acme/empty": _repo({"readme.md": b"hi"})})
    try:
        with pytest.raises(PullError) as e:
            list_gguf_files("acme/empty", api_base=base)
        assert "no .gguf" in str(e.value)
    finally:
        srv.shutdown()


# ---------------------------------------------------------------------------
# download: resume, restart, progress
# ---------------------------------------------------------------------------

def test_download_resume_sends_range_and_completes():
    payload = os.urandom(200_000)
    srv, base = _make_hub({"acme/r": _repo({"f.gguf": payload})})
    try:
        url = f"{base}/acme/r/resolve/main/f.gguf"
        dest = "/tmp/lacuna-test-resume/f.gguf"
        if os.path.exists(dest):
            os.remove(dest)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:  # simulate an interrupted download
            f.write(payload[:80_000])
        download(url, dest)
        with open(dest, "rb") as f:
            assert f.read() == payload
        ranges = [h.get("Range") for _, h in _FakeHubHandler.received
                  if _.endswith("f.gguf")]
        assert "bytes=80000-" in ranges, f"no Range resume header: {ranges}"
    finally:
        srv.shutdown()


def test_download_restarts_when_server_ignores_range():
    payload = os.urandom(50_000)
    srv, base = _make_hub({"acme/r": _repo({"f.gguf": payload})},
                          ignore_range=True)
    try:
        url = f"{base}/acme/r/resolve/main/f.gguf"
        dest = "/tmp/lacuna-test-restart/f.gguf"
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:  # stale partial content
            f.write(b"\x00" * 10_000)
        download(url, dest)
        with open(dest, "rb") as f:
            assert f.read() == payload
    finally:
        srv.shutdown()


def test_progress_callback_fires_monotonic():
    payload = os.urandom(100_000)
    srv, base = _make_hub({"acme/p": _repo({"f.gguf": payload})})
    try:
        seen = []
        download(f"{base}/acme/p/resolve/main/f.gguf",
                 "/tmp/lacuna-test-progress/f.gguf",
                 progress=lambda d, t: seen.append((d, t)))
        assert seen, "progress never fired"
        assert seen[-1] == (100_000, 100_000)
        vals = [d for d, _ in seen]
        assert all(b >= a for a, b in zip(vals, vals[1:], strict=False)), "not monotonic"
    finally:
        srv.shutdown()


# ---------------------------------------------------------------------------
# pull(): end-to-end with a REAL (tiny, valid) GGUF through the REAL converter
# ---------------------------------------------------------------------------

def test_pull_end_to_end_converts_real_gguf(tmp_path):
    gguf = _tiny_gguf()
    srv, base = _make_hub({"acme/tiny": _repo({"tiny-Q4_K_M.gguf": gguf})})
    try:
        lac = pull("acme/tiny", cache_dir=str(tmp_path), api_base=base)
        assert lac.endswith(".lac")
        with open(lac, "rb") as f:
            assert f.read(4) == b"LAC1"  # real .lac, written by LacWriter
        # idempotent: second pull() reuses the .lac — the file itself must
        # NOT be re-downloaded (the cheap API re-list is fine).
        before = dict(_FakeHubHandler.hits)
        lac2 = pull("acme/tiny", cache_dir=str(tmp_path), api_base=base)
        assert lac2 == lac
        dl_before = sum(v for k, v in before.items() if "resolve/main" in k)
        dl_after = sum(v for k, v in _FakeHubHandler.hits.items()
                       if "resolve/main" in k)
        assert dl_after == dl_before, "second pull re-downloaded the file"
    finally:
        srv.shutdown()


def test_pull_conversion_failure_is_loud_not_fake(tmp_path):
    srv, base = _make_hub({"acme/bad": _repo({"bad-Q4_K_M.gguf": b"not a gguf"})})
    try:
        with pytest.raises(PullError) as e:
            pull("acme/bad", cache_dir=str(tmp_path), api_base=base)
        assert "conversion" in str(e.value)
        # no half-written .lac left behind pretending to be a model
        leftovers = [p for p in tmp_path.rglob("*.lac")]
        assert leftovers == []
    finally:
        srv.shutdown()


# ---------------------------------------------------------------------------
# Ollama shim (mock backend, clearly labeled)
# ---------------------------------------------------------------------------

class _FakeBackend:
    mode = "mock"

    def generate(self, prompt: str, max_tokens: int = 32) -> str:
        return "[mock] test reply"


def _start_shim(tmp_path):
    be = _FakeBackend()
    assert isinstance(be, InferenceBackend)  # protocol conformance
    srv = build_ollama_server(be, port=0, model_name="lacuna",
                              cache_dir=str(tmp_path))
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def _post(base, path, obj):
    req = urllib.request.Request(
        base + path, data=json.dumps(obj).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    return urllib.request.urlopen(req)


def test_tags_lists_models(tmp_path):
    srv, base = _start_shim(tmp_path)
    try:
        with urllib.request.urlopen(base + "/api/tags") as r:
            data = json.loads(r.read())
        names = [m["name"] for m in data["models"]]
        assert "lacuna" in names
        assert data["models"][0]["details"]["lacuna_mode"] == "mock"
    finally:
        srv.shutdown()


def test_chat_nonstreaming_returns_mock_reply(tmp_path):
    srv, base = _start_shim(tmp_path)
    try:
        with _post(base, "/api/chat",
                   {"model": "lacuna",
                    "messages": [{"role": "user", "content": "hi"}]}) as r:
            data = json.loads(r.read())
        assert data["done"] is True
        assert "[mock]" in data["message"]["content"]
        assert data["message"]["role"] == "assistant"
    finally:
        srv.shutdown()


def test_chat_streaming_yields_valid_ndjson_ending_done(tmp_path):
    srv, base = _start_shim(tmp_path)
    try:
        with _post(base, "/api/chat",
                   {"model": "lacuna",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True}) as r:
            lines = [ln for ln in r.read().decode().splitlines() if ln.strip()]
        objs = [json.loads(ln) for ln in lines]  # every line valid JSON
        assert len(objs) >= 2
        assert all(o.get("done") is False for o in objs[:-1])
        assert objs[-1] == {"done": True}
        assert any("message" in o for o in objs[:-1])
    finally:
        srv.shutdown()


def test_generate_nonstreaming(tmp_path):
    srv, base = _start_shim(tmp_path)
    try:
        with _post(base, "/api/generate",
                   {"model": "lacuna", "prompt": "hi"}) as r:
            data = json.loads(r.read())
        assert data["done"] is True
        assert "[mock]" in data["response"]
    finally:
        srv.shutdown()


def test_unknown_model_404(tmp_path):
    srv, base = _start_shim(tmp_path)
    try:
        with pytest.raises(urllib.error.HTTPError) as e:
            _post(base, "/api/chat",
                  {"model": "nope", "messages": []})
        assert e.value.code == 404
    finally:
        srv.shutdown()


# ---------------------------------------------------------------------------
# auto_tune: pure function of HardwareInfo — synthetic profiles
# ---------------------------------------------------------------------------

def test_autotune_tiny_box():
    cfg = auto_tune(HardwareInfo(ram_total_gb=4.0, vram_gb=0.0, has_gpu=False))
    assert cfg.ram_cap <= 0.75 * 4.0
    assert cfg.ram_cap == 3.0
    assert cfg.vram_cap == 0.0
    assert cfg.page_size == 1 * 1024 * 1024
    assert cfg.precision_policy == "exact"  # never silently degrades quality


def test_autotune_laptop():
    cfg = auto_tune(HardwareInfo(ram_total_gb=8.0, vram_gb=4.0, has_gpu=True))
    assert cfg.ram_cap == 6.0
    assert cfg.vram_cap == 3.2
    assert cfg.page_size == 2 * 1024 * 1024
    assert cfg.k_lookahead == 4
    assert cfg.precision_policy == "exact"


def test_autotune_golden_reference_laptop():
    # GOLDEN REFERENCE: the machine Lacuna must feel perfect on —
    # 16GB system RAM + RTX 4050 6GB VRAM (laptop).
    cfg = auto_tune(HardwareInfo(ram_total_gb=16.0, vram_gb=6.0, has_gpu=True,
                                nvme_bw_gbs=3.5, disk_is_nvme=True))
    # Well under the 75% RAM ceiling: 6GB of 16GB leaves the system breathing.
    assert cfg.ram_cap == 6.0
    assert cfg.ram_cap <= 0.75 * 16.0
    assert cfg.ram_cap / 16.0 < 0.40  # under 40% of system RAM
    # VRAM cap leaves >=2GB headroom for the OS/display on a 6GB card.
    assert cfg.vram_cap == 3.2
    assert cfg.vram_cap <= 6.0 - 2.0
    assert cfg.page_size == 2 * 1024 * 1024
    assert cfg.k_lookahead == 4
    assert cfg.precision_policy == "exact"


def test_autotune_desktop():
    cfg = auto_tune(HardwareInfo(ram_total_gb=64.0, vram_gb=24.0, has_gpu=True))
    assert cfg.ram_cap == 6.0  # capped: Lacuna never hogs a big machine
    assert cfg.vram_cap == 3.2
    assert cfg.page_size == 4 * 1024 * 1024
    assert cfg.precision_policy == "exact"


def test_autotune_slow_disk_gets_more_lookahead():
    cfg = auto_tune(HardwareInfo(ram_total_gb=8.0, vram_gb=4.0, has_gpu=True,
                                nvme_bw_gbs=0.55, disk_is_nvme=False))
    assert cfg.k_lookahead == 6  # deeper lookahead hides slow-disk latency


def test_autotune_never_exceeds_75pct_ram():
    for ram in (2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0):
        cfg = auto_tune(HardwareInfo(ram_total_gb=ram))
        assert cfg.ram_cap <= 0.75 * ram + 1e-9, f"ram={ram}"


# ---------------------------------------------------------------------------
# CLI wiring smoke tests
# ---------------------------------------------------------------------------

def test_cli_pull_parser():
    a = build_parser().parse_args(["pull", "Qwen/Qwen3-32B"])
    assert a.cmd == "pull" and a.hf_id == "Qwen/Qwen3-32B"


def test_cli_serve_ollama_flag():
    a = build_parser().parse_args(["serve", "--ollama", "--port", "11434"])
    assert a.cmd == "serve" and a.ollama is True and a.port == 11434
    b = build_parser().parse_args(["serve"])
    assert b.ollama is False  # default stays OpenAI-compatible
