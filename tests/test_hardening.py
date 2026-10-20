"""Production hardening tests (round 2, workstream 4 + edge cases).

- HTTP servers: body-size cap, invalid/non-object JSON, bad max_tokens,
  bad prompt/messages shapes, query strings — all answered with clean
  JSON (status + one human sentence), never a dropped connection or a
  stack trace.
- MCP server: malformed messages get JSON-RPC errors; tool errors carry
  one clean line, never a traceback.
- CLI: normal failure paths print one clear sentence + next step and
  return a nonzero code, no traceback.
- pull: hub-supplied paths cannot escape the cache dir; resume edge
  cases (zero-byte file, early close, stale-prefix 416) behave.
- auto_tune: floor/tiny/no-GPU/huge/degenerate hardware profiles.
"""

import io
import json
import os
import socket
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

import lacuna.mcp_server as mcp
from lacuna import cli
from lacuna.engine import Engine
from lacuna.hardware import HardwareInfo, auto_tune
from lacuna.mcp_server import handle as mcp_handle
from lacuna.ollama_shim import build_ollama_server
from lacuna.pull import PullError, _cache_paths, download
from lacuna.server import _Handler


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _post_raw(base, path, data: bytes):
    """POST raw bytes; returns (status, body) without raising on 4xx/5xx."""
    req = urllib.request.Request(
        base + path, data=data,
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def _raw_http_status(port: int, request: bytes) -> str:
    with socket.create_connection(("127.0.0.1", port), timeout=5) as s:
        s.sendall(request)
        s.settimeout(5)
        buf = b""
        while b"\r\n\r\n" not in buf:
            chunk = s.recv(4096)
            if not chunk:
                break
            buf += chunk
    return buf.split(b"\r\n", 1)[0].decode("latin-1")


@pytest.fixture()
def openai_server():
    eng = Engine("mock")
    port = _free_port()
    _Handler.engine = eng
    _Handler.model_id = "lacuna-mock"
    srv = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}", port
    srv.shutdown()
    srv.server_close()
    eng.close()


class _FakeBackend:
    mode = "mock"

    def generate(self, prompt: str, max_tokens: int = 32) -> str:
        return "[mock] test reply"


@pytest.fixture()
def ollama_server(tmp_path):
    srv = build_ollama_server(_FakeBackend(), port=0, model_name="lacuna",
                              cache_dir=str(tmp_path))
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", srv.server_address[1]
    srv.shutdown()


# ---------------------------------------------------------------------------
# OpenAI-compatible server
# ---------------------------------------------------------------------------

def test_openai_invalid_json_is_clean_400(openai_server):
    base, _ = openai_server
    code, body = _post_raw(base, "/v1/completions", b"{not json")
    assert code == 400
    assert b"Traceback" not in body
    assert "not valid JSON" in json.loads(body)["error"]


def test_openai_non_object_json_is_clean_400(openai_server):
    base, _ = openai_server
    code, body = _post_raw(base, "/v1/completions", b"[1, 2, 3]")
    assert code == 400
    assert "must be a JSON object" in json.loads(body)["error"]


def test_openai_bad_max_tokens_is_400_and_huge_is_clamped(openai_server):
    base, _ = openai_server
    code, body = _post_raw(base, "/v1/completions",
                           json.dumps({"prompt": "hi", "max_tokens": "abc"}).encode())
    assert code == 400
    assert "max_tokens" in json.loads(body)["error"]
    code, body = _post_raw(base, "/v1/completions",
                           json.dumps({"prompt": "hi", "max_tokens": 10**9}).encode())
    assert code == 200
    code, body = _post_raw(base, "/v1/completions",
                           json.dumps({"prompt": "hi", "max_tokens": None}).encode())
    assert code == 200


def test_openai_bad_prompt_and_messages_shapes_are_400(openai_server):
    base, _ = openai_server
    code, _ = _post_raw(base, "/v1/completions",
                        json.dumps({"prompt": {"x": 1}}).encode())
    assert code == 400
    code, body = _post_raw(base, "/v1/chat/completions",
                           json.dumps({"messages": "nope"}).encode())
    assert code == 400
    assert "messages must be a list" in json.loads(body)["error"]


def test_openai_query_string_and_unknown_path(openai_server):
    base, _ = openai_server
    with urllib.request.urlopen(base + "/health?verbose=1", timeout=5) as r:
        assert json.loads(r.read()) == {"status": "ok"}
    code, body = _post_raw(base, "/v1/nope", b"{}")
    assert code == 404
    assert json.loads(body)["error"] == "not found"


def test_openai_oversized_body_is_413(openai_server):
    _, port = openai_server
    status = _raw_http_status(
        port,
        b"POST /v1/completions HTTP/1.1\r\nHost: x\r\n"
        b"Content-Type: application/json\r\n"
        b"Content-Length: 2097153\r\n\r\n{}",
    )
    assert "413" in status


# ---------------------------------------------------------------------------
# Ollama shim
# ---------------------------------------------------------------------------

def test_ollama_nonstring_model_is_clean_400(ollama_server):
    base, _ = ollama_server
    code, body = _post_raw(base, "/api/generate",
                           json.dumps({"model": ["x"], "prompt": "hi"}).encode())
    assert code == 400
    assert "model must be a string" in json.loads(body)["error"]


def test_ollama_invalid_json_and_bad_fields_are_400(ollama_server):
    base, _ = ollama_server
    code, body = _post_raw(base, "/api/generate", b"###")
    assert code == 400
    assert b"Traceback" not in body
    code, _ = _post_raw(base, "/api/generate",
                        json.dumps({"prompt": "hi", "max_tokens": "abc"}).encode())
    assert code == 400
    code, _ = _post_raw(base, "/api/generate",
                        json.dumps({"prompt": {"x": 1}}).encode())
    assert code == 400
    code, _ = _post_raw(base, "/api/chat",
                        json.dumps({"messages": 42}).encode())
    assert code == 400


def test_ollama_tags_with_query_string(ollama_server):
    base, _ = ollama_server
    with urllib.request.urlopen(base + "/api/tags?x=1", timeout=5) as r:
        data = json.loads(r.read())
    assert any(m["name"] == "lacuna" for m in data["models"])


def test_ollama_oversized_body_is_413(ollama_server):
    _, port = ollama_server
    status = _raw_http_status(
        port,
        b"POST /api/generate HTTP/1.1\r\nHost: x\r\n"
        b"Content-Type: application/json\r\n"
        b"Content-Length: 2097153\r\n\r\n{}",
    )
    assert "413" in status


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

def test_mcp_non_dict_message_and_params():
    r = mcp_handle([1, 2, 3])
    assert r["error"]["code"] == -32600
    r = mcp_handle({"jsonrpc": "2.0", "id": 1, "method": "ping",
                    "params": [1]})
    assert r["error"]["code"] == -32602
    r = mcp_handle({"jsonrpc": "2.0", "id": 2, "method": "nope"})
    assert r["error"]["code"] == -32601


def test_mcp_tool_error_has_no_traceback():
    r = mcp_handle({"jsonrpc": "2.0", "id": 9, "method": "tools/call",
                    "params": {"name": "lacuna_convert",
                               "arguments": {"input_gguf": "/no/such.gguf"}}})
    text = r["result"]["content"][0]["text"]
    assert r["result"]["isError"] is True
    assert "Traceback" not in text
    assert "not found" in text


def test_mcp_bad_tool_arguments_are_clean_errors():
    r = mcp_handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                    "params": {"name": "lacuna_generate",
                               "arguments": {"prompt": {"x": 1}}}})
    assert r["result"]["isError"] is True
    assert "prompt must be a string" in r["result"]["content"][0]["text"]
    r = mcp_handle({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                    "params": {"name": "lacuna_generate",
                               "arguments": {"prompt": "hi",
                                             "max_tokens": "abc"}}})
    assert r["result"]["isError"] is True
    r = mcp_handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                    "params": {"name": "lacuna_generate",
                               "arguments": "not-a-dict"}})
    assert r["result"]["isError"] is True
    assert "arguments" in r["result"]["content"][0]["text"]


def test_mcp_main_parse_error_and_recovery(monkeypatch):
    inp = io.StringIO(
        "this is not json\n"
        '{"jsonrpc": "2.0", "id": 1, "method": "ping"}\n'
        "[1, 2, 3]\n"
    )
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdin", inp)
    monkeypatch.setattr(sys, "stdout", out)
    assert mcp.main() == 0
    lines = [json.loads(l) for l in out.getvalue().strip().split("\n")]
    assert lines[0]["error"]["code"] == -32700
    assert lines[1]["result"] == {}
    assert lines[2]["error"]["code"] == -32600


# ---------------------------------------------------------------------------
# CLI error quality
# ---------------------------------------------------------------------------

def test_cli_run_missing_model_file_is_clean(capsys):
    rc = cli.main(["run", "definitely-missing.gguf", "hi"])
    out = capsys.readouterr().out
    assert rc == 2
    assert "model file not found" in out
    assert "lacuna pull" in out  # says what to do next
    assert "Traceback" not in out


def test_cli_run_bad_max_tokens_is_clean(capsys):
    rc = cli.main(["run", "mock", "hi", "--max-tokens", "0"])
    out = capsys.readouterr().out
    assert rc == 2
    assert "--max-tokens" in out
    assert "Traceback" not in out


def test_cli_run_corrupt_lac_is_clean(tmp_path, capsys):
    bad = tmp_path / "bad.lac"
    bad.write_bytes(b"garbage, not a lac file")
    rc = cli.main(["run", str(bad), "hi"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "could not run" in out
    assert "Traceback" not in out


def test_cli_run_mock_still_works(capsys):
    rc = cli.main(["run", "mock", "hello", "--max-tokens", "4"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "[mock]" in out


# ---------------------------------------------------------------------------
# pull: path traversal + resume edges
# ---------------------------------------------------------------------------

def test_cache_paths_reject_traversal(tmp_path):
    with pytest.raises(PullError):
        _cache_paths("owner/repo", "../../evil.gguf", str(tmp_path))
    with pytest.raises(PullError):
        _cache_paths("owner/repo", "../evil.gguf", str(tmp_path))
    with pytest.raises(PullError):
        _cache_paths("..", "x.gguf", str(tmp_path))
    with pytest.raises(PullError):
        _cache_paths("", "x.gguf", str(tmp_path))


def test_cache_paths_allow_legit_nested_filename(tmp_path):
    gguf, lac = _cache_paths("owner/repo", "sub/dir/model.gguf", str(tmp_path))
    assert gguf.startswith(str(tmp_path))
    assert gguf.endswith(os.path.join("sub", "dir", "model.gguf"))
    assert lac.endswith(".lac")


class _Resp:
    def __init__(self, body: bytes, status: int = 200, headers=None):
        self._buf = io.BytesIO(body)
        self.status = status
        self._headers = headers or {}

    def getheader(self, name):
        return self._headers.get(name)

    def read(self, n=-1):
        return self._buf.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _http_error(code, content_range=None):
    hdrs = {}
    if content_range:
        hdrs["Content-Range"] = content_range
    return urllib.error.HTTPError("http://x/f.gguf", code, "err", hdrs, None)


def test_download_zero_byte_dest_sends_no_range(tmp_path):
    dest = str(tmp_path / "f.gguf")
    open(dest, "wb").close()
    seen = {}

    def opener(req):
        seen["range"] = req.get_header("Range")
        return _Resp(b"payload")

    assert download("http://x/f.gguf", dest, opener=opener) == dest
    assert seen["range"] is None
    assert open(dest, "rb").read() == b"payload"


def test_download_early_close_then_resume_completes(tmp_path):
    dest = str(tmp_path / "f.gguf")
    full = b"0123456789" * 1000

    def opener1(req):
        assert req.get_header("Range") is None
        return _Resp(full[:400], status=200)  # server drops after 400 bytes

    download("http://x/f.gguf", dest, opener=opener1)
    assert open(dest, "rb").read() == full[:400]

    def opener2(req):
        assert req.get_header("Range") == "bytes=400-"
        return _Resp(full[400:], status=206,
                     headers={"Content-Length": str(len(full) - 400)})

    download("http://x/f.gguf", dest, opener=opener2)
    assert open(dest, "rb").read() == full


def test_download_stale_prefix_416_restarts(tmp_path):
    dest = str(tmp_path / "f.gguf")
    with open(dest, "wb") as f:
        f.write(b"STALE" * 30)  # 150 bytes held; remote file is only 100
    calls = {"n": 0}

    def opener(req):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _http_error(416, "bytes */100")
        return _Resp(b"R" * 100, status=200)

    download("http://x/f.gguf", dest, opener=opener)
    assert calls["n"] == 2
    assert open(dest, "rb").read() == b"R" * 100


def test_download_complete_416_returns_dest_unchanged(tmp_path):
    dest = str(tmp_path / "f.gguf")
    with open(dest, "wb") as f:
        f.write(b"C" * 100)

    def opener(req):
        raise _http_error(416, "bytes */100")

    assert download("http://x/f.gguf", dest, opener=opener) == dest
    assert open(dest, "rb").read() == b"C" * 100


def test_download_creates_nested_dirs_and_404_is_clean(tmp_path):
    dest = str(tmp_path / "a" / "b" / "c" / "f.gguf")
    download("http://x/f.gguf", dest, opener=lambda req: _Resp(b"ok"))
    assert open(dest, "rb").read() == b"ok"

    def opener404(req):
        raise _http_error(404)

    with pytest.raises(PullError, match="download failed"):
        download("http://x/f.gguf", str(tmp_path / "g.gguf"), opener=opener404)


# ---------------------------------------------------------------------------
# auto_tune edge profiles
# ---------------------------------------------------------------------------

def test_autotune_floor_profile_min_spec():
    # the documented floor: RTX 2050-class 4GB VRAM + 6GB RAM
    cfg = auto_tune(HardwareInfo(ram_total_gb=6.0, vram_gb=4.0, has_gpu=True))
    assert cfg.ram_cap == 4.5
    assert cfg.vram_cap == 3.2
    assert cfg.page_size == 2 * 1024 * 1024
    assert cfg.k_lookahead == 4
    assert cfg.precision_policy == "exact"


def test_autotune_tiny_ram_machine():
    cfg = auto_tune(HardwareInfo(ram_total_gb=2.0, has_gpu=False))
    assert cfg.ram_cap == 1.5
    assert cfg.page_size == 1 * 1024 * 1024
    assert cfg.vram_cap == 0.0


def test_autotune_no_gpu_machine():
    cfg = auto_tune(HardwareInfo(ram_total_gb=16.0, vram_gb=0.0, has_gpu=False))
    assert cfg.vram_cap == 0.0
    assert cfg.k_lookahead == 3
    assert cfg.ram_cap == 6.0


def test_autotune_huge_machine_caps_hold():
    cfg = auto_tune(HardwareInfo(ram_total_gb=128.0, vram_gb=24.0, has_gpu=True))
    assert cfg.ram_cap == 6.0  # the 6GB cap, not 75% of 128
    assert cfg.page_size == 4 * 1024 * 1024


def test_autotune_small_vram_gpu_gets_80pct():
    cfg = auto_tune(HardwareInfo(ram_total_gb=16.0, vram_gb=2.0, has_gpu=True))
    assert cfg.vram_cap == 1.6


def test_autotune_degenerate_ram_is_guarded():
    cfg = auto_tune(HardwareInfo(ram_total_gb=-5.0, has_gpu=False))
    assert cfg.ram_cap == 0.38  # 75% of the 0.5 GB floor, no nonsense values
    assert cfg.page_size == 1 * 1024 * 1024
    cfg0 = auto_tune(HardwareInfo(ram_total_gb=0.0, has_gpu=False))
    assert cfg0.ram_cap == 0.38


def test_autotune_slow_disk_beats_gpu_default_k():
    cfg = auto_tune(HardwareInfo(ram_total_gb=16.0, vram_gb=8.0, has_gpu=True,
                                 nvme_bw_gbs=0.5))
    assert cfg.k_lookahead == 6
