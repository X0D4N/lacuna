"""API tests: one-line Engine, HTTP endpoints, MCP stdio smoke test."""

import json
import subprocess
import sys
import threading
import urllib.request

import pytest

from lacuna.engine import Engine
from lacuna.mcp_server import handle as mcp_handle
from lacuna.server import _Handler
from http.server import ThreadingHTTPServer


def test_one_line_engine_api():
    eng = Engine("mock")
    try:
        out = eng.generate("hello world", max_tokens=8)
        assert isinstance(out, str) and out.startswith("[mock]")
        assert len(out.split()) >= 4
    finally:
        eng.close()


def _free_port():
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture()
def http_server():
    eng = Engine("mock")
    port = _free_port()
    _Handler.engine = eng
    srv = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}"
    srv.shutdown()
    srv.server_close()
    eng.close()


def _post(base, path, payload):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode())


def _get(base, path):
    with urllib.request.urlopen(base + path, timeout=10) as r:
        return json.loads(r.read().decode())


def test_health_and_models(http_server):
    assert _get(http_server, "/health") == {"status": "ok"}
    models = _get(http_server, "/v1/models")
    assert models["object"] == "list" and len(models["data"]) == 1


def test_completions(http_server):
    r = _post(http_server, "/v1/completions", {"prompt": "hi", "max_tokens": 8})
    assert r["object"] == "text_completion"
    assert isinstance(r["choices"][0]["text"], str)


def test_chat_completions(http_server):
    r = _post(
        http_server,
        "/v1/chat/completions",
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8},
    )
    assert r["object"] == "chat_completion"
    assert r["choices"][0]["message"]["role"] == "assistant"


def test_mcp_initialize_and_tools_list():
    r = mcp_handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert r["result"]["serverInfo"]["name"] == "lacuna"
    r = mcp_handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    names = {t["name"] for t in r["result"]["tools"]}
    assert names == {"lacuna_generate", "lacuna_convert", "lacuna_benchmark"}


def test_mcp_generate_tool():
    r = mcp_handle(
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "lacuna_generate",
                    "arguments": {"prompt": "hi", "max_tokens": 8}}},
    )
    text = r["result"]["content"][0]["text"]
    assert isinstance(text, str) and len(text) > 0


def test_mcp_stdio_smoke():
    """Real subprocess over stdio: initialize -> tools/list -> tools/call."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "lacuna.mcp_server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        cwd="/home/hatch/workspace/lacuna",
    )
    msgs = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "lacuna_benchmark", "arguments": {}}},
    ]
    inp = "\n".join(json.dumps(m) for m in msgs) + "\n"
    try:
        out, _ = proc.communicate(inp, timeout=60)
    finally:
        proc.kill()
    lines = [json.loads(l) for l in out.strip().split("\n") if l.strip()]
    assert len(lines) == 3
    assert lines[0]["result"]["serverInfo"]["name"] == "lacuna"
    assert "SIMULATED" in lines[2]["result"]["content"][0]["text"]
