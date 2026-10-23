"""LlamaServerBackend + Engine routing to real llama.cpp inference.

The unit tests are hermetic: a stub HTTP server stands in for llama-server
and ``run_llama_server`` is monkeypatched, so the suite stays green on
machines without a llama.cpp build. The integration test runs only when
LACUNA_LLAMA_SERVER and LACUNA_TEST_GGUF are both set, and then it demands
real model output (non-empty, no mock label).
"""

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from lacuna import engine as engine_mod
from lacuna.engine import Engine, LlamaServerBackend
from lacuna.ollama_shim import InferenceBackend

_CANNED = "Paris is the capital of France."


class _StubLlamaHandler(BaseHTTPRequestHandler):
    """Minimal stand-in for llama-server's /health and /completion."""

    def log_message(self, *a):  # quiet
        pass

    def do_GET(self):
        if self.path == "/health":
            body = b'{"status":"ok"}'
            self.send_response(200)
        else:
            body = b'{"error":"not found"}'
            self.send_response(404)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(n) or b"{}")
        assert self.path == "/completion"
        # the backend must pin greedy decoding for reproducibility
        assert payload["temperature"] == 0.0
        assert payload["top_k"] == 1
        body = json.dumps(
            {"content": _CANNED, "timings": {"predicted_n": 7}}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _DummyProc:
    def poll(self):
        return None

    def terminate(self):
        pass

    def wait(self, timeout=None):
        return 0


@pytest.fixture()
def stub_server(monkeypatch):
    """Run the stub llama-server; make the backend find and 'spawn' it."""
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _StubLlamaHandler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    monkeypatch.setattr(
        engine_mod, "run_llama_server", lambda *a, **k: _DummyProc()
    )
    yield port
    srv.shutdown()
    srv.server_close()


def test_backend_satisfies_protocol_and_generates(stub_server, tmp_path):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"GGUF")
    be = LlamaServerBackend(
        str(model), server_binary="/bin/true", port=stub_server
    )
    try:
        assert isinstance(be, InferenceBackend)
        assert be.mode == "libllama-dense"
        assert be.generate("The capital of France is", max_tokens=16) == _CANNED
        assert be.last_timings == {"predicted_n": 7}
    finally:
        be.close()


def test_backend_missing_binary_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("LACUNA_LLAMA_SERVER", raising=False)
    monkeypatch.setattr(engine_mod.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="llama-server not found"):
        LlamaServerBackend(str(tmp_path / "m.gguf"))


def test_engine_routes_gguf_to_real_backend(tmp_path, monkeypatch):
    """A .gguf + an available llama-server => Engine generates via it."""
    model = tmp_path / "m.gguf"
    model.write_bytes(b"GGUF")

    class _StubBackend:
        mode = "libllama-dense"

        def __init__(self, model_path, server_binary=None):
            self.model_path = model_path

        def generate(self, prompt, max_tokens=32):
            return f"real:{prompt}"

        def close(self):
            pass

    monkeypatch.setattr(
        engine_mod, "_find_llama_server", lambda explicit=None: "/fake/llama-server"
    )
    monkeypatch.setattr(engine_mod, "LlamaServerBackend", _StubBackend)
    eng = Engine(str(model))
    try:
        assert eng.mode == "libllama-dense"
        assert eng.generate("hello", max_tokens=4) == "real:hello"
    finally:
        eng.close()


@pytest.mark.skipif(
    not (os.environ.get("LACUNA_LLAMA_SERVER") and os.environ.get("LACUNA_TEST_GGUF")),
    reason="set LACUNA_LLAMA_SERVER + LACUNA_TEST_GGUF to run real inference",
)
def test_real_inference_end_to_end():
    eng = Engine(
        os.environ["LACUNA_TEST_GGUF"],
        llama_server=os.environ["LACUNA_LLAMA_SERVER"],
    )
    try:
        assert eng.mode == "libllama-dense"
        out = eng.generate("The capital of France is", max_tokens=16)
        assert out.strip()
        assert "[mock]" not in out
    finally:
        eng.close()
