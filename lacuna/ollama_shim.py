"""Ollama-compatible HTTP API — stdlib ``http.server`` only.

Lets the entire Ollama agent ecosystem (LangChain, LlamaIndex, Open WebUI,
ollama-python, Hermes-style agents, ...) talk to Lacuna unchanged: point the
client at this server instead of ``localhost:11434``.

Endpoints:
  POST /api/generate   {model, prompt, stream}  -> NDJSON or single JSON
  POST /api/chat       {model, messages, stream} -> NDJSON or single JSON
  GET  /api/tags       -> {"models": [...]}  (models found in the pull cache)

Streaming is NDJSON: one JSON object per line,
``{"model": ..., "response": "...", "done": false}`` ... ``{"done": true}``.
``/api/chat`` uses the ``message`` shape instead of ``response``.

ENGINE-SWAP INTERFACE
---------------------
The shim (and ``server.py``) only needs this tiny protocol::

    class InferenceBackend(Protocol):
        def generate(self, prompt: str, max_tokens: int = 32) -> str: ...
        @property
        def mode(self) -> str: ...   # "mock" | "lacuna" | "libllama-dense"

``lacuna.engine.Engine`` already satisfies it structurally. Swapping in real
inference later = pass a real backend object to ``serve_ollama()`` — a
one-line change at the call site, zero shim changes. The shim never assumes
anything about how generation happens; it only chunks and formats text.

HONESTY: the default backend is ``Engine("mock")`` — the clearly-labeled
mock. ``/api/tags`` reports ``"lacuna": {"mode": "mock"}`` etc. so clients
can see what they are talking to. Nothing here pretends the mock is a model.
"""

from __future__ import annotations

import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Protocol, runtime_checkable


@runtime_checkable
class InferenceBackend(Protocol):
    """Minimal contract any Lacuna inference backend must satisfy.

    ``generate`` returns the full completion text. ``mode`` reports which
    backend produced it ("mock" = clearly-labeled mock, "lacuna" = paged
    weights, "libllama-dense" = llama.cpp dense fallback).
    """

    def generate(self, prompt: str, max_tokens: int = 32) -> str: ...

    @property
    def mode(self) -> str: ...


def _default_backend():
    from .engine import Engine

    return Engine("mock")


def scan_cache_models(cache_dir: str) -> list[str]:
    """Model names available: every ``.lac`` in the pull cache, by stem."""
    models_dir = os.path.join(cache_dir, "models")
    names: list[str] = []
    if os.path.isdir(models_dir):
        for repo in sorted(os.listdir(models_dir)):
            rd = os.path.join(models_dir, repo)
            if not os.path.isdir(rd):
                continue
            for fn in sorted(os.listdir(rd)):
                if fn.endswith(".lac"):
                    names.append(os.path.splitext(fn)[0])
    return names


class _OllamaHandler(BaseHTTPRequestHandler):
    backend: InferenceBackend = None  # set by serve_ollama()
    registry: dict = {}               # model name -> InferenceBackend
    default_model: str = "lacuna"

    # -- helpers -------------------------------------------------
    def _send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
        except ValueError:
            n = 0
        if n <= 0:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8") or "{}")
        except (ValueError, UnicodeDecodeError):
            return {}

    def log_message(self, *a):  # quiet
        pass

    def _backend_for(self, name: str):
        if not name:
            name = self.default_model
        be = self.registry.get(name)
        if be is None:
            self._send_json(
                {"error": f"model '{name}' not found; available: "
                          f"{sorted(self.registry)}"}, 404)
        return be

    def _ndjson_stream(self, backend, prompt, max_tokens, chat: bool):
        text = backend.generate(prompt, max_tokens=max_tokens)
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        # stream word-by-word: simple, honest chunking of the final text
        for word in text.split(" "):
            if chat:
                obj = {"model": self.default_model,
                       "message": {"role": "assistant",
                                   "content": word + " "},
                       "done": False}
            else:
                obj = {"model": self.default_model,
                       "response": word + " ", "done": False}
            self.wfile.write((json.dumps(obj) + "\n").encode())
        self.wfile.write(b'{"done": true}\n')

    # -- routes --------------------------------------------------
    def do_GET(self):
        if self.path in ("/api/tags", "/api/tags/"):
            models = [
                {"name": name,
                 "model": name,
                 "modified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                              time.gmtime()),
                 "size": 0,
                 "digest": "lacuna",
                 "details": {"lacuna_mode": be.mode}}
                for name, be in sorted(self.registry.items())
            ]
            self._send_json({"models": models})
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        body = self._read_json()
        if self.path == "/api/generate":
            be = self._backend_for(body.get("model", ""))
            if be is None:
                return
            prompt = str(body.get("prompt", ""))
            max_tokens = int(body.get("max_tokens", 32) or 32)
            if body.get("stream"):
                self._ndjson_stream(be, prompt, max_tokens, chat=False)
            else:
                text = be.generate(prompt, max_tokens=max_tokens)
                self._send_json({"model": body.get("model") or self.default_model,
                                 "response": text, "done": True})
        elif self.path == "/api/chat":
            be = self._backend_for(body.get("model", ""))
            if be is None:
                return
            messages = body.get("messages", [])
            prompt = " ".join(
                str(m.get("content", "")) for m in messages if isinstance(m, dict))
            max_tokens = int(body.get("max_tokens", 32) or 32)
            if body.get("stream"):
                self._ndjson_stream(be, prompt, max_tokens, chat=True)
            else:
                text = be.generate(prompt, max_tokens=max_tokens)
                self._send_json(
                    {"model": body.get("model") or self.default_model,
                     "message": {"role": "assistant", "content": text},
                     "done": True})
        else:
            self._send_json({"error": "not found"}, 404)


def build_ollama_server(backend: InferenceBackend | None = None,
                        host: str = "127.0.0.1", port: int = 11434,
                        model_name: str = "lacuna",
                        cache_dir: str | None = None,
                        extra_models: dict | None = None) -> ThreadingHTTPServer:
    """Build (but do not start) the Ollama-compatible server.

    Useful for embedding and for tests (bind ``port=0`` to pick a free port,
    then read it back from ``server.server_address[1]``).
    """
    from .pull import DEFAULT_CACHE_DIR

    be = backend if backend is not None else _default_backend()
    assert isinstance(be, InferenceBackend), (
        "backend must satisfy the InferenceBackend protocol "
        "(generate(prompt, max_tokens) -> str, .mode)")
    cache = cache_dir or DEFAULT_CACHE_DIR
    registry = {model_name: be}
    for cached in scan_cache_models(cache):
        registry.setdefault(cached, be)
    if extra_models:
        registry.update(extra_models)
    _OllamaHandler.backend = be
    _OllamaHandler.registry = registry
    _OllamaHandler.default_model = model_name
    return ThreadingHTTPServer((host, port), _OllamaHandler)


def serve_ollama(backend: InferenceBackend | None = None,
                 host: str = "127.0.0.1", port: int = 11434,
                 model_name: str = "lacuna",
                 cache_dir: str | None = None,
                 extra_models: dict | None = None):
    """Start the Ollama-compatible server (blocking).

    ``backend`` is the default model served as ``model_name``; every ``.lac``
    in the pull cache is also registered (backed by the same backend object
    until per-model backends land — the registry maps names, so wiring a real
    per-model backend later is one dict entry). ``extra_models`` lets callers
    register additional name -> backend pairs.
    """
    srv = build_ollama_server(backend, host, port, model_name, cache_dir,
                              extra_models)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
