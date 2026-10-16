"""OpenAI-compatible HTTP server — stdlib ``http.server`` only.

Endpoints:
  GET  /health
  GET  /v1/models
  POST /v1/completions        (non-streaming)
  POST /v1/chat/completions   (non-streaming, or SSE when "stream": true)

Accepts requests from any OpenAI client library (openai-python, curl, ...).
Works against the mock engine in tests; the real-model path is ``Engine``
with a .lac/.gguf model (see engine.py docstring for what that means today).
"""

from __future__ import annotations

import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _now() -> int:
    return int(time.time())


class _Handler(BaseHTTPRequestHandler):
    engine = None  # set by serve()
    model_id = "lacuna-mock"

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

    # -- routes --------------------------------------------------
    def do_GET(self):
        if self.path == "/health":
            self._send_json({"status": "ok"})
        elif self.path in ("/v1/models", "/v1/models/"):
            self._send_json(
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.model_id,
                            "object": "model",
                            "created": _now(),
                            "owned_by": "lacuna",
                        }
                    ],
                }
            )
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        body = self._read_json()
        if self.path == "/v1/completions":
            prompt = body.get("prompt", "")
            if isinstance(prompt, list):
                prompt = " ".join(map(str, prompt))
            max_tokens = int(body.get("max_tokens", 32))
            text = self.engine.generate(str(prompt), max_tokens=max_tokens)
            self._send_json(
                {
                    "id": f"cmpl-{uuid.uuid4().hex[:12]}",
                    "object": "text_completion",
                    "created": _now(),
                    "model": self.model_id,
                    "choices": [
                        {"text": text, "index": 0, "finish_reason": "stop"}
                    ],
                }
            )
        elif self.path == "/v1/chat/completions":
            messages = body.get("messages", [])
            prompt = " ".join(
                str(m.get("content", "")) for m in messages if isinstance(m, dict)
            )
            max_tokens = int(body.get("max_tokens", 32))
            if body.get("stream"):
                self._send_sse(prompt, max_tokens)
            else:
                text = self.engine.generate(prompt, max_tokens=max_tokens)
                self._send_json(
                    {
                        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                        "object": "chat_completion",
                        "created": _now(),
                        "model": self.model_id,
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": text},
                                "finish_reason": "stop",
                            }
                        ],
                    }
                )
        else:
            self._send_json({"error": "not found"}, 404)

    def _send_sse(self, prompt: str, max_tokens: int):
        text = self.engine.generate(prompt, max_tokens=max_tokens)
        cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        # stream word-by-word: simple, honest chunking of the final text
        for word in text.split(" "):
            chunk = {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": _now(),
                "model": self.model_id,
                "choices": [
                    {"index": 0, "delta": {"content": word + " "}, "finish_reason": None}
                ],
            }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")


def serve(engine, host: str = "127.0.0.1", port: int = 8080, model_id: str = "lacuna-mock"):
    """Start the OpenAI-compatible server (blocking)."""
    _Handler.engine = engine
    _Handler.model_id = model_id
    srv = ThreadingHTTPServer((host, port), _Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
