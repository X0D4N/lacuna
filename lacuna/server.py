"""OpenAI-compatible HTTP server — stdlib ``http.server`` only.

Endpoints:
  GET  /health
  GET  /v1/models
  POST /v1/completions        (non-streaming)
  POST /v1/chat/completions   (non-streaming, or SSE when "stream": true)

Accepts requests from any OpenAI client library (openai-python, curl, ...).
Works against the mock engine in tests; the real-model path is ``Engine``
with a .lac/.gguf model (see engine.py docstring for what that means today).

Binds localhost by default and has NO authentication (see SECURITY.md).
Request bodies are capped at 1 MiB and ``max_tokens`` at 4096; client
errors return clean JSON (``lacuna/http_util.py``), never stack traces.
"""

from __future__ import annotations

import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING

from .http_util import (
    RequestError,
    chat_prompt,
    clamp_max_tokens,
    read_json_body,
    request_path,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .ollama_shim import InferenceBackend


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

    def log_message(self, *a):  # quiet
        pass

    # -- routes --------------------------------------------------
    def do_GET(self):
        path = request_path(self.path)
        if path == "/health":
            self._send_json({"status": "ok"})
        elif path in ("/v1/models", "/v1/models/"):
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
        # Every failure leaves as clean JSON: RequestError for client
        # problems (400/413 with one human sentence), a plain 500 for
        # engine failures. Stack traces never reach the client.
        try:
            body = read_json_body(self)
            path = request_path(self.path)
            if path == "/v1/completions":
                prompt = body.get("prompt", "")
                if isinstance(prompt, list):
                    prompt = " ".join(map(str, prompt))
                elif not isinstance(prompt, str):
                    raise RequestError(
                        400,
                        "prompt must be a string (or a list of strings); "
                        "send {\"prompt\": \"...\"}.",
                    )
                max_tokens = clamp_max_tokens(body)
                text = self.engine.generate(prompt, max_tokens=max_tokens)
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
            elif path == "/v1/chat/completions":
                prompt = chat_prompt(body)
                max_tokens = clamp_max_tokens(body)
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
        except RequestError as e:
            self._send_json({"error": e.message}, e.status)
        except (BrokenPipeError, ConnectionResetError):
            pass  # client went away mid-response: normal, not an error
        except Exception as e:  # engine failure: one clean line, no trace
            self._send_json({"error": f"generation failed: {e}"}, 500)

    def _send_sse(self, prompt: str, max_tokens: int):
        text = self.engine.generate(prompt, max_tokens=max_tokens)
        cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        # stream word-by-word: simple, honest chunking of the final text
        try:
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
        except (BrokenPipeError, ConnectionResetError):
            pass  # client disconnected mid-stream


def serve(engine: "InferenceBackend", host: str = "127.0.0.1", port: int = 8080,
          model_id: str = "lacuna-mock") -> None:
    """Start the OpenAI-compatible server (blocking).

    Binds ``host`` — localhost by default; pass a LAN address only if you
    mean to expose the server (it has no authentication, see SECURITY.md).
    """
    _Handler.engine = engine
    _Handler.model_id = model_id
    srv = ThreadingHTTPServer((host, port), _Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
