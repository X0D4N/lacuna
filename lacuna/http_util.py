"""Shared request validation for Lacuna's HTTP servers (server.py,
ollama_shim.py) — stdlib only.

Every client-caused problem raises :class:`RequestError` with an HTTP
status and ONE human sentence saying what to do instead; handlers turn it
into a clean JSON ``{"error": ...}`` response. No handler ever sends a
stack trace to a client, and no request can make a handler read an
unbounded body into memory.

Limits (documented, deliberate):
- ``MAX_BODY_BYTES`` = 1 MiB. Prompts are text; model weights are never
  POSTed. A larger body is a client bug or an attack, rejected with 413
  before a single byte is read.
- ``MAX_TOKENS_LIMIT`` = 4096. ``max_tokens`` above the cap is clamped,
  not rejected (matching hosted-API behavior); a non-numeric value is a
  400.
"""

from __future__ import annotations

import json
from http.server import ThreadingHTTPServer
from urllib.parse import urlparse

MAX_BODY_BYTES = 1024 * 1024
MAX_TOKENS_LIMIT = 4096


class LacunaHTTPServer(ThreadingHTTPServer):
    """Threading server tuned for a public showcase, not a quiet demo.

    The stdlib default listen backlog (``request_queue_size = 5``) drops
    connections with RST the moment a few dozen clients connect at once —
    measured in the round-3 load test (6 of 240 requests reset at 32
    concurrent clients). 128 absorbs realistic bursts; daemon threads keep
    a stuck client from blocking shutdown.
    """

    daemon_threads = True
    request_queue_size = 128


class JsonErrors:
    """Mixin for Lacuna's handlers: framework-level errors leave as JSON.

    ``BaseHTTPRequestHandler.send_error`` answers unimplemented methods
    (501), malformed request lines (400) and the like with an HTML page —
    an API client should never have to parse HTML to learn what went
    wrong. Handlers using this mixin must provide ``_send_json``.
    Unsupported-but-common methods get a clean 405 instead of a 501.
    """

    def send_error(self, code, message=None, explain=None):  # noqa: D102
        try:
            self._send_json(
                {"error": message or "request failed"}, int(code)
            )
        except Exception:
            pass  # the client is already gone; nothing sensible left to do

    def _method_not_allowed(self):
        self._send_json(
            {"error": "method not allowed on this endpoint; use GET or POST."},
            405,
        )

    do_PUT = _method_not_allowed
    do_DELETE = _method_not_allowed
    do_PATCH = _method_not_allowed


class RequestError(Exception):
    """A client-caused request problem: HTTP status + one clear sentence."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = int(status)
        self.message = message


def request_path(raw_path: str) -> str:
    """Path component of a request target (query string stripped)."""
    return urlparse(raw_path).path


def read_json_body(handler) -> dict:
    """Read and parse the request body as a JSON object, with limits.

    Returns ``{}`` when there is no body. Raises :class:`RequestError`
    (400/413) for an unparseable Content-Length, an oversized body,
    invalid JSON, or a JSON value that is not an object.
    """
    raw_len = handler.headers.get("Content-Length")
    if raw_len is None:
        return {}
    try:
        n = int(raw_len)
    except ValueError:
        raise RequestError(
            400,
            "the Content-Length header is not a number; send a JSON body "
            "with a valid Content-Length, or no body at all.",
        ) from None
    if n <= 0:
        return {}
    if n > MAX_BODY_BYTES:
        # Drain what the client is already sending (bounded, streamed to
        # /dev/null — memory stays flat) before answering. Replying while
        # the client is mid-upload makes the kernel reset the connection
        # and the client sees a broken pipe instead of this clean 413
        # (round-3 chaos test caught exactly that).
        _drain(handler, n)
        raise RequestError(
            413,
            f"the request body is {n} bytes, over the {MAX_BODY_BYTES}-byte "
            "limit; send a shorter prompt.",
        )
    raw = handler.rfile.read(n)
    try:
        body = json.loads(raw.decode("utf-8") or "{}")
    except (ValueError, UnicodeDecodeError):
        raise RequestError(
            400,
            'the request body is not valid JSON; send a JSON object like '
            '{"prompt": "hello"}.',
        ) from None
    if not isinstance(body, dict):
        raise RequestError(
            400,
            "the request body must be a JSON object; wrap your fields in "
            "{ ... }.",
        )
    return body


# How much of an oversized body to drain before answering 413. Enough for
# any honest client that slightly overshoots the cap; a client declaring
# more than this is abusive and gets its reset.
_DRAIN_CAP = 4 * MAX_BODY_BYTES

# Per-read stall tolerance while draining. The timeout applies to each
# recv, not the whole drain: a client steadily streaming a big body never
# trips it, while a client that declares a huge body and then sends
# nothing gets its 413 after ~1s instead of hanging a handler thread.
_DRAIN_STALL_SECS = 1.0


def _drain(handler, n: int) -> None:
    """Discard up to ``_DRAIN_CAP`` bytes of a request body, best-effort."""
    remaining = min(n, _DRAIN_CAP)
    try:
        handler.connection.settimeout(_DRAIN_STALL_SECS)
        while remaining > 0:
            chunk = handler.rfile.read(min(65536, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
    except OSError:
        pass  # timeout / reset while draining: the 413 still goes out


def clamp_max_tokens(body: dict, default: int = 32) -> int:
    """``max_tokens`` from a request body: numeric, clamped to [1, cap].

    Raises :class:`RequestError` (400) when the value is not a number.
    """
    v = body.get("max_tokens", default)
    if v is None:
        return default
    if isinstance(v, bool) or not isinstance(v, (int, float, str)):
        raise RequestError(400, "max_tokens must be a number; omit it to use the default.")
    try:
        n = int(v)
    except (TypeError, ValueError, OverflowError):
        raise RequestError(
            400, "max_tokens must be a number; omit it to use the default."
        ) from None
    return max(1, min(n, MAX_TOKENS_LIMIT))


def _content_text(content) -> str:
    """One message's content as text.

    ``None`` means "no content" (some clients send it) — it must flatten to
    an empty string, never to the literal text ``"None"`` (the round-3 chaos
    test caught exactly that silent wrongness). A list is the OpenAI
    content-parts shape: join the parts' ``text`` fields.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(p.get("text", "")) for p in content if isinstance(p, dict)
        )
    return str(content)


def chat_prompt(body: dict) -> str:
    """Flatten chat ``messages`` to a prompt string, validating the shape."""
    messages = body.get("messages", [])
    if not isinstance(messages, list):
        raise RequestError(
            400,
            'messages must be a list of {"role": ..., "content": ...} '
            "objects.",
        )
    return " ".join(
        _content_text(m.get("content", ""))
        for m in messages
        if isinstance(m, dict)
    )
