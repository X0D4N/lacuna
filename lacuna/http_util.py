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
from urllib.parse import urlparse

MAX_BODY_BYTES = 1024 * 1024
MAX_TOKENS_LIMIT = 4096


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
        str(m.get("content", "")) for m in messages if isinstance(m, dict)
    )
