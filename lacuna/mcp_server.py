"""MCP server over stdio — stdlib only, newline-delimited JSON-RPC 2.0.

Lets open-source agents (Hermes and others) call Lacuna with zero glue::

    {"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}
    {"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}
    {"jsonrpc":"2.0","id":3,"method":"tools/call",
     "params":{"name":"lacuna_generate","arguments":{"prompt":"hi"}}}

Tools:
  lacuna_generate  {prompt, max_tokens?} -> generated text
  lacuna_convert    {input_gguf, output_spp?} -> conversion stats
  lacuna_benchmark  {scenario?} -> simulator table (SIMULATED numbers)

Example client config (e.g. Claude Desktop / Hermes)::

    {
      "mcpServers": {
        "lacuna": {
          "command": "lacuna",
          "args": ["mcp"]
        }
      }
    }
"""

from __future__ import annotations

import json
import os
import sys
import traceback

_PROTOCOL_VERSION = "2024-11-05"
_SERVER_INFO = {"name": "lacuna", "version": "0.3.0"}
# One JSON-RPC message per line; a line past this cap is rejected, never
# buffered unboundedly or parsed.
MAX_LINE_BYTES = 10 * 1024 * 1024
MAX_TOKENS_LIMIT = 4096

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        from .engine import Engine

        _engine = Engine("mock")
    return _engine


def _tools():
    return [
        {
            "name": "lacuna_generate",
            "description": "Generate text with the Lacuna engine (mock target in v0.3.0).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "max_tokens": {"type": "integer", "default": 64},
                },
                "required": ["prompt"],
            },
        },
        {
            "name": "lacuna_convert",
            "description": "Convert a GGUF model to Lacuna .lac paged format.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input_gguf": {"type": "string"},
                    "output_spp": {"type": "string"},
                },
                "required": ["input_gguf"],
            },
        },
        {
            "name": "lacuna_benchmark",
            "description": "Run the discrete-event hardware simulator; returns SIMULATED tok/s table.",
            "inputSchema": {
                "type": "object",
                "properties": {"scenario": {"type": "string"}},
            },
        },
    ]


def _max_tokens(args: dict, default: int = 64) -> int:
    v = args.get("max_tokens", default)
    if isinstance(v, bool) or not isinstance(v, (int, float, str)):
        raise ValueError("max_tokens must be a number")
    try:
        n = int(v)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("max_tokens must be a number") from None
    return max(1, min(n, MAX_TOKENS_LIMIT))


def _call_tool(name: str, args: dict) -> dict:
    if name == "lacuna_generate":
        prompt = args.get("prompt", "")
        if not isinstance(prompt, str):
            raise ValueError("prompt must be a string")
        text = _get_engine().generate(prompt, _max_tokens(args))
        return {"content": [{"type": "text", "text": text}]}
    if name == "lacuna_convert":
        from .convert import convert as _convert

        src = args.get("input_gguf")
        if not isinstance(src, str) or not src:
            raise ValueError(
                "lacuna_convert needs 'input_gguf': the path to a .gguf file")
        if not os.path.exists(src):
            raise ValueError(f"input file not found: {src} — check the path")
        dst = args.get("output_spp") or (src + ".lac")
        if not isinstance(dst, str):
            raise ValueError("output_spp must be a string path")
        stats = _convert(src, dst, quiet=True)
        return {"content": [{"type": "text", "text": json.dumps(stats, default=str)}]}
    if name == "lacuna_benchmark":
        from simulator.simulate import markdown_table

        return {"content": [{"type": "text", "text": markdown_table() + "\n(all numbers SIMULATED)"}]}
    return {"content": [{"type": "text", "text": f"unknown tool: {name}"}], "isError": True}


def handle(msg: dict) -> dict | None:
    """Dispatch one JSON-RPC message. Returns response dict or None (notification)."""
    if not isinstance(msg, dict):
        return {
            "jsonrpc": "2.0",
            "id": None,
            "error": {"code": -32600,
                      "message": "invalid request: expected a JSON object"},
        }
    method = msg.get("method", "")
    mid = msg.get("id")
    params = msg.get("params", {}) or {}
    if not isinstance(params, dict):
        return {
            "jsonrpc": "2.0",
            "id": mid,
            "error": {"code": -32602,
                      "message": "invalid params: expected a JSON object"},
        }

    def ok(result):
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    def err(code, message):
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}}

    if method == "initialize":
        return ok(
            {
                "protocolVersion": _PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": _SERVER_INFO,
            }
        )
    if method == "notifications/initialized":
        return None
    if method == "tools/list":
        return ok({"tools": _tools()})
    if method == "tools/call":
        args = params.get("arguments", {}) or {}
        if not isinstance(args, dict):
            return ok(
                {
                    "content": [{"type": "text",
                                 "text": "tool error: 'arguments' must be an object"}],
                    "isError": True,
                }
            )
        try:
            return ok(_call_tool(params.get("name", ""), args))
        except Exception as e:  # noqa: BLE001 — surface tool errors over the wire
            # The client gets ONE clean line; the traceback goes to stderr
            # for the operator only (it can contain local paths).
            traceback.print_exc(limit=3, file=sys.stderr)
            return ok(
                {
                    "content": [{"type": "text", "text": f"tool error: {e}"}],
                    "isError": True,
                }
            )
    if method == "ping":
        return ok({})
    return err(-32601, f"method not found: {method}")


def main() -> int:
    inp, out = sys.stdin, sys.stdout
    for line in inp:
        line = line.strip()
        if not line:
            continue
        if len(line) > MAX_LINE_BYTES:
            out.write(json.dumps({
                "jsonrpc": "2.0", "id": None,
                "error": {"code": -32600,
                          "message": "request too large; send one smaller "
                                     "JSON-RPC message per line"},
            }) + "\n")
            out.flush()
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            out.write(json.dumps({
                "jsonrpc": "2.0", "id": None,
                "error": {"code": -32700, "message": "parse error: not valid JSON"},
            }) + "\n")
            out.flush()
            continue
        resp = handle(msg)
        if resp is not None:
            out.write(json.dumps(resp) + "\n")
            out.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
