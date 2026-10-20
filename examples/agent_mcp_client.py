"""Example 2 — agent integration: drive Lacuna's MCP server from Python.

This is exactly how an agent framework (Hermes-style) talks to Lacuna:
spawn `lacuna mcp` as a subprocess and speak newline-delimited JSON-RPC
2.0 over stdio. Stdlib only, no MCP client library needed.

    python examples/agent_mcp_client.py

It initializes the session, lists the tools, and calls lacuna_generate
against the clearly-labeled mock backend.
"""

import json
import subprocess
import sys


def rpc(proc: subprocess.Popen, method: str, params: dict | None = None,
        msg_id: int = 1) -> dict:
    msg = {"jsonrpc": "2.0", "id": msg_id, "method": method,
           "params": params or {}}
    proc.stdin.write(json.dumps(msg) + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError("lacuna mcp closed the connection unexpectedly")
    return json.loads(line)


def main() -> None:
    proc = subprocess.Popen(
        [sys.executable, "-m", "lacuna.mcp_server"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    try:
        hello = rpc(proc, "initialize", {}, 1)
        print("server:", hello["result"]["serverInfo"])

        tools = rpc(proc, "tools/list", {}, 2)
        print("tools:", [t["name"] for t in tools["result"]["tools"]])

        reply = rpc(proc, "tools/call", {
            "name": "lacuna_generate",
            "arguments": {"prompt": "hello from an agent", "max_tokens": 8},
        }, 3)
        print("lacuna_generate ->",
              reply["result"]["content"][0]["text"])
    finally:
        proc.stdin.close()
        proc.wait(timeout=10)


if __name__ == "__main__":
    main()
