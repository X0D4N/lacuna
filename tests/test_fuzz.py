"""Fuzz / property tests: randomized malformed inputs to the .lac reader,
the warm-start restore, both HTTP servers, the MCP dispatcher, and the
CLI parser. Invariants for every input: fail loud or answer cleanly —
never hang, never emit a stack trace to a client, never crash with an
unexpected exception type. All RNGs are seeded: failures reproduce.
"""

import json
import struct
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from lacuna.cli import build_parser
from lacuna.engine import Engine
from lacuna.mcp_server import handle as mcp_handle
from lacuna.ollama_shim import build_ollama_server
from lacuna.pager import LacWriter, PagedWeightStore
from lacuna.quant import Q4
from lacuna.server import _Handler

LOUD = (ValueError, KeyError, IndexError, struct.error, OSError)


@pytest.fixture()
def lac_bytes(tmp_path):
    p = str(tmp_path / "w.lac")
    rng = np.random.default_rng(3)
    with LacWriter(p, page_size=4096) as w:
        w.add_tensor("a", rng.standard_normal((32, 16)).astype(np.float32),
                     layer=0, precision=Q4)
        w.add_tensor("b", rng.standard_normal((48, 16)).astype(np.float32),
                     layer=0, precision=Q4)
    return open(p, "rb").read()


def test_lac_reader_mutation_fuzz(tmp_path, lac_bytes):
    rng = np.random.default_rng(1234)
    outcomes = {"rejected": 0, "opened": 0}
    for it in range(60):
        data = bytearray(lac_bytes)
        kind = it % 3
        if kind == 0:  # byte flips
            for _ in range(rng.integers(1, 6)):
                data[rng.integers(0, len(data))] = rng.integers(0, 256)
        elif kind == 1:  # truncate
            data = data[: int(rng.integers(0, len(data)))]
        else:  # append garbage
            data += rng.bytes(int(rng.integers(1, 512)))
        p = tmp_path / f"fuzz{it}.lac"
        p.write_bytes(bytes(data))
        try:
            s = PagedWeightStore(str(p), hot_budget_bytes=10**9)
        except LOUD:
            outcomes["rejected"] += 1
            continue
        outcomes["opened"] += 1
        try:
            for i in range(s.n_pages):
                try:
                    arr = s.get_page(i)
                    assert isinstance(arr, np.ndarray)
                except LOUD:
                    pass  # loud at fault time: correct
        finally:
            s.close()
    # the fuzz must actually exercise both paths to mean anything
    assert outcomes["rejected"] > 0
    assert outcomes["opened"] > 0


def test_warmstart_restore_fuzz(tmp_path, lac_bytes):
    p = tmp_path / "w.lac"
    p.write_bytes(lac_bytes)
    rng = np.random.default_rng(77)
    with PagedWeightStore(str(p), hot_budget_bytes=10**9) as s:
        for it in range(30):
            cache = tmp_path / f"c{it}.hotset"
            cache.write_bytes(rng.bytes(int(rng.integers(0, 300))))
            rep = s.restore_hotset(str(cache))  # must never raise
            assert rep["restored"] == 0
        s.get_page(0)  # store unharmed after every fuzzed restore
        assert s.stats["faults"] >= 1


# ---------------------------------------------------------------------------
# HTTP servers
# ---------------------------------------------------------------------------

def _rand_json_value(rng, depth=0):
    choices = ["int", "str", "bool", "none", "float"]
    if depth < 2:
        choices += ["list", "dict"]
    kind = choices[rng.integers(0, len(choices))]
    if kind == "int":
        return int(rng.integers(-10**6, 10**6))
    if kind == "str":
        return rng.bytes(int(rng.integers(0, 12))).decode("latin-1")
    if kind == "bool":
        return bool(rng.integers(0, 2))
    if kind == "none":
        return None
    if kind == "float":
        return float(rng.standard_normal())
    if kind == "list":
        return [_rand_json_value(rng, depth + 1)
                for _ in range(rng.integers(0, 3))]
    return {f"k{i}": _rand_json_value(rng, depth + 1)
            for i in range(rng.integers(0, 3))}


def _post(base, path, data: bytes):
    req = urllib.request.Request(
        base + path, data=data,
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def _fuzz_server(base, paths, seed):
    rng = np.random.default_rng(seed)
    for _ in range(30):
        path = paths[rng.integers(0, len(paths))]
        if rng.integers(0, 2):
            payload = rng.bytes(int(rng.integers(0, 200)))
        else:
            payload = json.dumps(_rand_json_value(rng)).encode("latin-1")
        code, body = _post(base, path, payload)
        assert code in (200, 400, 404, 413, 500)
        assert b"Traceback" not in body
        json.loads(body)  # every response is valid JSON


def test_openai_server_fuzz():
    eng = Engine("mock")
    _Handler.engine = eng
    _Handler.model_id = "lacuna-mock"
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        _fuzz_server(base, ["/v1/completions", "/v1/chat/completions",
                            "/v1/models", "/nope"], seed=42)
    finally:
        srv.shutdown()
        srv.server_close()
        eng.close()


class _FakeBackend:
    mode = "mock"

    def generate(self, prompt: str, max_tokens: int = 32) -> str:
        return "[mock] fuzz reply"


def test_ollama_server_fuzz(tmp_path):
    srv = build_ollama_server(_FakeBackend(), port=0, model_name="lacuna",
                              cache_dir=str(tmp_path))
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        _fuzz_server(base, ["/api/generate", "/api/chat", "/api/tags",
                            "/nope"], seed=43)
    finally:
        srv.shutdown()


# ---------------------------------------------------------------------------
# MCP + CLI
# ---------------------------------------------------------------------------

def test_mcp_handle_fuzz():
    rng = np.random.default_rng(9)
    methods = ["initialize", "tools/list", "tools/call", "ping", "bogus", ""]
    for _ in range(60):
        msg = _rand_json_value(rng)
        if isinstance(msg, dict) and rng.integers(0, 2):
            msg["method"] = methods[rng.integers(0, len(methods))]
            if msg["method"] == "tools/call":
                msg["params"] = {
                    "name": ["lacuna_generate", "bogus"][rng.integers(0, 2)],
                    "arguments": _rand_json_value(rng),
                }
        r = mcp_handle(msg)  # must never raise
        assert r is None or isinstance(r, dict)


def test_cli_parser_fuzz():
    rng = np.random.default_rng(4)
    tokens = ["run", "serve", "pull", "convert", "mcp", "bench", "mock",
              "--max-tokens", "--port", "--exact", "--extreme", "--ollama",
              "model.gguf", "x", "0", "-1", "--help", "oracle-recipe",
              "--list-families", "perplexity"]
    parser = build_parser()
    for _ in range(80):
        argv = [tokens[rng.integers(0, len(tokens))]
                for _ in range(rng.integers(0, 5))]
        try:
            parser.parse_args(argv)
        except SystemExit:
            pass  # argparse's clean exit: the only allowed way out
