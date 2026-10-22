"""Round-3 showcase tests (2026-10-22, Fahad's directive: release tomorrow,
1000s of users, no delay, no error).

Pinned here:
- pack_q2's float32 fast path is bit-identical to the original float64
  algorithm — checked against an inline float64 reference on random data
  AND adversarial values clustered on the decision midpoints.
- chat_prompt flattens content=None to "" (never the literal text "None")
  and joins OpenAI content-parts lists.
- The OpenAI server under concurrent load: every request succeeds, an
  unsupported method gets a clean 405 JSON (never the stdlib HTML page),
  and an oversized body gets a clean 413 the client can actually read
  (the body is drained first, so no broken pipe).
"""

import json
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from lacuna import quant
from lacuna.http_util import LacunaHTTPServer, chat_prompt
from lacuna.server import _Handler


class _FakeBackend:
    mode = "mock"

    def generate(self, prompt, max_tokens=32):
        return "[mock] test reply"


@pytest.fixture()
def server():
    _Handler.engine = _FakeBackend()
    _Handler.model_id = "lacuna-mock"
    srv = LacunaHTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def _req(base, method, path, body=None, raw=None):
    data = raw if raw is not None else (
        json.dumps(body).encode() if body is not None else None
    )
    r = urllib.request.Request(
        base + path, data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(r, timeout=15) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


# ------------------------------------------------------------- codecs ---


def _pack_q2_float64_reference(x):
    """The pre-round-3 pack_q2 algorithm, verbatim semantics (float64)."""
    x = np.asarray(x, dtype=np.float32)
    scale = np.float32(np.max(np.abs(x)))
    enc = scale if scale != 0 else np.float32(1.0)
    xn = x / enc
    xn64 = xn.astype(np.float64)
    codes = np.zeros(xn.shape, dtype=np.uint8)
    for m in quant._Q2_MIDS:
        codes += (xn64 > m).astype(np.uint8)
    near = np.zeros(xn.shape, dtype=bool)
    for m in quant._Q2_MIDS:
        near |= np.abs(xn64 - m) <= quant._Q2_FIXUP_BAND
    if near.any():
        sub = xn[near]
        dist = np.abs(sub[..., None] - quant._Q2_LEVELS)
        codes[near] = np.argmin(dist, axis=-1).astype(np.uint8)
    flat = codes.reshape(-1)
    pad = (-flat.size) % 4
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.uint8)])
    packed = (
        flat[0::4] | (flat[1::4] << 2) | (flat[2::4] << 4) | (flat[3::4] << 6)
    ).astype(np.uint8)
    return packed, scale, x.shape


def test_pack_q2_float32_path_bit_identical():
    rng = np.random.default_rng(42)
    cases = [
        rng.standard_normal(100_003).astype(np.float32),
        (rng.standard_normal(50_000) * 1000).astype(np.float32),
        np.zeros(1024, dtype=np.float32),
    ]
    for m in quant._Q2_MIDS:
        # adversarial: dense cluster within +/-3e-5 of each midpoint
        base = np.full(60_000, m, dtype=np.float64)
        offs = rng.uniform(-3e-5, 3e-5, size=60_000)
        cases.append((base + offs).astype(np.float32))
        cases.append(np.full(4096, np.float32(m), dtype=np.float32))
    for x in cases:
        got = quant.pack_q2(x)
        want = _pack_q2_float64_reference(x)
        assert np.array_equal(got[0], want[0])
        assert got[1] == want[1] and got[2] == want[2]


def test_unpack_q4_roundtrip_still_exact():
    rng = np.random.default_rng(7)
    x = rng.standard_normal((257, 131)).astype(np.float32)
    packed, scale, shape = quant.pack_q4(x)
    rt = quant.unpack_q4(packed, scale, shape)
    assert rt.shape == shape
    # Q4 grid: decoded values are exactly the codebook values of the input
    dec = quant.unpack_q4(*quant.pack_q4(rt))
    assert np.array_equal(dec, rt)


# -------------------------------------------------------- chat_prompt ---


def test_chat_prompt_none_content_is_empty_not_literal():
    body = {"messages": [{"role": "user", "content": None},
                         {"role": "user", "content": "hi"}]}
    out = chat_prompt(body)
    assert "None" not in out
    assert out.strip() == "hi"


def test_chat_prompt_content_parts_list():
    body = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "hello"},
        {"type": "text", "text": "world"},
    ]}]}
    assert chat_prompt(body) == "hello world"


# ------------------------------------------------------------- server ---


def test_concurrent_load_all_succeed(server):
    def one(i):
        return _req(server, "POST", "/v1/chat/completions",
                    {"model": "lacuna-mock",
                     "messages": [{"role": "user", "content": "hi"}],
                     "max_tokens": 8})
    with ThreadPoolExecutor(max_workers=16) as ex:
        results = list(ex.map(one, range(80)))
    assert all(s == 200 for s, _ in results), results[:3]


def test_unsupported_method_is_clean_json_405(server):
    status, body = _req(server, "DELETE", "/v1/models")
    assert status == 405
    assert "<html" not in body.lower()
    assert "error" in json.loads(body)


def test_oversized_body_gets_readable_413(server):
    status, body = _req(server, "POST", "/v1/completions",
                        raw=b"x" * (2 * 1024 * 1024))
    assert status == 413
    assert "error" in json.loads(body)
