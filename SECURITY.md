# Security

## Threat model

Lacuna is **local-first** software for running large models on your own
machine:

- **No telemetry.** Lacuna never phones home. The only network traffic it
  initiates is `lacuna pull` talking to the HuggingFace Hub (and only when
  you ask it to fetch a model). The servers and the MCP process make no
  outbound connections at all.
- **No credentials are stored.** Lacuna has no accounts and no API keys.
  If you download gated HuggingFace models you do that with your own
  tools/credentials; Lacuna never sees, stores, or transmits them.
- **Trust boundary = your user account.** Anyone who can run code as your
  OS user can already read your model files. Lacuna's job is to not widen
  that boundary: it does not expose your machine to the network by
  default, and it treats every file and request it parses as untrusted
  input that must fail loud, never silently corrupt.

## Network surfaces

| Surface | Transport | Default bind | Auth |
|---|---|---|---|
| OpenAI-compatible server (`lacuna serve`) | HTTP | `127.0.0.1:8080` | **none** |
| Ollama-compatible API (`lacuna serve --ollama`) | HTTP | `127.0.0.1:11434` | **none** |
| MCP server (`lacuna mcp`) | stdio | — (your agent spawns it) | process trust |

Both HTTP servers bind **localhost only** unless you explicitly pass
`--host`. There is **no authentication**: if you bind a server to a LAN
or public address (`--host 0.0.0.0`), anyone who can reach that port can
generate text with your models and read your model list. That may be
exactly what you want on a trusted LAN — it is your decision to make,
and it is never the default.

Request hardening (both HTTP servers, `lacuna/http_util.py`):

- Request bodies are capped at **1 MiB** (413 before the body is read).
- `max_tokens` is clamped to **4096**; non-numeric values are a 400.
- Bodies must be JSON **objects** with correctly-typed fields; every
  client error returns `{"error": "<one sentence + what to do>"}` with a
  4xx status. Stack traces are never sent to clients.
- Streaming responses tolerate clients disconnecting mid-stream.

## File handling

- **`.lac` store reader** (`lacuna/pager.py`): the header, page table and
  page headers are cross-validated on open and at fault time — bad magic,
  unknown version/policy, truncated tables, table/header precision
  mismatch, offsets outside the file, and declared payloads larger than
  the file all raise a clear `ValueError` instead of decoding garbage.
  (Payload *byte* corruption is not detectable from the format itself in
  v0.2 — no silent wrongness is claimed there; see below.)
- **Warm-start cache** (`lacuna/warmstart.py`): a convenience snapshot,
  never a source of truth. It is accepted only if the store fingerprint
  (size, mtime, page count, page size) matches, and every page is
  re-read from the `.lac` file and re-verified against its recorded
  CRC32 before use. Any mismatch degrades to a normal cold start. CRC32
  is an integrity check against corruption, not a security MAC — the
  threat model is a same-user local file.
- **Downloads** (`lacuna/pull.py`): filenames come from the hub's JSON
  and are treated as hostile: the resolved path must stay inside the
  model's cache directory or the pull is refused. Resumed downloads
  validate 416 responses against the server's declared total, so a stale
  over-long prefix is restarted instead of trusted. Downloads are plain
  HTTPS via urllib with the system CA store.
- **Conversion** (`lacuna/convert.py`): parses GGUF headers defensively;
  a corrupt or non-GGUF file fails the conversion loudly and the partial
  `.lac` is removed rather than left behind pretending to be a model.

## Deserialization

Lacuna uses **no pickle and no `eval`-style deserialization anywhere**.
Parsers are `struct` (fixed binary layouts with explicit length checks)
and `json` (with type validation at every trust boundary: HTTP bodies,
JSON-RPC messages, hub API responses).

## Reporting

Found a vulnerability? Open a private security advisory on the GitHub
repository (Security → Report a vulnerability) rather than a public
issue. Please include a minimal reproduction.
