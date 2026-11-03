"""One-command model fetch: ``lacuna pull <hf-id>``.

Downloads a GGUF from the HuggingFace Hub and auto-converts it to Lacuna's
``.lac`` paged format, so ``lacuna run <hf-id>`` just works with zero setup.

Stdlib only (``urllib``) — no ``requests``, no ``huggingface_hub``.

Pipeline: ``pull()``
  1. ``GET {api}/api/models/{hf_id}`` -> sibling file list (JSON).
  2. ``pick_gguf()`` -> best GGUF (Q4_K_M first, else smallest).
  3. ``download()`` -> ``~/.cache/lacuna/models/<owner>__<repo>/<file>.gguf``
     with progress callbacks and resume via ``Range`` headers.
  4. ``lacuna.convert.convert()`` -> ``<file>.lac`` (the REAL converter:
     parses the GGUF v3 header and streams tensor bytes; nothing is faked).
     ``pull()`` is idempotent: if the ``.lac`` already exists it is returned
     immediately without re-downloading.

Sharded repos (same ``lacuna pull <hf_id>`` command, no new interface):
when the picked file is one shard of a ``-NNNNN-of-MMMMM.gguf`` split,
``resolve_shard_set()`` expands it to the full set from the sibling list
— failing loud before any download when the set is incomplete or
inconsistent — and ``pull_sharded()`` downloads every shard (same resume
behavior), verifies each shard's presence and hub-listed size, then calls
``lacuna.convert.convert_sharded()`` for ONE ``.lac``. The converter's own
split validation (split.count / split.no / tensor counts / duplicates)
is the authority on shard consistency; pull never re-implements it.

``pull()`` takes ``api_base`` / ``opener`` parameters so tests run fully
offline against a local HTTP server; production defaults hit huggingface.co.

The Engine is still a clearly-labeled MOCK for inference (see engine.py);
this module builds the plumbing around it — fetching and converting real
weight files — without pretending the mock generates real model output.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

DEFAULT_API_BASE = "https://huggingface.co"
DEFAULT_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "lacuna")
CHUNK = 1024 * 256  # 256 KiB per read


class PullError(RuntimeError):
    """Clean, user-facing pull failure (network, 404, no GGUF, ...)."""


def _api_url(api_base: str, hf_id: str) -> str:
    return f"{api_base.rstrip('/')}/api/models/{hf_id}"


def _resolve_url(base: str, hf_id: str, filename: str) -> str:
    return f"{base.rstrip('/')}/{hf_id}/resolve/main/{filename}"


def list_gguf_files(hf_id: str, api_base: str = DEFAULT_API_BASE,
                    opener=None) -> list[dict]:
    """Return ``[{rfilename, size}]`` for GGUF siblings of a Hub repo.

    Raises ``PullError`` on network failure, non-200 status, or bad JSON.
    """
    opener = opener or urllib.request.urlopen
    url = _api_url(api_base, hf_id)
    try:
        with opener(url) as resp:
            status = getattr(resp, "status", 200) or 200
            if status != 200:
                raise PullError(f"model not found: {hf_id} (HTTP {status})")
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise PullError(f"model not found: '{hf_id}' — check the id "
                            f"(e.g. Qwen/Qwen3-32B)") from e
        raise PullError(f"hub request failed for '{hf_id}': HTTP {e.code}") from e
    except urllib.error.URLError as e:
        raise PullError(f"network error reaching the hub: {e.reason}") from e
    except (ValueError, KeyError) as e:
        raise PullError(f"unexpected hub response for '{hf_id}': {e}") from e
    siblings = payload.get("siblings", []) if isinstance(payload, dict) else []
    ggufs = [
        {"rfilename": s["rfilename"], "size": int(s.get("size") or 0)}
        for s in siblings
        if isinstance(s, dict) and str(s.get("rfilename", "")).lower().endswith(".gguf")
    ]
    if not ggufs:
        raise PullError(f"no .gguf files found in '{hf_id}'")
    return ggufs


# Filename tokens that mark an unquantized native-precision GGUF
# (BF16/F16). Token-based, not substring: "q4_k_m" tokenizes to
# {"q4", "k", "m"} and never matches, while "model-bf16.gguf" and
# "gemma-4-31B-it-bf16-00001-of-00002.gguf" both carry a "bf16" token.
_NATIVE_TOKENS = frozenset({"bf16", "f16", "fp16"})


def _is_native(filename: str) -> bool:
    """True when ``filename`` names an unquantized BF16/F16 GGUF."""
    tokens = set(re.split(r"[^A-Za-z0-9]+", filename.lower()))
    return bool(tokens & _NATIVE_TOKENS)


def pick_gguf(files: list[dict], prefer_native: bool = False) -> dict:
    """Pick the best GGUF: Q4_K_M first, else the smallest file.

    With ``prefer_native=True``, prefer an unquantized BF16/F16 file —
    or shard set, via :func:`resolve_shard_set` — over any quantized
    file; within native files the smallest wins. Falls back to the
    default preference when the listing holds no native file. Default
    (``False``) behavior is unchanged: existing callers and tests see
    exactly the old preference. Pure function — unit-testable without
    network.
    """
    if not files:
        raise PullError("no candidate files to pick from")
    if prefer_native:
        natives = [f for f in files if _is_native(f["rfilename"])]
        if natives:
            return min(natives, key=lambda f: f["size"])
    q4km = [f for f in files if "q4_k_m" in f["rfilename"].lower()]
    if q4km:
        return min(q4km, key=lambda f: f["size"])
    return min(files, key=lambda f: f["size"])


# Mirrors lacuna.convert's shard-filename pattern (``-NNNNN-of-MMMMM.gguf``).
_SHARD_RE = re.compile(r"-(\d+)-of-(\d+)\.gguf$", re.IGNORECASE)


def shard_info(filename: str):
    """``(index, total)`` from a ``-NNNNN-of-MMMMM.gguf`` filename, or None
    when the filename carries no shard numbering. Pure function."""
    m = _SHARD_RE.search(filename)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _shard_stem(filename: str) -> str:
    """Filename with the ``-NNNNN-of-MMMMM`` shard suffix stripped, so all
    shards of one split share one stem (``model-00001-of-00002.gguf`` and
    ``model-00002-of-00002.gguf`` both stem to ``model.gguf``)."""
    return re.sub(r"-\d+-of-\d+(\.gguf)$", r"\1", filename,
                  flags=re.IGNORECASE)


def resolve_shard_set(chosen: dict, files: list[dict]) -> list[dict] | None:
    """Shard set for ``chosen`` when it is one shard of a split model.

    Returns the set's files sorted by shard index, or None when ``chosen``
    is not a shard (the single-file path applies). Raises ``PullError``
    when the set is incomplete or inconsistent in the sibling listing —
    before any download starts, so a partial split never begins. A
    ``-00001-of-00001`` file is a degenerate single shard and flows
    through the single-file path (the converter accepts split.count=1).
    Pure function over the listing — unit-testable without network.
    """
    info = shard_info(chosen["rfilename"])
    if info is None:
        return None
    _, total = info
    if total <= 1:
        return None
    stem = _shard_stem(chosen["rfilename"])
    group = []
    for f in files:
        if _shard_stem(f["rfilename"]) != stem:
            continue
        fi = shard_info(f["rfilename"])
        if fi is None:
            continue
        group.append((fi[0], fi[1], f))
    totals = {t for _, t, _ in group}
    if totals != {total}:
        raise PullError(
            f"shard set '{stem}' is inconsistent: filename totals "
            f"{sorted(totals)} disagree with chosen shard total {total}")
    idxs = sorted(i for i, _, _ in group)
    if idxs != list(range(1, total + 1)):
        raise PullError(
            f"shard set '{stem}' is incomplete: shard indices {idxs} "
            f"do not cover 1..{total} (missing shard?)")
    return [f for _, _, f in sorted(group, key=lambda t: t[0])]


def _cache_paths(hf_id: str, filename: str,
                 cache_dir: str = DEFAULT_CACHE_DIR) -> tuple[str, str]:
    """Cache paths for a Hub file, traversal-proof.

    Both ``hf_id`` and ``filename`` ultimately come from a remote JSON
    response, so neither is trusted: the model directory must be a single
    plain path component, and the resolved file path must stay inside it
    (hub repos may legitimately nest GGUFs in subfolders — allowed, as
    long as the path cannot escape the model's cache directory).
    """
    safe = hf_id.replace("/", "__")
    if safe in ("", ".", "..") or "\\" in safe:
        raise PullError(f"unsafe model id: {hf_id!r}")
    model_dir = os.path.join(cache_dir, "models", safe)
    gguf_path = os.path.normpath(os.path.join(model_dir, filename))
    if os.path.commonpath(
        [os.path.abspath(gguf_path), os.path.abspath(model_dir)]
    ) != os.path.abspath(model_dir):
        raise PullError(
            f"unsafe filename from the hub: {filename!r} would escape the "
            f"cache directory; refusing to download it")
    lac_path = os.path.splitext(gguf_path)[0] + ".lac"
    return gguf_path, lac_path


def download(url: str, dest: str, progress=None, opener=None,
             download_base: str | None = None) -> str:
    """Download ``url`` -> ``dest`` with resume + progress.

    ``progress(downloaded_bytes, total_bytes_or_None)`` is called after every
    chunk. If ``dest`` already holds a prefix, a ``Range`` request resumes it;
    a server that ignores ``Range`` (HTTP 200 instead of 206) restarts cleanly.
    Returns ``dest``.
    """
    opener = opener or urllib.request.urlopen
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    have = os.path.getsize(dest) if os.path.exists(dest) else 0
    req = urllib.request.Request(url)
    if have > 0:
        req.add_header("Range", f"bytes={have}-")
    try:
        resp = opener(req)
    except urllib.error.HTTPError as e:
        if e.code == 416:
            # Range unsatisfiable. Genuinely complete only if the server's
            # total (Content-Range: bytes */TOTAL) equals what we hold; a
            # LONGER local file is stale garbage from an older/different
            # download and must restart, not be trusted as "complete".
            total_hdr = ""
            if e.headers is not None:
                total_hdr = e.headers.get("Content-Range", "") or ""
            m = re.search(r"/(\d+)\s*$", total_hdr)
            if m is None or int(m.group(1)) == have:
                return dest
            os.remove(dest)
            have = 0
            try:
                resp = opener(urllib.request.Request(url))
            except urllib.error.HTTPError as e2:
                raise PullError(f"download failed (HTTP {e2.code}): {url}") from e2
            except urllib.error.URLError as e2:
                raise PullError(
                    f"network error downloading {url}: {e2.reason}") from e2
        else:
            raise PullError(f"download failed (HTTP {e.code}): {url}") from e
    except urllib.error.URLError as e:
        raise PullError(f"network error downloading {url}: {e.reason}") from e
    with resp:
        status = getattr(resp, "status", 200) or 200
        if status == 206:
            mode, offset = "ab", have
        elif status == 200 and have > 0:
            # Server ignored Range: restart from scratch.
            mode, offset = "wb", 0
        elif status == 200:
            mode, offset = "wb", 0
        else:
            raise PullError(f"download failed (HTTP {status}): {url}")
        total = resp.getheader("Content-Length")
        total = (offset + int(total)) if total else None
        downloaded = offset
        with open(dest, mode) as f:
            while True:
                chunk = resp.read(CHUNK)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if progress is not None:
                    progress(downloaded, total)
    return dest


def pull_sharded(hf_id: str, shard_files: list[dict],
                 cache_dir: str = DEFAULT_CACHE_DIR, progress=None,
                 api_base: str = DEFAULT_API_BASE, opener=None,
                 precision_policy: str = "exact") -> str:
    """Fetch every shard of a split Hub model and convert to ONE ``.lac``.

    ``shard_files`` are sibling-list entries (``{rfilename, size}``) for one
    split; they are sorted by shard index here, so callers may pass them
    in listing order. Idempotent: if the ``.lac`` already exists it is
    returned immediately. Each shard download resumes via ``Range`` exactly
    like single-file pulls. Every shard is verified (present on disk, size
    equal to the hub-listed size) before conversion, and conversion itself
    reuses ``convert_sharded``'s split validation — a missing or
    mismatched shard fails loud and never yields a partial ``.lac``.
    """
    if not shard_files:
        raise PullError("no shard files given for sharded pull")
    ordered = []
    for f in shard_files:
        info = shard_info(f["rfilename"])
        if info is None:
            raise PullError(
                f"not a shard file: {f['rfilename']!r} (expected a "
                f"'-NNNNN-of-MMMMM.gguf' name)")
        ordered.append((info[0], f))
    ordered = [f for _, f in sorted(ordered, key=lambda t: t[0])]
    stem = _shard_stem(ordered[0]["rfilename"])
    _, lac_path = _cache_paths(hf_id, stem, cache_dir)
    if os.path.exists(lac_path):
        return lac_path
    sizes_known = all(f["size"] > 0 for f in ordered)
    total_all = sum(f["size"] for f in ordered) if sizes_known else None
    done_before = 0
    shard_paths = []
    for entry in ordered:
        gguf_path, _ = _cache_paths(hf_id, entry["rfilename"], cache_dir)
        url = _resolve_url(api_base, hf_id, entry["rfilename"])
        base = done_before
        if sizes_known:
            def _prog(done, total, _base=base):
                if progress is not None:
                    progress(_base + done, total_all)
            prog = _prog
        else:
            prog = progress
        try:
            download(url, gguf_path, progress=prog, opener=opener)
        except PullError:
            raise
        except OSError as e:
            raise PullError(f"cannot write to cache: {e}") from e
        if not os.path.exists(gguf_path):
            raise PullError(
                f"shard download missing on disk: {entry['rfilename']!r}")
        if entry["size"] > 0 and \
                os.path.getsize(gguf_path) != entry["size"]:
            raise PullError(
                f"shard size mismatch for {entry['rfilename']!r}: "
                f"downloaded {os.path.getsize(gguf_path)} bytes, hub "
                f"lists {entry['size']}")
        shard_paths.append(gguf_path)
        done_before += entry["size"]
    from .convert import convert_sharded as _convert_sharded

    try:
        _convert_sharded(shard_paths, lac_path, quiet=True,
                         precision_policy=precision_policy)
    except Exception as e:
        # Never leave a half-written .lac behind pretending to be a model.
        try:
            os.remove(lac_path)
        except OSError:
            pass
        raise PullError(
            f"conversion of sharded model '{hf_id}' failed: {e}") from e
    return lac_path


def pull(hf_id: str, cache_dir: str = DEFAULT_CACHE_DIR, progress=None,
         api_base: str = DEFAULT_API_BASE, opener=None,
         precision_policy: str = "exact",
         prefer_native: bool = False) -> str:
    """Fetch ``hf_id`` from the Hub and return the local ``.lac`` path.

    Idempotent: if the converted ``.lac`` already exists in the cache it is
    returned immediately (pull-if-missing). The GGUF download resumes across
    runs. Conversion uses the real ``lacuna.convert.convert`` — a corrupt or
    non-GGUF file raises here, loudly, instead of producing a fake ``.lac``.
    When the picked file is one shard of a split model, the whole shard set
    is fetched and converted via ``pull_sharded`` instead.
    ``prefer_native=True`` selects a BF16/F16 file (or shard set) over
    any quantized file; default (``False``) keeps the Q4_K_M-first
    preference.
    """
    files = list_gguf_files(hf_id, api_base=api_base, opener=opener)
    chosen = pick_gguf(files, prefer_native=prefer_native)
    shard_set = resolve_shard_set(chosen, files)
    if shard_set is not None:
        return pull_sharded(hf_id, shard_set, cache_dir=cache_dir,
                            progress=progress, api_base=api_base,
                            opener=opener,
                            precision_policy=precision_policy)
    gguf_path, lac_path = _cache_paths(hf_id, chosen["rfilename"], cache_dir)
    if os.path.exists(lac_path):
        return lac_path
    url = _resolve_url(api_base, hf_id, chosen["rfilename"])
    try:
        download(url, gguf_path, progress=progress, opener=opener)
    except PullError:
        raise
    except OSError as e:
        raise PullError(f"cannot write to cache: {e}") from e
    from .convert import convert as _convert

    try:
        _convert(gguf_path, lac_path, quiet=True,
                 precision_policy=precision_policy)
    except Exception as e:
        # Never leave a half-written .lac behind pretending to be a model.
        try:
            os.remove(lac_path)
        except OSError:
            pass
        raise PullError(f"conversion of '{gguf_path}' failed: {e}") from e
    return lac_path
