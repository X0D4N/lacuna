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

``pull()`` takes ``api_base`` / ``opener`` parameters so tests run fully
offline against a local HTTP server; production defaults hit huggingface.co.

The Engine is still a clearly-labeled MOCK for inference (see engine.py);
this module builds the plumbing around it — fetching and converting real
weight files — without pretending the mock generates real model output.
"""

from __future__ import annotations

import json
import os
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


def pick_gguf(files: list[dict]) -> dict:
    """Pick the best GGUF: Q4_K_M first, else the smallest file.

    Pure function — unit-testable without network.
    """
    if not files:
        raise PullError("no candidate files to pick from")
    q4km = [f for f in files if "q4_k_m" in f["rfilename"].lower()]
    if q4km:
        return min(q4km, key=lambda f: f["size"])
    return min(files, key=lambda f: f["size"])


def _cache_paths(hf_id: str, filename: str,
                 cache_dir: str = DEFAULT_CACHE_DIR) -> tuple[str, str]:
    safe = hf_id.replace("/", "__")
    model_dir = os.path.join(cache_dir, "models", safe)
    gguf_path = os.path.join(model_dir, filename)
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
        # 416 = range unsatisfiable (file already complete); treat as done.
        if e.code == 416:
            return dest
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


def pull(hf_id: str, cache_dir: str = DEFAULT_CACHE_DIR, progress=None,
         api_base: str = DEFAULT_API_BASE, opener=None,
         precision_policy: str = "exact") -> str:
    """Fetch ``hf_id`` from the Hub and return the local ``.lac`` path.

    Idempotent: if the converted ``.lac`` already exists in the cache it is
    returned immediately (pull-if-missing). The GGUF download resumes across
    runs. Conversion uses the real ``lacuna.convert.convert`` — a corrupt or
    non-GGUF file raises here, loudly, instead of producing a fake ``.lac``.
    """
    files = list_gguf_files(hf_id, api_base=api_base, opener=opener)
    chosen = pick_gguf(files)
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
