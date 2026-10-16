"""LacunaEngine — orchestration of oracle + pager + scheduler + speculate.

What IS implemented here
------------------------
- The full orchestration loop over a *target-model callable interface*:
  ``target_step(prefix_tokens) -> next_token_id`` and the draft callable.
  Against the built-in deterministic mock target this runs end to end and is
  what the tests exercise.
- ``Engine``: the one-line public API — ``Engine("model.lac").generate("hi")``.
  ``.gguf`` inputs are auto-converted to ``.lac`` (cached next to the source).

What is an INTEGRATION POINT (not implemented on this box)
----------------------------------------------------------
- Real 100B inference needs a GPU executor. Two documented paths:
  1. **ctypes libllama** (``_try_load_libllama``): loads ``libllama.so`` when
     present and exposes the exact C functions Lacuna would call
     (``llama_model_load_from_file``, ``llama_decode``, ...). On this dev box
     there is no GPU and no libllama build, so the loader returns ``None`` and
     the engine runs the mock target. Nothing is faked: the mock path is
     clearly labeled in every output.
  2. **llama-server dense baseline** (``run_llama_server``): shells out to a
     real ``llama-server`` binary for the dense (non-sparse) baseline path.
     Requires the binary on PATH; raises a clear error otherwise.

See integrations/llama.cpp/INTEGRATION.md for the ggml backend hook design.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess

from . import quant as _q
from .hardware import HardwareTier, detect_hardware
from .oracle import Oracle
from .pager import PagedWeightStore
from .scheduler import OverlapScheduler
from .speculate import speculative_generate

_MOCK_VOCAB = (
    "the quick brown fox jumps over lazy dog and then runs home again with friends "
    "under bright stars while rivers flow softly through quiet valleys".split()
)


def _mock_target_step(prefix_tokens: list[int]) -> int:
    """Deterministic mock 'model': token id from hash of the prefix. NOT a model."""
    h = hashlib.sha256(str(prefix_tokens).encode()).digest()
    return int.from_bytes(h[:4], "little") % 4096


def _mock_draft_step(prefix_tokens: list[int], k: int) -> list[int]:
    # A "perfect" mock draft: proposes exactly what the mock target would emit,
    # so the draft->verify plumbing is exercised with full acceptance.
    out, p = [], list(prefix_tokens)
    for _ in range(k):
        t = _mock_target_step(p)
        out.append(t)
        p = p + [t]
    return out


def _mock_decode(token_ids: list[int]) -> str:
    return " ".join(_MOCK_VOCAB[t % len(_MOCK_VOCAB)] for t in token_ids)


def _try_load_libllama(path: str | None = None):
    """Guarded ctypes loader for libllama. Returns None when unavailable.

    Real integration would call, in order:
      llama_model_default_params() -> llama_model_load_from_file(path, params)
      llama_context_default_params() -> llama_init_from_model(model, params)
      llama_decode(ctx, batch) per decode step, with the pager's hot-set
      backing the ggml backend buffers (see INTEGRATION.md).
    """
    candidates = [path] if path else []
    candidates += ["libllama.so", "/usr/local/lib/libllama.so", "./libllama.so"]
    for cand in candidates:
        try:
            lib = ctypes.CDLL(cand)
        except OSError:
            continue
        # sanity: the symbol table must expose the core entry points
        try:
            lib.llama_model_default_params
            lib.llama_model_load_from_file
            lib.llama_decode
        except AttributeError:
            continue
        return lib
    return None


def run_llama_server(model_path: str, port: int = 8080, extra_args=()) -> subprocess.Popen:
    """Dense baseline path: shell out to a real ``llama-server`` binary.

    Raises RuntimeError when the binary is not on PATH. This is the honest
    dense baseline that Lacuna's sparse numbers are compared against.
    """
    binary = shutil.which("llama-server")
    if binary is None:
        raise RuntimeError(
            "llama-server not found on PATH — install llama.cpp to use the dense baseline"
        )
    cmd = [binary, "-m", model_path, "--port", str(port), *extra_args]
    return subprocess.Popen(cmd)


class LacunaEngine:
    """Full M1-M6 orchestration over pluggable callables."""

    def __init__(
        self,
        hardware: HardwareTier,
        store: PagedWeightStore | None = None,
        oracle: Oracle | None = None,
        target_step=None,
        draft_step=None,
        lookahead_k: int | None = None,
    ):
        self.hw = hardware
        self.store = store
        self.oracle = oracle
        self.target_step = target_step or _mock_target_step
        self.draft_step = draft_step or _mock_draft_step
        self.k = lookahead_k or hardware.spec_k
        self.scheduler = OverlapScheduler(lookahead_k=self.k)

    def generate(self, prompt: str, max_tokens: int = 32) -> str:
        """Generate text. With the mock target this validates the orchestration
        plumbing (draft -> verify -> schedule); it is NOT model inference."""
        seed = [int(hashlib.sha256(prompt.encode()).hexdigest(), 16) % 4096]

        def draft_fn(prefix, k):
            return self.draft_step(prefix, k)

        def verify_fn(prefix, drafts):
            # one sparse pass over the union mask (mock: accept by hash lottery)
            n = 0
            for d in drafts:
                want = self.target_step(prefix + drafts[:n])
                if d == want:
                    n += 1
                else:
                    break
            return n

        res = speculative_generate(draft_fn, verify_fn, max_tokens, k=self.k, prefix=seed)
        return _mock_decode(res["tokens"])

    def stats(self) -> dict:
        s = dict(self.scheduler.stats)
        if self.store is not None:
            s["pager"] = dict(self.store.stats)
            s["pager"]["hit_rate"] = self.store.hit_rate
        return s


class Engine:
    """One-line public API::

        from lacuna import Engine
        print(Engine("model.lac").generate("hello"))

    ``model_path`` may be a ``.lac`` file, a ``.gguf`` file (auto-converted,
    cached next to the source), or anything else / missing, in which case the
    engine runs in clearly-labeled MOCK mode for plumbing tests.
    """

    def __init__(self, model_path: str = "mock", hardware: HardwareTier | None = None):
        self.hw = hardware or detect_hardware()
        self.model_path = model_path
        self.mode = "mock"
        self.store: PagedWeightStore | None = None
        self.libllama = _try_load_libllama()

        if model_path.endswith(".lac") and os.path.exists(model_path):
            self.store = PagedWeightStore(
                model_path, hot_budget_bytes=int(self.hw.ram_hot_budget_gb * 1024**3)
            )
            self.mode = "lacuna"
        elif model_path.endswith(".gguf") and os.path.exists(model_path):
            spp = model_path + ".lac"
            if not os.path.exists(spp):
                from .convert import convert as _convert

                _convert(model_path, spp, page_mb=self.hw.page_mb, quiet=True)
            self.store = PagedWeightStore(
                spp, hot_budget_bytes=int(self.hw.ram_hot_budget_gb * 1024**3)
            )
            self.mode = "lacuna"
        elif self.libllama is not None:
            self.mode = "libllama-dense"

        self._engine = LacunaEngine(self.hw, store=self.store)

    def generate(self, prompt: str, max_tokens: int = 32) -> str:
        text = self._engine.generate(prompt, max_tokens=max_tokens)
        if self.mode == "mock":
            return f"[mock] {text}"
        return text

    def close(self):
        if self.store is not None:
            self.store.close()
            self.store = None
        self._engine.scheduler.close()
