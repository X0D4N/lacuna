"""LacunaEngine — orchestration of oracle + pager + scheduler + speculate.

Exactness guarantee ("prediction guides memory, never math")
------------------------------------------------------------
In exact mode (default) the oracle is a PREFETCH SCHEDULER ONLY — it never
skips computation. This is the structural difference from PowerInfer-style
pruning, where the predictor decides which neurons participate in the math
(and quality degrades when it is wrong). Here:

1. Per decode step the oracle's union mask only decides the *order* pages
   stream from NVMe (async prefetch).
2. Before the compute callable runs, the engine calls
   ``store.ensure_resident(full_required_page_set)`` — a blocking fault of
   every page the step may touch. Compute therefore never observes a missing
   page, no matter how bad the prediction was.
3. Forced faults are counted as ``exact_faults``; ``exact_guarantee_held`` is
   True only if every step measured full residency via ``are_resident``.

Consequence: oracle quality affects speed only, never output. ``--extreme``
opts out (adaptive precision + no residency guarantee) for maximum tok/s.

Honest scaling note: the reference engine ensures the FULL page set per
decode step (a dense step's matmuls may touch any weight). The guarantee
therefore needs ``hot_budget_bytes`` >= the model's paged size; when the
budget is smaller, ``exact_guarantee_held`` measurably reads False instead
of pretending. A production deployment tiles per layer (or per expert for
MoE): the scheduler's ``required_pages_for_step`` is exactly that hook —
pass one layer's pages and the guarantee holds per tile. See
integrations/llama.cpp/INTEGRATION.md.

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
  2. **llama-server dense path** (``run_llama_server`` +
     ``LlamaServerBackend``): IMPLEMENTED. When a ``llama-server`` binary
     is available, ``Engine("model.gguf")`` serves the real model through
     it (greedy decoding) and reports mode ``"libllama-dense"``. This is
     dense inference inside the llama.cpp process — it does NOT route
     through Lacuna's pager — and it is labeled as such everywhere.
  3. **GPU paged path** (``lacuna.gpu_executor.GPUModel``, mode
     ``"lacuna-gpu"``): IMPLEMENTED 2026-10-25. Full-model Qwen2
     generation through the pager on a torch-CUDA or numpy backend;
     request it with ``Engine(path, gpu="cuda"|"auto"|"numpy")``. See
     ``docs/GPU_EXECUTOR_DESIGN.md`` for the backend decision, the
     floor arithmetic, and the Kaggle GPU test results. Gemma-4
     files dispatch to ``lacuna.gpu_executor.GPUGemmaModel`` in the
     same mode (hybrid sliding/global attention, V = K, full-dim
     global RoPE with stored rope_freqs, GELU-tanh MLP); see
     ``docs/GEMMA4_31B.md``.

See integrations/llama.cpp/INTEGRATION.md for the ggml backend hook design.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request

from . import quant as _q
from .hardware import HardwareTier, detect_hardware
from .oracle import Oracle
from .paged_model import PagedModelError, PagedQwen2
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
            for sym in ("llama_model_default_params", "llama_model_load_from_file", "llama_decode"):
                getattr(lib, sym)
        except AttributeError:
            continue
        return lib
    return None


def run_llama_server(
    model_path: str,
    port: int = 8080,
    extra_args=(),
    capture_output: bool = False,
) -> subprocess.Popen:
    """Dense baseline path: shell out to a real ``llama-server`` binary.

    Raises RuntimeError when the binary is not on PATH. This is the honest
    dense baseline that Lacuna's sparse numbers are compared against.
    ``capture_output=True`` silences the server's own console logging
    (used by LlamaServerBackend, where the server is an implementation
    detail and its log spam would bury the caller's output).
    """
    binary = shutil.which("llama-server")
    if binary is None:
        raise RuntimeError(
            "llama-server not found on PATH — install llama.cpp to use the dense baseline"
        )
    cmd = [binary, "-m", model_path, "--port", str(port), *extra_args]
    if capture_output:
        return subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    return subprocess.Popen(cmd)


def _find_llama_server(explicit: str | None = None) -> str | None:
    """Locate a ``llama-server`` binary, or return None.

    Search order: the explicit path, ``$LACUNA_LLAMA_SERVER``, then PATH.
    A candidate only counts if it is an executable file, so a stale env var
    degrades to "not found" instead of a confusing spawn failure later.
    """
    candidates = [explicit, os.environ.get("LACUNA_LLAMA_SERVER")]
    candidates.append(shutil.which("llama-server"))
    for cand in candidates:
        if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LlamaServerBackend:
    """Real-inference backend: drives a local llama.cpp ``llama-server``.

    Satisfies the :class:`~lacuna.ollama_shim.InferenceBackend` protocol
    (``generate`` + ``mode``). The server is started lazily-never: it is
    spawned by the constructor via :func:`run_llama_server` and the
    constructor blocks until the server answers ``/health`` (or raises).

    HONEST SCOPE: this is llama.cpp *dense* inference over HTTP. It does
    NOT route through Lacuna's pager — the weights live inside the
    llama-server process — so ``mode`` is ``"libllama-dense"``, never
    ``"lacuna"``. It exists so Lacuna's public API, CLI and servers can
    produce real model output today, through the documented engine-swap
    seam, while the paged executor is built.

    Decoding is pinned greedy (temperature 0, top_k 1) so output is
    reproducible and directly comparable to ``llama-cli --temp 0
    --top-k 1`` on the same prompt string.
    """

    def __init__(
        self,
        model_path: str,
        server_binary: str | None = None,
        port: int | None = None,
        startup_timeout: float = 180.0,
    ):
        binary = _find_llama_server(server_binary)
        if binary is None:
            raise RuntimeError(
                "llama-server not found — build llama.cpp and put llama-server "
                "on PATH or set LACUNA_LLAMA_SERVER to the binary"
            )
        self.model_path = model_path
        self.server_binary = binary
        self.port = port if port is not None else _free_port()
        self.last_timings: dict | None = None
        self._closed = False
        # run_llama_server() resolves the binary via PATH; make sure the
        # directory of the binary we actually found is on it.
        bindir = os.path.dirname(os.path.abspath(binary))
        path_entries = os.environ.get("PATH", "").split(os.pathsep)
        if bindir not in path_entries:
            os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")
        self._proc = run_llama_server(
            model_path, port=self.port, capture_output=True
        )
        try:
            self._wait_ready(startup_timeout)
        except Exception:
            self.close()
            raise

    # -- lifecycle -------------------------------------------------
    def _wait_ready(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        url = f"http://127.0.0.1:{self.port}/health"
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited early (code {self._proc.returncode}) "
                    f"while loading {self.model_path}"
                )
            try:
                with urllib.request.urlopen(url, timeout=2) as resp:
                    if resp.status == 200:
                        return
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(0.25)
        raise RuntimeError(
            f"llama-server did not become ready within {timeout:.0f}s"
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        proc = getattr(self, "_proc", None)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    # -- InferenceBackend protocol ----------------------------------
    @property
    def mode(self) -> str:
        return "libllama-dense"

    def generate(self, prompt: str, max_tokens: int = 32) -> str:
        """Greedy-generate from the real model served by llama-server.

        The prompt is sent verbatim to the server's ``/completion``
        endpoint (no chat template is applied by this backend — the
        string the caller passes is the string the model sees).
        """
        if self._closed:
            raise RuntimeError("LlamaServerBackend is closed")
        payload = {
            "prompt": prompt,
            "n_predict": int(max_tokens),
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "seed": 42,
            "cache_prompt": False,
            "stream": False,
        }
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/completion",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            body = json.loads(resp.read().decode())
        self.last_timings = body.get("timings")
        return body["content"]


class LacunaEngine:
    """Full M1-M6 orchestration over pluggable callables.

    ``exact=True`` (default): the exactness guarantee described in the module
    docstring. ``pages_for_step_fn(step_index) -> iterable of page ids`` is the
    oracle's prefetch schedule (may be None -> prefetch the full set, i.e. the
    dense schedule). The full required page set is always ALL pages of the
    store: a dense decode step's matmuls may touch any weight.
    """

    def __init__(
        self,
        hardware: HardwareTier,
        store: PagedWeightStore | None = None,
        oracle: Oracle | None = None,
        target_step=None,
        draft_step=None,
        lookahead_k: int | None = None,
        exact: bool = True,
        pages_for_step_fn=None,
    ):
        self.hw = hardware
        self.store = store
        self.oracle = oracle
        self.target_step = target_step or _mock_target_step
        self.draft_step = draft_step or _mock_draft_step
        self.k = lookahead_k or hardware.spec_k
        self.exact = bool(exact)
        self.pages_for_step_fn = pages_for_step_fn
        self.scheduler = OverlapScheduler(lookahead_k=self.k)
        self.exact_faults = 0
        self._guarantee_held = True
        self._all_pages: list[int] | None = None

    def _full_page_set(self) -> list[int]:
        if self._all_pages is None:
            self._all_pages = list(range(self.store.n_pages))
        return self._all_pages

    def generate(self, prompt: str, max_tokens: int = 32) -> str:
        """Generate text. With the mock target this validates the orchestration
        plumbing (draft -> verify -> schedule); it is NOT model inference.

        In exact mode with a store, every verify pass runs the guarantee
        protocol: prefetch the oracle's predicted set, then
        ``ensure_resident`` the full set BEFORE any target_step call.
        """
        seed = [int(hashlib.sha256(prompt.encode()).hexdigest(), 16) % 4096]
        full_set = self._full_page_set() if (self.exact and self.store) else None
        pass_idx = 0

        def draft_fn(prefix, k):
            return self.draft_step(prefix, k)

        def verify_fn(prefix, drafts):
            nonlocal pass_idx
            if full_set is not None:
                # 1. oracle = prefetch scheduler only: warm the predicted set.
                predicted = (
                    list(self.pages_for_step_fn(pass_idx))
                    if self.pages_for_step_fn is not None
                    else full_set
                )
                if predicted:
                    self.store.prefetch(predicted)
                # 2. guarantee: full required set resident BEFORE compute.
                #    The oracle never gates what compute may read.
                self.exact_faults += self.store.ensure_resident(full_set)
                # 3. measure the guarantee (not assumed): every page resident?
                if not self.store.are_resident(full_set):
                    self._guarantee_held = False
                pass_idx += 1
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

    @property
    def exact_guarantee_held(self) -> bool:
        """True when every exact-mode compute step measured full residency.

        Vacuously True when exact mode had no store to page (mock-only runs):
        with no pager there is nothing that could be missing.
        """
        return self._guarantee_held

    def stats(self) -> dict:
        s = dict(self.scheduler.stats)
        s["exact"] = self.exact
        s["exact_faults"] = self.exact_faults
        s["exact_guarantee_held"] = self.exact_guarantee_held
        if self.store is not None:
            s["pager"] = dict(self.store.stats)
            s["pager"]["hit_rate"] = self.store.hit_rate
            s["pager"]["precision_policy"] = self.store.precision_policy
        return s


class Engine:
    """One-line public API::

        from lacuna import Engine
        print(Engine("model.lac").generate("hello"))

    ``model_path`` may be a ``.lac`` file, a ``.gguf`` file, or anything
    else / missing, in which case the engine runs in clearly-labeled MOCK
    mode for plumbing tests. A ``.gguf`` file generates REAL model output
    when a ``llama-server`` binary is available (on PATH, via
    ``$LACUNA_LLAMA_SERVER``, or via the ``llama_server=`` argument):
    generation is then llama.cpp dense inference through
    :class:`LlamaServerBackend` and ``mode`` reads ``"libllama-dense"``.
    Without a binary, a supported architecture (Qwen2, see
    :mod:`lacuna.paged_model`, or Gemma-4, see :mod:`lacuna.gemma`) runs
    through the real paged executor and
    ``mode`` reads ``"lacuna"`` — genuine model output with every weight
    byte faulted through the pager. If the paged executor cannot load
    the model (unsupported architecture, missing tensors, bad file), the
    engine falls back to the paged orchestration path: it exercises the
    pager/oracle/scheduler over the real pages but does NOT execute the
    model's transformer math, so its text is a simulation of generation —
    always labeled ``[simulated]``, never presented as the model's own
    output. The fallback reason is kept inspectable on ``paged_error``.

    With ``gpu="cuda"``, ``"auto"``, or ``"numpy"``, a real model file
    instead runs through :class:`lacuna.gpu_executor.GPUModel` and
    ``mode`` reads ``"lacuna-gpu"``: full-model generation through the
    pager on a torch-CUDA backend (``"cuda"`` demands CUDA and raises
    instead of falling back), the same backend when present
    (``"auto"``), or the numpy backend (``"numpy"``, the CPU-executable
    GPU-mode math used by tests). ``vram_budget_bytes`` caps device
    memory (default 4 GiB); a breach raises instead of spilling. The
    last generation's end-response timings are on ``gpu_stats``
    (time-to-first-token, total seconds, tok/s, and whether the
    32-token response stayed within the 300 s bound).

    ``exact=True`` (default): the exactness guarantee — output identical to
    the dense baseline by construction; the oracle only schedules prefetches.
    ``precision_policy``: "exact" (default; bit-identical source bytes),
    "extreme" (adaptive Q4/Q2/ternary; fastest, documented tradeoff), or None
    to derive from ``exact`` ("exact" if exact else "extreme").
    ``warm_start=True`` (default): restore the persisted hot-set on open and
    save it on :meth:`close`, so repeat runs start with the working set
    already resident (see :mod:`lacuna.warmstart`; validated per page, a
    stale cache degrades to a cold start, never to wrong data).
    """

    def __init__(
        self,
        model_path: str = "mock",
        hardware: HardwareTier | None = None,
        exact: bool = True,
        precision_policy: str | None = None,
        warm_start: bool = True,
        llama_server: str | None = None,
        gpu: str | None = None,
        vram_budget_bytes: int | None = None,
    ):
        self.hw = hardware or detect_hardware()
        self.model_path = model_path
        self.exact = bool(exact)
        self.warm_start = bool(warm_start)
        self.precision_policy = precision_policy or (
            _q.POLICY_EXACT if self.exact else _q.POLICY_EXTREME
        )
        _q.validate_policy(self.precision_policy)
        self.mode = "mock"
        self.store: PagedWeightStore | None = None
        self._backend: LlamaServerBackend | None = None
        self._paged: PagedQwen2 | None = None
        self.paged_error: Exception | None = None
        self.gpu_mode = gpu
        self._gpu = None
        self.gpu_error: Exception | None = None
        self.gpu_stats: dict | None = None
        self.gpu_backend: str | None = None
        self._real_file = False  # a real model file was loaded from disk
        self.libllama = _try_load_libllama()

        # Try the real paged executor first on real model files; on any
        # load failure (unsupported arch, missing tensors, bad file) fall
        # back to the labeled simulated orchestration path below. The
        # failure is kept on paged_error so a silent downgrade is visible
        # to callers and tests instead of hidden.
        def _try_paged(gguf_path: str) -> bool:
            try:
                self._paged = PagedQwen2(
                    gguf_path,
                    hot_budget_bytes=int(self.hw.ram_hot_budget_gb * 1024**3),
                )
            except (PagedModelError, ValueError, OSError, KeyError) as qwen_err:
                # Not a Qwen2 file: try the Gemma-4 paged executor before
                # giving up to the simulated path. The reported error is
                # the one for the file's actual architecture when known.
                try:
                    from .gemma import GemmaModelError, PagedGemma4

                    self._paged = PagedGemma4(
                        gguf_path,
                        hot_budget_bytes=int(
                            self.hw.ram_hot_budget_gb * 1024**3
                        ),
                    )
                except (GemmaModelError, ValueError, OSError, KeyError) as gemma_err:
                    try:
                        from .gguf import parse_gguf as _parse

                        _arch, _, _ = _parse(gguf_path)
                        self.paged_error = (
                            gemma_err if _arch.get("general.architecture") == "gemma4"
                            else qwen_err
                        )
                    except Exception:
                        self.paged_error = qwen_err
                    return False
            self.mode = "lacuna"
            return True

        def _try_gpu(gguf_path: str) -> bool:
            """GPU paged path (mode "lacuna-gpu"). On failure the reason
            is kept on gpu_error and the caller falls through to the
            existing chain — except when gpu == "cuda", which demands
            CUDA and raises instead of silently degrading."""
            from .gpu_executor import (
                GPUExecutorError,
                GPUModel,
                NumpyBackend,
                TorchBackend,
                detect_backend,
            )

            try:
                if gpu == "cuda":
                    backend = TorchBackend("cuda")
                elif gpu == "numpy":
                    backend = NumpyBackend()
                else:  # "auto" or any other value
                    backend, _info = detect_backend("auto")
                # Gemma-4 files go to the Gemma GPU model, not the
                # Qwen2/MoE GPUModel (which raises on gemma4). The
                # header peek is cheap; on peek failure the call
                # below raises its own load error as before.
                try:
                    from .gguf import parse_gguf as _parse_gguf

                    _md, _, _ = _parse_gguf(gguf_path)
                    _arch = _md.get("general.architecture")
                except Exception:
                    _arch = None
                if _arch == "gemma4":
                    from .gpu_executor import GPUGemmaModel

                    self._gpu = GPUGemmaModel.from_gguf(
                        gguf_path,
                        backend=backend,
                        hot_budget_bytes=int(
                            self.hw.ram_hot_budget_gb * 1024**3
                        ),
                        vram_budget_bytes=int(vram_budget_bytes)
                        if vram_budget_bytes
                        else 4 * 1024**3,
                    )
                else:
                    self._gpu = GPUModel.from_gguf(
                        gguf_path,
                        backend=backend,
                        hot_budget_bytes=int(
                            self.hw.ram_hot_budget_gb * 1024**3
                        ),
                        vram_budget_bytes=int(vram_budget_bytes)
                        if vram_budget_bytes
                        else 4 * 1024**3,
                    )
            except (
                GPUExecutorError,
                PagedModelError,
                ValueError,
                OSError,
                KeyError,
            ) as exc:
                self.gpu_error = exc
                return False
            self.mode = "lacuna-gpu"
            self.gpu_backend = (
                f"{self._gpu.backend.name}:{self._gpu.backend.device}"
            )
            return True

        if model_path.endswith(".lac") and os.path.exists(model_path):
            self._real_file = True
            # Two .lac naming conventions exist: pull/convert use
            # splitext (model.gguf -> model.lac), while PagedQwen2's
            # auto-convert appends (model.gguf -> model.gguf.lac).
            # The sibling GGUF is the stripped stem when it already
            # ends in .gguf, else the stem + .gguf. The old code used
            # the bare stem unconditionally, so for pull-style .lac
            # files _try_paged opened a nonexistent path, failed with
            # FileNotFoundError, and Engine silently degraded to the
            # [simulated] mock path even though mode read "lacuna"
            # (CLI smoke on a pulled model printed mock gibberish).
            # Prefer an existing candidate so a bare stem that does
            # exist keeps its old behavior.
            stem = model_path[: -len(".lac")]
            candidates = (
                [stem]
                if stem.endswith(".gguf")
                else [stem + ".gguf", stem]
            )
            sibling = next(
                (c for c in candidates if os.path.exists(c)), candidates[0]
            )
            gpu_ok = (
                _try_gpu(sibling)
                if (gpu is not None and os.path.exists(sibling))
                else False
            )
            if gpu_ok:
                pass
            elif gpu == "cuda":
                raise RuntimeError(
                    f"GPU mode demanded but unavailable: {self.gpu_error}"
                )
            elif not _try_paged(sibling):
                self.store = PagedWeightStore(
                    model_path, hot_budget_bytes=int(self.hw.ram_hot_budget_gb * 1024**3)
                )
                self.mode = "lacuna"
        elif model_path.endswith(".gguf") and os.path.exists(model_path):
            self._real_file = True
            gpu_ok = _try_gpu(model_path) if gpu is not None else False
            binary = _find_llama_server(llama_server)
            if gpu_ok:
                pass
            elif gpu == "cuda":
                raise RuntimeError(
                    f"GPU mode demanded but unavailable: {self.gpu_error}"
                )
            elif binary is not None:
                # Real inference: the actual model, served by llama.cpp.
                self._backend = LlamaServerBackend(
                    model_path, server_binary=binary
                )
                self.mode = self._backend.mode
            elif not _try_paged(model_path):
                spp = model_path + ".lac"
                if not os.path.exists(spp):
                    from .convert import convert as _convert

                    _convert(
                        model_path,
                        spp,
                        page_mb=self.hw.page_mb,
                        quiet=True,
                        precision_policy=self.precision_policy,
                    )
                self.store = PagedWeightStore(
                    spp, hot_budget_bytes=int(self.hw.ram_hot_budget_gb * 1024**3)
                )
                self.mode = "lacuna"
        elif self.libllama is not None:
            self.mode = "libllama-dense"

        if self.store is not None and self.warm_start:
            from .warmstart import default_hotset_path

            self.store.restore_hotset(default_hotset_path(self.store.path))

        self._engine = LacunaEngine(self.hw, store=self.store, exact=self.exact)

    def generate(self, prompt: str, max_tokens: int = 32) -> str:
        if self._backend is not None:
            return self._backend.generate(prompt, max_tokens=max_tokens)
        if self._gpu is not None:
            text, _ids, stats = self._gpu.generate(
                prompt, max_tokens=max_tokens
            )
            self.gpu_stats = stats
            return text
        if self._paged is not None:
            text, _ = self._paged.generate(prompt, max_tokens=max_tokens)
            return text
        text = self._engine.generate(prompt, max_tokens=max_tokens)
        if self.mode == "mock":
            return f"[mock] {text}"
        if self._real_file:
            # A real model file is loaded, but generation is going through
            # the paged orchestration path, which does not execute the
            # model's transformer math. Label it — silently returning
            # simulated text as if it were the model's answer was a real
            # defect (caught 2026-10-23 by an independent verification run
            # on a real GGUF in a default environment).
            return f"[simulated] {text}"
        return text

    def close(self):
        if self._backend is not None:
            self._backend.close()
            self._backend = None
        if self._gpu is not None:
            self._gpu.close()
            self._gpu = None
        if self._paged is not None:
            self._paged.close()
            self._paged = None
        if self.store is not None:
            if self.warm_start:
                from .warmstart import default_hotset_path

                self.store.save_hotset(default_hotset_path(self.store.path))
            self.store.close()
            self.store = None
        self._engine.scheduler.close()
