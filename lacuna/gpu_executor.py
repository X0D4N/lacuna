"""GPU executor core: pager-driven weight streaming + on-device compute.

This is the speed closer for beating Colibri-class engines at big-model
scale. The CPU paged path measured ~36 min/token on dense 110B because it
is dequant-bound on the CPU: every token re-dequantizes every tensor in
numpy. The GPU path keeps the same pager discipline (every weight byte
arrives through ``PagedWeightStore``) but moves the two expensive steps
onto the device:

1. **Upload raw quantized bytes** (not dequantized float32) per weight
   matrix, via :func:`read_raw_tensor_bytes`.
2. **Dequantize on the device** and run the GEMMs there, then **free the
   matrix** before the next one. At no point is a whole layer's float32
   weights resident, which is what makes a 4 GB floor GPU (RTX 2050)
   viable for 100B-class models: the largest single 110B matrix is the
   49152x8192 MLP matrix at 1.61 GB float32, while a whole layer in
   float32 would be ~5.5 GB. (Gate and up are streamed as separate
   GEMMs, never fused on device: a fused gate+up matrix would be
   3.22 GB and leave almost no headroom under a 4 GB budget.)

Scope of this file (deliberate, 2026-10-25): a single Qwen2 layer
forward plus the streaming/budget machinery around it. Full-model
generation and ``Engine`` wiring come later, via the parent — this file
is the testable core, not the finished product.

Backend decision (evidence, not assumption — see
``docs/GPU_EXECUTOR_DESIGN.md`` for the full table):
- Primary: **PyTorch CUDA**. It is the only GPU tensor library with a
  realistic chance of already being present on a Kaggle GPU kernel, and
  it lets the layer math be validated (on CPU) before any GPU run.
- Fallback: **numpy** (``NumpyBackend``). Same layer math, same code
  path, CPU execution. This is what runs on this box (no GPU, no CUDA
  torch) and what every test here exercises.
- NOT implemented: CuPy. Writing a backend that cannot be executed
  anywhere available would be fake breadth; CuPy stays a documented
  fallback candidate, not code.

Exactness, stated honestly (see the design doc): the layer math is the
same operations in float32 as the CPU path, in the same order, but a
GPU GEMM (cuBLAS) need not reduce in the same order as a CPU BLAS.
Bit-identical output is therefore NOT promised. What is promised is
the same greedy choice except at near-ties — the exact class of
divergence the CPU paged-vs-dense comparison already shows (a 0.0022
probability margin flipping one token). The Kaggle test measures token
parity and reports any flip with its margin; it does not assume it.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

# GGML type ids for this llama.cpp checkout (see lacuna/gguf.py).
GGML_F32 = 0
GGML_F16 = 1
GGML_Q5_0 = 6
GGML_Q8_0 = 8
GGML_Q2_K = 10
GGML_Q3_K = 11
GGML_Q4_K = 12
GGML_Q5_K = 13
GGML_Q6_K = 14

# Types with a native on-device (torch) dequant kernel in this file.
# Everything else dequantizes on CPU via lacuna.gguf and uploads float32
# — exact, but the CPU-dequant cost remains for those types. That is a
# documented v1 gap (Q2_K/Q3_K/Q5_K/Q6_K torch kernels), not a silent
# approximation: no type is ever run through a lossier path than the
# CPU executor uses.
GPU_DEQUANT_TYPES = frozenset({GGML_F32, GGML_F16, GGML_Q8_0, GGML_Q4_K})

# Bytes per stored value, from gguf._TYPE_INFO (block_bytes / block_size).
_BYTES_PER_VALUE = {
    GGML_F32: 4.0,
    GGML_F16: 2.0,
    GGML_Q5_0: 22 / 32,
    GGML_Q8_0: 34 / 32,
    GGML_Q2_K: 84 / 256,
    GGML_Q3_K: 110 / 256,
    GGML_Q4_K: 144 / 256,
    GGML_Q5_K: 176 / 256,
    GGML_Q6_K: 210 / 256,
}


class GPUExecutorError(ValueError):
    """Raised for unsupported configs, missing weights, or budget breach."""


class GPUUnavailableError(GPUExecutorError):
    """Raised when a CUDA backend is demanded but no GPU is present."""


class GPUMemoryError(GPUExecutorError):
    """Raised when weights would exceed the VRAM budget. Fails loud —
    the executor never silently spills, downcasts, or drops weights."""


# ---------------------------------------------------------------- backend


@dataclass
class LayerConfig:
    """Qwen2 layer geometry. Mirrors the fields PagedQwen2 reads from the
    GGUF header, so a config can be built from parsed metadata."""

    hidden: int
    ffn: int
    n_head: int
    n_kv: int
    eps: float = 1e-6
    rope_theta: float = 1000000.0

    @property
    def head_dim(self) -> int:
        if self.hidden % self.n_head:
            raise GPUExecutorError("hidden size not divisible by head count")
        return self.hidden // self.n_head

    @property
    def group(self) -> int:
        if self.n_head % self.n_kv:
            raise GPUExecutorError("head count not divisible by kv head count")
        return self.n_head // self.n_kv


class Backend:
    """Minimal array interface the single-layer forward is written
    against. Numpy and torch both satisfy it, so one layer implementation
    serves both — the backend is a kernel choice, not a second math path.

    Arrays are opaque: numpy ndarrays for NumpyBackend, torch tensors
    for TorchBackend. Elementwise ops (``+``, ``*``, ``/``, ``@``) work
    identically on both, so only the operations whose APIs differ get
    named methods here.
    """

    name = "base"
    is_gpu = False
    device = "cpu"

    def asarray(self, arr: np.ndarray) -> object:
        raise NotImplementedError

    def to_numpy(self, arr: object) -> np.ndarray:
        raise NotImplementedError

    def zeros(self, shape) -> object:
        raise NotImplementedError

    def arange(self, n: int) -> object:
        """Float32 arange (RoPE angles)."""
        raise NotImplementedError

    def arange_int(self, n: int) -> object:
        """Integer arange (causal-mask positions; kept integer so mask
        comparisons stay exact at any sequence length)."""
        raise NotImplementedError

    def permute(self, arr: object, dims) -> object:
        """Transpose to axis order ``dims`` (numpy .transpose /
        torch .permute, same result)."""
        raise NotImplementedError

    def where(self, cond: object, x: object, y: object) -> object:
        raise NotImplementedError

    def cos(self, arr: object) -> object:
        raise NotImplementedError

    def sin(self, arr: object) -> object:
        raise NotImplementedError

    def exp(self, arr: object) -> object:
        raise NotImplementedError

    def sqrt(self, arr: object) -> object:
        raise NotImplementedError

    def cat(self, arrs, axis: int = 0) -> object:
        raise NotImplementedError

    def split(self, arr: object, sizes, axis: int = -1) -> list:
        raise NotImplementedError

    def repeat_kv(self, arr: object, group: int) -> object:
        raise NotImplementedError

    def reduce_max(self, arr: object, axis: int) -> object:
        raise NotImplementedError

    def reduce_sum(self, arr: object, axis: int) -> object:
        raise NotImplementedError

    def mean(self, arr: object, axis: int) -> object:
        raise NotImplementedError

    def dequantize(self, raw: bytes, ttype: int, n: int) -> object:
        """Dequantize ``n`` values of GGML type ``ttype`` to a flat
        float32 device array. Must equal the CPU executor's result up
        to float32 rounding (elementwise ops) — never a re-quantization."""
        raise NotImplementedError

    def memory_allocated(self) -> int:
        """Device bytes currently allocated (0 when not meaningful)."""
        return 0

    def synchronize(self) -> None:
        """Block until queued device work finishes (timing fence)."""

    def empty_cache(self) -> None:
        """Release cached-but-unused device memory (layer boundary)."""


class NumpyBackend(Backend):
    """CPU fallback. Dequantization delegates to lacuna.gguf, the same
    bit-exact code the CPU paged executor uses."""

    name = "numpy"
    is_gpu = False
    device = "cpu"

    def asarray(self, arr: np.ndarray) -> np.ndarray:
        return np.array(arr, dtype=np.float32, copy=True)

    def to_numpy(self, arr: np.ndarray) -> np.ndarray:
        return np.array(arr, dtype=np.float32, copy=True)

    def zeros(self, shape) -> np.ndarray:
        return np.zeros(shape, dtype=np.float32)

    def arange(self, n: int) -> np.ndarray:
        return np.arange(n, dtype=np.float32)

    def arange_int(self, n: int) -> np.ndarray:
        return np.arange(n, dtype=np.int64)

    def permute(self, arr: np.ndarray, dims) -> np.ndarray:
        return arr.transpose(*dims)

    def where(self, cond: np.ndarray, x: np.ndarray,
              y: np.ndarray) -> np.ndarray:
        return np.where(cond, x, y)

    def cos(self, arr: np.ndarray) -> np.ndarray:
        return np.cos(arr)

    def sin(self, arr: np.ndarray) -> np.ndarray:
        return np.sin(arr)

    def exp(self, arr: np.ndarray) -> np.ndarray:
        return np.exp(arr)

    def sqrt(self, arr: np.ndarray) -> np.ndarray:
        return np.sqrt(arr)

    def cat(self, arrs, axis: int = 0) -> np.ndarray:
        return np.concatenate(list(arrs), axis=axis)

    def split(self, arr: np.ndarray, sizes, axis: int = -1) -> list:
        idx = np.cumsum(list(sizes))[:-1]
        return list(np.split(arr, idx, axis=axis))

    def repeat_kv(self, arr: np.ndarray, group: int) -> np.ndarray:
        return np.repeat(arr, group, axis=1)

    def reduce_max(self, arr: np.ndarray, axis: int) -> np.ndarray:
        return np.max(arr, axis=axis, keepdims=True)

    def reduce_sum(self, arr: np.ndarray, axis: int) -> np.ndarray:
        return np.sum(arr, axis=axis, keepdims=True)

    def mean(self, arr: np.ndarray, axis: int) -> np.ndarray:
        return np.mean(arr, axis=axis, keepdims=True)

    def dequantize(self, raw: bytes, ttype: int, n: int) -> np.ndarray:
        from .gguf import dequantize as _dq

        return np.array(_dq(raw, ttype, n), dtype=np.float32, copy=True)


class TorchBackend(Backend):
    """PyTorch backend. ``device`` is ``"cuda"`` when CUDA is available,
    otherwise ``"cpu"`` — the CPU mode exists so the torch kernels'
    math can be validated on machines without a GPU. It is not a speed
    claim: torch-CPU is expected to be slower than the numpy path for
    these shapes, and nothing in this file benchmarks it as a win.

    Torch dequant kernels mirror lacuna/gguf.py exactly (same block
    layout, same float32 arithmetic order) for F32/F16/Q8_0/Q4_K. Other
    types fall back to CPU dequant + upload: exact, but not yet fast.
    """

    name = "torch"

    def __init__(self, device: str | None = None):
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - import guard
            raise GPUUnavailableError("torch is not installed") from exc
        self._torch = torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise GPUUnavailableError(
                "CUDA requested but torch.cuda.is_available() is False "
                "(no GPU, no CUDA torch build, or missing libcuda.so — "
                "see docs/GPU_EXECUTOR_DESIGN.md for the Kaggle libcuda fix)"
            )
        self.device = device
        self.is_gpu = device.startswith("cuda")

    # -- generic ops -------------------------------------------------
    def asarray(self, arr: np.ndarray) -> object:
        t = self._torch
        return t.from_numpy(np.array(arr, dtype=np.float32, copy=True)).to(
            self.device
        )

    def to_numpy(self, arr: object) -> np.ndarray:
        return np.array(arr.detach().cpu().numpy(), dtype=np.float32, copy=True)

    def zeros(self, shape) -> object:
        return self._torch.zeros(shape, dtype=self._torch.float32,
                                 device=self.device)

    def arange(self, n: int) -> object:
        return self._torch.arange(n, dtype=self._torch.float32,
                                  device=self.device)

    def arange_int(self, n: int) -> object:
        return self._torch.arange(n, dtype=self._torch.int64,
                                  device=self.device)

    def permute(self, arr: object, dims) -> object:
        return arr.permute(*dims)

    def where(self, cond: object, x: object, y: object) -> object:
        return self._torch.where(cond, x, y)

    def cos(self, arr: object) -> object:
        return self._torch.cos(arr)

    def sin(self, arr: object) -> object:
        return self._torch.sin(arr)

    def exp(self, arr: object) -> object:
        return self._torch.exp(arr)

    def sqrt(self, arr: object) -> object:
        return self._torch.sqrt(arr)

    def cat(self, arrs, axis: int = 0) -> object:
        return self._torch.cat(list(arrs), dim=axis)

    def split(self, arr: object, sizes, axis: int = -1) -> list:
        return list(self._torch.split(arr, list(sizes), dim=axis))

    def repeat_kv(self, arr: object, group: int) -> object:
        return arr.repeat_interleave(group, dim=1)

    def reduce_max(self, arr: object, axis: int) -> object:
        return arr.max(dim=axis, keepdim=True).values

    def reduce_sum(self, arr: object, axis: int) -> object:
        return arr.sum(dim=axis, keepdim=True)

    def mean(self, arr: object, axis: int) -> object:
        return arr.mean(dim=axis, keepdim=True)

    # -- dequant ------------------------------------------------------
    def _f16(self, two_byte_slice: object) -> object:
        """Float16 scale column from a (nb, 2) uint8 slice, as flat
        float32 (nb,). The slice is flattened first so the dtype view
        is unambiguous: (nb*2,) uint8 -> (nb,) float16 -> (nb,) f32."""
        t = self._torch
        flat = two_byte_slice.reshape(-1).contiguous()
        return flat.view(t.float16).float()

    @staticmethod
    def _q4_k_scales_min(scales: object, torch) -> tuple:
        """Mirror of gguf._q4_k_scales_min in torch ops. ``scales`` is
        (nb, 12) uint8; returns (sc, mn) float32 (nb, 8)."""
        nb = scales.shape[0]
        sc = torch.empty((nb, 8), dtype=torch.float32, device=scales.device)
        mn = torch.empty((nb, 8), dtype=torch.float32, device=scales.device)
        for j in range(8):
            if j < 4:
                sc[:, j] = (scales[:, j] & 63).float()
                mn[:, j] = (scales[:, j + 4] & 63).float()
            else:
                sc[:, j] = ((scales[:, j + 4] & 0x0F)
                            | ((scales[:, j - 4] >> 6) << 4)).float()
                mn[:, j] = ((scales[:, j + 4] >> 4)
                            | ((scales[:, j] >> 6) << 4)).float()
        return sc, mn

    def dequantize(self, raw: bytes, ttype: int, n: int) -> object:
        t = self._torch
        dev = self.device
        if ttype == GGML_F32:
            arr = np.frombuffer(raw, dtype="<f4", count=n).astype(np.float32)
            return t.from_numpy(np.array(arr, copy=True)).to(dev)
        if ttype == GGML_F16:
            try:
                u8 = t.frombuffer(bytearray(raw), dtype=t.uint8).to(dev)
                return u8.view(t.float16).float().reshape(-1)
            except Exception:
                # Fall back to the numpy path (exact, CPU conversion).
                from .gguf import dequantize as _dq

                return t.from_numpy(
                    np.array(_dq(raw, ttype, n), copy=True)).to(dev)
        if ttype == GGML_Q8_0:
            if n % 32:
                raise GPUExecutorError("Q8_0 count not a multiple of 32")
            nb = n // 32
            blocks = t.frombuffer(bytearray(raw), dtype=t.uint8).to(dev)
            blocks = blocks.reshape(nb, 34)
            d = self._f16(blocks[:, 0:2])
            qs = blocks[:, 2:34].contiguous().view(t.int8).float()
            return (qs * d[:, None]).reshape(-1)
        if ttype == GGML_Q4_K:
            if n % 256:
                raise GPUExecutorError("Q4_K count not a multiple of 256")
            nb = n // 256
            blocks = t.frombuffer(bytearray(raw), dtype=t.uint8).to(dev)
            blocks = blocks.reshape(nb, 144)
            d = self._f16(blocks[:, 0:2])
            dmin = self._f16(blocks[:, 2:4])
            sc, mn = self._q4_k_scales_min(blocks[:, 4:16], t)
            qs = blocks[:, 16:144]
            out = t.empty((nb, 256), dtype=t.float32, device=dev)
            for g in range(4):
                d1 = d * sc[:, 2 * g]
                m1 = dmin * mn[:, 2 * g]
                d2 = d * sc[:, 2 * g + 1]
                m2 = dmin * mn[:, 2 * g + 1]
                qb = qs[:, g * 32:(g + 1) * 32]
                lo = (qb & 0x0F).float()
                hi = (qb >> 4).float()
                out[:, g * 64:g * 64 + 32] = d1[:, None] * lo - m1[:, None]
                out[:, g * 64 + 32:g * 64 + 64] = d2[:, None] * hi - m2[:, None]
            return out.reshape(-1)
        # CPU fallback for Q5_0/Q2_K/Q3_K/Q5_K/Q6_K: exact, not yet fast.
        from .gguf import dequantize as _dq

        return t.from_numpy(np.array(_dq(raw, ttype, n), copy=True)).to(dev)

    # -- device management ---------------------------------------------
    def memory_allocated(self) -> int:
        if self.is_gpu:
            return int(self._torch.cuda.memory_allocated(self.device))
        return 0

    def synchronize(self) -> None:
        if self.is_gpu:
            self._torch.cuda.synchronize(self.device)

    def empty_cache(self) -> None:
        if self.is_gpu:
            self._torch.cuda.empty_cache()


# ------------------------------------------------------- environment


def gpu_available() -> bool:
    """True only if torch imports AND reports a CUDA device. Never
    crashes on a torch-less box: absence is a normal result."""
    try:
        import torch
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def describe_environment() -> dict:
    """Inspectable environment report: what is present, what is missing,
    and why. Used by the Kaggle test script's output header."""
    info: dict = {
        "has_gpu": False,
        "torch_version": None,
        "torch_cuda": False,
        "cuda_device_name": None,
        "cuda_device_count": 0,
        "cupy_version": None,
        "numpy_version": np.__version__,
        "nvidia_smi": False,
    }
    try:
        import torch

        info["torch_version"] = torch.__version__
        try:
            info["torch_cuda"] = bool(torch.cuda.is_available())
            info["cuda_device_count"] = int(torch.cuda.device_count())
            if info["torch_cuda"]:
                info["has_gpu"] = True
                info["cuda_device_name"] = torch.cuda.get_device_name(0)
        except Exception:
            pass
    except ImportError:
        pass
    try:
        import cupy

        info["cupy_version"] = cupy.__version__
    except ImportError:
        pass
    import shutil

    info["nvidia_smi"] = shutil.which("nvidia-smi") is not None
    return info


def detect_backend(prefer: str = "auto") -> tuple[Backend, dict]:
    """Pick the backend for this machine.

    ``prefer="auto"`` (default): CUDA torch if present, else numpy.
    ``prefer="torch-cpu"``: torch on CPU if torch imports (math
    validation on GPU-less machines), else numpy. Any other value
    behaves like ``"auto"``. Returns (backend, info dict) where info
    carries the backend name, device, and the reason for the choice —
    so a log line can show why a run was CPU, instead of silently
    being slow.
    """
    env = describe_environment()
    if env["torch_cuda"]:
        backend: Backend = TorchBackend("cuda")
        reason = "torch.cuda.is_available()"
    elif prefer == "torch-cpu" and env["torch_version"] is not None:
        backend = TorchBackend("cpu")
        reason = "prefer=torch-cpu, torch present without CUDA"
    else:
        backend = NumpyBackend()
        reason = "no CUDA torch; numpy fallback"
    info = dict(env)
    info["backend"] = backend.name
    info["backend_device"] = backend.device
    info["backend_reason"] = reason
    return backend, info


# ------------------------------------------------------- weight refs


@dataclass
class WeightRef:
    """One weight matrix, either as raw GGUF bytes (to be dequantized
    on the device) or as an already-float numpy array (synthetic tests
    and already-converted weights). Exactly one of ``raw``/``array``
    must be set."""

    shape: tuple
    raw: bytes | None = None
    ttype: int | None = None
    array: np.ndarray | None = None

    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= int(d)
        return n

    def quant_bytes(self) -> int:
        """Stored bytes on disk / in the pager for this matrix."""
        if self.raw is not None:
            return len(self.raw)
        if self.ttype is not None and self.ttype in _BYTES_PER_VALUE:
            return int(math.ceil(self.numel() * _BYTES_PER_VALUE[self.ttype]))
        return self.numel() * 4

    def f32_bytes(self) -> int:
        return self.numel() * 4

    def materialize(self, backend: Backend) -> object:
        """Produce the float32 device array. Raw bytes dequantize on
        the device when the backend has a kernel, else on CPU."""
        if self.array is not None:
            arr = np.array(self.array, dtype=np.float32, copy=True)
            if arr.shape != tuple(self.shape):
                raise GPUExecutorError(
                    f"weight array shape {arr.shape} != declared {self.shape}")
            return backend.asarray(arr)
        if self.raw is None or self.ttype is None:
            raise GPUExecutorError("WeightRef needs raw+ttype or array")
        flat = backend.dequantize(self.raw, int(self.ttype), self.numel())
        return flat.reshape(tuple(self.shape))


def read_raw_tensor_bytes(store, info) -> bytes:
    """Raw GGUF bytes for one tensor, through the pager only.

    This is the GPU path's answer to ``gguf.read_tensor``: same pager
    discipline (exact policy, every byte via ``store.get_page``, length
    checked against the declared size), but WITHOUT dequantization —
    the bytes go to the device and dequantize there. Nothing here reads
    the .lac file directly.
    """
    from .gguf import tensor_nbytes

    if getattr(store, "precision_policy", None) != "exact":
        raise GPUExecutorError(
            f"paged executor needs an exact-policy store, got "
            f"{getattr(store, 'precision_policy', None)!r}"
        )
    tid = store.tensor_id(info["name"])
    if tid is None:
        raise GPUExecutorError(f"tensor {info['name']} not present in store")
    pages = store.tensor_pages(tid)
    blob = b"".join(store.get_page(e["page_id"]).tobytes() for e in pages)
    want = tensor_nbytes(info)
    if len(blob) != want:
        raise GPUExecutorError(
            f"tensor {info['name']}: paged bytes {len(blob)} != "
            f"declared {want}"
        )
    return blob


# ------------------------------------------------------- layer forward


class GPULayer:
    """One Qwen2 decoder layer on a backend. The math mirrors
    ``PagedQwen2.forward``'s per-layer block exactly (RMSNorm, fused
    QKV, RoPE half-rotation, GQA repeat, causal attention, SwiGLU),
    so GPU-vs-CPU layer outputs are comparable element-for-element.

    Weight keys (``attn_*``/``ffn_*`` under one layer): attn_norm,
    attn_q/k/v + optional attn_q/k/v biases, attn_output, ffn_norm,
    ffn_gate, ffn_up, ffn_down. Missing biases default to zeros (some
    files lack them); missing weights raise.
    """

    def __init__(self, backend: Backend, config: LayerConfig):
        self.b = backend
        self.cfg = config
        head_dim = config.head_dim
        self._inv_freq = backend.asarray(
            np.array(config.rope_theta
                     ** (-np.arange(0, head_dim, 2, dtype=np.float32)
                         / head_dim), dtype=np.float32)
        )

    # -- primitives (backend-generic) --------------------------------
    def _rmsnorm(self, x: object, w: object) -> object:
        b = self.b
        ms = b.mean(x * x, axis=-1)
        return (x / b.sqrt(ms + np.float32(self.cfg.eps))) * w

    def _rope(self, t: object, positions: object) -> object:
        """RoPE, half-rotation form — the empirically verified form from
        paged_model (first half of each head against the second half),
        not the pair-rotation form that produced the wrong token.
        ``positions`` is a (T,) float32 backend array."""
        b = self.b
        ang = positions.reshape(-1, 1) * self._inv_freq.reshape(1, -1)
        cos = b.cos(ang)[:, None, :]
        sin = b.sin(ang)[:, None, :]
        h = t.shape[-1] // 2
        x0 = t[..., :h]
        x1 = t[..., h:]
        first = x0 * cos - x1 * sin
        second = x0 * sin + x1 * cos
        return b.cat([first, second], axis=-1)

    @staticmethod
    def _silu(b: Backend, x: object) -> object:
        return x / (np.float32(1.0) + b.exp(-x))

    # -- forward -------------------------------------------------------
    # The layer is split into phases so weights stream per MATRIX, not
    # per layer: attention matrices are materialized, used, and freed
    # before any MLP matrix is materialized, and gate/up/down are each
    # materialized, used, and freed in turn. Peak live weights are
    # therefore one MLP matrix (1.61 GB float32 at 110B scale), not a
    # whole layer (5.43 GB) — which is what fits a 4 GB floor budget.
    # Fusing gate+up (as the CPU path does) would break this: the fused
    # matrix alone is 3.22 GB, so the GPU path deliberately does not
    # fuse them.
    _ATTN_KEYS = ("attn_norm.weight", "attn_q.weight", "attn_q.bias",
                  "attn_k.weight", "attn_k.bias", "attn_v.weight",
                  "attn_v.bias", "attn_output.weight")

    def _attention(self, x: object, mats: dict, positions: object,
                   cache: dict | None, pos_offset: int) -> object:
        """Attention half of the layer. ``mats`` holds the attn_* device
        arrays; ``cache`` (optional dict) holds per-layer (k_all, v_all)
        device arrays across decode steps and is updated in place.
        Returns x after the attention residual, on device.

        ``pos_offset`` MUST equal the first element of ``positions``
        (the absolute position of the first query token): the causal
        mask is built from it, while RoPE is built from ``positions``.
        If they disagree, the mask silently attends to the wrong keys
        (a single-key mask makes the output position-independent — this
        exact inconsistency was caught by a test). ``forward`` derives
        ``pos_offset`` from ``positions`` so callers cannot disagree."""
        b = self.b
        cfg = self.cfg
        n_tok = x.shape[0]

        def w(key: str) -> object:
            if key not in mats:
                raise GPUExecutorError(f"layer weight {key!r} missing")
            return mats[key]

        def bias(key: str, n: int) -> object:
            if key in mats:
                return mats[key]
            return b.zeros((n,))

        h = self._rmsnorm(x, w("attn_norm.weight"))
        q = h @ w("attn_q.weight").T + bias("attn_q.bias", cfg.hidden)
        k = h @ w("attn_k.weight").T + bias("attn_k.bias",
                                            cfg.n_kv * cfg.head_dim)
        v = h @ w("attn_v.weight").T + bias("attn_v.bias",
                                            cfg.n_kv * cfg.head_dim)
        q = self._rope(q.reshape(n_tok, cfg.n_head, cfg.head_dim), positions)
        k = self._rope(k.reshape(n_tok, cfg.n_kv, cfg.head_dim), positions)
        v = v.reshape(n_tok, cfg.n_kv, cfg.head_dim)
        k_rep = b.repeat_kv(k, cfg.group)
        v_rep = b.repeat_kv(v, cfg.group)
        if cache is not None and "k" in cache and "v" in cache:
            k_all = b.cat([cache["k"], k_rep], axis=0)
            v_all = b.cat([cache["v"], v_rep], axis=0)
        else:
            k_all, v_all = k_rep, v_rep
        if cache is not None:
            cache["k"], cache["v"] = k_all, v_all
        # Causal attention, softmax in float32 (manual, backend-generic).
        # Layouts: q (T,H,D) -> (H,T,D); k_all (S,H,D) -> (H,D,S);
        # scores (H,T,S); att (H,T,D) -> (T,H,D) -> (T, hidden).
        q_b = b.permute(q, (1, 0, 2))
        k_b = b.permute(k_all, (1, 2, 0))
        scores = (q_b @ k_b) / np.float32(math.sqrt(cfg.head_dim))
        n_keys = k_all.shape[0]
        # Causal mask, integer positions: key j attends iff j <=
        # pos_offset + i for query i. Shapes (1,1,S) and (1,T,1)
        # broadcast over the head axis.
        col = b.arange_int(n_keys).reshape(1, 1, -1)
        row = (b.arange_int(n_tok) + int(pos_offset)).reshape(1, -1, 1)
        scores = b.where(col <= row, scores,
                         b.zeros(scores.shape) + np.float32(-1e30))
        scores = scores - b.reduce_max(scores, axis=-1)
        probs = b.exp(scores)
        probs = probs / b.reduce_sum(probs, axis=-1)
        v_b = b.permute(v_all, (1, 0, 2))
        att = b.permute(probs @ v_b, (1, 0, 2)).reshape(n_tok, cfg.hidden)
        return x + att @ w("attn_output.weight").T

    def forward(self, x_np: np.ndarray, refs: dict[str, WeightRef],
                positions_np: np.ndarray, cache: dict | None = None,
                pos_offset: int = 0) -> np.ndarray:
        """Run one layer with per-matrix streaming: each phase
        materializes only its own matrices, uses them, and drops them
        before the next phase materializes anything. The KV cache is
        the exception (it is working memory, bounded by sequence
        length, not weights). Returns numpy.

        The mask offset is derived from ``positions_np[0]`` (single
        source of truth); an explicit ``pos_offset`` argument is
        accepted for compatibility but never trusted over positions."""
        b = self.b
        x = b.asarray(np.array(x_np, dtype=np.float32, copy=True))
        pos_arr = np.array(positions_np, dtype=np.float32, copy=True)
        positions = b.asarray(pos_arr)
        if pos_arr.size:
            pos_offset = int(pos_arr.reshape(-1)[0])

        def mat(key: str) -> object:
            ref = refs.get(key)
            if ref is None:
                raise GPUExecutorError(f"layer weight {key!r} missing")
            return ref.materialize(b)

        # Phase 1: attention. Only attn_* matrices are live here.
        mats = {k: refs[k].materialize(b) for k in self._ATTN_KEYS
                if k in refs}
        for required in ("attn_norm.weight", "attn_q.weight",
                         "attn_k.weight", "attn_v.weight",
                         "attn_output.weight"):
            if required not in mats:
                raise GPUExecutorError(
                    f"layer weight {required!r} missing")
        try:
            x = self._attention(x, mats, positions, cache, pos_offset)
        finally:
            del mats
            b.empty_cache()
        # Phase 2: MLP, one matrix at a time. Gate's activation is kept
        # (it is activations, T x ffn, not weights); gate itself is
        # freed before up is materialized, and up before down.
        ffn_norm = mat("ffn_norm.weight")
        h2 = self._rmsnorm(x, ffn_norm)
        del ffn_norm
        gate = mat("ffn_gate.weight")
        gate_act = h2 @ gate.T
        del gate
        up = mat("ffn_up.weight")
        act = self._silu(b, gate_act) * (h2 @ up.T)
        del up, gate_act, h2
        down = mat("ffn_down.weight")
        try:
            out = x + act @ down.T
            return b.to_numpy(out)
        finally:
            del down, act
            b.empty_cache()


# ------------------------------------------------------- executor


class GPUExecutor:
    """Streaming single-layer executor with a hard VRAM budget.

    ``run_layer`` streams one layer's matrices in phases (raw bytes ->
    device float32, used, freed — see ``GPULayer.forward``): peak live
    weights are one phase's matrices, never the whole layer. The KV
    cache survives across calls (it is the model's working memory, not
    weights). If a phase's peak + cache + activations would exceed
    ``vram_budget_bytes``, it raises :class:`GPUMemoryError` before
    computing — never silently downcasts or spills.
    """

    def __init__(self, backend: Backend, config: LayerConfig,
                 vram_budget_bytes: int = 4 * 1024**3):
        self.backend = backend
        self.config = config
        self.layer = GPULayer(backend, config)
        self.vram_budget = int(vram_budget_bytes)
        self.cache: dict = {}

    def reset_cache(self) -> None:
        self.cache = {}

    def _cache_bytes(self) -> int:
        total = 0
        for arr in self.cache.values():
            # torch tensors have .numel(); numpy arrays have .size.
            n = arr.numel() if hasattr(arr, "numel") else arr.size
            total += int(n) * 4
        return total

    def check_budget(self, refs: dict[str, WeightRef],
                     n_tok: int) -> dict:
        """Byte accounting for one layer call, per streaming phase.
        Raises GPUMemoryError on breach; returns the breakdown
        otherwise (inspectable, like the pager's stats). The enforced
        figure is the PEAK live set — the larger of the attention
        phase's matrices and the MLP phase's single largest matrix
        (gate/up/down stream one at a time) — plus cache and an
        activations bound. The whole layer's float32 total is reported
        for information but is NOT the enforced figure: enforcing it
        would reject 110B-class layers on a 4 GB budget that phased
        streaming actually fits."""
        mats_bytes = sum(r.f32_bytes() for r in refs.values())
        largest = max((r.f32_bytes() for r in refs.values()), default=0)
        attn_bytes = sum(r.f32_bytes() for k, r in refs.items()
                         if k in GPULayer._ATTN_KEYS)
        mlp_peak = 0
        for key in ("ffn_norm.weight", "ffn_gate.weight",
                    "ffn_up.weight", "ffn_down.weight"):
            if key in refs:
                mlp_peak = max(mlp_peak, refs[key].f32_bytes())
        act_bytes = n_tok * self.config.hidden * 4 * 8  # activations, bound
        cache_bytes = self._cache_bytes()
        peak = max(attn_bytes, mlp_peak) + act_bytes + cache_bytes
        breakdown = {
            "matrices_f32_bytes": mats_bytes,
            "largest_matrix_f32_bytes": largest,
            "attention_phase_bytes": attn_bytes,
            "mlp_phase_peak_bytes": mlp_peak,
            "activations_bound_bytes": act_bytes,
            "kv_cache_bytes": cache_bytes,
            "peak_live_bytes": peak,
            "budget_bytes": self.vram_budget,
        }
        if largest > self.vram_budget:
            raise GPUMemoryError(
                f"single matrix {largest} bytes exceeds VRAM budget "
                f"{self.vram_budget} — no matrix can ever run here")
        if peak > self.vram_budget:
            raise GPUMemoryError(
                f"layer peak {peak} bytes > VRAM budget "
                f"{self.vram_budget} (phase peak "
                f"{max(attn_bytes, mlp_peak)}, cache {cache_bytes}, "
                f"activations bound {act_bytes})")
        return breakdown

    def run_layer(self, x_np: np.ndarray, refs: dict[str, WeightRef],
                  positions_np: np.ndarray,
                  pos_offset: int = 0) -> tuple[np.ndarray, dict]:
        """Run one layer with streaming + budget enforcement. Returns
        (output numpy, stats dict with timing and byte accounting).
        The mask offset is derived from ``positions_np[0]`` (see
        ``GPULayer.forward``); the ``pos_offset`` argument is accepted
        for compatibility and not trusted over positions."""
        n_tok = int(np.asarray(x_np).shape[0])
        budget = self.check_budget(refs, n_tok)
        t0 = time.perf_counter()
        out = self.layer.forward(x_np, refs, positions_np, self.cache,
                                 pos_offset)
        self.backend.synchronize()
        dt = time.perf_counter() - t0
        stats = dict(budget)
        stats["seconds"] = dt
        stats["live_bytes_after"] = self._cache_bytes()
        stats["backend"] = self.backend.name
        stats["backend_device"] = self.backend.device
        return out, stats

    @staticmethod
    def estimate_layer_vram(specs, config: LayerConfig | None = None) -> dict:
        """Floor arithmetic for one layer from (shape, ggml_type) specs:
        stored bytes, float32 bytes if all matrices were resident, the
        largest single matrix (the streaming unit), and KV bytes per
        cached token (when ``config`` is given). Pure arithmetic — no
        device needed.

        Example (110B-class layer, hidden 8192, ffn 49152, 64 Q / 8 KV
        heads, Q4_K weights): stored ~0.75 GB/layer, largest matrix
        1.61 GB float32 (the 49152x8192 MLP matrices), KV ~0.63
        MiB/token across all 80 layers. A 4 GB budget therefore holds
        per-matrix streaming + a multi-thousand-token cache, but never
        a whole layer in float32 (~5.5 GB), and never a fused gate+up
        matrix (3.22 GB)."""
        stored = 0
        f32 = 0
        largest = 0
        for shape, ttype in specs:
            n = 1
            for d in shape:
                n *= int(d)
            stored += int(math.ceil(n * _BYTES_PER_VALUE.get(int(ttype), 4.0)))
            f32 += n * 4
            largest = max(largest, n * 4)
        out = {
            "stored_bytes_per_layer": stored,
            "f32_bytes_all_matrices": f32,
            "largest_matrix_f32_bytes": largest,
        }
        if config is not None:
            # K and V per layer per token: 2 * n_kv * head_dim floats.
            out["kv_bytes_per_token_per_layer"] = (
                2 * config.n_kv * config.head_dim * 4)
        return out
