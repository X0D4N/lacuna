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

Scope of this file: a single Qwen2 layer forward plus the streaming /
budget machinery around it, AND (2026-10-25) full-model generation
(``GPUModel`` below) with ``Engine`` wiring in ``lacuna/engine.py``.
The layer core is unchanged by the wiring: the model class runs one
``GPUExecutor`` per layer so each layer's KV cache stays isolated
(the single shared ``GPUExecutor.cache`` dict is correct only for one
layer and must never be shared across layers).

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
GGML_BF16 = 30

# Types with a native on-device (torch) dequant kernel in this file.
# Everything else dequantizes on CPU via lacuna.gguf and uploads float32
# — exact, but the CPU-dequant cost remains for those types. Q5_0 is
# the remaining CPU-fallback type; Q2_K/Q3_K/Q5_K/Q6_K gained torch
# kernels (2026-10-26, parity-tested against lacuna.gguf). No type is
# ever run through a lossier path than the CPU executor uses.
GPU_DEQUANT_TYPES = frozenset({GGML_F32, GGML_F16, GGML_Q8_0, GGML_Q4_K,
                                GGML_BF16, GGML_Q2_K, GGML_Q3_K,
                                GGML_Q5_K, GGML_Q6_K})

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
    GGML_BF16: 2.0,
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
    GGUF header, so a config can be built from parsed metadata.

    MoE fields (qwen3moe/olmoe only; zero/False for dense): n_experts,
    top_k, and qk_norm (True for qwen3moe — per-layer QK-norm before
    RoPE — False for olmoe). Dense constructions omit them."""

    hidden: int
    ffn: int
    n_head: int
    n_kv: int
    eps: float = 1e-6
    rope_theta: float = 1000000.0
    n_experts: int = 0
    top_k: int = 0
    qk_norm: bool = False

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

    def scatter_add(self, base: object, idx, contrib: object) -> None:
        """In-place row scatter-add: ``base[idx[i]] += contrib[i]`` for
        each row i. Used by the MoE grouped GEMM (one expert's
        weighted contribution per position group). ``idx`` is a numpy
        integer array; implementations convert it to their own index
        type. Rows in ``idx`` must be unique per call (true for one
        expert's group: top-k ids are distinct per position), so the
        numpy path's ``np.add.at`` and torch's ``index_add_`` agree."""
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

    def scatter_add(self, base: np.ndarray, idx, contrib: np.ndarray) -> None:
        np.add.at(base, np.asarray(idx, dtype=int), contrib)

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
    layout, same float32 arithmetic order) for F32/F16/BF16/Q8_0/Q4_K
    and Q2_K/Q3_K/Q5_K/Q6_K. Only Q5_0 still falls back to CPU dequant
    + upload: exact, but not yet fast.
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

    def scatter_add(self, base: object, idx, contrib: object) -> None:
        t = self._torch
        long_idx = t.as_tensor(np.asarray(idx, dtype=np.int64),
                                device=base.device)
        base.index_add_(0, long_idx, contrib)

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
        if ttype == GGML_BF16:
            # BF16 -> float32 is a widening cast (gguf.dequant_bf16);
            # viewing the bytes as bfloat16 and casting is the same
            # operation on device, bit-exact including NaN payloads.
            try:
                u8 = t.frombuffer(bytearray(raw), dtype=t.uint8).to(dev)
                return u8.view(t.bfloat16).float().reshape(-1)
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
        if ttype == GGML_Q2_K:
            if n % 256:
                raise GPUExecutorError("Q2_K count not a multiple of 256")
            nb = n // 256
            blocks = t.frombuffer(bytearray(raw), dtype=t.uint8).to(dev)
            blocks = blocks.reshape(nb, 84)
            scales = blocks[:, 0:16]
            qs = blocks[:, 16:80]
            d = self._f16(blocks[:, 80:82])
            dmin = self._f16(blocks[:, 82:84])
            out = t.empty((nb, 256), dtype=t.float32, device=dev)
            for h in range(2):
                qh = qs[:, h * 32:(h + 1) * 32]
                for j in range(4):
                    shift = 2 * j
                    sc1 = scales[:, h * 8 + 2 * j]
                    sc2 = scales[:, h * 8 + 2 * j + 1]
                    dl1 = d * (sc1 & 0x0F).float()
                    ml1 = dmin * (sc1 >> 4).float()
                    dl2 = d * (sc2 & 0x0F).float()
                    ml2 = dmin * (sc2 >> 4).float()
                    base = h * 128 + j * 32
                    out[:, base:base + 16] = (
                        dl1[:, None]
                        * ((qh[:, 0:16] >> shift) & 3).float()
                        - ml1[:, None]
                    )
                    out[:, base + 16:base + 32] = (
                        dl2[:, None]
                        * ((qh[:, 16:32] >> shift) & 3).float()
                        - ml2[:, None]
                    )
            return out.reshape(-1)
        if ttype == GGML_Q3_K:
            if n % 256:
                raise GPUExecutorError("Q3_K count not a multiple of 256")
            nb = n // 256
            blocks = t.frombuffer(bytearray(raw), dtype=t.uint8).to(dev)
            blocks = blocks.reshape(nb, 110)
            hm = blocks[:, 0:32]
            qs = blocks[:, 32:96]
            raw_sc = blocks[:, 96:108].to(t.int64)
            # Little-endian words a0, a1, tmp from the 12 scale bytes.
            a0 = (raw_sc[:, 0] | (raw_sc[:, 1] << 8)
                  | (raw_sc[:, 2] << 16) | (raw_sc[:, 3] << 24))
            a1 = (raw_sc[:, 4] | (raw_sc[:, 5] << 8)
                  | (raw_sc[:, 6] << 16) | (raw_sc[:, 7] << 24))
            tmp = (raw_sc[:, 8] | (raw_sc[:, 9] << 8)
                   | (raw_sc[:, 10] << 16) | (raw_sc[:, 11] << 24))
            k1 = 0x03030303
            k2 = 0x0F0F0F0F
            w0 = (a0 & k2) | (((tmp >> 0) & k1) << 4)
            w1 = (a1 & k2) | (((tmp >> 2) & k1) << 4)
            w2 = ((a0 >> 4) & k2) | (((tmp >> 4) & k1) << 4)
            w3 = ((a1 >> 4) & k2) | (((tmp >> 6) & k1) << 4)
            words = t.stack([w0, w1, w2, w3], dim=1)  # (nb, 4)
            sc_bytes = t.empty((nb, 16), dtype=t.int64, device=dev)
            for wi in range(4):
                for bi in range(4):
                    sc_bytes[:, wi * 4 + bi] = (words[:, wi] >> (8 * bi)) & 0xFF
            # gguf views the bytes as int8 before subtracting 32.
            scales = t.where(sc_bytes >= 128, sc_bytes - 256,
                             sc_bytes).float()
            d = self._f16(blocks[:, 108:110])
            out = t.empty((nb, 256), dtype=t.float32, device=dev)
            for h in range(2):
                qh = qs[:, h * 32:(h + 1) * 32]
                for j in range(4):
                    bit = h * 4 + j
                    for s in range(2):
                        g = h * 8 + j * 2 + s
                        dl = d * (scales[:, g] - t.tensor(
                            32.0, dtype=t.float32, device=dev))
                        low = ((qh[:, s * 16:(s + 1) * 16] >> (2 * j))
                               & 3).float()
                        high = (hm[:, s * 16:(s + 1) * 16] >> bit) & 1
                        code = low - t.where(
                            high != 0,
                            t.tensor(0.0, dtype=t.float32, device=dev),
                            t.tensor(4.0, dtype=t.float32, device=dev))
                        base = h * 128 + j * 32 + s * 16
                        out[:, base:base + 16] = dl[:, None] * code
            return out.reshape(-1)
        if ttype == GGML_Q5_K:
            if n % 256:
                raise GPUExecutorError("Q5_K count not a multiple of 256")
            nb = n // 256
            blocks = t.frombuffer(bytearray(raw), dtype=t.uint8).to(dev)
            blocks = blocks.reshape(nb, 176)
            d = self._f16(blocks[:, 0:2])
            dmin = self._f16(blocks[:, 2:4])
            sc, mn = self._q4_k_scales_min(blocks[:, 4:16], t)
            qh = blocks[:, 16:48]
            qs = blocks[:, 48:176]
            out = t.empty((nb, 256), dtype=t.float32, device=dev)
            for g in range(4):
                d1 = d * sc[:, 2 * g]
                m1 = dmin * mn[:, 2 * g]
                d2 = d * sc[:, 2 * g + 1]
                m2 = dmin * mn[:, 2 * g + 1]
                ql = qs[:, g * 32:(g + 1) * 32]
                c1 = (ql & 0x0F).float() + t.where(
                    ((qh >> (2 * g)) & 1) != 0,
                    t.tensor(16.0, dtype=t.float32, device=dev),
                    t.tensor(0.0, dtype=t.float32, device=dev))
                c2 = (ql >> 4).float() + t.where(
                    ((qh >> (2 * g + 1)) & 1) != 0,
                    t.tensor(16.0, dtype=t.float32, device=dev),
                    t.tensor(0.0, dtype=t.float32, device=dev))
                base = g * 64
                out[:, base:base + 32] = d1[:, None] * c1 - m1[:, None]
                out[:, base + 32:base + 64] = d2[:, None] * c2 - m2[:, None]
            return out.reshape(-1)
        if ttype == GGML_Q6_K:
            if n % 256:
                raise GPUExecutorError("Q6_K count not a multiple of 256")
            nb = n // 256
            blocks = t.frombuffer(bytearray(raw), dtype=t.uint8).to(dev)
            blocks = blocks.reshape(nb, 210)
            d = self._f16(blocks[:, 208:210])
            sc_raw = blocks[:, 192:208].float()
            sc_all = t.where(sc_raw >= 128, sc_raw - 256, sc_raw)
            ql_all = blocks[:, 0:128]
            qh_all = blocks[:, 128:192]
            out = t.empty((nb, 256), dtype=t.float32, device=dev)
            for h in range(2):
                ql = ql_all[:, h * 64:(h + 1) * 64]
                qh = qh_all[:, h * 32:(h + 1) * 32]
                sc = sc_all[:, h * 8:(h + 1) * 8]
                for l in range(32):
                    is_ = l // 16
                    q1 = (((ql[:, l] & 0x0F).to(t.int16)
                           | ((qh[:, l] & 3).to(t.int16) << 4))
                          - 32).float()
                    q2 = (((ql[:, l + 32] & 0x0F).to(t.int16)
                           | (((qh[:, l] >> 2) & 3).to(t.int16) << 4))
                          - 32).float()
                    q3 = (((ql[:, l] >> 4).to(t.int16)
                           | (((qh[:, l] >> 4) & 3).to(t.int16) << 4))
                          - 32).float()
                    q4 = (((ql[:, l + 32] >> 4).to(t.int16)
                           | (((qh[:, l] >> 6) & 3).to(t.int16) << 4))
                          - 32).float()
                    base = h * 128
                    out[:, base + l] = d * sc[:, is_] * q1
                    out[:, base + 32 + l] = d * sc[:, is_ + 2] * q2
                    out[:, base + 64 + l] = d * sc[:, is_ + 4] * q3
                    out[:, base + 96 + l] = d * sc[:, is_ + 6] * q4
            return out.reshape(-1)
        # CPU fallback for Q5_0 only: exact, not yet fast.
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


def read_raw_stacked_rows_bytes(store, info, r0: int, r1: int) -> bytes:
    """Raw GGUF bytes for rows ``[r0, r1)`` of a stacked 3-D expert
    tensor, through the pager only.

    This is the MoE expert-paging primitive: a stacked tensor's dims
    fastest-first are ``[row_len, out_len, n_experts]``, so the
    flattened 2-D view ``(n_experts * out_len, row_len)`` gives
    expert ``e`` exactly rows ``[e * out_len, (e+1) * out_len)`` —
    the same view ``PagedMoE._expert_flat_info`` uses. Only the
    pages overlapping the requested byte range are faulted (the page
    walk mirrors ``gguf.read_tensor_rows``: ``page_shape`` reads
    headers only, so boundaries are known without faulting payloads).
    The bytes dequantize on the device like any other raw tensor;
    nothing here dequantizes or reads the .lac file directly.
    """
    from .gguf import _TYPE_INFO

    if getattr(store, "precision_policy", None) != "exact":
        raise GPUExecutorError(
            f"paged executor needs an exact-policy store, got "
            f"{getattr(store, 'precision_policy', None)!r}"
        )
    dims = [int(d) for d in info["dims"]]
    if len(dims) != 3:
        raise GPUExecutorError(
            f"tensor {info['name']}: stacked rows need 3-D dims, "
            f"got {dims}"
        )
    if info["type"] not in _TYPE_INFO:
        raise GPUExecutorError(
            f"tensor {info['name']}: unsupported GGML type {info['type']}"
        )
    row_len, out_len, n_experts = dims
    total_rows = out_len * n_experts
    if not (0 <= r0 <= r1 <= total_rows):
        raise GPUExecutorError(
            f"tensor {info['name']}: row range [{r0}, {r1}) outside "
            f"[0, {total_rows})"
        )
    per_block, bytes_per_block, _ = _TYPE_INFO[info["type"]]
    if row_len % per_block:
        raise GPUExecutorError(
            f"tensor {info['name']}: row length {row_len} not a "
            f"multiple of block size {per_block}; rows cannot be "
            f"dequantized alone"
        )
    row_bytes = (row_len // per_block) * bytes_per_block
    start, end = r0 * row_bytes, r1 * row_bytes
    tid = store.tensor_id(info["name"])
    if tid is None:
        raise GPUExecutorError(f"tensor {info['name']} not present in store")
    pieces: list[bytes] = []
    off = 0
    for e in store.tensor_pages(tid):
        _, plen = store.page_shape(e["page_id"])  # RAW: (0, payload bytes)
        p_start, p_end = off, off + plen
        if p_end > start and p_start < end:
            arr = store.get_page(e["page_id"])
            lo = max(start, p_start) - p_start
            hi = min(end, p_end) - p_start
            pieces.append(arr[lo:hi].tobytes())
        off = p_end
        if off >= end:
            break
    blob = b"".join(pieces)
    if len(blob) != end - start:
        raise GPUExecutorError(
            f"tensor {info['name']}: paged row bytes {len(blob)} != "
            f"wanted {end - start}"
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
                   cache: dict | None, pos_offset: int,
                   q_norm: object | None = None,
                   k_norm: object | None = None) -> object:
        """Attention half of the layer. ``mats`` holds the attn_* device
        arrays; ``cache`` (optional dict) holds per-layer (k_all, v_all)
        device arrays across decode steps and is updated in place.
        Returns x after the attention residual, on device.

        ``q_norm``/``k_norm`` (optional, qwen3moe only) are the QK-norm
        RMSNorm weights, shape (head_dim,): applied to Q (per Q head)
        and K (per KV head) BEFORE RoPE, with the model's epsilon —
        exactly as ``PagedMoE`` does. Without them attention scores
        are wrong, not near-tie noise. Dense layers pass neither.

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
        q = q.reshape(n_tok, cfg.n_head, cfg.head_dim)
        k = k.reshape(n_tok, cfg.n_kv, cfg.head_dim)
        if q_norm is not None:
            q = self._rmsnorm(q, q_norm)
        if k_norm is not None:
            k = self._rmsnorm(k, k_norm)
        q = self._rope(q, positions)
        k = self._rope(k, positions)
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

    # -- MoE forward ---------------------------------------------------
    # Same phase discipline as the dense forward, but the MLP phase
    # is routed: the router runs first (dense, resident), then only
    # the selected experts' matrices are materialized — one expert,
    # one role at a time — and freed before the next. Gate and up
    # stream separately (never fused: a fused per-expert gate+up
    # matrix would double the phase peak, the same reason the dense
    # path does not fuse them). Unselected experts cost zero faults
    # and zero FLOPs: ``expert_ref`` is only ever called for experts
    # the router selected for at least one position.
    _MOE_ATTN_KEYS = _ATTN_KEYS + ("attn_q_norm.weight",
                                   "attn_k_norm.weight")

    def forward_moe(self, x_np: np.ndarray, refs: dict[str, WeightRef],
                    expert_ref, positions_np: np.ndarray,
                    cache: dict | None = None,
                    pos_offset: int = 0) -> np.ndarray:
        """Run one MoE layer with per-expert streaming.

        ``refs`` holds the dense-per-layer matrices (attn_*, QK-norm
        weights for qwen3moe, ffn_norm, router, optional shared-expert
        weights); ``expert_ref(role, e)`` returns the WeightRef for
        one routed expert matrix (``role`` in {"gate", "up", "down"})
        and is called ONLY for selected experts. ``cache`` is this
        layer's own KV cache (per-layer isolation, same as dense).
        Returns numpy. The mask offset is derived from
        ``positions_np[0]``, as in :meth:`forward`."""
        b = self.b
        cfg = self.cfg
        if cfg.top_k < 1 or cfg.top_k > cfg.n_experts:
            raise GPUExecutorError(
                f"MoE layer needs 1 <= top_k <= n_experts, got "
                f"top_k={cfg.top_k}, n_experts={cfg.n_experts}"
            )
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

        # Phase 1: attention (QK-norm before RoPE when present).
        mats = {k: refs[k].materialize(b) for k in self._ATTN_KEYS
                if k in refs}
        for required in ("attn_norm.weight", "attn_q.weight",
                         "attn_k.weight", "attn_v.weight",
                         "attn_output.weight"):
            if required not in mats:
                raise GPUExecutorError(
                    f"layer weight {required!r} missing")
        q_norm = (refs["attn_q_norm.weight"].materialize(b)
                  if "attn_q_norm.weight" in refs else None)
        k_norm = (refs["attn_k_norm.weight"].materialize(b)
                  if "attn_k_norm.weight" in refs else None)
        try:
            x = self._attention(x, mats, positions, cache, pos_offset,
                                q_norm=q_norm, k_norm=k_norm)
        finally:
            del mats, q_norm, k_norm
            b.empty_cache()
        # Phase 2: router + routed experts, grouped by expert.
        ffn_norm = mat("ffn_norm.weight")
        h2 = self._rmsnorm(x, ffn_norm)
        del ffn_norm
        router = mat("ffn_gate_inp.weight")
        rlog = h2 @ router.T
        if "ffn_gate_inp.bias" in refs:
            rlog = rlog + mat("ffn_gate_inp.bias")
        del router
        # Router rule is PagedMoE's v1 rule, reused — not re-derived:
        # softmax -> top-k -> renormalize, higher score then lower
        # index on ties. (Lazy import: moe.py pulls the paged model;
        # the router itself is a pure function of the logits.)
        from .moe import PagedMoE

        ids, weights = PagedMoE.select_experts(b.to_numpy(rlog),
                                               cfg.top_k)
        del rlog
        out = b.zeros(x.shape)
        try:
            for e in range(cfg.n_experts):
                pos_idx, slot_idx = np.where(ids == e)
                if len(pos_idx) == 0:
                    continue  # unselected: zero faults, zero FLOPs
                h_sub = h2[pos_idx]
                gate = expert_ref("gate", e).materialize(b)
                gate_act = h_sub @ gate.T
                del gate
                up = expert_ref("up", e).materialize(b)
                act = self._silu(b, gate_act) * (h_sub @ up.T)
                del up, gate_act
                down = expert_ref("down", e).materialize(b)
                contrib = act @ down.T
                del down, act
                w = b.asarray(
                    np.array(weights[pos_idx, slot_idx][:, None],
                             dtype=np.float32, copy=True))
                b.scatter_add(out, pos_idx, contrib * w)
                del contrib, w
            # Shared expert (dense term), if present: same one-matrix-
            # at-a-time streaming as the dense MLP phase.
            if "ffn_gate_shexp.weight" in refs:
                gate = mat("ffn_gate_shexp.weight")
                gate_act = h2 @ gate.T
                del gate
                up = mat("ffn_up_shexp.weight")
                act = self._silu(b, gate_act) * (h2 @ up.T)
                del up, gate_act
                down = mat("ffn_down_shexp.weight")
                out = out + act @ down.T
                del down, act
        finally:
            del h2
            b.empty_cache()
        try:
            return b.to_numpy(x + out)
        finally:
            del out
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

    def check_budget_moe(self, refs: dict[str, WeightRef],
                         moe_peak_bytes: int, n_tok: int) -> dict:
        """Byte accounting for one MoE layer call, per streaming
        phase. Enforced peak = the largest single phase — attention
        (attn_* + QK-norm), router, one expert matrix, or one shared-
        expert matrix — plus cache and an activations bound. The
        enforced expert figure is the LARGEST single expert matrix,
        not the sum over experts: experts stream one matrix at a
        time, so the count of selected experts never enters the peak
        (selection is unknown until the router runs, and must not be
        needed for the budget). Raises GPUMemoryError on breach."""
        attn_bytes = sum(r.f32_bytes() for k, r in refs.items()
                         if k in GPULayer._MOE_ATTN_KEYS)
        router_bytes = sum(r.f32_bytes() for k, r in refs.items()
                           if k in ("ffn_gate_inp.weight",
                                    "ffn_gate_inp.bias"))
        act_bytes = n_tok * self.config.hidden * 4 * 8
        cache_bytes = self._cache_bytes()
        phase = max(attn_bytes, router_bytes, int(moe_peak_bytes))
        peak = phase + act_bytes + cache_bytes
        breakdown = {
            "attention_phase_bytes": attn_bytes,
            "router_phase_bytes": router_bytes,
            "moe_phase_peak_bytes": int(moe_peak_bytes),
            "activations_bound_bytes": act_bytes,
            "kv_cache_bytes": cache_bytes,
            "peak_live_bytes": peak,
            "budget_bytes": self.vram_budget,
        }
        if peak > self.vram_budget:
            raise GPUMemoryError(
                f"MoE layer peak {peak} bytes > VRAM budget "
                f"{self.vram_budget} (phase {phase}, cache "
                f"{cache_bytes}, activations bound {act_bytes})")
        return breakdown

    def run_moe_layer(self, x_np: np.ndarray, refs: dict[str, WeightRef],
                      expert_ref, positions_np: np.ndarray,
                      moe_peak_bytes: int,
                      pos_offset: int = 0) -> tuple[np.ndarray, dict]:
        """Run one MoE layer with streaming + budget enforcement.
        ``expert_ref`` and ``moe_peak_bytes`` are as in
        :meth:`GPULayer.forward_moe` and :meth:`check_budget_moe`.
        Returns (output numpy, stats dict)."""
        n_tok = int(np.asarray(x_np).shape[0])
        budget = self.check_budget_moe(refs, moe_peak_bytes, n_tok)
        t0 = time.perf_counter()
        out = self.layer.forward_moe(x_np, refs, expert_ref,
                                     positions_np, self.cache, pos_offset)
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


# ------------------------------------------------------- full model


# All twelve tensors of one Qwen2 decoder layer, in demand order.
_LAYER_KEYS = GPULayer._ATTN_KEYS + (
    "ffn_norm.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
)


class _DirectSource:
    """In-memory weights: synthetic tests and Kaggle synthetic
    generation. Embed/head are plain float32 arrays; layers are dicts
    of WeightRefs. No pager involved — this source exists so the
    generation math is testable without a model file."""

    def __init__(self, embed, head, out_norm, layers):
        self._embed = np.array(embed, dtype=np.float32, copy=True)
        self._head = np.array(head, dtype=np.float32, copy=True)
        self._out_norm = np.array(out_norm, dtype=np.float32, copy=True)
        self._layers = layers

    @property
    def vocab_size(self) -> int:
        return int(self._head.shape[0])

    @property
    def out_norm(self) -> np.ndarray:
        return self._out_norm

    def layer_refs(self, li: int) -> dict:
        return self._layers[li]

    def embed_rows(self, ids) -> np.ndarray:
        if len(ids) == 0:
            return np.zeros((0, self._embed.shape[1]), dtype=np.float32)
        return np.array(self._embed[np.asarray(ids, dtype=int)],
                        dtype=np.float32, copy=True)

    def head_block(self, b0: int, b1: int) -> np.ndarray:
        return np.array(self._head[b0:b1], dtype=np.float32, copy=True)

    def close(self) -> None:
        pass


class _PagedSource:
    """Real-file weights through the pager — the production source.

    Layer weights arrive as raw GGUF bytes per layer per forward call
    (via :func:`read_raw_tensor_bytes`, so they dequantize on the
    device); embed rows and lm_head blocks arrive sliced via
    ``read_tensor_rows`` (never the whole matrix). Nothing here reads
    the .lac file directly, and nothing here caches dequantized
    weights: device memory is the constraint, so every layer's raw
    bytes are transient and freed after the layer runs.
    """

    def __init__(self, store, by_name: dict):
        from .gguf import read_tensor

        self.store = store
        self.by_name = by_name
        self._embed_info = by_name["token_embd.weight"]
        self._head_info = by_name["output.weight"]
        self._out_norm = read_tensor(store, by_name["output_norm.weight"])

    @property
    def vocab_size(self) -> int:
        return int(self._head_info["dims"][1])

    @property
    def out_norm(self) -> np.ndarray:
        return self._out_norm

    def layer_refs(self, li: int) -> dict:
        from .gguf import tensor_shape

        base = f"blk.{li}."
        refs: dict[str, WeightRef] = {}
        for suffix in _LAYER_KEYS:
            info = self.by_name.get(base + suffix)
            if info is None:
                continue  # biases may be absent; the layer raises if
                # a required weight is missing
            refs[suffix] = WeightRef(
                shape=tensor_shape(info["dims"]),
                raw=read_raw_tensor_bytes(self.store, info),
                ttype=int(info["type"]),
            )
        return refs

    def embed_rows(self, ids) -> np.ndarray:
        """Gather embedding rows for ``ids`` without holding the matrix.

        Mirrors ``PagedQwen2._embed_rows`` (unique ids grouped into
        contiguous runs, one ``read_tensor_rows`` call per run) minus
        the dequantized-row cache: rows are tiny, and the GPU path
        must not hold dequantized weights outside the device budget.
        """
        from .gguf import read_tensor_rows

        if len(ids) == 0:
            hidden = int(self._embed_info["dims"][0])
            return np.zeros((0, hidden), dtype=np.float32)
        uniq = sorted(set(int(i) for i in ids))
        rows: dict[int, np.ndarray] = {}
        runs: list[tuple[int, int]] = []
        start = last = uniq[0]
        for u in uniq[1:]:
            if u == last + 1:
                last = u
            else:
                runs.append((start, last + 1))
                start = last = u
        runs.append((start, last + 1))
        for a, b in runs:
            arr = read_tensor_rows(self.store, self._embed_info, a, b)
            for k, u in enumerate(range(a, b)):
                rows[u] = np.array(arr[k], dtype=np.float32, copy=True)
        return np.stack([rows[int(i)] for i in ids]).astype(np.float32)

    def head_block(self, b0: int, b1: int) -> np.ndarray:
        from .gguf import read_tensor_rows

        return read_tensor_rows(self.store, self._head_info, b0, b1)

    def close(self) -> None:
        self.store.close()


class _DirectMoESource:
    """In-memory MoE weights for synthetic tests (no pager).

    ``layers`` is a list of dicts: the dense-per-layer WeightRefs
    (attn_*, QK-norm, ffn_norm, router, optional shared-expert) plus
    an ``"experts"`` dict keyed ``(role, e)`` -> WeightRef, where role
    is "gate"/"up"/"down". Embed/head/out_norm are plain arrays, as
    in :class:`_DirectSource`."""

    def __init__(self, embed, head, out_norm, layers):
        self._embed = np.array(embed, dtype=np.float32, copy=True)
        self._head = np.array(head, dtype=np.float32, copy=True)
        self._out_norm = np.array(out_norm, dtype=np.float32, copy=True)
        self._layers = layers

    @property
    def vocab_size(self) -> int:
        return int(self._head.shape[0])

    @property
    def out_norm(self) -> np.ndarray:
        return self._out_norm

    def layer_refs(self, li: int) -> dict:
        layer = self._layers[li]
        return {k: v for k, v in layer.items() if k != "experts"}

    def expert_ref(self, li: int, role: str, e: int) -> WeightRef:
        try:
            return self._layers[li]["experts"][(role, e)]
        except KeyError as exc:
            raise GPUExecutorError(
                f"layer {li}: expert ({role}, {e}) missing") from exc

    def moe_ffn_peak_bytes(self, li: int) -> int:
        """Largest single MoE-FFN matrix in float32 bytes (one routed
        expert matrix, or one shared-expert matrix): the streaming
        unit the budget enforces."""
        peak = 0
        for ref in self._layers[li]["experts"].values():
            peak = max(peak, ref.f32_bytes())
        for key in ("ffn_gate_shexp.weight", "ffn_up_shexp.weight",
                    "ffn_down_shexp.weight"):
            if key in self._layers[li]:
                peak = max(peak, self._layers[li][key].f32_bytes())
        return peak

    def embed_rows(self, ids) -> np.ndarray:
        if len(ids) == 0:
            return np.zeros((0, self._embed.shape[1]), dtype=np.float32)
        return np.array(self._embed[np.asarray(ids, dtype=int)],
                        dtype=np.float32, copy=True)

    def head_block(self, b0: int, b1: int) -> np.ndarray:
        return np.array(self._head[b0:b1], dtype=np.float32, copy=True)

    def close(self) -> None:
        pass


class _PagedMoESource(_PagedSource):
    """Real-file MoE weights through the pager — the production source.

    Dense-per-layer matrices arrive as raw GGUF bytes per layer per
    forward (as in :class:`_PagedSource`); routed expert matrices
    arrive ONLY for selected experts, as raw byte ranges of the
    stacked tensors via :func:`read_raw_stacked_rows_bytes` — never
    the whole stacked tensor (which would fault every expert and
    defeat paging). Embed/head slicing is inherited unchanged.

    Load-time validation mirrors ``PagedMoE`` where the GPU path can
    check it without running: qwen3moe layers must carry both QK-norm
    tensors with dims [head_dim]; attention-bias presence must be
    uniform across layers and complete across q/k/v (a partial set
    fails loud instead of silently mixing zeros). The router rule
    itself is ``PagedMoE.select_experts`` (v1), reused at forward
    time — DeepSeek sigmoid+bias routing and MLA raise elsewhere
    (unsupported arch), not here.
    """

    # Dense-per-layer MoE keys (attn + QK-norm + ffn_norm + router).
    _MOE_LAYER_SUFFIXES = (
        "attn_norm.weight", "attn_q.weight", "attn_q.bias",
        "attn_k.weight", "attn_k.bias", "attn_v.weight", "attn_v.bias",
        "attn_output.weight", "attn_q_norm.weight", "attn_k_norm.weight",
        "ffn_norm.weight", "ffn_gate_inp.weight", "ffn_gate_inp.bias",
        "ffn_gate_shexp.weight", "ffn_up_shexp.weight",
        "ffn_down_shexp.weight",
    )

    def __init__(self, store, by_name: dict, arch: str, n_layer: int,
                 n_experts: int, ffn: int, hidden: int, head_dim: int):
        super().__init__(store, by_name)
        self.arch = arch
        self.n_layer = int(n_layer)
        self.n_experts = int(n_experts)
        self.ffn = int(ffn)
        self.hidden = int(hidden)
        self.head_dim = int(head_dim)
        # Attention-bias uniformity (mirror of PagedMoE's rule).
        per_layer = []
        for li in range(n_layer):
            base = f"blk.{li}."
            per_layer.append(tuple(
                (base + s) in by_name
                for s in ("attn_q.bias", "attn_k.bias", "attn_v.bias")
            ))
        first = per_layer[0] if per_layer else (False, False, False)
        if any(flags != first for flags in per_layer):
            raise GPUExecutorError(
                "attention bias presence differs across layers "
                f"(layer 0: {first}) — corrupt or mixed file"
            )
        if not (all(first) or not any(first)):
            raise GPUExecutorError(
                f"partial attention bias set on layer 0: q/k/v "
                f"present = {first} — a layer needs all three or none"
            )
        # QK-norm presence (qwen3moe only) + stacked expert dims.
        for li in range(n_layer):
            base = f"blk.{li}."
            if arch == "qwen3moe":
                for role in ("attn_q_norm", "attn_k_norm"):
                    info = by_name.get(base + role + ".weight")
                    if info is None:
                        raise GPUExecutorError(
                            f"layer {li}: qwen3moe requires {base + role}"
                            f".weight (QK-norm), tensor missing"
                        )
                    dims = [int(d) for d in info["dims"]]
                    if len(dims) != 1 or dims[0] != head_dim:
                        raise GPUExecutorError(
                            f"layer {li}: {base + role}.weight has dims "
                            f"{dims}, expected [{head_dim}]"
                        )
            if base + "ffn_gate_inp.weight" not in by_name:
                raise GPUExecutorError(f"layer {li}: router weight missing")
            for role, row_len, out_len in (
                ("gate", hidden, ffn),
                ("up", hidden, ffn),
                ("down", ffn, hidden),
            ):
                name = base + f"ffn_{role}_exps.weight"
                info = by_name.get(name)
                if info is None:
                    raise GPUExecutorError(f"layer {li}: {name} missing")
                dims = [int(d) for d in info["dims"]]
                if (len(dims) != 3
                        or dims != [row_len, out_len, n_experts]):
                    raise GPUExecutorError(
                        f"layer {li}: {name} has dims {dims}, expected "
                        f"[{row_len}, {out_len}, {n_experts}]"
                    )
            # Shared experts (v1: intermediate size must equal the
            # routed ffn size, mirroring PagedMoE's load check).
            for role, row_len, out_len in (
                ("gate_shexp", hidden, ffn),
                ("up_shexp", hidden, ffn),
                ("down_shexp", ffn, hidden),
            ):
                name = base + f"ffn_{role}.weight"
                info = by_name.get(name)
                if info is None:
                    continue
                dims = [int(d) for d in info["dims"]]
                if len(dims) != 2 or dims != [row_len, out_len]:
                    raise GPUExecutorError(
                        f"layer {li}: shared expert {name} has dims "
                        f"{dims}, expected [{row_len}, {out_len}] "
                        f"(v1 requires shared ffn == routed ffn)"
                    )

    def layer_refs(self, li: int) -> dict:
        from .gguf import tensor_shape

        base = f"blk.{li}."
        refs: dict[str, WeightRef] = {}
        for suffix in self._MOE_LAYER_SUFFIXES:
            info = self.by_name.get(base + suffix)
            if info is None:
                continue  # biases / QK-norm (olmoe) may be absent;
                # the layer raises if a required weight is missing
            refs[suffix] = WeightRef(
                shape=tensor_shape(info["dims"]),
                raw=read_raw_tensor_bytes(self.store, info),
                ttype=int(info["type"]),
            )
        return refs

    def expert_ref(self, li: int, role: str, e: int) -> WeightRef:
        """Raw byte range for one routed expert matrix (demand-only).

        Stacked dims fastest-first are [row_len, out_len,
        n_experts]; the flattened view gives expert ``e`` rows
        [e * out_len, (e+1) * out_len). The returned shape is the
        logical matrix: gate/up (ffn, hidden), down (hidden, ffn)."""
        if role not in ("gate", "up", "down"):
            raise GPUExecutorError(f"unknown expert role {role!r}")
        if not 0 <= e < self.n_experts:
            raise GPUExecutorError(
                f"expert {e} outside [0, {self.n_experts})")
        base = f"blk.{li}."
        info = self.by_name[base + f"ffn_{role}_exps.weight"]
        dims = [int(d) for d in info["dims"]]
        row_len, out_len, _ = dims
        if role == "down":
            rows_per_expert = self.hidden
            shape = (self.hidden, self.ffn)
        else:
            rows_per_expert = self.ffn
            shape = (self.ffn, self.hidden)
        raw = read_raw_stacked_rows_bytes(
            self.store, info,
            e * rows_per_expert, (e + 1) * rows_per_expert)
        return WeightRef(shape=shape, raw=raw, ttype=int(info["type"]))

    def moe_ffn_peak_bytes(self, li: int) -> int:
        """Largest single MoE-FFN matrix in float32 bytes. Routed
        gate/up/down and shared-expert matrices all have ffn*hidden
        elements in v1, so the peak is that product — but it is
        computed from the layer's actual refs, not assumed."""
        return self.ffn * self.hidden * 4


class GPUModel:
    """Full Qwen2 generation on a Backend (numpy or torch-CUDA).

    This is the GPU-mode model behind ``Engine(..., gpu=...)``: the
    same decoder math as ``PagedQwen2`` (RMSNorm, GQA, RoPE
    half-rotation, SwiGLU, KV cache, greedy argmax), but every layer
    runs through :class:`GPULayer` on the backend, with one
    ``GPUExecutor`` per layer so KV caches stay per-layer. Embed rows
    and lm_head blocks are sliced (never materialized whole), exactly
    like the CPU paged path.

    MoE mode (``moe=True``, qwen3moe/olmoe): layers run through
    :meth:`GPULayer.forward_moe` — QK-norm before RoPE for qwen3moe,
    ``PagedMoE``'s v1 router rule reused, and routed experts paged
    per expert per layer (never the whole stacked tensor). The CPU
    paged MoE path is the reference: GPU vs CPU must give the same
    greedy choice except at near-ties; a wide-margin flip is a bug,
    not a tolerance.

    Native precision, stated explicitly (this is the priority path):
    F16 tensors flow as native bytes — ``read_raw_tensor_bytes``
    hands the file's F16 bytes to the backend, and the torch kernel
    views them as float16 and casts to float32 on the device; the
    numpy backend uses ``gguf.dequant_f16`` (also a cast). There is
    NO quantization step anywhere on this path: the bytes are the
    file's native bytes, and "dequantization" for F16 is a cast, not
    a re-quantization. F32 flows the same way (direct copy). K-quant
    types Q2_K/Q3_K/Q5_K/Q6_K use native torch dequant kernels;
    only Q5_0 still dequantizes on CPU via ``gguf`` and uploads as
    float32 — exact, but slow, and NOT the priority: on the headline
    path, native F16 avoids them entirely. BF16
    (GGML type 30, the Qwen3-235B native file) flows the same native
    way: ``gguf.dequant_bf16`` (a widening shift, bit-exact vs ggml)
    on the numpy backend, and a bfloat16 view + cast on the torch
    backend. Still no quantization step anywhere on this path.

    End-response speed (user's hard bound): :meth:`generate` reports
    time-to-first-token (prefill time, i.e. time until the first
    token's logits exist) and total time for the full response,
    alongside tok/s. The pass/fail line for the headline run is a
    32-token response within 300 seconds (``within_300s_bound``);
    a fast layer that still yields a slow end response does not pass.
    The route to that bound is GPU + speculative + prefetch combined:
    this class supplies the GPU leg and the speculative hook
    (:meth:`truncate_cache` rolls the KV cache back to an accepted
    prefix after a verify forward, so a speculative wrapper can
    verify K draft tokens in one batched forward and keep only the
    accepted rows). Speculative decoding itself lives outside this
    class; ``generate`` here is plain greedy.

    Exactness: same greedy choice as the CPU paged path except at
    near-ties (cuBLAS/CPU-BLAS reduction order may differ, so
    bit-identical logits are NOT promised). A wide-margin flip is a
    bug, not a tolerance — the Kaggle generation test reports flips
    with their top-2 margins.
    """

    def __init__(self, backend: Backend, config: LayerConfig,
                 n_layer: int, source, tokenizer, eos: int | None = None,
                 vram_budget_bytes: int = 4 * 1024**3,
                 head_block_rows: int = 8192, moe: bool = False):
        self.backend = backend
        self.config = config
        self.n_layer = int(n_layer)
        self.source = source
        self.tok = tokenizer
        self.eos = eos
        self.vram_budget = int(vram_budget_bytes)
        self.head_block_rows = int(head_block_rows)
        # MoE mode: layers run through GPULayer.forward_moe with
        # per-expert paging (qwen3moe/olmoe). Dense models pass
        # moe=False and are untouched by this flag.
        self.moe = bool(moe)
        self.vocab_size = int(source.vocab_size)
        # One executor per layer: each holds ONLY its own layer's KV
        # cache. Sharing one executor across layers would concatenate
        # every layer's keys into one cache — wrong, and invisible in
        # single-layer tests.
        self._executors = [
            GPUExecutor(backend, config, vram_budget_bytes)
            for _ in range(self.n_layer)
        ]
        self.stats: dict = {}
        self.last_layer_stats: list[dict] = []

    # -- construction --------------------------------------------------
    @classmethod
    def from_arrays(cls, backend: Backend, config: LayerConfig,
                    embed, head, out_norm, layers, tokenizer,
                    eos: int | None = None,
                    vram_budget_bytes: int = 4 * 1024**3,
                    head_block_rows: int = 8192) -> "GPUModel":
        """Build from in-memory arrays (synthetic tests, Kaggle
        synthetic generation). ``layers`` is a list of dicts of
        WeightRefs, one per layer."""
        source = _DirectSource(embed, head, out_norm, layers)
        return cls(backend, config, len(layers), source, tokenizer, eos,
                   vram_budget_bytes, head_block_rows)

    @classmethod
    def from_arrays_moe(cls, backend: Backend, config: LayerConfig,
                        embed, head, out_norm, layers, tokenizer,
                        eos: int | None = None,
                        vram_budget_bytes: int = 4 * 1024**3,
                        head_block_rows: int = 8192) -> "GPUModel":
        """Build an MoE model from in-memory arrays (synthetic
        tests). ``layers`` is a list of dicts of WeightRefs plus an
        ``"experts"`` dict keyed (role, e); see
        :class:`_DirectMoESource`."""
        if config.top_k < 1 or config.top_k > config.n_experts:
            raise GPUExecutorError(
                f"MoE model needs 1 <= top_k <= n_experts, got "
                f"top_k={config.top_k}, n_experts={config.n_experts}"
            )
        source = _DirectMoESource(embed, head, out_norm, layers)
        return cls(backend, config, len(layers), source, tokenizer, eos,
                   vram_budget_bytes, head_block_rows, moe=True)

    @classmethod
    def from_store(cls, backend: Backend, config: LayerConfig,
                   n_layer: int, store, by_name: dict, tokenizer,
                   eos: int | None = None,
                   vram_budget_bytes: int = 4 * 1024**3,
                   head_block_rows: int = 8192) -> "GPUModel":
        """Build from a paged store + GGUF tensor infos. This model
        OWNS the store: :meth:`close` closes it."""
        for required in ("token_embd.weight", "output.weight",
                         "output_norm.weight"):
            if required not in by_name:
                raise GPUExecutorError(f"required tensor {required} missing")
        if getattr(store, "precision_policy", None) != "exact":
            raise GPUExecutorError(
                "GPU paged model needs an exact-policy store, got "
                f"{getattr(store, 'precision_policy', None)!r}"
            )
        source = _PagedSource(store, by_name)
        return cls(backend, config, n_layer, source, tokenizer, eos,
                   vram_budget_bytes, head_block_rows)

    @classmethod
    def from_store_moe(cls, backend: Backend, config: LayerConfig,
                       n_layer: int, store, by_name: dict, arch: str,
                       tokenizer, eos: int | None = None,
                       vram_budget_bytes: int = 4 * 1024**3,
                       head_block_rows: int = 8192) -> "GPUModel":
        """Build an MoE model from a paged store + GGUF tensor infos
        (qwen3moe/olmoe). This model OWNS the store: :meth:`close`
        closes it. ``config`` must carry n_experts, top_k, and
        qk_norm (True for qwen3moe)."""
        if arch not in ("qwen3moe", "olmoe"):
            raise GPUExecutorError(
                f"GPU MoE model supports qwen3moe/olmoe, got {arch!r} "
                f"(DeepSeek sigmoid+bias routing and MLA are rung 3)"
            )
        for required in ("token_embd.weight", "output.weight",
                         "output_norm.weight"):
            if required not in by_name:
                raise GPUExecutorError(f"required tensor {required} missing")
        if getattr(store, "precision_policy", None) != "exact":
            raise GPUExecutorError(
                "GPU paged MoE model needs an exact-policy store, got "
                f"{getattr(store, 'precision_policy', None)!r}"
            )
        source = _PagedMoESource(
            store, by_name, arch, n_layer, config.n_experts,
            config.ffn, config.hidden, config.head_dim)
        return cls(backend, config, n_layer, source, tokenizer, eos,
                   vram_budget_bytes, head_block_rows, moe=True)

    @classmethod
    def from_gguf(cls, gguf_path: str, backend: Backend | None = None,
                  hot_budget_bytes: int = 256 * 1024 * 1024,
                  vram_budget_bytes: int = 4 * 1024**3,
                  lac_path: str | None = None) -> "GPUModel":
        """Build from a GGUF file: parse the header, convert to .lac
        if missing (like ``PagedQwen2``), open the paged store."""
        import os as _os

        from .gguf import parse_gguf
        from .pager import PagedWeightStore

        md, infos, _ = parse_gguf(gguf_path)
        arch = md.get("general.architecture")
        if arch not in ("qwen2", "qwen3moe", "olmoe"):
            raise GPUExecutorError(
                "GPU paged model supports qwen2/qwen3moe/olmoe, got "
                f"{arch!r} (DeepSeek sigmoid+bias routing and MLA are "
                f"rung 3, not v1)"
            )
        p = arch + "."
        config = LayerConfig(
            hidden=int(md[p + "embedding_length"]),
            ffn=int(md[p + "feed_forward_length"]),
            n_head=int(md[p + "attention.head_count"]),
            n_kv=int(md[p + "attention.head_count_kv"]),
            eps=float(md[p + "attention.layer_norm_rms_epsilon"]),
            rope_theta=float(md.get(p + "rope.freq_base", 1000000.0)),
            n_experts=int(md[p + "expert_count"]) if arch != "qwen2"
            else 0,
            top_k=int(md[p + "expert_used_count"]) if arch != "qwen2"
            else 0,
            qk_norm=(arch == "qwen3moe"),
        )
        by_name = {i["name"]: i for i in infos}
        lac = lac_path or (gguf_path + ".lac")
        if not _os.path.exists(lac):
            from .convert import convert as _convert

            _convert(gguf_path, lac, quiet=True)
        store = PagedWeightStore(lac, hot_budget_bytes=int(hot_budget_bytes))
        try:
            from .bpe import BPETokenizer

            tok = BPETokenizer.from_metadata(md)
            eos = md.get("tokenizer.ggml.eos_token_id")
            if backend is None:
                backend, _info = detect_backend("auto")
            if arch == "qwen2":
                return cls.from_store(
                    backend, config, int(md[p + "block_count"]), store,
                    by_name, tok, eos, vram_budget_bytes)
            return cls.from_store_moe(
                backend, config, int(md[p + "block_count"]), store,
                by_name, arch, tok, eos, vram_budget_bytes)
        except Exception:
            store.close()
            raise

    # -- cache -----------------------------------------------------------
    def reset_cache(self) -> None:
        for ex in self._executors:
            ex.reset_cache()

    def truncate_cache(self, n_kept: int) -> None:
        """Roll every layer's KV cache back to its first ``n_kept``
        positions. This is the speculative-verify hook: after a
        batched verify forward over K draft tokens, keep only the
        accepted prefix's rows so the next forward continues from
        exactly the accepted state."""
        for ex in self._executors:
            c = ex.cache
            if "k" in c:
                c["k"] = c["k"][:n_kept]
            if "v" in c:
                c["v"] = c["v"][:n_kept]

    # -- budget -----------------------------------------------------------
    def _check_total_budget(self, refs: dict, n_tok: int) -> dict:
        """Binding VRAM check for one layer call: this layer's phase
        peak + the KV caches of ALL layers + an activations bound.
        The per-layer ``GPUExecutor.check_budget`` (own cache only)
        runs too, but it under-counts total KV, so this is the check
        that actually protects the budget. Raises GPUMemoryError."""
        attn = sum(r.f32_bytes() for k, r in refs.items()
                   if k in GPULayer._ATTN_KEYS)
        mlp = 0
        for key in ("ffn_norm.weight", "ffn_gate.weight",
                    "ffn_up.weight", "ffn_down.weight"):
            if key in refs:
                mlp = max(mlp, refs[key].f32_bytes())
        act = n_tok * self.config.hidden * 4 * 8
        kv = sum(ex._cache_bytes() for ex in self._executors)
        peak = max(attn, mlp) + act + kv
        breakdown = {
            "layer_phase_peak_bytes": max(attn, mlp),
            "all_layers_kv_bytes": kv,
            "activations_bound_bytes": act,
            "peak_live_bytes": peak,
            "budget_bytes": self.vram_budget,
        }
        if peak > self.vram_budget:
            raise GPUMemoryError(
                f"GPU model layer peak {peak} bytes > VRAM budget "
                f"{self.vram_budget} (phase {max(attn, mlp)}, all-layer "
                f"KV {kv}, activations bound {act})"
            )
        return breakdown

    def _check_total_budget_moe(self, refs: dict, moe_peak: int,
                                n_tok: int) -> dict:
        """Binding VRAM check for one MoE layer call: this layer's
        phase peak (attention + QK-norm, router, or ONE expert
        matrix — never the sum over experts, which would reject
        every real MoE model) + the KV caches of ALL layers + an
        activations bound. Raises GPUMemoryError."""
        attn = sum(r.f32_bytes() for k, r in refs.items()
                   if k in GPULayer._MOE_ATTN_KEYS)
        router = sum(r.f32_bytes() for k, r in refs.items()
                     if k in ("ffn_gate_inp.weight", "ffn_gate_inp.bias"))
        act = n_tok * self.config.hidden * 4 * 8
        kv = sum(ex._cache_bytes() for ex in self._executors)
        peak = max(attn, router, int(moe_peak)) + act + kv
        breakdown = {
            "layer_phase_peak_bytes": max(attn, router, int(moe_peak)),
            "all_layers_kv_bytes": kv,
            "activations_bound_bytes": act,
            "peak_live_bytes": peak,
            "budget_bytes": self.vram_budget,
        }
        if peak > self.vram_budget:
            raise GPUMemoryError(
                f"GPU MoE model layer peak {peak} bytes > VRAM budget "
                f"{self.vram_budget} (phase "
                f"{max(attn, router, int(moe_peak))}, all-layer "
                f"KV {kv}, activations bound {act})"
            )
        return breakdown

    # -- forward -----------------------------------------------------------
    def forward(self, ids, pos_offset: int = 0) -> np.ndarray:
        """Run ``ids`` through the model; return last-position logits
        (float32 numpy, one per vocab row). ``pos_offset`` is the
        absolute position of ``ids[0]``; the KV cache appends these
        tokens after whatever is already cached."""
        ids = [int(i) for i in ids]
        n_tok = len(ids)
        positions = np.arange(pos_offset, pos_offset + n_tok,
                              dtype=np.float32)
        x = np.array(self.source.embed_rows(ids), dtype=np.float32,
                     copy=True)
        self.last_layer_stats = []
        for li, ex in enumerate(self._executors):
            refs = self.source.layer_refs(li)
            if self.moe:
                # MoE: experts fault on demand inside run_moe_layer
                # (only router-selected experts' raw bytes are ever
                # read); the enforced peak is one expert matrix, not
                # the sum over experts.
                peak = self.source.moe_ffn_peak_bytes(li)
                self._check_total_budget_moe(refs, peak, n_tok)

                def _expert_ref(role, e, li=li):
                    return self.source.expert_ref(li, role, e)

                x, lst = ex.run_moe_layer(x, refs, _expert_ref,
                                          positions, peak)
            else:
                self._check_total_budget(refs, n_tok)
                x, lst = ex.run_layer(x, refs, positions)
            self.last_layer_stats.append(lst)
            del refs  # raw bytes are transient per layer, freed here
        last_np = x[-1:]
        ms = np.mean(last_np * last_np, axis=-1, keepdims=True)
        last = (last_np / np.sqrt(ms + np.float32(self.config.eps))
                ) * self.source.out_norm
        # Blocked lm_head on the backend: only the logits vector is
        # ever assembled whole, exactly like the CPU paged path.
        last_dev = self.backend.asarray(
            np.array(last, dtype=np.float32, copy=True))
        logits = np.empty((self.vocab_size,), dtype=np.float32)
        for b0 in range(0, self.vocab_size, self.head_block_rows):
            b1 = min(b0 + self.head_block_rows, self.vocab_size)
            w = self.backend.asarray(
                np.array(self.source.head_block(b0, b1),
                         dtype=np.float32, copy=True))
            logits[b0:b1] = self.backend.to_numpy(last_dev @ w.T)[0]
            del w
        return logits

    def generate(self, prompt: str, max_tokens: int = 32):
        """Greedy-generate from ``prompt``. Returns (text, token ids,
        stats). Stats carry the end-response numbers the headline run
        is judged on: ``ttft_seconds`` (prefill time, i.e. time until
        the first token's logits exist), ``total_seconds`` for the
        whole response, ``tok_per_s_overall`` / ``tok_per_s_decode``,
        and ``within_300s_bound`` (total <= 300 s). Stops after
        emitting EOS (not included in the text), or at ``max_tokens``.
        The cache is reset per call."""
        self.reset_cache()
        t0 = time.perf_counter()
        ids = self.tok.encode(prompt)
        logits = self.forward(ids, 0)
        ttft = time.perf_counter() - t0
        out: list[int] = []
        for _ in range(int(max_tokens)):
            nxt = int(np.argmax(logits))
            if self.eos is not None and nxt == self.eos:
                break
            out.append(nxt)
            logits = self.forward([nxt], len(ids) + len(out) - 1)
        total = time.perf_counter() - t0
        decode = total - ttft
        stats = {
            "prompt_tokens": len(ids),
            "tokens_generated": len(out),
            "ttft_seconds": ttft,
            "decode_seconds": decode,
            "total_seconds": total,
            "tok_per_s_overall": (len(out) / total) if total > 0 else 0.0,
            "tok_per_s_decode": (len(out) / decode) if decode > 0 else 0.0,
            "within_300s_bound": total <= 300.0,
            "backend": self.backend.name,
            "backend_device": self.backend.device,
        }
        store = getattr(self.source, "store", None)
        if store is not None:
            stats["store_stats"] = dict(store.stats)
        self.stats = stats
        return self.tok.decode(out), out, stats

    def close(self) -> None:
        self.source.close()
        for ex in self._executors:
            ex.reset_cache()
        self.backend.empty_cache()


# ------------------------------------------------------- Gemma-4 GPU path


@dataclass
class GemmaConfig:
    """Gemma-4 text-decoder geometry, mirroring the fields
    :class:`lacuna.gemma.PagedGemma4` reads from the GGUF header.

    Sliding layers: window, n_kv, head_dim, rope_theta (full rotary).
    Global layers: n_kv_global, head_dim_global, rope_theta_global,
    V = K (no separate v projection), and the file's stored
    ``rope_freqs`` factors: global RoPE is the ggml formula (NEOX
    half-rotation over the FULL head dim, angle = position *
    theta**(-2i/head_dim) / rope_freqs[i] — see lacuna.gemma's
    module docstring). There is no partial rotary: the old
    partial-128-dim rotation differs from the reference kernel by
    O(1) on generic vectors for ANY factor choice, so the knobs
    are gone; ``rope_freqs`` (length head_dim_global // 2) is the
    specification. The parity knob that remains is ``embed_scale``
    (default sqrt(hidden), overridable)."""

    hidden: int
    ffn: int
    n_head: int
    n_kv: int
    n_kv_global: int
    head_dim: int = 256
    head_dim_global: int = 512
    window: int = 1024
    rope_theta: float = 10000.0
    rope_theta_global: float = 1000000.0
    rope_freqs: np.ndarray | None = None  # head_dim_global // 2
    eps: float = 1e-6
    embed_scale: float | None = None
    attn_logit_softcapping: float | None = None
    final_logit_softcapping: float | None = None
    layer_types: list[str] | None = None  # "sliding_attention"/"full_attention"

    def __post_init__(self):
        if self.embed_scale is None:
            self.embed_scale = float(math.sqrt(self.hidden))
        if self.rope_freqs is not None:
            arr = np.asarray(self.rope_freqs, dtype=np.float32)
            if arr.shape != (self.head_dim_global // 2,):
                raise GPUExecutorError(
                    f"rope_freqs has shape {arr.shape}, expected "
                    f"({self.head_dim_global // 2},)")
            if not np.all(np.isfinite(arr)) or np.any(arr == 0):
                raise GPUExecutorError(
                    "rope_freqs must be finite and nonzero "
                    "(global RoPE divides by it)")
            self.rope_freqs = arr

    def layer_dims(self, li: int) -> tuple[int, int, int, bool]:
        """(n_q, n_kv, head_dim, is_global) for layer li."""
        if self.layer_types is not None:
            is_global = self.layer_types[li] == "full_attention"
        else:
            is_global = (li % 6 == 5)
        if is_global:
            return self.n_head, self.n_kv_global, self.head_dim_global, True
        return self.n_head, self.n_kv, self.head_dim, False


class _DirectGemmaSource:
    """In-memory Gemma weights for synthetic tests (no pager)."""

    def __init__(self, embed, head, out_norm, layers):
        self._embed = np.array(embed, dtype=np.float32, copy=True)
        self._head = np.array(head, dtype=np.float32, copy=True)
        self._out_norm = np.array(out_norm, dtype=np.float32, copy=True)
        self._layers = layers

    @property
    def vocab_size(self) -> int:
        return int(self._head.shape[0])

    @property
    def out_norm(self) -> np.ndarray:
        return self._out_norm

    def layer_refs(self, li: int) -> dict:
        return self._layers[li]

    def embed_rows(self, ids) -> np.ndarray:
        if len(ids) == 0:
            return np.zeros((0, self._embed.shape[1]), dtype=np.float32)
        return np.array(self._embed[np.asarray(ids, dtype=int)],
                        dtype=np.float32, copy=True)

    def head_block(self, b0: int, b1: int) -> np.ndarray:
        return np.array(self._head[b0:b1], dtype=np.float32, copy=True)

    def close(self) -> None:
        pass


class _PagedGemmaSource:
    """Real-file Gemma weights through the pager — the production
    source. Layer weights arrive as raw GGUF bytes per layer per
    forward (via :func:`read_raw_tensor_bytes`, dequantized on the
    device); embed rows and lm_head blocks arrive sliced via
    ``read_tensor_rows`` (never the whole matrix). Tied embeddings:
    when ``output.weight`` is absent the head blocks are embedding
    rows. Nothing here reads the .lac file directly."""

    def __init__(self, store, by_name: dict):
        from .gguf import read_tensor

        self.store = store
        self.by_name = by_name
        self._embed_info = by_name["token_embd.weight"]
        self._head_info = by_name.get("output.weight", self._embed_info)
        self._out_norm = read_tensor(store, by_name["output_norm.weight"])

    @property
    def vocab_size(self) -> int:
        return int(self._head_info["dims"][1])

    @property
    def out_norm(self) -> np.ndarray:
        return self._out_norm

    def layer_refs(self, li: int) -> dict:
        from .gguf import tensor_shape

        base = f"blk.{li}."
        refs: dict[str, WeightRef] = {}
        for suffix in ("attn_norm.weight", "attn_q.weight", "attn_q.bias",
                       "attn_k.weight", "attn_k.bias", "attn_v.weight",
                       "attn_v.bias", "attn_q_norm.weight",
                       "attn_k_norm.weight", "attn_output.weight",
                       "post_attention_norm.weight",
                       "layer_output_scale.weight",
                       "ffn_norm.weight", "ffn_gate.weight",
                       "ffn_up.weight", "ffn_down.weight",
                       "post_ffw_norm.weight"):
            info = self.by_name.get(base + suffix)
            if info is None:
                continue  # biases / global attn_v may be absent;
                # the layer raises if a required weight is missing
            refs[suffix] = WeightRef(
                shape=tensor_shape(info["dims"]),
                raw=read_raw_tensor_bytes(self.store, info),
                ttype=int(info["type"]),
            )
        return refs

    def embed_rows(self, ids) -> np.ndarray:
        from .gguf import read_tensor_rows

        if len(ids) == 0:
            hidden = int(self._embed_info["dims"][0])
            return np.zeros((0, hidden), dtype=np.float32)
        uniq = sorted(set(int(i) for i in ids))
        rows: dict[int, np.ndarray] = {}
        runs: list[tuple[int, int]] = []
        start = last = uniq[0]
        for u in uniq[1:]:
            if u == last + 1:
                last = u
            else:
                runs.append((start, last + 1))
                start = last = u
        runs.append((start, last + 1))
        for a, b in runs:
            arr = read_tensor_rows(self.store, self._embed_info, a, b)
            for k, u in enumerate(range(a, b)):
                rows[u] = np.array(arr[k], dtype=np.float32, copy=True)
        return np.stack([rows[int(i)] for i in ids]).astype(np.float32)

    def head_block(self, b0: int, b1: int) -> np.ndarray:
        from .gguf import read_tensor_rows

        return read_tensor_rows(self.store, self._head_info, b0, b1)

    def close(self) -> None:
        self.store.close()


class GemmaGPULayer:
    """One Gemma-4 decoder layer on a backend. The math mirrors
    :class:`lacuna.gemma.PagedGemma4`'s per-layer block exactly
    (RMSNorm, separate Q/K/V GEMMs, QK-norm, RoPE half-rotation,
    GQA repeat, hybrid sliding/global attention, GELU-tanh gated
    MLP, Peri-LN post-norms + layer_output_scale, attention scale
    1.0), so GPU-vs-CPU layer outputs are comparable
    element-for-element.

    Differences from the CPU path are deliberate and documented:
    attention runs in the batched-head layout of :class:`GPULayer`
    (H,T,D) rather than the CPU path's einsum layout, and weights
    stream per matrix (never fused, never whole-layer). The greedy
    choice must still match except at near-ties; a wide-margin flip
    is a bug, not a tolerance."""

    def __init__(self, backend: Backend, config: GemmaConfig):
        self.b = backend
        self.cfg = config
        self._inv_freq = backend.asarray(
            np.array(config.rope_theta
                     ** (-np.arange(0, config.head_dim, 2, dtype=np.float32)
                         / config.head_dim), dtype=np.float32)
        )
        # Global: full head dim, ggml-style; the stored rope_freqs
        # factors divide the angles in _rope (kept separate from
        # inv_freq so the formula stays legible against
        # ggml_rope_cache_init).
        inv_g = np.array(config.rope_theta_global
                         ** (-np.arange(0, config.head_dim_global, 2,
                                        dtype=np.float32)
                             / config.head_dim_global), dtype=np.float32)
        self._inv_freq_global = backend.asarray(inv_g)
        self._rope_freqs = (
            backend.asarray(np.array(config.rope_freqs, dtype=np.float32))
            if config.rope_freqs is not None else None
        )

    # -- primitives (backend-generic) --------------------------------
    def _rmsnorm(self, x: object, w: object) -> object:
        b = self.b
        ms = b.mean(x * x, axis=-1)
        return (x / b.sqrt(ms + np.float32(self.cfg.eps))) * w

    @staticmethod
    def _tanh(b: Backend, x: object) -> object:
        """tanh via exp (the Backend interface has no tanh method):
        tanh(x) = 1 - 2 / (exp(2x) + 1). Exact in float32 up to the
        same rounding the CPU path's np.tanh gives; the GELU-tanh
        parity test pins the agreement."""
        return np.float32(1.0) - np.float32(2.0) / (
            b.exp(np.float32(2.0) * x) + np.float32(1.0))

    def _gelu_tanh(self, x: object) -> object:
        return (np.float32(0.5) * x * (np.float32(1.0) + self._tanh(
            self.b, np.float32(0.7978845608028654) * (
                x + np.float32(0.044715) * x * x * x))))

    def _rope(self, t: object, positions: object,
              is_global: bool) -> object:
        """RoPE, half-rotation form (HF ``rotate_half``), same as the
        CPU path. Sliding layers rotate the full head dim. Global
        layers rotate the FULL head dim too, ggml-style: angle =
        position * inv_freq / rope_freqs (the file's stored
        factors). There is no partial rotary and no passthrough
        on global layers."""
        b = self.b
        if is_global:
            rotary = self.cfg.head_dim_global
            inv_freq = self._inv_freq_global
            if self._rope_freqs is None:
                raise GPUExecutorError(
                    "global RoPE needs rope_freqs (file tensor); "
                    "GemmaConfig.rope_freqs is None")
            ang = (positions.reshape(-1, 1) * inv_freq.reshape(1, -1)
                   / self._rope_freqs.reshape(1, -1))
        else:
            rotary = t.shape[-1]
            inv_freq = self._inv_freq
            ang = positions.reshape(-1, 1) * inv_freq.reshape(1, -1)
        cos = b.cos(ang)[:, None, :]
        sin = b.sin(ang)[:, None, :]
        h = rotary // 2
        x0 = t[..., :h]
        x1 = t[..., h:rotary]
        first = x0 * cos - x1 * sin
        second = x0 * sin + x1 * cos
        parts = [first, second]
        if rotary < t.shape[-1]:
            parts.append(t[..., rotary:])
        return b.cat(parts, axis=-1)

    def _attention(self, q: object, k_all: object, v_all: object,
                   positions: object, col_start: int,
                   is_sliding: bool) -> object:
        """Dot-product attention over cached K/V, batched-head
        layout. ``q`` is (T,H,D); ``k_all``/``v_all`` are (S,H,D)
        after GQA repeat. ``positions`` is the (T,) absolute float32
        query positions; ``col_start`` is the absolute position of
        the first cached key row (sliding layers trim, so the cache
        start moves). Sliding layers additionally allow only the
        last ``window`` positions (current token included).

        Gemma-4 attention scale is 1.0, NOT 1/sqrt(head_dim): QK-norm
        provides the stability (see lacuna.gemma._attention)."""
        b = self.b
        n_tok = q.shape[0]
        n_keys = k_all.shape[0]
        q_b = b.permute(q, (1, 0, 2))
        k_b = b.permute(k_all, (1, 2, 0))
        scores = q_b @ k_b
        col = (b.arange_int(n_keys) + int(col_start)).reshape(1, 1, -1)
        row = (b.arange_int(n_tok)
               + int(self._pos_offset)).reshape(1, -1, 1)
        allowed = col <= row
        if is_sliding:
            allowed = allowed & ((row - col) < int(self.cfg.window))
        scores = b.where(allowed, scores,
                         b.zeros(scores.shape) + np.float32(-1e30))
        if self.cfg.attn_logit_softcapping is not None:
            cap = np.float32(self.cfg.attn_logit_softcapping)
            scores = cap * self._tanh(b, scores / cap)
        scores = scores - b.reduce_max(scores, axis=-1)
        probs = b.exp(scores)
        probs = probs / b.reduce_sum(probs, axis=-1)
        v_b = b.permute(v_all, (1, 0, 2))
        return b.permute(probs @ v_b, (1, 0, 2))

    # -- forward -------------------------------------------------------
    def forward(self, x_np: np.ndarray, refs: dict[str, WeightRef],
                positions_np: np.ndarray, cache: dict,
                trim: bool = True) -> np.ndarray:
        """Run one layer with per-matrix streaming. ``cache`` is this
        layer's own dict (``"k"``/``"v"`` backend arrays plus
        ``"start"``, the absolute position of the first cached row)
        and is updated in place — per-layer isolation is structural:
        no layer ever sees another layer's cache. Sliding layers trim
        AFTER attention (trimming before would leave early prefill
        positions with no allowed keys at all, the bug the CPU path
        documents) — unless ``trim`` is False, in which case the cache
        grows transiently and the caller must restore the window
        bound with a later ``truncate_cache``. That no-trim mode is
        what speculative verification needs: trimming during a
        verification block would move the window head past keys a
        rollback to the accepted prefix still needs (see
        :mod:`lacuna.speculative_gpu_gemma`). Returns numpy."""
        b = self.b
        cfg = self.cfg
        x = b.asarray(np.array(x_np, dtype=np.float32, copy=True))
        pos_arr = np.array(positions_np, dtype=np.float32, copy=True)
        positions = b.asarray(pos_arr)
        self._pos_offset = int(pos_arr.reshape(-1)[0]) if pos_arr.size else 0
        n_tok = x.shape[0]
        # Per-layer geometry is fixed by the layer's own dims, not by
        # a shared config: the caller passes n_q/n_kv/head_dim via
        # the refs' shapes, but the head counts live on the layer
        # object (set by GPUGemmaModel before each call).
        n_q, n_kv, d, is_global = self._dims
        group = n_q // n_kv
        if n_q % n_kv:
            raise GPUExecutorError(
                f"Gemma layer: Q heads {n_q} not divisible by KV heads "
                f"{n_kv}")

        def mat(key: str) -> object:
            ref = refs.get(key)
            if ref is None:
                raise GPUExecutorError(f"layer weight {key!r} missing")
            return ref.materialize(b)

        def bias(key: str, n: int) -> object:
            if key in refs:
                return refs[key].materialize(b)
            return b.zeros((n,))

        # Phase 1: attention. Only attn_* matrices are live here.
        attn_norm = mat("attn_norm.weight")
        h = self._rmsnorm(x, attn_norm)
        del attn_norm
        q = h @ mat("attn_q.weight").T + bias("attn_q.bias", n_q * d)
        k = h @ mat("attn_k.weight").T + bias("attn_k.bias", n_kv * d)
        has_v = "attn_v.weight" in refs
        if has_v:
            v_raw = h @ mat("attn_v.weight").T + bias("attn_v.bias",
                                                      n_kv * d)
        else:
            if not is_global:
                raise GPUExecutorError(
                    "sliding layer missing attn_v.weight")
            # Global V = K: V starts as the RAW k projection
            # (llama.cpp gemma4.cpp assigns Vcur = Kcur before any
            # norm is applied to either).
            v_raw = k
        del h
        q = q.reshape(n_tok, n_q, d)
        k = k.reshape(n_tok, n_kv, d)
        # QK-norm (Gemma-4 QKNorm): per-head RMSNorm before RoPE.
        # mat() fails loud on a file that lacks these tensors.
        q_norm = mat("attn_q_norm.weight")
        q = self._rmsnorm(q, q_norm)
        del q_norm
        k_norm = mat("attn_k_norm.weight")
        k = self._rmsnorm(k, k_norm)
        del k_norm
        # V-norm (llama.cpp gemma4.cpp): V is ALWAYS weightless
        # per-head RMSNormed (explicit v projection, or the raw k
        # projection on V=K global layers). V is never RoPE'd.
        ones = b.asarray(np.ones(d, dtype=np.float32))
        v = self._rmsnorm(v_raw.reshape(n_tok, n_kv, d), ones)
        del ones, v_raw
        q = self._rope(q, positions, is_global)
        k = self._rope(k, positions, is_global)
        k_rep = b.repeat_kv(k, group)
        v_rep = b.repeat_kv(v, group)
        del k, v
        if "k" in cache and "v" in cache:
            k_all = b.cat([cache["k"], k_rep], axis=0)
            v_all = b.cat([cache["v"], v_rep], axis=0)
            start = int(cache["start"])
        else:
            k_all, v_all = k_rep, v_rep
            start = self._pos_offset
        del k_rep, v_rep
        att = self._attention(q, k_all, v_all, positions, start,
                              not is_global)
        del q
        if trim and not is_global and k_all.shape[0] > cfg.window:
            drop = k_all.shape[0] - cfg.window
            k_all = k_all[drop:]
            v_all = v_all[drop:]
            start += drop
        cache["k"], cache["v"], cache["start"] = k_all, v_all, start
        del k_all, v_all
        attn_out = mat("attn_output.weight")
        try:
            # Peri-LN post-norm (see lacuna.gemma). The
            # layer_output_scale multiplies the WHOLE layer output
            # at the end (llama.cpp gemma4.cpp), not this branch.
            att_out = att.reshape(n_tok, n_q * d) @ attn_out.T
            post = mat("post_attention_norm.weight")
            att_out = self._rmsnorm(att_out, post)
            del post
            x1 = x + att_out
        finally:
            del attn_out, att
            b.empty_cache()
        # Phase 2: MLP, one matrix at a time (gate act kept, gate freed
        # before up, up freed before down — same discipline as dense).
        ffn_norm = mat("ffn_norm.weight")
        h2 = self._rmsnorm(x1, ffn_norm)
        del ffn_norm
        gate = mat("ffn_gate.weight")
        gate_act = self._gelu_tanh(h2 @ gate.T)
        del gate
        up = mat("ffn_up.weight")
        act = gate_act * (h2 @ up.T)
        del up, gate_act, h2
        down = mat("ffn_down.weight")
        try:
            mlp_out = act @ down.T
            post = mat("post_ffw_norm.weight")
            mlp_out = self._rmsnorm(mlp_out, post)
            del post
            # layer_output_scale multiplies the WHOLE layer output
            # (llama.cpp gemma4.cpp), residual included.
            scale = mat("layer_output_scale.weight")
            out = (x1 + mlp_out) * scale
            del scale
            return b.to_numpy(out)
        finally:
            del down, act
            b.empty_cache()


class GPUGemmaModel:
    """Full Gemma-4 generation on a Backend (numpy or torch-CUDA).

    The GPU-mode Gemma model behind ``Engine(path, gpu=...)`` when the
    file's architecture is gemma4: the same decoder math as
    :class:`lacuna.gemma.PagedGemma4` (hybrid sliding/global attention,
    V = K on global layers, ggml-style global RoPE with the file's
    stored rope_freqs factors, GELU-tanh gated MLP, tied
    embeddings, RMSNorm eps 1e-6, no attention bias by default), but
    every layer runs through :class:`GemmaGPULayer` on the backend,
    with one KV cache dict per layer (per-layer isolation, exactly
    as the dense GPU path). Embed rows and lm_head blocks are sliced
    (never materialized whole), and layer weights arrive as raw GGUF
    bytes per layer per forward, dequantized on the device.

    Parity knob (explicit parameter, from the CPU path): embed
    scale (default sqrt(hidden), overridable). Global RoPE has no
    knob: it is fully determined by the file's rope_freqs tensor.

    Exactness: same greedy choice as the CPU paged path except at
    near-ties (backend GEMM reduction order may differ, so
    bit-identical logits are NOT promised). A wide-margin flip is a
    bug, not a tolerance. End-response stats mirror
    :class:`GPUModel` (``ttft_seconds``, ``total_seconds``,
    ``within_300s_bound``); the 32-token / 300 s headline gate is
    judged on these fields, not on layer speed.
    """

    def __init__(self, backend: Backend, config: GemmaConfig,
                 n_layer: int, source, tokenizer, eos: int | None = None,
                 vram_budget_bytes: int = 4 * 1024**3,
                 head_block_rows: int = 8192):
        self.backend = backend
        self.config = config
        self.n_layer = int(n_layer)
        self.source = source
        self.tok = tokenizer
        self.eos = eos
        self.vram_budget = int(vram_budget_bytes)
        self.head_block_rows = int(head_block_rows)
        self.vocab_size = int(source.vocab_size)
        # One layer object per layer would duplicate RoPE tables for
        # no benefit; a single shared layer plus one cache dict per
        # layer gives the same isolation (caches are per-layer state,
        # never shared) with one set of frequency tables.
        self._layer = GemmaGPULayer(backend, config)
        self._caches: list[dict] = [dict() for _ in range(self.n_layer)]
        self.stats: dict = {}
        self.last_layer_stats: list[dict] = []

    # -- construction --------------------------------------------------
    @classmethod
    def from_arrays(cls, backend: Backend, config: GemmaConfig,
                    embed, head, out_norm, layers, tokenizer,
                    eos: int | None = None,
                    vram_budget_bytes: int = 4 * 1024**3,
                    head_block_rows: int = 8192) -> "GPUGemmaModel":
        """Build from in-memory arrays (synthetic tests). ``layers``
        is a list of dicts of WeightRefs, one per layer."""
        source = _DirectGemmaSource(embed, head, out_norm, layers)
        return cls(backend, config, len(layers), source, tokenizer, eos,
                   vram_budget_bytes, head_block_rows)

    @classmethod
    def from_store(cls, backend: Backend, config: GemmaConfig,
                   n_layer: int, store, by_name: dict, tokenizer,
                   eos: int | None = None,
                   vram_budget_bytes: int = 4 * 1024**3,
                   head_block_rows: int = 8192) -> "GPUGemmaModel":
        """Build from a paged store + GGUF tensor infos. This model
        OWNS the store: :meth:`close` closes it."""
        for required in ("token_embd.weight", "output_norm.weight"):
            if required not in by_name:
                raise GPUExecutorError(f"required tensor {required} missing")
        if getattr(store, "precision_policy", None) != "exact":
            raise GPUExecutorError(
                "GPU paged Gemma model needs an exact-policy store, got "
                f"{getattr(store, 'precision_policy', None)!r}"
            )
        # Global RoPE factors: read from the store when the caller
        # did not supply them (the from_gguf path) and the model has
        # global layers. Missing factors fail loud, never default —
        # ggml-style global RoPE is undefined without the file tensor.
        if config.rope_freqs is None and any(
                config.layer_dims(li)[3] for li in range(n_layer)):
            if "rope_freqs.weight" not in by_name:
                raise GPUExecutorError(
                    "required tensor rope_freqs.weight missing "
                    "(global RoPE needs the file's stored factors)")
            from dataclasses import replace as _replace

            from .gguf import read_tensor as _read_tensor

            config = _replace(
                config,
                rope_freqs=np.asarray(
                    _read_tensor(store, by_name["rope_freqs.weight"]),
                    dtype=np.float32).reshape(-1))
        source = _PagedGemmaSource(store, by_name)
        return cls(backend, config, n_layer, source, tokenizer, eos,
                   vram_budget_bytes, head_block_rows)

    @classmethod
    def from_gguf(cls, gguf_path: str, backend: Backend | None = None,
                  hot_budget_bytes: int = 256 * 1024 * 1024,
                  vram_budget_bytes: int = 4 * 1024**3,
                  lac_path: str | None = None,
                  embed_scale: float | None = None) -> "GPUGemmaModel":
        """Build from a GGUF file: parse the header, convert to .lac
        if missing (like ``PagedGemma4``), open the paged store. The
        layer pattern, global-KV key, and softcapping keys follow the
        CPU loader's contract exactly (see docs/GEMMA4_31B.md); the
        loader fails loud naming any missing architecture-defining
        key instead of defaulting it."""
        import os as _os

        from .gguf import parse_gguf
        from .pager import PagedWeightStore

        md, infos, _ = parse_gguf(gguf_path)
        if md.get("general.architecture") != "gemma4":
            raise GPUExecutorError(
                f"GPU Gemma model supports gemma4 only, got "
                f"{md.get('general.architecture')!r}")
        p = "gemma4."
        for key in ("block_count", "embedding_length",
                    "feed_forward_length", "attention.head_count",
                    "attention.head_count_kv",
                    "attention.layer_norm_rms_epsilon"):
            if p + key not in md:
                raise GPUExecutorError(f"required GGUF key {p + key} missing")
        n_layer = int(md[p + "block_count"])
        from .gemma import (
            _layer_types_from_metadata,
            resolve_gemma4_kv_geometry,
            GemmaModelError,
        )

        layer_types = _layer_types_from_metadata(md, p, n_layer)
        by_name = {i["name"]: i for i in infos}
        try:
            (n_kv, n_kv_global, head_dim,
             head_dim_global) = resolve_gemma4_kv_geometry(
                 md, p, n_layer, layer_types, by_name)
        except GemmaModelError as e:
            raise GPUExecutorError(str(e)) from e
        hidden = int(md[p + "embedding_length"])
        if embed_scale is not None:
            scale = float(embed_scale)
        elif p + "embedding_scale" in md:
            scale = float(md[p + "embedding_scale"])
        else:
            scale = float(math.sqrt(hidden))
        final_cap = md.get(p + "final_logit_softcapping")
        attn_cap = md.get(p + "attn_logit_softcapping")
        config = GemmaConfig(
            hidden=hidden,
            ffn=int(md[p + "feed_forward_length"]),
            n_head=int(md[p + "attention.head_count"]),
            n_kv=n_kv,
            n_kv_global=n_kv_global,
            head_dim=head_dim,
            head_dim_global=head_dim_global,
            window=int(md.get(p + "attention.sliding_window",
                              md.get(p + "sliding_window", 1024))),
            rope_theta=float(md.get(p + "rope.freq_base_swa",
                                   md.get(p + "rope.freq_base",
                                          md.get(p + "rope.theta",
                                                 10000.0)))),
            rope_theta_global=float(md.get(
                p + "rope.freq_base_global",
                md.get(p + "rope.theta_global",
                       md.get(p + "rope.freq_base", 1000000.0)))),
            eps=float(md[p + "attention.layer_norm_rms_epsilon"]),
            embed_scale=scale,
            attn_logit_softcapping=(float(attn_cap)
                                    if attn_cap is not None else None),
            final_logit_softcapping=(float(final_cap)
                                     if final_cap is not None else None),
            layer_types=layer_types,
        )
        lac = lac_path or (gguf_path + ".lac")
        if not _os.path.exists(lac):
            from .convert import convert as _convert

            _convert(gguf_path, lac, quiet=True)
        store = PagedWeightStore(lac, hot_budget_bytes=int(hot_budget_bytes))
        try:
            from .spm import SPMTokenizer

            tok = SPMTokenizer.from_metadata(md)
            eos = md.get("tokenizer.ggml.eos_token_id")
            if backend is None:
                backend, _info = detect_backend("auto")
            return cls.from_store(
                backend, config, n_layer, store, by_name, tok, eos,
                vram_budget_bytes)
        except Exception:
            store.close()
            raise

    # -- cache -----------------------------------------------------------
    def reset_cache(self) -> None:
        self._caches = [dict() for _ in range(self.n_layer)]

    def truncate_cache(self, n_kept: int) -> None:
        """Roll every layer's KV cache back to absolute position
        ``n_kept`` (speculative-verify hook, same semantics as
        :meth:`lacuna.speculative_gemma.BatchedGemma4.truncate_cache`
        on the CPU path). One count serves all layers: global caches
        hold exactly positions ``[0, n_kept)`` and sliding caches hold
        the last window of ``[0, n_kept)`` that was still cached —
        exactly what a fresh model would hold after that prefix, and
        exactly the keys any future query can still attend to.

        Fails loud when ``n_kept`` is at or before a layer's current
        window head (and nonzero): those keys were already trimmed by
        an earlier block and cannot be recovered from cache — the
        caller must re-prefill instead. ``n_kept == 0`` always
        succeeds (caches cleared to fresh state). This replaces the
        previous cache-relative semantics (sliding ``n_kept`` counted
        rows from the cache front), which no single count could make
        correct for both layer types once any sliding layer trimmed.
        """
        n_kept = int(n_kept)
        if n_kept < 0:
            raise GPUExecutorError(
                f"truncate_cache: negative n_kept {n_kept}")
        for li, c in enumerate(self._caches):
            if "k" not in c:
                continue
            k, v, start = c["k"], c["v"], int(c["start"])
            end = start + k.shape[0]
            if n_kept >= end:
                k2, v2, start2 = k, v, start
            elif n_kept == 0:
                c.clear()
                continue
            elif n_kept <= start:
                raise GPUExecutorError(
                    f"layer {li}: cannot roll back to {n_kept}: cache "
                    f"starts at {start} (earlier keys were trimmed); "
                    f"re-prefill required")
            else:
                k2, v2, start2 = (k[: n_kept - start],
                                  v[: n_kept - start], start)
            if not self.config.layer_dims(li)[3] and \
                    k2.shape[0] > self.config.window:
                drop = k2.shape[0] - self.config.window
                k2, v2, start2 = k2[drop:], v2[drop:], start2 + drop
            c["k"], c["v"], c["start"] = k2, v2, start2

    def cache_len(self) -> int:
        """Absolute position one past the last cached row (0 when
        empty). Sliding layers report window head + rows, global
        layers their full length; the max is the emitted-prefix
        length the cache as a whole covers."""
        best = 0
        for c in self._caches:
            if "k" in c:
                best = max(best, int(c["start"]) + int(c["k"].shape[0]))
        return best

    # -- budget -----------------------------------------------------------
    @staticmethod
    def _arr_bytes(arr) -> int:
        n = arr.numel() if hasattr(arr, "numel") else arr.size
        return int(n) * 4

    def _check_total_budget(self, refs: dict, n_tok: int) -> dict:
        """Binding VRAM check for one layer call: this layer's phase
        peak (attention matrices, or the single largest MLP matrix —
        gate/up/down stream one at a time) + the KV caches of ALL
        layers + an activations bound. Raises GPUMemoryError on
        breach; never silently spills, downcasts, or drops weights."""
        attn = sum(r.f32_bytes() for k, r in refs.items()
                   if k.startswith("attn_"))
        mlp = 0
        for key in ("ffn_norm.weight", "ffn_gate.weight",
                    "ffn_up.weight", "ffn_down.weight"):
            if key in refs:
                mlp = max(mlp, refs[key].f32_bytes())
        act = n_tok * self.config.hidden * 4 * 8
        kv = sum(self._arr_bytes(a)
                 for c in self._caches for k, a in c.items()
                 if k in ("k", "v"))
        peak = max(attn, mlp) + act + kv
        breakdown = {
            "layer_phase_peak_bytes": max(attn, mlp),
            "all_layers_kv_bytes": kv,
            "activations_bound_bytes": act,
            "peak_live_bytes": peak,
            "budget_bytes": self.vram_budget,
        }
        if peak > self.vram_budget:
            raise GPUMemoryError(
                f"GPU Gemma model layer peak {peak} bytes > VRAM "
                f"budget {self.vram_budget} (phase {max(attn, mlp)}, "
                f"all-layer KV {kv}, activations bound {act})"
            )
        return breakdown

    # -- forward -----------------------------------------------------------
    def forward(self, ids, pos_offset: int = 0) -> np.ndarray:
        """Run ``ids`` through the model; return last-position logits
        (float32 numpy, one per vocab row). ``pos_offset`` is the
        absolute position of ``ids[0]``; the KV caches append these
        tokens after whatever is already cached."""
        ids = [int(i) for i in ids]
        n_tok = len(ids)
        positions = np.arange(pos_offset, pos_offset + n_tok,
                              dtype=np.float32)
        x = np.array(self.source.embed_rows(ids), dtype=np.float32,
                     copy=True) * np.float32(self.config.embed_scale)
        self.last_layer_stats = []
        for li in range(self.n_layer):
            n_q, n_kv, d, is_global = self.config.layer_dims(li)
            self._layer._dims = (n_q, n_kv, d, is_global)
            refs = self.source.layer_refs(li)
            self.last_layer_stats.append(
                self._check_total_budget(refs, n_tok))
            x = self._layer.forward(x, refs, positions, self._caches[li])
            del refs  # raw bytes are transient per layer, freed here
        last_np = x[-1:]
        ms = np.mean(last_np * last_np, axis=-1, keepdims=True)
        last = (last_np / np.sqrt(ms + np.float32(self.config.eps))
                ) * self.source.out_norm
        last_dev = self.backend.asarray(
            np.array(last, dtype=np.float32, copy=True))
        logits = np.empty((self.vocab_size,), dtype=np.float32)
        for b0 in range(0, self.vocab_size, self.head_block_rows):
            b1 = min(b0 + self.head_block_rows, self.vocab_size)
            w = self.backend.asarray(
                np.array(self.source.head_block(b0, b1),
                         dtype=np.float32, copy=True))
            logits[b0:b1] = self.backend.to_numpy(last_dev @ w.T)[0]
            del w
        if self.config.final_logit_softcapping is not None:
            cap = np.float32(self.config.final_logit_softcapping)
            logits = cap * np.tanh(logits / cap)
        return logits

    def forward_all(self, ids, pos_offset: int = 0) -> np.ndarray:
        """Run ``ids`` through the model; return per-position logits
        (float32 numpy, one row per input token). This is the
        verification primitive speculative decoding needs: one call
        yields the target's prediction at EVERY draft position, so a
        whole draft block verifies at once.

        Deliberate difference from :meth:`forward`: layers run with
        ``trim=False`` (see :meth:`GemmaGPULayer.forward`) — the
        cache grows transiently by the block length and trimming is
        deferred to :meth:`truncate_cache`, which slices to the
        accepted prefix and re-trims to that prefix's window.
        Trimming during verification would move the window head past
        keys a rollback still needs. Callers that run
        ``forward_all`` without a following ``truncate_cache`` hold
        the full (untrimmed) cache, exactly as a single ``forward``
        call does within its own call. ``pos_offset`` is the absolute
        position of ``ids[0]``; K/V for ``ids`` are appended at those
        positions. The last row matches :meth:`forward` on the same
        input at 2e-3 with an exact argmax (pinned by test) — not
        bitwise, because a (T,H) head matmul need not reduce in the
        same order as a (1,H) one (measured 9.3e-10 on the tiny
        model); per-row logits use the same RMSNorm + blocked-head
        computation as ``forward``, applied to every position's
        final hidden state instead of only the last."""
        ids = [int(i) for i in ids]
        n_tok = len(ids)
        if n_tok == 0:
            return np.empty((0, self.vocab_size), dtype=np.float32)
        positions = np.arange(pos_offset, pos_offset + n_tok,
                              dtype=np.float32)
        x = np.array(self.source.embed_rows(ids), dtype=np.float32,
                     copy=True) * np.float32(self.config.embed_scale)
        self.last_layer_stats = []
        for li in range(self.n_layer):
            n_q, n_kv, d, is_global = self.config.layer_dims(li)
            self._layer._dims = (n_q, n_kv, d, is_global)
            refs = self.source.layer_refs(li)
            self.last_layer_stats.append(
                self._check_total_budget(refs, n_tok))
            x = self._layer.forward(x, refs, positions, self._caches[li],
                                    trim=False)
            del refs  # raw bytes are transient per layer, freed here
        ms = np.mean(x * x, axis=-1, keepdims=True)
        last = (x / np.sqrt(ms + np.float32(self.config.eps))
                ) * self.source.out_norm
        last_dev = self.backend.asarray(
            np.array(last, dtype=np.float32, copy=True))
        logits = np.empty((n_tok, self.vocab_size), dtype=np.float32)
        for b0 in range(0, self.vocab_size, self.head_block_rows):
            b1 = min(b0 + self.head_block_rows, self.vocab_size)
            w = self.backend.asarray(
                np.array(self.source.head_block(b0, b1),
                         dtype=np.float32, copy=True))
            logits[:, b0:b1] = self.backend.to_numpy(last_dev @ w.T)
            del w
        if self.config.final_logit_softcapping is not None:
            cap = np.float32(self.config.final_logit_softcapping)
            logits = cap * np.tanh(logits / cap)
        return logits

    def generate(self, prompt: str, max_tokens: int = 32):
        """Greedy-generate from ``prompt``. Returns (text, token ids,
        stats) with the same end-response stats as
        :meth:`GPUModel.generate` (``ttft_seconds``,
        ``total_seconds``, ``within_300s_bound``). Stops after
        emitting EOS (not included in the text), or at
        ``max_tokens``. The cache is reset per call. No BOS is
        prepended (same parity item as the CPU path)."""
        self.reset_cache()
        t0 = time.perf_counter()
        ids = self.tok.encode(prompt)
        logits = self.forward(ids, 0)
        ttft = time.perf_counter() - t0
        out: list[int] = []
        for _ in range(int(max_tokens)):
            nxt = int(np.argmax(logits))
            if self.eos is not None and nxt == self.eos:
                break
            out.append(nxt)
            logits = self.forward([nxt], len(ids) + len(out) - 1)
        total = time.perf_counter() - t0
        decode = total - ttft
        stats = {
            "prompt_tokens": len(ids),
            "tokens_generated": len(out),
            "ttft_seconds": ttft,
            "decode_seconds": decode,
            "total_seconds": total,
            "tok_per_s_overall": (len(out) / total) if total > 0 else 0.0,
            "tok_per_s_decode": (len(out) / decode) if decode > 0 else 0.0,
            "within_300s_bound": total <= 300.0,
            "backend": self.backend.name,
            "backend_device": self.backend.device,
        }
        store = getattr(self.source, "store", None)
        if store is not None:
            stats["store_stats"] = dict(store.stats)
        self.stats = stats
        return self.tok.decode(out), out, stats

    def close(self) -> None:
        self.source.close()
        self.reset_cache()
        self.backend.empty_cache()
