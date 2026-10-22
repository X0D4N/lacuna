"""Lacuna — lookahead sparse-paged LLM inference.

Run 100B-class models on 8 GB RAM + <4 GB VRAM by minimizing bytes-per-token
(sparsity oracle + adaptive precision + speculative amortization) and hiding
fetch latency (async NVMe paging overlapped with compute).

One-line API::

    from lacuna import Engine
    print(Engine("model.lac").generate("hello"))

Stdlib + numpy only.
"""

__version__ = "0.3.0"

from .buffer import BufferCapacityError, BufferError, PagerBackedBuffer, tiled_exact_matvec
from .engine import Engine, LacunaEngine
from .hardware import (
    HardwareInfo,
    HardwareTier,
    TuneConfig,
    auto_tune,
    detect_hardware,
    detect_hardware_info,
)
from .ollama_shim import InferenceBackend, build_ollama_server, serve_ollama
from .oracle import Oracle
from .pager import PagedWeightStore, LacWriter
from .prefetch import CacheExpertPredictor, NgramPrefetcher, NgramSpec, ShadowScorer
from .pull import PullError, download, list_gguf_files, pick_gguf, pull
from .quant import (
    POLICY_EXACT,
    POLICY_EXTREME,
    PRECISION_POLICIES,
    pack_q2,
    pack_q4,
    pack_ternary,
    roundtrip_max_error,
    unpack_q2,
    unpack_q4,
    unpack_ternary,
    validate_policy,
)
from .scheduler import OverlapScheduler, simulate_overlap
from .speculate import speculative_generate, speculative_tree_generate
from .warmstart import default_hotset_path, restore_hotset, save_hotset

__all__ = [
    "__version__",
    "BufferCapacityError",
    "BufferError",
    "CacheExpertPredictor",
    "Engine",
    "HardwareInfo",
    "HardwareTier",
    "InferenceBackend",
    "LacunaEngine",
    "NgramPrefetcher",
    "NgramSpec",
    "Oracle",
    "OverlapScheduler",
    "PagedWeightStore",
    "PagerBackedBuffer",
    "PullError",
    "TuneConfig",
    "LacWriter",
    "POLICY_EXACT",
    "POLICY_EXTREME",
    "PRECISION_POLICIES",
    "auto_tune",
    "build_ollama_server",
    "default_hotset_path",
    "detect_hardware",
    "detect_hardware_info",
    "download",
    "list_gguf_files",
    "pack_q2",
    "pack_q4",
    "pack_ternary",
    "pick_gguf",
    "pull",
    "restore_hotset",
    "roundtrip_max_error",
    "save_hotset",
    "ShadowScorer",
    "serve_ollama",
    "simulate_overlap",
    "speculative_generate",
    "speculative_tree_generate",
    "tiled_exact_matvec",
    "unpack_q2",
    "unpack_q4",
    "unpack_ternary",
    "validate_policy",
]
