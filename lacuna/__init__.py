"""Lacuna — lookahead sparse-paged LLM inference.

Run 100B-class models on 8 GB RAM + <4 GB VRAM by minimizing bytes-per-token
(sparsity oracle + adaptive precision + speculative amortization) and hiding
fetch latency (async NVMe paging overlapped with compute).

One-line API::

    from lacuna import Engine
    print(Engine("model.lac").generate("hello"))

Stdlib + numpy only.
"""

__version__ = "0.2.0"

from .engine import Engine, LacunaEngine
from .hardware import HardwareTier, detect_hardware
from .oracle import Oracle
from .pager import PagedWeightStore, LacWriter
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

__all__ = [
    "__version__",
    "Engine",
    "HardwareTier",
    "LacunaEngine",
    "Oracle",
    "OverlapScheduler",
    "PagedWeightStore",
    "LacWriter",
    "POLICY_EXACT",
    "POLICY_EXTREME",
    "PRECISION_POLICIES",
    "detect_hardware",
    "pack_q2",
    "pack_q4",
    "pack_ternary",
    "roundtrip_max_error",
    "simulate_overlap",
    "speculative_generate",
    "speculative_tree_generate",
    "unpack_q2",
    "unpack_q4",
    "unpack_ternary",
    "validate_policy",
]
