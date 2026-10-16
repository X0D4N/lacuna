"""Hardware auto-detection and tier policy (stdlib + numpy only).

SUPPORTED PLATFORMS: Linux (Debian/Arch-based) and Windows 11 only.
There are no macOS code paths, docs, or claims anywhere in this project.

- Linux: RAM from /proc/meminfo.
- Windows 11: RAM from kernel32.GlobalMemoryStatusEx via ctypes.
- nvidia-smi (guarded subprocess) on both; safe fallback when absent.
- Any other OS: a clear warning is emitted on import; detection falls back to
  documented defaults and pure-python paths (simulator, quant, oracle) still
  work. Nothing crashes.

Budget policy:
- RAM hot-set budget: min(6 GB, 75% of total RAM) — the OS and other apps keep
  the rest; the pager never pages the machine into swap.
- VRAM working set: 3.2 GB on a >=4 GB card, else 80% of detected VRAM.
- NVMe sequential bandwidth: default 3.5 GB/s, overridable (measured wins).
"""

from __future__ import annotations

import ctypes
import re
import shutil
import subprocess
import sys
import warnings
from dataclasses import dataclass

SUPPORTED_PLATFORMS = ("linux", "win32")


def _platform_check() -> None:
    if not sys.platform.startswith(SUPPORTED_PLATFORMS):
        warnings.warn(
            f"lacuna supports Linux and Windows 11 only; running on "
            f"{sys.platform}. Hardware detection falls back to defaults and "
            f"only pure-python paths (simulator, quant, oracle) are expected "
            f"to work.",
            RuntimeWarning,
            stacklevel=3,
        )


_platform_check()


@dataclass(frozen=True)
class HardwareTier:
    ram_total_gb: float
    ram_hot_budget_gb: float
    vram_gb: float
    vram_working_gb: float
    nvme_bw_gbs: float
    page_mb: int
    spec_k: int
    has_gpu: bool

    def summary(self) -> str:
        return (
            f"RAM {self.ram_total_gb:.1f} GB (hot budget {self.ram_hot_budget_gb:.1f} GB) | "
            f"VRAM {self.vram_gb:.1f} GB (working {self.vram_working_gb:.1f} GB) | "
            f"NVMe {self.nvme_bw_gbs:.1f} GB/s | page {self.page_mb} MB | K={self.spec_k}"
        )


def _read_ram_linux() -> float | None:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(re.findall(r"\d+", line)[0])
                    return kb / 1024 / 1024
    except (OSError, IndexError, ValueError):
        return None
    return None


def _read_ram_windows() -> float | None:
    try:
        kernel32 = ctypes.windll.kernel32

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return stat.ullTotalPhys / 1024**3
    except (OSError, AttributeError, ValueError):
        return None
    return None


def _read_ram_gb() -> float:
    if sys.platform.startswith("linux"):
        v = _read_ram_linux()
    elif sys.platform.startswith("win32"):
        v = _read_ram_windows()
    else:
        v = None
    if v is None or v <= 0:
        # documented safe fallback: assume the target-class machine
        return 8.0
    return v


def _read_vram_gb() -> tuple[float, bool]:
    """Returns (vram_gb, has_gpu). Never raises; 0.0/False when no NVIDIA GPU."""
    if shutil.which("nvidia-smi") is None:
        return 0.0, False
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            vals = [float(x) for x in re.findall(r"[\d.]+", out.stdout)]
            if vals:
                return max(vals) / 1024, True
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return 0.0, False


def detect_hardware(
    nvme_bw_gbs: float = 3.5,
    page_mb: int = 2,
    spec_k: int = 4,
) -> HardwareTier:
    """Detect the machine and return the Lacuna tier policy.

    All budgets are caps, not targets: on weaker machines every value degrades
    gracefully instead of failing.
    """
    ram = _read_ram_gb()
    vram, has_gpu = _read_vram_gb()
    ram_hot = min(6.0, 0.75 * ram)
    vram_working = 3.2 if vram >= 4.0 else 0.8 * vram
    return HardwareTier(
        ram_total_gb=round(ram, 2),
        ram_hot_budget_gb=round(ram_hot, 2),
        vram_gb=round(vram, 2),
        vram_working_gb=round(vram_working, 2),
        nvme_bw_gbs=float(nvme_bw_gbs),
        page_mb=int(page_mb),
        spec_k=int(spec_k),
        has_gpu=has_gpu,
    )
