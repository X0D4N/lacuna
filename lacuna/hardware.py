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
import os
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


# --------------------------------------------------------------------------
# auto_tune(): zero-config tuning. Pure function of HardwareInfo so it is
# unit-testable without real hardware: pass a synthetic HardwareInfo.
#
# Policy (documented, never silent):
# - ram_cap  = min(6.0, 75% of RAM). Never exceeds 75% — the OS keeps the rest.
# - vram_cap = 3.2 GB on a >=4 GB card, else 80% of VRAM, 0.0 with no GPU.
# - page_size: 1 MiB on tiny boxes (<6 GB RAM, less waste), 2 MiB default,
#   4 MiB on big machines (>=32 GB RAM, fewer syscalls per GB streamed).
# - k_lookahead: 4 default; 6 on slow disks (<1 GB/s, more lookahead hides
#   latency); 3 with no GPU (less speculation without device overlap).
# - precision_policy: ALWAYS "exact". auto_tune never silently trades quality
#   for speed — the user opts into "extreme" explicitly via CLI/API.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class HardwareInfo:
    """Everything auto_tune() needs. Build by hand for tests, or via
    detect_hardware_info() on a real machine."""

    ram_total_gb: float
    vram_gb: float = 0.0
    has_gpu: bool = False
    nvme_bw_gbs: float = 3.5
    disk_is_nvme: bool = True


@dataclass(frozen=True)
class TuneConfig:
    ram_cap: float          # GB of RAM Lacuna may hold (<= 75% of total)
    vram_cap: float         # GB of VRAM Lacuna may hold
    page_size: int          # weight-page size in bytes
    k_lookahead: int        # speculation lookahead depth
    precision_policy: str   # always "exact" from auto_tune (see above)


def auto_tune(info: HardwareInfo | None = None) -> TuneConfig:
    """Return the recommended zero-config tuning for this hardware."""
    hw = info if info is not None else detect_hardware_info()
    ram = max(hw.ram_total_gb, 0.5)  # never divide by nonsense
    ram_cap = min(6.0, 0.75 * ram)
    if hw.has_gpu and hw.vram_gb >= 4.0:
        vram_cap = 3.2
    elif hw.has_gpu:
        vram_cap = 0.8 * max(hw.vram_gb, 0.0)
    else:
        vram_cap = 0.0
    if ram < 6.0:
        page_size = 1 * 1024 * 1024
    elif ram >= 32.0:
        page_size = 4 * 1024 * 1024
    else:
        page_size = 2 * 1024 * 1024
    if hw.nvme_bw_gbs < 1.0:
        k = 6
    elif not hw.has_gpu:
        k = 3
    else:
        k = 4
    return TuneConfig(
        ram_cap=round(ram_cap, 2),
        vram_cap=round(vram_cap, 2),
        page_size=page_size,
        k_lookahead=k,
        precision_policy="exact",
    )


def _disk_is_nvme_linux(path: str) -> bool:
    """Best-effort: True if the disk backing `path` is non-rotational.

    Never raises; on any doubt returns True (the NVMe default) so behavior
    matches the long-standing simulator assumption.
    """
    try:
        st = os.stat(path if os.path.exists(path) else os.path.expanduser("~"))
        major = os.major(st.st_dev)
        # find the block device name via /sys/dev/block/<major>:<minor>
        link = f"/sys/dev/block/{major}:{os.minor(st.st_dev)}"
        dev = os.path.basename(os.readlink(link))
        # strip partition suffix -> base device (nvme0n1p2 -> nvme0n1, sda1 -> sda)
        base = re.sub(r"(nvme\d+n\d+)p\d+$", r"\1", dev)
        base = re.sub(r"^(sd[a-z]|hd[a-z]|vd[a-z]|mmcblk\d+|nvme\d+n\d+)p?\d*$",
                      r"\1", base)
        with open(f"/sys/block/{base}/queue/rotational") as f:
            return f.read().strip() == "0"
    except (OSError, ValueError):
        return True


def detect_hardware_info(cache_dir: str | None = None) -> HardwareInfo:
    """Detect real hardware into a HardwareInfo. Never raises."""
    ram = _read_ram_gb()
    vram, has_gpu = _read_vram_gb()
    if sys.platform.startswith("linux"):
        is_nvme = _disk_is_nvme_linux(
            cache_dir or os.path.expanduser("~/.cache/lacuna"))
        bw = 3.5 if is_nvme else 0.55  # NVMe default, SATA fallback
    else:
        # Windows 11: no cheap reliable disk-type probe via stdlib; keep the
        # documented default and let measured runs override it.
        is_nvme, bw = True, 3.5
    return HardwareInfo(
        ram_total_gb=ram,
        vram_gb=vram,
        has_gpu=has_gpu,
        nvme_bw_gbs=bw,
        disk_is_nvme=is_nvme,
    )
