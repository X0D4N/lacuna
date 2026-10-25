#!/usr/bin/env python3
"""Kaggle GPU-kernel test for Lacuna's GPU executor core.

What this measures (and what it does not): single-layer parity and
latency on a real GPU. It does NOT measure full-model generation —
`lacuna/gpu_executor.py` is single-layer by design, and token-vs-CPU
generation parity becomes runnable only after Engine wiring. Any tok/s
figure derived here from per-layer latency is an extrapolation and is
labeled as such in the output JSON.

Launch (parent runs these; this script never launches itself):
  1. On a Kaggle GPU kernel (or any CUDA box), copy the whole lacuna
     repo next to this script — the script adds the repo root to
     sys.path itself, so `python kaggle_gpu/gpu_executor_test.py`
     works from anywhere as long as the repo tree is intact.
  2. Baseline run (no installs, reports exactly what it finds):
       python kaggle_gpu/gpu_executor_test.py --out gpu_results.json
  3. If torch is absent and you want the script to try installing it:
       python kaggle_gpu/gpu_executor_test.py --install-torch
     (pip install torch from the cu128 wheel index; ~2.5 GB, minutes.
     Failure is reported, never hidden.)
  4. If GPUs are visible to nvidia-smi but torch reports no CUDA
     (the known Kaggle libcuda gap — see docs/GPU_EXECUTOR_DESIGN.md
     and REAL_WEIGHTS_104B_GPU.md), the script prints the diagnostic.
     To apply the documented driver-runfile fix, pass --fix-libcuda
     (needs root + network; every command is printed first).
  5. Real-weight layer parity (optional, needs a model file on the
     kernel; the .lac is converted if missing, like PagedQwen2 does):
       python kaggle_gpu/gpu_executor_test.py --gguf /tmp/model.gguf
  6. Full 110B-scale numpy parity (slow: minutes of CPU GEMMs;
     default is latency-only at that scale):
       python kaggle_gpu/gpu_executor_test.py --parity-110

Exit codes: 0 = all measured criteria pass; 2 = no CUDA backend
available (after any requested install attempt); 1 = any measured
criterion fails or the run errors. A nonzero exit with a clear message
is a result, not something to paper over.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def _ensure_lacuna_on_path(repo_arg: str | None) -> Path:
    """Find the lacuna package: explicit --repo, else the repo root
    (parent of this script's directory), else whatever is importable."""
    if repo_arg:
        root = Path(repo_arg).expanduser().resolve()
        if (root / "lacuna" / "__init__.py").exists():
            sys.path.insert(0, str(root))
            return root
        raise SystemExit(f"--repo {root} does not contain lacuna/__init__.py")
    try:
        import lacuna  # noqa: F401
        return Path(lacuna.__file__).resolve().parent.parent
    except ImportError:
        pass
    root = Path(__file__).resolve().parent.parent
    if (root / "lacuna" / "__init__.py").exists():
        sys.path.insert(0, str(root))
        return root
    raise SystemExit("lacuna package not found: pass --repo <repo root>")


def _run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, (proc.stdout + proc.stderr)[-2000:]


def _maybe_install_torch() -> dict:
    print("[install-torch] attempting: pip install torch (cu128 index)")
    rc, out = _run([sys.executable, "-m", "pip", "install", "torch",
                    "--index-url", "https://download.pytorch.org/whl/cu128"])
    print(out[-1000:])
    return {"attempted": True, "returncode": rc, "ok": rc == 0}


def _maybe_fix_libcuda() -> dict:
    """Documented fix from REAL_WEIGHTS_104B_GPU.md, applied only when
    explicitly flagged. Driver version is read from the loaded module
    when available, not assumed."""
    import re

    version = "580.178.04"
    try:
        text = Path("/proc/driver/nvidia/version").read_text()
        m = re.search(r"(\d+\.\d+\.\d+)", text)
        if m:
            version = m.group(1)
    except OSError:
        pass
    url = (f"https://us.download.nvidia.com/XFree86/Linux-x86_64/"
           f"{version}/NVIDIA-Linux-x86_64-{version}.run")
    steps = [
        ["curl", "-L", "-o", "/tmp/nvidia.run", url],
        ["sh", "/tmp/nvidia.run", "--extract-only",
         "--target", "/tmp/nvidia-driver"],
        ["cp", f"/tmp/nvidia-driver/libcuda.so.{version}",
         "/usr/lib/x86_64-linux-gnu/"],
        ["ln", "-sf", f"libcuda.so.{version}",
         "/usr/lib/x86_64-linux-gnu/libcuda.so.1"],
        ["ln", "-sf", "libcuda.so.1",
         "/usr/lib/x86_64-linux-gnu/libcuda.so"],
        ["ldconfig"],
    ]
    for cmd in steps:
        print("[fix-libcuda] $", " ".join(cmd))
        rc, out = _run(cmd)
        print(out[-500:])
        if rc != 0:
            return {"attempted": True, "ok": False,
                    "failed_step": " ".join(cmd), "driver_version": version}
    return {"attempted": True, "ok": True, "driver_version": version}


def _synthetic_refs(rng: np.random.RandomState, cfg, q4k: bool):
    """WeightRefs for one synthetic layer. Small configs use float
    arrays; the 110B-scale config uses raw Q4_K bytes (also exercises
    the torch Q4_K kernel + upload path at scale)."""
    from lacuna.gpu_executor import GGML_Q4_K, WeightRef

    def ref_float(shape):
        arr = (rng.standard_normal(shape) * 0.05).astype(np.float32)
        return WeightRef(shape=tuple(shape), array=arr)

    def ref_q4k(shape):
        n = 1
        for d in shape:
            n *= int(d)
        if n % 256:
            raise ValueError("Q4_K shape numel must be a multiple of 256")
        raw = rng.randint(0, 256, size=n * 144 // 256,
                          dtype=np.uint8).tobytes()
        return WeightRef(shape=tuple(shape), raw=raw, ttype=GGML_Q4_K)

    mk = ref_q4k if q4k else ref_float
    h, f, nh, nkv = cfg.hidden, cfg.ffn, cfg.n_head, cfg.n_kv
    hd = cfg.head_dim
    return {
        "attn_norm.weight": mk((h,)),
        "attn_q.weight": mk((h, h)),
        "attn_q.bias": mk((h,)),
        "attn_k.weight": mk((nkv * hd, h)),
        "attn_k.bias": mk((nkv * hd,)),
        "attn_v.weight": mk((nkv * hd, h)),
        "attn_v.bias": mk((nkv * hd,)),
        "attn_output.weight": mk((h, h)),
        "ffn_norm.weight": mk((h,)),
        "ffn_gate.weight": mk((f, h)),
        "ffn_up.weight": mk((f, h)),
        "ffn_down.weight": mk((h, f)),
    }


def _time_layer(layer, x, refs, pos, cache, backend, reps: int):
    """Synchronized per-layer seconds (mean over reps, after warmup).
    Synchronization is the honest fence: without it, queued GPU work
    would be timed as zero."""
    layer.forward(x, refs, pos, {} if cache is None else {})
    times = []
    for _ in range(reps):
        backend.synchronize()
        t0 = time.perf_counter()
        layer.forward(x, refs, pos, {} if cache is None else {})
        backend.synchronize()
        times.append(time.perf_counter() - t0)
    return float(np.mean(times)), float(np.min(times))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=None)
    ap.add_argument("--gguf", default=None,
                    help="optional real model file for layer-0 parity")
    ap.add_argument("--install-torch", action="store_true")
    ap.add_argument("--fix-libcuda", action="store_true")
    ap.add_argument("--parity-110", action="store_true",
                    help="also run numpy parity at 110B scale (slow)")
    ap.add_argument("--out", default="gpu_executor_results.json")
    args = ap.parse_args()

    repo = _ensure_lacuna_on_path(args.repo)
    from lacuna import gpu_executor as ge
    from lacuna.gpu_executor import (GGML_Q4_K, GGML_Q8_0, GPUExecutor,
                                    GPULayer, LayerConfig, NumpyBackend,
                                    TorchBackend)

    results: dict = {"repo": str(repo), "steps": {}}

    # 1. Environment -------------------------------------------------
    env = ge.describe_environment()
    results["environment"] = env
    print("[env]", json.dumps(env, indent=2))
    if args.fix_libcuda:
        results["steps"]["fix_libcuda"] = _maybe_fix_libcuda()
        env = ge.describe_environment()
        results["environment_after_fix"] = env
    if args.install_torch and not env["torch_cuda"]:
        results["steps"]["install_torch"] = _maybe_install_torch()
        env = ge.describe_environment()
        results["environment_after_install"] = env
    if not env["torch_cuda"]:
        print("[FAIL] no CUDA torch backend available. If nvidia-smi "
              "sees GPUs but torch does not, re-run with --fix-libcuda "
              "(libcuda gap, see docs). If torch is absent, re-run "
              "with --install-torch.")
        Path(args.out).write_text(json.dumps(results, indent=2))
        return 2

    import torch

    gpu = TorchBackend("cuda")
    cpu = NumpyBackend()
    print(f"[backend] torch {torch.__version__} on "
          f"{torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB)")

    # 2. Dequant parity ----------------------------------------------
    rng = np.random.RandomState(1234)
    from lacuna.gguf import dequantize as gguf_dq

    dq_rows = []
    for ttype, block_bytes, per in ((GGML_Q8_0, 34, 32),
                                    (GGML_Q4_K, 144, 256)):
        raw = rng.randint(0, 256, size=(8, block_bytes),
                          dtype=np.uint8)
        raw[:, 0:2] = np.array([0x00, 0x3C], dtype=np.uint8)
        if ttype == GGML_Q4_K:
            raw[:, 2:4] = np.array([0x00, 0x3C], dtype=np.uint8)
        raw = raw.tobytes()
        got = gpu.to_numpy(gpu.dequantize(raw, ttype, 8 * per))
        want = gguf_dq(raw, ttype, 8 * per)
        diff = float(np.abs(got - want).max())
        dq_rows.append({"type": ttype, "max_abs_diff": diff,
                        "pass": diff <= 1e-6})
        print(f"[dequant] type={ttype} max_abs_diff={diff:.3e}")
    results["dequant_parity"] = dq_rows

    # 3. Layer parity + latency ---------------------------------------
    configs = {
        "tiny": (LayerConfig(hidden=16, ffn=32, n_head=2, n_kv=1,
                             eps=1e-5, rope_theta=10000.0), False, 5),
        "qwen05_scale": (LayerConfig(hidden=896, ffn=4864, n_head=14,
                                     n_kv=2, eps=1e-6), False, 3),
        "qwen110_scale": (LayerConfig(hidden=8192, ffn=49152, n_head=64,
                                      n_kv=8, eps=1e-6), True, 2),
    }
    layer_rows = []
    for name, (cfg, q4k, reps) in configs.items():
        r = np.random.RandomState(99)
        refs = _synthetic_refs(r, cfg, q4k)
        x = r.standard_normal((1, cfg.hidden)).astype(np.float32)
        pos = np.array([0], dtype=np.float32)
        torch.cuda.reset_peak_memory_stats()
        mean_s, min_s = _time_layer(GPULayer(gpu, cfg), x, refs, pos,
                                    None, gpu, reps)
        peak = torch.cuda.max_memory_allocated()
        row = {"config": name, "mean_seconds_per_layer": mean_s,
               "min_seconds_per_layer": min_s,
               "peak_vram_bytes": peak,
               "extrapolated_tok_per_s_80_layers": 1.0 / (80 * mean_s)
               if mean_s > 0 else None,
               "extrapolation_note": "80 layers x per-layer seconds; "
               "NOT a generation measurement"}
        # Parity vs numpy (110-scale only with --parity-110: slow CPU).
        if name != "qwen110_scale" or args.parity_110:
            out_gpu = GPULayer(gpu, cfg).forward(x, refs, pos, {})
            out_cpu = GPULayer(cpu, cfg).forward(x, refs, pos, {})
            diff = float(np.abs(out_gpu - out_cpu).max())
            row["parity_max_abs_diff_vs_numpy"] = diff
            row["parity_pass"] = diff <= 1e-4
            print(f"[{name}] layer {mean_s:.4f}s peak={peak / 2**30:.2f} GiB "
                  f"parity_diff={diff:.3e}")
        else:
            print(f"[{name}] layer {mean_s:.4f}s peak={peak / 2**30:.2f} GiB "
                  f"(numpy parity skipped; pass --parity-110)")
        layer_rows.append(row)
    results["layers"] = layer_rows

    # 4. Real-weight layer parity --------------------------------------
    if args.gguf:
        from lacuna.convert import convert as _convert
        from lacuna.gguf import parse_gguf, tensor_shape
        from lacuna.gpu_executor import read_raw_tensor_bytes
        from lacuna.pager import PagedWeightStore

        gguf_path = args.gguf
        lac_path = gguf_path + ".lac"
        md, infos, _ = parse_gguf(gguf_path)
        p = "qwen2."
        cfg = LayerConfig(
            hidden=int(md[p + "embedding_length"]),
            ffn=int(md[p + "feed_forward_length"]),
            n_head=int(md[p + "attention.head_count"]),
            n_kv=int(md[p + "attention.head_count_kv"]),
            eps=float(md[p + "attention.layer_norm_rms_epsilon"]),
            rope_theta=float(md.get(p + "rope.freq_base", 1000000.0)))
        if not Path(lac_path).exists():
            print(f"[gguf] converting {gguf_path} -> {lac_path}")
            _convert(gguf_path, lac_path, quiet=True)
        store = PagedWeightStore(lac_path,
                                 hot_budget_bytes=2 * 1024**3)
        by_name = {i["name"]: i for i in infos}
        base = "blk.0."
        keys = ["attn_norm.weight", "attn_q.weight", "attn_q.bias",
                "attn_k.weight", "attn_k.bias", "attn_v.weight",
                "attn_v.bias", "attn_output.weight", "ffn_norm.weight",
                "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"]
        refs = {}
        for suffix in keys:
            info = by_name.get(base + suffix)
            if info is None:
                continue
            raw = read_raw_tensor_bytes(store, info)
            from lacuna.gpu_executor import WeightRef

            # tensor_shape is the CPU path's own logical shape
            # ((n_out, n_in) for 2-D) — same shape the CPU executor
            # reshapes to, so GPU/CPU parity compares like with like.
            refs[suffix] = WeightRef(shape=tensor_shape(info["dims"]),
                                     raw=raw, ttype=int(info["type"]))
        store.close()
        rng = np.random.RandomState(5)
        x = rng.standard_normal((1, cfg.hidden)).astype(np.float32)
        pos = np.array([0], dtype=np.float32)
        out_gpu = GPULayer(gpu, cfg).forward(x, refs, pos, {})
        out_cpu = GPULayer(cpu, cfg).forward(x, refs, pos, {})
        diff = float(np.abs(out_gpu - out_cpu).max())
        results["real_weight_layer0"] = {
            "gguf": gguf_path, "max_abs_diff": diff,
            "pass": diff <= 1e-4}
        print(f"[gguf] layer-0 real-weight parity max_abs_diff={diff:.3e}")

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"[done] wrote {args.out}")

    failed = [r for r in dq_rows if not r["pass"]]
    failed += [r for r in layer_rows if r.get("parity_pass") is False]
    if args.gguf and not results["real_weight_layer0"]["pass"]:
        failed.append(results["real_weight_layer0"])
    if failed:
        print(f"[FAIL] {len(failed)} criterion/criteria failed")
        return 1
    print("[PASS] all measured criteria pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
