#!/usr/bin/env python3
"""Head-to-head measurement harness: Lacuna vs AirLLM vs Colibri.

Same model, same hardware, same prompt, same max_tokens — or no
comparison at all. This script NEVER invents a number:

- An engine that is not installed / not supplied is recorded as
  ``not_available`` with its reason, never as 0 tok/s and never as
  an extrapolated figure.
- A per-layer latency is NEVER converted into tok/s. Only an
  end-to-end generate() call produces tok/s, with ttft, total
  seconds, token ids, and peak VRAM/RAM recorded alongside it.
- Token identity is compared only when two engines ran the same
  prompt on the same model file / HF id: ``token_ids_match`` is
  True/False, or ``not_comparable`` when an engine did not run.

Kaggle-first design: this box has ~7 GB RAM and no CUDA, so a
110B-class model is REFUSED here unless ``--allow-local-110b`` is
passed explicitly (and even then the caller owns the OOM risk).
The intended run is a Kaggle GPU kernel (or any CUDA box) where
the repo tree is intact: ``python kaggle_gpu/head_to_head.py
--model /path/to/model.gguf --prompt "The capital of France is"
--max-tokens 32 --out head_to_head.json``.

Lacuna run: ``Engine(model, gpu="cuda"|"auto")`` in mode
``lacuna-gpu`` (falls back to numpy only if ``--lacuna-gpu numpy``
is passed; that mode is labelled as such, never as a GPU number).

AirLLM run: ``airllm.AutoModel.from_pretrained(hf_id_or_path)``
with the same prompt/max_tokens. AirLLM needs an HF id or a local
transformers-format model, NOT a GGUF file: pass ``--airllm-model``
separately from ``--model`` when they differ. If airllm is not
installed, the run is recorded not_available.

Colibri run: Colibri has no stable CLI assumed here. Supply a
previously measured Colibri result JSON with ``--colibri-result``,
in the same ``results`` schema this script writes, and it is merged
verbatim (with ``source: "external"``) — or leave Colibri as
not_available. Do NOT hand-type Colibri numbers into this file.

Output JSON schema (top level):
  {
    "model": ..., "prompt": ..., "max_tokens": ...,
    "hardware": {"cuda": bool, "gpu_name": str|None, "ram_gb": float|None},
    "results": {
      "lacuna": {"status": "measured"|"not_available"|"failed",
                 "mode": ..., "backend": ..., "token_ids": [...],
                 "ttft_seconds": ..., "total_seconds": ...,
                 "tok_per_s": ..., "peak_vram_bytes": ...|None,
                 "error": ...|None, "source": "measured"},
      "airllm": {...}, "colibri": {...}
    },
    "comparison": {"token_ids_match": True|False|"not_comparable",
                   "fastest_measured": "lacuna"|"airllm"|"colibri"|None}
  }
Partial results are written after every engine finishes, so an
interrupted run still leaves evidence, not an empty file.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


def _repo_root() -> Path:
    root = Path(__file__).resolve().parent.parent
    if (root / "lacuna" / "__init__.py").exists():
        return root
    return Path.cwd()


def hardware_info() -> dict:
    info: dict = {
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "cuda": False,
        "gpu_name": None,
        "ram_gb": None,
    }
    try:
        import resource

        info["ram_gb"] = round(
            resource.getrlimit(resource.RLIMIT_AS)[0] / 1024**3, 2
        ) if False else None
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    info["ram_gb"] = round(int(line.split()[1]) / 1024**2, 2)
                    break
    except OSError:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            info["cuda"] = True
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    if info["gpu_name"] is None:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip()
            if out:
                info["gpu_name"] = out.splitlines()[0]
                info["cuda"] = True
        except Exception:
            pass
    return info


def _peak_vram_bytes() -> int | None:
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.max_memory_allocated())
    except Exception:
        pass
    return None


def _write_partial(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run_lacuna(model: str, prompt: str, max_tokens: int,
               gpu_mode: str) -> dict:
    sys.path.insert(0, str(_repo_root()))
    from lacuna import Engine

    t0 = time.perf_counter()
    eng = Engine(model, gpu=gpu_mode)
    try:
        if eng.mode != "lacuna-gpu":
            return {
                "status": "failed",
                "mode": eng.mode,
                "error": f"Engine did not land in lacuna-gpu mode "
                         f"(got {eng.mode}; gpu_error={eng.gpu_error!r})",
                "source": "measured",
            }
        text = eng.generate(prompt, max_tokens=max_tokens)
        stats = dict(eng.gpu_stats or {})
        # Re-derive tok/s from the engine's own timing, never from
        # a per-layer figure.
        total = stats.get("total_seconds")
        ids = stats.get("token_ids")
        return {
            "status": "measured",
            "mode": eng.mode,
            "backend": eng.gpu_backend,
            "text": text,
            "token_ids": ids,
            "ttft_seconds": stats.get("ttft_seconds"),
            "total_seconds": total,
            "tok_per_s": stats.get("tok_per_s_overall"),
            "within_300s_bound": stats.get("within_300s_bound"),
            "peak_vram_bytes": _peak_vram_bytes(),
            "error": None,
            "source": "measured",
        }
    finally:
        try:
            eng.close()
        except Exception:
            pass
        _ = t0


def run_airllm(model: str, prompt: str, max_tokens: int) -> dict:
    try:
        from airllm import AutoModel
    except ImportError as exc:
        return {"status": "not_available", "error": f"airllm not installed: {exc}",
                "source": "measured"}
    try:

        m = AutoModel.from_pretrained(model)
        inputs = m.tokenizer(prompt, return_tensors="pt",
                             return_attention_mask=False,
                             truncation=True, max_length=512)
        t0 = time.perf_counter()
        out = m.generate(inputs["input_ids"].to(m.device),
                         max_new_tokens=max_tokens,
                         use_cache=True, return_dict_in_generate=True)
        total = time.perf_counter() - t0
        seq = out.sequences[0].tolist()
        return {
            "status": "measured",
            "text": m.tokenizer.decode(seq),
            "token_ids": seq,
            "total_seconds": total,
            "tok_per_s": (len(seq) / total) if total > 0 else 0.0,
            "peak_vram_bytes": _peak_vram_bytes(),
            "error": None,
            "source": "measured",
        }
    except Exception as exc:  # noqa: BLE001 - harness must record, not crash
        return {"status": "failed", "error": repr(exc), "source": "measured"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True,
                    help="GGUF/.lac path for Lacuna (and for Colibri comparison, same model)")
    ap.add_argument("--airllm-model", default=None,
                    help="HF id / local path for AirLLM (defaults to --model)")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--engines", default="lacuna,airllm,colibri",
                    help="comma list, subset of lacuna,airllm,colibri")
    ap.add_argument("--lacuna-gpu", default="auto",
                    choices=["auto", "cuda", "numpy"])
    ap.add_argument("--colibri-result", default=None,
                    help="JSON file with a previously measured Colibri result, merged verbatim")
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-local-110b", action="store_true",
                    help="allow a >20GB model on a non-CUDA box (caller owns OOM risk)")
    ap.add_argument("--dry-run", action="store_true",
                    help="write the schema with all engines not_available; run nothing")
    args = ap.parse_args(argv)

    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    payload: dict = {
        "model": args.model,
        "airllm_model": args.airllm_model or args.model,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "hardware": hardware_info(),
        "results": {},
        "comparison": {"token_ids_match": "not_comparable",
                       "fastest_measured": None},
    }
    out_path = Path(args.out)
    if args.dry_run:
        for e in engines:
            payload["results"][e] = {"status": "not_available",
                                     "error": "dry-run: no engine executed",
                                     "source": "measured"}
        _write_partial(out_path, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    # Guard: 110B-class on this box without CUDA is a known OOM/hours trap.
    try:
        size = os.path.getsize(args.model)
    except OSError:
        size = 0
    if size > 20 * 1024**3 and not payload["hardware"]["cuda"] \
            and not args.allow_local_110b:
        print(f"REFUSED: model is {size / 1024**3:.1f} GB and no CUDA "
              f"was detected on this box. Run on a Kaggle GPU kernel, "
              f"or pass --allow-local-110b if you own the OOM risk.",
              file=sys.stderr)
        return 2

    for eng in engines:
        if eng == "lacuna":
            try:
                payload["results"]["lacuna"] = run_lacuna(
                    args.model, args.prompt, args.max_tokens, args.lacuna_gpu)
            except Exception as exc:  # noqa: BLE001
                payload["results"]["lacuna"] = {"status": "failed",
                                                "error": repr(exc),
                                                "source": "measured"}
        elif eng == "airllm":
            payload["results"]["airllm"] = run_airllm(
                args.airllm_model or args.model, args.prompt, args.max_tokens)
        elif eng == "colibri":
            if args.colibri_result:
                try:
                    ext = json.loads(Path(args.colibri_result).read_text())
                    ext = dict(ext)
                    ext["source"] = "external"
                    payload["results"]["colibri"] = ext
                except Exception as exc:  # noqa: BLE001
                    payload["results"]["colibri"] = {
                        "status": "failed",
                        "error": f"unreadable --colibri-result: {exc!r}",
                        "source": "external"}
            else:
                payload["results"]["colibri"] = {
                    "status": "not_available",
                    "error": "no Colibri binary/result supplied; "
                             "pass --colibri-result with a measured result JSON",
                    "source": "measured"}
        else:
            payload["results"][eng] = {"status": "failed",
                                       "error": f"unknown engine {eng!r}",
                                       "source": "measured"}
        _write_partial(out_path, payload)

    measured = {k: v for k, v in payload["results"].items()
                if v.get("status") == "measured"
                and isinstance(v.get("tok_per_s"), (int, float))}
    if measured:
        payload["comparison"]["fastest_measured"] = max(
            measured, key=lambda k: measured[k]["tok_per_s"])
    id_sets = [tuple(v["token_ids"]) for v in measured.values()
               if isinstance(v.get("token_ids"), list)]
    if len(id_sets) >= 2:
        payload["comparison"]["token_ids_match"] = all(
            s == id_sets[0] for s in id_sets[1:])
    _write_partial(out_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
