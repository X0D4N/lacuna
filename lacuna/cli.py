"""Zero-setup CLI: ``lacuna run model.gguf`` does everything.

Subcommands:
  run       auto-detect hardware, auto-convert GGUF->.lac if needed, generate
  serve     OpenAI-compatible HTTP server (stdlib http.server)
  convert   GGUF -> .lac explicitly
  mcp       MCP server over stdio (JSON-RPC) for agent integration
  bench     run the hardware simulator and print the honest tok/s table
  simulate  alias of bench
  oracle-recipe
            train/evaluate a per-family activation oracle (synthetic or
            captured calibration data); see docs/ORACLE_RECIPES.md
"""

from __future__ import annotations

import argparse
import sys

from .hardware import detect_hardware


def _cmd_run(a) -> int:
    from .engine import Engine

    hw = detect_hardware()
    print(f"lacuna v{__import__('lacuna').__version__} | {hw.summary()}", flush=True)
    eng = Engine(a.model, hardware=hw, exact=a.exact)
    print(f"mode: {eng.mode} | exact: {eng.exact} | precision: {eng.precision_policy}", flush=True)
    prompt = a.prompt or sys.stdin.read().strip() or "Hello"
    print(eng.generate(prompt, max_tokens=a.max_tokens), flush=True)
    eng.close()
    return 0


def _cmd_serve(a) -> int:
    from .engine import Engine
    from .server import serve

    eng = Engine(a.model)
    print(f"serving OpenAI-compatible API on :{a.port} (mode={eng.mode})", flush=True)
    serve(eng, host=a.host, port=a.port)
    return 0


def _cmd_convert(a) -> int:
    from .convert import main as convert_main

    return convert_main(
        [a.input, a.output, "--page-mb", str(a.page_mb),
         "--precision-policy", a.precision_policy]
    )


def _cmd_mcp(a) -> int:
    from .mcp_server import main as mcp_main

    return mcp_main()


def _cmd_bench(a) -> int:
    from simulator.simulate import main as sim_main

    return sim_main([])

def _cmd_perplexity(a) -> int:
    import sys as _sys

    _sys.path.insert(0, ".")
    from benchmarks.perplexity.measure import main as ppl_main

    return ppl_main(["--retrain"] if a.retrain else [])


def _cmd_oracle_recipe(a) -> int:
    from .recipes import (
        FAMILIES,
        format_report,
        gate_report,
        get_family,
        list_families,
        load_dataset,
        quick_spec,
        save_dataset,
        synthetic_calibration,
        train_family_oracle,
    )

    if a.list_families:
        print(f"{'family':>8} {'kind':>6} {'layers':>6} {'hidden':>6} "
              f"{'inter':>7} {'experts':>7}")
        for name in list_families():
            s = FAMILIES[name]
            exp = f"{s.n_experts}x top-{s.top_k}" if s.kind == "moe" else "--"
            print(f"{name:>8} {s.kind:>6} {s.n_layers:>6} {s.hidden_dim:>6} "
                  f"{s.intermediate:>7} {exp:>7}  {s.reference_model}")
        return 0

    spec = quick_spec(get_family(a.family)) if a.quick else get_family(a.family)
    if a.real_data:
        dataset = load_dataset(a.real_data)
        label = "REAL captured data (measured)"
    else:
        dataset = synthetic_calibration(
            spec, n_tokens=a.tokens, density=a.density, seed=a.seed)
        label = ("QUICK (scaled-down pipeline check)"
                 if a.quick else "SYNTHETIC (pipeline validation only)")
    if a.save_data:
        save_dataset(a.save_data, dataset)
        print(f"calibration dataset -> {a.save_data} [{label}]", flush=True)

    print(f"family={spec.name} kind={spec.kind} layers={spec.n_layers} "
          f"hidden={spec.hidden_dim} units/layer={dataset['layers'][0]['Y'].shape[1]} "
          f"[{label}]", flush=True)
    out = train_family_oracle(spec, dataset, epochs=a.epochs, lr=a.lr,
                              batch=a.batch)
    print(format_report(out["report"]), flush=True)
    ok, failures = gate_report(out["report"])
    if a.save:
        out["ffn"].save(a.save)
        print(f"FFN oracle -> {a.save}", flush=True)
        if out["router"] is not None:
            rp = a.save + ".router.npz"
            out["router"].save(rp)
            print(f"router oracle -> {rp}", flush=True)
    if ok:
        print("GATE: PASS -- every layer clears recall/density/union bars",
              flush=True)
        return 0
    print("GATE: FAIL", flush=True)
    for f in failures:
        print(f"  - {f}", flush=True)
    return 1


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="lacuna", description="Lookahead sparse-paged LLM inference")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="generate text from a model (one command)")
    p.add_argument("model", help=".gguf or .lac file (or 'mock')")
    p.add_argument("prompt", nargs="?", default=None)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument(
        "--exact", dest="exact", action="store_true", default=True,
        help="exact mode: output identical to the dense baseline by "
        "construction; oracle only schedules prefetches (default)",
    )
    p.add_argument(
        "--extreme", dest="exact", action="store_false",
        help="extreme mode: adaptive Q4/Q2/ternary precision, fastest, "
        "documented quality/speed tradeoff",
    )
    p.set_defaults(fn=_cmd_run)

    p = sub.add_parser("serve", help="OpenAI-compatible HTTP server")
    p.add_argument("model", nargs="?", default="mock")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.set_defaults(fn=_cmd_serve)

    p = sub.add_parser("convert", help="convert GGUF -> .lac")
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--page-mb", type=int, default=2)
    p.add_argument(
        "--precision-policy", choices=["exact", "extreme"], default="exact",
        help="exact: bit-identical source bytes (default); extreme: adaptive "
        "Q4/Q2/ternary tiering",
    )
    p.set_defaults(fn=_cmd_convert)

    p = sub.add_parser("mcp", help="MCP server over stdio")
    p.set_defaults(fn=_cmd_mcp)

    p = sub.add_parser("bench", help="simulated hardware benchmark table")
    p.set_defaults(fn=_cmd_bench)
    p = sub.add_parser("simulate", help="alias of bench")
    p.set_defaults(fn=_cmd_bench)

    p = sub.add_parser(
        "oracle-recipe",
        help="train/evaluate a per-family activation oracle "
             "(see docs/ORACLE_RECIPES.md)",
    )
    p.add_argument("--family", choices=["qwen", "llama", "moe"], default="qwen")
    p.add_argument("--list-families", action="store_true",
                   help="print the family presets and exit")
    p.add_argument("--tokens", type=int, default=4096,
                   help="synthetic calibration tokens (ignored with --real-data)")
    p.add_argument("--density", type=float, default=0.10,
                   help="synthetic target density (ignored with --real-data)")
    p.add_argument("--epochs", type=int, default=None,
                   help="training epochs (default: per-family validated regime)")
    p.add_argument("--lr", type=float, default=None,
                   help="learning rate (default: per-family validated regime)")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true",
                   help="scaled-down dims for a fast pipeline check")
    p.add_argument("--real-data", default=None, metavar="CALIB.NPZ",
                   help="train from a captured .npz dataset instead of synthetic")
    p.add_argument("--save", default=None, metavar="ORACLE.NPZ",
                   help="save the trained FFN oracle (.npz)")
    p.add_argument("--save-data", default=None, metavar="CALIB.NPZ",
                   help="save the calibration dataset (.npz)")
    p.set_defaults(fn=_cmd_oracle_recipe)

    p = sub.add_parser(
        "perplexity",
        help="measured exact-vs-extreme quality cost on a tiny trained LM "
             "(see benchmarks/perplexity.md)",
    )
    p.add_argument("--retrain", action="store_true",
                   help="retrain the tiny LM even if a trained checkpoint exists")
    p.set_defaults(fn=_cmd_perplexity)
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
