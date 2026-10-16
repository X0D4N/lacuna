"""Zero-setup CLI: ``lacuna run model.gguf`` does everything.

Subcommands:
  run       auto-detect hardware, auto-convert GGUF->.lac if needed, generate
  serve     OpenAI-compatible HTTP server (stdlib http.server)
  convert   GGUF -> .lac explicitly
  mcp       MCP server over stdio (JSON-RPC) for agent integration
  bench     run the hardware simulator and print the honest tok/s table
  simulate  alias of bench
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
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
