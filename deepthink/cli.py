"""
Console entry points for open-deepthink.

Usage::

    deepthink                 # start web UI (default)
    deepthink serve           # same
    deepthink qnn  --prompt "…" [--debug]
    deepthink qdad --prompt "…" [--debug]
    deepthink version
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys


def _print_log(msg: str) -> None:
    print(msg, file=sys.stderr)


def _build_llm(args: argparse.Namespace):
    if getattr(args, "debug", False):
        try:
            from app import CoderMockLLM

            return CoderMockLLM()
        except Exception:
            from langchain_core.runnables import Runnable

            class _Stub(Runnable):
                def invoke(self, input_data, config=None, **kwargs):
                    return asyncio.get_event_loop().run_until_complete(
                        self.ainvoke(input_data, config=config, **kwargs)
                    )

                async def ainvoke(self, input_data, config=None, **kwargs):
                    text = str(input_data).lower()
                    if "noun" in text or "verb" in text:
                        return json.dumps(
                            {
                                "nouns": ["canvas", "ink", "lamp", "desk"],
                                "verbs": ["write", "glow", "focus", "rest"],
                            }
                        )
                    if "complexity" in text:
                        return json.dumps(
                            {
                                "complexity_score": 4,
                                "recommended_layers": 2,
                                "recommended_width": 2,
                                "recommended_epochs": 1,
                                "reasoning": "debug",
                            }
                        )
                    if "solution-space" in text or "polisher" in text or "synthesizer" in text:
                        return "## Solution-Space Report\nDebug stub strategy with falsifiers."
                    return json.dumps(
                        {
                            "proposed_solution": "Debug stub solution.",
                            "reasoning": "debug",
                        }
                    )

            return _Stub()

    from deepthink.providers import create_llm

    return create_llm(
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        base_url=getattr(args, "base_url", None),
    )


def cmd_serve(_args: argparse.Namespace) -> int:
    from deepthink.__main__ import main as serve_main

    serve_main()
    return 0


def cmd_version(_args: argparse.Namespace) -> int:
    from deepthink import __release_name__, __version__

    print(f"open-deepthink {__version__} ({__release_name__})")
    return 0


def cmd_qnn(args: argparse.Namespace) -> int:
    from deepthink import default_qnn_params, run_qnn

    params = default_qnn_params()
    if args.layers:
        params["qnn_mode"] = "manual"
        params["manual_layers"] = args.layers
        params["manual_width"] = args.width or params["manual_width"]
    if args.epochs:
        params["num_epochs"] = args.epochs

    llm = _build_llm(args)

    async def _run():
        return await run_qnn(
            llm,
            args.prompt,
            params=params,
            log=_print_log if args.verbose else None,
        )

    result = asyncio.run(_run())
    text = result.get("proposed_solution") or result.get("final_solution") or result
    if isinstance(text, dict):
        text = text.get("proposed_solution") or json.dumps(text, indent=2)
    print(text)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Wrote {args.json_out}", file=sys.stderr)
    return 0


def cmd_qdad(args: argparse.Namespace) -> int:
    from deepthink import default_qdad_params, run_qdad

    params = default_qdad_params()
    if args.n:
        params["grid_size"] = params["n"] = args.n
    if args.steps:
        params["denoising_steps"] = args.steps
    if args.temperature is not None:
        params["temperature_scale"] = args.temperature

    llm = _build_llm(args)

    async def _run():
        return await run_qdad(
            llm,
            args.prompt,
            params=params,
            log=_print_log if args.verbose else None,
        )

    result = asyncio.run(_run())
    # QDAD final_solution often nests the build prompt
    text = result
    if isinstance(result, dict):
        text = (
            result.get("app_build_prompt")
            or result.get("proposed_solution")
            or result.get("final_solution")
            or result
        )
        if isinstance(text, dict):
            text = (
                text.get("app_build_prompt")
                or text.get("proposed_solution")
                or json.dumps(text, indent=2)
            )
    print(text if isinstance(text, str) else json.dumps(text, indent=2))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Wrote {args.json_out}", file=sys.stderr)
    return 0


def _add_provider_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--provider",
        choices=("openrouter", "llamacpp"),
        default=None,
        help="LLM provider (default: settings / openrouter)",
    )
    p.add_argument("--model", default=None, help="Model id override")
    p.add_argument("--api-key", default=None, help="API key (or OPENROUTER_API_KEY)")
    p.add_argument(
        "--base-url",
        default=None,
        help="Base URL override (llama.cpp or OpenRouter)",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Use mock LLM (no API key / network)",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print pipeline logs to stderr",
    )
    p.add_argument(
        "--json-out",
        default=None,
        metavar="PATH",
        help="Write full result JSON to PATH",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deepthink",
        description=(
            "open-deepthink — QNN, QDAD, and Knowledge Distillation as a library + optional web UI"
        ),
    )
    sub = parser.add_subparsers(dest="command")

    p_serve = sub.add_parser("serve", help="Start the FastAPI web UI")
    p_serve.set_defaults(func=cmd_serve)

    p_ver = sub.add_parser("version", help="Print package version")
    p_ver.set_defaults(func=cmd_version)

    p_qnn = sub.add_parser("qnn", help="Run Qualitative Neural Network (library)")
    p_qnn.add_argument("--prompt", "-p", required=True, help="Problem / impasse brief")
    p_qnn.add_argument("--layers", type=int, default=None)
    p_qnn.add_argument("--width", type=int, default=None)
    p_qnn.add_argument("--epochs", type=int, default=None)
    _add_provider_args(p_qnn)
    p_qnn.set_defaults(func=cmd_qnn)

    p_qdad = sub.add_parser("qdad", help="Run Qualitative Diffusion App Designer")
    p_qdad.add_argument("--prompt", "-p", required=True, help="App vibe / Midjourney-style prompt")
    p_qdad.add_argument("--n", type=int, default=None, help="Grid size N (N×N features)")
    p_qdad.add_argument("--steps", type=int, default=None, help="Denoising steps")
    p_qdad.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Noise temperature scale",
    )
    _add_provider_args(p_qdad)
    p_qdad.set_defaults(func=cmd_qdad)

    return parser


def main(argv: list[str] | None = None) -> None:
    """``open-deepthink`` / ``deepthink`` console scripts."""
    argv = list(sys.argv[1:] if argv is None else argv)
    # Bare invocation → serve web UI (backward compatible with python -m deepthink)
    if not argv:
        cmd_serve(argparse.Namespace())
        return

    parser = build_parser()
    # Allow `deepthink --help` and unknown top-level flags → serve is default only when empty
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        sys.exit(0)
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        sys.exit(0)
    code = func(args)
    if code:
        sys.exit(code)


if __name__ == "__main__":
    main()
