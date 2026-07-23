#!/usr/bin/env python3
"""
/qdad skill runner — execute Qualitative Diffusion (App Slot Machine) in code.

Harnesses should run THIS script (or import the same API) instead of only
simulating the markdown procedure.

Examples
--------
python run_qdad.py --prompt "cozy night writing app, soft dark mode, offline-first" \\
  --provider openrouter --api-key $OPENROUTER_API_KEY --n 3 --steps 2

python run_qdad.py --prompt "habit garden" --n 4 --temperature-scale 1.3 \\
  --denoising-steps 3 --noun-verb-temperature 0.6

python run_qdad.py --prompt "test" --debug

# Library import
#   from deepthink.qdad import run_qdad_pipeline
#   result = await run_qdad_pipeline(llm, params={...}, user_prompt="...")
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_CANDIDATES = [
    _HERE / "lib",
    _HERE.parent.parent,
    Path(os.environ.get("OPEN_DEEPTHINK_ROOT", "")),
]
for _p in _CANDIDATES:
    if _p and (_p / "deepthink").is_dir():
        sys.path.insert(0, str(_p))
        break


def _build_llm(args):
    if args.debug:
        try:
            sys.path.insert(0, str(_HERE.parent.parent))
            from app import CoderMockLLM  # type: ignore

            return CoderMockLLM()
        except Exception:
            from langchain_core.runnables import Runnable

            class _Stub(Runnable):
                async def ainvoke(self, input_data, config=None, **kwargs):
                    text = str(input_data).lower()
                    if "foundation" in text or "distinct nouns" in text:
                        return json.dumps(
                            {
                                "nouns": ["canvas", "lantern", "notebook", "harbor"][:4],
                                "verbs": ["whisper", "weave", "anchor", "glow"][:4],
                            }
                        )
                    if "featureagent" in text or "forward diffusion" in text:
                        return (
                            "A wild mock feature: ambient focus rituals with offline capture."
                        )
                    if "criticagent" in text or "reverse diffusion" in text:
                        return (
                            "A refined mock feature: offline-first focus mode with soft glow."
                        )
                    if "synthesizer" in text or "app build prompt" in text:
                        return (
                            "# App Build Prompt\n\n## High-Level Vision\n"
                            "Cozy offline writing app.\n\n## Core Features\n1. Focus timer\n"
                        )
                    return "mock feature"

            return _Stub()

    if args.provider == "openrouter":
        from langchain_openai import ChatOpenAI

        key = args.api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("API_KEY")
        if not key:
            raise SystemExit("OpenRouter requires --api-key or OPENROUTER_API_KEY")
        return ChatOpenAI(
            model=args.model,
            openai_api_key=key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.7,
        )

    if args.provider == "llamacpp":
        try:
            from deepthink.models import ChatLlamaCpp

            return ChatLlamaCpp(
                base_url=args.base_url.rstrip("/"),
                api_key="no-key-required",
                model=args.model,
                temperature=0.7,
                max_tokens=4096,
            )
        except Exception:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=args.model,
                openai_api_key="no-key",
                openai_api_base=args.base_url.rstrip("/"),
                temperature=0.7,
            )

    raise SystemExit(f"Unknown provider: {args.provider}")


def _params_from_args(args) -> dict:
    return {
        "grid_size": args.n,
        "n": args.n,
        "temperature_scale": args.temperature_scale,
        "denoising_steps": args.denoising_steps,
        "noun_verb_temperature": args.noun_verb_temperature,
    }


async def _amain(args) -> int:
    from deepthink.qdad import run_qdad_pipeline

    llm = _build_llm(args)
    params = _params_from_args(args)

    def log(msg: str):
        print(msg, file=sys.stderr)

    doc = ""
    if args.context_file:
        doc = Path(args.context_file).read_text(encoding="utf-8", errors="replace")

    result = await run_qdad_pipeline(
        llm=llm,
        params=params,
        user_prompt=args.prompt,
        document_context=doc,
        log=log,
        session_id="skill-cli",
    )

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote {args.out}", file=sys.stderr)

    print(result.get("proposed_solution") or json.dumps(result, indent=2))
    if args.json:
        print(json.dumps(result, indent=2))
    return 0


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run Qualitative Diffusion / App Slot Machine (portable /qdad entrypoint)"
    )
    p.add_argument("--prompt", "-p", required=True, help="Midjourney-style app intent")
    p.add_argument("--provider", choices=["openrouter", "llamacpp"], default="openrouter")
    p.add_argument("--api-key", default=None)
    p.add_argument("--model", default="stepfun/step-3.5-flash:free")
    p.add_argument("--base-url", default="http://localhost:8080/v1")
    p.add_argument("--n", type=int, default=4, help="Grid size N (N×N agents)")
    p.add_argument(
        "--temperature-scale",
        type=float,
        default=1.3,
        dest="temperature_scale",
        help="Forward diffusion noise temperature",
    )
    p.add_argument(
        "--denoising-steps",
        type=int,
        default=3,
        dest="denoising_steps",
        help="Reverse diffusion rounds",
    )
    p.add_argument(
        "--noun-verb-temperature",
        type=float,
        default=0.6,
        dest="noun_verb_temperature",
        help="Foundation basis temperature",
    )
    p.add_argument("--context-file", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--json", action="store_true")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()
    raise SystemExit(asyncio.run(_amain(args)))


if __name__ == "__main__":
    main()
