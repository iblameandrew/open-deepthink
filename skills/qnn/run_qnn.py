#!/usr/bin/env python3
"""
/qnn skill runner — execute the Qualitative Neural Network pipeline in code.

Harnesses should run THIS script (or import the same API) instead of only
simulating the markdown procedure.

Examples
--------
# OpenRouter
python run_qnn.py --prompt "explore this deadlock" \\
  --provider openrouter --api-key $OPENROUTER_API_KEY \\
  --model stepfun/step-3.5-flash:free --qnn-mode auto

# Manual topology
python run_qnn.py --prompt "richer metrics for export" \\
  --qnn-mode manual --layers 3 --width 3 --epochs 2

# Local llama.cpp
python run_qnn.py --prompt "..." --provider llamacpp \\
  --base-url http://localhost:8080/v1 --model local-model

# Debug / no cost
python run_qnn.py --prompt "test" --debug

# Library import (from open-deepthink repo root or installed package)
#   from deepthink.qnn import run_qnn_pipeline, default_qnn_params
#   result = await run_qnn_pipeline(llm, prompt, params={...})
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Allow running from a skill zip: prefer sibling package, then monorepo root.
_HERE = Path(__file__).resolve().parent
_CANDIDATES = [
    _HERE / "lib",  # optional vendored package layout
    _HERE.parent.parent,  # skills/qnn → repo root
    Path(os.environ.get("OPEN_DEEPTHINK_ROOT", "")),
]
for _p in _CANDIDATES:
    if _p and (_p / "deepthink").is_dir():
        sys.path.insert(0, str(_p))
        break


def _build_llm(args):
    if args.debug:
        # Import app mock only when debug; fallback to a tiny stub
        try:
            sys.path.insert(0, str(_HERE.parent.parent))
            from app import CoderMockLLM  # type: ignore

            return CoderMockLLM()
        except Exception:
            from langchain_core.runnables import Runnable

            class _Stub(Runnable):
                async def ainvoke(self, input_data, config=None, **kwargs):
                    text = str(input_data).lower()
                    if "json" in text and "complexity" in text:
                        return json.dumps(
                            {
                                "complexity_score": 4,
                                "recommended_layers": 2,
                                "recommended_width": 2,
                                "recommended_epochs": 1,
                                "reasoning": "debug stub",
                            }
                        )
                    if "space-separated" in text or "seed" in text:
                        return "distill ownership latch invariant probe reframe entropy horizon"
                    if "node generator" in text or "guiding_words" in text:
                        return json.dumps(
                            {
                                "name": "Debug Expert",
                                "specialty": "Stub specialist",
                                "emoji": "🤖",
                                "guiding_words": "distill ownership",
                                "attributes": ["Analytical"],
                                "skills": ["probing"],
                                "system_prompt": "You are a debug QNN expert. Map strategies with falsifiers.",
                            }
                        )
                    if "solution-space" in text or "master synthesizer" in text:
                        return "## 1. Impasse\nDebug run.\n## 3. Strategies\n**Probe first** — log boundaries."
                    if "new_problem" in text or "re-framer" in text:
                        return json.dumps({"new_problem": "Harder challenge under concurrency."})
                    return json.dumps(
                        {
                            "original_problem": "debug",
                            "proposed_solution": "Mock strategy: instrument ownership boundaries.",
                            "reasoning": "debug mode",
                            "falsifiers": "logs show no interleaving",
                            "risks": "noise",
                            "skills_used": [],
                        }
                    )

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
        except Exception:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=args.model,
                openai_api_key="no-key",
                openai_api_base=args.base_url.rstrip("/"),
                temperature=0.7,
            )
        return ChatLlamaCpp(
            base_url=args.base_url.rstrip("/"),
            api_key="no-key-required",
            model=args.model,
            temperature=0.7,
            max_tokens=4096,
        )

    raise SystemExit(f"Unknown provider: {args.provider}")


def _params_from_args(args) -> dict:
    return {
        "qnn_mode": args.qnn_mode,
        "manual_layers": args.layers,
        "manual_width": args.width,
        "num_epochs": args.epochs,
        "vector_word_size": args.vector_word_size,
        "learning_rate": args.learning_rate,
        "attention_top_k": args.attention_top_k,
        "enable_self_attention": not args.no_attention,
    }


async def _amain(args) -> int:
    from deepthink.qnn import run_qnn_pipeline

    llm = _build_llm(args)
    params = _params_from_args(args)

    def log(msg: str):
        print(msg, file=sys.stderr)

    doc = ""
    if args.context_file:
        doc = Path(args.context_file).read_text(encoding="utf-8", errors="replace")

    result = await run_qnn_pipeline(
        llm,
        user_prompt=args.prompt,
        params=params,
        document_context=doc,
        log=log,
        session_id="skill-cli",
    )

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote {args.out}", file=sys.stderr)

    # Primary deliverable for the harness / user
    print(result.get("proposed_solution") or result.get("final_solution", {}).get("proposed_solution", ""))
    if args.json:
        print(json.dumps(result, indent=2))
    return 0


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run Qualitative Neural Network (portable /qnn skill code entrypoint)"
    )
    p.add_argument("--prompt", "-p", required=True, help="Impasse / feature brief")
    p.add_argument("--provider", choices=["openrouter", "llamacpp"], default="openrouter")
    p.add_argument("--api-key", default=None)
    p.add_argument("--model", default="stepfun/step-3.5-flash:free")
    p.add_argument("--base-url", default="http://localhost:8080/v1")
    p.add_argument("--qnn-mode", choices=["auto", "manual"], default="auto")
    p.add_argument("--layers", type=int, default=3, help="Manual L")
    p.add_argument("--width", type=int, default=3, help="Manual W")
    p.add_argument("--epochs", type=int, default=2, help="E (auto may override)")
    p.add_argument("--vector-word-size", type=int, default=6, dest="vector_word_size")
    p.add_argument("--learning-rate", type=float, default=0.5)
    p.add_argument("--attention-top-k", type=int, default=5)
    p.add_argument("--no-attention", action="store_true")
    p.add_argument("--context-file", default=None, help="Optional document context path")
    p.add_argument("--out", default=None, help="Write full JSON result to path")
    p.add_argument("--json", action="store_true", help="Also print full JSON to stdout")
    p.add_argument("--debug", action="store_true", help="CoderMockLLM / stub (no API cost)")
    args = p.parse_args()
    raise SystemExit(asyncio.run(_amain(args)))


if __name__ == "__main__":
    main()
