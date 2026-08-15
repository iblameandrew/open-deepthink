#!/usr/bin/env python3
"""
QNN runner — materialized on-the-fly by the /qnn skill.
Discovers deepthink from the workspace tree; runs the QNN pipeline.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path


def _discover_deepthink() -> Path | None:
    """Walk cwd parents, env, and common clones for a deepthink package root."""
    env = os.environ.get("OPEN_DEEPTHINK_ROOT", "").strip()
    candidates = []
    if env:
        candidates.append(Path(env))
    here = Path.cwd().resolve()
    candidates.append(here)
    candidates.extend(here.parents)
    # Skill install dir → monorepo (…/skills/qnn → repo)
    skill_file = Path(__file__).resolve()
    candidates.append(skill_file.parent)
    candidates.extend(skill_file.parents)
    for c in candidates:
        if not c:
            continue
        if (c / "deepthink" / "qnn" / "pipeline.py").is_file():
            return c
        if (c / "open-deepthink" / "deepthink" / "qnn" / "pipeline.py").is_file():
            return c / "open-deepthink"
    return None


def _ensure_path() -> None:
    root = _discover_deepthink()
    if root is None:
        print(
            "ERROR: Could not find open-deepthink (deepthink/qnn). "
            "Run from a workspace that contains the repo, or set OPEN_DEEPTHINK_ROOT.",
            file=sys.stderr,
        )
        sys.exit(2)
    sys.path.insert(0, str(root))
    print(f"LOG: using deepthink root {root}", file=sys.stderr)


def _build_llm(args):
    if args.debug:
        try:
            from app import CoderMockLLM  # type: ignore

            return CoderMockLLM()
        except Exception:
            from langchain_core.runnables import Runnable

            class _Stub(Runnable):
                async def ainvoke(self, input_data, config=None, **kwargs):
                    t = str(input_data).lower()
                    if "complexity" in t and "json" in t:
                        return json.dumps(
                            {
                                "complexity_score": 4,
                                "recommended_layers": 2,
                                "recommended_width": 2,
                                "recommended_epochs": 1,
                                "reasoning": "debug",
                            }
                        )
                    if "seed" in t or "space-separated" in t:
                        return "distill ownership latch invariant probe reframe entropy horizon"
                    if "guiding_words" in t or "node generator" in t:
                        return json.dumps(
                            {
                                "name": "Debug Expert",
                                "specialty": "Stub",
                                "emoji": "🤖",
                                "guiding_words": "distill ownership",
                                "attributes": ["Analytical"],
                                "skills": ["probing"],
                                "system_prompt": "You are a debug QNN expert. Map strategies with falsifiers.",
                            }
                        )
                    if "synthesizer" in t or "solution-space" in t or "polisher" in t:
                        return (
                            "## 1. Impasse / Goal\nDebug QNN run.\n"
                            "## 3. Divergent Strategy Map\n**Probe first** — logs at ownership boundaries.\n"
                            "## 5. Recommended Next Steps\n1. Instrument 2. Minimal test 3. Implement after probe."
                        )
                    if "re-framer" in t or "new_problem" in t:
                        return json.dumps({"new_problem": "Harder challenge under concurrency."})
                    return json.dumps(
                        {
                            "original_problem": "debug",
                            "proposed_solution": "Instrument ownership boundaries with ordered logs.",
                            "reasoning": "debug",
                            "falsifiers": "no interleaving under load",
                            "risks": "noise",
                            "skills_used": [],
                        }
                    )

            return _Stub()

    if args.provider == "openrouter":
        from langchain_openai import ChatOpenAI

        key = args.api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("API_KEY")
        if not key:
            raise SystemExit("Need --api-key or OPENROUTER_API_KEY")
        return ChatOpenAI(
            model=args.model,
            openai_api_key=key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.7,
        )

    if args.provider == "llamacpp":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=args.model,
            openai_api_key="no-key",
            openai_api_base=args.base_url.rstrip("/"),
            temperature=0.7,
        )
    raise SystemExit(f"Unknown provider {args.provider}")


async def _main(args) -> int:
    _ensure_path()
    from deepthink.qnn import run_qnn_pipeline

    llm = _build_llm(args)
    params = {
        "qnn_mode": args.qnn_mode,
        "manual_layers": args.layers,
        "manual_width": args.width,
        "num_epochs": args.epochs,
        "vector_word_size": args.vector_word_size,
        "learning_rate": args.learning_rate,
        "attention_top_k": args.attention_top_k,
        "enable_self_attention": not args.no_attention,
    }
    doc = ""
    if args.context_file:
        doc = Path(args.context_file).read_text(encoding="utf-8", errors="replace")

    def log(msg: str):
        print(msg, file=sys.stderr)

    result = await run_qnn_pipeline(
        llm,
        user_prompt=args.prompt,
        params=params,
        document_context=doc,
        log=log,
        session_id="skill-on-the-fly",
    )
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote {args.out}", file=sys.stderr)
    print(result.get("proposed_solution") or "")
    if args.json:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="QNN pipeline (skill-materialized runner)")
    p.add_argument("--prompt", "-p", required=True)
    p.add_argument("--provider", choices=["openrouter", "llamacpp"], default="openrouter")
    p.add_argument("--api-key", default=None)
    p.add_argument("--model", default="stepfun/step-3.5-flash:free")
    p.add_argument("--base-url", default="http://localhost:8080/v1")
    p.add_argument("--qnn-mode", choices=["auto", "manual"], default="auto")
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--width", type=int, default=3)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--vector-word-size", type=int, default=6)
    p.add_argument("--learning-rate", type=float, default=0.5)
    p.add_argument("--attention-top-k", type=int, default=5)
    p.add_argument("--no-attention", action="store_true")
    p.add_argument("--context-file", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--json", action="store_true")
    p.add_argument("--debug", action="store_true")
    raise SystemExit(asyncio.run(_main(p.parse_args())))
