#!/usr/bin/env python3
"""
QDAD / App Slot Machine runner — materialized on-the-fly by the /qdad skill.
Discovers deepthink from the workspace tree; runs qualitative diffusion.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path


def _discover_deepthink() -> Path | None:
    env = os.environ.get("OPEN_DEEPTHINK_ROOT", "").strip()
    candidates = []
    if env:
        candidates.append(Path(env))
    here = Path.cwd().resolve()
    candidates.append(here)
    candidates.extend(here.parents)
    skill_file = Path(__file__).resolve()
    candidates.append(skill_file.parent)
    candidates.extend(skill_file.parents)
    for c in candidates:
        if not c:
            continue
        if (c / "deepthink" / "qdad" / "pipeline.py").is_file():
            return c
        if (c / "open-deepthink" / "deepthink" / "qdad" / "pipeline.py").is_file():
            return c / "open-deepthink"
    return None


def _ensure_path() -> None:
    root = _discover_deepthink()
    if root is None:
        print(
            "ERROR: Could not find open-deepthink (deepthink/qdad). "
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
                    if "foundation" in t or "distinct nouns" in t:
                        return json.dumps(
                            {
                                "nouns": ["canvas", "lantern", "notebook", "harbor"],
                                "verbs": ["whisper", "weave", "anchor", "glow"],
                            }
                        )
                    if "featureagent" in t or "forward diffusion" in t:
                        return "A wild mock feature: ambient focus with offline capture."
                    if "criticagent" in t or "reverse diffusion" in t:
                        return "A refined mock feature: offline-first focus mode with soft glow."
                    if "synthesizer" in t or "app build" in t:
                        return (
                            "# App Build Prompt\n\n## High-Level Vision\n"
                            "Cozy offline writing app.\n\n## Core Features\n1. Focus timer\n"
                            "2. Offline draft capture\n"
                        )
                    return "mock feature"

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
    from deepthink.qdad import run_qdad_pipeline

    llm = _build_llm(args)
    params = {
        "grid_size": args.n,
        "n": args.n,
        "temperature_scale": args.temperature_scale,
        "denoising_steps": args.denoising_steps,
        "noun_verb_temperature": args.noun_verb_temperature,
    }
    doc = ""
    if args.context_file:
        doc = Path(args.context_file).read_text(encoding="utf-8", errors="replace")

    def log(msg: str):
        print(msg, file=sys.stderr)

    result = await run_qdad_pipeline(
        llm=llm,
        params=params,
        user_prompt=args.prompt,
        document_context=doc,
        log=log,
        session_id="skill-on-the-fly",
    )
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote {args.out}", file=sys.stderr)
    print(result.get("proposed_solution") or json.dumps(result, indent=2))
    if args.json:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="QDAD pipeline (skill-materialized runner)")
    p.add_argument("--prompt", "-p", required=True)
    p.add_argument("--provider", choices=["openrouter", "llamacpp"], default="openrouter")
    p.add_argument("--api-key", default=None)
    p.add_argument("--model", default="stepfun/step-3.5-flash:free")
    p.add_argument("--base-url", default="http://localhost:8080/v1")
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--temperature-scale", type=float, default=1.3)
    p.add_argument("--denoising-steps", type=int, default=3)
    p.add_argument("--noun-verb-temperature", type=float, default=0.6)
    p.add_argument("--context-file", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--json", action="store_true")
    p.add_argument("--debug", action="store_true")
    raise SystemExit(asyncio.run(_main(p.parse_args())))
