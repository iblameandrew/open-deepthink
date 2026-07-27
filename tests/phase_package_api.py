"""Phase package API: algorithms available as a pure Python library (no app server)."""

from __future__ import annotations

import asyncio
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(ROOT))

results = []


def chk(name, fn):
    try:
        fn()
        results.append((name, "OK", None))
    except Exception as e:
        tb = traceback.format_exc().splitlines()[-3:]
        results.append((name, "FAIL", f"{type(e).__name__}: {e} | " + " | ".join(tb)))


def t1():
    import deepthink as dt

    assert dt.__version__
    # Public algorithm surface
    for name in (
        "run_qnn",
        "run_qdad",
        "run_distillation",
        "run_qnn_pipeline",
        "run_qdad_pipeline",
        "DistillationGraph",
        "DistillationAgent",
        "create_llm",
        "compute_self_attention",
        "default_qnn_params",
        "default_qdad_params",
        "get_settings",
        "QNNResult",
        "QDADState",
        "DISTILLATION_ARCHETYPES",
    ):
        assert hasattr(dt, name), f"missing deepthink.{name}"
        assert getattr(dt, name) is not None


chk("deepthink public API symbols export", t1)


def t2():
    from deepthink.api import run_distillation, run_qdad, run_qnn
    from deepthink.distillation import DistillationAgent, DistillationGraph
    from deepthink.providers import create_llm
    from deepthink.qdad import QDADState, default_qdad_params, run_qdad_pipeline
    from deepthink.qnn import QNNResult, default_qnn_params, run_qnn_pipeline

    assert callable(run_qnn_pipeline) and callable(run_qdad_pipeline)
    assert callable(run_qnn) and callable(run_qdad) and callable(run_distillation)
    assert callable(create_llm)
    assert isinstance(default_qnn_params(), dict)
    assert isinstance(default_qdad_params(), dict)
    assert DistillationGraph is not None
    assert DistillationAgent is not None
    assert QNNResult is not None
    assert QDADState is not None


chk("subpackage imports for all algorithms", t2)


def t3():
    from deepthink import DISTILLATION_ARCHETYPES, default_qdad_params, default_qnn_params

    assert len(DISTILLATION_ARCHETYPES) == 12
    qnn = default_qnn_params()
    assert qnn["qnn_mode"] in ("auto", "manual")
    qdad = default_qdad_params()
    assert qdad["grid_size"] >= 1 and "denoising_steps" in qdad


chk("defaults + archetypes", t3)


def t4():
    """Library QNN + QDAD run with mock LLM — no app import required for QDAD/QNN cores."""
    from deepthink import run_qdad, run_qnn
    from langchain_core.runnables import Runnable

    class MockLLM(Runnable):
        def invoke(self, input_data, config=None, **kwargs):
            return asyncio.get_event_loop().run_until_complete(
                self.ainvoke(input_data, config=config, **kwargs)
            )

        async def ainvoke(self, input_data, config=None, **kwargs):
            t = str(input_data).lower()
            if "noun" in t or "verb" in t or "foundation" in t:
                return (
                    '{"nouns": ["canvas", "ink", "lamp", "desk"], '
                    '"verbs": ["write", "glow", "focus", "rest"]}'
                )
            if "complexity" in t:
                return (
                    '{"complexity_score": 3, "recommended_layers": 2, '
                    '"recommended_width": 2, "recommended_epochs": 1, "reasoning": "x"}'
                )
            if "seed" in t or "space-separated" in t:
                return "distill ownership latch invariant probe reframe"
            if "guiding_words" in t or "node generator" in t or "persona" in t:
                return (
                    '{"name": "Debug", "specialty": "x", "emoji": "x", '
                    '"guiding_words": "distill", "attributes": ["A"], '
                    '"skills": ["s"], "system_prompt": "You are a test expert."}'
                )
            if "re-framer" in t or "new_problem" in t:
                return '{"new_problem": "Harder under concurrency."}'
            if "mirror" in t:
                return (
                    '{"updated_system_prompt": "You are refined.", '
                    '"updated_attributes": ["A"], "updated_skills": ["s"]}'
                )
            if "epoch map" in t or "compact" in t:
                return "Epoch map stub."
            if "solution-space" in t or "polisher" in t or "synthesiz" in t:
                return (
                    "## 1. Impasse\nStub.\n## 3. Strategies\nProbe first.\n"
                    "## 5. Next Steps\n1. Log 2. Test"
                )
            if "feature" in t or "noise" in t or "critic" in t:
                return "A concrete offline-first writing desk feature with soft dark mode."
            if "build prompt" in t or "app build" in t:
                return "# App Build Prompt\n\nBuild a cozy night writing app."
            return (
                '{"original_problem": "x", "proposed_solution": "Probe ownership.", '
                '"reasoning": "x", "falsifiers": "x", "risks": "x", "skills_used": []}'
            )

    llm = MockLLM()

    async def _run():
        qnn = await run_qnn(
            llm,
            "Deadlock in async ownership",
            params={
                "qnn_mode": "manual",
                "manual_layers": 2,
                "manual_width": 2,
                "num_epochs": 1,
                "enable_self_attention": False,
            },
        )
        qdad = await run_qdad(
            llm,
            "cozy night writing app",
            params={"grid_size": 2, "n": 2, "denoising_steps": 1},
        )
        return qnn, qdad

    qnn_res, qdad_res = asyncio.run(_run())
    assert isinstance(qnn_res, dict)
    assert qnn_res.get("proposed_solution") or qnn_res.get("final_solution")
    assert isinstance(qdad_res, dict)


chk("run_qnn + run_qdad library pipelines (mock LLM)", t4)


def t5():
    from deepthink.distillation import DistillationGraph

    try:
        from app import DistillationMockLLM

        mock = DistillationMockLLM()
    except Exception:
        from langchain_core.runnables import Runnable

        class DistMock(Runnable):
            def invoke(self, input_data, config=None, **kwargs):
                return asyncio.get_event_loop().run_until_complete(
                    self.ainvoke(input_data, config=config, **kwargs)
                )

            async def ainvoke(self, input_data, config=None, **kwargs):
                t = str(input_data).lower()
                if "task master" in t or "decompose" in t or "sub-question" in t:
                    return (
                        '{"questions": {"1": "q1", "2": "q2", "3": "q3", "4": "q4", '
                        '"5": "q5", "6": "q6", "7": "q7", "8": "q8", "9": "q9", '
                        '"10": "q10", "11": "q11", "12": "q12"}}'
                    )
                if "mirror" in t or "difficulty" in t:
                    return '{"difficulty": "Easy", "reasoning": "ok"}'
                if "seed" in t or "topic" in t:
                    return '["t1","t2","t3","t4","t5","t6","t7","t8","t9","t10","t11","t12"]'
                if "mix" in t:
                    return '{"system_prompt": "child", "attributes": ["a"], "skills": ["s"]}'
                if "follow" in t:
                    return '{"followup": "deeper?"}'
                if "perplex" in t:
                    return '{"perplexity": 2.5}'
                return "Answer stub for distillation sub-question."

        mock = DistMock()

    g = DistillationGraph(
        mock,
        ["topic_a"],
        "anchor?",
        token_budget=50_000,
        debug_mode=True,
        output_dir=str(ROOT / "distillation_output" / "_pkg_api_test"),
    )
    assert len(g._flat_agents()) == 12
    ok = asyncio.run(g.run_epoch())
    assert ok is True or ok is False  # completed epoch
    assert len(g.distilled_data) >= 1


chk("DistillationGraph library import + one epoch (mock)", t5)


def t6():
    import os

    from deepthink.providers import create_llm

    # Without key, openrouter should raise
    try:
        create_llm(provider="openrouter", api_key=None)
        # If settings have a key in the environment, that's fine
        key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("API_KEY")
        if not key:
            raise AssertionError("create_llm should raise without API key")
    except ValueError as e:
        assert "API key" in str(e) or "key" in str(e).lower()

    # llamacpp never requires a real key
    llm = create_llm(provider="llamacpp", model="test-model")
    assert llm is not None


chk("create_llm provider helpers", t6)


def t7():
    """Algorithm API module must not depend on the web app module."""
    import inspect

    import deepthink.api as api_mod

    assert hasattr(api_mod, "run_qnn")
    src = inspect.getsource(api_mod)
    assert "from app import" not in src
    assert "import app" not in src
    # providers / api must not import the web stack
    import deepthink.providers as prov_mod

    psrc = inspect.getsource(prov_mod)
    assert "from fastapi" not in psrc
    assert "import fastapi" not in psrc
    assert "import uvicorn" not in psrc
    assert "from uvicorn" not in psrc


chk("algorithm API independent of web app module", t7)


def t8():
    from deepthink.cli import build_parser

    p = build_parser()
    # subcommands exist
    help_txt = p.format_help()
    assert "qnn" in help_txt and "qdad" in help_txt and "serve" in help_txt


chk("CLI exposes qnn / qdad / serve", t8)


for name, status, err in results:
    line = f"  [{status}] {name}"
    if err:
        line += f" :: {err}"
    print(line)
ok = sum(1 for _, s, _ in results if s == "OK")
print(f"\nPHASE PACKAGE API: {ok}/{len(results)} OK")
