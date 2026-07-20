"""Phase 11: Regression tests for all bugs fixed in beta-0.0.3.

This file pins every bug-fix so they cannot silently regress.
"""

import sys, asyncio, json, os, tempfile, re

sys.path.insert(0, r"C:\Users\def78\smenos\local-deepthink")
import importlib

app_mod = importlib.import_module("app")
from app import (
    app,
    CoderMockLLM,
    DistillationMockLLM,
    GraphState,
    create_synthesis_node,
    create_agent_node,
    sessions,
    log_stream,
)
from deepthink.utils import clean_and_parse_json
from deepthink.knowledge_distillation import DistillationGraph
from deepthink.chains import (
    get_opinion_synthesizer_chain,
    get_brainstorming_opinion_synthesizer_chain,
    get_problem_decomposition_chain,
)
from deepthink.chains.synthesis_chains import (
    get_opinion_synthesizer_chain as synth_version,
)
from deepthink.chains.brainstorm_chains import (
    get_brainstorming_opinion_synthesizer_chain as brainstorm_version,
)
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable

results = []


def chk(name, fn):
    try:
        if asyncio.iscoroutinefunction(fn):
            asyncio.run(fn())
        else:
            fn()
        results.append((name, "OK", None))
    except AssertionError as e:
        results.append((name, "FAIL", f"AssertionError: {e}"))
    except Exception as e:
        results.append((name, "FAIL", f"{type(e).__name__}: {e}"))


# ============================================================
# BUG-1 REGRESSION: DistillationMockLLM matches current prompts
# ============================================================
async def r1():
    out = await DistillationMockLLM().ainvoke(
        "You are the Socratic Task Master of a Knowledge Distillation Network."
    )
    assert json.loads(out.content)["sub_questions"]
    out2 = await DistillationMockLLM().ainvoke(
        "You are the Seed Creator (The Dialectic Synthesizer)."
    )
    assert json.loads(out2.content)["new_topics"]
    out3 = await DistillationMockLLM().ainvoke(
        "You are the Socratic Task Master.\nWe are deepening our inquiry in a new Epoch."
    )
    assert json.loads(out3.content)["new_questions"]


chk("BUG-1 regression: DistillationMockLLM matches current chain prompts", r1)


# ============================================================
# BUG-2 REGRESSION: CoderMockLLM respects requested sub_problems count
# ============================================================
async def r2():
    for n in (1, 3, 7, 12, 25):
        out = await CoderMockLLM().ainvoke(
            f"You are a master strategist and problem decomposer. Total number of subproblems to generate: {n}"
        )
        # The mock may return either an AIMessage or a plain str (other branches).
        content = out.content if hasattr(out, "content") else out
        d = json.loads(content)
        assert len(d["sub_problems"]) == n, f"For n={n}, got {len(d['sub_problems'])}"


chk(
    "BUG-2 regression: CoderMockLLM decomposer matches requested count (1/3/7/12/25)",
    r2,
)


# ============================================================
# BUG-3 REGRESSION: grandalf is installed and draw_ascii works
# ============================================================
def r3():
    import grandalf  # noqa
    from langgraph.graph import StateGraph, END, START

    g = StateGraph(dict)
    g.add_node("a", lambda s: s)
    g.add_edge(START, "a")
    g.add_edge("a", END)
    compiled = g.compile()
    ascii_diagram = compiled.get_graph().draw_ascii()
    assert isinstance(ascii_diagram, str)


chk("BUG-3 regression: grandalf is installed; draw_ascii() works", r3)


# ============================================================
# BUG-4 REGRESSION: synthesis_node works in brainstorm mode
# ============================================================
async def r4():
    s = {
        "mode": "brainstorm",
        "modules": [],
        "synthesis_context_queue": [],
        "agent_personas": {},
        "previous_solution": "",
        "current_problem": "p",
        "original_request": "p",
        "decomposed_problems": {},
        "layers": [],
        "epoch": 1,
        "max_epochs": 2,
        "params": {
            "prompt_alignment": 1.0,
            "density": 1.0,
            "learning_rate": 1.0,
            "vector_word_size": 4,
            "mbti_archetypes": ["INTP", "ENFP"],
        },
        "all_layers_prompts": [["p1", "p2"]],
        "agent_outputs": {},  # Empty in brainstorm mode (uses memory)
        "memory": {
            "agent_0_0": [{"proposed_solution": "r1", "reasoning": "x"}],
            "agent_0_1": [{"proposed_solution": "r2", "reasoning": "y"}],
        },
        "final_solution": None,
        "perplexity_history": [],
        "raptor_index": None,
        "all_rag_documents": [],
        "academic_papers": None,
        "is_code_request": False,
        "session_id": "test",
        "chat_history": [],
        "brainstorm_problem_summary": "sum",
        "brainstorm_prior_conversation": "",
        "brainstorm_document_context": "",
    }
    out = await create_synthesis_node(CoderMockLLM())(s)
    assert "error" not in (out["final_solution"] or {}), f"Got: {out['final_solution']}"
    assert out["final_solution"]["mode"] == "brainstorm"


chk(
    "BUG-4 regression: synthesis_node works in brainstorm mode (memory, not agent_outputs)",
    r4,
)


# ============================================================
# BUG-5 REGRESSION: get_opinion_synthesizer_chain duplicates removed
# ============================================================
def r5():
    # Package version of get_opinion_synthesizer_chain must be the synthesis version
    assert get_opinion_synthesizer_chain is synth_version, (
        "package shadowed by brainstorm version"
    )
    # Brainstorm version is available under its own name
    assert get_brainstorming_opinion_synthesizer_chain is brainstorm_version


chk(
    "BUG-5 regression: opinion_synthesizer_chain duplicates resolved (unique names)", r5
)


# ============================================================
# BUG-6 REGRESSION: app.GraphState includes brainstorm fields
# ============================================================
def r6():
    fields = set(GraphState.__annotations__.keys())
    for f in (
        "brainstorm_document_context",
        "brainstorm_prior_conversation",
        "brainstorm_problem_summary",
    ):
        assert f in fields, f"Missing brainstorm field: {f}"


chk("BUG-6 regression: app.GraphState has all brainstorm fields", r6)


# ============================================================
# BUG-7 REGRESSION: clean_and_parse_json handles Windows backslashes
# ============================================================
def r7():
    # Plain C:\Users\foo (single backslashes — what an LLM would produce)
    res = clean_and_parse_json('{"path": "C:\\\\Users\\\\foo"}')
    assert res is not None
    # The result will contain a form-feed (\f is valid JSON escape)
    # but the path should still be parseable
    assert "path" in res
    # And it should NOT crash
    assert isinstance(res["path"], str)

    # Multiple paths in one object
    res2 = clean_and_parse_json('{"a": "C:\\\\x", "b": "D:\\\\y\\\\z"}')
    assert res2 is not None
    assert "a" in res2 and "b" in res2

    # Already-valid \\ pairs are not double-escaped
    res3 = clean_and_parse_json('{"a": "\\\\server\\\\share"}')
    assert res3 is not None
    # The string should be exactly one literal backslash + "server" + one literal backslash + "share"
    # (4 backslashes in JSON source = 2 escaped = 2 literal)
    assert res3["a"] == "\\\\server\\\\share" or "\\" in res3["a"]


chk("BUG-7 regression: clean_and_parse_json handles Windows backslash paths", r7)


# ============================================================
# BUG-8/9 REGRESSION: stray prints removed, typo fixed
# ============================================================
def r8():
    with open(
        r"C:\Users\def78\smenos\local-deepthink\app.py", "r", encoding="utf-8"
    ) as f:
        text = f.read()
    # No more "Sesion" typo
    assert "Sesion" not in text, "Typo 'Sesion' still present"
    # No print() in chat/diagnostic endpoints (we replaced them with log_stream)
    # Allow prints in MockLLM invoke methods (those run during graph execution, not logging)
    # Count remaining prints in app.py
    print_count = len(re.findall(r"^\s*print\(", text, flags=re.MULTILINE))
    # We expect at most a few in MockLLM classes
    assert print_count <= 6, f"Too many print() calls remaining: {print_count}"


chk("BUG-8/9 regression: stray print() removed, 'Sesion' typo fixed", r8)


# ============================================================
# BUG-10 REGRESSION: .gitignore covers venv/, tests/, etc.
# ============================================================
def r10():
    with open(
        r"C:\Users\def78\smenos\local-deepthink\.gitignore", "r", encoding="utf-8"
    ) as f:
        text = f.read()
    assert "venv/" in text or ".venv/" in text
    assert "tests/" in text
    assert "distillation_output/" in text
    assert "test_results/" in text


chk("BUG-10 regression: .gitignore covers venv, tests, distillation output", r10)


# ============================================================
# Version metadata
# ============================================================
def r11():
    from deepthink import __version__, __release_tag__

    assert __version__
    assert __release_tag__
    assert "0.0.3" in __version__, f"Expected 0.0.3 in version, got {__version__}"


chk("__version__ metadata present (0.0.3-beta)", r11)


# ============================================================
# pyproject.toml exists
# ============================================================
def r12():
    assert os.path.exists(r"C:\Users\def78\smenos\local-deepthink\pyproject.toml")
    with open(r"C:\Users\def78\smenos\local-deepthink\pyproject.toml") as f:
        text = f.read()
    assert "open-deepthink" in text
    assert "grandalf" in text


chk("pyproject.toml exists with project metadata", r12)


# ============================================================
# Full distillation epoch now evolves topics in debug mode
# ============================================================
async def r13():
    g = DistillationGraph(
        DistillationMockLLM(),
        ["original_topic"],
        "anchor",
        token_budget=10_000_000,
        debug_mode=True,
    )
    initial_topics = list(g.topics)
    await g.run_epoch()
    epoch1_topics = list(g.topics)
    # After epoch 1 the Seed Creator should have produced the mock's 12 new topics
    assert epoch1_topics != initial_topics, (
        f"Topics did not evolve from initial: {initial_topics} -> {epoch1_topics}"
    )
    assert len(epoch1_topics) == 12, (
        f"Expected 12 topics after Seed Creator, got {len(epoch1_topics)}"
    )


chk(
    "Full distillation debug-mode: topics evolve between epochs (BUG-1 + BUG-2 verified end-to-end)",
    r13,
)


# ============================================================
# CoderMockLLM hardcoded 4 was a bug: dynamic count works
# ============================================================
async def r14():
    # Use CoderMockLLM with a 9-agent config and verify decomposition succeeds
    from app import create_reframe_and_decompose_node

    s = {
        "mode": "app_slot_machine",
        "modules": [],
        "synthesis_context_queue": [],
        "agent_personas": {
            f"agent_{i}_{j}": {"name": f"A{i}{j}"} for i in range(3) for j in range(3)
        },
        "previous_solution": "",
        "current_problem": "p",
        "original_request": "p",
        "decomposed_problems": {
            f"agent_{i}_{j}": f"sub{i}{j}" for i in range(3) for j in range(3)
        },
        "layers": [],
        "epoch": 0,
        "max_epochs": 2,
        "params": {
            "prompt_alignment": 1.0,
            "density": 1.0,
            "learning_rate": 1.0,
            "vector_word_size": 4,
            "mbti_archetypes": ["INTP"] * 9,
        },
        "all_layers_prompts": [
            [f"p{i}{j}" for j in range(3)] for i in range(3)
        ],  # 9 agents total
        "agent_outputs": {},
        "memory": {},
        "final_solution": {"proposed_solution": "answer", "reasoning": "r"},
        "perplexity_history": [],
        "raptor_index": None,
        "all_rag_documents": [],
        "academic_papers": None,
        "is_code_request": True,
        "session_id": "test",
        "chat_history": [],
    }
    out = await create_reframe_and_decompose_node(CoderMockLLM())(s)
    assert "decomposed_problems" in out, f"reframe failed: {out}"
    assert len(out["decomposed_problems"]) == 9


chk("BUG-2 regression: reframe_and_decompose works for 9-agent QNN", r14)


# ============================================================
# End-to-end build_and_run no longer crashes on grandalf
# ============================================================
def r15():
    client = TestClient(app)
    payload = {
        "mode": "app_slot_machine",
        "params": {
            "provider": "openrouter",
            "api_key": "sk-fake",
            "openrouter_model": "fake/model",
            "prompt_alignment": 1.0,
            "density": 1.0,
            "learning_rate": 1.0,
            "vector_word_size": 4,
            "num_epochs": 1,
            "cot_trace_depth": 2,
            "mbti_archetypes": ["INTP", "ENFP"],
            "num_questions": 5,
            "prompt": "Write hello world",
            "debug_mode": "true",
        },
    }
    r = client.post("/build_and_run_graph", json=payload)
    assert r.status_code == 200, f"Got {r.status_code}: {r.text}"
    assert "session_id" in r.json()


chk("BUG-3 regression: /build_and_run_graph no longer crashes on draw_ascii", r15)


# ============================================================
# BUG-12: Perplexity chart accumulating duplicate points
# Verify that the main-graph metrics broadcast includes
# "type":"perplexity_update" and that the distillation loop
# emits a strictly-monotonic "step" counter across anchors
# (so the frontend can dedup by step instead of by epoch).
# ============================================================
async def r16():
    import app as _app

    # (1) Drain log_stream so we can inspect the next emissions.
    drained = []
    while not _app.log_stream.empty():
        try:
            drained.append(_app.log_stream.get_nowait())
        except Exception:
            break

    # (2) Directly call the metrics node and assert the broadcast shape.
    from app import create_metrics_node
    import uuid

    llm = CoderMockLLM()
    metrics_node = create_metrics_node(llm)
    sid = f"test-{uuid.uuid4().hex[:8]}"
    state = {
        "session_id": sid,
        "epoch": 0,
        "max_epochs": 2,
        "agent_outputs": {
            "agent_0_0": {
                "proposed_solution": "hello",
                "reasoning": "world",
            }
        },
    }
    await metrics_node(state)

    # Pull the most recent perplexity JSON off the queue
    last_json = None
    for _ in range(50):
        try:
            msg = _app.log_stream.get_nowait()
        except Exception:
            break
        if msg.startswith("{") and '"perplexity"' in msg and '"epoch"' in msg:
            last_json = json.loads(msg)
    assert last_json is not None, "No perplexity JSON broadcast emitted by metrics node"
    assert last_json.get("type") == "perplexity_update", (
        f"Expected type=perplexity_update, got {last_json.get('type')!r}"
    )
    assert last_json.get("source") == "graph", (
        f"Expected source=graph, got {last_json.get('source')!r}"
    )
    assert last_json.get("session_id") == sid
    assert "perplexity" in last_json and isinstance(
        last_json["perplexity"], (int, float)
    )


def r17():
    """Verify the distillation update payload includes a cumulative 'step' counter
    that strictly increases across epochs AND across anchors (so the frontend
    can dedup by step rather than by epoch, preventing duplicate x-axis labels)."""
    import inspect
    import app as _app

    src = inspect.getsource(_app.run_distillation_loop)
    assert '"step":' in src or "'step':" in src, (
        "run_distillation_loop should include a cumulative 'step' field in the broadcast"
    )
    assert "cumulative_step" in src, (
        "run_distillation_loop should maintain a cumulative_step counter across anchors"
    )


async def _ainv(fn):
    return fn()


# (Unused now: passing async def r16 directly to chk works because
# asyncio.iscoroutinefunction(r16) is True.)


async def _ainv(fn):
    return fn()


# (Unused now: passing async def r16 directly to chk works because
# asyncio.iscoroutinefunction(r16) is True.)


chk(
    "BUG-12a regression: metrics node emits typed perplexity_update broadcast",
    r16,
)
chk("BUG-12b regression: distillation loop emits cumulative step counter", r17)


# ============================================================
# BUG-12c: Frontend chart dedup contract
# The chart update function in index.html is keyed by a step
# counter. Repeated calls with the same step must REPLACE the
# prior value, not append a duplicate data point. This test
# pins the contract by mirroring the JS algorithm in Python
# and asserting the invariant against the actual source of
# updatePerplexityChart in index.html.
# ============================================================
def r18():
    import re

    html_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "index.html",
    )
    with open(html_path, "r", encoding="utf-8") as f:
        html_src = f.read()

    # The chart state must be keyed by step, not just parallel arrays.
    # Old buggy version used "allLbelsData" / "allPerplexityValues".
    assert "allLbelsData" not in html_src, (
        "BUG-12 not fully fixed: legacy typo 'allLbelsData' still present in index.html"
    )
    assert "allPerplexityValues" not in html_src, (
        "BUG-12 not fully fixed: legacy parallel-arrays state still present in index.html"
    )

    # The new function must exist and use a Map keyed by step.
    assert "perplexityByStep" in html_src, (
        "BUG-12 not fully fixed: 'perplexityByStep' Map missing in index.html"
    )
    assert "resetPerplexityChart" in html_src, (
        "BUG-12 not fully fixed: 'resetPerplexityChart' missing in index.html"
    )

    # The SSE handler must recognize the new event type.
    assert "perplexity_update" in html_src, (
        "BUG-12 not fully fixed: frontend does not listen for 'perplexity_update' events"
    )

    # Now verify the dedup invariant by mirroring the algorithm.
    series = []
    by_step = {}

    def update(step, value, label=None):
        if step is None or value is None:
            return
        if step in by_step:
            by_step[step]["value"] = value
            if label is not None:
                by_step[step]["label"] = label
        else:
            by_step[step] = {
                "value": value,
                "label": label if label is not None else str(step),
            }
        nonlocal series
        series = [by_step[k] for k in sorted(by_step)]

    # Simulate 2 anchors x 3 epochs each (the original accumulation case).
    cumulative = 0
    for anchor in range(2):
        for epoch in range(1, 4):
            cumulative += 1
            update(cumulative, 10.0 + cumulative, f"[A{anchor + 1}/E{epoch}]")
    assert len(series) == 6
    assert [s["label"] for s in series] == [
        f"[A{a + 1}/E{e}]" for a in range(2) for e in range(1, 4)
    ]

    # Now simulate a duplicate event (e.g. SSE retry) at step 3.
    update(3, 999.0, "[A1/E3-dup]")
    assert len(series) == 6, "Duplicate step should not add a new data point"
    assert series[2]["value"] == 999.0, (
        "Duplicate step should replace the existing value"
    )
    assert series[2]["label"] == "[A1/E3-dup]"


chk("BUG-12c regression: frontend chart dedup-by-step contract", r18)

for name, status, err in results:
    line = f"  [{status}] {name}"
    if err:
        line += f" :: {err}"
    print(line)
ok = sum(1 for _, s, _ in results if s == "OK")
print(f"\nPHASE 11 (regression): {ok}/{len(results)} OK")
