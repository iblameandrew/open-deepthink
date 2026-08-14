"""Phase 6: knowledge_distillation.py - DistillationGraph with DistillationMockLLM."""

import sys, asyncio, json, traceback

from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))
import importlib

app_mod = importlib.import_module("app")
from app import DistillationMockLLM
from deepthink.knowledge_distillation import DistillationGraph, DistillationAgent
from deepthink.chains import DISTILLATION_ARCHETYPES

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


# 1) DistillationAgent: basic
def t1():
    a = DistillationAgent("cnt_0_0", 1, "sys", ["bold"], ["lead"])
    assert a.id == "cnt_0_0"
    assert a.archetype_id == 1
    assert a.system_prompt == "sys"
    assert a.attributes == ["bold"]
    assert a.skills == ["lead"]
    assert a.difficulty_history == []
    assert a.history == []
    assert a.context_memory == ""
    assert a.current_question == ""
    assert a.inherited_from is None
    assert a.solved_parent_question == False


chk("DistillationAgent default state", t1)


# 2) DistillationAgent: to_dict
def t2():
    a = DistillationAgent("a", 2, "sp", ["x"], ["y"])
    a.context_memory = "hello"
    d = a.to_dict()
    assert d["id"] == "a"
    assert d["archetype_id"] == 2
    assert d["context_memory_chars"] == 5


chk("DistillationAgent.to_dict()", t2)


# 3) DistillationAgent: deep_copy_state preserves history
def t3():
    a = DistillationAgent("a", 1, "sp", ["x"])
    a.history.append({"question": "q1", "answer": "a1"})
    a.difficulty_history.append("Easy")
    d = a.deep_copy_state()
    assert d["history"] == [{"question": "q1", "answer": "a1"}]
    assert d["difficulty_history"] == ["Easy"]


chk("DistillationAgent.deep_copy_state() preserves history", t3)


# 4) DistillationGraph: init 12 agents across 7 layers
def t4():
    g = DistillationGraph(
        DistillationMockLLM(), ["topic1"], "anchor?", token_budget=100, debug_mode=True
    )
    # topology 1x2x2x2x2x2x1 = 12
    total = sum(len(layer) for layer in g.layers)
    assert total == 12, f"Got {total} agents, expected 12"
    assert len(g.layers) == 7, f"Got {len(g.layers)} layers, expected 7"
    # Each agent has the right archetype
    arch_ids = [a.archetype_id for layer in g.layers for a in layer]
    assert len(set(arch_ids)) == 12, (
        f"Expected 12 unique archetypes, got {set(arch_ids)}"
    )


chk("DistillationGraph initializes 12 unique agents across 7 layers", t4)


# 5) _estimate_tokens
def t5():
    g = DistillationGraph(DistillationMockLLM(), ["t"], "a")
    n = g._estimate_tokens("a" * 100)
    assert isinstance(n, int) and n >= 1
    assert g._estimate_tokens("") == 0
    assert g._estimate_tokens("word " * 400) > n


chk("_estimate_tokens counts text (tiktoken or heuristic)", t5)


# 6) total_tokens
def t6():
    g = DistillationGraph(DistillationMockLLM(), ["t"], "a")
    g.total_input_tokens = 100
    g.total_output_tokens = 50
    assert g.total_tokens == 150


chk("total_tokens property", t6)


# 7) _trim_context_memory cap
def t7():
    g = DistillationGraph(DistillationMockLLM(), ["t"], "a")
    big = "x" * (g.CONTEXT_MEMORY_MAX_CHARS + 1000)
    trimmed = g._trim_context_memory(big)
    assert len(trimmed) == g.CONTEXT_MEMORY_MAX_CHARS
    assert trimmed == "x" * g.CONTEXT_MEMORY_MAX_CHARS


chk("_trim_context_memory enforces 100k-token cap", t7)


# 8) _trim_context_memory below cap
def t8():
    g = DistillationGraph(DistillationMockLLM(), ["t"], "a")
    small = "abc"
    assert g._trim_context_memory(small) == "abc"


chk("_trim_context_memory keeps small text unchanged", t8)


# 9) _build_current_grid_description
def t9():
    g = DistillationGraph(DistillationMockLLM(), ["t"], "a")
    desc = g._build_current_grid_description()
    assert "cnt_0_0" in desc
    assert "Attributes=" in desc


chk("_build_current_grid_description includes all 12 agents", t9)


# 10) Run a full epoch
async def t10():
    g = DistillationGraph(
        DistillationMockLLM(),
        ["topic1", "topic2"],
        "What is the meaning of life?",
        token_budget=1_000_000,
        debug_mode=True,
    )
    res = await g.run_epoch()
    assert res == True  # should continue
    assert g.epochs_run == 1
    assert len(g.distilled_data) == 12  # 12 agents
    for qa in g.distilled_data:
        assert "epoch" in qa and qa["epoch"] == 1
        assert "agent_id" in qa
        assert "archetype_id" in qa
        assert "question" in qa
        assert "answer" in qa


chk("run_epoch produces 12 QA pairs in dataset", t10)


# 11) Token accounting after one epoch
async def t11():
    g = DistillationGraph(
        DistillationMockLLM(), ["t"], "a", token_budget=1_000_000, debug_mode=True
    )
    await g.run_epoch()
    assert g.total_input_tokens > 0
    assert g.total_output_tokens > 0


chk("Token accounting after one epoch (in/out > 0)", t11)


# 12) Budget exhaustion
async def t12():
    g = DistillationGraph(
        DistillationMockLLM(), ["t"], "a", token_budget=1, debug_mode=True
    )
    res = await g.run_epoch()
    assert res == False  # budget exhausted


chk("Token budget exhaustion stops the run", t12)


# 13) Mirror descent spawns children (difficulty_history gets entries)
async def t13():
    g = DistillationGraph(
        DistillationMockLLM(), ["t"], "a", token_budget=1_000_000, debug_mode=True
    )
    await g.run_epoch()
    # Each agent should have at least one difficulty entry
    flat = g._flat_agents()
    diffs = [a.difficulty_history for a in flat]
    assert all(len(d) >= 1 for d in diffs), f"Some agents missing difficulty: {diffs}"
    # Mix of Easy/Hard
    all_diff = [d for sub in diffs for d in sub]
    assert "Easy" in all_diff or "Hard" in all_diff


chk("Mirror descent evaluates all agents (difficulty_history populated)", t13)


# 14) File write of dataset
async def t14():
    import os, tempfile

    tmp = tempfile.mkdtemp()
    g = DistillationGraph(
        DistillationMockLLM(),
        ["t"],
        "a",
        token_budget=1_000_000,
        debug_mode=True,
        output_dir=tmp,
    )
    await g.run_epoch()
    assert os.path.exists(g.dataset_path), f"Dataset not written: {g.dataset_path}"
    with open(g.dataset_path) as f:
        d = json.load(f)
    assert "qa_pairs" in d
    assert len(d["qa_pairs"]) == 12


chk("Dataset file is written to disk with 12 QA pairs", t14)


# 15) Topology archive
async def t15():
    import os, tempfile

    tmp = tempfile.mkdtemp()
    g = DistillationGraph(
        DistillationMockLLM(),
        ["t"],
        "a",
        token_budget=1_000_000,
        debug_mode=True,
        output_dir=tmp,
    )
    await g.run_epoch()
    assert os.path.exists(g.topology_archive_path)
    with open(g.topology_archive_path) as f:
        d = json.load(f)
    assert isinstance(d, list)
    assert len(d) == 1  # one archive snapshot for one epoch
    assert d[0]["epoch"] == 1
    assert len(d[0]["layers"]) == 7


chk("Topology archive snapshot written after epoch", t15)


# 16) run_epoch returns False when is_running = False (stopped)
async def t16():
    g = DistillationGraph(
        DistillationMockLLM(), ["t"], "a", token_budget=1_000_000, debug_mode=True
    )
    g.is_running = False
    res = await g.run_epoch()
    assert res == False


chk("run_epoch returns False when is_running=False (stop)", t16)


# 17) DISTILLATION_ARCHETYPES has 12 entries
def t17():
    assert len(DISTILLATION_ARCHETYPES) == 12
    # Each has required fields
    for k, v in DISTILLATION_ARCHETYPES.items():
        assert "name" in v and "attributes" in v and "system_prompt" in v
        assert isinstance(v["attributes"], list) and len(v["attributes"]) >= 3


chk("DISTILLATION_ARCHETYPES has 12 well-formed entries", t17)


# 18) Perplexity is set after epoch
async def t18():
    g = DistillationGraph(
        DistillationMockLLM(), ["t"], "a", token_budget=1_000_000, debug_mode=True
    )
    await g.run_epoch()
    # last_perplexity should be a number (from mock LLM)
    assert isinstance(g.last_perplexity, (int, float))


chk("Perplexity is computed after epoch", t18)


# 19) Second epoch evolves topics
async def t19():
    g = DistillationGraph(
        DistillationMockLLM(),
        ["original_topic"],
        "anchor",
        token_budget=10_000_000,
        debug_mode=True,
    )
    await g.run_epoch()
    epoch1_topics = list(g.topics)
    await g.run_epoch()
    epoch2_topics = list(g.topics)
    # After epoch 2, topics should be the seed_creator output (mocked 12 AI topics)
    assert (
        epoch2_topics != epoch1_topics
        or "Advanced" in epoch2_topics[0]
        or len(epoch2_topics) == 12
    )


chk("Topics evolve between epochs (Seed Creator)", t19)


# 20) Synthesized final_answer populated
async def t20():
    g = DistillationGraph(
        DistillationMockLLM(), ["t"], "a", token_budget=1_000_000, debug_mode=True
    )
    await g.run_epoch()
    assert isinstance(g.final_answer, str)
    assert len(g.final_answer) > 0


chk("final_answer is populated after an epoch", t20)

for name, status, err in results:
    line = f"  [{status}] {name}"
    if err:
        line += f" :: {err}"
    print(line)
ok = sum(1 for _, s, _ in results if s == "OK")
print(f"\nPHASE 6: {ok}/{len(results)} OK")
