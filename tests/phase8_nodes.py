"""Phase 8: app.py graph nodes (agent_node, synthesis_node, code_execution_node, etc.) with mock LLM."""

import sys, asyncio, traceback

sys.path.insert(0, r"C:\Users\def78\smenos\local-deepthink")
import importlib

app_mod = importlib.import_module("app")
from app import (
    app,
    CoderMockLLM,
    GraphState,
    create_agent_node,
    create_synthesis_node,
    create_code_execution_node,
    create_archive_epoch_outputs_node,
    create_metrics_node,
    create_reframe_and_decompose_node,
    create_update_agent_prompts_node,
    log_stream,
    sessions,
)
import json

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
        results.append(
            (name, "FAIL", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
        )


# Base state for tests
def base_state(**overrides):
    s = {
        "mode": "algorithm",
        "modules": [],
        "synthesis_context_queue": [],
        "agent_personas": {
            "agent_0_0": {"name": "Alice", "mbti_type": "INTP", "specialty": "Logic"},
            "agent_0_1": {
                "name": "Bob",
                "mbti_type": "ENFP",
                "specialty": "Creativity",
            },
            "agent_1_0": {
                "name": "Carol",
                "mbti_type": "INTJ",
                "specialty": "Strategy",
            },
        },
        "previous_solution": "",
        "current_problem": "Build a calculator",
        "original_request": "Build a calculator",
        "decomposed_problems": {
            "agent_0_0": "sub1",
            "agent_0_1": "sub2",
            "agent_1_0": "sub3",
        },
        "layers": [],
        "epoch": 0,
        "max_epochs": 2,
        "params": {
            "prompt_alignment": 1.0,
            "density": 1.0,
            "learning_rate": 1.0,
            "vector_word_size": 4,
            "mbti_archetypes": ["INTP", "ENFP", "INTJ"],
        },
        "all_layers_prompts": [
            ["You are Agent 0.0, Logic.", "You are Agent 0.1, Creativity."],
            ["You are Agent 1.0, Strategy."],
        ],
        "agent_outputs": {},
        "memory": {},
        "final_solution": None,
        "perplexity_history": [],
        "raptor_index": None,
        "all_rag_documents": [],
        "academic_papers": None,
        "is_code_request": True,
        "session_id": "test_session",
        "chat_history": [],
    }
    s.update(overrides)
    return s


# 1) Agent node - basic (layer 0)
async def t1():
    llm = CoderMockLLM()
    node = create_agent_node(llm, "agent_0_0")(base_state())
    # Returns a coroutine - need to await
    out = await node
    assert "agent_outputs" in out
    assert "agent_0_0" in out["agent_outputs"]
    agent_out = out["agent_outputs"]["agent_0_0"]
    assert "proposed_solution" in agent_out or "original_problem" in agent_out


chk("create_agent_node layer 0 produces output", t1)


# 2) Agent node - layer 1 (uses previous layer output)
async def t2():
    llm = CoderMockLLM()
    s = base_state()
    s["agent_outputs"] = {
        "agent_0_0": {"proposed_solution": "sol1", "reasoning": "r1"},
        "agent_0_1": {"proposed_solution": "sol2", "reasoning": "r2"},
    }
    node = create_agent_node(llm, "agent_1_0")(s)
    out = await node
    assert "agent_1_0" in out["agent_outputs"]


chk("create_agent_node layer 1 uses previous layer outputs", t2)


# 3) Agent node with invalid JSON falls back gracefully
async def t3():
    # Use a custom LLM that returns garbage
    from langchain_core.runnables import Runnable
    from langchain_core.messages import AIMessage

    class GarbageLLM(Runnable):
        def invoke(self, *a, **k):
            return AIMessage(content="not valid json at all")

        async def ainvoke(self, *a, **k):
            return AIMessage(content="not valid json at all")

    node = create_agent_node(GarbageLLM(), "agent_0_0")(base_state())
    out = await node
    assert "agent_0_0" in out["agent_outputs"]
    # Should have a fallback structure
    assert "proposed_solution" in out["agent_outputs"]["agent_0_0"]
    assert "node_id" in out["agent_outputs"]["agent_0_0"]


chk("Agent node falls back on invalid JSON (graceful error)", t3)


# 4) Synthesis node - algorithm mode, code
async def t4():
    llm = CoderMockLLM()
    s = base_state()
    s["agent_outputs"] = {
        "agent_1_0": {
            "proposed_solution": "```python\nx = 1\n```",
            "reasoning": "simple",
        }
    }
    # s["is_code_request"] is True (from base_state)
    node = create_synthesis_node(llm)(s)
    out = await node
    assert "final_solution" in out
    assert out["final_solution"]["mode"] == "algorithm"


chk("synthesis_node algorithm+code builds final_solution", t4)


# 5) Synthesis node - algorithm mode, non-code
async def t5():
    llm = CoderMockLLM()
    s = base_state()
    s["is_code_request"] = False
    s["agent_outputs"] = {
        "agent_1_0": {"proposed_solution": "the answer is 42", "reasoning": "obvs"}
    }
    node = create_synthesis_node(llm)(s)
    out = await node
    assert "final_solution" in out


chk("synthesis_node algorithm+non-code builds final_solution", t5)


# 6) Synthesis node - brainstorm mode, final epoch
async def t6():
    llm = CoderMockLLM()
    s = base_state(mode="brainstorm", is_code_request=False)
    s["epoch"] = 1  # final epoch (max_epochs=2)
    s["memory"] = {
        "agent_0_0": [{"proposed_solution": "reflection 1", "reasoning": "r1"}],
        "agent_0_1": [{"proposed_solution": "reflection 2", "reasoning": "r2"}],
        "agent_1_0": [{"proposed_solution": "reflection 3", "reasoning": "r3"}],
    }
    s["brainstorm_problem_summary"] = "summary of problem"
    s["brainstorm_prior_conversation"] = "prior chat"
    s["brainstorm_document_context"] = "doc context"
    node = create_synthesis_node(llm)(s)
    out = await node
    assert "final_solution" in out
    assert out["final_solution"]["mode"] == "brainstorm"
    assert "proposed_solution" in out["final_solution"]


chk("synthesis_node brainstorm+final_epoch builds final_solution", t6)


# 7) Synthesis node - brainstorm mode, intermediate epoch (QNN epoch map)
async def t7():
    llm = CoderMockLLM()
    s = base_state(mode="brainstorm", is_code_request=False)
    s["epoch"] = 0  # intermediate
    s["memory"] = {
        "agent_0_0": [{"proposed_solution": "reflection 1", "reasoning": "r1"}],
        "agent_0_1": [{"proposed_solution": "reflection 2", "reasoning": "r2"}],
    }
    node = create_synthesis_node(llm)(s)
    out = await node
    assert "final_solution" in out
    assert out["final_solution"] is not None
    assert out["final_solution"].get("mode") == "brainstorm"
    assert out["final_solution"].get("epoch_map") is True
    assert "previous_solution" in out


chk("synthesis_node brainstorm+intermediate_epoch builds QNN epoch map", t7)


# 8) Synthesis node with no agent outputs
async def t8():
    llm = CoderMockLLM()
    s = base_state()
    s["agent_outputs"] = {}
    node = create_synthesis_node(llm)(s)
    out = await node
    assert "error" in out["final_solution"]


chk("synthesis_node with no inputs returns error", t8)


# 9) Code execution node
async def t9():
    llm = CoderMockLLM()
    s = base_state()
    s["final_solution"] = {"proposed_solution": "```python\nprint('hi')\n```"}
    node = create_code_execution_node(llm)(s)
    out = await node
    assert "modules" in out
    assert len(out["modules"]) == 1


chk("code_execution_node adds module on success", t9)


# 10) Code execution node - non-code path
async def t10():
    llm = CoderMockLLM()
    s = base_state()
    s["is_code_request"] = False
    node = create_code_execution_node(llm)(s)
    out = await node
    assert out.get("synthesis_execution_success") == True
    assert "modules" not in out


chk("code_execution_node is no-op for non-code tasks", t10)


# 11) Archive epoch outputs
async def t11():
    s = base_state()
    s["agent_outputs"] = {
        "agent_0_0": {
            "proposed_solution": "sol1",
            "reasoning": "r1",
            "original_problem": "sub1",
        },
        "agent_0_1": {
            "proposed_solution": "sol2",
            "reasoning": "r2",
            "original_problem": "sub2",
        },
    }
    node = create_archive_epoch_outputs_node()(s)
    out = await node
    assert "all_rag_documents" in out
    assert len(out["all_rag_documents"]) == 2


chk("archive_epoch_outputs_node archives outputs as Documents", t11)


# 12) Archive skips brainstorm
async def t12():
    s = base_state(mode="brainstorm")
    s["agent_outputs"] = {"agent_0_0": {"proposed_solution": "sol1"}}
    node = create_archive_epoch_outputs_node()(s)
    out = await node
    assert out == {}  # no archiving


chk("archive_epoch_outputs_node skips brainstorm mode", t12)


# 13) Metrics node
async def t13():
    llm = CoderMockLLM()
    s = base_state()
    s["agent_outputs"] = {
        "agent_0_0": {"proposed_solution": "Hello", "reasoning": "World"},
        "agent_0_1": {"proposed_solution": "Foo", "reasoning": "Bar"},
    }
    node = create_metrics_node(llm)(s)
    out = await node
    assert "perplexity_history" in out
    assert len(out["perplexity_history"]) == 1


chk("metrics_node computes perplexity", t13)


# 14) Metrics node empty outputs
async def t14():
    llm = CoderMockLLM()
    s = base_state()
    s["agent_outputs"] = {}
    node = create_metrics_node(llm)(s)
    out = await node
    # Should not have added any perplexity
    assert "perplexity_history" not in out or out["perplexity_history"] == []


chk("metrics_node is no-op for empty outputs", t14)


# 15) Reframe and decompose - algorithm mode
async def t15():
    llm = CoderMockLLM()
    s = base_state()
    s["final_solution"] = {"proposed_solution": "the answer"}
    node = create_reframe_and_decompose_node(llm)(s)
    out = await node
    assert "decomposed_problems" in out
    assert "current_problem" in out


chk("reframe_and_decompose_node updates problems in algorithm mode", t15)


# 16) Reframe and decompose - brainstorm mode (QNN Step 4D — harder challenge)
async def t16():
    llm = CoderMockLLM()
    s = base_state(mode="brainstorm", is_code_request=False)
    s["final_solution"] = {
        "proposed_solution": "epoch map: ownership vs ordering tension",
        "mode": "brainstorm",
        "epoch_map": True,
    }
    s["original_request"] = "fix the deadlock"
    s["current_problem"] = "fix the deadlock"
    node = create_reframe_and_decompose_node(llm)(s)
    out = await node
    assert "current_problem" in out, f"expected QNN reframe, got {out}"
    assert "decomposed_problems" in out
    assert out["original_request"] == "fix the deadlock"  # ground truth preserved
    assert out["current_problem"] != "fix the deadlock" or "Harder" in out["current_problem"]
    # All agents share the new thinking challenge
    assert len(out["decomposed_problems"]) == sum(
        len(layer) for layer in s["all_layers_prompts"]
    )


chk("reframe_and_decompose_node QNN reframe in brainstorm mode", t16)


# 17) Update agent prompts - algorithm mode
async def t17():
    llm = CoderMockLLM()
    s = base_state()
    s["agent_outputs"] = {
        "agent_0_0": {"proposed_solution": "x", "reasoning": "y"},
        "agent_0_1": {"proposed_solution": "x2", "reasoning": "y2"},
        "agent_1_0": {"proposed_solution": "x3", "reasoning": "y3"},
    }
    node = create_update_agent_prompts_node(llm)(s)
    out = await node
    assert "all_layers_prompts" in out
    assert out["epoch"] == 1


chk("update_agent_prompts_node advances epoch in algorithm mode", t17)


# 18) Update agent prompts - brainstorm mode
async def t18():
    llm = CoderMockLLM()
    s = base_state(mode="brainstorm")
    s["agent_outputs"] = {
        "agent_0_0": {"proposed_solution": "x"},
        "agent_0_1": {"proposed_solution": "x2"},
    }
    node = create_update_agent_prompts_node(llm)(s)
    out = await node
    assert "all_layers_prompts" in out


chk("update_agent_prompts_node advances epoch in brainstorm mode", t18)

for name, status, err in results:
    line = f"  [{status}] {name}"
    if err:
        line += f" :: {err}"
    print(line)
ok = sum(1 for _, s, _ in results if s == "OK")
print(f"\nPHASE 8: {ok}/{len(results)} OK")
