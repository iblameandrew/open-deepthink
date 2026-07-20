"""Phase 4: state.py and other structural types."""

import sys, traceback

sys.path.insert(0, r"C:\Users\def78\smenos\local-deepthink")
results = []


def chk(name, fn):
    try:
        fn()
        results.append((name, "OK", None))
    except AssertionError as e:
        results.append((name, "FAIL", f"AssertionError: {e}"))
    except Exception as e:
        results.append((name, "FAIL", f"{type(e).__name__}: {e}"))


# GraphState
def t1():
    from deepthink.state import GraphState

    s: GraphState = {
        "mode": "app_slot_machine",
        "modules": [],
        "synthesis_context_queue": [],
        "agent_personas": {},
        "previous_solution": "",
        "current_problem": "",
        "original_request": "",
        "decomposed_problems": {},
        "layers": [],
        "epoch": 0,
        "max_epochs": 1,
        "params": {},
        "all_layers_prompts": [],
        "agent_outputs": {},
        "memory": {},
        "final_solution": {},
        "perplexity_history": [],
        "raptor_index": None,
        "all_rag_documents": [],
        "academic_papers": None,
        "is_code_request": False,
        "session_id": "x",
        "chat_history": [],
        "brainstorm_document_context": "",
        "brainstorm_prior_conversation": "",
        "brainstorm_problem_summary": "",
    }
    assert s["mode"] == "app_slot_machine"


chk("GraphState TypedDict can be instantiated with all keys", t1)


# BRAINSTORM_EXPERTS
def t2():
    from deepthink.state import BRAINSTORM_EXPERTS

    assert isinstance(BRAINSTORM_EXPERTS, list)
    assert len(BRAINSTORM_EXPERTS) == 5
    for ex in BRAINSTORM_EXPERTS:
        assert "name" in ex and "specialty" in ex and "emoji" in ex


chk("BRAINSTORM_EXPERTS structure (5 experts, name/specialty/emoji)", t2)


# app.py GraphState - separate definition with extra 'llm' and 'summarizer_llm' fields
def t3():
    import importlib

    mod = importlib.import_module("app")
    fields = mod.GraphState.__annotations__.keys()
    expected_core = {
        "mode",
        "modules",
        "agent_personas",
        "previous_solution",
        "current_problem",
        "original_request",
        "decomposed_problems",
        "layers",
        "epoch",
        "max_epochs",
        "params",
        "all_layers_prompts",
        "agent_outputs",
        "memory",
        "final_solution",
        "perplexity_history",
        "raptor_index",
        "all_rag_documents",
        "academic_papers",
        "is_code_request",
        "session_id",
        "chat_history",
    }
    missing = expected_core - set(fields)
    assert not missing, f"app.GraphState missing: {missing}"


chk("app.GraphState has all expected TypedDict fields", t3)


# Note: app.GraphState does NOT include brainstorm_document_context etc.
def t4():
    import importlib

    mod = importlib.import_module("app")
    fields = set(mod.GraphState.__annotations__.keys())
    # These are accessed in the code but are NOT in app.GraphState TypedDict
    expected_brainstorm = {
        "brainstorm_document_context",
        "brainstorm_prior_conversation",
        "brainstorm_problem_summary",
    }
    missing = expected_brainstorm - fields
    # This is more of a soft warning - TypedDict is structural so missing keys don't fail
    # at runtime, but it's a smell
    if missing:
        print(
            f"  INFO: app.GraphState is missing brainstorm fields {missing} (defined in deepthink.state.GraphState)"
        )
    # No assertion - it's an inconsistency, not a hard failure


chk("app.GraphState (informational) brainstorm fields (NOT a fail)", t4)


# Test state initialization - what happens when you try to instantiate with mode=brainstorm but state has no brainstorm fields
def t5():
    import importlib

    mod = importlib.import_module("app")
    GS = mod.GraphState
    # Setting an undeclared key is allowed in TypedDict at runtime
    s: GS = {
        "mode": "brainstorm",
        "modules": [],
        "synthesis_context_queue": [],
        "agent_personas": {},
        "previous_solution": "",
        "current_problem": "x",
        "original_request": "x",
        "decomposed_problems": {},
        "layers": [],
        "epoch": 0,
        "max_epochs": 1,
        "params": {},
        "all_layers_prompts": [],
        "agent_outputs": {},
        "memory": {},
        "final_solution": {},
        "perplexity_history": [],
        "raptor_index": None,
        "all_rag_documents": [],
        "academic_papers": None,
        "is_code_request": False,
        "session_id": "x",
        "chat_history": [],
    }
    # The agent_node accesses these via state.get(...)
    # In create_agent_node (app.py), the state.get is used for these
    # So this works at runtime
    assert s.get("mode") == "brainstorm"


chk("app.GraphState mode='brainstorm' instantiation works", t5)

for name, status, err in results:
    line = f"  [{status}] {name}"
    if err:
        line += f" :: {err}"
    print(line)
ok = sum(1 for _, s, _ in results if s == "OK")
print(f"\nPHASE 4: {ok}/{len(results)} OK")
