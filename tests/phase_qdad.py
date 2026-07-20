"""QDAD / App Slot Machine unit tests — package structure + LangGraph pipeline."""
import sys
import asyncio
import traceback

sys.path.insert(0, r"C:\Users\def78\smenos\open-deepthink")

results = []


def chk(name, fn):
    try:
        fn()
        results.append((name, "OK", None))
    except AssertionError as e:
        results.append((name, "FAIL", f"AssertionError: {e}"))
    except Exception as e:
        results.append((name, "FAIL", f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))


def t1():
    from deepthink.qdad import QDADState, build_qdad_graph, run_qdad_pipeline
    from deepthink.qdad.utils import clamp_params, parse_word_list, format_feature_matrix
    from deepthink.qdad.graph import draw_qdad_ascii

    assert callable(build_qdad_graph)
    assert callable(run_qdad_pipeline)
    assert "foundation" in draw_qdad_ascii().lower()
    n, nt, ds, nv = clamp_params(99, 9.0, 99, 0.0)
    assert n == 8 and nt == 1.8 and ds == 6 and nv == 0.3
    words = parse_word_list(["a", "a", "b", "c"], 4, "noun")
    assert len(words) == 4
    assert len(set(w.lower() for w in words)) == 4


chk("qdad package imports + utils", t1)


def t2():
    from deepthink.chains import (
        get_qdad_foundation_chain,
        get_qdad_noise_chain,
        get_qdad_critic_chain,
        get_qdad_synthesis_chain,
    )
    from app import CoderMockLLM

    llm = CoderMockLLM()
    for factory in (
        get_qdad_foundation_chain,
        get_qdad_noise_chain,
        get_qdad_critic_chain,
        get_qdad_synthesis_chain,
    ):
        chain = factory(llm)
        assert chain is not None


chk("qdad chains construct with MockLLM", t2)


def t3():
    from app import CoderMockLLM
    from deepthink.qdad import build_qdad_graph

    llm = CoderMockLLM()
    graph = build_qdad_graph(llm)
    # LangGraph compiled graph exposes get_graph
    g = graph.get_graph()
    nodes = set(g.nodes)
    for required in ("foundation", "grid", "noise", "denoise", "synthesize"):
        assert required in nodes, f"missing node {required} in {nodes}"


chk("LangGraph QDAD has all phase nodes", t3)


def t4():
    from app import CoderMockLLM, sessions
    from deepthink.qdad import run_qdad_pipeline

    async def run():
        llm = CoderMockLLM()
        sid = "qdad-unit-1"
        sessions[sid] = {
            "session_id": sid,
            "mode": "app_slot_machine",
            "qdad_matrices": {},
            "all_rag_documents": [],
        }
        logs = []

        async def log(msg):
            logs.append(msg)

        sol = await run_qdad_pipeline(
            llm=llm,
            params={
                "grid_size": 2,
                "temperature_scale": 1.3,
                "denoising_steps": 2,
                "noun_verb_temperature": 0.6,
            },
            user_prompt="a cozy night writing app, soft dark mode, offline-first",
            session_id=sid,
            log=log,
            session_store=sessions,
        )
        assert sol["mode"] == "app_slot_machine"
        assert sol["grid_size"] == 2
        assert len(sol["nouns"]) == 2
        assert len(sol["verbs"]) == 2
        assert len(sol["feature_matrix"]) == 2
        assert "App Build Prompt" in sol["proposed_solution"] or "app" in sol[
            "proposed_solution"
        ].lower()
        assert sessions[sid]["final_solution"] is not None
        assert any("PHASE 0" in m or "Foundation" in m for m in logs)
        assert any("PHASE 2" in m or "Noise" in m for m in logs)
        assert any("PHASE 3" in m or "Denois" in m for m in logs)
        assert any("PHASE 4" in m or "Synthes" in m for m in logs)
        assert any("FINAL_ANSWER" in m for m in logs)
        return sol

    sol = asyncio.run(run())
    assert sol["denoising_steps"] == 2


chk("full QDAD LangGraph pipeline (mock) produces build prompt", t4)


# summary
ok = sum(1 for _, s, _ in results if s == "OK")
total = len(results)
for name, status, err in results:
    print(f"  [{status}] {name}" + (f" :: {err}" if err else ""))
print(f"\nPHASE QDAD: {ok}/{total} OK")
sys.exit(0 if ok == total else 1)
