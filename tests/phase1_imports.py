"""Phase 1: import + module structure checks."""
import sys, os, traceback
sys.path.insert(0, r"C:\Users\def78\smenos\local-deepthink")
results = []
def chk(name, fn):
    try:
        fn()
        results.append((name, "OK", None))
    except Exception as e:
        tb = traceback.format_exc().splitlines()[-3:]
        results.append((name, "FAIL", f"{type(e).__name__}: {e} | " + " | ".join(tb)))

# Import all deepthink modules
def t1():
    from deepthink import utils
    assert hasattr(utils, "clean_and_parse_json")
    assert hasattr(utils, "execute_code_in_sandbox")
chk("import deepthink.utils", t1)

def t2():
    from deepthink.state import GraphState, BRAINSTORM_EXPERTS
    assert isinstance(BRAINSTORM_EXPERTS, list)
chk("import deepthink.state", t2)

def t3():
    from deepthink.chains import (get_input_spanner_chain, get_attribute_and_hard_request_generator_chain, get_seed_generation_chain, get_dense_spanner_chain, get_synthesis_chain, get_code_synthesis_chain, get_problem_decomposition_chain, get_problem_reframer_chain, get_opinion_synthesizer_chain, get_memory_summarizer_chain, get_perplexity_heuristic_chain, get_module_card_chain, get_code_detector_chain, get_request_is_code_chain, get_interrogator_chain, get_paper_formatter_chain, get_rag_chat_chain, get_complexity_estimator_chain, get_expert_reflection_chain, get_brainstorming_agent_chain, get_brainstorming_mirror_descent_chain, get_brainstorming_synthesis_chain, get_brainstorming_seed_chain, get_brainstorming_spanner_chain, get_problem_summarizer_chain, get_brainstorming_polisher_chain, get_brainstorming_reframer_chain, get_brainstorming_epoch_map_chain, get_task_master_chain, get_seed_creator_chain, get_mirror_descent_chain, get_mixing_chain, get_followup_question_chain, DISTILLATION_ARCHETYPES, get_qdad_foundation_chain, get_qdad_noise_chain, get_qdad_critic_chain, get_qdad_synthesis_chain)
    assert isinstance(DISTILLATION_ARCHETYPES, dict) and len(DISTILLATION_ARCHETYPES) == 12
    assert callable(get_qdad_foundation_chain) and callable(get_qdad_synthesis_chain)
chk("import deepthink.chains (all factories incl. QDAD)", t3)

def t4():
    from deepthink.models import ChatLlamaCpp
chk("import deepthink.models", t4)

def t5():
    from deepthink.chains.perplexity_chain import PerplexityChain
chk("import perplexity_chain", t5)

def t6():
    from deepthink.knowledge_distillation import DistillationGraph, DistillationAgent
chk("import knowledge_distillation", t6)

def t7():
    import importlib
    mod = importlib.import_module("app")
    assert mod.app is not None
    paths = [r.path for r in mod.app.routes if hasattr(r, "path")]
    expected = ["/", "/run_inference_from_state", "/build_and_run_graph", "/export_qnn/{session_id}", "/import_qnn", "/upload_documents", "/upload_code_files", "/upload_repository", "/chat", "/diagnostic_chat", "/harvest", "/stream_log", "/log_stream", "/download_report/{session_id}", "/start_distillation", "/stop_distillation", "/distillation_data", "/download_distillation"]
    missing = [e for e in expected if e not in paths]
    assert not missing, f"Missing endpoints: {missing}"
chk("import app.py + verify 18 endpoints exist", t7)

for name, status, err in results:
    line = f"  [{status}] {name}"
    if err:
        line += f" :: {err}"
    print(line)
ok = sum(1 for _,s,_ in results if s == "OK")
print(f"\nPHASE 1: {ok}/{len(results)} OK")
