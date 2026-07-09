"""Phase 3: All chain factories should construct without error using a mock LLM."""

import sys, traceback, asyncio

sys.path.insert(0, r"C:\Users\def78\smenos\local-deepthink")
from langchain_core.runnables import Runnable
from langchain_core.messages import AIMessage


class _NoopLLM(Runnable):
    def invoke(self, input_data, config=None, **kwargs):
        return AIMessage(content="{}")

    async def ainvoke(self, input_data, config=None, **kwargs):
        return AIMessage(content="{}")


results = []


def chk(name, fn):
    try:
        fn()
        results.append((name, "OK", None))
    except AssertionError as e:
        results.append((name, "FAIL", f"AssertionError: {e}"))
    except Exception as e:
        results.append((name, "FAIL", f"{type(e).__name__}: {e}"))


# agent_chains
from deepthink.chains import (
    get_input_spanner_chain,
    get_attribute_and_hard_request_generator_chain,
    get_seed_generation_chain,
    get_dense_spanner_chain,
    get_synthesis_chain,
    get_code_synthesis_chain,
    get_problem_decomposition_chain,
    get_problem_reframer_chain,
    get_opinion_synthesizer_chain,
    get_memory_summarizer_chain,
    get_perplexity_heuristic_chain,
    get_module_card_chain,
    get_code_detector_chain,
    get_request_is_code_chain,
    get_interrogator_chain,
    get_paper_formatter_chain,
    get_rag_chat_chain,
    get_complexity_estimator_chain,
    get_expert_reflection_chain,
    get_brainstorming_agent_chain,
    get_brainstorming_mirror_descent_chain,
    get_brainstorming_synthesis_chain,
    get_brainstorming_seed_chain,
    get_brainstorming_spanner_chain,
    get_problem_summarizer_chain,
    get_brainstorming_polisher_chain,
    get_brainstorming_reframer_chain,
    get_brainstorming_epoch_map_chain,
    get_task_master_chain,
    get_seed_creator_chain,
    get_mirror_descent_chain,
    get_mixing_chain,
    get_followup_question_chain,
)

llm = _NoopLLM()

chk("get_input_spanner_chain", lambda: (get_input_spanner_chain(llm, 1.0, 1.0),)[0])
chk(
    "get_attribute_and_hard_request_generator_chain",
    lambda: (get_attribute_and_hard_request_generator_chain(llm, 5),)[0],
)
chk("get_seed_generation_chain", lambda: (get_seed_generation_chain(llm),)[0])
chk(
    "get_dense_spanner_chain", lambda: (get_dense_spanner_chain(llm, 1.0, 1.0, 1.0),)[0]
)
chk("get_synthesis_chain", lambda: (get_synthesis_chain(llm),)[0])
chk("get_code_synthesis_chain", lambda: (get_code_synthesis_chain(llm),)[0])
chk(
    "get_problem_decomposition_chain",
    lambda: (get_problem_decomposition_chain(llm),)[0],
)
chk("get_problem_reframer_chain", lambda: (get_problem_reframer_chain(llm),)[0])
chk(
    "get_opinion_synthesizer_chain (synthesis_chains)",
    lambda: (get_opinion_synthesizer_chain(llm),)[0],
)
chk("get_memory_summarizer_chain", lambda: (get_memory_summarizer_chain(llm),)[0])
chk("get_perplexity_heuristic_chain", lambda: (get_perplexity_heuristic_chain(llm),)[0])
chk("get_module_card_chain", lambda: (get_module_card_chain(llm),)[0])
chk("get_code_detector_chain", lambda: (get_code_detector_chain(llm),)[0])
chk("get_request_is_code_chain", lambda: (get_request_is_code_chain(llm),)[0])
chk("get_interrogator_chain", lambda: (get_interrogator_chain(llm),)[0])
chk("get_paper_formatter_chain", lambda: (get_paper_formatter_chain(llm),)[0])
chk("get_rag_chat_chain", lambda: (get_rag_chat_chain(llm),)[0])
chk("get_complexity_estimator_chain", lambda: (get_complexity_estimator_chain(llm),)[0])
chk(
    "get_expert_reflection_chain",
    lambda: (get_expert_reflection_chain(llm, "Dr. X", "Logic", "🧠"),)[0],
)
chk("get_brainstorming_agent_chain", lambda: (get_brainstorming_agent_chain(llm),)[0])
chk(
    "get_brainstorming_mirror_descent_chain",
    lambda: (get_brainstorming_mirror_descent_chain(llm, 0.5),)[0],
)
chk(
    "get_brainstorming_synthesis_chain",
    lambda: (get_brainstorming_synthesis_chain(llm),)[0],
)
chk("get_brainstorming_seed_chain", lambda: (get_brainstorming_seed_chain(llm),)[0])
chk(
    "get_brainstorming_spanner_chain",
    lambda: (get_brainstorming_spanner_chain(llm),)[0],
)
chk("get_problem_summarizer_chain", lambda: (get_problem_summarizer_chain(llm),)[0])
chk(
    "get_brainstorming_polisher_chain",
    lambda: (get_brainstorming_polisher_chain(llm),)[0],
)
chk(
    "get_brainstorming_reframer_chain",
    lambda: (get_brainstorming_reframer_chain(llm),)[0],
)
chk(
    "get_brainstorming_epoch_map_chain",
    lambda: (get_brainstorming_epoch_map_chain(llm),)[0],
)
chk("get_task_master_chain", lambda: (get_task_master_chain(llm),)[0])
chk("get_seed_creator_chain", lambda: (get_seed_creator_chain(llm),)[0])
chk("get_mirror_descent_chain", lambda: (get_mirror_descent_chain(llm),)[0])
chk("get_mixing_chain", lambda: (get_mixing_chain(llm),)[0])
chk("get_followup_question_chain", lambda: (get_followup_question_chain(llm),)[0])

# Verify each chain is a Runnable
from langchain_core.runnables import Runnable


def verify_runnable():
    for fn in [
        get_input_spanner_chain(llm, 1.0, 1.0),
        get_synthesis_chain(llm),
        get_code_synthesis_chain(llm),
        get_problem_decomposition_chain(llm),
        get_problem_reframer_chain(llm),
        get_brainstorming_synthesis_chain(llm),
        get_task_master_chain(llm),
        get_mixing_chain(llm),
    ]:
        assert isinstance(fn, Runnable), f"{fn} is not Runnable"


chk("all chains are Runnable instances", verify_runnable)


# Verify ainvoke works
async def ainvoke_test():
    chain = get_synthesis_chain(llm)
    out = await chain.ainvoke(
        {"original_request": "x", "current_problem": "y", "agent_solutions": "z"}
    )
    assert out == "{}", f"got {out!r}"


chk("chain.ainvoke works", lambda: asyncio.run(ainvoke_test()))


# Test the duplicate import - opinion_synthesizer is defined in BOTH synthesis_chains.py and brainstorm_chains.py
# After BUG-5 fix, the brainstorm version was renamed to get_brainstorming_opinion_synthesizer_chain.
def dup_test():
    from deepthink.chains import get_opinion_synthesizer_chain as p
    from deepthink.chains import get_brainstorming_opinion_synthesizer_chain as q
    from deepthink.chains.synthesis_chains import get_opinion_synthesizer_chain as s
    from deepthink.chains.brainstorm_chains import (
        get_brainstorming_opinion_synthesizer_chain as b,
    )

    assert p is s, f"package version ({p}) should be synthesis_chains ({s})"
    assert q is b, f"package version ({q}) should be brainstorm_chains ({b})"


chk(
    "opinion_synthesizer & brainstorming_opinion_synthesizer resolve to correct modules (BUG-5 fix)",
    dup_test,
)

for name, status, err in results:
    line = f"  [{status}] {name}"
    if err:
        line += f" :: {err}"
    print(line)
ok = sum(1 for _, s, _ in results if s == "OK")
print(f"\nPHASE 3: {ok}/{len(results)} OK")
