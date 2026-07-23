"""Phase 7: CoderMockLLM and DistillationMockLLM - check the prompt pattern recognition.
After bug-fix BUG-1, the patterns match the *current* chain prompts.
"""

import sys, asyncio, traceback

sys.path.insert(0, r"C:\Users\def78\smenos\local-deepthink")
import importlib

app_mod = importlib.import_module("app")
from app import CoderMockLLM, DistillationMockLLM

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


# === DistillationMockLLM checks ===
DMLL = DistillationMockLLM


async def t1():
    # After BUG-1 fix, the pattern should match "Socratic Task Master" + "Knowledge Distillation Network"
    out = await DMLL().ainvoke(
        "You are the Socratic Task Master of a Knowledge Distillation Network.\nYour goal is to orchestrate a logical inquiry."
    )
    content = out.content if hasattr(out, "content") else out
    import json

    d = json.loads(content)
    assert "sub_questions" in d
    assert len(d["sub_questions"]) == 12


chk("DistillationMockLLM recognizes Task Master pattern (FIXED)", t1)


async def t2():
    out = await DMLL().ainvoke(
        "You are the Seed Creator (The Dialectic Synthesizer).\nYour role is to evolve the knowledge graph"
    )
    content = out.content if hasattr(out, "content") else out
    import json

    d = json.loads(content)
    assert "new_topics" in d
    assert len(d["new_topics"]) == 12


chk("DistillationMockLLM recognizes Seed Creator pattern (FIXED)", t2)


async def t3():
    out = await DMLL().ainvoke(
        "You are the Socratic Task Master.\nWe are deepening our inquiry in a new Epoch"
    )
    content = out.content if hasattr(out, "content") else out
    import json

    d = json.loads(content)
    assert "new_questions" in d


chk("DistillationMockLLM recognizes Followup pattern (FIXED)", t3)


async def t4():
    out = await DMLL().ainvoke(
        "You are the Mirror Descent Agent.\nYour task is to evaluate"
    )
    content = out.content if hasattr(out, "content") else out
    import json

    d = json.loads(content)
    assert d["difficulty"] in ("Easy", "Hard")


chk("DistillationMockLLM recognizes Mirror Descent pattern", t4)


async def t5():
    out = await DMLL().ainvoke(
        "You are a Mixing Agent.\nYour goal is to spawn a new Child agent"
    )
    content = out.content if hasattr(out, "content") else out
    import json

    d = json.loads(content)
    assert "new_system_prompt" in d
    assert "new_attributes" in d
    assert "new_skills" in d


chk("DistillationMockLLM recognizes Mixing Agent pattern", t5)


async def t6():
    out = await DMLL().ainvoke("Answer your sub-question deeply.\nBe thorough.")
    content = out.content if hasattr(out, "content") else out
    assert isinstance(content, str)
    assert len(content) > 0


chk("DistillationMockLLM returns string for sub-question answer", t6)


async def t7():
    out = await DMLL().ainvoke("Estimate the perplexity score for the following batch")
    content = out.content if hasattr(out, "content") else out
    import json

    d = json.loads(content)
    assert "score" in d


chk("DistillationMockLLM returns perplexity score", t7)


async def t8():
    out = await DMLL().ainvoke("Random unknown prompt that has no match anywhere")
    content = out.content if hasattr(out, "content") else out
    import json

    d = json.loads(content)
    assert "error" in d


chk("DistillationMockLLM returns error JSON for unrecognized prompt", t8)

# === CoderMockLLM checks ===
CMLL = CoderMockLLM


async def t9():
    out = await CMLL().ainvoke(
        "You are a helpful AI assistant with access to a knowledge base.\nUse the following context"
    )
    content = out.content if hasattr(out, "content") else out
    assert isinstance(content, str)


chk("CoderMockLLM RAG chat pattern", t9)


async def t10():
    out = await CMLL().ainvoke(
        "Analyze the following text and determine if it contains any programming code"
    )
    content = out.content if hasattr(out, "content") else out
    assert content == "yes"


chk("CoderMockLLM code detector returns 'yes'", t10)


async def t11():
    out = await CMLL().ainvoke(
        "Analyze the following user request. Is this primarily a request for code"
    )
    content = out.content if hasattr(out, "content") else out
    assert content == "yes"


chk("CoderMockLLM request_is_code returns 'yes'", t11)


async def t12():
    out = await CMLL().ainvoke(
        "You are a master strategist and problem decomposer. Break down a complex problem"
    )
    content = out.content if hasattr(out, "content") else out
    import json

    d = json.loads(content)
    assert "sub_problems" in d
    # After BUG-2 fix, the mock should default to 4 when no count is in prompt
    assert len(d["sub_problems"]) == 4


chk("CoderMockLLM problem decomposer (4 by default)", t12)


async def t12b():
    out = await CMLL().ainvoke(
        "You are a master strategist and problem decomposer.\nTotal number of subproblems to generate: 7"
    )
    content = out.content if hasattr(out, "content") else out
    import json

    d = json.loads(content)
    assert "sub_problems" in d
    assert len(d["sub_problems"]) == 7, f"expected 7, got {len(d['sub_problems'])}"


chk("CoderMockLLM problem decomposer respects requested count (BUG-2 fix)", t12b)


async def t13():
    out = await CMLL().ainvoke("You are a strategic problem re-framer")
    content = out.content if hasattr(out, "content") else out
    import json

    d = json.loads(content)
    assert "new_problem" in d


chk("CoderMockLLM re-framer returns new_problem", t13)


async def t14():
    out = await CMLL().ainvoke(
        "You are an expert code synthesis agent.\nSynthesize the final code"
    )
    content = out.content if hasattr(out, "content") else out
    assert "```python" in content


chk("CoderMockLLM code synthesis returns Python code block", t14)


async def t15():
    out = await CMLL().ainvoke(
        "Analyze the complexity of the following user input/question for a brainstorming session"
    )
    content = out.content if hasattr(out, "content") else out
    import json

    d = json.loads(content)
    assert "complexity_score" in d
    assert "recommended_layers" in d
    assert "recommended_width" in d


chk("CoderMockLLM complexity estimator returns JSON", t15)


async def t16():
    out = await CMLL().ainvoke("You are a QNN Node Generator. Create an expert persona")
    content = out.content if hasattr(out, "content") else out
    import json

    d = json.loads(content)
    assert "name" in d and "specialty" in d and "emoji" in d and "system_prompt" in d


chk("CoderMockLLM QNN node generator returns persona JSON", t16)


async def t17():
    out = await CMLL().ainvoke(
        "You are a Concept Spanner.\nGenerate exactly 3 distinct concepts"
    )
    content = out.content if hasattr(out, "content") else out
    assert isinstance(content, str)
    assert len(content.split()) >= 1


chk("CoderMockLLM Concept Spanner returns space-separated concepts", t17)


async def t18():
    out = await CMLL().ainvoke("Reflect on the input from your specific persona")
    content = out.content if hasattr(out, "content") else out
    assert isinstance(content, str)
    assert len(content) > 0


chk("CoderMockLLM brainstorming agent reflection", t18)


async def t19():
    chunks = []
    async for c in CMLL().astream(
        "You are a helpful AI assistant with access to a knowledge base."
    ):
        chunks.append(c)
    assert len(chunks) > 0


chk("CoderMockLLM astream yields chunks", t19)


async def t20():
    chunks = []
    async for c in CMLL().astream("You are an expert code synthesis agent"):
        chunks.append(c)
    assert len(chunks) >= 1


chk("CoderMockLLM astream fallback for non-streamable prompt", t20)


# === QDAD / App Slot Machine mock patterns ===
async def t21():
    import json

    out = await CMLL().ainvoke(
        "You are the QDAD Foundation Generator for a Qualitative Diffusion App Designer.\n"
        "Generate exactly 4 distinct nouns and exactly 4 distinct verbs."
    )
    content = out.content if hasattr(out, "content") else out
    d = json.loads(content)
    assert "nouns" in d and "verbs" in d
    assert len(d["nouns"]) == 4 and len(d["verbs"]) == 4


chk("CoderMockLLM QDAD foundation returns nouns/verbs JSON", t21)


async def t22():
    out = await CMLL().ainvoke(
        "You are FeatureAgent_0_1.\nFORWARD DIFFUSION (noise induction):\n"
        "Embrace controlled qualitative noise."
    )
    content = out.content if hasattr(out, "content") else out
    assert isinstance(content, str)
    assert "mock feature" in content.lower()
    assert not content.strip().startswith("{")


chk("CoderMockLLM QDAD noise agent returns plain-text feature", t22)


async def t23():
    out = await CMLL().ainvoke(
        "You are CriticAgent_1_2.\nYou are the inverse of noise induction: "
        "qualitative reverse diffusion / score matching."
    )
    content = out.content if hasattr(out, "content") else out
    assert isinstance(content, str)
    assert "refined mock feature" in content.lower()


chk("CoderMockLLM QDAD critic agent returns refined feature", t23)


async def t24():
    out = await CMLL().ainvoke(
        "You are the QDAD Synthesizer Agent for the Qualitative Diffusion App Designer."
    )
    content = out.content if hasattr(out, "content") else out
    assert "# App Build Prompt" in content
    assert "## High-Level Vision" in content


chk("CoderMockLLM QDAD synthesizer returns App Build Prompt", t24)


def t25():
    mock = CMLL()
    bound = mock.bind(temperature=1.3)
    assert bound is mock


chk("CoderMockLLM.bind(temperature=...) returns self (debug temp path)", t25)


for name, status, err in results:
    line = f"  [{status}] {name}"
    if err:
        line += f" :: {err}"
    print(line)
ok = sum(1 for _, s, _ in results if s == "OK")
print(f"\nPHASE 7: {ok}/{len(results)} OK")
