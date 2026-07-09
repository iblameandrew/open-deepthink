"""
Brainstorming mode chain factories.

The expert panel is a full Qualitative Neural Network (QNN), not a flat
panel of static experts. Chains implement the QNN algorithm step-by-step:

  0. Brief (summarizer / complexity context)
  1. Topology (complexity estimator → L × W × E)
  2. Guiding concepts (seed chain)
  3. Personas (spanner chain, layer 0 diverge / deeper converge)
  4. Epoch loop: layered forward → epoch map → Mirror Descent → reframe
  5. Solution-Space Report (final synthesis + polisher)
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def get_complexity_estimator_chain(llm):
    """QNN Step 1 (Auto): estimate topology L × W × E from problem complexity.

    Manual/Massive UI mode bypasses this and uses user-specified dimensions.
    """
    prompt = ChatPromptTemplate.from_template("""
You size a Qualitative Neural Network (QNN) expert panel for AUTO mode.
Recommend a *small* affordable topology. Users can override with Manual/Massive for huge nets.

User Input:
---
{user_input}
---

Prior Conversation (if any):
---
{prior_conversation}
---

Document Context (if any):
---
{document_context}
---

Score complexity 1–10 using: distinct domains, depth needed, conflicting perspectives,
technical vs conceptual load, prior conversation continuity, document complexity.

Map score to topology (prefer small for AUTO):
- score 1–3 → layers 2, width 2, epochs 1
- score 4–6 → layers 3, width 3, epochs 2
- score 7–8 → layers 3, width 4, epochs 2
- score 9–10 → layers 4, width 5, epochs 3

Respond with JSON only:
{{
    "complexity_score": <1-10 integer>,
    "recommended_layers": <2-5 integer>,
    "recommended_epochs": <1-3 integer>,
    "recommended_width": <2-5 integer>,
    "reasoning": "<brief explanation including unstick vs enrich if relevant>"
}}
""")
    return prompt | llm | StrOutputParser()


def get_expert_reflection_chain(llm, expert_name, expert_specialty, expert_emoji):
    """Legacy single-expert reflection (tests / fallback). Prefer full QNN spanner+agent path."""
    prompt = ChatPromptTemplate.from_template(f"""
You are {expert_name} ({expert_emoji}), an expert in {expert_specialty}.

Your role is to provide your unique perspective on the user's question or idea, filtered through your specialty of {expert_specialty}.

User's Question/Idea:
---
{{user_input}}
---

Previous Expert Opinions (if any):
---
{{previous_opinions}}
---

Provide your thoughtful reflection from your area of expertise. Be concise but insightful.
Focus on what your specialty ({expert_specialty}) uniquely contributes to this discussion.

Your response should be 2-4 sentences of substantive analysis.
""")
    return prompt | llm | StrOutputParser()


def get_brainstorming_opinion_synthesizer_chain(llm):
    """Legacy flat-panel synthesizer. Prefer get_brainstorming_synthesis_chain (QNN)."""
    prompt = ChatPromptTemplate.from_template("""
You are a master synthesizer. You have received opinions from multiple experts on a user's question.
Your task is to synthesize these diverse perspectives into a coherent, actionable response.

Original User Question:
---
{user_input}
---

Expert Opinions:
---
{all_opinions}
---

Create a synthesized response that:
1. Identifies key areas of agreement
2. Acknowledges valuable tensions or trade-offs
3. Provides a balanced, actionable conclusion
4. Is concise but comprehensive (3-5 sentences)

Synthesized Response:
""")
    return prompt | llm | StrOutputParser()


# ==================== QNN BRAINSTORMING CHAINS ====================


def get_brainstorming_seed_chain(llm):
    """QNN Step 2: seed verbs + nouns from the problem space (same spirit as algorithm mode).

    Algorithm mode uses get_seed_generation_chain for abstract verbs, then samples
    per-agent guiding_words for the input spanner. Brainstorm uses the same idea:
    a pool of linguistically loaded verbs AND nouns — some tight to the problem,
    some from far semantic fields — so personas span the problem space rather than
    a flat list of expert labels.
    """
    prompt = ChatPromptTemplate.from_template("""
You are the QNN Seed Generator (same role as Algorithm Mode seed generation).

Given the problem, generate exactly {word_count} unique seed words that will become
guiding_words for spanning agent personas.

Problem:
---
{problem}
---

Requirements (match original algorithm spanning):
1. About half should be **verbs** — abstract, linguistically loaded, related to the problem.
2. About half should be **nouns** — entities, forces, structures, or domains in/near the problem space.
3. Include words tightly related to the problem AND words from **far semantic fields** of knowledge
   (so the network can invent unexpected but useful specializations).
4. Single tokens only (no phrases). Unique. No filler (the, and, solve, problem).

Output ONLY a single space-separated string of words.
Example: distill reconverge entangle ownership latch invariant horizon entropy braid crystallize
""")
    return prompt | llm | StrOutputParser()


def get_brainstorming_spanner_chain(llm):
    """QNN Step 3: span one persona from guiding_words (verbs/nouns), like input_spanner.

    Mirrors algorithm-mode get_input_spanner_chain: career + attributes shaped by
    guiding_words, skills as extensions — plus layer diverge/converge role for brainstorm.
    """
    prompt = ChatPromptTemplate.from_template("""
You are a QNN Node Generator / Agent Architect (same spanning method as Algorithm Mode input spanner).

You design one specialized agent by blending **guiding_words** (verbs and nouns sampled from
the problem space) into a realistic professional persona. Do NOT invent a generic "Senior Engineer".

Topic / sub-problem:
---
{problem}
---

Guiding words (seed verbs + nouns for THIS column — treat as the agent's word-vector):
---
{guiding_words}
---

QNN Position: Layer {layer_index}, Node {node_index}

Document Context (optional domain grounding):
---
{document_context}
---

Layer roles (mandatory):
- Layer 0: DIVERGENT — breadth, "what if", unusual strategies from your word-vector.
- Layer 1+: CONVERGENT / CRITICAL — critique, refine, stress-test upstream using your word-vector.

Spanning procedure (same as original algorithm):
1. **Career** — realistic professional role specialized for the problem, colored by guiding_words.
2. **Attributes** — derive a personality/cognition profile whose descriptors are clearly
   influenced by the guiding_words (verbs → action style; nouns → domain objects/forces).
3. **Skills** — 4–6 methodologies that are logical extensions of the Career + guiding_words.
4. The agent must answer only through its specializations; map strategies with mechanisms
   and falsifiers; do NOT ship production patches.

Respond with JSON only:
{{
    "name": "<Creative human name>",
    "specialty": "<Niche career/specialty derived from guiding_words + problem>",
    "emoji": "<Emoji>",
    "guiding_words": "{guiding_words}",
    "attributes": ["<12 short attribute phrases shaped by the guiding words>"],
    "skills": ["<4-6 practical skills/methodologies>"],
    "system_prompt": "<Second-person system prompt: who they are, career, how guiding_words shape their cognition, layer goal (diverge vs converge), mandate to produce strategy angles with falsifiers not production patches. 4-8 sentences.>"
}}
""")
    return prompt | llm | StrOutputParser()


def get_brainstorming_agent_chain(llm):
    """Optional standalone QNN agent reflection chain (prompt-compatible tests)."""
    prompt = ChatPromptTemplate.from_template("""
<System>
{system_prompt}
</System>

Concept to Explore / User Input:
---
{input}
---

Prior Conversation Context:
---
{prior_conversation}
---

Document Context (Reference Material):
---
{document_context}
---

Your Task (QNN node reflection):
Reflect on the input from your persona only.
- Do NOT write production patches or full file diffs.
- Do NOT converge prematurely on "the" fix.
- Explore "why" and "what if".
Provide a unique strategic angle, why it might break the impasse (or enrich the artifact),
what evidence would confirm or kill it, and risks.

Your Reflection:
""")
    return prompt | llm | StrOutputParser()


def get_brainstorming_mirror_descent_chain(llm, learning_rate):
    """QNN Step 4C: evolve personas between epochs (qualitative Mirror Descent)."""
    prompt = ChatPromptTemplate.from_template(f"""
You are a Persona Evolver for a QNN. Rewrite the agent's system prompt based on last output.

Original System Prompt:
---
{{current_prompt}}
---

Last Output from Agent:
---
{{last_output}}
---

Mutation rules:
- Too generic → narrow specialty; name concrete subsystems from the problem if present.
- Too patch-shaped / algorithmic → push toward mechanisms, invariants, failure modes, falsifiers.
- Strong unique insight → reinforce that niche; demand sharper falsifiers next epoch.
- Contradicted or thin → require explicit dissent with evidence or reconciling upstream views.
- Learning rate ({learning_rate}): 0.0 = almost no change, 0.5 = moderate, 2.0 = radical reinvention.
- Keep layer role (diverge vs converge). Keep "no production patches" discipline.

Output ONLY the new system prompt. No explanation.
""")
    return prompt | llm | StrOutputParser()


def get_brainstorming_reframer_chain(llm):
    """QNN Step 4D: harder thinking challenge for next epoch (ground truth unchanged)."""
    prompt = ChatPromptTemplate.from_template("""
You are the QNN Problem Re-framer for brainstorming mode.

The expert network produced an intermediate solution-space map. Your job is to formulate a
*harder, more advanced thinking challenge* for the next epoch — NOT a different product goal.

Rules:
1. Preserve the user's original success criteria and product intent (ground truth).
2. Remove a simplifying assumption the network may have leaned on.
3. Force consideration of scale, concurrency, partial failure, migration, evaluation, or adversarial use as appropriate.
4. The reframe is a *thinking tool* for the next forward pass only.
5. Do not ask for a production patch; ask for deeper strategy exploration.

Original Request (ground truth — do not replace):
---
{original_request}
---

Current Thinking Challenge:
---
{current_problem}
---

Latest Epoch Map / Synthesis (may be partial):
---
{final_solution}
---

Prior Conversation (optional):
---
{prior_conversation}
---

Respond with JSON only:
{{
    "new_problem": "<harder thinking challenge for next epoch, still grounded in original request>"
}}
""")
    return prompt | llm | StrOutputParser()


def get_brainstorming_epoch_map_chain(llm):
    """QNN Step 4B intermediate: compact epoch map (not the final user answer)."""
    prompt = ChatPromptTemplate.from_template("""
You are the QNN Epoch Cartographer.
After one forward pass of a layered expert network, produce a compact EPOCH MAP.

Original Request:
---
{original_request}
---

Thinking Challenge (this epoch):
---
{current_problem}
---

Expert Reflections (this epoch and history):
---
{agent_solutions}
---

Produce a compact markdown map with exactly these sections:
1. **Clusters of agreement**
2. **Productive tensions / trade-offs**
3. **Novel mechanisms** (angles nobody started with)
4. **Dead ends** (collapsed under critique)
5. **Open questions / missing evidence**

Keep it dense (roughly half a page). Do NOT write a polished final answer.
Do NOT invent files or APIs not supported by the reflections.
""")
    return prompt | llm | StrOutputParser()


def get_brainstorming_synthesis_chain(llm):
    """QNN Step 5 core: draft Solution-Space Report from evolved expert history."""
    prompt = ChatPromptTemplate.from_template("""
You are a Master Synthesizer for a Qualitative Neural Network (QNN) brainstorm.

You have layered, multi-epoch expert reflections. Produce a SOLUTION-SPACE REPORT draft —
a map of divergent strategies — NOT a single premature patch and NOT a flat "expert panel average".

Core Concept & Context:
---
{original_request}
---

Prior Conversation Context:
---
{prior_conversation}
---

Document / Reference Material:
---
{document_context}
---

Expert Reflections (History of Ideas across layers/epochs):
---
{agent_solutions}
---

CRITICAL INSTRUCTIONS:
1. Incorporate document/reference details when provided.
2. Build only on Expert Reflections — cite agent ids or specialties when possible.
3. Prefer 3–7 distinct strategies with mechanisms and falsifiers over one vague winner.
4. Include dead ends so the user does not re-circle.
5. Rank top 1–3 next probes (smallest experiment first), not full implementations.
6. Explicitly note: the QNN does not ship the fix; hand off to edit→run→debug (or design→spike).

Draft the report in markdown with clear headings.
""")
    return prompt | llm | StrOutputParser()


def get_problem_summarizer_chain(llm):
    """QNN Step 0: concise Impasse / Enrich brief from user request + documents."""
    prompt = ChatPromptTemplate.from_template("""
You are a Research Director briefing a QNN expert network.

They need the core of the user's request AND key constraints from reference documents —
not the full document text.

User's Request:
---
{user_input}
---

Reference Documents (Full Text):
---
{document_context}
---

Create a concise "QNN Brief" (1–2 paragraphs) covering:
1. Goal / problem statement (stuck debug vs feature enrichment if clear)
2. Critical facts, constraints, and relevant loci from documents
3. What success would look like
4. Any failed approaches or thin areas mentioned

QNN Brief:
""")
    return prompt | llm | StrOutputParser()


def get_brainstorming_polisher_chain(llm):
    """QNN Step 5 polish: format Solution-Space Report for the user."""
    prompt = ChatPromptTemplate.from_template("""
You are a master technical communicator for QNN brainstorm outputs.

Original User Request:
---
{original_request}
---

Initial Solution-Space Draft:
---
{initial_synthesis}
---

Transform the draft into a clear, engaging Solution-Space Report with this structure:

## 1. Impasse / Goal
Restate what we are stuck on or enriching, and why local approaches were thin.

## 2. Topology & Process
If the draft lacks topology detail, note that a layered multi-epoch QNN expert network produced this map.

## 3. Divergent Strategy Map
For each promising strategy (typically 3–7): Name, Mechanism, Why it might work,
Falsifiers, Risks, First probe, Confidence (Low/Med/High).

## 4. Dead Ends
Angles discarded and why.

## 5. Recommended Next Steps (Handoff)
Top 1–3 strategies ordered for the grounded coding loop:
probe/instrument → minimal spike or failing test → implement only after a probe succeeds.

Close with: **The QNN does not ship the fix. Pick a direction, then resume edit → run → debug.**

Use markdown. Keep technical precision. Warm professional tone. Do not invent unsupported claims.
""")
    return prompt | llm | StrOutputParser()
