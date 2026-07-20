"""
QDAD chain factories — qualitative diffusion as language computation.

Philosophy (enforced in every prompt):
  • Language is the computational medium.
  • Nouns and verbs act as orthogonal basis directions.
  • High temperature = controlled qualitative noise.
  • Critic agents = qualitative reverse diffusion / score matching.
  • Vague aesthetic prompt → concrete, buildable app specification
    (as Midjourney turns a vague prompt into an image).
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


PHILOSOPHY_PREAMBLE = """
QUALITATIVE COMPUTATION CONTRACT
- Language is the computational medium (not numbers).
- Nouns and verbs are orthogonal basis directions of feature space.
- A feature is a language-vector at the intersection of one noun and one verb.
- High temperature injects controlled qualitative noise (forward diffusion).
- Critics perform reverse diffusion / score matching in language space.
"""


def get_qdad_foundation_chain(llm):
    """Phase 0: shared qualitative basis — N nouns + N verbs."""
    prompt = ChatPromptTemplate.from_template(
        """You are the QDAD Foundation Generator for a Qualitative Diffusion App Designer.
"""
        + PHILOSOPHY_PREAMBLE
        + """
Your task is to invent a shared qualitative basis for an N × N feature grid.

User prompt (Midjourney-style aesthetic / app intent):
---
{user_prompt}
---

Generate exactly {n} distinct nouns and exactly {n} distinct verbs.
They are orthogonal basis directions:
- Nouns = object / substance / place / affordance axes (rows)
- Verbs = action / process / transformation axes (columns)
- Concrete enough to ground features; diverse; mutually distinct
- Together they should span the aesthetic space of the user's intent

Output ONLY a single valid JSON object with exactly these keys:
{{
  "nouns": ["noun1", "noun2", ...],
  "verbs": ["verb1", "verb2", ...]
}}
Each array length must be exactly {n}. No markdown fences. No commentary. JSON only.
"""
    )
    return prompt | llm | StrOutputParser()


def get_qdad_noise_chain(llm):
    """Phase 2: forward diffusion — one noisy feature per grid cell."""
    prompt = ChatPromptTemplate.from_template(
        """You are FeatureAgent_{i}_{j}.
"""
        + PHILOSOPHY_PREAMBLE
        + """
Your unique qualitative signature is the orthogonal product of:
  noun basis "{noun}"  ×  verb basis "{verb}"

Your sole purpose is to invent exactly ONE concrete, implementable feature for an application.
The feature must feel like a natural expression of the interaction between "{noun}" and "{verb}"
in the context of the user's high-level intent.

User high-level intent:
---
{user_prompt}
---

FORWARD DIFFUSION (noise induction):
Invent one wild, imperfect, slightly hallucinated but still related feature.
Embrace controlled qualitative noise. Rough edges, odd metaphors, and over-ambition are allowed —
they are the language analogue of Gaussian noise. Still stay in the orbit of the user's intent.

Output ONLY the feature description (2–6 sentences). No JSON. No headings. No preamble.
"""
    )
    return prompt | llm | StrOutputParser()


def get_qdad_critic_chain(llm):
    """Phase 3: reverse diffusion — critic score-matches language noise away."""
    prompt = ChatPromptTemplate.from_template(
        """You are CriticAgent_{i}_{j}.
"""
        + PHILOSOPHY_PREAMBLE
        + """
You share the exact same qualitative signature as the feature agent:
  noun basis "{noun}"  ×  verb basis "{verb}"

You are the inverse of noise induction: qualitative reverse diffusion / score matching.
Clean up imperfections, remove contradictions, sharpen the original intent, and make the
feature coherent, useful, and implementable — while remaining a true expression of
"{noun}" + "{verb}".

User high-level intent:
---
{user_prompt}
---

Denoising step: {step} of {total_steps}
(Earlier steps remove gross noise; later steps fine-tune fidelity to intent.)

Current noisy / partially denoised feature:
---
{current_feature}
---

Output ONLY the refined feature description (2–6 sentences). No JSON. No headings. No preamble.
"""
    )
    return prompt | llm | StrOutputParser()


def get_qdad_synthesis_chain(llm):
    """Phase 4: collapse the clean matrix into one agentic coding prompt."""
    prompt = ChatPromptTemplate.from_template(
        """You are the QDAD Synthesizer Agent for the Qualitative Diffusion App Designer.
"""
        + PHILOSOPHY_PREAMBLE
        + """
You receive the original user prompt and the full N × N matrix of clean features produced by
qualitative diffusion (noun × verb feature agents after iterative reverse diffusion).

Your job is the Midjourney analogue of "decode latent → image": decode the clean feature
matrix into one high-quality, structured agentic coding prompt for any app-building agent
(Grok-Build, Claude Artifacts, Cursor, etc.).

Original user prompt:
---
{user_prompt}
---

Nouns (row basis): {nouns}
Verbs (column basis): {verbs}

Clean feature matrix (each cell is a language-vector on a noun×verb basis pair):
---
{feature_matrix}
---

You MUST output in this exact markdown format (no extra wrapper text before the heading):

# App Build Prompt

## High-Level Vision
[1-2 sentence summary]

## Core Features (synthesized & prioritized from the diffusion matrix)
1. ...
2. ...
...

## Technical Architecture Suggestions
- ...

## UI/UX Direction
- ...

## Non-Functional Requirements
- ...

## Implementation Notes for the Coding Agent
- Build this as a complete, runnable application.
- Prefer modern, clean tech (React/Next.js + Tailwind, or Streamlit, or whatever fits best).
- Make it beautiful and immediately usable.

Style:
- Deduplicate and prioritize; merge related matrix cells into strong product features.
- Features should feel like coherent expressions of user intent, not a random laundry list.
- Be concrete and implementable.
"""
    )
    return prompt | llm | StrOutputParser()
