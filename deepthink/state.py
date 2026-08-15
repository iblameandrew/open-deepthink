"""
State definitions and type hints for DeepThink.
"""

from typing import Annotated, Any, TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict):
    """The state object passed through the graph during execution."""

    mode: str  # "app_slot_machine" or "brainstorm"
    modules: list[dict]
    synthesis_context_queue: list[str]
    agent_personas: dict
    previous_solution: str
    current_problem: str
    original_request: str
    decomposed_problems: dict[str, str]
    layers: list[dict]
    epoch: int
    max_epochs: int
    params: dict
    all_layers_prompts: list[list[str]]
    agent_outputs: Annotated[dict, lambda a, b: {**a, **b}]
    memory: Annotated[dict, lambda a, b: {**a, **b}]
    final_solution: dict
    perplexity_history: list[float]
    raptor_index: Any | None  # RAPTOR type - circular import prevention
    all_rag_documents: list[Document]
    academic_papers: dict | None
    is_code_request: bool
    session_id: str
    chat_history: list[dict]
    brainstorm_document_context: str
    brainstorm_prior_conversation: str
    brainstorm_problem_summary: str
    # Qualitative Self-Attention edges (brainstorm mode): agent_id → edge list
    attention_edges: Annotated[dict, lambda a, b: {**a, **b}]


# Legacy static labels kept for import/tests only.
# Live brainstorm expert panels are dynamically spanned QNN personas
# (seeds → L×W nodes with layer 0 diverge / deeper converge) — not this list.
BRAINSTORM_EXPERTS = [
    {"name": "Dr. Synthia Logic", "specialty": "Logical Analysis", "emoji": "🧠"},
    {"name": "Marcus Visionary", "specialty": "Creative Ideation", "emoji": "💡"},
    {"name": "Elena Pragmatic", "specialty": "Practical Implementation", "emoji": "🔧"},
    {"name": "Professor Critique", "specialty": "Devil's Advocate", "emoji": "🎭"},
    {"name": "Aria Empathy", "specialty": "Human-Centered Design", "emoji": "❤️"},
]
