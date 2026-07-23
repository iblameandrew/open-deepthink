"""
State definitions and type hints for DeepThink.
"""
from typing import TypedDict, Annotated, List, Optional, Dict, Any
from langchain_core.documents import Document


class GraphState(TypedDict):
    """The state object passed through the graph during execution."""
    mode: str  # "app_slot_machine" or "brainstorm"
    modules: List[dict]
    synthesis_context_queue: List[str] 
    agent_personas: dict
    previous_solution: str
    current_problem: str
    original_request: str
    decomposed_problems: Dict[str, str]
    layers: List[dict]
    epoch: int
    max_epochs: int
    params: dict
    all_layers_prompts: List[List[str]]
    agent_outputs: Annotated[dict, lambda a, b: {**a, **b}]
    memory: Annotated[dict, lambda a, b: {**a, **b}]
    final_solution: dict
    perplexity_history: List[float] 
    raptor_index: Optional[Any]  # RAPTOR type - circular import prevention
    all_rag_documents: List[Document]
    academic_papers: Optional[dict]
    is_code_request: bool
    session_id: str
    chat_history: List[dict]
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
