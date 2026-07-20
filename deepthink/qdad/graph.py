"""
LangGraph construction for QDAD (App Slot Machine).

Topology:
  START → foundation → grid → noise → denoise ⟲ → synthesize → END

Noise induction and each denoise round fan out over the N×N grid in parallel
inside their nodes (asyncio.gather). The denoise node loops until
denoising_steps is exhausted (conditional edge).
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from langgraph.graph import StateGraph, END, START

from .state import QDADState
from .nodes import (
    create_foundation_node,
    create_grid_node,
    create_noise_node,
    create_denoise_node,
    create_synthesis_node,
    should_continue_denoise,
)


def build_qdad_graph(
    llm,
    synthesis_llm=None,
    log: Optional[Callable[[str], Any]] = None,
):
    """Compile the QDAD qualitative diffusion graph.

    Args:
        llm: model-agnostic chat model for foundation / noise / critics
        synthesis_llm: optional separate model for final synthesis
        log: async or sync callable receiving log lines
    """
    synth = synthesis_llm or llm
    workflow = StateGraph(QDADState)

    workflow.add_node("foundation", create_foundation_node(llm, log=log))
    workflow.add_node("grid", create_grid_node(log=log))
    workflow.add_node("noise", create_noise_node(llm, log=log))
    workflow.add_node("denoise", create_denoise_node(llm, log=log))
    workflow.add_node("synthesize", create_synthesis_node(synth, log=log))

    workflow.add_edge(START, "foundation")
    workflow.add_edge("foundation", "grid")
    workflow.add_edge("grid", "noise")
    workflow.add_edge("noise", "denoise")
    workflow.add_conditional_edges(
        "denoise",
        should_continue_denoise,
        {
            "denoise": "denoise",
            "synthesize": "synthesize",
        },
    )
    workflow.add_edge("synthesize", END)

    return workflow.compile()


def draw_qdad_ascii() -> str:
    """Static ASCII of the QDAD graph for UI / logs."""
    return (
        "QDAD Qualitative Diffusion Graph\n"
        "================================\n"
        "START\n"
        "  │\n"
        "  ▼\n"
        "foundation   (Phase 0: N nouns + N verbs basis)\n"
        "  │\n"
        "  ▼\n"
        "grid         (Phase 1: N×N FeatureAgent assignment)\n"
        "  │\n"
        "  ▼\n"
        "noise        (Phase 2: parallel forward diffusion @ high T)\n"
        "  │\n"
        "  ▼\n"
        "denoise ⟲    (Phase 3: parallel reverse diffusion / critics)\n"
        "  │\n"
        "  ▼\n"
        "synthesize   (Phase 4: agentic coding prompt)\n"
        "  │\n"
        "  ▼\n"
        "END\n"
    )
