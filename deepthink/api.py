"""
Public high-level API for open-deepthink algorithms.

Prefer importing from the top-level package::

    from deepthink import run_qnn, run_qdad, DistillationGraph, create_llm

This module re-exports stable entry points so the library surface stays
discoverable without pulling the optional web UI.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# ── Providers / config ───────────────────────────────────────────────────
from deepthink.config import Settings, get_settings, reload_settings

# ── Distillation ─────────────────────────────────────────────────────────
from deepthink.distillation import (
    DISTILLATION_ARCHETYPES,
    DistillationAgent,
    DistillationGraph,
)
from deepthink.providers import create_chat_model, create_llm

# ── QDAD ─────────────────────────────────────────────────────────────────
from deepthink.qdad import (
    QDADState,
    build_qdad_graph,
    default_qdad_params,
    run_qdad_pipeline,
)
from deepthink.qdad.utils import clamp_params as clamp_qdad_params

# ── QNN ──────────────────────────────────────────────────────────────────
from deepthink.qnn import QNNResult, default_qnn_params, run_qnn_pipeline

# ── Self-attention ───────────────────────────────────────────────────────
from deepthink.self_attention import (
    AttentionCandidate,
    AttentionEdge,
    compute_self_attention,
    format_attention_context,
)

# ── Shared utils / state ─────────────────────────────────────────────────
from deepthink.state import BRAINSTORM_EXPERTS, GraphState
from deepthink.utils import clean_and_parse_json, execute_code_in_sandbox

LogFn = Callable[[str], Any] | None


async def run_qnn(
    llm,
    user_prompt: str,
    params: dict[str, Any] | None = None,
    *,
    synthesis_llm=None,
    document_context: str = "",
    chat_history: list[dict] | None = None,
    log: LogFn = None,
    session_id: str = "",
    session_store: dict | None = None,
) -> dict[str, Any]:
    """
    Run a Qualitative Neural Network (brainstorm / solution-space) pipeline.

    Thin alias of :func:`deepthink.qnn.run_qnn_pipeline`.
    """
    return await run_qnn_pipeline(
        llm,
        user_prompt,
        params=params,
        synthesis_llm=synthesis_llm,
        document_context=document_context,
        chat_history=chat_history,
        log=log,
        session_id=session_id,
        session_store=session_store,
    )


async def run_qdad(
    llm,
    user_prompt: str,
    params: dict[str, Any] | None = None,
    *,
    synthesis_llm=None,
    document_context: str = "",
    chat_history: list[dict] | None = None,
    log: LogFn = None,
    session_id: str = "",
    session_store: dict | None = None,
) -> dict[str, Any]:
    """
    Run Qualitative Diffusion App Designer (App Slot Machine).

    Thin alias of :func:`deepthink.qdad.run_qdad_pipeline`.
    """
    p = {**default_qdad_params(), **(params or {})}
    return await run_qdad_pipeline(
        llm=llm,
        params=p,
        user_prompt=user_prompt,
        session_id=session_id,
        synthesis_llm=synthesis_llm,
        document_context=document_context,
        chat_history=chat_history,
        log=log,
        session_store=session_store,
    )


async def run_distillation(
    llm,
    anchor_question: str,
    topics: list[str] | None = None,
    *,
    token_budget: int | None = None,
    output_dir: str | None = None,
    debug_mode: bool = False,
    max_epochs: int | None = None,
    log: LogFn = None,
) -> DistillationGraph:
    """
    Run Knowledge Distillation until the token budget is exhausted (or max_epochs).

    Returns the finished :class:`DistillationGraph` (dataset + topology archive
    written under ``output_dir``).
    """
    cfg = get_settings()
    graph = DistillationGraph(
        llm=llm,
        topics=topics or ["reasoning", "decomposition", "critique"],
        anchor_question=anchor_question,
        token_budget=token_budget or cfg.distillation_token_budget,
        debug_mode=debug_mode,
        output_dir=output_dir or cfg.distillation_output_dir,
    )

    epoch = 0
    while graph.is_running:
        if max_epochs is not None and epoch >= max_epochs:
            graph.is_running = False
            break
        cont = await graph.run_epoch()
        epoch += 1
        if log is not None:
            msg = (
                f"LOG: [DISTILL] epoch={epoch} "
                f"tokens_in={graph.total_input_tokens} "
                f"tokens_out={graph.total_output_tokens} "
                f"qa_pairs={len(graph.distilled_data)}"
            )
            result = log(msg)
            if hasattr(result, "__await__"):
                await result
        if not cont:
            break

    return graph


__all__ = [
    # Algorithms
    "run_qnn",
    "run_qdad",
    "run_distillation",
    "run_qnn_pipeline",
    "run_qdad_pipeline",
    "default_qnn_params",
    "default_qdad_params",
    "clamp_qdad_params",
    "QNNResult",
    "QDADState",
    "build_qdad_graph",
    "DistillationGraph",
    "DistillationAgent",
    "DISTILLATION_ARCHETYPES",
    "compute_self_attention",
    "format_attention_context",
    "AttentionEdge",
    "AttentionCandidate",
    # Providers / config
    "create_llm",
    "create_chat_model",
    "get_settings",
    "reload_settings",
    "Settings",
    # Shared
    "GraphState",
    "BRAINSTORM_EXPERTS",
    "clean_and_parse_json",
    "execute_code_in_sandbox",
]
