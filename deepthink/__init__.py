"""
open-deepthink — Qualitative Neural Networks, Qualitative Diffusion, and
evolutionary knowledge distillation as a **Python library**.

Quick start (library)::

    import asyncio
    from deepthink import create_llm, run_qnn, run_qdad, DistillationGraph

    async def main():
        llm = create_llm()  # needs OPENROUTER_API_KEY or provider='llamacpp'
        report = await run_qnn(llm, "Break this ownership deadlock…")
        print(report["proposed_solution"])

    asyncio.run(main())

Web UI (optional)::

    open-deepthink          # or: python -m deepthink serve
"""

from __future__ import annotations

from typing import Any

__version__ = "0.3.0"
__release_name__ = "honest-engine"
__release_tag__ = "0.3.0"

# Lightweight imports — always safe, no LangChain graph spin-up
from deepthink.state import BRAINSTORM_EXPERTS, GraphState
from deepthink.utils import clean_and_parse_json, execute_code_in_sandbox

__all__ = [
    # Version
    "__version__",
    "__release_name__",
    "__release_tag__",
    # Algorithms (lazy)
    "run_qnn",
    "run_qdad",
    "run_distillation",
    "run_qnn_pipeline",
    "run_qdad_pipeline",
    "default_qnn_params",
    "default_qdad_params",
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
    "clamp_qnn_topology",
    "estimate_qnn_cost",
    "estimate_qdad_cost",
    "estimate_distillation_cost",
    "run_structural_eval",
    "CoderMockLLM",
    "DistillationMockLLM",
    # Providers / config (lazy)
    "create_llm",
    "create_chat_model",
    "get_settings",
    "reload_settings",
    "Settings",
    # Always available
    "GraphState",
    "BRAINSTORM_EXPERTS",
    "clean_and_parse_json",
    "execute_code_in_sandbox",
]

# Map public names → (module, attribute) for lazy loading
_LAZY: dict[str, tuple[str, str]] = {
    "run_qnn": ("deepthink.api", "run_qnn"),
    "run_qdad": ("deepthink.api", "run_qdad"),
    "run_distillation": ("deepthink.api", "run_distillation"),
    "run_qnn_pipeline": ("deepthink.qnn", "run_qnn_pipeline"),
    "run_qdad_pipeline": ("deepthink.qdad", "run_qdad_pipeline"),
    "default_qnn_params": ("deepthink.qnn", "default_qnn_params"),
    "default_qdad_params": ("deepthink.qdad", "default_qdad_params"),
    "QNNResult": ("deepthink.qnn", "QNNResult"),
    "QDADState": ("deepthink.qdad", "QDADState"),
    "build_qdad_graph": ("deepthink.qdad", "build_qdad_graph"),
    "DistillationGraph": ("deepthink.distillation", "DistillationGraph"),
    "DistillationAgent": ("deepthink.distillation", "DistillationAgent"),
    "DISTILLATION_ARCHETYPES": ("deepthink.distillation", "DISTILLATION_ARCHETYPES"),
    "compute_self_attention": ("deepthink.self_attention", "compute_self_attention"),
    "format_attention_context": (
        "deepthink.self_attention",
        "format_attention_context",
    ),
    "AttentionEdge": ("deepthink.self_attention", "AttentionEdge"),
    "AttentionCandidate": ("deepthink.self_attention", "AttentionCandidate"),
    "clamp_qnn_topology": ("deepthink.qnn", "clamp_qnn_topology"),
    "estimate_qnn_cost": ("deepthink.cost", "estimate_qnn_cost"),
    "estimate_qdad_cost": ("deepthink.cost", "estimate_qdad_cost"),
    "estimate_distillation_cost": ("deepthink.cost", "estimate_distillation_cost"),
    "run_structural_eval": ("deepthink.eval_structural", "run_structural_eval"),
    "CoderMockLLM": ("deepthink.mocks", "CoderMockLLM"),
    "DistillationMockLLM": ("deepthink.mocks", "DistillationMockLLM"),
    "create_llm": ("deepthink.providers", "create_llm"),
    "create_chat_model": ("deepthink.providers", "create_chat_model"),
    "get_settings": ("deepthink.config", "get_settings"),
    "reload_settings": ("deepthink.config", "reload_settings"),
    "Settings": ("deepthink.config", "Settings"),
}


def __getattr__(name: str) -> Any:
    """Lazy-load algorithm / provider symbols to keep ``import deepthink`` light."""
    if name in _LAZY:
        mod_name, attr = _LAZY[name]
        import importlib

        mod = importlib.import_module(mod_name)
        value = getattr(mod, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(__all__))
