"""Web-runtime helpers (log bus + leftover LangGraph nodes)."""

from __future__ import annotations

from typing import Any

from deepthink.runtime.bus import emit, emit_nowait, get_log_queue, set_log_queue

__all__ = [
    "set_log_queue",
    "get_log_queue",
    "emit",
    "emit_nowait",
    "create_agent_node",
    "create_synthesis_node",
    "create_code_execution_node",
    "create_archive_epoch_outputs_node",
    "create_update_rag_index_node",
    "create_metrics_node",
    "create_reframe_and_decompose_node",
    "create_update_agent_prompts_node",
    "create_final_harvest_node",
]

_LAZY = {
    "create_agent_node": ("deepthink.runtime.nodes", "create_agent_node"),
    "create_synthesis_node": ("deepthink.runtime.nodes", "create_synthesis_node"),
    "create_code_execution_node": (
        "deepthink.runtime.nodes",
        "create_code_execution_node",
    ),
    "create_archive_epoch_outputs_node": (
        "deepthink.runtime.nodes",
        "create_archive_epoch_outputs_node",
    ),
    "create_update_rag_index_node": (
        "deepthink.runtime.nodes",
        "create_update_rag_index_node",
    ),
    "create_metrics_node": ("deepthink.runtime.nodes", "create_metrics_node"),
    "create_reframe_and_decompose_node": (
        "deepthink.runtime.nodes",
        "create_reframe_and_decompose_node",
    ),
    "create_update_agent_prompts_node": (
        "deepthink.runtime.nodes",
        "create_update_agent_prompts_node",
    ),
    "create_final_harvest_node": (
        "deepthink.runtime.nodes",
        "create_final_harvest_node",
    ),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        mod_name, attr = _LAZY[name]
        import importlib

        value = getattr(importlib.import_module(mod_name), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
