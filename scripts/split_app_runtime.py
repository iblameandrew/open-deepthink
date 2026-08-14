#!/usr/bin/env python3
"""One-shot split: extract mocks / RAPTOR / legacy nodes from app.py.

Idempotent if the extracted files already exist *and* app.py already
re-exports them. Safe to re-run only on the pre-split app.py.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app.py"

# 1-indexed inclusive line ranges in the pre-split app.py
RAPTOR_RANGE = (166, 313)
MOCKS_RANGE = (315, 1326)
NODES_RANGE = (1362, 2457)
# Dead brainstorm/algorithm graph after early returns
DEAD_START = 3003  # "decomposed_problems_map = {}"
# Keep run_graph_background? It's only used by the dead path.
# We'll delete through the end of run_graph_background (starts 3531).
# After split we search for markers instead if line numbers drift.


RAG_HEADER = '''"""RAPTOR hierarchical index used by the web UI RAG paths."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.cluster import KMeans

from deepthink.runtime.bus import emit, emit_nowait


'''

MOCKS_HEADER = '''"""Mock LLMs for debug mode, tests, and `deepthink --debug`.

No network. Prompt-substring routers that keep CI and local loops free.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
from typing import Optional

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig


'''

NODES_HEADER = '''"""Legacy LangGraph node factories (inference replay + node unit tests).

Live Brainstorm / QDAD runs use ``deepthink.qnn`` and ``deepthink.qdad``.
These nodes remain for ``/run_inference_from_state`` and the phase-8/11 tests.
"""

from __future__ import annotations

import json
import random
import traceback
from typing import List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from deepthink.chains import (
    get_brainstorming_mirror_descent_chain,
    get_brainstorming_synthesis_chain,
    get_code_synthesis_chain,
    get_dense_spanner_chain,
    get_interrogator_chain,
    get_memory_summarizer_chain,
    get_module_card_chain,
    get_opinion_synthesizer_chain,
    get_paper_formatter_chain,
    get_perplexity_heuristic_chain,
    get_problem_decomposition_chain,
    get_problem_reframer_chain,
    get_rag_chat_chain,
    get_synthesis_chain,
)
from deepthink.rag import RAPTOR
from deepthink.runtime.bus import emit
from deepthink.self_attention import compute_self_attention
from deepthink.state import GraphState
from deepthink.utils import clean_and_parse_json, execute_code_in_sandbox


'''


def _slice(lines: list[str], start: int, end: int) -> str:
    return "".join(lines[start - 1 : end])


def _rewrite_log(src: str) -> str:
    src = src.replace("await log_stream.put(", "await emit(")
    src = src.replace("loop.create_task(log_stream.put(msg))", "emit_nowait(msg)")
    return src


def main() -> None:
    text = APP.read_text(encoding="utf-8")
    if "from deepthink.mocks import" in text and (ROOT / "deepthink" / "mocks.py").is_file():
        print("app.py already split; skipping extract")
        return

    lines = text.splitlines(keepends=True)
    rag_body = _rewrite_log(_slice(lines, *RAPTOR_RANGE))
    mocks_body = _slice(lines, *MOCKS_RANGE)
    nodes_body = _rewrite_log(_slice(lines, *NODES_RANGE))

    (ROOT / "deepthink" / "rag.py").write_text(RAG_HEADER + rag_body, encoding="utf-8")
    (ROOT / "deepthink" / "mocks.py").write_text(MOCKS_HEADER + mocks_body, encoding="utf-8")
    (ROOT / "deepthink" / "runtime" / "nodes.py").write_text(
        NODES_HEADER + nodes_body, encoding="utf-8"
    )
    (ROOT / "deepthink" / "runtime" / "__init__.py").write_text(
        '''"""Web-runtime helpers (log bus + leftover LangGraph nodes)."""

from deepthink.runtime.bus import emit, emit_nowait, get_log_queue, set_log_queue
from deepthink.runtime.nodes import (
    create_agent_node,
    create_archive_epoch_outputs_node,
    create_code_execution_node,
    create_final_harvest_node,
    create_metrics_node,
    create_reframe_and_decompose_node,
    create_synthesis_node,
    create_update_agent_prompts_node,
    create_update_rag_index_node,
)

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
''',
        encoding="utf-8",
    )
    print("wrote deepthink/rag.py, mocks.py, runtime/nodes.py")


if __name__ == "__main__":
    main()
