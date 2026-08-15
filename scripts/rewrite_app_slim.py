#!/usr/bin/env python3
"""Replace extracted bodies in app.py with imports; drop dead graph path."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app.py"

MARKER = "from deepthink.mocks import"


def main() -> None:
    text = APP.read_text(encoding="utf-8")
    if MARKER in text:
        print("app.py already rewritten")
        return

    lines = text.splitlines(keepends=True)

    # Cut ranges (1-indexed inclusive) from the original file
    # Keep 1-165 (through sessions comment / active_distillation start)
    # Drop RAPTOR 166-313, mocks 315-1326, GraphState 1328-1359, nodes 1362-2457
    # Keep 2459+ (get_index) but drop dead graph 3003-3569

    head = "".join(lines[0:165])  # through line 165 inclusive (index 164)
    tail_start = "".join(lines[2458:3002])  # get_index through brainstorm return
    # After brainstorm JSONResponse return at ~3001, skip dead path until
    # export_qnn at 3573 (index 3572)
    rest = "".join(lines[3572:])

    extra_imports = """from deepthink.cost import estimate_qdad_cost, estimate_qnn_cost
from deepthink.mocks import CoderMockLLM, DistillationMockLLM, MockLLM
from deepthink.rag import RAPTOR, RAPTORRetriever
from deepthink.runtime.bus import set_log_queue
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
from deepthink.sessions import SessionStore
from deepthink.state import GraphState

"""

    # Insert extra imports after self_attention import if present
    needle = "from deepthink.self_attention import compute_self_attention\n"
    if needle in head:
        head = head.replace(needle, needle + extra_imports)
    else:
        head = extra_imports + head

    # Replace in-memory sessions dict
    head = head.replace("sessions = {}\n", "sessions = SessionStore()\n")
    head += "set_log_queue(log_stream)\n\n"

    # After brainstorm/qdad early returns, reject unknown modes
    reject = """
    return JSONResponse(
        content={
            "message": (
                f"Unsupported mode '{mode}'. "
                "Use 'brainstorm' or 'app_slot_machine'."
            )
        },
        status_code=400,
    )


"""

    new = head + tail_start + reject + rest
    APP.write_text(new, encoding="utf-8")
    print(f"rewrote app.py ({len(new.splitlines())} lines)")


if __name__ == "__main__":
    main()
