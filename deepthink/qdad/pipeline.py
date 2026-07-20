"""
QDAD pipeline entrypoint — wire params → LangGraph → final_solution.

Model-agnostic: any LangChain-compatible chat model works (OpenAI, Anthropic
via OpenRouter, local llama.cpp, mocks, etc.).
"""
from __future__ import annotations

import json
import traceback
from typing import Any, Callable, Dict, List, Optional

from .graph import build_qdad_graph, draw_qdad_ascii
from .utils import clamp_params, enrich_prompt
from .state import QDADState


async def run_qdad_pipeline(
    llm,
    params: Dict[str, Any],
    user_prompt: str,
    session_id: str = "",
    synthesis_llm=None,
    document_context: str = "",
    chat_history: Optional[List[dict]] = None,
    log: Optional[Callable[[str], Any]] = None,
    session_store: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    Run the full Qualitative Diffusion App Designer pipeline.

    Returns the final_solution dict (mode=app_slot_machine).
    Side effects: optional log stream + session_store updates.
    """

    async def _log(msg: str):
        if log is None:
            return
        result = log(msg)
        if hasattr(result, "__await__"):
            await result

    n, noise_temp, denoising_steps, nv_temp = clamp_params(
        n=params.get("grid_size", params.get("n", 4)),
        noise_temp=params.get("temperature_scale", 1.3),
        denoising_steps=params.get("denoising_steps", 3),
        nv_temp=params.get("noun_verb_temperature", 0.6),
    )
    enriched = enrich_prompt(user_prompt, document_context, chat_history)

    await _log(
        f"--- [QDAD] App Slot Machine: N={n}, noise_T={noise_temp}, "
        f"steps={denoising_steps}, noun_verb_T={nv_temp} ---"
    )
    await _log(
        "LOG: [QDAD] Philosophy: language=medium; nouns/verbs=basis; "
        "high-T=noise; critics=reverse diffusion."
    )
    await _log(f"__start__ {draw_qdad_ascii()}")

    graph = build_qdad_graph(
        llm=llm,
        synthesis_llm=synthesis_llm or llm,
        log=log,
    )

    initial: QDADState = {
        "user_prompt": enriched,
        "session_id": session_id,
        "n": n,
        "noise_temperature": noise_temp,
        "noun_verb_temperature": nv_temp,
        "denoising_steps": denoising_steps,
        "denoise_step": 0,
        "nouns": [],
        "verbs": [],
        "grid": [],
        "features": [],
        "noisy_features": [],
        "clean_features": [],
        "matrices": {},
        "app_build_prompt": "",
        "final_solution": None,
        "phase_log": [],
    }

    try:
        # recursion_limit: foundation+grid+noise + denoise_steps + synth + margin
        limit = max(25, 10 + denoising_steps * 2)
        final_state: QDADState = await graph.ainvoke(
            initial, {"recursion_limit": limit}
        )

        # Persist intermediate matrices into session store for inspection
        if session_store is not None and session_id in session_store:
            session_store[session_id]["qdad_matrices"] = final_state.get(
                "matrices"
            ) or {}
            session_store[session_id]["nouns"] = final_state.get("nouns")
            session_store[session_id]["verbs"] = final_state.get("verbs")
            session_store[session_id]["final_solution"] = final_state.get(
                "final_solution"
            )
            session_store[session_id]["phase_log"] = final_state.get("phase_log")

        final_solution = final_state.get("final_solution") or {
            "mode": "app_slot_machine",
            "proposed_solution": "QDAD completed without a synthesizer payload.",
            "reasoning": "empty final_solution",
        }

        await _log(f"FINAL_ANSWER: {json.dumps(final_solution)}")
        await _log(
            "SUCCESS: App Slot Machine (QDAD) LangGraph execution completed."
        )
        return final_solution

    except Exception as e:
        await _log(f"QDAD Pipeline Error: {e}")
        await _log(traceback.format_exc())
        err = {
            "mode": "app_slot_machine",
            "proposed_solution": f"QDAD pipeline failed: {e}",
            "reasoning": str(e),
        }
        if session_store is not None and session_id in session_store:
            session_store[session_id]["final_solution"] = err
        await _log(f"FINAL_ANSWER: {json.dumps(err)}")
        return err
