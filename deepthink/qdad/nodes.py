"""
QDAD LangGraph node factories.

Phase 0 Foundation → Phase 1 Grid → Phase 2 Noise → Phase 3 Denoise* → Phase 4 Synthesis
Noise and each denoise round parallelize the full N×N grid.
"""
from __future__ import annotations

import asyncio
import re
from typing import Any, Callable, Optional

from deepthink.utils import clean_and_parse_json
from deepthink.chains.qdad_chains import (
    get_qdad_foundation_chain,
    get_qdad_noise_chain,
    get_qdad_critic_chain,
    get_qdad_synthesis_chain,
)
from .utils import (
    llm_with_temperature,
    parse_word_list,
    empty_matrix,
    format_feature_matrix,
    ascii_grid_preview,
)
from .state import QDADState


LogFn = Optional[Callable[[str], Any]]


async def _emit(log: LogFn, msg: str):
    if log is None:
        return
    result = log(msg)
    if asyncio.iscoroutine(result):
        await result


def create_foundation_node(llm, log: LogFn = None):
    """Phase 0 — generate N nouns + N verbs (qualitative basis)."""

    async def foundation_node(state: QDADState) -> dict:
        n = state["n"]
        nv_temp = state["noun_verb_temperature"]
        user_prompt = state["user_prompt"]
        await _emit(
            log,
            f"LOG: [QDAD PHASE 0] Foundation — {n} nouns + {n} verbs "
            f"as orthogonal basis (T={nv_temp})...",
        )
        chain = get_qdad_foundation_chain(llm_with_temperature(llm, nv_temp))
        raw = await chain.ainvoke({"user_prompt": user_prompt, "n": n})
        parsed = clean_and_parse_json(raw) or {}
        nouns = parse_word_list(parsed.get("nouns", raw), n, "noun")
        verbs = parse_word_list(parsed.get("verbs", []), n, "verb")

        if all(v.startswith("verb") for v in verbs) and isinstance(raw, str):
            all_words = [
                w.strip().strip(".,;:\"'")
                for w in re.split(r"[\s,]+", raw)
                if w.strip() and len(w.strip()) > 2 and w.strip().isalpha()
            ]
            if len(all_words) >= 2 * n:
                nouns = parse_word_list(all_words[:n], n, "noun")
                verbs = parse_word_list(all_words[n : 2 * n], n, "verb")

        await _emit(log, f"LOG: [QDAD PHASE 0] nouns (row basis) = {nouns}")
        await _emit(log, f"LOG: [QDAD PHASE 0] verbs (col basis) = {verbs}")
        phase_log = list(state.get("phase_log") or [])
        phase_log.append(f"foundation n={n} nouns={nouns} verbs={verbs}")
        return {
            "nouns": nouns,
            "verbs": verbs,
            "matrices": {
                **(state.get("matrices") or {}),
                "nouns": nouns,
                "verbs": verbs,
            },
            "phase_log": phase_log,
        }

    return foundation_node


def create_grid_node(log: LogFn = None):
    """Phase 1 — assign permanent noun[i]×verb[j] to each FeatureAgent."""

    async def grid_node(state: QDADState) -> dict:
        nouns = state["nouns"]
        verbs = state["verbs"]
        n = state["n"]
        await _emit(
            log,
            f"LOG: [QDAD PHASE 1] Constructing {n}×{n} FeatureAgent grid "
            f"({n * n} agents; language = latent)...",
        )
        grid = []
        for i in range(n):
            for j in range(n):
                agent = {
                    "i": i,
                    "j": j,
                    "noun": nouns[i],
                    "verb": verbs[j],
                    "agent_id": f"FeatureAgent_{i}_{j}",
                }
                grid.append(agent)
                await _emit(
                    log,
                    f"LOG: [QDAD PHASE 1] {agent['agent_id']} ← "
                    f"noun={nouns[i]!r} × verb={verbs[j]!r}",
                )
        preview = ascii_grid_preview(nouns, verbs)
        await _emit(log, f"LOG: [QDAD PHASE 1] Basis grid preview:\n{preview}")
        phase_log = list(state.get("phase_log") or [])
        phase_log.append(f"grid {n}x{n} ready")
        return {
            "grid": grid,
            "features": empty_matrix(n),
            "phase_log": phase_log,
        }

    return grid_node


def create_noise_node(llm, log: LogFn = None):
    """Phase 2 — parallel forward diffusion (noise induction) over the grid."""

    async def noise_node(state: QDADState) -> dict:
        n = state["n"]
        nouns = state["nouns"]
        verbs = state["verbs"]
        noise_temp = state["noise_temperature"]
        user_prompt = state["user_prompt"]
        await _emit(
            log,
            f"LOG: [QDAD PHASE 2] Noise induction (forward diffusion) at T={noise_temp} "
            f"— {n * n} parallel FeatureAgent calls...",
        )
        chain = get_qdad_noise_chain(llm_with_temperature(llm, noise_temp))

        async def cell(i: int, j: int):
            try:
                text = await chain.ainvoke(
                    {
                        "i": i,
                        "j": j,
                        "noun": nouns[i],
                        "verb": verbs[j],
                        "user_prompt": user_prompt,
                    }
                )
                return i, j, (text or "").strip()
            except Exception as e:
                return i, j, f"[noise error] {e}"

        results = await asyncio.gather(
            *[cell(i, j) for i in range(n) for j in range(n)]
        )
        noisy = empty_matrix(n)
        for i, j, text in results:
            noisy[i][j] = text
            preview = text[:120].replace("\n", " ")
            await _emit(
                log,
                f"LOG: [QDAD PHASE 2] noisy[{i}][{j}] "
                f"({nouns[i]}×{verbs[j]}): {preview}...",
            )

        matrices = dict(state.get("matrices") or {})
        matrices["noisy"] = [row[:] for row in noisy]
        phase_log = list(state.get("phase_log") or [])
        phase_log.append("noise_induction complete")
        return {
            "noisy_features": noisy,
            "features": [row[:] for row in noisy],
            "denoise_step": 0,
            "matrices": matrices,
            "phase_log": phase_log,
        }

    return noise_node


def create_denoise_node(llm, log: LogFn = None):
    """Phase 3 — one reverse-diffusion round (parallel CriticAgents)."""

    async def denoise_node(state: QDADState) -> dict:
        n = state["n"]
        nouns = state["nouns"]
        verbs = state["verbs"]
        user_prompt = state["user_prompt"]
        total_steps = state["denoising_steps"]
        step = int(state.get("denoise_step") or 0) + 1
        current = state.get("features") or empty_matrix(n)
        # Critics run cooler than noise (score matching prefers lower T)
        critic_temp = max(0.3, min(1.0, float(state["noise_temperature"]) * 0.5))

        await _emit(
            log,
            f"LOG: [QDAD PHASE 3] Denoising step {step}/{total_steps} "
            f"(reverse diffusion / score matching, T={critic_temp}) — "
            f"{n * n} parallel CriticAgent calls...",
        )
        chain = get_qdad_critic_chain(llm_with_temperature(llm, critic_temp))

        async def cell(i: int, j: int, features=current, step=step):
            try:
                text = await chain.ainvoke(
                    {
                        "i": i,
                        "j": j,
                        "noun": nouns[i],
                        "verb": verbs[j],
                        "user_prompt": user_prompt,
                        "step": step,
                        "total_steps": total_steps,
                        "current_feature": features[i][j],
                    }
                )
                return i, j, (text or "").strip()
            except Exception:
                return i, j, features[i][j]

        results = await asyncio.gather(
            *[cell(i, j) for i in range(n) for j in range(n)]
        )
        refined = empty_matrix(n)
        for i, j, text in results:
            refined[i][j] = text
            preview = text[:100].replace("\n", " ")
            await _emit(
                log,
                f"LOG: [QDAD PHASE 3 step {step}] clean[{i}][{j}]: {preview}...",
            )

        matrices = dict(state.get("matrices") or {})
        matrices[f"step_{step}"] = [row[:] for row in refined]
        out: dict = {
            "features": refined,
            "denoise_step": step,
            "matrices": matrices,
        }
        if step >= total_steps:
            out["clean_features"] = [row[:] for row in refined]
            matrices["clean"] = [row[:] for row in refined]
            out["matrices"] = matrices
            await _emit(
                log,
                f"LOG: [QDAD PHASE 3] Reverse diffusion complete "
                f"({total_steps} steps).",
            )
        phase_log = list(state.get("phase_log") or [])
        phase_log.append(f"denoise step {step}/{total_steps}")
        out["phase_log"] = phase_log
        return out

    return denoise_node


def create_synthesis_node(llm, log: LogFn = None):
    """Phase 4 — synthesizer collapses clean matrix → agentic coding prompt."""

    async def synthesis_node(state: QDADState) -> dict:
        nouns = state["nouns"]
        verbs = state["verbs"]
        clean = state.get("clean_features") or state.get("features") or []
        user_prompt = state["user_prompt"]
        n = state["n"]
        await _emit(
            log,
            "LOG: [QDAD PHASE 4] Synthesizer — decode clean feature matrix "
            "→ agentic coding prompt (Midjourney latent→image analogue)...",
        )
        matrix_text = format_feature_matrix(nouns, verbs, clean)
        chain = get_qdad_synthesis_chain(llm)
        app_build_prompt = (
            await chain.ainvoke(
                {
                    "user_prompt": user_prompt,
                    "nouns": ", ".join(nouns),
                    "verbs": ", ".join(verbs),
                    "feature_matrix": matrix_text,
                }
            )
            or ""
        ).strip()

        appendix = (
            "\n\n---\n\n## Diffusion Feature Matrix (transparency)\n\n"
            f"**Nouns (row basis):** {', '.join(nouns)}\n\n"
            f"**Verbs (column basis):** {', '.join(verbs)}\n\n"
            f"{matrix_text}"
        )
        full_output = app_build_prompt + appendix
        final_solution = {
            "mode": "app_slot_machine",
            "proposed_solution": full_output,
            "app_build_prompt": app_build_prompt,
            "feature_matrix": clean,
            "noisy_features": state.get("noisy_features") or [],
            "nouns": nouns,
            "verbs": verbs,
            "grid_size": n,
            "denoising_steps": state["denoising_steps"],
            "temperature_scale": state["noise_temperature"],
            "noun_verb_temperature": state["noun_verb_temperature"],
            "reasoning": (
                f"QDAD qualitative diffusion on {n}×{n} noun×verb grid "
                f"({state['denoising_steps']} reverse-diffusion steps). "
                "Language = medium; critics = score matching."
            ),
            "matrices": state.get("matrices") or {},
        }
        phase_log = list(state.get("phase_log") or [])
        phase_log.append("synthesis complete")
        await _emit(log, "LOG: [QDAD PHASE 4] Agentic coding prompt ready.")
        return {
            "app_build_prompt": app_build_prompt,
            "final_solution": final_solution,
            "phase_log": phase_log,
        }

    return synthesis_node


def should_continue_denoise(state: QDADState) -> str:
    """Route: more reverse-diffusion steps or proceed to synthesis."""
    step = int(state.get("denoise_step") or 0)
    total = int(state.get("denoising_steps") or 1)
    if step < total:
        return "denoise"
    return "synthesize"
