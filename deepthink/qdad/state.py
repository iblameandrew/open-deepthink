"""QDAD graph state — the qualitative diffusion state tensor."""

from typing import Any, TypedDict


class QDADState(TypedDict, total=False):
    """State passed through the QDAD LangGraph.

    The feature matrix is the qualitative analogue of a noisy latent:
    each cell is a language-vector at coordinates (noun_i, verb_j).
    """

    # ── Inputs / hyperparams (GUI) ──
    user_prompt: str
    session_id: str
    n: int  # grid size N (2–8)
    noise_temperature: float  # Temperature Scale (0.7–1.8)
    noun_verb_temperature: float  # Noun/Verb Temperature (0.3–1.0)
    denoising_steps: int  # 1–6
    denoise_step: int  # current step (1-indexed during denoise loop)

    # ── Phase 0: qualitative basis ──
    nouns: list[str]  # row basis, length N
    verbs: list[str]  # column basis, length N

    # ── Phase 1: agent grid metadata ──
    # list of {i, j, noun, verb, agent_id}
    grid: list[dict[str, Any]]

    # ── Feature matrices (language as the latent) ──
    features: list[list[str]]  # current N×N matrix
    noisy_features: list[list[str]]  # after forward noise
    clean_features: list[list[str]]  # after final denoise step

    # Intermediate snapshots for transparency / inspection
    # keys: "noisy", "step_1", ..., "clean"
    matrices: dict[str, Any]

    # ── Phase 4 output ──
    app_build_prompt: str
    final_solution: dict[str, Any] | None

    # Runtime logs collected for optional inspection
    phase_log: list[str]
