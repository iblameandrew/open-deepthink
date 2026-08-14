"""Deterministic cost / call-count estimates. No network, no API key.

These numbers count *LLM invocations the pipelines will issue*, not dollars.
Token totals are order-of-magnitude (default ~2k in / 800 out per call).
They exist so a run can be sized *before* money is spent.

This is not a quality benchmark.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

# Conservative mid-size prompt / completion. Real models vary widely.
DEFAULT_TOKENS_IN = 2000
DEFAULT_TOKENS_OUT = 800

# Auto mode: keep topologies laptop-sized unless the user opts into manual.
AUTO_AGENT_CAP = 24


@dataclass(frozen=True)
class CostEstimate:
    """One estimated run."""

    kind: str
    llm_calls: int
    est_tokens_in: int
    est_tokens_out: int
    est_tokens_total: int
    breakdown: dict[str, int]
    notes: list[str]
    topology: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary_line(self) -> str:
        return (
            f"{self.kind}: ~{self.llm_calls} LLM calls, "
            f"~{self.est_tokens_total:,} tokens "
            f"({self.est_tokens_in:,} in / {self.est_tokens_out:,} out)"
        )


def estimate_qnn_cost(
    layers: int = 3,
    width: int = 3,
    epochs: int = 2,
    *,
    qnn_mode: str = "manual",
    has_document_context: bool = False,
    enable_self_attention: bool = True,
    tokens_in: int = DEFAULT_TOKENS_IN,
    tokens_out: int = DEFAULT_TOKENS_OUT,
) -> CostEstimate:
    """Count QNN pipeline LLM calls for a given L×W×E.

    Matches ``deepthink.qnn.pipeline.run_qnn_pipeline``:

    * optional brief summarizer (documents only)
    * optional complexity estimator (auto mode)
    * 1 seed-word call
    * L×W persona spans
    * E × L × W forward cells
    * (E−1) epoch maps + (E−1)×L×W persona rewrites + (E−1) reframes
    * 2 final calls (synthesis + polish)
    """
    layers = max(1, int(layers))
    width = max(1, int(width))
    epochs = max(1, int(epochs))
    agents = layers * width
    brief = 1 if has_document_context else 0
    complexity = 1 if str(qnn_mode).lower() != "manual" else 0
    seeds = 1
    span = agents
    forward = epochs * agents
    maps = max(0, epochs - 1)
    mirror = max(0, epochs - 1) * agents
    reframes = max(0, epochs - 1)
    final = 2  # synthesis + polish
    calls = brief + complexity + seeds + span + forward + maps + mirror + reframes + final
    breakdown = {
        "brief": brief,
        "complexity_estimator": complexity,
        "seeds": seeds,
        "persona_span": span,
        "forward_cells": forward,
        "epoch_maps": maps,
        "persona_rewrite": mirror,
        "reframe": reframes,
        "final_report": final,
    }
    notes = [
        "Counts LLM invocations the library pipeline will issue.",
        "Self-attention is lexical overlap — it does not add LLM calls.",
        "Quality versus a single long prompt is unevaluated.",
    ]
    if str(qnn_mode).lower() != "manual" and agents > AUTO_AGENT_CAP:
        notes.append(
            f"Auto mode will shrink width so L×W ≤ {AUTO_AGENT_CAP} (requested {agents} agents)."
        )
    if not enable_self_attention:
        notes.append("Self-attention disabled; call count unchanged.")
    if agents * epochs > 40:
        notes.append("This topology is expensive. Prefer 2×2×1 or 3×3×2 first.")
    return CostEstimate(
        kind="qnn",
        llm_calls=calls,
        est_tokens_in=calls * tokens_in,
        est_tokens_out=calls * tokens_out,
        est_tokens_total=calls * (tokens_in + tokens_out),
        breakdown=breakdown,
        notes=notes,
        topology={
            "layers": layers,
            "width": width,
            "epochs": epochs,
            "agents": agents,
            "qnn_mode": qnn_mode,
        },
    )


def estimate_qdad_cost(
    n: int = 3,
    denoising_steps: int = 2,
    *,
    tokens_in: int = DEFAULT_TOKENS_IN,
    tokens_out: int = DEFAULT_TOKENS_OUT,
) -> CostEstimate:
    """Count QDAD LLM calls for an N×N grid and S critic steps.

    * 1 foundation (nouns + verbs)
    * N×N noise (forward invent)
    * S × N×N critic rewrites
    * 1 synthesizer
    """
    n = max(2, min(8, int(n)))
    steps = max(1, min(6, int(denoising_steps)))
    cells = n * n
    foundation = 1
    noise = cells
    critics = steps * cells
    synth = 1
    calls = foundation + noise + critics + synth
    return CostEstimate(
        kind="qdad",
        llm_calls=calls,
        est_tokens_in=calls * tokens_in,
        est_tokens_out=calls * tokens_out,
        est_tokens_total=calls * (tokens_in + tokens_out),
        breakdown={
            "foundation": foundation,
            "noise_cells": noise,
            "critic_cells": critics,
            "synthesizer": synth,
        },
        notes=[
            "Counts LLM invocations the QDAD LangGraph will issue.",
            "High temperature is a sampling knob, not a diffusion SDE.",
            "Quality versus a single app-brief prompt is unevaluated.",
        ],
        topology={"n": n, "cells": cells, "denoising_steps": steps},
    )


def estimate_distillation_cost(
    max_epochs: int = 1,
    *,
    token_budget: int = 500_000,
    tokens_in: int = DEFAULT_TOKENS_IN,
    tokens_out: int = DEFAULT_TOKENS_OUT,
) -> CostEstimate:
    """Rough per-epoch call count for the fixed 1×2×2×2×2×2×1 graph (12 agents)."""
    epochs = max(1, int(max_epochs))
    per_epoch = {
        "task_master": 1,
        "forward_agents": 12,
        "mirror_descent": 12,
        "mixing_expected": 3,  # mock mixes ~30% hard; this is a mid guess
        "seed_creator": 1,
        "followup": 1,
        "perplexity": 1,
    }
    per = sum(per_epoch.values())
    calls = per * epochs
    return CostEstimate(
        kind="distillation",
        llm_calls=calls,
        est_tokens_in=calls * tokens_in,
        est_tokens_out=calls * tokens_out,
        est_tokens_total=calls * (tokens_in + tokens_out),
        breakdown={k: v * epochs for k, v in per_epoch.items()},
        notes=[
            "Fixed 12-agent topology. Mixing calls vary with 'hard' judgments.",
            f"A token_budget of {token_budget:,} is the real stop condition.",
            "Output is a QA trace + topology archive, not a proven training set.",
        ],
        topology={
            "agents": 12,
            "layers": [1, 2, 2, 2, 2, 2, 1],
            "max_epochs": epochs,
            "token_budget": token_budget,
        },
    )
