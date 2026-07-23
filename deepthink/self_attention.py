"""
Qualitative Self-Attention (QSA) for QNN brainstorm mode.

Inspired by colony's AttentionAgent (Qualitative Self-Attention):
  • Neurons ≈ token sequence
  • Pairwise qualitative scoring ≈ Q·Kᵀ (not numeric MatMul)
  • Strength buckets ≈ softmax (none / low / med / high)
  • Distance buckets ≈ receptive field (near / mid / far)

Unlike the layered feed-forward graph (each agent only sees the previous
layer), self-attention lets a neuron in the current epoch **attend past
neurons that are not graph neighbors** — earlier layers this epoch, and
memory from other agents in prior epochs.

Colony reference: smenos/colony/backend/app/agents/attention.py
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

# Cap scoring work (colony uses ≤96 pairs / tick)
MAX_ATTENTION_CANDIDATES = 48
# How many non-local past neurons to inject into the prompt
DEFAULT_TOP_K = 5

STRENGTH_ORDER = {"none": 0, "low": 1, "med": 2, "high": 3}
VALID_STRENGTHS = frozenset(STRENGTH_ORDER)
VALID_DISTANCES = frozenset({"near", "mid", "far"})

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_\-]{2,}")


@dataclass
class AttentionEdge:
    """One directed attention judgment: query neuron → past neuron (key/value)."""

    from_id: str
    to_id: str
    strength: str  # none | low | med | high
    qualitative_distance: str  # near | mid | far
    kind: str  # conceptual affinity bucket
    rationale: str
    source: str  # "memory" | "agent_outputs"
    epoch_hint: Optional[int] = None
    excerpt: str = ""
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AttentionCandidate:
    agent_id: str
    output: Dict[str, Any]
    source: str
    epoch_hint: Optional[int] = None
    entry_index: int = -1


def _tokenize(text: str) -> Set[str]:
    if not text:
        return set()
    return {t.lower() for t in TOKEN_RE.findall(str(text))}


def _persona_tokens(persona: Optional[dict]) -> Set[str]:
    if not persona:
        return set()
    bits: List[str] = []
    for key in ("name", "specialty", "guiding_words", "system_prompt"):
        val = persona.get(key)
        if isinstance(val, str):
            bits.append(val)
        elif isinstance(val, list):
            bits.extend(str(x) for x in val)
    for key in ("attributes", "skills"):
        val = persona.get(key)
        if isinstance(val, list):
            bits.extend(str(x) for x in val)
        elif isinstance(val, str):
            bits.append(val)
    return _tokenize(" ".join(bits))


def _output_tokens(output: Dict[str, Any]) -> Set[str]:
    parts = [
        str(output.get("original_problem") or ""),
        str(output.get("proposed_solution") or ""),
        str(output.get("reasoning") or ""),
        str(output.get("falsifiers") or ""),
        str(output.get("risks") or ""),
    ]
    skills = output.get("skills_used")
    if isinstance(skills, list):
        parts.append(" ".join(str(s) for s in skills))
    return _tokenize(" ".join(parts))


def _excerpt(output: Dict[str, Any], limit: int = 280) -> str:
    text = (
        str(output.get("proposed_solution") or "")
        or str(output.get("reasoning") or "")
        or str(output.get("summary_of_past_epochs") or "")
        or str(output)[:limit]
    )
    text = " ".join(text.split())
    if len(text) > limit:
        return text[: limit - 1] + "…"
    return text


def _normalize_output(raw: Any) -> Optional[Dict[str, Any]]:
    if raw is None:
        return None
    if isinstance(raw, list):
        if not raw:
            return None
        raw = raw[-1]
    if not isinstance(raw, dict):
        return {"proposed_solution": str(raw)}
    # Skip pure memory-summary shells with no substance
    if "summary_of_past_epochs" in raw and not raw.get("proposed_solution"):
        return {
            "proposed_solution": raw.get("summary_of_past_epochs", ""),
            "reasoning": raw.get("note", "summarized past epochs"),
            "is_summary": True,
        }
    return raw


def graph_neighbor_ids(node_id: str, all_layers_prompts: Sequence[Sequence[Any]]) -> Set[str]:
    """Immediate feed-forward neighbors: entire previous layer (graph edges)."""
    try:
        layer_index = int(node_id.split("_")[1])
        agent_index = int(node_id.split("_")[2])
    except (IndexError, ValueError):
        return set()

    if layer_index <= 0:
        return set()
    prev = layer_index - 1
    if prev >= len(all_layers_prompts):
        return set()
    return {f"agent_{prev}_{j}" for j in range(len(all_layers_prompts[prev]))}


def collect_attention_candidates(
    state: dict,
    node_id: str,
    neighbor_ids: Optional[Set[str]] = None,
    max_candidates: int = MAX_ATTENTION_CANDIDATES,
) -> List[AttentionCandidate]:
    """
    Build a pool of past / non-local neurons the query agent may attend to.

    Sources:
      1. ``agent_outputs`` from neurons already finished this epoch that are
         **not** immediate previous-layer neighbors (skipped layers, side paths).
      2. ``memory`` of other neurons (prior-epoch reflections). Neighbors'
         *past* epochs are included; their current-epoch output is not
         (already in the feed-forward upstream block).
    """
    neighbor_ids = neighbor_ids if neighbor_ids is not None else set()
    candidates: List[AttentionCandidate] = []
    seen: Set[Tuple[str, str, int]] = set()

    def _add(agent_id: str, output: Any, source: str, epoch_hint: Optional[int], entry_index: int = -1):
        if agent_id == node_id:
            return
        norm = _normalize_output(output)
        if not norm:
            return
        key = (agent_id, source, entry_index if entry_index >= 0 else hash(str(norm)[:120]))
        if key in seen:
            return
        seen.add(key)
        candidates.append(
            AttentionCandidate(
                agent_id=agent_id,
                output=norm,
                source=source,
                epoch_hint=epoch_hint,
                entry_index=entry_index,
            )
        )

    current_epoch = state.get("epoch", 0)
    agent_outputs = state.get("agent_outputs") or {}

    # (1) Non-neighbor outputs already produced this epoch
    for agent_id, raw in agent_outputs.items():
        if agent_id in neighbor_ids:
            continue
        _add(agent_id, raw, "agent_outputs", current_epoch)

    # (2) Memory from other agents (and neighbors' older epochs)
    memory = state.get("memory") or {}
    for agent_id, history in memory.items():
        if not isinstance(history, list):
            continue
        # Own personal memory is already injected separately — skip self
        if agent_id == node_id:
            continue
        for idx, entry in enumerate(history):
            # Neighbors: skip the latest entry if it matches current agent_outputs
            # (current-epoch feed-forward path). Keep older entries.
            if agent_id in neighbor_ids and agent_id in agent_outputs:
                if idx == len(history) - 1:
                    # Likely the entry just written this epoch or last write
                    current = _normalize_output(agent_outputs.get(agent_id))
                    this = _normalize_output(entry)
                    if current and this and current.get("proposed_solution") == this.get(
                        "proposed_solution"
                    ):
                        continue
            _add(agent_id, entry, "memory", None if agent_id in neighbor_ids else None, idx)

    # Prefer current-epoch non-local first, then denser memories
    candidates.sort(
        key=lambda c: (
            0 if c.source == "agent_outputs" else 1,
            -(c.epoch_hint if c.epoch_hint is not None else -1),
            c.agent_id,
        )
    )
    return candidates[:max_candidates]


def score_pair_heuristic(
    query_id: str,
    query_persona: Optional[dict],
    candidate: AttentionCandidate,
) -> AttentionEdge:
    """
    Colony-style qualitative pair score (no MatMul).

    Shared persona / content tokens → near+high; partial → mid+med; else far+low.
    """
    q_tokens = _persona_tokens(query_persona)
    # Also let the query's specialty words bias matching
    k_tokens = _output_tokens(candidate.output) | _persona_tokens(
        # re-use output tokens only; key persona may be unknown for memory-only
        {}
    )
    # Prefer richer key side from solution text
    shared = q_tokens & k_tokens if q_tokens else set()
    # Fallback: content richness as weak signal
    k_len = len(k_tokens)

    if len(shared) >= 3:
        strength, distance = "high", "near"
        kind = "affinity"
        rationale = (
            f"{query_id} attends {candidate.agent_id}: shared concepts "
            f"{sorted(list(shared))[:5]} — strong non-local bond."
        )
        score = 3.0 + 0.1 * len(shared)
    elif len(shared) >= 1:
        strength, distance = "med", "mid"
        kind = "resonance"
        rationale = (
            f"{query_id} attends {candidate.agent_id}: partial overlap "
            f"{sorted(list(shared))[:4]} — moderate cross-layer affinity."
        )
        score = 2.0 + 0.1 * len(shared)
    elif k_len >= 8:
        strength, distance = "low", "far"
        kind = "exploration"
        rationale = (
            f"{query_id} attends {candidate.agent_id}: no shared lexicon, "
            f"but substantive past output — exploratory far attention."
        )
        score = 1.0
    else:
        strength, distance = "none", "far"
        kind = "ignore"
        rationale = f"No useful affinity between {query_id} and {candidate.agent_id}."
        score = 0.0

    # Slight boost for same-epoch non-neighbor (fresh non-local signal)
    if candidate.source == "agent_outputs" and strength != "none":
        score += 0.35

    return AttentionEdge(
        from_id=query_id,
        to_id=candidate.agent_id,
        strength=strength,
        qualitative_distance=distance,
        kind=kind,
        rationale=rationale,
        source=candidate.source,
        epoch_hint=candidate.epoch_hint,
        excerpt=_excerpt(candidate.output),
        score=score,
    )


def select_top_edges(
    edges: Iterable[AttentionEdge],
    top_k: int = DEFAULT_TOP_K,
    min_strength: str = "low",
) -> List[AttentionEdge]:
    min_ord = STRENGTH_ORDER.get(min_strength, 1)
    filtered = [
        e
        for e in edges
        if STRENGTH_ORDER.get(e.strength, 0) >= min_ord and e.strength != "none"
    ]
    filtered.sort(key=lambda e: (-e.score, -STRENGTH_ORDER.get(e.strength, 0), e.to_id))
    # One edge per target neuron (best score wins)
    best: Dict[str, AttentionEdge] = {}
    for e in filtered:
        if e.to_id not in best:
            best[e.to_id] = e
    ranked = sorted(best.values(), key=lambda e: (-e.score, e.to_id))
    return ranked[:top_k]


def format_attention_context(edges: Sequence[AttentionEdge]) -> str:
    """Markdown block injected into the agent prompt (V analogue)."""
    if not edges:
        return ""
    lines = [
        "## Qualitative Self-Attention (non-local past neurons)",
        "These are past / non-neighbor neurons your feed-forward edges do **not**",
        "connect you to. Use them as optional keys/values — cite `agent_id` when useful.",
        "Do not merely restate them; integrate, contrast, or falsify.",
        "",
    ]
    for i, e in enumerate(edges, 1):
        ep = f" epoch≈{e.epoch_hint}" if e.epoch_hint is not None else ""
        lines.append(
            f"### Attended {i}: `{e.to_id}` "
            f"[{e.strength}/{e.qualitative_distance}/{e.kind}] source={e.source}{ep}"
        )
        lines.append(f"- Why: {e.rationale}")
        lines.append(f"- Value: {e.excerpt}")
        lines.append("")
    return "\n".join(lines)


def compute_self_attention(
    state: dict,
    node_id: str,
    top_k: int = DEFAULT_TOP_K,
    max_candidates: int = MAX_ATTENTION_CANDIDATES,
) -> Tuple[List[AttentionEdge], str]:
    """
    Full QSA step for one neuron in brainstorm mode.

    Returns (edges, formatted_context_block).
    """
    personas = state.get("agent_personas") or {}
    query_persona = personas.get(node_id) or {}
    neighbors = graph_neighbor_ids(node_id, state.get("all_layers_prompts") or [])

    candidates = collect_attention_candidates(
        state, node_id, neighbors, max_candidates=max_candidates
    )
    if not candidates:
        return [], ""

    edges = [
        score_pair_heuristic(node_id, query_persona, cand) for cand in candidates
    ]
    top = select_top_edges(edges, top_k=top_k)
    return top, format_attention_context(top)
