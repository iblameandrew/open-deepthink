"""Helpers for QDAD qualitative algebra (word lists, matrices, temperature)."""
from __future__ import annotations

import re
from typing import Any, List, Optional


def llm_with_temperature(llm, temperature: float):
    """Best-effort temperature bind. Mock / some wrappers ignore it."""
    try:
        return llm.bind(temperature=float(temperature))
    except Exception:
        return llm


def parse_word_list(raw: Any, n: int, kind: str = "word") -> List[str]:
    """Normalize LLM output into exactly `n` distinct words (basis vectors)."""
    words: List[str] = []
    if isinstance(raw, list):
        words = [str(w).strip() for w in raw if str(w).strip()]
    elif isinstance(raw, str):
        words = [
            w.strip().strip(".,;:\"'[]")
            for w in re.split(r"[\s,]+", raw)
            if w.strip() and len(w.strip()) > 1
        ]

    seen = set()
    cleaned: List[str] = []
    for w in words:
        key = w.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(w)

    fi = 0
    while len(cleaned) < n:
        cand = f"{kind}{fi + 1}"
        fi += 1
        if cand.lower() not in seen:
            seen.add(cand.lower())
            cleaned.append(cand)
    return cleaned[:n]


def empty_matrix(n: int) -> List[List[str]]:
    return [["" for _ in range(n)] for _ in range(n)]


def format_feature_matrix(
    nouns: List[str], verbs: List[str], matrix: List[List[str]]
) -> str:
    """Human-readable N×N feature matrix for synthesis + transparency."""
    lines = []
    n = len(nouns)
    for i in range(n):
        for j in range(n):
            cell = matrix[i][j] if i < len(matrix) and j < len(matrix[i]) else ""
            lines.append(
                f"[({i},{j}) basis: noun={nouns[i]!r} × verb={verbs[j]!r}]\n{cell}\n"
            )
    return "\n".join(lines)


def ascii_grid_preview(nouns: List[str], verbs: List[str]) -> str:
    """Compact ASCII of the noun×verb basis for log / graph viz."""
    n = len(nouns)
    header = "        " + " | ".join(f"{v[:8]:^8}" for v in verbs)
    rows = [header, "      " + "-" * max(8, len(header) - 6)]
    for i, noun in enumerate(nouns):
        cells = " | ".join(f"A{i}{j}".center(8) for j in range(n))
        rows.append(f"{noun[:6]:>6} | {cells}")
    return "\n".join(rows)


def clamp_params(
    n: Any = 4,
    noise_temp: Any = 1.3,
    denoising_steps: Any = 3,
    nv_temp: Any = 0.6,
) -> tuple:
    n = max(2, min(8, int(n)))
    noise_temp = max(0.7, min(1.8, float(noise_temp)))
    denoising_steps = max(1, min(6, int(denoising_steps)))
    nv_temp = max(0.3, min(1.0, float(nv_temp)))
    return n, noise_temp, denoising_steps, nv_temp


def enrich_prompt(
    user_prompt: str,
    document_context: str = "",
    chat_history: Optional[list] = None,
) -> str:
    enriched = user_prompt or ""
    if document_context:
        enriched = (
            f"{enriched}\n\n--- Attached Context ---\n{document_context[:30000]}"
        )
    if chat_history:
        prior = "\n".join(
            f"{'User' if m.get('role') == 'user' else 'Assistant'}: {m.get('content', '')}"
            for m in chat_history
        )
        if prior:
            enriched = f"{enriched}\n\n--- Prior conversation ---\n{prior[-8000:]}"
    return enriched
