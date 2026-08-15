"""
Qualitative Diffusion App Designer (QDAD) — App Slot Machine Mode.

Philosophy (strict):
  • Language is the computational medium.
  • Nouns and verbs act as orthogonal basis directions.
  • High temperature = controlled qualitative noise.
  • Critic agents = qualitative reverse diffusion / score matching.
  • Vague aesthetic prompt → concrete buildable app spec
    (the way Midjourney turns a vague prompt into an image).
"""

from .graph import build_qdad_graph
from .pipeline import run_qdad_pipeline
from .state import QDADState


def default_qdad_params():
    """Documented QDAD defaults for harnesses / library callers."""
    return {
        "grid_size": 3,
        "n": 3,
        "temperature_scale": 1.3,
        "denoising_steps": 2,
        "noun_verb_temperature": 0.6,
    }


__all__ = [
    "QDADState",
    "build_qdad_graph",
    "run_qdad_pipeline",
    "default_qdad_params",
]
