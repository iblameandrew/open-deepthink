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

from .state import QDADState
from .graph import build_qdad_graph
from .pipeline import run_qdad_pipeline

__all__ = [
    "QDADState",
    "build_qdad_graph",
    "run_qdad_pipeline",
]
