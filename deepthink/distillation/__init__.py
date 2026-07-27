"""
Knowledge Distillation — evolutionary multi-agent topology (1×2×2×2×2×2×1).

Public library entry::

    from deepthink.distillation import DistillationGraph, DistillationAgent
    # or: from deepthink import DistillationGraph

Implementation remains in ``deepthink.knowledge_distillation`` for
backward compatibility; this package is the preferred import path.
"""

from __future__ import annotations

from deepthink.chains.distillation_chains import DISTILLATION_ARCHETYPES
from deepthink.knowledge_distillation import DistillationAgent, DistillationGraph

__all__ = [
    "DistillationAgent",
    "DistillationGraph",
    "DISTILLATION_ARCHETYPES",
]
