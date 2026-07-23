"""
Qualitative Neural Network (QNN) — brainstorm / strategy-map engine.

Portable entrypoint for skills and harnesses:

    from deepthink.qnn import run_qnn_pipeline

See ``run_qnn_pipeline`` for the full parameter surface.
"""

from .pipeline import run_qnn_pipeline, QNNResult, default_qnn_params

__all__ = [
    "run_qnn_pipeline",
    "QNNResult",
    "default_qnn_params",
]
