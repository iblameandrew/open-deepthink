"""Shared path helpers for the legacy chk()-based test suite."""
from __future__ import annotations

import sys
from pathlib import Path

# Repository root (parent of tests/)
ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def root_path(*parts: str) -> Path:
    """Join path segments onto the repo root."""
    return ROOT.joinpath(*parts)


def root_str(*parts: str) -> str:
    """String form of a path under the repo root."""
    return str(root_path(*parts))
