"""Process-wide log bus for web-runtime nodes (legacy graph + RAPTOR).

The FastAPI app installs an asyncio.Queue at startup. Library callers that
never start the UI leave the queue unset; ``emit`` then becomes a no-op.
"""

from __future__ import annotations

import asyncio

_log_queue: asyncio.Queue | None = None


def set_log_queue(queue: asyncio.Queue | None) -> None:
    """Install or clear the process log queue."""
    global _log_queue
    _log_queue = queue


def get_log_queue() -> asyncio.Queue | None:
    return _log_queue


async def emit(msg: str) -> None:
    """Put a log line on the queue if one is installed."""
    q = _log_queue
    if q is None:
        return
    await q.put(msg)


def emit_nowait(msg: str) -> None:
    """Best-effort put from sync code (RAPTOR clustering)."""
    q = _log_queue
    if q is None:
        return
    try:
        q.put_nowait(msg)
    except Exception:
        pass
