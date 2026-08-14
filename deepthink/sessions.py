"""Dict-like session store with optional JSON persistence.

Web UI sessions stay in memory for live RAPTOR / LLM objects. JSON-safe
fields are also written under ``state_dir/sessions`` so a process restart
can recover exportable traces. Persistence failures are ignored.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SKIP_KEYS = frozenset(
    {
        "raptor_index",
        "llm",
        "summarizer_llm",
        "embeddings_model",
        "synthesis_llm",
    }
)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items() if k not in _SKIP_KEYS}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return None


class SessionStore(dict):
    """``sessions[id] = state`` compatible mapping with optional disk writes."""

    def __init__(self, persist_dir: str | Path | None = None, *, persist: bool = True):
        super().__init__()
        self.persist = persist
        if persist_dir is None:
            try:
                from deepthink.config import get_settings

                persist_dir = Path(get_settings().state_dir) / "sessions"
            except Exception:
                persist_dir = Path(".deepthink-state") / "sessions"
        self.persist_dir = Path(persist_dir)

    def __setitem__(self, key: Any, value: Any) -> None:
        super().__setitem__(key, value)
        self._maybe_write(str(key), value)

    def update(self, *args, **kwargs) -> None:
        super().update(*args, **kwargs)
        if self.persist:
            for key in dict(*args, **kwargs):
                self._maybe_write(str(key), self.get(key))

    def _maybe_write(self, session_id: str, value: Any) -> None:
        if not self.persist or not session_id:
            return
        if not isinstance(value, dict):
            return
        try:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            payload = _json_safe(value)
            path = self.persist_dir / f"{session_id}.json"
            path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        except Exception as exc:
            logger.debug("session persist skipped for %s: %s", session_id, exc)

    def load(self, session_id: str) -> dict[str, Any] | None:
        """Load a persisted session if memory is empty."""
        if session_id in self:
            value = self[session_id]
            return value if isinstance(value, dict) else None
        path = self.persist_dir / f"{session_id}.json"
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if isinstance(data, dict):
            super().__setitem__(session_id, data)
            return data
        return None
