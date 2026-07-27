"""Allow ``python -m deepthink`` to start the open-deepthink web server."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    # Ensure the repo root (for app.py + static assets) is on sys.path and cwd-friendly
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    # Prefer running from repo root so StaticFiles("js") etc. resolve
    try:
        os.chdir(root)
    except OSError:
        pass

    from deepthink.config import get_settings

    settings = get_settings()

    import uvicorn

    # Import app after path/chdir so relative mounts work
    from app import app  # intentional late import after chdir

    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
    )


if __name__ == "__main__":
    main()
