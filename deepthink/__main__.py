"""
``python -m deepthink`` entry.

* No args / ``serve`` → FastAPI web UI
* ``qnn`` / ``qdad`` / ``version`` → library CLI (see ``deepthink.cli``)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def serve() -> None:
    """Start the open-deepthink web server."""
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        os.chdir(root)
    except OSError:
        pass

    from deepthink.config import get_settings

    settings = get_settings()
    import uvicorn
    from app import app  # intentional late import after chdir

    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
    )


def main() -> None:
    # Delegate to CLI when subcommands/args are present
    if len(sys.argv) > 1:
        from deepthink.cli import main as cli_main

        cli_main(sys.argv[1:])
        return
    serve()


if __name__ == "__main__":
    main()
