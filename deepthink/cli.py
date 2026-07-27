"""Console-script entry points for open-deepthink."""

from __future__ import annotations


def main() -> None:
    """``open-deepthink`` / ``deepthink`` console scripts."""
    from deepthink.__main__ import main as run

    run()


if __name__ == "__main__":
    main()
