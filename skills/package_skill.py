#!/usr/bin/env python3
"""Build release zip(s) for portable skills under skills/."""

from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKILLS_DIR = ROOT / "skills"
DIST_DIR = ROOT / "dist" / "skills"


def read_version() -> str:
    init = (ROOT / "deepthink" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'__version__\s*=\s*"([^"]+)"', init)
    if not match:
        raise SystemExit("Could not read deepthink.__version__")
    return match.group(1)


def package_qnn(version: str) -> Path:
    skill_dir = SKILLS_DIR / "qnn"
    if not (skill_dir / "SKILL.md").is_file():
        raise SystemExit(f"Missing {skill_dir / 'SKILL.md'}")

    DIST_DIR.mkdir(parents=True, exist_ok=True)
    out = DIST_DIR / f"qnn-skill-{version}.zip"
    members = ["SKILL.md", "INSTALL.md"]
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in members:
            path = skill_dir / name
            if not path.is_file():
                raise SystemExit(f"Missing {path}")
            # Layout: qnn/SKILL.md so unzip into ~/.grok/skills works
            zf.write(path, arcname=f"qnn/{name}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        default=None,
        help="Version string for the zip name (default: deepthink.__version__)",
    )
    args = parser.parse_args()
    version = args.version or read_version()
    out = package_qnn(version)
    print(f"Wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
