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

# skill_folder_name -> zip name prefix
SKILLS = {
    "qnn": "qnn-skill",
    "qdad": "qdad-skill",
}


def read_version() -> str:
    init = (ROOT / "deepthink" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'__version__\s*=\s*"([^"]+)"', init)
    if not match:
        raise SystemExit("Could not read deepthink.__version__")
    return match.group(1)


def package_skill(skill_id: str, version: str) -> Path:
    if skill_id not in SKILLS:
        raise SystemExit(f"Unknown skill {skill_id!r}; choose from {list(SKILLS)}")
    skill_dir = SKILLS_DIR / skill_id
    if not (skill_dir / "SKILL.md").is_file():
        raise SystemExit(f"Missing {skill_dir / 'SKILL.md'}")

    DIST_DIR.mkdir(parents=True, exist_ok=True)
    prefix = SKILLS[skill_id]
    out = DIST_DIR / f"{prefix}-{version}.zip"
    members = ["SKILL.md", "INSTALL.md"]
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in members:
            path = skill_dir / name
            if not path.is_file():
                raise SystemExit(f"Missing {path}")
            # Layout: {skill_id}/SKILL.md so unzip into ~/.grok/skills works
            zf.write(path, arcname=f"{skill_id}/{name}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        default=None,
        help="Version string for the zip name (default: deepthink.__version__)",
    )
    parser.add_argument(
        "--skill",
        default="all",
        choices=["all", *SKILLS.keys()],
        help="Which skill to package (default: all)",
    )
    args = parser.parse_args()
    version = args.version or read_version()
    targets = list(SKILLS.keys()) if args.skill == "all" else [args.skill]
    for skill_id in targets:
        out = package_skill(skill_id, version)
        print(f"Wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
