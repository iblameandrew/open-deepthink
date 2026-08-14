#!/usr/bin/env python3
"""
Run the full open-deepthink phase test suite.

Usage (from repo root)::

    python tests/run_all.py
    python -m tests.run_all

Exits non-zero if any phase reports failures or a phase process crashes.
Does not require external API keys (mocks / debug mode).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"

# Order mirrors historical phase numbering
PHASE_FILES = [
    "phase1_imports.py",
    "phase2_utils.py",
    "phase3_chains.py",
    "phase4_state.py",
    "phase5_endpoints.py",
    "phase6_distillation.py",
    "phase7_mocks.py",
    "phase8_nodes.py",
    "phase9_e2e.py",
    "phase10_static.py",
    "phase11_regression.py",
    "phase_qdad.py",
    "phase_self_attention.py",
    "phase_config.py",
    "phase_package_api.py",
    "phase_honesty.py",
]


def _parse_summary(output: str) -> tuple[int, int] | None:
    """Parse lines like 'PHASE 1: 7/7 OK' → (ok, total)."""
    matches = re.findall(
        r"PHASE[^\n:]*:\s*(\d+)\s*/\s*(\d+)\s*OK",
        output,
        flags=re.IGNORECASE,
    )
    if not matches:
        return None
    # Use the last summary line in the file
    ok, total = matches[-1]
    return int(ok), int(total)


def main() -> int:
    print(f"open-deepthink test runner — root={ROOT}")
    print(f"Python {sys.version.split()[0]}\n")

    grand_ok = 0
    grand_total = 0
    failed_phases: list[str] = []

    for name in PHASE_FILES:
        path = TESTS / name
        if not path.is_file():
            print(f"  [SKIP] {name} (missing)")
            continue

        print(f"======== {name} ========")
        proc = subprocess.run(
            [sys.executable, str(path)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        # Stream condensed output
        for line in out.splitlines():
            if line.strip().startswith("[") or "PHASE" in line or "FAIL" in line:
                print(line)
            elif "Error" in line or "Traceback" in line:
                print(line)

        summary = _parse_summary(out)
        if proc.returncode != 0 and summary is None:
            print(f"  [CRASH] exit={proc.returncode}")
            # Print tail for debugging
            tail = "\n".join(out.splitlines()[-40:])
            if tail.strip():
                print(tail)
            failed_phases.append(name)
            continue

        if summary is None:
            print("  [WARN] no PHASE summary line found")
            if proc.returncode != 0:
                failed_phases.append(name)
            continue

        ok, total = summary
        grand_ok += ok
        grand_total += total
        status = "OK" if ok == total else "FAIL"
        print(f"  → {ok}/{total} {status}\n")
        if ok != total:
            failed_phases.append(name)
            # Show failure lines
            for line in out.splitlines():
                if "FAIL" in line:
                    print(f"    {line}")

    extra = TESTS / "test_control_flow.py"
    if extra.is_file():
        print("======== test_control_flow.py ========")
        proc = subprocess.run(
            [sys.executable, str(extra)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        print(out[-2000:] if len(out) > 2000 else out)
        if proc.returncode == 0:
            grand_ok += 1
            grand_total += 1
            print("  → 1/1 OK\n")
        else:
            grand_ok += 0
            grand_total += 1
            failed_phases.append("test_control_flow.py")
            print("  → 0/1 FAIL\n")

    print("=" * 50)
    print(f"TOTAL: {grand_ok}/{grand_total} OK")
    if failed_phases:
        print(f"FAILED PHASES: {', '.join(failed_phases)}")
        return 1
    if grand_total == 0:
        print("No tests executed.")
        return 1
    print("All phases green.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
