"""Phase honesty: cost estimator, structural eval, GraphState identity, docs."""

from __future__ import annotations

import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(ROOT))

results = []


def chk(name, fn):
    try:
        fn()
        results.append((name, "OK", None))
    except Exception as e:
        tb = traceback.format_exc().splitlines()[-3:]
        results.append((name, "FAIL", f"{type(e).__name__}: {e} | " + " | ".join(tb)))


def t1():
    from deepthink.cost import estimate_qdad_cost, estimate_qnn_cost

    qnn = estimate_qnn_cost(2, 2, 1, qnn_mode="manual")
    assert qnn.llm_calls == 11
    qdad = estimate_qdad_cost(3, 2)
    assert qdad.llm_calls == 2 + 9 * 3


chk("cost estimator formulas", t1)


def t2():
    from app import GraphState as AppGS
    from deepthink.state import GraphState as LibGS

    assert AppGS is LibGS


chk("app.GraphState is deepthink.state.GraphState", t2)


def t3():
    from deepthink.mocks import CoderMockLLM, DistillationMockLLM

    assert callable(CoderMockLLM().ainvoke)
    assert callable(DistillationMockLLM().ainvoke)


chk("mocks live in deepthink.mocks (library, not only app)", t3)


def t4():
    design = (ROOT / "docs" / "DESIGN_NOTES.md").read_text(encoding="utf-8")
    eval_doc = (ROOT / "docs" / "EVAL.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "lexical overlap" in design.lower() or "token-overlap" in design.lower()
    assert "unevaluated" in eval_doc.lower()
    assert "unevaluated" in readme.lower()
    assert "0.3.0" in readme or "0.3" in readme


chk("docs state metaphors vs ops and unevaluated quality", t4)


def t5():
    from deepthink.eval_structural import run_structural_eval_sync

    result = run_structural_eval_sync()
    assert result.total >= 8
    assert result.passed >= result.total - 2, result.to_dict()


chk("structural eval mostly green on mocks", t5)


def t6():
    from deepthink.cli import build_parser

    help_txt = build_parser().format_help()
    assert "estimate" in help_txt and "eval" in help_txt


chk("CLI exposes estimate + eval", t6)


for name, status, err in results:
    line = f"  [{status}] {name}"
    if err:
        line += f" :: {err}"
    print(line)
ok = sum(1 for _, s, _ in results if s == "OK")
print(f"\nPHASE HONESTY: {ok}/{len(results)} OK")
if ok != len(results):
    raise SystemExit(1)
