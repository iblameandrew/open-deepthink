# Eval protocol (quality is unevaluated)

We do **not** claim QNN / QDAD / distillation beat a single-shot prompt.
There is no paid-model quality benchmark in CI.

## What you can run for free

```bash
deepthink eval
# or: python -m deepthink.cli eval
```

This is a **structural** eval: mock LLMs, no API key. It checks that
pipelines produce the artifacts they advertise (personas, epoch maps,
dataset files, N×N basis). See `deepthink/eval_structural.py`.

```bash
deepthink estimate qnn --layers 2 --width 2 --epochs 1
deepthink estimate qdad --n 3 --steps 2
deepthink estimate distill --epochs 1
```

Call-count estimates only. They exist so you can refuse a 20×20 run
before you spend money.

## If you later have budget for a quality check

Hold the **problem set**, **judge**, and **token budget** fixed.

1. **Problems** — 8–20 items you care about (sticky bugs, design briefs).
   Write them down *before* looking at outputs.
2. **Conditions** (same model, same temperature):
   - A: single long prompt (the synthesis prompt alone)
   - B: flat panel (one layer, one epoch, no rewrite)
   - C: QNN 2×2×2 (default small)
3. **Blind rank** — shuffle A/B/C; a human or a held-out judge model
   scores usefulness 1–5 and picks a winner.
4. **Report** — mean score, win rate, **tokens used**, wall time.
   Do not report C as better unless the numbers say so.

Suggested free-tier start (still costs tokens):

```bash
deepthink qnn -p "<problem>" --layers 2 --width 2 --epochs 1 --verbose
```

Do not use 10×10 or distillation-until-budget for a first comparison.
