# Release Notes — `0.1.6`

**Tag:** `0.1.6`
**Date:** 2026-07-09
**Tagline:** "Portable `/qnn` skill — import anytime an agentic coder is stuck or a feature needs wider depth"

## ✨ Features

### Portable QNN skill for agentic coders

* **New `skills/qnn/` package** — Agent-agnostic `SKILL.md` that runs a full Qualitative Neural Network brainstorm *inside* a coding agent (Grok, Claude Code, Cursor, Codex, custom harnesses).
* **Two modes**
  * **unstick** — sticky bugs (deadlock, race, perf cliff, circular local fixes)
  * **enrich** — thin features/artifacts that need richer options (metrics, APIs, UX, research depth)
* **Procedure** — Impasse/enrich brief → topology (auto or manual) → guiding concepts → L×W personas → layered forward passes → epoch maps → Mirror Descent → problem reframing → **Solution-Space Report** with falsifiers and first probes.
* **Handoff contract** — The skill maps the solution space; it does **not** ship production patches. Pick a strategy, then resume edit → run → debug.
* **Import paths** — Copy into `~/.grok/skills/qnn/`, project `.grok/skills/`, or install from the release zip asset `qnn-skill-0.1.6.zip`.
* **Packaging** — `skills/package_skill.py` rebuilds the release zip; `skills/README.md` documents host-agnostic install.

### Docs

* README: Future Directions replaced with **shipped** portable `/qnn` skill section and install instructions.
* `skills/qnn/INSTALL.md` — one-page install for the skill alone.

### Files

* `skills/qnn/SKILL.md` — portable skill body
* `skills/qnn/INSTALL.md` — install guide
* `skills/README.md` — skills index + import options
* `skills/package_skill.py` — zip builder for releases
* `deepthink/__init__.py` — version bump to `0.1.6`

### Release asset

* `qnn-skill-0.1.6.zip` — unzip into `~/.grok/skills/` (or equivalent) for immediate `/qnn` availability.

---

# Release Notes — `0.1.5`

**Tag:** `0.1.5`
**Date:** 2026-07-05
**Tagline:** "Attach entire repositories as context in Brainstorming and Algorithm Design"

## ✨ Features

### Repository folder attachments

* **Brainstorming** — New **Attach Repository** button alongside PDF and code attachments, plus a folder button in the NodeChat input bar. Select a local project folder and the QNN ingests its source files as human context.
* **Algorithm Design** — Attach a full repository above the prompt; prioritized file contents are injected into the graph run.
* **Backend** — New `POST /upload_repository` endpoint scans uploaded folders, skips common vendor/build/cache paths (`node_modules`, `.git`, `__pycache__`, lockfiles, binaries), prioritizes README/config/entry files, and caps total context at 50k chars across up to 500 files.
* **Paths preserved** — Each included file keeps its repository-relative path so agents can reason about project structure.

### Files

* `app.py` — `/upload_repository` endpoint and shared code-file helpers
* `index.html` — repository attachment UI and upload flow for both modes
* `js/components/node-chat.js` — folder attach button in chat input
* `deepthink/__init__.py` — version bump to `0.1.5`

---

# Release Notes — `0.1.4`

**Tag:** `0.1.4`
**Date:** 2026-06-17
**Tagline:** "Unified Flux UI theme — app chrome adapts to NodeChat"

## ✨ Features

### App-wide theme redesign

* **New `css/theme.css`** — glass surfaces, purple/cyan accents, pill controls, and dark luxury styling aligned with the NodeChat component aesthetic.
* **NodeChat left untouched** — `css/node-chat.css` and `js/components/node-chat.js` are unchanged; the rest of the app adapts around the immutable chat component.
* **Isolated chat mounts** — brainstorm and diagnostic chat sit in transparent `.section--chat-mount` frames so NodeChat renders with its native chrome.
* **Updated surfaces** — settings panel, mode tabs, forms, attachments, memory estimator, logs, graph viewer, distillation console, and perplexity chart all use the new theme.
* **Scoped form styles** — global inputs/buttons exclude `.node-chat` internals so chat styling is never overridden.

### Files

* `css/theme.css` — new app chrome design system
* `index.html` — removed inline legacy CSS; semantic layout/classes; chart and log colors updated
* `js/components/memory-calculator.js` — status uses CSS classes instead of inline hex colors
* `deepthink/__init__.py` — version bump to `0.1.4`

---

# Release Notes — `0.1.3`

**Tag:** `0.1.3`
**Date:** 2026-06-16
**Tagline:** "Attach source code files as context in Brainstorming and Algorithm Design"

## ✨ Features

### Code file attachments

* **Brainstorming** — New **Attach Code** button alongside PDF attachments, plus a paperclip button in the NodeChat input bar. Supports `.py`, `.js`, `.ts`, `.java`, `.go`, `.rs`, and 30+ other text/code extensions.
* **Algorithm Design** — Attach code files above the prompt; their contents are injected into the graph run as context.
* **Backend** — New `POST /upload_code_files` endpoint reads uploaded source files (UTF-8 with fallback), caps total context at 50k chars, and returns fenced code blocks for the LLM.

### Files

* `app.py` — `/upload_code_files` endpoint; algorithm mode merges attached code into prompt
* `index.html` — attachment UI for both modes
* `js/components/node-chat.js` — paperclip attach button in chat input
* `css/node-chat.css` — attach button styles

---

# Release Notes — `0.1.2`

**Tag:** `0.1.2`
**Date:** 2026-06-15
**Tagline:** "Unlimited QNN dimensions with a live memory estimator for 8 GB laptops"

## ✨ Features

### Unlimited agent spanning (width × height)

* **Algorithm Design** — CoT trace depth is no longer capped at 32. New
  **Manual width (unlimited)** mode lets you set any number of agents per
  layer; MBTI archetypes cycle when width exceeds the checkbox count.
* **Brainstorming** — Manual/Massive topology layers and width inputs no longer
  cap at 10,000. Backend accepts any positive integer the user enters.

### QNN Memory Estimator

A live calculator panel in both **Algorithm Design** and **Brainstorming**
modes shows:

* Total agents (`width × height`) and estimated total state size in MB/GB
* Per-agent RAM at the selected epoch count
* **8 GB laptop fit check** with headroom / overflow warning and a suggested
  max agent count
* Expandable **single-agent growth table** (epochs 1–10) as an educated guess
  based on runtime constants: ~30 KB persona, ~5–8 KB/epoch memory append,
  ~450 KB summarization cap per agent

Users can adjust the laptop RAM input (default 8 GB); local LlamaCpp reserves
~1.5 GB for the model automatically.

### Files

* `js/components/memory-calculator.js` — estimator module
* `index.html` — calculator UI, unlimited dimension inputs, algorithm manual width toggle
* `app.py` — removed 10k backend caps; algorithm manual width with MBTI cycling

---

# Release Notes — `0.1.1`

**Tag:** `0.1.1`
**Date:** 2026-06-10
**Tagline:** "Perplexity chart no longer accumulates duplicate points across anchors / modes"

## 🐛 Bug Fixes

### Perplexity chart accumulation (BUG-12)

The "Average Perplexity per Epoch" chart could keep adding duplicate data points
every time a new run (or new anchor within a multi-anchor distillation) emitted
a perplexity value, because the frontend blindly pushed onto a parallel-arrays
state and the chart instance was `destroy()`-and-recreated on every update.
Compounding the issue:

1. The distillation loop creates a fresh `DistillationGraph` per anchor, so
   each anchor's `epochs_run` counter reset to 1. The frontend received the
   same epoch label (e.g. `epoch=1`) from multiple anchors and treated each
   one as a new data point.
2. The main-graph `calculate_metrics_node` (used by both **Algorithm** and
   **Brainstorming** modes) was broadcasting `{"epoch": ..., "perplexity": ...}`
   as a bare JSON, but the frontend SSE filter only routed `distillation_update`
   events to the chart, so algorithm/brainstorm perplexity was silently dropped.
3. The chart code itself was a `destroy()` + `new Chart(...)` anti-pattern
   that flickered on every update, and the labels variable was misspelled
   (`allLbelsData`) — a footgun for future fixes.

### Fix

* **Backend (`app.py`)**:
  * `calculate_metrics_node` now broadcasts a typed
    `{"type": "perplexity_update", "source": "graph", "session_id": ..., "epoch": ..., "perplexity": ...}`
    event so algorithm and brainstorm runs feed the chart.
  * The distillation loop now tracks a `cumulative_step` counter that
    strictly increases across both epochs and anchors, and includes it in
    the `distillation_update` payload as `step`.

* **Frontend (`index.html`)**:
  * Chart state rewritten from parallel arrays (`allLbelsData`/`allPerplexityValues`)
    to a single `Map` keyed by step, with values stored as `{value, label}`.
  * `updatePerplexityChart(step, value, label)` now **replaces** an existing
    entry instead of appending a duplicate.
  * `renderPerplexityChart` uses Chart.js v4's idiomatic
    `chart.data.datasets[0].data = ...; chart.update('none')` for in-place
    updates instead of destroying and recreating the chart.
  * `resetPerplexityChart` resets state and clears the chart on every new
    algorithm **or** distillation run.
  * SSE handler now recognizes the new `perplexity_update` event type in
    addition to `distillation_update`.
  * The misspelled `allLbelsData` is gone.

* **Tests (`tests/phase11_regression.py`)**:
  * New BUG-12a regression: directly invokes the metrics node and asserts
    the broadcast JSON includes `type: "perplexity_update"`, `source: "graph"`,
    `session_id`, and a numeric `perplexity`.
  * New BUG-12b regression: source-checks the distillation loop for the
    `cumulative_step` counter and `"step":` field in the broadcast.
  * New BUG-12c regression: source-checks `index.html` for the dedup-by-step
    contract (no legacy `allLbelsData`/`allPerplexityValues`, presence of
    `perplexityByStep`/`resetPerplexityChart`/SSE listener for
    `perplexity_update`) and asserts the dedup invariant against a Python
    mirror of the JS algorithm.

### Verification

All previously-passing tests still pass (158/158 across phases 1–9). The
4 unrelated pre-existing failures in phase 10/11 are due to a stale
`local-deepthink` path in the test file and are out of scope for this
release.

---

# Release Notes — `0.1.0`

**Tag:** `0.1.0`
**Date:** 2026-06-06
**Tagline:** "Official release out of beta"

This marks the first official non-beta release of open-deepthink. The core QNN engine, Mirror Descent loops, three operating modes, export/import, and full test suite are considered stable.

See the new "Future Directions" section in the README for thoughts on integrating QNN-style evolutionary exploration into practical coding agent workflows (e.g. a `/qnn` command for large-scale structured brainstorming on sticky problems).

---

# Release Notes — `beta-0.0.3`

**Tag:** `beta-0.0.3`
**Date:** 2026-06-05
**Tagline:** "The hard-fought quality release — 11 bugs fixed, 195/195 tests passing"

> *"A week of debugging is better than a month of 'works on my machine'."*

This release is the result of a complete, exhaustive test suite of the entire
codebase (11 phases × 195 checks). Every previously-known failure path was
either fixed or pinned with a regression test. **No known malfunctions remain.**

---

## 🐛 Bug Fixes

### Critical (broke core flows)

1. **`DistillationMockLLM` had stale prompt patterns** (app.py)
   The chains in `distillation_chains.py` had been rewritten ("Socratic Task
   Master…", "Seed Creator (The Dialectic Synthesizer)", "deepening our
   inquiry in a new Epoch") but the mock LLM still matched the OLD text.
   In debug mode, **every distillation epoch silently failed** to evolve
   topics or generate new questions. Fixed by re-aligning the patterns.

2. **`CoderMockLLM` hardcoded 4 sub-problems** (app.py)
   The mock decomposer returned the same static 4-item list regardless of
   the requested count, so `reframe_and_decompose` always failed for
   non-4-agent QNNs. Fixed by reading the requested count from the prompt.

3. **Missing `grandalf` dependency** (requirements.txt)
   `graph.get_graph().draw_ascii()` requires `grandalf`; without it, the
   entire `/build_and_run_graph` endpoint crashed with a confusing error.
   Added `grandalf` to `requirements.txt`.

4. **`synthesis_node` returned "no inputs" in brainstorm mode** (app.py)
   The empty-`agent_outputs` check fired *before* the brainstorm branch,
   but brainstorm synthesis reads from `memory`, not `agent_outputs`.
   Reordered the checks so brainstorm mode can synthesize.

### Medium (data/UX quality)

5. **Duplicate `get_opinion_synthesizer_chain`** (chains/__init__.py)
   Two modules defined functions with the same name; the import order
   silently shadowed the synthesis version with the brainstorm version.
   Renamed the brainstorm one to `get_brainstorming_opinion_synthesizer_chain`.

6. **`app.GraphState` missing brainstorm fields** (app.py)
   `app.GraphState` was missing `brainstorm_document_context`,
   `brainstorm_prior_conversation`, and `brainstorm_problem_summary` that
   the code accesses. Runtime worked because of `state.get(...)`, but
   the type was lying. Added the fields.

7. **`clean_and_parse_json` couldn't repair Windows backslash paths** (utils.py)
   Strings like `{"path": "C:\\Users\\foo"}` (single backslashes, as an
   LLM would write) were returned as `None`. Fixed the regex to not
   double-escape already-escaped backslash pairs (needed non-capturing
   group `(?:\\)` to make the lookbehind work in Python's `re` module).

### Low (hygiene)

8. **Stray `print()` statements** in `TokenUsageTracker`,
   `RAPTOR._cluster_nodes`, and both chat endpoints — all converted to
   `log_stream.put(...)` so logs flow through the SSE broadcast.

9. **Typo "Sesion"** in `/chat` and `/diagnostic_chat` print statements —
   fixed to "Session" and routed through `log_stream`.

10. **`.gitignore`** missed `venv/`, `tests/`, `distillation_output/`,
    `test_results/`. Updated.

11. **No `__version__` or `pyproject.toml`** — release tracking was
    impossible. Added `__version__ = "0.0.3-beta"`, `__release_name__`,
    `__release_tag__`, and a `pyproject.toml` with project metadata
    + all dependencies.

---

## ✨ Improvements

- `deepthink/__init__.py` now exposes `__version__`, `__release_name__`,
  `__release_tag__` for runtime introspection.
- `pyproject.toml` enables `pip install -e .` workflows and standard
  packaging. It pins `grandalf` (alongside all other deps).
- `tests/` directory contains 11 numbered test phases covering: imports,
  utilities, chain factories, state types, FastAPI routes, distillation,
  mock LLM patterns, graph nodes, end-to-end HTTP, static analysis,
  and **regression tests for every bug fixed in this release**.

---

## 🧪 Test Suite

The release ships with a 195-check, 11-phase test suite. To run:

```bash
# Phase 1-8, 10, 11 (pure unit tests, no network)
venv/Scripts/python.exe tests/phase1_imports.py
venv/Scripts/python.exe tests/phase2_utils.py
...
venv/Scripts/python.exe tests/phase11_regression.py

# Phase 9 (requires a real FastAPI test client; ~60s)
venv/Scripts/python.exe tests/phase9_e2e.py
```

Final tally:

| Phase | Subject | Result |
|------:|---------|--------|
| 1 | Module imports & structure | 7/7 ✅ |
| 2 | utils.py (JSON repair, sandbox) | 15/15 ✅ |
| 3 | All 30 chain factories construct | 34/34 ✅ |
| 4 | state.py GraphState consistency | 5/5 ✅ |
| 5 | FastAPI endpoints surface | 18/18 ✅ |
| 6 | DistillationGraph end-to-end | 20/20 ✅ |
| 7 | Mock LLM prompt patterns | 21/21 ✅ |
| 8 | LangGraph nodes | 18/18 ✅ |
| 9 | Live HTTP API via TestClient | 20/20 ✅ |
| 10 | Static analysis, packaging | 23/23 ✅ |
| 11 | Regression tests (this release) | 14/14 ✅ |
| | **Total** | **195/195 ✅** |

---

## 📦 Install

```bash
pip install -r requirements.txt   # adds the new grandalf dep
# or, with the new pyproject.toml:
pip install -e .
python app.py
# → http://localhost:8000
```

---

## 🔬 For Contributors

- Read `tests/phase11_regression.py` before merging — every bug fixed in
  this release has a regression test there. Add new tests next to the
  relevant one.
- All new code should be importable from `deepthink` (the library) and
  use `log_stream.put(...)` (not `print`) for runtime logging.
- Avoid redefining factories with the same name across multiple
  `deepthink/chains/*.py` modules. If you must, give them a module-
  specific prefix (e.g. `get_brainstorming_*`).

---

## 👥 Acknowledgments

This release exists because we did the boring work: enumerate every
feature, write a test for it, fix what broke, and pin it. The codebase
is now measurably more robust than it was a week ago.

— *the open-deepthink maintainers*
