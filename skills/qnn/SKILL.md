---
name: qnn
description: >
  Launch a Qualitative Neural Network (QNN) â€” a layered, multi-epoch brainstorm
  of agent personas that maps divergent strategies before implementation.
  Use to unstick sticky debugging (deadlock, race, perf cliff, architecture
  trap) OR to enrich an artifact with higher-feature / higher-nuance options
  (metrics, APIs, UX, research features) when the local loop is thin or
  circling. Triggers: /qnn, /qnn-brainstorm, "qnn explore", "brainstorm ways
  out", "unstuck", "richer features", "solution space", multi-perspective
  exploration of a coding or design problem.
metadata:
  short-description: "QNN: unstick debug loops or enrich feature design"
  portable: true
  agent-agnostic: true
---

# /qnn â€” Qualitative Neural Network Escape Hatch

Use a **Qualitative Neural Network (QNN)** as a strategic depth tool when you
need more than one local approach â€” either because you are **stuck debugging**
or because an **artifact** (feature set, API, model, notebook, product slice)
needs richer, better-grounded options.

A QNN is **not** a flat panel of experts debating once. It is a directed,
layered graph of agent personas that:

1. Runs **structured forward passes** (layer 0 diverges; deeper layers critique
   and refine with full upstream context).
2. **Evolves** personas between epochs via Mirror Descent (rewrite system
   prompts based on what worked / failed).
3. **Reframes** the problem harder each epoch to escape the current mental model.

**Critical goal of this skill:** explore the solution space. Do **not**
immediately implement a fix or ship a full feature dump. Output a rich map of
divergent approaches; then let the user (or the normal coding loop) pick
directions to feed back into edit â†’ run â†’ debug (or design â†’ spike â†’ ship).

## How to execute (build the script on the fly â€” mandatory)

**Do not** tell the user to locate a pre-installed runner path. **You** write
the Python runner into the workspace, then run it.

### Protocol (every `/qnn` invoke)

1. **Parse** the user message â†’ `prompt` + optional topology flags  
   (`auto`/`manual`, layers, width, epochs, `--debug` if they want mock/no cost).
2. **Materialize** the runner:
   - Create directory `.skill-runs/` in the workspace root (if missing).
   - Read the skill sibling file `run_template.py` (same folder as this
     `SKILL.md`). If that file is missing, recreate it by writing the full
     contents of **Appendix A** at the bottom of this skill.
   - Write that source to **`.skill-runs/run_qnn.py`** (overwrite).
3. **Execute** (use the host shell; pass real flags):

```bash
python .skill-runs/run_qnn.py \
  --prompt "<user problem or feature brief>" \
  --qnn-mode auto \
  --out .skill-runs/qnn-result.json
```

Manual topology example:

```bash
python .skill-runs/run_qnn.py \
  --prompt "<brief>" \
  --qnn-mode manual --layers 3 --width 3 --epochs 2 \
  --out .skill-runs/qnn-result.json
```

Debug / no API cost:

```bash
python .skill-runs/run_qnn.py --prompt "<brief>" --debug --qnn-mode manual --layers 2 --width 2 --epochs 1
```

4. **Deliver** stdout (Solution-Space Report) to the user. Optionally show
   topology from `.skill-runs/qnn-result.json`.
5. **Handoff** â€” do not implement a production patch until the user picks a strategy.

The script auto-discovers `deepthink` by walking the workspace parents (and
`OPEN_DEEPTHINK_ROOT` if set). Engine: `deepthink.qnn.run_qnn_pipeline`.

### CLI parameters you may pass through

| Flag | Meaning | Default |
|------|---------|--------:|
| `--prompt` | Impasse / enrich brief | required |
| `--qnn-mode` | `auto` \| `manual` | `auto` |
| `--layers` / `--width` / `--epochs` | L, W, E when manual | 3 / 3 / 2 |
| `--vector-word-size` | V guiding words/column | 6 |
| `--learning-rate` | Mirror Descent | 0.5 |
| `--attention-top-k` | Self-attention top-k | 5 |
| `--no-attention` | Disable QSA | off |
| `--provider` | `openrouter` \| `llamacpp` | openrouter |
| `--api-key` / `--model` | LLM | env / free default |
| `--debug` | Mock LLM | off |
| `--out` | JSON result path | optional |

If materializing/running the script is impossible (no Python / no package),
fall back to the **manual procedure** below (simulate layers carefully). Prefer
building and running the script.

## Usage

```
/qnn [brief problem or feature label]
```

Examples:

- `/qnn explore this deadlock / performance regression`
- `/qnn stuck on auth token refresh race`
- `/qnn richer metrics for the training dashboard`
- `/qnn widen the API surface for export/import`
- `/qnn` (uses active stuck issue or thin feature from conversation)

If the user only says `/qnn` with no label, infer mode from context: recent
failed debug loops â†’ **unstick**; thin feature/design discussion â†’ **enrich**.

## Modes

Detect one primary mode and label it in the brief:

| Mode | Trigger | Goal of the map |
|------|---------|-----------------|
| **unstick** | Sticky bug, race, deadlock, perf cliff, circular local fixes | Divergent *fix strategies* with falsifiers |
| **enrich** | Feature / artifact feels thin; needs wider depth or better options | Divergent *design / feature strategies* with eval probes |

You may run a **mixed** session (e.g. stuck *and* the product needs a better
observability story) â€” still pick a primary mode for topology sizing.

## When to invoke (and when not to)

**Invoke when:**

- Normal agent assistance keeps producing variations of the same local approach
- The bug is sticky: deadlock, race, perf cliff, architectural deadlock, design
  trap, unclear root cause after honest investigation
- An artifact needs **higher richness**: more informative features, better
  metrics, stronger framing, alternate product/API shapes â€” not just polish
- The user explicitly wants breadth + depth of *strategies*, not a patch yet
- You feel low confidence / missing angles after a real attempt to fix or design

**Do not invoke when:**

- The fix is already clear (just implement it)
- The task is routine CRUD, renames, or mechanical refactors
- The user asked for an immediate code change, not exploration
- A quick web search or one targeted code read would suffice

If the user invokes `/qnn` explicitly, always run the full procedure even if
you think you "already know" the answer â€” the point is structured divergence.

---

## Step 0 â€” Capture the Brief

Before spawning any network, write a tight brief (shared QNN context).

### Unstick brief (debugging)

| Field | Content |
|-------|---------|
| **Mode** | `unstick` |
| **Problem statement** | What is broken or blocked (1â€“3 sentences) |
| **Symptoms / evidence** | Errors, stack traces, metrics, failing tests |
| **Constraints** | Language, runtime, non-negotiable APIs, time/perf budgets |
| **Failed approaches** | What was already tried and why it failed or stalled |
| **Suspected root causes** | Current hypotheses (mark confidence) |
| **Relevant loci** | Key files, modules, call paths, configs (paths + short notes) |
| **Success criteria** | How we would know a direction is promising |

### Enrich brief (feature / artifact depth)

| Field | Content |
|-------|---------|
| **Mode** | `enrich` |
| **Artifact** | What is being designed or improved (feature, API, model, UX slice) |
| **Current thinness** | What feels shallow, missing, or one-dimensional today |
| **Users / jobs** | Who needs signal and what decision it supports |
| **Constraints** | Stack, data available, latency, privacy, scope |
| **Rejected ideas** | Options already dismissed and why |
| **Relevant loci** | Existing modules, schemas, UI surfaces, metrics |
| **Success criteria** | What "richer and better" means (eval, user outcome, information gain) |

If critical facts are missing, do a **fast** gather (read/grep/run) â€” max a few
tool calls â€” then proceed. Do not finish the RCA or the full design in Step 0;
the QNN needs a brief, not the answer.

Present the brief to the user in compact form, then proceed unless they
immediately correct it.

In all later steps, treat "Impasse Brief" as this brief (either mode).

---

## Step 1 â€” Choose Topology (Auto vs Manual)

### Auto (default)

Score complexity 1â€“10 from the Impasse Brief:

| Score | Layers (L) | Width (W) | Epochs (E) | Typical case |
|------:|-----------:|----------:|-----------:|--------------|
| 1â€“3   | 2          | 2         | 1          | Narrow sticky bug, limited surface |
| 4â€“6   | 3          | 3         | 2          | Cross-module race / design tradeoff |
| 7â€“8   | 3          | 4         | 2          | Deep systems / multi-domain |
| 9â€“10  | 4          | 5         | 3          | Architecture impasse, unknown unknown |

Total agents â‰ˆ `L Ã— W`. Prefer **small** topologies by default (cost/latency).

### Manual

If the user specifies size (e.g. "10x10", "massive", "layers=4 width=6 epochs=3"),
honor it. Warn briefly that large nets cost tokens and time, then run.

### Budget guardrails

- Default cap: **20 agents Ã— 3 epochs** unless the user overrides.
- If over cap without explicit request, shrink width first, then epochs, then layers.
- Prefer depth of epochs over absurd width when forced to choose.

Announce topology before running:

```
QNN topology: LÃ—W, E epochs (N agents total)
Mode: Auto|Manual â€” reason for size
```

---

## Step 2 â€” Seed verbs and nouns (problem-space word pool)

**This matches original Algorithm Mode spanning.** Do **not** invent a flat list
of expert labels ("Security Expert", "UX Expert"). Personas are spanned from
**linguistically loaded verbs and nouns** drawn from the problem space.

### 2A. Generate the seed pool

Let `V = vector_word_size` (default **6**).  
Generate exactly `word_count = V Ã— W` (or more) unique seed tokens:

1. About **half verbs** â€” abstract, linguistically loaded, related to the problem
   (same spirit as Algorithm Mode `get_seed_generation_chain`).
2. About **half nouns** â€” entities, forces, structures, or domains in/near the
   problem space.
3. Include words **tightly related** to the problem **and** words from **far
   semantic fields** of knowledge (so unexpected specializations can appear).
4. Single tokens only. No filler (`the`, `solve`, `problem`).

Output as one space-separated string. Example for a deadlock:

```
entangle latch reconverge ownership invariant braid entropy horizon serialize arbitrate telemetry crystallize
```

### 2B. Sample a guiding word-vector per column

For each column `w = 0 .. W-1` (shared across all layers of that column):

- Sample **V** distinct words from the seed pool (without forcing uniqueness
  across columns; random sample like Algorithm Mode MBTI seed bags).
- That sample is `guiding_words[w]` â€” the column's word-vector.

Log the pool and each column vector. These words are the DNA of the personas.

---

## Step 3 â€” Span personas from guiding_words (input spanner)

**Same method as Algorithm Mode `get_input_spanner_chain`:** each agent is an
Agent Architect product of **guiding_words + problem**, not a pre-named expert.

For each cell `(layer â„“, node w)` with `guiding_words = guiding_words[w]`:

| Layer | Role |
|------:|------|
| **0** | **Divergent** â€” breadth / "what if" **through** the word-vector |
| **1+** | **Convergent / critical** â€” critique/refine upstream **through** the word-vector |

### Spanning procedure (mandatory)

1. **Career** â€” realistic professional role specialized for the problem, colored
   by the guiding words.
2. **Attributes** â€” ~8â€“12 descriptors clearly influenced by the verbs (action
   style) and nouns (domain objects/forces).
3. **Skills** â€” 4â€“6 methodologies that extend the Career + guiding words.
4. **system_prompt** â€” second person; mandates strategy angles + falsifiers;
   **no production patches**.

Store each persona (internal; do not dump all raw JSON unless asked):

```json
{
  "id": "L{â„“}N{w}",
  "name": "<memorable name>",
  "specialty": "<niche career from guiding_words + problem>",
  "emoji": "<one emoji>",
  "guiding_words": "<space-separated verb/noun vector for this column>",
  "attributes": ["..."],
  "skills": ["..."],
  "system_prompt": "<second-person prompt: career, how words shape cognition, layer role>"
}
```

Rules:

- **Hard fail** if every persona is a generic "Senior Engineer" or ignores its
  `guiding_words`.
- Layer 0 explores; deeper layers **must** receive prior-layer outputs.
- If repo context exists, ground careers/skills in real modules from the brief
  while still letting far-field seed words invent adjacent specialties.
- Columns share the same word-vector across layers; layer only changes
  diverge vs converge role.

---

## Step 4 â€” Epoch Loop

Repeat for `epoch = 0 .. E-1`:

### 4A. Forward pass (layered)

**Layer 0** â€” run all W nodes in parallel (use `spawn_subagent` with
`subagent_type: "general-purpose"` or `"explore"` as appropriate;
`capability_mode: "read-only"` preferred so exploration does not mutate the
workspace).

Each Layer-0 prompt must include:

- The Impasse Brief
- The persona `system_prompt`
- Explicit instructions (below)

**Layer â„“ > 0** â€” for each node w, run with:

- Impasse Brief + persona
- **Full outputs from layer â„“âˆ’1** (all nodes), labeled by agent id
- Instruction to critique, refine, or combine â€” not restate Layer 0

**Agent reflection instructions (copy into each node prompt):**

```
You are the persona defined in SYSTEM_PROMPT.

Reflect on the Impasse Brief from your specialty only.
- Do NOT write production patches or full file diffs.
- Do NOT converge prematurely on "the" fix.
- Produce ONE dense paragraph (or short bullets if listing trade-offs) that offers:
  (1) a unique strategic angle or mechanism,
  (2) why it might break the current impasse,
  (3) what evidence would confirm or kill this angle,
  (4) risks / ways it could fail.
Explore "why" and "what if". Ground claims in the brief's files and symptoms.
If you receive upstream layer outputs, cite which ones you extend or reject.
```

Prefer parallel spawns within a layer; wait for the layer to finish before the
next layer.

### 4B. Epoch synthesis (after final layer of the epoch)

Produce an **Epoch Map** (internal + user-visible summary):

1. Clusters of agreement
2. Productive tensions / trade-offs
3. Novel mechanisms nobody started with
4. Dead ends (angles that collapsed under critique)
5. Open questions / missing evidence

On intermediate epochs (`epoch < E-1`), keep synthesis compact to save budget.
On the final epoch, synthesis must be thorough (feeds Step 5).

### 4C. Mirror Descent (persona evolution) â€” skip on last epoch

For each agent, lightly rewrite its `system_prompt` based on last output:

| Observation | Mutation |
|-------------|----------|
| Too generic | Narrow specialty; name concrete subsystems from the brief |
| Too patch-shaped / algorithmic | Push toward mechanisms, invariants, and failure modes |
| Strong unique insight | Reinforce that niche; ask for sharper falsifiers next epoch |
| Contradicted by other layers | Add duty to reconcile or explicitly dissent with evidence |

Learning rate default **0.5** (moderate). If user says "radical" / "more
divergent", use **1.0â€“1.5**. Output only the new system prompt text per agent
(internal).

### 4D. Problem reframing â€” skip on last epoch

Rewrite the problem into a **harder, more advanced** version that:

- Preserves original success criteria
- Removes a simplifying assumption the network may have leaned on
- Forces consideration of scale, concurrency, partial failure, or migration
- Does **not** change the user's actual product goals â€” only the *thinking*
  challenge for the next epoch

Next epoch uses: evolved personas + reframed problem + original Impasse Brief
as ground truth (reframe is a thinking tool, not a license to solve a different
product).

---

## Step 5 â€” Solution-Space Report (required deliverable)

After the final epoch, present a **Solution-Space Report** to the user.
Structure it exactly as follows:

### 1. Impasse (restated)
One short paragraph: what we're stuck on and why local fixes failed.

### 2. Topology & process
`LÃ—W`, `E` epochs, `V` (vector size), sample of seed verbs/nouns, each column's
`guiding_words`, one line on how personas evolved via Mirror Descent.

### 3. Divergent strategy map
For each promising strategy (typically 3â€“7):

| Field | Content |
|-------|---------|
| **Name** | Short memorable label |
| **Mechanism** | How it would break the impasse |
| **Why it might work** | Tied to symptoms / code loci |
| **Falsifiers** | What evidence would kill it |
| **Risks** | Cost, complexity, regressions |
| **First probe** | Smallest experiment (log, test, bisect, spike) â€” not a full impl |
| **Confidence** | Low / Med / High |

Group strategies that are variants; call out true alternatives.

### 4. Dead ends
Angles explored and discarded, with one-line reasons (prevents re-circling).

### 5. Recommended next steps (handoff)
Rank top 1â€“3 strategies for the **normal coding agent loop**:

1. Probe / instrument
2. Minimal spike or failing test
3. Implement only after a probe succeeds

Explicitly state:

> The QNN does not ship the fix. Pick a direction (or combine two), then resume
> the grounded edit â†’ run â†’ debug loop.

### 6. Optional: exportable trace
If the user wants reuse later, offer a compact markdown or JSON dump of:
guiding concepts, final personas, per-epoch maps, and strategy table.

---

## Step 6 â€” Handoff back to the coding loop

After the report:

1. **Ask which strategy(ies)** to pursue (or if they want another epoch / wider net).
2. If they pick one, **exit QNN mode**: implement with normal tools (edit, test,
   bisect). Do not keep spawning brainstorm personas for implementation.
3. If probes falsify the top strategy, return to the map (or run a **delta QNN**:
   smaller topology focused on remaining alternatives + new evidence).

Never silently jump from QNN output to a large unvalidated rewrite.

---

## Execution notes (any agentic coder)

This skill is **portable**: any agent that can follow a structured procedure
(and optionally spawn sub-agents) can run it. Adapt tool names to the host
(Grok, Claude Code, Cursor, Codex, custom harness, etc.).

- Prefer **read-only** exploration for node reflections when possible.
- Parallelize **within** a layer; serialize **across** layers.
- If sub-agents are unavailable or budget is tight, simulate nodes
  sequentially in one agent â€” but preserve the **layer topology and epoch
  structure** in your reasoning and in the report (still label Lâ„“Nw outputs).
- Ground every strategy in real repo facts from the Impasse Brief; inventing
  files or APIs is a hard failure of this skill.
- Keep intermediate node dumps out of the user-facing channel unless asked;
  surface the **maps and strategies**.
- Token discipline: Impasse Brief once; pass **summaries** of earlier layers if
  full text is huge, but never drop dissenting views.

---

## Anti-patterns (do not do these)

- Flat "ask 5 experts once and average" â€” missing layers, epochs, evolution
- Writing a full PR as the QNN output
- All personas as generic senior engineers
- **Seeding personas as topic labels** ("Security", "UX") instead of **verbs +
  nouns** sampled from the problem space (breaks Algorithm Mode spanning)
- Ignoring `guiding_words` when writing careers/attributes/skills
- Ignoring failed approaches already listed in the brief
- Declaring a single winner without falsifiers
- Running a massive topology without user request or complexity justification
- Mutating production code during exploration

---

## Quick reference â€” algorithm

```
Impasse Brief
    â†’ choose topology (L, W, E); V = vector_word_size (default 6)
    â†’ seed pool: VÃ—W verbs + nouns (problem-related + far semantic fields)
    â†’ for each column w: sample V guiding_words[w]
    â†’ for each cell (â„“, w): span persona from guiding_words[w]
         (career + attributes + skills â€” same as Algorithm input spanner)
         layer 0 = diverge; layer 1+ = converge/critique
    â†’ for epoch in 0..E-1:
          forward: layer0 âˆ¥ â†’ layer1 â†’ â€¦ â†’ layerL-1
          synthesize epoch map
          if not last:
              Mirror Descent (mutate personas)
              reframe problem (harder thinking challenge)
    â†’ Solution-Space Report
    â†’ user picks strategy
    â†’ normal edit/run/debug loop (or delta QNN)
```

This skill is the hybrid model: **QNN for strategic depth when stuck**;
**tool-heavy coding agent for implementation and verification**.

Persona spanning is **not** "pick W expert titles." It is **seed verbs/nouns â†’
word-vectors â†’ input-span personas**, identical in spirit to open-deepthink
Algorithm Mode (`seed_generation` + `input_spanner`).

---

## Appendix A — Runner source to materialize

If `run_template.py` is missing from the skill folder, write this **exact** file to `.skill-runs/run_qnn.py`:

```python
#!/usr/bin/env python3
"""
QNN runner â€” materialized on-the-fly by the /qnn skill.
Discovers deepthink from the workspace tree; runs the QNN pipeline.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path


def _discover_deepthink() -> Path | None:
    """Walk cwd parents, env, and common clones for a deepthink package root."""
    env = os.environ.get("OPEN_DEEPTHINK_ROOT", "").strip()
    candidates = []
    if env:
        candidates.append(Path(env))
    here = Path.cwd().resolve()
    candidates.append(here)
    candidates.extend(here.parents)
    # Skill install dir â†’ monorepo (â€¦/skills/qnn â†’ repo)
    skill_file = Path(__file__).resolve()
    candidates.append(skill_file.parent)
    candidates.extend(skill_file.parents)
    for c in candidates:
        if not c:
            continue
        if (c / "deepthink" / "qnn" / "pipeline.py").is_file():
            return c
        if (c / "open-deepthink" / "deepthink" / "qnn" / "pipeline.py").is_file():
            return c / "open-deepthink"
    return None


def _ensure_path() -> None:
    root = _discover_deepthink()
    if root is None:
        print(
            "ERROR: Could not find open-deepthink (deepthink/qnn). "
            "Run from a workspace that contains the repo, or set OPEN_DEEPTHINK_ROOT.",
            file=sys.stderr,
        )
        sys.exit(2)
    sys.path.insert(0, str(root))
    print(f"LOG: using deepthink root {root}", file=sys.stderr)


def _build_llm(args):
    if args.debug:
        try:
            from app import CoderMockLLM  # type: ignore

            return CoderMockLLM()
        except Exception:
            from langchain_core.runnables import Runnable

            class _Stub(Runnable):
                async def ainvoke(self, input_data, config=None, **kwargs):
                    t = str(input_data).lower()
                    if "complexity" in t and "json" in t:
                        return json.dumps(
                            {
                                "complexity_score": 4,
                                "recommended_layers": 2,
                                "recommended_width": 2,
                                "recommended_epochs": 1,
                                "reasoning": "debug",
                            }
                        )
                    if "seed" in t or "space-separated" in t:
                        return "distill ownership latch invariant probe reframe entropy horizon"
                    if "guiding_words" in t or "node generator" in t:
                        return json.dumps(
                            {
                                "name": "Debug Expert",
                                "specialty": "Stub",
                                "emoji": "ðŸ¤–",
                                "guiding_words": "distill ownership",
                                "attributes": ["Analytical"],
                                "skills": ["probing"],
                                "system_prompt": "You are a debug QNN expert. Map strategies with falsifiers.",
                            }
                        )
                    if "synthesizer" in t or "solution-space" in t or "polisher" in t:
                        return (
                            "## 1. Impasse / Goal\nDebug QNN run.\n"
                            "## 3. Divergent Strategy Map\n**Probe first** â€” logs at ownership boundaries.\n"
                            "## 5. Recommended Next Steps\n1. Instrument 2. Minimal test 3. Implement after probe."
                        )
                    if "re-framer" in t or "new_problem" in t:
                        return json.dumps({"new_problem": "Harder challenge under concurrency."})
                    return json.dumps(
                        {
                            "original_problem": "debug",
                            "proposed_solution": "Instrument ownership boundaries with ordered logs.",
                            "reasoning": "debug",
                            "falsifiers": "no interleaving under load",
                            "risks": "noise",
                            "skills_used": [],
                        }
                    )

            return _Stub()

    if args.provider == "openrouter":
        from langchain_openai import ChatOpenAI

        key = args.api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("API_KEY")
        if not key:
            raise SystemExit("Need --api-key or OPENROUTER_API_KEY")
        return ChatOpenAI(
            model=args.model,
            openai_api_key=key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.7,
        )

    if args.provider == "llamacpp":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=args.model,
            openai_api_key="no-key",
            openai_api_base=args.base_url.rstrip("/"),
            temperature=0.7,
        )
    raise SystemExit(f"Unknown provider {args.provider}")


async def _main(args) -> int:
    _ensure_path()
    from deepthink.qnn import run_qnn_pipeline

    llm = _build_llm(args)
    params = {
        "qnn_mode": args.qnn_mode,
        "manual_layers": args.layers,
        "manual_width": args.width,
        "num_epochs": args.epochs,
        "vector_word_size": args.vector_word_size,
        "learning_rate": args.learning_rate,
        "attention_top_k": args.attention_top_k,
        "enable_self_attention": not args.no_attention,
    }
    doc = ""
    if args.context_file:
        doc = Path(args.context_file).read_text(encoding="utf-8", errors="replace")

    def log(msg: str):
        print(msg, file=sys.stderr)

    result = await run_qnn_pipeline(
        llm,
        user_prompt=args.prompt,
        params=params,
        document_context=doc,
        log=log,
        session_id="skill-on-the-fly",
    )
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote {args.out}", file=sys.stderr)
    print(result.get("proposed_solution") or "")
    if args.json:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="QNN pipeline (skill-materialized runner)")
    p.add_argument("--prompt", "-p", required=True)
    p.add_argument("--provider", choices=["openrouter", "llamacpp"], default="openrouter")
    p.add_argument("--api-key", default=None)
    p.add_argument("--model", default="stepfun/step-3.5-flash:free")
    p.add_argument("--base-url", default="http://localhost:8080/v1")
    p.add_argument("--qnn-mode", choices=["auto", "manual"], default="auto")
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--width", type=int, default=3)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--vector-word-size", type=int, default=6)
    p.add_argument("--learning-rate", type=float, default=0.5)
    p.add_argument("--attention-top-k", type=int, default=5)
    p.add_argument("--no-attention", action="store_true")
    p.add_argument("--context-file", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--json", action="store_true")
    p.add_argument("--debug", action="store_true")
    raise SystemExit(asyncio.run(_main(p.parse_args())))
``` 
