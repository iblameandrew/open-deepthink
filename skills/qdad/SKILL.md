---
name: qdad
description: >
  Run Qualitative Diffusion App Designer (QDAD) â€” a qualitative re-implementation
  of diffusion for app design. Turns a vague Midjourney-style product prompt into
  a concrete, buildable agentic coding prompt via an NÃ—N nounÃ—verb feature grid,
  high-temperature noise induction, iterative critic reverse diffusion, and final
  synthesis. Use when designing a new app, expanding a vague product idea, or
  generating a high-quality build brief for Grok-Build / Cursor / Claude Artifacts.
  Triggers: /qdad, /app-slot-machine, /qualitative-diffusion, "diffuse this app",
  "slot machine this idea", "turn this vibe into a build prompt", Midjourney-style
  app prompt â†’ coding prompt.
metadata:
  short-description: "QDAD: vague app vibe â†’ agentic coding prompt via qualitative diffusion"
  portable: true
  agent-agnostic: true
  technique: qualitative-diffusion
---

# /qdad â€” Qualitative Diffusion App Designer

Use **Qualitative Diffusion (QDAD)** when you need to turn a **vague aesthetic or
product vibe** into a **concrete, buildable app specification** â€” the same job
Midjourney does for images, but the latent is **language** and the decode target
is an **agentic coding prompt**.

This skill is **portable**. You do not need the open-deepthink **HTTP server**.
Same algorithm as **App Slot Machine Mode** (`deepthink/qdad/`).

## How to execute (build the script on the fly â€” mandatory)

**Do not** point the user at a pre-installed CLI path. **You** write the runner
into the workspace, then run it with parsed parameters.

### Protocol (every `/qdad` invoke)

1. **Parse** the user message â†’ Midjourney-style `prompt` + optional  
   `N=â€¦`, `steps=â€¦`, temperature flags, or `--debug`.
2. **Materialize** the runner:
   - Create `.skill-runs/` in the workspace root if missing.
   - Read skill sibling `run_template.py` (next to this `SKILL.md`). If missing,
     write **Appendix A** from the end of this skill.
   - Write source to **`.skill-runs/run_qdad.py`** (overwrite).
3. **Execute**:

```bash
python .skill-runs/run_qdad.py \
  --prompt "<Midjourney-style app intent>" \
  --n 3 \
  --temperature-scale 1.3 \
  --denoising-steps 2 \
  --noun-verb-temperature 0.6 \
  --out .skill-runs/qdad-result.json
```

Debug / no API cost:

```bash
python .skill-runs/run_qdad.py --prompt "<intent>" --debug --n 2 --denoising-steps 1
```

4. **Deliver** stdout (`# App Build Prompt`) as the primary answer.  
5. **Handoff** â€” only implement the app after the user approves.

The script auto-discovers `deepthink` (walk parents / `OPEN_DEEPTHINK_ROOT`).  
Engine: `deepthink.qdad.run_qdad_pipeline` (foundation â†’ grid â†’ noise â†’ denoise â†’ synth).

### CLI parameters

| Flag | Param | Default |
|------|-------|--------:|
| `--prompt` | user intent | required |
| `--n` | grid size N | 4 |
| `--temperature-scale` | forward noise T | 1.3 |
| `--denoising-steps` | reverse rounds | 3 |
| `--noun-verb-temperature` | foundation T | 0.6 |
| `--provider` / `--api-key` / `--model` | LLM | openrouter + env |
| `--debug` | mock LLM | off |
| `--out` | JSON dump | optional |

If you cannot write/run Python, fall back to the **manual QDAD procedure**
below. Prefer materializing and running the script.

## Technique analysis (read this; it is the algorithm)

### What classical diffusion does

1. Start from noise in a continuous latent.
2. Iteratively **denoise** toward the data manifold (score matching / reverse SDE).
3. Decode to pixels (or tokens).

### What Qualitative Diffusion does

| Classical object | Qualitative analogue |
|------------------|----------------------|
| Continuous latent vector | **Feature text** at a grid cell |
| Coordinate basis of latent space | **Nouns (rows) Ã— verbs (columns)** â€” orthogonal *language* basis |
| Gaussian noise at high Ïƒ | **High-temperature LLM generation** (â€œwild, imperfect, slightly hallucinatedâ€) |
| Denoiser / score network | **CriticAgent** with the *same* nounÃ—verb signature (reverse diffusion in language) |
| Decode network | **Synthesizer** â†’ structured App Build Prompt |
| Vague caption â†’ image | Vague Midjourney-style **app intent** â†’ **buildable coding prompt** |

### Why this is not â€œjust brainstorm featuresâ€

1. **Basis structure** â€” Features are not free-floating ideas. Each sits at a
   forced intersection `noun_i Ã— verb_j`. That is the qualitative analogue of a
   tensor product / coordinate system: diversity is *systematic*, not random.
2. **Forward then reverse** â€” Noise induction *explores*; critics *project back*
   toward intent + implementability. One-shot ideation skips the reverse process.
3. **Signature-locked critics** â€” Critic *(i,j)* shares the exact signature of
   FeatureAgent *(i,j)*. Denoising cannot â€œedit awayâ€ the basis; it cleans *along*
   that direction (score matching in one local chart of feature space).
4. **Temperature as Ïƒ** â€” GUI/params map: high **Temperature Scale** â‰ˆ more
   qualitative noise; more **Denoising Steps** â‰ˆ longer reverse chain.
5. **Decode is separate** â€” Synthesis is not â€œpick the best cell.â€ It *merges,
   prioritizes, and architectures* the clean matrix into one shippable brief.

### Philosophy (strict â€” do not dilute)

- **Language is the computational medium** (not numbers, not embeddings you
  manipulate by hand).
- **Nouns and verbs act as orthogonal basis directions.**
- **High temperature = controlled qualitative noise.**
- **Critic agents = qualitative reverse diffusion / score matching.**
- The whole process turns a vague aesthetic prompt into a concrete, buildable
  app specification **the same way Midjourney turns a vague prompt into an image.**

### When to invoke (and when not)

**Invoke when:**

- User has a **vibe / Midjourney-style** app idea (â€œcozy night writing appâ€¦â€)
- You need a **full app build brief**, not a single function
- Product surface is under-specified (features, UX, NFRs all fuzzy)
- User says `/qdad`, â€œdiffuse thisâ€, â€œslot machineâ€, â€œturn this into a build promptâ€

**Do not invoke when:**

- Task is a local bugfix or a single clear feature already specified
- User asked for an immediate small code edit only
- `/qnn` is more appropriate (stuck **debug** strategy map, not app design)

If the user invokes `/qdad` explicitly, always run the full procedure.

---

## Usage

```
/qdad [Midjourney-style app intent]
```

Examples:

- `/qdad a cozy productivity app for writers who work at night, soft dark mode, gentle notifications, offline-first`
- `/qdad N=3 steps=2 â€” minimal habit tracker that feels like a garden`
- `/qdad expand this into a full build prompt: marketplace for local makers`
- `/qdad` (uses the last vague product idea in the conversation)

### Parameters (optional; parse from user text or defaults)

| Param | Default | Range | Role |
|-------|--------:|-------|------|
| **N** (grid size) | 4 | 2â€“8 | NÃ—N feature agents |
| **Temperature Scale** (noise T) | 1.3 | 0.7â€“1.8 | Forward diffusion only |
| **Denoising Steps** | 3 | 1â€“6 | Reverse diffusion rounds |
| **Noun/Verb Temperature** | 0.6 | 0.3â€“1.0 | Foundation basis generation |

Budget: agents per noise/denoise round = **NÂ²**. Total heavy LLM calls â‰ˆ  
`1 (foundation) + NÂ² (noise) + StepsÃ—NÂ² (critics) + 1 (synth)`.  
Prefer **N=3, steps=2** when cost-sensitive; **N=4â€“5, steps=3** for rich apps.

Announce params before running:

```
QDAD: NÃ—N grid, noise_T=â€¦, steps=â€¦, noun_verb_T=â€¦
Philosophy: language=medium; nounsÃ—verbs=basis; high-T=noise; critics=reverse diffusion
```

---

## Step 0 â€” Capture the Intent Brief

Write a short brief (do not solve the product yet):

| Field | Content |
|-------|---------|
| **User prompt** | Raw Midjourney-style intent (verbatim) |
| **Users / jobs** | Who and what outcome (infer lightly if missing) |
| **Constraints** | Platform, offline, privacy, stack prefs (if any) |
| **Aesthetic** | Mood, visual language, interaction feel |
| **Non-goals** | What this is *not* (if stated or obvious) |
| **Params** | N, noise T, steps, noun/verb T |

If attachments/repo context exist, note paths that should ground features.
Present the brief compactly, then proceed unless the user corrects it.

---

## Step 1 â€” Phase 0: Foundation (qualitative basis)

Generate **exactly N distinct nouns** and **exactly N distinct verbs**.

### Basis rules

- **Nouns** = object / substance / place / affordance axes (**rows**)
- **Verbs** = action / process / transformation axes (**columns**)
- Concrete enough to ground features; mutually distinct; span the *aesthetic
  space* of the intent (not just synonyms of â€œappâ€ / â€œuserâ€)
- Prefer evocative, implementable words over pure abstractions

### Prompt skeleton (foundation)

```
You are the QDAD Foundation Generator.

QUALITATIVE COMPUTATION CONTRACT
- Language is the computational medium (not numbers).
- Nouns and verbs are orthogonal basis directions of feature space.
- A feature is a language-vector at the intersection of one noun and one verb.

User prompt:
---
{user_prompt}
---

Generate exactly {N} distinct nouns and {N} distinct verbs.
Output ONLY JSON: {"nouns":[...], "verbs":[...]}
```

Use **noun_verb_temperature** if the host supports temperature; otherwise ask
for slightly more diverse / surprising basis words when N is small.

**Log:** `nouns = [â€¦]`, `verbs = [â€¦]`.

Hard fail: fewer than N unique items, or all generic (â€œdataâ€, â€œmanageâ€, â€œsystemâ€).

---

## Step 2 â€” Phase 1: Agent grid construction

For each `i in 0..N-1`, `j in 0..N-1`:

```
FeatureAgent_{i}_{j}
  noun = nouns[i]
  verb = verbs[j]
  signature = noun Ã— verb
```

Permanent assignment. No reassignment later. Log a compact grid:

```
        verb0    verb1    â€¦
noun0   A00      A01
noun1   A10      A11
â€¦
```

---

## Step 3 â€” Phase 2: Noise induction (forward diffusion)

**In parallel** (or sequential if no sub-agents), for every cell `(i,j)`:

### FeatureAgent system prompt

```
You are FeatureAgent_{i}_{j}.
Your unique qualitative signature is noun "{noun}" Ã— verb "{verb}".
Your sole purpose is to invent exactly ONE concrete, implementable feature
for an application. The feature must feel like a natural expression of the
interaction between "{noun}" and "{verb}" given the user intent.

User intent:
---
{user_prompt}
---

FORWARD DIFFUSION: Invent one wild, imperfect, slightly hallucinated but still
related feature. Embrace controlled qualitative noise. Rough edges and odd
metaphors are allowed â€” they are the language analogue of Gaussian noise.
Stay in orbit of the intent.

Output ONLY the feature (2â€“6 sentences). No JSON. No headings.
```

Use **noise temperature** (Temperature Scale) for these calls.

Collect `noisy_features[i][j]`.

**Do not** polish yet. Noise is a feature of the algorithm.

---

## Step 4 â€” Phase 3: Iterative qualitative denoising

For `step = 1 .. Denoising_Steps`:

**In parallel**, for every cell `(i,j)`, spawn **CriticAgent_{i}_{j}** with the
**exact same** noun+verb signature:

```
You are CriticAgent_{i}_{j}.
You share signature noun "{noun}" + verb "{verb}".
You are the inverse of noise induction: qualitative reverse diffusion / score matching.
Clean imperfections, remove contradictions, sharpen original intent, make the
feature coherent, useful, and implementable â€” while remaining a true expression
of "{noun}" + "{verb}".

User intent:
---
{user_prompt}
---

Denoising step: {step} of {total_steps}
(Early steps: gross noise. Late steps: fidelity to intent.)

Current feature:
---
{current_feature}
---

Output ONLY the refined feature (2â€“6 sentences).
```

Use a **cooler** temperature than noise (â‰ˆ `0.5 Ã— noise_T`, clamped to ~0.3â€“1.0).

Replace `features[i][j]` with the critic output after each full parallel round.

Optional: keep snapshots `step_1`, `step_2`, â€¦ for transparency.

After the final step: `clean_features = features`.

---

## Step 5 â€” Phase 4: Synthesis (decode)

One **Synthesizer** agent receives:

- Original user prompt
- Nouns, verbs
- Full clean NÃ—N matrix (each cell labeled with nounÃ—verb)

### Required output format (exact)

```markdown
# App Build Prompt

## High-Level Vision
[1-2 sentence summary]

## Core Features (synthesized & prioritized from the diffusion matrix)
1. ...
2. ...
...

## Technical Architecture Suggestions
- ...

## UI/UX Direction
- ...

## Non-Functional Requirements
- ...

## Implementation Notes for the Coding Agent
- Build this as a complete, runnable application.
- Prefer modern, clean tech (React/Next.js + Tailwind, or Streamlit, or whatever fits best).
- Make it beautiful and immediately usable.
```

### Synthesizer rules

- **Deduplicate and prioritize** â€” merge related cells; do not dump NÂ² features.
- Features must feel like **coherent expressions of intent**, not a laundry list.
- Be **concrete and implementable**.
- Prefer modern clean stacks; match constraints from the brief.
- Optionally append a **## Diffusion Feature Matrix (transparency)** section
  with nouns, verbs, and each clean cell for auditability.

---

## Step 6 â€” Handoff to the coding loop

After the App Build Prompt:

1. Show the **App Build Prompt** as the primary deliverable.
2. Optionally show a **compact matrix summary** (not all raw noise unless asked).
3. Ask: **Build now?** / tweak params (N, steps) / re-diffuse a subspace?
4. If the user says build:
   - **Exit QDAD mode**
   - Implement with normal agentic coding tools (edit, run, test)
   - Use the App Build Prompt as the system of record for scope
5. Do **not** re-run full diffusion for every code tweak unless the product
   direction changed.

---

## Execution notes (Grok-Build and any host)

| Capability | How to run QDAD |
|------------|-----------------|
| **Parallel sub-agents** | Spawn NÂ² FeatureAgents / CriticAgents per phase; gather results |
| **Single agent only** | Simulate the grid sequentially; still label every cell `(i,j)` and preserve phases |
| **Temperature** | Set per phase if API allows; else prompt for â€œwilderâ€ vs â€œstricterâ€ |
| **Model-agnostic** | Any capable chat model works; stronger models â†’ better basis + synth |
| **Read-only** | Prefer no workspace mutation until Step 6 handoff |
| **Cost control** | Default N=3â€“4, steps=2â€“3; never N=8Ã—steps=6 without explicit ask |

### Parallelization contract

```
foundation: 1 call
noise:      NÂ² calls in parallel
for step in 1..Steps:
    denoise: NÂ² calls in parallel
synthesize: 1 call
```

### Sub-agent prompt packaging

When spawning, always include: cell `(i,j)`, noun, verb, user prompt, phase
instructions, and (for critics) current feature + step index.

---

## Anti-patterns (do not do these)

- Flat â€œlist 10 featuresâ€ without a nounÃ—verb grid
- Skipping noise (going straight to â€œgoodâ€ features) â€” kills exploration
- Skipping critics (shipping raw noise) â€” kills implementability
- Critics that **change** the noun/verb signature
- Synthesizer that pastes all NÂ² cells without prioritization
- Implementing a full app **inside** the diffusion steps
- Using only generic basis words (data, user, manage, system)
- Running N=8 by default

---

## Quick reference â€” algorithm

```
Intent Brief + params (N, noise_T, steps, nv_T)
    â†’ Phase 0 Foundation: N nouns, N verbs   [nv_T]
    â†’ Phase 1 Grid: FeatureAgent_i_j := (nouns[i], verbs[j])
    â†’ Phase 2 Noise: âˆ€(i,j) parallel feature @ noise_T
    â†’ Phase 3 Denoise: for step in 1..steps:
          âˆ€(i,j) parallel CriticAgent_i_j (same signature) @ cooler T
    â†’ Phase 4 Synthesize: clean matrix + prompt â†’ App Build Prompt
    â†’ Handoff: user approves â†’ normal build loop
```

### Relation to open-deepthink

| Artifact | Role |
|----------|------|
| This skill (`/qdad`) | Portable procedure for any agentic coder |
| App Slot Machine Mode UI | Full server UI + logs + matrix persistence |
| `deepthink/qdad/` | LangGraph reference implementation |
| `/qnn` skill | Different technique: layered strategy maps for **stuck debug / enrich** |

**QDAD designs apps from vibes. QNN maps strategies when stuck. Do not conflate.**

---

## Minimal worked sketch (N=2, steps=1)

Intent: *â€œcozy night writing app, soft dark mode, offline-firstâ€*

```
nouns: [lantern, notebook]
verbs: [whisper, weave]

noisy[0][0] lanternÃ—whisper â†’ wild ambient voice notes idea
noisy[0][1] lanternÃ—weave   â†’ wild link-glow between drafts
â€¦
critic cleans each cell toward offline-first + calm UX
synth â†’ App Build Prompt with prioritized features + architecture
```

End of skill.

---

## Appendix A — Runner source to materialize

If `run_template.py` is missing from the skill folder, write this **exact** file to `.skill-runs/run_qdad.py`:

```python
#!/usr/bin/env python3
"""
QDAD / App Slot Machine runner â€” materialized on-the-fly by the /qdad skill.
Discovers deepthink from the workspace tree; runs qualitative diffusion.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path


def _discover_deepthink() -> Path | None:
    env = os.environ.get("OPEN_DEEPTHINK_ROOT", "").strip()
    candidates = []
    if env:
        candidates.append(Path(env))
    here = Path.cwd().resolve()
    candidates.append(here)
    candidates.extend(here.parents)
    skill_file = Path(__file__).resolve()
    candidates.append(skill_file.parent)
    candidates.extend(skill_file.parents)
    for c in candidates:
        if not c:
            continue
        if (c / "deepthink" / "qdad" / "pipeline.py").is_file():
            return c
        if (c / "open-deepthink" / "deepthink" / "qdad" / "pipeline.py").is_file():
            return c / "open-deepthink"
    return None


def _ensure_path() -> None:
    root = _discover_deepthink()
    if root is None:
        print(
            "ERROR: Could not find open-deepthink (deepthink/qdad). "
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
                    if "foundation" in t or "distinct nouns" in t:
                        return json.dumps(
                            {
                                "nouns": ["canvas", "lantern", "notebook", "harbor"],
                                "verbs": ["whisper", "weave", "anchor", "glow"],
                            }
                        )
                    if "featureagent" in t or "forward diffusion" in t:
                        return "A wild mock feature: ambient focus with offline capture."
                    if "criticagent" in t or "reverse diffusion" in t:
                        return "A refined mock feature: offline-first focus mode with soft glow."
                    if "synthesizer" in t or "app build" in t:
                        return (
                            "# App Build Prompt\n\n## High-Level Vision\n"
                            "Cozy offline writing app.\n\n## Core Features\n1. Focus timer\n"
                            "2. Offline draft capture\n"
                        )
                    return "mock feature"

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
    from deepthink.qdad import run_qdad_pipeline

    llm = _build_llm(args)
    params = {
        "grid_size": args.n,
        "n": args.n,
        "temperature_scale": args.temperature_scale,
        "denoising_steps": args.denoising_steps,
        "noun_verb_temperature": args.noun_verb_temperature,
    }
    doc = ""
    if args.context_file:
        doc = Path(args.context_file).read_text(encoding="utf-8", errors="replace")

    def log(msg: str):
        print(msg, file=sys.stderr)

    result = await run_qdad_pipeline(
        llm=llm,
        params=params,
        user_prompt=args.prompt,
        document_context=doc,
        log=log,
        session_id="skill-on-the-fly",
    )
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote {args.out}", file=sys.stderr)
    print(result.get("proposed_solution") or json.dumps(result, indent=2))
    if args.json:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="QDAD pipeline (skill-materialized runner)")
    p.add_argument("--prompt", "-p", required=True)
    p.add_argument("--provider", choices=["openrouter", "llamacpp"], default="openrouter")
    p.add_argument("--api-key", default=None)
    p.add_argument("--model", default="stepfun/step-3.5-flash:free")
    p.add_argument("--base-url", default="http://localhost:8080/v1")
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--temperature-scale", type=float, default=1.3)
    p.add_argument("--denoising-steps", type=int, default=3)
    p.add_argument("--noun-verb-temperature", type=float, default=0.6)
    p.add_argument("--context-file", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--json", action="store_true")
    p.add_argument("--debug", action="store_true")
    raise SystemExit(asyncio.run(_main(p.parse_args())))
``` 
