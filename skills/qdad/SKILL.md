---
name: qdad
description: >
  Run Qualitative Diffusion App Designer (QDAD) — a qualitative re-implementation
  of diffusion for app design. Turns a vague Midjourney-style product prompt into
  a concrete, buildable agentic coding prompt via an N×N noun×verb feature grid,
  high-temperature noise induction, iterative critic reverse diffusion, and final
  synthesis. Use when designing a new app, expanding a vague product idea, or
  generating a high-quality build brief for Grok-Build / Cursor / Claude Artifacts.
  Triggers: /qdad, /app-slot-machine, /qualitative-diffusion, "diffuse this app",
  "slot machine this idea", "turn this vibe into a build prompt", Midjourney-style
  app prompt → coding prompt.
metadata:
  short-description: "QDAD: vague app vibe → agentic coding prompt via qualitative diffusion"
  portable: true
  agent-agnostic: true
  technique: qualitative-diffusion
---

# /qdad — Qualitative Diffusion App Designer

Use **Qualitative Diffusion (QDAD)** when you need to turn a **vague aesthetic or
product vibe** into a **concrete, buildable app specification** — the same job
Midjourney does for images, but the latent is **language** and the decode target
is an **agentic coding prompt**.

This skill is **portable**. You do not need the open-deepthink **HTTP server**.
Prefer running the **reference code** (below). Procedural steps later in this
file are the fallback when the package is unavailable. Same algorithm as
open-deepthink **App Slot Machine Mode** (`deepthink/qdad/`).

## Run the code (preferred over pure simulation)

When **open-deepthink** is on `PYTHONPATH` / installed / `OPEN_DEEPTHINK_ROOT`,
the harness **must execute the QDAD engine** and pass structured parameters.

| | |
|--|--|
| **Library** | `from deepthink.qdad import run_qdad_pipeline` |
| **Source** | `deepthink/qdad/pipeline.py` (+ `graph.py`, `nodes.py`) |
| **CLI** | `python skills/qdad/run_qdad.py --prompt "…" [flags]` |
| **Full contract** | [`CODE_REFERENCE.md`](./CODE_REFERENCE.md) |

```bash
export OPEN_DEEPTHINK_ROOT=/path/to/open-deepthink
export OPENROUTER_API_KEY=sk-...
python skills/qdad/run_qdad.py \
  --prompt "cozy night writing app, soft dark mode, offline-first" \
  --n 3 --temperature-scale 1.3 --denoising-steps 2 \
  --noun-verb-temperature 0.6 \
  --out /tmp/qdad.json
```

```python
result = await run_qdad_pipeline(
    llm=llm,
    params={
        "grid_size": 4,                # N  (also accept "n")
        "n": 4,
        "temperature_scale": 1.3,      # forward diffusion noise T
        "denoising_steps": 3,          # reverse diffusion rounds
        "noun_verb_temperature": 0.6,  # foundation basis T
    },
    user_prompt=midjourney_style_intent,
    document_context=optional_docs,
    log=print_progress,
)
# Deliver result["proposed_solution"]  # # App Build Prompt
```

Parse user text for overrides (`N=3`, `steps=2`, etc.) into these params.
If the package is not importable, fall back to the multi-step procedure below.

## Technique analysis (read this; it is the algorithm)

### What classical diffusion does

1. Start from noise in a continuous latent.
2. Iteratively **denoise** toward the data manifold (score matching / reverse SDE).
3. Decode to pixels (or tokens).

### What Qualitative Diffusion does

| Classical object | Qualitative analogue |
|------------------|----------------------|
| Continuous latent vector | **Feature text** at a grid cell |
| Coordinate basis of latent space | **Nouns (rows) × verbs (columns)** — orthogonal *language* basis |
| Gaussian noise at high σ | **High-temperature LLM generation** (“wild, imperfect, slightly hallucinated”) |
| Denoiser / score network | **CriticAgent** with the *same* noun×verb signature (reverse diffusion in language) |
| Decode network | **Synthesizer** → structured App Build Prompt |
| Vague caption → image | Vague Midjourney-style **app intent** → **buildable coding prompt** |

### Why this is not “just brainstorm features”

1. **Basis structure** — Features are not free-floating ideas. Each sits at a
   forced intersection `noun_i × verb_j`. That is the qualitative analogue of a
   tensor product / coordinate system: diversity is *systematic*, not random.
2. **Forward then reverse** — Noise induction *explores*; critics *project back*
   toward intent + implementability. One-shot ideation skips the reverse process.
3. **Signature-locked critics** — Critic *(i,j)* shares the exact signature of
   FeatureAgent *(i,j)*. Denoising cannot “edit away” the basis; it cleans *along*
   that direction (score matching in one local chart of feature space).
4. **Temperature as σ** — GUI/params map: high **Temperature Scale** ≈ more
   qualitative noise; more **Denoising Steps** ≈ longer reverse chain.
5. **Decode is separate** — Synthesis is not “pick the best cell.” It *merges,
   prioritizes, and architectures* the clean matrix into one shippable brief.

### Philosophy (strict — do not dilute)

- **Language is the computational medium** (not numbers, not embeddings you
  manipulate by hand).
- **Nouns and verbs act as orthogonal basis directions.**
- **High temperature = controlled qualitative noise.**
- **Critic agents = qualitative reverse diffusion / score matching.**
- The whole process turns a vague aesthetic prompt into a concrete, buildable
  app specification **the same way Midjourney turns a vague prompt into an image.**

### When to invoke (and when not)

**Invoke when:**

- User has a **vibe / Midjourney-style** app idea (“cozy night writing app…”)
- You need a **full app build brief**, not a single function
- Product surface is under-specified (features, UX, NFRs all fuzzy)
- User says `/qdad`, “diffuse this”, “slot machine”, “turn this into a build prompt”

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
- `/qdad N=3 steps=2 — minimal habit tracker that feels like a garden`
- `/qdad expand this into a full build prompt: marketplace for local makers`
- `/qdad` (uses the last vague product idea in the conversation)

### Parameters (optional; parse from user text or defaults)

| Param | Default | Range | Role |
|-------|--------:|-------|------|
| **N** (grid size) | 4 | 2–8 | N×N feature agents |
| **Temperature Scale** (noise T) | 1.3 | 0.7–1.8 | Forward diffusion only |
| **Denoising Steps** | 3 | 1–6 | Reverse diffusion rounds |
| **Noun/Verb Temperature** | 0.6 | 0.3–1.0 | Foundation basis generation |

Budget: agents per noise/denoise round = **N²**. Total heavy LLM calls ≈  
`1 (foundation) + N² (noise) + Steps×N² (critics) + 1 (synth)`.  
Prefer **N=3, steps=2** when cost-sensitive; **N=4–5, steps=3** for rich apps.

Announce params before running:

```
QDAD: N×N grid, noise_T=…, steps=…, noun_verb_T=…
Philosophy: language=medium; nouns×verbs=basis; high-T=noise; critics=reverse diffusion
```

---

## Step 0 — Capture the Intent Brief

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

## Step 1 — Phase 0: Foundation (qualitative basis)

Generate **exactly N distinct nouns** and **exactly N distinct verbs**.

### Basis rules

- **Nouns** = object / substance / place / affordance axes (**rows**)
- **Verbs** = action / process / transformation axes (**columns**)
- Concrete enough to ground features; mutually distinct; span the *aesthetic
  space* of the intent (not just synonyms of “app” / “user”)
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

**Log:** `nouns = […]`, `verbs = […]`.

Hard fail: fewer than N unique items, or all generic (“data”, “manage”, “system”).

---

## Step 2 — Phase 1: Agent grid construction

For each `i in 0..N-1`, `j in 0..N-1`:

```
FeatureAgent_{i}_{j}
  noun = nouns[i]
  verb = verbs[j]
  signature = noun × verb
```

Permanent assignment. No reassignment later. Log a compact grid:

```
        verb0    verb1    …
noun0   A00      A01
noun1   A10      A11
…
```

---

## Step 3 — Phase 2: Noise induction (forward diffusion)

**In parallel** (or sequential if no sub-agents), for every cell `(i,j)`:

### FeatureAgent system prompt

```
You are FeatureAgent_{i}_{j}.
Your unique qualitative signature is noun "{noun}" × verb "{verb}".
Your sole purpose is to invent exactly ONE concrete, implementable feature
for an application. The feature must feel like a natural expression of the
interaction between "{noun}" and "{verb}" given the user intent.

User intent:
---
{user_prompt}
---

FORWARD DIFFUSION: Invent one wild, imperfect, slightly hallucinated but still
related feature. Embrace controlled qualitative noise. Rough edges and odd
metaphors are allowed — they are the language analogue of Gaussian noise.
Stay in orbit of the intent.

Output ONLY the feature (2–6 sentences). No JSON. No headings.
```

Use **noise temperature** (Temperature Scale) for these calls.

Collect `noisy_features[i][j]`.

**Do not** polish yet. Noise is a feature of the algorithm.

---

## Step 4 — Phase 3: Iterative qualitative denoising

For `step = 1 .. Denoising_Steps`:

**In parallel**, for every cell `(i,j)`, spawn **CriticAgent_{i}_{j}** with the
**exact same** noun+verb signature:

```
You are CriticAgent_{i}_{j}.
You share signature noun "{noun}" + verb "{verb}".
You are the inverse of noise induction: qualitative reverse diffusion / score matching.
Clean imperfections, remove contradictions, sharpen original intent, make the
feature coherent, useful, and implementable — while remaining a true expression
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

Output ONLY the refined feature (2–6 sentences).
```

Use a **cooler** temperature than noise (≈ `0.5 × noise_T`, clamped to ~0.3–1.0).

Replace `features[i][j]` with the critic output after each full parallel round.

Optional: keep snapshots `step_1`, `step_2`, … for transparency.

After the final step: `clean_features = features`.

---

## Step 5 — Phase 4: Synthesis (decode)

One **Synthesizer** agent receives:

- Original user prompt
- Nouns, verbs
- Full clean N×N matrix (each cell labeled with noun×verb)

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

- **Deduplicate and prioritize** — merge related cells; do not dump N² features.
- Features must feel like **coherent expressions of intent**, not a laundry list.
- Be **concrete and implementable**.
- Prefer modern clean stacks; match constraints from the brief.
- Optionally append a **## Diffusion Feature Matrix (transparency)** section
  with nouns, verbs, and each clean cell for auditability.

---

## Step 6 — Handoff to the coding loop

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
| **Parallel sub-agents** | Spawn N² FeatureAgents / CriticAgents per phase; gather results |
| **Single agent only** | Simulate the grid sequentially; still label every cell `(i,j)` and preserve phases |
| **Temperature** | Set per phase if API allows; else prompt for “wilder” vs “stricter” |
| **Model-agnostic** | Any capable chat model works; stronger models → better basis + synth |
| **Read-only** | Prefer no workspace mutation until Step 6 handoff |
| **Cost control** | Default N=3–4, steps=2–3; never N=8×steps=6 without explicit ask |

### Parallelization contract

```
foundation: 1 call
noise:      N² calls in parallel
for step in 1..Steps:
    denoise: N² calls in parallel
synthesize: 1 call
```

### Sub-agent prompt packaging

When spawning, always include: cell `(i,j)`, noun, verb, user prompt, phase
instructions, and (for critics) current feature + step index.

---

## Anti-patterns (do not do these)

- Flat “list 10 features” without a noun×verb grid
- Skipping noise (going straight to “good” features) — kills exploration
- Skipping critics (shipping raw noise) — kills implementability
- Critics that **change** the noun/verb signature
- Synthesizer that pastes all N² cells without prioritization
- Implementing a full app **inside** the diffusion steps
- Using only generic basis words (data, user, manage, system)
- Running N=8 by default

---

## Quick reference — algorithm

```
Intent Brief + params (N, noise_T, steps, nv_T)
    → Phase 0 Foundation: N nouns, N verbs   [nv_T]
    → Phase 1 Grid: FeatureAgent_i_j := (nouns[i], verbs[j])
    → Phase 2 Noise: ∀(i,j) parallel feature @ noise_T
    → Phase 3 Denoise: for step in 1..steps:
          ∀(i,j) parallel CriticAgent_i_j (same signature) @ cooler T
    → Phase 4 Synthesize: clean matrix + prompt → App Build Prompt
    → Handoff: user approves → normal build loop
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

Intent: *“cozy night writing app, soft dark mode, offline-first”*

```
nouns: [lantern, notebook]
verbs: [whisper, weave]

noisy[0][0] lantern×whisper → wild ambient voice notes idea
noisy[0][1] lantern×weave   → wild link-glow between drafts
…
critic cleans each cell toward offline-first + calm UX
synth → App Build Prompt with prioritized features + architecture
```

End of skill.
