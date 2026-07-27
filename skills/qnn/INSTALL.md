# Install the `/qnn` skill

Portable Qualitative Neural Network procedure for agentic coders.

Personas are **not** a flat expert list. They are spanned the same way as
open-deepthink Algorithm Mode: seed **verbs + nouns** from the problem space
→ sample word-vectors per column → input-span careers/attributes/skills.

Use it when:

- You are **stuck** (deadlock, race, perf cliff, circular local fixes), or
- A **feature / artifact** needs wider depth (richer metrics, APIs, UX options).

## 1. Place skill files where your agent loads skills

Copy the whole skill folder (not only `SKILL.md`):

| File | Role |
|------|------|
| `SKILL.md` | Procedure + **materialize-then-run** protocol (+ Appendix A) |
| `run_template.py` | Source the agent writes to `.skill-runs/run_qnn.py` |
| `CODE_REFERENCE.md` | Parameter table |
| `INSTALL.md` | This file |

### Grok (user-global)

```bash
mkdir -p ~/.grok/skills/qnn
cp SKILL.md CODE_REFERENCE.md run_template.py INSTALL.md ~/.grok/skills/qnn/
```

Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.grok\skills\qnn"
Copy-Item SKILL.md,CODE_REFERENCE.md,run_template.py,INSTALL.md "$env:USERPROFILE\.grok\skills\qnn\"
```

On invoke, the agent **builds** `.skill-runs/run_qnn.py` from `run_template.py`
and runs it. The script discovers the open-deepthink package by walking the
workspace tree (optional: `OPEN_DEEPTHINK_ROOT`).

## 2. Invoke

```
/qnn explore this deadlock / performance regression
/qnn richer metrics for the export dashboard
/qnn stuck on token refresh race
```

Natural language also works if the host matches the skill description
(“brainstorm ways out”, “solution space”, “unstuck”, “richer features”).

## 3. What you get

A **Solution-Space Report**: divergent strategies with mechanisms, falsifiers,
risks, and smallest first probes — **not** an immediate production patch.

Pick a direction, then resume your normal edit → run → debug (or design →
spike → ship) loop.

## 4. From GitHub Release

Download `qnn-skill-*.zip` from
[open-deepthink releases](https://github.com/iblameandrew/open-deepthink/releases),
unzip, and copy the `qnn/` folder into your skills root.

```bash
gh release download --repo iblameandrew/open-deepthink --pattern "qnn-skill-*.zip"
unzip qnn-skill-*.zip -d ~/.grok/skills
```

## Source

Canonical tree in the open-deepthink repo: `skills/qnn/`.
Full QNN engine (Brainstorming / Algorithm / Distillation UI): repository root.
