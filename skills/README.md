# Portable Skills

Agent-agnostic skill packages you can drop into any agentic coding harness
(Grok Build, Claude Code, Cursor, Codex, custom runners, etc.).

These skills encode **procedures**, not host-specific APIs. Tool names in the
body are examples — adapt them to your runtime.

## Skills in this release

| Skill | Path | When to use |
|-------|------|-------------|
| **qnn** | [`qnn/SKILL.md`](./qnn/SKILL.md) | Sticky debug loops **or** features that need wider depth — layered strategy maps from problem-space **verbs + nouns** |
| **qdad** | [`qdad/SKILL.md`](./qdad/SKILL.md) | Vague Midjourney-style **app vibe** → concrete **agentic coding prompt** via **Qualitative Diffusion** (noun×verb grid, noise, critic reverse diffusion, synthesize) |

| Technique | Skill | Output |
|-----------|-------|--------|
| Qualitative Neural Network | `/qnn` | Solution-space / strategy map (then implement) |
| Qualitative Diffusion | `/qdad` | App Build Prompt (then implement) |

## Quick import

### Option A — Copy into the host skills directory

**Grok Build / Grok agent skills** (user scope, all projects):

```bash
# Linux / macOS — both skills
mkdir -p ~/.grok/skills/qnn ~/.grok/skills/qdad
cp skills/qnn/SKILL.md ~/.grok/skills/qnn/SKILL.md
cp skills/qdad/SKILL.md ~/.grok/skills/qdad/SKILL.md

# Windows (PowerShell)
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.grok\skills\qnn"
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.grok\skills\qdad"
Copy-Item skills\qnn\SKILL.md "$env:USERPROFILE\.grok\skills\qnn\SKILL.md"
Copy-Item skills\qdad\SKILL.md "$env:USERPROFILE\.grok\skills\qdad\SKILL.md"
```

**Project scope** (share with a single repo):

```bash
mkdir -p .grok/skills/qnn .grok/skills/qdad
cp skills/qnn/SKILL.md .grok/skills/qnn/SKILL.md
cp skills/qdad/SKILL.md .grok/skills/qdad/SKILL.md
```

### Option B — Install from a GitHub Release asset

Download `qnn-skill-*.zip` and/or `qdad-skill-*.zip` from
[Releases](https://github.com/iblameandrew/open-deepthink/releases), unzip into
your host’s skills root (e.g. `~/.grok/skills/`).

```bash
gh release download --repo iblameandrew/open-deepthink --pattern "*-skill-*.zip"
unzip qnn-skill-*.zip -d ~/.grok/skills
unzip qdad-skill-*.zip -d ~/.grok/skills
```

### Option C — Point the agent at the file

```
Read and follow: path/to/open-deepthink/skills/qdad/SKILL.md
```

Invoke:

```
/qdad a cozy night writing app, soft dark mode, offline-first
/qnn explore this deadlock
```

## Verify

After install, skills should appear as `/qnn` and `/qdad` (or via natural
language triggers in each skill’s frontmatter).

## Packaging layout

```
skills/
  README.md
  package_skill.py
  qnn/
    SKILL.md
    INSTALL.md
  qdad/
    SKILL.md
    INSTALL.md
```

Release assets:

```
qnn-skill-<version>.zip   → qnn/SKILL.md, qnn/INSTALL.md
qdad-skill-<version>.zip  → qdad/SKILL.md, qdad/INSTALL.md
```

Rebuild:

```bash
python skills/package_skill.py
python skills/package_skill.py --skill qdad
python skills/package_skill.py --version 0.1.9
```

## Philosophy

- **`/qdad`** = design an app from a *vibe* (Qualitative Diffusion → build prompt).
- **`/qnn`** = map *strategies* when stuck or when a feature needs depth.
- **Coding agent** = grounded implementation after you pick a direction / brief.
- Skills stay portable: no hard dependency on the open-deepthink server UI.
  The full engine remains in this repo for long runs with logs and matrix persistence.
