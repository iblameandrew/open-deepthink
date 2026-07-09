# Portable Skills

Agent-agnostic skill packages you can drop into any agentic coding harness
(Grok, Claude Code, Cursor, Codex, custom runners, etc.).

These skills encode **procedures**, not host-specific APIs. Tool names in the
body are examples — adapt them to your runtime.

## Skills in this release

| Skill | Path | When to use |
|-------|------|-------------|
| **qnn** | [`qnn/SKILL.md`](./qnn/SKILL.md) | Sticky debug loops **or** features that need wider depth — personas spanned from problem-space **verbs + nouns** (Algorithm Mode method) |

## Quick import

### Option A — Copy into the host skills directory

**Grok Build / Grok agent skills** (user scope, all projects):

```bash
# Linux / macOS
mkdir -p ~/.grok/skills/qnn
cp skills/qnn/SKILL.md ~/.grok/skills/qnn/SKILL.md

# Windows (PowerShell)
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.grok\skills\qnn"
Copy-Item skills\qnn\SKILL.md "$env:USERPROFILE\.grok\skills\qnn\SKILL.md"
```

**Project scope** (share with a single repo):

```bash
mkdir -p .grok/skills/qnn
cp skills/qnn/SKILL.md .grok/skills/qnn/SKILL.md
```

### Option B — Install from a GitHub Release asset

Download `qnn-skill-<version>.zip` from the
[Releases](https://github.com/iblameandrew/open-deepthink/releases) page, unzip,
and place the `qnn/` folder under your host’s skills root (e.g. `~/.grok/skills/`).

```bash
# Example: latest release asset via gh
gh release download --repo iblameandrew/open-deepthink --pattern "qnn-skill-*.zip"
unzip qnn-skill-*.zip -d ~/.grok/skills
```

### Option C — Point the agent at the file

Some harnesses accept an explicit skill path or system attachment:

```
Read and follow: path/to/open-deepthink/skills/qnn/SKILL.md
```

Or paste the skill when stuck:

```
/qnn explore this deadlock
```

## Verify

After install, the skill should appear as `/qnn` (or via natural language
triggers listed in the skill frontmatter). Invoke once on a dry-run problem to
confirm the host loads frontmatter + body.

## Packaging layout

```
skills/
  README.md                 # this file
  qnn/
    SKILL.md                # skill body (YAML frontmatter + procedure)
    INSTALL.md              # one-page install for the qnn skill alone
```

A release asset is built as:

```
qnn-skill-<version>.zip
  qnn/SKILL.md
  qnn/INSTALL.md
```

Rebuild with:

```bash
python skills/package_skill.py
# or: python skills/package_skill.py --version 0.1.6
```

## Philosophy

- **QNN skill** = strategic depth when the local edit–run–debug (or design)
  loop is thin or circling.
- **Coding agent** = grounded implementation and verification after you pick a
  direction from the solution-space map.
- Skills stay portable: no hard dependency on the open-deepthink server UI.
  The full engine remains available in this repo for long evolutionary runs.
