# Install the `/qdad` skill

Portable **Qualitative Diffusion App Designer (QDAD)** for agentic coders
(Grok-Build, Claude Code, Cursor, Codex, custom harnesses).

Turns a vague Midjourney-style app prompt into a structured **agentic coding
prompt** via noun×verb basis grids, high-T noise, critic reverse diffusion,
and synthesis — without needing the open-deepthink server.

## 1. Place `SKILL.md` where your agent loads skills

### Grok Build / Grok agent (user-global)

```bash
mkdir -p ~/.grok/skills/qdad
cp SKILL.md ~/.grok/skills/qdad/SKILL.md
```

Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.grok\skills\qdad"
Copy-Item SKILL.md "$env:USERPROFILE\.grok\skills\qdad\SKILL.md"
```

### Grok (this project only)

```bash
mkdir -p .grok/skills/qdad
cp SKILL.md .grok/skills/qdad/SKILL.md
```

### Other agents (Claude Code, Cursor, Codex, custom)

Copy `SKILL.md` into that product’s skills / prompts directory, or attach the
file at session start. The procedure is host-agnostic.

## 2. Invoke

```
/qdad a cozy productivity app for writers who work at night, soft dark mode, offline-first
/qdad N=3 steps=2 — garden-like habit tracker
/qdad expand into a full build prompt: local makers marketplace
```

Natural language also works if the host matches the skill description
(“diffuse this app”, “slot machine this idea”, “turn this vibe into a build prompt”).

## 3. What you get

1. Logged **noun/verb basis** and N×N grid  
2. Optional intermediate noisy / cleaned features  
3. Primary deliverable: **`# App Build Prompt`** (vision, features, architecture, UX, NFRs, implementation notes)  
4. Handoff: implement with your normal coding loop  

## 4. From GitHub Release

Download `qdad-skill-*.zip` from
[open-deepthink releases](https://github.com/iblameandrew/open-deepthink/releases),
unzip, and copy the `qdad/` folder into your skills root.

```bash
gh release download --repo iblameandrew/open-deepthink --pattern "qdad-skill-*.zip"
unzip qdad-skill-*.zip -d ~/.grok/skills
```

## Source

Canonical tree: `skills/qdad/` in open-deepthink.  
Reference engine: `deepthink/qdad/` (LangGraph).  
Full UI: **App Slot Machine Mode** in the open-deepthink app.
