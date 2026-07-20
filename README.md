<img width="1248" height="832" alt="open-deepthink QNN visualization" src="https://github.com/user-attachments/assets/ffd223ee-a875-4213-a65f-23c2f7a7807c" />

# open-deepthink: Evolvable Agent Networks for Deep, Structured Reasoning

**Not another flat panel of 16 agents brainstorming once.**  
A **Qualitative Neural Network (QNN)** that runs layered forward passes, reflects on its own performance, mutates its agents' cognitive identities, raises the difficulty of the problem, and records the entire developmental history as high-signal training data.

Most agentic systems give you breadth through parallelism. open-deepthink gives you **depth through structured iteration and self-modification**.

---

## 🎰 New in 0.1.8 — Qualitative Diffusion (App Slot Machine Mode)

**Diffusion, re-implemented at a purely qualitative scale.**

Algorithm Design Mode is gone. In its place: **Qualitative Diffusion App Designer (QDAD)** — a new technique that treats *language* as the computational medium the way Midjourney treats pixels as the medium for images.

| Classical diffusion | Qualitative Diffusion (QDAD) |
|---------------------|------------------------------|
| Continuous noise in latent space | **Controlled qualitative noise** at high temperature |
| Denoising network | **Critic agents** (reverse diffusion / score matching in language) |
| Pixel / embedding basis | **Nouns × verbs** as orthogonal basis directions |
| Image from a vague prompt | **Buildable agentic coding prompt** from a vague aesthetic app prompt |

**How it works**

1. **Foundation** — From your Midjourney-style intent, sample **N nouns** and **N verbs** (shared qualitative basis).
2. **Grid** — Build an **N×N** grid of FeatureAgents. Cell *(i, j)* is permanently bound to `nouns[i] × verbs[j]`.
3. **Forward diffusion** — In parallel, each agent invents one wild, imperfect feature at **noise temperature**.
4. **Reverse diffusion** — For each denoising step, CriticAgents with the *same* noun+verb signature clean, sharpen, and implementabilize the matrix.
5. **Decode** — A Synthesizer collapses the clean matrix into a structured **# App Build Prompt** you can hand to Grok-Build, Cursor, Claude Artifacts, etc.

**Philosophy (strict)**

- Language is the computational medium.
- Nouns and verbs act as orthogonal basis directions.
- High temperature = controlled qualitative noise.
- Critic agents = qualitative reverse diffusion / score matching.
- The whole process turns a vague aesthetic prompt into a concrete, buildable app specification — the same way Midjourney turns a vague prompt into an image.

**Implementation:** LangGraph pipeline in `deepthink/qdad/` (`foundation → grid → noise → denoise⟲ → synthesize`), model-agnostic LLM calls, full feature-matrix transparency on every run. Frontend matches Brainstorming Mode; only the backend algorithm is different.

---

## The Core Problem with "Just Add More Agents"

Typical multi-agent setups (including many "16 expert" or "army of agents" brainstorming interfaces) work like this:

- Spawn N agents with static or lightly templated personas.
- Run them in parallel or loose conversation.
- Synthesize once (or a few turns).
- Done.

You get diversity of perspective, but the agents themselves do not become meaningfully better at the *specific* problem over time. There is no topology, no persistent specialization, no mechanism that rewires *how* the system thinks, and almost never a reusable artifact of the reasoning process.

open-deepthink treats agents like **neurons in a network** whose "weights" are rich natural-language personas, and whose learning rule is **Mirror Descent** (qualitative backpropagation).

---

## What Makes a QNN Different

A QNN is a directed, layered graph of LLM agents with three repeating phases per epoch:

1. **Forward Pass** — Problem is decomposed across the topology. Layer 0 runs in parallel. Each subsequent layer receives context from the previous layer and builds deeper analysis. Information flows structurally, not just through a shared chat.

2. **Reflection + Mirror Descent** — After synthesis, the system does not just "critique the answer." It:
   - Evaluates which agents struggled vs. succeeded on their specific sub-problems.
   - Extracts attributes and "hard requests" from current personas.
   - Uses a dense-spanner mechanism (or explicit mixing in Distillation) to **rewrite the system prompts, attributes, and skills** of agents for the next round.
   - In Knowledge Distillation mode, literally **spawns evolved child agents** that inherit context memory and replace struggling parents in the live topology.

3. **Problem Reframing** — A dedicated re-framer node looks at the current solution and formulates a *harder, more advanced version* of the problem. The network is then forced to solve the harder problem in the next epoch with its newly evolved agents.

This loop (decompose → structured forward → synthesize → reframe the goal → mutate the thinkers) is repeated for as many epochs as you allocate. The result is compounding depth rather than repeated breadth.

---

## Three Powerful Operating Modes

### 1. ⚗️ Knowledge Distillation Mode (The Data Engine)

The most distinctive and high-leverage mode.

- Fixed powerful topology: **1×2×2×2×2×2×1** (7 layers, 12 agents).
- 12 distinct cognitive archetypes (The Initiator, Builder, Connector, Preserver, Performer, Analyst, Diplomat, Transformer, Explorer, Architect, Visionary, Dreamer) with hand-crafted system prompts, attributes, and skills.
- **Task Master** decomposes the anchor question into 12 Socratically-linked sub-questions.
- Full forward pass with layer-to-layer context.
- **Mirror Descent** evaluates every agent-question pair. Hard agents trigger **live evolutionary replacement**: a Mixing Agent combines the struggling agent with the best resonant helper from the *current* grid. The child inherits the parent's 100k-token context memory and keeps the difficult question.
- **Seed Creator** evolves the topic set itself each epoch, generating ontologically adjacent new topics.
- Runs until your token budget is exhausted.

**Primary output**: A structured JSON dataset of every (epoch, agent, archetype, question, answer) pair, plus a complete `topology_archive.json` containing the full evolutionary history (every system prompt mutation, every inheritance, every difficulty judgment).

This is not generic chat logs. This is **developmental trace data** explicitly designed for training the next generation of reasoning models — models that can internalize patterns of collaboration, critique, specialization, and progressive deepening.

### 2. 🧠 Brainstorming Mode (Full QNN Expert Panel)

A chat-first interface that runs the **same QNN deepthink algorithm** as the portable `/qnn` skill — not a flat static expert panel.

**Step-by-step each run:**

0. **Brief** — Impasse/enrich summary from prompt + attachments  
1. **Topology** — Auto (complexity → L×W×E) or Manual/Massive  
2. **Seeds** — Verbs + nouns from the problem space (related + far semantic fields), sampled into per-column word-vectors — same spanning DNA as qualitative verb/noun bases  

3. **Personas** — Input-span careers/attributes/skills from those guiding_words (layer 0 diverge; deeper layers converge/critique)  
4. **Epoch loop** — Layered forward pass → epoch map → Mirror Descent → harder reframe  
5. **Solution-Space Report** — Divergent strategies with falsifiers and first probes (handoff to edit→run→debug)

- **Auto mode**: Complexity estimator recommends a small topology (skill-aligned score bands).
- **Manual / Massive mode**: Any Layers × Width; spawn a genuine "army" when justified.
- Intermediate epochs produce compact **epoch maps**; the final epoch polishes the full report.
- Rich markdown chat interface for the report and logs for each QNN step.

Use this when you want deeper insight than a single model or a flat expert panel can deliver.

### 3. 🎰 App Slot Machine Mode (Qualitative Diffusion App Designer)

Replaces Algorithm Design Mode with **Qualitative Diffusion App Designer (QDAD)** — diffusion re-implemented at a purely qualitative scale.

**Philosophy:** language = computational medium · nouns/verbs = orthogonal basis · high T = controlled noise · critics = reverse diffusion / score matching · vague prompt → buildable app spec (Midjourney for apps).

| Layer | Path |
|-------|------|
| Package | `deepthink/qdad/` (`state`, `nodes`, `graph`, `pipeline`, `utils`) |
| Chains | `deepthink/chains/qdad_chains.py` |
| GUI | Same chat shell as Brainstorming + N / Temperature Scale / Denoising Steps / Noun-Verb T |
| Graph | LangGraph: `foundation → grid → noise → denoise⟲ → synthesize` |

- **Phase 0** — N nouns + N verbs (shared qualitative basis, noun/verb temperature)
- **Phase 1** — N×N FeatureAgents permanently bound to `nouns[i] × verbs[j]`
- **Phase 2** — Parallel forward diffusion (noise temperature)
- **Phase 3** — Iterative reverse diffusion via CriticAgents (same signature)
- **Phase 4** — Synthesizer → structured **agentic coding prompt** + transparent matrix

Feed a Midjourney-style app prompt; hand the result to Grok-Build, Cursor, Claude Artifacts, etc.

---

## The Outputs That Actually Matter

A single deep open-deepthink run produces far more than an answer:

- **Evolved QNN artifacts** — Portable, versionable "trained" multi-agent systems you can share and reuse.
- **Full evolutionary traces** — Every prompt before/after Mirror Descent, every difficulty classification, every child/parent relationship, every reframed problem.
- **Structured distillation datasets** — Purpose-built for fine-tuning or synthetic data pipelines targeting advanced reasoning and multi-agent behavior.
- **Interpretable intermediate state** — Because everything is explicit natural-language personas and traceable sub-problems, you can diagnose *why* the system thought what it thought at any layer and epoch.
- **Accumulated executable knowledge** (in code modes) — Real modules that survived sandbox validation and were re-used.

These artifacts are the real product. The final synthesized answer is a byproduct.

---

## Why This Matters for Agentic Coding and Reasoning Research

- **Test-time compute, done right and observably.** Many frontier systems hide their long reasoning inside a single model. open-deepthink makes the structure, specialization, and adaptation explicit and archivable.
- **A credible path to better base models.** The highest-leverage use of current powerful models may be generating traces of *how* hard problems should be decomposed, attacked by specialized perspectives, critiqued, and progressively deepened. open-deepthink is purpose-built to produce that class of data at scale on consumer hardware.
- **Reusable specialized reasoners.** An exported QNN that has spent 10–20 epochs evolving on a domain is qualitatively different from prompting a base model with a long system prompt. The specialization is *baked into the network structure and the evolved personas*.
- **Local and long-horizon by design.** Runs for hours or days on a 32 GB laptop or a modest rig. No requirement for frontier API spend to get compounding returns.

---

## Technical Strengths

- Built on **LangGraph** with real cyclic graphs, conditional routing (epoch gateway), parallel layer execution, and proper state management — not a pile of sequential chains.
- 195 passing tests across 11 phases, including regression tests for every bug fixed in the 0.0.3 quality release.
- Clean provider model: only OpenRouter (cloud) and llama.cpp server (local). Per-agent and per-synthesis model selection supported.
- Robust JSON handling, token tracking, streaming logs, RAPTOR hierarchical indexing, and safe(ish) code execution.
- Real export/import of full QNN state.
- Massive scale supported when you ask for it (Manual/Massive mode will happily run 50×50+ if your budget allows).

---

## Quick Start

```bash
git clone https://github.com/iblameandrew/open-deepthink
cd open-deepthink
python -m venv venv
.\venv\Scripts\activate          # Windows
# source venv/bin/activate       # macOS/Linux
pip install -r requirements.txt
launch.bat                       # or: python app.py
```

Open http://127.0.0.1:8000.

**Supported providers**: OpenRouter (bring your own key) and local llama.cpp server.

See the in-app UI for the three modes, topology visualization, token budgeting (especially important for Distillation), and export controls.

---

## Hyperparameters & Hardware Reality

- **Layers / Width**: Control depth vs. breadth of the network. 3–6 layers with 3–8 width is already deep on most problems. Massive mode exists for when you want to go further.
- **Epochs**: The number of full forward + reflection + reframing cycles. This is where the power law lives.
- **Learning rate / Density / Prompt alignment**: Control how aggressively agents mutate and how strongly the original problem shapes their identities.
- **Token budget** (Distillation): The real governor. Set it high if you want serious evolutionary runs.

**Practical guidance**:
- 32 GB RAM CPU laptop: 2×2 to 4×4 topologies, 2–4 epochs.
- 64 GB + decent GPU: 6×6 to 10×10, more epochs, or serious Distillation runs.
- The system is explicitly designed so that **more time and more epochs beats needing a bigger base model**.

---

## Vision

Every serious open-deepthink run is a small laboratory experiment in collective intelligence. The structured traces it produces — complete with evolutionary dynamics, difficulty signals, and topology mutations — are some of the richest open data currently being generated about *how* LLMs can be orchestrated to think harder.

The long-term bet is that collecting thousands of such runs will let us train models that no longer need elaborate hand-written system prompts or external scaffolding, because they have internalized the patterns of decomposition, specialization, critique, and progressive deepening directly.

This is why the distillation dataset + full topology archives are treated as first-class outputs.

---

## Portable `/qnn` Skill for Agentic Coders

When a coding agent is **stuck** (deadlock, race, perf cliff, circular local fixes) or a **feature needs wider depth** (richer metrics, APIs, UX options), drop in the portable QNN skill instead of grinding more one-shot prompts.

```
/qnn explore this deadlock / performance regression
/qnn richer metrics for the training dashboard
```

The skill runs a structured Qualitative Neural Network procedure inside the host agent: guiding concepts → layered personas → multi-epoch forward passes → Mirror Descent persona evolution → harder reframes → a **Solution-Space Report**. The goal is not to ship the patch immediately — it is a map of divergent strategies with falsifiers and smallest first probes. You pick a direction and resume the grounded edit → run → debug loop.

| Artifact | Location |
|----------|----------|
| Skill body | [`skills/qnn/SKILL.md`](./skills/qnn/SKILL.md) |
| Install notes | [`skills/qnn/INSTALL.md`](./skills/qnn/INSTALL.md) |
| Skills index | [`skills/README.md`](./skills/README.md) |
| Release zip | `qnn-skill-<version>.zip` on [GitHub Releases](https://github.com/iblameandrew/open-deepthink/releases) |

**Install (Grok user skills):**

```bash
mkdir -p ~/.grok/skills/qnn
cp skills/qnn/SKILL.md ~/.grok/skills/qnn/SKILL.md
```

Or download the release asset and unzip into `~/.grok/skills/`. Works with any agent that can follow a skill file — tool names adapt to the host.

The full open-deepthink server remains the place for long evolutionary runs, export/import of trained QNNs, and distillation datasets. The portable skill is the lightweight escape hatch for day-to-day agentic coding.

---

## Contributing & Benchmarking

This is research software with a stable core (195/195 tests, all core loops functional). The most valuable contributions right now are:

- Deep, long runs on interesting problems (especially with local models) and sharing the exported QNNs + distillation datasets.
- Bug reports that include the graph trace / logs.
- Ideas for tightening the code execution loop, adding real tool use inside agents, or improving the Mirror Descent signal.
- P2P/distributed ideas for running truly massive topologies across machines.

Open an issue with your traces and thoughts.

---

## License & Credits

Open-source research project. The goal is to push forward what small teams and individuals can do with structured, long-horizon agentic systems.

If open-deepthink helps you go deeper on hard problems or generates useful traces, star the repo and share what you built with the exported QNNs or distillation data.

---

**open-deepthink** — Turn time and structure into depth.  
Not more agents. Better *becoming* agents.

---

*Version 0.1.7 — See [RELEASE_NOTES.md](./RELEASE_NOTES.md) for the full history.*