<img width="1248" height="832" alt="open-deepthink QNN visualization" src="https://github.com/user-attachments/assets/ffd223ee-a875-4213-a65f-23c2f7a7807c" />

# open-deepthink: Evolvable Agent Networks for Deep, Structured Reasoning

**Not another flat panel of 16 agents brainstorming once.**  
A **Qualitative Neural Network (QNN)** that runs layered forward passes, reflects on its own performance, mutates its agents' cognitive identities, raises the difficulty of the problem, and records the entire developmental history as high-signal training data.

Most agentic systems give you breadth through parallelism. open-deepthink gives you **depth through structured iteration and self-modification**.

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

### 2. 🧠 Brainstorming Mode (Deep Conceptual Exploration)

A chat-first interface over the full QNN engine.

- **Auto mode**: Complexity estimator recommends a small, efficient panel (typically 2–5 agents, 1–3 epochs).
- **Manual / Massive mode**: You directly specify Layers × Width (e.g. 8×12, 20×20, or larger). No artificial caps. Spawn a genuine "army" when the problem justifies it.
- Agents are dynamically generated expert personas (not generic "helpful assistant" variants).
- Full multi-epoch Mirror Descent evolution of the expert panel.
- Final synthesis represents the *evolved collective intelligence*, not a single round of debate.
- Rich markdown chat interface for inspecting intermediate reflections.

Use this when you want deeper insight than a single model or a flat expert panel can deliver, without manually designing a full algorithm topology.

### 3. 🧬 Algorithm Design Mode (Maximum Control + Code Generation)

The original deep QNN mode for hard algorithmic and software problems.

- Design arbitrary layer × width topologies.
- Full hyperparameter control (prompt alignment, density, learning rate, vector word size, etc.).
- Automatic problem decomposition into per-agent sub-problems.
- Code-aware synthesis + real (restricted) Python sandbox execution.
- Successful modules are documented as "module cards" and fed forward as context for future epochs.
- Complete RAG (RAPTOR) indexing of every agent output across every epoch.
- Post-run interactive chat directly into the network's memory ("what did agent_2_1 think about X?").
- Final harvest produces structured research-style reports.
- **Export / Import trained QNNs**: Save the entire evolved network (all layer prompts + state) as a compact JSON. Load it later and run new problems against the *already-evolved* specialists.

This is the mode for when you want to treat the QNN as a trainable, reusable reasoning artifact.

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

## Future Directions: QNNs in Practical Coding Agents

As a programmer, one of the most exciting follow-up projects I can imagine is taking the core ideas from open-deepthink and embedding them into a real, tool-using coding agent.

Here's the kind of workflow I'm thinking about:

You're deep in a codebase, "vibe coding" some rough, high-level instructions because you don't have perfect nuance on the solution yet. You hit a particularly sticky bug or architectural problem — the kind where normal agent assistance (even strong models) keeps circling around variations of the same local approach.

Instead of grinding it out with more prompting, you type something like:

```
/qnn explore this deadlock / performance regression
```

The system spans a large structured network — for example a 10×10 QNN with 100 agents — and runs multiple epochs of thinking. These agents don't just brainstorm in a flat chat. They perform layered forward passes, decompose the problem across the topology, explore genuinely different strategies in parallel and in depth, evolve their own specializations through Mirror Descent, and deliberately reframe the problem to help escape the current mental model.

The goal isn't for the QNN to immediately write the fix. The output is a rich map of divergent approaches, with reasoning about why certain paths might break the current impasse. You review the explored solution space, pick the directions that feel promising (or that surface angles you never would have considered), and feed the best ones back into the tight, grounded edit-run-debug loop of your normal coding agent.

This hybrid model — using large-scale evolutionary QNN exploration as a "strategic depth tool" precisely when you're stuck and lacking nuance, while keeping a fast, tool-heavy agent for actual implementation and verification — feels like a natural and powerful evolution of the ideas here.

I plan to explore building something like this in a dedicated coding agent project in the future.

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

*Version 0.1.5 — See [RELEASE_NOTES.md](./RELEASE_NOTES.md) for the full history.*