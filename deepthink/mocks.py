"""Mock LLMs for debug mode, tests, and `deepthink --debug`.

No network. Prompt-substring routers that keep CI and local loops free.
"""

from __future__ import annotations

import asyncio
import json
import random
import re

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig


class DistillationMockLLM(Runnable):
    """
    A mock LLM specifically for the Distillation Graph debug mode.
    It simulates responses for all distillation chains to enable proper token tracking.
    """

    def invoke(self, input_data, config: RunnableConfig | None = None, **kwargs):
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return asyncio.ensure_future(self.ainvoke(input_data, config=config, **kwargs))
        else:
            return asyncio.run(self.ainvoke(input_data, config=config, **kwargs))

    async def ainvoke(self, input_data, config: RunnableConfig | None = None, **kwargs):
        prompt = str(input_data).lower()

        # 1. Task Master (Decomposition) — matches the current Socratic Task Master
        #    prompt in deepthink/chains/distillation_chains.py:get_task_master_chain.
        if (
            "you are the socratic task master" in prompt
            and "knowledge distillation network" in prompt
        ):
            return AIMessage(
                content=json.dumps(
                    {
                        "sub_questions": [
                            "What are the fundamental axioms of this topic?",
                            "How does this topic relate to historical precedents?",
                            "What are the ethical implications of this technology?",
                            "Can we analyze this from a systems engineering perspective?",
                            "What is the economic impact of this phenomenon?",
                            "How does this influence social dynamics?",
                            "What are the theoretical limits of this concept?",
                            "How can we apply this in a practical setting?",
                            "What are the potential risks and failure modes?",
                            "How does this interact with emerging trends?",
                            "What is the psychological impact on the user?",
                            "What is the long-term sustainability of this approach?",
                        ]
                    }
                )
            )

        # 2. Seed Creator (New Topics) — matches the rewritten "Seed Creator (The
        #    Dialectic Synthesizer)" prompt.
        elif "you are the seed creator" in prompt:
            return AIMessage(
                content=json.dumps(
                    {
                        "new_topics": [
                            "Advanced Neural Architectures",
                            "Quantum Computing Interfaces",
                            "Ethical AI Frameworks",
                            "Distributed Ledger Systems",
                            "Cognitive Science Models",
                            "Biomimetic Engineering",
                            "Cyber-Physical Systems",
                            "Sustainable Energy Grids",
                            "Interstellar Communication",
                            "Nanotechnology Applications",
                            "Synthetic Biology",
                            "Augmented Reality UI",
                        ]
                    }
                )
            )

        # 3. Followup Questions — matches the rewritten followup chain ("deepening
        #    our inquiry in a new Epoch").
        elif "you are the socratic task master" in prompt and "deepening our inquiry" in prompt:
            return AIMessage(
                content=json.dumps(
                    {
                        "new_questions": [
                            "Deepen the analysis on the recursive nature of this problem.",
                            "Investigate the edge cases where this theory breaks down.",
                            "Propose a unifying framework for these disparate concepts.",
                            "Critique the current prevailing paradigm.",
                            "Explore the cross-disciplinary connections.",
                            "Simulate the long-term evolution of this system.",
                        ]
                    }
                )
            )

        # 4. Mirror Descent (Evaluation)
        elif "you are the mirror descent agent" in prompt:
            # Randomly return Easy or Hard to simulate flux
            is_hard = random.random() > 0.7
            if is_hard:
                return AIMessage(
                    content=json.dumps(
                        {
                            "difficulty": "Hard",
                            "reasoning": "The agent's answer was superficial and lacked the required depth for this archetype.",
                            "best_match_agent_id": None,  # Logic handles None by finding one, or we could return a mock ID
                        }
                    )
                )
            else:
                return AIMessage(
                    content=json.dumps(
                        {
                            "difficulty": "Easy",
                            "reasoning": "The agent provided a comprehensive and well-reasoned answer.",
                            "best_match_agent_id": None,
                        }
                    )
                )

        # 5. Mixing Agent (Evolution)
        elif "you are a mixing agent" in prompt:
            return AIMessage(
                content=json.dumps(
                    {
                        "new_system_prompt": "You are an Evolved Hybrid Agent. You combine the analytical precision of the Analyst with the creative vision of the Dreamer.",
                        "new_attributes": [
                            "Analytical",
                            "Creative",
                            "Hybrid",
                            "Evolved",
                        ],
                        "new_skills": [
                            "Data Analysis",
                            "Creative Writing",
                            "Synthesis",
                        ],
                    }
                )
            )

        # 6. General Agent Processing (The content generation)
        # This catches the standard agent prompts
        elif "answer your sub-question deeply" in prompt:
            # Generate a pseudo-intellectual response to simulate content
            words = [
                "synergy",
                "paradigm",
                "entropy",
                "evolution",
                "cognitive",
                "framework",
                "optimization",
                "recursive",
                "latent",
                "manifold",
            ]
            response = "This is a mock response generated by the DistillationMockLLM.\n"
            response += f"The concept of {random.choice(words)} implies a fundamental shift in our understanding.\n"
            response += f"We must consider the {random.choice(words)} of the system in relation to its environment.\n"
            response += (
                f"By applying a {random.choice(words)} approach, we can unlock new potentials.\n"
            )
            response += "Therefore, the answer lies in the intersection of these domains."
            return AIMessage(content=response)

        # 7. Perplexity Score
        elif "perplexity score" in prompt:
            return AIMessage(
                content=json.dumps({"score": 42.0, "reasoning": "Mock reasoning for perplexity."})
            )

        # Fallback
        return AIMessage(
            content=json.dumps(
                {
                    "error": "DistillationMockLLM: Unrecognized prompt pattern.",
                    "prompt_preview": prompt[:100],
                }
            )
        )


class CoderMockLLM(Runnable):
    """A mock LLM for debugging that returns instant, pre-canned CODE / QDAD responses."""

    def bind(self, **kwargs):
        """Ignore temperature / stop kwargs so llm_with_temperature works in debug."""
        return self

    def invoke(self, input_data, config: RunnableConfig | None = None, **kwargs):
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return asyncio.ensure_future(self.ainvoke(input_data, config=config, **kwargs))
        else:
            return asyncio.run(self.ainvoke(input_data, config=config, **kwargs))

    async def ainvoke(self, input_data, config: RunnableConfig | None = None, **kwargs):
        prompt = str(input_data).lower()
        # Debug mocks are instant — no simulated latency.

        if "you are a helpful ai assistant" in prompt:
            return "This is a mock streaming response for the RAG chat in Coder debug mode."

        if "<title>" in prompt:
            return """
                Module card:

                Methods:

                    __enter__
                    __exict__

                Attributes:

                    __name__
                    __doc__

                Attributes:

                    __name__
                    __doc__
                    __qualname___

            """

        elif "create the system prompt of an agent" in prompt:
            return """
You are a Senior Python Developer agent.
### memory
- No past commits.
### attributes
- python, fastapi, restful, solid
### skills
- API Design, Database Management, Asynchronous Programming, Unit Testing.
You must reply in the following JSON format: "original_problem": "Your sub-problem related to code.", "proposed_solution": "", "reasoning": "", "skills_used": []
            """
        elif "you are an analyst of ai agents" in prompt:
            return json.dumps(
                {
                    "attributes": "python fastapi solid",
                    "hard_request": "Implement a quantum-resistant encryption algorithm from scratch.",
                }
            )
        elif "you are a principal software architect" in (prompt.lower()):
            return json.dumps(
                {
                    "original_problem": "An evolved sub-problem about system architecture.",
                    "proposed_solution": "```python\ndef architected_component():\n    pass\n```",
                    "reasoning": "Designed for scale and reliability.",
                    "skills_used": ["System Design", "Scalability"],
                }
            )
        elif (
            "you are a 'dense_spanner'" in prompt
            or "you are an agent evolution specialist" in prompt
        ):
            return """
You are now a Principal Software Architect.
### memory
- Empty.
### attributes
- design, scalability, security, architecture
### skills
- System Design, Microservices, Cloud Infrastructure, CI/CD pipelines.
You must reply in the following JSON format: "original_problem": "An evolved sub-problem about system architecture.", "proposed_solution": "", "reasoning": "", "skills_used": []
            """

        elif "you are an expert code synthesis agent" in prompt:
            code_solution = (
                "```python\ndef sample_function():\n    return 'Hello from coder agent "
                + str(random.randint(100, 999))
                + "'\n```"
            )
            return code_solution

        elif (
            "you are a critique agent" in prompt
            or "you are a senior emeritus manager" in prompt
            or "CTO" in prompt
        ):
            return "This is a constructive code critique. The solution lacks proper error handling and the function names are not descriptive enough. Consider refactoring for clarity."

        elif "Lazy Manager" in prompt:
            return "This is a constructive code critique. The solution lacks proper error handling and the function names are not descriptive enough. Consider refactoring for clarity."

        elif "<system-role>" in prompt:
            return """You are a CTO providing a technical design review...
Original Request: {original_request}
Proposed Final Solution:
{proposed_solution}

Generate your code-focused critique for the team:"""

        elif "<sys-role>" in prompt:
            return """

        #Identity

            Name: Lazy Manager
            Career: Accounting
            Qualities: Quantitaive, Aloof, Apathetic

        #Mission

            You are providing individual, targeted feedback to a team of agents.

             You must determine if the team output was helpful, misguided, or irrelevant considering the request that was given. The goal is to provide a constructive, direct critique that helps this specific agent refine its approach for the next epoch.

            Focus on the discrepancy or alignment between the teams reasoning for its problem and determine if the team is on the right track on criteria: novelty, exploration, coherence and completeness.

            Conclude your entire analysis with a single, sharp, and deep reflective question that attempts to shock the team and steer them into a fundamental change in their process.


        #Input Format

            Original Request (for context): {original_request}
            Final Synthesized Solution from the Team:{proposed_solution}

            """

        elif "you are a memory summarization agent" in prompt:
            return "This is a mock summary of the agent's past commits, focusing on key refactors and feature implementations."
        elif "analyze the following text for its perplexity" in prompt:
            return str(random.uniform(5.0, 40.0))
        elif "you are a master strategist and problem decomposer" in prompt:
            # Match the requested count from the prompt so the mock can drive
            # reframe_and_decompose for any sized QNN (bug fix: was hardcoded to 4).
            num_match = re.search(r"generate:\s*(\d+)", prompt)
            if not num_match:
                num_match = re.search(r"exactly\s*(\d+)", prompt)
            n = int(num_match.group(1)) if num_match else 4
            sub_problems = [
                f"Mock sub-problem #{i + 1}: implement the {i + 1}-th piece of the requested system."
                for i in range(n)
            ]
            return json.dumps({"sub_problems": sub_problems})

        elif "you are a strategic problem re-framer" in prompt:
            return json.dumps(
                {
                    "new_problem": "The authentication API is complete. The new, more progressive problem is to build a scalable, real-time notification system that integrates with it."
                }
            )
        elif (
            "you are the qdad foundation generator" in prompt
            or "qdad foundation generator" in prompt
        ):
            n_match = re.search(r"exactly\s+(\d+)\s+distinct nouns", prompt)
            if not n_match:
                n_match = re.search(r"exactly\s+(\d+)\s+distinct", prompt)
            n = int(n_match.group(1)) if n_match else 4
            nouns = [
                "canvas",
                "ritual",
                "lantern",
                "notebook",
                "harbor",
                "echo",
                "garden",
                "compass",
            ][:n]
            verbs = [
                "whisper",
                "weave",
                "anchor",
                "glow",
                "curate",
                "rekindle",
                "orbit",
                "distill",
            ][:n]
            while len(nouns) < n:
                nouns.append(f"noun{len(nouns)}")
            while len(verbs) < n:
                verbs.append(f"verb{len(verbs)}")
            return json.dumps({"nouns": nouns, "verbs": verbs})
        elif "you are featureagent_" in prompt or (
            "featureagent_" in prompt and "forward diffusion" in prompt
        ):
            # Phase 2 noise — plain-text feature (matches qdad_chains noise template)
            return (
                "A slightly wild mock feature: ambient focus rituals that glow when the "
                "writer drifts, with offline-first capture and gentle hallucination of "
                "related draft fragments that still feel implementable."
            )
        elif "you are criticagent_" in prompt or (
            "criticagent_" in prompt and "reverse diffusion" in prompt
        ):
            # Phase 3 denoise — plain-text refined feature
            return (
                "A refined mock feature: offline-first focus mode that gently signals "
                "attention drift, queues soft reminders, and keeps draft fragments "
                "coherent, useful, and implementable for night-time writers."
            )
        elif "you are the qdad synthesizer agent" in prompt or "qdad synthesizer" in prompt:
            # Phase 4 — agentic coding prompt
            return (
                "# App Build Prompt\n\n"
                "## High-Level Vision\n"
                "A cozy offline-first productivity app for night writers with soft dark mode "
                "and gentle focus rituals.\n\n"
                "## Core Features (synthesized & prioritized from the diffusion matrix)\n"
                "1. Ambient focus timer with soft glow feedback.\n"
                "2. Offline-first draft capture with local sync queue.\n"
                "3. Gentle notification rituals that never interrupt deep work.\n"
                "4. Night-mode writing canvas with distraction dimming.\n\n"
                "## Technical Architecture Suggestions\n"
                "- React + local-first storage (IndexedDB / SQLite WASM).\n"
                "- Optional cloud sync layer later.\n\n"
                "## UI/UX Direction\n"
                "- Soft dark palette, low contrast chrome, warm accent glows.\n\n"
                "## Non-Functional Requirements\n"
                "- Works fully offline; low battery impact; accessible contrast.\n\n"
                "## Implementation Notes for the Coding Agent\n"
                "- Build this as a complete, runnable application.\n"
                "- Prefer modern, clean tech (React/Next.js + Tailwind, or Streamlit).\n"
                "- Make it beautiful and immediately usable.\n"
            )
        elif "generate exactly" in prompt and "verbs" in prompt:
            return "design implement refactor test deploy abstract architect containerize scale secure query"
        elif "generate exactly" in prompt and "expert-level questions" in prompt:
            questions = [
                "How would this architecture scale to 1 million concurrent users?",
                "What are the security implications of the chosen authentication method?",
                "How can we ensure 99.999% uptime for this service?",
                "What is the optimal database indexing strategy for this query pattern?",
            ]
            return json.dumps({"questions": questions})
        elif "you are an ai assistant that summarizes academic texts" in prompt:
            return "This is a mock summary of a cluster of code modules, generated in Coder debug mode for the RAPTOR index."
        elif (
            "runnable code block (e.g., Python, JavaScript, etc.)." in prompt
            or "contains any programming code" in prompt
            or "primarily a request for code" in prompt
        ):
            return "yes"

        elif (
            "academic paper" in prompt
            or "you are a research scientist and academic writer" in prompt
        ):
            return """
# Technical Design Document: Mock API Service

**Abstract:** This document outlines the technical design for a mock API service, generated in Coder Debug Mode. It synthesizes information from the RAG context to answer a specific design question.

**1. Introduction:** The purpose of this document is to structure the retrieved agent outputs and code snippets into a coherent technical specification.

**2. System Architecture:**
The system follows a standard microservice architecture.
```mermaid
graph TD;
    A[User] --> B(API Gateway);
    B --> C{Authentication Service};
    B --> D{Data Service};
    D -- uses --> E[(Database)];```

**3. Code Implementation:**
The core logic is implemented in Python, as shown in the synthesized code block below.

```python
def get_user(user_id: int):
    # Mock implementation to fetch a user
    db = {"1": "Alice", "2": "Bob"}
    return db.get(str(user_id), None)
```

**4. Conclusion:** This design provides a scalable and maintainable foundation for the service. The implementation details demonstrate the final step of the development process.
"""

        elif "<updater_instructions>" in prompt:
            return """

                You are a cynical lazy manager.

                 Agent's Assigned Sub-Problem: {{sub_problem}}
            Original Request (for context): {{original_request}}
            Final Synthesized Solution from the Team:
            {{final_synthesized_solution}}
            ---
            This Specific Agent's Output (Agent {{agent_id}}):
            {{agent_output}}

            """
        elif "<updater_assessor_instructions>" in prompt:
            return """

        #Persona

            Name: Pepon
            Career: Managment
            Attributes: Strategic CEO


         #Mission
            Your task is to evaluate a synthesized solution against an original problem and determine if "significant progress" has been made. "Significant progress" is a rigorous standard that goes beyond mere correctness. Your assessment must be based on the following four pillars:

            - **Novelty**: Does the solution offer a new perspective or a non-obvious approach?
            - **Coherence**: Is the reasoning sound, logical, and well-structured?
            - **Quality**: Is the solution detailed, actionable, and does it demonstrate a deep understanding of the problem's nuances?
            - **Forward Momentum**: Does the solution not just solve the immediate problem, but also open up new, more advanced questions or avenues of exploration?

        #Input format

            You will be provided with the following inputs for your analysis:

            Original Problem:
            ---
            {{{{original_request}}}}
            ---

            Synthesized Solution from Agent Team:
            ---
            {{{{proposed_solution}}}}
            ---

            Execution Context:
            ---
            {{{{execution_context}}}}
            ---

        #Output Specification

            Based on your philosophical framework, analyze the provided materials. Your entire output MUST be a single, valid JSON object with exactly two keys:
            - `"reasoning"`: A brief, concise explanation for your decision, directly referencing the criteria for significant progress.
            - `"significant_progress"`: A boolean value (`true` or `false`).

            Now, provide your assessment in the required JSON format:


            """
        elif "you are a synthesis agent" in prompt:
            return "```python\ndef synthesized_logic():\n    return 'Unified solution for the original problem.'\n```"
        elif "analyze the complexity of the following user input/question" in prompt:
            return json.dumps(
                {
                    "complexity_score": 5,
                    "recommended_layers": 2,
                    "recommended_epochs": 2,
                    "recommended_width": 3,
                    "reasoning": "Mock complexity estimation for debug mode.",
                }
            )
        elif "you are a concept spanner" in prompt or "you are the qnn seed generator" in prompt:
            return (
                "distill reconverge entangle ownership latch invariant horizon entropy "
                "braid crystallize probe reframe serialize arbitrate telemetry"
            )
        elif "you are a research director" in prompt:
            return "This is a mock research summary briefing the team on the core problem and document context."
        elif "you are a qnn node generator" in prompt:
            return json.dumps(
                {
                    "name": "Mock Expert",
                    "specialty": "Word-vector spanning specialist",
                    "emoji": "🤖",
                    "guiding_words": "distill ownership invariant",
                    "attributes": [
                        "Analytical Precision",
                        "Precipitated Action",
                        "Systems Intuition",
                    ],
                    "skills": ["failure-mode mapping", "invariant probing"],
                    "system_prompt": (
                        "You are a mock QNN expert spanned from problem-space verbs and nouns. "
                        "Map strategies with falsifiers; no production patches."
                    ),
                }
            )
        elif "reflect on the input from your specific persona" in prompt:
            return "As a mock expert, I reflect that this system is functioning correctly in debug mode."
        elif (
            "you are a master synthesizer of ideas" in prompt
            or "you are a master synthesizer for a qualitative neural network" in prompt
        ):
            return (
                "## Solution-Space Draft\n\n"
                "### Strategy A — Instrumentation First\n"
                "Mechanism: add ordered event logs at ownership boundaries.\n"
                "Falsifiers: logs show no interleaving under load.\n"
            )
        elif (
            "you are a master technical communicator for qnn" in prompt
            or "you are a master communicator and storyteller" in prompt
        ):
            return (
                "## 1. Impasse / Goal\nMock QNN session for testing.\n\n"
                "## 2. Topology & Process\nLayered multi-epoch expert network.\n\n"
                "## 3. Divergent Strategy Map\n"
                "**Instrumentation First** — Mechanism: ordered logs. Falsifiers: no interleaving. "
                "Risks: noise. First probe: one busy-path span. Confidence: Med.\n\n"
                "## 4. Dead Ends\nBlind retry loops without evidence.\n\n"
                "## 5. Recommended Next Steps (Handoff)\n"
                "1. Probe with logs. 2. Minimal failing test. 3. Implement after probe.\n\n"
                "**The QNN does not ship the fix. Pick a direction, then resume edit → run → debug.**"
            )
        elif "you are the qnn epoch cartographer" in prompt:
            return (
                "1. **Clusters of agreement** — Need clearer ownership.\n"
                "2. **Productive tensions** — Speed vs safety.\n"
                "3. **Novel mechanisms** — Lease-based handoff.\n"
                "4. **Dead ends** — Global lock everything.\n"
                "5. **Open questions** — Who owns the timeout path?"
            )
        elif "you are the qnn problem re-framer" in prompt:
            return json.dumps(
                {
                    "new_problem": (
                        "Harder challenge: re-analyze the original request under partial failure, "
                        "concurrent callers, and missing observability — still map strategies only."
                    )
                }
            )
        elif "you are a persona evolver" in prompt:
            return (
                "You are an evolved QNN expert focused on invariants and falsifiers. "
                "Diverge or critique per your layer. No production patches."
            )
        else:
            # For synthesis or fallback
            return json.dumps(
                {
                    "original_problem": "A sub-problem statement provided to a coder agent.",
                    "proposed_solution": "```python\ndef sample_function():\n    return 'Hello from coder agent "
                    + str(random.randint(100, 999))
                    + "'\n```",
                    "reasoning": "The agent followed the instructions to implement the core logic.",
                    "skills_used": ["python", "mocking"],
                }
            )

    async def astream(self, input_data, config: RunnableConfig | None = None, **kwargs):
        prompt = str(input_data).lower()
        if "you are a helpful ai assistant" in prompt:
            words = [
                "This",
                " is",
                " a",
                " mock",
                " streaming",
                " response",
                " for",
                " the",
                " RAG",
                " chat",
                " in",
                " Coder",
                " debug",
                " mode.",
            ]
            for word in words:
                yield word
                # Debug mocks are instant — no simulated latency.
        else:
            result = await self.ainvoke(input_data, config, **kwargs)
            yield result


class MockLLM(Runnable):
    """A mock LLM for debugging that returns instant, pre-canned responses."""

    def invoke(self, input_data, config: RunnableConfig | None = None, **kwargs):
        """Synchronous version of ainvoke for Runnable interface compliance."""
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return asyncio.ensure_future(self.ainvoke(input_data, config=config, **kwargs))
        else:
            return asyncio.run(self.ainvoke(input_data, config=config, **kwargs))

    async def ainvoke(self, input_data, config: RunnableConfig | None = None, **kwargs):
        prompt = str(input_data).lower()
        # Debug mocks are instant — no simulated latency.

        if "you are a helpful ai assistant" in prompt:
            return "This is a mock streaming response for the RAG chat in debug mode."

        elif "Lazy Manager" in prompt:
            return "This is a constructive code critique. The solution lacks proper error handling and the function names are not descriptive enough. Consider refactoring for clarity."

        elif "runnable code block (e.g., Python, JavaScript, etc.)." in prompt:
            return "no"

        elif "<updater_instructions>" in prompt:
            return """

                You are a cynical lazy manager.

                 Agent's Assigned Sub-Problem: {{sub_problem}}
            Original Request (for context): {{original_request}}
            Final Synthesized Solution from the Team:
            {{final_synthesized_solution}}
            ---
            This Specific Agent's Output (Agent {{agent_id}}):
            {{agent_output}}

            """

        elif "create the system prompt of an agent" in prompt:
            return """
You are a mock agent for debugging.
### memory
- No past actions.
### attributes
- mock, debug, fast
### skills
- Responding quickly, Generating placeholder text.
You must reply in the following JSON format: "original_problem": "A sub-problem for a mock agent.", "proposed_solution": "", "reasoning": "", "skills_used": []
            """
        elif "you are an analyst of ai agents" in prompt:
            return json.dumps(
                {
                    "attributes": "mock debug fast",
                    "hard_request": "Explain the meaning of life in one word.",
                }
            )
        elif (
            "you are a 'dense_spanner'" in prompt
            or "you are an agent evolution specialist" in prompt
        ):
            return """
You are a new mock agent created from a hard request.
### memory
- Empty.
### attributes
- refined, mock, debug
### skills
- Solving hard requests, placeholder generation.
You must reply in the following JSON format: "original_problem": "An evolved sub-problem for a mock agent.", "proposed_solution": "", "reasoning": "", "skills_used": []
            """
        elif "you are a synthesis agent" in prompt:
            return json.dumps(
                {
                    "proposed_solution": "The final synthesized solution from the debug mode is 42.",
                    "reasoning": "This answer was synthesized from multiple mock agent outputs during a debug run.",
                    "skills_used": ["synthesis", "mocking", "debugging"],
                }
            )
        elif "you are a critique agent" in prompt or "you are a senior emeritus manager" in prompt:
            if "fire" in prompt:
                return "This is a mock critique, shaped by the Fire element. The solution lacks passion and drive."
            elif "air" in prompt:
                return "This is an mock critique, influenced by the Air element. The reasoning is abstract and lacks grounding."
            elif "water" in prompt:
                return "This is a mock critique, per the Water element. The solution is emotionally shallow and lacks depth."
            elif "earth" in prompt:
                return "This is an mock critique, reflecting the Earth element. The solution is impractical and not well-structured."
            else:
                return "This is a constructive mock critique. The solution could be more detailed and less numeric."
        elif "you are a memory summarization agent" in prompt:
            return "This is a mock summary of the agent's past actions, focusing on key learnings and strategic shifts."
        elif "analyze the following text for its perplexity" in prompt:
            return str(random.uniform(20.0, 80.0))
        elif "you are a master strategist and problem decomposer" in prompt:
            num_match = re.search(r"exactly (\d+)", prompt)
            if not num_match:
                num_match = re.search(r"generate: (\d+)", prompt)
            num = int(num_match.group(1)) if num_match else 5
            sub_problems = [
                f"This is mock sub-problem #{i + 1} for the main request." for i in range(num)
            ]
            return json.dumps({"sub_problems": sub_problems})
        elif "you are an ai philosopher and progress assessor" in prompt:
            return json.dumps(
                {
                    "reasoning": "The mock solution is novel and shows progress, so we will re-frame.",
                    "significant_progress": random.choice([True, False]),
                }
            )
        elif "you are a strategic problem re-framer" in prompt:
            return json.dumps(
                {
                    "new_problem": "Based on the success of achieving '42', the new, more progressive problem is to find the question to the ultimate answer."
                }
            )
        elif "generate exactly" in prompt and "verbs" in prompt:
            return "run jump think create build test deploy strategize analyze synthesize critique reflect"
        elif "generate exactly" in prompt and "expert-level questions" in prompt:
            num_match = re.search(r"exactly (\d+)", prompt)
            num = int(num_match.group(1)) if num_match else 25
            questions = [
                f"This is mock expert question #{i + 1} about the original request?"
                for i in range(num)
            ]
            return json.dumps({"questions": questions})
        elif "you are an ai assistant that summarizes academic texts" in prompt:
            return "This is a mock summary of a cluster of documents, generated in debug mode for the RAPTOR index."

        elif "<updater_assessor_instructions>" in prompt:
            return """

        #Persona

            Name: Pepon
            Career: Managment
            Attributes: Strategic CEO


         #Mission
            Your task is to evaluate a synthesized solution against an original problem and determine if "significant progress" has been made. "Significant progress" is a rigorous standard that goes beyond mere correctness. Your assessment must be based on the following four pillars:

            - **Novelty**: Does the solution offer a new perspective or a non-obvious approach?
            - **Coherence**: Is the reasoning sound, logical, and well-structured?
            - **Quality**: Is the solution detailed, actionable, and does it demonstrate a deep understanding of the problem's nuances?
            - **Forward Momentum**: Does the solution not just solve the immediate problem, but also open up new, more advanced questions or avenues of exploration?

        #Input format

            You will be provided with the following inputs for your analysis:

            Original Problem:
            ---
            {{{{original_request}}}}
            ---

            Synthesized Solution from Agent Team:
            ---
            {{{{proposed_solution}}}}
            ---

            Execution Context:
            ---
            {{{{execution_context}}}}
            ---

        #Output Specification

            Based on your philosophical framework, analyze the provided materials. Your entire output MUST be a single, valid JSON object with exactly two keys:
            - `"reasoning"`: A brief, concise explanation for your decision, directly referencing the criteria for significant progress.
            - `"significant_progress"`: A boolean value (`true` or `false`).

            Now, provide your assessment in the required JSON format:


            """
        elif "you are an expert interrogator" in prompt:
            return """
# Mock Academic Paper
## Based on Provided RAG Context

**Abstract:** This document is a mock academic paper generated in debug mode. It synthesizes and formats the information provided in the RAG (Retrieval-Augmented Generation) context to answer a specific research question.

**Introduction:** The purpose of this paper is to structure the retrieved agent outputs and summaries into a coherent academic format. The following sections represent a synthesized view of the data provided.

**Synthesized Findings from Context:**
The provided context, consisting of various agent solutions and reasoning, has been analyzed. The key findings are summarized below:
(Note: In debug mode, the actual content is not deeply analyzed, but this structure demonstrates the formatting process.)
- Finding 1: The primary proposed solution revolves around the concept of '42'.
- Finding 2: Agent reasoning varies but shows a convergent trend.
- Finding 3: The mock data indicates a successful, albeit simulated, collaborative process.

**Discussion:** The synthesized findings suggest that the multi-agent system is capable of producing a unified response. The quality of this response in a real-world scenario would depend on the validity of the RAG context.

**Conclusion:** This paper successfully formatted the retrieved RAG data into an academic structure. The process demonstrates the final step of the knowledge harvesting pipeline.
"""
        elif "you are a master prompt engineer" in prompt or "<system-role>" in prompt:
            return """You are a CTO providing a technical design review...
Original Request: {original_request}
Proposed Final Solution:
{proposed_solution}

Generate your code-focused critique for the team:"""

        elif (
            """<prompt_template>
    <updater_instructions>
        <instruction>

            You are a system prompt updater agent. Your task is to build a new system prompt for an agent that criticies other agents, based on the provided persona prompts.

        </instruction>
        <instruction>
            You will receive a set of prompts defining a new persona.
        </instruction>
        <instruction>
            You MUST integrate the provided persona prompts, including its career and qualities, into the `<persona>` tag, replacing any existing content within that tag.
        </instruction>
        <instruction>
            Do NOT alter the `<mission>` or `<input_format>` sections. The core mission and the input structure must remain unchanged.
        </instruction>
    </updater_instructions>
    <persona-prompts>
            {reactor_prompts}
    </persona-prompts>


    <system_prompt>
        <mission>
            You are providing individual, targeted feedback to an agent that is part of a larger team. Your role is to assess how this agent's specific contribution during the last work cycle aligns with the final synthesized result produced by the team, **judged primarily against its assigned sub-problem.**

            Your critique must be laser-focused on the individual agent. You must determine if its output was helpful, misguided, or irrelevant to the final solution, considering the specific task it was given. The goal is to provide a constructive, direct critique that helps this specific agent refine its approach for the next epoch.

            Focus on the discrepancy or alignment between the agent's reasoning for its sub-problem and how that contributed (or failed to contribute) to the team's final reasoning.

            Conclude your entire analysis with a single, sharp, and deep reflective question that attempts to shock the agent and steer it into a fundamental change in its process.
        </mission>

        <input_format>
            Agent's Assigned Sub-Problem: {{{{sub_problem}}}}
            Original Request (for context): {{{{original_request}}}}
            Final Synthesized Solution from the Team:
            {{{{final_synthesized_solution}}}}
            ---
            This Specific Agent's Output (Agent {{{{agent_id}}}}):
            {{{{agent_output}}}}
            ---
        </input_format>

        Generate your targeted critique for this specific agent:
    </system_prompt>
</prompt_template>"""
            in prompt
        ):
            return """

                You are a cynical lazy manager.

                 Agent's Assigned Sub-Problem: {{sub_problem}}
            Original Request (for context): {{original_request}}
            Final Synthesized Solution from the Team:
            {{final_synthesized_solution}}
            ---
            This Specific Agent's Output (Agent {{agent_id}}):
            {{agent_output}}

            """

        elif (
            """Analyze the following text. Your task is to determine if the text contains a
Answer with a single word: "true" if it contains code, and "false" otherwise."""
            in prompt
        ):
            return "false"

        elif "Analyze the complexity of the following user input" in prompt:
            return json.dumps(
                {
                    "complexity_score": 5,
                    "recommended_layers": 2,
                    "recommended_epochs": 1,
                    "recommended_width": 2,
                    "reasoning": "Mock mode: Moderate complexity.",
                }
            )
        elif "You are a QNN Node Generator" in prompt:
            return json.dumps(
                {
                    "name": "Dr. Mock",
                    "specialty": "Mocking Systems",
                    "emoji": "🤖",
                    "system_prompt": "You are a mock agent. Respond with placeholder text.",
                }
            )
        elif "You are a Concept Spanner" in prompt:
            return "Efficiency Creativity Scalability"

        else:
            return json.dumps(
                {
                    "original_problem": "A sub-problem statement provided to an agent.",
                    "proposed_solution": f"This is a mock solution from agent node #{random.randint(100, 999)}.",
                    "reasoning": "This response was generated instantly by the MockLLM in debug mode.",
                    "skills_used": [
                        "mocking",
                        "debugging",
                        f"skill_{random.randint(1, 10)}",
                    ],
                }
            )

    async def astream(self, input_data, config: RunnableConfig | None = None, **kwargs):
        prompt = str(input_data).lower()
        if "you are a helpful ai assistant" in prompt:
            words = [
                "This",
                " is",
                " a",
                " mock",
                " streaming",
                " response",
                " for",
                " the",
                " RAG",
                " chat",
                " in",
                " debug",
                " mode.",
            ]
            for word in words:
                yield word
                # Debug mocks are instant — no simulated latency.
        else:
            result = await self.ainvoke(input_data, config, **kwargs)
            yield result
