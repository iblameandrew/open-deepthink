"""Legacy LangGraph node factories (inference replay + node unit tests).

Live Brainstorm / QDAD runs use ``deepthink.qnn`` and ``deepthink.qdad``.
These nodes remain for ``/run_inference_from_state`` and the phase-8/11 tests.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import traceback

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from deepthink.chains import (
    get_attribute_and_hard_request_generator_chain,
    get_brainstorming_epoch_map_chain,
    get_brainstorming_mirror_descent_chain,
    get_brainstorming_polisher_chain,
    get_brainstorming_reframer_chain,
    get_brainstorming_synthesis_chain,
    get_code_synthesis_chain,
    get_dense_spanner_chain,
    get_interrogator_chain,
    get_memory_summarizer_chain,
    get_module_card_chain,
    get_paper_formatter_chain,
    get_perplexity_heuristic_chain,
    get_problem_decomposition_chain,
    get_problem_reframer_chain,
    get_synthesis_chain,
)
from deepthink.rag import RAPTOR
from deepthink.runtime.bus import emit
from deepthink.self_attention import compute_self_attention
from deepthink.state import GraphState
from deepthink.utils import clean_and_parse_json, execute_code_in_sandbox


def create_agent_node(llm, node_id):
    """
    Creates a node in the graph that represents an agent.
    Each agent is powered by an LLM and has a specific system prompt.
    """
    agent_chain = ChatPromptTemplate.from_template("{input}") | llm | StrOutputParser()

    async def agent_node(state: GraphState):
        """
        The function that will be executed when the node is called in the graph.
        """
        await emit(f"--- [FORWARD PASS] Invoking Agent: {node_id} ---")

        try:
            layer_index_str, agent_index_str = node_id.split("_")[1:]
            layer_index, agent_index = int(layer_index_str), int(agent_index_str)
            agent_prompt = state["all_layers_prompts"][layer_index][agent_index]

            # Problem 2: Prepend name and specialty to prompt for better agent identity
            agent_personas = state.get("agent_personas", {})
            persona = agent_personas.get(node_id, {})
            if persona:
                name = persona.get("name", "Expert")
                specialty = persona.get("specialty", "Specialist")
                agent_prompt = f"YOU ARE {name.upper()}, A {specialty.upper()}.\n\n{agent_prompt}"
        except (ValueError, IndexError):
            await emit(f"ERROR: Could not find prompt for {node_id} in state. Halting agent.")
            return {}

        prev_layer_outputs = []
        if layer_index == 0:
            await emit(f"LOG: Agent {node_id} (Layer 0) is processing its sub-problem.")
            input_data = state["decomposed_problems"].get(node_id, state["original_request"])
        else:
            prev_layer_index = layer_index - 1
            num_agents_prev_layer = len(state["all_layers_prompts"][prev_layer_index])

            for i in range(num_agents_prev_layer):
                prev_node_id = f"agent_{prev_layer_index}_{i}"
                if prev_node_id in state["agent_outputs"]:
                    # Label upstream outputs so deeper layers can cite them
                    upstream = state["agent_outputs"][prev_node_id]
                    if isinstance(upstream, dict):
                        prev_layer_outputs.append({"agent_id": prev_node_id, **upstream})
                    else:
                        prev_layer_outputs.append({"agent_id": prev_node_id, "output": upstream})

            await emit(
                f"LOG: Agent {node_id} (Layer {layer_index}) is processing {len(prev_layer_outputs)} outputs from Layer {prev_layer_index}."
            )
            input_data = json.dumps(prev_layer_outputs, indent=2)

        current_memory = state.get("memory", {}).copy()
        agent_memory_history = current_memory.get(node_id, [])

        MEMORY_THRESHOLD_CHARS = 450000
        NUM_RECENT_ENTRIES_TO_KEEP = 10

        memory_as_string = json.dumps(agent_memory_history)
        if (
            len(memory_as_string) > MEMORY_THRESHOLD_CHARS
            and len(agent_memory_history) > NUM_RECENT_ENTRIES_TO_KEEP
        ):
            await emit(
                f"WARNING: Memory for agent {node_id} exceeds threshold ({len(memory_as_string)} chars). Summarizing..."
            )

            entries_to_summarize = agent_memory_history[:-NUM_RECENT_ENTRIES_TO_KEEP]
            recent_entries = agent_memory_history[-NUM_RECENT_ENTRIES_TO_KEEP:]

            history_to_summarize_str = json.dumps(entries_to_summarize, indent=2)

            summarizer_chain = get_memory_summarizer_chain(llm)
            summary_text = await summarizer_chain.ainvoke({"history": history_to_summarize_str})

            summary_entry = {
                "summary_of_past_epochs": summary_text,
                "note": f"This is a summary of epochs up to {state['epoch'] - NUM_RECENT_ENTRIES_TO_KEEP - 1}.",
            }

            agent_memory_history = [summary_entry] + recent_entries
            await emit(
                f"SUCCESS: Memory for agent {node_id} has been summarized. New memory length: {len(json.dumps(agent_memory_history))} chars."
            )

        memory_str = "\n".join([f"- {json.dumps(mem)}" for mem in agent_memory_history])

        # Brainstorm mode: full QNN layered forward pass (do NOT flatten to original_request).
        # Algorithm mode keeps decomposed_problems (L0) / upstream outputs (L1+).
        brainstorm_context = ""
        attention_block = ""
        attention_edge_dicts: list[dict] = []
        json_schema_block = "#Your JSON formatted response:"
        if state.get("mode") == "brainstorm":
            prior_conv = state.get("brainstorm_prior_conversation", "") or ""
            brief = state.get("brainstorm_problem_summary") or state.get("original_request") or ""
            thinking_challenge = state.get("current_problem") or state.get("original_request") or ""
            epoch_n = state.get("epoch", 0)

            if prior_conv:
                brainstorm_context = f"""
# Prior Conversation Context:
---
{prior_conv[:20000]}
---
"""

            # Qualitative Self-Attention (colony QSA analogue):
            # attend past / non-neighbor neurons — not only previous-layer edges.
            try:
                att_edges, attention_block = compute_self_attention(state, node_id, top_k=5)
                attention_edge_dicts = [e.to_dict() for e in att_edges]
                if att_edges:
                    attended = ", ".join(
                        f"{e.to_id}({e.strength}/{e.qualitative_distance})" for e in att_edges
                    )
                    await emit(
                        f"LOG: [QNN ATTEND] {node_id} self-attention → {len(att_edges)} "
                        f"non-local past neuron(s): {attended}"
                    )
                else:
                    await emit(
                        f"LOG: [QNN ATTEND] {node_id} self-attention → no eligible past neurons yet."
                    )
            except Exception as att_err:
                await emit(f"WARNING: [QNN ATTEND] {node_id} self-attention failed: {att_err}")
                attention_block = ""
                attention_edge_dicts = []

            if layer_index == 0:
                await emit(
                    f"LOG: [QNN FORWARD] {node_id} Layer 0 DIVERGENT pass (epoch {epoch_n})."
                )
                input_data = f"""## QNN Brief
{brief}

## Original Request (ground truth — do not replace)
{state.get("original_request", "")}

## Thinking Challenge (epoch {epoch_n})
{thinking_challenge}

## Layer 0 Role
Divergent exploration. Span strategies and mechanisms. Do NOT write production patches or full file diffs.

{attention_block}
"""
            else:
                await emit(
                    f"LOG: [QNN FORWARD] {node_id} Layer {layer_index} CONVERGENT pass "
                    f"({len(prev_layer_outputs)} upstream) epoch {epoch_n}."
                )
                input_data = f"""## QNN Brief
{brief}

## Original Request (ground truth — do not replace)
{state.get("original_request", "")}

## Thinking Challenge (epoch {epoch_n})
{thinking_challenge}

## Layer {layer_index} Role
Convergent / critical. Critique, refine, reject, or combine upstream outputs. Cite agent_id values.
Do NOT restate Layer 0. Do NOT write production patches.

## Upstream Layer Outputs (graph neighbors — previous layer)
{json.dumps(prev_layer_outputs, indent=2)}

{attention_block}
"""

            json_schema_block = """# Your JSON response (required keys):
{
  "original_problem": "<brief or challenge you addressed>",
  "proposed_solution": "<strategic angle / mechanism — NOT a production patch>",
  "reasoning": "<why this might break the impasse or enrich the artifact>",
  "falsifiers": "<evidence that would kill this angle>",
  "risks": "<ways it could fail>",
  "skills_used": []
}
"""

        full_prompt = f"""
#System Prompt (Your Persona & Task):
---
{agent_prompt}
---
{brainstorm_context}
#Your Memory (Your Past Actions from Previous Epochs):
---
{memory_str if memory_str else "You have no past actions in memory."}
---
#Input Data to Process:
---
{input_data}
---
{json_schema_block}
"""
        await emit(f"LOG: Agent {node_id} prompt:\n{full_prompt}")

        response_str = await agent_chain.ainvoke({"input": full_prompt})

        try:
            response_json = clean_and_parse_json(response_str)

            if response_json is None:
                raise ValueError("JSON parsing failed (returned None)")

            await emit(
                f"SUCCESS: Agent {node_id} produced output:\n{json.dumps(response_json, indent=2)}"
            )
        except (json.JSONDecodeError, AttributeError, ValueError) as e:
            await emit(
                f"ERROR: Agent {node_id} produced invalid JSON. Raw output: {response_str}. Error: {e}"
            )

            # FALLBACK STRATEGY:
            # If parsing fails, try to return a "safe" no-op response or the original request to keep graph alive
            agent_sub_problem = state.get("decomposed_problems", {}).get(
                node_id, state["original_request"]
            )

            # Try to recover a previous valid state if possible, otherwise generic error
            response_json = {
                "original_problem": agent_sub_problem,
                "proposed_solution": f"System Fallback: The neuron {node_id} failed to format its response correctly. Raw output was captured.",
                "reasoning": f"JSON Parsing Error. Raw Content: {response_str[:500]}...",
                "skills_used": [],
                "node_id": node_id,
            }

        if state.get("is_code_request") and layer_index > 0:
            await emit(f"--- [SANDBOX] Testing code from Agent {node_id} ---")
            code_to_test = response_json.get("proposed_solution", "")
            success, output = execute_code_in_sandbox(code_to_test)
            sandbox_log = {"sandbox_execution_log": {"success": success, "output": output}}
            agent_memory_history.append(sandbox_log)
            await emit(
                f"--- [SANDBOX] Agent {node_id} Result: {'Success' if success else 'Failure'} ---"
            )
            await emit(output)

        agent_memory_history.append(response_json)
        current_memory[node_id] = agent_memory_history

        result_delta = {
            "agent_outputs": {node_id: response_json},
            "memory": {node_id: agent_memory_history},  # RETURN DELTA ONLY to avoid race conditions
        }
        if state.get("mode") == "brainstorm" and attention_edge_dicts:
            result_delta["attention_edges"] = {node_id: attention_edge_dicts}
        return result_delta

    return agent_node


def create_synthesis_node(llm):
    async def synthesis_node(state: GraphState):
        await emit("--- [FORWARD PASS] Entering Synthesis Node ---")

        is_code = state.get("is_code_request", False)
        previous_solution = state.get("final_solution")

        if state.get("mode") == "brainstorm":
            await emit("LOG: [BRAINSTORM] Synthesizing expert reflections...")
            synthesis_chain = get_brainstorming_synthesis_chain(llm)
            # Brainstorm synthesis context (will be populated below)
            synthesis_context = ""
        elif is_code:
            await emit(
                "LOG: Original request detected as a code generation task. Using code synthesis prompt."
            )
            synthesis_chain = get_code_synthesis_chain(llm)

            synthesis_context = "\n\n".join(state.get("synthesis_context_queue", []))
            if not synthesis_context:
                synthesis_context = "No modules have been successfully built yet."
            await emit(
                f"LOG: Providing synthesis agent with context from {len(state.get('synthesis_context_queue', []))} modules."
            )
        else:
            await emit("LOG: Original request is not a code task. Using standard synthesis prompt.")
            synthesis_chain = get_synthesis_chain(llm)
            synthesis_context = ""

        last_agent_layer_idx = len(state["all_layers_prompts"]) - 1
        num_agents_last_layer = len(state["all_layers_prompts"][last_agent_layer_idx])

        last_layer_outputs = []
        for i in range(num_agents_last_layer):
            node_id = f"agent_{last_agent_layer_idx}_{i}"
            if node_id in state["agent_outputs"]:
                out = state["agent_outputs"][node_id]
                if isinstance(out, list):
                    if not out:
                        continue
                    out = out[-1]
                if isinstance(out, dict):
                    last_layer_outputs.append(out)

        await emit(
            f"LOG: Synthesizing {len(last_layer_outputs)} outputs from the final agent layer (Layer {last_agent_layer_idx})."
        )

        if state.get("mode") == "brainstorm":
            # QNN brainstorm synthesis uses full `memory` (layered multi-epoch history).
            if not state.get("memory"):
                await emit("WARNING: Synthesis node received no inputs (brainstorm memory empty).")
                return {"final_solution": {"error": "Synthesis node received no inputs."}}

            is_final_epoch = state["epoch"] >= state["max_epochs"] - 1
            agent_reflections = ""
            memory = state.get("memory", {})

            for layer_idx, layer in enumerate(state.get("all_layers_prompts", [])):
                for agent_idx in range(len(layer)):
                    node_id = f"agent_{layer_idx}_{agent_idx}"
                    history = memory.get(node_id, [])

                    for hist_idx, entry in enumerate(history):
                        if isinstance(entry, dict):
                            sol = entry.get("proposed_solution")
                            reas = entry.get("reasoning")
                            falsifiers = entry.get("falsifiers", "")
                            risks = entry.get("risks", "")
                            if sol and not str(sol).startswith("Error"):
                                agent_reflections += (
                                    f"Agent {node_id} (Epoch {hist_idx}):\n"
                                    f"Reflection: {sol}\n"
                                    f"Reasoning: {reas}\n"
                                )
                                if falsifiers:
                                    agent_reflections += f"Falsifiers: {falsifiers}\n"
                                if risks:
                                    agent_reflections += f"Risks: {risks}\n"
                                agent_reflections += "\n"

            doc_ctx = state.get("brainstorm_document_context", "") or ""
            prior_conv = state.get("brainstorm_prior_conversation", "") or ""
            synthesis_input_concept = (
                state.get("brainstorm_problem_summary") or state["original_request"]
            )
            thinking_challenge = state.get("current_problem") or state["original_request"]

            await emit(
                f"LOG: [QNN SYNTHESIS] epoch={state['epoch']} final={is_final_epoch} "
                f"reflections_chars={len(agent_reflections)} memory_keys={list(memory.keys())}"
            )

            if not is_final_epoch:
                # Step 4B: compact epoch map (feeds reframe + next epoch; no FINAL_ANSWER)
                await emit(
                    f"LOG: [QNN STEP 4B] Epoch map for intermediate epoch {state['epoch']}..."
                )
                epoch_map_chain = get_brainstorming_epoch_map_chain(llm)
                epoch_map_str = await epoch_map_chain.ainvoke(
                    {
                        "original_request": synthesis_input_concept,
                        "current_problem": thinking_challenge,
                        "agent_solutions": agent_reflections,
                    }
                )
                final_solution = {
                    "proposed_solution": epoch_map_str,
                    "reasoning": f"QNN epoch map (epoch {state['epoch']}, intermediate).",
                    "mode": "brainstorm",
                    "epoch_map": True,
                    "epoch": state["epoch"],
                }
                await emit(f"SUCCESS: [QNN] Intermediate epoch map ready (epoch {state['epoch']}).")
                # Stream epoch map for UI; epoch_map=true keeps the diffusion vortex spinning
                await emit(f"FINAL_ANSWER: {json.dumps(final_solution)}")
                return {
                    "final_solution": final_solution,
                    "previous_solution": epoch_map_str,
                }

            # Step 5: final Solution-Space Report
            await emit("LOG: [QNN STEP 5] Final epoch — Solution-Space Report synthesis...")
            final_solution_str = await synthesis_chain.ainvoke(
                {
                    "original_request": synthesis_input_concept,
                    "agent_solutions": agent_reflections,
                    "prior_conversation": prior_conv[:15000],
                    "document_context": doc_ctx[:20000],
                }
            )

            await emit("LOG: [QNN STEP 5] Polishing Solution-Space Report for delivery...")
            polisher_chain = get_brainstorming_polisher_chain(llm)
            final_solution_str = await polisher_chain.ainvoke(
                {
                    "original_request": synthesis_input_concept,
                    "initial_synthesis": final_solution_str,
                }
            )

            final_solution = {
                "proposed_solution": final_solution_str,
                "reasoning": "QNN Solution-Space Report complete.",
                "mode": "brainstorm",
                "epoch_map": False,
            }
            await emit(
                f"LOG: [DEBUG] Emitting FINAL_ANSWER token to frontend. Solution length: {len(final_solution_str)}"
            )
            await emit("SUCCESS: [QNN] Brainstorm Solution-Space Report complete.")
            # Emit full dict so frontend can read mode / epoch_map and stop the vortex correctly
            await emit(f"FINAL_ANSWER: {json.dumps(final_solution)}")

        else:
            # Algorithm / Code Synthesis — uses `last_layer_outputs` (agent_outputs).
            # We only get here in algorithm mode; the brainstorm branch above
            # already returned. Defensive check: in algorithm mode last_layer_outputs
            # must be non-empty (set above).
            if not last_layer_outputs:
                await emit("WARNING: Synthesis node received no inputs.")
                return {"final_solution": {"error": "Synthesis node received no inputs."}}
            invoke_params = {
                "original_request": state["original_request"],
                "agent_solutions": json.dumps(last_layer_outputs, indent=2),
                "current_problem": state["current_problem"],
            }
            if is_code:
                invoke_params["synthesis_context"] = synthesis_context

            final_solution_str = await synthesis_chain.ainvoke(invoke_params)

            try:
                if is_code:
                    final_solution = {
                        "proposed_solution": final_solution_str,
                        "reasoning": "Synthesized multiple agent code outputs into a single application.",
                        "skills_used": ["code_synthesis"],
                        "mode": "algorithm",
                    }
                else:
                    final_solution = clean_and_parse_json(final_solution_str)
                    if isinstance(final_solution, dict):
                        final_solution["mode"] = "algorithm"
                    else:
                        # Fallback if it's just a string
                        final_solution = {
                            "proposed_solution": str(final_solution),
                            "mode": "algorithm",
                        }
                await emit("SUCCESS: Synthesis complete.")
            except (json.JSONDecodeError, AttributeError):
                await emit(
                    f"ERROR: Could not decode JSON from synthesis chain. Result: {final_solution_str}"
                )
                final_solution = {
                    "error": "Failed to synthesize final solution.",
                    "raw": final_solution_str,
                }

        return {
            "final_solution": final_solution,
            "previous_solution": previous_solution,
        }

    return synthesis_node


def create_code_execution_node(llm):
    async def code_execution_node(state: GraphState):
        if not state.get("is_code_request"):
            return {"synthesis_execution_success": True}

        await emit("--- [SANDBOX] Testing Synthesized Code ---")
        synthesized_code = state.get("final_solution", {}).get("proposed_solution", "")

        success, output = execute_code_in_sandbox(synthesized_code)

        await emit(
            f"--- [SANDBOX] Synthesized Code Result: {'Success' if success else 'Failure'} ---"
        )
        await emit(output)

        module_card_chain = get_module_card_chain(llm)
        module_card = await module_card_chain.ainvoke({"code": synthesized_code})

        await emit("--- [MODULE CARD] ---")
        await emit(module_card)

        new_modules = state.get("modules", []) + [{"code": synthesized_code, "card": module_card}]
        new_context_queue = state.get("synthesis_context_queue", []) + [module_card]

        return {
            "synthesis_execution_success": True,
            "modules": new_modules,
            "synthesis_context_queue": new_context_queue,
        }

    return code_execution_node


def create_archive_epoch_outputs_node():
    async def archive_epoch_outputs_node(state: GraphState):
        if state.get("mode") == "brainstorm":
            # await emit("LOG: [BRAINSTORM] Skipping RAG archival pass.") # Optional: Reduce noise
            return {}

        await emit("--- [ARCHIVAL PASS] Archiving agent outputs for RAG ---")

        current_epoch_outputs = state.get("agent_outputs", {})
        if not current_epoch_outputs:
            await emit("LOG: No new agent outputs in this epoch to archive. Skipping.")
            return {}

        await emit(
            f"LOG: Found {len(current_epoch_outputs)} new agent outputs from epoch {state['epoch']} to process for RAG."
        )

        new_docs = []
        all_prompts = state.get("all_layers_prompts", [])

        for agent_id, output in current_epoch_outputs.items():
            try:
                # Robustness check: if output is a list (due to merge_dicts or multiple runs), take the last one
                if isinstance(output, list):
                    if not output:
                        continue  # empty list
                    output = output[-1]

                if not isinstance(output, dict):
                    await emit(
                        f"WARNING: Output for {agent_id} is not a dict or list of dicts. Skipping. Type: {type(output)}"
                    )
                    continue

                layer_idx, agent_idx = map(int, agent_id.split("_")[1:])
                system_prompt = all_prompts[layer_idx][agent_idx]

                content = (
                    f"Agent ID: {agent_id}\n"
                    f"Epoch: {state['epoch']}\n\n"
                    f"System Prompt:\n---\n{system_prompt}\n---\n\n"
                    f"Sub-Problem: {output.get('original_problem', 'N/A')}\n\n"
                    f"Proposed Solution: {output.get('proposed_solution', 'N/A')}\n\n"
                    f"Reasoning: {output.get('reasoning', 'N/A')}"
                )

                metadata = {"agent_id": agent_id, "epoch": state["epoch"]}

                new_docs.append(Document(page_content=content, metadata=metadata))
            except (ValueError, IndexError) as e:
                await emit(
                    f"WARNING: Could not process output for {agent_id} to create RAG document. Error: {e}"
                )

        all_rag_documents = state.get("all_rag_documents", []) + new_docs
        await emit(
            f"LOG: Archived {len(new_docs)} documents. Total RAG documents now: {len(all_rag_documents)}."
        )

        return {"all_rag_documents": all_rag_documents}

    return archive_epoch_outputs_node


def create_update_rag_index_node(llm, embeddings_model):
    async def update_rag_index_node(state: GraphState, end_of_run: bool = False):
        node_name = "Final RAG Index" if end_of_run else f"Epoch {state['epoch']} RAG Index"
        await emit(f"--- [RAG PASS] Building {node_name} ---")

        all_rag_documents = state.get("all_rag_documents", [])
        if not all_rag_documents:
            await emit("WARNING: No documents were archived. Cannot build RAG index.")
            return {"raptor_index": None}

        if not embeddings_model:
            await emit("WARNING: No embeddings model configured. Skipping RAG index build.")
            return {"raptor_index": None}

        await emit(
            f"LOG: Total documents to index: {len(all_rag_documents)}. Building RAPTOR index..."
        )

        raptor_index = RAPTOR(llm=llm, embeddings_model=embeddings_model)

        try:
            await raptor_index.add_documents(all_rag_documents)
            await emit(f"SUCCESS: {node_name} built successfully.")
            await emit(f"__session_id__ {state.get('session_id')}")
            return {"raptor_index": raptor_index}
        except Exception as e:
            await emit(f"ERROR: Failed to build {node_name}. Error: {e}")
            await emit(traceback.format_exc())
            return {"raptor_index": state.get("raptor_index")}

    return update_rag_index_node


def create_metrics_node(llm):
    """
    NEW: This node calculates the perplexity heuristic for the epoch's agent outputs.
    """

    async def calculate_metrics_node(state: GraphState):
        await emit("--- [METRICS PASS] Calculating Perplexity Heuristic ---")

        all_outputs = state.get("agent_outputs", {})
        if not all_outputs:
            await emit("LOG: No agent outputs to analyze. Skipping perplexity calculation.")
            return {}

        combined_text_parts = []
        for agent_id, output in all_outputs.items():
            if isinstance(output, list):
                if not output:
                    continue
                output = output[-1]
            if not isinstance(output, dict):
                continue

            combined_text_parts.append(
                f"Agent {agent_id}:\nSolution: {output.get('proposed_solution', '')}\nReasoning: {output.get('reasoning', '')}"
            )

        combined_text = "\n\n---\n\n".join(combined_text_parts)

        perplexity_chain = get_perplexity_heuristic_chain(llm)

        try:
            score_str = await perplexity_chain.ainvoke({"text_to_analyze": combined_text})
            score = float(re.sub(r"[^\d.]", "", score_str))
            await emit(
                f"SUCCESS: Calculated perplexity heuristic for Epoch {state['epoch']}: {score}"
            )
        except (ValueError, TypeError) as e:
            score = 100.0
            await emit(
                f"ERROR: Could not parse perplexity score. Defaulting to 100. Raw output: '{score_str}'. Error: {e}"
            )

        await emit(
            json.dumps(
                {
                    "type": "perplexity_update",
                    "source": "graph",
                    "session_id": state.get("session_id"),
                    "epoch": state["epoch"],
                    "perplexity": score,
                }
            )
        )

        new_history = state.get("perplexity_history", []) + [score]
        return {"perplexity_history": new_history}

    return calculate_metrics_node


def create_reframe_and_decompose_node(llm):
    """
    QNN Step 4D + algorithm decomposition:
    - Brainstorm: harder thinking challenge; original_request stays ground truth.
    - Algorithm: reframe + full sub-problem re-decomposition.
    """

    async def reframe_and_decompose_node(state: GraphState):
        await emit("--- [REFLECTION PASS] Re-framing Problem and Decomposing ---")

        final_solution = state.get("final_solution")
        original_request = state.get("original_request")

        if state.get("mode") == "brainstorm":
            await emit(
                "LOG: [QNN STEP 4D] Reframing thinking challenge for next epoch "
                "(original request remains ground truth)..."
            )
            reframer_chain = get_brainstorming_reframer_chain(llm)
            fs_payload = final_solution
            if isinstance(final_solution, dict):
                fs_payload = final_solution.get(
                    "proposed_solution", json.dumps(final_solution, indent=2)
                )
            new_problem_str = await reframer_chain.ainvoke(
                {
                    "original_request": original_request,
                    "current_problem": state.get("current_problem") or original_request,
                    "final_solution": fs_payload
                    if isinstance(fs_payload, str)
                    else json.dumps(fs_payload, indent=2),
                    "prior_conversation": state.get("brainstorm_prior_conversation", "") or "",
                }
            )
            try:
                new_problem_data = clean_and_parse_json(new_problem_str)
                new_problem = (new_problem_data or {}).get("new_problem")
                if not new_problem:
                    raise ValueError("Brainstorm re-framer did not return new_problem.")
                await emit(f"SUCCESS: [QNN] Thinking challenge re-framed to: '{new_problem}'")
            except (json.JSONDecodeError, AttributeError, ValueError, TypeError) as e:
                await emit(
                    f"ERROR: [QNN] Brainstorm re-frame failed. Raw: {new_problem_str}. Error: {e}."
                )
                return {}

            # All nodes share the harder challenge; ground truth stays original_request.
            new_decomposed = {
                f"agent_{i}_{j}": new_problem
                for i, layer in enumerate(state["all_layers_prompts"])
                for j in range(len(layer))
            }
            return {
                "decomposed_problems": new_decomposed,
                "original_request": original_request,
                "current_problem": new_problem,
            }

        reframer_chain = get_problem_reframer_chain(llm)
        new_problem_str = await reframer_chain.ainvoke(
            {
                "original_request": original_request,
                "final_solution": json.dumps(final_solution, indent=2),
                "current_problem": state.get("current_problem"),
                "previous_solution": state.get("previous_solution"),
                "module_cards": state.get("synthesis_context_queue"),
            }
        )
        try:
            new_problem_data = clean_and_parse_json(new_problem_str)
            new_problem = new_problem_data.get("new_problem")
            if not new_problem:
                raise ValueError("Re-framer did not return a new problem.")
            await emit(f"SUCCESS: Problem re-framed to: '{new_problem}'")
        except (json.JSONDecodeError, AttributeError, ValueError) as e:
            await emit(
                f"ERROR: Failed to re-frame problem. Raw: {new_problem_str}. Error: {e}. Aborting re-frame."
            )
            return {}

        num_agents_total = sum(len(layer) for layer in state["all_layers_prompts"])
        decomposition_chain = get_problem_decomposition_chain(llm)
        try:
            sub_problems_str = await decomposition_chain.ainvoke(
                {"problem": new_problem, "num_sub_problems": num_agents_total}
            )
            sub_problems_list = clean_and_parse_json(sub_problems_str).get("sub_problems", [])
            if len(sub_problems_list) != num_agents_total:
                raise ValueError(
                    f"Decomposition failed: Expected {num_agents_total} subproblems, but got {len(sub_problems_list)}."
                )
            await emit(
                f"SUCCESS: Decomposed new problem into {len(sub_problems_list)} subproblems."
            )
            await emit(f"Subproblems: {sub_problems_list}")
        except Exception as e:
            await emit(f"ERROR: Failed to decompose new problem. Error: {e}. Aborting re-frame.")
            return {}

        new_decomposed_problems_map = {}
        problem_idx = 0
        for i, layer in enumerate(state["all_layers_prompts"]):
            for j in range(len(layer)):
                agent_id = f"agent_{i}_{j}"
                new_decomposed_problems_map[agent_id] = sub_problems_list[problem_idx]
                problem_idx += 1

        return {
            "decomposed_problems": new_decomposed_problems_map,
            "original_request": original_request,
            "current_problem": new_problem,
        }

    return reframe_and_decompose_node


def create_update_agent_prompts_node(llm):
    """Creates the mirror descent node that updates agent prompts based on reflection."""

    async def update_agent_prompts_node(state: GraphState):
        await emit("--- [MIRROR DESCENT] Entering Agent Prompt Update Node ---")

        params = state.get("params", {})
        all_prompts_copy = [layer[:] for layer in state.get("all_layers_prompts", [])]

        if state.get("mode") == "brainstorm":
            await emit("LOG: [QNN STEP 4C] Mirror Descent — evolving expert personas...")
            mirror_chain = get_brainstorming_mirror_descent_chain(
                llm, params.get("learning_rate", 0.5)
            )

            for i in range(len(all_prompts_copy) - 1, -1, -1):
                await emit(f"LOG: [QNN STEP 4C] Evolving personas in Layer {i}...")

                update_tasks = []
                for j, agent_prompt in enumerate(all_prompts_copy[i]):
                    agent_id = f"agent_{i}_{j}"

                    async def evolve_persona(layer_idx, agent_idx, prompt, agent_id):
                        # Get last output for this agent
                        last_output = (
                            state.get("agent_outputs", {})
                            .get(agent_id, {})
                            .get("proposed_solution", "No output")
                        )

                        try:
                            new_prompt = await mirror_chain.ainvoke(
                                {"current_prompt": prompt, "last_output": last_output}
                            )
                            await emit(f"LOG: [EVOLUTION] Persona for {agent_id} evolved.")
                            return layer_idx, agent_idx, new_prompt
                        except Exception as e:
                            await emit(f"WARNING: Failed to evolve persona for {agent_id}: {e}")
                            return layer_idx, agent_idx, prompt

                    update_tasks.append(evolve_persona(i, j, agent_prompt, agent_id))

                updated_prompts_data = await asyncio.gather(*update_tasks)
                for layer_idx, agent_idx, new_prompt in updated_prompts_data:
                    all_prompts_copy[layer_idx][agent_idx] = new_prompt
        else:
            # Algorithm Mode - Standard Mirror Descent
            dense_spanner_chain = get_dense_spanner_chain(
                llm,
                params["prompt_alignment"],
                params["density"],
                params["learning_rate"],
            )
            attribute_chain = get_attribute_and_hard_request_generator_chain(
                llm, params["vector_word_size"]
            )

            for i in range(len(all_prompts_copy) - 1, -1, -1):
                await emit(f"LOG: [MIRROR_DESCENT] Reflecting on Layer {i}...")

                update_tasks = []

                for j, agent_prompt in enumerate(all_prompts_copy[i]):
                    agent_id = f"agent_{i}_{j}"

                    async def update_single_prompt(layer_idx, agent_idx, prompt, agent_id):
                        await emit(
                            f"[PRE-UPDATE PROMPT] System prompt for {agent_id}:\n---\n{prompt}\n---"
                        )

                        analysis_str = await attribute_chain.ainvoke({"agent_prompt": prompt})
                        try:
                            analysis = clean_and_parse_json(analysis_str)
                        except (json.JSONDecodeError, AttributeError):
                            analysis = {"attributes": "", "hard_request": ""}

                        agent_personas = state.get("agent_personas", {})
                        mbti_type = agent_personas.get(agent_id, {}).get("mbti_type")
                        name = agent_personas.get(agent_id, {}).get("name")

                        if not mbti_type:
                            mbti_type = random.choice(params.get("mbti_archetypes", ["INTP"]))
                            await emit(
                                f"WARNING: Could not find persistent MBTI for {agent_id}. Using random: {mbti_type}"
                            )

                        agent_sub_problem = state.get("decomposed_problems", {}).get(
                            agent_id, state["original_request"]
                        )
                        new_prompt = await dense_spanner_chain.ainvoke(
                            {
                                "attributes": analysis.get("attributes"),
                                "hard_request": analysis.get("hard_request"),
                                "sub_problem": agent_sub_problem,
                                "mbti_type": mbti_type,
                                "name": name,
                            }
                        )

                        await emit(
                            f"[POST-UPDATE PROMPT] Updated system prompt for {agent_id}:\n---\n{new_prompt}\n---"
                        )
                        await emit(
                            f"LOG: [MIRROR_DESCENT] System prompt for {agent_id} has been updated."
                        )
                        return layer_idx, agent_idx, new_prompt

                    update_tasks.append(update_single_prompt(i, j, agent_prompt, agent_id))

                updated_prompts_data = await asyncio.gather(*update_tasks)

                for layer_idx, agent_idx, new_prompt in updated_prompts_data:
                    all_prompts_copy[layer_idx][agent_idx] = new_prompt

        new_epoch = state["epoch"] + 1
        await emit(f"--- Epoch {state['epoch']} Finished. Starting Epoch {new_epoch} ---")

        return {
            "all_layers_prompts": all_prompts_copy,
            "epoch": new_epoch,
            "agent_outputs": {},
            "critiques": {},
            "memory": state.get("memory", {}),
            "final_solution": {},
        }

    return update_agent_prompts_node


def create_final_harvest_node(llm, formatter_llm, num_questions):
    async def final_harvest_node(state: GraphState):
        await emit("--- [FINAL HARVEST] Starting Interrogation and Paper Generation ---")

        raptor_index = state.get("raptor_index")
        if not raptor_index or not raptor_index.vector_store:
            await emit("ERROR: No valid RAPTOR index found. Cannot perform final harvest.")
            return {"academic_papers": {}}

        await emit(
            "LOG: [HARVEST] Instantiating interrogator chain to generate expert questions..."
        )
        interrogator_chain = get_interrogator_chain(llm)
        user_questions = [doc["content"] for doc in state["chat_history"] if doc["role"] == "user"]

        try:
            questions_str = await interrogator_chain.ainvoke(
                {
                    "original_request": state["original_request"],
                    "num_questions": num_questions,
                    "further_questions": user_questions,
                }
            )
            questions_data = clean_and_parse_json(questions_str)
            questions = questions_data.get("questions", [])
            if not questions:
                raise ValueError("No questions generated by interrogator.")
            await emit(f"SUCCESS: Generated {len(questions)} expert questions.")
        except Exception as e:
            await emit(
                f"ERROR: Failed to generate questions for harvesting. Error: {e}. Aborting harvest."
            )
            return {"academic_papers": {}}

        paper_formatter_chain = get_paper_formatter_chain(formatter_llm)
        academic_papers = {}

        MAX_CONTEXT_CHARS = 250000

        generation_tasks = []

        for question in questions:

            async def generate_paper(q):
                try:
                    await emit(f"LOG: [HARVEST] Processing Question: '{q[:100]}...'")
                    retrieved_docs = raptor_index.retrieve(q, k=40)

                    if not retrieved_docs:
                        await emit(
                            f"WARNING: No relevant documents found for question '{q[:50]}...'. Skipping paper generation."
                        )
                        return None, None

                    await emit(
                        f"LOG: Retrieved {len(retrieved_docs)} documents from RAG index for question."
                    )
                    rag_context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])

                    if len(rag_context) > MAX_CONTEXT_CHARS:
                        await emit(
                            f"WARNING: RAG context length ({len(rag_context)} chars) exceeds limit. Truncating to {MAX_CONTEXT_CHARS} chars."
                        )
                        rag_context = rag_context[:MAX_CONTEXT_CHARS]

                    paper_content = await paper_formatter_chain.ainvoke(
                        {"question": q, "rag_context": rag_context}
                    )
                    await emit(f"SUCCESS: Generated document for question '{q[:50]}...'.")
                    return q, paper_content
                except Exception as e:
                    await emit(
                        f"ERROR: Failed during document generation for question '{q[:50]}...'. Error: {e}"
                    )
                    return None, None

            generation_tasks.append(generate_paper(question))

        results = await asyncio.gather(*generation_tasks)
        for question, paper_content in results:
            if question and paper_content:
                academic_papers[question] = paper_content

        await emit(f"--- [FINAL HARVEST] Finished. Generated {len(academic_papers)} papers. ---")
        return {"academic_papers": academic_papers}

    return final_harvest_node
