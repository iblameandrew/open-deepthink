"""
QNN pipeline entrypoint — params → layered epochs → Solution-Space Report.

Model-agnostic: any LangChain-compatible chat model (OpenRouter, llama.cpp,
mocks, etc.). Used by the open-deepthink Brainstorm UI *and* the portable
``/qnn`` skill runner (``skills/qnn/run_qnn.py``).
"""
from __future__ import annotations

import json
import random
import traceback
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

from deepthink.utils import clean_and_parse_json
from deepthink.self_attention import compute_self_attention
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from deepthink.chains.brainstorm_chains import (
    get_complexity_estimator_chain,
    get_brainstorming_seed_chain,
    get_brainstorming_spanner_chain,
    get_brainstorming_mirror_descent_chain,
    get_brainstorming_reframer_chain,
    get_brainstorming_epoch_map_chain,
    get_brainstorming_synthesis_chain,
    get_brainstorming_polisher_chain,
    get_problem_summarizer_chain,
)


LogFn = Optional[Callable[[str], Any]]


def default_qnn_params() -> Dict[str, Any]:
    """Documented defaults for harnesses / CLI."""
    return {
        "qnn_mode": "auto",  # "auto" | "manual"
        "manual_layers": 3,
        "manual_width": 3,
        "num_epochs": 2,
        "vector_word_size": 6,
        "learning_rate": 0.5,
        "attention_top_k": 5,
        "enable_self_attention": True,
    }


@dataclass
class QNNResult:
    """Structured result returned to the skill / harness."""

    mode: str = "brainstorm"
    proposed_solution: str = ""
    reasoning: str = ""
    topology: Dict[str, Any] = field(default_factory=dict)
    seed_pool: List[str] = field(default_factory=list)
    column_guiding_words: List[str] = field(default_factory=list)
    agent_personas: Dict[str, Any] = field(default_factory=dict)
    attention_edges: Dict[str, Any] = field(default_factory=dict)
    epoch_maps: List[str] = field(default_factory=list)
    final_solution: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


async def _log(log: LogFn, msg: str) -> None:
    if log is None:
        return
    result = log(msg)
    if hasattr(result, "__await__"):
        await result


def _clamp_topology(layers: int, width: int, epochs: int) -> tuple:
    layers = max(1, min(20, int(layers)))
    width = max(1, min(20, int(width)))
    epochs = max(1, min(10, int(epochs)))
    # Soft budget unless caller already chose manual large
    if layers * width > 60:
        width = max(1, 60 // layers)
    return layers, width, epochs


async def run_qnn_pipeline(
    llm,
    user_prompt: str,
    params: Optional[Dict[str, Any]] = None,
    *,
    synthesis_llm=None,
    document_context: str = "",
    chat_history: Optional[List[dict]] = None,
    log: LogFn = None,
    session_id: str = "",
) -> Dict[str, Any]:
    """
    Run the full Qualitative Neural Network (brainstorm) pipeline.

    Parameters
    ----------
    llm :
        LangChain chat model used for most agents (seed, span, forward, MD).
    user_prompt : str
        Impasse / feature brief from the user.
    params : dict, optional
        Topology and knobs (see ``default_qnn_params``):

        * ``qnn_mode``: ``"auto"`` | ``"manual"``
        * ``manual_layers``, ``manual_width``: used when mode is manual
        * ``num_epochs``: E (overridden by estimator in auto mode)
        * ``vector_word_size``: V (words per column vector)
        * ``learning_rate``: Mirror Descent intensity (default 0.5)
        * ``attention_top_k``: non-local past neurons per agent (default 5)
        * ``enable_self_attention``: bool (default True)
    synthesis_llm :
        Optional separate model for final synthesis / polish.
    document_context : str
        Optional attached PDF/code/repo text.
    chat_history : list[dict]
        Optional ``[{role, content}, ...]``.
    log :
        Optional ``async/sync (str) -> None`` callback for progress lines.
    session_id : str
        Opaque id for callers that persist sessions.

    Returns
    -------
    dict
        ``QNNResult.to_dict()`` shape; primary text in
        ``proposed_solution`` / ``final_solution``.
    """
    p = {**default_qnn_params(), **(params or {})}
    synth = synthesis_llm or llm
    chat_history = chat_history or []

    chat_history_str = "\n".join(
        f"{'User' if m.get('role') == 'user' else 'Assistant'}: {m.get('content', '')}"
        for m in chat_history
    )

    await _log(log, "--- [QNN] Qualitative Neural Network pipeline ---")
    await _log(log, f"LOG: [QNN] session={session_id or 'local'} params={p}")

    # ── Step 0: Brief ──────────────────────────────────────────────
    await _log(log, "LOG: [QNN STEP 0] Building Impasse/Enrich brief...")
    if document_context:
        try:
            brief = await get_problem_summarizer_chain(llm).ainvoke(
                {
                    "user_input": user_prompt,
                    "document_context": document_context[:50000],
                }
            )
        except Exception:
            brief = user_prompt
    else:
        brief = user_prompt
    await _log(log, f"LOG: [QNN STEP 0] Brief ready ({len(brief)} chars).")

    # ── Step 1: Topology ───────────────────────────────────────────
    qnn_mode = str(p.get("qnn_mode", "auto")).lower()
    epochs = max(1, int(p.get("num_epochs", 2)))
    layers, width = 2, 3

    if qnn_mode == "manual":
        layers = int(p.get("manual_layers", 3))
        width = int(p.get("manual_width", 3))
        await _log(
            log,
            f"LOG: [QNN STEP 1] Manual topology request: {layers}L × {width}W × {epochs}E",
        )
    else:
        await _log(log, "LOG: [QNN STEP 1] Auto topology via complexity estimator...")
        try:
            raw = await get_complexity_estimator_chain(llm).ainvoke(
                {
                    "user_input": user_prompt,
                    "prior_conversation": chat_history_str,
                    "document_context": (document_context or "")[:10000],
                }
            )
            data = clean_and_parse_json(raw) or {}
            layers = int(data.get("recommended_layers", 2))
            width = int(data.get("recommended_width", 3))
            epochs = int(data.get("recommended_epochs", epochs))
            await _log(
                log,
                f"LOG: [QNN STEP 1] Auto: {layers}L×{width}W×{epochs}E "
                f"(score={data.get('complexity_score', '?')})",
            )
        except Exception as e:
            await _log(log, f"WARNING: complexity estimator failed ({e}); defaults.")
            layers, width, epochs = 2, 3, 2

    layers, width, epochs = _clamp_topology(layers, width, epochs)
    V = max(2, int(p.get("vector_word_size", 6)))
    top_k = max(0, int(p.get("attention_top_k", 5)))
    enable_attn = bool(p.get("enable_self_attention", True))
    lr = float(p.get("learning_rate", 0.5))

    topology = {
        "layers": layers,
        "width": width,
        "epochs": epochs,
        "vector_word_size": V,
        "agents": layers * width,
        "qnn_mode": qnn_mode,
        "enable_self_attention": enable_attn,
        "attention_top_k": top_k,
    }
    await _log(
        log,
        f"LOG: [QNN STEP 1] Final topology: {layers}L × {width}W × {epochs}E "
        f"({layers * width} agents, V={V}, attention={enable_attn})",
    )

    # ── Step 2: Seeds ──────────────────────────────────────────────
    total_seed = max(V * width, V * 2)
    await _log(log, f"LOG: [QNN STEP 2] Seeding {total_seed} verbs+nouns...")
    seeds_str = await get_brainstorming_seed_chain(llm).ainvoke(
        {"problem": user_prompt, "word_count": total_seed}
    )
    all_seed_words = list(
        {
            w.strip()
            for w in seeds_str.replace(",", " ").split()
            if w.strip() and len(w.strip()) > 1
        }
    )
    fallback = [
        "distill",
        "reconverge",
        "entangle",
        "ownership",
        "latch",
        "invariant",
        "horizon",
        "entropy",
        "braid",
        "crystallize",
        "probe",
        "reframe",
    ]
    while len(all_seed_words) < total_seed:
        all_seed_words.append(fallback[len(all_seed_words) % len(fallback)])
    random.shuffle(all_seed_words)

    column_guiding_words: List[str] = []
    for j in range(width):
        sample = (
            random.sample(all_seed_words, V)
            if len(all_seed_words) >= V
            else list(all_seed_words)
        )
        column_guiding_words.append(" ".join(sample))
        await _log(log, f"LOG: [QNN STEP 2] Column {j} guiding_words: {column_guiding_words[-1]}")

    # ── Step 3: Span personas ──────────────────────────────────────
    await _log(log, f"LOG: [QNN STEP 3] Spanning {layers}×{width} personas...")
    spanner = get_brainstorming_spanner_chain(llm)
    agent_personas: Dict[str, Any] = {}
    all_layers_prompts: List[List[str]] = []

    for i in range(layers):
        layer_prompts: List[str] = []
        for j in range(width):
            node_id = f"agent_{i}_{j}"
            raw = await spanner.ainvoke(
                {
                    "problem": user_prompt,
                    "guiding_words": column_guiding_words[j],
                    "layer_index": i,
                    "node_index": j,
                    "document_context": brief,
                }
            )
            persona = clean_and_parse_json(raw) or {}
            if not isinstance(persona, dict):
                persona = {}
            system_prompt = persona.get("system_prompt") or (
                f"You are a QNN expert spanned from: {column_guiding_words[j]}. "
                f"Layer {i} ({'diverge' if i == 0 else 'converge'}). Map strategies with falsifiers."
            )
            persona.setdefault("name", f"Agent_{i}_{j}")
            persona.setdefault("specialty", "Word-vector specialist")
            persona.setdefault("guiding_words", column_guiding_words[j])
            persona["system_prompt"] = system_prompt
            agent_personas[node_id] = persona
            layer_prompts.append(system_prompt)
            await _log(
                log,
                f"LOG: [QNN STEP 3] {node_id} → {persona.get('name')} / {persona.get('specialty')}",
            )
        all_layers_prompts.append(layer_prompts)

    # ── Step 4: Epoch loop ─────────────────────────────────────────
    memory: Dict[str, List[Any]] = {nid: [] for nid in agent_personas}
    attention_edges: Dict[str, Any] = {}
    epoch_maps: List[str] = []
    current_problem = user_prompt
    previous_solution = ""

    agent_chain = ChatPromptTemplate.from_template("{input}") | llm | StrOutputParser()

    for epoch in range(epochs):
        await _log(log, f"--- [QNN STEP 4] Epoch {epoch}/{epochs - 1} forward ---")
        agent_outputs: Dict[str, Any] = {}

        for i in range(layers):
            for j in range(width):
                node_id = f"agent_{i}_{j}"
                persona = agent_personas[node_id]
                agent_prompt = (
                    f"YOU ARE {str(persona.get('name', 'Expert')).upper()}, "
                    f"A {str(persona.get('specialty', 'Specialist')).upper()}.\n\n"
                    f"{persona.get('system_prompt', '')}"
                )

                # Upstream = previous layer (graph neighbors)
                prev_layer_outputs: List[Any] = []
                if i > 0:
                    for k in range(width):
                        prev_id = f"agent_{i - 1}_{k}"
                        if prev_id in agent_outputs:
                            up = agent_outputs[prev_id]
                            if isinstance(up, dict):
                                prev_layer_outputs.append({"agent_id": prev_id, **up})
                            else:
                                prev_layer_outputs.append(
                                    {"agent_id": prev_id, "output": up}
                                )

                # Qualitative self-attention over non-local past neurons
                attention_block = ""
                if enable_attn and top_k > 0:
                    state_snap = {
                        "epoch": epoch,
                        "all_layers_prompts": all_layers_prompts,
                        "agent_personas": agent_personas,
                        "agent_outputs": agent_outputs,
                        "memory": memory,
                    }
                    try:
                        edges, attention_block = compute_self_attention(
                            state_snap, node_id, top_k=top_k
                        )
                        if edges:
                            attention_edges[node_id] = [e.to_dict() for e in edges]
                            await _log(
                                log,
                                f"LOG: [QNN ATTEND] {node_id} → "
                                + ", ".join(
                                    f"{e.to_id}({e.strength})" for e in edges
                                ),
                            )
                    except Exception as ae:
                        await _log(log, f"WARNING: [QNN ATTEND] {node_id}: {ae}")

                if i == 0:
                    input_data = f"""## QNN Brief
{brief}

## Original Request (ground truth — do not replace)
{user_prompt}

## Thinking Challenge (epoch {epoch})
{current_problem}

## Layer 0 Role
Divergent exploration. Span strategies and mechanisms. Do NOT write production patches.

{attention_block}
"""
                else:
                    input_data = f"""## QNN Brief
{brief}

## Original Request (ground truth — do not replace)
{user_prompt}

## Thinking Challenge (epoch {epoch})
{current_problem}

## Layer {i} Role
Convergent / critical. Critique or combine upstream. Cite agent_id. No production patches.

## Upstream Layer Outputs (graph neighbors)
{json.dumps(prev_layer_outputs, indent=2)}

{attention_block}
"""

                mem_str = "\n".join(
                    f"- {json.dumps(m)}" for m in memory.get(node_id, [])[-5:]
                )
                full_prompt = f"""
#System Prompt (Your Persona & Task):
---
{agent_prompt}
---
#Your Memory (Past Actions):
---
{mem_str or "No past actions."}
---
#Input Data to Process:
---
{input_data}
---
# Your JSON response (required keys):
{{
  "original_problem": "<brief or challenge you addressed>",
  "proposed_solution": "<strategic angle / mechanism — NOT a production patch>",
  "reasoning": "<why this might break the impasse>",
  "falsifiers": "<evidence that would kill this angle>",
  "risks": "<ways it could fail>",
  "skills_used": []
}}
"""
                raw_out = await agent_chain.ainvoke({"input": full_prompt})
                parsed = clean_and_parse_json(raw_out)
                if not isinstance(parsed, dict):
                    parsed = {
                        "original_problem": current_problem,
                        "proposed_solution": str(raw_out)[:2000],
                        "reasoning": "unparsed agent output",
                        "falsifiers": "",
                        "risks": "",
                        "skills_used": [],
                    }
                agent_outputs[node_id] = parsed
                memory.setdefault(node_id, []).append(parsed)
                await _log(
                    log,
                    f"SUCCESS: {node_id} epoch={epoch} "
                    f"sol={str(parsed.get('proposed_solution', ''))[:80]}…",
                )

        # Epoch map / synthesis
        reflections = []
        for i in range(layers):
            for j in range(width):
                nid = f"agent_{i}_{j}"
                hist = memory.get(nid, [])
                if hist:
                    last = hist[-1]
                    reflections.append(
                        f"### {nid} ({agent_personas[nid].get('name', '')})\n"
                        f"{last.get('proposed_solution', '')}\n"
                        f"Reasoning: {last.get('reasoning', '')}\n"
                        f"Falsifiers: {last.get('falsifiers', '')}"
                    )
        agent_reflections = "\n\n".join(reflections)

        is_final = epoch >= epochs - 1
        if not is_final:
            await _log(log, f"LOG: [QNN STEP 4B] Epoch map (epoch {epoch})...")
            epoch_map = await get_brainstorming_epoch_map_chain(synth).ainvoke(
                {
                    "original_request": user_prompt,
                    "current_problem": current_problem,
                    "agent_solutions": agent_reflections[:80000],
                }
            )
            epoch_maps.append(epoch_map)
            previous_solution = epoch_map

            # Mirror Descent
            await _log(log, "LOG: [QNN STEP 4C] Mirror Descent (persona evolution)...")
            md = get_brainstorming_mirror_descent_chain(llm, lr)
            for nid, persona in list(agent_personas.items()):
                try:
                    out = agent_outputs.get(nid, {})
                    new_prompt = await md.ainvoke(
                        {
                            "current_prompt": persona.get("system_prompt", ""),
                            "last_output": json.dumps(out)[:8000],
                        }
                    )
                    if isinstance(new_prompt, str) and len(new_prompt.strip()) > 40:
                        persona["system_prompt"] = new_prompt.strip()
                        li, wi = map(int, nid.split("_")[1:])
                        all_layers_prompts[li][wi] = persona["system_prompt"]
                except Exception as me:
                    await _log(log, f"WARNING: Mirror Descent failed for {nid}: {me}")

            # Reframe
            await _log(log, "LOG: [QNN STEP 4D] Reframe thinking challenge...")
            try:
                reframed = await get_brainstorming_reframer_chain(llm).ainvoke(
                    {
                        "original_request": user_prompt,
                        "current_problem": current_problem,
                        "final_solution": previous_solution[:12000],
                        "prior_conversation": chat_history_str[:8000],
                    }
                )
                data = clean_and_parse_json(reframed)
                if isinstance(data, dict) and data.get("new_problem"):
                    current_problem = data["new_problem"]
                elif isinstance(reframed, str) and reframed.strip():
                    current_problem = reframed.strip()
                await _log(log, f"LOG: [QNN STEP 4D] New challenge: {current_problem[:160]}…")
            except Exception as re:
                await _log(log, f"WARNING: reframe failed: {re}")
        else:
            await _log(log, "LOG: [QNN STEP 5] Final Solution-Space Report...")
            draft = await get_brainstorming_synthesis_chain(synth).ainvoke(
                {
                    "original_request": user_prompt,
                    "prior_conversation": chat_history_str[:8000],
                    "document_context": (document_context or "")[:20000],
                    "agent_solutions": agent_reflections[:100000],
                }
            )
            polished = await get_brainstorming_polisher_chain(synth).ainvoke(
                {
                    "initial_synthesis": draft,
                    "original_request": user_prompt,
                }
            )
            report = polished if isinstance(polished, str) and polished.strip() else draft
            final = {
                "mode": "brainstorm",
                "proposed_solution": report,
                "reasoning": "QNN Solution-Space Report complete.",
                "topology": topology,
                "epoch": epoch,
            }
            result = QNNResult(
                mode="brainstorm",
                proposed_solution=report,
                reasoning=final["reasoning"],
                topology=topology,
                seed_pool=all_seed_words,
                column_guiding_words=column_guiding_words,
                agent_personas=agent_personas,
                attention_edges=attention_edges,
                epoch_maps=epoch_maps,
                final_solution=final,
                params=p,
            )
            await _log(log, "SUCCESS: [QNN] Solution-Space Report complete.")
            return result.to_dict()

    # Fallback if epochs==0 somehow
    return QNNResult(
        proposed_solution="QNN completed without a final report.",
        topology=topology,
        params=p,
    ).to_dict()
