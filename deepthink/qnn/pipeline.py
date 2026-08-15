"""
QNN pipeline entrypoint — params → layered epochs → Solution-Space Report.

Model-agnostic: any LangChain-compatible chat model (OpenRouter, llama.cpp,
mocks, etc.). Used by the open-deepthink Brainstorm UI *and* the portable
``/qnn`` skill runner (``skills/qnn/run_qnn.py``).
"""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from deepthink.chains.brainstorm_chains import (
    get_brainstorming_epoch_map_chain,
    get_brainstorming_mirror_descent_chain,
    get_brainstorming_polisher_chain,
    get_brainstorming_reframer_chain,
    get_brainstorming_seed_chain,
    get_brainstorming_spanner_chain,
    get_brainstorming_synthesis_chain,
    get_complexity_estimator_chain,
    get_problem_summarizer_chain,
)
from deepthink.self_attention import compute_self_attention
from deepthink.utils import clean_and_parse_json

LogFn = Callable[[str], Any] | None


def default_qnn_params() -> dict[str, Any]:
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
    topology: dict[str, Any] = field(default_factory=dict)
    seed_pool: list[str] = field(default_factory=list)
    column_guiding_words: list[str] = field(default_factory=list)
    agent_personas: dict[str, Any] = field(default_factory=dict)
    attention_edges: dict[str, Any] = field(default_factory=dict)
    epoch_maps: list[str] = field(default_factory=list)
    final_solution: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


async def _log(log: LogFn, msg: str) -> None:
    if log is None:
        return
    result = log(msg)
    if hasattr(result, "__await__"):
        await result


def _clamp_topology(layers: int, width: int, epochs: int, *, manual: bool = False) -> tuple:
    """Bound topology. Auto mode keeps a 24-agent laptop cap; manual does not."""
    from deepthink.cost import AUTO_AGENT_CAP

    layers = max(1, min(100, int(layers)))
    width = max(1, min(100, int(width)))
    epochs = max(1, min(20, int(epochs)))
    if not manual and layers * width > AUTO_AGENT_CAP:
        width = max(1, AUTO_AGENT_CAP // layers)
    return layers, width, epochs


def clamp_qnn_topology(layers: int, width: int, epochs: int, *, manual: bool = False) -> tuple:
    """Public alias of the topology clamp used by the pipeline."""
    return _clamp_topology(layers, width, epochs, manual=manual)


def _persist_session(session_store, session_id: str, payload: dict[str, Any]) -> None:
    if session_store is None or not session_id or session_id not in session_store:
        return
    session_store[session_id].update(payload)


async def _span_persona(spanner, user_prompt: str, guiding_words: str, i: int, j: int, brief: str):
    raw = await spanner.ainvoke(
        {
            "problem": user_prompt,
            "guiding_words": guiding_words,
            "layer_index": i,
            "node_index": j,
            "document_context": brief,
        }
    )
    persona = clean_and_parse_json(raw) or {}
    if not isinstance(persona, dict):
        persona = {}
    system_prompt = persona.get("system_prompt") or (
        f"You are a QNN expert spanned from: {guiding_words}. "
        f"Layer {i} ({'diverge' if i == 0 else 'converge'}). Map strategies with falsifiers."
    )
    persona.setdefault("name", f"Agent_{i}_{j}")
    persona.setdefault("specialty", "Word-vector specialist")
    persona.setdefault("guiding_words", guiding_words)
    persona["system_prompt"] = system_prompt
    return f"agent_{i}_{j}", persona, system_prompt


async def _run_agent_cell(
    agent_chain,
    *,
    node_id: str,
    layer_index: int,
    persona: dict[str, Any],
    brief: str,
    user_prompt: str,
    current_problem: str,
    epoch: int,
    prev_layer_outputs: list[Any],
    memory: dict[str, list[Any]],
    agent_outputs_snapshot: dict[str, Any],
    all_layers_prompts: list[list[str]],
    agent_personas: dict[str, Any],
    enable_attn: bool,
    top_k: int,
    log: LogFn,
) -> tuple:
    agent_prompt = (
        f"YOU ARE {str(persona.get('name', 'Expert')).upper()}, "
        f"A {str(persona.get('specialty', 'Specialist')).upper()}.\n\n"
        f"{persona.get('system_prompt', '')}"
    )

    attention_block = ""
    edge_dicts: list[dict] = []
    if enable_attn and top_k > 0:
        state_snap = {
            "epoch": epoch,
            "all_layers_prompts": all_layers_prompts,
            "agent_personas": agent_personas,
            "agent_outputs": agent_outputs_snapshot,
            "memory": memory,
        }
        try:
            edges, attention_block = compute_self_attention(state_snap, node_id, top_k=top_k)
            if edges:
                edge_dicts = [e.to_dict() for e in edges]
                await _log(
                    log,
                    f"LOG: [QNN ATTEND] {node_id} → "
                    + ", ".join(f"{e.to_id}({e.strength})" for e in edges),
                )
        except Exception as ae:
            await _log(log, f"WARNING: [QNN ATTEND] {node_id}: {ae}")

    if layer_index == 0:
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

## Layer {layer_index} Role
Convergent / critical. Critique or combine upstream. Cite agent_id. No production patches.

## Upstream Layer Outputs (graph neighbors)
{json.dumps(prev_layer_outputs, indent=2)}

{attention_block}
"""

    mem_str = "\n".join(f"- {json.dumps(m)}" for m in memory.get(node_id, [])[-5:])
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
    return node_id, parsed, edge_dicts


async def run_qnn_pipeline(
    llm,
    user_prompt: str,
    params: dict[str, Any] | None = None,
    *,
    synthesis_llm=None,
    document_context: str = "",
    chat_history: list[dict] | None = None,
    log: LogFn = None,
    session_id: str = "",
    session_store: dict | None = None,
) -> dict[str, Any]:
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
    session_store : dict, optional
        Optional ``{session_id: state}`` map updated in place (web UI).

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
    try:
        from deepthink.cost import estimate_qnn_cost

        est = estimate_qnn_cost(
            layers=int(p.get("manual_layers", 3) or 3),
            width=int(p.get("manual_width", 3) or 3),
            epochs=int(p.get("num_epochs", 2) or 2),
            qnn_mode=str(p.get("qnn_mode", "auto")),
            has_document_context=bool(document_context),
            enable_self_attention=bool(p.get("enable_self_attention", True)),
        )
        await _log(log, f"LOG: [COST] {est.summary_line()}")
    except Exception:
        pass

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
    manual = qnn_mode == "manual"

    if manual:
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

    layers, width, epochs = _clamp_topology(layers, width, epochs, manual=manual)
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
        {w.strip() for w in seeds_str.replace(",", " ").split() if w.strip() and len(w.strip()) > 1}
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

    column_guiding_words: list[str] = []
    for j in range(width):
        sample = (
            random.sample(all_seed_words, V) if len(all_seed_words) >= V else list(all_seed_words)
        )
        column_guiding_words.append(" ".join(sample))
        await _log(log, f"LOG: [QNN STEP 2] Column {j} guiding_words: {column_guiding_words[-1]}")

    # ── Step 3: Span personas (parallel per layer) ─────────────────
    await _log(log, f"LOG: [QNN STEP 3] Spanning {layers}×{width} personas...")
    spanner = get_brainstorming_spanner_chain(llm)
    agent_personas: dict[str, Any] = {}
    all_layers_prompts: list[list[str]] = []

    for i in range(layers):
        spanned = await asyncio.gather(
            *[
                _span_persona(spanner, user_prompt, column_guiding_words[j], i, j, brief)
                for j in range(width)
            ]
        )
        layer_prompts: list[str] = []
        for node_id, persona, system_prompt in spanned:
            agent_personas[node_id] = persona
            layer_prompts.append(system_prompt)
            await _log(
                log,
                f"LOG: [QNN STEP 3] {node_id} → {persona.get('name')} / {persona.get('specialty')}",
            )
        all_layers_prompts.append(layer_prompts)

    # ── Step 4: Epoch loop ─────────────────────────────────────────
    memory: dict[str, list[Any]] = {nid: [] for nid in agent_personas}
    attention_edges: dict[str, Any] = {}
    epoch_maps: list[str] = []
    current_problem = user_prompt
    previous_solution = ""

    agent_chain = ChatPromptTemplate.from_template("{input}") | llm | StrOutputParser()

    def _build_result(report: str, epoch: int, reasoning: str) -> dict[str, Any]:
        final = {
            "mode": "brainstorm",
            "proposed_solution": report,
            "reasoning": reasoning,
            "topology": topology,
            "epoch": epoch,
        }
        return QNNResult(
            mode="brainstorm",
            proposed_solution=report,
            reasoning=reasoning,
            topology=topology,
            seed_pool=all_seed_words,
            column_guiding_words=column_guiding_words,
            agent_personas=agent_personas,
            attention_edges=attention_edges,
            epoch_maps=epoch_maps,
            final_solution=final,
            params=p,
        ).to_dict()

    for epoch in range(epochs):
        await _log(log, f"--- [QNN STEP 4] Epoch {epoch}/{epochs - 1} forward ---")
        agent_outputs: dict[str, Any] = {}

        for i in range(layers):
            prev_layer_outputs: list[Any] = []
            if i > 0:
                for k in range(width):
                    prev_id = f"agent_{i - 1}_{k}"
                    if prev_id in agent_outputs:
                        up = agent_outputs[prev_id]
                        if isinstance(up, dict):
                            prev_layer_outputs.append({"agent_id": prev_id, **up})
                        else:
                            prev_layer_outputs.append({"agent_id": prev_id, "output": up})

            snapshot = dict(agent_outputs)
            layer_results = await asyncio.gather(
                *[
                    _run_agent_cell(
                        agent_chain,
                        node_id=f"agent_{i}_{j}",
                        layer_index=i,
                        persona=agent_personas[f"agent_{i}_{j}"],
                        brief=brief,
                        user_prompt=user_prompt,
                        current_problem=current_problem,
                        epoch=epoch,
                        prev_layer_outputs=prev_layer_outputs,
                        memory=memory,
                        agent_outputs_snapshot=snapshot,
                        all_layers_prompts=all_layers_prompts,
                        agent_personas=agent_personas,
                        enable_attn=enable_attn,
                        top_k=top_k,
                        log=log,
                    )
                    for j in range(width)
                ]
            )
            for node_id, parsed, edge_dicts in layer_results:
                agent_outputs[node_id] = parsed
                memory.setdefault(node_id, []).append(parsed)
                if edge_dicts:
                    attention_edges[node_id] = edge_dicts
                await _log(
                    log,
                    f"SUCCESS: {node_id} epoch={epoch} "
                    f"sol={str(parsed.get('proposed_solution', ''))[:80]}…",
                )

        _persist_session(
            session_store,
            session_id,
            {
                "agent_outputs": dict(agent_outputs),
                "memory": {k: list(v) for k, v in memory.items()},
                "attention_edges": dict(attention_edges),
                "agent_personas": agent_personas,
                "all_layers_prompts": all_layers_prompts,
                "epoch": epoch,
                "current_problem": current_problem,
            },
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

            # Mirror Descent — evolve every persona in parallel
            await _log(log, "LOG: [QNN STEP 4C] Mirror Descent (persona evolution)...")
            md = get_brainstorming_mirror_descent_chain(llm, lr)

            async def _evolve(nid: str, persona: dict):
                try:
                    out = agent_outputs.get(nid, {})
                    new_prompt = await md.ainvoke(
                        {
                            "current_prompt": persona.get("system_prompt", ""),
                            "last_output": json.dumps(out)[:8000],
                        }
                    )
                    if isinstance(new_prompt, str) and len(new_prompt.strip()) > 40:
                        return nid, new_prompt.strip(), None
                    return nid, None, None
                except Exception as me:
                    return nid, None, me

            evolved = await asyncio.gather(
                *[_evolve(nid, persona) for nid, persona in list(agent_personas.items())]
            )
            for nid, new_prompt, err in evolved:
                if err is not None:
                    await _log(log, f"WARNING: Mirror Descent failed for {nid}: {err}")
                    continue
                if new_prompt:
                    agent_personas[nid]["system_prompt"] = new_prompt
                    li, wi = map(int, nid.split("_")[1:])
                    all_layers_prompts[li][wi] = new_prompt

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
            except Exception as re_err:
                await _log(log, f"WARNING: reframe failed: {re_err}")
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
            result = _build_result(report, epoch, "QNN Solution-Space Report complete.")
            _persist_session(
                session_store,
                session_id,
                {
                    "final_solution": result["final_solution"],
                    "agent_personas": agent_personas,
                    "attention_edges": attention_edges,
                    "all_layers_prompts": all_layers_prompts,
                    "column_guiding_words": column_guiding_words,
                    "epoch_maps": epoch_maps,
                },
            )
            await _log(log, f"FINAL_ANSWER: {json.dumps(result['final_solution'])}")
            await _log(log, "SUCCESS: [QNN] Solution-Space Report complete.")
            return result

    # Fallback if epochs==0 somehow
    fallback_result = QNNResult(
        proposed_solution="QNN completed without a final report.",
        topology=topology,
        params=p,
    ).to_dict()
    _persist_session(
        session_store,
        session_id,
        {"final_solution": fallback_result.get("final_solution") or fallback_result},
    )
    return fallback_result
