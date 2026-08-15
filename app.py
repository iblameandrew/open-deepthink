import asyncio
import io
import json
import random
import re
import time
import traceback
import uuid
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from typing import Any

import fitz  # PyMuPDF for PDF text extraction
import names
import uvicorn
from deepthink.chains import (
    get_attribute_and_hard_request_generator_chain,
    get_brainstorming_agent_chain,
    get_brainstorming_epoch_map_chain,
    get_brainstorming_mirror_descent_chain,
    get_brainstorming_polisher_chain,
    get_brainstorming_reframer_chain,
    get_brainstorming_seed_chain,
    get_brainstorming_spanner_chain,
    get_brainstorming_synthesis_chain,
    get_code_detector_chain,
    get_code_synthesis_chain,
    get_complexity_estimator_chain,
    get_dense_spanner_chain,
    get_expert_reflection_chain,
    get_input_spanner_chain,
    get_interrogator_chain,
    get_memory_summarizer_chain,
    get_module_card_chain,
    get_opinion_synthesizer_chain,
    get_paper_formatter_chain,
    get_perplexity_heuristic_chain,
    get_problem_decomposition_chain,
    get_problem_reframer_chain,
    get_problem_summarizer_chain,
    get_rag_chat_chain,
    get_seed_generation_chain,
    get_synthesis_chain,
)
from deepthink.cost import estimate_qdad_cost, estimate_qnn_cost
from deepthink.knowledge_distillation import DistillationGraph
from deepthink.mocks import CoderMockLLM, DistillationMockLLM, MockLLM
from deepthink.models import ChatLlamaCpp
from deepthink.qdad import run_qdad_pipeline
from deepthink.qnn import run_qnn_pipeline
from deepthink.rag import RAPTOR, RAPTORRetriever
from deepthink.runtime.bus import set_log_queue
from deepthink.runtime.nodes import (
    create_agent_node,
    create_archive_epoch_outputs_node,
    create_code_execution_node,
    create_final_harvest_node,
    create_metrics_node,
    create_reframe_and_decompose_node,
    create_synthesis_node,
    create_update_agent_prompts_node,
    create_update_rag_index_node,
)
from deepthink.self_attention import compute_self_attention
from deepthink.sessions import SessionStore
from deepthink.state import GraphState
from deepthink.utils import clean_and_parse_json, execute_code_in_sandbox
from dotenv import load_dotenv
from fastapi import Body, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# langchain_community is deprecated / being sunset (see warning on import).
# We still depend on it for the FAISS vectorstore integration (widely used pattern).
# Migration path: https://github.com/langchain-ai/langchain-community/issues/674
# For now we keep it; if FAISS support moves to langchain-faiss we can switch later.
from langchain_community.vectorstores import FAISS
from langchain_core.callbacks import (
    AsyncCallbackHandler,
    BaseCallbackHandler,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.outputs import LLMResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from sklearn.cluster import KMeans
from sse_starlette.sse import EventSourceResponse


class TokenUsageTracker(AsyncCallbackHandler):
    def __init__(self, log_stream):
        self.log_stream = log_stream
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        try:
            # Aggregate usage from all generations
            if response.llm_output and "token_usage" in response.llm_output:
                usage = response.llm_output["token_usage"]
                self.total_tokens += usage.get("total_tokens", 0)
                self.prompt_tokens += usage.get("prompt_tokens", 0)
                self.completion_tokens += usage.get("completion_tokens", 0)

            # Check for standard usage_metadata in generations
            if hasattr(response, "generations"):
                for generation_list in response.generations:
                    for generation in generation_list:
                        if hasattr(generation, "message") and hasattr(
                            generation.message, "usage_metadata"
                        ):
                            usage = generation.message.usage_metadata
                            self.total_tokens += usage.get("input_tokens", 0) + usage.get(
                                "output_tokens", 0
                            )
                            self.prompt_tokens += usage.get("input_tokens", 0)
                            self.completion_tokens += usage.get("output_tokens", 0)

            # Emit Update
            data = {
                "total": self.total_tokens,
                "prompt": self.prompt_tokens,
                "completion": self.completion_tokens,
            }
            await self.log_stream.put(f"TOKEN_USAGE: {json.dumps(data)}")

        except Exception as e:
            await self.log_stream.put(f"WARNING: Token tracking error: {e}")


load_dotenv()
# Only OpenRouter and LlamaCpp server providers are supported.
# API keys: UI params (per-request) take precedence; otherwise Settings /
# environment (OPENROUTER_API_KEY). Never hard-code secrets.

from deepthink.config import get_settings  # noqa: E402

_settings = get_settings()

app = FastAPI(title="open-deepthink", version=__import__("deepthink").__version__)
app.mount("/js", StaticFiles(directory="js"), name="js")
app.mount("/css", StaticFiles(directory="css"), name="css")
app.mount("/static", StaticFiles(directory="static"), name="static")


log_stream = asyncio.Queue()
connected_log_clients = set()


async def broadcast_log(message: str):
    """Broadcasts a log message to all connected SSE clients."""
    if connected_log_clients:
        # Create a list of tasks for parallel putting
        tasks = [asyncio.create_task(q.put(message)) for q in connected_log_clients]
        if tasks:
            await asyncio.wait(tasks, timeout=0.1)
    # Also put in the main queue for any fallback/legacy listeners (optional)
    # await log_stream.put(message)


sessions = SessionStore()
final_reports = {}

# --- Custom Embedding Classes ---


active_distillation_graph = None


set_log_queue(log_stream)


@app.get("/")
def get_index():
    with open("index.html", encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), status_code=200)


@app.post("/run_inference_from_state")
async def run_inference_from_state(payload: dict = Body(...)):
    await log_stream.put(
        "--- [INFERENCE-ONLY] Received request to run inference from imported state. ---"
    )
    try:
        imported_state = payload.get("imported_state")
        user_prompt = payload.get("prompt")
        params = imported_state.get("params", {})

        if not imported_state or not user_prompt:
            return JSONResponse(
                content={"error": "Invalid payload. 'imported_state' and 'prompt' are required."},
                status_code=400,
            )

        is_debug = (
            params.get("coder_debug_mode") == "true"
            or params.get("debug_mode") == "true"
            or params.get("coder_debug_mode") is True
            or params.get("debug_mode") is True
        )
        if is_debug:
            llm = CoderMockLLM()
        else:
            return JSONResponse(
                content={"error": "No valid LLM provider configured."}, status_code=400
            )

        imported_state["original_request"] = user_prompt
        imported_state["current_problem"] = user_prompt
        imported_state["agent_outputs"] = {}

        workflow = StateGraph(GraphState)
        all_layers_prompts = imported_state["all_layers_prompts"]
        cot_trace_depth = len(all_layers_prompts)

        agent_chain = ChatPromptTemplate.from_template("{input}") | llm | StrOutputParser()

        async def inference_agent_logic(state: GraphState, node_id: str):
            await log_stream.put(f"--- [INFERENCE] Invoking Agent: {node_id} ---")
            layer_index_str, agent_index_str = node_id.split("_")[1:]
            layer_index = int(layer_index_str)
            agent_prompt = state["all_layers_prompts"][layer_index][int(agent_index_str)]

            if layer_index == 0:
                input_data = state["original_request"]
            else:
                prev_layer_index = layer_index - 1
                num_agents_prev_layer = len(state["all_layers_prompts"][prev_layer_index])
                prev_layer_outputs = [
                    state["agent_outputs"].get(f"agent_{prev_layer_index}_{k}", {})
                    for k in range(num_agents_prev_layer)
                ]
                input_data = json.dumps(prev_layer_outputs)

            full_prompt = f"{agent_prompt}\n\nInput Data to Process:\n---\n{input_data}\n---\nYour JSON formatted response:"
            response_str = await agent_chain.ainvoke({"input": full_prompt})

            try:
                response_json = clean_and_parse_json(response_str)
            except Exception:
                response_json = {
                    "proposed_solution": response_str,
                    "reasoning": "Inference output could not be parsed as JSON.",
                }

            current_outputs = state.get("agent_outputs", {}).copy()
            current_outputs[node_id] = response_json
            return {"agent_outputs": current_outputs}

        def create_inference_node_function(node_id_for_closure: str):
            async def node_function(state: GraphState):
                return await inference_agent_logic(state, node_id_for_closure)

            return node_function

        for i, layer_prompts in enumerate(all_layers_prompts):
            for j, _ in enumerate(layer_prompts):
                node_id = f"agent_{i}_{j}"
                workflow.add_node(node_id, create_inference_node_function(node_id))

        workflow.add_node("synthesis", create_synthesis_node(llm))

        first_layer_nodes = [f"agent_0_{j}" for j in range(len(all_layers_prompts[0]))]
        workflow.set_entry_point(first_layer_nodes[0])
        if len(first_layer_nodes) > 1:
            for node in first_layer_nodes[1:]:
                workflow.add_edge(first_layer_nodes[0], node)

        for i in range(cot_trace_depth - 1):
            for current_node in [f"agent_{i}_{j}" for j in range(len(all_layers_prompts[i]))]:
                for next_node in [
                    f"agent_{i + 1}_{k}" for k in range(len(all_layers_prompts[i + 1]))
                ]:
                    workflow.add_edge(current_node, next_node)

        for node in [
            f"agent_{cot_trace_depth - 1}_{j}"
            for j in range(len(all_layers_prompts[cot_trace_depth - 1]))
        ]:
            workflow.add_edge(node, "synthesis")

        workflow.add_edge("synthesis", END)
        graph = workflow.compile()

        ascii_diagram = graph.get_graph().draw_ascii()
        await log_stream.put(ascii_diagram)

        final_result_node = None
        async for output in graph.astream(imported_state):
            if "synthesis" in output:
                final_result_node = output["synthesis"]

        await log_stream.put("--- [INFERENCE-ONLY] Run complete. ---")

        return JSONResponse(
            content={
                "message": "Inference complete.",
                "code_solution": final_result_node.get("final_solution", {}).get(
                    "proposed_solution", "No solution generated."
                ),
                "reasoning": final_result_node.get("final_solution", {}).get(
                    "reasoning", "No reasoning provided."
                ),
                "is_inference": True,
            }
        )

    except Exception as e:
        error_message = f"An error occurred during inference: {e}"
        await log_stream.put(error_message)
        await log_stream.put(traceback.format_exc())
        return JSONResponse(
            content={"message": error_message, "traceback": traceback.format_exc()},
            status_code=500,
        )


async def run_qnn_background(
    llm,
    synthesis_llm,
    params,
    user_prompt: str,
    session_id: str,
    document_context: str = "",
    chat_history=None,
):
    """
    Brainstorm background runner.

    Delegates to deepthink.qnn.run_qnn_pipeline (same engine as the /qnn skill).
    """

    async def _log(msg: str):
        await log_stream.put(msg)

    await run_qnn_pipeline(
        llm=llm,
        user_prompt=user_prompt or "",
        params=params or {},
        synthesis_llm=synthesis_llm or llm,
        document_context=document_context or "",
        chat_history=chat_history or [],
        log=_log,
        session_id=session_id,
        session_store=sessions,
    )


async def run_qdad_background(
    llm,
    synthesis_llm,
    params,
    user_prompt: str,
    session_id: str,
    document_context: str = "",
    chat_history=None,
    is_debug: bool = False,
    provider: str = "openrouter",
    api_key: str = "",
    default_agent_model: str = "",
    agent_model_list=None,
    llamacpp_url: str = "",
    llamacpp_api_key: str = "",
    token_tracker=None,
):
    """
    App Slot Machine background runner.

    Delegates to deepthink.qdad.run_qdad_pipeline (LangGraph):
      foundation → grid → noise → denoise* → synthesize
    """

    async def _log(msg: str):
        await log_stream.put(msg)

    await run_qdad_pipeline(
        llm=llm,
        synthesis_llm=synthesis_llm or llm,
        params=params or {},
        user_prompt=user_prompt or "",
        session_id=session_id,
        document_context=document_context or "",
        chat_history=chat_history or [],
        log=_log,
        session_store=sessions,
    )


@app.post("/build_and_run_graph")
async def build_and_run_graph(payload: dict = Body(...)):
    llm = None
    params = payload.get("params", {})
    mode = payload.get("mode", "brainstorm")

    # Initialize Token Tracker
    token_tracker = TokenUsageTracker(log_stream)

    try:
        # Determine Provider - only OpenRouter and LlamaCpp supported
        # Settings / env provide defaults; request params override (backward compatible).
        cfg = get_settings()
        provider = params.get("provider", cfg.default_provider) or cfg.default_provider
        api_key = params.get("api_key", "") or cfg.resolved_api_key() or ""

        # Hoist common config and model choices for per-agent / synthesis support (visible in all branches)
        openrouter_model = params.get("openrouter_model", cfg.openrouter_model)
        llamacpp_url = params.get("llamacpp_url", cfg.llamacpp_base_url)
        llamacpp_model = params.get("llamacpp_model", cfg.llamacpp_model)
        # normalize llamacpp url early
        llamacpp_url = cfg.normalize_llamacpp_url(llamacpp_url)
        llamacpp_api_key = (
            params.get("llamacpp_api_key", cfg.llamacpp_api_key) or cfg.llamacpp_api_key
        )

        default_agent_model = openrouter_model if provider == "openrouter" else llamacpp_model

        synthesis_model = params.get("synthesis_model", "").strip()
        agent_models_raw = params.get("agent_models", "").strip()
        agent_model_list = (
            [m.strip() for m in agent_models_raw.split(",") if m.strip()]
            if agent_models_raw
            else []
        )

        # Debug / simulation mode: use Mock LLM immediately — no API key or network required.
        is_debug = (
            params.get("coder_debug_mode") == "true"
            or params.get("debug_mode") == "true"
            or params.get("coder_debug_mode") is True
            or params.get("debug_mode") is True
        )

        if is_debug:
            if mode == "app_slot_machine":
                await log_stream.put(
                    "--- 🎰 APP SLOT MACHINE DEBUG MODE ENABLED "
                    "(CoderMockLLM / QDAD — no API cost) 🎰 ---"
                )
            else:
                await log_stream.put(
                    "--- 💻 CODER DEBUG MODE ENABLED (Mock LLM — no API cost) 💻 ---"
                )
            llm = CoderMockLLM()
            synthesis_llm = CoderMockLLM()
            await log_stream.put("--- ⚠️ Debug Mode: Embeddings skipped. RAG will be skipped. ---")

        elif provider == "openrouter":
            if not api_key:
                return JSONResponse(
                    content={"message": "OpenRouter API Key required"}, status_code=400
                )
            # use hoisted openrouter_model as default for agents
            default_agent_model = openrouter_model
            llm = ChatOpenAI(
                model=default_agent_model,
                openai_api_key=api_key,
                openai_api_base=cfg.openrouter_base_url,
                temperature=cfg.temperature,
                callbacks=[token_tracker],
            )
            # Use OpenAIEmbeddings with OpenRouter base URL (works for many OpenRouter embedding models)
            try:
                OpenAIEmbeddings(
                    model=cfg.openrouter_embedding_model,
                    openai_api_key=api_key,
                    openai_api_base=cfg.openrouter_base_url,
                    check_embedding_ctx_length=False,
                )
                await log_stream.put(
                    f"--- Initializing Main Agent LLM: OpenRouter ({default_agent_model}) & Embeddings ---"
                )
            except Exception as e:
                await log_stream.put(f"WARNING: Failed to initialize OpenRouter embeddings: {e}")

            # Create synthesis LLM if user specified a different model for synthesis
            synthesis_llm = llm
            if synthesis_model and synthesis_model != default_agent_model:
                try:
                    synthesis_llm = ChatOpenAI(
                        model=synthesis_model,
                        openai_api_key=api_key,
                        openai_api_base=cfg.openrouter_base_url,
                        temperature=cfg.temperature,
                        callbacks=[token_tracker],
                    )
                    await log_stream.put(
                        f"--- Using separate SYNTHESIS model: {synthesis_model} ---"
                    )
                except Exception as e:
                    await log_stream.put(
                        f"WARNING: Could not init separate synthesis LLM, falling back: {e}"
                    )

        elif provider == "llamacpp":
            # use hoisted + normalized llamacpp_url and model
            default_agent_model = llamacpp_model
            llm = ChatLlamaCpp(
                base_url=llamacpp_url,
                api_key=llamacpp_api_key,
                model=default_agent_model,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
            )
            # Use OpenAIEmbeddings pointing to local server (assumes embedding capable server)
            llamacpp_emb_url = cfg.normalize_llamacpp_url(
                params.get("llamacpp_embedding_url", cfg.llamacpp_embedding_url)
            )
            try:
                OpenAIEmbeddings(
                    model=cfg.llamacpp_embedding_model,
                    openai_api_base=llamacpp_emb_url,
                    openai_api_key=llamacpp_api_key or "sk-no-key-required",
                    check_embedding_ctx_length=False,
                )
                await log_stream.put(
                    f"--- Initializing Main Agent LLM: LlamaCpp & Embeddings ({llamacpp_emb_url}) ---"
                )
            except Exception as e:
                await log_stream.put(f"WARNING: Failed to initialize LlamaCpp embeddings: {e}")

            synthesis_llm = llm
            if synthesis_model and synthesis_model != default_agent_model:
                try:
                    synthesis_llm = ChatLlamaCpp(
                        base_url=llamacpp_url,
                        api_key=llamacpp_api_key,
                        model=synthesis_model,
                        temperature=cfg.temperature,
                        max_tokens=cfg.max_tokens,
                    )
                    await log_stream.put(
                        f"--- Using separate SYNTHESIS model: {synthesis_model} ---"
                    )
                except Exception as e:
                    await log_stream.put(
                        f"WARNING: Could not init separate synthesis LLM, falling back: {e}"
                    )

        else:
            return JSONResponse(
                content={"message": "Invalid provider. Please select openrouter or llamacpp."},
                status_code=400,
            )

    except Exception as e:
        error_message = f"Failed to initialize LLM: {e}. Please ensure the selected provider is configured correctly."
        await log_stream.put(error_message)
        return JSONResponse(
            content={"message": error_message, "traceback": traceback.format_exc()},
            status_code=500,
        )

    document_context = payload.get("document_context", "")
    user_prompt = params.get("prompt")

    if document_context and mode in ("app_slot_machine", "algorithm"):
        capped_context = document_context[:50000]
        user_prompt = (
            f"{user_prompt}\n\n--- Attached Context ---\n{capped_context}"
            if user_prompt
            else capped_context
        )
        params["prompt"] = user_prompt
        await log_stream.put(f"LOG: Context attached to prompt ({len(capped_context)} characters).")

    await log_stream.put(f"--- Starting Graph Build and Run Process (Mode: {mode}) ---")
    await log_stream.put(f"Parameters: {params}")

    # ── App Slot Machine (QDAD): dedicated qualitative diffusion pipeline ──
    if mode == "app_slot_machine":
        session_id = str(uuid.uuid4())
        sessions[session_id] = {
            "session_id": session_id,
            "mode": "app_slot_machine",
            "original_request": user_prompt,
            "params": params,
            "final_solution": None,
            "qdad_matrices": {},
            "all_rag_documents": [],
            "all_layers_prompts": [],
            "agent_personas": {},
            "agent_outputs": {},
            "memory": {},
            "epoch": 0,
            "max_epochs": 1,
            "raptor_index": None,
        }
        await log_stream.put(f"__session_id__ {session_id}")
        await log_stream.put(
            "__start__ QDAD N×N Feature Grid\n"
            "  Phase 0 Foundation → Phase 1 Grid → Phase 2 Noise\n"
            "  → Phase 3 Denoise → Phase 4 Synthesis"
        )
        qdad_est = estimate_qdad_cost(
            n=int(params.get("grid_size", params.get("n", 3)) or 3),
            denoising_steps=int(params.get("denoising_steps", 2) or 2),
        )
        await log_stream.put(f"LOG: [COST] {qdad_est.summary_line()}")
        asyncio.create_task(
            run_qdad_background(
                llm=llm,
                synthesis_llm=synthesis_llm,
                params=params,
                user_prompt=user_prompt or "",
                session_id=session_id,
                document_context=document_context or "",
                chat_history=payload.get("chat_history", []) or [],
                is_debug=is_debug,
                provider=provider,
                api_key=api_key,
                default_agent_model=default_agent_model,
                agent_model_list=agent_model_list,
                llamacpp_url=llamacpp_url,
                llamacpp_api_key=llamacpp_api_key,
                token_tracker=token_tracker,
            )
        )
        return JSONResponse(
            content={
                "message": "Graph started.",
                "session_id": session_id,
                "cost_estimate": qdad_est.to_dict(),
            }
        )

    # ── Brainstorm (QNN): same library engine as `deepthink qnn` / /qnn skill ──
    if mode == "brainstorm":
        session_id = str(uuid.uuid4())
        sessions[session_id] = {
            "session_id": session_id,
            "mode": "brainstorm",
            "original_request": user_prompt,
            "params": params,
            "final_solution": None,
            "all_rag_documents": [],
            "all_layers_prompts": [],
            "agent_personas": {},
            "agent_outputs": {},
            "memory": {},
            "epoch": 0,
            "max_epochs": int(params.get("num_epochs", 2) or 2),
            "raptor_index": None,
            "attention_edges": {},
        }
        await log_stream.put(f"__session_id__ {session_id}")
        await log_stream.put(
            "__start__ QNN Brainstorm\n"
            "  Brief → Topology → Seeds → Personas\n"
            "  → Epochs (forward / map / Mirror Descent / reframe)\n"
            "  → Solution-Space Report"
        )
        qnn_est = estimate_qnn_cost(
            layers=int(params.get("manual_layers", 3) or 3),
            width=int(params.get("manual_width", 3) or 3),
            epochs=int(params.get("num_epochs", 2) or 2),
            qnn_mode=str(params.get("qnn_mode", "auto")),
            has_document_context=bool(document_context),
        )
        await log_stream.put(f"LOG: [COST] {qnn_est.summary_line()}")
        asyncio.create_task(
            run_qnn_background(
                llm=llm,
                synthesis_llm=synthesis_llm,
                params=params,
                user_prompt=user_prompt or "",
                session_id=session_id,
                document_context=document_context or "",
                chat_history=payload.get("chat_history", []) or [],
            )
        )
        return JSONResponse(
            content={
                "message": "Graph started.",
                "session_id": session_id,
                "cost_estimate": qnn_est.to_dict(),
            }
        )

    return JSONResponse(
        content={
            "message": (f"Unsupported mode '{mode}'. Use 'brainstorm' or 'app_slot_machine'.")
        },
        status_code=400,
    )


@app.get("/export_qnn/{session_id}")
async def export_qnn(session_id: str):
    """
    Exports the current state of a session graph to a JSON file.
    """
    if session_id not in sessions:
        return JSONResponse(content={"error": "Session not found."}, status_code=404)

    state_to_export = sessions[session_id].copy()

    state_to_export.pop("llm", None)
    state_to_export.pop("summarizer_llm", None)
    state_to_export.pop("embeddings_model", None)
    state_to_export.pop("raptor_index", None)

    rag_docs = state_to_export.get("all_rag_documents") or []
    serialized_docs = []
    for document in rag_docs:
        if hasattr(document, "dict"):
            serialized_docs.append(document.dict())
        elif hasattr(document, "model_dump"):
            serialized_docs.append(document.model_dump())
        elif isinstance(document, dict):
            serialized_docs.append(document)
        else:
            serialized_docs.append({"page_content": str(document), "metadata": {}})
    state_to_export["all_rag_documents"] = serialized_docs

    await log_stream.put(
        f"--- [EXPORT] Exporting session {session_id} (mode={state_to_export.get('mode', '?')}) ---"
    )

    return JSONResponse(
        content=state_to_export,
        headers={"Content-Disposition": f"attachment; filename=qnn_state_{session_id}.json"},
    )


@app.post("/import_qnn")
async def import_qnn(file: UploadFile = File(...)):
    """
    Imports a QNN JSON file to initialize a new session.
    """
    try:
        content = await file.read()
        imported_state = json.loads(content)

        session_id = str(uuid.uuid4())
        imported_state["session_id"] = session_id

        rag_docs = imported_state.get("all_rag_documents") or []
        restored = []
        for document in rag_docs:
            if isinstance(document, dict):
                try:
                    restored.append(Document.from_dict(document))
                except Exception:
                    restored.append(
                        Document(
                            page_content=document.get("page_content", ""),
                            metadata=document.get("metadata") or {},
                        )
                    )
            else:
                restored.append(document)
        imported_state["all_rag_documents"] = restored

        sessions[session_id] = imported_state
        await log_stream.put(
            f"--- [IMPORT] Successfully imported QNN file. New Session ID: {session_id} ---"
        )

        return JSONResponse(
            content={
                "message": "QNN file imported successfully.",
                "session_id": session_id,
                "imported_params": imported_state.get("params", {}),
            }
        )
    except Exception as e:
        error_message = f"Failed to import QNN file: {e}"
        await log_stream.put(error_message)
        return JSONResponse(
            content={"message": error_message, "traceback": traceback.format_exc()},
            status_code=500,
        )


@app.post("/upload_documents")
async def upload_documents(files: list[UploadFile] = File(...)):
    """
    Uploads PDF documents and extracts their text content.
    Returns extracted text to be used as context in brainstorm mode.
    """
    MAX_TOTAL_CHARS = 50000  # Limit to prevent token overflow

    extracted_texts = []
    total_chars = 0

    try:
        for file in files:
            if not file.filename.lower().endswith(".pdf"):
                await log_stream.put(f"WARNING: Skipping non-PDF file: {file.filename}")
                continue

            content = await file.read()

            # Use PyMuPDF to extract text
            try:
                pdf_document = fitz.open(stream=content, filetype="pdf")
                file_text = ""

                for page_num in range(len(pdf_document)):
                    page = pdf_document[page_num]
                    file_text += page.get_text()

                pdf_document.close()

                # Truncate if needed
                remaining_chars = MAX_TOTAL_CHARS - total_chars
                if remaining_chars <= 0:
                    await log_stream.put(
                        "WARNING: Character limit reached. Skipping remaining files."
                    )
                    break

                if len(file_text) > remaining_chars:
                    file_text = file_text[:remaining_chars]
                    await log_stream.put(
                        f"WARNING: Truncated {file.filename} to fit character limit."
                    )

                total_chars += len(file_text)
                extracted_texts.append(
                    {
                        "filename": file.filename,
                        "text": file_text,
                        "char_count": len(file_text),
                    }
                )

                await log_stream.put(
                    f"SUCCESS: Extracted {len(file_text)} characters from {file.filename}"
                )

            except Exception as pdf_error:
                await log_stream.put(
                    f"ERROR: Failed to extract text from {file.filename}: {pdf_error}"
                )
                continue

        # Combine all extracted texts
        combined_text = "\n\n---\n\n".join(
            [f"[Document: {doc['filename']}]\n{doc['text']}" for doc in extracted_texts]
        )

        return JSONResponse(
            content={
                "message": f"Successfully extracted text from {len(extracted_texts)} document(s).",
                "documents": extracted_texts,
                "combined_text": combined_text,
                "total_chars": total_chars,
            }
        )

    except Exception as e:
        error_message = f"Failed to process documents: {e}"
        await log_stream.put(error_message)
        return JSONResponse(
            content={"message": error_message, "traceback": traceback.format_exc()},
            status_code=500,
        )


CODE_FILE_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".sh",
    ".bash",
    ".sql",
    ".html",
    ".css",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".md",
    ".txt",
    ".xml",
    ".vue",
    ".svelte",
    ".kt",
    ".swift",
    ".r",
    ".scala",
    ".lua",
    ".pl",
    ".zig",
    ".cs",
    ".m",
    ".mm",
    ".ipynb",
}


def _decode_text_file(content: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="replace")


def _file_extension(filename: str) -> str:
    if filename and "." in filename:
        return "." + filename.rsplit(".", 1)[-1].lower()
    return ""


def _format_code_block(filename: str, text: str, extension: str = "") -> str:
    ext = extension or _file_extension(filename).lstrip(".")
    return f"[Code File: {filename}]\n```{ext}\n{text}\n```"


REPO_IGNORE_DIR_NAMES = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "dist",
    "build",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".tox",
    ".eggs",
    ".next",
    ".nuxt",
    "target",
    "coverage",
    ".idea",
    ".vscode",
    "vendor",
    ".gradle",
    ".svn",
    ".hg",
}

REPO_IGNORE_BASENAMES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "Pipfile.lock",
    "Cargo.lock",
    "go.sum",
}

REPO_ALLOWED_HIDDEN_FILES = {
    ".gitignore",
    ".dockerignore",
    ".env.example",
    ".editorconfig",
}

REPO_PRIORITY_BASENAMES = [
    "readme.md",
    "readme",
    "package.json",
    "pyproject.toml",
    "requirements.txt",
    "setup.py",
    "cargo.toml",
    "go.mod",
    "makefile",
    "dockerfile",
    "app.py",
    "main.py",
    "index.js",
    "index.ts",
    "__init__.py",
]


def _normalize_repo_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("/")


def _repo_path_should_skip(rel_path: str) -> bool:
    normalized = _normalize_repo_path(rel_path)
    if not normalized:
        return True

    parts = normalized.split("/")
    for part in parts[:-1]:
        if part.lower() in REPO_IGNORE_DIR_NAMES:
            return True

    basename = parts[-1]
    basename_lower = basename.lower()
    if basename_lower in REPO_IGNORE_BASENAMES:
        return True

    if basename.startswith("."):
        if basename_lower not in REPO_ALLOWED_HIDDEN_FILES:
            return True

    ext = _file_extension(basename)
    if ext not in CODE_FILE_EXTENSIONS:
        return True

    return False


def _repo_priority_key(rel_path: str) -> tuple:
    normalized = _normalize_repo_path(rel_path)
    parts = normalized.split("/")
    basename = parts[-1].lower()
    depth = len(parts)

    try:
        priority = REPO_PRIORITY_BASENAMES.index(basename)
    except ValueError:
        priority = len(REPO_PRIORITY_BASENAMES)

    return (priority, depth, normalized.lower())


def _extract_repo_name(paths: list[str]) -> str:
    for path in paths:
        normalized = _normalize_repo_path(path)
        if normalized:
            return normalized.split("/")[0]
    return "repository"


@app.post("/upload_code_files")
async def upload_code_files(files: list[UploadFile] = File(...)):
    """
    Uploads source code / text files and returns their contents for use as context.
    """
    MAX_TOTAL_CHARS = 50000

    extracted_files = []
    total_chars = 0

    try:
        for file in files:
            ext = _file_extension(file.filename or "")

            if ext not in CODE_FILE_EXTENSIONS:
                await log_stream.put(
                    f"WARNING: Skipping unsupported code file type: {file.filename}"
                )
                continue

            content = await file.read()
            file_text = _decode_text_file(content)

            remaining_chars = MAX_TOTAL_CHARS - total_chars
            if remaining_chars <= 0:
                await log_stream.put(
                    "WARNING: Character limit reached. Skipping remaining code files."
                )
                break

            if len(file_text) > remaining_chars:
                file_text = file_text[:remaining_chars]
                await log_stream.put(f"WARNING: Truncated {file.filename} to fit character limit.")

            total_chars += len(file_text)
            extracted_files.append(
                {
                    "filename": file.filename,
                    "text": file_text,
                    "char_count": len(file_text),
                    "extension": ext.lstrip("."),
                }
            )

            await log_stream.put(
                f"SUCCESS: Loaded {len(file_text)} characters from code file {file.filename}"
            )

        combined_text = "\n\n---\n\n".join(
            [
                _format_code_block(doc["filename"], doc["text"], doc.get("extension", ""))
                for doc in extracted_files
            ]
        )

        return JSONResponse(
            content={
                "message": f"Successfully loaded {len(extracted_files)} code file(s).",
                "files": extracted_files,
                "combined_text": combined_text,
                "total_chars": total_chars,
            }
        )

    except Exception as e:
        error_message = f"Failed to process code files: {e}"
        await log_stream.put(error_message)
        return JSONResponse(
            content={"message": error_message, "traceback": traceback.format_exc()},
            status_code=500,
        )


@app.post("/upload_repository")
async def upload_repository(
    files: list[UploadFile] = File(...),
    paths: list[str] = Form(default=[]),
):
    """
    Uploads an entire repository folder and returns prioritized source files as context.
    Skips common vendor/build/cache directories and caps total context at 50k chars.
    """
    MAX_TOTAL_CHARS = 50000
    MAX_FILES = 500

    try:
        if not files:
            return JSONResponse(
                content={"message": "No files received from repository upload."},
                status_code=400,
            )

        resolved_paths: list[str] = []
        for index, file in enumerate(files):
            if index < len(paths) and paths[index]:
                resolved_paths.append(_normalize_repo_path(paths[index]))
            else:
                resolved_paths.append(_normalize_repo_path(file.filename or f"file_{index}"))

        repo_name = _extract_repo_name(resolved_paths)

        candidates = []
        skipped_count = 0
        for file, rel_path in zip(files, resolved_paths):
            if _repo_path_should_skip(rel_path):
                skipped_count += 1
                continue

            content = await file.read()
            file_text = _decode_text_file(content)
            ext = _file_extension(rel_path)

            candidates.append(
                {
                    "filename": rel_path,
                    "text": file_text,
                    "char_count": len(file_text),
                    "extension": ext.lstrip("."),
                    "repo_name": repo_name,
                }
            )

        candidates.sort(key=lambda item: _repo_priority_key(item["filename"]))

        extracted_files = []
        total_chars = 0
        truncated_count = 0

        for candidate in candidates:
            if len(extracted_files) >= MAX_FILES:
                skipped_count += 1
                continue

            remaining_chars = MAX_TOTAL_CHARS - total_chars
            if remaining_chars <= 0:
                skipped_count += 1
                continue

            file_text = candidate["text"]
            if len(file_text) > remaining_chars:
                file_text = file_text[:remaining_chars]
                truncated_count += 1
                await log_stream.put(
                    f"WARNING: Truncated repository file {candidate['filename']} to fit character limit."
                )

            total_chars += len(file_text)
            extracted_files.append(
                {
                    "filename": candidate["filename"],
                    "text": file_text,
                    "char_count": len(file_text),
                    "extension": candidate["extension"],
                    "repo_name": repo_name,
                }
            )

            await log_stream.put(
                f"SUCCESS: Loaded {len(file_text)} characters from repository file {candidate['filename']}"
            )

        combined_text = "\n\n---\n\n".join(
            [
                f"[Repository: {doc['repo_name']}/{doc['filename']}]\n```{doc.get('extension', '')}\n{doc['text']}\n```"
                for doc in extracted_files
            ]
        )

        message = (
            f"Successfully loaded {len(extracted_files)} file(s) from repository '{repo_name}'."
        )
        if skipped_count:
            message += (
                f" Skipped {skipped_count} file(s) (ignored paths, unsupported types, or limits)."
            )
        if truncated_count:
            message += f" Truncated {truncated_count} file(s) to fit the character budget."

        return JSONResponse(
            content={
                "message": message,
                "repo_name": repo_name,
                "files": extracted_files,
                "combined_text": combined_text,
                "total_chars": total_chars,
                "included_count": len(extracted_files),
                "skipped_count": skipped_count,
            }
        )

    except Exception as e:
        error_message = f"Failed to process repository: {e}"
        await log_stream.put(error_message)
        return JSONResponse(
            content={"message": error_message, "traceback": traceback.format_exc()},
            status_code=500,
        )


@app.post("/chat")
async def chat_with_index(payload: dict = Body(...)):
    message = payload.get("message")
    session_id = payload.get("session_id")

    await log_stream.put(f"LOG: [CHAT] session_id={session_id}, active_sessions={len(sessions)}")

    if not session_id or session_id not in list(sessions.keys()):
        return JSONResponse(content={"error": "Invalid session ID"}, status_code=404)

    state = sessions[session_id]

    raptor_index = state.get("raptor_index")
    llm = state["llm"]

    if not raptor_index:
        return JSONResponse(
            content={"error": "RAG index not found for this session"}, status_code=500
        )

    async def stream_response():
        try:
            retrieved_docs = await asyncio.to_thread(raptor_index.retrieve, message, k=10)
            context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])

            chat_chain = get_rag_chat_chain(llm)
            full_response = ""
            async for chunk in chat_chain.astream({"context": context, "question": message}):
                content = chunk.content if hasattr(chunk, "content") else chunk
                yield content
                full_response += content

            state["chat_history"].append({"role": "user", "content": message})
            state["chat_history"].append({"role": "ai", "content": full_response})

        except Exception as e:
            await log_stream.put(f"ERROR: Error during chat streaming: {e}")
            yield f"Error: Could not generate response. {e}"

    return StreamingResponse(stream_response(), media_type="text/event-stream")


@app.post("/diagnostic_chat")
async def diagnostic_chat_with_index(payload: dict = Body(...)):
    message = payload.get("message")
    session_id = payload.get("session_id")
    message = payload.get("message")

    await log_stream.put(
        f"LOG: [DIAGNOSTIC_CHAT] session_id={session_id}, active_sessions={len(sessions)}"
    )

    if not session_id or session_id not in list(sessions.keys()):
        return JSONResponse(content={"error": "Invalid session ID"}, status_code=404)

    await log_stream.put("LOG: [DIAGNOSTIC_CHAT] entering diagnostic handler")

    state = sessions[session_id]
    raptor_index = state.get("raptor_index")

    if not raptor_index:

        async def stream_error():
            yield "The RAG index for this session is not yet available. Please wait for the first epoch to complete."

        return StreamingResponse(stream_error(), media_type="text/event-stream")

    async def stream_response():
        try:
            query = message.strip()[5:]
            await log_stream.put(f"--- [DIAGNOSTIC] Raw RAG query received: '{query}' ---")

            retrieved_docs = await asyncio.to_thread(raptor_index.retrieve, query, k=10)

            if not retrieved_docs:
                yield "No relevant documents found in the RAPTOR index for that query."
                return

            yield "--- Top Relevant Documents (Raw Retrieval) ---\n\n"
            for i, doc in enumerate(retrieved_docs):
                content_preview = doc.page_content.replace("\n", " ").strip()
                metadata_str = json.dumps(doc.metadata)
                response_chunk = (
                    f"DOCUMENT #{i + 1}\n"
                    f"-----------------\n"
                    f"METADATA: {metadata_str}\n"
                    f"CONTENT: {content_preview}...\n\n"
                )
                yield response_chunk

        except Exception as e:
            await log_stream.put(f"ERROR: Error during diagnostic chat streaming: {e}")
            yield f"Error: Could not generate response. {e}"

    return StreamingResponse(stream_response(), media_type="text/event-stream")


@app.post("/harvest")
async def harvest_session(payload: dict = Body(...)):
    if not payload.get("session_id") or payload.get("session_id") not in list(sessions.keys()):
        return JSONResponse(content={"error": "Invalid request"}, status_code=404)

    session = sessions.get(payload.get("session_id"))

    if not session:
        return JSONResponse(content={"error": "Invalid request"}, status_code=404)

    try:
        await log_stream.put("--- [HARVEST] Initiating Final Harvest Process ---")
        state = session
        chat_history = session["chat_history"]
        llm = session["llm"]
        summarizer_llm = session["summarizer_llm"]
        embeddings_model = session["embeddings_model"]
        params = session["params"]

        chat_docs = []
        if chat_history:
            for i, turn in enumerate(chat_history):
                if turn["role"] == "ai":
                    user_turn = chat_history[i - 1]
                    content = (
                        f"User Question: {user_turn['content']}\n\nAI Answer: {turn['content']}"
                    )
                    chat_docs.append(
                        Document(
                            page_content=content,
                            metadata={"source": "chat_session", "turn": i // 2},
                        )
                    )
            await log_stream.put(
                f"LOG: Converted {len(chat_history)} chat turns into {len(chat_docs)} documents."
            )
            state["all_rag_documents"].extend(chat_docs)
            await log_stream.put(
                f"LOG: Added chat documents. Total RAG documents now: {len(state['all_rag_documents'])}."
            )

            await log_stream.put(
                "--- [RAG PASS] Re-building Final RAPTOR Index with Chat History ---"
            )
            update_rag_node = create_update_rag_index_node(summarizer_llm, embeddings_model)
            update_result = await update_rag_node(state, end_of_run=True)
            state.update(update_result)

        num_questions = int(params.get("num_questions", 25))
        final_harvest_node = create_final_harvest_node(llm, summarizer_llm, num_questions)
        final_harvest_result = await final_harvest_node(state)
        state.update(final_harvest_result)

        academic_papers = state.get("academic_papers", {})
        session_id = state.get("session_id", "")

        if academic_papers:
            final_reports[session_id] = academic_papers
            await log_stream.put(
                f"SUCCESS: Final report with {len(academic_papers)} papers created."
            )
        else:
            await log_stream.put("WARNING: No academic papers were generated in the final harvest.")

        return JSONResponse(
            content={
                "message": "Harvest complete.",
            }
        )

    except Exception as e:
        error_message = f"An error occurred during harvest: {e}"
        await log_stream.put(error_message)
        await log_stream.put(traceback.format_exc())
        return JSONResponse(
            content={"message": error_message, "traceback": traceback.format_exc()},
            status_code=500,
        )


@app.get("/stream_log")
async def stream_log(request: Request):
    client_queue = asyncio.Queue()
    connected_log_clients.add(client_queue)

    async def event_generator():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    # Wait for a message with a heartbeat timeout
                    log = await asyncio.wait_for(client_queue.get(), timeout=15.0)
                    yield f"data: {log}\n\n"
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
        finally:
            connected_log_clients.remove(client_queue)

    return EventSourceResponse(event_generator())


@app.get("/log_stream")
async def stream_logs_legacy(request: Request):
    """Legacy endpoint redirecting to the new broadcast stream."""
    return await stream_log(request)


@app.get("/download_report/{session_id}")
async def download_report(session_id: str):
    papers = final_reports.get(session_id, {})

    if not papers:
        return JSONResponse(content={"error": "Report not found or expired."}, status_code=404)

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        for i, (question, content) in enumerate(papers.items()):
            safe_question = re.sub(r"[^\w\s-]", "", question).strip().replace(" ", "_")
            filename = f"paper_{i + 1}_{safe_question[:50]}.md"
            zip_file.writestr(filename, content)

    zip_buffer.seek(0)

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=NOA_Report_{session_id}.zip"},
    )


@app.post("/start_distillation")
async def start_distillation(payload: dict = Body(...)):
    global active_distillation_graph

    topics_str = payload.get("topics", "")
    topics_list = [t.strip() for t in topics_str.split(",") if t.strip()]

    # Handle both single anchor (legacy) and multiple anchors
    anchors_payload = payload.get("anchors")
    if not anchors_payload:
        anchor_question = payload.get("anchor_question", "")
        token_budget = payload.get("token_budget", 1_000_000)
        anchors = [{"question": anchor_question, "budget": token_budget}]
    else:
        anchors = anchors_payload

    debug_mode = payload.get("debug_mode", False)
    provider = payload.get("provider", "openrouter")
    api_key = payload.get("api_key", "")

    await log_stream.put(
        f"--- ⚗️ DISTILLATION: Initializing (provider: {provider}, debug: {debug_mode}) ---"
    )

    try:
        if debug_mode:
            llm = DistillationMockLLM()
            await log_stream.put("--- ⚗️ Distillation Debug Mode: using DistillationMockLLM ---")
        else:
            # For distillation, use synthesis_model as override if provided, else main model
            distil_model = (
                payload.get("synthesis_model", "").strip()
                or payload.get("openrouter_model", "stepfun/step-3.5-flash:free")
                if provider == "openrouter"
                else payload.get("llamacpp_model", "llama-3.2-1b-instruct")
            )
            if provider == "openrouter":
                if not api_key:
                    return JSONResponse(
                        content={"message": "OpenRouter API Key required"},
                        status_code=400,
                    )
                llm = ChatOpenAI(
                    model=distil_model,
                    openai_api_key=api_key,
                    openai_api_base="https://openrouter.ai/api/v1",
                    temperature=0.7,
                )
                await log_stream.put(f"--- Distillation LLM: OpenRouter ({distil_model}) ---")
            elif provider == "llamacpp":
                llamacpp_url = payload.get("llamacpp_url", "http://localhost:8080/v1")
                llamacpp_url = llamacpp_url.rstrip("/")
                if "/chat/completions" in llamacpp_url:
                    llamacpp_url = llamacpp_url.replace("/chat/completions", "")
                llamacpp_api_key = "no-key-required"
                llm = ChatLlamaCpp(
                    base_url=llamacpp_url,
                    api_key=llamacpp_api_key,
                    model=distil_model,
                    temperature=0.7,
                    max_tokens=4096,
                )
                await log_stream.put(f"--- Distillation LLM: LlamaCpp ({llamacpp_url}) ---")
            else:
                return JSONResponse(
                    content={"message": "Invalid provider. Please select openrouter or llamacpp."},
                    status_code=400,
                )
    except Exception as e:
        await log_stream.put(f"Distillation LLM Init Error: {e}")
        return JSONResponse(content={"message": f"Failed to initialize LLM: {e}"}, status_code=500)

    asyncio.create_task(run_distillation_loop(llm, topics_list, anchors, debug_mode))

    return {
        "status": "started",
        "message": f"Knowledge Distillation started with {len(anchors)} anchors.",
    }


async def run_distillation_loop(llm, topics, anchors, debug_mode):
    """Background loop that runs epochs for each anchor until budgets exhausted."""
    global active_distillation_graph

    total_qa_pairs = 0
    all_dataset_paths = []
    cumulative_step = 0

    for i, anchor in enumerate(anchors):
        question = anchor.get("question")
        budget = anchor.get("budget", 1_000_000)

        await log_stream.put(
            f"--- ⚗️ Starting Distillation for Anchor {i + 1}/{len(anchors)}: '{question[:50]}...' (Budget: {budget}) ---"
        )

        active_distillation_graph = DistillationGraph(
            llm=llm,
            topics=topics,
            anchor_question=question,
            token_budget=budget,
            debug_mode=debug_mode,
            log_queue=log_stream,
        )

        while active_distillation_graph.is_running:
            try:
                should_continue = await active_distillation_graph.run_epoch()
                cumulative_step += 1

                # Broadcast structured update to SSE
                data = {
                    "type": "distillation_update",
                    "source": "distillation",
                    "anchor_index": i,
                    "anchor_count": len(anchors),
                    "anchor_question": question,
                    "step": cumulative_step,
                    "epoch": active_distillation_graph.epochs_run,
                    "topology": [
                        [a.to_dict() for a in layer] for layer in active_distillation_graph.layers
                    ],
                    "token_count": active_distillation_graph.total_tokens,
                    "input_tokens": active_distillation_graph.total_input_tokens,
                    "output_tokens": active_distillation_graph.total_output_tokens,
                    "token_budget": active_distillation_graph.token_budget,
                    "qa_pairs_count": len(active_distillation_graph.distilled_data),
                    "total_qa_pairs_count": total_qa_pairs
                    + len(active_distillation_graph.distilled_data),
                    "dataset_file": active_distillation_graph.dataset_path,
                    "perplexity": active_distillation_graph.last_perplexity,
                }
                await log_stream.put(json.dumps(data))

                if not should_continue:
                    total_qa_pairs += len(active_distillation_graph.distilled_data)
                    all_dataset_paths.append(active_distillation_graph.dataset_path)
                    await log_stream.put(
                        f"--- ⚗️ Anchor {i + 1} Complete. Total QA so far: {total_qa_pairs} ---"
                    )
                    break

            except Exception as e:
                await log_stream.put(f"Distillation Error for Anchor {i + 1}: {e}")
                import traceback

                await log_stream.put(traceback.format_exc())
                break

        if not active_distillation_graph.is_running:
            await log_stream.put("--- ⚗️ Distillation halted by user. ---")
            break

    await log_stream.put(
        json.dumps(
            {
                "type": "distillation_complete",
                "total_qa_pairs_count": total_qa_pairs,
                "dataset_files": all_dataset_paths,
            }
        )
    )
    active_distillation_graph = None


@app.post("/stop_distillation")
async def stop_distillation():
    """Gracefully stop a running distillation."""
    global active_distillation_graph
    if not active_distillation_graph:
        return JSONResponse(status_code=404, content={"message": "No active distillation."})
    active_distillation_graph.is_running = False
    await log_stream.put("--- ⚗️ Distillation stop requested. Will halt after current epoch. ---")
    return {
        "status": "stopping",
        "message": "Distillation will stop after current epoch.",
    }


@app.get("/distillation_data")
async def get_distillation_data():
    """Return the current distilled dataset and metrics."""
    global active_distillation_graph
    if not active_distillation_graph:
        return JSONResponse(status_code=404, content={"message": "No active distillation."})

    return JSONResponse(
        content={
            "distilled_data": active_distillation_graph.distilled_data,
            "final_answer": active_distillation_graph.final_answer[:5000],
            "epochs_run": active_distillation_graph.epochs_run,
            "total_tokens": active_distillation_graph.total_tokens,
            "input_tokens": active_distillation_graph.total_input_tokens,
            "output_tokens": active_distillation_graph.total_output_tokens,
            "token_budget": active_distillation_graph.token_budget,
            "is_running": active_distillation_graph.is_running,
            "qa_pairs_count": len(active_distillation_graph.distilled_data),
        }
    )


@app.get("/download_distillation")
async def download_distillation():
    """Download the distilled dataset as a JSON file."""
    global active_distillation_graph
    if not active_distillation_graph:
        return JSONResponse(status_code=404, content={"message": "No active distillation."})

    dataset = {
        "anchor_question": active_distillation_graph.anchor_question,
        "topics": active_distillation_graph.topics,
        "total_epochs": active_distillation_graph.epochs_run,
        "total_input_tokens": active_distillation_graph.total_input_tokens,
        "total_output_tokens": active_distillation_graph.total_output_tokens,
        "total_tokens": active_distillation_graph.total_tokens,
        "qa_pairs": active_distillation_graph.distilled_data,
        "topology_archive": active_distillation_graph.topology_archive,
    }

    json_content = json.dumps(dataset, indent=2, ensure_ascii=False)

    return StreamingResponse(
        iter([json_content]),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=distilled_dataset.json"},
    )


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(log_broadcaster_worker())


async def log_broadcaster_worker():
    """Continuously pipes messages from the legacy log_stream queue to all broadcast clients."""
    while True:
        try:
            msg = await log_stream.get()
            await broadcast_log(msg)
        except Exception:
            # Prevent the worker from dying on unexpected errors
            pass
            await asyncio.sleep(1)


if __name__ == "__main__":
    _cfg = get_settings()
    uvicorn.run(app, host=_cfg.host, port=_cfg.port)
