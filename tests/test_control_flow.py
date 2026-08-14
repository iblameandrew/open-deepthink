"""Pytest: pipeline control flow with mock LLMs (no API keys)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepthink.cost import AUTO_AGENT_CAP, estimate_qdad_cost, estimate_qnn_cost
from deepthink.mocks import CoderMockLLM
from deepthink.qdad.utils import clamp_params
from deepthink.qnn.pipeline import _clamp_topology, clamp_qnn_topology


def test_auto_topology_caps_at_laptop_limit():
    layers, width, epochs = clamp_qnn_topology(10, 20, 3, manual=False)
    assert layers * width <= AUTO_AGENT_CAP
    assert epochs == 3


def test_manual_topology_not_shrunk():
    layers, width, epochs = _clamp_topology(10, 10, 4, manual=True)
    assert (layers, width, epochs) == (10, 10, 4)


def test_qdad_clamp_defaults_small():
    n, noise, steps, nv = clamp_params()
    assert n == 3
    assert steps == 2
    assert 0.7 <= noise <= 1.8


def test_qnn_cost_formula_2x2x1():
    est = estimate_qnn_cost(2, 2, 1, qnn_mode="manual")
    # seeds + 4 span + 4 forward + 2 final = 11
    assert est.llm_calls == 11
    assert est.breakdown["persona_rewrite"] == 0
    assert est.breakdown["epoch_maps"] == 0


def test_qnn_cost_includes_rewrite_when_multi_epoch():
    est = estimate_qnn_cost(2, 2, 2, qnn_mode="manual")
    assert est.breakdown["persona_rewrite"] == 4
    assert est.breakdown["epoch_maps"] == 1
    assert est.breakdown["reframe"] == 1
    assert est.llm_calls == 21


def test_qdad_cost_formula():
    est = estimate_qdad_cost(2, 2)
    assert est.llm_calls == 2 + 4 * 3  # foundation + synth + 4*(noise+2 critics)
    assert est.breakdown["critic_cells"] == 8


@pytest.mark.asyncio
async def test_qnn_skips_mirror_descent_on_single_epoch():
    from deepthink.qnn import run_qnn_pipeline

    rewritten = {"count": 0}

    class Probe(CoderMockLLM):
        async def ainvoke(self, input_data, config=None, **kwargs):
            text = str(input_data).lower()
            if "mirror descent" in text or "evolve" in text and "system prompt" in text:
                rewritten["count"] += 1
            return await super().ainvoke(input_data, config=config, **kwargs)

    out = await run_qnn_pipeline(
        Probe(),
        "cancellation safety on a latch",
        params={
            "qnn_mode": "manual",
            "manual_layers": 2,
            "manual_width": 2,
            "num_epochs": 1,
            "enable_self_attention": False,
        },
    )
    assert out["topology"]["epochs"] == 1
    assert out["epoch_maps"] == []
    assert rewritten["count"] == 0
    assert out.get("proposed_solution")


@pytest.mark.asyncio
async def test_qnn_two_epochs_writes_one_map():
    from deepthink.qnn import run_qnn_pipeline

    out = await run_qnn_pipeline(
        CoderMockLLM(),
        "cancellation safety on a latch",
        params={
            "qnn_mode": "manual",
            "manual_layers": 2,
            "manual_width": 2,
            "num_epochs": 2,
            "enable_self_attention": True,
        },
    )
    assert len(out.get("epoch_maps") or []) == 1
    assert len(out.get("agent_personas") or {}) == 4


@pytest.mark.asyncio
async def test_qdad_runs_requested_denoise_steps():
    from deepthink.qdad import run_qdad_pipeline

    logs: list[str] = []

    def log(msg: str) -> None:
        logs.append(str(msg))

    out = await run_qdad_pipeline(
        llm=CoderMockLLM(),
        params={"grid_size": 2, "n": 2, "denoising_steps": 2},
        user_prompt="cozy night writing app",
        log=log,
    )
    assert isinstance(out, dict)
    denoise_rounds = [m for m in logs if "Denoising step" in m]
    assert len(denoise_rounds) == 2, logs[-20:]


def test_session_store_persists_json_safe_fields(tmp_path):
    from deepthink.sessions import SessionStore

    store = SessionStore(persist_dir=tmp_path, persist=True)
    store["abc"] = {"session_id": "abc", "mode": "brainstorm", "raptor_index": object()}
    loaded = SessionStore(persist_dir=tmp_path, persist=False).load("abc")
    assert loaded["session_id"] == "abc"
    assert "raptor_index" not in loaded


if __name__ == "__main__":
    # Allow `python tests/test_control_flow.py` without pytest-asyncio plugin
    test_auto_topology_caps_at_laptop_limit()
    test_manual_topology_not_shrunk()
    test_qdad_clamp_defaults_small()
    test_qnn_cost_formula_2x2x1()
    test_qnn_cost_includes_rewrite_when_multi_epoch()
    test_qdad_cost_formula()
    asyncio.run(test_qnn_skips_mirror_descent_on_single_epoch())
    asyncio.run(test_qnn_two_epochs_writes_one_map())
    asyncio.run(test_qdad_runs_requested_denoise_steps())
    import tempfile

    test_session_store_persists_json_safe_fields(Path(tempfile.mkdtemp()))
    print("test_control_flow: ok")
