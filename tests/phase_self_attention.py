"""Unit tests for Qualitative Self-Attention (brainstorm QSA)."""

import sys

sys.path.insert(0, r".")

from deepthink.self_attention import (
    AttentionCandidate,
    collect_attention_candidates,
    compute_self_attention,
    format_attention_context,
    graph_neighbor_ids,
    score_pair_heuristic,
    select_top_edges,
)

results = []


def chk(name, fn):
    try:
        fn()
        results.append((name, "OK", None))
    except AssertionError as e:
        results.append((name, "FAIL", f"AssertionError: {e}"))
    except Exception as e:
        results.append((name, "FAIL", f"{type(e).__name__}: {e}"))


def t1():
    layers = [["p0", "p1"], ["p2", "p3"], ["p4"]]
    assert graph_neighbor_ids("agent_0_0", layers) == set()
    assert graph_neighbor_ids("agent_1_0", layers) == {"agent_0_0", "agent_0_1"}
    assert graph_neighbor_ids("agent_2_0", layers) == {"agent_1_0", "agent_1_1"}


chk("graph_neighbor_ids returns previous layer only", t1)


def t2():
    state = {
        "epoch": 1,
        "all_layers_prompts": [["a", "b"], ["c", "d"]],
        "agent_personas": {
            "agent_1_0": {
                "name": "Invariant Probe",
                "specialty": "Ownership boundaries",
                "guiding_words": "lease ownership timeout invariant",
                "skills": ["falsifier design"],
            }
        },
        "agent_outputs": {
            # previous layer neighbors — should be excluded from attention pool
            "agent_0_0": {
                "proposed_solution": "Use lease-based ownership transfer.",
                "reasoning": "timeouts on busy path",
            },
            "agent_0_1": {
                "proposed_solution": "Add global lock.",
                "reasoning": "serialize everything",
            },
            # non-neighbor would be same layer — usually empty mid-pass; simulate skip
        },
        "memory": {
            "agent_0_0": [
                {
                    "proposed_solution": "Earlier epoch: probe with ordered event logs at ownership boundaries.",
                    "reasoning": "lease ownership invariant timeout",
                }
            ],
            "agent_1_1": [
                {
                    "proposed_solution": "Map strategies with falsifiers for concurrent callers.",
                    "reasoning": "ownership invariant probing",
                }
            ],
            "agent_1_0": [
                {
                    "proposed_solution": "My own past (should not appear in attention).",
                    "reasoning": "self",
                }
            ],
        },
    }
    neighbors = graph_neighbor_ids("agent_1_0", state["all_layers_prompts"])
    cands = collect_attention_candidates(state, "agent_1_0", neighbors)
    ids = {c.agent_id for c in cands}
    assert "agent_1_0" not in ids, "must not attend self"
    # Neighbors' current outputs excluded; their past memory may still appear
    assert "agent_1_1" in ids, "should attend non-neighbor peer memory"
    # Current epoch neighbor outputs should not be double-counted as agent_outputs candidates
    current_n = [
        c for c in cands if c.source == "agent_outputs" and c.agent_id in neighbors
    ]
    assert current_n == [], f"neighbor current outputs leaked: {current_n}"


chk("collect_attention_candidates excludes self & neighbor current outputs", t2)


def t3():
    persona = {
        "specialty": "ownership invariants",
        "guiding_words": "lease ownership timeout",
        "skills": ["falsifier"],
    }
    high = AttentionCandidate(
        agent_id="agent_0_2",
        output={
            "proposed_solution": "lease-based ownership with timeout on busy path",
            "reasoning": "ownership invariant falsifier",
        },
        source="memory",
    )
    edge = score_pair_heuristic("agent_1_0", persona, high)
    assert edge.strength in ("med", "high"), edge
    assert edge.score > 0
    none_edge = score_pair_heuristic(
        "agent_1_0",
        persona,
        AttentionCandidate(
            agent_id="x",
            output={"proposed_solution": "zz", "reasoning": "qq"},
            source="memory",
        ),
    )
    assert none_edge.strength in ("none", "low")


chk("score_pair_heuristic rewards shared lexicon", t3)


def t4():
    edges = []
    for i, s in enumerate(["high", "med", "low", "none", "high"]):
        from deepthink.self_attention import AttentionEdge

        edges.append(
            AttentionEdge(
                from_id="a",
                to_id=f"agent_0_{i}",
                strength=s,
                qualitative_distance="mid",
                kind="affinity",
                rationale="r",
                source="memory",
                score=float(i),
            )
        )
    top = select_top_edges(edges, top_k=2, min_strength="low")
    assert len(top) <= 2
    assert all(e.strength != "none" for e in top)


chk("select_top_edges filters none and caps k", t4)


def t5():
    from deepthink.self_attention import AttentionEdge

    block = format_attention_context(
        [
            AttentionEdge(
                from_id="agent_1_0",
                to_id="agent_0_2",
                strength="high",
                qualitative_distance="near",
                kind="affinity",
                rationale="shared ownership concepts",
                source="memory",
                excerpt="Use lease-based handoff.",
            )
        ]
    )
    assert "Qualitative Self-Attention" in block
    assert "agent_0_2" in block
    assert "lease-based" in block
    assert format_attention_context([]) == ""


chk("format_attention_context renders markdown block", t5)


def t6():
    state = {
        "epoch": 1,
        "all_layers_prompts": [["p0", "p1"], ["p2", "p3"]],
        "agent_personas": {
            "agent_1_0": {
                "specialty": "systems ownership",
                "guiding_words": "ownership lease invariant",
            }
        },
        "agent_outputs": {
            "agent_0_0": {
                "proposed_solution": "neighbor only",
                "reasoning": "upstream",
            },
            "agent_0_1": {
                "proposed_solution": "neighbor two",
                "reasoning": "upstream",
            },
        },
        "memory": {
            "agent_0_0": [
                {
                    "proposed_solution": "prior epoch ownership lease strategy",
                    "reasoning": "invariant probing for ownership",
                }
            ],
            "agent_1_1": [
                {
                    "proposed_solution": "peer: concurrent ownership falsifiers",
                    "reasoning": "lease timeout ownership",
                }
            ],
        },
    }
    edges, block = compute_self_attention(state, "agent_1_0", top_k=3)
    assert block
    assert edges
    # Must not only be pure neighbors without past value — peer agent_1_1 or past of 0_0
    targets = {e.to_id for e in edges}
    assert targets & {"agent_1_1", "agent_0_0"}


chk("compute_self_attention end-to-end produces non-local edges", t6)


for name, status, err in results:
    line = f"  [{status}] {name}"
    if err:
        line += f" :: {err}"
    print(line)
ok = sum(1 for _, s, _ in results if s == "OK")
print(f"\nPHASE SELF-ATTENTION: {ok}/{len(results)} OK")
raise SystemExit(0 if ok == len(results) else 1)
