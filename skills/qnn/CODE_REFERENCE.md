# `/qnn` — Code reference for agent harnesses

**Do not only simulate the procedure in `SKILL.md`.** When the open-deepthink
package is available (clone, `pip install -e .`, or `OPEN_DEEPTHINK_ROOT`),
**run the reference implementation** and pass structured parameters.

## Canonical implementation

| Piece | Path |
|-------|------|
| **Library API** | `deepthink/qnn/pipeline.py` → `run_qnn_pipeline` |
| **Package export** | `from deepthink.qnn import run_qnn_pipeline, default_qnn_params` |
| **CLI runner (this skill)** | `skills/qnn/run_qnn.py` |
| **Self-attention** | `deepthink/self_attention.py` |
| **Chains** | `deepthink/chains/brainstorm_chains.py` |
| **UI mode** | open-deepthink **Brainstorming Mode** |

## Python API

```python
import asyncio
from langchain_openai import ChatOpenAI
from deepthink.qnn import run_qnn_pipeline, default_qnn_params

llm = ChatOpenAI(
    model="stepfun/step-3.5-flash:free",
    openai_api_key="...",
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.7,
)

params = default_qnn_params()
params.update({
    "qnn_mode": "auto",          # or "manual"
    "manual_layers": 3,          # L when manual
    "manual_width": 3,           # W when manual
    "num_epochs": 2,             # E (auto may override)
    "vector_word_size": 6,       # V
    "learning_rate": 0.5,        # Mirror Descent
    "attention_top_k": 5,        # non-local past neurons
    "enable_self_attention": True,
})

async def main():
    result = await run_qnn_pipeline(
        llm,
        user_prompt="explore this deadlock on the ownership path",
        params=params,
        document_context="",          # optional attached text
        chat_history=[],              # optional [{role, content}]
        log=print,                    # optional progress
        session_id="harness-1",
    )
    print(result["proposed_solution"])  # Solution-Space Report
    # Full dict also has: topology, agent_personas, attention_edges, epoch_maps

asyncio.run(main())
```

## Parameter contract

| Param | Type | Default | Meaning |
|-------|------|--------:|---------|
| `qnn_mode` | str | `auto` | `auto` uses complexity estimator; `manual` uses L/W |
| `manual_layers` | int | 3 | Layers L (depth) |
| `manual_width` | int | 3 | Width W (agents per layer) |
| `num_epochs` | int | 2 | Epochs E |
| `vector_word_size` | int | 6 | Guiding words per column (V) |
| `learning_rate` | float | 0.5 | Mirror Descent mutation intensity |
| `attention_top_k` | int | 5 | Self-attention edges per neuron |
| `enable_self_attention` | bool | true | Attend non-neighbor past neurons |

## CLI (harness shell)

From **repo root** (or with `OPEN_DEEPTHINK_ROOT` set):

```bash
export OPEN_DEEPTHINK_ROOT=/path/to/open-deepthink
export OPENROUTER_API_KEY=sk-...

python skills/qnn/run_qnn.py \
  --prompt "explore this deadlock" \
  --provider openrouter \
  --model stepfun/step-3.5-flash:free \
  --qnn-mode auto \
  --out /tmp/qnn-result.json

# Manual topology
python skills/qnn/run_qnn.py -p "richer metrics" \
  --qnn-mode manual --layers 3 --width 3 --epochs 2

# Free simulation
python skills/qnn/run_qnn.py -p "test" --debug
```

### CLI flags → params

| Flag | Param |
|------|--------|
| `--qnn-mode auto\|manual` | `qnn_mode` |
| `--layers` | `manual_layers` |
| `--width` | `manual_width` |
| `--epochs` | `num_epochs` |
| `--vector-word-size` | `vector_word_size` |
| `--learning-rate` | `learning_rate` |
| `--attention-top-k` | `attention_top_k` |
| `--no-attention` | `enable_self_attention=false` |
| `--context-file PATH` | `document_context` file body |
| `--debug` | Mock LLM (no API) |

## What the harness must do

1. **Resolve code**: monorepo root on `PYTHONPATH`, or `pip install -e .` from open-deepthink, or `OPEN_DEEPTHINK_ROOT`.
2. **Parse user intent** from `/qnn …` into `--prompt` + optional topology flags.
3. **Execute** `run_qnn.py` **or** `await run_qnn_pipeline(...)`.
4. **Surface** `proposed_solution` (Solution-Space Report) to the user.
5. **Do not implement** a production patch until the user picks a strategy.

## Return shape (JSON)

```json
{
  "mode": "brainstorm",
  "proposed_solution": "## 1. Impasse / Goal\\n...",
  "reasoning": "QNN Solution-Space Report complete.",
  "topology": {"layers": 2, "width": 3, "epochs": 2, "agents": 6, "...": "..."},
  "seed_pool": ["distill", "..."],
  "column_guiding_words": ["...", "..."],
  "agent_personas": {"agent_0_0": {"name": "...", "system_prompt": "..."}},
  "attention_edges": {"agent_1_0": [{"to_id": "agent_0_1", "strength": "med"}]},
  "epoch_maps": ["..."],
  "final_solution": {"mode": "brainstorm", "proposed_solution": "..."},
  "params": {}
}
```
