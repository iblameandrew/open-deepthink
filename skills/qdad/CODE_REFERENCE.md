# `/qdad` — Code reference for agent harnesses

**Do not only simulate the procedure in `SKILL.md`.** When the open-deepthink
package is available, **run the Qualitative Diffusion engine** and pass
structured parameters.

## Canonical implementation

| Piece | Path |
|-------|------|
| **Library API** | `deepthink/qdad/pipeline.py` → `run_qdad_pipeline` |
| **Package export** | `from deepthink.qdad import run_qdad_pipeline` |
| **LangGraph** | `deepthink/qdad/graph.py`, `nodes.py` |
| **Chains** | `deepthink/chains/qdad_chains.py` |
| **CLI runner (this skill)** | `skills/qdad/run_qdad.py` |
| **UI mode** | open-deepthink **App Slot Machine Mode** |

## Python API

```python
import asyncio
from langchain_openai import ChatOpenAI
from deepthink.qdad import run_qdad_pipeline

llm = ChatOpenAI(
    model="stepfun/step-3.5-flash:free",
    openai_api_key="...",
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.7,
)

params = {
    "grid_size": 4,                 # N  (alias: "n")
    "n": 4,
    "temperature_scale": 1.3,       # forward diffusion noise T
    "denoising_steps": 3,           # reverse diffusion rounds
    "noun_verb_temperature": 0.6,   # foundation basis T
}

async def main():
    result = await run_qdad_pipeline(
        llm=llm,
        params=params,
        user_prompt="cozy night writing app, soft dark mode, offline-first",
        document_context="",
        chat_history=[],
        log=print,
        session_id="harness-1",
    )
    print(result["proposed_solution"])  # # App Build Prompt ...

asyncio.run(main())
```

## Parameter contract

| Param | Type | Default | Range | Role |
|-------|------|--------:|-------|------|
| `grid_size` / `n` | int | 4 | 2–8 | N×N feature agents |
| `temperature_scale` | float | 1.3 | 0.7–1.8 | Forward diffusion (noise) temperature |
| `denoising_steps` | int | 3 | 1–6 | Critic reverse-diffusion rounds |
| `noun_verb_temperature` | float | 0.6 | 0.3–1.0 | Foundation nouns/verbs temperature |

Clamped in `deepthink.qdad.utils.clamp_params`.

## CLI (harness shell)

```bash
export OPEN_DEEPTHINK_ROOT=/path/to/open-deepthink
export OPENROUTER_API_KEY=sk-...

python skills/qdad/run_qdad.py \
  --prompt "cozy night writing app, soft dark mode, offline-first" \
  --provider openrouter \
  --n 3 \
  --temperature-scale 1.3 \
  --denoising-steps 2 \
  --noun-verb-temperature 0.6 \
  --out /tmp/qdad-result.json

# Free simulation
python skills/qdad/run_qdad.py -p "test app" --debug --n 2 --denoising-steps 1
```

### CLI flags → params

| Flag | Param |
|------|--------|
| `--n` | `grid_size` / `n` |
| `--temperature-scale` | `temperature_scale` |
| `--denoising-steps` | `denoising_steps` |
| `--noun-verb-temperature` | `noun_verb_temperature` |
| `--context-file PATH` | `document_context` |
| `--debug` | Mock LLM (no API) |

## What the harness must do

1. **Resolve code** via monorepo / `pip install -e .` / `OPEN_DEEPTHINK_ROOT`.
2. Parse `/qdad …` (and optional `N=`, `steps=`) into CLI or `params`.
3. **Execute** `run_qdad.py` **or** `await run_qdad_pipeline(...)`.
4. **Surface** `proposed_solution` (`# App Build Prompt`) to the user.
5. Hand off to normal build only after user approval.

## Pipeline phases (code)

```
foundation → grid → noise → denoise×steps → synthesize
```

Implemented in `deepthink/qdad/nodes.py` + `graph.py`, orchestrated by
`run_qdad_pipeline`.

## Return shape (JSON)

```json
{
  "mode": "app_slot_machine",
  "proposed_solution": "# App Build Prompt\\n\\n## High-Level Vision\\n...",
  "reasoning": "...",
  "feature_matrix": "...",
  "nouns": ["..."],
  "verbs": ["..."]
}
```

(Exact keys may include extra matrix fields depending on graph version; always
prefer `proposed_solution` as the user-facing App Build Prompt.)
