# `/qdad` — code contract (on-the-fly runner)

## Harness flow

1. Materialize `.skill-runs/run_qdad.py` from skill `run_template.py` (or SKILL Appendix A).
2. Run it with CLI flags (`--n`, `--denoising-steps`, …).
3. Show `proposed_solution` (`# App Build Prompt`).

Do **not** require the user to know monorepo paths. The runner discovers `deepthink` by walking parents.

## Engine

`deepthink.qdad.run_qdad_pipeline` — LangGraph: foundation → grid → noise → denoise → synthesize.

## Parameters (CLI → pipeline)

| CLI | `params` key | Default |
|-----|--------------|--------:|
| `--n` | `grid_size` / `n` | 4 |
| `--temperature-scale` | `temperature_scale` | 1.3 |
| `--denoising-steps` | `denoising_steps` | 3 |
| `--noun-verb-temperature` | `noun_verb_temperature` | 0.6 |

## Example

```bash
python .skill-runs/run_qdad.py --prompt "cozy night writing app, offline-first" --n 3 --denoising-steps 2 --out .skill-runs/qdad-result.json
```
