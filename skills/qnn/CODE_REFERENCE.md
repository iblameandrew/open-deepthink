# `/qnn` — code contract (on-the-fly runner)

## Harness flow

1. Materialize `.skill-runs/run_qnn.py` from skill `run_template.py` (or SKILL Appendix A).
2. Run it with CLI flags derived from the user message.
3. Show `proposed_solution` (Solution-Space Report).

Do **not** require the user to know monorepo paths. The runner discovers `deepthink` by walking parents.

## Engine

`deepthink.qnn.run_qnn_pipeline` in `deepthink/qnn/pipeline.py`.

## Parameters (CLI → pipeline)

| CLI | `params` key | Default |
|-----|--------------|--------:|
| `--qnn-mode` | `qnn_mode` | `auto` |
| `--layers` | `manual_layers` | 3 |
| `--width` | `manual_width` | 3 |
| `--epochs` | `num_epochs` | 2 |
| `--vector-word-size` | `vector_word_size` | 6 |
| `--learning-rate` | `learning_rate` | 0.5 |
| `--attention-top-k` | `attention_top_k` | 5 |
| `--no-attention` | `enable_self_attention=false` | |

## Example

```bash
python .skill-runs/run_qnn.py --prompt "explore this deadlock" --qnn-mode auto --out .skill-runs/qnn-result.json
```
