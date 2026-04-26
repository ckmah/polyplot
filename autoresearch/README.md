# Autoresearch

Experiment drivers and benchmarks for the meshify pipeline. This tree is **not** part of the installable `polyplot` package.

| Path | Role |
|------|------|
| `scripts/meshify_benchmark_measure.py` | Runs `notebook.py`, prints seconds per cell |
| `scripts/run_experiments.py` | Main commit / measure / revert loop |
| `scripts/autoresearch_alg.py` | Fixed catalog of algorithmic experiments |
| `scripts/autoresearch_1k.py` | Structural grid over mesh knobs |
| `watch.py` | Marimo notebook: plot `results.tsv` |
| `layouts/watch.slides.json` | Optional marimo slides layout |
| `results.tsv` | Append-only log (create locally; not ignored by default) |

Logs go under `autoresearch/logs/` (gitignored).

Examples:

```bash
uv run python autoresearch/scripts/meshify_benchmark_measure.py
uv run python autoresearch/scripts/run_experiments.py
uv run marimo edit autoresearch/watch.py
```
