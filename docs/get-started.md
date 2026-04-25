# Get started

## Requirements

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/) recommended (see the repo `pyproject.toml` for direct dependencies)

## Install

```bash
git clone https://github.com/ckmah/polyplot.git
cd polyplot
uv sync
```

## Sample data

The repository includes a small **`sample_data/liver_crop_sample.parquet`** (about 50 cells in one spatial patch) for demos and CI. To build a new subset from a local full `liver_crop.parquet`, use:

```bash
uv run python scripts/make_liver_subset.py
```

Larger local parquets in `sample_data/` are not tracked by git (see `sample_data/README.md` in the repository).

## Quick start (marimo)

The shortest path is [`quickstart.py`](https://github.com/ckmah/polyplot/blob/main/quickstart.py) at the repository root (marimo). From a clone of the repository:

```bash
uv run marimo edit quickstart.py
```

A **“Open in molab”** badge in the [README](https://github.com/ckmah/polyplot#readme) links to a hosted run once the file is on the default branch.

## Full demo

For a longer walkthrough, see `notebook.py`:

```bash
uv run marimo edit notebook.py
```

## Build this documentation site locally

```bash
uvx zensical serve
```

Open the URL printed in the terminal (by default [http://127.0.0.1:8000](http://127.0.0.1:8000)).

## Publish

Documentation for this site is built with [Zensical](https://zensical.org/docs/) in CI and published to **GitHub Pages** at the [`site_url`](https://ckmah.github.io/polyplot/) configured in `zensical.toml`.
