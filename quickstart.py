# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "marimo>=0.22",
#   "geopandas>=1.0",
#   "pyarrow",
#   "polyplot @ git+https://github.com/ckmah/polyplot.git@ec17ca181279b30b4b2befddb15d5c15d118b3b0",
# ]
# ///

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")


@app.cell
def _():
    import os
    import textwrap
    import urllib.parse

    import geopandas as gpd
    import marimo as mo
    import polyplot as po

    dataset_repo = os.getenv("POLYPLOT_HF_DATASET_REPO", "ckmah/polyplot-data")
    dataset_file = os.getenv("POLYPLOT_HF_DATASET_FILE", "liver_crop_sample.parquet")
    dataset_ref = os.getenv("POLYPLOT_HF_DATASET_REF", "main")
    hf_data_url = os.getenv(
        "POLYPLOT_HF_PARQUET_URL",
        f"https://huggingface.co/datasets/{dataset_repo}/resolve/{urllib.parse.quote(dataset_ref, safe='')}/{dataset_file}",
    )
    fallback_data_url = os.getenv(
        "POLYPLOT_PARQUET_FALLBACK_URL",
        "https://raw.githubusercontent.com/ckmah/polyplot/main/sample_data/liver_crop_sample.parquet",
    )
    hf_token = os.getenv("HF_TOKEN")
    return fallback_data_url, gpd, hf_data_url, hf_token, mo, po, textwrap


@app.cell
def _(fallback_data_url, gpd, hf_data_url, hf_token, mo, po, textwrap):
    import pathlib
    import tempfile
    import urllib.request

    def _read_remote_parquet(url: str):
        storage_options = None
        headers: dict[str, str] = {}
        if hf_token and "huggingface.co" in url:
            headers["Authorization"] = f"Bearer {hf_token}"
            storage_options = headers
        try:
            return gpd.read_parquet(url, storage_options=storage_options)
        except Exception:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response:
                with tempfile.TemporaryDirectory() as temp_dir:
                    parquet_path = pathlib.Path(temp_dir) / "dataset.parquet"
                    parquet_path.write_bytes(response.read())
                    return gpd.read_parquet(parquet_path)

    errors = {}
    gdf = None
    loaded_url = None
    for data_url in (hf_data_url, fallback_data_url):
        try:
            gdf = _read_remote_parquet(data_url)
            loaded_url = data_url
            break
        except Exception as exc:
            errors[data_url] = str(exc)
    if gdf is None or loaded_url is None:
        failure_log = "\n".join(f"- {url}: {error}" for url, error in errors.items())
        raise RuntimeError(f"Unable to load dataset from configured URLs:\n{failure_log}")

    intro = mo.md(
        textwrap.dedent(
            f"""
            # Polyplot quick start

            This notebook loads sample data over HTTP and targets Hugging Face as
            the primary source so it works in hosted MoLab sessions where only
            this notebook file is copied.

            Source priority:
            1. `POLYPLOT_HF_PARQUET_URL`
            2. built URL from `POLYPLOT_HF_DATASET_REPO`, `POLYPLOT_HF_DATASET_REF`, `POLYPLOT_HF_DATASET_FILE`
            3. `POLYPLOT_PARQUET_FALLBACK_URL` (defaults to raw GitHub sample)

            If your HF dataset is private, set `HF_TOKEN`.

            Current source: `{loaded_url}`

            Tip: In `on_demand=True` mode, click a cell in the minimap to build a
            on-demand tile view. The camera also clamps max zoom-out so you cannot
            pull back far enough to frame the entire dataset at once (keeps big
            datasets from loading everything).
            """
        ).strip()
    )
    viewer = po.plot(gdf, on_demand=True)
    return mo.vstack(intro, viewer)


if __name__ == "__main__":
    app.run()
