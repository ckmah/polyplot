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

    dataset_repo = os.getenv("POLYPLOT_HF_DATASET_REPO", "ckmah/polyplot")
    dataset_file = os.getenv("POLYPLOT_HF_DATASET_FILE", "liver_crop_sample.parquet")
    dataset_ref = os.getenv(
        "POLYPLOT_HF_DATASET_REF", "92678be92f8e0b06fc2a32b53885c4fdf3419ee3"
    )
    hf_data_url = os.getenv(
        "POLYPLOT_HF_PARQUET_URL",
        f"https://huggingface.co/datasets/{dataset_repo}/resolve/{urllib.parse.quote(dataset_ref, safe='')}/{dataset_file}",
    )
    hf_token = os.getenv("HF_TOKEN")
    return gpd, hf_data_url, hf_token, mo, po, textwrap


@app.cell
def _(gpd, hf_data_url, hf_token, mo, po, textwrap):
    storage_options = None
    if hf_token and "huggingface.co" in hf_data_url:
        storage_options = {"Authorization": f"Bearer {hf_token}"}
    gdf = gpd.read_parquet(hf_data_url, storage_options=storage_options)

    intro = mo.md(
        textwrap.dedent(
            f"""
            # Polyplot quick start

            This notebook loads sample data from Hugging Face only, so it works
            in hosted MoLab sessions where only this notebook file is copied.

            Source configuration:
            1. `POLYPLOT_HF_PARQUET_URL`
            2. built URL from `POLYPLOT_HF_DATASET_REPO`,
               `POLYPLOT_HF_DATASET_REF`, `POLYPLOT_HF_DATASET_FILE`

            Defaults point to Hugging Face dataset `ckmah/polyplot` at commit
            `92678be92f8e0b06fc2a32b53885c4fdf3419ee3`.

            If your HF dataset is private, set `HF_TOKEN`.

            Current source: `{hf_data_url}`

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
