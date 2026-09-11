# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "marimo>=0.22",
#   "geopandas>=1.0",
#   "pyarrow",
#   "polyrender @ git+https://github.com/ckmah/polyplot.git@e2adf041c417b0ffb0121a78364f368e19cc2999",
# ]
# ///

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")


@app.cell
def _():
    import os
    import textwrap

    import geopandas as gpd
    import marimo as mo
    import polyrender as po

    parquet_url = os.getenv(
        "POLYPLOT_PARQUET_URL",
        "https://huggingface.co/datasets/ckmah/polyplot/resolve/92678be92f8e0b06fc2a32b53885c4fdf3419ee3/liver_crop_sample.parquet",
    )
    return gpd, mo, parquet_url, po, textwrap


@app.cell
def _(gpd, mo, parquet_url, po, textwrap):
    import pathlib
    import tempfile
    import urllib.parse
    import urllib.request

    scheme = urllib.parse.urlparse(parquet_url).scheme.lower()
    if scheme in {"http", "https"}:
        with urllib.request.urlopen(parquet_url, timeout=120) as response:
            with tempfile.TemporaryDirectory() as temp_dir:
                parquet_path = pathlib.Path(temp_dir) / "dataset.parquet"
                parquet_path.write_bytes(response.read())
                gdf = gpd.read_parquet(parquet_path)
    else:
        gdf = gpd.read_parquet(parquet_url)

    intro = mo.md(
        textwrap.dedent(
            f"""
            # Polyplot quick start

            This notebook loads sample data from Hugging Face only, so it works
            in hosted MoLab sessions where only this notebook file is copied.

            Set `POLYPLOT_PARQUET_URL` to any public parquet URL.
            Current source: `{parquet_url}`

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
