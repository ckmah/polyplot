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

    import geopandas as gpd
    import marimo as mo
    import polyplot as po

    data_url = os.getenv(
        "POLYPLOT_PARQUET_URL",
        "https://raw.githubusercontent.com/ckmah/polyplot/main/sample_data/liver_crop_sample.parquet",
    )
    return data_url, gpd, mo, po, textwrap


@app.cell
def _(data_url, gpd, mo, po, textwrap):
    intro = mo.md(
        textwrap.dedent(
            """
            # Polyplot quick start

            This notebook loads sample data from a raw GitHub URL so it works in
            hosted MoLab sessions where only this notebook file is copied.
            Optionally override the URL with `POLYPLOT_PARQUET_URL`.

            Tip: In `on_demand=True` mode, click a cell in the minimap to build a
            on-demand tile view. The camera also clamps max zoom-out so you cannot
            pull back far enough to frame the entire dataset at once (keeps big
            datasets from loading everything).
            """
        ).strip()
    )
    gdf = gpd.read_parquet(data_url)
    viewer = po.plot(gdf, on_demand=True)
    return mo.vstack(intro, viewer)


if __name__ == "__main__":
    app.run()
