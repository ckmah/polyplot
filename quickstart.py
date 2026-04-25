# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "marimo>=0.22",
#   "geopandas>=1.0",
#   "pyarrow",
#   "polyplot",
# ]
# [tool.uv.sources]
# polyplot = { path = ".", editable = true }
# ///

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")


@app.cell
def _():
    import pathlib
    import textwrap

    import geopandas as gpd
    import marimo as mo
    import polyplot as po

    return gpd, mo, pathlib, po, textwrap


@app.cell
def _(gpd, mo, pathlib, po, textwrap):
    intro = mo.md(
        textwrap.dedent(
            """
            # Polyplot quick start

            This notebook uses the tracked `sample_data/liver_crop_sample.parquet`. From
            the repository root (so paths resolve) run `uv run marimo edit quickstart.py`.
            """
        ).strip()
    )
    gdf = gpd.read_parquet(
        pathlib.Path(__file__).resolve().parent
        / "sample_data"
        / "liver_crop_sample.parquet"
    )
    with mo.status.spinner(title="Meshify (or cache load)…", remove_on_exit=True):
        po.meshify(gdf, show_progress=True, use_cache=True)
    viewer = po.plot(gdf, use_cache=True)
    return mo.vstack(intro, viewer)


if __name__ == "__main__":
    app.run()
