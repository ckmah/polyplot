# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "polyplot",
#     "marimo>=0.22",
#     "geopandas>=1.0",
#     "pyarrow",
# ]
# ///

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")


@app.cell
def _():
    import pathlib

    import geopandas as gpd
    import marimo as mo
    import polyplot as po

    return gpd, mo, pathlib, po


@app.cell(hide_code=True)
def _(mo):
    diagram = mo.mermaid("""
    flowchart LR
        subgraph pre["po.meshify()"]
            direction LR
            B["Simplify\ngeometry"]
        end
        subgraph mesh["mesh building"]
            direction LR
            E["Align +\ncorrespondence"] --> H["Strip +\nearcut caps"] --> G["3D Taubin\nsmooth"]
        end
        subgraph serve["po.plot()"]
            direction LR
            I["Tile server"] --> J["WebGL viewer"]
        end
        A[("liver_crop_sample.parquet")] --> pre
        pre --> mesh
        mesh --> serve
    """)
    diagram
    return


@app.cell
def _(gpd, pathlib):
    """Load parquet; optionally crop to a few cells for fast tests.

    Set ``CROP_MAX_CELLS = None`` to use the full file.
    """
    CROP_MAX_CELLS = None
    CROP_MAX_ROWS = None

    gdf = gpd.read_parquet(
        pathlib.Path(__file__).parent / "sample_data" / "liver_crop_sample.parquet"
    )
    if CROP_MAX_CELLS is not None:
        _take = gdf["cell_id"].unique()[: int(CROP_MAX_CELLS)]
        gdf = gdf[gdf["cell_id"].isin(_take)].copy()
    if CROP_MAX_ROWS is not None and len(gdf) > CROP_MAX_ROWS:
        gdf = gdf.iloc[: int(CROP_MAX_ROWS)].copy()
    return (gdf,)


@app.cell
def _(gdf, po):
    po.meshify(gdf, use_cache=False)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
