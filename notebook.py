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
        A[("liver_crop_sample.parquet\n500 cells")] --> pre
        pre --> mesh
        mesh --> serve
    """)
    diagram
    return


@app.cell
def _(gpd, pathlib):
    """Load the 500-cell subset (``scripts/make_liver_subset.py -n 500``)."""
    gdf = gpd.read_parquet(
        pathlib.Path(__file__).parent / "sample_data" / "liver_crop_sample.parquet"
    )
    return (gdf,)


@app.cell
def _(gdf, po):
    import time

    t0 = time.perf_counter()
    po.meshify(gdf, use_cache=False)
    elapsed = time.perf_counter() - t0
    print(f"MESHIFY_SECONDS={elapsed:.6f}", flush=True)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
