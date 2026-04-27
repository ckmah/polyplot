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

    return gpd, pathlib, po


@app.cell
def _(gpd, pathlib):
    """Load the 500-cell subset (``scripts/make_liver_subset.py -n 500``)."""
    gdf = gpd.read_parquet(
        pathlib.Path(__file__).parent / "sample_data" / "liver_crop.parquet"
    )
    return (gdf,)


@app.cell
def _(gdf, po):
    import time

    n_cells = int(gdf["cell_id"].nunique())
    t0 = time.perf_counter()
    po.meshify(gdf, use_cache=False)
    elapsed = time.perf_counter() - t0
    per_cell = elapsed / n_cells if n_cells else 0.0
    print(f"MESHIFY_SECONDS={elapsed:.6f}", flush=True)
    print(f"MESHIFY_PER_CELL_SECONDS={per_cell:.6f}", flush=True)
    return


@app.cell
def _(gdf, po):
    viewer = po.plot(gdf, on_demand=True, max_orbit_distance=2000)
    viewer
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
