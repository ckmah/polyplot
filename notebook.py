# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.22",
#     "geopandas>=1.0",
#     "pyarrow",
#     "polyplot @ git+https://github.com/ckmah/polyplot.git@ec17ca181279b30b4b2befddb15d5c15d118b3b0",
# ]
# ///

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")


@app.cell
def _():
    import os

    import geopandas as gpd
    import polyplot as po

    data_url = os.getenv(
        "POLYPLOT_PARQUET_URL",
        "https://raw.githubusercontent.com/ckmah/polyplot/main/sample_data/liver_crop_sample.parquet",
    )
    return data_url, gpd, po


@app.cell
def _(data_url, gpd):
    """Load sample data from raw GitHub (or POLYPLOT_PARQUET_URL override)."""
    gdf = gpd.read_parquet(data_url)
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
