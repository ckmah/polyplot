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
    import pathlib

    import geopandas as gpd
    import polyplot as po

    repo_root = (
        pathlib.Path(__file__).resolve().parent
        if "__file__" in globals()
        else pathlib.Path.cwd()
    )
    return gpd, po, repo_root


@app.cell
def _(gpd, repo_root):
    """Load local full data when present, otherwise the tracked sample."""
    data_dir = repo_root / "sample_data"
    full_path = data_dir / "liver_crop.parquet"
    sample_path = data_dir / "liver_crop_sample.parquet"
    data_path = full_path if full_path.exists() else sample_path
    gdf = gpd.read_parquet(data_path)
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
