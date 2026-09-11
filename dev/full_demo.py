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
    import urllib.parse

    import geopandas as gpd
    import polyplot as po

    dataset_repo = os.getenv("POLYPLOT_HF_DATASET_REPO", "ckmah/polyplot")
    dataset_file = os.getenv("POLYPLOT_HF_DATASET_FILE", "liver_crop.parquet")
    dataset_ref = os.getenv(
        "POLYPLOT_HF_DATASET_REF", "92678be92f8e0b06fc2a32b53885c4fdf3419ee3"
    )
    data_url = os.getenv(
        "POLYPLOT_HF_PARQUET_URL",
        f"https://huggingface.co/datasets/{dataset_repo}/resolve/{urllib.parse.quote(dataset_ref, safe='')}/{dataset_file}",
    )
    hf_token = os.getenv("HF_TOKEN")
    return data_url, gpd, hf_token, po


@app.cell
def _(data_url, gpd, hf_token):
    """Load dataset from Hugging Face only (no fallbacks)."""
    storage_options = None
    if hf_token and "huggingface.co" in data_url:
        storage_options = {"Authorization": f"Bearer {hf_token}"}
    gdf = gpd.read_parquet(data_url, storage_options=storage_options)
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


if __name__ == "__main__":
    app.run()
