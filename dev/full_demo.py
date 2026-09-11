# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.22",
#     "geopandas>=1.0",
#     "pyarrow",
#     "polyrender @ git+https://github.com/ckmah/polyplot.git@e2adf041c417b0ffb0121a78364f368e19cc2999",
# ]
# ///

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")


@app.cell
def _():
    import os

    import geopandas as gpd
    import polyrender as po

    parquet_url = os.getenv(
        "POLYPLOT_PARQUET_URL",
        "https://huggingface.co/datasets/ckmah/polyplot/resolve/92678be92f8e0b06fc2a32b53885c4fdf3419ee3/liver_crop.parquet",
    )
    return gpd, parquet_url, po


@app.cell
def _(gpd, parquet_url):
    """Load dataset from Hugging Face only (no fallbacks)."""
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
