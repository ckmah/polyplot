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

    dataset_repo = os.getenv("POLYPLOT_HF_DATASET_REPO", "ckmah/polyplot-data")
    dataset_file = os.getenv("POLYPLOT_HF_DATASET_FILE", "liver_crop_sample.parquet")
    dataset_ref = os.getenv("POLYPLOT_HF_DATASET_REF", "main")
    hf_data_url = os.getenv(
        "POLYPLOT_HF_PARQUET_URL",
        f"https://huggingface.co/datasets/{dataset_repo}/resolve/{urllib.parse.quote(dataset_ref, safe='')}/{dataset_file}",
    )
    fallback_data_url = os.getenv(
        "POLYPLOT_PARQUET_FALLBACK_URL",
        "https://raw.githubusercontent.com/ckmah/polyplot/main/sample_data/liver_crop_sample.parquet",
    )
    hf_token = os.getenv("HF_TOKEN")
    return fallback_data_url, gpd, hf_data_url, hf_token, po


@app.cell
def _(fallback_data_url, gpd, hf_data_url, hf_token):
    """Load sample data from Hugging Face URL with GitHub fallback."""
    import pathlib
    import tempfile
    import urllib.request

    def _read_remote_parquet(url: str):
        storage_options = None
        headers: dict[str, str] = {}
        if hf_token and "huggingface.co" in url:
            headers["Authorization"] = f"Bearer {hf_token}"
            storage_options = headers
        try:
            return gpd.read_parquet(url, storage_options=storage_options)
        except Exception:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response:
                with tempfile.TemporaryDirectory() as temp_dir:
                    parquet_path = pathlib.Path(temp_dir) / "dataset.parquet"
                    parquet_path.write_bytes(response.read())
                    return gpd.read_parquet(parquet_path)

    errors = {}
    for data_url in (hf_data_url, fallback_data_url):
        try:
            gdf = _read_remote_parquet(data_url)
            print(f"Loaded dataset from {data_url}", flush=True)
            return (gdf,)
        except Exception as exc:
            errors[data_url] = str(exc)
    failure_log = "\n".join(f"- {url}: {error}" for url, error in errors.items())
    raise RuntimeError(f"Unable to load dataset from configured URLs:\n{failure_log}")


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
