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


@app.cell
def _(mo):
    intro = mo.md(
        """
    # Cross-section mesh viewer

    **In plain terms:** your dataset is a list of 2D outlines stacked at different heights (`ZIndex`), grouped by biological **cell**. Think of it like a CAT scan turned into closed curves per slice.

    **`po.meshify(gdf)`** exports **GLB** tiles under `.polyplot/<content hash>/` (override with `out_dir=`). Use **`smooth=False`** for no 3D smoothing. **`use_cache=True`** skips work when that fingerprint already exists. In notebooks, pass **`show_progress=False`** if you wrap the call in your own spinner.

    **`po.plot(gdf)`** ensures tiles exist (via `meshify`), serves them locally, and opens the WebGL viewer. **Wireframe**, **opacity**, and **background** are controlled in the viewer toolbar, not from Python.
    """
    )
    intro
    return


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
def _(mo):
    param_doc = mo.md(
        """
    ## Public API

    | Function | Main parameters | Notes |
    |----------|-----------------|--------|
    | `meshify(..., show_progress=True)` | `gdf`, `out_dir`, `smooth`, `use_cache`, `show_progress` | Returns `MeshifyInfo` (tile index + `out_dir`, `_cache_hit`). |
    | `plot(gdf, max_concurrent_fetches=4, use_cache=True)` | Same as signature | Runs `meshify` with `smooth=True`, `out_dir=".polyplot"`, `show_progress=False`. Viewer toolbar: wireframe / opacity / background. |
    """
    )
    param_doc
    return


@app.cell
def _(gdf, mo, po):
    with mo.status.spinner(title="Meshify (or cache load)…", remove_on_exit=True):
        tiles_info = po.meshify(gdf, show_progress=True, use_cache=True)

    _tiles = tiles_info["tiles"]
    _hit = tiles_info.get("_cache_hit", False)
    mo.md(
        f"{'Cache hit' if _hit else 'Built'} · **{len(_tiles)}** tile(s) · \n\n"
        f"tile_size_xy=**{tiles_info['tile_size_xy']:.0f}** · \n\n"
        f"avg **{sum(t['cell_count'] for t in _tiles) / max(len(_tiles), 1):.1f}** cells/tile · \n\n"
        f"`out_dir={tiles_info['out_dir']}`"
    )
    return


@app.cell
def _(gdf, po):
    po.plot(gdf, use_cache=False)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
