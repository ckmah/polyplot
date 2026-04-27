# User guide

## Data model

`polyplot` expects a [GeoDataFrame](https://geopandas.org/) with at least:

- `cell_id` – stable identifier for each biological cell (or object).
- `ZIndex` – ordering along the stack (height or slice).
- `geometry` – 2D polygons in a consistent map CRS.

Each row is one 2D outline; multiple rows per `cell_id` with different `ZIndex` form the 2.5D stack.

## `meshify`

```python
import geopandas as gpd
import polyplot as po

gdf = gpd.read_parquet("sample_data/liver_crop_sample.parquet")
info = po.meshify(
    gdf,
    out_dir=".polyplot",
    smooth=True,
    use_cache=True,
    show_progress=True,
)
```

- Writes `**tiles.json**` and per-tile **GLB** files under a content-hashed subfolder of `out_dir` (default `.polyplot`).
- `smooth` – 3D Taubin smoothing (on) or raw surfaces (off).
- `use_cache` – if the same fingerprint was built before, reuses the cache and sets `_cache_hit` in the return value.
- `show_progress` – in marimo, a progress display while building.

## `plot`

```python
po.plot(gdf, use_cache=True, max_concurrent_fetches=4)
```

Starts or reuses a local **tile server** and returns a marimo **anywidget** with a Three.js / WebGL view. The toolbar adjusts wireframe, opacity, and background.

Additional optional controls:

```python
# On-demand tile streaming: bound what can load by distance cap.
po.plot(gdf, on_demand=True, max_orbit_distance=250.0)

# Clamp how far you can zoom out (max distance from orbit target).
po.plot(gdf, max_orbit_distance=250.0)
```

## Caching and disk use

`meshify` keeps a content-addressed cache. Older digests under the same `out_dir` can be pruned; the set used by an active `plot` server is kept.

## Optional tooling

- [gltfpack](https://github.com/zeux/meshopt) on your `PATH` can help shrink GLB output; meshify enables compression when available.

