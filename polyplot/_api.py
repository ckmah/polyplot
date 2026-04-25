from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast

import geopandas as gpd

if TYPE_CHECKING:
    from marimo._plugins.ui._impl.from_anywidget import anywidget as MarimoAnywidgetUI


class TileRecord(TypedDict):
    """One spatial tile entry from ``tiles.json``."""

    col: int
    row: int
    bbox: list[float]
    center_xy: list[float]
    cell_count: int
    glb: str


class MeshifyInfo(TypedDict):
    """Tile index written by :func:`meshify` (same keys as ``tiles.json`` plus meshify metadata)."""

    version: int
    tile_size_xy: float
    scene_bbox: list[float]
    tiles: list[TileRecord]
    out_dir: str
    _cache_hit: bool


def meshify(
    gdf: gpd.GeoDataFrame,
    out_dir: str | Path = ".polyplot",
    *,
    smooth: bool = True,
    use_cache: bool = True,
    show_progress: bool = True,
) -> MeshifyInfo:
    """Export GLB tiles for ``gdf`` into a content-addressed cache directory.

    Writes::

        <out_dir>/<sha256>/tiles.json
        <out_dir>/<sha256>/tiles/*.glb

    The hash includes geometry, cell ids, Z indices, and the ``smooth`` flag.

    Args:
        gdf: GeoDataFrame with columns ``cell_id``, ``ZIndex``, and ``geometry``.
        out_dir: Root cache directory (default ``.polyplot`` under the process cwd).
        smooth: If ``True``, apply 3D Taubin smoothing (1 iteration); if ``False``, none.
        use_cache: If ``True`` and this fingerprint already exists, load ``tiles.json`` and skip rebuild.
        show_progress: If ``True`` and running in marimo, show a progress bar while building tiles.

    Returns:
        Parsed tile index (``tiles.json``) with ``out_dir`` set to the run directory
        and ``_cache_hit`` indicating whether the cache was reused.

    After each call, older cache shards under ``out_dir`` may be deleted (LRU by
    ``tiles.json`` mtime, capped). The current digest and any shard still served
    by an active :func:`plot` tile server are never removed.
    """
    from polyplot._cache import gdf_cache_key, prune_stale_cache_shards
    from polyplot._preprocess import preprocess_gdf
    from polyplot._tile_export import export_tiles

    root = Path(out_dir).expanduser().resolve()
    fp = gdf_cache_key(gdf, smooth)
    cache_dir = root / fp
    tiles_path = cache_dir / "tiles.json"

    if use_cache and tiles_path.is_file():
        tiles_path.touch()
        raw = json.loads(tiles_path.read_text(encoding="utf-8"))
        info = cast(
            MeshifyInfo,
            {
                "version": int(raw["version"]),
                "tile_size_xy": float(raw["tile_size_xy"]),
                "scene_bbox": [float(x) for x in raw["scene_bbox"]],
                "tiles": raw["tiles"],
                "out_dir": str(cache_dir),
                "_cache_hit": True,
            },
        )
        prune_stale_cache_shards(root, keep_digest=fp)
        return info

    cache_dir.mkdir(parents=True, exist_ok=True)

    gdf_render = preprocess_gdf(gdf)
    cfg = {"smooth_iters": 1 if smooth else 0}
    tiles_info = export_tiles(gdf_render, cfg, cache_dir, show_progress=show_progress)
    out = cast(
        MeshifyInfo,
        {
            "version": int(tiles_info["version"]),
            "tile_size_xy": float(tiles_info["tile_size_xy"]),
            "scene_bbox": [float(x) for x in tiles_info["scene_bbox"]],
            "tiles": tiles_info["tiles"],
            "out_dir": str(cache_dir),
            "_cache_hit": False,
        },
    )
    prune_stale_cache_shards(root, keep_digest=fp)
    return out


def plot(
    gdf: gpd.GeoDataFrame,
    smooth: bool = True,
    max_concurrent_fetches: int = 4,
    use_cache: bool = True,
    show_progress: bool = True,
) -> MarimoAnywidgetUI:
    """Open the 3D viewer for ``gdf``, building from cache or exporting first.

    Calls :func:`meshify` with ``out_dir=".polyplot"`` and ``show_progress=False``.
    ``show_progress=False``. Wireframe, opacity, and background are adjusted in the
    widget toolbar.

    Args:
        gdf: GeoDataFrame with columns ``cell_id``, ``ZIndex``, and ``geometry``.
        smooth: If ``True``, apply 3D Taubin smoothing; if ``False``, none.
        max_concurrent_fetches: Maximum parallel HTTP fetches for tile GLBs.
        use_cache: Forwarded to :func:`meshify`.
        show_progress: Forwarded to :func:`meshify`.

    Returns:
        A marimo ``anywidget`` UI element wrapping :class:`~polyplot.PolyFiberWidget`.
    """
    import marimo as mo
    from polyplot._tile_server import get_or_start
    from polyplot._widget import PolyFiberWidget

    tiles_info = meshify(
        gdf,
        ".polyplot",
        smooth=smooth,
        use_cache=use_cache,
        show_progress=show_progress,
    )
    srv = get_or_start(Path(tiles_info["out_dir"]))
    widget_model = PolyFiberWidget(
        tile_server_url=srv.url,
        tiles_json_path="tiles.json",
        bbox=tiles_info["scene_bbox"],
        max_concurrent_fetches=max_concurrent_fetches,
    )
    return mo.ui.anywidget(widget_model)
