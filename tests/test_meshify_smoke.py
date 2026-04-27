from __future__ import annotations

import json
import tempfile
from pathlib import Path

import geopandas as gpd
import shapely.geometry as sg

from polyplot._api import meshify


def test_meshify_writes_tiles_json_and_glbs():
    poly = sg.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    gdf = gpd.GeoDataFrame(
        [
            {"cell_id": "x", "ZIndex": 0.0, "geometry": poly},
            {"cell_id": "x", "ZIndex": 1.0, "geometry": poly},
        ],
        geometry="geometry",
    )
    with tempfile.TemporaryDirectory() as td:
        out = meshify(gdf, out_dir=td, smooth=False, use_cache=False, show_progress=False)
        run_dir = Path(out["out_dir"])
        tiles_json = run_dir / "tiles.json"
        assert tiles_json.is_file()
        raw = json.loads(tiles_json.read_text(encoding="utf-8"))
        assert raw["tiles"]
        first = raw["tiles"][0]
        glb_rel = first["glb"]
        glb_path = run_dir / glb_rel
        assert glb_path.is_file()
        assert glb_path.stat().st_size > 64

