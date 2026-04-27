from __future__ import annotations

import pathlib

import anywidget
import traitlets


class PolyFiberWidget(anywidget.AnyWidget):
    """Streaming 3D cross-section viewer backed by GLB tile files."""

    _esm = pathlib.Path(__file__).parent / "widget.js"
    _css = pathlib.Path(__file__).parent / "widget.css"

    tile_server_url       = traitlets.Unicode("").tag(sync=True)
    tiles_json_path       = traitlets.Unicode("tiles.json").tag(sync=True)
    bbox                  = traitlets.List(trait=traitlets.Float()).tag(sync=True)
    color                 = traitlets.Unicode("#4aa3ff").tag(sync=True)
    max_concurrent_fetches = traitlets.Int(4).tag(sync=True)
    # Base64-encoded packed float32 XY pairs: [x0,y0,x1,y1,...]
    # This is used only for the 2D minimap overlay.
    centroids_xy_b64      = traitlets.Unicode("").tag(sync=True)

    # JSON-encoded list of cell ids aligned with centroids_xy_b64 pairs.
    centroids_cell_ids_json = traitlets.Unicode("[]").tag(sync=True)

    # On-demand tile streaming mode: enforce a distance cap so only nearby tiles load.
    on_demand             = traitlets.Bool(False).tag(sync=True)

    # Cap zoom-out (OrbitControls max distance). 0 means "derive in JS from bbox".
    max_orbit_distance    = traitlets.Float(0.0).tag(sync=True)
