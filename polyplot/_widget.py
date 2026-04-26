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
