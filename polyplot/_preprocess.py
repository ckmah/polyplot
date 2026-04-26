from __future__ import annotations

import geopandas as gpd
import pandas as pd

from polyplot import _autoresearch_knobs as _ak


def preprocess_gdf(
    gdf: gpd.GeoDataFrame,
    simplify_tol: float | None = None,
) -> gpd.GeoDataFrame:
    """Simplify geometry.

    Args:
        gdf: Raw GeoDataFrame with columns cell_id, ZIndex, geometry.
        simplify_tol: Shapely simplify tolerance in CRS units (often microns for
            microscopy segmentations); 0 disables.
    """
    if simplify_tol is None:
        simplify_tol = float(_ak.PREPROCESS_SIMPLIFY_TOL)
    out = gdf.copy()
    if simplify_tol > 0:
        out["geometry"] = out["geometry"].simplify(simplify_tol, preserve_topology=True)
    return out
