#!/usr/bin/env python3
"""Build a spatially local subset of sample_data/liver_crop.parquet for the repo.

Picks a seed cell (smallest cell_id by default), keeps the ``n`` cells whose
dissolved centroids are nearest to the seed in the plane, then writes all rows
for those cell_ids (default ``n=50``; use ``-n 500`` for the meshify benchmark
notebook).

Requires a local (untracked) copy of the full file at sample_data/liver_crop.parquet.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np


def _centroids_per_cell(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if gdf.empty:
        raise SystemExit("no valid geometries")
    by = gdf.dissolve(by="cell_id", as_index=True)
    by = by[by.geometry.notna() & ~by.geometry.is_empty]
    cent = by.geometry.centroid
    c = gpd.GeoDataFrame(
        {
            "cell_id": by.index,
            "geometry": cent,
        },
        geometry="geometry",
        crs=gdf.crs,
    )
    c = c.reset_index(drop=True)
    c = c[c.geometry.notna()]
    if c.empty:
        raise SystemExit("failed to build centroids (empty after dissolve)")
    return c


def select_nearest_cell_ids(
    gdf: gpd.GeoDataFrame, n: int, seed: str | int | None
) -> list:
    c = _centroids_per_cell(gdf)
    ids = sorted(c["cell_id"].unique())
    if len(ids) < n:
        raise SystemExit(
            f"only {len(ids)} unique cell_id values; need at least {n}"
        )
    if seed is None:
        seed_id = min(ids, key=str)
    else:
        if seed not in ids:
            raise SystemExit(f"seed cell_id {seed!r} not in data")
        seed_id = seed
    seed_pt = c.loc[c["cell_id"] == seed_id, "geometry"].iloc[0]
    others = c[c["cell_id"] != seed_id].copy()
    # squared distance in map units (avoids sqrt)
    ox = others.geometry.x.values - seed_pt.x
    oy = others.geometry.y.values - seed_pt.y
    dist2 = ox * ox + oy * oy
    order = np.argsort(dist2)[: n - 1]
    take = [seed_id] + others.iloc[order]["cell_id"].tolist()
    return take


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--source",
        type=Path,
        help="Path to full liver_crop.parquet (default: <repo>/sample_data/liver_crop.parquet)",
    )
    p.add_argument(
        "--out",
        type=Path,
        help="Output path (default: <repo>/sample_data/liver_crop_sample.parquet)",
    )
    p.add_argument(
        "-n", "--n-cells", type=int, default=50, help="Number of cell_id values to keep"
    )
    p.add_argument(
        "--seed",
        type=str,
        default=None,
        help="cell_id to start from; default: numerically/lexicographically smallest",
    )
    args = p.parse_args()

    root = Path(__file__).resolve().parent.parent
    src = args.source or (root / "sample_data" / "liver_crop.parquet")
    dst = args.out or (root / "sample_data" / "liver_crop_sample.parquet")

    if not src.is_file():
        print(f"Missing source file: {src}", file=sys.stderr)
        sys.exit(1)

    gdf = gpd.read_parquet(src)
    if "cell_id" not in gdf.columns:
        print("expected column 'cell_id'", file=sys.stderr)
        sys.exit(1)

    keep = select_nearest_cell_ids(gdf, args.n_cells, args.seed)
    keep_set = set(keep)
    sub = gdf[gdf["cell_id"].isin(keep_set)].copy()
    dst.parent.mkdir(parents=True, exist_ok=True)
    sub.to_parquet(dst, index=False)
    nrows = len(sub)
    nbytes = dst.stat().st_size
    print(
        f"Wrote {dst}  ({nrows} rows, {len(keep)} cells, {nbytes / 1024 / 1024:.2f} MiB)"
    )


if __name__ == "__main__":
    main()
