"""Debug exp 232: test bulk array extract timing."""
import time, geopandas as gpd, polyplot as po
from polyplot._preprocess import preprocess_gdf
from polyplot._tile_export import auto_tile_size, compute_tile_grid, _build_tile_data
from polyplot._mesh_build import _cell_max_turning, cell_color, _collect_rings_arrays
from collections import defaultdict
import numpy as np

gdf = gpd.read_parquet("sample_data/liver_crop_sample.parquet")
po.meshify(gdf, "/tmp/_wm_232", use_cache=False)  # warmup Numba

gdf_r = preprocess_gdf(gdf)
cfg = {"smooth_iters": 1}
all_cell_ids = sorted(gdf_r["cell_id"].unique().tolist())
cell_colors = {cid: cell_color(i) for i, cid in enumerate(all_cell_ids)}

t0 = time.perf_counter()
geoms_col = gdf_r.geometry.values
zvals_col = gdf_r["ZIndex"].to_numpy(dtype=np.float64)
cids_col = gdf_r["cell_id"].values
geoms_map: dict = defaultdict(list)
zvals_map: dict = defaultdict(list)
for cid, g, z in zip(cids_col, geoms_col, zvals_col):
    geoms_map[cid].append(g)
    zvals_map[cid].append(z)
t1 = time.perf_counter()

rings_by_cid = {}
zs_by_cid = {}
scores_by_cid = {}
for cid in all_cell_ids:
    r, z = _collect_rings_arrays(
        np.asarray(geoms_map[cid]),
        np.asarray(zvals_map[cid], dtype=np.float64),
        2.0,
    )
    rings_by_cid[cid] = r
    zs_by_cid[cid] = z
    scores_by_cid[cid] = _cell_max_turning(r) if r else 0.0
t2 = time.perf_counter()

ts = auto_tile_size(gdf_r, cfg, 5.0)
assignments = compute_tile_grid(gdf_r, ts)
tc: dict = defaultdict(list)
for cid, key in assignments.items():
    tc[key].append(cid)
sorted_tiles = sorted(tc.items())
t3 = time.perf_counter()

tile_data = [
    (key, cells) + _build_tile_data(cells, rings_by_cid, zs_by_cid, scores_by_cid, cfg, cell_colors)
    for key, cells in sorted_tiles
]
t4 = time.perf_counter()
print(f"bulk_extract={1000*(t1-t0):.0f}ms collect_rings={1000*(t2-t1):.0f}ms "
      f"grid={1000*(t3-t2):.0f}ms build_tiles={1000*(t4-t3):.0f}ms "
      f"n_tiles={len(sorted_tiles)} total={1000*(t4-t0):.0f}ms")
