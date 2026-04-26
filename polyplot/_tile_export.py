"""GLB tile export pipeline.

Partitions cells into a spatial XY grid, builds one per-tile GLB file, and
writes a tiles.json spatial index. The exported directory layout:

    <out_dir>/
        tiles.json
        tiles/
            tile_<col>_<row>.glb
            ...
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from polyplot import _tune as _poly_tune
from polyplot._mesh_build import (
    _adaptive_ring_targets_from_scores,
    _cell_max_turning,
    _collect_rings_for_cell,
    build_loft_mesh_from_rings,
    cell_color,
)


# ---------------------------------------------------------------------------
# GLB writing
# ---------------------------------------------------------------------------

def _build_glb_bytes(
    positions_f32: np.ndarray,   # (N, 3) float32
    indices_u32: np.ndarray,     # (M*3,) uint32
    colors_rgb_f32: np.ndarray,  # (N, 3) float32 in [0, 1]
    normals_f32: np.ndarray,     # (N, 3) float32, unit length
) -> bytes:
    """Pack mesh data into a binary GLB (glTF 2.0).

    COLOR_0 is written as VEC4 UNSIGNED_BYTE with normalized=True, which is
    what Three.js GLTFLoader expects for per-vertex vertex colors. NORMAL is
    written as VEC3 FLOAT so the renderer can skip ``computeVertexNormals()``
    (avoids shading artifacts on flat caps once meshopt quantizes positions).
    """
    import pygltflib

    n_verts = len(positions_f32)

    # COLOR_0: VEC4 UNSIGNED_BYTE — alpha channel set to 255 (opaque)
    colors_rgba = np.ones((n_verts, 4), dtype=np.uint8)
    colors_rgba[:, :3] = (colors_rgb_f32 * 255).clip(0, 255).astype(np.uint8)

    # Binary blob layout: [positions][indices][colors][normals] — each part is
    # naturally 4-byte aligned (float32 × 3, uint32, uint8 × 4, float32 × 3).
    pos_blob = positions_f32.tobytes()
    idx_blob = indices_u32.tobytes()
    col_blob = colors_rgba.tobytes()
    nrm_blob = normals_f32.astype(np.float32, copy=False).tobytes()
    blob = pos_blob + idx_blob + col_blob + nrm_blob

    p_off, p_len = 0, len(pos_blob)
    i_off, i_len = p_len, len(idx_blob)
    c_off, c_len = p_len + i_len, len(col_blob)
    n_off, n_len = p_len + i_len + c_len, len(nrm_blob)

    gltf = pygltflib.GLTF2(
        scene=0,
        scenes=[pygltflib.Scene(nodes=[0])],
        nodes=[pygltflib.Node(mesh=0)],
        meshes=[pygltflib.Mesh(primitives=[
            pygltflib.Primitive(
                attributes=pygltflib.Attributes(POSITION=0, COLOR_0=2, NORMAL=3),
                indices=1,
                mode=pygltflib.TRIANGLES,
            )
        ])],
        accessors=[
            pygltflib.Accessor(                      # 0: POSITION
                bufferView=0,
                componentType=pygltflib.FLOAT,
                count=n_verts,
                type=pygltflib.VEC3,
                min=positions_f32.min(axis=0).tolist(),
                max=positions_f32.max(axis=0).tolist(),
            ),
            pygltflib.Accessor(                      # 1: indices
                bufferView=1,
                componentType=pygltflib.UNSIGNED_INT,
                count=int(len(indices_u32)),
                type=pygltflib.SCALAR,
            ),
            pygltflib.Accessor(                      # 2: COLOR_0
                bufferView=2,
                componentType=pygltflib.UNSIGNED_BYTE,
                count=n_verts,
                type=pygltflib.VEC4,
                normalized=True,    # required — without this Three reads 0-255 as-is
            ),
            pygltflib.Accessor(                      # 3: NORMAL
                bufferView=3,
                componentType=pygltflib.FLOAT,
                count=n_verts,
                type=pygltflib.VEC3,
            ),
        ],
        bufferViews=[
            pygltflib.BufferView(
                buffer=0, byteOffset=p_off, byteLength=p_len,
                target=pygltflib.ARRAY_BUFFER,
            ),
            pygltflib.BufferView(
                buffer=0, byteOffset=i_off, byteLength=i_len,
                target=pygltflib.ELEMENT_ARRAY_BUFFER,
            ),
            pygltflib.BufferView(
                buffer=0, byteOffset=c_off, byteLength=c_len,
                target=pygltflib.ARRAY_BUFFER,
            ),
            pygltflib.BufferView(
                buffer=0, byteOffset=n_off, byteLength=n_len,
                target=pygltflib.ARRAY_BUFFER,
            ),
        ],
        buffers=[pygltflib.Buffer(byteLength=len(blob))],
    )
    gltf.set_binary_blob(blob)
    return b"".join(gltf.save_to_bytes())


# ---------------------------------------------------------------------------
# Tile grid partitioning
# ---------------------------------------------------------------------------

def auto_tile_size(gdf_render, cfg: dict, target_tile_mb: float = 100.0) -> float:
    """Compute tile_size_xy so each LOD0 tile is ~target_tile_mb MB.

    Estimates per-cell vertex count from actual slice counts and average ring
    vertex count observed in the input, then derives the tile grid spacing from
    spatial cell density.
    Falls back to 200.0 for degenerate inputs.
    """
    n_cells = gdf_render["cell_id"].nunique()
    if n_cells < 2:
        return 200.0

    avg_slices = float(gdf_render.groupby("cell_id").size().mean())
    try:
        avg_ring_verts = float(
            gdf_render.geometry.apply(lambda g: len(getattr(g, "exterior", ()).coords) - 1).mean()
        )
    except Exception:
        avg_ring_verts = 48.0
    n_verts = max(1.0, avg_ring_verts) * avg_slices
    # positions: 12 B/vert, COLOR_0: 4 B/vert, indices: 2 tris/quad × 3 × 4 B
    bytes_per_cell = n_verts * (12 + 4) + n_verts * 2 * 3 * 4

    target_cells = max(1, int(target_tile_mb * 1024 * 1024 / bytes_per_cell))

    cx = gdf_render.geometry.centroid.x
    cy = gdf_render.geometry.centroid.y
    tmp = gdf_render[["cell_id"]].copy()
    tmp["cx"] = cx.values
    tmp["cy"] = cy.values
    centers = tmp.groupby("cell_id")[["cx", "cy"]].mean()
    x_range = float(centers["cx"].max() - centers["cx"].min())
    y_range = float(centers["cy"].max() - centers["cy"].min())
    scene_area = x_range * y_range
    if scene_area < 1.0:
        return 200.0

    density = n_cells / scene_area
    tile_area = target_cells / density
    return float(np.sqrt(tile_area))


def compute_tile_grid(gdf_render, tile_size_xy: float = 50.0) -> dict:
    """Map each cell_id to a (col, row) tile based on mean centroid XY."""
    cx = gdf_render.geometry.centroid.x
    cy = gdf_render.geometry.centroid.y

    tmp = gdf_render[["cell_id"]].copy()
    tmp["cx"] = cx.values
    tmp["cy"] = cy.values
    centers = tmp.groupby("cell_id")[["cx", "cy"]].mean()

    min_x = float(centers["cx"].min())
    min_y = float(centers["cy"].min())

    assignments: dict = {}
    for cell_id, row in centers.iterrows():
        col = int((row["cx"] - min_x) / tile_size_xy)
        tile_row = int((row["cy"] - min_y) / tile_size_xy)
        assignments[cell_id] = (col, tile_row)

    return assignments


def _build_tile_data(
    cell_ids: list,
    rings_by_cid: dict,
    zs_by_cid: dict,
    scores_by_cid: dict,
    cfg: dict,
    cell_colors: dict,
) -> tuple[bytes, list] | tuple[bytes, None]:
    """Build a merged GLB for the given cell_ids.

    Returns (glb_bytes, bbox) where bbox is [minX, minY, minZ, maxX, maxY, maxZ].
    Returns (b"", None) if no valid cell meshes were produced.
    """
    z_scale = cfg.get("z_scale", 2.0)
    smooth_iters = cfg.get("smooth_iters", 1)
    smooth_factor = cfg.get("smooth_factor", 0.5)
    ring_target = cfg.get("ring_target", 48)
    ring_cb = float(cfg.get("ring_curvature_base", 0.28))
    n_by_cid = _adaptive_ring_targets_from_scores(
        cell_ids,
        scores_by_cid,
        ring_target,
        adaptive=cfg.get("ring_adaptive", True),
        min_mul=float(cfg.get("ring_adaptive_min_mul", 0.55)),
        max_mul=float(cfg.get("ring_adaptive_max_mul", 2.25)),
        exponent=float(cfg.get("ring_adaptive_exponent", 0.75)),
    )

    pos_parts: list[np.ndarray] = []
    idx_parts: list[np.ndarray] = []
    nrm_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    all_bboxes: list[tuple] = []

    def _loft_cell(cid):
        rt = ring_target if n_by_cid is None else n_by_cid[cid]
        pos, idx, nrm, bbox = build_loft_mesh_from_rings(
            rings_by_cid[cid],
            zs_by_cid[cid],
            smooth_iters=smooth_iters,
            smooth_factor=smooth_factor,
            ring_vertex_target=rt,
            ring_curvature_base=ring_cb,
        )
        if pos.shape[0] == 0:
            return None
        nv = pos.shape[0]
        r, g, b = cell_colors[cid]
        rgb = np.empty((nv, 3), dtype=np.float32)
        rgb[:, 0] = r
        rgb[:, 1] = g
        rgb[:, 2] = b
        return cid, pos, idx, nrm, rgb, bbox

    if len(cell_ids) > 1:
        raw = Parallel(**_poly_tune.loft_parallel_kw())(
            delayed(_loft_cell)(cid) for cid in cell_ids
        )
    else:
        raw = [_loft_cell(cell_ids[0])] if cell_ids else []

    by_cid: dict = {}
    for item in raw:
        if item is None:
            continue
        cid, pos, idx, nrm, rgb, bbox = item
        by_cid[cid] = (pos, idx, nrm, rgb, bbox)

    vert_offset = 0
    for cell_id in cell_ids:
        if cell_id not in by_cid:
            continue
        pos, idx, nrm, rgb, bbox = by_cid[cell_id]
        nv = pos.shape[0]
        pos_parts.append(pos)
        idx_parts.append(idx + vert_offset)
        nrm_parts.append(nrm)
        col_parts.append(rgb)
        all_bboxes.append(bbox)
        vert_offset += nv

    if not pos_parts:
        return b"", None

    positions_arr = np.vstack(pos_parts).astype(np.float32, copy=False)
    indices_arr = np.concatenate(idx_parts).astype(np.uint32, copy=False)
    normals_arr = np.vstack(nrm_parts).astype(np.float32, copy=False)
    colors_arr = np.vstack(col_parts).astype(np.float32, copy=False)

    glb = _build_glb_bytes(positions_arr, indices_arr, colors_arr, normals_arr)

    b_arr = np.array(all_bboxes)
    bbox = [
        float(b_arr[:, 0].min()), float(b_arr[:, 1].min()), float(b_arr[:, 2].min()),
        float(b_arr[:, 3].max()), float(b_arr[:, 4].max()), float(b_arr[:, 5].max()),
    ]
    return glb, bbox


# ---------------------------------------------------------------------------
# gltfpack compression (optional)
# ---------------------------------------------------------------------------

def _compress_with_gltfpack(src: Path, dst: Path) -> bool:
    """Compress a GLB with gltfpack meshopt. Returns True on success."""
    if not shutil.which("gltfpack"):
        return False
    cmd = ["gltfpack", "-i", str(src), "-o", str(dst), "-cc", "-kn"]
    result = subprocess.run(cmd, capture_output=True, timeout=120)
    return result.returncode == 0


def _write_glb(
    path: Path,
    data: bytes,
    compress: bool,
) -> None:
    if compress:
        tmp = path.with_suffix(".tmp.glb")
        tmp.write_bytes(data)
        if _compress_with_gltfpack(tmp, path):
            tmp.unlink()
        else:
            tmp.rename(path)
    else:
        path.write_bytes(data)


# ---------------------------------------------------------------------------
# Main export entry point
# ---------------------------------------------------------------------------

def export_tiles(
    gdf_render,
    cfg: dict,
    out_dir: Path,
    tile_size_xy: float | None = None,
    target_tile_mb: float = 100.0,
    compress: bool = True,
    n_jobs: int = -1,
    show_progress: bool = True,
) -> dict:
    """Export spatial tiles as GLB files and write tiles.json.

    tile_size_xy: XY grid cell size in CRS units. Pass None (default) to have
        auto_tile_size() compute a value targeting ~target_tile_mb MB per tile.
    show_progress: When True and running inside a marimo notebook, shows a
        progress bar (processes tiles sequentially). Falls back to parallel
        processing outside marimo.

    Returns the parsed tiles.json dict (also written to <out_dir>/tiles.json).
    """
    out_dir = Path(out_dir)
    tile_dir = out_dir / "tiles"
    tile_dir.mkdir(parents=True, exist_ok=True)

    if tile_size_xy is None:
        tile_size_xy = auto_tile_size(gdf_render, cfg, target_tile_mb)

    all_cell_ids = sorted(gdf_render["cell_id"].unique().tolist())
    cell_colors = {cid: cell_color(i) for i, cid in enumerate(all_cell_ids)}

    # Group once: avoids O(n_cells * n_rows) boolean slicing.
    groups = {cid: df for cid, df in gdf_render.groupby("cell_id", sort=False)}

    # Pre-extract rings once per cell (used for adaptive sizing + meshing).
    z_scale = cfg.get("z_scale", 2.0)
    rings_by_cid: dict = {}
    zs_by_cid: dict = {}
    scores_by_cid: dict = {}
    for cid in all_cell_ids:
        rings, zs = _collect_rings_for_cell(groups[cid], z_scale)
        rings_by_cid[cid] = rings
        zs_by_cid[cid] = zs
        scores_by_cid[cid] = _cell_max_turning(rings) if rings else 0.0

    assignments = compute_tile_grid(gdf_render, tile_size_xy)

    tile_cells: dict = defaultdict(list)
    for cell_id, key in assignments.items():
        tile_cells[key].append(cell_id)

    sorted_tiles = sorted(tile_cells.items())

    def _process_tile(tile_key, cell_ids):
        col, row = tile_key
        glb_bytes, bbox = _build_tile_data(
            cell_ids,
            rings_by_cid,
            zs_by_cid,
            scores_by_cid,
            cfg,
            cell_colors,
        )
        if not glb_bytes:
            return None
        name = f"tile_{col}_{row}.glb"
        _write_glb(tile_dir / name, glb_bytes, compress)

        cx = (bbox[0] + bbox[3]) / 2
        cy = (bbox[1] + bbox[4]) / 2
        return {
            "col": col,
            "row": row,
            "bbox": bbox,
            "center_xy": [cx, cy],
            "cell_count": len(cell_ids),
            "glb": f"tiles/{name}",
        }

    # Detect marimo context for progress bar
    _in_marimo = False
    if show_progress:
        try:
            import marimo as mo
            _in_marimo = mo.running_in_notebook()
        except (ImportError, Exception):
            pass

    if _in_marimo:
        import marimo as mo
        results = []
        with mo.status.progress_bar(
            total=len(sorted_tiles),
            title="Building tiles…",
            remove_on_exit=True,
        ) as bar:
            for key, cells in sorted_tiles:
                results.append(_process_tile(key, cells))
                bar.update()
    else:
        results = Parallel(**_poly_tune.tile_parallel_kw(n_jobs))(
            delayed(_process_tile)(key, cells) for key, cells in sorted_tiles
        )

    tiles = [r for r in results if r is not None]

    if tiles:
        all_bboxes = np.array([t["bbox"] for t in tiles])
        scene_bbox = [
            float(all_bboxes[:, 0].min()), float(all_bboxes[:, 1].min()), float(all_bboxes[:, 2].min()),
            float(all_bboxes[:, 3].max()), float(all_bboxes[:, 4].max()), float(all_bboxes[:, 5].max()),
        ]
    else:
        scene_bbox = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    index = {
        "version": 1,
        "tile_size_xy": tile_size_xy,
        "scene_bbox": scene_bbox,
        "tiles": tiles,
    }

    (out_dir / "tiles.json").write_text(json.dumps(index, separators=(",", ":")))
    return index
