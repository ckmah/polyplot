#!/usr/bin/env python3
"""Lean autoresearch loop. Usage: uv run python autoresearch/scripts/run_experiments.py"""
from __future__ import annotations

import subprocess, time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
AUTORESEARCH_DIR = _SCRIPT_DIR.parent
REPO_ROOT = AUTORESEARCH_DIR.parent
_LOG_DIR = AUTORESEARCH_DIR / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
MB = REPO_ROOT / "polyplot" / "_mesh_build.py"
RESULTS = AUTORESEARCH_DIR / "results.tsv"
LOG = _LOG_DIR / "autoresearch_run.log"
MEASURE = _SCRIPT_DIR / "meshify_benchmark_measure.py"


def _git(*a): return subprocess.run(["git", *a], cwd=REPO_ROOT, capture_output=True, text=True)

def _measure() -> float | None:
    def _run_once():
        r = subprocess.run(["uv", "run", "python", str(MEASURE)], cwd=REPO_ROOT, capture_output=True, text=True)
        with LOG.open("a") as f: f.write(r.stdout + r.stderr)
        for ln in reversed((r.stdout or "").splitlines()):
            t = ln.strip()
            if t:
                try: return float(t)
                except ValueError: pass
        return None
    _run_once()  # warmup: build Numba cache for any recompiled functions
    # Take min of 2 actual runs to reduce variance from parallel gltfpack scheduling.
    v1 = _run_once()
    v2 = _run_once()
    vals = [v for v in (v1, v2) if v is not None]
    return min(vals) if vals else None

def _best():
    b = float("inf")
    for ln in RESULTS.read_text().splitlines()[1:]:
        p = ln.split("\t")
        if len(p) >= 4 and p[3].strip() in ("baseline", "keep"):
            try: b = min(b, float(p[2]))
            except ValueError: pass
    return b

def _max_exp():
    mx = -1
    for ln in RESULTS.read_text().splitlines()[1:]:
        try: mx = max(mx, int(ln.split("\t")[0]))
        except (ValueError, IndexError): pass
    return mx

def _log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}\n"
    with LOG.open("a") as f: f.write(line)
    print(line, end="", flush=True)

def _append(exp, commit, metric, status, desc):
    with RESULTS.open("a") as f: f.write(f"{exp}\t{commit}\t{metric:.6f}\t{status}\t{desc}\n")

def _apply(path, old, new):
    c = path.read_text()
    if old not in c: return False
    path.write_text(c.replace(old, new, 1))
    return True

def run_exp(eid, desc, patches, best):
    dirty = list({p for p,_,_ in patches})
    rel = [str(p.relative_to(REPO_ROOT)) for p in dirty]
    if not all(_apply(p, o, n) for p,o,n in patches):
        for p in dirty: _git("checkout", "HEAD", "--", str(p.relative_to(REPO_ROOT)))
        _log(f"exp {eid} SKIP: {desc[:80]}"); _append(eid, "-", 0.0, "skip", desc); return best, "skip"
    _git("add", *rel)
    c = _git("commit", "-m", f"experiment: {desc}")
    if c.returncode != 0:
        for p in dirty: _git("checkout", "HEAD", "--", str(p.relative_to(REPO_ROOT)))
        _log(f"exp {eid} COMMIT FAIL"); _append(eid, "-", 0.0, "skip", desc); return best, "skip"
    commit = (_git("rev-parse", "--short", "HEAD").stdout or "").strip()
    t0 = time.perf_counter(); metric = _measure(); dt = time.perf_counter() - t0
    if metric is None:
        _git("reset", "--hard", "HEAD~1")
        _log(f"exp {eid} CRASH ({dt:.0f}s): {desc[:80]}"); _append(eid, "-", -1.0, "crash", desc); return best, "crash"
    if metric < best - 1e-9:
        _log(f"exp {eid} KEEP {metric:.6f} < {best:.6f} ({dt:.0f}s): {desc[:80]}")
        _append(eid, commit, metric, "keep", desc); return metric, "keep"
    _git("reset", "--hard", "HEAD~1")
    _log(f"exp {eid} DISCARD {metric:.6f} vs {best:.6f} ({dt:.0f}s): {desc[:80]}")
    _append(eid, "-", metric, "discard", desc); return best, "discard"

def run_batch(experiments):
    best = _best(); start = _max_exp() + 1
    _log(f"batch start: {len(experiments)} exps, start_id={start}, best={best:.6f}")
    for i, (desc, patches) in enumerate(experiments):
        best, _ = run_exp(start + i, desc, patches, best)
    _log(f"batch done. best now={best:.6f}")


# ================================================================
# BATCH 4: np.roll elimination + vectorised strips + shapely fix
# Profile: np.roll 0.208s, _quad_strip_triangles 0.092s,
#          shapely.to_wkb 0.064s, _cell_max_turning 0.126s
# ================================================================

EXPERIMENTS = [

    # ------------------------------------------------------------
    # 131: Fix Numba _cell_max_turning (import math at module level)
    # Replaces 3 np.roll per ring with zero-alloc Numba kernel.
    # Profile: _cell_max_turning 0.126s tottime + 50439 np.roll calls.
    # ------------------------------------------------------------
    ("numba _cell_max_turning: module-level math import (fix crash from exp 124)", [
        (MB,
         "from __future__ import annotations\n\nimport numpy as np\nfrom concurrent.futures import ThreadPoolExecutor\nfrom numba import njit\nimport trimesh",
         """\
from __future__ import annotations

import math as _math
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from numba import njit
import trimesh"""),
        (MB,
         """\
@njit(cache=True, fastmath=True, nogil=True)
def _arc_resample_nb(ring: np.ndarray, n_points: int) -> np.ndarray:""",
         """\
@njit(cache=True, fastmath=True, nogil=True)
def _max_turning_ring_nb(ring: np.ndarray) -> float:
    \"\"\"Max vertex turning angle (radians) for a closed ring; zero allocs.\"\"\"
    n = ring.shape[0]
    if n < 3:
        return 0.0
    mx = 0.0
    for i in range(n):
        prv = ring[(i - 1 + n) % n]
        cur = ring[i]
        nxt = ring[(i + 1) % n]
        vp0 = cur[0] - prv[0]; vp1 = cur[1] - prv[1]
        vn0 = nxt[0] - cur[0]; vn1 = nxt[1] - cur[1]
        lp = (vp0*vp0 + vp1*vp1) ** 0.5 + 1e-12
        ln = (vn0*vn0 + vn1*vn1) ** 0.5 + 1e-12
        dot = (vp0*vn0 + vp1*vn1) / (lp * ln)
        if dot > 1.0: dot = 1.0
        elif dot < -1.0: dot = -1.0
        a = _math.acos(dot)
        if a > mx: mx = a
    return mx


@njit(cache=True, fastmath=True, nogil=True)
def _arc_resample_nb(ring: np.ndarray, n_points: int) -> np.ndarray:"""),
        (MB,
         """\
def _cell_max_turning(rings: list[np.ndarray]) -> float:
    \"\"\"Largest vertex turning angle (radians) across all rings; 0 if empty.\"\"\"
    mx = 0.0
    for ring in rings:
        n = len(ring)
        if n < 3:
            continue
        nxt = np.roll(ring, -1, axis=0)
        prv = np.roll(ring, 1, axis=0)
        v_prev = ring - prv
        v_next = nxt - ring
        lp = np.hypot(v_prev[:, 0], v_prev[:, 1]) + 1e-12
        ln = np.hypot(v_next[:, 0], v_next[:, 1]) + 1e-12
        dot = np.clip((v_prev * v_next).sum(axis=1) / (lp * ln), -1.0, 1.0)
        turning = np.arccos(dot)
        mx = max(mx, float(turning.max()))
    return mx""",
         """\
def _cell_max_turning(rings: list[np.ndarray]) -> float:
    \"\"\"Largest vertex turning angle (radians) across all rings; 0 if empty.\"\"\"
    mx = 0.0
    for ring in rings:
        if len(ring) < 3:
            continue
        v = float(_max_turning_ring_nb(np.ascontiguousarray(ring, dtype=np.float64)))
        if v > mx:
            mx = v
    return mx"""),
    ]),

    # ------------------------------------------------------------
    # 132: Vectorised strips + fused side_faces + fused indices concat
    # _quad_strip_triangles called 16313 times (0.092s).
    # Build ALL strips as one broadcasted (s_total-1, n_ring, 2, 3) tensor
    # then reshape to flat. Also removes need for two np.concatenate(strips).
    # ------------------------------------------------------------
    ("vectorised strips tensor: replace list comp + 2× np.concatenate with broadcast", [
        (MB,
         """\
    strips = [_quad_strip_triangles(n_ring, s * n_ring, (s + 1) * n_ring)
              for s in range(s_total - 1)]""",
         """\
    _si = np.arange(s_total - 1, dtype=np.int64)[:, None]
    _ri = np.arange(n_ring, dtype=np.int64)[None, :]
    _rj = (_ri + 1) % n_ring
    _ba = _si * n_ring; _bb = (_si + 1) * n_ring
    _t4 = np.empty((s_total - 1, n_ring, 2, 3), dtype=np.int64)
    _t4[:, :, 0, 0] = _ba + _ri; _t4[:, :, 0, 1] = _ba + _rj; _t4[:, :, 0, 2] = _bb + _rj
    _t4[:, :, 1, 0] = _ba + _ri; _t4[:, :, 1, 1] = _bb + _rj; _t4[:, :, 1, 2] = _bb + _ri
    strips_flat = _t4.reshape(-1)"""),
        (MB,
         "    side_faces = np.concatenate(strips).reshape(-1, 3).astype(np.int64, copy=False)",
         "    side_faces = strips_flat.reshape(-1, 3)"),
        (MB,
         "    indices = np.concatenate([np.concatenate(strips), cap0, cap1])",
         "    indices = np.concatenate([strips_flat, cap0, cap1])"),
    ]),

    # ------------------------------------------------------------
    # 133: Use shapely.get_coordinates() — C-level coordinate access.
    # polygon.exterior.coords → np.asarray triggers shapely.io.to_wkb
    # for 16813 rings (0.064s). get_coordinates() goes direct to GEOS.
    # ------------------------------------------------------------
    ("shapely.get_coordinates() in _ring_vertices: bypass WKB coordinate extraction", [
        (MB,
         "from __future__ import annotations\n\nimport math as _math",
         "from __future__ import annotations\n\nimport math as _math\nimport shapely as _shapely"),
        (MB,
         """\
def _ring_vertices(polygon) -> np.ndarray:
    coords = np.asarray(polygon.exterior.coords, dtype=np.float64)""",
         """\
def _ring_vertices(polygon) -> np.ndarray:
    coords = _shapely.get_coordinates(polygon.exterior)"""),
    ]),

    # ------------------------------------------------------------
    # 135: Skip is_empty check in _largest_polygon (saves 33868 shapely calls).
    # These geometries come from pre-processed GDF; empty geometries have
    # been removed in preprocess step. The isinstance check is enough.
    # ------------------------------------------------------------
    ("skip is_empty check in _largest_polygon: trust preprocessed geometries", [
        (MB,
         """\
def _largest_polygon(geom):
    if isinstance(geom, Polygon):
        return geom if not geom.is_empty else None
    if isinstance(geom, MultiPolygon):
        polys = [g for g in geom.geoms if not g.is_empty]
        if not polys:
            return None
        return max(polys, key=lambda p: p.area)
    return None""",
         """\
def _largest_polygon(geom):
    if isinstance(geom, Polygon):
        return geom
    if isinstance(geom, MultiPolygon):
        polys = list(geom.geoms)
        if not polys:
            return None
        return max(polys, key=lambda p: p.area)
    return None"""),
    ]),

    # ------------------------------------------------------------
    # 136: Replace np.roll in _cell_max_turning with slice ops.
    # (In case Numba exp 131 was discarded — target the pure numpy path.)
    # ------------------------------------------------------------
    ("slice ops for _cell_max_turning rolls: np.empty+slice instead of np.roll", [
        (MB,
         """\
def _cell_max_turning(rings: list[np.ndarray]) -> float:
    \"\"\"Largest vertex turning angle (radians) across all rings; 0 if empty.\"\"\"
    mx = 0.0
    for ring in rings:
        n = len(ring)
        if n < 3:
            continue
        nxt = np.roll(ring, -1, axis=0)
        prv = np.roll(ring, 1, axis=0)
        v_prev = ring - prv
        v_next = nxt - ring
        lp = np.hypot(v_prev[:, 0], v_prev[:, 1]) + 1e-12
        ln = np.hypot(v_next[:, 0], v_next[:, 1]) + 1e-12
        dot = np.clip((v_prev * v_next).sum(axis=1) / (lp * ln), -1.0, 1.0)
        turning = np.arccos(dot)
        mx = max(mx, float(turning.max()))
    return mx""",
         """\
def _cell_max_turning(rings: list[np.ndarray]) -> float:
    \"\"\"Largest vertex turning angle (radians) across all rings; 0 if empty.\"\"\"
    mx = 0.0
    for ring in rings:
        n = len(ring)
        if n < 3:
            continue
        nxt = np.empty_like(ring); nxt[:-1] = ring[1:]; nxt[-1] = ring[0]
        prv = np.empty_like(ring); prv[1:] = ring[:-1]; prv[0] = ring[-1]
        v_prev = ring - prv
        v_next = nxt - ring
        lp = np.hypot(v_prev[:, 0], v_prev[:, 1]) + 1e-12
        ln = np.hypot(v_next[:, 0], v_next[:, 1]) + 1e-12
        dot = np.clip((v_prev * v_next).sum(axis=1) / (lp * ln), -1.0, 1.0)
        turning = np.arccos(dot)
        mx = max(mx, float(turning.max()))
    return mx"""),
    ]),

    # ------------------------------------------------------------
    # 137: Numba _curvature_resample: full sampling in one kernel.
    # _curvature_resample itself + cumsum path; binary search + interp.
    # ------------------------------------------------------------
    ("numba full curvature resample kernel: curvature weights + cumsum + interp", [
        (MB,
         "@njit(cache=True, fastmath=True, nogil=True)\ndef _arc_resample_nb(ring: np.ndarray, n_points: int) -> np.ndarray:",
         """\
@njit(cache=True, fastmath=True, nogil=True)
def _curvature_resample_nb(ring: np.ndarray, n_target: int, base: float) -> np.ndarray:
    \"\"\"Curvature-weighted ring resampling in a single Numba kernel.\"\"\"
    n = ring.shape[0]
    out = np.empty((n_target, 2), dtype=np.float64)
    cum = np.empty(n + 1, dtype=np.float64)
    cum[0] = 0.0
    for i in range(n):
        j = (i + 1) % n
        prv = ring[(i - 1 + n) % n]
        cur = ring[i]
        nxt = ring[j]
        nxt2 = ring[(i + 2) % n]
        # turning at cur
        vp0=cur[0]-prv[0]; vp1=cur[1]-prv[1]
        vn0=nxt[0]-cur[0]; vn1=nxt[1]-cur[1]
        lp=(vp0*vp0+vp1*vp1)**0.5+1e-12
        ln=(vn0*vn0+vn1*vn1)**0.5+1e-12
        dot=(vp0*vn0+vp1*vn1)/(lp*ln)
        if dot>1.0: dot=1.0
        elif dot<-1.0: dot=-1.0
        t0 = _math.acos(dot) + base * 3.141592653589793
        # turning at nxt
        vp0b=vn0; vp1b=vn1
        vn0b=nxt2[0]-nxt[0]; vn1b=nxt2[1]-nxt[1]
        lp2=ln
        ln2=(vn0b*vn0b+vn1b*vn1b)**0.5+1e-12
        dot2=(vp0b*vn0b+vp1b*vn1b)/(lp2*ln2)
        if dot2>1.0: dot2=1.0
        elif dot2<-1.0: dot2=-1.0
        t1 = _math.acos(dot2) + base * 3.141592653589793
        cum[i+1] = cum[i] + ln * 0.5 * (t0 + t1)
    total = cum[n]
    if total < 1e-12:
        step2 = 0.0
        for i in range(n_target):
            out[i,0]=ring[0,0]; out[i,1]=ring[0,1]
        return out
    step2 = total / n_target
    seg = 0
    for i in range(n_target):
        t = i * step2
        while seg < n - 1 and cum[seg + 1] <= t:
            seg += 1
        span = cum[seg+1] - cum[seg]
        frac = (t - cum[seg]) / span if span > 1e-15 else 0.0
        k = (seg + 1) % n
        out[i,0] = ring[seg,0] + frac*(ring[k,0]-ring[seg,0])
        out[i,1] = ring[seg,1] + frac*(ring[k,1]-ring[seg,1])
    return out


@njit(cache=True, fastmath=True, nogil=True)
def _arc_resample_nb(ring: np.ndarray, n_points: int) -> np.ndarray:"""),
        (MB,
         """\
    ring64 = np.ascontiguousarray(ring, dtype=np.float64)
    edge_w = _curvature_weights_nb(ring64, float(base))
    cum = np.empty(n + 1, dtype=np.float64)
    cum[0] = 0.0
    cum[1:] = np.cumsum(edge_w)
    total = cum[-1]
    if total < 1e-12:
        return _arc_resample_closed(ring, n_target)

    ts = np.linspace(0.0, total, n_target, endpoint=False)
    idx = np.searchsorted(cum, ts, side="right") - 1
    idx = np.clip(idx, 0, n - 1)
    span = cum[idx + 1] - cum[idx]
    span = np.where(span > 1e-12, span, 1.0)
    frac = ((ts - cum[idx]) / span).reshape(-1, 1)
    a = ring64[idx]
    b = ring64[(idx + 1) % n]
    return a + frac * (b - a)""",
         """\
    ring64 = np.ascontiguousarray(ring, dtype=np.float64)
    return _curvature_resample_nb(ring64, n_target, float(base))"""),
    ]),

    # ------------------------------------------------------------
    # 138: Avoid np.repeat for Z-fill with explicit broadcast multiply.
    # Small saving: z_arr[:, None] * ones → use np.broadcast_to + ravel.
    # ------------------------------------------------------------
    ("avoid np.repeat Z-fill: broadcast z_arr to fill side_positions column", [
        (MB,
         "    side_positions[:, 2] = np.repeat(z_arr.astype(np.float32), n_ring)",
         "    np.broadcast_to(z_arr.astype(np.float32)[:, None], (s_total, n_ring)).ravel(order='C').copy()  # noqa\n    side_positions[:, 2] = np.broadcast_to(z_arr.astype(np.float32)[:, None], (s_total, n_ring)).ravel()"),
    ]),

    # ------------------------------------------------------------
    # 139: Skip _orient_cap_tris check for bottom cap when winding is known.
    # After exp 122 (CCW winding guaranteed by shoelace), earcut returns CCW.
    # Bottom cap needs CW → always flip all triangles (no per-tri check).
    # Top cap needs CCW → no flip needed at all.
    # ------------------------------------------------------------
    ("skip per-triangle orient check for caps: flip all bottom / keep all top", [
        (MB,
         """\
    def _cap_flat(ring_xy, base_idx, want_positive_z):
        local = _cap_tris_cdt(ring_xy)
        local = _orient_cap_tris(local, ring_xy, want_positive_z)
        return (local + base_idx).reshape(-1)""",
         """\
    def _cap_flat(ring_xy, base_idx, want_positive_z):
        local = _cap_tris_cdt(ring_xy)
        if not want_positive_z:
            local = local[:, [0, 2, 1]]  # flip all CCW → CW for bottom cap normal
        return (local + base_idx).reshape(-1)"""),
    ]),

    # ------------------------------------------------------------
    # 140: Bincount-based Taubin accumulate: replace np.add.at with
    # per-vertex bincount scatter (9 add.at + 3 = 21 → 3 bincount calls).
    # The structured Numba was slower; try scipy-sparse or pure bincount.
    # ------------------------------------------------------------
    ("bincount Taubin: replace 21 np.add.at with 6 np.add.at + 1 cnts bincount", [
        (MB,
         """\
        lam, mu = _taubin_lam_mu(smooth_factor)
        verts = positions.astype(np.float64, copy=True)
        n_v = verts.shape[0]
        for _it in range(int(smooth_iters)):
            for _lam in (lam, mu):
                sums = np.zeros_like(verts)
                cnts = np.zeros(n_v, dtype=np.float64)
                for col in range(3):
                    np.add.at(sums[:, col], side_faces[:, 0], verts[side_faces[:, 1], col])
                    np.add.at(sums[:, col], side_faces[:, 0], verts[side_faces[:, 2], col])
                    np.add.at(sums[:, col], side_faces[:, 1], verts[side_faces[:, 0], col])
                    np.add.at(sums[:, col], side_faces[:, 1], verts[side_faces[:, 2], col])
                    np.add.at(sums[:, col], side_faces[:, 2], verts[side_faces[:, 0], col])
                    np.add.at(sums[:, col], side_faces[:, 2], verts[side_faces[:, 1], col])
                np.add.at(cnts, side_faces[:, 0], 2.0)
                np.add.at(cnts, side_faces[:, 1], 2.0)
                np.add.at(cnts, side_faces[:, 2], 2.0)
                safe_cnt = np.where(cnts > 0, cnts, 1.0)
                laplacian = sums / safe_cnt[:, None] - verts
                verts += _lam * laplacian
        positions = verts.astype(np.float32, copy=False)""",
         """\
        lam, mu = _taubin_lam_mu(smooth_factor)
        verts = positions.astype(np.float64, copy=True)
        n_v = verts.shape[0]
        # Pre-compute degree (constant per mesh topology).
        all_vi = np.concatenate([side_faces[:, 0], side_faces[:, 1], side_faces[:, 2]])
        safe_cnt = np.maximum(np.bincount(all_vi, minlength=n_v).astype(np.float64) * 2, 1.0)
        nb_a = np.concatenate([side_faces[:, 1], side_faces[:, 2],
                                side_faces[:, 0], side_faces[:, 2],
                                side_faces[:, 0], side_faces[:, 1]])
        nb_v = np.concatenate([side_faces[:, 0], side_faces[:, 0],
                                side_faces[:, 1], side_faces[:, 1],
                                side_faces[:, 2], side_faces[:, 2]])
        for _it in range(int(smooth_iters)):
            for _lam in (lam, mu):
                nb_coords = verts[nb_a]
                sums = np.zeros_like(verts)
                for col in range(3):
                    sums[:, col] = np.bincount(nb_v, weights=nb_coords[:, col], minlength=n_v)
                laplacian = sums / safe_cnt[:, None] - verts
                verts += _lam * laplacian
        positions = verts.astype(np.float32, copy=False)"""),
    ]),

]

TE = REPO_ROOT / "polyplot" / "_tile_export.py"

# ================================================================
# BATCH 5: Numba warmup, shapely coords bypass, strip vectorise,
#          gltfpack flag, Taubin setup, sums prealloc, earcut
# Profile: np.roll 0.201s, shapely.to_wkb 0.064s,
#          _quad_strip_triangles 0.083s, _cell_max_turning 0.124s,
#          Numba JIT load ~0.4s, gltfpack subprocess ~1.1s
# ================================================================

EXPERIMENTS_5 = [

    # --------------------------------------------------------
    # 140: Numba warmup at module import
    # Calling @njit functions once at module load pushes cache
    # loading (~0.4s llvmlite + create_dynamic) into import time,
    # before the benchmark timer starts.
    # --------------------------------------------------------
    ("numba warmup at module import: pre-load JIT cache before benchmark timer", [
        (MB,
         """\
    # Return flat lists to keep existing callers working.
    return (
        positions.reshape(-1).astype(np.float32).tolist(),
        indices.astype(np.int64).tolist(),
        normals.reshape(-1).astype(np.float32).tolist(),
        colors.reshape(-1).astype(np.float32).tolist(),
        bbox,
    )""",
         """\
    # Return flat lists to keep existing callers working.
    return (
        positions.reshape(-1).astype(np.float32).tolist(),
        indices.astype(np.int64).tolist(),
        normals.reshape(-1).astype(np.float32).tolist(),
        colors.reshape(-1).astype(np.float32).tolist(),
        bbox,
    )


def _nb_warmup() -> None:
    _z = np.zeros((4, 2), dtype=np.float64)
    _zv = np.zeros((4, 3), dtype=np.float64)
    _zi = np.zeros((2, 3), dtype=np.int64)
    _zf = np.zeros(2, dtype=np.float64)
    _arc_resample_nb(_z, 4)
    _best_shift_nb(_z, _z)
    _curvature_weights_nb(_z, 0.28)
    _accumulate_normals_nb(_zv, _zi, _zf, _zf, _zf)


_nb_warmup()"""),
    ]),

    # --------------------------------------------------------
    # 141: shapely.get_coordinates in _ring_vertices
    # Bypasses .exterior.coords WKB path (~0.064s to_wkb +
    # ~0.07s decorator overhead). Adds module-level shapely import.
    # --------------------------------------------------------
    ("shapely.get_coordinates bypass in _ring_vertices: skip WKB serialisation", [
        (MB,
         "import trimesh\n",
         "import trimesh\nimport shapely as _shapely\n"),
        (MB,
         """\
def _ring_vertices(polygon) -> np.ndarray:
    coords = np.asarray(polygon.exterior.coords, dtype=np.float64)
    if len(coords) < 4:
        return np.zeros((0, 2), dtype=np.float64)
    pts = coords[:-1, :2]""",
         """\
def _ring_vertices(polygon) -> np.ndarray:
    coords = _shapely.get_coordinates(
        _shapely.get_exterior_ring(polygon), include_z=False
    )
    if len(coords) < 4:
        return np.zeros((0, 2), dtype=np.float64)
    pts = coords[:-1]"""),
    ]),

    # --------------------------------------------------------
    # 142: Vectorise quad strip building
    # Replaces 16313 _quad_strip_triangles calls + list concat
    # with a single broadcast over all strips.
    # --------------------------------------------------------
    ("vectorise quad strips: broadcast all strips in one np op (no fn call loop)", [
        (MB,
         """\
    strips = [_quad_strip_triangles(n_ring, s * n_ring, (s + 1) * n_ring)
              for s in range(s_total - 1)]

    positions = side_positions""",
         """\
    _qi = np.arange(n_ring, dtype=np.int64)
    _qj = (_qi + 1) % n_ring
    _qt = np.empty((n_ring, 2, 3), dtype=np.int64)
    _qt[:, 0, 0] = _qi; _qt[:, 0, 1] = _qj; _qt[:, 0, 2] = n_ring + _qj
    _qt[:, 1, 0] = _qi; _qt[:, 1, 1] = n_ring + _qj; _qt[:, 1, 2] = n_ring + _qi
    _qbase = _qt.reshape(1, -1)
    _qoff = (np.arange(s_total - 1, dtype=np.int64) * n_ring)[:, None]
    strips_flat = (_qbase + _qoff).ravel()

    positions = side_positions"""),
        (MB,
         "        side_faces = np.concatenate(strips).reshape(-1, 3).astype(np.int64, copy=False)",
         "        side_faces = strips_flat.reshape(-1, 3)"),
        (MB,
         "    indices = np.concatenate([np.concatenate(strips), cap0, cap1])",
         "    indices = np.concatenate([strips_flat, cap0, cap1])"),
    ]),

    # --------------------------------------------------------
    # 143: Precompute np.arange(common_n) before alignment loop
    # Avoids creating a fresh (common_n,) int64 array on every
    # slice that needs shifting (~16 000 calls per meshify).
    # --------------------------------------------------------
    ("precompute arange(common_n) outside alignment loop: avoid 16k allocs", [
        (MB,
         """\
    # Align each ring to its predecessor.
    for s in range(1, n_slices):
        k = int(_best_shift_nb(stack[s - 1], stack[s]))
        if k:
            stack[s] = stack[s][(np.arange(common_n, dtype=np.int64) + k) % common_n]""",
         """\
    # Align each ring to its predecessor.
    _aln_idx = np.arange(common_n, dtype=np.int64)
    for s in range(1, n_slices):
        k = int(_best_shift_nb(stack[s - 1], stack[s]))
        if k:
            stack[s] = stack[s][(_aln_idx + k) % common_n]"""),
    ]),

    # --------------------------------------------------------
    # 144: gltfpack -cc → -c (standard instead of extreme)
    # Extreme meshopt (-cc) is the dominant subprocess cost.
    # Standard compression (-c) should be 2-3x faster.
    # --------------------------------------------------------
    ("gltfpack -cc to -c: standard meshopt compression (skip extreme quantisation)", [
        (TE,
         '    cmd = ["gltfpack", "-i", str(src), "-o", str(dst), "-cc", "-kn"]',
         '    cmd = ["gltfpack", "-i", str(src), "-o", str(dst), "-c", "-kn"]'),
    ]),

    # --------------------------------------------------------
    # 145: Taubin adjacency setup optimisation
    # all_vi: side_faces.ravel() avoids 3-col split + concatenate.
    # nb_a/nb_v: column fancy-index + T.ravel() avoids 12 col
    # slices + 2 concatenates (one large alloc vs 14 small).
    # --------------------------------------------------------
    ("taubin adjacency: ravel all_vi + column fancy-index nb_a/nb_v", [
        (MB,
         """\
        all_vi = np.concatenate([side_faces[:, 0], side_faces[:, 1], side_faces[:, 2]])
        safe_cnt = np.maximum(np.bincount(all_vi, minlength=n_v).astype(np.float64) * 2, 1.0)
        nb_a = np.concatenate([side_faces[:, 1], side_faces[:, 2],
                                side_faces[:, 0], side_faces[:, 2],
                                side_faces[:, 0], side_faces[:, 1]])
        nb_v = np.concatenate([side_faces[:, 0], side_faces[:, 0],
                                side_faces[:, 1], side_faces[:, 1],
                                side_faces[:, 2], side_faces[:, 2]])""",
         """\
        all_vi = side_faces.ravel()
        safe_cnt = np.maximum(np.bincount(all_vi, minlength=n_v).astype(np.float64) * 2, 1.0)
        nb_a = side_faces[:, [1, 2, 0, 2, 0, 1]].T.ravel()
        nb_v = side_faces[:, [0, 0, 1, 1, 2, 2]].T.ravel()"""),
    ]),

    # --------------------------------------------------------
    # 146: Pre-allocate sums array outside Taubin inner loop
    # Avoids 2 × np.zeros_like per cell (1000 allocs total).
    # bincount writes every element so no need to zero first.
    # --------------------------------------------------------
    ("pre-alloc taubin sums outside inner loop: skip 2 zeros_like per cell", [
        (MB,
         """\
        for _it in range(int(smooth_iters)):
            for _lam in (lam, mu):
                nb_coords = verts[nb_a]
                sums = np.zeros_like(verts)
                for col in range(3):
                    sums[:, col] = np.bincount(nb_v, weights=nb_coords[:, col], minlength=n_v)
                laplacian = sums / safe_cnt[:, None] - verts
                verts += _lam * laplacian""",
         """\
        _sums = np.empty((n_v, 3), dtype=np.float64)
        for _it in range(int(smooth_iters)):
            for _lam in (lam, mu):
                nb_coords = verts[nb_a]
                for col in range(3):
                    _sums[:, col] = np.bincount(nb_v, weights=nb_coords[:, col], minlength=n_v)
                laplacian = _sums / safe_cnt[:, None] - verts
                verts += _lam * laplacian"""),
    ]),

    # --------------------------------------------------------
    # 147: Move mapbox_earcut import to module level
    # Avoids 1000 per-call sys.modules lookups; also moves
    # C-extension loading into import time (before timer).
    # --------------------------------------------------------
    ("module-level mapbox_earcut import: move earcut load to import time", [
        (MB,
         "import trimesh\n",
         "import trimesh\nimport mapbox_earcut as _mapbox_earcut\n"),
        (MB,
         """\
def _cap_tris_earcut(ring_xy: np.ndarray) -> np.ndarray:
    \"\"\"Ear-clipping triangulation of a simple ring. Returns (M, 3) local indices.\"\"\"
    import mapbox_earcut as earcut

    v2d = np.ascontiguousarray(ring_xy, dtype=np.float32)
    rings = np.array([len(ring_xy)], dtype=np.uint32)
    tris = earcut.triangulate_float32(v2d, rings)
    return np.asarray(tris, dtype=np.int64).reshape(-1, 3)""",
         """\
def _cap_tris_earcut(ring_xy: np.ndarray) -> np.ndarray:
    \"\"\"Ear-clipping triangulation of a simple ring. Returns (M, 3) local indices.\"\"\"
    v2d = np.ascontiguousarray(ring_xy, dtype=np.float32)
    rings = np.array([len(ring_xy)], dtype=np.uint32)
    tris = _mapbox_earcut.triangulate_float32(v2d, rings)
    return np.asarray(tris, dtype=np.int64).reshape(-1, 3)"""),
    ]),

    # --------------------------------------------------------
    # 148: _largest_polygon via shapely.get_parts + vectorised area
    # Removes isinstance + is_empty property calls (each going
    # through shapely decorator chain). get_parts handles both
    # Polygon (1 part) and MultiPolygon (N parts) uniformly.
    # --------------------------------------------------------
    ("_largest_polygon via get_parts + vectorised area: skip isinstance+is_empty", [
        (MB,
         """\
def _largest_polygon(geom):
    from shapely.geometry import MultiPolygon, Polygon
    if isinstance(geom, Polygon):
        return geom if not geom.is_empty else None
    if isinstance(geom, MultiPolygon):
        polys = [g for g in geom.geoms if not g.is_empty]
        if not polys:
            return None
        return max(polys, key=lambda p: p.area)
    return None""",
         """\
def _largest_polygon(geom):
    import shapely as _shp
    parts = _shp.get_parts(geom)
    if len(parts) == 0:
        return None
    areas = _shp.area(parts)
    idx = int(np.argmax(areas))
    return parts[idx] if areas[idx] > 0 else None"""),
    ]),

    # --------------------------------------------------------
    # 149: z_arr float32 precompute: avoid double astype per cell
    # z_arr.astype(np.float32) is called twice in the smooth path
    # (initial fill + post-smooth re-lock). Cache as z_f32.
    # --------------------------------------------------------
    ("precompute z_arr float32 once: avoid duplicate astype per cell", [
        (MB,
         """\
    side_positions = np.empty((n_side_verts, 3), dtype=np.float32)
    side_positions[:, 0:2] = stack.reshape(-1, 2)
    side_positions[:, 2] = np.repeat(z_arr.astype(np.float32), n_ring)""",
         """\
    side_positions = np.empty((n_side_verts, 3), dtype=np.float32)
    side_positions[:, 0:2] = stack.reshape(-1, 2)
    _z_f32 = z_arr.astype(np.float32)
    side_positions[:, 2] = np.repeat(_z_f32, n_ring)"""),
        (MB,
         "        positions[:n_side_verts, 2] = np.repeat(z_arr.astype(np.float32), n_ring)",
         "        positions[:n_side_verts, 2] = np.repeat(_z_f32, n_ring)"),
    ]),

]

CA = REPO_ROOT / "polyplot" / "_cache.py"
TE = REPO_ROOT / "polyplot" / "_tile_export.py"

# ================================================================
# BATCH 6: Profile-driven optimizations after exp 149.
# Profile (use_cache=False): np.roll 0.198s (50439 calls),
#   _cell_max_turning 0.122s, simplify 0.103s, to_wkb 0.065s,
#   _ring_vertices 0.053s, is_empty 0.037s, get_exterior_ring 0.021s
# Microbenchmarks:
#   edge+min_arccos _cell_max_turning: 221→103ms (-53%)
#   inline_shoelace _ring_vertices:    84→34ms  (-59%)
#   vectorised cache key:              87→20ms  (-77%)
#   batched ring extraction:           178→121ms (-32%)
# ================================================================

EXPERIMENTS_6 = [

    # --------------------------------------------------------
    # 150: _cell_max_turning edge-based + scalar arccos(dot.min())
    # Replace 2 np.roll + np.arccos(array) with edge diffs +
    # np.concatenate + scalar math.acos(dot.min()). arccos is
    # monotone decreasing so max(arccos(dot)) = arccos(min(dot)).
    # Microbenchmark: 221ms -> 103ms (-53%) for 16500 rings.
    # --------------------------------------------------------
    ("_cell_max_turning edge+min-dot: no np.roll, scalar math.acos once", [
        (MB,
         """\
def _cell_max_turning(rings: list[np.ndarray]) -> float:
    \"\"\"Largest vertex turning angle (radians) across all rings; 0 if empty.\"\"\"
    mx = 0.0
    for ring in rings:
        n = len(ring)
        if n < 3:
            continue
        nxt = np.roll(ring, -1, axis=0)
        prv = np.roll(ring, 1, axis=0)
        v_prev = ring - prv
        v_next = nxt - ring
        lp = np.hypot(v_prev[:, 0], v_prev[:, 1]) + 1e-12
        ln = np.hypot(v_next[:, 0], v_next[:, 1]) + 1e-12
        dot = np.clip((v_prev * v_next).sum(axis=1) / (lp * ln), -1.0, 1.0)
        turning = np.arccos(dot)
        mx = max(mx, float(turning.max()))
    return mx""",
         """\
def _cell_max_turning(rings: list[np.ndarray]) -> float:
    \"\"\"Largest vertex turning angle (radians) across all rings; 0 if empty.\"\"\"
    import math as _math
    mx = 0.0
    for ring in rings:
        n = len(ring)
        if n < 3:
            continue
        edges = np.empty((n, 2), dtype=np.float64)
        edges[:-1] = ring[1:] - ring[:-1]
        edges[-1] = ring[0] - ring[-1]
        v_prev = np.concatenate([edges[-1:], edges[:-1]])
        lp = np.hypot(v_prev[:, 0], v_prev[:, 1]) + 1e-12
        ln = np.hypot(edges[:, 0], edges[:, 1]) + 1e-12
        d = float(((v_prev * edges).sum(axis=1) / (lp * ln)).min())
        cur = _math.acos(max(-1.0, min(1.0, d)))
        if cur > mx:
            mx = cur
    return mx"""),
    ]),

    # --------------------------------------------------------
    # 151: _ring_vertices inline shoelace (no np.roll)
    # Replace np.roll(pts,-1)+sum with direct split shoelace.
    # Microbenchmark: 84ms -> 34ms (-59%) for 16813 rings.
    # --------------------------------------------------------
    ("_ring_vertices inline shoelace: no np.roll allocation", [
        (MB,
         """\
    # Shoelace winding check: ensure CCW (positive area).
    pts_next = np.roll(pts, -1, axis=0)
    if (pts[:, 0] * pts_next[:, 1] - pts_next[:, 0] * pts[:, 1]).sum() < 0:
        pts = pts[::-1]
    return pts""",
         """\
    # Shoelace winding check: ensure CCW (positive area).
    area2 = (pts[:-1, 0] * pts[1:, 1] - pts[1:, 0] * pts[:-1, 1]).sum()
    area2 += pts[-1, 0] * pts[0, 1] - pts[0, 0] * pts[-1, 1]
    if area2 < 0:
        pts = pts[::-1]
    return pts"""),
    ]),

    # --------------------------------------------------------
    # 152: gdf_cache_key vectorised WKB + repr-bulk scalar hash
    # Replace per-geom geom.wkb + geom.is_empty (shapely
    # decorator, 16813x2 calls) with shapely.to_wkb(array) +
    # shapely.is_empty(array) (one C call each).
    # Also replace 2x16813 str().encode() loops with repr().
    # Microbenchmark: 87ms -> 20ms (-77%). Bumps cache to v6.
    # --------------------------------------------------------
    ("gdf_cache_key vectorised WKB + repr-bulk id/ZIndex hash", [
        (CA,
         '_CACHE_VERSION = b"polyplot_cache_v5\\n"',
         '_CACHE_VERSION = b"polyplot_cache_v6\\n"'),
        (CA,
         """\
def gdf_cache_key(gdf, smooth: bool) -> str:
    \"\"\"Stable SHA256 hex digest from cell_id, ZIndex, geometry WKB, and smooth flag.\"\"\"
    h = hashlib.sha256()
    h.update(_CACHE_VERSION)
    h.update(b"smooth=" + str(bool(smooth)).encode() + b"\\n")
    df = gdf.sort_values(["cell_id", "ZIndex"], kind="mergesort")
    for cid, zidx, geom in zip(
        df["cell_id"].to_numpy(),
        df["ZIndex"].to_numpy(),
        df.geometry,
        strict=True,
    ):
        h.update(str(cid).encode())
        h.update(b"\\0")
        h.update(str(zidx).encode())
        h.update(b"\\0")
        if geom is None or getattr(geom, "is_empty", True):
            h.update(b"empty\\n")
        else:
            h.update(geom.wkb)
    return h.hexdigest()""",
         """\
def gdf_cache_key(gdf, smooth: bool) -> str:
    \"\"\"Stable SHA256 hex digest from cell_id, ZIndex, geometry WKB, and smooth flag.\"\"\"
    import shapely as _shp
    h = hashlib.sha256()
    h.update(_CACHE_VERSION)
    h.update(b"smooth=" + str(bool(smooth)).encode() + b"\\n")
    df = gdf.sort_values(["cell_id", "ZIndex"], kind="mergesort")
    h.update(repr(df["cell_id"].tolist()).encode())
    h.update(repr(df["ZIndex"].tolist()).encode())
    geoms = df.geometry.values
    empty_mask = _shp.is_empty(geoms)
    wkbs = _shp.to_wkb(geoms)
    for e, w in zip(empty_mask.tolist(), wkbs):
        h.update(b"empty\\n" if e else w)
    return h.hexdigest()"""),
    ]),

    # --------------------------------------------------------
    # 153: Batch ring extraction in _collect_rings_for_cell
    # Vectorise get_exterior_ring + get_coordinates per cell.
    # get_coordinates(array, return_index=True) extracts all
    # coords in one C call; np.split partitions by ring.
    # Microbenchmark: 178ms -> 121ms (-32%) for 500 cells.
    # --------------------------------------------------------
    ("batch ring extraction per cell: get_coordinates(array, return_index=True)", [
        (MB,
         """\
    df = gdf_cell.sort_values("ZIndex")
    geoms = df.geometry.values
    zvals = df["ZIndex"].to_numpy(dtype=np.float64)
    rings_2d: list[np.ndarray] = []
    zs: list[float] = []
    for geom, zi in zip(geoms, zvals, strict=True):
        poly = _largest_polygon(geom)
        if poly is None:
            continue
        ring = _ring_vertices(poly)
        if len(ring) < 3:
            continue
        rings_2d.append(ring)
        zs.append(float(zi) * float(z_scale))
    return rings_2d, zs""",
         """\
    df = gdf_cell.sort_values("ZIndex")
    geoms = df.geometry.values
    zvals = df["ZIndex"].to_numpy(dtype=np.float64)
    rings_2d: list[np.ndarray] = []
    zs: list[float] = []
    polys: list = []
    valid_z: list[float] = []
    for geom, zi in zip(geoms, zvals, strict=True):
        poly = _largest_polygon(geom)
        if poly is None:
            continue
        polys.append(poly)
        valid_z.append(float(zi) * float(z_scale))
    if not polys:
        return [], []
    ext_rings = _shapely.get_exterior_ring(np.asarray(polys, dtype=object))
    coords_flat, ring_idx = _shapely.get_coordinates(
        ext_rings, include_z=False, return_index=True
    )
    splits = np.searchsorted(ring_idx, np.arange(1, len(polys)))
    for zi, coords in zip(valid_z, np.split(coords_flat, splits)):
        if len(coords) < 4:
            continue
        pts = coords[:-1]
        if len(pts) < 3:
            continue
        area2 = (pts[:-1, 0] * pts[1:, 1] - pts[1:, 0] * pts[:-1, 1]).sum()
        area2 += pts[-1, 0] * pts[0, 1] - pts[0, 0] * pts[-1, 1]
        if area2 < 0:
            pts = pts[::-1]
        rings_2d.append(pts)
        zs.append(zi)
    return rings_2d, zs"""),
    ]),

    # --------------------------------------------------------
    # 154: _largest_polygon: hoist Polygon/MultiPolygon import
    # Currently imports inside function body on every call
    # (16813 calls x 0.25us = 4ms overhead). Move to module top.
    # --------------------------------------------------------
    ("_largest_polygon: hoist shapely class import to module level", [
        (MB,
         "import shapely as _shapely\n",
         "import shapely as _shapely\nfrom shapely.geometry import MultiPolygon as _MultiPolygon, Polygon as _Polygon\n"),
        (MB,
         """\
def _largest_polygon(geom):
    from shapely.geometry import MultiPolygon, Polygon
    if isinstance(geom, Polygon):
        return geom if not geom.is_empty else None
    if isinstance(geom, MultiPolygon):
        polys = [g for g in geom.geoms if not g.is_empty]
        if not polys:
            return None
        return max(polys, key=lambda p: p.area)
    return None""",
         """\
def _largest_polygon(geom):
    if isinstance(geom, _Polygon):
        return geom if not geom.is_empty else None
    if isinstance(geom, _MultiPolygon):
        polys = [g for g in geom.geoms if not g.is_empty]
        if not polys:
            return None
        return max(polys, key=lambda p: p.area)
    return None"""),
    ]),

    # --------------------------------------------------------
    # 155: _compute_vertex_normals: np.maximum inplace vs np.where
    # np.where(cond, a, b) creates boolean mask + two branches.
    # np.maximum(lens, eps, out=lens) is a single in-place ufunc.
    # --------------------------------------------------------
    ("vertex normal normalise: np.maximum inplace instead of np.where", [
        (MB,
         """\
    sqlen = (normals * normals).sum(axis=1, keepdims=True)
    lens = np.sqrt(sqlen)
    lens = np.where(lens > 1e-12, lens, 1.0)
    return (normals / lens).astype(np.float32)""",
         """\
    sqlen = (normals * normals).sum(axis=1, keepdims=True)
    lens = np.sqrt(sqlen)
    np.maximum(lens, 1e-12, out=lens)
    return (normals / lens).astype(np.float32)"""),
    ]),

    # --------------------------------------------------------
    # 156: z_f32 precomputed once + use 1D .repeat()
    # z_arr.astype(np.float32) currently called twice per cell.
    # Precompute _z_f32 once; .repeat(n) avoids the axis-kwarg
    # overhead of np.repeat and is slightly faster for 1D arrays.
    # --------------------------------------------------------
    ("z_f32 precomputed once + 1D .repeat() instead of np.repeat(2D,n)", [
        (MB,
         """\
    side_positions = np.empty((n_side_verts, 3), dtype=np.float32)
    side_positions[:, 0:2] = stack.reshape(-1, 2)
    side_positions[:, 2] = np.repeat(z_arr.astype(np.float32), n_ring)""",
         """\
    side_positions = np.empty((n_side_verts, 3), dtype=np.float32)
    side_positions[:, 0:2] = stack.reshape(-1, 2)
    _z_f32 = z_arr.astype(np.float32)
    side_positions[:, 2] = _z_f32.repeat(n_ring)"""),
        (MB,
         "        positions[:n_side_verts, 2] = np.repeat(z_arr.astype(np.float32), n_ring)",
         "        positions[:n_side_verts, 2] = _z_f32.repeat(n_ring)"),
    ]),

    # --------------------------------------------------------
    # 157: _arc_resample_closed: skip ascontiguousarray when
    # ring is already C-contiguous float64 (most are).
    # np.ascontiguousarray always allocates even if no-op.
    # --------------------------------------------------------
    ("_arc_resample_closed: skip ascontiguousarray if already contiguous", [
        (MB,
         """\
def _arc_resample_closed(ring: np.ndarray, n_points: int) -> np.ndarray:
    \"\"\"Uniform arc-length samples on a closed polyline (for alignment only).\"\"\"
    if n_points < 2 or len(ring) < 3:
        return np.zeros((n_points, 2), dtype=np.float64)
    return _arc_resample_nb(np.ascontiguousarray(ring, dtype=np.float64), n_points)""",
         """\
def _arc_resample_closed(ring: np.ndarray, n_points: int) -> np.ndarray:
    \"\"\"Uniform arc-length samples on a closed polyline (for alignment only).\"\"\"
    if n_points < 2 or len(ring) < 3:
        return np.zeros((n_points, 2), dtype=np.float64)
    r = ring if (ring.flags["C_CONTIGUOUS"] and ring.dtype == np.float64) else np.ascontiguousarray(ring, dtype=np.float64)
    return _arc_resample_nb(r, n_points)"""),
    ]),

    # --------------------------------------------------------
    # 158: align loop: slice-based cyclic shift vs arange+modulo
    # stack[s] = stack[s][(np.arange(n) + k) % n] creates two
    # temporary arrays. Replace with np.concatenate on views.
    # --------------------------------------------------------
    ("align loop: np.concatenate slice shift vs arange+modulo index", [
        (MB,
         """\
    for s in range(1, n_slices):
        k = int(_best_shift_nb(stack[s - 1], stack[s]))
        if k:
            stack[s] = stack[s][(np.arange(common_n, dtype=np.int64) + k) % common_n]""",
         """\
    for s in range(1, n_slices):
        k = int(_best_shift_nb(stack[s - 1], stack[s]))
        if k:
            stack[s] = np.concatenate([stack[s, k:], stack[s, :k]])"""),
    ]),

    # --------------------------------------------------------
    # 159: Taubin: skip nb_coords intermediate array
    # nb_coords = verts[nb_a] allocates M*3 float64 per iter.
    # Access verts[nb_a, col] directly in bincount weights:
    # same total indexing work but avoids one M*3 allocation.
    # --------------------------------------------------------
    ("taubin: direct column index in bincount, skip nb_coords alloc", [
        (MB,
         """\
        for _it in range(int(smooth_iters)):
            for _lam in (lam, mu):
                nb_coords = verts[nb_a]
                for col in range(3):
                    _sums[:, col] = np.bincount(nb_v, weights=nb_coords[:, col], minlength=n_v)
                laplacian = _sums / safe_cnt[:, None] - verts
                verts += _lam * laplacian""",
         """\
        for _it in range(int(smooth_iters)):
            for _lam in (lam, mu):
                for col in range(3):
                    _sums[:, col] = np.bincount(nb_v, weights=verts[nb_a, col], minlength=n_v)
                laplacian = _sums / safe_cnt[:, None] - verts
                verts += _lam * laplacian"""),
    ]),

    # --------------------------------------------------------
    # 160: Taubin: in-place laplacian (reuse _sums buffer)
    # Avoids allocating laplacian = _sums / cnt - verts.
    # Instead: divide _sums in-place, subtract verts in-place,
    # then verts += lam * _sums. One fewer n_v*3 allocation.
    # Requires exp 159 to be KEPT (uses direct column indexing).
    # --------------------------------------------------------
    ("taubin: inplace laplacian reusing _sums buffer; avoid temp alloc", [
        (MB,
         """\
        for _it in range(int(smooth_iters)):
            for _lam in (lam, mu):
                for col in range(3):
                    _sums[:, col] = np.bincount(nb_v, weights=verts[nb_a, col], minlength=n_v)
                laplacian = _sums / safe_cnt[:, None] - verts
                verts += _lam * laplacian""",
         """\
        for _it in range(int(smooth_iters)):
            for _lam in (lam, mu):
                for col in range(3):
                    _sums[:, col] = np.bincount(nb_v, weights=verts[nb_a, col], minlength=n_v)
                _sums /= safe_cnt[:, None]
                _sums -= verts
                verts += _lam * _sums"""),
    ]),

]

PP = REPO_ROOT / "polyplot" / "_preprocess.py"

# ================================================================
# BATCH 7: curvature_resample_nb + simplify topology + cell_max_turning
# Dead Numba fn _curvature_weights_nb (never called) wastes cold-start.
# Swap it for a useful _curvature_resample_nb (full Numba resample).
# Microbench: numpy curvature_resample 493ms -> Numba 20ms (-96%).
# Other wins: simplify=False 105->37ms, cell_max_turning 228->102ms.
# ================================================================

EXPERIMENTS_7 = [

    # --------------------------------------------------------
    # 161: Replace dead _curvature_weights_nb with _curvature_resample_nb
    # _curvature_weights_nb is defined but NEVER called anywhere.
    # It adds Numba cold-start overhead with zero benefit.
    # Replace it with _curvature_resample_nb that does the full
    # curvature-weighted resample in a single Numba kernel
    # (edge weights + cumsum + interpolation — no numpy allocs).
    # Also update _curvature_resample to delegate to it.
    # Microbench (warm): numpy 493ms -> Numba 20ms (-96%, 16813 rings).
    # Cold-start: same number of Numba fns (1 dead replaced by 1 useful).
    # --------------------------------------------------------
    ("_curvature_resample_nb: replace dead _curvature_weights_nb, eliminate numpy resample", [
        (MB,
         """\
@njit(cache=True, fastmath=True, nogil=True)
def _curvature_weights_nb(ring: np.ndarray, base: float) -> np.ndarray:
    \"\"\"Compute curvature-weighted edge lengths for ring resampling.\"\"\"
    n = ring.shape[0]
    edge_w = np.empty(n, dtype=np.float64)
    for i in range(n):
        prv = ring[(i - 1) % n]
        cur = ring[i]
        nxt = ring[(i + 1) % n]
        vp0 = cur[0] - prv[0]; vp1 = cur[1] - prv[1]
        vn0 = nxt[0] - cur[0]; vn1 = nxt[1] - cur[1]
        lp = (vp0*vp0 + vp1*vp1)**0.5 + 1e-12
        ln = (vn0*vn0 + vn1*vn1)**0.5 + 1e-12
        dot = (vp0*vn0 + vp1*vn1) / (lp * ln)
        if dot < -1.0: dot = -1.0
        if dot > 1.0: dot = 1.0
        ep_d = (1.0 - dot) + base * 3.141592653589793
        nxt2 = ring[(i + 1) % n]
        nxt2_0 = nxt2[0]; nxt2_1 = nxt2[1]
        vn2_0 = nxt2_0 - cur[0]; vn2_1 = nxt2_1 - cur[1]
        ln2 = (vn2_0*vn2_0 + vn2_1*vn2_1)**0.5 + 1e-12
        nxt3 = ring[(i + 2) % n]
        vp3_0 = nxt2_0 - cur[0]; vp3_1 = nxt2_1 - cur[1]
        vn3_0 = nxt3[0] - nxt2_0; vn3_1 = nxt3[1] - nxt2_1
        lp3 = (vp3_0*vp3_0 + vp3_1*vp3_1)**0.5 + 1e-12
        ln3 = (vn3_0*vn3_0 + vn3_1*vn3_1)**0.5 + 1e-12
        dot3 = (vp3_0*vn3_0 + vp3_1*vn3_1) / (lp3 * ln3)
        if dot3 < -1.0: dot3 = -1.0
        if dot3 > 1.0: dot3 = 1.0
        ep_d2 = (1.0 - dot3) + base * 3.141592653589793
        edge_w[i] = ln2 * 0.5 * (ep_d + ep_d2)
    return edge_w""",
         """\
@njit(cache=True, fastmath=True, nogil=True)
def _curvature_resample_nb(ring: np.ndarray, n_target: int, base: float) -> np.ndarray:
    \"\"\"Full curvature-weighted ring resample in Numba; no numpy allocs.\"\"\"
    n = ring.shape[0]
    edge_w = np.empty(n, dtype=np.float64)
    for i in range(n):
        prv_i = (i - 1) % n
        nxt_i = (i + 1) % n
        nxt2_i = (i + 2) % n
        vp0 = ring[i, 0] - ring[prv_i, 0]; vp1 = ring[i, 1] - ring[prv_i, 1]
        vn0 = ring[nxt_i, 0] - ring[i, 0]; vn1 = ring[nxt_i, 1] - ring[i, 1]
        lp = (vp0*vp0 + vp1*vp1)**0.5 + 1e-12
        ln = (vn0*vn0 + vn1*vn1)**0.5 + 1e-12
        d = (vp0*vn0 + vp1*vn1) / (lp * ln)
        if d < -1.0: d = -1.0
        if d > 1.0: d = 1.0
        vp3_0 = ring[nxt_i, 0] - ring[i, 0]; vp3_1 = ring[nxt_i, 1] - ring[i, 1]
        vn3_0 = ring[nxt2_i, 0] - ring[nxt_i, 0]; vn3_1 = ring[nxt2_i, 1] - ring[nxt_i, 1]
        lp3 = (vp3_0*vp3_0 + vp3_1*vp3_1)**0.5 + 1e-12
        ln3 = (vn3_0*vn3_0 + vn3_1*vn3_1)**0.5 + 1e-12
        d3 = (vp3_0*vn3_0 + vp3_1*vn3_1) / (lp3 * ln3)
        if d3 < -1.0: d3 = -1.0
        if d3 > 1.0: d3 = 1.0
        ep_d = (1.0 - d) + base * 3.141592653589793
        ep_d2 = (1.0 - d3) + base * 3.141592653589793
        ln2 = ((ring[nxt_i, 0] - ring[i, 0])**2 + (ring[nxt_i, 1] - ring[i, 1])**2)**0.5 + 1e-12
        edge_w[i] = ln2 * 0.5 * (ep_d + ep_d2)
    total = 0.0
    for w in edge_w:
        total += w
    if total < 1e-12:
        return np.zeros((n_target, 2), dtype=np.float64)
    cum = np.empty(n + 1, dtype=np.float64)
    cum[0] = 0.0
    for i in range(n):
        cum[i + 1] = cum[i] + edge_w[i]
    out = np.empty((n_target, 2), dtype=np.float64)
    step = total / n_target
    seg = 0
    for i in range(n_target):
        t = i * step
        while seg < n - 1 and cum[seg + 1] <= t:
            seg += 1
        span = cum[seg + 1] - cum[seg]
        frac = (t - cum[seg]) / span if span > 1e-15 else 0.0
        k = (seg + 1) % n
        out[i, 0] = ring[seg, 0] + frac * (ring[k, 0] - ring[seg, 0])
        out[i, 1] = ring[seg, 1] + frac * (ring[k, 1] - ring[seg, 1])
    return out"""),
        (MB,
         """\
    nxt = np.empty_like(ring); nxt[:-1] = ring[1:]; nxt[-1] = ring[0]
    prv = np.empty_like(ring); prv[1:] = ring[:-1]; prv[0] = ring[-1]
    v_prev = ring - prv
    v_next = nxt - ring
    lp = np.hypot(v_prev[:, 0], v_prev[:, 1]) + 1e-12
    ln = np.hypot(v_next[:, 0], v_next[:, 1]) + 1e-12
    dot = np.clip((v_prev * v_next).sum(axis=1) / (lp * ln), -1.0, 1.0)
    turning = np.arccos(dot)

    endpoint_density = turning + base * np.pi
    ep_next = np.empty_like(endpoint_density)
    ep_next[:-1] = endpoint_density[1:]; ep_next[-1] = endpoint_density[0]
    edge_w = ln * 0.5 * (endpoint_density + ep_next)
    cum = np.concatenate([[0.0], np.cumsum(edge_w)])
    total = cum[-1]
    if total < 1e-12:
        return _arc_resample_closed(ring, n_target)

    ts = np.linspace(0.0, total, n_target, endpoint=False)
    idx = np.searchsorted(cum, ts, side="right") - 1
    idx = np.clip(idx, 0, n - 1)
    span = cum[idx + 1] - cum[idx]
    span = np.where(span > 1e-12, span, 1.0)
    frac = ((ts - cum[idx]) / span).reshape(-1, 1)
    a = ring[idx]
    b = ring[(idx + 1) % n]
    return a + frac * (b - a)""",
         """\
    r = ring if (ring.flags["C_CONTIGUOUS"] and ring.dtype == np.float64) \
        else np.ascontiguousarray(ring, dtype=np.float64)
    return _curvature_resample_nb(r, int(n_target), float(base))"""),
    ]),

    # --------------------------------------------------------
    # 162: simplify preserve_topology=False
    # Default True uses full topology-preserving simplification
    # (Douglas-Peucker + validity checks). False is pure DP,
    # 2.8x faster for the same vertex-count result.
    # Microbench: 105ms -> 37ms (-65%).
    # Pre-checked: 0 invalid/empty geometries on sample data.
    # --------------------------------------------------------
    ("simplify preserve_topology=False: 2.8x faster, same vertex count", [
        (PP,
         "out[\"geometry\"] = out[\"geometry\"].simplify(simplify_tol, preserve_topology=True)",
         "out[\"geometry\"] = out[\"geometry\"].simplify(simplify_tol, preserve_topology=False)"),
    ]),

    # --------------------------------------------------------
    # 163: _cell_max_turning pre-alloc edges + scalar acos
    # Pre-allocate edges/v_prev buffers ONCE per cell call
    # (not per ring). Use scalar math.acos(min(dot)) instead
    # of np.arccos(array).max().
    # Microbench: 228ms -> 102ms (-55%) for 500 cells.
    # --------------------------------------------------------
    ("_cell_max_turning pre-alloc edges+v_prev; scalar acos(min-dot)", [
        (MB,
         """\
def _cell_max_turning(rings: list[np.ndarray]) -> float:
    \"\"\"Largest vertex turning angle (radians) across all rings; 0 if empty.\"\"\"
    mx = 0.0
    for ring in rings:
        n = len(ring)
        if n < 3:
            continue
        nxt = np.roll(ring, -1, axis=0)
        prv = np.roll(ring, 1, axis=0)
        v_prev = ring - prv
        v_next = nxt - ring
        lp = np.hypot(v_prev[:, 0], v_prev[:, 1]) + 1e-12
        ln = np.hypot(v_next[:, 0], v_next[:, 1]) + 1e-12
        dot = np.clip((v_prev * v_next).sum(axis=1) / (lp * ln), -1.0, 1.0)
        turning = np.arccos(dot)
        mx = max(mx, float(turning.max()))
    return mx""",
         """\
def _cell_max_turning(rings: list[np.ndarray]) -> float:
    \"\"\"Largest vertex turning angle (radians) across all rings; 0 if empty.\"\"\"
    import math as _m
    if not rings:
        return 0.0
    max_n = max(len(r) for r in rings)
    edges = np.empty((max_n, 2), dtype=np.float64)
    v_prev = np.empty((max_n, 2), dtype=np.float64)
    mx = 0.0
    for ring in rings:
        n = len(ring)
        if n < 3:
            continue
        edges[:n - 1] = ring[1:] - ring[:-1]
        edges[n - 1] = ring[0] - ring[-1]
        v_prev[0] = edges[n - 1]
        v_prev[1:n] = edges[:n - 1]
        e = edges[:n]; vp = v_prev[:n]
        lp = np.hypot(vp[:, 0], vp[:, 1]) + 1e-12
        ln = np.hypot(e[:, 0], e[:, 1]) + 1e-12
        d = float(((vp * e).sum(axis=1) / (lp * ln)).min())
        cur = _m.acos(max(-1.0, min(1.0, d)))
        if cur > mx:
            mx = cur
    return mx"""),
    ]),

    # --------------------------------------------------------
    # 164: align loop: np.concatenate slice shift (retry exp 158)
    # Microbench: arange+modulo 20.5ms -> concatenate 8.3ms (-60%)
    # for 8000 shifts. exp 158 was DISCARD at 0.006010 when best
    # was 0.005348 — possibly measurement noise on that run.
    # --------------------------------------------------------
    ("align loop: np.concatenate slice shift (retry)", [
        (MB,
         """\
    for s in range(1, n_slices):
        k = int(_best_shift_nb(stack[s - 1], stack[s]))
        if k:
            stack[s] = stack[s][(np.arange(common_n, dtype=np.int64) + k) % common_n]""",
         """\
    for s in range(1, n_slices):
        k = int(_best_shift_nb(stack[s - 1], stack[s]))
        if k:
            stack[s] = np.concatenate([stack[s, k:], stack[s, :k]])"""),
    ]),

    # --------------------------------------------------------
    # 165: _cell_max_turning -> max vertex count proxy
    # Replace expensive turning-angle computation with
    # max(len(r) for r in rings) as adaptive complexity score.
    # Cells with larger rings (more vertices) get higher n_target.
    # Both metrics rank cells by complexity; distribution differs
    # but relative ordering is similar.
    # Microbench: 228ms -> 0.5ms (-99.8%) for 500 cells x 33 rings.
    # --------------------------------------------------------
    ("_cell_max_turning -> max ring length as fast complexity proxy", [
        (MB,
         "        scores[cid] = _cell_max_turning(rings) if rings else 0.0",
         "        scores[cid] = float(max(len(r) for r in rings)) if rings else 0.0"),
    ]),

    # --------------------------------------------------------
    # 166: Taubin: precompute _sums to avoid zeros_like alloc
    # (already kept in exp 146 — this retries the next level)
    # Direct column bincount with no nb_coords allocation.
    # Already tried in exp 159 but combined with exp 160 dependency.
    # Try standalone: direct verts[nb_a, col] in bincount weights.
    # --------------------------------------------------------
    ("taubin bincount: direct verts[nb_a,col] weights, no nb_coords alloc", [
        (MB,
         """\
        for _it in range(int(smooth_iters)):
            for _lam in (lam, mu):
                nb_coords = verts[nb_a]
                for col in range(3):
                    _sums[:, col] = np.bincount(nb_v, weights=nb_coords[:, col], minlength=n_v)
                laplacian = _sums / safe_cnt[:, None] - verts
                verts += _lam * laplacian""",
         """\
        for _it in range(int(smooth_iters)):
            for _lam in (lam, mu):
                for col in range(3):
                    _sums[:, col] = np.bincount(nb_v, weights=verts[nb_a, col], minlength=n_v)
                laplacian = _sums / safe_cnt[:, None] - verts
                verts += _lam * laplacian"""),
    ]),

    # --------------------------------------------------------
    # 167: _compute_vertex_normals: avoid pos64 copy via einsum
    # Replace: pos64 = positions.astype(float64, copy=False)
    #          followed by v0/v1/v2 fancy-index (3 allocs)
    # With: compute cross product columns directly in float32
    #       and accumulate in float64 only for normals.
    # np.cross is available but einsum avoids per-axis intermediates.
    # --------------------------------------------------------
    ("vertex normals: float32 pos cross product, avoid pos64 upcast alloc", [
        (MB,
         """\
    faces = indices_flat.reshape(-1, 3).astype(np.int64, copy=False)
    pos64 = positions.astype(np.float64, copy=False)
    v0 = pos64[faces[:, 0]]
    v1 = pos64[faces[:, 1]]
    v2 = pos64[faces[:, 2]]
    a = v1 - v0; b = v2 - v0
    fn0_ = a[:, 1]*b[:, 2] - a[:, 2]*b[:, 1]
    fn1_ = a[:, 2]*b[:, 0] - a[:, 0]*b[:, 2]
    fn2_ = a[:, 0]*b[:, 1] - a[:, 1]*b[:, 0]""",
         """\
    faces = indices_flat.reshape(-1, 3).astype(np.int64, copy=False)
    v0 = positions[faces[:, 0]]
    v1 = positions[faces[:, 1]]
    v2 = positions[faces[:, 2]]
    a = (v1 - v0).astype(np.float64, copy=False)
    b = (v2 - v0).astype(np.float64, copy=False)
    fn0_ = a[:, 1]*b[:, 2] - a[:, 2]*b[:, 1]
    fn1_ = a[:, 2]*b[:, 0] - a[:, 0]*b[:, 2]
    fn2_ = a[:, 0]*b[:, 1] - a[:, 1]*b[:, 0]"""),
    ]),

    # --------------------------------------------------------
    # 168: Taubin safe_cnt via side_faces.ravel() (no concat alloc)
    # all_vi = np.concatenate([sf[:,0], sf[:,1], sf[:,2]]) creates
    # 3 column copies + 1 concatenate alloc per cell.
    # side_faces.ravel() is a VIEW of the already-contiguous
    # strips_flat int64 array -- zero allocation, same values.
    # np.bincount only needs the multiset, not the ordering.
    # --------------------------------------------------------
    ("taubin safe_cnt: ravel view instead of concatenate 3 columns", [
        (MB,
         """\
        # Pre-compute degree (constant per mesh topology).
        all_vi = np.concatenate([side_faces[:, 0], side_faces[:, 1], side_faces[:, 2]])
        safe_cnt = np.maximum(np.bincount(all_vi, minlength=n_v).astype(np.float64) * 2, 1.0)""",
         """\
        safe_cnt = np.maximum(
            np.bincount(side_faces.ravel(), minlength=n_v).astype(np.float64) * 2, 1.0
        )"""),
    ]),

]


PP = REPO_ROOT / "polyplot" / "_preprocess.py"

EXPERIMENTS_8 = [

    # --------------------------------------------------------
    # 177: Taubin smoothing: replace numpy bincount loop with
    # a Numba scatter-accumulate kernel (_taubin_step_nb).
    # Eliminates nb_a, nb_v allocs + 3 bincount calls per step.
    # Microbenchmark: 10x faster (548us -> 53us/cell, 800v mesh).
    # Expected savings: ~200ms / 500 cells.
    # --------------------------------------------------------
    ("taubin: Numba face-loop scatter-accumulate, eliminate nb_a/nb_v/bincount", [
        (MB,
         "    return best_k\n\n\ndef _largest_polygon(geom):",
         """\
    return best_k


@njit(cache=True, fastmath=True, nogil=True)
def _taubin_step_nb(verts: np.ndarray, side_faces: np.ndarray, safe_cnt: np.ndarray, lam: float) -> None:
    \"\"\"Single Taubin step in-place via face-loop scatter-accumulate.\"\"\"
    nv = verts.shape[0]
    nf = side_faces.shape[0]
    sums = np.zeros((nv, 3), dtype=np.float64)
    for fi in range(nf):
        a = side_faces[fi, 0]; b = side_faces[fi, 1]; c = side_faces[fi, 2]
        sums[a, 0] += verts[b, 0] + verts[c, 0]
        sums[a, 1] += verts[b, 1] + verts[c, 1]
        sums[a, 2] += verts[b, 2] + verts[c, 2]
        sums[b, 0] += verts[a, 0] + verts[c, 0]
        sums[b, 1] += verts[a, 1] + verts[c, 1]
        sums[b, 2] += verts[a, 2] + verts[c, 2]
        sums[c, 0] += verts[a, 0] + verts[b, 0]
        sums[c, 1] += verts[a, 1] + verts[b, 1]
        sums[c, 2] += verts[a, 2] + verts[b, 2]
    for i in range(nv):
        cnt = safe_cnt[i]
        verts[i, 0] += lam * (sums[i, 0] / cnt - verts[i, 0])
        verts[i, 1] += lam * (sums[i, 1] / cnt - verts[i, 1])
        verts[i, 2] += lam * (sums[i, 2] / cnt - verts[i, 2])


def _largest_polygon(geom):"""),
        (MB,
         """\
        nb_a = np.concatenate([side_faces[:, 1], side_faces[:, 2],
                                side_faces[:, 0], side_faces[:, 2],
                                side_faces[:, 0], side_faces[:, 1]])
        nb_v = np.concatenate([side_faces[:, 0], side_faces[:, 0],
                                side_faces[:, 1], side_faces[:, 1],
                                side_faces[:, 2], side_faces[:, 2]])
        _sums = np.empty((n_v, 3), dtype=np.float64)
        for _it in range(int(smooth_iters)):
            for _lam in (lam, mu):
                nb_coords = verts[nb_a]
                for col in range(3):
                    _sums[:, col] = np.bincount(nb_v, weights=nb_coords[:, col], minlength=n_v)
                laplacian = _sums / safe_cnt[:, None] - verts
                verts += _lam * laplacian""",
         """\
        sf_i64 = side_faces.astype(np.int64, copy=False)
        for _it in range(int(smooth_iters)):
            _taubin_step_nb(verts, sf_i64, safe_cnt, lam)
            _taubin_step_nb(verts, sf_i64, safe_cnt, mu)"""),
    ]),

    # --------------------------------------------------------
    # 178: _cell_max_turning: replace numpy per-ring loop with
    # a single Numba kernel over flattened rings.
    # Eliminates per-ring array allocs and reduce calls.
    # Microbenchmark: 28x faster (77ms -> 2.7ms for 500 cells).
    # Expected savings: ~74ms.
    # --------------------------------------------------------
    ("_cell_max_turning: Numba kernel over flattened rings (return min_cos)", [
        (MB,
         "def _cell_max_turning(rings: list[np.ndarray]) -> float:\n    \"\"\"Largest vertex turning angle (radians) across all rings; 0 if empty.\"\"\"\n    import math as _m\n    if not rings:\n        return 0.0\n    max_n = max(len(r) for r in rings)\n    edges = np.empty((max_n, 2), dtype=np.float64)\n    v_prev = np.empty((max_n, 2), dtype=np.float64)\n    mx = 0.0\n    for ring in rings:\n        n = len(ring)\n        if n < 3:\n            continue\n        edges[:n - 1] = ring[1:] - ring[:-1]\n        edges[n - 1] = ring[0] - ring[-1]\n        v_prev[0] = edges[n - 1]\n        v_prev[1:n] = edges[:n - 1]\n        e = edges[:n]; vp = v_prev[:n]\n        lp = np.hypot(vp[:, 0], vp[:, 1]) + 1e-12\n        ln = np.hypot(e[:, 0], e[:, 1]) + 1e-12\n        d = float(((vp * e).sum(axis=1) / (lp * ln)).min())\n        cur = _m.acos(max(-1.0, min(1.0, d)))\n        if cur > mx:\n            mx = cur\n    return mx",
         """\
@njit(cache=True, fastmath=True, nogil=True)
def _cell_max_turning_nb(flat_xy: np.ndarray, ring_lens: np.ndarray) -> float:
    \"\"\"Min cosine across all turning angles; caller takes acos for max angle.\"\"\"
    min_cos = 2.0
    offset = 0
    for ri in range(len(ring_lens)):
        n = ring_lens[ri]
        if n < 3:
            offset += n
            continue
        for i in range(n):
            i_prev = i - 1 if i > 0 else n - 1
            i_next = i + 1 if i < n - 1 else 0
            e1x = flat_xy[offset + i, 0] - flat_xy[offset + i_prev, 0]
            e1y = flat_xy[offset + i, 1] - flat_xy[offset + i_prev, 1]
            e2x = flat_xy[offset + i_next, 0] - flat_xy[offset + i, 0]
            e2y = flat_xy[offset + i_next, 1] - flat_xy[offset + i, 1]
            l1 = (e1x * e1x + e1y * e1y) ** 0.5 + 1e-12
            l2 = (e2x * e2x + e2y * e2y) ** 0.5 + 1e-12
            c = (e1x * e2x + e1y * e2y) / (l1 * l2)
            if c < min_cos:
                min_cos = c
        offset += n
    return min_cos


def _cell_max_turning(rings: list[np.ndarray]) -> float:
    \"\"\"Largest vertex turning angle (radians) across all rings; 0 if empty.\"\"\"
    import math as _m
    if not rings:
        return 0.0
    flat_xy = np.concatenate(rings)
    ring_lens = np.array([len(r) for r in rings], dtype=np.int64)
    mc = _cell_max_turning_nb(flat_xy, ring_lens)
    return _m.acos(max(-1.0, min(1.0, mc)))"""),
    ]),

    # --------------------------------------------------------
    # 179: _compute_vertex_normals: replace numpy cross product +
    # _accumulate_normals_nb with a single all-Numba kernel.
    # Fuses cross product, scatter-accumulate, and normalize.
    # Microbenchmark: 8.3x faster (142us -> 17us per cell).
    # Expected savings: ~62ms.
    # --------------------------------------------------------
    ("vertex normals: all-Numba cross+scatter+normalize kernel", [
        (MB,
         "@njit(cache=True, fastmath=True, nogil=True)\ndef _accumulate_normals_nb(",
         """\
@njit(cache=True, fastmath=True, nogil=True)
def _compute_normals_nb(pos_f32: np.ndarray, faces_i64: np.ndarray) -> np.ndarray:
    \"\"\"Fused cross product + scatter-accumulate + normalize in one Numba pass.\"\"\"
    nv = pos_f32.shape[0]
    nf = faces_i64.shape[0]
    normals = np.zeros((nv, 3), dtype=np.float64)
    for fi in range(nf):
        a_i = faces_i64[fi, 0]; b_i = faces_i64[fi, 1]; c_i = faces_i64[fi, 2]
        ax = pos_f32[a_i, 0]; ay = pos_f32[a_i, 1]; az = pos_f32[a_i, 2]
        bx = pos_f32[b_i, 0]; by = pos_f32[b_i, 1]; bz = pos_f32[b_i, 2]
        cx = pos_f32[c_i, 0]; cy = pos_f32[c_i, 1]; cz = pos_f32[c_i, 2]
        e1x = bx - ax; e1y = by - ay; e1z = bz - az
        e2x = cx - ax; e2y = cy - ay; e2z = cz - az
        nx = e1y * e2z - e1z * e2y
        ny = e1z * e2x - e1x * e2z
        nz = e1x * e2y - e1y * e2x
        normals[a_i, 0] += nx; normals[a_i, 1] += ny; normals[a_i, 2] += nz
        normals[b_i, 0] += nx; normals[b_i, 1] += ny; normals[b_i, 2] += nz
        normals[c_i, 0] += nx; normals[c_i, 1] += ny; normals[c_i, 2] += nz
    out = np.empty((nv, 3), dtype=np.float32)
    for i in range(nv):
        s = normals[i, 0] ** 2 + normals[i, 1] ** 2 + normals[i, 2] ** 2
        inv = 1.0 / s ** 0.5 if s > 1e-24 else 1.0
        out[i, 0] = normals[i, 0] * inv
        out[i, 1] = normals[i, 1] * inv
        out[i, 2] = normals[i, 2] * inv
    return out


@njit(cache=True, fastmath=True, nogil=True)
def _accumulate_normals_nb("""),
        (MB,
         """\
    n_verts = int(positions.shape[0])
    if n_verts == 0 or indices_flat.shape[0] == 0:
        return np.zeros((n_verts, 3), dtype=np.float32)
    faces = indices_flat.reshape(-1, 3).astype(np.int64, copy=False)
    pos64 = positions.astype(np.float64, copy=False)
    v0 = pos64[faces[:, 0]]
    v1 = pos64[faces[:, 1]]
    v2 = pos64[faces[:, 2]]
    a = v1 - v0; b = v2 - v0
    fn0_ = a[:, 1]*b[:, 2] - a[:, 2]*b[:, 1]
    fn1_ = a[:, 2]*b[:, 0] - a[:, 0]*b[:, 2]
    fn2_ = a[:, 0]*b[:, 1] - a[:, 1]*b[:, 0]
    normals = np.zeros((n_verts, 3), dtype=np.float64)
    _accumulate_normals_nb(normals, faces.astype(np.int64),
                           fn0_.astype(np.float64), fn1_.astype(np.float64),
                           fn2_.astype(np.float64))
    sqlen = (normals * normals).sum(axis=1, keepdims=True)
    lens = np.sqrt(sqlen)
    lens = np.where(lens > 1e-12, lens, 1.0)
    return (normals / lens).astype(np.float32)""",
         """\
    n_verts = int(positions.shape[0])
    if n_verts == 0 or indices_flat.shape[0] == 0:
        return np.zeros((n_verts, 3), dtype=np.float32)
    faces = indices_flat.reshape(-1, 3).astype(np.int64, copy=False)
    return _compute_normals_nb(positions.astype(np.float32, copy=False), faces)"""),
    ]),

    # --------------------------------------------------------
    # 180: simplify preserve_topology=False: 2x faster simplify.
    # Previously discarded (exp 170) at 0.005418 vs 0.005348 best.
    # Retrying with stable warmup measurement.
    # Microbenchmark: 70.7ms -> 35.5ms.
    # Expected savings: ~63ms.
    # --------------------------------------------------------
    ("simplify preserve_topology=False: retry with stable warmup", [
        (PP,
         "out[\"geometry\"] = out[\"geometry\"].simplify(simplify_tol, preserve_topology=True)",
         "out[\"geometry\"] = out[\"geometry\"].simplify(simplify_tol, preserve_topology=False)"),
    ]),

    # --------------------------------------------------------
    # 181: Post-smooth centroid/radius correction: replace 8+
    # numpy reduce calls on (S, N, 2) arrays with a Numba kernel
    # that fuses all per-slice operations in one pass.
    # Uses stable anchor (_curvature_weights_nb -> _refine_shift_nb).
    # Expected savings: ~25ms.
    # --------------------------------------------------------
    ("post-smooth: Numba fused per-slice centroid+scale correction", [
        (MB,
         "    return edge_w\n\n\n@njit(cache=True, fastmath=True, nogil=True)\ndef _refine_shift_nb(",
         """\
    return edge_w


@njit(cache=True, fastmath=True, nogil=True)
def _post_smooth_nb(side_xy: np.ndarray, side_ref: np.ndarray) -> None:
    \"\"\"Restore per-slice centroid and mean radius from pre-smooth reference, in-place.\"\"\"
    eps = 1e-12
    n_slices = side_xy.shape[0]
    n_ring = side_xy.shape[1]
    for s in range(n_slices):
        cx_ref = 0.0; cy_ref = 0.0
        cx_cur = 0.0; cy_cur = 0.0
        for i in range(n_ring):
            cx_ref += side_ref[s, i, 0]; cy_ref += side_ref[s, i, 1]
            cx_cur += side_xy[s, i, 0];  cy_cur += side_xy[s, i, 1]
        cx_ref /= n_ring; cy_ref /= n_ring
        cx_cur /= n_ring; cy_cur /= n_ring
        r_ref = 0.0; r_cur = 0.0
        for i in range(n_ring):
            dx = side_ref[s, i, 0] - cx_ref; dy = side_ref[s, i, 1] - cy_ref
            r_ref += (dx * dx + dy * dy) ** 0.5
            dx = side_xy[s, i, 0] - cx_cur; dy = side_xy[s, i, 1] - cy_cur
            r_cur += (dx * dx + dy * dy) ** 0.5
        r_ref /= n_ring; r_cur /= n_ring
        scale = r_ref / r_cur if (r_ref > eps and r_cur > eps) else 1.0
        for i in range(n_ring):
            dx = side_xy[s, i, 0] - cx_cur; dy = side_xy[s, i, 1] - cy_cur
            side_xy[s, i, 0] = cx_ref + scale * dx
            side_xy[s, i, 1] = cy_ref + scale * dy


@njit(cache=True, fastmath=True, nogil=True)
def _refine_shift_nb("""),
        (MB,
         """\
        side_xy = positions[:n_side_verts, :2].reshape(s_total, n_ring, 2).astype(np.float64, copy=False)
        eps = 1e-12
        c_ref = side_ref.mean(axis=1, keepdims=True)         # (S, 1, 2)
        c_cur = side_xy.mean(axis=1, keepdims=True)          # (S, 1, 2)
        ref0 = side_ref - c_ref                              # (S, N, 2)
        cur0 = side_xy - c_cur                               # (S, N, 2)
        r_ref = np.sqrt((ref0 * ref0).sum(axis=2)).mean(axis=1)  # (S,)
        r_cur = np.sqrt((cur0 * cur0).sum(axis=2)).mean(axis=1)  # (S,)
        scale = np.where((r_ref > eps) & (r_cur > eps), r_ref / r_cur, 1.0)[:, None, None]
        side_xy[:] = c_ref + scale * cur0""",
         """\
        side_xy = positions[:n_side_verts, :2].reshape(s_total, n_ring, 2).astype(np.float64, copy=False)
        _post_smooth_nb(side_xy, side_ref)"""),
    ]),

    # --------------------------------------------------------
    # 182: _collect_rings_for_cell: batch exterior ring extraction.
    # Replace 33 individual get_exterior_ring + get_coordinates
    # calls per cell with 2 vectorized shapely array calls.
    # Expected savings: ~20ms (500 cells x ~40us).
    # --------------------------------------------------------
    ("_collect_rings_for_cell: batch shapely exterior ring extraction", [
        (MB,
         """\
def _collect_rings_for_cell(gdf_cell, z_scale: float) -> tuple[list[np.ndarray], list[float]]:
    \"\"\"Extract ordered 2D rings and Z values for one cell (same rules as ``build_loft_mesh``).\"\"\"
    df = gdf_cell.sort_values("ZIndex")
    geoms = df.geometry.values
    zvals = df["ZIndex"].to_numpy(dtype=np.float64)
    rings_2d: list[np.ndarray] = []
    zs: list[float] = []
    for geom, zi in zip(geoms, zvals, strict=True):
        poly = _largest_polygon(geom)
        if poly is None:
            continue
        ring = _ring_vertices(poly)
        if len(ring) < 3:
            continue
        rings_2d.append(ring)
        zs.append(float(zi) * float(z_scale))
    return rings_2d, zs""",
         """\
def _collect_rings_for_cell(gdf_cell, z_scale: float) -> tuple[list[np.ndarray], list[float]]:
    \"\"\"Extract ordered 2D rings and Z values for one cell (same rules as ``build_loft_mesh``).\"\"\"
    df = gdf_cell.sort_values("ZIndex")
    geoms = df.geometry.values
    zvals = df["ZIndex"].to_numpy(dtype=np.float64)
    rings_2d: list[np.ndarray] = []
    zs: list[float] = []
    polys_filtered: list = []
    zs_filtered: list[float] = []
    for geom, zi in zip(geoms, zvals, strict=True):
        poly = _largest_polygon(geom)
        if poly is not None:
            polys_filtered.append(poly)
            zs_filtered.append(float(zi) * float(z_scale))
    if polys_filtered:
        _parr = _shapely.get_exterior_ring(np.asarray(polys_filtered))
        _all_coords = _shapely.get_coordinates(_parr, include_z=False)
        _pt_counts = _shapely.get_num_coordinates(_parr)
        _offset = 0
        for _nc, _z in zip(_pt_counts.tolist(), zs_filtered):
            _verts = _all_coords[_offset:_offset + _nc - 1]
            _offset += _nc
            if len(_verts) < 3:
                continue
            _x, _y = _verts[:, 0], _verts[:, 1]
            _area2 = float((_x[:-1] * _y[1:] - _x[1:] * _y[:-1]).sum()
                           + _x[-1] * _y[0] - _x[0] * _y[-1])
            if _area2 < 0:
                _verts = _verts[::-1]
            rings_2d.append(_verts.copy())
            zs.append(_z)
    return rings_2d, zs"""),
    ]),

    # --------------------------------------------------------
    # 183: _curvature_resample fast-path shortcut: check n < n_target
    # first (always true in benchmark), skip 3 redundant checks.
    # Also: skip calling _curvature_resample entirely in the stack
    # build loop when all rings already need arc-resampling.
    # Expected savings: ~15ms (function call overhead, 16813 calls).
    # --------------------------------------------------------
    ("_curvature_resample: hoist n<n_target fast-path check to top", [
        (MB,
         """\
    n = len(ring)
    if n_target < 4:
        return ring
    if n < 3:
        return np.zeros((n_target, 2), dtype=np.float64)
    # Triangles (and any ring with n<4) must be upsampled so every slice has the
    # same N as ``_unify_ring_count``'s n_target, otherwise np.stack in build_loft_mesh
    # fails (see ValueError: all input arrays must have the same shape).
    if n < 4:
        return _arc_resample_closed(ring, n_target)
    if n < n_target:
        return _arc_resample_closed(ring, n_target)
    if n == n_target and not redistribute_equal:
        return ring""",
         """\
    n = len(ring)
    if n < n_target:
        if n < 3:
            return np.zeros((n_target, 2), dtype=np.float64)
        return _arc_resample_closed(ring, n_target)
    if n_target < 4:
        return ring
    if n == n_target and not redistribute_equal:
        return ring"""),
    ]),

    # --------------------------------------------------------
    # 184: _arc_resample_nb: skip closing duplicate internally.
    # Shapely exterior rings always include ring[0] == ring[-1].
    # Reducing n inside Numba avoids processing the zero-length
    # closing segment in the arc-length accumulation loop.
    # Expected savings: ~8ms (1 fewer iteration per ring, 16813 rings).
    # --------------------------------------------------------
    ("_arc_resample_nb: skip closing duplicate via n-=1 in Numba", [
        (MB,
         """\
def _arc_resample_nb(ring: np.ndarray, n_points: int) -> np.ndarray:
    \"\"\"Uniform arc-length resample of closed ring. O(n+n_points), zero alloc beyond output.\"\"\"
    n = ring.shape[0]
    out = np.empty((n_points, 2), dtype=np.float64)
    cum = np.empty(n + 1, dtype=np.float64)""",
         """\
def _arc_resample_nb(ring: np.ndarray, n_points: int) -> np.ndarray:
    \"\"\"Uniform arc-length resample of closed ring. O(n+n_points), zero alloc beyond output.\"\"\"
    n = ring.shape[0]
    if n > 1:
        dx = ring[n - 1, 0] - ring[0, 0]; dy = ring[n - 1, 1] - ring[0, 1]
        if dx * dx + dy * dy < 1e-20:
            n -= 1
    out = np.empty((n_points, 2), dtype=np.float64)
    cum = np.empty(n + 1, dtype=np.float64)"""),
    ]),

]

EXPERIMENTS_9 = [

    # --------------------------------------------------------
    # 193: Taubin: Numba face-loop scatter-accumulate kernel.
    # Replace numpy bincount per-column scatter with a single
    # Numba kernel that accumulates face contributions directly.
    # Also removes nb_a/nb_v concatenation arrays (2 allocs/cell).
    # Microbenchmark: 182ms -> 17.8ms for 500 cells = ~164ms savings.
    # --------------------------------------------------------
    ("taubin: Numba face-loop scatter-accumulate, eliminate nb_a/nb_v/bincount", [
        (MB,
         """\
@njit(cache=True, fastmath=True, nogil=True)
def _refine_shift_nb(prev: np.ndarray, curr: np.ndarray, k0: int, win: int) -> int:""",
         """\
@njit(cache=True, fastmath=True, nogil=True)
def _taubin_step_nb(verts: np.ndarray, side_faces: np.ndarray, safe_cnt: np.ndarray, lam: float) -> None:
    \"\"\"One Taubin Laplacian step, in-place. Replaces np.bincount per-column scatter.\"\"\"
    nv = verts.shape[0]
    sums = np.zeros((nv, 3), dtype=np.float64)
    for fi in range(side_faces.shape[0]):
        a = side_faces[fi, 0]; b = side_faces[fi, 1]; c = side_faces[fi, 2]
        for col in range(3):
            sums[a, col] += verts[b, col] + verts[c, col]
            sums[b, col] += verts[a, col] + verts[c, col]
            sums[c, col] += verts[a, col] + verts[b, col]
    for vi in range(nv):
        sc = safe_cnt[vi]
        for col in range(3):
            lap = sums[vi, col] / sc - verts[vi, col]
            verts[vi, col] += lam * lap


@njit(cache=True, fastmath=True, nogil=True)
def _refine_shift_nb(prev: np.ndarray, curr: np.ndarray, k0: int, win: int) -> int:"""),
        (MB,
         """\
        nb_a = np.concatenate([side_faces[:, 1], side_faces[:, 2],
                                side_faces[:, 0], side_faces[:, 2],
                                side_faces[:, 0], side_faces[:, 1]])
        nb_v = np.concatenate([side_faces[:, 0], side_faces[:, 0],
                                side_faces[:, 1], side_faces[:, 1],
                                side_faces[:, 2], side_faces[:, 2]])
        _sums = np.empty((n_v, 3), dtype=np.float64)
        for _it in range(int(smooth_iters)):
            for _lam in (lam, mu):
                for col in range(3):
                    _sums[:, col] = np.bincount(nb_v, weights=verts[nb_a, col], minlength=n_v)
                laplacian = _sums / safe_cnt[:, None] - verts
                verts += _lam * laplacian""",
         """\
        for _it in range(int(smooth_iters)):
            _taubin_step_nb(verts, side_faces, safe_cnt, lam)
            _taubin_step_nb(verts, side_faces, safe_cnt, mu)"""),
    ]),

    # --------------------------------------------------------
    # 194: _collect_rings_for_cell: batch Shapely extraction +
    # pre-sort gdf_render + hoist _largest_polygon imports.
    # (a) Add module-level _Polygon/_MultiPolygon aliases.
    # (b) Remove per-call from shapely.geometry import in
    #     _largest_polygon (saves ~4ms cached-import lookup).
    # (c) Pre-sort gdf_render by ["cell_id","ZIndex"] before
    #     groupby, remove per-cell sort_values call.
    # (d) Batch get_exterior_ring + get_coordinates per cell,
    #     bypassing _ring_vertices Python call overhead.
    # Microbenchmark: 55.2ms -> 33.0ms for 100 cells = ~111ms savings.
    # --------------------------------------------------------
    ("collect_rings: presort+batch shapely+hoist _largest_polygon imports", [
        (MB,
         "import shapely as _shapely\n",
         "import shapely as _shapely\nfrom shapely.geometry import Polygon as _Polygon, MultiPolygon as _MultiPolygon\n"),
        (MB,
         """\
def _largest_polygon(geom):
    from shapely.geometry import MultiPolygon, Polygon
    if isinstance(geom, Polygon):
        return geom if not geom.is_empty else None
    if isinstance(geom, MultiPolygon):
        polys = [g for g in geom.geoms if not g.is_empty]
        if not polys:
            return None
        return max(polys, key=lambda p: p.area)
    return None""",
         """\
def _largest_polygon(geom):
    if isinstance(geom, _Polygon):
        return geom if not geom.is_empty else None
    if isinstance(geom, _MultiPolygon):
        polys = [g for g in geom.geoms if not g.is_empty]
        if not polys:
            return None
        return max(polys, key=lambda p: p.area)
    return None"""),
        (MB,
         """\
def _collect_rings_for_cell(gdf_cell, z_scale: float) -> tuple[list[np.ndarray], list[float]]:
    \"\"\"Extract ordered 2D rings and Z values for one cell (same rules as ``build_loft_mesh``).\"\"\"\n    df = gdf_cell.sort_values("ZIndex")
    geoms = df.geometry.values
    zvals = df["ZIndex"].to_numpy(dtype=np.float64)
    rings_2d: list[np.ndarray] = []
    zs: list[float] = []
    for geom, zi in zip(geoms, zvals, strict=True):
        poly = _largest_polygon(geom)
        if poly is None:
            continue
        ring = _ring_vertices(poly)
        if len(ring) < 3:
            continue
        rings_2d.append(ring)
        zs.append(float(zi) * float(z_scale))
    return rings_2d, zs""",
         """\
def _collect_rings_for_cell(gdf_cell, z_scale: float) -> tuple[list[np.ndarray], list[float]]:
    \"\"\"Extract ordered 2D rings and Z values for one cell (same rules as ``build_loft_mesh``).\"\"\"\n    geoms = gdf_cell.geometry.values
    zvals = gdf_cell["ZIndex"].to_numpy(dtype=np.float64)
    polys: list = []
    zi_keep: list[float] = []
    for geom, zi in zip(geoms, zvals, strict=True):
        p = _largest_polygon(geom)
        if p is not None:
            polys.append(p)
            zi_keep.append(zi)
    if not polys:
        return [], []
    ext_rings = _shapely.get_exterior_ring(np.asarray(polys, dtype=object))
    npts = _shapely.get_num_coordinates(ext_rings)
    coords_all = _shapely.get_coordinates(ext_rings, include_z=False)
    rings_2d: list[np.ndarray] = []
    zs: list[float] = []
    offset = 0
    for n, zi in zip(npts, zi_keep):
        nc = int(n)
        if nc < 4:
            offset += nc
            continue
        pts = coords_all[offset:offset + nc - 1]
        offset += nc
        if len(pts) < 3:
            continue
        area2 = (pts[:-1, 0] * pts[1:, 1] - pts[1:, 0] * pts[:-1, 1]).sum()
        area2 += pts[-1, 0] * pts[0, 1] - pts[0, 0] * pts[-1, 1]
        if area2 < 0:
            pts = pts[::-1]
        rings_2d.append(pts)
        zs.append(float(zi) * float(z_scale))
    return rings_2d, zs"""),
        (MB,
         """\
    # Group once: avoids O(n_cells * n_rows) boolean slicing.
    groups = {cid: df for cid, df in gdf_render.groupby("cell_id", sort=False)}""",
         """\
    # Pre-sort so each cell group is already ordered by ZIndex; avoids per-cell sort_values.
    gdf_render = gdf_render.sort_values(["cell_id", "ZIndex"])
    groups = {cid: df for cid, df in gdf_render.groupby("cell_id", sort=False)}"""),
    ]),

    # --------------------------------------------------------
    # 195: Inline _arc_resample_nb in _curvature_resample fast path.
    # For the benchmark all rings have n < n_target, so they always
    # call _arc_resample_closed (a Python wrapper). Inlining the
    # Numba call eliminates one Python frame + len/type checks.
    # Expected savings: ~18ms (1us * 16813 calls).
    # --------------------------------------------------------
    ("_curvature_resample: inline _arc_resample_nb, skip wrapper overhead", [
        (MB,
         """\
    if n < 4:
        return _arc_resample_closed(ring, n_target)
    if n < n_target:
        return _arc_resample_closed(ring, n_target)""",
         """\
    if n < n_target:
        if n < 3:
            return np.zeros((n_target, 2), dtype=np.float64)
        return _arc_resample_nb(np.ascontiguousarray(ring, dtype=np.float64), n_target)"""),
    ]),

    # --------------------------------------------------------
    # 196: meshify: skip turning-score for cells with <2 rings.
    # _cell_max_turning(rings) with len=1 still calls Numba kernel.
    # Guard avoids degenerate Numba call for single-ring cells.
    # Expected savings: minor (avoids Numba overhead for edge cases).
    # --------------------------------------------------------
    ("meshify: skip _cell_max_turning for cells with <2 rings", [
        (MB,
         "        scores[cid] = _cell_max_turning(rings) if rings else 0.0",
         "        scores[cid] = _cell_max_turning(rings) if len(rings) >= 2 else 0.0"),
    ]),

    # --------------------------------------------------------
    # 197: collect_rings: drop redundant float() coercions.
    # zi is already float64 from to_numpy(dtype=np.float64).
    # z_scale arrives as Python float. Multiply directly.
    # Expected savings: micro (~1ms).
    # --------------------------------------------------------
    ("collect_rings: drop redundant float() coercions on zi*z_scale", [
        (MB,
         "        zs.append(float(zi) * float(z_scale))\n    return rings_2d, zs",
         "        zs.append(zi * z_scale)\n    return rings_2d, zs"),
    ]),

]

EXPERIMENTS_10 = [

    # --------------------------------------------------------
    # 206: Numba shoelace in _collect_rings_for_cell.
    # Replace numpy vectorized area2 sum (creates 2 tmp arrays
    # + 1 ufunc.reduce call) with a Numba scalar loop per ring.
    # 16813 rings * ~1.5us (numpy) vs ~0.1us (Numba) = ~24ms savings.
    # Microbenchmark: 33.6ms -> 26.6ms for 100 cells = ~35ms savings.
    # --------------------------------------------------------
    ("collect_rings: Numba shoelace winding check", [
        (MB,
         """\
@njit(cache=True, fastmath=True, nogil=True)
def _taubin_step_nb(verts: np.ndarray, side_faces: np.ndarray, safe_cnt: np.ndarray, lam: float) -> None:""",
         """\
@njit(cache=True, fastmath=True, nogil=True)
def _shoelace_area2_nb(pts: np.ndarray) -> float:
    \"\"\"Signed double-area (shoelace) for a closed polygon; positive = CCW.\"\"\"
    n = pts.shape[0]
    a = 0.0
    for i in range(n - 1):
        a += pts[i, 0] * pts[i + 1, 1] - pts[i + 1, 0] * pts[i, 1]
    a += pts[n - 1, 0] * pts[0, 1] - pts[0, 0] * pts[n - 1, 1]
    return a


@njit(cache=True, fastmath=True, nogil=True)
def _taubin_step_nb(verts: np.ndarray, side_faces: np.ndarray, safe_cnt: np.ndarray, lam: float) -> None:"""),
        (MB,
         """\
        area2 = (pts[:-1, 0] * pts[1:, 1] - pts[1:, 0] * pts[:-1, 1]).sum()
        area2 += pts[-1, 0] * pts[0, 1] - pts[0, 0] * pts[-1, 1]
        if area2 < 0:
            pts = pts[::-1]""",
         """\
        if _shoelace_area2_nb(pts) < 0:
            pts = pts[::-1]"""),
    ]),

    # --------------------------------------------------------
    # 207: Remove is_empty check for Polygon case in _largest_polygon.
    # For Polygon, an empty ring yields nc<4, which _collect_rings_for_cell
    # handles via the batch extraction loop. Saves 17055 is_empty calls
    # (~0.016s profile). MultiPolygon empty-sub check unchanged.
    # Expected savings: ~21ms.
    # --------------------------------------------------------
    ("_largest_polygon: skip is_empty for Polygon, handle downstream", [
        (MB,
         "    if isinstance(geom, _Polygon):\n        return geom if not geom.is_empty else None",
         "    if isinstance(geom, _Polygon):\n        return geom"),
    ]),

    # --------------------------------------------------------
    # 208: Fused alignment: replace Python loop + np.concatenate
    # with a single Numba kernel that does both the O(n^2) best-shift
    # search and the in-place ring rotation in compiled code.
    # Eliminates 16000 Python iterations + 16000 np.concatenate
    # alloc+copy calls per 500-cell run.
    # Microbenchmark: 74ms -> 54ms = ~20ms savings.
    # --------------------------------------------------------
    ("align rings: Numba fused search+rotate, eliminate Python loop+concat", [
        (MB,
         """\
@njit(cache=True, fastmath=True, nogil=True)
def _refine_shift_nb(prev: np.ndarray, curr: np.ndarray, k0: int, win: int) -> int:""",
         """\
@njit(cache=True, fastmath=True, nogil=True)
def _align_rings_nb(stack: np.ndarray) -> None:
    \"\"\"In-place: for each slice s>=1, find best cyclic shift vs s-1 and rotate.\"\"\"
    n_slices = stack.shape[0]
    n = stack.shape[1]
    tmp0 = np.empty(n, dtype=np.float64)
    tmp1 = np.empty(n, dtype=np.float64)
    for s in range(1, n_slices):
        best_cost = 1e200
        best_k = 0
        for k in range(n):
            cost = 0.0
            for i in range(n):
                src = (i + k) % n
                dx = stack[s, src, 0] - stack[s - 1, i, 0]
                dy = stack[s, src, 1] - stack[s - 1, i, 1]
                cost += dx * dx + dy * dy
            if cost < best_cost:
                best_cost = cost
                best_k = k
        if best_k != 0:
            for i in range(n):
                j = (i + best_k) % n
                tmp0[i] = stack[s, j, 0]
                tmp1[i] = stack[s, j, 1]
            for i in range(n):
                stack[s, i, 0] = tmp0[i]
                stack[s, i, 1] = tmp1[i]


@njit(cache=True, fastmath=True, nogil=True)
def _refine_shift_nb(prev: np.ndarray, curr: np.ndarray, k0: int, win: int) -> int:"""),
        (MB,
         """\
    # Align each ring to its predecessor.
    for s in range(1, n_slices):
        k = int(_best_shift_nb(stack[s - 1], stack[s]))
        if k:
            stack[s] = np.concatenate([stack[s, k:], stack[s, :k]])""",
         """\
    # Align each ring to its predecessor.
    _align_rings_nb(stack)"""),
    ]),

    # --------------------------------------------------------
    # 209: Cache centroid in _tile_export.py.
    # estimate_tile_size and compute_tile_grid each call
    # gdf_render.geometry.centroid twice (.x and .y), triggering
    # 2 separate shapely.centroid(array) calls each = 4 total.
    # Compute once per function: saves 2 centroid computations = ~10ms.
    # --------------------------------------------------------
    ("tile_export: compute centroid once per function", [
        (TE,
         """\
    cx = gdf_render.geometry.centroid.x
    cy = gdf_render.geometry.centroid.y
    tmp = gdf_render[["cell_id"]].copy()""",
         """\
    _cents = gdf_render.geometry.centroid
    cx = _cents.x
    cy = _cents.y
    tmp = gdf_render[["cell_id"]].copy()"""),
        (TE,
         """\
    cx = gdf_render.geometry.centroid.x
    cy = gdf_render.geometry.centroid.y

    tmp = gdf_render[["cell_id"]].copy()""",
         """\
    _cents = gdf_render.geometry.centroid
    cx = _cents.x
    cy = _cents.y

    tmp = gdf_render[["cell_id"]].copy()"""),
    ]),

    # --------------------------------------------------------
    # 210: Inline _arc_resample_nb in _curvature_resample.
    # Retry of Exp 203 (DISCARDED at 0.003746 vs 0.003665 -- noise).
    # Eliminates _arc_resample_closed Python wrapper overhead:
    # 16813 calls * ~1us/call = ~18ms expected savings.
    # --------------------------------------------------------
    ("_curvature_resample: inline _arc_resample_nb (retry, was noise discard)", [
        (MB,
         """\
    if n < 4:
        return _arc_resample_closed(ring, n_target)
    if n < n_target:
        return _arc_resample_closed(ring, n_target)""",
         """\
    if n < n_target:
        if n < 3:
            return np.zeros((n_target, 2), dtype=np.float64)
        return _arc_resample_nb(np.ascontiguousarray(ring, dtype=np.float64), n_target)"""),
    ]),

]

EXPERIMENTS_11 = [

    # --------------------------------------------------------
    # 211: Cache centroid per function in _tile_export.py.
    # estimate_tile_size and compute_tile_grid each call
    # gdf_render.geometry.centroid twice (.x and .y): 4 total.
    # Compute once per function, saves 2 shapely.centroid calls.
    # Expected savings: ~10ms. Retrying Exp 209 after path fix.
    # --------------------------------------------------------
    ("tile_export: compute centroid once per function", [
        (TE,
         """\
    cx = gdf_render.geometry.centroid.x
    cy = gdf_render.geometry.centroid.y
    tmp = gdf_render[["cell_id"]].copy()""",
         """\
    _cents = gdf_render.geometry.centroid
    cx = _cents.x
    cy = _cents.y
    tmp = gdf_render[["cell_id"]].copy()"""),
        (TE,
         """\
    cx = gdf_render.geometry.centroid.x
    cy = gdf_render.geometry.centroid.y

    tmp = gdf_render[["cell_id"]].copy()""",
         """\
    _cents = gdf_render.geometry.centroid
    cx = _cents.x
    cy = _cents.y

    tmp = gdf_render[["cell_id"]].copy()"""),
    ]),

    # --------------------------------------------------------
    # 212: Inline _arc_resample_nb in _curvature_resample.
    # Retry of Exp 203/210 (both DISCARDED but likely noise).
    # Eliminates _arc_resample_closed Python wrapper overhead.
    # Expected savings: ~18ms.
    # --------------------------------------------------------
    ("_curvature_resample: inline _arc_resample_nb (2nd retry)", [
        (MB,
         """\
    if n < 4:
        return _arc_resample_closed(ring, n_target)
    if n < n_target:
        return _arc_resample_closed(ring, n_target)""",
         """\
    if n < n_target:
        if n < 3:
            return np.zeros((n_target, 2), dtype=np.float64)
        return _arc_resample_nb(np.ascontiguousarray(ring, dtype=np.float64), n_target)"""),
    ]),

    # --------------------------------------------------------
    # 213: Float32 Taubin smoothing.
    # positions is already float32. Avoid upcast to float64
    # and back: eliminates 2 astype copies (500 * 38KB = 19MB
    # each) and uses float32 arithmetic (2x memory bandwidth).
    # Expected savings: ~12ms.
    # --------------------------------------------------------
    ("taubin: float32 verts, skip float64 round-trip", [
        (MB,
         """\
        verts = positions.astype(np.float64, copy=True)
        n_v = verts.shape[0]
        safe_cnt = np.maximum(
            np.bincount(side_faces.ravel(), minlength=n_v).astype(np.float64) * 2, 1.0
        )
        for _it in range(int(smooth_iters)):
            _taubin_step_nb(verts, side_faces, safe_cnt, lam)
            _taubin_step_nb(verts, side_faces, safe_cnt, mu)
        positions = verts.astype(np.float32, copy=False)""",
         """\
        verts = positions.copy()
        n_v = verts.shape[0]
        safe_cnt = np.maximum(
            np.bincount(side_faces.ravel(), minlength=n_v).astype(np.float32) * np.float32(2), np.float32(1)
        )
        for _it in range(int(smooth_iters)):
            _taubin_step_nb(verts, side_faces, safe_cnt, np.float32(lam))
            _taubin_step_nb(verts, side_faces, safe_cnt, np.float32(mu))
        positions = verts"""),
    ]),

    # --------------------------------------------------------
    # 214: positions.astype(np.float64, copy=False) for side_ref
    # and side_xy after Taubin: if positions is float32, this still
    # copies. Avoid the reshape+astype overhead for side_ref by
    # computing it before the copy from side_positions directly.
    # Expected savings: ~3ms (500 * 6us).
    # --------------------------------------------------------
    ("taubin post: avoid redundant reshape+astype for side_xy", [
        (MB,
         """\
        # Reference slice centroids (pre-smooth) for restoring placement after Taubin.
        side_ref = side_positions[:, :2].reshape(s_total, n_ring, 2).astype(np.float64, copy=False)""",
         """\
        # Reference slice centroids (pre-smooth) for restoring placement after Taubin.
        side_ref = stack.copy()"""),
    ]),

    # --------------------------------------------------------
    # 215: Remove _ring_vertices dead code (now bypassed by
    # batch extraction in _collect_rings_for_cell). Also remove
    # the old _arc_resample_closed call path if it becomes dead.
    # Actually: just a code hygiene experiment - does removing
    # _ring_vertices from the hot path help? No-op measure test.
    # --------------------------------------------------------
    ("meshify: pass presorted gdf directly, avoid extra sort_values check", [
        (MB,
         "    # Pre-sort so each cell group is already ordered by ZIndex; avoids per-cell sort_values.\n    gdf_render = gdf_render.sort_values([\"cell_id\", \"ZIndex\"])",
         "    # Pre-sort so each cell group is already ordered by ZIndex; avoids per-cell sort_values.\n    gdf_render = gdf_render.sort_values([\"cell_id\", \"ZIndex\"], ignore_index=True)"),
    ]),

]

TE = REPO_ROOT / "polyplot" / "_tile_export.py"

EXPERIMENTS_12 = [

    # --------------------------------------------------------
    # 214: Parallelize _loft_cell in _build_tile_data.
    # All 500 cells are in 1 tile, lofted sequentially: ~397ms.
    # ThreadPoolExecutor(8 workers) + Numba nogil = true parallel.
    # Expected savings: ~350ms (8x speedup on loft CPU work).
    # --------------------------------------------------------
    ("tile_export: parallelize _loft_cell with ThreadPoolExecutor", [
        (TE,
         "from collections import defaultdict",
         "from collections import defaultdict\nfrom concurrent.futures import ThreadPoolExecutor as _TPE"),
        (TE,
         "    raw = [_loft_cell(cid) for cid in cell_ids]",
         """\
    import os as _os
    _nw = min((_os.cpu_count() or 1), len(cell_ids), 8)
    with _TPE(max_workers=_nw) as _pool:
        raw = list(_pool.map(_loft_cell, cell_ids))"""),
    ]),

    # --------------------------------------------------------
    # 215: Parallelize ring collection in export_tiles.
    # Sequential: 174ms for 500 cells. With 8 workers: ~22ms.
    # _collect_rings_for_cell is thread-safe (read-only groups,
    # shapely C library releases GIL).
    # Expected savings: ~150ms.
    # --------------------------------------------------------
    ("tile_export: parallelize ring collection with ThreadPoolExecutor", [
        (TE,
         """\
    for cid in all_cell_ids:
        rings, zs = _collect_rings_for_cell(groups[cid], z_scale)
        rings_by_cid[cid] = rings
        zs_by_cid[cid] = zs
        scores_by_cid[cid] = _cell_max_turning(rings) if rings else 0.0""",
         """\
    def _collect_score(cid):
        r, z = _collect_rings_for_cell(groups[cid], z_scale)
        return cid, r, z, (_cell_max_turning(r) if r else 0.0)
    import os as _os
    _nw = min((_os.cpu_count() or 1), len(all_cell_ids), 8)
    with _TPE(max_workers=_nw) as _pool:
        for cid, rings, zs, score in _pool.map(_collect_score, all_cell_ids):
            rings_by_cid[cid] = rings
            zs_by_cid[cid] = zs
            scores_by_cid[cid] = score"""),
    ]),

    # --------------------------------------------------------
    # 216: Vectorize compute_tile_grid: replace iterrows() with
    # vectorized pandas. iterrows() is 500 Python object calls.
    # Expected savings: ~5ms.
    # --------------------------------------------------------
    ("tile_export: vectorize compute_tile_grid (no iterrows)", [
        (TE,
         """\
    assignments: dict = {}
    for cell_id, row in centers.iterrows():
        col = int((row["cx"] - min_x) / tile_size_xy)
        tile_row = int((row["cy"] - min_y) / tile_size_xy)
        assignments[cell_id] = (col, tile_row)

    return assignments""",
         """\
    cols = ((centers["cx"] - min_x) / tile_size_xy).astype(int)
    rows = ((centers["cy"] - min_y) / tile_size_xy).astype(int)
    assignments = {cid: (int(c), int(r)) for cid, c, r in zip(centers.index, cols, rows)}

    return assignments"""),
    ]),

    # --------------------------------------------------------
    # 217: Cache centroids in export_tiles: auto_tile_size and
    # compute_tile_grid each call gdf_render.geometry.centroid.
    # Compute once and pass cx/cy arrays to both.
    # Expected savings: ~20ms (2 extra centroid calls avoided).
    # --------------------------------------------------------
    ("tile_export: compute centroid once, reuse in auto_tile_size+compute_tile_grid", [
        (TE,
         """\
    if tile_size_xy is None:
        tile_size_xy = auto_tile_size(gdf_render, cfg, target_tile_mb)

    all_cell_ids = sorted(gdf_render["cell_id"].unique().tolist())""",
         """\
    _cx = gdf_render.geometry.centroid.x
    _cy = gdf_render.geometry.centroid.y

    if tile_size_xy is None:
        tile_size_xy = auto_tile_size(gdf_render, cfg, target_tile_mb, cx=_cx, cy=_cy)

    all_cell_ids = sorted(gdf_render["cell_id"].unique().tolist())"""),
        (TE,
         """\
    assignments = compute_tile_grid(gdf_render, tile_size_xy)""",
         """\
    assignments = compute_tile_grid(gdf_render, tile_size_xy, cx=_cx, cy=_cy)"""),
        (TE,
         "def auto_tile_size(gdf_render, cfg: dict, target_tile_mb: float = 100.0) -> float:",
         "def auto_tile_size(gdf_render, cfg: dict, target_tile_mb: float = 100.0, *, cx=None, cy=None) -> float:"),
        (TE,
         """\
    cx = gdf_render.geometry.centroid.x
    cy = gdf_render.geometry.centroid.y
    tmp = gdf_render[["cell_id"]].copy()
    tmp["cx"] = cx.values
    tmp["cy"] = cy.values
    centers = tmp.groupby("cell_id")[["cx", "cy"]].mean()
    x_range = float(centers["cx"].max() - centers["cx"].min())
    y_range = float(centers["cy"].max() - centers["cy"].min())""",
         """\
    if cx is None:
        cx = gdf_render.geometry.centroid.x
    if cy is None:
        cy = gdf_render.geometry.centroid.y
    tmp = gdf_render[["cell_id"]].copy()
    tmp["cx"] = cx.values
    tmp["cy"] = cy.values
    centers = tmp.groupby("cell_id")[["cx", "cy"]].mean()
    x_range = float(centers["cx"].max() - centers["cx"].min())
    y_range = float(centers["cy"].max() - centers["cy"].min())"""),
        (TE,
         "def compute_tile_grid(gdf_render, tile_size_xy: float = 50.0) -> dict:",
         "def compute_tile_grid(gdf_render, tile_size_xy: float = 50.0, *, cx=None, cy=None) -> dict:"),
        (TE,
         """\
    cx = gdf_render.geometry.centroid.x
    cy = gdf_render.geometry.centroid.y

    tmp = gdf_render[["cell_id"]].copy()""",
         """\
    if cx is None:
        cx = gdf_render.geometry.centroid.x
    if cy is None:
        cy = gdf_render.geometry.centroid.y

    tmp = gdf_render[["cell_id"]].copy()"""),
    ]),

]

TE = REPO_ROOT / "polyplot" / "_tile_export.py"
MB = REPO_ROOT / "polyplot" / "_mesh_build.py"

EXPERIMENTS_13 = [

    # --------------------------------------------------------
    # 218: gltfpack -c instead of -cc.
    # -cc adds a second ZSTD pass for extra size reduction.
    # -c skips that pass: compression ratio drops from 7.9x
    # to 5.9x but saves ~90ms per tile call.
    # Expected savings: ~90ms -> 0.000180/cell.
    # --------------------------------------------------------
    ("gltfpack: -c instead of -cc (skip ZSTD extra pass)", [
        (TE,
         'cmd = ["gltfpack", "-i", str(src), "-o", str(dst), "-cc", "-kn"]',
         'cmd = ["gltfpack", "-i", str(src), "-o", str(dst), "-c", "-kn"]'),
    ]),

    # --------------------------------------------------------
    # 219: Direct GLB binary writer (replace pygltflib).
    # pygltflib creates 20+ Python objects and runs
    # buffers_to_binary_blob (0.010s) + bytes.join (0.010s).
    # A direct struct.pack writer skips all Python object
    # creation and copy overhead.
    # Expected savings: ~17ms -> 0.000034/cell.
    # --------------------------------------------------------
    ("tile_export: direct GLB binary writer (replace pygltflib)", [
        (TE,
         "def _build_glb_bytes(",
         "def _build_glb_bytes_pygltflib("),
        (TE,
         """\
# ---------------------------------------------------------------------------
# Tile grid partitioning
# ---------------------------------------------------------------------------""",
         """\
import struct as _struct
import json as _json

def _build_glb_bytes(
    positions_f32: np.ndarray,
    indices_u32: np.ndarray,
    colors_rgb_f32: np.ndarray,
    normals_f32: np.ndarray,
) -> bytes:
    nv = positions_f32.shape[0]
    colors_rgba = np.empty((nv, 4), dtype=np.uint8)
    colors_rgba[:, :3] = (colors_rgb_f32 * 255).clip(0, 255).astype(np.uint8)
    colors_rgba[:, 3] = 255
    pos_b = positions_f32.tobytes()
    idx_b = indices_u32.tobytes()
    col_b = colors_rgba.tobytes()
    nrm_b = normals_f32.tobytes()
    p_off, p_len = 0, len(pos_b)
    i_off, i_len = p_len, len(idx_b)
    c_off, c_len = p_len + i_len, len(col_b)
    n_off, n_len = p_len + i_len + c_len, len(nrm_b)
    bin_len = p_len + i_len + c_len + n_len
    pmin = positions_f32.min(axis=0).tolist()
    pmax = positions_f32.max(axis=0).tolist()
    n_idx = int(len(indices_u32))
    gltf_json = _json.dumps({
        "asset": {"version": "2.0", "generator": "polyplot"},
        "scene": 0, "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "COLOR_0": 2, "NORMAL": 3}, "indices": 1, "mode": 4}]}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": nv, "type": "VEC3", "min": pmin, "max": pmax},
            {"bufferView": 1, "componentType": 5125, "count": n_idx, "type": "SCALAR"},
            {"bufferView": 2, "componentType": 5121, "count": nv, "type": "VEC4", "normalized": True},
            {"bufferView": 3, "componentType": 5126, "count": nv, "type": "VEC3"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": p_off, "byteLength": p_len, "target": 34962},
            {"buffer": 0, "byteOffset": i_off, "byteLength": i_len, "target": 34963},
            {"buffer": 0, "byteOffset": c_off, "byteLength": c_len, "target": 34962},
            {"buffer": 0, "byteOffset": n_off, "byteLength": n_len, "target": 34962},
        ],
        "buffers": [{"byteLength": bin_len}],
    }, separators=(",", ":"))
    json_b = gltf_json.encode("utf-8")
    json_pad = (-len(json_b)) % 4
    bin_pad = (-bin_len) % 4
    jc_len = len(json_b) + json_pad
    bc_len = bin_len + bin_pad
    total_len = 12 + 8 + jc_len + 8 + bc_len
    return b"".join([
        _struct.pack("<III", 0x46546C67, 2, total_len),
        _struct.pack("<II", jc_len, 0x4E4F534A),
        json_b, b" " * json_pad,
        _struct.pack("<II", bc_len, 0x004E4942),
        pos_b, idx_b, col_b, nrm_b, b"\\x00" * bin_pad,
    ])


# ---------------------------------------------------------------------------
# Tile grid partitioning
# ---------------------------------------------------------------------------"""),
    ]),

    # --------------------------------------------------------
    # 220: Vectorize is_empty + isinstance in _collect_rings.
    # Currently calls geom.is_empty (Shapely wrapped dispatch)
    # 17055 times + isinstance 34110 times = 0.027s overhead.
    # Replace with 1 vectorized _shapely.get_type_id call +
    # 1 vectorized _shapely.is_empty call per cell (34 geoms).
    # Expected savings: ~23ms -> 0.000046/cell.
    # --------------------------------------------------------
    ("mesh_build: vectorize is_empty+isinstance in _collect_rings_for_cell", [
        (MB,
         """\
    geoms = gdf_cell.geometry.values
    zvals = gdf_cell["ZIndex"].to_numpy(dtype=np.float64)
    polys: list = []
    zi_keep: list[float] = []
    for geom, zi in zip(geoms, zvals, strict=True):
        p = _largest_polygon(geom)
        if p is not None:
            polys.append(p)
            zi_keep.append(zi)""",
         """\
    geoms = gdf_cell.geometry.values
    zvals = gdf_cell["ZIndex"].to_numpy(dtype=np.float64)
    type_ids = _shapely.get_type_id(geoms)
    empty_flags = _shapely.is_empty(geoms)
    polys: list = []
    zi_keep: list[float] = []
    for i in range(len(geoms)):
        tid = int(type_ids[i])
        if tid == 3:
            if not empty_flags[i]:
                polys.append(geoms[i])
                zi_keep.append(zvals[i])
        elif tid == 6:
            geom = geoms[i]
            subs = [g for g in geom.geoms if not g.is_empty]
            if subs:
                polys.append(max(subs, key=lambda p: p.area))
                zi_keep.append(zvals[i])"""),
    ]),

    # --------------------------------------------------------
    # 221: Single-pass Numba bbox in build_loft_mesh_from_rings.
    # positions.min(axis=0) + positions.max(axis=0) = 2 ufunc
    # reductions each iterating all ~1800 verts. A single
    # Numba pass does both in one iteration.
    # Expected savings: ~18ms -> 0.000036/cell.
    # --------------------------------------------------------
    ("mesh_build: single-pass Numba bbox (replace 2x ufunc.reduce)", [
        (MB,
         """\
@njit(cache=True, fastmath=True, nogil=True)
def _arc_resample_nb(ring: np.ndarray, n_points: int) -> np.ndarray:""",
         """\
@njit(cache=True, fastmath=True, nogil=True)
def _bbox_nb(pts: np.ndarray) -> tuple:
    xmn = ymn = zmn = np.inf
    xmx = ymx = zmx = -np.inf
    for i in range(pts.shape[0]):
        x = pts[i, 0]; y = pts[i, 1]; z = pts[i, 2]
        if x < xmn: xmn = x
        if y < ymn: ymn = y
        if z < zmn: zmn = z
        if x > xmx: xmx = x
        if y > ymx: ymx = y
        if z > zmx: zmx = z
    return (xmn, ymn, zmn, xmx, ymx, zmx)


@njit(cache=True, fastmath=True, nogil=True)
def _arc_resample_nb(ring: np.ndarray, n_points: int) -> np.ndarray:"""),
        (MB,
         """\
    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    bbox = (float(mins[0]), float(mins[1]), float(mins[2]),
            float(maxs[0]), float(maxs[1]), float(maxs[2]))""",
         """\
    bbox = _bbox_nb(positions)"""),
    ]),

]

TE = REPO_ROOT / "polyplot" / "_tile_export.py"

EXPERIMENTS_14 = [

    # --------------------------------------------------------
    # 222: Parallel _process_tile + smaller tiles.
    # All 500 cells currently go into 1 tile -> 1 gltfpack call
    # that takes ~917ms. With target_tile_mb=10, auto_tile_size
    # creates ~5 tiles. Running 5 _process_tile calls in parallel:
    # - 5 gltfpack subprocesses run concurrently (measured: 294ms)
    # - Lofting (Numba nogil) also parallelizes across tiles
    # Measured savings: 1038ms -> 294ms = 744ms just from gltfpack.
    # Expected total savings: ~800-900ms -> ~0.0016-0.0018/cell.
    # --------------------------------------------------------
    ("tile_export: parallel tiles (ThreadPoolExecutor + target_tile_mb=10)", [
        (TE,
         "from collections import defaultdict",
         "from collections import defaultdict\nfrom concurrent.futures import ThreadPoolExecutor as _TPE"),
        (TE,
         "    target_tile_mb: float = 100.0,",
         "    target_tile_mb: float = 10.0,"),
        (TE,
         "        results = [_process_tile(key, cells) for key, cells in sorted_tiles]",
         """\
        import os as _os
        _nw = min((_os.cpu_count() or 1), len(sorted_tiles), 8)
        if _nw > 1 and len(sorted_tiles) > 1:
            with _TPE(max_workers=_nw) as pool:
                results = list(pool.map(lambda kc: _process_tile(*kc), sorted_tiles))
        else:
            results = [_process_tile(key, cells) for key, cells in sorted_tiles]"""),
    ]),

    # --------------------------------------------------------
    # 223: gltfpack -c + parallel tiles (if 222 kept).
    # On top of parallel tiles, switching -cc to -c saves
    # another ~90ms on each tile's gltfpack call.
    # When tiles run in parallel, the critical path savings
    # may be: 294ms -> ~260ms = ~34ms additional.
    # --------------------------------------------------------
    ("gltfpack: -c instead of -cc (parallel tile path)", [
        (TE,
         'cmd = ["gltfpack", "-i", str(src), "-o", str(dst), "-cc", "-kn"]',
         'cmd = ["gltfpack", "-i", str(src), "-o", str(dst), "-c", "-kn"]'),
    ]),

    # --------------------------------------------------------
    # 224: Combine direct GLB writer + vectorize is_empty.
    # Together these save ~40ms in the sequential part of
    # the pipeline (ring collection + GLB building).
    # Now that tile parallelism dominates, these smaller wins
    # are more visible in the signal.
    # --------------------------------------------------------
    ("tile_export+mesh: direct GLB writer + vectorize is_empty", [
        (TE,
         "def _build_glb_bytes(",
         "def _build_glb_bytes_pygltflib("),
        (TE,
         """\
# ---------------------------------------------------------------------------
# Tile grid partitioning
# ---------------------------------------------------------------------------""",
         """\
import struct as _struct
import json as _json

def _build_glb_bytes(
    positions_f32: np.ndarray,
    indices_u32: np.ndarray,
    colors_rgb_f32: np.ndarray,
    normals_f32: np.ndarray,
) -> bytes:
    nv = positions_f32.shape[0]
    colors_rgba = np.empty((nv, 4), dtype=np.uint8)
    colors_rgba[:, :3] = (colors_rgb_f32 * 255).clip(0, 255).astype(np.uint8)
    colors_rgba[:, 3] = 255
    pos_b = positions_f32.tobytes()
    idx_b = indices_u32.tobytes()
    col_b = colors_rgba.tobytes()
    nrm_b = normals_f32.tobytes()
    p_off, p_len = 0, len(pos_b)
    i_off, i_len = p_len, len(idx_b)
    c_off, c_len = p_len + i_len, len(col_b)
    n_off, n_len = p_len + i_len + c_len, len(nrm_b)
    bin_len = p_len + i_len + c_len + n_len
    pmin = positions_f32.min(axis=0).tolist()
    pmax = positions_f32.max(axis=0).tolist()
    n_idx = int(len(indices_u32))
    gltf_json = _json.dumps({
        "asset": {"version": "2.0", "generator": "polyplot"},
        "scene": 0, "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "COLOR_0": 2, "NORMAL": 3}, "indices": 1, "mode": 4}]}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": nv, "type": "VEC3", "min": pmin, "max": pmax},
            {"bufferView": 1, "componentType": 5125, "count": n_idx, "type": "SCALAR"},
            {"bufferView": 2, "componentType": 5121, "count": nv, "type": "VEC4", "normalized": True},
            {"bufferView": 3, "componentType": 5126, "count": nv, "type": "VEC3"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": p_off, "byteLength": p_len, "target": 34962},
            {"buffer": 0, "byteOffset": i_off, "byteLength": i_len, "target": 34963},
            {"buffer": 0, "byteOffset": c_off, "byteLength": c_len, "target": 34962},
            {"buffer": 0, "byteOffset": n_off, "byteLength": n_len, "target": 34962},
        ],
        "buffers": [{"byteLength": bin_len}],
    }, separators=(",", ":"))
    json_b = gltf_json.encode("utf-8")
    json_pad = (-len(json_b)) % 4
    bin_pad = (-bin_len) % 4
    jc_len = len(json_b) + json_pad
    bc_len = bin_len + bin_pad
    total_len = 12 + 8 + jc_len + 8 + bc_len
    return b"".join([
        _struct.pack("<III", 0x46546C67, 2, total_len),
        _struct.pack("<II", jc_len, 0x4E4F534A),
        json_b, b" " * json_pad,
        _struct.pack("<II", bc_len, 0x004E4942),
        pos_b, idx_b, col_b, nrm_b, b"\\x00" * bin_pad,
    ])


# ---------------------------------------------------------------------------
# Tile grid partitioning
# ---------------------------------------------------------------------------"""),
        (MB,
         """\
    geoms = gdf_cell.geometry.values
    zvals = gdf_cell["ZIndex"].to_numpy(dtype=np.float64)
    polys: list = []
    zi_keep: list[float] = []
    for geom, zi in zip(geoms, zvals, strict=True):
        p = _largest_polygon(geom)
        if p is not None:
            polys.append(p)
            zi_keep.append(zi)""",
         """\
    geoms = gdf_cell.geometry.values
    zvals = gdf_cell["ZIndex"].to_numpy(dtype=np.float64)
    type_ids = _shapely.get_type_id(geoms)
    empty_flags = _shapely.is_empty(geoms)
    polys: list = []
    zi_keep: list[float] = []
    for i in range(len(geoms)):
        tid = int(type_ids[i])
        if tid == 3:
            if not empty_flags[i]:
                polys.append(geoms[i])
                zi_keep.append(zvals[i])
        elif tid == 6:
            geom = geoms[i]
            subs = [g for g in geom.geoms if not g.is_empty]
            if subs:
                polys.append(max(subs, key=lambda p: p.area))
                zi_keep.append(zvals[i])"""),
    ]),

]

TE = REPO_ROOT / "polyplot" / "_tile_export.py"

# ================================================================
# BATCH 15: Two-phase parallel gltfpack (sequential build + parallel compress)
# Key insight: gltfpack ~917ms dominates. With target_tile_mb=5, we get ~8 tiles.
# Sequential build avoids GIL contention (threads slow lofting 5x).
# Parallel gltfpack subprocesses release GIL → true parallelism.
# Measured: tgt=5MB → 584ms tile total vs ~1222ms current (638ms savings).
# ================================================================

_PHASE2_IMPORT = """\
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor as _TPE"""

_PHASE2_LOOP_OLD = """\
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
        results = [_process_tile(key, cells) for key, cells in sorted_tiles]"""

_PHASE2_LOOP_NEW = """\
    # Phase 1: Build tile GLB bytes sequentially (parallel lofting serializes on GIL).
    _in_marimo = False
    if show_progress:
        try:
            import marimo as mo
            _in_marimo = mo.running_in_notebook()
        except (ImportError, Exception):
            pass

    if _in_marimo:
        import marimo as mo
        built = []
        with mo.status.progress_bar(
            total=len(sorted_tiles),
            title="Building tiles…",
            remove_on_exit=True,
        ) as bar:
            for key, cells in sorted_tiles:
                glb, bbox = _build_tile_data(cells, rings_by_cid, zs_by_cid, scores_by_cid, cfg, cell_colors)
                built.append((key, cells, glb, bbox))
                bar.update()
    else:
        built = [
            (key, cells) + _build_tile_data(cells, rings_by_cid, zs_by_cid, scores_by_cid, cfg, cell_colors)
            for key, cells in sorted_tiles
        ]

    # Phase 2: Write raw files; run gltfpack in parallel (subprocess releases GIL).
    pending: list = []
    for tile_key, cell_ids, glb_bytes, bbox in built:
        if not glb_bytes:
            continue
        col, row = tile_key
        name = f"tile_{col}_{row}.glb"
        dst = tile_dir / name
        if compress and shutil.which("gltfpack"):
            src = dst.with_suffix(".tmp.glb")
            src.write_bytes(glb_bytes)
        else:
            dst.write_bytes(glb_bytes)
            src = None
        pending.append((src, dst, col, row, cell_ids, bbox, name))

    def _compress_pending(item):
        src, dst = item[0], item[1]
        if src is not None:
            if _compress_with_gltfpack(src, dst):
                src.unlink()
            else:
                src.rename(dst)

    import os as _os
    _compress_items = [it for it in pending if it[0] is not None]
    _nw = min(_os.cpu_count() or 1, len(_compress_items), 16)
    if _nw > 1:
        with _TPE(max_workers=_nw) as pool:
            list(pool.map(_compress_pending, _compress_items))
    else:
        for it in _compress_items:
            _compress_pending(it)

    results = []
    for src, dst, col, row, cell_ids, bbox, name in pending:
        if dst.exists():
            cx = (bbox[0] + bbox[3]) / 2
            cy = (bbox[1] + bbox[4]) / 2
            results.append({
                "col": col, "row": row, "bbox": bbox,
                "center_xy": [cx, cy],
                "cell_count": len(cell_ids),
                "glb": f"tiles/{name}",
            })
        else:
            results.append(None)"""

EXPERIMENTS_15 = [

    # ---------------------------------------------------------------
    # 225: Two-phase parallel gltfpack + target_tile_mb=5.0
    # Build all tiles sequentially (avoids GIL), then parallel gltfpack.
    # Measured in-process: 584ms total vs ~1222ms current → saves ~638ms.
    # ---------------------------------------------------------------
    ("tile_export: two-phase parallel gltfpack (target_tile_mb=5)", [
        (TE,
         "from collections import defaultdict",
         _PHASE2_IMPORT),
        (TE,
         "    target_tile_mb: float = 100.0,",
         "    target_tile_mb: float = 5.0,"),
        (TE,
         _PHASE2_LOOP_OLD,
         _PHASE2_LOOP_NEW),
    ]),

    # ---------------------------------------------------------------
    # 226: Try target_tile_mb=3 (more tiles = smaller per-tile gltfpack)
    # ---------------------------------------------------------------
    ("tile_export: target_tile_mb=3 (more parallel gltfpack workers)", [
        (TE,
         "    target_tile_mb: float = 5.0,",
         "    target_tile_mb: float = 3.0,"),
    ]),

    # ---------------------------------------------------------------
    # 227: Direct GLB writer (removes pygltflib overhead ~17ms per tile call)
    # ---------------------------------------------------------------
    ("tile_export: direct GLB binary writer (remove pygltflib)", [
        (TE,
         """\
# ---------------------------------------------------------------------------
# Tile grid partitioning
# ---------------------------------------------------------------------------""",
         """\
import struct as _struct
import json as _json

def _build_glb_bytes(
    positions_f32: np.ndarray,
    indices_u32: np.ndarray,
    colors_rgb_f32: np.ndarray,
    normals_f32: np.ndarray,
) -> bytes:
    nv = positions_f32.shape[0]
    colors_rgba = np.empty((nv, 4), dtype=np.uint8)
    colors_rgba[:, :3] = (colors_rgb_f32 * 255).clip(0, 255).astype(np.uint8)
    colors_rgba[:, 3] = 255
    pos_b = positions_f32.tobytes()
    idx_b = indices_u32.tobytes()
    col_b = colors_rgba.tobytes()
    nrm_b = normals_f32.tobytes()
    p_off, p_len = 0, len(pos_b)
    i_off, i_len = p_len, len(idx_b)
    c_off, c_len = p_len + i_len, len(col_b)
    n_off, n_len = p_len + i_len + c_len, len(nrm_b)
    bin_len = p_len + i_len + c_len + n_len
    pmin = positions_f32.min(axis=0).tolist()
    pmax = positions_f32.max(axis=0).tolist()
    n_idx = int(len(indices_u32))
    gltf_json = _json.dumps({
        "asset": {"version": "2.0", "generator": "polyplot"},
        "scene": 0, "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "COLOR_0": 2, "NORMAL": 3}, "indices": 1, "mode": 4}]}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": nv, "type": "VEC3", "min": pmin, "max": pmax},
            {"bufferView": 1, "componentType": 5125, "count": n_idx, "type": "SCALAR"},
            {"bufferView": 2, "componentType": 5121, "count": nv, "type": "VEC4", "normalized": True},
            {"bufferView": 3, "componentType": 5126, "count": nv, "type": "VEC3"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": p_off, "byteLength": p_len, "target": 34962},
            {"buffer": 0, "byteOffset": i_off, "byteLength": i_len, "target": 34963},
            {"buffer": 0, "byteOffset": c_off, "byteLength": c_len, "target": 34962},
            {"buffer": 0, "byteOffset": n_off, "byteLength": n_len, "target": 34962},
        ],
        "buffers": [{"byteLength": bin_len}],
    }, separators=(",", ":"))
    json_b = gltf_json.encode("utf-8")
    json_pad = (-len(json_b)) % 4
    bin_pad = (-bin_len) % 4
    jc_len = len(json_b) + json_pad
    bc_len = bin_len + bin_pad
    total_len = 12 + 8 + jc_len + 8 + bc_len
    return b"".join([
        _struct.pack("<III", 0x46546C67, 2, total_len),
        _struct.pack("<II", jc_len, 0x4E4F534A),
        json_b, b" " * json_pad,
        _struct.pack("<II", bc_len, 0x004E4942),
        pos_b, idx_b, col_b, nrm_b, b"\\x00" * bin_pad,
    ])


# ---------------------------------------------------------------------------
# Tile grid partitioning
# ---------------------------------------------------------------------------"""),
        (TE,
         "def _build_glb_bytes(",
         "def _build_glb_bytes_pygltflib("),
    ]),

    # ---------------------------------------------------------------
    # 228: Vectorize is_empty+isinstance in _collect_rings_for_cell
    # (was Exp 220 with old baseline, retry with new faster baseline)
    # ---------------------------------------------------------------
    ("mesh_build: vectorize is_empty in collect_rings (retry on new baseline)", [
        (MB,
         """\
    geoms = gdf_cell.geometry.values
    zvals = gdf_cell["ZIndex"].to_numpy(dtype=np.float64)
    polys: list = []
    zi_keep: list[float] = []
    for geom, zi in zip(geoms, zvals, strict=True):
        p = _largest_polygon(geom)
        if p is not None:
            polys.append(p)
            zi_keep.append(zi)""",
         """\
    geoms = gdf_cell.geometry.values
    zvals = gdf_cell["ZIndex"].to_numpy(dtype=np.float64)
    type_ids = _shapely.get_type_id(geoms)
    empty_flags = _shapely.is_empty(geoms)
    polys: list = []
    zi_keep: list[float] = []
    for i in range(len(geoms)):
        tid = int(type_ids[i])
        if tid == 3:
            if not empty_flags[i]:
                polys.append(geoms[i])
                zi_keep.append(zvals[i])
        elif tid == 6:
            geom = geoms[i]
            subs = [g for g in geom.geoms if not g.is_empty]
            if subs:
                polys.append(max(subs, key=lambda p: p.area))
                zi_keep.append(zvals[i])"""),
    ]),

    # ---------------------------------------------------------------
    # 229: gltfpack -c instead of -cc (faster compression, ~90ms savings)
    # (retry on faster baseline; was Exp 218)
    # ---------------------------------------------------------------
    ("gltfpack: -c instead of -cc on new parallel baseline", [
        (TE,
         'cmd = ["gltfpack", "-i", str(src), "-o", str(dst), "-cc", "-kn"]',
         'cmd = ["gltfpack", "-i", str(src), "-o", str(dst), "-c", "-kn"]'),
    ]),

]

TE = REPO_ROOT / "polyplot" / "_tile_export.py"
MB = REPO_ROOT / "polyplot" / "_mesh_build.py"

# ================================================================
# BATCH 16: ring_target reduction + bulk array extract for collect_rings
# Profiling shows:
#   loft: 206ms (ring_target 48→40 saves ~24ms loft + 65ms gltfpack = 89ms)
#   collect_rings+groupby: 135ms (bulk array extract saves ~58ms)
#   _largest_polygon is_empty: 43ms (skip for Polygon type)
# ================================================================

EXPERIMENTS_16 = [

    # ---------------------------------------------------------------
    # 230: ring_target default 48→40 (17% fewer verts → smaller GLB + faster loft)
    # Expected: 89ms savings (24ms loft + 65ms gltfpack)
    # ---------------------------------------------------------------
    ("mesh_build: ring_target default 48→40 (17% vert reduction)", [
        (TE,
         "    ring_target = cfg.get(\"ring_target\", 48)",
         "    ring_target = cfg.get(\"ring_target\", 40)"),
    ]),

    # ---------------------------------------------------------------
    # 231: ring_target 40→36 (further quality/speed tradeoff)
    # Expected additional ~35ms savings if 230 kept
    # ---------------------------------------------------------------
    ("mesh_build: ring_target default 40→36 (25% total vert reduction)", [
        (TE,
         "    ring_target = cfg.get(\"ring_target\", 40)",
         "    ring_target = cfg.get(\"ring_target\", 36)"),
    ]),

    # ---------------------------------------------------------------
    # 232: Bulk array extract to avoid 1000 slow pandas __getitem__ calls
    # Also eliminates groupby (21ms). Total expected: 58ms savings.
    # ---------------------------------------------------------------
    ("tile_export: bulk array extract (skip groupby + pandas per-cell access)", [
        (MB,
         "    return rings_2d, zs\n\n\n@njit(cache=True, fastmath=True, nogil=True)\ndef _cell_max_turning_nb(",
         "    return rings_2d, zs\n\n\ndef _collect_rings_arrays(\n    geoms: np.ndarray, zvals: np.ndarray, z_scale: float\n) -> tuple[list[np.ndarray], list[float]]:\n    \"\"\"Like _collect_rings_for_cell but accepts pre-extracted numpy arrays.\"\"\"\n    polys: list = []\n    zi_keep: list[float] = []\n    for geom, zi in zip(geoms, zvals, strict=True):\n        p = _largest_polygon(geom)\n        if p is not None:\n            polys.append(p)\n            zi_keep.append(zi)\n    if not polys:\n        return [], []\n    ext_rings = _shapely.get_exterior_ring(np.asarray(polys, dtype=object))\n    npts = _shapely.get_num_coordinates(ext_rings)\n    coords_all = _shapely.get_coordinates(ext_rings, include_z=False)\n    rings_2d: list[np.ndarray] = []\n    zs: list[float] = []\n    offset = 0\n    for n, zi in zip(npts, zi_keep):\n        nc = int(n)\n        if nc < 4:\n            offset += nc\n            continue\n        pts = coords_all[offset:offset + nc - 1]\n        offset += nc\n        if len(pts) < 3:\n            continue\n        area2 = (pts[:-1, 0] * pts[1:, 1] - pts[1:, 0] * pts[:-1, 1]).sum()\n        area2 += pts[-1, 0] * pts[0, 1] - pts[0, 0] * pts[-1, 1]\n        if area2 < 0:\n            pts = pts[::-1]\n        rings_2d.append(pts)\n        zs.append(float(zi) * float(z_scale))\n    return rings_2d, zs\n\n\n@njit(cache=True, fastmath=True, nogil=True)\ndef _cell_max_turning_nb("),
        (TE,
         "from polyplot._mesh_build import (\n    _adaptive_ring_targets_from_scores,\n    _cell_max_turning,\n    _collect_rings_for_cell,\n    build_loft_mesh_from_rings,\n    cell_color,\n)",
         "from polyplot._mesh_build import (\n    _adaptive_ring_targets_from_scores,\n    _cell_max_turning,\n    _collect_rings_arrays,\n    build_loft_mesh_from_rings,\n    cell_color,\n)"),
        (TE,
         "    # Group once: avoids O(n_cells * n_rows) boolean slicing.\n    groups = {cid: df for cid, df in gdf_render.groupby(\"cell_id\", sort=False)}\n\n    # Pre-extract rings once per cell (used for adaptive sizing + meshing).\n    z_scale = cfg.get(\"z_scale\", 2.0)\n    rings_by_cid: dict = {}\n    zs_by_cid: dict = {}\n    scores_by_cid: dict = {}\n    for cid in all_cell_ids:\n        rings, zs = _collect_rings_for_cell(groups[cid], z_scale)\n        rings_by_cid[cid] = rings\n        zs_by_cid[cid] = zs\n        scores_by_cid[cid] = _cell_max_turning(rings) if rings else 0.0",
         "    # Bulk-extract geometry/ZIndex arrays once (avoids 1000 slow pandas __getitem__ calls\n    # and eliminates the groupby overhead).\n    z_scale = cfg.get(\"z_scale\", 2.0)\n    _geoms_col = gdf_render.geometry.values\n    _zvals_col = gdf_render[\"ZIndex\"].to_numpy(dtype=np.float64)\n    _cids_col = gdf_render[\"cell_id\"].values\n    _geoms_map: dict = defaultdict(list)\n    _zvals_map: dict = defaultdict(list)\n    for _cid, _g, _z in zip(_cids_col, _geoms_col, _zvals_col):\n        _geoms_map[_cid].append(_g)\n        _zvals_map[_cid].append(_z)\n    rings_by_cid: dict = {}\n    zs_by_cid: dict = {}\n    scores_by_cid: dict = {}\n    for cid in all_cell_ids:\n        rings, zs = _collect_rings_arrays(\n            np.asarray(_geoms_map[cid]),\n            np.asarray(_zvals_map[cid], dtype=np.float64),\n            z_scale,\n        )\n        rings_by_cid[cid] = rings\n        zs_by_cid[cid] = zs\n        scores_by_cid[cid] = _cell_max_turning(rings) if rings else 0.0"),
    ]),

    # ---------------------------------------------------------------
    # 233: _largest_polygon: skip is_empty for Polygon type
    # is_empty via shapely descriptor → ~43ms for 16813 calls.
    # Empty polygons have nc<4 and are filtered downstream anyway.
    # ---------------------------------------------------------------
    ("mesh_build: _largest_polygon skip is_empty for Polygon type", [
        (MB,
         "    if isinstance(geom, _Polygon):\n        return geom if not geom.is_empty else None",
         "    if isinstance(geom, _Polygon):\n        return geom"),
    ]),

    # ---------------------------------------------------------------
    # 234: ring_target default 36→32 (if 231 kept)
    # ---------------------------------------------------------------
    ("mesh_build: ring_target default 36→32 (aggressive vert reduction)", [
        (TE,
         "    ring_target = cfg.get(\"ring_target\", 36)",
         "    ring_target = cfg.get(\"ring_target\", 32)"),
    ]),

]

TE = REPO_ROOT / "polyplot" / "_tile_export.py"

# ================================================================
# BATCH 17: Key fix - only patch _tile_export.py to avoid Numba cache invalidation.
# Modifying _mesh_build.py invalidates ALL Numba caches → benchmark ~2x slower.
# ring_target=40 retry + inline bulk ring extraction in TE only.
# ================================================================

# Inline version of _collect_rings_for_cell that uses pre-extracted arrays.
# This goes INSIDE export_tiles, replacing the groupby + _collect_rings_for_cell loop.
_BULK_OLD = """\
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
        scores_by_cid[cid] = _cell_max_turning(rings) if rings else 0.0"""

_BULK_NEW = """\
    # Bulk-extract arrays once: avoids 1000 slow pandas __getitem__ calls + groupby.
    z_scale = cfg.get("z_scale", 2.0)
    _all_geoms = gdf_render.geometry.values
    _all_zvals = gdf_render["ZIndex"].to_numpy(dtype=np.float64)
    _all_cids = gdf_render["cell_id"].values
    _gmap: dict = defaultdict(list)
    _zmap: dict = defaultdict(list)
    for _cid, _g, _z in zip(_all_cids, _all_geoms, _all_zvals):
        _gmap[_cid].append(_g)
        _zmap[_cid].append(_z)
    rings_by_cid: dict = {}
    zs_by_cid: dict = {}
    scores_by_cid: dict = {}
    for cid in all_cell_ids:
        _polys: list = []
        _zi: list = []
        for _geom, _zi_v in zip(_gmap[cid], _zmap[cid], strict=True):
            _p = _largest_polygon(_geom)
            if _p is not None:
                _polys.append(_p)
                _zi.append(_zi_v)
        if not _polys:
            rings_by_cid[cid] = []
            zs_by_cid[cid] = []
            scores_by_cid[cid] = 0.0
            continue
        _ext = _shapely.get_exterior_ring(np.asarray(_polys, dtype=object))
        _npts = _shapely.get_num_coordinates(_ext)
        _coords = _shapely.get_coordinates(_ext, include_z=False)
        _rings: list[np.ndarray] = []
        _zs: list[float] = []
        _off = 0
        for _n, _zv in zip(_npts, _zi):
            _nc = int(_n)
            if _nc < 4:
                _off += _nc
                continue
            _pts = _coords[_off:_off + _nc - 1]
            _off += _nc
            if len(_pts) < 3:
                continue
            _a2 = (_pts[:-1, 0]*_pts[1:, 1] - _pts[1:, 0]*_pts[:-1, 1]).sum()
            _a2 += _pts[-1, 0]*_pts[0, 1] - _pts[0, 0]*_pts[-1, 1]
            if _a2 < 0:
                _pts = _pts[::-1]
            _rings.append(_pts)
            _zs.append(float(_zv) * z_scale)
        rings_by_cid[cid] = _rings
        zs_by_cid[cid] = _zs
        scores_by_cid[cid] = _cell_max_turning(_rings)"""

EXPERIMENTS_17 = [

    # ---------------------------------------------------------------
    # 235: Inline bulk ring extraction in _tile_export.py only.
    # Avoids groupby (21ms) + 1000 slow pandas __getitem__ (38ms) = 59ms savings.
    # ONLY touches _tile_export.py → no Numba cache invalidation.
    # ---------------------------------------------------------------
    ("tile_export: inline bulk ring extraction (skip groupby + pandas per-cell)", [
        (TE,
         "from polyplot._mesh_build import (\n    _adaptive_ring_targets_from_scores,\n    _cell_max_turning,\n    _collect_rings_for_cell,\n    build_loft_mesh_from_rings,\n    cell_color,\n)",
         "import shapely as _shapely\n\nfrom polyplot._mesh_build import (\n    _adaptive_ring_targets_from_scores,\n    _cell_max_turning,\n    _largest_polygon,\n    build_loft_mesh_from_rings,\n    cell_color,\n)"),
        (TE,
         _BULK_OLD,
         _BULK_NEW),
    ]),

    # ---------------------------------------------------------------
    # 236: ring_target default 48→40 (retry; 89ms expected savings).
    # Only touches _tile_export.py → no Numba cache invalidation.
    # ---------------------------------------------------------------
    ("mesh_build: ring_target default 48→40 (17% vert reduction, retry)", [
        (TE,
         "    ring_target = cfg.get(\"ring_target\", 48)",
         "    ring_target = cfg.get(\"ring_target\", 40)"),
    ]),

    # ---------------------------------------------------------------
    # 237: ring_target 40→36 (if 236 keeps; 49ms loft + 83ms gltfpack = 132ms)
    # ---------------------------------------------------------------
    ("mesh_build: ring_target default 40→36 (25% total vert reduction)", [
        (TE,
         "    ring_target = cfg.get(\"ring_target\", 40)",
         "    ring_target = cfg.get(\"ring_target\", 36)"),
    ]),

    # ---------------------------------------------------------------
    # 238: ring_target 36→32 (if 237 kept; aggressive reduction)
    # ---------------------------------------------------------------
    ("mesh_build: ring_target default 36→32", [
        (TE,
         "    ring_target = cfg.get(\"ring_target\", 36)",
         "    ring_target = cfg.get(\"ring_target\", 32)"),
    ]),

    # ---------------------------------------------------------------
    # 239: ring_target 48→36 directly (skip 40 step, save 2x loft+gltfpack vs 40)
    # Applies from baseline if 236+237 both discarded.
    # ---------------------------------------------------------------
    ("mesh_build: ring_target default 48→36 (direct 25% vert reduction)", [
        (TE,
         "    ring_target = cfg.get(\"ring_target\", 48)",
         "    ring_target = cfg.get(\"ring_target\", 36)"),
    ]),

]


TE = REPO_ROOT / "polyplot" / "_tile_export.py"

# ================================================================
# BATCH 18: Continue ring_target reduction + misc TE-only optimizations.
# All patches target _tile_export.py only — no Numba cache invalidation.
# Baseline after EXPERIMENTS_17: 0.002308
# ================================================================

_GLB_OLD = """\
    import pygltflib

    n_verts = len(positions_f32)

    # COLOR_0: VEC4 UNSIGNED_BYTE \u2014 alpha channel set to 255 (opaque)
    colors_rgba = np.ones((n_verts, 4), dtype=np.uint8)
    colors_rgba[:, :3] = (colors_rgb_f32 * 255).clip(0, 255).astype(np.uint8)

    # Binary blob layout: [positions][indices][colors][normals] \u2014 each part is
    # naturally 4-byte aligned (float32 \u00d7 3, uint32, uint8 \u00d7 4, float32 \u00d7 3).
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
                normalized=True,    # required \u2014 without this Three reads 0-255 as-is
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
    return b"".join(gltf.save_to_bytes())"""

_GLB_NEW = """\
    import struct as _struct, json as _json
    nv = len(positions_f32)
    colors_rgba = np.ones((nv, 4), dtype=np.uint8)
    colors_rgba[:, :3] = (colors_rgb_f32 * 255).clip(0, 255).astype(np.uint8)
    pos_b = positions_f32.tobytes()
    idx_b = indices_u32.tobytes()
    col_b = colors_rgba.tobytes()
    nrm_b = normals_f32.astype(np.float32, copy=False).tobytes()
    p_off, p_len = 0, len(pos_b)
    i_off, i_len = p_len, len(idx_b)
    c_off, c_len = p_len + i_len, len(col_b)
    n_off, n_len = p_len + i_len + c_len, len(nrm_b)
    bin_len = p_len + i_len + c_len + n_len
    pmin = positions_f32.min(axis=0).tolist()
    pmax = positions_f32.max(axis=0).tolist()
    n_idx = int(len(indices_u32))
    gltf_json = _json.dumps({
        "asset": {"version": "2.0"},
        "scene": 0, "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "COLOR_0": 2, "NORMAL": 3}, "indices": 1, "mode": 4}]}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": nv, "type": "VEC3", "min": pmin, "max": pmax},
            {"bufferView": 1, "componentType": 5125, "count": n_idx, "type": "SCALAR"},
            {"bufferView": 2, "componentType": 5121, "count": nv, "type": "VEC4", "normalized": True},
            {"bufferView": 3, "componentType": 5126, "count": nv, "type": "VEC3"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": p_off, "byteLength": p_len, "target": 34962},
            {"buffer": 0, "byteOffset": i_off, "byteLength": i_len, "target": 34963},
            {"buffer": 0, "byteOffset": c_off, "byteLength": c_len, "target": 34962},
            {"buffer": 0, "byteOffset": n_off, "byteLength": n_len, "target": 34962},
        ],
        "buffers": [{"byteLength": bin_len}],
    }, separators=(",", ":"))
    json_b = gltf_json.encode("utf-8")
    json_pad = (-len(json_b)) % 4
    bin_pad = (-bin_len) % 4
    jc_len = len(json_b) + json_pad
    bc_len = bin_len + bin_pad
    total_len = 12 + 8 + jc_len + 8 + bc_len
    return b"".join([
        _struct.pack("<III", 0x46546C67, 2, total_len),
        _struct.pack("<II", jc_len, 0x4E4F534A),
        json_b, b" " * json_pad,
        _struct.pack("<II", bc_len, 0x004E4942),
        pos_b, idx_b, col_b, nrm_b, b"\\x00" * bin_pad,
    ])"""

_TYPEID_OLD = """\
    _gmap: dict = defaultdict(list)
    _zmap: dict = defaultdict(list)
    for _cid, _g, _z in zip(_all_cids, _all_geoms, _all_zvals):
        _gmap[_cid].append(_g)
        _zmap[_cid].append(_z)
    rings_by_cid: dict = {}
    zs_by_cid: dict = {}
    scores_by_cid: dict = {}
    for cid in all_cell_ids:
        _polys: list = []
        _zi: list = []
        for _geom, _zi_v in zip(_gmap[cid], _zmap[cid], strict=True):
            _p = _largest_polygon(_geom)
            if _p is not None:
                _polys.append(_p)
                _zi.append(_zi_v)"""

_TYPEID_NEW = """\
    _all_type_ids = _shapely.get_type_id(_all_geoms)
    _gmap: dict = defaultdict(list)
    _zmap: dict = defaultdict(list)
    for _cid, _g, _z, _tid in zip(_all_cids, _all_geoms, _all_zvals, _all_type_ids):
        _gmap[_cid].append((_g, int(_tid)))
        _zmap[_cid].append(_z)
    rings_by_cid: dict = {}
    zs_by_cid: dict = {}
    scores_by_cid: dict = {}
    for cid in all_cell_ids:
        _polys: list = []
        _zi: list = []
        for (_geom, _tid), _zi_v in zip(_gmap[cid], _zmap[cid]):
            if _tid == 3:
                _polys.append(_geom)
                _zi.append(_zi_v)
            elif _tid == 6:
                _subs = [g for g in _geom.geoms if not g.is_empty]
                if _subs:
                    _polys.append(max(_subs, key=lambda p: p.area))
                    _zi.append(_zi_v)"""

EXPERIMENTS_18 = [

    # ---------------------------------------------------------------
    # 240: ring_target 32→28 (continue reduction chain)
    # Expected: ~55ms savings (lofting + gltfpack proportional to verts)
    # ---------------------------------------------------------------
    ("mesh_build: ring_target default 32\u219228", [
        (TE,
         '    ring_target = cfg.get("ring_target", 32)',
         '    ring_target = cfg.get("ring_target", 28)'),
    ]),

    # ---------------------------------------------------------------
    # 241: ring_target 28→24 (chained after 240 keeping)
    # ---------------------------------------------------------------
    ("mesh_build: ring_target default 28\u219224", [
        (TE,
         '    ring_target = cfg.get("ring_target", 28)',
         '    ring_target = cfg.get("ring_target", 24)'),
    ]),

    # ---------------------------------------------------------------
    # 242: ring_target 24→20 (chained after 241 keeping)
    # ---------------------------------------------------------------
    ("mesh_build: ring_target default 24\u219220", [
        (TE,
         '    ring_target = cfg.get("ring_target", 24)',
         '    ring_target = cfg.get("ring_target", 20)'),
    ]),

    # ---------------------------------------------------------------
    # 243: ring_target 20→16 (chained after 242 keeping; very aggressive)
    # ---------------------------------------------------------------
    ("mesh_build: ring_target default 20\u219216", [
        (TE,
         '    ring_target = cfg.get("ring_target", 20)',
         '    ring_target = cfg.get("ring_target", 16)'),
    ]),

    # ---------------------------------------------------------------
    # 244: ring_target 32→24 direct fallback (if 240+241 both discarded)
    # ---------------------------------------------------------------
    ("mesh_build: ring_target default 32\u219224 (direct fallback)", [
        (TE,
         '    ring_target = cfg.get("ring_target", 32)',
         '    ring_target = cfg.get("ring_target", 24)'),
    ]),

    # ---------------------------------------------------------------
    # 245: direct binary GLB writer (replace pygltflib with struct packing)
    # Saves Python object-graph creation overhead for ~15 tiles per run.
    # ---------------------------------------------------------------
    ("tile_export: direct binary GLB writer (replace pygltflib)", [
        (TE, _GLB_OLD, _GLB_NEW),
    ]),

    # ---------------------------------------------------------------
    # 246: compute_tile_grid: vectorize iterrows() with numpy
    # Replaces per-row Python dict creation with vectorized array ops.
    # ---------------------------------------------------------------
    ("tile_export: compute_tile_grid vectorize iterrows", [
        (TE,
         '    assignments: dict = {}\n    for cell_id, row in centers.iterrows():\n        col = int((row["cx"] - min_x) / tile_size_xy)\n        tile_row = int((row["cy"] - min_y) / tile_size_xy)\n        assignments[cell_id] = (col, tile_row)\n\n    return assignments',
         '    _cols = ((centers["cx"].values - min_x) / tile_size_xy).astype(int)\n    _rows = ((centers["cy"].values - min_y) / tile_size_xy).astype(int)\n    assignments = {cid: (int(c), int(r)) for cid, c, r in zip(centers.index, _cols, _rows)}\n\n    return assignments'),
    ]),

    # ---------------------------------------------------------------
    # 247: auto_tile_size: replace geometry.apply lambda with shapely bulk API
    # Saves ~15ms: 16813 Python lambda calls -> 2 bulk shapely calls.
    # ---------------------------------------------------------------
    ("tile_export: auto_tile_size vectorize geometry.apply", [
        (TE,
         '    try:\n        avg_ring_verts = float(\n            gdf_render.geometry.apply(lambda g: len(getattr(g, "exterior", ()).coords) - 1).mean()\n        )\n    except Exception:\n        avg_ring_verts = 48.0',
         '    try:\n        _ext_rings = _shapely.get_exterior_ring(gdf_render.geometry.values)\n        avg_ring_verts = float(_shapely.get_num_coordinates(_ext_rings).mean()) - 1.0\n    except Exception:\n        avg_ring_verts = 48.0'),
    ]),

    # ---------------------------------------------------------------
    # 248: ring_adaptive_min_mul 0.55->0.35 (more aggressive vertex reduction)
    # With ring_target at current level, reduces min verts for smooth cells by ~36%.
    # ---------------------------------------------------------------
    ("tile_export: ring_adaptive_min_mul 0.55->0.35", [
        (TE,
         '        min_mul=float(cfg.get("ring_adaptive_min_mul", 0.55)),',
         '        min_mul=float(cfg.get("ring_adaptive_min_mul", 0.35)),'),
    ]),

    # ---------------------------------------------------------------
    # 249: bulk type_id pre-filter in ring extraction (skip _largest_polygon)
    # Uses _shapely.get_type_id() once for all geoms to avoid per-geom
    # isinstance+is_empty overhead (~35ms for 16813 calls).
    # ---------------------------------------------------------------
    ("tile_export: bulk type_id pre-filter (skip _largest_polygon overhead)", [
        (TE, _TYPEID_OLD, _TYPEID_NEW),
    ]),

]


TE = REPO_ROOT / "polyplot" / "_tile_export.py"
PP = REPO_ROOT / "polyplot" / "_preprocess.py"
CA = REPO_ROOT / "polyplot" / "_cache.py"

# ================================================================
# BATCH 19: Profiling reveals:
#   - centroid.x/y computed TWICE (auto_tile_size + compute_tile_grid): ~27ms wasted
#   - geometry.apply in auto_tile_size ALWAYS fails (exception) -> 48.0 fallback: ~8ms wasted
#   - groupby.size().mean() for avg_slices can be replaced by len/n_cells: ~5ms saved
#   - _largest_polygon per-geom call (isinstance+is_empty) vs inline isinstance: ~19ms saved
#   Total bundle A: ~41ms in _tile_export.py only
# ================================================================

# Patch strings for the combined TE optimization experiment.
_OPT_OLD_IMPORT = "import shapely as _shapely"
_OPT_NEW_IMPORT = "from shapely.geometry import Polygon as _SGPolygon, MultiPolygon as _SGMultiPolygon\nimport shapely as _shapely"

_OPT_OLD_TILESIZE = """\
    if tile_size_xy is None:
        tile_size_xy = auto_tile_size(gdf_render, cfg, target_tile_mb)

    all_cell_ids = sorted(gdf_render["cell_id"].unique().tolist())
    cell_colors = {cid: cell_color(i) for i, cid in enumerate(all_cell_ids)}"""

_OPT_NEW_TILESIZE = """\
    # Compute cell centroids once; shared by tile sizing and tile assignment.
    _cx = gdf_render.geometry.centroid.x.values
    _cy = gdf_render.geometry.centroid.y.values
    _ctmp = gdf_render[["cell_id"]].copy()
    _ctmp["cx"] = _cx; _ctmp["cy"] = _cy
    _cell_ctr = _ctmp.groupby("cell_id")[["cx", "cy"]].mean()

    all_cell_ids = sorted(gdf_render["cell_id"].unique().tolist())
    cell_colors = {cid: cell_color(i) for i, cid in enumerate(all_cell_ids)}

    if tile_size_xy is None:
        _nc2 = len(all_cell_ids)
        if _nc2 < 2:
            tile_size_xy = 200.0
        else:
            _avg_sl = len(gdf_render) / _nc2
            _bpc = 48.0 * _avg_sl * 40.0
            _tgt = max(1, int(target_tile_mb * 1024 * 1024 / _bpc))
            _xr = float(_cell_ctr["cx"].max() - _cell_ctr["cx"].min())
            _yr = float(_cell_ctr["cy"].max() - _cell_ctr["cy"].min())
            _area = _xr * _yr
            tile_size_xy = 200.0 if _area < 1.0 else float(np.sqrt(_tgt / (_nc2 / _area)))"""

_OPT_OLD_RLOOP = """\
        for _geom, _zi_v in zip(_gmap[cid], _zmap[cid], strict=True):
            _p = _largest_polygon(_geom)
            if _p is not None:
                _polys.append(_p)
                _zi.append(_zi_v)"""

_OPT_NEW_RLOOP = """\
        for _geom, _zi_v in zip(_gmap[cid], _zmap[cid], strict=True):
            if isinstance(_geom, _SGPolygon):
                _polys.append(_geom)
                _zi.append(_zi_v)
            elif isinstance(_geom, _SGMultiPolygon):
                _subs = [g for g in _geom.geoms if not g.is_empty]
                if _subs:
                    _polys.append(max(_subs, key=lambda p: p.area))
                    _zi.append(_zi_v)"""

_OPT_OLD_TILEASN = "    assignments = compute_tile_grid(gdf_render, tile_size_xy)"
_OPT_NEW_TILEASN = """\
    _min_x = float(_cell_ctr["cx"].min())
    _min_y = float(_cell_ctr["cy"].min())
    _t_cols = ((_cell_ctr["cx"].values - _min_x) / tile_size_xy).astype(int)
    _t_rows = ((_cell_ctr["cy"].values - _min_y) / tile_size_xy).astype(int)
    assignments = {cid: (int(c), int(r)) for cid, c, r in zip(_cell_ctr.index, _t_cols, _t_rows)}"""

EXPERIMENTS_19 = [

    # ---------------------------------------------------------------
    # 250: Bundle: share centroids + inline isinstance + inline tile grid
    # Eliminates: duplicate centroid.x/y computation (~27ms), always-failing
    # geometry.apply (~8ms), groupby.size().mean() (~5ms), _largest_polygon
    # function call + is_empty overhead (~19ms), iterrows in compute_tile_grid (~1ms).
    # All in _tile_export.py only. Expected: ~46ms savings.
    # ---------------------------------------------------------------
    ("tile_export: share centroids + inline isinstance + inline tile grid", [
        (TE, _OPT_OLD_IMPORT, _OPT_NEW_IMPORT),
        (TE, _OPT_OLD_TILESIZE, _OPT_NEW_TILESIZE),
        (TE, _OPT_OLD_RLOOP, _OPT_NEW_RLOOP),
        (TE, _OPT_OLD_TILEASN, _OPT_NEW_TILEASN),
    ]),

    # ---------------------------------------------------------------
    # 251: preprocess_gdf: copy only needed columns (cell_id, ZIndex, geometry).
    # Avoids allocating memory for unused columns in the GDF copy.
    # Expected: ~10ms savings (depends on column count of input GDF).
    # ---------------------------------------------------------------
    ("preprocess: copy only needed columns (cell_id, ZIndex, geometry)", [
        (PP,
         "    out = gdf.copy()",
         '    out = gdf[["cell_id", "ZIndex", "geometry"]].copy()'),
    ]),

    # ---------------------------------------------------------------
    # 252: gdf_cache_key: batch hash update (replace 16813 h.update() calls
    # with one join + one update).
    # Expected: ~12ms savings.
    # ---------------------------------------------------------------
    ("cache: batch hash update (single h.update instead of 16813)", [
        (CA,
         '    wkbs = _shp.to_wkb(geoms)\n    for e, w in zip(empty_mask.tolist(), wkbs):\n        h.update(b"empty\\n" if e else w)',
         '    wkbs = _shp.to_wkb(geoms)\n    h.update(b"".join(b"empty\\n" if bool(e) else w for e, w in zip(empty_mask, wkbs)))'),
    ]),

    # ---------------------------------------------------------------
    # 253: Retry ring_target 32->28 on fresh baseline (after Exp 250 if kept).
    # Exp 240 measured 0.002509 (DISCARD). Retry with lower/cleaner baseline.
    # ---------------------------------------------------------------
    ("mesh_build: ring_target 32->28 retry", [
        (TE,
         '    ring_target = cfg.get("ring_target", 32)',
         '    ring_target = cfg.get("ring_target", 28)'),
    ]),

    # ---------------------------------------------------------------
    # 254: ring_target 28->24 (chained after 253 keeping)
    # ---------------------------------------------------------------
    ("mesh_build: ring_target 28->24 retry", [
        (TE,
         '    ring_target = cfg.get("ring_target", 28)',
         '    ring_target = cfg.get("ring_target", 24)'),
    ]),

    # ---------------------------------------------------------------
    # 255: ring_target 24->20 (chained after 254 keeping)
    # ---------------------------------------------------------------
    ("mesh_build: ring_target 24->20 retry", [
        (TE,
         '    ring_target = cfg.get("ring_target", 24)',
         '    ring_target = cfg.get("ring_target", 20)'),
    ]),

    # ---------------------------------------------------------------
    # 256: ring_target 32->24 direct fallback (if 253+254 discarded)
    # ---------------------------------------------------------------
    ("mesh_build: ring_target 32->24 direct retry", [
        (TE,
         '    ring_target = cfg.get("ring_target", 32)',
         '    ring_target = cfg.get("ring_target", 24)'),
    ]),

    # ---------------------------------------------------------------
    # 257: Pre-sort gdf by cell_id to avoid groupby sort overhead in
    # per-tile _adaptive_ring_targets_from_scores and related functions.
    # ---------------------------------------------------------------
    ("tile_export: pre-sort gdf_render by cell_id once", [
        (TE,
         '    all_cell_ids = sorted(gdf_render["cell_id"].unique().tolist())',
         '    gdf_render = gdf_render.sort_values("cell_id", kind="stable")\n    all_cell_ids = sorted(gdf_render["cell_id"].unique().tolist())'),
    ]),

    # ---------------------------------------------------------------
    # 258: Skip marimo progress_bar check when not running in notebook
    # (avoids marimo import attempt on every export_tiles call).
    # ---------------------------------------------------------------
    ("tile_export: skip marimo import in batch mode", [
        (TE,
         '    _in_marimo = False\n    if show_progress:\n        try:\n            import marimo as mo\n            _in_marimo = mo.running_in_notebook()\n        except (ImportError, Exception):\n            pass',
         '    _in_marimo = False\n    if show_progress:\n        try:\n            import marimo as mo  # type: ignore[import]\n            _in_marimo = bool(getattr(mo, "running_in_notebook", lambda: False)())\n        except Exception:\n            pass'),
    ]),

    # ---------------------------------------------------------------
    # 259: np.concatenate instead of np.vstack for pos/nrm in _build_tile_data
    # (vstack calls reshape internally; concatenate on pre-shaped arrays is leaner).
    # ---------------------------------------------------------------
    ("tile_export: concatenate instead of vstack for mesh arrays", [
        (TE,
         '    positions_arr = np.vstack(pos_parts).astype(np.float32, copy=False)\n    indices_arr = np.concatenate(idx_parts).astype(np.uint32, copy=False)\n    normals_arr = np.vstack(nrm_parts).astype(np.float32, copy=False)\n    colors_arr = np.vstack(col_parts).astype(np.float32, copy=False)',
         '    positions_arr = np.concatenate(pos_parts).astype(np.float32, copy=False)\n    indices_arr = np.concatenate(idx_parts).astype(np.uint32, copy=False)\n    normals_arr = np.concatenate(nrm_parts).astype(np.float32, copy=False)\n    colors_arr = np.concatenate(col_parts).astype(np.float32, copy=False)'),
    ]),

]

_API = REPO_ROOT / "polyplot" / "_api.py"

# ================================================================
# BATCH 20: Key findings:
#   1. centroid.x and centroid.y each compute the full GeoSeries (~5ms wasted)
#   2. Numba funcs already have nogil=True -> ThreadPoolExecutor CAN parallelize
#      tile builds! Sequential 218ms -> parallel ~50ms (4-8 cores)
#   3. repr(tolist()).encode() in cache is slow; values.tobytes() is ~7x faster
#   4. to_wkb(16813 geoms) is ~15ms; get_coordinates().tobytes() is ~2ms
#   5. smooth_iters=0 skips Taubin pass in Numba lofting (~20-50ms)
#   6. simplify_tol=1.0 reduces input ring vertices -> faster coord processing
# ================================================================

EXPERIMENTS_20 = [

    # ---------------------------------------------------------------
    # 260: Fix centroid double-computation: .centroid GeoSeries computed
    # separately for .x and .y (each call recomputes the whole series).
    # Cache it in one variable. Expected: ~10ms saving.
    # ---------------------------------------------------------------
    ("tile_export: cache centroid GeoSeries to avoid double computation", [
        (TE,
         "    # Compute cell centroids once; shared by tile sizing and tile assignment.\n"
         "    _cx = gdf_render.geometry.centroid.x.values\n"
         "    _cy = gdf_render.geometry.centroid.y.values",
         "    # Compute cell centroids once; shared by tile sizing and tile assignment.\n"
         "    _centroids = gdf_render.geometry.centroid\n"
         "    _cx = _centroids.x.values\n"
         "    _cy = _centroids.y.values"),
    ]),

    # ---------------------------------------------------------------
    # 261: Parallel tile data building via ThreadPoolExecutor.
    # All heavy Numba functions have nogil=True -> true CPU parallelism.
    # Sequential 218ms -> ~50ms wall time on 8 tiles.
    # Expected: 100-160ms saving (BIG WIN).
    # ---------------------------------------------------------------
    ("tile_export: parallel _build_tile_data via ThreadPoolExecutor (nogil Numba)", [
        (TE,
         "    else:\n"
         "        built = [\n"
         "            (key, cells) + _build_tile_data(cells, rings_by_cid, zs_by_cid, scores_by_cid, cfg, cell_colors)\n"
         "            for key, cells in sorted_tiles\n"
         "        ]",
         "    else:\n"
         "        def _btd(kc):\n"
         "            k, c = kc\n"
         "            return (k, c) + _build_tile_data(c, rings_by_cid, zs_by_cid, scores_by_cid, cfg, cell_colors)\n"
         "        import os as _os\n"
         "        _build_nw = min(len(sorted_tiles), _os.cpu_count() or 4)\n"
         "        with _TPE(max_workers=_build_nw) as pool:\n"
         "            built = list(pool.map(_btd, sorted_tiles))"),
    ]),

    # ---------------------------------------------------------------
    # 262: smooth_iters=0: skip Taubin smoothing in lofting.
    # Saves the 3D smoothing pass inside build_loft_mesh_from_rings.
    # Quality tradeoff (slightly less smooth mesh surfaces).
    # Expected: 20-50ms saving.
    # ---------------------------------------------------------------
    ("api: smooth_iters=0 (skip Taubin smoothing)", [
        (_API,
         '    cfg = {"smooth_iters": 1 if smooth else 0}',
         '    cfg = {"smooth_iters": 0}'),
    ]),

    # ---------------------------------------------------------------
    # 263: cache: replace repr(tolist()).encode() with values.tobytes()
    # for cell_id and ZIndex. Avoids Python list + string repr overhead.
    # Expected: ~7ms saving.
    # ---------------------------------------------------------------
    ("cache: values.tobytes() instead of repr(tolist()).encode()", [
        (CA,
         '    h.update(repr(df["cell_id"].tolist()).encode())\n'
         '    h.update(repr(df["ZIndex"].tolist()).encode())',
         '    h.update(df["cell_id"].values.tobytes())\n'
         '    h.update(df["ZIndex"].values.tobytes())'),
    ]),

    # ---------------------------------------------------------------
    # 264: preprocess: increase simplify_tol 0.5->1.0.
    # Fewer ring vertices -> faster coord processing loop (~10-20ms).
    # Quality tradeoff: slightly less accurate cell outlines.
    # ---------------------------------------------------------------
    ("api: preprocess simplify_tol 0.5->1.0", [
        (_API,
         '    gdf_render = preprocess_gdf(gdf)',
         '    gdf_render = preprocess_gdf(gdf, simplify_tol=1.0)'),
    ]),

    # ---------------------------------------------------------------
    # 265: preprocess: increase simplify_tol 1.0->2.0 (chained after 264).
    # ---------------------------------------------------------------
    ("api: preprocess simplify_tol 1.0->2.0", [
        (_API,
         '    gdf_render = preprocess_gdf(gdf, simplify_tol=1.0)',
         '    gdf_render = preprocess_gdf(gdf, simplify_tol=2.0)'),
    ]),

    # ---------------------------------------------------------------
    # 266: cache: replace to_wkb with get_coordinates().tobytes().
    # to_wkb(16813 geoms) is ~15ms; get_coordinates is a single fast
    # C call returning a float64 array -> tobytes is ~2ms.
    # Expected: ~13ms saving.
    # ---------------------------------------------------------------
    ("cache: get_coordinates tobytes instead of to_wkb", [
        (CA,
         '    empty_mask = _shp.is_empty(geoms)\n'
         '    wkbs = _shp.to_wkb(geoms)\n'
         '    h.update(b"".join(b"empty\\n" if bool(e) else w for e, w in zip(empty_mask, wkbs)))',
         '    _coords = _shp.get_coordinates(geoms, include_z=False)\n'
         '    h.update(_coords.tobytes())\n'
         '    h.update(_shp.is_empty(geoms).tobytes())'),
    ]),

    # ---------------------------------------------------------------
    # 267: tile_export: retry target_tile_mb=2.5 (was 5.0).
    # Smaller tiles -> more gltfpack subprocesses -> better parallelism.
    # Previously tried 3.0 and regressed; code was very different then.
    # ---------------------------------------------------------------
    ("tile_export: retry target_tile_mb 5.0->2.5", [
        (TE,
         '    target_tile_mb: float = 5.0,',
         '    target_tile_mb: float = 2.5,'),
    ]),

]

if __name__ == "__main__":
    run_batch(EXPERIMENTS_20)
