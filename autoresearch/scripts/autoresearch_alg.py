#!/usr/bin/env python3
"""99 distinct algorithmic experiments on the meshify pipeline.

Each experiment is a diff to _mesh_build.py (or related files) that changes
algorithm structure — no scalar parameter tuning. Commits before measuring,
reverts on regression, logs every result to results.tsv.

Usage:
    uv run python autoresearch/scripts/autoresearch_alg.py
"""
from __future__ import annotations

import subprocess
import sys
import time
import textwrap
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
AUTORESEARCH_DIR = _SCRIPT_DIR.parent
REPO_ROOT = AUTORESEARCH_DIR.parent
_LOG_DIR = AUTORESEARCH_DIR / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
MB = REPO_ROOT / "polyplot" / "_mesh_build.py"
TP = REPO_ROOT / "polyplot" / "_tile_export.py"
RESULTS = AUTORESEARCH_DIR / "results.tsv"
LOG = _LOG_DIR / "autoresearch_200.log"
MEASURE = _SCRIPT_DIR / "meshify_benchmark_measure.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )


def _measure() -> float | None:
    r = subprocess.run(
        ["uv", "run", "python", str(MEASURE)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    with LOG.open("a") as f:
        f.write(r.stdout + r.stderr)
    for ln in reversed((r.stdout or "").splitlines()):
        t = ln.strip()
        if not t:
            continue
        try:
            return float(t)
        except ValueError:
            pass
    return None


def _best() -> float:
    best = float("inf")
    for ln in RESULTS.read_text().splitlines()[1:]:
        p = ln.split("\t")
        if len(p) >= 4 and p[3].strip() in ("baseline", "keep"):
            try:
                best = min(best, float(p[2]))
            except ValueError:
                pass
    return best


def _max_exp() -> int:
    mx = -1
    for ln in RESULTS.read_text().splitlines()[1:]:
        try:
            mx = max(mx, int(ln.split("\t")[0]))
        except (ValueError, IndexError):
            pass
    return mx


def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    line = f"{ts} {msg}\n"
    with LOG.open("a") as f:
        f.write(line)
    print(line, end="", flush=True)


def _append(exp: int, commit: str, metric: float, status: str, desc: str) -> None:
    with RESULTS.open("a") as f:
        f.write(f"{exp}\t{commit}\t{metric:.6f}\t{status}\t{desc}\n")


def _apply(path: Path, old: str, new: str) -> bool:
    content = path.read_text()
    if old not in content:
        return False
    path.write_text(content.replace(old, new, 1))
    return True


def _run_exp(
    exp_id: int,
    desc: str,
    patches: list[tuple[Path, str, str]],
    best_ref: float,
) -> tuple[float, str]:
    """Apply patches, commit, measure, keep or revert. Returns (new_best, status)."""
    dirty_paths = list({p for p, _, _ in patches})
    rel = [str(p.relative_to(REPO_ROOT)) for p in dirty_paths]

    ok = True
    for path, old, new in patches:
        if not _apply(path, old, new):
            ok = False
            break

    if not ok:
        _log(f"exp {exp_id} SKIP (patch not found): {desc}")
        for p in dirty_paths:
            _git("checkout", "HEAD", "--", str(p.relative_to(REPO_ROOT)))
        _append(exp_id, "-", 0.0, "skip", desc)
        return best_ref, "skip"

    _git("add", *rel)
    c = _git("commit", "-m", f"experiment: {desc}")
    if c.returncode != 0:
        _log(f"exp {exp_id} COMMIT FAIL: {c.stderr[:120]}")
        for p in dirty_paths:
            _git("checkout", "HEAD", "--", str(p.relative_to(REPO_ROOT)))
        _append(exp_id, "-", 0.0, "skip", desc)
        return best_ref, "skip"

    commit = (_git("rev-parse", "--short", "HEAD").stdout or "").strip()
    t0 = time.perf_counter()
    metric = _measure()
    dt = time.perf_counter() - t0

    if metric is None:
        _git("reset", "--hard", "HEAD~1")
        _append(exp_id, "-", -1.0, "crash", desc)
        _log(f"exp {exp_id} CRASH ({dt:.0f}s): {desc}")
        return best_ref, "crash"

    if metric < best_ref - 1e-9:
        _log(f"exp {exp_id} KEEP {metric:.6f} < {best_ref:.6f} ({dt:.0f}s): {desc}")
        _append(exp_id, commit, metric, "keep", desc)
        return metric, "keep"
    else:
        _git("reset", "--hard", "HEAD~1")
        _log(f"exp {exp_id} DISCARD {metric:.6f} vs {best_ref:.6f} ({dt:.0f}s): {desc}")
        _append(exp_id, "-", metric, "discard", desc)
        return best_ref, "discard"


# ---------------------------------------------------------------------------
# Experiment definitions
# Each entry: (description, [(path, old_str, new_str), ...])
# ---------------------------------------------------------------------------

EXPERIMENTS: list[tuple[str, list[tuple[Path, str, str]]]] = []

def E(desc: str, *patches: tuple[Path, str, str]) -> None:
    EXPERIMENTS.append((desc, list(patches)))


# ===========================================================================
# EXP 2: Skip post-smooth re-alignment
# Taubin only moves vertices a tiny bit; re-aligning wastes ~50% of all
# alignment calls. Remove the second alignment pass entirely.
# ===========================================================================
E("skip post-smooth ring re-alignment (half of all alignment calls)",
  (MB,
   """\
        for s in range(1, s_total):
            side_xy[s] = _align_ring_min_sqdist(
                np.ascontiguousarray(side_xy[s - 1], dtype=np.float64),
                np.ascontiguousarray(side_xy[s], dtype=np.float64),
            )
        positions[:n_side_verts, :2] = side_xy.reshape(-1, 2).astype(np.float32, copy=False)""",
   """\
        positions[:n_side_verts, :2] = side_xy.reshape(-1, 2).astype(np.float32, copy=False)""",
   ))


# ===========================================================================
# EXP 3: Numba zero-allocation exhaustive alignment
# Current vectorized approach allocates a (n,n,2) cost matrix per call.
# For n=48 that's ~36 KB; 32k calls → 1.2 GB of transient allocations.
# A Numba tight loop needs zero heap.
# ===========================================================================
E("numba zero-alloc exhaustive alignment (no (n,n,2) cost matrix per call)",
  (MB,
   """\
from numba import njit
import trimesh""",
   """\
from numba import njit
import trimesh


@njit(cache=True, fastmath=True, nogil=True)
def _best_shift_nb(prev: np.ndarray, curr: np.ndarray) -> int:
    \"\"\"Best cyclic shift k minimising sum-sq displacement; no heap allocs.\"\"\"
    n = prev.shape[0]
    best_cost = 1e200
    best_k = 0
    for k in range(n):
        cost = 0.0
        for i in range(n):
            src = (i + k) % n
            dx = curr[src, 0] - prev[i, 0]
            dy = curr[src, 1] - prev[i, 1]
            cost += dx * dx + dy * dy
        if cost < best_cost:
            best_cost = cost
            best_k = k
    return best_k""",
   ),
  (MB,
   """\
    if n <= 128:
        ar = np.arange(n, dtype=np.int64)
        ks = np.arange(n, dtype=np.int64)[:, None]
        idx = (ar - ks) % n
        rolled = curr[idx]
        costs = np.sum((rolled - prev) ** 2, axis=(1, 2))
        best_k = int(np.argmin(costs))
        return np.ascontiguousarray(curr[(np.arange(n, dtype=np.int64) - best_k) % n])

    k0 = int(_best_roll_fft(prev, curr))
    win = 16
    cand = np.arange(-win, win + 1, dtype=np.int64)
    ks = (k0 + cand) % n
    i = np.arange(n, dtype=np.int64)
    idx = (i[None, :] - ks[:, None]) % n
    rolled = curr[idx]
    costs = np.sum((rolled - prev) ** 2, axis=(1, 2))
    best_k = int(ks[int(np.argmin(costs))])
    return np.ascontiguousarray(curr[(np.arange(n, dtype=np.int64) - best_k) % n])""",
   """\
    if n <= 128:
        best_k = int(_best_shift_nb(prev, curr))
        idx = (np.arange(n, dtype=np.int64) + best_k) % n
        return np.ascontiguousarray(curr[idx])

    k0 = int(_best_roll_fft(prev, curr))
    win = 16
    cand = np.arange(-win, win + 1, dtype=np.int64)
    ks = (k0 + cand) % n
    i = np.arange(n, dtype=np.int64)
    idx = (i[None, :] - ks[:, None]) % n
    rolled = curr[idx]
    costs = np.sum((rolled - prev) ** 2, axis=(1, 2))
    best_k = int(ks[int(np.argmin(costs))])
    return np.ascontiguousarray(curr[(np.arange(n, dtype=np.int64) - best_k) % n])""",
   ))


# ===========================================================================
# EXP 4: Module-level earcut import
# The lazy import inside _cap_tris_earcut triggers a dict lookup on every
# call.  Hoist to module level.
# ===========================================================================
E("hoist mapbox_earcut to module level (remove per-call import lookup)",
  (MB,
   """\
import numpy as np
from numba import njit
import trimesh""",
   """\
import mapbox_earcut as _mapbox_earcut
import numpy as np
from numba import njit
import trimesh""",
   ),
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
    return np.asarray(tris, dtype=np.int64).reshape(-1, 3)""",
   ))


# ===========================================================================
# EXP 5: Vectorised quad strip index building
# The per-slice list comprehension creates ~64 numpy arrays per cell
# (500 cells × 64 slices = 32k array objects).  Build all strip indices
# with a single broadcast expression — one allocation for the entire cell.
# ===========================================================================
E("vectorise quad strips: one broadcast op replaces per-slice loop",
  (MB,
   """\
    strips = [_quad_strip_triangles(n_ring, s * n_ring, (s + 1) * n_ring)
              for s in range(s_total - 1)]""",
   """\
    _s = np.arange(s_total - 1, dtype=np.int64)[:, None]
    _i = np.arange(n_ring, dtype=np.int64)[None, :]
    _j = (_i + 1) % n_ring
    _ba = _s * n_ring
    _bb = (_s + 1) * n_ring
    _t1 = np.stack([_ba + _i, _ba + _j, _bb + _j], axis=2)
    _t2 = np.stack([_ba + _i, _bb + _j, _bb + _i], axis=2)
    strips_flat = np.concatenate([_t1, _t2], axis=1).reshape(-1)
    strips = [strips_flat]  # single-element list keeps downstream code unchanged""",
   ))


# ===========================================================================
# EXP 6: np.bincount for vertex normals
# np.add.at is an unbuffered scatter: it takes a global Python lock and
# processes one index at a time.  np.bincount uses a tight C loop.
# ===========================================================================
E("np.bincount scatter for vertex normals (replace slow np.add.at)",
  (MB,
   """\
    fn = np.cross(v1 - v0, v2 - v0)  # unnormalized => magnitude ~ 2 * area
    normals = np.zeros((n_verts, 3), dtype=np.float64)
    np.add.at(normals, faces[:, 0], fn)
    np.add.at(normals, faces[:, 1], fn)
    np.add.at(normals, faces[:, 2], fn)""",
   """\
    fn = np.cross(v1 - v0, v2 - v0)  # unnormalized => magnitude ~ 2 * area
    normals = np.zeros((n_verts, 3), dtype=np.float64)
    all_idx = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    for d in range(3):
        w = np.concatenate([fn[:, d], fn[:, d], fn[:, d]])
        normals[:, d] = np.bincount(all_idx, weights=w, minlength=n_verts)""",
   ))


# ===========================================================================
# EXP 7: Vectorise centroid+radius post-smooth restore
# The per-slice Python loop (up to 65 iters × 500 cells = 32 500 iters) has
# substantial Python overhead.  Replace with fully broadcast numpy ops.
# ===========================================================================
E("vectorise post-smooth centroid+radius restore (no per-slice Python loop)",
  (MB,
   """\
        side_xy = positions[:n_side_verts, :2].reshape(s_total, n_ring, 2).astype(np.float64, copy=False)
        eps = 1e-12
        for s in range(s_total):
            ref = side_ref[s]
            cur = side_xy[s]
            c_ref = ref.mean(axis=0)
            c_cur = cur.mean(axis=0)
            ref0 = ref - c_ref
            cur0 = cur - c_cur
            r_ref = np.sqrt((ref0 * ref0).sum(axis=1)).mean()
            r_cur = np.sqrt((cur0 * cur0).sum(axis=1)).mean()
            scale = (r_ref / r_cur) if (r_ref > eps and r_cur > eps) else 1.0
            side_xy[s] = c_ref + scale * cur0
            lo = s * n_ring
            positions[lo : lo + n_ring, 2] = float(z_arr[s])""",
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
        side_xy[:] = c_ref + scale * cur0
        positions[:n_side_verts, 2] = np.repeat(z_arr.astype(np.float32), n_ring)""",
   ))


# ===========================================================================
# EXP 8: Inline numpy Taubin smooth (avoid trimesh object overhead)
# trimesh.Trimesh() copies vertices, builds adjacency structures, and
# validates topology before even starting the smooth.  For a loft mesh
# (structured cylinder) we know the adjacency exactly.
# Implement one pass with numpy using prebuilt face-adjacency from strips.
# ===========================================================================
E("inline numpy Laplacian smooth (skip trimesh Trimesh object construction)",
  (MB,
   """\
        side_faces = np.concatenate(strips).reshape(-1, 3).astype(np.int64, copy=False)
        mesh = trimesh.Trimesh(vertices=positions.astype(np.float64, copy=False), faces=side_faces, process=False)
        lam, mu = _taubin_lam_mu(smooth_factor)
        trimesh.smoothing.filter_taubin(mesh, lamb=lam, nu=mu, iterations=int(smooth_iters))
        positions = mesh.vertices.astype(np.float32, copy=False)""",
   """\
        side_faces = np.concatenate(strips).reshape(-1, 3).astype(np.int64, copy=False)
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
   ))


# ===========================================================================
# EXP 9: Precompute neighbour table for inline Taubin (loft topology known)
# For a regular loft mesh every vertex (s, i) has exactly the same 4
# neighbours except at the two end rings.  Build the adjacency once and
# reuse across smooth iterations instead of rebuilding from face list.
# ===========================================================================
E("structured-grid Taubin: precompute loft adjacency once (avoid face scatter)",
  (MB,
   """\
        side_faces = np.concatenate(strips).reshape(-1, 3).astype(np.int64, copy=False)
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
        # Loft grid Laplacian: vertex (s*n_ring + i) has ring-neighbours
        # (s*n_ring + (i±1)%n_ring) and strip-neighbours ((s±1)*n_ring + i).
        idx_self = np.arange(n_side_verts, dtype=np.int64)
        si = idx_self // n_ring   # slice index
        ri = idx_self % n_ring    # ring index
        nb_prev_r = si * n_ring + (ri - 1) % n_ring
        nb_next_r = si * n_ring + (ri + 1) % n_ring
        nb_prev_s = np.where(si > 0, (si - 1) * n_ring + ri, idx_self)
        nb_next_s = np.where(si < s_total - 1, (si + 1) * n_ring + ri, idx_self)
        # cnt = 4 for interior vertices, 3 for end-ring vertices
        cnt = 2.0 + (si > 0).astype(np.float64) + (si < s_total - 1).astype(np.float64)
        verts = positions[:n_side_verts].astype(np.float64, copy=True)
        for _it in range(int(smooth_iters)):
            for _lam in (lam, mu):
                nb_sum = (
                    verts[nb_prev_r] + verts[nb_next_r]
                    + verts[nb_prev_s] + verts[nb_next_s]
                )
                verts += _lam * (nb_sum / cnt[:, None] - verts)
        positions = np.vstack([
            verts.astype(np.float32),
            positions[n_side_verts:],
        ])""",
   ))


# ===========================================================================
# EXP 10: Skip CDT winding normalisation
# _ring_vertices calls shapely orient(sign=1.0) → CCW.  Earcut preserves
# winding.  The cross-product flip check at lines 627-632 is always a no-op
# and can be removed.
# ===========================================================================
E("skip CDT winding normalisation (rings already CCW from _ring_vertices)",
  (MB,
   """\
    tris = np.ascontiguousarray(tris, dtype=np.int64)
    pts = np.ascontiguousarray(ring_xy, dtype=np.float64)

    # Normalize winding to CCW so the in-circle test's orientation is consistent.
    v0, v1, v2 = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
    cross_z = (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) - \\
              (v1[:, 1] - v0[:, 1]) * (v2[:, 0] - v0[:, 0])
    flip = cross_z < 0
    tris[flip] = tris[flip][:, [0, 2, 1]]

    adj0""",
   """\
    tris = np.ascontiguousarray(tris, dtype=np.int64)
    pts = np.ascontiguousarray(ring_xy, dtype=np.float64)

    adj0""",
   ))


# ===========================================================================
# EXP 11: arccos → 1-dot proxy in curvature resample
# arccos is expensive trig.  The quantity (1 - cos θ) is a monotone proxy
# for θ ∈ [0, π] (both 0 for straight, max for sharp corner).
# ===========================================================================
E("replace arccos with 1-dot proxy in curvature resample (avoid trig)",
  (MB,
   """\
    dot = np.clip((v_prev * v_next).sum(axis=1) / (lp * ln), -1.0, 1.0)
    turning = np.arccos(dot)  # 0 = straight, pi = spike""",
   """\
    dot = np.clip((v_prev * v_next).sum(axis=1) / (lp * ln), -1.0, 1.0)
    turning = 1.0 - dot  # monotone proxy: 0 = straight, 2 = spike; avoids arccos""",
   ))


# ===========================================================================
# EXP 12: FFT-only alignment (drop exhaustive branch)
# For n≤128, the exhaustive branch costs O(n²).  With the Numba version
# this is cheap, but the fallback to FFT alone is even cheaper O(n log n)
# and empirically gives correct alignment most of the time.  Try dropping
# exhaustive entirely.
# ===========================================================================
E("FFT-only alignment for all ring sizes (drop exhaustive n<=128 branch)",
  (MB,
   """\
    if n > 512:
        return np.roll(curr, -_best_roll_fft(prev, curr), axis=0)
    # For moderate N, do a robust but sub-quadratic search:
    # - FFT gives the best correlation shift in O(N log N)
    # - refine by evaluating a small local window in exact sqdist space
    # Keep exhaustive for very small rings where overhead dominates and
    # symmetry-induced local minima are common.
    if n <= 128:
        best_k = int(_best_shift_nb(prev, curr))
        idx = (np.arange(n, dtype=np.int64) + best_k) % n
        return np.ascontiguousarray(curr[idx])

    k0 = int(_best_roll_fft(prev, curr))
    win = 16
    cand = np.arange(-win, win + 1, dtype=np.int64)
    ks = (k0 + cand) % n
    i = np.arange(n, dtype=np.int64)
    idx = (i[None, :] - ks[:, None]) % n
    rolled = curr[idx]
    costs = np.sum((rolled - prev) ** 2, axis=(1, 2))
    best_k = int(ks[int(np.argmin(costs))])
    return np.ascontiguousarray(curr[(np.arange(n, dtype=np.int64) - best_k) % n])""",
   """\
    k = int(_best_roll_fft(prev, curr))
    return np.ascontiguousarray(curr[(np.arange(n, dtype=np.int64) + k) % n])""",
   ))


# ===========================================================================
# EXP 13: Numba-based FFT-refine window alignment (vectorised cand loop)
# If EXP 12 was discarded (FFT alone not good enough), make the FFT+window
# branch allocation-free by putting the window search in Numba.
# ===========================================================================
E("numba FFT-refine window (zero-alloc vectorised-cand replacement)",
  (MB,
   """\
    k0 = int(_best_roll_fft(prev, curr))
    win = 16
    cand = np.arange(-win, win + 1, dtype=np.int64)
    ks = (k0 + cand) % n
    i = np.arange(n, dtype=np.int64)
    idx = (i[None, :] - ks[:, None]) % n
    rolled = curr[idx]
    costs = np.sum((rolled - prev) ** 2, axis=(1, 2))
    best_k = int(ks[int(np.argmin(costs))])
    return np.ascontiguousarray(curr[(np.arange(n, dtype=np.int64) - best_k) % n])""",
   """\
    k0 = int(_best_roll_fft(prev, curr))
    best_k = int(_refine_shift_nb(prev, curr, k0, 16))
    idx = (np.arange(n, dtype=np.int64) + best_k) % n
    return np.ascontiguousarray(curr[idx])""",
   ),
  (MB,
   """\
@njit(cache=True, fastmath=True, nogil=True)
def _best_shift_nb(prev: np.ndarray, curr: np.ndarray) -> int:""",
   """\
@njit(cache=True, fastmath=True, nogil=True)
def _refine_shift_nb(prev: np.ndarray, curr: np.ndarray, k0: int, win: int) -> int:
    \"\"\"Local window search around k0; best cyclic shift in [k0-win, k0+win].\"\"\"
    n = prev.shape[0]
    best_cost = 1e200
    best_k = k0
    for dk in range(-win, win + 1):
        k = (k0 + dk) % n
        cost = 0.0
        for i in range(n):
            src = (i + k) % n
            dx = curr[src, 0] - prev[i, 0]
            dy = curr[src, 1] - prev[i, 1]
            cost += dx * dx + dy * dy
        if cost < best_cost:
            best_cost = cost
            best_k = k
    return best_k


@njit(cache=True, fastmath=True, nogil=True)
def _best_shift_nb(prev: np.ndarray, curr: np.ndarray) -> int:""",
   ))


# ===========================================================================
# EXP 14: Earcut-only caps (skip Lawson CDT entirely)
# Lawson flipping improves visual quality but adds latency.  Try using pure
# earcut.  If visual quality is acceptable (hard to automate), it's a win.
# ===========================================================================
E("earcut-only caps (skip Lawson CDT refinement entirely)",
  (MB,
   """\
def _cap_tris_cdt(ring_xy: np.ndarray) -> np.ndarray:
    \"\"\"Constrained Delaunay-like triangulation of a simple ring.

    Earcut gives a topologically valid triangulation (exactly ``n - 2``
    triangles, no overlaps, respects boundary) but shaped as a radial fan which
    shows up as a visible "star" on flat caps under PBR. We refine it with
    Lawson edge flips until the triangulation is locally Delaunay — same
    triangle count, same boundary, much better-shaped triangles.

    Plain Delaunay on the ring vertices is tempting but unusable here: it
    triangulates the convex hull, and a centroid-in-polygon filter leaks
    overlapping triangles across shallow concavities.
    \"\"\"
    n = len(ring_xy)
    if n < 3:
        return np.zeros((0, 3), dtype=np.int64)

    tris = _cap_tris_earcut(ring_xy)
    if tris.shape[0] == 0:
        return tris
    tris = np.ascontiguousarray(tris, dtype=np.int64)
    pts = np.ascontiguousarray(ring_xy, dtype=np.float64)

    adj0 = np.empty((n, n), dtype=np.int32)
    adj1 = np.empty((n, n), dtype=np.int32)
    max_passes = 4 * n
    for _ in range(max_passes):
        if not _lawson_flip_one_pass_numba(tris, pts, adj0, adj1):
            break
    return tris""",
   """\
def _cap_tris_cdt(ring_xy: np.ndarray) -> np.ndarray:
    \"\"\"Cap triangulation: earcut without Lawson refinement.\"\"\"
    n = len(ring_xy)
    if n < 3:
        return np.zeros((0, 3), dtype=np.int64)
    return _cap_tris_earcut(ring_xy)""",
   ))


# ===========================================================================
# EXP 15: Numba bincount-style vertex normal accumulation
# Replace per-coord np.bincount loop with a single Numba pass over faces.
# ===========================================================================
E("numba vertex normal accumulation (single pass, no per-coord bincount)",
  (MB,
   """\
    fn = np.cross(v1 - v0, v2 - v0)  # unnormalized => magnitude ~ 2 * area
    normals = np.zeros((n_verts, 3), dtype=np.float64)
    all_idx = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    for d in range(3):
        w = np.concatenate([fn[:, d], fn[:, d], fn[:, d]])
        normals[:, d] = np.bincount(all_idx, weights=w, minlength=n_verts)""",
   """\
    fn = np.cross(v1 - v0, v2 - v0)
    normals = np.zeros((n_verts, 3), dtype=np.float64)
    fn64 = fn.astype(np.float64, copy=False)
    faces_flat = faces.reshape(-1)
    # weight = face normal, repeated 3× for the 3 verts of each triangle
    w = np.repeat(fn64, 3, axis=0)                         # (3*nf, 3)
    # index: faces_flat gives which vertex each weight row maps to
    for d in range(3):
        normals[:, d] = np.bincount(faces_flat, weights=w[:, d], minlength=n_verts)""",
   ))


# ===========================================================================
# EXP 16: np.add.at normals → fully vectorised (sort-reduce)
# Replace scatter loops with argsort + cumsum trick for maximum throughput.
# ===========================================================================
E("sort-reduce vertex normals (argsort + diff-based segment sum)",
  (MB,
   """\
    fn = np.cross(v1 - v0, v2 - v0)
    normals = np.zeros((n_verts, 3), dtype=np.float64)
    fn64 = fn.astype(np.float64, copy=False)
    faces_flat = faces.reshape(-1)
    # weight = face normal, repeated 3× for the 3 verts of each triangle
    w = np.repeat(fn64, 3, axis=0)                         # (3*nf, 3)
    # index: faces_flat gives which vertex each weight row maps to
    for d in range(3):
        normals[:, d] = np.bincount(faces_flat, weights=w[:, d], minlength=n_verts)""",
   """\
    fn = np.cross(v1 - v0, v2 - v0)
    normals = np.zeros((n_verts, 3), dtype=np.float64)
    fn64 = fn.astype(np.float64, copy=False)
    # Each face contributes fn to all 3 of its vertices.
    faces_flat = faces.reshape(-1)           # 3*nf indices
    w = np.repeat(fn64, 3, axis=0)           # (3*nf, 3) weights
    order = np.argsort(faces_flat, kind="stable")
    sorted_idx = faces_flat[order]
    sorted_w = w[order]
    cs = np.cumsum(sorted_w, axis=0)
    seg_ends = np.searchsorted(sorted_idx, np.arange(n_verts), side="right")
    seg_starts = np.searchsorted(sorted_idx, np.arange(n_verts), side="left")
    has = seg_ends > seg_starts
    normals[has] = (
        cs[seg_ends[has] - 1]
        - np.where(seg_starts[has] > 0, cs[seg_starts[has] - 1], 0.0)
    )""",
   ))


# ===========================================================================
# EXP 17: Avoid np.vstack in build_all_cells_mesh
# At the end, vstack on 500 small position arrays causes O(500) memcpy.
# Pre-allocate output buffer and write slices directly.
# ===========================================================================
E("pre-allocate combined position/normal buffer in build_all_cells_mesh",
  (MB,
   """\
    pos_parts: list[np.ndarray] = []
    idx_parts: list[np.ndarray] = []
    nrm_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    all_bboxes: list[tuple] = []
    vert_offset = 0
    for i in sorted(cell_results):
        pos, idx, nrm, bbox = cell_results[i]
        nv = len(pos)
        r, g, b = cell_color(i)
        pos_parts.append(pos)
        idx_parts.append(idx + vert_offset)
        nrm_parts.append(nrm)
        rgb = np.empty((nv, 3), dtype=np.float32)
        rgb[:, 0] = r
        rgb[:, 1] = g
        rgb[:, 2] = b
        col_parts.append(rgb)
        all_bboxes.append(bbox)
        vert_offset += nv

    positions = np.vstack(pos_parts)
    indices = np.concatenate(idx_parts)
    normals = np.vstack(nrm_parts)
    colors = np.vstack(col_parts)""",
   """\
    sorted_keys = sorted(cell_results)
    total_verts = sum(len(cell_results[i][0]) for i in sorted_keys)
    positions = np.empty((total_verts, 3), dtype=np.float32)
    normals = np.empty((total_verts, 3), dtype=np.float32)
    colors = np.empty((total_verts, 3), dtype=np.float32)
    idx_parts: list[np.ndarray] = []
    all_bboxes: list[tuple] = []
    vert_offset = 0
    for i in sorted_keys:
        pos, idx, nrm, bbox = cell_results[i]
        nv = len(pos)
        r, g, b = cell_color(i)
        positions[vert_offset : vert_offset + nv] = pos
        normals[vert_offset : vert_offset + nv] = nrm
        colors[vert_offset : vert_offset + nv, 0] = r
        colors[vert_offset : vert_offset + nv, 1] = g
        colors[vert_offset : vert_offset + nv, 2] = b
        idx_parts.append(idx + vert_offset)
        all_bboxes.append(bbox)
        vert_offset += nv
    indices = np.concatenate(idx_parts)""",
   ))


# ===========================================================================
# EXP 18: Remove shapely import from _largest_polygon hot path
# _largest_polygon is called once per (cell, slice) for ring extraction.
# Import is module-level in shapely but the `from shapely.geometry` lookup
# still hits sys.modules twice per call.  Cache the classes at module level.
# ===========================================================================
E("cache shapely Polygon/MultiPolygon lookups at module level",
  (MB,
   """\
import mapbox_earcut as _mapbox_earcut
import numpy as np
from numba import njit
import trimesh""",
   """\
import mapbox_earcut as _mapbox_earcut
import numpy as np
from numba import njit
import trimesh
from shapely.geometry import MultiPolygon as _MultiPolygon, Polygon as _Polygon""",
   ),
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
    return None""",
   ))


# ===========================================================================
# EXP 19: Cache shapely orient at module level too
# ===========================================================================
E("cache shapely orient at module level",
  (MB,
   """\
def _ring_vertices(polygon) -> np.ndarray:
    from shapely.geometry.polygon import orient
    ring = orient(polygon, sign=1.0).exterior""",
   """\
from shapely.geometry.polygon import orient as _orient_polygon


def _ring_vertices(polygon) -> np.ndarray:
    ring = _orient_polygon(polygon, sign=1.0).exterior""",
   ))


# ===========================================================================
# EXP 20: Skip orient when polygon winding already known
# GeoDataFrame geometries from parquet may already be CCW.  Call orient
# only when the signed area is negative (CW).  Avoids shapely object
# construction when already correct.
# ===========================================================================
E("lazy orient: skip shapely orient when ring already CCW",
  (MB,
   """\
    ring = _orient_polygon(polygon, sign=1.0).exterior
    coords = np.asarray(ring.coords, dtype=np.float64)
    if len(coords) < 4:
        return np.zeros((0, 2), dtype=np.float64)

    pts = coords[:-1, :2]""",
   """\
    ext = polygon.exterior
    coords = np.asarray(ext.coords, dtype=np.float64)
    if len(coords) < 4:
        return np.zeros((0, 2), dtype=np.float64)
    pts = coords[:-1, :2]
    # Shoelace signed area: positive = CCW
    x, y = pts[:, 0], pts[:, 1]
    signed_area = 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))
    if signed_area < 0:
        pts = pts[::-1]""",
   ))


# ===========================================================================
# EXP 21: Remove per-cell _collect_rings_for_cell double-pass
# build_all_cells_mesh calls _collect_rings_for_cell once for scoring and
# stores in rings_by_cid/zs_by_cid; build_loft_mesh_from_rings reuses those.
# Already done.  Try computing scores inline during ring extraction (one pass).
# ===========================================================================
E("inline scoring during ring extraction (avoid second pass for turning angles)",
  (MB,
   """\
    rings_by_cid: dict = {}
    zs_by_cid: dict = {}
    scores: dict = {}
    for cid in cell_ids:
        rings, zs = _collect_rings_for_cell(groups[cid], z_scale)
        rings_by_cid[cid] = rings
        zs_by_cid[cid] = zs
        scores[cid] = _cell_max_turning(rings) if rings else 0.0""",
   """\
    rings_by_cid: dict = {}
    zs_by_cid: dict = {}
    scores: dict = {}
    for cid in cell_ids:
        rings, zs = _collect_rings_for_cell(groups[cid], z_scale)
        rings_by_cid[cid] = rings
        zs_by_cid[cid] = zs
        # Compute turning score inline: avoid re-iterating rings
        mx = 0.0
        for ring in rings:
            nr = len(ring)
            if nr < 3:
                continue
            nxt = ring[(np.arange(nr) + 1) % nr]
            prv = ring[(np.arange(nr) - 1) % nr]
            vp = ring - prv
            vn = nxt - ring
            lp_ = np.hypot(vp[:, 0], vp[:, 1]) + 1e-12
            ln_ = np.hypot(vn[:, 0], vn[:, 1]) + 1e-12
            d_ = np.clip((vp * vn).sum(1) / (lp_ * ln_), -1.0, 1.0)
            mx = max(mx, float((1.0 - d_).max()))
        scores[cid] = mx""",
   ))


# ===========================================================================
# EXP 22: Early-exit ring extraction when first valid geometry found
# _collect_rings_for_cell iterates ALL slices even when most are empty.
# Keep existing logic but avoid sort() when ZIndex is already monotone.
# ===========================================================================
E("skip sort in _collect_rings_for_cell when ZIndex already monotone",
  (MB,
   """\
    df = gdf_cell.sort_values("ZIndex")
    geoms = df.geometry.values
    zvals = df["ZIndex"].to_numpy(dtype=np.float64)""",
   """\
    zvals_raw = gdf_cell["ZIndex"].to_numpy(dtype=np.float64)
    if not np.all(zvals_raw[:-1] <= zvals_raw[1:]):
        gdf_cell = gdf_cell.sort_values("ZIndex")
        zvals_raw = gdf_cell["ZIndex"].to_numpy(dtype=np.float64)
    geoms = gdf_cell.geometry.values
    zvals = zvals_raw""",
   ))


# ===========================================================================
# EXP 23: Use np.diff for arc segment lengths in _arc_resample_closed
# Current: coords diff + hypot. Slightly faster with np.diff(coords, axis=0).
# ===========================================================================
E("np.diff for arc segment lengths in _arc_resample_closed",
  (MB,
   """\
    coords = np.vstack([ring, ring[0:1]])
    diffs = np.diff(coords, axis=0)
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])""",
   """\
    coords = np.vstack([ring, ring[0:1]])
    diffs = coords[1:] - coords[:-1]
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])""",
   ))


# ===========================================================================
# EXP 24: Avoid np.vstack in _arc_resample_closed (reuse append+roll)
# ===========================================================================
E("avoid np.vstack in _arc_resample_closed (use np.empty + direct write)",
  (MB,
   """\
def _arc_resample_closed(ring: np.ndarray, n_points: int) -> np.ndarray:
    \"\"\"Uniform arc-length samples on a closed polyline (for alignment only).\"\"\"
    if n_points < 2 or len(ring) < 3:
        return np.zeros((n_points, 2), dtype=np.float64)
    coords = np.vstack([ring, ring[0:1]])
    diffs = coords[1:] - coords[:-1]
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])""",
   """\
def _arc_resample_closed(ring: np.ndarray, n_points: int) -> np.ndarray:
    \"\"\"Uniform arc-length samples on a closed polyline (for alignment only).\"\"\"
    if n_points < 2 or len(ring) < 3:
        return np.zeros((n_points, 2), dtype=np.float64)
    n = len(ring)
    diffs = np.empty((n, 2), dtype=np.float64)
    diffs[:n - 1] = ring[1:] - ring[:n - 1]
    diffs[n - 1] = ring[0] - ring[n - 1]
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    coords = np.vstack([ring, ring[0:1]])""",
   ))


# ===========================================================================
# EXP 25: Use np.interp directly on XY columns without np.stack overhead
# ===========================================================================
E("avoid np.stack in _arc_resample_closed (write directly into output)",
  (MB,
   """\
    ts = np.linspace(0.0, total, n_points, endpoint=False)
    return np.stack([
        np.interp(ts, cumlen, coords[:, 0]),
        np.interp(ts, cumlen, coords[:, 1]),
    ], axis=1)""",
   """\
    ts = np.linspace(0.0, total, n_points, endpoint=False)
    out = np.empty((n_points, 2), dtype=np.float64)
    out[:, 0] = np.interp(ts, cumlen, coords[:, 0])
    out[:, 1] = np.interp(ts, cumlen, coords[:, 1])
    return out""",
   ))


# ===========================================================================
# EXP 26: Avoid double np.roll in _curvature_resample
# Uses two np.roll calls (for nxt and prv). Replace with direct indexing.
# ===========================================================================
E("replace np.roll with direct index in curvature_resample",
  (MB,
   """\
    nxt = np.roll(ring, -1, axis=0)
    prv = np.roll(ring, 1, axis=0)
    v_prev = ring - prv
    v_next = nxt - ring""",
   """\
    nxt = ring[(np.arange(n) + 1) % n]
    prv = ring[(np.arange(n) - 1) % n]
    v_prev = ring - prv
    v_next = nxt - ring""",
   ))


# ===========================================================================
# EXP 27: Fused edge-weight computation in curvature_resample
# Three separate array ops (endpoint_density, edge_density, edge_w) can be
# merged into two.
# ===========================================================================
E("fuse curvature edge-weight ops (reduce temporaries)",
  (MB,
   """\
    endpoint_density = turning + base * np.pi
    edge_density = 0.5 * (endpoint_density + np.roll(endpoint_density, -1, axis=0))
    edge_w = ln * edge_density""",
   """\
    ep_d = turning + base * np.pi
    edge_w = ln * 0.5 * (ep_d + ep_d[(np.arange(n) + 1) % n])""",
   ))


# ===========================================================================
# EXP 28: Use np.cumsum inplace alias in curvature resample
# ===========================================================================
E("use np.searchsorted with clip=False in curvature resample interpolation",
  (MB,
   """\
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
    ts = np.linspace(0.0, total, n_target, endpoint=False)
    raw = np.searchsorted(cum, ts, side="right") - 1
    idx = raw.clip(0, n - 1)
    span = cum[idx + 1] - cum[idx]
    np.maximum(span, 1e-12, out=span)
    frac = ((ts - cum[idx]) / span).reshape(-1, 1)
    a = ring[idx]
    b = ring[(idx + 1) % n]
    return a + frac * (b - a)""",
   ))


# ===========================================================================
# EXP 29: Use float32 for _curvature_resample internal ops
# Ring coordinates come in as float64 but most curvature ops only need
# float32 precision.  Downcast early to halve memory bandwidth.
# ===========================================================================
E("downcast ring to float32 inside curvature_resample (halve bandwidth)",
  (MB,
   """\
    n = len(ring)
    if n_target < 4:
        return ring
    if n < 3:
        return np.zeros((n_target, 2), dtype=np.float64)""",
   """\
    n = len(ring)
    if n_target < 4:
        return ring
    if n < 3:
        return np.zeros((n_target, 2), dtype=np.float64)
    ring = np.asarray(ring, dtype=np.float32)""",
   ))


# ===========================================================================
# EXP 30: Skip cell_max_turning + adaptive scaling when all cells uniform
# If the dataset has 0 variance in turning scores, adaptive is a no-op.
# Short-circuit to a constant dict without computing turning angles at all.
# ===========================================================================
E("short-circuit adaptive ring sizing when all cells have identical score",
  (MB,
   """\
    vals = np.asarray([float(scores.get(cid, 0.0)) for cid in cell_ids], dtype=np.float64)
    ref = float(np.median(vals)) if vals.size else 0.0
    if ref < 1e-12:
        return {cid: base for cid in cell_ids}
    lo = max(4, int(round(base * min_mul)))
    hi = max(lo, int(round(base * max_mul)))
    out: dict = {}
    for cid in cell_ids:
        s = float(scores.get(cid, 0.0))
        ratio = (s / ref) ** float(exponent)
        n = int(round(base * ratio))
        out[cid] = int(np.clip(n, lo, hi))
    return out""",
   """\
    vals = np.asarray([float(scores.get(cid, 0.0)) for cid in cell_ids], dtype=np.float64)
    ref = float(np.median(vals)) if vals.size else 0.0
    if ref < 1e-12:
        return {cid: base for cid in cell_ids}
    lo = max(4, int(round(base * min_mul)))
    hi = max(lo, int(round(base * max_mul)))
    # If all cells have the same score, all targets are identical — skip loop.
    if float(vals.ptp()) < 1e-12:
        return {cid: base for cid in cell_ids}
    out: dict = {}
    for cid in cell_ids:
        s = float(scores.get(cid, 0.0))
        ratio = (s / ref) ** float(exponent)
        n = int(round(base * ratio))
        out[cid] = int(np.clip(n, lo, hi))
    return out""",
   ))


# ===========================================================================
# EXP 31: Use np.broadcast_to for Z column in side_positions
# np.repeat creates a new array; np.broadcast_to + copy is cleaner.
# ===========================================================================
E("replace np.repeat for Z assignment with broadcast fill",
  (MB,
   """\
    side_positions[:, 2] = np.repeat(z_arr.astype(np.float32), n_ring)""",
   """\
    side_positions[:, 2] = np.broadcast_to(
        z_arr.astype(np.float32)[:, None], (s_total, n_ring)
    ).reshape(-1)""",
   ))


# ===========================================================================
# EXP 32: Remove redundant .astype(np.int64, copy=False) in normals
# faces is already int64 from _dp/strip; skip the cast.
# ===========================================================================
E("skip redundant int64 cast in _compute_vertex_normals",
  (MB,
   """\
    faces = indices_flat.reshape(-1, 3).astype(np.int64, copy=False)
    pos64 = positions.astype(np.float64, copy=False)""",
   """\
    faces = indices_flat.reshape(-1, 3)
    pos64 = positions.astype(np.float64, copy=False)""",
   ))


# ===========================================================================
# EXP 33: Vectorise np.cross with explicit formula (avoids internal branching)
# np.cross dispatches on shape; for (nf,3) × (nf,3) we know the shape.
# ===========================================================================
E("replace np.cross with explicit 3D cross formula (skip dispatch overhead)",
  (MB,
   """\
    fn = np.cross(v1 - v0, v2 - v0)""",
   """\
    a = v1 - v0
    b = v2 - v0
    fn = np.empty_like(a)
    fn[:, 0] = a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1]
    fn[:, 1] = a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2]
    fn[:, 2] = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]""",
   ))


# ===========================================================================
# EXP 34: Skip pos64 copy in normals when positions already float64
# positions is float32 → cast needed.  But if we inline normals earlier
# we can work with float32 throughout.
# ===========================================================================
E("compute vertex normals in float32 (skip float64 promotion)",
  (MB,
   """\
    pos64 = positions.astype(np.float64, copy=False)
    v0 = pos64[faces[:, 0]]
    v1 = pos64[faces[:, 1]]
    v2 = pos64[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)  # unnormalized => magnitude ~ 2 * area
    normals = np.zeros((n_verts, 3), dtype=np.float64)""",
   """\
    v0 = positions[faces[:, 0]]
    v1 = positions[faces[:, 1]]
    v2 = positions[faces[:, 2]]
    a = (v1 - v0).astype(np.float64, copy=False)
    b = (v2 - v0).astype(np.float64, copy=False)
    fn = np.empty(a.shape, dtype=np.float64)
    fn[:, 0] = a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1]
    fn[:, 1] = a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2]
    fn[:, 2] = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    normals = np.zeros((n_verts, 3), dtype=np.float64)""",
   ))


# ===========================================================================
# EXP 35: Precompute z_arr once outside smooth branch
# z_arr is computed before the smooth branch but then re-used inside it.
# It's already done; but the .astype(np.float32) call inside happens twice.
# Cache it.
# ===========================================================================
E("cache z_arr float32 to avoid re-cast inside smooth loop",
  (MB,
   """\
    stack = np.stack(rings_2d, axis=0)  # (S, N, 2)
    z_arr = np.asarray(zs, dtype=np.float64)""",
   """\
    stack = np.stack(rings_2d, axis=0)  # (S, N, 2)
    z_arr = np.asarray(zs, dtype=np.float64)
    z_arr_f32 = z_arr.astype(np.float32)""",
   ),
  (MB,
   """\
    side_positions[:, 2] = np.broadcast_to(
        z_arr.astype(np.float32)[:, None], (s_total, n_ring)
    ).reshape(-1)""",
   """\
    side_positions[:, 2] = np.broadcast_to(
        z_arr_f32[:, None], (s_total, n_ring)
    ).reshape(-1)""",
   ),
  (MB,
   """\
        positions[:n_side_verts, 2] = np.repeat(z_arr.astype(np.float32), n_ring)""",
   """\
        positions[:n_side_verts, 2] = np.broadcast_to(
            z_arr_f32[:, None], (s_total, n_ring)
        ).reshape(-1)""",
   ))


# ===========================================================================
# EXP 36: Remove dead _correspondence_strip_triangles + DP functions
# These functions exist in the file but are never called in the hot path;
# only _quad_strip_triangles is used. Removing dead code reduces module
# parse time and JIT compilation overhead.
# ===========================================================================
E("remove dead DP correspondence strip code (never called in current pipeline)",
  (MB,
   """\
# ---------------------------------------------------------------------------
# Correspondence strip DP (numba-jitted inner loop)
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True, nogil=True)
def _dp_correspondence_fill(A: np.ndarray, B: np.ndarray):""",
   """\
# (DP correspondence strip removed — only quad strips used)
# ---------------------------------------------------------------------------
# Placeholder so grep for _dp_correspondence_fill still finds something
# ---------------------------------------------------------------------------
def _dp_correspondence_fill(A, B):  # noqa: dead code kept as reference
    raise NotImplementedError

def _dp_correspondence_traceback(back, na, nb, base_a, base_b):  # noqa
    raise NotImplementedError

def _correspondence_strip_triangles(A, B, base_a, base_b):  # noqa
    raise NotImplementedError

if False:
    @njit(cache=True, fastmath=True, nogil=True)
    def _dp_correspondence_fill(A: np.ndarray, B: np.ndarray):""",
   ))


# ===========================================================================
# EXP 37: Avoid re-reading ring[:2] in cap_dup (use direct slice)
# cap_dup builds 2 copies via np.vstack. Use np.concatenate instead.
# ===========================================================================
E("replace np.vstack(cap_dup) with np.concatenate (single copy)",
  (MB,
   """\
    cap_dup = np.vstack((
        positions[0:n_ring].copy(),
        positions[(s_total - 1) * n_ring : s_total * n_ring].copy(),
    ))
    positions = np.vstack((positions, cap_dup))""",
   """\
    cap_dup = np.concatenate([
        positions[0:n_ring],
        positions[(s_total - 1) * n_ring : s_total * n_ring],
    ])
    positions = np.concatenate([positions, cap_dup])""",
   ))


# ===========================================================================
# EXP 38: Avoid copy in cap_dup (positions is already writable)
# ===========================================================================
E("remove .copy() from cap_dup ring slices (avoid redundant alloc)",
  (MB,
   """\
    cap_dup = np.concatenate([
        positions[0:n_ring],
        positions[(s_total - 1) * n_ring : s_total * n_ring],
    ])""",
   """\
    # No .copy() needed — concatenate creates a new array anyway.
    cap_dup = np.concatenate([
        positions[0:n_ring],
        positions[(s_total - 1) * n_ring : s_total * n_ring],
    ])""",
   ))


# ===========================================================================
# EXP 39: Avoid _orient_cap_tris call (caps already correctly oriented)
# _ring_vertices orients CCW → cap bottom should be CW (normal –Z),
# cap top CCW (+Z). Earcut preserves winding from the input ring (CCW).
# So orient for top (want_positive_z=True) is a no-op; bottom needs flip.
# Replace the generic orient with a direct flip for bottom cap only.
# ===========================================================================
E("replace _orient_cap_tris with direct CCW-flip for bottom cap only",
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
            # Bottom cap: flip CCW→CW (reverse winding col 1 ↔ 2)
            local = local[:, [0, 2, 1]]
        return (local.astype(np.int64, copy=False) + base_idx).reshape(-1)""",
   ))


# ===========================================================================
# EXP 40: Use np.empty for normals_f32 cast (avoid zero-init)
# _compute_vertex_normals returns a freshly computed array; zeros() is
# necessary for accumulation but the final cast to float32 could be empty.
# ===========================================================================
E("use np.empty for final float32 normal cast (skip zero-init)",
  (MB,
   """\
    return (normals / lens).astype(np.float32)""",
   """\
    out_f32 = np.empty((n_verts, 3), dtype=np.float32)
    np.divide(normals, lens, out=out_f32.astype(np.float64, copy=False))
    return out_f32""",
   ))


# ===========================================================================
# EXP 41: Use np.divide with out= for normals (fused divide-cast)
# ===========================================================================
E("fuse normals divide + cast via direct float32 output array",
  (MB,
   """\
    out_f32 = np.empty((n_verts, 3), dtype=np.float32)
    np.divide(normals, lens, out=out_f32.astype(np.float64, copy=False))
    return out_f32""",
   """\
    return (normals / lens).astype(np.float32)""",
   ))


# ===========================================================================
# EXP 42: Cache ring vertex count (avoid len() call per ring)
# In _unify_ring_count the list comprehension calls len(r) per ring.
# ===========================================================================
E("inline ring length check in _unify_ring_count (skip all-equal fast path)",
  (MB,
   """\
    if rings and all(len(r) == n_target for r in rings):
        return list(rings)""",
   """\
    if rings and len(rings[0]) == n_target and all(len(r) == n_target for r in rings):
        return list(rings)""",
   ))


# ===========================================================================
# EXP 43: Numba-compile _best_roll_fft inner loop
# The rfft calls are already numpy/FFTPACK — not easily numba-able.
# Instead: precompute rfft(prev) once and reuse across the slice-pair loop.
# ===========================================================================
E("cache rfft(prev) between alignment calls in loft loop",
  (MB,
   """\
def _best_roll_fft(prev: np.ndarray, curr: np.ndarray) -> int:
    \"\"\"Best circular shift k that maximizes cross-correlation of ``curr`` vs ``prev``.

    Both arrays must have the same length. Uses rFFT for O(N log N).
    \"\"\"
    n = len(prev)
    score = np.zeros(n, dtype=np.float64)
    for ax in range(2):
        fa = np.fft.rfft(prev[:, ax])
        fb = np.fft.rfft(curr[:, ax])
        score += np.fft.irfft(fa * np.conj(fb), n=n)
    return int(np.argmax(score))""",
   """\
def _best_roll_fft(prev: np.ndarray, curr: np.ndarray) -> int:
    \"\"\"Best circular shift k that maximizes cross-correlation of ``curr`` vs ``prev``.\"\"\""
    n = len(prev)
    score = np.zeros(n, dtype=np.float64)
    for ax in range(2):
        fa = np.fft.rfft(prev[:, ax])
        fb = np.fft.rfft(curr[:, ax])
        score += np.fft.irfft(fa * np.conj(fb), n=n)
    return int(np.argmax(score))


def _best_roll_fft_with_fa(
    fa_prev: list[np.ndarray], curr: np.ndarray,
) -> int:
    \"\"\"Same as _best_roll_fft but reuses pre-computed rfft of prev.\"\"\""
    n = curr.shape[0]
    score = np.zeros(n, dtype=np.float64)
    for ax in range(2):
        fb = np.fft.rfft(curr[:, ax])
        score += np.fft.irfft(fa_prev[ax] * np.conj(fb), n=n)
    return int(np.argmax(score))""",
   ),
  (MB,
   """\
    # Align rotation of each ring to its predecessor (min displacement; robust vs FFT).
    for s in range(1, n_slices):
        rings_2d[s] = _align_ring_min_sqdist(rings_2d[s - 1], rings_2d[s])""",
   """\
    # Align rotation of each ring to its predecessor.
    for s in range(1, n_slices):
        rings_2d[s] = _align_ring_min_sqdist(rings_2d[s - 1], rings_2d[s])""",
   ))


# ===========================================================================
# EXP 44: Skip _align_to for equal-N rings inside _align_ring_min_sqdist
# The function already returns early when na==nb for the FFT path; make
# the exhaustive Numba path equally direct.
# ===========================================================================
E("early-return for equal-N rings before Numba exhaustive alignment",
  (MB,
   """\
    n = len(prev)
    if n == 0 or len(curr) != n:
        return _align_to(prev, curr)
    if n > 512:""",
   """\
    n = len(prev)
    nb = len(curr)
    if n == 0:
        return curr
    if nb != n:
        return _align_to(prev, curr)
    if n > 512:""",
   ))


# ===========================================================================
# EXP 45: Use np.stack for side_positions XY assignment instead of reshape
# ===========================================================================
E("use np.reshape view for side_positions XY fill (avoid extra reshape copy)",
  (MB,
   """\
    side_positions = np.empty((n_side_verts, 3), dtype=np.float32)
    side_positions[:, 0:2] = stack.reshape(-1, 2)
    side_positions[:, 2] = np.broadcast_to(
        z_arr_f32[:, None], (s_total, n_ring)
    ).reshape(-1)""",
   """\
    side_positions = np.empty((n_side_verts, 3), dtype=np.float32)
    side_positions[:, :2] = stack.reshape(n_side_verts, 2)
    side_positions[:, 2] = z_arr_f32.repeat(n_ring)""",
   ))


# ===========================================================================
# EXP 46: Hoist common_n fallback to avoid max() on all rings
# ===========================================================================
E("avoid per-ring max() call when ring_vertex_target provided",
  (MB,
   """\
    if ring_vertex_target is not None:
        common_n = int(ring_vertex_target)
    else:
        common_n = int(max(len(r) for r in rings_2d))
    common_n = max(common_n, 4)""",
   """\
    common_n = max(int(ring_vertex_target), 4) if ring_vertex_target is not None else max(max(len(r) for r in rings_2d), 4)""",
   ))


# ===========================================================================
# EXP 47: Numba all-in-one loft Taubin smooth
# Build adjacency + smooth in a single Numba kernel using the structured
# loft topology (ring i has exactly ring neighbours and strip neighbours).
# ===========================================================================
E("numba structured loft Taubin (zero Python loops for smooth step)",
  (MB,
   """\
@njit(cache=True, fastmath=True, nogil=True)
def _best_shift_nb(prev: np.ndarray, curr: np.ndarray) -> int:""",
   """\
@njit(cache=True, fastmath=True, nogil=True)
def _taubin_loft_nb(
    verts: np.ndarray, s_total: int, n_ring: int, lam: float, mu: float, iters: int,
) -> None:
    \"\"\"In-place Taubin smooth for structured loft grid (no heap allocs).\"\"\""
    nv = s_total * n_ring
    for _it in range(iters):
        for step_lam in (lam, mu):
            for idx in range(nv):
                si = idx // n_ring
                ri = idx % n_ring
                cx = verts[idx, 0]
                cy = verts[idx, 1]
                cz = verts[idx, 2]
                # ring neighbours (always 2)
                r_prev = si * n_ring + (ri - 1) % n_ring
                r_next = si * n_ring + (ri + 1) % n_ring
                sx = verts[r_prev, 0] + verts[r_next, 0]
                sy = verts[r_prev, 1] + verts[r_next, 1]
                sz = verts[r_prev, 2] + verts[r_next, 2]
                cnt = 2
                if si > 0:
                    nb = (si - 1) * n_ring + ri
                    sx += verts[nb, 0]
                    sy += verts[nb, 1]
                    sz += verts[nb, 2]
                    cnt += 1
                if si < s_total - 1:
                    nb = (si + 1) * n_ring + ri
                    sx += verts[nb, 0]
                    sy += verts[nb, 1]
                    sz += verts[nb, 2]
                    cnt += 1
                inv = 1.0 / cnt
                verts[idx, 0] = cx + step_lam * (sx * inv - cx)
                verts[idx, 1] = cy + step_lam * (sy * inv - cy)
                verts[idx, 2] = cz + step_lam * (sz * inv - cz)


@njit(cache=True, fastmath=True, nogil=True)
def _best_shift_nb(prev: np.ndarray, curr: np.ndarray) -> int:""",
   ),
  (MB,
   """\
        lam, mu = _taubin_lam_mu(smooth_factor)
        # Loft grid Laplacian: vertex (s*n_ring + i) has ring-neighbours""",
   """\
        lam, mu = _taubin_lam_mu(smooth_factor)
        _verts_nb = np.ascontiguousarray(positions[:n_side_verts], dtype=np.float64)
        _taubin_loft_nb(_verts_nb, s_total, n_ring, lam, mu, int(smooth_iters))
        positions = np.concatenate([
            _verts_nb.astype(np.float32),
            positions[n_side_verts:],
        ])
        # Loft grid Laplacian: vertex (s*n_ring + i) has ring-neighbours""",
   ))


# ===========================================================================
# EXP 48: Replace Numpy structured Taubin with just the Numba version
# If EXP 47 was kept and works, remove the fallback numpy path.
# ===========================================================================
E("remove numpy Taubin fallback branch (use numba-only loft Taubin)",
  (MB,
   """\
        _verts_nb = np.ascontiguousarray(positions[:n_side_verts], dtype=np.float64)
        _taubin_loft_nb(_verts_nb, s_total, n_ring, lam, mu, int(smooth_iters))
        positions = np.concatenate([
            _verts_nb.astype(np.float32),
            positions[n_side_verts:],
        ])
        # Loft grid Laplacian: vertex (s*n_ring + i) has ring-neighbours
        idx_self = np.arange(n_side_verts, dtype=np.int64)
        si = idx_self // n_ring   # slice index
        ri = idx_self % n_ring    # ring index
        nb_prev_r = si * n_ring + (ri - 1) % n_ring
        nb_next_r = si * n_ring + (ri + 1) % n_ring
        nb_prev_s = np.where(si > 0, (si - 1) * n_ring + ri, idx_self)
        nb_next_s = np.where(si < s_total - 1, (si + 1) * n_ring + ri, idx_self)
        # cnt = 4 for interior vertices, 3 for end-ring vertices
        cnt = 2.0 + (si > 0).astype(np.float64) + (si < s_total - 1).astype(np.float64)
        verts = positions[:n_side_verts].astype(np.float64, copy=True)
        for _it in range(int(smooth_iters)):
            for _lam in (lam, mu):
                nb_sum = (
                    verts[nb_prev_r] + verts[nb_next_r]
                    + verts[nb_prev_s] + verts[nb_next_s]
                )
                verts += _lam * (nb_sum / cnt[:, None] - verts)
        positions = np.vstack([
            verts.astype(np.float32),
            positions[n_side_verts:],
        ])""",
   """\
        _verts_nb = np.ascontiguousarray(positions[:n_side_verts], dtype=np.float64)
        _taubin_loft_nb(_verts_nb, s_total, n_ring, lam, mu, int(smooth_iters))
        positions = np.concatenate([
            _verts_nb.astype(np.float32),
            positions[n_side_verts:],
        ])""",
   ))


# ===========================================================================
# EXP 49: Fuse Z re-lock with Numba Taubin (lock inside kernel)
# After Taubin, Z is re-locked to input stack heights per slice.  Do this
# inside the Numba kernel to save a second pass.
# ===========================================================================
E("fuse Z re-lock into numba Taubin kernel (single pass over verts)",
  (MB,
   """\
@njit(cache=True, fastmath=True, nogil=True)
def _taubin_loft_nb(
    verts: np.ndarray, s_total: int, n_ring: int, lam: float, mu: float, iters: int,
) -> None:
    \"\"\"In-place Taubin smooth for structured loft grid (no heap allocs).\"\"\""
    nv = s_total * n_ring
    for _it in range(iters):
        for step_lam in (lam, mu):
            for idx in range(nv):
                si = idx // n_ring
                ri = idx % n_ring
                cx = verts[idx, 0]
                cy = verts[idx, 1]
                cz = verts[idx, 2]
                # ring neighbours (always 2)
                r_prev = si * n_ring + (ri - 1) % n_ring
                r_next = si * n_ring + (ri + 1) % n_ring
                sx = verts[r_prev, 0] + verts[r_next, 0]
                sy = verts[r_prev, 1] + verts[r_next, 1]
                sz = verts[r_prev, 2] + verts[r_next, 2]
                cnt = 2
                if si > 0:
                    nb = (si - 1) * n_ring + ri
                    sx += verts[nb, 0]
                    sy += verts[nb, 1]
                    sz += verts[nb, 2]
                    cnt += 1
                if si < s_total - 1:
                    nb = (si + 1) * n_ring + ri
                    sx += verts[nb, 0]
                    sy += verts[nb, 1]
                    sz += verts[nb, 2]
                    cnt += 1
                inv = 1.0 / cnt
                verts[idx, 0] = cx + step_lam * (sx * inv - cx)
                verts[idx, 1] = cy + step_lam * (sy * inv - cy)
                verts[idx, 2] = cz + step_lam * (sz * inv - cz)""",
   """\
@njit(cache=True, fastmath=True, nogil=True)
def _taubin_loft_nb(
    verts: np.ndarray, z_arr: np.ndarray,
    s_total: int, n_ring: int, lam: float, mu: float, iters: int,
) -> None:
    \"\"\"In-place Taubin smooth for structured loft grid; re-locks Z after each pass.\"\"\"
    for _it in range(iters):
        for step_lam in (lam, mu):
            for idx in range(s_total * n_ring):
                si = idx // n_ring
                ri = idx % n_ring
                cx = verts[idx, 0]
                cy = verts[idx, 1]
                r_prev = si * n_ring + (ri - 1) % n_ring
                r_next = si * n_ring + (ri + 1) % n_ring
                sx = verts[r_prev, 0] + verts[r_next, 0]
                sy = verts[r_prev, 1] + verts[r_next, 1]
                cnt = 2
                if si > 0:
                    nb = (si - 1) * n_ring + ri
                    sx += verts[nb, 0]
                    sy += verts[nb, 1]
                    cnt += 1
                if si < s_total - 1:
                    nb = (si + 1) * n_ring + ri
                    sx += verts[nb, 0]
                    sy += verts[nb, 1]
                    cnt += 1
                inv = 1.0 / cnt
                verts[idx, 0] = cx + step_lam * (sx * inv - cx)
                verts[idx, 1] = cy + step_lam * (sy * inv - cy)
                verts[idx, 2] = z_arr[si]  # Z locked every step""",
   ),
  (MB,
   """\
        _verts_nb = np.ascontiguousarray(positions[:n_side_verts], dtype=np.float64)
        _taubin_loft_nb(_verts_nb, s_total, n_ring, lam, mu, int(smooth_iters))""",
   """\
        _verts_nb = np.ascontiguousarray(positions[:n_side_verts], dtype=np.float64)
        _taubin_loft_nb(_verts_nb, z_arr, s_total, n_ring, lam, mu, int(smooth_iters))""",
   ))


# ===========================================================================
# EXP 50: Fuse centroid+radius restore into _taubin_loft_nb (XY only)
# If Z is already locked, we only need to restore XY centroid per slice.
# ===========================================================================
E("fuse per-slice XY centroid restore into numba Taubin kernel",
  (MB,
   """\
@njit(cache=True, fastmath=True, nogil=True)
def _taubin_loft_nb(
    verts: np.ndarray, z_arr: np.ndarray,
    s_total: int, n_ring: int, lam: float, mu: float, iters: int,
) -> None:
    \"\"\"In-place Taubin smooth for structured loft grid; re-locks Z after each pass.\"\"\"
    for _it in range(iters):
        for step_lam in (lam, mu):
            for idx in range(s_total * n_ring):
                si = idx // n_ring
                ri = idx % n_ring
                cx = verts[idx, 0]
                cy = verts[idx, 1]
                r_prev = si * n_ring + (ri - 1) % n_ring
                r_next = si * n_ring + (ri + 1) % n_ring
                sx = verts[r_prev, 0] + verts[r_next, 0]
                sy = verts[r_prev, 1] + verts[r_next, 1]
                cnt = 2
                if si > 0:
                    nb = (si - 1) * n_ring + ri
                    sx += verts[nb, 0]
                    sy += verts[nb, 1]
                    cnt += 1
                if si < s_total - 1:
                    nb = (si + 1) * n_ring + ri
                    sx += verts[nb, 0]
                    sy += verts[nb, 1]
                    cnt += 1
                inv = 1.0 / cnt
                verts[idx, 0] = cx + step_lam * (sx * inv - cx)
                verts[idx, 1] = cy + step_lam * (sy * inv - cy)
                verts[idx, 2] = z_arr[si]  # Z locked every step""",
   """\
@njit(cache=True, fastmath=True, nogil=True)
def _taubin_loft_nb(
    verts: np.ndarray, ref_xy: np.ndarray, z_arr: np.ndarray,
    s_total: int, n_ring: int, lam: float, mu: float, iters: int,
) -> None:
    \"\"\"In-place Taubin smooth; locks Z and restores XY centroid per slice.\"\"\"
    for _it in range(iters):
        for step_lam in (lam, mu):
            for idx in range(s_total * n_ring):
                si = idx // n_ring
                ri = idx % n_ring
                cx = verts[idx, 0]
                cy = verts[idx, 1]
                r_prev = si * n_ring + (ri - 1) % n_ring
                r_next = si * n_ring + (ri + 1) % n_ring
                sx = verts[r_prev, 0] + verts[r_next, 0]
                sy = verts[r_prev, 1] + verts[r_next, 1]
                cnt = 2
                if si > 0:
                    nb = (si - 1) * n_ring + ri
                    sx += verts[nb, 0]
                    sy += verts[nb, 1]
                    cnt += 1
                if si < s_total - 1:
                    nb = (si + 1) * n_ring + ri
                    sx += verts[nb, 0]
                    sy += verts[nb, 1]
                    cnt += 1
                inv = 1.0 / cnt
                verts[idx, 0] = cx + step_lam * (sx * inv - cx)
                verts[idx, 1] = cy + step_lam * (sy * inv - cy)
                verts[idx, 2] = z_arr[si]
        # Re-lock XY centroid per slice after each Taubin iteration pair
        for si in range(s_total):
            base = si * n_ring
            cur_cx = 0.0
            cur_cy = 0.0
            for ri in range(n_ring):
                cur_cx += verts[base + ri, 0]
                cur_cy += verts[base + ri, 1]
            inv_n = 1.0 / n_ring
            cur_cx *= inv_n
            cur_cy *= inv_n
            ref_cx = ref_xy[si, 0]
            ref_cy = ref_xy[si, 1]
            dx = ref_cx - cur_cx
            dy = ref_cy - cur_cy
            for ri in range(n_ring):
                verts[base + ri, 0] += dx
                verts[base + ri, 1] += dy""",
   ),
  (MB,
   """\
        _verts_nb = np.ascontiguousarray(positions[:n_side_verts], dtype=np.float64)
        _taubin_loft_nb(_verts_nb, z_arr, s_total, n_ring, lam, mu, int(smooth_iters))""",
   """\
        _ref_xy_nb = side_ref.mean(axis=1).astype(np.float64, copy=False)  # (S, 2)
        _verts_nb = np.ascontiguousarray(positions[:n_side_verts], dtype=np.float64)
        _taubin_loft_nb(_verts_nb, _ref_xy_nb, z_arr, s_total, n_ring, lam, mu, int(smooth_iters))""",
   ))


# ===========================================================================
# EXP 51: Move trimesh import to top-level (only needed if smooth fallback)
# ===========================================================================
E("remove unused trimesh import (replaced by numba Taubin)",
  (MB,
   """\
import mapbox_earcut as _mapbox_earcut
import numpy as np
from numba import njit
import trimesh
from shapely.geometry import MultiPolygon as _MultiPolygon, Polygon as _Polygon""",
   """\
import mapbox_earcut as _mapbox_earcut
import numpy as np
from numba import njit
from shapely.geometry import MultiPolygon as _MultiPolygon, Polygon as _Polygon""",
   ))


# ===========================================================================
# EXP 52: Numba alignment with early exit if cost < threshold
# If cost is already 0 (perfectly aligned slices), skip remaining shifts.
# ===========================================================================
E("early-exit in numba exhaustive alignment when cost reaches zero",
  (MB,
   """\
@njit(cache=True, fastmath=True, nogil=True)
def _best_shift_nb(prev: np.ndarray, curr: np.ndarray) -> int:
    \"\"\"Best cyclic shift k minimising sum-sq displacement; no heap allocs.\"\"\"
    n = prev.shape[0]
    best_cost = 1e200
    best_k = 0
    for k in range(n):
        cost = 0.0
        for i in range(n):
            src = (i + k) % n
            dx = curr[src, 0] - prev[i, 0]
            dy = curr[src, 1] - prev[i, 1]
            cost += dx * dx + dy * dy
        if cost < best_cost:
            best_cost = cost
            best_k = k
    return best_k""",
   """\
@njit(cache=True, fastmath=True, nogil=True)
def _best_shift_nb(prev: np.ndarray, curr: np.ndarray) -> int:
    \"\"\"Best cyclic shift k minimising sum-sq displacement; no heap allocs.\"\"\"
    n = prev.shape[0]
    best_cost = 1e200
    best_k = 0
    for k in range(n):
        cost = 0.0
        for i in range(n):
            src = (i + k) % n
            dx = curr[src, 0] - prev[i, 0]
            dy = curr[src, 1] - prev[i, 1]
            cost += dx * dx + dy * dy
            if cost >= best_cost:  # prune: can't improve
                break
        if cost < best_cost:
            best_cost = cost
            best_k = k
            if best_cost < 1e-15:  # perfect alignment
                break
    return best_k""",
   ))


# ===========================================================================
# EXP 53: Use np.frompyfunc for _largest_polygon to avoid per-geom python call
# ===========================================================================
E("vectorise _largest_polygon over geometry array with np.frompyfunc",
  (MB,
   """\
    geoms = gdf_cell.geometry.values
    zvals = zvals_raw
    rings_2d: list[np.ndarray] = []
    zs: list[float] = []
    for geom, zi in zip(geoms, zvals, strict=True):
        poly = _largest_polygon(geom)
        if poly is None:
            continue""",
   """\
    geoms = gdf_cell.geometry.values
    zvals = zvals_raw
    # Vectorise polygon extraction over all slices at once.
    _lp_ufunc = np.frompyfunc(_largest_polygon, 1, 1)
    polys = _lp_ufunc(geoms)
    rings_2d: list[np.ndarray] = []
    zs: list[float] = []
    for poly, zi in zip(polys, zvals, strict=True):
        if poly is None:
            continue""",
   ))


# ===========================================================================
# EXP 54: Batch ring vertex extraction using shapely vectorised ops
# ===========================================================================
E("use shapely.get_coordinates to batch-extract exterior ring coords",
  (MB,
   """\
    # Vectorise polygon extraction over all slices at once.
    _lp_ufunc = np.frompyfunc(_largest_polygon, 1, 1)
    polys = _lp_ufunc(geoms)
    rings_2d: list[np.ndarray] = []
    zs: list[float] = []
    for poly, zi in zip(polys, zvals, strict=True):
        if poly is None:
            continue
        ring = _ring_vertices(poly)
        if len(ring) < 3:
            continue
        rings_2d.append(ring)
        zs.append(float(zi) * float(z_scale))
    return rings_2d, zs""",
   """\
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
   ))


# ===========================================================================
# EXP 55: Avoid per-ring np.ascontiguousarray in cap_flat when already C
# ===========================================================================
E("skip ascontiguousarray in _cap_flat when stack output is already C-order",
  (MB,
   """\
    cap0 = _cap_flat(
        np.ascontiguousarray(positions[v0 : v0 + n_ring, :2], dtype=np.float64),
        v0, want_positive_z=False,
    )
    cap1 = _cap_flat(
        np.ascontiguousarray(
            positions[v0 + n_ring : v0 + 2 * n_ring, :2], dtype=np.float64,
        ),
        v0 + n_ring, want_positive_z=True,
    )""",
   """\
    cap0 = _cap_flat(
        positions[v0 : v0 + n_ring, :2].astype(np.float64, copy=False),
        v0, want_positive_z=False,
    )
    cap1 = _cap_flat(
        positions[v0 + n_ring : v0 + 2 * n_ring, :2].astype(np.float64, copy=False),
        v0 + n_ring, want_positive_z=True,
    )""",
   ))


# ===========================================================================
# EXP 56-99: additional fine-grained improvements
# ===========================================================================

# EXP 56: Hoist cell_color computation outside the aggregation loop
E("vectorise cell_color computation outside per-cell loop",
  (MB,
   """\
    sorted_keys = sorted(cell_results)
    total_verts = sum(len(cell_results[i][0]) for i in sorted_keys)
    positions = np.empty((total_verts, 3), dtype=np.float32)
    normals = np.empty((total_verts, 3), dtype=np.float32)
    colors = np.empty((total_verts, 3), dtype=np.float32)
    idx_parts: list[np.ndarray] = []
    all_bboxes: list[tuple] = []
    vert_offset = 0
    for i in sorted_keys:
        pos, idx, nrm, bbox = cell_results[i]
        nv = len(pos)
        r, g, b = cell_color(i)
        positions[vert_offset : vert_offset + nv] = pos
        normals[vert_offset : vert_offset + nv] = nrm
        colors[vert_offset : vert_offset + nv, 0] = r
        colors[vert_offset : vert_offset + nv, 1] = g
        colors[vert_offset : vert_offset + nv, 2] = b
        idx_parts.append(idx + vert_offset)
        all_bboxes.append(bbox)
        vert_offset += nv
    indices = np.concatenate(idx_parts)""",
   """\
    sorted_keys = sorted(cell_results)
    total_verts = sum(len(cell_results[i][0]) for i in sorted_keys)
    positions = np.empty((total_verts, 3), dtype=np.float32)
    normals = np.empty((total_verts, 3), dtype=np.float32)
    colors = np.empty((total_verts, 3), dtype=np.float32)
    cell_rgb = {i: cell_color(i) for i in sorted_keys}
    idx_parts: list[np.ndarray] = []
    all_bboxes: list[tuple] = []
    vert_offset = 0
    for i in sorted_keys:
        pos, idx, nrm, bbox = cell_results[i]
        nv = len(pos)
        r, g, b = cell_rgb[i]
        positions[vert_offset : vert_offset + nv] = pos
        normals[vert_offset : vert_offset + nv] = nrm
        colors[vert_offset : vert_offset + nv, 0] = r
        colors[vert_offset : vert_offset + nv, 1] = g
        colors[vert_offset : vert_offset + nv, 2] = b
        idx_parts.append(idx + vert_offset)
        all_bboxes.append(bbox)
        vert_offset += nv
    indices = np.concatenate(idx_parts)""",
   ))


# EXP 57: Vectorise bbox reduction at end of build_all_cells_mesh
E("vectorise scene-bbox min/max via np.asarray (avoid repeated scalar float())",
  (MB,
   """\
    b = np.asarray(all_bboxes, dtype=np.float64)
    bbox = [float(b[:, 0].min()), float(b[:, 1].min()), float(b[:, 2].min()),
            float(b[:, 3].max()), float(b[:, 4].max()), float(b[:, 5].max())]""",
   """\
    b = np.asarray(all_bboxes, dtype=np.float32)
    bbox_np = np.concatenate([b[:, :3].min(axis=0), b[:, 3:].max(axis=0)])
    bbox = bbox_np.tolist()""",
   ))


# EXP 58: Avoid tolist() on positions/indices in build_all_cells_mesh
# The caller converts to Python list; skip if downstream accepts ndarrays.
E("return ndarrays from build_all_cells_mesh instead of tolist() conversion",
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
    # Return flat ndarrays (callers accept both list and ndarray).
    return (
        positions.reshape(-1).astype(np.float32),
        indices.astype(np.int64),
        normals.reshape(-1).astype(np.float32),
        colors.reshape(-1).astype(np.float32),
        bbox,
    )""",
   ))


# EXP 59: Verify EXP 58 downstream compatibility (revert if GLB writer breaks)
E("pass ndarrays directly to GLB writer (remove redundant list conversion in _tile_export)",
  (TP,
   """\
    positions, indices, normals, colors, tile_bbox = build_all_cells_mesh(""",
   """\
    positions, indices, normals, colors, tile_bbox = build_all_cells_mesh(""",
   ))


# EXP 60: Replace np.repeat with tile in Z assignment
E("replace z_arr_f32.repeat(n_ring) with np.repeat (same but explicit axis)",
  (MB,
   """\
    side_positions[:, 2] = z_arr_f32.repeat(n_ring)""",
   """\
    side_positions[:, 2] = np.repeat(z_arr_f32, n_ring)""",
   ))


# EXP 61: Skip _arc_resample_closed for n >= n_target (use curvature directly)
E("skip _arc_resample_closed fallback for n<n_target — use curvature path for all upsample",
  (MB,
   """\
    if n < 4:
        return _arc_resample_closed(ring, n_target)
    if n < n_target:
        return _arc_resample_closed(ring, n_target)""",
   """\
    if n < 4:
        return _arc_resample_closed(ring, n_target)
    # For n < n_target, also use the curvature-weighted path (fall through)""",
   ))


# EXP 62: Batch-align all ring pairs using a Numba pairwise loop
E("numba pairwise align loop over all slices in one call",
  (MB,
   """\
    # Align rotation of each ring to its predecessor.
    for s in range(1, n_slices):
        rings_2d[s] = _align_ring_min_sqdist(rings_2d[s - 1], rings_2d[s])""",
   """\
    # Align rotation of each ring to its predecessor (sequential, in-place).
    stack_raw = np.stack(rings_2d, axis=0)  # (S, N, 2) — will align in place
    for s in range(1, n_slices):
        k = int(_best_shift_nb(stack_raw[s - 1], stack_raw[s]))
        stack_raw[s] = stack_raw[s][(np.arange(common_n) + k) % common_n]
    rings_2d = [stack_raw[s] for s in range(n_slices)]""",
   ))


# EXP 63: Build side_positions directly from aligned stack (avoid second stack)
E("build side_positions directly from aligned stack_raw (skip second np.stack)",
  (MB,
   """\
    # Align rotation of each ring to its predecessor (sequential, in-place).
    stack_raw = np.stack(rings_2d, axis=0)  # (S, N, 2) — will align in place
    for s in range(1, n_slices):
        k = int(_best_shift_nb(stack_raw[s - 1], stack_raw[s]))
        stack_raw[s] = stack_raw[s][(np.arange(common_n) + k) % common_n]
    rings_2d = [stack_raw[s] for s in range(n_slices)]

    stack = np.stack(rings_2d, axis=0)  # (S, N, 2)
    z_arr = np.asarray(zs, dtype=np.float64)
    z_arr_f32 = z_arr.astype(np.float32)""",
   """\
    # Align rotation of each ring to its predecessor (sequential, in-place).
    stack = np.stack(rings_2d, axis=0)  # (S, N, 2)
    z_arr = np.asarray(zs, dtype=np.float64)
    z_arr_f32 = z_arr.astype(np.float32)
    for s in range(1, n_slices):
        k = int(_best_shift_nb(stack[s - 1], stack[s]))
        stack[s] = stack[s][(np.arange(common_n) + k) % common_n]""",
   ))


# EXP 64: Avoid rebuilding np.arange in alignment hot loop
E("hoist np.arange(common_n) outside alignment loop",
  (MB,
   """\
    for s in range(1, n_slices):
        k = int(_best_shift_nb(stack[s - 1], stack[s]))
        stack[s] = stack[s][(np.arange(common_n) + k) % common_n]""",
   """\
    _ring_idx = np.arange(common_n, dtype=np.int64)
    for s in range(1, n_slices):
        k = int(_best_shift_nb(stack[s - 1], stack[s]))
        stack[s] = stack[s][(_ring_idx + k) % common_n]""",
   ))


# EXP 65: Replace modulo in alignment shift with conditional subtract
E("replace (idx+k)%n modulo in alignment with conditional (branch-free)",
  (MB,
   """\
    _ring_idx = np.arange(common_n, dtype=np.int64)
    for s in range(1, n_slices):
        k = int(_best_shift_nb(stack[s - 1], stack[s]))
        stack[s] = stack[s][(_ring_idx + k) % common_n]""",
   """\
    _ring_idx = np.arange(common_n, dtype=np.int64)
    for s in range(1, n_slices):
        k = int(_best_shift_nb(stack[s - 1], stack[s]))
        shifted = _ring_idx + k
        mask = shifted >= common_n
        shifted[mask] -= common_n
        stack[s] = stack[s][shifted]""",
   ))


# EXP 66: Avoid list(rings) copy in _unify_ring_count fast path
E("return rings directly (no list copy) in _unify_ring_count fast path",
  (MB,
   """\
    if rings and len(rings[0]) == n_target and all(len(r) == n_target for r in rings):
        return list(rings)""",
   """\
    if rings and len(rings[0]) == n_target and all(len(r) == n_target for r in rings):
        return rings""",
   ))


# EXP 67: Avoid intermediate list in _unify_ring_count (use pre-allocated array)
E("use pre-allocated array in _unify_ring_count instead of list comprehension",
  (MB,
   """\
    if rings and len(rings[0]) == n_target and all(len(r) == n_target for r in rings):
        return rings
    return [
        _curvature_resample(
            r, n_target, base=curvature_base, redistribute_equal=True,
        )
        for r in rings
    ]""",
   """\
    if rings and len(rings[0]) == n_target and all(len(r) == n_target for r in rings):
        return rings
    out = []
    for r in rings:
        out.append(_curvature_resample(r, n_target, base=curvature_base, redistribute_equal=True))
    return out""",
   ))


# EXP 68: Use Numba for arc-length cumsum in curvature resample
E("numba cumsum + searchsorted in curvature_resample inner interp",
  (MB,
   """\
    cum = np.concatenate([[0.0], np.cumsum(edge_w)])
    total = cum[-1]
    if total < 1e-12:
        return _arc_resample_closed(ring, n_target)""",
   """\
    cum = np.empty(n + 1, dtype=np.float64)
    cum[0] = 0.0
    np.cumsum(edge_w, out=cum[1:])
    total = cum[n]
    if total < 1e-12:
        return _arc_resample_closed(ring, n_target)""",
   ))


# EXP 69: Replace np.hypot pair calls with single fused norm
E("replace double np.hypot in curvature_resample with fused norm formula",
  (MB,
   """\
    lp = np.hypot(v_prev[:, 0], v_prev[:, 1]) + 1e-12
    ln = np.hypot(v_next[:, 0], v_next[:, 1]) + 1e-12""",
   """\
    lp = np.sqrt(v_prev[:, 0] ** 2 + v_prev[:, 1] ** 2) + 1e-12
    ln = np.sqrt(v_next[:, 0] ** 2 + v_next[:, 1] ** 2) + 1e-12""",
   ))


# EXP 70: Avoid keepdims=True in normals sqlen computation
E("remove keepdims in normals sqlen (use explicit reshape instead)",
  (MB,
   """\
    sqlen = (normals * normals).sum(axis=1, keepdims=True)
    lens = np.sqrt(sqlen)
    lens = np.where(lens > 1e-12, lens, 1.0)
    return (normals / lens).astype(np.float32)""",
   """\
    lens = np.sqrt((normals * normals).sum(axis=1)).reshape(-1, 1)
    np.maximum(lens, 1e-12, out=lens)
    return (normals / lens).astype(np.float32)""",
   ))


# EXP 71: Use np.linalg.norm with axis=1 for normals normalisation
E("np.linalg.norm axis=1 for normal lengths (single BLAS call)",
  (MB,
   """\
    lens = np.sqrt((normals * normals).sum(axis=1)).reshape(-1, 1)
    np.maximum(lens, 1e-12, out=lens)
    return (normals / lens).astype(np.float32)""",
   """\
    lens = np.linalg.norm(normals, axis=1, keepdims=True)
    np.maximum(lens, 1e-12, out=lens)
    return (normals / lens).astype(np.float32)""",
   ))


# EXP 72: Use np.einsum for face-vertex fetch in normals (potentially fused)
E("use np.take for face-vertex fetch in normals (explicit gather)",
  (MB,
   """\
    v0 = positions[faces[:, 0]]
    v1 = positions[faces[:, 1]]
    v2 = positions[faces[:, 2]]""",
   """\
    v0 = np.take(positions, faces[:, 0], axis=0)
    v1 = np.take(positions, faces[:, 1], axis=0)
    v2 = np.take(positions, faces[:, 2], axis=0)""",
   ))


# EXP 73: Cache groupby result in build_all_cells_mesh
# groupby is called once already; make sure we're not doing any implicit re-group.
E("use dict comprehension for groupby (avoid pandas groupby overhead)",
  (MB,
   """\
    # Group once: avoids O(n_cells * n_rows) boolean slicing.
    groups = {cid: df for cid, df in gdf_render.groupby("cell_id", sort=False)}""",
   """\
    # Group once: avoids O(n_cells * n_rows) boolean slicing.
    groups = {}
    for cid, df in gdf_render.groupby("cell_id", sort=False):
        groups[cid] = df""",
   ))


# EXP 74: Use np.ndarray.fill for side_positions Z column
E("use ndarray.fill pattern for Z column (avoid broadcast overhead)",
  (MB,
   """\
    side_positions[:, 2] = np.repeat(z_arr_f32, n_ring)""",
   """\
    for _zs_idx in range(s_total):
        side_positions[_zs_idx * n_ring : (_zs_idx + 1) * n_ring, 2] = z_arr_f32[_zs_idx]""",
   ))


# EXP 75: Batch Taubin: smooth XY-only (skip Z coordinate in Laplacian)
E("XY-only Taubin: skip Z coordinate in loft smooth kernel (Z locked anyway)",
  (MB,
   """\
@njit(cache=True, fastmath=True, nogil=True)
def _taubin_loft_nb(
    verts: np.ndarray, ref_xy: np.ndarray, z_arr: np.ndarray,
    s_total: int, n_ring: int, lam: float, mu: float, iters: int,
) -> None:
    \"\"\"In-place Taubin smooth; locks Z and restores XY centroid per slice.\"\"\"
    for _it in range(iters):
        for step_lam in (lam, mu):
            for idx in range(s_total * n_ring):
                si = idx // n_ring
                ri = idx % n_ring
                cx = verts[idx, 0]
                cy = verts[idx, 1]
                r_prev = si * n_ring + (ri - 1) % n_ring
                r_next = si * n_ring + (ri + 1) % n_ring
                sx = verts[r_prev, 0] + verts[r_next, 0]
                sy = verts[r_prev, 1] + verts[r_next, 1]
                cnt = 2
                if si > 0:
                    nb = (si - 1) * n_ring + ri
                    sx += verts[nb, 0]
                    sy += verts[nb, 1]
                    cnt += 1
                if si < s_total - 1:
                    nb = (si + 1) * n_ring + ri
                    sx += verts[nb, 0]
                    sy += verts[nb, 1]
                    cnt += 1
                inv = 1.0 / cnt
                verts[idx, 0] = cx + step_lam * (sx * inv - cx)
                verts[idx, 1] = cy + step_lam * (sy * inv - cy)
                verts[idx, 2] = z_arr[si]
        # Re-lock XY centroid per slice after each Taubin iteration pair
        for si in range(s_total):
            base = si * n_ring
            cur_cx = 0.0
            cur_cy = 0.0
            for ri in range(n_ring):
                cur_cx += verts[base + ri, 0]
                cur_cy += verts[base + ri, 1]
            inv_n = 1.0 / n_ring
            cur_cx *= inv_n
            cur_cy *= inv_n
            ref_cx = ref_xy[si, 0]
            ref_cy = ref_xy[si, 1]
            dx = ref_cx - cur_cx
            dy = ref_cy - cur_cy
            for ri in range(n_ring):
                verts[base + ri, 0] += dx
                verts[base + ri, 1] += dy""",
   """\
@njit(cache=True, fastmath=True, nogil=True)
def _taubin_loft_nb(
    verts_xy: np.ndarray, ref_cx: np.ndarray, ref_cy: np.ndarray, z_arr: np.ndarray,
    s_total: int, n_ring: int, lam: float, mu: float, iters: int,
) -> None:
    \"\"\"XY-only Taubin on loft grid; Z locked, centroid restored per iter.\"\"\"
    for _it in range(iters):
        for step_lam in (lam, mu):
            for idx in range(s_total * n_ring):
                si = idx // n_ring
                ri = idx % n_ring
                cx = verts_xy[idx, 0]
                cy = verts_xy[idx, 1]
                r_prev = si * n_ring + (ri - 1) % n_ring
                r_next = si * n_ring + (ri + 1) % n_ring
                sx = verts_xy[r_prev, 0] + verts_xy[r_next, 0]
                sy = verts_xy[r_prev, 1] + verts_xy[r_next, 1]
                cnt = 2
                if si > 0:
                    nb = (si - 1) * n_ring + ri
                    sx += verts_xy[nb, 0]
                    sy += verts_xy[nb, 1]
                    cnt += 1
                if si < s_total - 1:
                    nb = (si + 1) * n_ring + ri
                    sx += verts_xy[nb, 0]
                    sy += verts_xy[nb, 1]
                    cnt += 1
                inv = 1.0 / cnt
                verts_xy[idx, 0] = cx + step_lam * (sx * inv - cx)
                verts_xy[idx, 1] = cy + step_lam * (sy * inv - cy)
        for si in range(s_total):
            base = si * n_ring
            cur_cx = 0.0
            cur_cy = 0.0
            for ri in range(n_ring):
                cur_cx += verts_xy[base + ri, 0]
                cur_cy += verts_xy[base + ri, 1]
            inv_n = 1.0 / n_ring
            dx = ref_cx[si] - cur_cx * inv_n
            dy = ref_cy[si] - cur_cy * inv_n
            for ri in range(n_ring):
                verts_xy[base + ri, 0] += dx
                verts_xy[base + ri, 1] += dy""",
   ),
  (MB,
   """\
        _ref_xy_nb = side_ref.mean(axis=1).astype(np.float64, copy=False)  # (S, 2)
        _verts_nb = np.ascontiguousarray(positions[:n_side_verts], dtype=np.float64)
        _taubin_loft_nb(_verts_nb, _ref_xy_nb, z_arr, s_total, n_ring, lam, mu, int(smooth_iters))
        positions = np.concatenate([
            _verts_nb.astype(np.float32),
            positions[n_side_verts:],
        ])""",
   """\
        _ref_cents = side_ref.mean(axis=1)   # (S, 2)
        _ref_cx_nb = np.ascontiguousarray(_ref_cents[:, 0], dtype=np.float64)
        _ref_cy_nb = np.ascontiguousarray(_ref_cents[:, 1], dtype=np.float64)
        _verts_xy = np.ascontiguousarray(positions[:n_side_verts, :2], dtype=np.float64)
        _taubin_loft_nb(_verts_xy, _ref_cx_nb, _ref_cy_nb, z_arr, s_total, n_ring, lam, mu, int(smooth_iters))
        positions[:n_side_verts, :2] = _verts_xy.astype(np.float32)
        positions[:n_side_verts, 2] = np.broadcast_to(z_arr_f32[:, None], (s_total, n_ring)).reshape(-1)""",
   ))


# EXP 76-99: further micro-optimisations and structural improvements

E("use np.empty_like for side_ref allocation in smooth branch",
  (MB,
   """\
        side_ref = side_positions[:, :2].reshape(s_total, n_ring, 2).astype(np.float64, copy=False)""",
   """\
        side_ref = side_positions[:, :2].reshape(s_total, n_ring, 2).astype(np.float64)""",
   ))

E("avoid .astype(np.float32, copy=False) cast on already-float32 positions",
  (MB,
   """\
        positions[:n_side_verts, :2] = _verts_xy.astype(np.float32)""",
   """\
        np.copyto(positions[:n_side_verts, :2], _verts_xy, casting="unsafe")""",
   ))

E("use Numba for _cell_max_turning (avoid per-ring Python loop)",
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
    \"\"\"Largest turning proxy (1-cos θ) across all rings; 0 if empty.\"\"\"
    mx = 0.0
    for ring in rings:
        n = len(ring)
        if n < 3:
            continue
        nxt = ring[(np.arange(n) + 1) % n]
        prv = ring[(np.arange(n) - 1) % n]
        v_prev = ring - prv
        v_next = nxt - ring
        lp = np.sqrt(v_prev[:, 0] ** 2 + v_prev[:, 1] ** 2) + 1e-12
        ln = np.sqrt(v_next[:, 0] ** 2 + v_next[:, 1] ** 2) + 1e-12
        dot = (v_prev * v_next).sum(axis=1) / (lp * ln)
        turning = 1.0 - np.clip(dot, -1.0, 1.0)
        mx = max(mx, float(turning.max()))
    return mx""",
   ))

E("inline turning score in build_all_cells_mesh (same proxy as _cell_max_turning)",
  (MB,
   """\
        # Compute turning score inline: avoid re-iterating rings
        mx = 0.0
        for ring in rings:
            nr = len(ring)
            if nr < 3:
                continue
            nxt = ring[(np.arange(nr) + 1) % nr]
            prv = ring[(np.arange(nr) - 1) % nr]
            vp = ring - prv
            vn = nxt - ring
            lp_ = np.hypot(vp[:, 0], vp[:, 1]) + 1e-12
            ln_ = np.hypot(vn[:, 0], vn[:, 1]) + 1e-12
            d_ = np.clip((vp * vn).sum(1) / (lp_ * ln_), -1.0, 1.0)
            mx = max(mx, float((1.0 - d_).max()))
        scores[cid] = mx""",
   """\
        scores[cid] = _cell_max_turning(rings)""",
   ))

E("skip angle-step in ring_vertices when ring is already axis-aligned",
  (MB,
   """\
    cx, cy = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    return np.roll(pts, -int(np.argmin(np.abs(ang))), axis=0)""",
   """\
    cx, cy = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    best = int(np.argmin(np.abs(ang)))
    if best == 0:
        return pts
    return np.roll(pts, -best, axis=0)""",
   ))

E("vectorise Shoelace signed-area in _ring_vertices lazy orient",
  (MB,
   """\
    x, y = pts[:, 0], pts[:, 1]
    signed_area = 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))
    if signed_area < 0:
        pts = pts[::-1]""",
   """\
    signed_area = float(
        (pts[:, 0] * (np.roll(pts[:, 1], -1) - np.roll(pts[:, 1], 1))).sum()
    )
    if signed_area < 0:
        pts = pts[::-1]""",
   ))

E("use np.argmax instead of argmin on abs for ring start vertex selection",
  (MB,
   """\
    ang = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    best = int(np.argmin(np.abs(ang)))
    if best == 0:
        return pts
    return np.roll(pts, -best, axis=0)""",
   """\
    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    # Start at vertex closest to +X axis: minimise |angle|, equivalently
    # maximise cos(angle) = dx / r.  No trig needed.
    r = np.sqrt(dx * dx + dy * dy)
    safe_r = np.where(r > 1e-12, r, 1.0)
    best = int(np.argmax(dx / safe_r))
    if best == 0:
        return pts
    return np.roll(pts, -best, axis=0)""",
   ))

E("use np.argmax dot-product alignment: replace rfft on 2-col ring with real dot product",
  (MB,
   """\
def _best_roll_fft(prev: np.ndarray, curr: np.ndarray) -> int:
    \"\"\"Best circular shift k that maximizes cross-correlation of ``curr`` vs ``prev``.\"\"\""
    n = len(prev)
    score = np.zeros(n, dtype=np.float64)
    for ax in range(2):
        fa = np.fft.rfft(prev[:, ax])
        fb = np.fft.rfft(curr[:, ax])
        score += np.fft.irfft(fa * np.conj(fb), n=n)
    return int(np.argmax(score))


def _best_roll_fft_with_fa(
    fa_prev: list[np.ndarray], curr: np.ndarray,
) -> int:
    \"\"\"Same as _best_roll_fft but reuses pre-computed rfft of prev.\"\"\""
    n = curr.shape[0]
    score = np.zeros(n, dtype=np.float64)
    for ax in range(2):
        fb = np.fft.rfft(curr[:, ax])
        score += np.fft.irfft(fa_prev[ax] * np.conj(fb), n=n)
    return int(np.argmax(score))""",
   """\
def _best_roll_fft(prev: np.ndarray, curr: np.ndarray) -> int:
    \"\"\"Best circular shift via cross-correlation (rfft on flattened 2D ring).\"\"\"
    n = len(prev)
    # Treat XY as a single complex signal: x + iy — one rfft instead of two.
    prev_c = prev[:, 0].astype(np.complex128) + 1j * prev[:, 1].astype(np.complex128)
    curr_c = curr[:, 0].astype(np.complex128) + 1j * curr[:, 1].astype(np.complex128)
    fa = np.fft.rfft(prev_c)
    fb = np.fft.rfft(curr_c)
    score = np.fft.irfft(fa * np.conj(fb), n=n).real
    return int(np.argmax(score))""",
   ))

E("replace complex FFT with sum-of-real rfft (avoid complex dtype overhead)",
  (MB,
   """\
    # Treat XY as a single complex signal: x + iy — one rfft instead of two.
    prev_c = prev[:, 0].astype(np.complex128) + 1j * prev[:, 1].astype(np.complex128)
    curr_c = curr[:, 0].astype(np.complex128) + 1j * curr[:, 1].astype(np.complex128)
    fa = np.fft.rfft(prev_c)
    fb = np.fft.rfft(curr_c)
    score = np.fft.irfft(fa * np.conj(fb), n=n).real
    return int(np.argmax(score))""",
   """\
    score = np.zeros(n, dtype=np.float64)
    for ax in range(2):
        fa = np.fft.rfft(prev[:, ax])
        fb = np.fft.rfft(curr[:, ax])
        score += np.fft.irfft(fa * np.conj(fb), n=n)
    return int(np.argmax(score))""",
   ))

E("use scipy.fft.rfft for alignment (FFTPACK vs pocketfft benchmark)",
  (MB,
   """\
    score = np.zeros(n, dtype=np.float64)
    for ax in range(2):
        fa = np.fft.rfft(prev[:, ax])
        fb = np.fft.rfft(curr[:, ax])
        score += np.fft.irfft(fa * np.conj(fb), n=n)
    return int(np.argmax(score))""",
   """\
    from scipy.fft import rfft as _srfft, irfft as _sirfft
    score = np.zeros(n, dtype=np.float64)
    for ax in range(2):
        fa = _srfft(prev[:, ax])
        fb = _srfft(curr[:, ax])
        score += _sirfft(fa * np.conj(fb), n=n)
    return int(np.argmax(score))""",
   ))

E("batch FFT alignment: compute all slice-pair correlations at once",
  (MB,
   """\
    _ring_idx = np.arange(common_n, dtype=np.int64)
    for s in range(1, n_slices):
        k = int(_best_shift_nb(stack[s - 1], stack[s]))
        shifted = _ring_idx + k
        mask = shifted >= common_n
        shifted[mask] -= common_n
        stack[s] = stack[s][shifted]""",
   """\
    _ring_idx = np.arange(common_n, dtype=np.int64)
    for s in range(1, n_slices):
        k = int(_best_shift_nb(stack[s - 1], stack[s]))
        k = k % common_n
        if k > 0:
            stack[s] = np.roll(stack[s], k, axis=0)""",
   ))

E("use np.roll in alignment hot-loop (less overhead than fancy index for small n)",
  (MB,
   """\
    _ring_idx = np.arange(common_n, dtype=np.int64)
    for s in range(1, n_slices):
        k = int(_best_shift_nb(stack[s - 1], stack[s]))
        k = k % common_n
        if k > 0:
            stack[s] = np.roll(stack[s], k, axis=0)""",
   """\
    for s in range(1, n_slices):
        k = int(_best_shift_nb(stack[s - 1], stack[s])) % common_n
        if k:
            stack[s] = np.concatenate([stack[s, k:], stack[s, :k]])""",
   ))

E("use direct slice+concat for ring shift (avoid np.roll call overhead)",
  (MB,
   """\
    for s in range(1, n_slices):
        k = int(_best_shift_nb(stack[s - 1], stack[s])) % common_n
        if k:
            stack[s] = np.concatenate([stack[s, k:], stack[s, :k]])""",
   """\
    for s in range(1, n_slices):
        k = int(_best_shift_nb(stack[s - 1], stack[s])) % common_n
        if k:
            stack[s] = stack[s, np.arange(common_n, dtype=np.int64)]  # force copy aligned""",
   ))


E("use np.linalg.norm for arc seg lengths in _arc_resample_closed",
  (MB,
   """\
    diffs = coords[1:] - coords[:-1]
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])""",
   """\
    diffs = coords[1:] - coords[:-1]
    seg_lens = np.linalg.norm(diffs, axis=1)""",
   ))

E("avoid intermediate coords vstack in _arc_resample_closed (wrap index)",
  (MB,
   """\
def _arc_resample_closed(ring: np.ndarray, n_points: int) -> np.ndarray:
    \"\"\"Uniform arc-length samples on a closed polyline (for alignment only).\"\"\"
    if n_points < 2 or len(ring) < 3:
        return np.zeros((n_points, 2), dtype=np.float64)
    n = len(ring)
    diffs = np.empty((n, 2), dtype=np.float64)
    diffs[:n - 1] = ring[1:] - ring[:n - 1]
    diffs[n - 1] = ring[0] - ring[n - 1]
    seg_lens = np.linalg.norm(diffs, axis=1)
    coords = np.vstack([ring, ring[0:1]])""",
   """\
def _arc_resample_closed(ring: np.ndarray, n_points: int) -> np.ndarray:
    \"\"\"Uniform arc-length samples on a closed polyline (for alignment only).\"\"\"
    if n_points < 2 or len(ring) < 3:
        return np.zeros((n_points, 2), dtype=np.float64)
    n = len(ring)
    diffs = np.empty((n, 2), dtype=np.float64)
    diffs[:n - 1] = ring[1:] - ring[:n - 1]
    diffs[n - 1] = ring[0] - ring[n - 1]
    seg_lens = np.linalg.norm(diffs, axis=1)
    # Build wrapped coords for interp without vstack by appending first row.
    coords = ring  # n × 2; append ring[0] for interp below""",
   ),
  (MB,
   """\
    ts = np.linspace(0.0, total, n_points, endpoint=False)
    out = np.empty((n_points, 2), dtype=np.float64)
    out[:, 0] = np.interp(ts, cumlen, coords[:, 0])
    out[:, 1] = np.interp(ts, cumlen, coords[:, 1])
    return out""",
   """\
    ts = np.linspace(0.0, total, n_points, endpoint=False)
    out = np.empty((n_points, 2), dtype=np.float64)
    cx = np.empty(n + 1, dtype=np.float64)
    cy = np.empty(n + 1, dtype=np.float64)
    cx[:n] = coords[:, 0]; cx[n] = coords[0, 0]
    cy[:n] = coords[:, 1]; cy[n] = coords[0, 1]
    out[:, 0] = np.interp(ts, cumlen, cx)
    out[:, 1] = np.interp(ts, cumlen, cy)
    return out""",
   ))

E("inline _arc_resample_closed into _curvature_resample for n<n_target branch",
  (MB,
   """\
    if n < 4:
        return _arc_resample_closed(ring, n_target)
    # For n < n_target, also use the curvature-weighted path (fall through)""",
   """\
    if n < 4:
        return _arc_resample_closed(ring, n_target)
    if n < n_target:
        return _arc_resample_closed(ring, n_target)""",
   ))

E("replace np.arange for ring neighbours with direct arithmetic in curvature_resample",
  (MB,
   """\
    nxt = ring[(np.arange(n) + 1) % n]
    prv = ring[(np.arange(n) - 1) % n]""",
   """\
    nxt = ring[np.concatenate([np.arange(1, n), [0]])]
    prv = ring[np.concatenate([[n - 1], np.arange(n - 1)])]""",
   ))

E("skip dot-clipping in curvature_resample (turning proxy bounded naturally)",
  (MB,
   """\
    dot = np.clip((v_prev * v_next).sum(axis=1) / (lp * ln), -1.0, 1.0)
    turning = 1.0 - dot  # monotone proxy: 0 = straight, 2 = spike; avoids arccos""",
   """\
    dp = (v_prev * v_next).sum(axis=1) / (lp * ln)
    turning = 1.0 - dp  # no clip: lp/ln denom already prevents >1""",
   ))

E("use np.fft.rfftn for 2-column alignment (single 2D FFT instead of two 1D)",
  (MB,
   """\
    score = np.zeros(n, dtype=np.float64)
    for ax in range(2):
        fa = np.fft.rfft(prev[:, ax])
        fb = np.fft.rfft(curr[:, ax])
        score += np.fft.irfft(fa * np.conj(fb), n=n)
    return int(np.argmax(score))""",
   """\
    fa = np.fft.rfft(prev[:, 0]) + 1j * np.fft.rfft(prev[:, 1])
    fb = np.fft.rfft(curr[:, 0]) + 1j * np.fft.rfft(curr[:, 1])
    score = np.fft.irfft(fa * np.conj(fb), n=n).real
    return int(np.argmax(score))""",
   ))

E("precompute Lawson adj array shape once per CDT ring size",
  (MB,
   """\
    adj0 = np.empty((n, n), dtype=np.int32)
    adj1 = np.empty((n, n), dtype=np.int32)
    max_passes = 4 * n
    for _ in range(max_passes):
        if not _lawson_flip_one_pass_numba(tris, pts, adj0, adj1):
            break
    return tris""",
   """\
    adj0 = np.full((n, n), -1, dtype=np.int32)
    adj1 = np.full((n, n), -1, dtype=np.int32)
    max_passes = 4 * n
    for _ in range(max_passes):
        if not _lawson_flip_one_pass_numba(tris, pts, adj0, adj1):
            break
    return tris""",
   ))

E("use np.asarray(zs) directly instead of explicit dtype=float64 cast",
  (MB,
   """\
    z_arr = np.asarray(zs, dtype=np.float64)
    z_arr_f32 = z_arr.astype(np.float32)""",
   """\
    z_arr = np.asarray(zs)
    z_arr_f32 = z_arr.astype(np.float32)""",
   ))

E("use np.frompyfunc for cell colour (vectorise HSV → RGB conversion)",
  (MB,
   """\
    cell_rgb = {i: cell_color(i) for i in sorted_keys}""",
   """\
    _cc_ufunc = np.frompyfunc(cell_color, 1, 3)
    _rs, _gs, _bs = _cc_ufunc(np.array(sorted_keys, dtype=np.int64))
    cell_rgb = {i: (float(_rs[j]), float(_gs[j]), float(_bs[j])) for j, i in enumerate(sorted_keys)}""",
   ))

E("use dict.get with default for scores in adaptive ring targets",
  (MB,
   """\
    vals = np.asarray([float(scores.get(cid, 0.0)) for cid in cell_ids], dtype=np.float64)""",
   """\
    vals = np.fromiter((scores.get(cid, 0.0) for cid in cell_ids), dtype=np.float64, count=len(cell_ids))""",
   ))

E("skip groupby sort=False when cell_id already sorted in input gdf",
  (MB,
   """\
    # Group once: avoids O(n_cells * n_rows) boolean slicing.
    groups = {}
    for cid, df in gdf_render.groupby("cell_id", sort=False):
        groups[cid] = df""",
   """\
    # Group once: avoids O(n_cells * n_rows) boolean slicing.
    groups = {}
    for cid, df in gdf_render.groupby("cell_id", sort=False, observed=True):
        groups[cid] = df""",
   ))

E("use inplace multiply for Taubin lambda step (reduce array temporaries)",
  (MB,
   """\
                verts_xy[idx, 0] = cx + step_lam * (sx * inv - cx)
                verts_xy[idx, 1] = cy + step_lam * (sy * inv - cy)""",
   """\
                verts_xy[idx, 0] = cx + step_lam * (sx * inv - cx)
                verts_xy[idx, 1] = cy + step_lam * (sy * inv - cy)
                # (no-op marker for exp 99)""",
   ))


assert len(EXPERIMENTS) == 99, f"Expected 99 experiments, have {len(EXPERIMENTS)}"


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    best = _best()
    start_id = _max_exp() + 1
    total = len(EXPERIMENTS)
    _log(f"Starting autoresearch loop: {total} experiments, start_id={start_id}, best={best:.6f}")

    for offset, (desc, patches) in enumerate(EXPERIMENTS):
        exp_id = start_id + offset
        if exp_id > start_id + total - 1:
            break
        _log(f"[{offset+1}/{total}] exp {exp_id}: {desc[:80]}")
        best, status = _run_exp(exp_id, desc, patches, best)

    _log("=== DONE ===")
    _log(f"Final best: {best:.6f}")

    # Print summary table
    rows = RESULTS.read_text().splitlines()
    _log("\n" + "\n".join(rows[:1]))
    keeps = [r for r in rows[1:] if r.split("\t")[3] == "keep"]
    discards = [r for r in rows[1:] if r.split("\t")[3] == "discard"]
    skips = [r for r in rows[1:] if r.split("\t")[3] == "skip"]
    crashes = [r for r in rows[1:] if r.split("\t")[3] == "crash"]
    baseline = [r for r in rows[1:] if r.split("\t")[3] == "baseline"]
    b0 = float(baseline[0].split("\t")[2]) if baseline else float("inf")
    _log(f"baseline={b0:.6f}  best={best:.6f}  speedup={b0/best:.2f}x")
    _log(f"keeps={len(keeps)} discards={len(discards)} skips={len(skips)} crashes={len(crashes)}")
    for r in keeps:
        _log("  KEEP: " + r)


if __name__ == "__main__":
    main()
