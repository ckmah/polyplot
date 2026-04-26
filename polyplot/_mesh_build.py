from __future__ import annotations

import math
import os

import numpy as np
from numba import njit
import trimesh


def _largest_polygon(geom):
    from shapely.geometry import MultiPolygon, Polygon
    if isinstance(geom, Polygon):
        return geom if not geom.is_empty else None
    if isinstance(geom, MultiPolygon):
        polys = [g for g in geom.geoms if not g.is_empty]
        if not polys:
            return None
        return max(polys, key=lambda p: p.area)
    return None


def _ring_vertices(polygon) -> np.ndarray:
    from shapely.geometry.polygon import orient
    ring = orient(polygon, sign=1.0).exterior
    coords = np.asarray(ring.coords, dtype=np.float64)
    if len(coords) < 4:
        return np.zeros((0, 2), dtype=np.float64)

    pts = coords[:-1, :2]
    if len(pts) < 3:
        return np.zeros((0, 2), dtype=np.float64)

    cx, cy = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    return np.roll(pts, -int(np.argmin(np.abs(ang))), axis=0)


def _curvature_resample(
    ring: np.ndarray,
    n_target: int,
    base: float = 0.35,
    *,
    redistribute_equal: bool = False,
) -> np.ndarray:
    """Resample a closed XY ring to ``n_target`` vertices, density-weighted by curvature.

    Samples uniformly in cumulative edge weight
    ``length * (average turning angle at endpoints + base * pi)``.
    Lower ``base`` allocates more samples to high-curvature regions.

    If ``n < n_target``, upsamples with uniform arc length. If ``n == n_target`` and
    ``redistribute_equal`` is False, returns ``ring`` unchanged. If True, recomputes
    ``n_target`` samples with the same curvature weighting (useful after unify).
    """
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
        return ring

    nxt = np.roll(ring, -1, axis=0)
    prv = np.roll(ring, 1, axis=0)
    v_prev = ring - prv
    v_next = nxt - ring
    lp = np.hypot(v_prev[:, 0], v_prev[:, 1]) + 1e-12
    ln = np.hypot(v_next[:, 0], v_next[:, 1]) + 1e-12
    dot = np.clip((v_prev * v_next).sum(axis=1) / (lp * ln), -1.0, 1.0)
    turning = np.arccos(dot)  # 0 = straight, pi = spike

    # Edge weight: average of turning at its two endpoints, times a density factor,
    # times edge length. Higher curvature => tighter sampling.
    endpoint_density = turning + base * np.pi
    edge_density = 0.5 * (endpoint_density + np.roll(endpoint_density, -1, axis=0))
    edge_w = ln * edge_density
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
    return a + frac * (b - a)


def _arc_resample_closed(ring: np.ndarray, n_points: int) -> np.ndarray:
    """Uniform arc-length samples on a closed polyline (for alignment only)."""
    if n_points < 2 or len(ring) < 3:
        return np.zeros((n_points, 2), dtype=np.float64)
    coords = np.vstack([ring, ring[0:1]])
    diffs = np.diff(coords, axis=0)
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    cumlen = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = cumlen[-1]
    if total < 1e-15:
        return np.zeros((n_points, 2), dtype=np.float64)
    ts = np.linspace(0.0, total, n_points, endpoint=False)
    return np.stack([
        np.interp(ts, cumlen, coords[:, 0]),
        np.interp(ts, cumlen, coords[:, 1]),
    ], axis=1)


def _best_roll_fft(prev: np.ndarray, curr: np.ndarray) -> int:
    """Best circular shift k that maximizes cross-correlation of ``curr`` vs ``prev``.

    Both arrays must have the same length. Uses rFFT for O(N log N).
    """
    n = len(prev)
    score = np.zeros(n, dtype=np.float64)
    for ax in range(2):
        fa = np.fft.rfft(prev[:, ax])
        fb = np.fft.rfft(curr[:, ax])
        score += np.fft.irfft(fa * np.conj(fb), n=n)
    return int(np.argmax(score))


def _align_to(prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
    """Rotation-align ``curr`` to ``prev`` (closed rings, possibly different N)."""
    na, nb = len(prev), len(curr)
    if nb == 0:
        return curr
    if na == nb:
        return np.roll(curr, -_best_roll_fft(prev, curr), axis=0)

    # Unequal N: compare on a common-N arc resampling, then map back to curr's
    # original grid. FFT runs on the common grid, result is scaled by nb/n_cmp.
    n_cmp = max(na, nb)
    prev_u = _arc_resample_closed(prev, n_cmp)
    curr_u = _arc_resample_closed(curr, n_cmp)
    k_cmp = _best_roll_fft(prev_u, curr_u)
    k = int(round(k_cmp * nb / n_cmp)) % nb
    return np.roll(curr, -k, axis=0)


def _align_ring_min_sqdist(prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
    """Rotate ``curr`` (closed XY ring) to minimize sum of squared edge lengths to ``prev``.

    FFT correlation can lock onto a wrong period on elongated or nearly symmetric
    slices, which shows up as twisted side strips. Exhaustive search over shifts
    is O(N^2) and robust for typical ring sizes after resampling.
    """
    n = len(prev)
    if n == 0 or len(curr) != n:
        return _align_to(prev, curr)
    if n > 512:
        return np.roll(curr, -_best_roll_fft(prev, curr), axis=0)
    # For moderate N, do a robust but sub-quadratic search:
    # - FFT gives the best correlation shift in O(N log N)
    # - refine by evaluating a small local window in exact sqdist space
    # Keep exhaustive for very small rings where overhead dominates and
    # symmetry-induced local minima are common.
    if n <= 128:
        best_k = 0
        best_cost = np.inf
        for k in range(n):
            rolled = np.roll(curr, -k, axis=0)
            cost = float(np.sum((rolled - prev) ** 2))
            if cost < best_cost:
                best_cost = cost
                best_k = k
        return np.roll(curr, -best_k, axis=0)

    k0 = int(_best_roll_fft(prev, curr))
    win = 16
    best_k = k0
    best_cost = np.inf
    for dk in range(-win, win + 1):
        k = (k0 + dk) % n
        rolled = np.roll(curr, -k, axis=0)
        cost = float(np.sum((rolled - prev) ** 2))
        if cost < best_cost:
            best_cost = cost
            best_k = k
    return np.roll(curr, -best_k, axis=0)


# ---------------------------------------------------------------------------
# Correspondence strip DP (numba-jitted inner loop)
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True, nogil=True)
def _dp_correspondence_fill(A: np.ndarray, B: np.ndarray):
    """Fill DP tables for min-area unfolding between closed rings A, B.

    Returns (dp, back) as int8/float64 ndarrays. back codes:
        1 = came from (i-1, j), 2 = came from (i, j-1).
    """
    na = A.shape[0]
    nb = B.shape[0]
    inf = 1e100
    dp = np.full((na + 1, nb + 1), inf)
    back = np.zeros((na + 1, nb + 1), dtype=np.int8)
    dp[0, 0] = 0.0

    for i in range(na + 1):
        for j in range(nb + 1):
            c_here = dp[i, j]
            if c_here >= inf:
                continue
            ia = i % na
            ib = j % nb

            if i < na:
                ia2 = (i + 1) % na
                # triangle (a0, b0, a1) — 2x area via cross product
                ax = A[ia2, 0] - A[ia, 0]
                ay = A[ia2, 1] - A[ia, 1]
                bx = B[ib, 0] - A[ia, 0]
                by = B[ib, 1] - A[ia, 1]
                tc = c_here + 0.5 * abs(ax * by - ay * bx)
                if tc < dp[i + 1, j]:
                    dp[i + 1, j] = tc
                    back[i + 1, j] = 1
            if j < nb:
                ib2 = (j + 1) % nb
                ax = B[ib2, 0] - A[ia, 0]
                ay = B[ib2, 1] - A[ia, 1]
                bx = B[ib, 0] - A[ia, 0]
                by = B[ib, 1] - A[ia, 1]
                tc = c_here + 0.5 * abs(ax * by - ay * bx)
                if tc < dp[i, j + 1]:
                    dp[i, j + 1] = tc
                    back[i, j + 1] = 2
    return dp, back


@njit(cache=True, nogil=True)
def _dp_correspondence_traceback(back: np.ndarray, na: int, nb: int,
                                 base_a: int, base_b: int) -> np.ndarray:
    """Walk back pointers from (na, nb) to (0, 0), emit flat triangle indices."""
    tris = np.empty((na + nb) * 3, dtype=np.int64)
    w = tris.shape[0]
    i, j = na, nb
    while i > 0 or j > 0:
        b = back[i, j]
        w -= 3
        if b == 1:
            tris[w + 0] = base_a + (i - 1) % na
            tris[w + 1] = base_b + j % nb
            tris[w + 2] = base_a + i % na
            i -= 1
        elif b == 2:
            tris[w + 0] = base_a + i % na
            tris[w + 1] = base_b + (j - 1) % nb
            tris[w + 2] = base_b + j % nb
            j -= 1
        else:
            # broken path — shouldn't happen for non-degenerate rings
            return tris[w + 3:]
    return tris


def _correspondence_strip_triangles(
    A: np.ndarray,
    B: np.ndarray,
    base_a: int,
    base_b: int,
) -> np.ndarray:
    """Triangulate the side surface between two closed XY rings.

    Returns a flat int64 array of vertex indices (length ``(na + nb) * 3``).
    Returns an empty array if either ring has fewer than 3 points.
    """
    na, nb = len(A), len(B)
    if na < 3 or nb < 3:
        return np.empty(0, dtype=np.int64)
    Ac = np.ascontiguousarray(A, dtype=np.float64)
    Bc = np.ascontiguousarray(B, dtype=np.float64)
    _, back = _dp_correspondence_fill(Ac, Bc)
    return _dp_correspondence_traceback(back, na, nb, base_a, base_b)


# ---------------------------------------------------------------------------
# Smoothing (Taubin): volume-preserving variant of Laplacian
# ---------------------------------------------------------------------------

# Taubin parameters: one positive (shrinking) step with lambda, one negative
# (expanding) step with mu. Choose mu so that 1/lambda - 1/mu = kpb (pass band);
# this keeps low-frequency shape while smoothing high-frequency noise.
_TAUBIN_LAMBDA = 0.5
_TAUBIN_MU = -0.53


def _laplacian_step(pts: np.ndarray, lam: float) -> np.ndarray:
    fwd = np.roll(pts, -1, axis=0)
    bck = np.roll(pts, 1, axis=0)
    return pts + lam * (0.5 * (fwd + bck) - pts)


def _smooth_ring(ring: np.ndarray, iters: int, factor: float = 0.3) -> np.ndarray:
    """Taubin smooth a single XY ring. ``factor`` controls lambda."""
    if iters <= 0:
        return ring
    pts = ring.astype(np.float64, copy=True)
    lam = float(np.clip(factor + 0.2, 0.2, 0.85))
    mu = lam / (lam * 0.1 - 1.0)  # keeps (1/lam - 1/mu) small, preserves volume
    for _ in range(iters):
        pts = _laplacian_step(pts, lam)
        pts = _laplacian_step(pts, mu)
    return pts


def _taubin_lam_mu(factor: float) -> tuple[float, float]:
    """Taubin (lambda, mu) pair with a stable pass-band frequency kPB=0.1.

    ``factor`` in [0, 1] sets lambda in [0.2, 0.65]; mu is derived from
    ``1/mu + 1/lam = kPB`` so each pass has unit low-frequency gain.
    """
    lam = float(np.clip(0.2 + 0.45 * factor, 0.2, 0.65))
    kPB = 0.1
    mu = 1.0 / (kPB - 1.0 / lam)
    return lam, mu


### NOTE
# The earlier implementation included:
# - per-column Z smoothing on the (S, N, 2) ring stack, and
# - Catmull-Rom Z subdivision (inserting extra rings).
# This was removed in favor of a single 3D Taubin smoothing pass applied to the
# full triangle mesh via trimesh (simpler pipeline, less custom code).


def _collect_rings_for_cell(gdf_cell, z_scale: float) -> tuple[list[np.ndarray], list[float]]:
    """Extract ordered 2D rings and Z values for one cell (same rules as ``build_loft_mesh``)."""
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
    return rings_2d, zs


def _cell_max_turning(rings: list[np.ndarray]) -> float:
    """Largest vertex turning angle (radians) across all rings; 0 if empty."""
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
    return mx


def _adaptive_ring_targets_from_scores(
    cell_ids: list,
    scores: dict,
    ring_target: int | None,
    *,
    adaptive: bool = True,
    min_mul: float = 0.55,
    max_mul: float = 2.25,
    exponent: float = 0.75,
) -> dict | None:
    """Like ``_adaptive_ring_targets_for_cells`` but uses precomputed scores."""
    if ring_target is None:
        return None
    base = max(4, int(ring_target))
    if not adaptive or not cell_ids:
        return {cid: base for cid in cell_ids}
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
    return out


def _adaptive_ring_targets_for_cells(
    cell_ids: list,
    cell_gdfs: dict,
    z_scale: float,
    ring_target: int | None,
    *,
    adaptive: bool = True,
    min_mul: float = 0.55,
    max_mul: float = 2.25,
    exponent: float = 0.75,
) -> dict | None:
    """Map ``cell_id`` to ring vertex count. Returns ``None`` if ``ring_target`` is None.

    When ``adaptive`` is True, scales each cell's target vs the median max-turning
    score in ``cell_ids`` so spikier boundaries get more vertices (clamped).
    """
    if ring_target is None:
        return None
    base = max(4, int(ring_target))
    if not adaptive or not cell_ids:
        return {cid: base for cid in cell_ids}
    scores: dict = {}
    for cid in cell_ids:
        rings, _ = _collect_rings_for_cell(cell_gdfs[cid], z_scale)
        scores[cid] = _cell_max_turning(rings) if rings else 0.0
    vals = np.asarray(list(scores.values()), dtype=np.float64)
    ref = float(np.median(vals)) if vals.size else 0.0
    if ref < 1e-12:
        return {cid: base for cid in cell_ids}
    lo = max(4, int(round(base * min_mul)))
    hi = max(lo, int(round(base * max_mul)))
    out: dict = {}
    for cid in cell_ids:
        s = scores[cid]
        ratio = (s / ref) ** float(exponent)
        n = int(round(base * ratio))
        out[cid] = int(np.clip(n, lo, hi))
    return out


def _unify_ring_count(
    rings: list[np.ndarray],
    n_target: int,
    *,
    curvature_base: float = 0.28,
) -> list[np.ndarray]:
    """Resample every ring to exactly ``n_target`` vertices (curvature-weighted)."""
    return [
        _curvature_resample(
            r, n_target, base=curvature_base, redistribute_equal=True,
        )
        for r in rings
    ]


def cell_color(i: int) -> tuple[float, float, float]:
    """Golden-ratio HSV color for cell index i — (r, g, b) in [0, 1]."""
    h = (i * 0.618033988749895) % 1.0
    s, v = 0.72, 0.88
    h6 = h * 6
    idx = int(h6)
    f = h6 - idx
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    return [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][idx % 6]


# ---------------------------------------------------------------------------
# Cap triangulation (Delaunay + polygon filter) and vertex normals
# ---------------------------------------------------------------------------

def _cap_tris_earcut(ring_xy: np.ndarray) -> np.ndarray:
    """Ear-clipping triangulation of a simple ring. Returns (M, 3) local indices."""
    import mapbox_earcut as earcut

    v2d = np.ascontiguousarray(ring_xy, dtype=np.float32)
    rings = np.array([len(ring_xy)], dtype=np.uint32)
    tris = earcut.triangulate_float32(v2d, rings)
    return np.asarray(tris, dtype=np.int64).reshape(-1, 3)


def _orient2d_ccw(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
    """Twice signed area / CCW turn test: >0 iff ``q`` is left of directed ``pr``."""
    return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])


def _in_circumcircle(a: np.ndarray, b: np.ndarray, c: np.ndarray,
                     d: np.ndarray) -> bool:
    """Strict Delaunay in-circle test for a CCW-oriented triangle (a, b, c).

    Returns True iff point ``d`` lies strictly inside the circumcircle of
    ``a, b, c``. Uses the standard 3x3 determinant.
    """
    ax, ay = a[0] - d[0], a[1] - d[1]
    bx, by = b[0] - d[0], b[1] - d[1]
    cx, cy = c[0] - d[0], c[1] - d[1]
    det = (
        (ax * ax + ay * ay) * (bx * cy - by * cx)
        - (bx * bx + by * by) * (ax * cy - ay * cx)
        + (cx * cx + cy * cy) * (ax * by - ay * bx)
    )
    return det > 1e-12


def _cap_tris_cdt(ring_xy: np.ndarray) -> np.ndarray:
    """Constrained Delaunay-like triangulation of a simple ring.

    Earcut gives a topologically valid triangulation (exactly ``n - 2``
    triangles, no overlaps, respects boundary) but shaped as a radial fan which
    shows up as a visible "star" on flat caps under PBR. We refine it with
    Lawson edge flips until the triangulation is locally Delaunay — same
    triangle count, same boundary, much better-shaped triangles.

    Plain Delaunay on the ring vertices is tempting but unusable here: it
    triangulates the convex hull, and a centroid-in-polygon filter leaks
    overlapping triangles across shallow concavities.
    """
    from collections import defaultdict

    n = len(ring_xy)
    if n < 3:
        return np.zeros((0, 3), dtype=np.int64)

    tris = _cap_tris_earcut(ring_xy)
    if tris.shape[0] == 0:
        return tris
    pts = np.ascontiguousarray(ring_xy, dtype=np.float64)

    # Normalize winding to CCW so the in-circle test's orientation is consistent.
    v0, v1, v2 = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
    cross_z = (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) - \
              (v1[:, 1] - v0[:, 1]) * (v2[:, 0] - v0[:, 0])
    flip = cross_z < 0
    tris[flip] = tris[flip][:, [0, 2, 1]]

    # Lawson flipping: up to O(n^2) flips in worst case; tiny n (<=128).
    max_passes = 4 * n
    for _ in range(max_passes):
        edge_to_tri: dict[tuple[int, int], list[int]] = defaultdict(list)
        for ti, t in enumerate(tris):
            for a, b in ((int(t[0]), int(t[1])),
                         (int(t[1]), int(t[2])),
                         (int(t[2]), int(t[0]))):
                key = (a, b) if a < b else (b, a)
                edge_to_tri[key].append(ti)

        did_flip = False
        for (a, b), adj in edge_to_tri.items():
            if len(adj) != 2:
                continue
            t0i, t1i = adj
            t0 = tris[t0i].tolist()
            t1 = tris[t1i].tolist()
            c = next(x for x in t0 if x != a and x != b)
            d = next(x for x in t1 if x != a and x != b)

            # Skip if the quadrilateral (a, c, b, d) is non-convex — flipping
            # would produce a triangle outside the current coverage.
            pa, pb, pc, pd = pts[a], pts[b], pts[c], pts[d]
            if _orient2d_ccw(pa, pc, pb) * _orient2d_ccw(pa, pd, pb) >= 0:
                continue
            if _orient2d_ccw(pc, pa, pd) * _orient2d_ccw(pc, pb, pd) >= 0:
                continue

            # Delaunay condition: d must lie outside circumcircle of t0.
            # Re-order t0 to CCW(a, b, c) for the in-circle test.
            if _orient2d_ccw(pa, pb, pc) > 0:
                abc = (pa, pb, pc)
            else:
                abc = (pa, pc, pb)
            if not _in_circumcircle(abc[0], abc[1], abc[2], pd):
                continue

            # Flip diagonal (a, b) -> (c, d). Preserve CCW winding.
            new0 = [a, c, d] if _orient2d_ccw(pa, pc, pd) > 0 else [a, d, c]
            new1 = [b, d, c] if _orient2d_ccw(pb, pd, pc) > 0 else [b, c, d]
            tris[t0i] = new0
            tris[t1i] = new1
            did_flip = True
            break
        if not did_flip:
            break
    return tris


def _orient_cap_tris(tris: np.ndarray, ring_xy: np.ndarray,
                     want_positive_z: bool) -> np.ndarray:
    """Flip any CW/CCW mismatch so every triangle has the requested Z-normal sign."""
    if tris.shape[0] == 0:
        return tris
    v0 = ring_xy[tris[:, 0]]
    v1 = ring_xy[tris[:, 1]]
    v2 = ring_xy[tris[:, 2]]
    cross_z = (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) - \
              (v1[:, 1] - v0[:, 1]) * (v2[:, 0] - v0[:, 0])
    flip = cross_z < 0 if want_positive_z else cross_z > 0
    out = tris.copy()
    out[flip] = out[flip][:, [0, 2, 1]]
    return out


def _compute_vertex_normals(positions: np.ndarray,
                            indices_flat: np.ndarray) -> np.ndarray:
    """Area-weighted per-vertex normals. Writes explicit normals into the GLB.

    Storing these as a NORMAL attribute lets :js:func:`computeVertexNormals` on
    the client be skipped, so meshopt position quantization (``gltfpack -cc``)
    no longer produces shading streaks on flat caps.
    """
    n_verts = int(positions.shape[0])
    if n_verts == 0 or indices_flat.shape[0] == 0:
        return np.zeros((n_verts, 3), dtype=np.float32)
    faces = indices_flat.reshape(-1, 3).astype(np.int64, copy=False)
    pos64 = positions.astype(np.float64, copy=False)
    v0 = pos64[faces[:, 0]]
    v1 = pos64[faces[:, 1]]
    v2 = pos64[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)  # unnormalized => magnitude ~ 2 * area
    normals = np.zeros((n_verts, 3), dtype=np.float64)
    np.add.at(normals, faces[:, 0], fn)
    np.add.at(normals, faces[:, 1], fn)
    np.add.at(normals, faces[:, 2], fn)
    lens = np.linalg.norm(normals, axis=1, keepdims=True)
    lens = np.where(lens > 1e-12, lens, 1.0)
    return (normals / lens).astype(np.float32)


# ---------------------------------------------------------------------------
# Per-cell loft
# ---------------------------------------------------------------------------

def _quad_strip_triangles(na: int, base_a: int, base_b: int) -> np.ndarray:
    """Triangulate a shared-N quad strip between two aligned rings (2*na tris).

    Assumes ring vertices are CCW in XY when viewed from +Z (guaranteed by
    :func:`_ring_vertices`) and ``base_b`` is the ring one Z step above
    ``base_a``. Triangle winding is chosen so face normals point OUTWARD from
    the tube axis — required for :js:attr:`THREE.FrontSide` culling to render
    the outside of each cell instead of the inside.
    """
    i = np.arange(na, dtype=np.int64)
    j = (i + 1) % na
    tris = np.empty((na, 2, 3), dtype=np.int64)
    tris[:, 0, 0] = base_a + i
    tris[:, 0, 1] = base_a + j
    tris[:, 0, 2] = base_b + j
    tris[:, 1, 0] = base_a + i
    tris[:, 1, 1] = base_b + j
    tris[:, 1, 2] = base_b + i
    return tris.reshape(-1)


def build_loft_mesh(
    gdf_cell,
    z_scale: float = 2.0,
    smooth_iters: int = 1,
    smooth_factor: float = 0.5,
    ring_vertex_target: int | None = 48,
    *,
    ring_curvature_base: float = 0.28,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, ...]]:
    """Loft a watertight surface through one cell's cross-section stack.

    Returns ``(positions_f32 (Nv, 3), indices_i64 (Nt*3,), normals_f32 (Nv, 3),
    bbox)``. Empty arrays + zero bbox when the cell has fewer than 2 valid
    slices.

    The pipeline unifies every ring to a common vertex count using
    curvature-weighted resampling (``ring_curvature_base``: higher => closer to
    uniform arc length), rotation-aligns each ring to its predecessor by minimum
    vertex displacement, builds the mesh, and optionally applies **3D Taubin
    smoothing** (via trimesh). Shared topology means sides are fast quad strips.
    Cap triangulation uses a Delaunay + interior-centroid filter (not earcut)
    so flat caps do not display a radial fan of sliver triangles under PBR.

    After Taubin, the first/last side rings are **duplicated** in the position
    buffer (same XYZ) and cap triangles are re-indexed to the copy so cap
    vertex normals are pure +Z/-Z and never averaged with the tube's side
    normals (which would show as diagonal specular streaks under PBR). Explicit
    per-vertex normals are returned so the GLB loader can skip
    ``computeVertexNormals()`` (meshopt-quantized positions otherwise produce
    subtle facet shading even on a planar cap).

    ``ring_vertex_target``: common ring vertex count. If ``None``, uses the
    max input ring count.
    """
    rings_2d, zs = _collect_rings_for_cell(gdf_cell, z_scale)
    return build_loft_mesh_from_rings(
        rings_2d,
        zs,
        smooth_iters=smooth_iters,
        smooth_factor=smooth_factor,
        ring_vertex_target=ring_vertex_target,
        ring_curvature_base=ring_curvature_base,
    )


def build_loft_mesh_from_rings(
    rings_2d: list[np.ndarray],
    zs: list[float],
    *,
    smooth_iters: int = 1,
    smooth_factor: float = 0.5,
    ring_vertex_target: int | None = 48,
    ring_curvature_base: float = 0.28,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, ...]]:
    """Like ``build_loft_mesh`` but consumes pre-extracted ``rings_2d`` + ``zs``."""

    n_slices = len(rings_2d)
    empty_pos = np.zeros((0, 3), dtype=np.float32)
    empty_idx = np.zeros(0, dtype=np.int64)
    empty_nrm = np.zeros((0, 3), dtype=np.float32)
    zero_bbox = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if n_slices < 2:
        return empty_pos, empty_idx, empty_nrm, zero_bbox

    if ring_vertex_target is not None:
        common_n = int(ring_vertex_target)
    else:
        common_n = int(max(len(r) for r in rings_2d))
    common_n = max(common_n, 4)

    rings_2d = _unify_ring_count(
        rings_2d, common_n, curvature_base=ring_curvature_base,
    )

    # Align rotation of each ring to its predecessor (min displacement; robust vs FFT).
    for s in range(1, n_slices):
        rings_2d[s] = _align_ring_min_sqdist(rings_2d[s - 1], rings_2d[s])
 
    stack = np.stack(rings_2d, axis=0)  # (S, N, 2)
    z_arr = np.asarray(zs, dtype=np.float64)

    s_total, n_ring = stack.shape[0], stack.shape[1]
    n_side_verts = s_total * n_ring

    side_positions = np.empty((n_side_verts, 3), dtype=np.float32)
    side_positions[:, 0:2] = stack.reshape(-1, 2)
    side_positions[:, 2] = np.repeat(z_arr.astype(np.float32), n_ring)

    strips = [_quad_strip_triangles(n_ring, s * n_ring, (s + 1) * n_ring)
              for s in range(s_total - 1)]

    positions = side_positions

    def _cap_flat(ring_xy, base_idx, want_positive_z):
        local = _cap_tris_cdt(ring_xy)
        local = _orient_cap_tris(local, ring_xy, want_positive_z)
        return (local + base_idx).reshape(-1)

    if smooth_iters > 0 and positions.shape[0] > 0:
        # Reference slice centroids (pre-smooth) for restoring placement after Taubin.
        side_ref = side_positions[:, :2].reshape(s_total, n_ring, 2).astype(np.float64, copy=False)

        # Smooth only the side strips. Including cap diagonals in the graph
        # pulls end-ring vertices across the ring (their cap neighbors are far
        # from them in XY), which can collapse or self-intersect the cap ring.
        # The end rings still get Laplacian averaging from strip neighbors.
        side_faces = np.concatenate(strips).reshape(-1, 3).astype(np.int64, copy=False)
        mesh = trimesh.Trimesh(vertices=positions.astype(np.float64, copy=False), faces=side_faces, process=False)
        lam, mu = _taubin_lam_mu(smooth_factor)
        trimesh.smoothing.filter_taubin(mesh, lamb=lam, nu=mu, iterations=int(smooth_iters))
        positions = mesh.vertices.astype(np.float32, copy=False)

        # Single post-smooth pass:
        # - re-lock Z to input stack heights
        # - restore per-slice XY centroid + mean radius to pre-smooth values
        # - sequentially roll slice s to match s-1 for strip correspondence
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
            positions[lo : lo + n_ring, 2] = float(z_arr[s])

        for s in range(1, s_total):
            side_xy[s] = _align_ring_min_sqdist(
                np.ascontiguousarray(side_xy[s - 1], dtype=np.float64),
                np.ascontiguousarray(side_xy[s], dtype=np.float64),
            )
        positions[:n_side_verts, :2] = side_xy.reshape(-1, 2).astype(np.float32, copy=False)

    # Split end-cap vs tube at shared rings for shading (see docstring). Geometry
    # is still the same (XYZ); only the index buffer for cap tris changes.
    v0 = n_side_verts
    cap_dup = np.vstack((
        positions[0:n_ring].copy(),
        positions[(s_total - 1) * n_ring : s_total * n_ring].copy(),
    ))
    positions = np.vstack((positions, cap_dup))
    cap0 = _cap_flat(
        np.ascontiguousarray(positions[v0 : v0 + n_ring, :2], dtype=np.float64),
        v0, want_positive_z=False,
    )
    cap1 = _cap_flat(
        np.ascontiguousarray(
            positions[v0 + n_ring : v0 + 2 * n_ring, :2], dtype=np.float64,
        ),
        v0 + n_ring, want_positive_z=True,
    )
    indices = np.concatenate([np.concatenate(strips), cap0, cap1])

    normals = _compute_vertex_normals(positions, indices)

    # Caps are guaranteed planar in Z (we re-locked Z at line above); overwrite
    # cap normals with exact +Z/-Z so float roundoff from cross products cannot
    # nudge them off-axis after meshopt quantization.
    normals[v0 : v0 + n_ring] = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    normals[v0 + n_ring : v0 + 2 * n_ring] = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    bbox = (float(mins[0]), float(mins[1]), float(mins[2]),
            float(maxs[0]), float(maxs[1]), float(maxs[2]))
    return positions, indices, normals, bbox


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

def build_all_cells_mesh(
    gdf_render,
    z_scale: float = 2.0,
    smooth_iters: int = 1,
    smooth_factor: float = 0.5,
    ring_vertex_target: int | None = 48,
    *,
    ring_curvature_base: float = 0.28,
    ring_adaptive: bool = True,
    ring_adaptive_min_mul: float = 0.55,
    ring_adaptive_max_mul: float = 2.25,
    ring_adaptive_exponent: float = 0.75,
    _on_progress=None,
):
    """Build a combined mesh for every cell.

    Returns ``(positions, indices, colors, bbox)`` as flat Python lists (for
    backward compatibility with GLB writers that expect list/ndarray-agnostic
    inputs).
    """
    # Group once: avoids O(n_cells * n_rows) boolean slicing.
    groups = {cid: df for cid, df in gdf_render.groupby("cell_id", sort=False)}
    cell_ids = sorted(groups.keys())
    n = len(cell_ids)
    if n == 0:
        return [], [], [], [0, 0, 0, 0, 0, 0]

    n_workers = os.cpu_count() or 4
    batch_size = max(1, math.ceil(n / (n_workers * 4)))
    indexed = list(enumerate(cell_ids))
    batches = [indexed[s: s + batch_size] for s in range(0, n, batch_size)]

    # Extract rings once (used for adaptive sizing + meshing).
    rings_by_cid: dict = {}
    zs_by_cid: dict = {}
    scores: dict = {}
    for cid in cell_ids:
        rings, zs = _collect_rings_for_cell(groups[cid], z_scale)
        rings_by_cid[cid] = rings
        zs_by_cid[cid] = zs
        scores[cid] = _cell_max_turning(rings) if rings else 0.0

    n_by_cid = _adaptive_ring_targets_from_scores(
        cell_ids,
        scores,
        ring_vertex_target,
        adaptive=ring_adaptive,
        min_mul=ring_adaptive_min_mul,
        max_mul=ring_adaptive_max_mul,
        exponent=ring_adaptive_exponent,
    )

    def _build_batch(batch):
        out = []
        for i, cid in batch:
            if _on_progress is not None:
                _on_progress(i, n, cid)
            rt = ring_vertex_target if n_by_cid is None else n_by_cid[cid]
            pos, idx, nrm, bbox = build_loft_mesh_from_rings(
                rings_by_cid[cid],
                zs_by_cid[cid],
                smooth_iters=smooth_iters,
                smooth_factor=smooth_factor,
                ring_vertex_target=rt,
                ring_curvature_base=ring_curvature_base,
            )
            out.append((i, pos, idx, nrm, bbox))
        return out

    batch_results = [_build_batch(b) for b in batches]

    cell_results = {}
    for batch_out in batch_results:
        for i, pos, idx, nrm, bbox in batch_out:
            if len(pos) > 0:
                cell_results[i] = (pos, idx, nrm, bbox)

    if not cell_results:
        return [], [], [], [], [0, 0, 0, 0, 0, 0]

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
    colors = np.vstack(col_parts)

    b = np.asarray(all_bboxes, dtype=np.float64)
    bbox = [float(b[:, 0].min()), float(b[:, 1].min()), float(b[:, 2].min()),
            float(b[:, 3].max()), float(b[:, 4].max()), float(b[:, 5].max())]

    # Return flat lists to keep existing callers working.
    return (
        positions.reshape(-1).astype(np.float32).tolist(),
        indices.astype(np.int64).tolist(),
        normals.reshape(-1).astype(np.float32).tolist(),
        colors.reshape(-1).astype(np.float32).tolist(),
        bbox,
    )
