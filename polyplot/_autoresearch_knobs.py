"""Internal scalar defaults for mesh export; rewritten by autoresearch scripts.

Public API is unchanged: these replace former literal defaults in :mod:`polyplot`.
"""

from __future__ import annotations

# Tile / ring mesh (defaults match historical literals)
RING_TARGET_DEFAULT: int = 48
RING_CURVATURE_BASE_DEFAULT: float = 0.28
Z_SCALE_DEFAULT: float = 2.0
ADAPTIVE_MIN_MUL: float = 0.55
ADAPTIVE_MAX_MUL: float = 2.25
ADAPTIVE_EXPONENT: float = 0.75

# Alignment (_align_ring_min_sqdist)
ALIGN_EXHAUSTIVE_MAX_N: int = 128
ALIGN_FFT_ONLY_MIN_N: int = 512
ALIGN_REFINE_WINDOW: int = 16

# Cap Lawson flips: max_passes = CDT_PASS_MULTIPLIER * n
CDT_PASS_MULTIPLIER: int = 4

# preprocess_gdf default when meshify calls without second arg
PREPROCESS_SIMPLIFY_TOL: float = 0.5
