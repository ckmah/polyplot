"""Internal Joblib knobs for parallel mesh export.

Defaults are empty: behavior matches historical ``Parallel(n_jobs=-1, prefer="threads")``.
Autoresearch scripts may rewrite ``EXTRA_*`` dicts and commit; public API is unchanged.
"""

from __future__ import annotations

EXTRA_LOFT_PARALLEL: dict = {}
EXTRA_TILE_PARALLEL: dict = {'batch_size': 6}


def loft_parallel_kw() -> dict:
    kw: dict = {"n_jobs": -1, "prefer": "threads"}
    kw.update(EXTRA_LOFT_PARALLEL)
    return kw


def tile_parallel_kw(n_jobs: int) -> dict:
    kw: dict = {"n_jobs": n_jobs, "prefer": "threads"}
    kw.update(EXTRA_TILE_PARALLEL)
    return kw
