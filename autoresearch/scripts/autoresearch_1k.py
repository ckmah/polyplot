#!/usr/bin/env python3
"""Run up to N autoresearch experiments (default 1000) on meshify.

Each experiment: derived from index as (refine_win, thread_cap, fft_align_threshold).
Not a blind grid for its own sake — these are three independent *algorithmic*
knobs (local search width, embarrassingly-parallel batching, FFT vs exhaustive
split) that change work and complexity class. Commits before measure; reverts
on regression. Appends results.tsv.

Usage:
    uv run python autoresearch/scripts/autoresearch_1k.py
    uv run python autoresearch/scripts/autoresearch_1k.py --count 200
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
AUTORESEARCH_DIR = _SCRIPT_DIR.parent
REPO_ROOT = AUTORESEARCH_DIR.parent
_LOG_DIR = AUTORESEARCH_DIR / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
MB = REPO_ROOT / "polyplot" / "_mesh_build.py"
RESULTS = AUTORESEARCH_DIR / "results.tsv"
LOG = _LOG_DIR / "autoresearch_1k.log"
MEASURE = _SCRIPT_DIR / "meshify_benchmark_measure.py"


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


def _append(exp: int, commit: str, metric: float, status: str, desc: str) -> None:
    with RESULTS.open("a") as f:
        f.write(f"{exp}\t{commit}\t{metric:.6f}\t{status}\t{desc}\n")


def params_from_index(idx: int) -> tuple[int, int, int]:
    """Map 0..1175 to (refine_win 4..24, thread_cap 1..8, fft_threshold 256..640 step 64)."""
    fft_i = idx % 7
    idx //= 7
    cap = 1 + (idx % 8)
    idx //= 8
    win = 4 + (idx % 21)
    fft_th = 256 + fft_i * 64
    return win, cap, fft_th


def apply_mesh_variants(content: str, win: int, cap: int, fft_th: int) -> str:
    """Rewrite _mesh_build.py text: refine window, thread cap, FFT-only split."""
    c2 = content
    c2 = re.sub(
        r"best_k = int\(_refine_shift_nb\(prev, curr, k0, \d+\)\)",
        f"best_k = int(_refine_shift_nb(prev, curr, k0, {win}))",
        c2,
        count=1,
    )
    c2 = re.sub(
        r"_nw = min\(\(_os\.cpu_count\(\) or 1\), len\(cell_ids\), \d+\)",
        f"_nw = min((_os.cpu_count() or 1), len(cell_ids), {cap})",
        c2,
        count=1,
    )
    c2 = re.sub(
        r"if n > \d+:\n        return np\.roll\(curr, -_best_roll_fft",
        f"if n > {fft_th}:\n        return np.roll(curr, -_best_roll_fft",
        c2,
        count=1,
    )
    return c2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=1000, help="experiments to run")
    args = ap.parse_args()

    tsv_min = _best()
    session0 = _measure()
    if session0 is None:
        print("refusing to start: could not measure current HEAD (meshify)", file=sys.stderr)
        sys.exit(1)
    # TSV may include pre-fix-parallel-bug numbers; improve vs current tree.
    best = session0
    start = _max_exp() + 1
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    with LOG.open("a") as f:
        f.write(
            f"{ts} autoresearch_1k start exp {start}..{start + args.count - 1} "
            f"session_baseline={session0} tsv_min_keep={tsv_min}\n"
        )

    for k in range(args.count):
        exp_id = start + k
        win, cap, fft_th = params_from_index(k)
        desc = (
            f"structural grid k={k}: refine_win={win} thread_cap={cap} "
            f"fft_align_if_n_gt={fft_th}"
        )
        raw = MB.read_text(encoding="utf-8")
        new = apply_mesh_variants(raw, win, cap, fft_th)
        if new == raw:
            _append(exp_id, "-", 0.0, "skip", desc + " (no regex match)")
            continue
        MB.write_text(new, encoding="utf-8")
        _git("add", "polyplot/_mesh_build.py")
        c = _git("commit", "-m", f"experiment {exp_id}: mesh W={win} C={cap} F={fft_th}")
        if c.returncode != 0:
            _git("checkout", "HEAD", "--", "polyplot/_mesh_build.py")
            _append(exp_id, "-", 0.0, "skip", desc + " commit_fail")
            continue
        commit = (_git("rev-parse", "--short", "HEAD").stdout or "").strip()
        t0 = time.perf_counter()
        metric = _measure()
        dt = time.perf_counter() - t0
        if metric is None:
            _git("reset", "--hard", "HEAD~1")
            _append(exp_id, "-", -1.0, "crash", desc)
            with LOG.open("a") as f:
                f.write(f"exp {exp_id} crash {dt:.1f}s\n")
            continue
        if metric < best - 1e-12:
            best = metric
            _append(exp_id, commit, metric, "keep", desc)
            with LOG.open("a") as f:
                f.write(f"exp {exp_id} KEEP {metric:.6f} {dt:.1f}s\n")
        else:
            _git("reset", "--hard", "HEAD~1")
            _append(exp_id, "-", metric, "discard", desc)
            with LOG.open("a") as f:
                f.write(f"exp {exp_id} discard {metric:.6f} vs {best:.6f} {dt:.1f}s\n")

    with LOG.open("a") as f:
        f.write(f"autoresearch_1k done best={best:.6f}\n")
    print(f"done. best={best:.6f}")


if __name__ == "__main__":
    main()
