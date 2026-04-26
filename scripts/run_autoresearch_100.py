#!/usr/bin/env python3
"""Run 100 autoresearch experiments on ``polyplot/_tune.py`` (Joblib batch_size grid).

Each experiment: commit tune dicts, run ``meshify_benchmark_measure.py``, keep if
metric improves strictly vs best-so-far else ``git reset --hard HEAD~1``. Appends
``results.tsv`` (must exist with header). Lower metric is better.
"""

from __future__ import annotations

import itertools
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TUNE_FILE = ROOT / "polyplot" / "_tune.py"
RESULTS = ROOT / "results.tsv"
MEASURE = ROOT / "scripts" / "meshify_benchmark_measure.py"

_ALLOWED_DIRTY_SUFFIX = (
    "results.tsv",
    "run.log",
    "autoresearch_run.log",
)


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, check=False, **kw)


def _git_short() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
    ).strip()


def read_best_metric() -> float:
    lines = RESULTS.read_text(encoding="utf-8").strip().splitlines()
    best = float("inf")
    for ln in lines[1:]:
        parts = ln.split("\t")
        if len(parts) < 4:
            continue
        if parts[3] not in ("baseline", "keep"):
            continue
        best = min(best, float(parts[2]))
    return best


def append_result(
    experiment: int, commit: str, metric: float, status: str, description: str
) -> None:
    with RESULTS.open("a", encoding="utf-8") as f:
        f.write(
            f"{experiment}\t{commit}\t{metric:.6f}\t{status}\t{description}\n"
        )


def write_tune(loft: dict, tile: dict) -> None:
    text = TUNE_FILE.read_text(encoding="utf-8")
    text = re.sub(
        r"^EXTRA_LOFT_PARALLEL: dict = .*$",
        "EXTRA_LOFT_PARALLEL: dict = " + repr(loft),
        text,
        count=1,
        flags=re.M,
    )
    text = re.sub(
        r"^EXTRA_TILE_PARALLEL: dict = .*$",
        "EXTRA_TILE_PARALLEL: dict = " + repr(tile),
        text,
        count=1,
        flags=re.M,
    )
    TUNE_FILE.write_text(text, encoding="utf-8")


def measure() -> float:
    out = subprocess.check_output(
        ["uv", "run", "python", str(MEASURE)], cwd=ROOT, text=True
    )
    return float(out.strip())


def build_schedule(n: int) -> list[tuple[dict, dict]]:
    def opt(b: int | None) -> dict:
        return {} if b is None else {"batch_size": int(b)}

    loft_bs = [
        None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48,
    ]
    tile_bs = [None, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
    pairs: list[tuple[dict, dict]] = []
    for a, b in itertools.product(loft_bs, tile_bs):
        if a is None and b is None:
            continue
        pairs.append((opt(a), opt(b)))
        if len(pairs) >= n:
            return pairs
    k = 64
    while len(pairs) < n:
        pairs.append((opt(k), opt(k // 2)))
        k += 8
    return pairs[:n]


def assert_clean_git() -> None:
    st = _run(["git", "status", "--porcelain"], capture_output=True, text=True)
    bad: list[str] = []
    for ln in st.stdout.splitlines():
        path = ln[3:].strip() if len(ln) > 3 else ln
        if any(path.endswith(s) or s in path for s in _ALLOWED_DIRTY_SUFFIX):
            continue
        bad.append(ln)
    if bad:
        print(
            "commit or stash before running (unrelated changes):\n" + "\n".join(bad[:30]),
            file=sys.stderr,
        )
        sys.exit(1)


def main() -> None:
    if not RESULTS.is_file():
        print("missing results.tsv", file=sys.stderr)
        sys.exit(1)
    assert_clean_git()

    best = read_best_metric()
    print(f"best metric from results.tsv: {best:.6f}", flush=True)

    schedule = build_schedule(100)
    for i, (loft, tile) in enumerate(schedule, start=1):
        desc = f"loft{loft!s} tile{tile!s}"
        write_tune(loft, tile)
        _run(["git", "add", "polyplot/_tune.py"])
        c = _run(
            ["git", "commit", "-m", f"experiment: tune joblib loft={loft} tile={tile}"],
            capture_output=True,
            text=True,
        )
        if c.returncode != 0:
            print(f"experiment {i}: commit failed:\n{c.stderr}", file=sys.stderr)
            append_result(i, "-", 0.0, "crash", f"git commit failed {desc}")
            _run(["git", "checkout", "--", "polyplot/_tune.py"])
            continue
        commit = _git_short()
        try:
            m = measure()
        except Exception as e:
            print(f"experiment {i}: measure failed {e}", file=sys.stderr)
            _run(["git", "reset", "--hard", "HEAD~1"])
            append_result(i, commit, 0.0, "crash", str(e)[:120])
            best = read_best_metric()
            continue
        if m < best - 1e-12:
            best = m
            append_result(i, commit, m, "keep", desc)
            print(f"{i}/100 KEEP {m:.6f} {desc}", flush=True)
        else:
            _run(["git", "reset", "--hard", "HEAD~1"])
            append_result(i, "-", m, "discard", desc)
            print(f"{i}/100 discard {m:.6f} {desc}", flush=True)

    print("done. Branch tip is best-so-far tune commits (discard resets applied).", flush=True)


if __name__ == "__main__":
    main()
