#!/usr/bin/env python3
"""Run ``notebook.py`` once and print meshify wall time **per cell** (seconds).

Reads the last line matching ``^MESHIFY_PER_CELL_SECONDS=(.+)$`` from combined
stdout/stderr. Also writes the full capture to ``run.log`` at the repo root.

Usage (autoresearch metric; lower is better)::

    METRIC=$(uv run python scripts/meshify_benchmark_measure.py)
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

_PER_CELL = re.compile(r"^MESHIFY_PER_CELL_SECONDS=(.+)$")


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    proc = subprocess.run(
        ["uv", "run", "python", str(root / "notebook.py")],
        cwd=root,
        capture_output=True,
        text=True,
    )
    blob = proc.stdout + proc.stderr
    (root / "run.log").write_text(blob, encoding="utf-8")
    metric = None
    for line in blob.splitlines():
        m = _PER_CELL.match(line.strip())
        if m:
            metric = m.group(1).strip()
    if metric is None:
        print(
            "error: no MESHIFY_PER_CELL_SECONDS= line in notebook output",
            file=sys.stderr,
        )
        sys.exit(1)
    print(metric)
    if proc.returncode != 0:
        sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
