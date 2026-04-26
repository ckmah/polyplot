#!/usr/bin/env python3
"""Run ``notebook.py`` once and print meshify-only seconds.

Reads the last line matching ``^MESHIFY_SECONDS=(.+)$`` from combined stdout/stderr.
Also writes the full capture to ``run.log`` at the repo root for debugging.

Usage (metric for autoresearch)::

    METRIC=$(uv run python scripts/meshify_benchmark_measure.py)
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

_PATTERN = re.compile(r"^MESHIFY_SECONDS=(.+)$")


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
        m = _PATTERN.match(line.strip())
        if m:
            metric = m.group(1).strip()
    if metric is None:
        print("error: no MESHIFY_SECONDS= line in notebook output", file=sys.stderr)
        sys.exit(1)
    print(metric)
    if proc.returncode != 0:
        sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
