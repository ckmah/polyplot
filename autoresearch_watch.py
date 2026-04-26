# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.22",
#     "matplotlib>=3.8",
# ]
# ///

"""Live plot of autoresearch `results.tsv` (repo root). Run: `uv run marimo edit autoresearch_watch.py`."""

import base64
import io

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")


@app.cell
def _():
    import pathlib

    import marimo as mo
    import matplotlib.pyplot as plt

    return mo, pathlib, plt


@app.cell
def _(mo, pathlib):
    root = pathlib.Path(__file__).resolve().parent
    results_path = root / "results.tsv"
    mo.watch.file(str(results_path))
    return (results_path,)


@app.cell
def _(mo, pathlib, plt, results_path):
    if not results_path.is_file():
        return mo.md(f"No `{results_path.name}` yet. Start the autoresearch loop.")

    lines = results_path.read_text(encoding="utf-8").strip().splitlines()
    if len(lines) < 2:
        return mo.md("Results file has header only; waiting for experiments.")

    header = lines[0].split("\t")
    rows = [ln.split("\t") for ln in lines[1:] if ln.strip()]
    exp_i = header.index("experiment")
    met_i = header.index("metric")
    stat_i = header.index("status")
    xs = [int(r[exp_i]) for r in rows]
    ys = [float(r[met_i]) for r in rows]
    st = [r[stat_i] for r in rows]
    baseline_y = ys[0] if st and st[0] == "baseline" else None

    fig, ax = plt.subplots(figsize=(9, 4))
    colors = [
        "#2ca02c" if s == "keep" else "#d62728" if s == "discard" else "#7f7f7f"
        for s in st
    ]
    ax.scatter(xs, ys, c=colors, s=36, zorder=3)
    ax.plot(xs, ys, color="#1f77b4", linewidth=1, alpha=0.7, zorder=2)
    if baseline_y is not None:
        ax.axhline(baseline_y, color="#ff7f0e", linestyle="--", linewidth=1, label="baseline")
    best = min((y for y, s in zip(ys, st) if s != "crash"), default=None)
    if best is not None:
        ax.axhline(best, color="#9467bd", linestyle=":", linewidth=1, label="best so far")
    ax.set_xlabel("experiment")
    ax.set_ylabel("real seconds (lower is better)")
    ax.set_title("Autoresearch: `uv run python notebook.py` runtime")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return mo.Html(
        f'<img src="data:image/png;base64,{b64}" alt="runtime plot" style="max-width:100%" />'
    )


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
