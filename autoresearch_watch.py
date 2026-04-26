# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.22",
#     "matplotlib>=3.8",
# ]
# ///

"""Live plot of autoresearch `results.tsv` (repo root). Run: `uv run marimo edit autoresearch_watch.py`."""

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import html
    import pathlib

    import marimo as mo
    import plotly.graph_objects as go

    return go, html, mo, pathlib


@app.cell
def _(mo, pathlib):
    root = pathlib.Path(__file__).resolve().parent
    results_path = mo.watch.file(root / "results.tsv")
    return (results_path,)


@app.cell
def _(go, html, mo, results_path):
    if not results_path.is_file():
        out = mo.md(f"No `{results_path.name}` yet. Start the autoresearch loop.")
    else:
        lines = results_path.resolve().read_text(encoding="utf-8").strip().splitlines()
        if len(lines) < 2:
            out = mo.md("Results file has header only; waiting for experiments.")
        else:
            header = lines[0].split("\t")
            rows = [ln.split("\t") for ln in lines[1:] if ln.strip()]
            exp_i = header.index("experiment")
            com_i = header.index("commit")
            met_i = header.index("metric")
            stat_i = header.index("status")
            desc_i = header.index("description")

            xs = [int(r[exp_i]) for r in rows]
            ys = [float(r[met_i]) for r in rows]
            st = [r[stat_i] for r in rows]
            commits = [r[com_i] for r in rows]
            descs = [r[desc_i] for r in rows]

            color_for = {
                "keep": "#2ca02c",
                "discard": "#d62728",
                "baseline": "#7f7f7f",
                "crash": "#bcbd22",
            }
            marker_colors = [color_for.get(s, "#333333") for s in st]

            hover_html: list[str] = []
            for x, y, cmt, stat, dsc in zip(
                xs, ys, commits, st, descs, strict=True
            ):
                safe_d = html.escape(dsc, quote=True)
                hover_html.append(
                    "<b>experiment "
                    + str(x)
                    + "</b> ("
                    + html.escape(stat, quote=True)
                    + ")<br>"
                    "<b>metric</b> "
                    + f"{y:.6f}"
                    + "<br>"
                    "<b>commit</b> "
                    + html.escape(cmt, quote=True)
                    + "<br><br>"
                    + safe_d
                )

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines+markers",
                    line=dict(color="rgba(31, 119, 180, 0.75)", width=2),
                    marker=dict(
                        size=11,
                        color=marker_colors,
                        line=dict(width=1, color="rgba(255,255,255,0.7)"),
                    ),
                    text=hover_html,
                    hovertemplate="%{text}<extra></extra>",
                )
            )

            if st and st[0] == "baseline":
                fig.add_hline(
                    y=ys[0],
                    line_dash="dash",
                    line_color="#ff7f0e",
                    line_width=2,
                    annotation_text="baseline",
                    annotation_position="right",
                )
            finite = [y for y, s in zip(ys, st, strict=True) if s != "crash"]
            if finite:
                best_y = min(finite)
                fig.add_hline(
                    y=best_y,
                    line_dash="dot",
                    line_color="#9467bd",
                    line_width=2,
                    annotation_text="best so far",
                    annotation_position="left",
                )

            fig.update_layout(
                title="Autoresearch: meshify wall time per cell",
                xaxis_title="experiment",
                yaxis_title="MESHIFY_PER_CELL_SECONDS (lower is better)",
                hovermode="closest",
                height=440,
                margin=dict(l=60, r=40, t=56, b=52),
                showlegend=False,
            )
            fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.25)")
            fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.25)")
            out = mo.ui.plotly(fig)
    out
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
