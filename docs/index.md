---
icon: lucide/rocket
---

# Polyplot

**Polyplot** is a [Three.js](https://threejs.org/) (React Three Fiber) [anywidget](https://anywidget.dev/) for **2.5D** polygon stacks: it preprocesses stacked 2D outlines per `cell_id` and `ZIndex`, builds triangulated meshes, exports **GLB** tiles, and serves them to a **WebGL** viewer in the browser (marimo, Jupyter, and similar).

- [**Get started**](get-started.md) – install, sample data, and the quickstart notebook.
- [**User guide**](user-guide.md) – `meshify`, `plot`, and caching.
- [PyPI / source](https://github.com/ckmah/polyplot) – `uv sync` from the repository root; API docstrings in `polyplot/`.

## At a glance

| Piece | Role |
|-------|------|
| `meshify(gdf, ...)` | Preprocess, tile, and write `tiles.json` and GLB shards under a content-hashed directory (default `.polyplot/`) |
| `plot(gdf, ...)` | Ensure tiles exist, run a local tile server, and return a marimo `anywidget` WebGL view |

The viewer controls **wireframe**, **opacity**, and **background** in the UI; they are not set from Python.
