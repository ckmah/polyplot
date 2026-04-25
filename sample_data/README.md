# Sample data

- **`liver_crop_sample.parquet`** (tracked) — small subset (~50 cells in one spatial neighborhood) for docs, CI, and `quickstart.py`. Generated with `uv run python scripts/make_liver_subset.py` from a local full export.
- **`liver_crop.parquet`** (not committed) — full dataset; place it here locally if you use `scripts/make_liver_subset.py` or the full [`notebook.py`](../notebook.py) against the large file.

Other large parquets in this directory are ignored by `.gitignore` and stay on your machine only.
