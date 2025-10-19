
# Hitter Archetypes — Walkthrough

This repository contains a pipeline and interactive Streamlit app for clustering and analyzing MLB hitter archetypes using aggregated Statcast and tracking data. It includes data extraction and cleaning scripts, a clustering pipeline (preprocessing, PCA, KMeans), per-cluster exports, and a Streamlit showcase that lets you search players, inspect cluster centroids, and visualize why players belong in particular archetypes.

This README documents every step taken — from data preparation to clustering and deployment — so someone new to the repo can reproduce and extend the results.

---

## Project overview

- Goal: identify and visualize meaningful hitter archetypes using Statcast metrics and derived features (exit velocity, launch angle, bat speed, swing geometry, intercept offsets, etc.).
- What it does: aggregates per-player features, preprocesses them (circular-angle handling, count transforms, winsorization, scaling), reduces dimensionality with PCA for visualization, clusters players (KMeans), and exports per-cluster CSVs and representative players.
- Why it’s interesting: combines modern tracking features with classic contact/power metrics, uses careful feature engineering for circular variables, and provides an interactive UI to explore clusters and player-level diagnostics.

## Quick start (run the app locally)

1. Create a Python environment and install dependencies. Example with pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# optional Streamlit extras
pip install -r requirements-streamlit.txt
```

2. Make sure required data files are present in `data/`:
- `data/player_archetypes.csv` — per-player aggregated features used by the app.
- `data/raw/unique_batters_with_names.csv` — optional player id -> name mapping for UI labels.

3. Start the Streamlit app:

```bash
streamlit run app.py
```

Open the host URL printed by Streamlit (typically http://localhost:8501).

## Data preparation — detailed

All data preparation steps are implemented as scripts under `scripts/` and aided by notebooks in `notebooks/` for exploration.

1. Raw extraction and sources
	- Raw Statcast / pitch-level / swing-level CSVs are stored in `data/raw/`.
	- `notebooks/get_savant_data.ipynb` contains examples showing how the raw data was obtained and inspected.

2. Aggregation
	- `scripts/cluster_archetypes.py` (and helper notebooks) compute per-player aggregates: mean, std, median, and high-percentile stats (e.g., p95 of exit velocity).
	- Aggregates include `launch_speed_mean`, `launch_angle_mean`, `bat_speed_mean`, `pct_hard_hit`, `swing_length_mean`, `intercept_ball_minus_batter_pos_*_mean`, and `n_swings`.

3. Handling missing values & outliers
	- Missing numeric values are replaced with reasonable defaults during preprocessing (e.g. 0 for missing counts) or handled via drop/ignore in operations requiring complete data.
	- Winsorize each numeric column at 1st/99th percentiles to limit extreme values' effects on PCA and clustering.

4. Circular features
	- Mean angle columns (e.g., `attack_direction_mean`, `launch_angle_mean`) are converted to sin/cos components before distance computations or PCA to avoid wrap-around discontinuities.

5. Name mapping
	- The code tries to enrich numeric player ids using `data/raw/unique_batters_with_names.csv`. If found, it creates `name` or `name_display` columns used by the UI. If missing, the app warns and falls back to ids (unless you opt into ID fallback).

Scripts of note:
- `scripts/add_player_names.py` — create mapping CSVs or enrich the main per-player table.
- `scripts/cluster_archetypes.py` — runs preprocessing and clustering, writes outputs to `data/clusters/`.

## Feature selection & engineering

Key features used and why:

- Power/contact: `launch_speed_mean`, `pct_hard_hit` — proxies for exit velocity and hard contact.
- Swing mechanics: `bat_speed_mean`, `swing_length_mean`, `swing_path_tilt_mean` — represent swing speed and geometry.
- Timing/intercept: `intercept_ball_minus_batter_pos_x_inches_mean`, `intercept_ball_minus_batter_pos_y_inches_mean` — lateral/vertical intercept offsets.
- Angles: `launch_angle_mean`, `attack_angle_mean`, `attack_direction_mean` — converted to sin & cos for model usage.
- Sample size: `n_swings` — used for filtering and log transforms to avoid small-sample artifacts.

Transforms:
- For angle features, create two columns per angle: `_sin` and `_cos`.
- Log-transform count-like columns with `log1p`.
- Winsorize per-column at [0.01, 0.99].
- Use `RobustScaler` for final scaling (less sensitive to outliers than StandardScaler).

## Clustering pipeline

High-level flow:

1. Select features to include (the code defaults to a diagnostic-provided list or numeric columns).
2. Preprocess: angle expansion, log1p counts, winsorize, robust scale.
3. (Optional) PCA for visualization; usually 2 components for scatter plots.
4. KMeans clustering (choose k experimentally; the repo used k=5 in examples).
5. Save cluster assignments back to `data/player_archetypes.csv` and export per-cluster CSVs to `data/clusters/`.

Practical tips:
- Fix `random_state` in KMeans for reproducibility.
- Save scalers and transformation metadata so new players can be assigned to clusters identically later.

## Evaluation & diagnostics

Because clustering is unsupervised, focus on diagnostics and qualitative checks:

- Centroid heatmap: compute z-score of centroid features vs overall mean; use a heatmap to show distinguishing features for each cluster.
- Representative players: find nearest players to each centroid (by z-score distance) and use these to label cluster archetypes.
- PCA scatter: visually inspect cluster separation and compactness.
- Optional metrics: silhouette score or Davies-Bouldin index for objective comparison of clusterings.

## Streamlit app / model usage

`app.py` is the primary interactive interface. It:

- Loads `data/player_archetypes.csv` and merges names if mapping files exist.
- Renders an interactive PCA scatter of all players colored by cluster. Hovering shows a player's name (if available) and id plus cluster.
- Renders centroid heatmaps and a plain-English generated summary for each cluster using z-scored centroids.
- Allows searching for a player by name and displays player statistics and how they deviate from the cluster mean (with special handling for circular angles).

Use cases:
- Explore cluster archetypes (e.g., find the cluster of high-exit-velocity hitters).
- Inspect a particular player and see how they differ from their cluster.
- Export representative players or full cluster CSVs for downstream analysis.

## Project files (what each file does)

- `app.py` — Streamlit UI: plotting, search, cluster descriptions, player detail views.
- `data/player_archetypes.csv` — aggregated per-player features used as the main input.
- `data/raw/unique_batters_with_names.csv` — optional name mapping to replace numeric ids with friendly names.
- `data/clusters/*` — per-cluster CSVs and summary diagnostics.
- `scripts/` — data preparation and clustering scripts (e.g., `cluster_archetypes.py`, `add_player_names.py`, `sort_players_by_cluster.py`).
- `notebooks/` — exploratory analysis and visualization notebooks.
- `utils/` — helper modules for UI selection and small utilities.
- `requirements.txt`, `requirements-streamlit.txt` — dependency lists.

## Reproducing the pipeline (step-by-step)

1. Obtain raw data (Statcast / pitch-level CSVs) and place them in `data/raw/`.
2. Run the aggregation script that produces `data/player_archetypes.csv`. Example (adjust to your script arguments):

```bash
python scripts/cluster_archetypes.py --input data/raw --output data/player_archetypes.csv
```

3. Optionally prepare mapping CSVs (`data/raw/unique_batters_with_names.csv`) so the Streamlit UI displays names instead of ids.
4. Tune k and run the clustering step:

```bash
python scripts/cluster_archetypes.py --players data/player_archetypes.csv --k 5 --out data/clusters/
```

5. Start the app and explore:

```bash
streamlit run app.py
```

## Future improvements & next steps

- Add supervised prediction tasks: predict next-season wOBA, HR probability, or plate discipline outcomes using these features as inputs.
- Improve feature engineering: per-pitch context, zone-specific metrics, and temporal (rolling) aggregations.
- Deploy a hosted interactive dashboard with hosted data or server-side cluster assignment endpoints.
- Add unit tests for feature transformations (angle handling, winsorization), clustering reproducibility, and UI selection logic.

---

If you want, I can expand this README with concrete example commands for each script, add a short tutorial notebook that runs the pipeline end-to-end, or add a `deploy` guide for Streamlit Cloud. Tell me which you'd like next and I'll add it.
