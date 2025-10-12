#!/usr/bin/env python3
"""
Cluster players into archetypes using per-swing Statcast data.

What it does:
- Reads a swings CSV (default: data/raw/2025_swings.csv)
- Aggregates per-player features (means/rates) for selected statcast columns
- Derives whiff/contact rates when possible from event/description columns
- Standardizes features, runs PCA for visualization
- Chooses K (clusters) by maximizing silhouette score over k=2..10
- Runs KMeans, writes per-player cluster assignments to CSV and saves plots

Usage:
  python3 scripts/cluster_archetypes.py \
      --input data/raw/2025_swings.csv \
      --player-col batter \
      --out data/player_archetypes.csv --plots out/plots

Notes:
- The script is defensive: if a named column isn't present in the CSV it will skip
  that feature and proceed with the ones available.
- You can pass alternate column names for launch speed/angle/bat speed via CLI.
"""
from pathlib import Path
import argparse
import sys
import re
import warnings

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def parse_args():
    p = argparse.ArgumentParser(description='Cluster players into archetypes from Statcast swings')
    p.add_argument('--input', '-i', default='data/raw/2025_swings.csv', help='Input swings CSV')
    p.add_argument('--player-col', default='batter', help='Column name for player id')
    p.add_argument('--launch-speed', default='launch_speed', help='launch speed column name')
    p.add_argument('--launch-angle', default='launch_angle', help='launch angle column name')
    p.add_argument('--bat-speed', default='bat_speed', help='bat speed column name (if available)')
    p.add_argument('--description-col', default='description', help='column with event/description to infer whiffs')
    p.add_argument('--out', default='data/player_archetypes.csv', help='Output CSV with cluster assignments')
    p.add_argument('--plots', default='out/cluster_plots', help='Directory to write diagnostic plots')
    p.add_argument('--min-k', type=int, default=2, help='Minimum k to try')
    p.add_argument('--max-k', type=int, default=10, help='Maximum k to try')
    p.add_argument('--k', type=int, default=None, help='If set, force this k and skip automatic selection')
    p.add_argument('--min-swings', type=int, default=5, help='Minimum swings per player to include')
    p.add_argument('--features', default=None, help='Comma-separated list of feature column names to use (default: auto detect numeric)')
    p.add_argument('--n-init', type=int, default=10, help='n_init parameter for KMeans')
    p.add_argument('--imbalance-weight', type=float, default=0.0, help='Penalty weight for cluster size imbalance when selecting k (0 = ignore imbalance)')
    p.add_argument('--sample-frac', type=float, default=1.0, help='Fraction sample of players for quick runs (0-1]')
    return p.parse_args()


def safe_head(df, n=5):
    with pd.option_context('display.max_rows', n):
        return df.head(n)


def infer_whiff_flag(desc):
    """Return True if description string indicates a swinging strike / whiff.
    This is a heuristic covering common MLBAM description values.
    """
    if pd.isna(desc):
        return False
    d = str(desc).lower()
    # common swinging strike descriptions
    patterns = [r"swinging_strike", r"swinging strike", r"miss", r"whiff", r"swinging_strike_blocked"]
    for p in patterns:
        if re.search(p, d):
            return True
    return False


def aggregate_features(df, player_col, desc_col, launch_speed_col, launch_angle_col, bat_speed_col):
    # Ensure player id present
    if player_col not in df.columns:
        raise ValueError(f"Player column '{player_col}' not found in data. Columns: {list(df.columns)}")

    grouped = df.groupby(player_col)

    features = pd.DataFrame(index=grouped.size().index)

    # total swings/rows per player
    features['n_swings'] = grouped.size()

    # whiff rate (from description)
    if desc_col in df.columns:
        whiff = df[desc_col].apply(infer_whiff_flag)
        features['whiff_rate'] = grouped[desc_col].apply(lambda s: whiff.loc[s.index].mean())
    else:
        warnings.warn(f"Description column '{desc_col}' not found; skipping whiff_rate")

    # launch speed/angle means and stds
    if launch_speed_col in df.columns:
        features['launch_speed_mean'] = grouped[launch_speed_col].mean()
        features['launch_speed_std'] = grouped[launch_speed_col].std().fillna(0.0)
    else:
        warnings.warn(f"Launch speed column '{launch_speed_col}' not found; skipping")

    if launch_angle_col in df.columns:
        features['launch_angle_mean'] = grouped[launch_angle_col].mean()
        features['launch_angle_std'] = grouped[launch_angle_col].std().fillna(0.0)
    else:
        warnings.warn(f"Launch angle column '{launch_angle_col}' not found; skipping")

    # bat speed (if present)
    if bat_speed_col in df.columns:
        features['bat_speed_mean'] = grouped[bat_speed_col].mean()
        features['bat_speed_std'] = grouped[bat_speed_col].std().fillna(0.0)
    else:
        warnings.warn(f"Bat speed column '{bat_speed_col}' not found; skipping")

    # contact rate: heuristic if description contains 'contact' keywords
    if desc_col in df.columns:
        contact_flags = df[desc_col].fillna('').str.lower().str.contains('contact|foul|hit|ball in play|in play')
        features['contact_rate'] = grouped[desc_col].apply(lambda s: contact_flags.loc[s.index].mean())
    else:
        warnings.warn("No description column for contact_rate; skipping")

    # replace NaNs from players with single swing
    features = features.fillna(0.0)

    return features


def choose_k(X, min_k=2, max_k=10, n_init=10, imbalance_weight=0.0):
    """Choose k by silhouette score with optional penalty for imbalance.

    imbalance_weight: multiplies the imbalance penalty (0 disables). Penalty is
    computed as the coefficient of variation (std/mean) of cluster sizes.
    """
    best_k = None
    best_score = -1e9
    diagnostics = {}
    for k in range(min_k, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=n_init)
        labels = km.fit_predict(X)
        uniq = np.unique(labels)
        if len(uniq) == 1:
            diagnostics[k] = {'silhouette': -1.0, 'imbalance': 1.0, 'combined': -1.0}
            continue
        sil = silhouette_score(X, labels)
        # imbalance penalty: cv = std(counts) / mean(counts)
        counts = np.array([np.sum(labels == u) for u in uniq], dtype=float)
        cv = counts.std() / counts.mean() if counts.mean() > 0 else 1.0
        combined = sil - imbalance_weight * cv
        diagnostics[k] = {'silhouette': float(sil), 'imbalance': float(cv), 'combined': float(combined), 'counts': counts.tolist()}
        if combined > best_score:
            best_score = combined
            best_k = k
    return best_k, best_score, diagnostics


def plot_diagnostics(X_pca, labels, scores, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

    # PCA scatter
    dfp = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    dfp['cluster'] = labels.astype(str)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=dfp, x='PC1', y='PC2', hue='cluster', palette='tab10', s=30, linewidth=0)
    plt.title('PCA projection colored by cluster')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(outdir / 'pca_clusters.png', dpi=150)
    plt.close()

    # silhouette by k plot
    ks = sorted(scores.keys())
    vals = [scores[k] for k in ks]
    plt.figure()
    plt.plot(ks, vals, marker='o')
    plt.xlabel('k')
    plt.ylabel('silhouette score')
    plt.title('Silhouette score by k')
    plt.grid(True)
    plt.savefig(outdir / 'silhouette_by_k.png', dpi=150)
    plt.close()


def main():
    args = parse_args()
    inp = Path(args.input)
    if not inp.exists():
        print(f"Input file not found: {inp}", file=sys.stderr)
        sys.exit(2)

    print('Loading data (this may take a minute)')
    df = pd.read_csv(inp, dtype=str)
    # convert numeric candidate columns to numeric if present
    for col in [args.launch_speed, args.launch_angle, args.bat_speed]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # aggregate per player
    feats = aggregate_features(df, args.player_col, args.description_col,
                               args.launch_speed, args.launch_angle, args.bat_speed)

    # optional sampling for speed
    if args.sample_frac < 1.0:
        feats = feats.sample(frac=args.sample_frac, random_state=42)

    # drop players with very few swings
    feats = feats[feats['n_swings'] >= args.min_swings]

    # features to use (exclude n_swings)
    X_df = feats.drop(columns=['n_swings'])

    # if user provided a feature list, restrict to those (and ensure they exist)
    if args.features:
        requested = [c.strip() for c in args.features.split(',') if c.strip()]
        missing = [c for c in requested if c not in X_df.columns]
        if missing:
            print(f'Warning: requested features not found and will be ignored: {missing}')
        keep = [c for c in requested if c in X_df.columns]
        if keep:
            X_df = X_df[keep]
        else:
            print('No requested features available; using auto-detected features')
    print('Using features:', list(X_df.columns))

    # standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X_df.values)

    # PCA for visualization
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    # choose k (or use forced k)
    diagnostics = {}
    if args.k is not None:
        best_k = int(args.k)
        print(f'Forcing k = {best_k}')
        km = KMeans(n_clusters=best_k, random_state=42, n_init=args.n_init)
        labels = km.fit_predict(X)
    else:
        best_k, best_score, diagnostics = choose_k(X, args.min_k, args.max_k, n_init=args.n_init, imbalance_weight=args.imbalance_weight)
        print(f'Chosen k = {best_k} (combined score={best_score:.3f})')
        km = KMeans(n_clusters=best_k, random_state=42, n_init=args.n_init)
        labels = km.fit_predict(X)

    # save diagnostics and params
    outdir = Path(args.plots)
    outdir.mkdir(parents=True, exist_ok=True)
    plot_diagnostics(X_pca, labels, {k: diagnostics[k]['silhouette'] if k in diagnostics else None for k in diagnostics}, outdir)
    # write diagnostics.json
    import json
    params = {
        'min_k': args.min_k,
        'max_k': args.max_k,
        'forced_k': args.k,
        'min_swings': args.min_swings,
        'features': X_df.columns.tolist(),
        'n_init': args.n_init,
        'imbalance_weight': args.imbalance_weight,
    }
    with open(outdir / 'diagnostics.json', 'w') as fh:
        json.dump({'params': params, 'diagnostics': diagnostics}, fh, indent=2)

    # write per-player CSV
    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    result = feats.copy()
    result = result.drop(columns=['n_swings'])
    result['cluster'] = labels
    result.reset_index(inplace=True)
    result = result.rename(columns={'index': args.player_col})
    result.to_csv(out_csv, index=False)

    print(f'Wrote clusters for {len(result)} players to {out_csv}')
    print(f'Plots written to {outdir}')


if __name__ == '__main__':
    main()
