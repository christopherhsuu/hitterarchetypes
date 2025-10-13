#!/usr/bin/env python3
"""
Cluster players into archetypes using per-swing Statcast data.

Enhancements:
- Cleans zero / constant features before clustering
- Removes player ID from feature matrix
- Filters all-zero or missing rows
- Preserves your silhouette diagnostics and PCA plots
"""

from pathlib import Path
import argparse, sys, re, warnings, json
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


# ---------------- Argument parsing ----------------
def parse_args():
    p = argparse.ArgumentParser(description='Cluster players into archetypes from Statcast swings')
    p.add_argument('--input', '-i', default='data/raw/2025_swings.csv')
    p.add_argument('--player-col', default='batter')
    p.add_argument('--launch-speed', default='launch_speed')
    p.add_argument('--launch-angle', default='launch_angle')
    p.add_argument('--bat-speed', default='bat_speed')
    p.add_argument('--description-col', default='description')
    p.add_argument('--out', default='data/player_archetypes.csv')
    p.add_argument('--plots', default='out/cluster_plots')
    p.add_argument('--min-k', type=int, default=3)
    p.add_argument('--max-k', type=int, default=8)
    p.add_argument('--k', type=int, default=None)
    p.add_argument('--min-swings', type=int, default=15)
    p.add_argument('--features', default=None)
    p.add_argument('--n-init', type=int, default=10)
    p.add_argument('--imbalance-weight', type=float, default=0.5,
                   help='Penalty weight for cluster size imbalance (higher penalizes uneven sizes)')
    # NEW:
    p.add_argument('--silhouette-tol', type=float, default=0.03,
                   help='Accept any k with silhouette >= best - tol (e.g., 0.03 ≈ within 3 points)')
    p.add_argument('--prefer-larger-k', action='store_true', default=True,
                   help='If multiple ks are within tolerance, pick the largest k')
    p.add_argument('--min-cluster-size', type=int, default=22,
                   help='Reject ks where any cluster has fewer than this many players')
    p.add_argument('--min-cluster-frac', type=float, default=0.03,
                   help='Also reject ks where any cluster has fewer than this fraction of players')
    p.add_argument('--sample-frac', type=float, default=1.0)
    return p.parse_args()



# ---------------- Helper functions ----------------
def infer_whiff_flag(desc):
    if pd.isna(desc): return False
    d = str(desc).lower()
    patterns = [r"swinging_strike", r"swinging strike", r"miss", r"whiff", r"swinging_strike_blocked"]
    return any(re.search(p, d) for p in patterns)


def aggregate_features(df, player_col, desc_col, launch_speed_col, launch_angle_col, bat_speed_col):
    if player_col not in df.columns:
        raise ValueError(f"Player column '{player_col}' not found in data. Columns: {list(df.columns)}")

    grouped = df.groupby(player_col)
    features = pd.DataFrame(index=grouped.size().index)
    features['n_swings'] = grouped.size()

    # Robust whiff/miss detection combining description, events, and type columns
    whiff_flag = pd.Series(False, index=df.index)
    if desc_col in df.columns:
        desc = df[desc_col].fillna('').astype(str).str.lower()
        whiff_flag = whiff_flag | desc.str.contains(r"swinging_strike|swinging strike|swinging_strike_blocked|whiff|swing and miss|miss")
        # capture patterns like 'swing' + 'miss'
        whiff_flag = whiff_flag | (desc.str.contains(r"swing") & desc.str.contains(r"miss|whiff"))
    if 'events' in df.columns:
        ev = df['events'].fillna('').astype(str).str.lower()
        whiff_flag = whiff_flag | ev.str.contains(r"swinging_strike|whiff|miss")
    if 'type' in df.columns and desc_col in df.columns:
        t = df['type'].fillna('').astype(str)
        desc = df[desc_col].fillna('').astype(str).str.lower()
        whiff_flag = whiff_flag | ((t == 'S') & desc.str.contains('swing'))

    try:
        features['whiff_rate'] = grouped.apply(lambda s: whiff_flag.loc[s.index].mean())
    except Exception:
        features['whiff_rate'] = 0.0

    # If whiff_rate is all zero (no descriptions/events present), compute a fallback 'miss_rate'
    if features['whiff_rate'].max() == 0:
        miss_flag = pd.Series(False, index=df.index)
        if desc_col in df.columns:
            desc = df[desc_col].fillna('').astype(str).str.lower()
            miss_flag = miss_flag | desc.str.contains(r"miss|whiff|swing and miss")
        if 'events' in df.columns:
            ev = df['events'].fillna('').astype(str).str.lower()
            miss_flag = miss_flag | ev.str.contains(r"miss|whiff")
        try:
            features['whiff_rate'] = grouped.apply(lambda s: miss_flag.loc[s.index].mean())
        except Exception:
            features['whiff_rate'] = 0.0

    # Launch metrics and additional EV stats
    if launch_speed_col in df.columns:
        ls = df[launch_speed_col]
        features['launch_speed_mean'] = grouped[launch_speed_col].mean()
        features['launch_speed_std'] = grouped[launch_speed_col].std().fillna(0.0)
        # Advanced exit-velocity features
        features['launch_speed_median'] = grouped[launch_speed_col].median()
        try:
            features['launch_speed_p95'] = grouped[launch_speed_col].quantile(0.95)
        except Exception:
            features['launch_speed_p95'] = grouped[launch_speed_col].apply(lambda s: s.quantile(0.95) if len(s) > 0 else 0)
        # percent hard-hit (>=95 mph) — guard against missing values
        try:
            features['pct_hard_hit'] = grouped[launch_speed_col].apply(lambda s: (s >= 95).mean() if len(s.dropna())>0 else 0.0)
        except Exception:
            features['pct_hard_hit'] = 0.0

    if launch_angle_col in df.columns:
        features['launch_angle_mean'] = grouped[launch_angle_col].mean()
        features['launch_angle_std'] = grouped[launch_angle_col].std().fillna(0.0)
        # Batted-ball distribution mix: ground/line/flat (gb/ld/fb)
        def bb_mix(s):
            s = s.dropna()
            return pd.Series({
                'gb_pct': (s < 10).mean(),
                'ld_pct': ((s >= 10) & (s < 25)).mean(),
                'fb_pct': (s >= 25).mean(),
            })
        try:
            mix = df[[player_col, launch_angle_col]].groupby(player_col)[launch_angle_col].apply(bb_mix)
            # join mix (it returns a DataFrame with multiindex sometimes)
            features = features.join(mix)
        except Exception:
            # fallback: compute per-player via apply
            mix2 = grouped[launch_angle_col].apply(bb_mix)
            features = features.join(mix2)
    if bat_speed_col in df.columns:
        features['bat_speed_mean'] = grouped[bat_speed_col].mean()
        features['bat_speed_std'] = grouped[bat_speed_col].std().fillna(0.0)

    # Contact rate heuristic
    if desc_col in df.columns:
        contact_flags = df[desc_col].fillna('').str.lower().str.contains('contact|foul|hit|ball in play|in play')
        features['contact_rate'] = grouped[desc_col].apply(lambda s: contact_flags.loc[s.index].mean())

    # Fill NaNs and return
    return features.fillna(0.0)


def choose_k(X, min_k=2, max_k=10, n_init=10, imbalance_weight=0.0,
             silhouette_tol=0.02, prefer_larger_k=True,
             min_cluster_size=5, min_cluster_frac=0.01):
    """Pick k via silhouette with tolerance window and (optional) size/imbalance constraints."""
    best_sil = -1e9
    stats = {}

    N = X.shape[0]
    for k in range(min_k, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=n_init)
        labels = km.fit_predict(X)

        # counts & size checks
        counts = np.array([(labels == i).sum() for i in range(k)], dtype=int)
        min_ok = (counts.min() >= min_cluster_size) and (counts.min() >= int(np.floor(min_cluster_frac * N)))
        if not min_ok:
            stats[k] = {'silhouette': -1.0, 'imbalance': None, 'combined': -1e9, 'counts': counts.tolist()}
            continue

        sil = silhouette_score(X, labels)
        cv = counts.std() / counts.mean() if counts.mean() > 0 else 1.0
        combined = sil - imbalance_weight * cv

        stats[k] = {'silhouette': float(sil), 'imbalance': float(cv),
                    'combined': float(combined), 'counts': counts.tolist()}

        if sil > best_sil:
            best_sil = sil

    # candidates within tolerance of best silhouette
    candidates = []
    for k, s in stats.items():
        if s['silhouette'] < 0:
            continue
        if s['silhouette'] >= best_sil - silhouette_tol:
            candidates.append((k, s))

    if not candidates:
        # fall back: pick argmax combined among valid entries
        valid = [(k, s) for k, s in stats.items() if s['combined'] > -1e8]
        if not valid:
            # as a last resort, pick the max k tried
            fallback_k = max(range(min_k, max_k + 1))
            return fallback_k, -1e9, stats
        best_k = max(valid, key=lambda kv: kv[1]['combined'])[0]
        return best_k, stats[best_k]['combined'], stats

    # within tolerance: either largest k or best combined (sil - imbalance*cv)
    if prefer_larger_k:
        best_k = max(candidates, key=lambda kv: kv[0])[0]
    else:
        best_k = max(candidates, key=lambda kv: kv[1]['combined'])[0]

    return best_k, stats[best_k]['combined'], stats


def plot_diagnostics(X_pca, labels, scores, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    dfp = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    dfp['cluster'] = labels.astype(str)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=dfp, x='PC1', y='PC2', hue='cluster', palette='tab10', s=30, linewidth=0)
    plt.title('PCA projection colored by cluster')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(outdir / 'pca_clusters.png', dpi=150)
    plt.close()

    if scores:
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


# ---------------- Main ----------------
def main():
    args = parse_args()
    inp = Path(args.input)
    if not inp.exists():
        sys.exit(f"Input file not found: {inp}")

    print('Loading data...')
    df = pd.read_csv(inp)
    for col in [args.launch_speed, args.launch_angle, args.bat_speed]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    feats = aggregate_features(df, args.player_col, args.description_col,
                               args.launch_speed, args.launch_angle, args.bat_speed)

    # Drop players with few swings
    feats = feats[feats['n_swings'] >= args.min_swings]
    if args.sample_frac < 1.0:
        feats = feats.sample(frac=args.sample_frac, random_state=42)

    # ---- Clean features before scaling ----
    X_df = feats.drop(columns=['n_swings'], errors='ignore')
    X_df = X_df.select_dtypes(include=[np.number])

    if args.player_col in X_df.columns:
        X_df = X_df.drop(columns=[args.player_col])

    # Drop near-constant features
    low_var = X_df.std() < 1e-3
    if low_var.any():
        print("Dropping near-constant features:", list(X_df.columns[low_var]))
        X_df = X_df.loc[:, ~low_var]

    # Drop rows with all-zero core stats
    core_cols = [c for c in ['launch_speed_mean','bat_speed_mean','launch_angle_mean'] if c in X_df.columns]
    if core_cols:
        X_df = X_df[X_df[core_cols].sum(axis=1) != 0]

    # Drop NaN rows
    X_df = X_df.dropna()

    # ---- Winsorize / clip outliers to 1st-99th percentiles ----
    for col in X_df.columns:
        lo, hi = X_df[col].quantile([0.01, 0.99])
        X_df[col] = X_df[col].clip(lo, hi)

    # ---- Drop highly correlated features (> 0.92) ----
    corr = X_df.corr().abs()
    to_drop = set()
    cols = list(corr.columns)
    for i, c1 in enumerate(cols):
        for c2 in cols[i+1:]:
            try:
                if corr.loc[c1, c2] > 0.92:
                    to_drop.add(c2)
            except Exception:
                continue
    if to_drop:
        print("Dropping highly correlated:", sorted(to_drop))
        X_df = X_df.drop(columns=sorted(to_drop))

    # ---- Standardize ----
    scaler = StandardScaler()
    X = scaler.fit_transform(X_df.values)

    # ---- PCA ----
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    # ---- Clustering ----
    if args.k is not None:
        best_k = int(args.k)
        print(f"Forcing k = {best_k}")
        km = KMeans(n_clusters=best_k, random_state=42, n_init=args.n_init)
        labels = km.fit_predict(X)
        diagnostics = {}
        best_score = 0.0
    else:
        best_k, best_score, diagnostics = choose_k(
            X,
            min_k=args.min_k,
            max_k=args.max_k,
            n_init=args.n_init,
            imbalance_weight=args.imbalance_weight,
            silhouette_tol=args.silhouette_tol,
            prefer_larger_k=args.prefer_larger_k,
            min_cluster_size=args.min_cluster_size,
            min_cluster_frac=args.min_cluster_frac,
        )

    # fallback if choose_k couldn't find valid k
    if best_k is None:
        print("⚠️ No valid k found — defaulting to k=3")
        best_k = 3

    print(f"Chosen k = {best_k} "
          f"(sil≈{diagnostics.get(best_k, {}).get('silhouette', float('nan')):.3f}, "
          f"combined={best_score:.3f})")

    km = KMeans(n_clusters=best_k, random_state=42, n_init=args.n_init)
    labels = km.fit_predict(X)


    # ---- Output ----
    outdir = Path(args.plots)
    plot_diagnostics(X_pca, labels,
                     {k: diagnostics[k]['silhouette'] for k in diagnostics} if diagnostics else {},
                     outdir)

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

    result = feats.loc[X_df.index].copy()
    result['cluster'] = labels
    result.reset_index(inplace=True)
    result = result.rename(columns={'index': args.player_col})
    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_csv, index=False)

    print(f"Wrote clusters for {len(result)} players to {out_csv}")
    print(f"Plots written to {outdir}")

    # ---- Post-run summaries: centroids (inverse-transformed), sizes, and top diffs ----
    try:
        centroids = pd.DataFrame(
            scaler.inverse_transform(km.cluster_centers_), columns=X_df.columns
        )
        counts = pd.Series(labels).value_counts().sort_index()
        print("Cluster sizes:", counts.to_dict())
        print(centroids.round(2))

        global_mean, global_std = X_df.mean(), X_df.std().replace(0, 1.0)
        z = (centroids - global_mean) / global_std
        for i in range(km.n_clusters):
            top = z.iloc[i].abs().sort_values(ascending=False).head(4)
            print(f"\ncluster {i} top diffs:")
            print(top.round(2).to_string())
    except Exception as e:
        print('Post-run summary failed:', e)


if __name__ == '__main__':
    main()
