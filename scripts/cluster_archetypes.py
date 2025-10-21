#!/usr/bin/env python3
from pathlib import Path
import argparse, sys, re, warnings, json
import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


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
    p.add_argument('--imbalance-weight', type=float, default=0.5)
    p.add_argument('--silhouette-tol', type=float, default=0.03)
    p.add_argument('--prefer-larger-k', action='store_true', default=True)
    p.add_argument('--min-cluster-size', type=int, default=22)
    p.add_argument('--min-cluster-frac', type=float, default=0.03)
    p.add_argument('--sample-frac', type=float, default=1.0)
    return p.parse_args()


def infer_whiff_flag(desc):
    if pd.isna(desc):
        return False
    d = str(desc).lower()
    patterns = [r"swinging_strike", r"swinging strike", r"miss", r"whiff", r"swinging_strike_blocked"]
    return any(re.search(p, d) for p in patterns)


def aggregate_features(df, player_col, desc_col, launch_speed_col, launch_angle_col, bat_speed_col):
    if player_col not in df.columns:
        raise ValueError(f"Player column '{player_col}' not found in data. Columns: {list(df.columns)}")

    grouped = df.groupby(player_col)
    features = pd.DataFrame(index=grouped.size().index)
    features['n_swings'] = grouped.size()

    whiff_flag = pd.Series(False, index=df.index)
    if desc_col in df.columns:
        desc = df[desc_col].fillna('').astype(str).str.lower()
        whiff_flag = whiff_flag | desc.str.contains(r"swinging_strike|swinging strike|swinging_strike_blocked|whiff|swing and miss|miss")
        whiff_flag = whiff_flag | (desc.str.contains(r"swing") & desc.str.contains(r"miss|whiff"))
    if 'events' in df.columns:
        ev = df['events'].fillna('').astype(str).str.lower()
        whiff_flag = whiff_flag | ev.str.contains(r"swinging_strike|whiff|miss")
    if 'type' in df.columns and desc_col in df.columns:
        t = df['type'].fillna('').astype(str)
        desc = df[desc_col].fillna('').astype(str).str.lower()
        whiff_flag = whiff_flag | ((t == 'S') & desc.str.contains('swing'))

    try:
        whiff_counts = grouped.apply(lambda s: int(whiff_flag.loc[s.index].sum()))
        swings_counts = grouped.size()
    except Exception:
        whiff_counts = df.groupby(player_col).apply(lambda s: int(whiff_flag.loc[s.index].sum()))
        swings_counts = df.groupby(player_col).size()

    total_whiffs = whiff_counts.sum()
    total_swings = swings_counts.sum()
    p0 = float(total_whiffs) / float(total_swings) if total_swings > 0 else 0.0
    prior_n = 50.0
    alpha = p0 * prior_n
    beta = (1.0 - p0) * prior_n

    shrunk_whiff = (whiff_counts + alpha) / (swings_counts + prior_n)
    features['whiff_rate'] = shrunk_whiff.fillna(0.0)

    if launch_speed_col in df.columns:
        features['launch_speed_mean'] = grouped[launch_speed_col].mean()
        features['launch_speed_std'] = grouped[launch_speed_col].std().fillna(0.0)
        features['launch_speed_median'] = grouped[launch_speed_col].median()
        try:
            features['launch_speed_p95'] = grouped[launch_speed_col].quantile(0.95)
        except Exception:
            features['launch_speed_p95'] = grouped[launch_speed_col].apply(lambda s: s.quantile(0.95) if len(s) > 0 else 0)
        try:
            features['pct_hard_hit'] = grouped[launch_speed_col].apply(lambda s: (s >= 95).mean() if len(s.dropna())>0 else 0.0)
        except Exception:
            features['pct_hard_hit'] = 0.0

    if launch_angle_col in df.columns:
        features['launch_angle_mean'] = grouped[launch_angle_col].mean()
        features['launch_angle_std'] = grouped[launch_angle_col].std().fillna(0.0)
        def bb_mix(s):
            s = s.dropna()
            return pd.Series({
                'gb_pct': (s < 10).mean(),
                'ld_pct': ((s >= 10) & (s < 25)).mean(),
                'fb_pct': (s >= 25).mean(),
            })
        mix_df = grouped[launch_angle_col].apply(lambda s: bb_mix(s))
        if isinstance(mix_df, pd.Series) and isinstance(mix_df.index, pd.MultiIndex):
            mix_df = mix_df.unstack(level=-1)
        mix_df.index = mix_df.index.astype(str)
        features = features.join(mix_df)
    if bat_speed_col in df.columns:
        features['bat_speed_mean'] = grouped[bat_speed_col].mean()
        features['bat_speed_std'] = grouped[bat_speed_col].std().fillna(0.0)

    if desc_col in df.columns:
        contact_flags = df[desc_col].fillna('').str.lower().str.contains('contact|foul|hit|ball in play|in play')
        try:
            contact_counts = grouped.apply(lambda s: int(contact_flags.loc[s.index].sum()))
            total_contacts = contact_counts.sum()
            p0c = float(total_contacts) / float(total_swings) if total_swings > 0 else 0.0
            alpha_c = p0c * prior_n
            beta_c = (1.0 - p0c) * prior_n
            shrunk_contact = (contact_counts + alpha_c) / (swings_counts + prior_n)
            features['contact_rate'] = shrunk_contact.fillna(0.0)
        except Exception:
            features['contact_rate'] = grouped[desc_col].apply(lambda s: contact_flags.loc[s.index].mean())

    for col in ['bat_speed', 'swing_length', 'attack_angle', 'swing_path_tilt', 'intercept_ball_minus_batter_pos_x_inches', 'intercept_ball_minus_batter_pos_y_inches']:
        if col in df.columns:
            features[col + '_mean'] = grouped[col].mean()

    if 'attack_direction' in df.columns:
        def circ_mean_deg(s):
            s = s.dropna()
            if len(s) == 0:
                return 0.0
            r = np.deg2rad(s.values)
            mean_angle = np.arctan2(np.mean(np.sin(r)), np.mean(np.cos(r)))
            return np.rad2deg(mean_angle) % 360
        features['attack_direction_mean'] = grouped['attack_direction'].apply(circ_mean_deg)

    for pcol in ['whiff_rate', 'contact_rate']:
        if pcol in features.columns:
            vals = features[pcol].clip(0.0, 1.0).astype(float)
            features[pcol] = np.arcsin(np.sqrt(vals))

    return features.fillna(0.0)


def choose_k(X, min_k=2, max_k=10, n_init=10, imbalance_weight=0.0,
             silhouette_tol=0.02, prefer_larger_k=True,
             min_cluster_size=5, min_cluster_frac=0.01):
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

    
    candidates = []
    for k, s in stats.items():
        if s['silhouette'] < 0:
            continue
        if s['silhouette'] >= best_sil - silhouette_tol:
            candidates.append((k, s))

    if not candidates:
        
        valid = [(k, s) for k, s in stats.items() if s['combined'] > -1e8]
        if not valid:
            fallback_k = max(range(min_k, max_k + 1))
            return fallback_k, -1e9, stats
        best_k = max(valid, key=lambda kv: kv[1]['combined'])[0]
        return best_k, stats[best_k]['combined'], stats

    
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


def main():
    args = parse_args()
    inp = Path(args.input)
    if not inp.exists():
        sys.exit(f"Input file not found: {inp}")

    print('Loading data...')
    df = pd.read_csv(inp)
    
    extra_numeric = [
        args.launch_speed, args.launch_angle, args.bat_speed,
        'swing_length', 'attack_angle', 'swing_path_tilt', 'attack_direction',
        'intercept_ball_minus_batter_pos_x_inches', 'intercept_ball_minus_batter_pos_y_inches'
    ]
    for col in extra_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    feats = aggregate_features(df, args.player_col, args.description_col,
                               args.launch_speed, args.launch_angle, args.bat_speed)
    
    feats = feats[feats['n_swings'] >= args.min_swings]
    if args.sample_frac < 1.0:
        feats = feats.sample(frac=args.sample_frac, random_state=42)
    # Keep numeric features including sample-size columns (n_swings) so they can be used if desired.
    X_df = feats.copy()
    X_df = X_df.select_dtypes(include=[np.number])

    if args.player_col in X_df.columns:
        X_df = X_df.drop(columns=[args.player_col])

    
    # Preserve contact_rate by default (do not drop it automatically). Optionally scale whiff_rate slightly
    if 'contact_rate' in X_df.columns:
        try:
            if 'whiff_rate' in X_df.columns:
                X_df['whiff_rate'] = X_df['whiff_rate'] * 0.2
        except Exception:
            pass
    
    # Protect a canonical set of features from being dropped by variance/correlation heuristics.
    CANONICAL_FEATURES = [
        'n_swings', 'whiff_rate',
        'launch_speed_mean', 'launch_speed_std', 'launch_speed_median', 'launch_speed_p95', 'pct_hard_hit',
        'launch_angle_mean', 'launch_angle_std', 'gb_pct', 'ld_pct', 'fb_pct',
        'bat_speed_mean', 'bat_speed_std', 'contact_rate',
        'swing_length_mean', 'attack_angle_mean', 'swing_path_tilt_mean',
        'intercept_ball_minus_batter_pos_x_inches_mean', 'intercept_ball_minus_batter_pos_y_inches_mean',
        'attack_direction_mean'
    ]
    canonical_present = [c for c in CANONICAL_FEATURES if c in X_df.columns]

    low_var = X_df.std() < 1e-3
    # Only drop low-variance columns that are NOT in the canonical set
    drop_low_cols = [c for c, v in low_var.items() if v and c not in canonical_present]
    if drop_low_cols:
        print("Dropping near-constant features:", drop_low_cols)
        X_df = X_df.drop(columns=drop_low_cols)
    
    core_cols = [c for c in ['launch_speed_mean','bat_speed_mean','launch_angle_mean'] if c in X_df.columns]
    if core_cols:
        X_df = X_df[X_df[core_cols].sum(axis=1) != 0]
    
    X_df = X_df.dropna()
    # Log-transform count-like columns (including n_swings) so counts are on a comparable scale
    count_cols = [c for c in X_df.columns if c.lower().startswith('n_') or c.lower().endswith('_count') or c.lower() == 'n_swings']
    for c in count_cols:
        try:
            X_df[c] = np.log1p(X_df[c].astype(float).fillna(0))
        except Exception:
            pass

    for col in X_df.columns:
        lo, hi = X_df[col].quantile([0.01, 0.99])
        X_df[col] = X_df[col].clip(lo, hi)
    corr = X_df.corr().abs()
    to_drop = set()
    cols = list(corr.columns)
    for i, c1 in enumerate(cols):
        for c2 in cols[i+1:]:
            try:
                if corr.loc[c1, c2] > 0.92:
                    # Prefer to drop the non-canonical column when a canonical and non-canonical column are highly correlated.
                    if c2 in canonical_present and c1 not in canonical_present:
                        to_drop.add(c1)
                    elif c1 in canonical_present and c2 not in canonical_present:
                        to_drop.add(c2)
                    else:
                        # If neither or both are canonical, drop c2 (original behavior) but ensure canonical columns are not removed below.
                        to_drop.add(c2)
            except Exception:
                continue
    # Ensure we do not drop canonical features
    to_drop = [c for c in to_drop if c not in canonical_present]
    if to_drop:
        print("Dropping highly correlated:", sorted(to_drop))
        X_df = X_df.drop(columns=sorted(to_drop))
    mad = X_df.apply(lambda x: (x - x.median()).abs().median())
    
    stds = X_df.std()
    mad2 = mad.copy()
    for c in mad2.index:
        if mad2[c] == 0 or pd.isna(mad2[c]):
            if stds[c] > 0:
                mad2[c] = stds[c]
            else:
                mad2[c] = 1.0
    k_mad = 5.0
    med = X_df.median()
    for col in X_df.columns:
        lo = med[col] - k_mad * mad2[col]
        hi = med[col] + k_mad * mad2[col]
        X_df[col] = X_df[col].clip(lo, hi)
    scaler = RobustScaler()
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
    out_feats = []
    if 'n_swings' in result.columns:
        out_feats.append('n_swings')
    out_feats += [c for c in X_df.columns if c in result.columns]
    result = result[out_feats]
    result['cluster'] = labels
    result.reset_index(inplace=True)
    result = result.rename(columns={'index': args.player_col})
    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_csv, index=False)

    print(f"Wrote clusters for {len(result)} players to {out_csv}")
    print(f"Plots written to {outdir}")

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
