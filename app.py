import streamlit as st
# Guarded imports for optional plotting libraries
HAS_PLOTLY = False
px = None
go = None
try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False
from pathlib import Path
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# repository root resolved relative to this file — define at module scope so deployments that change cwd still work
ROOT = Path(__file__).resolve().parent


# Human-friendly feature descriptions for the Feature Index shown with the centroid heatmap
FEATURE_DESCRIPTIONS = {
    'launch_speed_mean': 'Average exit velocity (mph) of batted balls.',
    'launch_speed_std': 'Std dev of exit velocity; hit-to-hit consistency of speed.',
    'launch_speed_median': 'Median exit velocity (mph).',
    'launch_speed_p95': '95th percentile exit velocity — how often they hit very hard.',
    'pct_hard_hit': 'Proportion of batted balls with hard-hit threshold (empirical).',
    'launch_angle_mean': 'Average launch angle (degrees) of batted balls.',
    'launch_angle_std': 'Std dev of launch angle; consistency of launch angle.',
    'bat_speed_mean': 'Average bat speed (mph) at impact.',
    'bat_speed_std': 'Std dev of bat speed; consistency of swing speed.',
    'swing_length_mean': 'Mean swing length (units depend on pipeline — relative arc length).',
    'attack_angle_mean': 'Mean attack angle (degrees) — bat path angle relative to the ground at contact.',
    'swing_path_tilt_mean': 'Tilt of the swing path (degrees) — how tilted the bat path is during swing.',
    'intercept_ball_minus_batter_pos_x_inches_mean': 'Mean horizontal offset (in) between ball and batter at intercept — negative/positive indicate side offsets.',
    'intercept_ball_minus_batter_pos_y_inches_mean': 'Mean vertical offset (in) between ball and batter at intercept — how early/late contact tends to be.',
    'attack_direction_mean': 'Average attack direction (degrees) — direction of bat travel at impact (circular).',
    'n_swings': 'Number of swings used to compute these per-player aggregates (sample size).'
}

# Friendly feature groups: map a readable label to a list of raw feature columns.
# The UI will let the user pick which raw column to show under each friendly label.
FEATURE_GROUPS = {
    'Launch speed': ['launch_speed_mean', 'launch_speed_median', 'launch_speed_std', 'launch_speed_p95'],
    'Launch angle': ['launch_angle_mean', 'launch_angle_std'],
    'Bat speed': ['bat_speed_mean', 'bat_speed_std'],
    'Swing': ['swing_length_mean', 'swing_path_tilt_mean'],
    'Attack': ['attack_angle_mean', 'attack_direction_mean'],
    'Intercept (x)': ['intercept_ball_minus_batter_pos_x_inches_mean'],
    'Intercept (y)': ['intercept_ball_minus_batter_pos_y_inches_mean'],
    'Hard contact': ['pct_hard_hit'],
    'Sample size': ['n_swings']
}


def get_display_name_series(df):
    """Return a safe series to use as the player display name.
    Tries common columns; falls back to the first column (id).
    """
    candidates = ['name_display', 'name', 'full_name', 'display_name']
    for c in candidates:
        if c in df.columns:
            return df[c].fillna('').astype(str)
    # try any column with 'name' in it
    for c in df.columns:
        if 'name' in c.lower():
            return df[c].fillna('').astype(str)
    # fallback to first column
    first = df.columns[0]
    return df[first].fillna('').astype(str)


def merge_candidate_names(main_df):
    """Attempt to enrich main_df with name columns from candidate files.
    Returns a new DataFrame (or the original if no merge succeeded).
    """
    # Prefer the raw mapping file first. Use module-level ROOT to resolve paths.
    preferred = ROOT.joinpath('data/raw/unique_batters_with_names.csv')
    idcol_main = main_df.columns[0]
    if preferred.exists():
        try:
            nm = pd.read_csv(preferred, dtype=str)
            # find id column in mapping (common names)
            id_candidates = [c for c in nm.columns if c.lower() in ('batter', 'playerid', 'mlbam', 'id', 'key')]
            if not id_candidates:
                # fallback: pick the column whose values intersect the main id column
                for c in nm.columns:
                    if nm[c].dropna().isin(main_df[idcol_main].astype(str).unique()).any():
                        id_candidates = [c]
                        break
            # find name-like column
            name_candidates = [c for c in nm.columns if 'name' in c.lower() or 'full' in c.lower() or 'display' in c.lower()]
            if not name_candidates and nm.shape[1] >= 2:
                name_candidates = [nm.columns[1]]
            if id_candidates and name_candidates:
                idc = id_candidates[0]
                namec = name_candidates[0]
                # build mapping dict (string keys)
                map_series = nm[[idc, namec]].dropna()
                map_series[idc] = map_series[idc].astype(str).str.strip()
                map_series[namec] = map_series[namec].astype(str).str.strip()
                mapping = map_series.set_index(idc)[namec].to_dict()
                # create a name column by mapping the main id column
                # DO NOT fill missing names with the raw id (this caused numeric ids to appear in the UI)
                main_df = main_df.copy()
                main_df['name'] = main_df[idcol_main].astype(str).str.strip().map(mapping)
                # leave unmatched names as NaN/empty; display logic will prefer human names and avoid showing raw ids
                return main_df
        except Exception:
            pass

    # Fallback: try other candidate files using the previous heuristic
    candidates = [ROOT.joinpath('data/unique_batters_with_names.csv'), ROOT.joinpath('data/raw/unique_batters.csv'), ROOT.joinpath('data/unique_batters.csv')]
    for pth in candidates:
        if not pth.exists():
            continue
        try:
            nm = pd.read_csv(pth, dtype=str)
        except Exception:
            continue
        id_candidates = [c for c in nm.columns if c.lower() in ('batter', 'playerid', 'mlbam', 'id', 'key')]
        name_candidates = [c for c in nm.columns if 'name' in c.lower() or 'full' in c.lower() or 'display' in c.lower()]
        if not id_candidates:
            for c in nm.columns:
                if nm[c].dropna().isin(main_df[idcol_main].astype(str).unique()).any():
                    id_candidates = [c]
                    break
        if not name_candidates and nm.shape[1] >= 2:
            name_candidates = [nm.columns[1]]
        if not id_candidates or not name_candidates:
            continue
        idc = id_candidates[0]
        namec = name_candidates[0]
        nm_tmp = nm[[idc, namec]].copy()
        nm_tmp['__merge_key'] = nm_tmp[idc].apply(lambda x: str(x).strip() if pd.notna(x) else '')
        main_tmp = main_df.copy()
        main_tmp['__merge_key'] = main_tmp[idcol_main].apply(lambda x: str(x).strip() if pd.notna(x) else '')
        merged = main_tmp.merge(nm_tmp[['__merge_key', namec]].rename(columns={namec: 'name'}), on='__merge_key', how='left')
        merged = merged.drop(columns=['__merge_key'])
        if merged['name'].notna().any():
            return merged
    return main_df


def summarize_cluster_features(df, features_list):
    """Compute cluster centroids and z-score differences vs overall mean.
    Returns a DataFrame of centroids and a z-score DataFrame (clusters x features).
    """
    feats = [c for c in features_list if c in df.columns]
    if not feats:
        return None, None
    centroids = df.set_index('cluster')[feats].groupby(level=0).mean()
    # overall mean/std across clusters (for z-scoring centroids)
    overall_mean = centroids.mean(axis=0)
    overall_std = centroids.std(axis=0).replace(0, 1)
    z = (centroids - overall_mean) / overall_std
    return centroids, z


def cluster_representatives(df, cluster_label, features_list, n_closest=3, n_farthest=2):
    """Return representative players for a cluster: closest to centroid and farthest (contrast).
    Uses z-scored features computed across players (not clusters) to compute distances.
    """
    feats = [c for c in features_list if c in df.columns]
    if not feats:
        return [], []
    # compute per-player z using overall mean/std across all players
    X = df[feats].fillna(0).astype(float)
    mean = X.mean(axis=0)
    std = X.std(axis=0).replace(0, 1)
    Xz = (X - mean) / std
    # centroid for this cluster in z-space
    centroid = Xz[df['cluster'].astype(str) == str(cluster_label)].mean(axis=0)
    # compute Euclidean distance
    dists = Xz.subtract(centroid, axis=1).pow(2).sum(axis=1).pow(0.5)
    # select indices from df matching cluster
    mask = df['cluster'].astype(str) == str(cluster_label)
    if mask.sum() == 0:
        return [], []
    cluster_idx = dists[mask].sort_values()
    closest_idx = cluster_idx.head(n_closest).index.tolist()
    far_idx = cluster_idx.tail(n_farthest).index.tolist()
    # return display names for these indices (prefer name_display if available)
    if 'name_display' in df.columns:
        names = df['name_display'].fillna('').astype(str)
    else:
        names = get_display_name_series(df)
    closest = [(i, names.loc[i]) for i in closest_idx]
    far = [(i, names.loc[i]) for i in far_idx]
    return closest, far


def render_cluster_pca_overview(df, features, use_plotly=HAS_PLOTLY, height=600):
    """Render a clear PCA overview of all players colored by cluster.
    Shows one trace per cluster plus centroid markers and returns the PCA embedding and centroids.
    """
    try:
        feats = [c for c in features if c in df.columns]
        # If automatic feature selection fails, fall back to a canonical small set
        canonical = ['launch_speed_mean', 'launch_angle_mean', 'bat_speed_mean', 'pct_hard_hit', 'n_swings']
        fallback = [c for c in canonical if c in df.columns]
        if len(feats) < 2:
            feats = fallback
        if len(feats) < 2 or df.shape[0] < 2:
            st.info('Not enough data to render cluster overview')
            return None, None

        # prepare feature matrix
        feat_df = df[feats].select_dtypes(include=[np.number]).fillna(0).copy()
        # basic preprocessing: log counts, winsorize, robust scale (keep consistent with PCA block)
        angle_cols = [c for c in feat_df.columns if ('angle' in c.lower() or 'direction' in c.lower()) and c.lower().endswith('_mean')]
        for c in angle_cols:
            rad = np.deg2rad(feat_df[c].astype(float).fillna(0).values)
            feat_df[c + '_sin'] = np.sin(rad)
            feat_df[c + '_cos'] = np.cos(rad)
            feat_df.drop(columns=[c], inplace=True)
        count_cols = [c for c in feat_df.columns if c.lower().startswith('n_') or c.lower().endswith('_count') or c.lower() == 'n_swings']
        for c in count_cols:
            feat_df[c] = np.log1p(feat_df[c].astype(float).fillna(0))
        for c in feat_df.columns:
            lo = feat_df[c].quantile(0.01)
            hi = feat_df[c].quantile(0.99)
            if pd.notna(lo) and pd.notna(hi) and lo < hi:
                feat_df[c] = feat_df[c].clip(lower=lo, upper=hi)
        from sklearn.preprocessing import RobustScaler
        from sklearn.decomposition import PCA

        scaler = RobustScaler()
        Xs = scaler.fit_transform(feat_df.values)
        pca = PCA(n_components=2)
        Xp = pca.fit_transform(Xs)
        emb = pd.DataFrame(Xp, columns=['PC1', 'PC2'], index=df.index)
        emb['cluster'] = df['cluster'].astype(str).values
        # display with Plotly as one trace per cluster for clarity
        if use_plotly and px is not None:
            fig = go.Figure()
            clusters_unique = sorted(emb['cluster'].unique(), key=lambda x: (int(x) if str(x).isdigit() else x))
            palette = px.colors.qualitative.Safe if hasattr(px.colors.qualitative, 'Safe') else px.colors.qualitative.Plotly
            # ensure enough colors
            colors = palette * ((len(clusters_unique) // len(palette)) + 1)
            for i, cl in enumerate(clusters_unique):
                sub = emb[emb['cluster'] == cl]
                # hover info
                # build a safe names Series (never None) aligned to sub.index
                idcol = df.columns[0]
                if 'name_display' in df.columns:
                    names = df.loc[sub.index, 'name_display'].astype(object).fillna('').astype(str)
                elif 'name' in df.columns:
                    names = df.loc[sub.index, 'name'].astype(object).fillna('').astype(str)
                else:
                    names = pd.Series([''] * len(sub), index=sub.index, dtype=str)
                ids = df.loc[sub.index, idcol].astype(str)
                hover = list(zip(names.tolist(), ids.tolist()))
                fig.add_trace(go.Scattergl(x=sub['PC1'], y=sub['PC2'], mode='markers', name=f'Cluster {cl}',
                                           marker=dict(color=colors[i], size=6, line=dict(width=0.5, color='black')),
                                           hovertemplate='<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}',
                                           text=[f"{t[0]} ({t[1]})" for t in hover]))
            # centroids in PCA space
            centroids = emb.groupby('cluster')[['PC1','PC2']].mean()
            fig.add_trace(go.Scattergl(x=centroids['PC1'], y=centroids['PC2'], mode='markers+text',
                                       marker=dict(symbol='x-open', color='black', size=14),
                                       text=[f'Centroid {c}' for c in centroids.index], textposition='top center', name='Centroids'))
            fig.update_layout(title='Cluster PCA overview', xaxis_title='PC1', yaxis_title='PC2', height=height)
            st.plotly_chart(fig, use_container_width=True)
        else:
            # matplotlib fallback: plot each cluster separately with edgecolors for clarity
            import matplotlib.pyplot as _plt
            fig, ax = _plt.subplots(figsize=(10, 6))
            clusters_unique = sorted(emb['cluster'].unique(), key=lambda x: (int(x) if str(x).isdigit() else x))
            palette = sns.color_palette(n_colors=len(clusters_unique))
            for i, cl in enumerate(clusters_unique):
                sub = emb[emb['cluster'] == cl]
                ax.scatter(sub['PC1'], sub['PC2'], s=30, color=palette[i], label=f'Cluster {cl}', edgecolors='black', linewidths=0.4, alpha=0.85)
            centroids = emb.groupby('cluster')[['PC1','PC2']].mean()
            ax.scatter(centroids['PC1'], centroids['PC2'], s=150, c='black', marker='X')
            ax.legend()
            ax.set_title('Cluster PCA overview')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            st.pyplot(fig)

        return emb, centroids
    except Exception as e:
        st.write('Cluster overview failed:', e)
        return None, None


def generate_cluster_descriptions(df, features, top_n=3):
    """Return a dict of textual descriptions for each cluster using z-scored centroids."""
    feats = [c for c in features if c in df.columns]
    if not feats:
        return {}
    centroids = df.set_index('cluster')[feats].groupby(level=0).mean()
    overall_mean = centroids.mean(axis=0)
    overall_std = centroids.std(axis=0).replace(0, 1)
    z = (centroids - overall_mean) / overall_std
    descriptions = {}
    for cl in centroids.index:
        row = z.loc[cl].sort_values(ascending=False)
        top_pos = row[row > 0].head(top_n).index.tolist()
        top_neg = row[row < 0].head(top_n).index.tolist()
        pos_text = ', '.join(top_pos) if top_pos else 'no strong positive distinguishing features'
        neg_text = ', '.join(top_neg) if top_neg else 'no strong negative distinguishing features'
        desc = f"Cluster {cl}: Players in this cluster tend to have higher-than-average {pos_text} and lower-than-average {neg_text}. "
        desc += "In short, membership is driven by these feature differences — players with similar values across these features are grouped together."
        descriptions[str(cl)] = desc
    return descriptions


@st.cache_data
def load_player_archetypes(path='data/player_archetypes.csv'):
    p = Path(path)
    if not p.exists():
        # Diagnostics to help deployed environments: print working dir and data folder contents
        try:
            print('DEBUG: load_player_archetypes - cwd=', Path.cwd())
            print('DEBUG: checking path ->', p, 'resolved ->', p.resolve())
            data_dir = Path('data')
            if data_dir.exists():
                print('DEBUG: data/ directory listing:')
                for f in sorted([str(x) for x in data_dir.iterdir()] )[:200]:
                    print('DEBUG:  ', f)
            else:
                print('DEBUG: data/ directory does not exist')
        except Exception as _:
            print('DEBUG: diagnostics failed', _)
        return None
    df = pd.read_csv(p)
    return df


@st.cache_data
def load_diagnostics(path='out/cluster_plots/diagnostics.json'):
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def main():
    st.set_page_config(layout='wide', page_title='Cluster Archetypes')
    st.title('Cluster Archetypes Showcase')


    df = load_player_archetypes()
    diag = load_diagnostics()

    if df is None:
        st.error('Could not find data/player_archetypes.csv')
        return

    # Render the clustering overview first (top of page)
    # Choose a conservative feature set for the overview: use diagnostic features if available, otherwise numeric columns
    diag_feats = diag.get('params', {}).get('features') if diag else None
    overview_features = [f for f in (diag_feats or df.columns.tolist()) if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
    if overview_features:
        st.header('Cluster overview')
        emb, centroids = render_cluster_pca_overview(df, overview_features, use_plotly=HAS_PLOTLY, height=500)
    else:
        st.info('No numeric features available for cluster overview')

    # try to merge external name mappings early so search uses human names
    df = merge_candidate_names(df)

    # Normalize any human name columns on load so selection helpers get canonical strings
    # If a 'name' column exists, normalize it and keep missing as empty string (do not fill with ids)
    if 'name' in df.columns:
        df['name'] = df['name'].fillna('').astype(str).str.strip()

    st.sidebar.header('Options')
    clusters = sorted(df['cluster'].astype(str).unique(), key=lambda x: (int(x) if x.isdigit() else x))
    sel_cluster = st.sidebar.selectbox('Cluster', ['All'] + clusters)
    show_pca = st.sidebar.checkbox('Show PCA', value=True)
    show_centroid = st.sidebar.checkbox('Show centroid heatmap', value=True)
    show_scatter = st.sidebar.checkbox('Show feature scatter', value=True)

    raw_feats = diag.get('params', {}).get('features') if diag else None
    if raw_feats:
        features = [f for f in raw_feats if f in df.columns]
    else:
        features = [c for c in df.columns if c not in ('cluster','name') and pd.api.types.is_numeric_dtype(df[c])]

    st.sidebar.markdown('Features used:')
    st.sidebar.write(features)
    # Friendly labels selector: let the user pick which raw column appears under each friendly name
    st.sidebar.markdown('### Friendly feature labels')
    chosen_mapping = {}
    for label, candidates in FEATURE_GROUPS.items():
        available = [c for c in candidates if c in df.columns]
        if not available:
            continue
        chosen = st.sidebar.selectbox(f'{label} (raw)', options=available, index=0)
        chosen_mapping[label] = chosen

    # Player selection helpers
    from utils.player_select import build_player_choices, build_player_summary, stable_widget_key

    # Create a robust display name column and choices derived from it so UI shows names only
    idcol_main = df.columns[0]
    mapping = {}
    for map_path in [Path('data/raw/unique_batters_with_names.csv'), Path('data/unique_batters_with_names.csv')]:
        if map_path.exists():
            try:
                mm = pd.read_csv(map_path, dtype=str)
                idc = next((c for c in mm.columns if c.lower() in ('batter','playerid','mlbam','id','key')), None)
                namec = next((c for c in mm.columns if 'name' in c.lower() or 'full' in c.lower()), None)
                if idc and namec:
                    mm[idc] = mm[idc].astype(str).str.strip()
                    mm[namec] = mm[namec].astype(str).str.strip()
                    mapping.update(mm.set_index(idc)[namec].to_dict())
            except Exception:
                pass

    def canonical_display(row):
        # prefer an explicit display column if present
        if 'name_display' in df.columns and row.get('name_display') and str(row.get('name_display')).strip():
            nm0 = str(row.get('name_display')).strip()
            if not nm0.isdigit():
                return nm0
        # prefer existing human name if available and non-numeric
        if 'name' in df.columns:
            nm = row.get('name')
            if isinstance(nm, str) and nm.strip() and not nm.strip().isdigit():
                return nm.strip()
        # try mapping via id
        rid = str(row[idcol_main]).strip()
        if rid in mapping:
            return mapping[rid]
        # fallback: empty string to avoid showing raw id
        return ''

    df['name_display'] = df.apply(canonical_display, axis=1)
    # Build choices from a fallback sequence so we show human names when available.
    # Preference order: name_display -> name -> any other column containing 'name'
    # IMPORTANT: do NOT fall back to showing raw ids by default (this prevents numeric ids from appearing in the UI)
    allow_id_fallback = False

    def build_choices_from_df(df, idcol):
        # helper to collect a list of display strings using fallbacks
        vals = []
        # primary: explicit name_display
        if 'name_display' in df.columns:
            vals.extend(df['name_display'].dropna().astype(str).str.strip().replace('', pd.NA).dropna().tolist())
        # secondary: normalized 'name' column
        if 'name' in df.columns:
            vals.extend(df['name'].dropna().astype(str).str.strip().replace('', pd.NA).dropna().tolist())
        # tertiary: any other column with 'name' in the column name
        for c in df.columns:
            if c not in ('name_display','name') and 'name' in c.lower():
                vals.extend(df[c].dropna().astype(str).str.strip().replace('', pd.NA).dropna().tolist())
        # optionally fall back to id column
        if allow_id_fallback and idcol in df.columns:
            vals.extend(df[idcol].astype(str).dropna().astype(str).str.strip().replace('', pd.NA).dropna().tolist())
        # unique + sort
        uniq = sorted(set([v for v in vals if v is not None and str(v).strip()]), key=lambda s: s.lower())
        return uniq

    choices = build_choices_from_df(df, idcol_main)
    # Forced-name pass: ensure choices contain only human names by mapping ids -> names
    # Build a single mapping dict from any available mapping CSVs (resolved relative to repo root)
    forced_map = {}
    for map_path in [ROOT.joinpath('data/raw/unique_batters_with_names.csv'), ROOT.joinpath('data/unique_batters_with_names.csv'), ROOT.joinpath('data/raw/unique_batters.csv')]:
        if not map_path.exists():
            continue
        try:
            mm = pd.read_csv(map_path, dtype=str)
        except Exception:
            continue
        idc = next((c for c in mm.columns if c.lower() in ('batter','playerid','mlbam','id','key')), None)
        namec = next((c for c in mm.columns if 'name' in c.lower() or 'full' in c.lower()), None)
        if not idc or not namec:
            if mm.shape[1] >= 2:
                idc = mm.columns[0]
                namec = mm.columns[1]
            else:
                continue
        mm[idc] = mm[idc].astype(str).str.strip()
        mm[namec] = mm[namec].astype(str).str.strip()
        # update mapping; later files can override earlier ones
        forced_map.update(mm.set_index(idc)[namec].to_dict())

    # Build forced name list: prefer existing name_display, then name, then mapping via id
    forced_names = []
    id_keys = df[idcol_main].astype(str).astype(object)
    for i, row in df.iterrows():
        nm = ''
        if 'name_display' in df.columns:
            nm = str(row.get('name_display') or '').strip()
        if not nm and 'name' in df.columns:
            nm = str(row.get('name') or '').strip()
        if not nm:
            rid = str(row[idcol_main]).strip()
            if rid in forced_map:
                nm = forced_map[rid]
        # Only include non-numeric, non-empty names
        if nm and not nm.isdigit():
            forced_names.append(nm)
            # write back into name_display so downstream filtering works
            try:
                df.at[i, 'name_display'] = nm
            except Exception:
                pass

    # Deduplicate and sort
    forced_names = sorted(set(forced_names), key=lambda s: s.lower())
    if forced_names:
        choices = forced_names
    # quick in-app diagnostics to help deployments where mapping files may be missing
    total_players = len(df)
    mapped_names = 0
    if 'name_display' in df.columns:
        mapped_names = df['name_display'].astype(str).str.strip().replace('', pd.NA).dropna().shape[0]
    name_columns = [c for c in df.columns if 'name' in c.lower()]
    map_files = [ROOT.joinpath('data/raw/unique_batters_with_names.csv'), ROOT.joinpath('data/unique_batters_with_names.csv')]
    present_map_files = [p.as_posix() for p in map_files if p.exists()]
    st.sidebar.markdown('### Data diagnostics')
    st.sidebar.write(f'- total players: {total_players}')
    st.sidebar.write(f'- players with mapped display name: {mapped_names}')
    st.sidebar.write(f'- name-like columns: {name_columns}')
    if present_map_files:
        st.sidebar.write(f'- mapping files found: {present_map_files}')
    else:
        st.sidebar.warning('No mapping CSVs found in data/raw or data/. The dropdown may be empty on deploy.')
    # and surface a visible warning so the deploy owner can fix mapping files.
    if not choices:
        # Try a direct mapping-file fallback: read mapping CSVs (resolved relative to repo root) and build names for ids present in df
        direct_names = []
        for map_path in [ROOT.joinpath('data/raw/unique_batters_with_names.csv'), ROOT.joinpath('data/unique_batters_with_names.csv'), ROOT.joinpath('data/raw/unique_batters.csv')]:
            if not map_path.exists():
                continue
            try:
                mm = pd.read_csv(map_path, dtype=str)
            except Exception:
                continue
            idc = next((c for c in mm.columns if c.lower() in ('batter','playerid','mlbam','id','key')), None)
            namec = next((c for c in mm.columns if 'name' in c.lower() or 'full' in c.lower()), None)
            if not idc or not namec:
                # attempt heuristic: use first two columns
                if mm.shape[1] >= 2:
                    idc = mm.columns[0]
                    namec = mm.columns[1]
                else:
                    continue
            mm[idc] = mm[idc].astype(str).str.strip()
            mm[namec] = mm[namec].astype(str).str.strip()
            # only keep mapping for ids present in df
            ids_in_df = set(df[idcol_main].astype(str).str.strip().unique())
            filtered = mm[mm[idc].isin(ids_in_df)]
            if filtered.empty:
                continue
            direct_names.extend(filtered[namec].dropna().astype(str).str.strip().tolist())
        direct_names = sorted(set([n for n in direct_names if n and not str(n).strip().isdigit()]), key=lambda s: s.lower())
        if direct_names:
            choices = direct_names
            # Build a mapping dict from the mapping CSVs for ids -> name and write back into df['name_display']
            direct_map = {}
            for map_path in [ROOT.joinpath('data/raw/unique_batters_with_names.csv'), ROOT.joinpath('data/unique_batters_with_names.csv'), ROOT.joinpath('data/raw/unique_batters.csv')]:
                if not map_path.exists():
                    continue
                try:
                    mm = pd.read_csv(map_path, dtype=str)
                except Exception:
                    continue
                idc = next((c for c in mm.columns if c.lower() in ('batter','playerid','mlbam','id','key')), None)
                namec = next((c for c in mm.columns if 'name' in c.lower() or 'full' in c.lower()), None)
                if not idc or not namec:
                    if mm.shape[1] >= 2:
                        idc = mm.columns[0]
                        namec = mm.columns[1]
                    else:
                        continue
                mm[idc] = mm[idc].astype(str).str.strip()
                mm[namec] = mm[namec].astype(str).str.strip()
                # only keep mapping for ids present in df
                ids_in_df = set(df[idcol_main].astype(str).str.strip().unique())
                filtered = mm[mm[idc].isin(ids_in_df)]
                for _, r in filtered[[idc, namec]].dropna().iterrows():
                    rid = str(r[idc]).strip()
                    rname = str(r[namec]).strip()
                    if rname and not rname.isdigit():
                        direct_map[rid] = rname
            # ensure name_display column exists and is string
            if 'name_display' not in df.columns:
                df['name_display'] = ''
            # write mapped names into df where name_display is empty
            try:
                key_series = df[idcol_main].astype(str).str.strip()
                mapped_vals = key_series.map(direct_map).fillna('')
                df.loc[df['name_display'].astype(str).str.strip() == '', 'name_display'] = mapped_vals[df['name_display'].astype(str).str.strip() == '']
            except Exception:
                pass
            st.sidebar.success('Populated dropdown choices using mapping CSVs found in the repo.')
        else:
            # no mapping names available — do NOT silently fall back to ids unless user enabled it
            if allow_id_fallback and idcol_main in df.columns:
                choices = sorted(df[idcol_main].astype(str).dropna().unique().tolist(), key=lambda s: s.lower())
                st.sidebar.warning('No human-readable player names found — using player ids because you enabled the ID fallback.')
            else:
                st.sidebar.error('No human-readable player names found. Provide data/raw/unique_batters_with_names.csv or enable "Show IDs when no name available" in the sidebar to use ids.')
    # show a small sample so you can see what the dropdown will contain
    try:
        st.sidebar.markdown('**Sample dropdown entries**')
        st.sidebar.write(choices[:10])
    except Exception:
        pass
    summaries = build_player_summary(df, name_col='name_display', season_col='Season')

    st.sidebar.markdown('### Player selection')
    select_mode = st.sidebar.radio('Selection mode', ['Primary (simple)', 'Disambiguation (name + year)', 'Substring filter', 'Fuzzy (optional)'])

    selected_name = None
    widget_key_base = stable_widget_key('player', df)
    # Clear any persisted session state for player widgets so old numeric selections don't reappear
    try:
        for k in list(st.session_state.keys()):
            if isinstance(k, str) and k.startswith(widget_key_base):
                del st.session_state[k]
    except Exception:
        pass

    if select_mode == 'Primary (simple)':
        # simple selectbox with deduplicated, alphabetically sorted names (display-only)
        selected_display = st.selectbox('Player', options=choices, key=widget_key_base + ':player-simple')
        selected_name = selected_display

    elif select_mode == 'Disambiguation (name + year)':
        # Build options like 'Name (last: YEAR)' but return canonical Name
        labels = []
        for n in choices:
            last, cnt = summaries.get(n, (None, 0))
            label = f"{n} (last: {last})" if last is not None else f"{n}"
            labels.append((label, n))
        # Map displayed labels to canonical names
        display_opts = [lab for lab, _ in labels]
        choice_map = {lab: canon for lab, canon in labels}
        sel = st.selectbox('Player', options=display_opts, key=widget_key_base + ':player-disamb')
        selected_display = sel
        selected_name = choice_map.get(sel)

    elif select_mode == 'Substring filter':
        q = st.text_input('Filter by substring (case-insensitive)', value='', key=widget_key_base + ':player-substr-input')
        if q:
            ql = q.strip().lower()
            filtered = [c for c in choices if ql in c.lower()]
        else:
            filtered = choices
        selected_display = st.selectbox('Player (filtered)', options=filtered, key=widget_key_base + ':player-substr-select')
        selected_name = selected_display

    elif select_mode == 'Fuzzy (optional)':
        try:
            from rapidfuzz import process
            qf = st.text_input('Fuzzy search query', value='', key=widget_key_base + ':player-fuzzy-input')
            if qf:
                results = process.extract(qf, choices, limit=20)
                # results: list of tuples (choice, score, idx)
                top = [r[0] for r in results]
            else:
                top = []
            if not top:
                st.info('Enter a query above to get fuzzy matches')
            else:
                selected_display = st.selectbox('Fuzzy matches', options=top, key=widget_key_base + ':player-fuzzy-select')
                selected_name = selected_display
        except Exception:
            st.warning('rapidfuzz not installed: install rapidfuzz to enable fuzzy search')
            selected_display = st.selectbox('Player', options=choices, key=widget_key_base + ':player-fallback')
            selected_name = selected_display

    # After selection, filter the main DataFrame to the selected player (if any)
    if selected_name:
        # choose rows matching the display name we created
        df_search = df[df['name_display'].astype(str).str.strip() == selected_name].copy()
    else:
        df_search = df.copy()

    # ensure there's a 'name' column (fallback to id)
    idcol = df.columns[0]
    if 'name' not in df.columns:
        df['name'] = df[idcol].astype(str)

    if sel_cluster == 'All':
        sub = df_search.copy()
    else:
        sub = df_search[df_search['cluster'].astype(str) == str(sel_cluster)].copy()

    st.header(f'Cluster selection: {sel_cluster}  —  {len(sub)} players')

    # Cluster summaries — main project section: auto-describe each cluster and show examples
    st.header('Cluster summaries')
    st.write('Below are automatic summaries for each cluster: top distinguishing features and representative players.')
    features_list = features
    centroids, zcent = summarize_cluster_features(df, features_list)
    if centroids is None:
        st.info('No numeric features available to summarize clusters.')
    else:
        def sort_key(x):
            try:
                return int(x)
            except Exception:
                return str(x)

        # generate plain-English cluster descriptions
        cluster_texts = generate_cluster_descriptions(df, features_list, top_n=3)

        for cl in sorted(centroids.index.tolist(), key=sort_key):
            with st.expander(f'Cluster {cl} summary', expanded=False):
                st.subheader(f'Cluster {cl}')
                # show the plain-English explanation we generated
                txt = cluster_texts.get(str(cl), '')
                if txt:
                    st.markdown(f"**Summary:** {txt}")
                # top positive z-score features for this cluster
                # use the actual index value `cl` when indexing zcent (preserves dtype)
                row = zcent.loc[cl]
                top_pos = row.sort_values(ascending=False).head(5)
                top_neg = row.sort_values().head(5)
                st.write('Top distinguishing features (positive):')
                st.write(top_pos.to_frame('z').round(2))
                st.write('Top distinguishing features (negative):')
                st.write(top_neg.to_frame('z').round(2))

                # bar chart for top positive features
                try:
                    if 'px' in globals() and px is not None and 'HAS_PLOTLY' in globals() and HAS_PLOTLY:
                        figc = px.bar(x=top_pos.index, y=top_pos.values, labels={'x': 'feature', 'y': 'z-score'}, title='Top positive z-features')
                        st.plotly_chart(figc, use_container_width=True)
                    else:
                        import matplotlib.pyplot as _plt
                        figc, axc = _plt.subplots(figsize=(8, 3))
                        axc.bar(top_pos.index, top_pos.values)
                        axc.set_xticklabels(top_pos.index, rotation=45, ha='right')
                        axc.set_ylabel('z-score')
                        axc.set_title('Top positive z-features')
                        st.pyplot(figc)
                except Exception:
                    pass

                # representative players
                closest, far = cluster_representatives(df, cl, features_list)
                if closest:
                    st.write('Representative players (closest to centroid):')
                    for i, name in closest:
                        st.write(f'- {name}  (id: {i})')
                if far:
                    st.write('Contrast players (far from centroid within cluster):')
                    for i, name in far:
                        st.write(f'- {name}  (id: {i})')

    # Player table (compact preview + expand)
    # Build display columns: cluster, name, and friendly labels mapped to chosen raw columns
    idcol = df.columns[0]
    display_df = sub.copy()
    # Prefer the mapped human-readable name_display for the UI table; fall back to previous heuristics
    if 'name_display' in display_df.columns:
        display_df['name'] = display_df['name_display'].astype(str).str.strip()
    else:
        display_df['name'] = get_display_name_series(sub)
    display_cols = ['cluster', 'name']
    for label, raw in chosen_mapping.items():
        if raw in display_df.columns:
            display_df[label] = display_df[raw]
            display_cols.append(label)
    display_df = display_df[display_cols].sort_values(by='cluster')

    # compact preview with optional expand
    preview_rows = st.number_input('Rows to preview', min_value=5, max_value=200, value=10)
    expand_table = st.checkbox('Expand to full table', value=False)
    to_show = display_df if expand_table else display_df.head(preview_rows)

    try:
        disp = to_show.copy()
        # format numeric columns to 2 decimals but keep minimal styling (no loud colors)
        numcols = [c for c in disp.columns if pd.api.types.is_numeric_dtype(disp[c])]
        fmt = {c: '{:.2f}' for c in numcols}
        sty = disp.style.format(fmt).set_table_styles([
            {'selector': 'th', 'props': [('text-align', 'left'), ('font-weight', '600'), ('padding', '6px')]},
            {'selector': 'td', 'props': [('text-align', 'left'), ('padding', '4px 6px')]},
            {'selector': 'table', 'props': [('border-collapse', 'collapse'), ('width', '100%')]}
        ])
        html = sty.hide_index().to_html()
        st.markdown(html, unsafe_allow_html=True)
    except Exception:
        st.dataframe(to_show)

    # PCA plot (preprocess numeric features to avoid domination by raw scales/outliers)
    if show_pca:
        feat_df = sub[features].select_dtypes(include=[np.number]).copy()
        feat_df = feat_df.fillna(0)
        if feat_df.shape[1] >= 2 and feat_df.shape[0] >= 2:
            try:
                # 1) convert circular mean angles (only mean columns) into sin/cos pairs
                angle_cols = [c for c in feat_df.columns if ('angle' in c.lower() or 'direction' in c.lower()) and c.lower().endswith('_mean')]
                for c in angle_cols:
                    # assume degrees; convert to radians then to sin/cos
                    rad = np.deg2rad(feat_df[c].astype(float).fillna(0).values)
                    feat_df[c + '_sin'] = np.sin(rad)
                    feat_df[c + '_cos'] = np.cos(rad)
                    feat_df.drop(columns=[c], inplace=True)

                # 2) log-transform count-like columns (n_swings etc.) to reduce scale differences
                count_cols = [c for c in feat_df.columns if c.lower().startswith('n_') or c.lower().endswith('_count') or c.lower() == 'n_swings']
                for c in count_cols:
                    feat_df[c] = np.log1p(feat_df[c].astype(float).fillna(0))

                # 3) winsorize extreme outliers at 1st/99th percentiles per column
                for c in feat_df.columns:
                    lo = feat_df[c].quantile(0.01)
                    hi = feat_df[c].quantile(0.99)
                    if pd.notna(lo) and pd.notna(hi) and lo < hi:
                        feat_df[c] = feat_df[c].clip(lower=lo, upper=hi)

                # 4) robust scale (median & IQR-like) to reduce influence of remaining outliers
                from sklearn.preprocessing import RobustScaler
                from sklearn.decomposition import PCA

                scaler = RobustScaler()
                Xs = scaler.fit_transform(feat_df.values)

                # 5) PCA on preprocessed, scaled features
                pca = PCA(n_components=2)
                Xp = pca.fit_transform(Xs)

                pxdf = pd.DataFrame(Xp, columns=['PC1','PC2'])
                # safe name series: prefer name_display, then 'name', then get_display_name_series
                idcol_local = sub.columns[0]
                if 'name_display' in sub.columns:
                    name_series = sub['name_display'].astype(object)
                elif 'name' in sub.columns:
                    name_series = sub['name'].astype(object)
                else:
                    name_series = get_display_name_series(sub)
                # ensure no nulls in hover name (use empty string rather than id)
                name_filled = name_series.fillna('').astype(str)
                pxdf['name'] = name_filled.values
                pxdf['id'] = sub[idcol_local].astype(str).values
                pxdf['cluster'] = sub['cluster'].astype(str).values
                # show explained variance for transparency
                evr = pca.explained_variance_ratio_
                st.caption(f'PCA explained variance ratio: PC1={evr[0]:.2f}, PC2={evr[1]:.2f}')
                # Plot: prefer Plotly if available, otherwise fallback to matplotlib + seaborn
                if use_plotly and px is not None:
                    fig = go.Figure()
                    clusters_unique = sorted(emb['cluster'].unique(), key=lambda x: (int(x) if str(x).isdigit() else x))
                    # pick a high-contrast qualitative palette
                    base_palette = px.colors.qualitative.Plotly
                    colors = (base_palette * ((len(clusters_unique) // len(base_palette)) + 1))[:len(clusters_unique)]
                    for i, cl in enumerate(clusters_unique):
                        sub = emb[emb['cluster'] == cl]
                        idcol = df.columns[0]
                        if 'name_display' in df.columns:
                            names = df.loc[sub.index, 'name_display'].astype(object).fillna('').astype(str)
                        elif 'name' in df.columns:
                            names = df.loc[sub.index, 'name'].astype(object).fillna('').astype(str)
                        else:
                            names = pd.Series([''] * len(sub), index=sub.index, dtype=str)
                        ids = df.loc[sub.index, idcol].astype(str)
                        hover_text = [f"{n} ({i0})" if n else f"{i0}" for n, i0 in zip(names.tolist(), ids.tolist())]
                        fig.add_trace(go.Scattergl(x=sub['PC1'], y=sub['PC2'], mode='markers', name=f'Cluster {cl}',
                                                   marker=dict(color=colors[i], size=9, opacity=0.85, line=dict(width=0.7, color='#222')),
                                                   hoverinfo='text', text=hover_text))
                    # centroids in PCA space
                    centroids = emb.groupby('cluster')[['PC1','PC2']].mean()
                    fig.add_trace(go.Scattergl(x=centroids['PC1'], y=centroids['PC2'], mode='markers+text',
                                               marker=dict(symbol='x', color='black', size=16),
                                               text=[f'Centroid {c}' for c in centroids.index], textposition='top center', name='Centroids'))
                    # add annotations for centroids for clarity
                    annotations = []
                    for c_idx, row in centroids.reset_index().iterrows():
                        annotations.append(dict(x=row['PC1'], y=row['PC2'], text=f"C{row['cluster']}", showarrow=False, yshift=10, font=dict(color='black', size=12)))
                    fig.update_layout(title='Cluster PCA overview', xaxis_title='PC1', yaxis_title='PC2', height=height, annotations=annotations)
                    st.plotly_chart(fig, use_container_width=True)
                scatter_df = df[[c1, c2, 'cluster']].copy()
                # ensure a name column for hover (prefer name_display)
                if 'name_display' in df.columns:
                    scatter_df['name'] = df['name_display']
                elif 'name' in df.columns:
                    scatter_df['name'] = df['name']
                else:
                    scatter_df['name'] = df[df.columns[0]].astype(str)
                scatter_df = scatter_df.dropna(subset=[c1, c2])
                scatter_df['cluster'] = scatter_df['cluster'].astype(str)
                if HAS_PLOTLY and px is not None:
                    fig2 = px.scatter(scatter_df, x=c1, y=c2, color='cluster', hover_data=['name'], height=600)
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    import matplotlib.pyplot as _plt
                    fig2, ax2 = _plt.subplots(figsize=(8, 6))
                    clusters_unique = scatter_df['cluster'].unique()
                    palette = sns.color_palette(n_colors=len(clusters_unique))
                    color_map = {c: palette[i] for i, c in enumerate(clusters_unique)}
                    colors = scatter_df['cluster'].map(color_map)
                    ax2.scatter(scatter_df[c1], scatter_df[c2], c=list(colors), s=40, alpha=0.8)
                    ax2.set_xlabel(c1)
                    ax2.set_ylabel(c2)
                    ax2.set_title(f'{c1} vs {c2}')
                    st.pyplot(fig2)
            except Exception as e:
                st.write('Scatter failed:', e)
        else:
            st.info('Not enough numeric features for scatter')

    # Player detail: driven by top-search selections (no separate search box)
    idcol = df.columns[0]
    # Player detail driven by the new selection UI: `selected_name` (canonical Name string)
    if selected_name:
        # find the row in the filtered `df_search` (which contains rows for selected_name) or fall back to df
        try:
            prow = df_search.iloc[0]
        except Exception:
            # fallback: find first matching row in df using name_display
            prow = df[df['name_display'].astype(str).str.strip() == selected_name].iloc[0]
        # show only the human-friendly display name if present
        display_val = ''
        if 'name_display' in prow.index and str(prow.get('name_display')).strip():
            display_val = prow.get('name_display')
        elif 'name' in prow.index and str(prow.get('name')).strip():
            display_val = prow.get('name')
        else:
            display_val = ''
        st.write('Player:', display_val)
        player_feats = prow[features].to_frame(name='player')
        # cluster centroid
        try:
            centroids = df.groupby('cluster')[features].mean()
            centroids.index = centroids.index.astype(str)
            player_cluster = str(prow['cluster'])
            centroid = centroids.loc[player_cluster]
            centroid = centroid.to_frame('cluster_mean')
            player_feats = player_feats.astype(float)
            comp = player_feats.join(centroid).dropna()
            st.dataframe(comp)
            # Compute differences robustly: handle circular angle features and show z-scored diffs
            # We'll compute cluster mean/std per-feature for the player's cluster using circular stats for angles.
            player_cluster = str(prow['cluster'])
            cluster_df = df[df['cluster'].astype(str) == player_cluster]
            rows = []
            for feat in comp.index:
                pval = float(comp.loc[feat, 'player'])
                # default cluster mean from centroid table
                cval = float(centroid.loc[feat, 'cluster_mean'])
                # detect circular feature
                is_circular = (('angle' in feat.lower() or 'direction' in feat.lower()) and feat.lower().endswith('_mean'))
                if is_circular:
                    # compute circular mean/std from raw player values in the cluster
                    vals = cluster_df[feat].dropna().astype(float)
                    if len(vals) >= 1:
                        rad = np.deg2rad(vals.values)
                        ms = np.sin(rad).mean()
                        mc = np.cos(rad).mean()
                        R = np.hypot(ms, mc)
                        # mean angle in radians
                        mean_rad = np.arctan2(ms, mc)
                        mean_deg = np.rad2deg(mean_rad)
                        cval = float(mean_deg)
                        # circular std (radians): sqrt(-2 ln R) ; convert to degrees
                        if R <= 0 or np.isnan(R):
                            cstd = 1.0
                        else:
                            std_rad = np.sqrt(max(0.0, -2.0 * np.log(min(max(R, 1e-12), 1.0))))
                            cstd = float(std_rad * 180.0 / np.pi)
                            if not (cstd and pd.notna(cstd)):
                                cstd = 1.0
                    else:
                        # fallback to arithmetic centroid/std
                        try:
                            cstd = float(cluster_df[feat].std()) if pd.notna(cluster_df[feat].std()) else 1.0
                        except Exception:
                            cstd = 1.0
                else:
                    # numeric feature: compute arithmetic mean/std in cluster
                    try:
                        cstd = float(cluster_df[feat].std()) if pd.notna(cluster_df[feat].std()) else 1.0
                        # use centroid arithmetic mean as cval (already present)
                    except Exception:
                        cstd = 1.0
                # compute delta with circular wrapping when needed
                if is_circular:
                    delta = ((pval - cval + 180.0) % 360.0) - 180.0
                else:
                    delta = (pval - cval)
                diff_z = delta / cstd if cstd != 0 else delta
                rows.append({'feature': feat, 'diff': delta, 'diff_z': diff_z})
            diffdf = pd.DataFrame(rows).set_index('feature')
            # Bar chart comparing player vs cluster mean (z-scores)
            st.caption('Values shown are player - cluster_mean, normalized by cluster std (z-score). Circular angle means are compared using minimal signed angle difference.')
            if HAS_PLOTLY and px is not None:
                fig3 = px.bar(diffdf.reset_index(), x='feature', y='diff_z', title='Player - cluster mean (z-score)', labels={'diff_z': 'diff (z-score)'})
                fig3.update_xaxes(tickangle=-45)
                st.plotly_chart(fig3, use_container_width=True)
            else:
                import matplotlib.pyplot as _plt
                fig3, ax3 = _plt.subplots(figsize=(8, 3))
                ax3.bar(diffdf.index, diffdf['diff_z'])
                ax3.set_xticklabels(diffdf.index, rotation=45, ha='right')
                ax3.set_ylabel('diff (z-score)')
                ax3.set_title('Player - cluster mean (z-score)')
                st.pyplot(fig3)
        except Exception as e:
            st.write('Player detail failed:', e)

    # Centroid heatmap (z-scored per feature) + Feature index
    if show_centroid and diag:
        try:
            features_list = features
            df_feats = df.set_index('cluster')[features_list].select_dtypes(include=[np.number]).groupby(level=0).mean()
            if not df_feats.empty:
                # z-score normalize features across clusters so shading shows relative differences
                z = (df_feats - df_feats.mean(axis=0)) / (df_feats.std(axis=0).replace(0, 1))
                # create a heatmap: prefer Plotly if available, otherwise use seaborn
                if HAS_PLOTLY and go is not None:
                    heat = go.Figure(data=go.Heatmap(
                        z=z.values,
                        x=z.columns.tolist(),
                        y=z.index.astype(str).tolist(),
                        colorscale='RdBu',
                        zmid=0,
                        colorbar=dict(title='z-score')))
                    heat.update_layout(title='Cluster centroid (feature z-scores)', xaxis_tickangle=-45, height=300 + 30 * len(z))
                    st.plotly_chart(heat, use_container_width=True)
                else:
                    import matplotlib.pyplot as _plt
                    fig_h, ax_h = _plt.subplots(figsize=(max(8, len(z.columns) * 0.6), 2 + len(z) * 0.5))
                    sns.heatmap(z, ax=ax_h, cmap='RdBu', center=0, cbar_kws={'label': 'z-score'})
                    ax_h.set_title('Cluster centroid (feature z-scores)')
                    _plt.xticks(rotation=45, ha='right')
                    st.pyplot(fig_h)

                # Feature index: readable descriptions for each feature (with filter and styling)
                with st.expander('Feature index (definitions)'):
                    st.write('This list explains what each feature means so the heatmap is easier to read. Use the filter to quickly find a feature by name or keyword.')
                    # show descriptions for features in the order used
                    rows = []
                    for f in features_list:
                        desc = FEATURE_DESCRIPTIONS.get(f, '')
                        rows.append({'feature': f, 'description': desc})
                    fi = pd.DataFrame(rows)
                    # small search box to filter features
                    q = st.text_input('Filter features (name or keyword)', value='', placeholder='e.g. launch_speed, angle, attack')
                    if q:
                        ql = q.strip().lower()
                        fi = fi[fi['feature'].str.lower().str.contains(ql) | fi['description'].str.lower().str.contains(ql)]
                    if fi.empty:
                        st.info('No features match that filter')
                    else:
                        # present a compact, left-aligned styled table without the index
                        try:
                            styled = fi.style.set_properties(**{'text-align': 'left'}).hide_index()
                            st.dataframe(styled, use_container_width=True)
                        except Exception:
                            # fallback
                            st.dataframe(fi, use_container_width=True)
        except Exception as e:
            st.write('Centroid heatmap failed:', e)

    # Links to static showcase files
    idx = Path('out/cluster_showcase/index.html')
    if idx.exists():
        st.sidebar.markdown('### Static showcase')
        st.sidebar.markdown(f"[Open cluster showcase]({idx.as_posix()})")

    


if __name__ == '__main__':
    main()
