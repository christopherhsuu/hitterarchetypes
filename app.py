import streamlit as st
from pathlib import Path
import pandas as pd
import json
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns


@st.cache_data
def load_player_archetypes(path='data/player_archetypes.csv'):
    p = Path(path)
    if not p.exists():
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

    # robust name merge: try several candidate files and heuristics
    def try_merge_names(main_df):
        candidates = [Path('data/raw/unique_batters_with_names.csv'), Path('data/unique_batters_with_names.csv'), Path('data/raw/unique_batters.csv'), Path('data/unique_batters.csv')]
        idcol = main_df.columns[0]
        for pth in candidates:
            if not pth.exists():
                continue
            try:
                nm = pd.read_csv(pth, dtype=str)
            except Exception:
                continue
            # heuristics to pick id column in names file
            id_candidates = [c for c in nm.columns if c.lower() in ('batter','playerid','mlbam','id','key')]
            name_candidates = [c for c in nm.columns if 'name' in c.lower() or 'full' in c.lower() or 'display' in c.lower()]
            if not id_candidates:
                # try intersecting values with main id column
                common_col = None
                for c in nm.columns:
                    if nm[c].dropna().isin(main_df[idcol].astype(str).unique()).any():
                        common_col = c
                        break
                if common_col:
                    id_candidates = [common_col]
            if not name_candidates and nm.shape[1] >= 2:
                # second column fallback
                name_candidates = [nm.columns[1]]
            if not id_candidates or not name_candidates:
                continue
            idc = id_candidates[0]
            namec = name_candidates[0]
            # normalize id column to string for merge
            # create temporary string keys to avoid dtype mismatch
            nm_tmp = nm[[idc, namec]].copy()
            nm_tmp['__merge_key'] = nm_tmp[idc].apply(lambda x: str(x).strip() if pd.notna(x) else '')
            main_tmp = main_df.copy()
            main_tmp['__merge_key'] = main_tmp[idcol].apply(lambda x: str(x).strip() if pd.notna(x) else '')
            merged = main_tmp.merge(nm_tmp[['__merge_key', namec]].rename(columns={namec: 'name'}), on='__merge_key', how='left')
            # drop temporary key
            merged = merged.drop(columns=['__merge_key'])
            # if merge added any names, return merged
            if merged['name'].notna().any():
                return merged
        return main_df

    df = try_merge_names(df)

    # ensure there's a 'name' column (fallback to id)
    idcol = df.columns[0]
    if 'name' not in df.columns:
        df['name'] = df[idcol].astype(str)

    if sel_cluster == 'All':
        sub = df.copy()
    else:
        sub = df[df['cluster'].astype(str) == str(sel_cluster)].copy()

    st.header(f'Cluster selection: {sel_cluster}  —  {len(sub)} players')

    # Player table
    cols = ['cluster', 'name'] + features
    cols = [c for c in cols if c in sub.columns]
    st.dataframe(sub[cols].sort_values(by='cluster'))

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
                pxdf['name'] = sub.get('name', sub.index.astype(str))
                pxdf['cluster'] = sub['cluster'].astype(str).values
                # show explained variance for transparency
                evr = pca.explained_variance_ratio_
                st.caption(f'PCA explained variance ratio: PC1={evr[0]:.2f}, PC2={evr[1]:.2f}')
                fig = px.scatter(pxdf, x='PC1', y='PC2', color='cluster', hover_data=['name'], height=600)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.write('PCA failed:', e)
        else:
            st.info('Not enough numeric features to compute PCA')

    # Interactive scatter between any two selected features
    if show_scatter:
        numeric_features = [c for c in features if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_features) >= 2:
            # use two dropdowns for clarity
            c1 = st.selectbox('X feature', options=numeric_features, index=0)
            c2 = st.selectbox('Y feature', options=numeric_features, index=1 if len(numeric_features) > 1 else 0)
            try:
                scatter_df = df[[c1, c2, 'cluster', 'name']].copy()
                scatter_df = scatter_df.dropna(subset=[c1, c2])
                scatter_df['cluster'] = scatter_df['cluster'].astype(str)
                fig2 = px.scatter(scatter_df, x=c1, y=c2, color='cluster', hover_data=['name'], height=600)
                st.plotly_chart(fig2, use_container_width=True)
            except Exception as e:
                st.write('Scatter failed:', e)
        else:
            st.info('Not enough numeric features for scatter')

    # Player detail: select a player and show comparison vs cluster centroid
    st.subheader('Player detail')
    idcol = df.columns[0]
    # present only names to the user
    player_list = df['name'].fillna(df[idcol].astype(str)).tolist()
    sel_player = st.selectbox('Select player', ['None'] + player_list)
    if sel_player and sel_player != 'None':
        prow = df[df['name'] == sel_player].iloc[0]
        st.write('Player:', prow.get('name', prow[idcol]))
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
            diff = (comp['player'] - comp['cluster_mean']).reset_index()
            diff.columns = ['feature', 'diff']
            fig3 = px.bar(diff, x='feature', y='diff', title='Player - cluster mean')
            st.plotly_chart(fig3, use_container_width=True)
        except Exception as e:
            st.write('Player detail failed:', e)

    # Centroid heatmap
    if show_centroid and diag:
        try:
            features_list = features
            df_feats = df.set_index('cluster')[features_list].select_dtypes(include=[np.number]).groupby(level=0).mean()
            if not df_feats.empty:
                fig, ax = plt.subplots(figsize=(8, max(4, len(df_feats) * 0.4)))
                sns.heatmap(df_feats, cmap='vlag', center=df_feats.values.mean(), ax=ax)
                ax.set_ylabel('cluster')
                st.pyplot(fig)
        except Exception as e:
            st.write('Centroid heatmap failed:', e)

    # Links to static showcase files
    idx = Path('out/cluster_showcase/index.html')
    if idx.exists():
        st.sidebar.markdown('### Static showcase')
        st.sidebar.markdown(f"[Open cluster showcase]({idx.as_posix()})")


if __name__ == '__main__':
    main()
