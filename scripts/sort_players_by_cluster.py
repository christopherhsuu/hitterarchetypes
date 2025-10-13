#!/usr/bin/env python3
"""
Sort players into files by their assigned cluster.

Reads `data/player_archetypes.csv` (or --input) and writes one CSV per cluster
into the output directory (default: data/clusters/). If a names file is found
it will join player names so you can see who is in each group.

Usage:
  python3 scripts/sort_players_by_cluster.py
  python3 scripts/sort_players_by_cluster.py --cluster-col cluster_k5 --names data/raw/unique_batters_with_names.csv
"""
from pathlib import Path
import argparse
import sys
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description='Split players into per-cluster files')
    p.add_argument('--input', '-i', default='data/player_archetypes.csv', help='Per-player archetypes CSV (output of clustering)')
    p.add_argument('--out-dir', '-o', default='data/clusters', help='Directory to write per-cluster CSVs')
    p.add_argument('--id-col', default=None, help='Player id column (defaults to first column)')
    p.add_argument('--cluster-col', default=None, help='Column name with cluster labels (auto-detect if omitted)')
    p.add_argument('--names', default=None, help='Optional CSV with id->name mapping (defaults to common repo paths)')
    return p.parse_args()


def find_names_file(user_path=None):
    candidates = []
    if user_path:
        candidates.append(Path(user_path))
    # common locations
    candidates += [Path('data/raw/unique_batters_with_names.csv'), Path('data/unique_batters_with_names.csv'), Path('data/raw/unique_batters.csv'), Path('data/unique_batters.csv')]
    for p in candidates:
        if p.exists():
            return p
    return None


def main():
    args = parse_args()
    inp = Path(args.input)
    if not inp.exists():
        print(f'Input file not found: {inp}', file=sys.stderr)
        sys.exit(2)

    df = pd.read_csv(inp, dtype=str)
    if df.shape[0] == 0:
        print('Input file is empty', file=sys.stderr)
        sys.exit(2)

    id_col = args.id_col or df.columns[0]
    if id_col not in df.columns:
        print(f'ID column {id_col} not found in input. Columns: {list(df.columns)}', file=sys.stderr)
        sys.exit(2)

    # detect cluster column
    cluster_col = args.cluster_col
    if cluster_col is None:
        # prefer exact 'cluster' or 'cluster_k5', else any column starting with 'cluster'
        if 'cluster' in df.columns:
            cluster_col = 'cluster'
        else:
            cands = [c for c in df.columns if c.lower().startswith('cluster')]
            cluster_col = cands[0] if cands else None

    if cluster_col is None or cluster_col not in df.columns:
        print('Could not find a cluster column. Provide --cluster-col', file=sys.stderr)
        print('Columns available:', list(df.columns))
        sys.exit(2)

    # try to attach names
    names_path = find_names_file(args.names)
    names_df = None
    if names_path:
        try:
            names_df = pd.read_csv(names_path, dtype=str)
            # guess columns
            name_col = None
            idname_col = None
            for c in names_df.columns:
                if c.lower() in ('name','player','full_name','display_name'):
                    name_col = c
                if c.lower() in (id_col.lower(), 'batter', 'playerid', 'mlbam'):
                    idname_col = c
            if idname_col is None:
                idname_col = names_df.columns[0]
            if name_col is None and names_df.shape[1] >= 2:
                name_col = names_df.columns[1]
            names_df = names_df[[idname_col, name_col]].rename(columns={idname_col: id_col, name_col: 'name'})
            print(f'Using names file: {names_path} (joined on {id_col})')
        except Exception as e:
            print('Failed to read names file:', e)
            names_df = None
    else:
        print('No names file found; output will contain ids only')

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # merge names if present
    merged = df.copy()
    if names_df is not None:
        merged = merged.merge(names_df, on=id_col, how='left')

    # ensure cluster labels are sortable
    merged[cluster_col] = merged[cluster_col].astype(str)
    clusters = sorted(merged[cluster_col].unique(), key=lambda x: (int(x) if x.isdigit() else x))

    # determine which feature columns were used for clustering (try diagnostics.json)
    features_front = []
    try:
        import json
        diag_path = Path('out/cluster_plots/diagnostics.json')
        if diag_path.exists():
            diag = json.loads(diag_path.read_text())
            feats = diag.get('params', {}).get('features')
            if feats:
                # keep only those that are present in the merged dataframe
                features_front = [f for f in feats if f in merged.columns]
    except Exception:
        features_front = []

    # fallback heuristic order if diagnostics not available
    if not features_front:
        heur = ['whiff_rate', 'contact_rate', 'launch_speed_mean', 'launch_angle_mean', 'bat_speed_mean', 'launch_speed_std', 'launch_angle_std', 'bat_speed_std']
        features_front = [f for f in heur if f in merged.columns]

    # all other columns (preserve original order)
    other_cols = [c for c in merged.columns if c not in ([id_col, 'name', cluster_col] + features_front)]

    mapping_rows = []
    for cl in clusters:
        sub = merged[merged[cluster_col] == cl]
        out_fp = outdir / f'cluster_{cl}.csv'
        # write id, name (if present), clustering features first, then the rest, and finally the cluster label
        cols = [id_col]
        if 'name' in merged.columns:
            cols.append('name')
        cols += features_front
        cols += other_cols
        # ensure cluster_col is last
        if cluster_col not in cols:
            cols.append(cluster_col)

        # only keep columns that exist
        cols = [c for c in cols if c in merged.columns]
        sub[cols].to_csv(out_fp, index=False)
        mapping_rows.append({'cluster': cl, 'count': len(sub), 'file': str(out_fp)})

    # also write full mapping
    mapping_df = merged[[id_col, 'name', cluster_col]].rename(columns={cluster_col: 'cluster'})
    mapping_df.to_csv(outdir / 'batter_to_cluster.csv', index=False)

    summary = pd.DataFrame(mapping_rows)
    summary.to_csv(outdir / 'clusters_summary.csv', index=False)

    print('Wrote per-cluster files to', outdir)
    print('Summary:')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
