#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input', '-i', default='data/player_archetypes.csv')
    p.add_argument('--out-dir', '-o', default='out/cluster_showcase')
    p.add_argument('--id-col', default=None)
    p.add_argument('--cluster-col', default=None)
    p.add_argument('--features', default=None, help='Comma-separated feature list to include')
    return p.parse_args()


def infer_columns(df, id_col, cluster_col):
    if id_col is None:
        id_col = df.columns[0]
    if cluster_col is None:
        if 'cluster' in df.columns:
            cluster_col = 'cluster'
        else:
            cands = [c for c in df.columns if c.lower().startswith('cluster')]
            cluster_col = cands[0] if cands else None
    return id_col, cluster_col


def load_features(df, features_arg):
    if features_arg:
        return [f.strip() for f in features_arg.split(',') if f.strip() in df.columns]
    diag = Path('out/cluster_plots/diagnostics.json')
    if diag.exists():
        try:
            j = json.loads(diag.read_text())
            feats = j.get('params', {}).get('features') or []
            return [f for f in feats if f in df.columns]
        except Exception:
            pass
    nums = df.select_dtypes(include=['number']).columns.tolist()
    return [c for c in nums if c in df.columns]


def make_showcase(inp, outdir, id_col=None, cluster_col=None, features_arg=None):
    inp = Path(inp)
    if not inp.exists():
        raise FileNotFoundError(inp)
    df = pd.read_csv(inp, dtype=str)
    id_col, cluster_col = infer_columns(df, id_col, cluster_col)
    if cluster_col is None or cluster_col not in df.columns:
        raise ValueError('No cluster column found; provide --cluster-col')

    num_df = df.select_dtypes(include=['number'])
    df[num_df.columns] = df[num_df.columns].apply(pd.to_numeric, errors='coerce')

    features = load_features(df, features_arg)
    if 'n_swings' in df.columns and 'n_swings' not in features:
        features = ['n_swings'] + features

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    clusters = sorted(df[cluster_col].astype(str).unique(), key=lambda x: (int(x) if x.isdigit() else x))
    rows = []
    index_lines = ['<html><head><meta charset="utf-8"><title>Cluster showcase</title></head><body>','<h1>Cluster showcase</h1>','<ul>']
    for cl in clusters:
        sub = df[df[cluster_col].astype(str) == str(cl)].copy()
        if sub.shape[0] == 0:
            continue
        stats = sub[features].agg(['mean','std','median']).transpose().round(3)
        stats.index.name = 'feature'
        stats.reset_index(inplace=True)
        stats_csv = outdir / f'cluster_{cl}_feature_summary.csv'
        stats.to_csv(stats_csv, index=False)

        table_html = sub[[id_col] + ([c for c in ['name'] if 'name' in sub.columns]) + features].fillna('').to_html(index=False)
        stats_html = stats.to_html(index=False)
        page = f"""<html><head><meta charset='utf-8'><title>Cluster {cl}</title></head><body>
<h1>Cluster {cl} — {len(sub)} players</h1>
<h2>Feature summary (mean / std / median)</h2>
{stats_html}
<h2>Players</h2>
{table_html}
</body></html>"""
        (outdir / f'cluster_{cl}.html').write_text(page)
        index_lines.append(f"<li><a href='cluster_{cl}.html'>Cluster {cl} ({len(sub)})</a></li>")
        rows.append({'cluster': cl, 'count': len(sub), 'summary_csv': str(stats_csv)})

    index_lines.append('</ul></body></html>')
    (outdir / 'index.html').write_text('\n'.join(index_lines))

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(outdir / 'clusters_overview.csv', index=False)
    return outdir


def main():
    args = parse_args()
    try:
        outdir = make_showcase(args.input, args.out_dir, args.id_col, args.cluster_col, args.features)
        print('Wrote showcase to', outdir)
    except Exception as e:
        print('Error:', e)


if __name__ == '__main__':
    main()
