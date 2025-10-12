#!/usr/bin/env python3
"""
Read a CSV of player ids (default: data/unique_batters.csv),
use pybaseball to lookup player names, and write a CSV with a
new `name` column placed immediately after the id column.

Usage:
  python scripts/add_player_names.py \
      --input data/unique_batters.csv \
      --output data/unique_batters_with_names.csv \
      --id-col batter --id-type mlbam

Notes:
  - Requires `pybaseball` and `pandas` to be installed.
    Install with: pip install pybaseball pandas
  - Default id type is 'mlbam'. If your ids are different, change
    --id-type to the appropriate key (e.g. 'bbref') if supported.
"""
from pathlib import Path
import argparse
import sys
import time


def try_imports():
    try:
        import pandas as pd
    except Exception as e:
        print("This script requires pandas. Install with: pip install pandas", file=sys.stderr)
        raise

    try:
        # pybaseball has functions to reverse lookup player ids
        from pybaseball import playerid_reverse_lookup
    except Exception as e:
        print("This script requires pybaseball. Install with: pip install pybaseball", file=sys.stderr)
        raise

    return pd, playerid_reverse_lookup


def build_full_name(row):
    # Attempt common column names
    first = row.get('name_first') or row.get('first_name') or row.get('firstname')
    last = row.get('name_last') or row.get('last_name') or row.get('lastname')
    if first and last:
        return f"{first} {last}"
    # fallback to any single-named column
    for k in ('name', 'full_name', 'display_name'):
        if row.get(k):
            return row.get(k)
    # last resort: join all string-like columns
    vals = [str(v).strip() for v in row.values() if v and isinstance(v, str)]
    return vals[0] if vals else ''


def add_names(input_csv, output_csv, id_col='batter', id_type='mlbam', sleep=0.1):
    pd, playerid_reverse_lookup = try_imports()

    input_path = Path(input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path, dtype=str)
    if id_col not in df.columns:
        raise ValueError(f"Input CSV does not contain the id column '{id_col}'. Available columns: {list(df.columns)}")

    # Prepare new column
    names = []

    # Keep a simple cache to avoid duplicate network calls
    cache = {}

    for i, raw_id in enumerate(df[id_col].fillna('')):
        if raw_id == '':
            names.append('')
            continue

        if raw_id in cache:
            names.append(cache[raw_id])
            continue

        # pybaseball.playerid_reverse_lookup accepts an id and returns a DataFrame
        try:
            # Some ids might be numeric strings; normalize to string for consistent lookup
            lookup_id = str(int(raw_id)) if raw_id.isdigit() else str(raw_id)
            # playerid_reverse_lookup expects a list-like of ids
            res = playerid_reverse_lookup([lookup_id], key_type=id_type)
        except TypeError:
            # older pybaseball versions accept a list-like without key_type
            try:
                res = playerid_reverse_lookup([lookup_id])
            except Exception:
                res = None
        except Exception:
            res = None

        name = ''
        try:
            if res is None or res.empty:
                # Fallback: try MLB Stats API directly (public endpoint)
                try:
                    import requests
                    mlb_api_url = f"https://statsapi.mlb.com/api/v1/people/{lookup_id}"
                    r = requests.get(mlb_api_url, timeout=10)
                    if r.status_code == 200:
                        j = r.json()
                        people = j.get('people') or []
                        if people:
                            p0 = people[0]
                            first = p0.get('firstName') or p0.get('nameFirst')
                            last = p0.get('lastName') or p0.get('nameLast')
                            if first and last:
                                name = f"{first} {last}"
                            elif p0.get('fullName'):
                                name = p0.get('fullName')
                except Exception:
                    # ignore and leave name blank
                    name = ''
            else:
                # res is a DataFrame-like; take first row and build a name
                row = res.iloc[0].to_dict()
                name = build_full_name(row)
        except Exception:
            name = ''

        cache[raw_id] = name
        names.append(name)

        # Be polite to the service
        time.sleep(sleep)

    # Insert name column next to id_col
    out_df = df.copy()
    insert_at = list(out_df.columns).index(id_col) + 1
    out_df.insert(insert_at, 'name', names)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)

    return len(names), sum(1 for n in names if n)


def main():
    p = argparse.ArgumentParser(description='Add player names to a CSV of player ids using pybaseball')
    p.add_argument('--input', '-i', default='/Users/christopherhsu/projects/hitterarchetypes/data/raw/unique_batters.csv', help='Input CSV with id column')
    p.add_argument('--output', '-o', default='/Users/christopherhsu/projects/hitterarchetypes/data/raw/unique_batters_with_names.csv', help='Output CSV path')
    p.add_argument('--id-col', default='batter', help='Column name containing player ids')
    p.add_argument('--id-type', default='mlbam', help='Type of id (e.g. mlbam, bbref) - passed to pybaseball lookup as key_type')
    p.add_argument('--sleep', type=float, default=0.1, help='Seconds to sleep between lookups (politeness)')
    args = p.parse_args()

    try:
        total, found = add_names(args.input, args.output, args.id_col, args.id_type, args.sleep)
    except Exception as e:
        print('Error:', e, file=sys.stderr)
        sys.exit(2)

    print(f'Processed {total} ids, found {found} names. Wrote to {args.output}')


if __name__ == '__main__':
    main()
