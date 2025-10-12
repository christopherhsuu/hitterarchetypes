#!/usr/bin/env python3
"""
Read a swings CSV, extract unique values from 'batter' and 'pitcher' columns,
and write them to two CSV files.

Usage:
  python scripts/extract_uniques.py \
      --input data/2025_swings.csv \
      --batters-out data/unique_batters.csv \
      --pitchers-out data/unique_pitchers.csv
"""
import argparse
import csv
from pathlib import Path
import sys


def find_column(fieldnames, target):
    if not fieldnames:
        return None
    for fn in fieldnames:
        if fn.lower() == target.lower():
            return fn
    return None


def extract_uniques(input_path, batters_out, pitchers_out):
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    batters = set()
    pitchers = set()

    with input_path.open(newline='') as f:
        reader = csv.DictReader(f)
        batter_col = find_column(reader.fieldnames, 'batter')
        pitcher_col = find_column(reader.fieldnames, 'pitcher')

        if batter_col is None:
            raise ValueError("Could not find a 'batter' column in input CSV headers: " + str(reader.fieldnames))
        if pitcher_col is None:
            raise ValueError("Could not find a 'pitcher' column in input CSV headers: " + str(reader.fieldnames))

        for i, row in enumerate(reader, start=1):
            # Use raw values; caller can normalize if needed
            b = row.get(batter_col)
            p = row.get(pitcher_col)
            if b is not None and b != '':
                batters.add(b)
            if p is not None and p != '':
                pitchers.add(p)

    batters = sorted(batters)
    pitchers = sorted(pitchers)

    # Ensure output directory exists
    Path(batters_out).parent.mkdir(parents=True, exist_ok=True)
    Path(pitchers_out).parent.mkdir(parents=True, exist_ok=True)

    with open(batters_out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['batter'])
        for v in batters:
            writer.writerow([v])

    with open(pitchers_out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['pitcher'])
        for v in pitchers:
            writer.writerow([v])

    return len(batters), len(pitchers)


def main():
    p = argparse.ArgumentParser(description='Extract unique batters and pitchers from swings CSV')
    p.add_argument('--input', '-i', default='/Users/christopherhsu/projects/hitterarchetypes/data/raw/2025_swings.csv', help='Input CSV file')
    p.add_argument('--batters-out', default='/Users/christopherhsu/projects/hitterarchetypes/data/raw/unique_batters.csv', help='Output CSV for batters')
    p.add_argument('--pitchers-out', default='/Users/christopherhsu/projects/hitterarchetypes/data/raw/unique_pitchers.csv', help='Output CSV for pitchers')
    args = p.parse_args()

    try:
        nb, np_ = extract_uniques(args.input, args.batters_out, args.pitchers_out)
    except Exception as e:
        print('Error:', e, file=sys.stderr)
        sys.exit(2)

    print(f'Wrote {nb} unique batters to {args.batters_out}')
    print(f'Wrote {np_} unique pitchers to {args.pitchers_out}')


if __name__ == '__main__':
    main()
