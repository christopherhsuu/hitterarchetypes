#!/usr/bin/env python3
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

        for row in reader:
            b = row.get(batter_col)
            p = row.get(pitcher_col)
            if b:
                batters.add(b)
            if p:
                pitchers.add(p)

    batters = sorted(batters)
    pitchers = sorted(pitchers)

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
    p.add_argument('--input', '-i', required=True, help='Input CSV file')
    p.add_argument('--batters-out', required=True, help='Output CSV for batters')
    p.add_argument('--pitchers-out', required=True, help='Output CSV for pitchers')
    args = p.parse_args()

    try:
        num_batters, num_pitchers = extract_uniques(args.input, args.batters_out, args.pitchers_out)
    except Exception as e:
        print('Error:', e, file=sys.stderr)
        sys.exit(2)

    print(f'Wrote {num_batters} unique batters to {args.batters_out}')
    print(f'Wrote {num_pitchers} unique pitchers to {args.pitchers_out}')


if __name__ == '__main__':
    main()
