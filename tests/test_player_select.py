import sys
import pathlib
import pandas as pd

# Ensure repo root is on sys.path so tests can import utils
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.player_select import build_player_choices, build_player_summary


def test_build_player_choices_basic():
    df = pd.DataFrame({'Name': [' Alice ', 'Bob', 'alice', None, '']})
    out = build_player_choices(df, name_col='Name', sort=True)
    assert out == ['Alice', 'alice', 'Bob'] or out == ['alice', 'Alice', 'Bob']


def test_build_player_summary_season():
    df = pd.DataFrame({
        'Name': ['Alice', 'Alice', 'Bob'],
        'Season': [2020, 2022, 2021]
    })
    s = build_player_summary(df, name_col='Name', season_col='Season')
    assert s['Alice'][0] == 2022 and s['Alice'][1] == 2
    assert s['Bob'][0] == 2021 and s['Bob'][1] == 1
