from typing import List, Tuple
import pandas as pd


def build_player_choices(df: pd.DataFrame, name_col: str = 'Name', sort: bool = True) -> List[str]:
    """Return a deduplicated list of canonical player names.

    - df: DataFrame containing a player name column
    - name_col: column name to use
    - sort: whether to sort alphabetically (case-insensitive)

    Returns a list of strings (canonical names).
    """
    if name_col not in df.columns:
        return []
    # Drop true NA values first, then normalize strings
    names = df[name_col].dropna()
    names = names.astype(str).str.strip()
    # filter out empty-like strings
    bad = set(['', 'none', 'nan'])
    names = names[~names.str.lower().isin(bad)]
    uniq = names.drop_duplicates()
    if sort:
        uniq = uniq.sort_values(key=lambda s: s.str.lower())
    return uniq.tolist()


def build_player_summary(df: pd.DataFrame, name_col: str = 'Name', season_col: str = 'Season') -> dict:
    """Build a summary mapping canonical Name -> (last_seen_season, count)

    - Returns a dict where keys are canonical names and values are tuples (last_season, count)
    - last_season is the max of season_col for that player (if season_col exists), otherwise None
    """
    if name_col not in df.columns:
        return {}
    names = df[name_col].astype(str).str.strip()
    if season_col in df.columns:
        seasons = df[season_col]
        grp = pd.DataFrame({'name': names, 'season': seasons}).groupby('name')
        res = {}
        for name, g in grp:
            # compute last seen season if possible
            try:
                last = g['season'].dropna().astype(int).max()
            except Exception:
                last = None
            res[name] = (last, int(g.shape[0]))
        return res
    else:
        grp = names.groupby(names)
        return {name: (None, int(g.shape[0])) for name, g in grp}


def stable_widget_key(prefix: str, df: pd.DataFrame) -> str:
    """Return a stable widget key string derived from prefix and DataFrame contents.

    The function is pure: it does not mutate df. Key changes when row count or column names change.
    """
    cols = ','.join(map(str, df.columns.tolist()))
    # lightweight fingerprint: row count + columns
    return f"{prefix}:{len(df)}:{cols}"
