"""Per player-week actual results — the backtest targets.

The weekly basic/advanced FantasyPoints exports already contain the realised stats, so this
is just a union + column tidy.
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from matchup_model import ingest

_KEEP = ["season", "week", "name_key", "name", "pos", "team", "opp", "fp"]


@lru_cache(maxsize=1)
def actuals() -> pd.DataFrame:
    """season, week, name_key, pos, team, opp, fp, and the stat lines that exist per group:
    rec/rec_yds/rec_td (WR/TE), rush_yds/rush_td/att (RB), pass_yds/pass_td/int (QB)."""
    frames = []
    rw = ingest.receiving_weekly()
    if not rw.empty:
        frames.append(rw[[c for c in _KEEP + ["rec", "rec_yds", "rec_td", "routes", "tgt"] if c in rw.columns]])
    ruw = ingest.rushing_weekly()
    if not ruw.empty:
        frames.append(ruw[[c for c in _KEEP + ["att", "rush_yds", "rush_td", "rush_rec", "rush_rec_yds"] if c in ruw.columns]])
    pw = ingest.passing_weekly()
    if not pw.empty:
        frames.append(pw[[c for c in _KEEP + ["att", "cmp", "pass_yds", "pass_td", "int", "dropbacks"] if c in pw.columns]])
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["pos"] = out["pos"].fillna("").str.upper()
    # one row per player-week-pos (a RB can appear in receiving + rushing files — merge)
    num = out.select_dtypes("number").columns
    agg = {c: "sum" for c in num if c not in ("season", "week")}
    agg["name"] = "first"
    out = out.groupby(["season", "week", "name_key", "pos", "team", "opp"], as_index=False).agg(agg)
    return out
