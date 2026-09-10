"""A deliberately simple base projection — the thing the matchup nudge sits on top of and
the thing the backtest must beat.

`base_fp`  : recency-weighted mean of the player's recent weekly fantasy points (the naive
             baseline in backtest.py).
`opp_*`    : trailing opportunity (routes / carries / dropbacks) used to scale the nudge.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from matchup_model.config import EWMA_HALFLIFE_GAMES, EWMA_LOOKBACK_GAMES
from matchup_model import ingest


def _ewma(v: np.ndarray) -> float:
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)][-EWMA_LOOKBACK_GAMES:]
    if len(v) == 0:
        return np.nan
    age = np.arange(len(v))[::-1]
    w = 0.5 ** (age / EWMA_HALFLIFE_GAMES)
    return float(np.sum(w * v) / np.sum(w))


@lru_cache(maxsize=4)
def _weekly_for(pos: str) -> pd.DataFrame:
    if pos in ("WR", "TE"):
        d = ingest.receiving_weekly().copy()
        d["opp_unit"] = d.get("routes")
    elif pos == "RB":
        d = ingest.rushing_weekly().copy()
        d["opp_unit"] = d.get("att")
    else:  # QB
        d = ingest.passing_weekly().copy()
        d["opp_unit"] = d.get("dropbacks")
    return d.sort_values(["name_key", "season", "week"]).reset_index(drop=True)


def base_row(name_key: str, pos: str, season: int, week: int) -> dict:
    """Trailing base projection for one player entering (season, week)."""
    d = _weekly_for(pos)
    hist = d[(d.name_key == name_key) & (
        (d.season < season) | ((d.season == season) & (d.week < week))
    )].sort_values(["season", "week"])
    if hist.empty:
        return {"base_fp": np.nan, "opp_unit": np.nan, "n_games": 0}
    return {
        "base_fp": _ewma(hist["fp"].to_numpy()),
        "opp_unit": _ewma(hist["opp_unit"].to_numpy()),
        "route_pct": _ewma(hist["route_pct"].to_numpy()) if "route_pct" in hist else np.nan,
        "yprr": _ewma(hist["yprr"].to_numpy()) if "yprr" in hist else np.nan,
        "n_games": int(len(hist)),
    }
