"""Predict a defense's scheme tendencies for an upcoming game.

Base = recency-weighted average of that defense's recent games (coverage-matrix weekly).
Early season → back off to the same defensive coordinator's prior-season rate, then to the
league average. Output feeds `matchup.py` and the Player Lookup "projected scheme" table.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from matchup_model.config import (
    EWMA_HALFLIFE_GAMES,
    EWMA_LOOKBACK_GAMES,
    MIN_GAMES_FOR_TREND,
)
from matchup_model import ingest

# axes we predict for the nudge + extra rates we show in the UI
PRED_AXES = ["man", "two_high"]
DISPLAY_RATES = ["man", "zone", "one_high", "two_high", "press",
                 "cover0", "cover1", "cover2", "cover3", "cover4", "cover6"]


def _ewma(series: pd.Series) -> float:
    """Recency-weighted mean of the last N values (series ordered oldest→newest)."""
    v = series.dropna().to_numpy(dtype=float)[-EWMA_LOOKBACK_GAMES:]
    if len(v) == 0:
        return np.nan
    age = np.arange(len(v))[::-1]                       # 0 = newest
    w = 0.5 ** (age / EWMA_HALFLIFE_GAMES)
    return float(np.sum(w * v) / np.sum(w))


@lru_cache(maxsize=1)
def _league_avg() -> pd.DataFrame:
    cm = ingest.coverage_matrix()
    return cm.groupby("season")[DISPLAY_RATES].mean().reset_index()


@lru_cache(maxsize=1)
def _team_dc_by_season() -> dict:
    """(team, season) -> modal DC name, from the coordinator split file."""
    cp = ingest.coordinator_profiles()
    if cp.empty or "dc" not in cp.columns:
        return {}
    m = (cp.dropna(subset=["dc"]).groupby(["team", "season"])["dc"]
           .agg(lambda s: s.mode().iat[0] if not s.mode().empty else None))
    return m.to_dict()


def league_rate(season: int, rate: str) -> float:
    la = _league_avg()
    row = la[la.season == season]
    if row.empty:
        row = la
    return float(row[rate].mean())


def predict(team: str, season: int, week: int) -> dict:
    """Predicted rate per DISPLAY_RATES for `team`'s defense entering (season, week),
    plus `_n_games` used and `_source` ('trend' | 'coord' | 'league')."""
    cm = ingest.coverage_matrix()
    hist = cm[(cm.team == team) & (cm.season == season) & (cm.week < week)].sort_values("week")
    out = {}
    n = len(hist)
    src = "trend" if n >= MIN_GAMES_FOR_TREND else ("coord" if n == 0 else "blend")

    # prior-season same-DC fallback
    dc_map = _team_dc_by_season()
    prior_ok = dc_map.get((team, season)) is not None and \
        dc_map.get((team, season)) == dc_map.get((team, season - 1))
    prior = cm[(cm.team == team) & (cm.season == season - 1)] if prior_ok else pd.DataFrame()

    for r in DISPLAY_RATES:
        trend = _ewma(hist[r]) if n else np.nan
        pri = float(prior[r].mean()) if not prior.empty and r in prior else np.nan
        lg = league_rate(season, r)
        if np.isfinite(trend) and n >= MIN_GAMES_FOR_TREND:
            val = trend
        elif np.isfinite(trend) and np.isfinite(pri):
            k = MIN_GAMES_FOR_TREND
            val = (n * trend + k * pri) / (n + k)          # blend thin trend with last year
        elif np.isfinite(pri):
            val = pri
        elif np.isfinite(trend):
            val = trend
        else:
            val = lg
        out[r] = round(float(val), 2)
    out["_n_games"] = n
    out["_source"] = src if not (src == "coord" and prior.empty) else "league"
    return out


def scheme_table(team: str, season: int, week: int) -> pd.DataFrame:
    """UI table: predicted rate vs league average + lean, for the opponent this week."""
    p = predict(team, season, week)
    rows = []
    for r in DISPLAY_RATES:
        lg = round(league_rate(season, r), 2)
        rows.append({"look": r, "predicted_%": p[r], "league_%": lg,
                     "lean": round(p[r] - lg, 1)})
    return pd.DataFrame(rows)


def axis_lean(team: str, season: int, week: int) -> dict:
    """{axis: predicted_rate − league_avg} in rate points, for the nudge in matchup.py."""
    p = predict(team, season, week)
    return {ax: round(p[ax] - league_rate(season, ax), 2) for ax in PRED_AXES}
