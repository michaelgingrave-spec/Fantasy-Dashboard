"""Regression-to-expected lean — the one adjustment that survived backtesting.

A player's recent *expected* fantasy points (xFP, from FantasyPoints' advanced tables)
predicts next week slightly better than their recent *actual* FP. Blending the two — or
equivalently, nudging the actual-FP trend toward the xFP trend — improves ranking,
especially for QBs (Spearman 0.447 -> 0.460; top-decile of the lean beats the bottom by
~3 fp). Effect is real but small for WR/RB (~1 fp). See backtest_report.md §"regression".

This needs current-season weekly game logs. The app has them only if weekly FantasyPoints
exports are dropped into data/dfs/matchup/ (the ingest globs by year, so newer files just
work). Absent that, `projection_lean` returns lean=0 with reason="no recent game logs".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from matchup_model import ingest

XFP_WEIGHT = 0.5           # lean = XFP_WEIGHT * (xfp_trail - fp_trail); backtest-insensitive 0.35-0.65
EWMA_HALFLIFE = 4
MIN_GAMES = 4
_POS_GROUP = {"WR": "rec", "TE": "rec", "RB": "rush", "QB": "pass"}


def _ewma(v: np.ndarray, hl: float = EWMA_HALFLIFE) -> float:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan
    w = 0.5 ** (np.arange(len(v))[::-1] / hl)
    return float((w * v).sum() / w.sum())


def _weekly(pos: str) -> pd.DataFrame:
    g = _POS_GROUP.get((pos or "").upper())
    if g == "rec":
        return ingest.receiving_weekly().loc[:, ~ingest.receiving_weekly().columns.duplicated()]
    if g == "rush":
        return ingest.rushing_weekly()
    if g == "pass":
        return ingest.passing_weekly()
    return pd.DataFrame()


def projection_lean(name_key: str, pos: str, as_of_season: int | None = None,
                    as_of_week: int | None = None) -> dict:
    """{lean, fp_trail, xfp_trail, n_games, reason}. `lean` is fantasy points to add to a
    projection (small: roughly -4..+4). Positive = under-performed expected lately -> bounce-back."""
    d = _weekly(pos)
    empty = {"lean": 0.0, "fp_trail": None, "xfp_trail": None, "n_games": 0,
             "reason": "no recent game logs (drop current-season weekly exports into data/dfs/matchup/)"}
    if d.empty or "xfp" not in d.columns:
        return empty
    h = d[d["name_key"] == name_key]
    if as_of_season is not None:
        h = h[(h["season"] < as_of_season) |
              ((h["season"] == as_of_season) & (h["week"] < (as_of_week or 99)))]
    h = h.sort_values(["season", "week"])
    if len(h) < MIN_GAMES or h["xfp"].notna().sum() < MIN_GAMES:
        return empty
    fp_t = _ewma(h["fp"].to_numpy())
    xfp_t = _ewma(h["xfp"].to_numpy())
    # a player's own long-run FP−xFP gap is often structural, not luck (e.g. rushing QBs
    # always outscore xFP). Only fade the *excess* over their own norm, lightly shrunk to 0.
    gap = h["fp"].to_numpy() - h["xfp"].to_numpy()
    gap = gap[np.isfinite(gap)]
    own_gap = float(np.mean(gap)) * (len(gap) / (len(gap) + 20)) if len(gap) else 0.0
    lean = XFP_WEIGHT * ((xfp_t + own_gap) - fp_t)
    norm = "" if abs(own_gap) < 1 else f" (own norm FP{own_gap:+.1f} vs xFP)"
    if lean > 1.0:
        reason = f"recent FP {fp_t:.1f} < expected {xfp_t + own_gap:.1f}{norm} — bounce-back lean"
    elif lean < -1.0:
        reason = f"recent FP {fp_t:.1f} > expected {xfp_t + own_gap:.1f}{norm} — regression/fade lean"
    else:
        reason = f"recent FP {fp_t:.1f} ≈ expected {xfp_t + own_gap:.1f}{norm} — in line"
    return {"lean": round(float(lean), 2), "fp_trail": round(float(fp_t), 1),
            "xfp_trail": round(float(xfp_t), 1), "n_games": int(len(h)), "reason": reason}
