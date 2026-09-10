"""Descriptive data for the DFS Player Lookup panel.

This is a *research / eyeball* tool, not a projection model. The coverage-matchup
projection edge did not validate (see matchup_model/backtest_report.md), so nothing here
feeds the optimizer or claims to beat the FantasyPoints projection. It surfaces:
  - the player's recent usage (route %, target rate, snap share, aDOT, alignment, RZ looks)
  - expected vs actual fantasy points (regression flag)
  - historical efficiency splits by coverage (2022-25)
  - the opponent defense's scheme tendencies (most recent full season in the data)
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

try:
    from matchup_model import ingest as _mi
    from matchup_model import player_splits as _ps
    from matchup_model import defense_model as _dm
    from matchup_model import project as _pj
    from matchup_model import project_stats as _pjs
    _OK = True
except Exception:  # matchup_model data / deps missing
    _OK = False

from dfs.names import normalize_name, norm_team

RECENT_WEEKS = 6


def available() -> bool:
    return _OK and not _mi.receiving_weekly().empty


@lru_cache(maxsize=1)
def _weekly_all() -> pd.DataFrame:
    frames = []
    for f, unit in ((_mi.receiving_weekly, "routes"), (_mi.rushing_weekly, "att"),
                    (_mi.passing_weekly, "dropbacks")):
        d = f()
        if not d.empty:
            d = d.copy()
            d["opp_unit"] = d.get(unit)
            frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _latest_season() -> int:
    w = _weekly_all()
    return int(w["season"].max()) if not w.empty else 0


def recent_usage(name_key: str) -> pd.DataFrame:
    """Week-by-week usage for the player's most recent weeks in the data."""
    if not _OK:
        return pd.DataFrame()
    w = _weekly_all()
    g = w[w["name_key"] == name_key].sort_values(["season", "week"]).tail(RECENT_WEEKS)
    if g.empty:
        return g
    show = ["season", "week", "opp", "opp_unit", "route_pct", "tprr", "adot",
            "slot_pct", "wide_pct", "eztgt", "yprr", "succ_pct", "epa_a",
            "fp", "xfp", "diff"]
    out = g[[c for c in show if c in g.columns]].copy()
    rename = {"opp_unit": "routes/att", "route_pct": "route%", "tprr": "tgt/route",
              "slot_pct": "slot%", "wide_pct": "wide%", "eztgt": "EZ tgt",
              "succ_pct": "succ%", "diff": "FP-XFP"}
    out = out.rename(columns=rename).round(2)
    keep_id = {"season", "week", "opp"}
    return out[[c for c in out.columns if c in keep_id or out[c].notna().any()]]


def usage_summary(name_key: str) -> dict:
    """One-line trailing averages + a regression flag from FP vs XFP."""
    if not _OK:
        return {}
    w = _weekly_all()
    g = w[w["name_key"] == name_key].sort_values(["season", "week"]).tail(RECENT_WEEKS)
    if g.empty:
        return {}
    d = {"games": len(g)}
    for c, lbl in [("route_pct", "route%"), ("tprr", "tgt/route"), ("adot", "aDOT"),
                   ("slot_pct", "slot%"), ("opp_unit", "routes/att"), ("fp", "FP/g"),
                   ("xfp", "xFP/g")]:
        if c in g.columns and g[c].notna().any():
            d[lbl] = round(float(g[c].mean()), 2)
    if "fp" in g and "xfp" in g and g["xfp"].notna().any():
        diff = float((g["fp"] - g["xfp"]).mean())
        d["FP - xFP"] = round(diff, 2)
        d["regression_flag"] = ("overperforming expected — fade risk" if diff > 2
                                else "underperforming expected — bounce-back" if diff < -2
                                else "in line with expected")
    return d


def coverage_splits(name_key: str, stat: str = "tprr") -> pd.DataFrame:
    """Historical (2022-25) efficiency by coverage split. Descriptive only."""
    if not _OK:
        return pd.DataFrame()
    try:
        df = _ps.best_spots(name_key, stat)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    return df.rename(columns={"eff_overall": "overall", "eff": "vs split",
                              "delta": "diff", "n": "sample"}).round(3)


def opponent_scheme(opp_team: str) -> pd.DataFrame:
    """The opponent defense's coverage tendencies — most recent full season in the data."""
    if not _OK:
        return pd.DataFrame()
    team = norm_team(opp_team)
    cm = _mi.coverage_matrix()
    if cm.empty or team not in set(cm["team"]):
        return pd.DataFrame()
    yr = int(cm[cm["team"] == team]["season"].max())
    return _dm.scheme_table(team, yr, 99).assign(season=yr).round(1)


def opponent_season(opp_team: str) -> int:
    if not _OK:
        return 0
    cm = _mi.coverage_matrix()
    t = norm_team(opp_team)
    sub = cm[cm["team"] == t]
    return int(sub["season"].max()) if not sub.empty else 0


def regression_lean(name_key: str, pos: str) -> dict:
    """The one backtested adjustment: nudge toward recent expected FP. Needs current-season
    game logs in data/dfs/matchup/ — returns lean=0 with a note otherwise."""
    if not _OK:
        return {"lean": 0.0, "reason": "matchup_model unavailable"}
    return _pj.projection_lean(name_key, pos)


def projected_line(name_key: str, pos: str) -> dict:
    """Reconstructed box-score line + DK points from trailing usage x efficiency.
    Descriptive cross-check, not a market projection. {"fp": None, "reason": ...} if thin."""
    if not _OK:
        return {"fp": None, "reason": "matchup_model unavailable"}
    return _pjs.projected_line(name_key, pos)


def model_projection(fp_proj: float, name_key: str, pos: str) -> dict:
    """FantasyPoints projection + the backtested regression lean = our number.
    Returns {"proj": float, "lean": float, "reason": str}."""
    lean = regression_lean(name_key, pos) if _OK else {"lean": 0.0, "reason": ""}
    lv = float(lean.get("lean", 0.0) or 0.0)
    return {"proj": round(float(fp_proj) + lv, 1), "lean": lv, "reason": lean.get("reason", "")}

