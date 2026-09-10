"""Descriptive data for the DFS Player Lookup panel.

This is a *research / eyeball* tool, not a projection model. The coverage-matchup
projection edge did not validate (see matchup_model/backtest_report.md), so nothing here
feeds the optimizer or claims to beat the FantasyPoints projection. It surfaces:
  - the player's recent usage (route %, target rate, snap share, aDOT, alignment, RZ looks)
  - expected vs actual fantasy points (regression flag)
  - efficiency splits by coverage over a user-chosen season window (2022-25)
  - the opponent defense's scheme tendencies for that same window
  - whether the player has faced the opponent's current DC before (any team, any year)
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
    from matchup_model import coordinators as _co
    from matchup_model import scheme as _sch
    _OK = True
except Exception:  # matchup_model data / deps missing
    _OK = False

from dfs.names import normalize_name, norm_team

RECENT_WEEKS = 6

# Window picker options for the splits / scheme tables -> the seasons each covers.
WINDOWS: dict[str, tuple[int, ...] | None] = {
    "2025": (2025,), "2024": (2024,), "2023": (2023,), "2022": (2022,),
    "2024–25": (2024, 2025), "2023–25": (2023, 2024, 2025), "All": None,
}


def window_seasons(label: str) -> tuple[int, ...] | None:
    return WINDOWS.get(label, None)


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


def coverage_splits(name_key: str, stat: str = "tprr",
                    seasons: tuple[int, ...] | None = None) -> pd.DataFrame:
    """Efficiency by coverage split over the chosen season window. Descriptive only."""
    if not _OK:
        return pd.DataFrame()
    try:
        df = _ps.best_spots(name_key, stat, seasons=seasons)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    return df.rename(columns={"eff_overall": "overall", "eff": "vs split",
                              "delta": "diff", "n": "sample"}).round(3)


def opponent_scheme(opp_team: str, seasons: tuple[int, ...] | None = None) -> pd.DataFrame:
    """Opponent defense's coverage rates, averaged over `seasons` (default: latest year
    in the data), each vs the league average for the same years."""
    if not _OK:
        return pd.DataFrame()
    team = norm_team(opp_team)
    cm = _mi.coverage_matrix()
    if cm.empty or team not in set(cm["team"]):
        return pd.DataFrame()
    sub = cm[cm["team"] == team]
    if seasons is not None:
        sub = sub[sub["season"].isin(seasons)]
    if sub.empty:
        return pd.DataFrame()
    if seasons is None:
        sub = sub[sub["season"] == sub["season"].max()]
    yrs = sorted(int(s) for s in sub["season"].unique())
    la = cm[cm["season"].isin(yrs)]
    rate_cols = [c for c in _mi.SCHEME_RATE_COLS if c in sub.columns and sub[c].notna().any()]
    rows = []
    for c in rate_cols:
        t_rate, l_rate = sub[c].mean(), la[c].mean()
        rows.append({"look": c, "team_%": round(float(t_rate), 1),
                     "league_%": round(float(l_rate), 1),
                     "lean": round(float(t_rate - l_rate), 1)})
    out = pd.DataFrame(rows)
    out.attrs["years"] = yrs
    return out


def opponent_season(opp_team: str) -> int:
    if not _OK:
        return 0
    cm = _mi.coverage_matrix()
    t = norm_team(opp_team)
    sub = cm[cm["team"] == t]
    return int(sub["season"].max()) if not sub.empty else 0


def vs_coordinator(name_key: str, opp_team: str) -> dict:
    """Has this player faced the opponent's current defensive coordinator before — at any
    team, any year in the data? Returns {dc, dc_since, games (DataFrame), summary (dict)}
    or {} when the DC isn't on record."""
    if not _OK:
        return {}
    cd = _co.current_dc(opp_team)
    if not cd:
        return {}
    spots = set(_co.dc_team_seasons(cd["dc"]))
    w = _weekly_all()
    if w.empty:
        return {"dc": cd["dc"], "dc_since": cd["season"], "games": pd.DataFrame(), "summary": {}}
    g = w[w["name_key"] == name_key].copy()
    if not g.empty:
        mask = pd.Series([(int(s), t) in spots for s, t in zip(g["season"], g["opp"])],
                         index=g.index)
        g = g[mask].sort_values(["season", "week"])
    show = [c for c in ["season", "week", "opp", "tgt", "rec", "rec_yds", "rec_td",
                        "att", "rush_yds", "rush_td", "pass_yds", "pass_td", "fp", "xfp"]
            if c in g.columns]
    games = g[show].copy()
    keep_id = {"season", "week", "opp"}
    games = games[[c for c in games.columns if c in keep_id or games[c].notna().any()]]
    summ = {"games": int(len(g))}
    if len(g):
        if "fp" in g:
            summ["fp_avg"] = round(float(g["fp"].mean()), 1)
        if "xfp" in g and g["xfp"].notna().any():
            summ["xfp_avg"] = round(float(g["xfp"].mean()), 1)
    return {"dc": cd["dc"], "dc_since": cd["season"], "games": games, "summary": summ}


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


# ── Matchup Machine: scheme-by-scheme offense vs defense ───────────────────
def scheme_available() -> bool:
    return _OK and _sch.available()


def pass_matchup(name_key: str, opp_team: str) -> dict:
    """{player: df by coverage, defense: df by coverage} for the passing grid."""
    if not scheme_available():
        return {}
    return {"player": _sch.player_pass_by_coverage(name_key),
            "defense": _sch.defense_pass_allowed_by_coverage(opp_team)}


def run_matchup(name_key: str, team: str, opp_team: str) -> dict:
    """{player, team_offense, defense} frames by run concept for the rushing grid."""
    if not scheme_available():
        return {}
    return {"player": _sch.player_run_by_concept(name_key),
            "team_offense": _sch.team_run_by_concept(team, "offense"),
            "defense": _sch.team_run_by_concept(opp_team, "defense")}


def defense_alignment_grid() -> pd.DataFrame:
    """All 32 defenses x yds/route allowed to wide/slot/inline/backfield."""
    return _sch.defense_alignment_grid() if scheme_available() else pd.DataFrame()


def heat(df: pd.DataFrame, cols, good_high=True):
    """Return a pandas Styler with `cols` shaded red→green by rank within each column
    (green = better for the offense; flip with good_high=False). No matplotlib needed."""
    cols = [c for c in cols if c in df.columns]

    def _style(col):
        s = pd.to_numeric(col, errors="coerce")
        lo, hi = s.min(), s.max()
        cells = []
        for v in s:
            if pd.isna(v) or hi == lo:
                cells.append("")
                continue
            t = (v - lo) / (hi - lo)
            if not good_high:
                t = 1.0 - t
            red = int(255 * min(1.0, 2 * (1 - t)))
            grn = int(255 * min(1.0, 2 * t))
            cells.append(f"background-color: rgba({red},{grn},110,0.32)")
        return cells

    sty = df.style
    for c in cols:
        sty = sty.apply(_style, subset=[c])
    return sty.format(precision=2)

