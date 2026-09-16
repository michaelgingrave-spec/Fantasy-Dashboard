"""Real accuracy check, not a backtest: for a completed week, how did our own live
projection (what the app actually showed pregame) compare to FantasyPoints' own weekly
projection, against what really happened?

Reuses the exact same projection call the app makes (`blend.blended_line`, walk-forward —
only data strictly before the target week) and the exact FantasyPoints export file the app
reads (`dfs/projections.week{N}.csv`), so this is judging the real historical calls, not a
simulation.

    python -m matchup_model.opp.week_accuracy 2026 1
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from dfs.names import normalize_name
from dfs.projections import load_weekly_projections
from matchup_model.opp import data as D
from matchup_model.opp import blend as B


def run(season: int, week: int) -> pd.DataFrame:
    pw = D.player_weeks()
    actual = pw[(pw.season == season) & (pw.week == week) &
               (pw.position.isin(["QB", "RB", "WR", "TE"]))].copy()
    actual = actual[(actual.targets + actual.carries + actual.attempts) >= 1]
    actual["name_key"] = actual["player_display_name"].map(normalize_name)

    try:
        fpj = load_weekly_projections(week)
        fp_map = {normalize_name(n): float(p) for n, p in zip(fpj["name"], fpj["proj"])}
    except Exception as e:  # noqa: BLE001
        print(f"no FantasyPoints projections file for week {week}: {e}")
        fp_map = {}

    rows = []
    for _, r in actual.iterrows():
        nk, pos = r["name_key"], r["position"]
        bl = B.blended_line(nk, pos, as_of_season=season, as_of_week=week)
        ours = bl.get("fp")
        fp = fp_map.get(nk)
        if ours is None and fp is None:
            continue
        rows.append(dict(player=r["player_display_name"], pos=pos, team=r["team"],
                         actual=float(r["dk_fp"]), ours=ours, fantasypoints=fp))
    return pd.DataFrame(rows)


def _rmse(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2))) if m.any() else np.nan


def _mae(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.mean(np.abs(a[m] - b[m]))) if m.any() else np.nan


def _spearman(a, b):
    s = pd.DataFrame({"a": a, "b": b}).dropna()
    return float(s["a"].corr(s["b"], method="spearman")) if len(s) > 5 else np.nan


def report(df: pd.DataFrame, season: int, week: int) -> str:
    L = [f"# Week {week} ({season}) projection accuracy — ours vs FantasyPoints vs actual", ""]
    both = df.dropna(subset=["ours", "fantasypoints"])
    L.append(f"- {len(df)} players with either projection, {len(both)} with BOTH "
             "(the fair, same-player comparison below).")
    L.append(f"- We had no number for {df['ours'].isna().sum()} players FantasyPoints "
             f"projected (usually: not enough trailing games — new to the league/team, "
             f"under our {3}-game minimum). FantasyPoints had no number for "
             f"{df['fantasypoints'].isna().sum()} we projected.")
    L.append("")
    if both.empty:
        L.append("**no overlapping players — nothing to compare.**")
        return "\n".join(L)
    L.append("## Head to head (same players only)")
    L.append("")
    L.append("| source | n | RMSE | MAE | rho (rank corr) |")
    L.append("|---|--:|--:|--:|--:|")
    for lbl, col in [("Ours", "ours"), ("FantasyPoints", "fantasypoints")]:
        r, m = _rmse(both[col], both.actual), _mae(both[col], both.actual)
        rho = _spearman(both[col], both.actual)
        L.append(f"| {lbl} | {len(both)} | {r:.2f} | {m:.2f} | {rho:.3f} |")
    L.append("")
    by_pos = []
    for pos in ["QB", "RB", "WR", "TE"]:
        g = both[both.pos == pos]
        if len(g) < 5:
            continue
        ro, ru = _rmse(g.ours, g.actual), _rmse(g.fantasypoints, g.actual)
        by_pos.append((pos, len(g), ro, ru))
    if by_pos:
        L.append("## By position (RMSE, lower better)")
        L.append("")
        L.append("| pos | n | ours | fantasypoints | edge |")
        L.append("|---|--:|--:|--:|---|")
        for pos, n, ro, ru in by_pos:
            edge = "ours" if ro < ru else "fantasypoints" if ru < ro else "tie"
            L.append(f"| {pos} | {n} | {ro:.2f} | {ru:.2f} | {edge} |")
        L.append("")
    r_ours, r_fp = _rmse(both.ours, both.actual), _rmse(both.fantasypoints, both.actual)
    if abs(r_ours - r_fp) < 0.3:
        v = "**essentially tied** — within noise for this sample size."
    elif r_ours < r_fp:
        v = f"**ours was more accurate** ({r_fp - r_ours:+.2f} RMSE better)."
    else:
        v = f"**FantasyPoints was more accurate** ({r_ours - r_fp:+.2f} RMSE worse for ours)."
    L.append(f"Verdict: {v}")
    return "\n".join(L)


def main():
    season = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    week = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    df = run(season, week)
    if df.empty:
        print("no rows"); return
    txt = report(df, season, week)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(txt)
    df.to_csv(f"matchup_model/week{week}_{season}_accuracy_rows.csv", index=False)


if __name__ == "__main__":
    main()
