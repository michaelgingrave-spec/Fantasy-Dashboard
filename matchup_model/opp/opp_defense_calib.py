"""Cheap first check on "is there edge in opponent efficiency, even without scheme
granularity?" — before building anything on top of the coverage/concept splits.

Idea: opp_line()'s ypt/ypc is a player's trailing efficiency with ZERO opponent
conditioning — same number walk into a tough or soft defense. This asks: does simply
knowing "this defense allows X% more/less yards per target|carry than league average"
(trailing, walk-forward, no leakage) improve the projection, and by how much?

Method, mirroring opp/backtest.py's own validation style:
  1. defense_allowed(): per (season, week, defense) yards allowed per target (receiving)
     and per carry (rushing), summed across every offensive player who faced them —
     derived straight from the same nflverse player_weeks() the live model already uses.
  2. For each defense, at each (season, week), a trailing EWMA of that allowed rate
     (last 10 games, half-life 4 — same constants as the rest of opp/), strictly prior
     games only, ratio'd to a league average computed from EARLIER SEASONS only.
  3. adjusted_yds = base_yds * (1 + beta*(factor-1)), beta fit by OLS on train seasons
     (2023-24), judged on a 2025 holdout the fit never saw — same split as opp/backtest.py.
  4. Compare RMSE/MAE of base vs adjusted on the yardage number itself (not full DK
     points — TD variance would dilute a yardage-specific signal).

    python -m matchup_model.opp.opp_defense_calib
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from matchup_model.opp import data as D
from matchup_model.opp import model as M

TRAIN_SEASONS = [2023, 2024]
HOLDOUT_SEASON = 2025
TEST_WEEKS = range(4, 19)
MIN_GAMES, LOOKBACK, HALFLIFE = 3, 10, 4

REPORT = Path(__file__).resolve().parents[1] / "opp_defense_calibration.md"
ROWS_CSV = Path(__file__).resolve().parents[1] / "opp_defense_calibration_rows.csv"


def _ewma(v: np.ndarray) -> float:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)][-LOOKBACK:]
    if v.size == 0:
        return np.nan
    w = 0.5 ** (np.arange(v.size)[::-1] / HALFLIFE)
    return float((w * v).sum() / w.sum())


def defense_allowed(pw: pd.DataFrame) -> pd.DataFrame:
    """Per (season, week, defense), yards allowed per target / per carry — summed
    across every offensive player who faced that defense that week."""
    g = (pw.groupby(["season", "week", "opponent_team"])
           .agg(rec_yds=("receiving_yards", "sum"), tgt=("targets", "sum"),
                rush_yds=("rushing_yards", "sum"), car=("carries", "sum"))
           .reset_index().rename(columns={"opponent_team": "defense"}))
    g["ypt_allowed"] = g.rec_yds / g.tgt.replace(0, np.nan)
    g["ypc_allowed"] = g.rush_yds / g.car.replace(0, np.nan)
    return g.sort_values(["defense", "season", "week"])


def _league_avg(dg: pd.DataFrame, col: str, before_season: int) -> float:
    d = dg[dg.season < before_season]
    if d.empty:
        d = dg
    return float(d[col].mean())


def _trailing_factor(dg: pd.DataFrame, defense: str, season: int, week: int,
                     col: str, league_avg: float) -> float:
    """This defense's trailing allowed-rate / league average, strictly prior games only.
    1.0 (neutral) when there isn't enough sample yet."""
    h = dg[(dg.defense == defense) &
          ((dg.season < season) | ((dg.season == season) & (dg.week < week)))].tail(LOOKBACK)
    if len(h) < MIN_GAMES or not np.isfinite(league_avg) or league_avg <= 0:
        return 1.0
    val = _ewma(h[col].to_numpy())
    return val / league_avg if np.isfinite(val) else 1.0


def run() -> pd.DataFrame:
    pw = M._pw_with_team()
    dg = defense_allowed(D.player_weeks())
    t0 = time.time()
    rows = []
    for season in TRAIN_SEASONS + [HOLDOUT_SEASON]:
        lg_ypt = _league_avg(dg, "ypt_allowed", season)
        lg_ypc = _league_avg(dg, "ypc_allowed", season)
        sl = pw[(pw.season == season) & (pw.week.isin(TEST_WEEKS)) &
               (pw.position.isin(["WR", "TE", "RB"]))]
        sl = sl[(sl.targets + sl.carries) >= 2]
        for _, r in sl.iterrows():
            pid, pos, wk, opp = r["player_id"], r["position"], int(r["week"]), r["opponent_team"]
            o = M.opp_line(pid, pos, season, wk, by="id", injury_adj=True)
            if o.get("dk_fp") is None:
                continue
            line = o["line"]
            if pos in ("WR", "TE"):
                base = line.get("rec_yds")
                actual = float(r["receiving_yards"])
                factor = _trailing_factor(dg, opp, season, wk, "ypt_allowed", lg_ypt)
                axis = "rec_yds"
            else:
                base = line.get("rush_yds")
                actual = float(r["rushing_yards"])
                factor = _trailing_factor(dg, opp, season, wk, "ypc_allowed", lg_ypc)
                axis = "rush_yds"
            if base is None or not np.isfinite(base):
                continue
            rows.append(dict(season=season, week=wk, player=r["player_display_name"],
                             pos=pos, axis=axis, defense=opp, base=float(base),
                             factor=round(float(factor), 4), actual=actual))
        print(f"  {season}: {len(rows)} rows ({time.time()-t0:.0f}s)")
    return pd.DataFrame(rows)


def _rmse(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2))) if m.any() else np.nan


def _mae(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.mean(np.abs(a[m] - b[m]))) if m.any() else np.nan


def _fit_beta(base, actual, factor) -> float:
    """OLS: minimise RMSE of base*(1+beta*(factor-1)) vs actual. Closed form — projects
    the residual onto the candidate delta direction, same style as opp/backtest._fit_blend."""
    d = (base * (factor - 1)).to_numpy()
    y = (actual - base).to_numpy()
    denom = float(np.dot(d, d))
    return float(np.dot(d, y) / denom) if denom > 0 else 0.0


def report(df: pd.DataFrame) -> str:
    L = ["# Opponent-efficiency-allowed calibration — cheap check before scheme splits", ""]
    L.append(f"- {len(df)} player-games, seasons {TRAIN_SEASONS}+{HOLDOUT_SEASON}, "
             f"wk {TEST_WEEKS.start}-{TEST_WEEKS.stop-1}.")
    L.append("- `base` = opp_line's own rec_yds/rush_yds (trailing usage x efficiency, "
             "ZERO opponent conditioning today). `factor` = this defense's trailing "
             "allowed-yards-per-target(carry) / league average (>1 = soft matchup, <1 = "
             "tough), strictly walk-forward. `adjusted` = base*(1+beta*(factor-1)), beta "
             f"fit by OLS on {TRAIN_SEASONS} only, judged on a {HOLDOUT_SEASON} holdout "
             "it never saw.")
    L.append("")
    for axis, label in [("rec_yds", "WR/TE receiving yards"), ("rush_yds", "RB rushing yards")]:
        g = df[df.axis == axis]
        train = g[g.season.isin(TRAIN_SEASONS)]
        hold = g[g.season == HOLDOUT_SEASON]
        if len(train) < 100 or len(hold) < 50:
            L.append(f"## {label} — insufficient rows, skipped"); L.append(""); continue
        beta = _fit_beta(train.base, train.actual, train.factor)
        hold = hold.assign(adjusted=hold.base * (1 + beta * (hold.factor - 1)))
        train_adj = train.assign(adjusted=train.base * (1 + beta * (train.factor - 1)))
        L.append(f"## {label}  (n={len(g)}: {len(train)} train, {len(hold)} holdout)")
        L.append("")
        L.append(f"**fitted beta: {beta:+.3f}**  "
                 f"({'defense signal moved the fit toward using it' if abs(beta) > 0.05 else 'fit landed near zero — no real signal found'})")
        L.append("")
        L.append("| slice | n | RMSE base | RMSE adjusted | Δ | MAE base | MAE adjusted | Δ |")
        L.append("|---|--:|--:|--:|--:|--:|--:|--:|")
        for lbl, sub in [(f"train {TRAIN_SEASONS}", train_adj), (f"HOLDOUT {HOLDOUT_SEASON}", hold)]:
            rb, ra = _rmse(sub.base, sub.actual), _rmse(sub.adjusted, sub.actual)
            mb, ma = _mae(sub.base, sub.actual), _mae(sub.adjusted, sub.actual)
            L.append(f"| {lbl} | {len(sub)} | {rb:.2f} | {ra:.2f} | **{rb-ra:+.2f}** | "
                     f"{mb:.2f} | {ma:.2f} | **{mb-ma:+.2f}** |")
        L.append("")
        # split the holdout by how extreme the matchup is — is the edge concentrated
        # in real mismatches (factor far from 1.0), same style as the ATTD dev-tercile cut
        hold["dev"] = (hold.factor - 1).abs()
        try:
            hold["_t"] = pd.qcut(hold["dev"], 3, labels=["soft/tough closest to avg", "mid", "biggest mismatch"], duplicates="drop")
            L.append("By how extreme the matchup is (holdout only):")
            L.append("")
            L.append("| tercile | n | factor range | RMSE base | RMSE adjusted | Δ |")
            L.append("|---|--:|---|--:|--:|--:|")
            for t in hold["_t"].cat.categories:
                s = hold[hold._t == t]
                if s.empty:
                    continue
                rb, ra = _rmse(s.base, s.actual), _rmse(s.adjusted, s.actual)
                L.append(f"| {t} | {len(s)} | {s.factor.min():.2f}-{s.factor.max():.2f} | "
                         f"{rb:.2f} | {ra:.2f} | **{rb-ra:+.2f}** |")
            L.append("")
        except ValueError:
            pass
        rb_h, ra_h = _rmse(hold.base, hold.actual), _rmse(hold.adjusted, hold.actual)
        if ra_h < rb_h - 0.5:
            v = "**real edge — worth wiring into opp_line()**"
        elif ra_h < rb_h - 0.05:
            v = "**small but real improvement — marginal, keep it in mind**"
        else:
            v = "**no improvement on the holdout — opponent-efficiency-allowed alone isn't enough signal here**"
        L.append(f"Verdict: {v}")
        L.append("")
    return "\n".join(L)


def main():
    df = run()
    if df.empty:
        print("no rows"); return
    df.to_csv(ROWS_CSV, index=False)
    txt = report(df)
    REPORT.write_text(txt, encoding="utf-8")
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("\n" + txt + f"\n\nwrote {REPORT}")


if __name__ == "__main__":
    main()
