"""Does a defense's personnel tendency (nickel/dime sub-package rate vs base) predict
receiving efficiency, on top of the trailing-usage model? Same question shape as
opp_defense_calib.py (yards-allowed) and scheme_calib.py (man/zone) — this is the third
and last opponent-conditioning axis with enough real historical depth to backtest
properly. Press-rate was also requested but isn't trackable: it's uniformly 0% in the
Data Suite's coverage-matrix for 2022-2024, only populated starting 2025 — one season,
no walk-forward holdout possible, so it's not run here.

Source: data/dfs/matchup/coverage-matrix_{season}_week.csv (BASE %/NICKEL %/DIME %),
already committed at weekly granularity 2022-2025 — no new data needed.

Method — identical to opp_defense_calib.py, only the axis changes:
  1. sub_pct = 100 - BASE% (fraction of snaps in nickel/dime, i.e. "pass-funnel" personnel),
     trailing EWMA per defense, walk-forward, vs a league average from earlier seasons only.
  2. adjusted = base_rec_yds * (1 + beta*(factor-1)), factor = trailing sub_pct / league avg,
     beta fit by OLS on 2022-24, judged on a 2025 holdout it never saw.

    python -m matchup_model.opp.personnel_calib
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from dfs.names import norm_team
from matchup_model.config import DATA as MATCHUP_DATA
from matchup_model.opp import model as M

TRAIN_SEASONS = [2022, 2023, 2024]
HOLDOUT_SEASON = 2025
TEST_WEEKS = range(4, 19)
MIN_GAMES, LOOKBACK, HALFLIFE = 3, 10, 4

REPORT = Path(__file__).resolve().parents[1] / "opp_personnel_calibration.md"
ROWS_CSV = Path(__file__).resolve().parents[1] / "opp_personnel_calibration_rows.csv"


def _ewma(v: np.ndarray) -> float:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)][-LOOKBACK:]
    if v.size == 0:
        return np.nan
    w = 0.5 ** (np.arange(v.size)[::-1] / HALFLIFE)
    return float((w * v).sum() / w.sum())


def load_personnel() -> pd.DataFrame:
    frames = []
    for season in TRAIN_SEASONS + [HOLDOUT_SEASON]:
        p = MATCHUP_DATA / f"coverage-matrix_{season}_week.csv"
        d = pd.read_csv(p, header=1)
        d = d[d["Name"].notna() & (d["Name"] != "League Avg")]
        d = d.rename(columns={"Name": "team_full", "WEEK": "week", "BASE %": "base_pct"})
        d = d[["team_full", "week", "base_pct"]].copy()
        d["season"] = season
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["defense"] = df["team_full"].map(norm_team)
    df["week"] = pd.to_numeric(df["week"], errors="coerce")
    df["base_pct"] = pd.to_numeric(df["base_pct"], errors="coerce")
    df["sub_pct"] = 100.0 - df["base_pct"]     # nickel+dime — "pass funnel" personnel
    return df.dropna(subset=["week"]).sort_values(["defense", "season", "week"])


def _league_avg(dg: pd.DataFrame, before_season: int) -> float:
    d = dg[dg.season < before_season]
    if d.empty:
        d = dg
    return float(d["sub_pct"].mean())


def _trailing_factor(dg: pd.DataFrame, defense: str, season: int, week: int,
                     league_avg: float) -> float:
    h = dg[(dg.defense == defense) &
          ((dg.season < season) | ((dg.season == season) & (dg.week < week)))].tail(LOOKBACK)
    if len(h) < MIN_GAMES or not np.isfinite(league_avg) or league_avg <= 0:
        return 1.0
    val = _ewma(h["sub_pct"].to_numpy())
    return val / league_avg if np.isfinite(val) else 1.0


def run() -> pd.DataFrame:
    pw = M._pw_with_team()
    dg = load_personnel()
    print(f"  loaded {len(dg)} defense-week rows")
    t0 = time.time()
    rows = []
    for season in TRAIN_SEASONS + [HOLDOUT_SEASON]:
        lg = _league_avg(dg, season)
        sl = pw[(pw.season == season) & (pw.week.isin(TEST_WEEKS)) &
               (pw.position.isin(["WR", "TE"]))]
        sl = sl[sl.targets >= 2]
        for _, r in sl.iterrows():
            pid, wk, opp = r["player_id"], int(r["week"]), r["opponent_team"]
            o = M.opp_line(pid, r["position"], season, wk, by="id", injury_adj=True)
            if o.get("dk_fp") is None:
                continue
            base = o["line"].get("rec_yds")
            if base is None or not np.isfinite(base):
                continue
            factor = _trailing_factor(dg, opp, season, wk, lg)
            rows.append(dict(season=season, week=wk, player=r["player_display_name"],
                             pos=r["position"], defense=opp, base=float(base),
                             factor=round(float(factor), 4),
                             actual=float(r["receiving_yards"])))
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
    d = (base * (factor - 1)).to_numpy()
    y = (actual - base).to_numpy()
    denom = float(np.dot(d, d))
    return float(np.dot(d, y) / denom) if denom > 0 else 0.0


def report(df: pd.DataFrame) -> str:
    train = df[df.season.isin(TRAIN_SEASONS)]
    hold = df[df.season == HOLDOUT_SEASON]
    L = ["# Defensive personnel (nickel/dime rate) vs receiving yards", ""]
    L.append(f"- {len(df)} WR/TE player-games ({len(train)} train {TRAIN_SEASONS}, "
             f"{len(hold)} holdout {HOLDOUT_SEASON}), wk {TEST_WEEKS.start}-{TEST_WEEKS.stop-1}.")
    L.append("- `factor` = defense's trailing (100-BASE%) [nickel+dime rate] / league "
             "average, walk-forward. `adjusted` = base*(1+beta*(factor-1)), beta fit by "
             f"OLS on {TRAIN_SEASONS}, judged on a {HOLDOUT_SEASON} holdout it never saw. "
             "Press-rate is NOT included — 0% populated in 2022-2024, no walk-forward "
             "holdout possible with one season of real data.")
    L.append("")
    if len(train) < 100 or len(hold) < 50:
        L.append("**insufficient rows — stopping here.**")
        return "\n".join(L)
    beta = _fit_beta(train.base, train.actual, train.factor)
    hold = hold.assign(adjusted=hold.base * (1 + beta * (hold.factor - 1)))
    train = train.assign(adjusted=train.base * (1 + beta * (train.factor - 1)))
    L.append(f"**fitted beta: {beta:+.3f}**")
    L.append("")
    L.append("| slice | n | RMSE base | RMSE adjusted | Δ | MAE base | MAE adjusted | Δ |")
    L.append("|---|--:|--:|--:|--:|--:|--:|--:|")
    for lbl, sub in [(f"train {TRAIN_SEASONS}", train), (f"HOLDOUT {HOLDOUT_SEASON}", hold)]:
        rb, ra = _rmse(sub.base, sub.actual), _rmse(sub.adjusted, sub.actual)
        mb, ma = _mae(sub.base, sub.actual), _mae(sub.adjusted, sub.actual)
        L.append(f"| {lbl} | {len(sub)} | {rb:.2f} | {ra:.2f} | **{rb-ra:+.2f}** | "
                 f"{mb:.2f} | {ma:.2f} | **{mb-ma:+.2f}** |")
    L.append("")
    hold["absdev"] = (hold.factor - 1).abs()
    try:
        hold["_t"] = pd.qcut(hold["absdev"], 3, labels=["closest to avg", "mid", "biggest mismatch"],
                             duplicates="drop")
        L.append("By how far from league-average personnel usage (holdout only):")
        L.append("")
        L.append("| tercile | n | RMSE base | RMSE adjusted | Δ |")
        L.append("|---|--:|--:|--:|--:|")
        for t in hold["_t"].cat.categories:
            s = hold[hold._t == t]
            if s.empty:
                continue
            rb, ra = _rmse(s.base, s.actual), _rmse(s.adjusted, s.actual)
            L.append(f"| {t} | {len(s)} | {rb:.2f} | {ra:.2f} | **{rb-ra:+.2f}** |")
        L.append("")
    except ValueError:
        pass
    rb_h, ra_h = _rmse(hold.base, hold.actual), _rmse(hold.adjusted, hold.actual)
    if ra_h < rb_h - 0.5:
        v = "**real edge**"
    elif ra_h < rb_h - 0.05:
        v = "**small improvement — marginal**"
    else:
        v = "**no improvement**"
    L.append(f"Verdict: {v}")
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
