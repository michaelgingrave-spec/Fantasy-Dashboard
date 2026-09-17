"""Applies three things from the literature review to the ONE opponent signal that's
actually confirmed real this session (RB defense strength -- persists year-over-year and
within-season, see opp_defense_calibration.md and the team-persistence check), which has
never actually been wired into the live model. Every earlier RB test was about adding
granularity ON TOP of a raw, unadjusted defense-strength read; this instead cleans up the
raw signal itself, the way real opponent-adjustment metrics do:

  1. Schedule adjustment (DVOA / PFF's SAFPA): a defense that happened to face light
     workloads doesn't look as tough as one that faced bellcow backs, even at equal
     fp-allowed. Walk-forward-safe leave-one-out: each game's fp-allowed is compared to
     the OPPOSING OFFENSE'S OWN trailing RB pace *entering that game* (not a full-season
     average, which would leak future games) -- "did this offense do better or worse
     than they were already trending, against this defense."
  2. Empirical-Bayes shrinkage with a pseudo-count DERIVED from measured reliability, not
     hand-picked. Split-half (odd/even week) correlation on our own data averages ~0.265
     across 2021-2025 -- Tango's method (Spearman-Brown to a full ~17-game season, then
     n(1-r)/r) implies K ~= 24 games, well above the K=100 *routes* used elsewhere in
     this codebase for a very different kind of stat (that K is in routes/attempts, this
     one's in games -- not directly comparable, but the derivation method is the same
     one Tango's "The Book" uses for platoon splits).
  3. Tested as a genuinely new signal, not a refinement of a failed one -- same rigor as
     every other calibration this session (2023-24 train, 2025 holdout, tail-concentration
     check, direction agreement, yards-isolated-from-volume where relevant).

    python -m matchup_model.opp.def_strength_calib
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
MIN_GAMES, LOOKBACK, HALFLIFE = 4, 10, 4
K_GAMES = 24.0   # derived from measured split-half reliability, see module docstring

REPORT = Path(__file__).resolve().parents[1] / "opp_defstrength_calibration.md"
ROWS_CSV = Path(__file__).resolve().parents[1] / "opp_defstrength_calibration_rows.csv"


def _ewma(v: np.ndarray) -> float:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)][-LOOKBACK:]
    if v.size == 0:
        return np.nan
    w = 0.5 ** (np.arange(v.size)[::-1] / HALFLIFE)
    return float((w * v).sum() / w.sum())


def _rb_defense_allowed(pw: pd.DataFrame) -> pd.DataFrame:
    """Per (season, week, defense), total RB fantasy points allowed that week."""
    d = pw[pw.position == "RB"]
    g = (d.groupby(["season", "week", "opponent_team"])
          .agg(fp_allowed=("dk_fp", "sum"))
          .reset_index().rename(columns={"opponent_team": "defense"}))
    return g


def _rb_offense_scored(pw: pd.DataFrame) -> pd.DataFrame:
    """Per (season, week, team), total RB fantasy points that team's RBs scored --
    the 'offense pace' side of the schedule adjustment."""
    d = pw[pw.position == "RB"]
    g = (d.groupby(["season", "week", "team"])
          .agg(fp_scored=("dk_fp", "sum"))
          .reset_index())
    return g


def _trailing(df: pd.DataFrame, key_col: str, key: str, season: int, week: int,
             val_col: str) -> float:
    h = df[(df[key_col] == key) &
          ((df.season < season) | ((df.season == season) & (df.week < week)))].tail(LOOKBACK)
    if h.empty:
        return np.nan
    return _ewma(h[val_col].to_numpy())


def build_schedule_adjusted_deltas() -> pd.DataFrame:
    """One row per (season, week, defense, offense): actual fp the offense's RBs scored
    against that defense, minus the offense's own trailing RB pace entering that game --
    walk-forward safe (only games strictly before that week feed the trailing pace)."""
    pw = D.player_weeks()
    allowed = _rb_defense_allowed(pw)
    scored = _rb_offense_scored(pw)

    rows = []
    for season in sorted(allowed.season.unique()):
        s_allowed = allowed[allowed.season == season]
        for _, r in s_allowed.iterrows():
            wk, defense, fp_allowed = int(r.week), r.defense, float(r.fp_allowed)
            # who did this defense play that week? -- look up via player_weeks' own team/opp
            opp_rows = pw[(pw.season == season) & (pw.week == wk) &
                         (pw.opponent_team == defense) & (pw.position == "RB")]
            if opp_rows.empty:
                continue
            offense = opp_rows["team"].iloc[0]
            trailing_pace = _trailing(scored, "team", offense, season, wk, "fp_scored")
            if not np.isfinite(trailing_pace):
                continue
            rows.append(dict(season=season, week=wk, defense=defense, offense=offense,
                             fp_allowed=fp_allowed, offense_trailing_pace=trailing_pace,
                             saf_delta=fp_allowed - trailing_pace))
    return pd.DataFrame(rows)


def _defense_dev_shrunk(deltas: pd.DataFrame, defense: str, season: int, week: int) -> float:
    """Trailing EWMA of this defense's schedule-adjusted delta, EB-shrunk toward 0
    (neutral, by construction of the delta) with the derived K_GAMES pseudo-count."""
    h = deltas[(deltas.defense == defense) &
              ((deltas.season < season) | ((deltas.season == season) & (deltas.week < week)))].tail(LOOKBACK)
    if len(h) < MIN_GAMES:
        return 0.0
    raw = _ewma(h["saf_delta"].to_numpy())
    if not np.isfinite(raw):
        return 0.0
    n = float(len(h))
    return (n * raw) / (n + K_GAMES)   # shrink toward 0, not toward a separate prior


def run() -> pd.DataFrame:
    pw = M._pw_with_team()
    deltas = build_schedule_adjusted_deltas()
    print(f"  built {len(deltas)} schedule-adjusted defense-game deltas")
    t0 = time.time()
    rows = []
    for season in TRAIN_SEASONS + [HOLDOUT_SEASON]:
        sl = pw[(pw.season == season) & (pw.week.isin(TEST_WEEKS)) & (pw.position == "RB")]
        sl = sl[sl.carries >= 2]
        for _, r in sl.iterrows():
            pid, wk, opp = r["player_id"], int(r["week"]), r["opponent_team"]
            o = M.opp_line(pid, "RB", season, wk, by="id", injury_adj=True)
            if o.get("dk_fp") is None:
                continue
            base = o["line"].get("rush_yds")
            if base is None or not np.isfinite(base):
                continue
            dev = _defense_dev_shrunk(deltas, opp, season, wk)
            rows.append(dict(season=season, week=wk, player=r["player_display_name"],
                             defense=opp, base=float(base), dev=round(dev, 4),
                             actual=float(r["rushing_yards"]),
                             actual_carries=float(r["carries"])))
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


def _fit_beta(base, actual, dev) -> float:
    d = (base * dev).to_numpy()
    y = (actual - base).to_numpy()
    denom = float(np.dot(d, d))
    return float(np.dot(d, y) / denom) if denom > 0 else 0.0


def report(df: pd.DataFrame) -> str:
    L = ["# Schedule-adjusted, reliability-shrunk RB defense strength", ""]
    L.append(f"- {len(df)} RB player-games, wk {TEST_WEEKS.start}-{TEST_WEEKS.stop-1}, "
             f"seasons {TRAIN_SEASONS}+{HOLDOUT_SEASON}.")
    L.append("- `dev` = defense's trailing (walk-forward) schedule-adjusted rush-fp-allowed "
             "delta -- each game compared to the OPPOSING OFFENSE'S OWN trailing pace "
             "entering that game (SAFPA-style, made walk-forward safe), then EB-shrunk "
             f"toward 0 with K={K_GAMES:.0f} games, derived from measured split-half "
             "reliability (~0.265) via Tango's method, not hand-picked.")
    L.append("")

    train = df[df.season.isin(TRAIN_SEASONS)].dropna(subset=["dev"])
    hold = df[df.season == HOLDOUT_SEASON].dropna(subset=["dev"])
    if len(train) < 100 or len(hold) < 50:
        L.append(f"**insufficient rows (train={len(train)}, holdout={len(hold)}), aborting**")
        return "\n".join(L)

    beta = _fit_beta(train.base, train.actual, train.dev)
    hold = hold.assign(adjusted=hold.base * (1 + beta * hold.dev))
    train_adj = train.assign(adjusted=train.base * (1 + beta * train.dev))
    L.append(f"## Whole-population RMSE (n={len(train)} train, {len(hold)} holdout)")
    L.append("")
    L.append(f"fitted beta (train): **{beta:+.4f}**")
    L.append("")
    L.append("| slice | n | RMSE base | RMSE adjusted | Δ | MAE base | MAE adjusted | Δ |")
    L.append("|---|--:|--:|--:|--:|--:|--:|--:|")
    for lbl, sub in [(f"train {TRAIN_SEASONS}", train_adj), (f"HOLDOUT {HOLDOUT_SEASON}", hold)]:
        rb, ra = _rmse(sub.base, sub.actual), _rmse(sub.adjusted, sub.actual)
        mb, ma = _mae(sub.base, sub.actual), _mae(sub.adjusted, sub.actual)
        L.append(f"| {lbl} | {len(sub)} | {rb:.2f} | {ra:.2f} | **{rb-ra:+.2f}** | "
                 f"{mb:.2f} | {ma:.2f} | **{mb-ma:+.2f}** |")
    L.append("")

    hold = hold.assign(abs_dev=hold.dev.abs())
    L.append("## Holdout RMSE by |dev| tercile")
    L.append("")
    try:
        hold["tercile"] = pd.qcut(hold["abs_dev"], 3, labels=["low", "mid", "high"], duplicates="drop")
        L.append("| tercile | n | dev range | RMSE base | RMSE adjusted | Δ |")
        L.append("|---|--:|---|--:|--:|--:|")
        for t in ["low", "mid", "high"]:
            g = hold[hold.tercile == t]
            if g.empty:
                continue
            rb, ra = _rmse(g.base, g.actual), _rmse(g.adjusted, g.actual)
            L.append(f"| {t} | {len(g)} | {g.abs_dev.min():.2f}-{g.abs_dev.max():.2f} | "
                     f"{rb:.2f} | {ra:.2f} | **{rb-ra:+.3f}** |")
        L.append("")
    except ValueError:
        L.append("(not enough spread in dev to tercile)")
        L.append("")

    sub = hold[np.sign(hold.adjusted - hold.base) != 0]
    if len(sub):
        agree = (np.sign(sub.adjusted - sub.base) == np.sign(sub.actual - sub.base)).mean()
        L.append(f"Direction agreement (n={len(sub)}): **{agree:.1%}** (50% = coin flip)")
        L.append("")

    rb_h, ra_h = _rmse(hold.base, hold.actual), _rmse(hold.adjusted, hold.actual)
    if ra_h < rb_h - 0.5:
        v = "**real edge — worth wiring into opp_line()**"
    elif ra_h < rb_h - 0.05:
        v = "**small improvement — marginal, keep it in mind**"
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
