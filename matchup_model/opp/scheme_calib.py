"""The real test of "scheme-specific efficiency vs opponent tendency," after the cheap
'yards allowed' version (opp_defense_calib.py) came back negative.

Uses FantasyPoints Data Suite splits already committed at weekly granularity, 2022-2025 —
no live data needed:
  * data/dfs/matchup/receiving-manvszone_{season}wk{batch}_week.csv — player YPRR vs man
    vs zone coverage, per game.
  * data/dfs/matchup/coverage-matrix_{season}_week.csv — each defense's own MAN% rate,
    per game.

Hypothesis: a defense's man-rate is a coaching decision (stable, process-based), unlike
"yards allowed" (a noisy, opponent-confounded outcome) — so it might carry real signal
where the cheap version didn't.

Method (mirrors opp_defense_calib.py exactly, only the axis changes):
  1. Trailing (walk-forward) EWMA of the DEFENSE's own MAN% (last 10 games, half-life 4),
     vs a league average from earlier seasons only.
  2. Trailing EWMA of the PLAYER's own YPRR-vs-man and YPRR-vs-zone, each vs his own
     trailing overall YPRR (his "edge" running that specific route mix).
  3. scheme_signal = (defense man-rate - league avg) * (player's man edge - zone edge)
     — positive when a player who's relatively better vs man faces a more-man-than-usual
     defense (or the mirror case for zone).
  4. adjusted = base*(1+beta*factor), beta fit by OLS on 2022-24 train, judged on a 2025
     holdout it never saw — identical scoring to opp_defense_calib.py so the two are
     directly comparable.

    python -m matchup_model.opp.scheme_calib
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from dfs.names import norm_team, normalize_name
from matchup_model.config import DATA as MATCHUP_DATA
from matchup_model.opp import data as D
from matchup_model.opp import model as M

TRAIN_SEASONS = [2022, 2023, 2024]
HOLDOUT_SEASON = 2025
TEST_WEEKS = range(4, 19)
MIN_GAMES, LOOKBACK, HALFLIFE = 4, 10, 4
_BATCHES = ["wk1-6", "wk7-12", "wk13-18"]

REPORT = Path(__file__).resolve().parents[1] / "opp_scheme_calibration.md"
ROWS_CSV = Path(__file__).resolve().parents[1] / "opp_scheme_calibration_rows.csv"


def _ewma(v: np.ndarray) -> float:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)][-LOOKBACK:]
    if v.size == 0:
        return np.nan
    w = 0.5 ** (np.arange(v.size)[::-1] / HALFLIFE)
    return float((w * v).sum() / w.sum())


# ── load the two Data Suite tables into tidy, walk-forward-ready frames ─────────
def load_player_splits() -> pd.DataFrame:
    frames = []
    for season in TRAIN_SEASONS + [HOLDOUT_SEASON]:
        for batch in _BATCHES:
            p = MATCHUP_DATA / f"receiving-manvszone_{season}{batch}_week.csv"
            if not p.exists():
                continue
            d = pd.read_csv(p, header=1)
            d = d[d["Name"].notna() & (d["Name"] != "League Avg")]
            d = d.rename(columns={"Name": "player", "Team": "team", "WEEK": "week",
                                  "YPRR": "yprr_all", "RTE": "rte_all",
                                  "YPRR.1": "yprr_man", "RTE.1": "rte_man",
                                  "YPRR.2": "yprr_zone", "RTE.2": "rte_zone"})
            keep = ["player", "team", "POS", "week", "yprr_all", "rte_all",
                   "yprr_man", "rte_man", "yprr_zone", "rte_zone"]
            d = d[[c for c in keep if c in d.columns]].copy()
            d["season"] = season
            frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["name_key"] = df["player"].map(normalize_name)
    df["week"] = pd.to_numeric(df["week"], errors="coerce")
    for c in ("yprr_all", "yprr_man", "yprr_zone", "rte_man", "rte_zone"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["week"]).sort_values(["name_key", "season", "week"])


def load_defense_scheme() -> pd.DataFrame:
    frames = []
    for season in TRAIN_SEASONS + [HOLDOUT_SEASON]:
        p = MATCHUP_DATA / f"coverage-matrix_{season}_week.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p, header=1)
        d = d[d["Name"].notna() & (d["Name"] != "League Avg")]
        d = d.rename(columns={"Name": "team_full", "WEEK": "week", "MAN %": "man_pct"})
        d = d[["team_full", "week", "man_pct"]].copy()
        d["season"] = season
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["defense"] = df["team_full"].map(norm_team)
    df["week"] = pd.to_numeric(df["week"], errors="coerce")
    df["man_pct"] = pd.to_numeric(df["man_pct"], errors="coerce")
    return df.dropna(subset=["week"]).sort_values(["defense", "season", "week"])


# ── walk-forward trailing lookups ────────────────────────────────────────────
K_SPLIT = 100.0   # pseudo-count, in ROUTES, for shrinking a player's split edge toward
                  # the population prior — same style as model.py's K_SHARE/K_YDS/K_TD


def _pop_priors(ps: pd.DataFrame, before_season: int) -> tuple[float, float]:
    """Population-level (route-weighted) man/zone edge, pooled across every WR/TE —
    the prior a thin personal sample gets shrunk toward. Walk-forward: only seasons
    strictly before the test season, so this never leaks the future."""
    d = ps[ps.season < before_season]
    if d.empty:
        d = ps
    d = d.dropna(subset=["yprr_all"])
    dm = d.dropna(subset=["yprr_man", "rte_man"])
    dz = d.dropna(subset=["yprr_zone", "rte_zone"])
    man_prior = float(np.average(dm.yprr_man - dm.yprr_all, weights=dm.rte_man.clip(lower=0.1))) \
        if len(dm) else 0.0
    zone_prior = float(np.average(dz.yprr_zone - dz.yprr_all, weights=dz.rte_zone.clip(lower=0.1))) \
        if len(dz) else 0.0
    return man_prior, zone_prior


def _player_edge(ps: pd.DataFrame, name_key: str, season: int, week: int) -> tuple[float, float]:
    """v1 (unchanged): (man_edge, zone_edge) = this player's own trailing YPRR vs
    man/zone minus his own trailing overall YPRR — no pooling across players.
    NaN, NaN if not enough prior games with split data."""
    h = ps[(ps.name_key == name_key) &
          ((ps.season < season) | ((ps.season == season) & (ps.week < week)))].tail(LOOKBACK)
    h_man = h.dropna(subset=["yprr_man"])
    h_zone = h.dropna(subset=["yprr_zone"])
    if len(h_man) < MIN_GAMES or len(h_zone) < MIN_GAMES:
        return np.nan, np.nan
    base = _ewma(h["yprr_all"].to_numpy())
    if not np.isfinite(base):
        return np.nan, np.nan
    man = _ewma(h_man["yprr_man"].to_numpy())
    zone = _ewma(h_zone["yprr_zone"].to_numpy())
    return (man - base if np.isfinite(man) else np.nan,
           zone - base if np.isfinite(zone) else np.nan)


def _player_edge_shrunk(ps: pd.DataFrame, name_key: str, season: int, week: int,
                        man_prior: float, zone_prior: float) -> tuple[float, float]:
    """v2: same raw per-game deltas as _player_edge, but empirical-Bayes shrunk toward
    the POPULATION prior (pooled across every WR/TE using years of data), weighted by
    how many routes of that coverage type this player has actually run — the technique
    the rest of opp_line() already uses for share/efficiency, extended to splits.
    A player with zero man-coverage routes on file gets exactly the population prior
    (assume average until proven otherwise) instead of being dropped."""
    h = ps[(ps.name_key == name_key) &
          ((ps.season < season) | ((ps.season == season) & (ps.week < week)))].tail(LOOKBACK)
    if h.empty:
        return man_prior, zone_prior
    base = _ewma(h["yprr_all"].to_numpy())
    if not np.isfinite(base):
        return man_prior, zone_prior

    def _shrink(col_yprr: str, col_rte: str, prior: float) -> float:
        hh = h.dropna(subset=[col_yprr, col_rte])
        if hh.empty:
            return prior
        n = float(hh[col_rte].sum())
        raw = float(np.average(hh[col_yprr], weights=hh[col_rte].clip(lower=0.1))) - base
        return (n * raw + K_SPLIT * prior) / (n + K_SPLIT) if (n + K_SPLIT) > 0 else prior

    return (_shrink("yprr_man", "rte_man", man_prior),
           _shrink("yprr_zone", "rte_zone", zone_prior))


def _defense_man_dev(dg: pd.DataFrame, defense: str, season: int, week: int,
                     league_avg: float) -> float:
    h = dg[(dg.defense == defense) &
          ((dg.season < season) | ((dg.season == season) & (dg.week < week)))].tail(LOOKBACK)
    if len(h) < MIN_GAMES or not np.isfinite(league_avg):
        return 0.0
    val = _ewma(h["man_pct"].to_numpy())
    return (val - league_avg) / 100.0 if np.isfinite(val) else 0.0    # MAN % is 0-100


def _league_avg_man(dg: pd.DataFrame, before_season: int) -> float:
    d = dg[dg.season < before_season]
    if d.empty:
        d = dg
    return float(d["man_pct"].mean())


def run() -> pd.DataFrame:
    pw = M._pw_with_team()
    ps = load_player_splits()
    dg = load_defense_scheme()
    print(f"  loaded {len(ps)} player-split rows, {len(dg)} defense-week rows")
    t0 = time.time()
    rows = []
    for season in TRAIN_SEASONS + [HOLDOUT_SEASON]:
        lg_man = _league_avg_man(dg, season)
        man_prior, zone_prior = _pop_priors(ps, season)
        sl = pw[(pw.season == season) & (pw.week.isin(TEST_WEEKS)) &
               (pw.position.isin(["WR", "TE"]))]
        sl = sl[sl.targets >= 2]
        for _, r in sl.iterrows():
            pid, wk, opp = r["player_id"], int(r["week"]), r["opponent_team"]
            nk = r["name_key"]
            o = M.opp_line(pid, "WR" if r.position == "WR" else "TE", season, wk,
                           by="id", injury_adj=True)
            if o.get("dk_fp") is None:
                continue
            base = o["line"].get("rec_yds")
            if base is None or not np.isfinite(base):
                continue
            man_edge_v1, zone_edge_v1 = _player_edge(ps, nk, season, wk)
            v1_ok = np.isfinite(man_edge_v1) and np.isfinite(zone_edge_v1)
            man_edge_v2, zone_edge_v2 = _player_edge_shrunk(ps, nk, season, wk,
                                                            man_prior, zone_prior)
            man_dev = _defense_man_dev(dg, opp, season, wk, lg_man)
            signal_v1 = man_dev * (man_edge_v1 - zone_edge_v1) if v1_ok else np.nan
            signal_v2 = man_dev * (man_edge_v2 - zone_edge_v2)
            rows.append(dict(season=season, week=wk, player=r["player_display_name"],
                             pos=r["position"], defense=opp, base=float(base),
                             v1_available=v1_ok,
                             signal_v1=round(signal_v1, 4) if v1_ok else np.nan,
                             signal_v2=round(signal_v2, 4),
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


def _fit_beta(base, actual, signal) -> float:
    d = (base * signal).to_numpy()
    y = (actual - base).to_numpy()
    denom = float(np.dot(d, d))
    return float(np.dot(d, y) / denom) if denom > 0 else 0.0


def _score(df: pd.DataFrame, signal_col: str, label: str, L: list[str]) -> None:
    train = df[df.season.isin(TRAIN_SEASONS)].dropna(subset=[signal_col])
    hold = df[df.season == HOLDOUT_SEASON].dropna(subset=[signal_col])
    if len(train) < 100 or len(hold) < 50:
        L.append(f"### {label} — insufficient rows, skipped"); L.append(""); return
    beta = _fit_beta(train.base, train.actual, train[signal_col])
    hold = hold.assign(adjusted=hold.base * (1 + beta * hold[signal_col]))
    train = train.assign(adjusted=train.base * (1 + beta * train[signal_col]))
    L.append(f"### {label}  (n={len(train)+len(hold)}: {len(train)} train, {len(hold)} holdout)")
    L.append("")
    L.append(f"fitted beta: **{beta:+.3f}**")
    L.append("")
    L.append("| slice | n | RMSE base | RMSE adjusted | Δ | MAE base | MAE adjusted | Δ |")
    L.append("|---|--:|--:|--:|--:|--:|--:|--:|")
    for lbl, sub in [(f"train {TRAIN_SEASONS}", train), (f"HOLDOUT {HOLDOUT_SEASON}", hold)]:
        rb, ra = _rmse(sub.base, sub.actual), _rmse(sub.adjusted, sub.actual)
        mb, ma = _mae(sub.base, sub.actual), _mae(sub.adjusted, sub.actual)
        L.append(f"| {lbl} | {len(sub)} | {rb:.2f} | {ra:.2f} | **{rb-ra:+.2f}** | "
                 f"{mb:.2f} | {ma:.2f} | **{mb-ma:+.2f}** |")
    rb_h, ra_h = _rmse(hold.base, hold.actual), _rmse(hold.adjusted, hold.actual)
    if ra_h < rb_h - 0.5:
        v = "**real edge**"
    elif ra_h < rb_h - 0.05:
        v = "**small improvement — marginal**"
    else:
        v = "**no improvement**"
    L.append("")
    L.append(f"Verdict: {v}")
    L.append("")


def report(df: pd.DataFrame) -> str:
    L = ["# Scheme-conditioned efficiency (man vs zone) vs opponent tendency — v1 vs v2", ""]
    L.append(f"- {len(df)} WR/TE player-games total, wk {TEST_WEEKS.start}-{TEST_WEEKS.stop-1}, "
             f"seasons {TRAIN_SEASONS}+{HOLDOUT_SEASON}.")
    L.append("- **v1** = each player's own trailing man/zone YPRR minus his own trailing "
             "overall YPRR — no pooling across players. Only scored where he has "
             f"{MIN_GAMES}+ prior games with split data ({int(df.v1_available.sum())}/{len(df)} rows).")
    L.append("- **v2** = the same per-game deltas, empirical-Bayes shrunk toward a "
             f"POPULATION prior (pooled across every WR/TE, years of data) with pseudo-count "
             f"K={K_SPLIT:.0f} routes — the same shrinkage style `opp_line()` already uses "
             "for share/efficiency, extended to the coverage split. Scored on every row — "
             "a player with no man-route history on file just gets the population prior "
             "instead of being dropped.")
    L.append("- `adjusted` = base*(1+beta*signal), beta fit by OLS on train, judged on a "
             f"{HOLDOUT_SEASON} holdout it never saw.")
    L.append("")
    L.append("## v1 vs v2 on the SAME rows (only where v1 had enough personal data — fair fight)")
    L.append("")
    matched = df[df.v1_available]
    _score(matched, "signal_v1", "v1 — own-player only", L)
    _score(matched, "signal_v2", "v2 — population-shrunk, same rows", L)
    L.append("## v2 on its full reach (every row, including thin/zero personal history)")
    L.append("")
    _score(df, "signal_v2", "v2 — population-shrunk, full data", L)
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
