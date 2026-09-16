"""Does a defense's tendency to stack the box -- independent of its overall run-defense
strength -- refine the RB run-defense signal that's ALREADY confirmed real this session
(see opp_defense_calibration.md / the team-persistence check: rho ~0.2-0.3, consistent
every year and every half-season split)? Same method as scheme_calib.py/alignment_calib.py,
axis = light (<=6 men) vs stacked (7+ men) box.

  1. Player split: a RB's own trailing YPC vs a stacked box and vs a light box, each minus
     his own trailing OVERALL YPC (from the existing unfiltered rushing-advanced weekly
     files already on disk) -- population-shrunk (EB, K_SPLIT attempts) toward a league
     prior, same style opp_line() already uses.
  2. Defense split: the opposing defense's trailing average men-in-the-box faced, minus a
     league average from train seasons -- walk-forward.
  3. signal = defense_box_dev * (stacked_edge - light_edge): positive when a defense that
     stacks the box more than average faces a back who's relatively BETTER against stacked
     fronts than his own light-box number would predict (and the mirror case).
  4. adjusted = base_rush_yds * (1 + beta*signal), beta fit by OLS on 2023-24 train,
     judged on a 2025 holdout it never saw.

Data: rushing-boxcount-{light,stacked}_{season}_week.csv (player side, "Men in the Box"
play-filter + week split) and team-defense-box_{season}_week.csv (defense side, Team
Defense's own BOX column -- average men in the box faced, no filter needed since it's
already the tendency metric directly). Baseline (unfiltered) YPC reuses
rushing-advanced_{season}_week.csv, already on disk from earlier this session.

    python -m matchup_model.opp.box_calib
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

TRAIN_SEASONS = [2023, 2024]
HOLDOUT_SEASON = 2025
TEST_WEEKS = range(4, 19)
MIN_GAMES, LOOKBACK, HALFLIFE = 4, 10, 4
K_SPLIT = 100.0   # pseudo-count in ATTEMPTS, same style/scale as scheme/alignment_calib.py

REPORT = Path(__file__).resolve().parents[1] / "opp_box_calibration.md"
ROWS_CSV = Path(__file__).resolve().parents[1] / "opp_box_calibration_rows.csv"


def _ewma(v: np.ndarray) -> float:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)][-LOOKBACK:]
    if v.size == 0:
        return np.nan
    w = 0.5 ** (np.arange(v.size)[::-1] / HALFLIFE)
    return float((w * v).sum() / w.sum())


def _read_export(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, header=1)
    d = d[d["Name"].notna() & (d["Name"] != "League Avg")]
    return d


def load_player_box(bucket: str) -> pd.DataFrame:
    """{name_key, Team, POS, season, week, att_<b>, yds_<b>, ypc_<b>} per player-week,
    `<b>` = 'light' or 'stacked'."""
    b = bucket.lower()
    frames = []
    for season in TRAIN_SEASONS + [HOLDOUT_SEASON]:
        p = MATCHUP_DATA / f"rushing-boxcount-{b}_{season}_week.csv"
        if not p.exists():
            continue
        d = _read_export(p)
        d = d.rename(columns={"WEEK": "week", "ATT": f"att_{b}", "YDS": f"yds_{b}",
                              "YPC": f"ypc_{b}"})
        keep = ["Name", "Team", "POS", "week", f"att_{b}", f"yds_{b}", f"ypc_{b}"]
        d = d[[c for c in keep if c in d.columns]].copy()
        d["season"] = season
        frames.append(d)
    cols = ["name_key", "Team", "POS", "season", "week", f"att_{b}", f"yds_{b}", f"ypc_{b}"]
    if not frames:
        return pd.DataFrame(columns=cols)
    df = pd.concat(frames, ignore_index=True)
    df["name_key"] = df["Name"].map(normalize_name)
    df["week"] = pd.to_numeric(df["week"], errors="coerce")
    for c in (f"att_{b}", f"yds_{b}", f"ypc_{b}"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["week"]).sort_values(["name_key", "season", "week"])


def load_player_splits() -> pd.DataFrame:
    light = load_player_box("light")
    stacked = load_player_box("stacked")
    m = light.merge(stacked.drop(columns=["Team", "POS"], errors="ignore"),
                    on=["name_key", "week", "season"], how="outer")

    base_frames = []
    for season in TRAIN_SEASONS + [HOLDOUT_SEASON]:
        p = MATCHUP_DATA / f"rushing-advanced_{season}_week.csv"
        if not p.exists():
            continue
        d = _read_export(p)
        d = d.rename(columns={"WEEK": "week", "ATT": "att_all", "YPC": "ypc_all"})
        keep = ["Name", "week", "att_all", "ypc_all"]
        d = d[[c for c in keep if c in d.columns]].copy()
        d["season"] = season
        base_frames.append(d)
    base = pd.concat(base_frames, ignore_index=True) if base_frames else pd.DataFrame()
    if not base.empty:
        base["name_key"] = base["Name"].map(normalize_name)
        base["week"] = pd.to_numeric(base["week"], errors="coerce")
        base["att_all"] = pd.to_numeric(base["att_all"], errors="coerce")
        base["ypc_all"] = pd.to_numeric(base["ypc_all"], errors="coerce")
        base = base.dropna(subset=["week"])[["name_key", "season", "week", "att_all", "ypc_all"]]

    m = m.merge(base, on=["name_key", "season", "week"], how="left")
    return m.sort_values(["name_key", "season", "week"])


def load_defense_box() -> pd.DataFrame:
    """{defense, season, week, box} per team-week -- BOX is Team Defense's own column,
    average men in the box faced, no play-filter needed."""
    frames = []
    for season in TRAIN_SEASONS + [HOLDOUT_SEASON]:
        p = MATCHUP_DATA / f"team-defense-box_{season}_week.csv"
        if not p.exists():
            continue
        d = _read_export(p)
        d = d.rename(columns={"WEEK": "week", "BOX": "box"})
        keep = ["Name", "week", "box"]
        d = d[[c for c in keep if c in d.columns]].copy()
        d["season"] = season
        frames.append(d)
    cols = ["defense", "season", "week", "box"]
    if not frames:
        return pd.DataFrame(columns=cols)
    df = pd.concat(frames, ignore_index=True)
    df["defense"] = df["Name"].map(norm_team)
    df["week"] = pd.to_numeric(df["week"], errors="coerce")
    df["box"] = pd.to_numeric(df["box"], errors="coerce")
    return df.dropna(subset=["week"]).sort_values(["defense", "season", "week"])


# ── walk-forward trailing lookups (mirrors scheme_calib.py / alignment_calib.py) ──
def _pop_priors(ps: pd.DataFrame, before_season: int) -> tuple[float, float]:
    d = ps[ps.season < before_season]
    if d.empty:
        d = ps
    d = d.dropna(subset=["ypc_all"])
    dl = d.dropna(subset=["ypc_light", "att_light"])
    ds = d.dropna(subset=["ypc_stacked", "att_stacked"])
    light_prior = float(np.average(dl.ypc_light - dl.ypc_all, weights=dl.att_light.clip(lower=0.1))) \
        if len(dl) else 0.0
    stacked_prior = float(np.average(ds.ypc_stacked - ds.ypc_all, weights=ds.att_stacked.clip(lower=0.1))) \
        if len(ds) else 0.0
    return light_prior, stacked_prior


def _player_edge_shrunk(ps: pd.DataFrame, name_key: str, season: int, week: int,
                        light_prior: float, stacked_prior: float) -> tuple[float, float]:
    h = ps[(ps.name_key == name_key) &
          ((ps.season < season) | ((ps.season == season) & (ps.week < week)))].tail(LOOKBACK)
    if h.empty:
        return light_prior, stacked_prior
    base = _ewma(h["ypc_all"].to_numpy())
    if not np.isfinite(base):
        return light_prior, stacked_prior

    def _shrink(col_ypc: str, col_att: str, prior: float) -> float:
        hh = h.dropna(subset=[col_ypc, col_att])
        if hh.empty:
            return prior
        n = float(hh[col_att].sum())
        raw = float(np.average(hh[col_ypc], weights=hh[col_att].clip(lower=0.1))) - base
        return (n * raw + K_SPLIT * prior) / (n + K_SPLIT) if (n + K_SPLIT) > 0 else prior

    return (_shrink("ypc_light", "att_light", light_prior),
           _shrink("ypc_stacked", "att_stacked", stacked_prior))


def _defense_box_dev(dg: pd.DataFrame, defense: str, season: int, week: int,
                     league_avg: float) -> float:
    h = dg[(dg.defense == defense) &
          ((dg.season < season) | ((dg.season == season) & (dg.week < week)))].tail(LOOKBACK)
    if len(h) < MIN_GAMES or not np.isfinite(league_avg) or league_avg <= 0:
        return 0.0
    val = _ewma(h["box"].dropna().to_numpy())
    return (val - league_avg) if np.isfinite(val) else 0.0


def _league_avg_box(dg: pd.DataFrame, before_season: int) -> float:
    d = dg[dg.season < before_season]
    if d.empty:
        d = dg
    return float(d["box"].mean())


def run() -> pd.DataFrame:
    pw = M._pw_with_team()
    ps = load_player_splits()
    dg = load_defense_box()
    print(f"  loaded {len(ps)} player-split rows, {len(dg)} defense-week rows")
    t0 = time.time()
    rows = []
    for season in TRAIN_SEASONS + [HOLDOUT_SEASON]:
        lg_box = _league_avg_box(dg, season)
        light_prior, stacked_prior = _pop_priors(ps, season)
        sl = pw[(pw.season == season) & (pw.week.isin(TEST_WEEKS)) & (pw.position == "RB")]
        sl = sl[sl.carries >= 2]
        for _, r in sl.iterrows():
            pid, wk, opp = r["player_id"], int(r["week"]), r["opponent_team"]
            nk = r["name_key"]
            o = M.opp_line(pid, "RB", season, wk, by="id", injury_adj=True)
            if o.get("dk_fp") is None:
                continue
            base = o["line"].get("rush_yds")
            if base is None or not np.isfinite(base):
                continue
            light_edge, stacked_edge = _player_edge_shrunk(ps, nk, season, wk,
                                                            light_prior, stacked_prior)
            box_dev = _defense_box_dev(dg, opp, season, wk, lg_box)
            signal = box_dev * (stacked_edge - light_edge)
            rows.append(dict(season=season, week=wk, player=r["player_display_name"],
                             defense=opp, base=float(base),
                             light_edge=round(light_edge, 4), stacked_edge=round(stacked_edge, 4),
                             box_dev=round(box_dev, 4), signal=round(signal, 4),
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


def _fit_beta(base, actual, signal) -> float:
    d = (base * signal).to_numpy()
    y = (actual - base).to_numpy()
    denom = float(np.dot(d, d))
    return float(np.dot(d, y) / denom) if denom > 0 else 0.0


def report(df: pd.DataFrame) -> str:
    L = ["# Box-count-conditioned RB efficiency vs defensive box tendency", ""]
    L.append(f"- {len(df)} RB player-games, wk {TEST_WEEKS.start}-{TEST_WEEKS.stop-1}, "
             f"seasons {TRAIN_SEASONS}+{HOLDOUT_SEASON}.")
    L.append("- `signal` = defense's trailing avg-men-in-box deviation from league avg x "
             "(RB's shrunk stacked-box edge minus light-box edge) -- positive when a "
             "defense that stacks the box more than usual faces a back who's relatively "
             "better against stacked fronts (or the mirror: light-box defense, "
             "light-box-dependent back).")
    L.append(f"- Player edges are EB-shrunk toward a population prior (K={K_SPLIT:.0f} "
             "attempts), same style opp_line() already uses -- and this session's other "
             "scheme calibrations used.")
    L.append("")

    train = df[df.season.isin(TRAIN_SEASONS)].dropna(subset=["signal"])
    hold = df[df.season == HOLDOUT_SEASON].dropna(subset=["signal"])
    if len(train) < 100 or len(hold) < 50:
        L.append(f"**insufficient rows (train={len(train)}, holdout={len(hold)}), aborting**")
        return "\n".join(L)

    beta = _fit_beta(train.base, train.actual, train.signal)
    hold = hold.assign(adjusted=hold.base * (1 + beta * hold.signal))
    train_adj = train.assign(adjusted=train.base * (1 + beta * train.signal))
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

    hold = hold.assign(abs_signal=hold.signal.abs())
    L.append("## Holdout RMSE by |signal| tercile")
    L.append("")
    try:
        hold["tercile"] = pd.qcut(hold["abs_signal"], 3, labels=["low", "mid", "high"], duplicates="drop")
        L.append("| tercile | n | signal range | RMSE base | RMSE adjusted | Δ |")
        L.append("|---|--:|---|--:|--:|--:|")
        for t in ["low", "mid", "high"]:
            g = hold[hold.tercile == t]
            if g.empty:
                continue
            rb, ra = _rmse(g.base, g.actual), _rmse(g.adjusted, g.actual)
            L.append(f"| {t} | {len(g)} | {g.abs_signal.min():.4f}-{g.abs_signal.max():.4f} | "
                     f"{rb:.2f} | {ra:.2f} | **{rb-ra:+.3f}** |")
        L.append("")
    except ValueError:
        L.append("(not enough spread in signal to tercile)")
        L.append("")

    sub = hold[np.sign(hold.adjusted - hold.base) != 0]
    if len(sub):
        agree = (np.sign(sub.adjusted - sub.base) == np.sign(sub.actual - sub.base)).mean()
        L.append(f"Direction agreement (n={len(sub)}): **{agree:.1%}** (50% = coin flip)")
        L.append("")

    yc = df.dropna(subset=["signal", "actual_carries"])
    yc = yc[yc.actual_carries >= 2].copy()
    yc["actual_ypc"] = yc.actual / yc.actual_carries
    yc["base_ypc"] = yc.base / yc.actual_carries
    yc_train = yc[yc.season.isin(TRAIN_SEASONS)]
    yc_hold = yc[yc.season == HOLDOUT_SEASON]
    if len(yc_train) > 50 and len(yc_hold) > 50:
        d = (yc_train.base_ypc * yc_train.signal).to_numpy()
        y = (yc_train.actual_ypc - yc_train.base_ypc).to_numpy()
        denom = float(np.dot(d, d))
        beta_ypc = float(np.dot(d, y) / denom) if denom > 0 else 0.0
        corr_hold = yc_hold[["signal"]].assign(resid=yc_hold.actual_ypc - yc_hold.base_ypc).corr().iloc[0, 1]
        corr_train = yc_train[["signal"]].assign(resid=yc_train.actual_ypc - yc_train.base_ypc).corr().iloc[0, 1]
        L.append("## Yards/carry isolated (strips out attempt-volume noise)")
        L.append("")
        L.append(f"beta fit directly on y/c residual (train): {beta_ypc:+.4f}")
        L.append(f"raw correlation(signal, actual_ypc - base_ypc): train {corr_train:+.4f}, "
                 f"holdout {corr_hold:+.4f}")
        L.append("")

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
