"""Does a defense's 1-high/2-high shell tendency -- distinct from the man/zone coverage-
TYPE axis scheme_calib.py already tested and found null -- refine QB passing-yard
projections? Same method as box_calib.py/scheme_calib.py, axis = Single-High vs Two-High
safety shell (not man vs zone: a defense can play zone under either shell).

  1. Player split: a QB's own trailing YPA vs a Two-High shell and vs a Single-High shell,
     each minus his own trailing OVERALL YPA -- population-shrunk (EB, K_SPLIT dropbacks)
     toward a league prior, same style opp_line()/box_calib.py already use.
  2. Defense split: the opposing defense's trailing 2-HIGH % faced, minus a league average
     from train seasons -- walk-forward.
  3. signal = defense_2high_dev * (two_high_edge - single_high_edge): positive when a
     defense that plays 2-high more than average faces a QB who's relatively BETTER
     against two-high shells than his own single-high number would predict (and the
     mirror case).
  4. adjusted = base_pass_yds * (1 + beta*signal), beta fit by OLS on 2023-24 train,
     judged on a 2025 holdout it never saw.

Data: passing-situation_{season}_week.csv (player side, Single-High/Two-High EPA+YPA
splits by dropback -- already on disk from earlier this session) and
team-defense-box_{season}_week.csv (defense side, Team Defense's own 2-HIGH % column --
already on disk from the box-count pull; its MAN %/ZONE %/BLITZ % columns went unused
there, this is the first calibration to read 2-HIGH % from it).

    python -m matchup_model.opp.shell_calib
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
MIN_ATT = 10
K_SPLIT = 100.0   # pseudo-count in DROPBACKS, same style/scale as box_calib.py

REPORT = Path(__file__).resolve().parents[1] / "opp_shell_calibration.md"
ROWS_CSV = Path(__file__).resolve().parents[1] / "opp_shell_calibration_rows.csv"


def _ewma(v: np.ndarray) -> float:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)][-LOOKBACK:]
    if v.size == 0:
        return np.nan
    w = 0.5 ** (np.arange(v.size)[::-1] / HALFLIFE)
    return float((w * v).sum() / w.sum())


def _read_export(path: Path, header: int = 1) -> pd.DataFrame:
    d = pd.read_csv(path, header=header)
    d = d[d["Name"].notna() & (d["Name"] != "League Avg")]
    return d


def load_player_splits() -> pd.DataFrame:
    """{name_key, season, week, db_all, ypa_all, db_sh, ypa_sh, db_th, ypa_th} per QB-week.
    sh = Single-High, th = Two-High (passing-situation's 4th and 5th column blocks --
    Overall/Man/Zone/Single-High/Two-High, each DB/CMP%/YPA/TD/INT/RATE/EPA/FP -- so
    Single-High is the '.3' suffix block and Two-High is '.4')."""
    frames = []
    for season in TRAIN_SEASONS + [HOLDOUT_SEASON]:
        p = MATCHUP_DATA / f"passing-situation_{season}_week.csv"
        if not p.exists():
            continue
        d = _read_export(p)
        d = d.rename(columns={"WEEK": "week", "DB": "db_all", "YPA": "ypa_all",
                              "DB.3": "db_sh", "YPA.3": "ypa_sh",
                              "DB.4": "db_th", "YPA.4": "ypa_th"})
        keep = ["Name", "week", "db_all", "ypa_all", "db_sh", "ypa_sh", "db_th", "ypa_th"]
        d = d[[c for c in keep if c in d.columns]].copy()
        d["season"] = season
        frames.append(d)
    cols = ["name_key", "season", "week", "db_all", "ypa_all", "db_sh", "ypa_sh", "db_th", "ypa_th"]
    if not frames:
        return pd.DataFrame(columns=cols)
    df = pd.concat(frames, ignore_index=True)
    df["name_key"] = df["Name"].map(normalize_name)
    df["week"] = pd.to_numeric(df["week"], errors="coerce")
    for c in ("db_all", "ypa_all", "db_sh", "ypa_sh", "db_th", "ypa_th"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["week"]).sort_values(["name_key", "season", "week"])


def load_defense_shell() -> pd.DataFrame:
    """{defense, season, week, two_high} per team-week -- Team Defense's own 2-HIGH %."""
    frames = []
    for season in TRAIN_SEASONS + [HOLDOUT_SEASON]:
        p = MATCHUP_DATA / f"team-defense-box_{season}_week.csv"
        if not p.exists():
            continue
        d = _read_export(p)
        d = d.rename(columns={"WEEK": "week", "2-HIGH %": "two_high"})
        keep = ["Name", "week", "two_high"]
        d = d[[c for c in keep if c in d.columns]].copy()
        d["season"] = season
        frames.append(d)
    cols = ["defense", "season", "week", "two_high"]
    if not frames:
        return pd.DataFrame(columns=cols)
    df = pd.concat(frames, ignore_index=True)
    df["defense"] = df["Name"].map(norm_team)
    df["week"] = pd.to_numeric(df["week"], errors="coerce")
    df["two_high"] = pd.to_numeric(df["two_high"], errors="coerce")
    return df.dropna(subset=["week"]).sort_values(["defense", "season", "week"])


# ── walk-forward trailing lookups (mirrors box_calib.py / scheme_calib.py) ──
def _pop_priors(ps: pd.DataFrame, before_season: int) -> tuple[float, float]:
    d = ps[ps.season < before_season]
    if d.empty:
        d = ps
    d = d.dropna(subset=["ypa_all"])
    dsh = d.dropna(subset=["ypa_sh", "db_sh"])
    dth = d.dropna(subset=["ypa_th", "db_th"])
    sh_prior = float(np.average(dsh.ypa_sh - dsh.ypa_all, weights=dsh.db_sh.clip(lower=0.1))) \
        if len(dsh) else 0.0
    th_prior = float(np.average(dth.ypa_th - dth.ypa_all, weights=dth.db_th.clip(lower=0.1))) \
        if len(dth) else 0.0
    return sh_prior, th_prior


def _player_edge_shrunk(ps: pd.DataFrame, name_key: str, season: int, week: int,
                        sh_prior: float, th_prior: float) -> tuple[float, float]:
    h = ps[(ps.name_key == name_key) &
          ((ps.season < season) | ((ps.season == season) & (ps.week < week)))].tail(LOOKBACK)
    if h.empty:
        return sh_prior, th_prior
    base = _ewma(h["ypa_all"].to_numpy())
    if not np.isfinite(base):
        return sh_prior, th_prior

    def _shrink(col_ypa: str, col_db: str, prior: float) -> float:
        hh = h.dropna(subset=[col_ypa, col_db])
        if hh.empty:
            return prior
        n = float(hh[col_db].sum())
        raw = float(np.average(hh[col_ypa], weights=hh[col_db].clip(lower=0.1))) - base
        return (n * raw + K_SPLIT * prior) / (n + K_SPLIT) if (n + K_SPLIT) > 0 else prior

    return (_shrink("ypa_sh", "db_sh", sh_prior), _shrink("ypa_th", "db_th", th_prior))


def _defense_shell_dev(dg: pd.DataFrame, defense: str, season: int, week: int,
                       league_avg: float) -> float:
    h = dg[(dg.defense == defense) &
          ((dg.season < season) | ((dg.season == season) & (dg.week < week)))].tail(LOOKBACK)
    if len(h) < MIN_GAMES or not np.isfinite(league_avg):
        return 0.0
    val = _ewma(h["two_high"].dropna().to_numpy())
    return (val - league_avg) / 100.0 if np.isfinite(val) else 0.0    # 2-HIGH % is 0-100


def _league_avg_shell(dg: pd.DataFrame, before_season: int) -> float:
    d = dg[dg.season < before_season]
    if d.empty:
        d = dg
    return float(d["two_high"].mean())


def run() -> pd.DataFrame:
    pw = M._pw_with_team()
    ps = load_player_splits()
    dg = load_defense_shell()
    print(f"  loaded {len(ps)} QB-split rows, {len(dg)} defense-week rows")
    t0 = time.time()
    rows = []
    for season in TRAIN_SEASONS + [HOLDOUT_SEASON]:
        lg_shell = _league_avg_shell(dg, season)
        sh_prior, th_prior = _pop_priors(ps, season)
        sl = pw[(pw.season == season) & (pw.week.isin(TEST_WEEKS)) & (pw.position == "QB")]
        sl = sl[sl.attempts >= MIN_ATT]
        for _, r in sl.iterrows():
            pid, wk, opp = r["player_id"], int(r["week"]), r["opponent_team"]
            nk = r["name_key"]
            o = M.opp_line(pid, "QB", season, wk, by="id", injury_adj=True)
            if o.get("dk_fp") is None:
                continue
            base = o["line"].get("pass_yds")
            if base is None or not np.isfinite(base):
                continue
            sh_edge, th_edge = _player_edge_shrunk(ps, nk, season, wk, sh_prior, th_prior)
            shell_dev = _defense_shell_dev(dg, opp, season, wk, lg_shell)
            signal = shell_dev * (th_edge - sh_edge)
            rows.append(dict(season=season, week=wk, player=r["player_display_name"],
                             defense=opp, base=float(base),
                             sh_edge=round(sh_edge, 4), th_edge=round(th_edge, 4),
                             shell_dev=round(shell_dev, 4), signal=round(signal, 4),
                             actual=float(r["passing_yards"]),
                             actual_attempts=float(r["attempts"])))
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
    L = ["# Shell-conditioned (1-high/2-high) QB passing-yard calibration", ""]
    L.append(f"- {len(df)} QB player-games, wk {TEST_WEEKS.start}-{TEST_WEEKS.stop-1}, "
             f"seasons {TRAIN_SEASONS}+{HOLDOUT_SEASON}, min {MIN_ATT} attempts.")
    L.append("- `signal` = defense's trailing 2-HIGH % deviation from league avg x (QB's "
             "shrunk two-high-shell edge minus single-high-shell edge) -- positive when a "
             "defense that plays 2-high more than usual faces a QB who's relatively better "
             "against two-high shells than his own single-high number would predict.")
    L.append(f"- Player edges are EB-shrunk toward a population prior (K={K_SPLIT:.0f} "
             "dropbacks), same style opp_line()/box_calib.py already use.")
    L.append("- Distinct from scheme_calib.py's man-vs-zone axis (coverage TYPE, found "
             "null): this is safety-shell alignment (1-high vs 2-high), which cuts across "
             "man and zone alike.")
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

    ya = df.dropna(subset=["signal", "actual_attempts"])
    ya = ya[ya.actual_attempts >= MIN_ATT].copy()
    ya["actual_ypa"] = ya.actual / ya.actual_attempts
    ya["base_ypa"] = ya.base / ya.actual_attempts
    ya_train = ya[ya.season.isin(TRAIN_SEASONS)]
    ya_hold = ya[ya.season == HOLDOUT_SEASON]
    if len(ya_train) > 50 and len(ya_hold) > 50:
        d = (ya_train.base_ypa * ya_train.signal).to_numpy()
        y = (ya_train.actual_ypa - ya_train.base_ypa).to_numpy()
        denom = float(np.dot(d, d))
        beta_ypa = float(np.dot(d, y) / denom) if denom > 0 else 0.0
        corr_hold = ya_hold[["signal"]].assign(resid=ya_hold.actual_ypa - ya_hold.base_ypa).corr().iloc[0, 1]
        corr_train = ya_train[["signal"]].assign(resid=ya_train.actual_ypa - ya_train.base_ypa).corr().iloc[0, 1]
        L.append("## Yards/attempt isolated (strips out attempt-volume noise)")
        L.append("")
        L.append(f"beta fit directly on y/a residual (train): {beta_ypa:+.4f}")
        L.append(f"raw correlation(signal, actual_ypa - base_ypa): train {corr_train:+.4f}, "
                 f"holdout {corr_hold:+.4f}")
        L.append("")

    rb_h, ra_h = _rmse(hold.base, hold.actual), _rmse(hold.adjusted, hold.actual)
    if ra_h < rb_h - 1.0:
        v = "**real edge**"
    elif ra_h < rb_h - 0.1:
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
