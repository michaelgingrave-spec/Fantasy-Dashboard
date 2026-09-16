"""Does a defense's tendency to be strong/weak specifically against SLOT-aligned (vs
WIDE-aligned) receivers -- independent of its overall pass-defense strength -- predict
receiving production, the way man/zone coverage rate didn't (see scheme_calib.py, which
tested this exact mechanism for coverage type and came back clean-null, including on a
tail-concentration recheck)?

Same method as scheme_calib.py, only the axis changes from man/zone to wide/slot:
  1. Player split: his own trailing YPRR when targeted from the SLOT vs from WIDE, each
     minus his own trailing OVERALL YPRR (from the existing unfiltered receiving-advanced
     weekly files already on disk) -- population-shrunk (EB, K_SPLIT routes) toward a
     league-wide prior, same style as opp_line() already uses for share/efficiency.
  2. Defense split: the opposing defense's trailing SLOT-allowed YPT vs WIDE-allowed YPT,
     each vs a league average from train seasons only -- walk-forward.
  3. signal = defense_slot_dev * (slot_edge - wide_edge): positive when a player who's
     relatively better from the slot faces a defense that's relatively softer against
     slot-targeted throws.
  4. adjusted = base_rec_yds * (1 + beta*signal), beta fit by OLS on 2023-24 train,
     judged on a 2025 holdout it never saw.

Data: receiving-targetalign-{wide,slot}_{season}wk{batch}_week.csv (player side, pulled
this session via the site's "Targeted Alignment" play-filter + "Split: Week") and
receiving-targetalign-{wide,slot}-defense_{season}_week.csv (defense side, same filter,
mode=defense). Baseline (unfiltered) YPRR reuses receiving-advanced_*_week.csv, already
on disk from earlier this session -- same export, just without the alignment filter.

    python -m matchup_model.opp.alignment_calib
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
_BATCHES = ["wk1-6", "wk7-12", "wk13-18"]
K_SPLIT = 100.0   # pseudo-count in ROUTES, same constant/style as scheme_calib.py

REPORT = Path(__file__).resolve().parents[1] / "opp_alignment_calibration.md"
ROWS_CSV = Path(__file__).resolve().parents[1] / "opp_alignment_calibration_rows.csv"


def _ewma(v: np.ndarray) -> float:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)][-LOOKBACK:]
    if v.size == 0:
        return np.nan
    w = 0.5 ** (np.arange(v.size)[::-1] / HALFLIFE)
    return float((w * v).sum() / w.sum())


# ── load weekly Data Suite exports (2 header rows, trailing footer) into tidy frames ──
def _read_export(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, header=1)
    d = d[d["Name"].notna() & (d["Name"] != "League Avg")]
    return d


def load_player_alignment(alignment: str) -> pd.DataFrame:
    """{name_key, team, pos, season, week, rte_<a>, tgt_<a>, yprr_<a>} for every
    player-week, `<a>` = 'wide' or 'slot'."""
    a = alignment.lower()
    frames = []
    for season in TRAIN_SEASONS + [HOLDOUT_SEASON]:
        for batch in _BATCHES:
            p = MATCHUP_DATA / f"receiving-targetalign-{a}_{season}{batch}_week.csv"
            if not p.exists():
                continue
            d = _read_export(p)
            d = d.rename(columns={"WEEK": "week", "RTE": f"rte_{a}", "TGT": f"tgt_{a}",
                                  "YPRR": f"yprr_{a}"})
            keep = ["Name", "Team", "POS", "week", f"rte_{a}", f"tgt_{a}", f"yprr_{a}"]
            d = d[[c for c in keep if c in d.columns]].copy()
            d["season"] = season
            frames.append(d)
    cols = ["name_key", "Team", "POS", "season", "week", f"rte_{a}", f"tgt_{a}", f"yprr_{a}"]
    if not frames:
        return pd.DataFrame(columns=cols)
    df = pd.concat(frames, ignore_index=True)
    df["name_key"] = df["Name"].map(normalize_name)
    df["week"] = pd.to_numeric(df["week"], errors="coerce")
    for c in (f"rte_{a}", f"tgt_{a}", f"yprr_{a}"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["week"]).sort_values(["name_key", "season", "week"])


def load_player_splits() -> pd.DataFrame:
    """Wide + slot player splits merged on (name_key, team, season, week), plus the
    unfiltered baseline (yprr_all, rte_all) from the already-on-disk receiving-advanced
    weekly files."""
    wide = load_player_alignment("wide")
    slot = load_player_alignment("slot")
    m = wide.merge(slot.drop(columns=["Team", "POS"], errors="ignore"),
                   on=["name_key", "week", "season"], how="outer")

    base_frames = []
    for season in TRAIN_SEASONS + [HOLDOUT_SEASON]:
        for batch in _BATCHES:
            p = MATCHUP_DATA / f"receiving-advanced_{season}{batch}_week.csv"
            if not p.exists():
                continue
            d = _read_export(p)
            d = d.rename(columns={"WEEK": "week", "RTE": "rte_all", "YPRR": "yprr_all"})
            keep = ["Name", "week", "rte_all", "yprr_all"]
            d = d[[c for c in keep if c in d.columns]].copy()
            d["season"] = season
            base_frames.append(d)
    base = pd.concat(base_frames, ignore_index=True)
    base["name_key"] = base["Name"].map(normalize_name)
    base["week"] = pd.to_numeric(base["week"], errors="coerce")
    base["rte_all"] = pd.to_numeric(base["rte_all"], errors="coerce")
    base["yprr_all"] = pd.to_numeric(base["yprr_all"], errors="coerce")
    base = base.dropna(subset=["week"])[["name_key", "season", "week", "rte_all", "yprr_all"]]

    m = m.merge(base, on=["name_key", "season", "week"], how="left")
    return m.sort_values(["name_key", "season", "week"])


def load_defense_alignment(alignment: str) -> pd.DataFrame:
    """{defense, season, week, ypt_<a>} per team-week, `<a>` = 'wide' or 'slot'."""
    a = alignment.lower()
    frames = []
    for season in TRAIN_SEASONS + [HOLDOUT_SEASON]:
        p = MATCHUP_DATA / f"receiving-targetalign-{a}-defense_{season}_week.csv"
        if not p.exists():
            continue
        d = _read_export(p)
        d = d.rename(columns={"WEEK": "week", "YPT": f"ypt_{a}"})
        keep = ["Name", "week", f"ypt_{a}"]
        d = d[[c for c in keep if c in d.columns]].copy()
        d["season"] = season
        frames.append(d)
    cols = ["defense", "season", "week", f"ypt_{a}"]
    if not frames:
        return pd.DataFrame(columns=cols)
    df = pd.concat(frames, ignore_index=True)
    df["defense"] = df["Name"].map(norm_team)
    df["week"] = pd.to_numeric(df["week"], errors="coerce")
    df[f"ypt_{a}"] = pd.to_numeric(df[f"ypt_{a}"], errors="coerce")
    return df.dropna(subset=["week"]).sort_values(["defense", "season", "week"])


def load_defense_splits() -> pd.DataFrame:
    wide = load_defense_alignment("wide")
    slot = load_defense_alignment("slot")
    return wide.merge(slot, on=["defense", "season", "week"], how="outer")


# ── walk-forward trailing lookups (mirrors scheme_calib.py exactly) ────────────────
def _pop_priors(ps: pd.DataFrame, before_season: int) -> tuple[float, float]:
    d = ps[ps.season < before_season]
    if d.empty:
        d = ps
    d = d.dropna(subset=["yprr_all"])
    dw = d.dropna(subset=["yprr_wide", "rte_wide"])
    ds = d.dropna(subset=["yprr_slot", "rte_slot"])
    wide_prior = float(np.average(dw.yprr_wide - dw.yprr_all, weights=dw.rte_wide.clip(lower=0.1))) \
        if len(dw) else 0.0
    slot_prior = float(np.average(ds.yprr_slot - ds.yprr_all, weights=ds.rte_slot.clip(lower=0.1))) \
        if len(ds) else 0.0
    return wide_prior, slot_prior


def _player_edge_shrunk(ps: pd.DataFrame, name_key: str, season: int, week: int,
                        wide_prior: float, slot_prior: float) -> tuple[float, float]:
    h = ps[(ps.name_key == name_key) &
          ((ps.season < season) | ((ps.season == season) & (ps.week < week)))].tail(LOOKBACK)
    if h.empty:
        return wide_prior, slot_prior
    base = _ewma(h["yprr_all"].to_numpy())
    if not np.isfinite(base):
        return wide_prior, slot_prior

    def _shrink(col_yprr: str, col_rte: str, prior: float) -> float:
        hh = h.dropna(subset=[col_yprr, col_rte])
        if hh.empty:
            return prior
        n = float(hh[col_rte].sum())
        raw = float(np.average(hh[col_yprr], weights=hh[col_rte].clip(lower=0.1))) - base
        return (n * raw + K_SPLIT * prior) / (n + K_SPLIT) if (n + K_SPLIT) > 0 else prior

    return (_shrink("yprr_wide", "rte_wide", wide_prior),
           _shrink("yprr_slot", "rte_slot", slot_prior))


def _defense_dev(dg: pd.DataFrame, defense: str, season: int, week: int,
                 col: str, league_avg: float) -> float:
    h = dg[(dg.defense == defense) &
          ((dg.season < season) | ((dg.season == season) & (dg.week < week)))].tail(LOOKBACK)
    if len(h) < MIN_GAMES or not np.isfinite(league_avg) or league_avg <= 0:
        return 0.0
    val = _ewma(h[col].dropna().to_numpy())
    return (val / league_avg - 1.0) if np.isfinite(val) else 0.0


def _league_avg(dg: pd.DataFrame, col: str, before_season: int) -> float:
    d = dg[dg.season < before_season]
    if d.empty:
        d = dg
    return float(d[col].mean())


def run() -> pd.DataFrame:
    pw = M._pw_with_team()
    ps = load_player_splits()
    dg = load_defense_splits()
    print(f"  loaded {len(ps)} player-split rows, {len(dg)} defense-week rows")
    t0 = time.time()
    rows = []
    for season in TRAIN_SEASONS + [HOLDOUT_SEASON]:
        lg_slot = _league_avg(dg, "ypt_slot", season)
        lg_wide = _league_avg(dg, "ypt_wide", season)
        wide_prior, slot_prior = _pop_priors(ps, season)
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
            wide_edge, slot_edge = _player_edge_shrunk(ps, nk, season, wk, wide_prior, slot_prior)
            slot_dev = _defense_dev(dg, opp, season, wk, "ypt_slot", lg_slot)
            wide_dev = _defense_dev(dg, opp, season, wk, "ypt_wide", lg_wide)
            signal = slot_dev * (slot_edge - wide_edge)
            rows.append(dict(season=season, week=wk, player=r["player_display_name"],
                             pos=r["position"], defense=opp, base=float(base),
                             wide_edge=round(wide_edge, 4), slot_edge=round(slot_edge, 4),
                             slot_dev=round(slot_dev, 4), wide_dev=round(wide_dev, 4),
                             signal=round(signal, 4),
                             actual=float(r["receiving_yards"]),
                             actual_targets=float(r["targets"])))
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
    L = ["# Slot vs wide alignment-conditioned efficiency vs opponent tendency", ""]
    L.append(f"- {len(df)} WR/TE player-games, wk {TEST_WEEKS.start}-{TEST_WEEKS.stop-1}, "
             f"seasons {TRAIN_SEASONS}+{HOLDOUT_SEASON}.")
    L.append("- `signal` = defense's trailing slot-allowed-YPT deviation from league avg x "
             "(player's shrunk slot-edge minus wide-edge) -- positive when a "
             "relatively-better-from-the-slot player faces a defense that's relatively "
             "softer against slot-targeted throws.")
    L.append(f"- Player edges are EB-shrunk toward a population prior (K={K_SPLIT:.0f} routes), "
             "same style as opp_line() already uses -- and scheme_calib.py used for man/zone.")
    L.append("")

    train = df[df.season.isin(TRAIN_SEASONS)].dropna(subset=["signal"])
    hold = df[df.season == HOLDOUT_SEASON].dropna(subset=["signal"])
    if len(train) < 100 or len(hold) < 50:
        L.append("**insufficient rows, aborting**")
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

    # tail-concentration cut -- the check that would've caught a real-but-averaged-away
    # effect, same as the man/zone recheck and the featured-target test this session.
    hold = hold.assign(abs_signal=hold.signal.abs())
    L.append("## Holdout RMSE by |signal| tercile (does it concentrate in the biggest mismatches?)")
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

    # direction agreement -- does the nudge even point the right way?
    sub = hold[np.sign(hold.adjusted - hold.base) != 0]
    if len(sub):
        agree = (np.sign(sub.adjusted - sub.base) == np.sign(sub.actual - sub.base)).mean()
        L.append(f"Direction agreement (nudge sign vs actual-vs-base sign, n={len(sub)}): "
                 f"**{agree:.1%}** (50% = coin flip)")
        L.append("")

    # yards/target isolated -- strip out target-volume noise, test the mechanism directly
    yt = df.dropna(subset=["signal", "actual_targets"])
    yt = yt[yt.actual_targets >= 2].copy()
    yt["actual_ypt"] = yt.actual / yt.actual_targets
    yt["base_ypt"] = yt.base / yt.actual_targets
    yt_train = yt[yt.season.isin(TRAIN_SEASONS)]
    yt_hold = yt[yt.season == HOLDOUT_SEASON]
    if len(yt_train) > 50 and len(yt_hold) > 50:
        d = (yt_train.base_ypt * yt_train.signal).to_numpy()
        y = (yt_train.actual_ypt - yt_train.base_ypt).to_numpy()
        denom = float(np.dot(d, d))
        beta_ypt = float(np.dot(d, y) / denom) if denom > 0 else 0.0
        corr_hold = yt_hold[["signal"]].assign(resid=yt_hold.actual_ypt - yt_hold.base_ypt).corr().iloc[0, 1]
        corr_train = yt_train[["signal"]].assign(resid=yt_train.actual_ypt - yt_train.base_ypt).corr().iloc[0, 1]
        L.append("## Yards/target isolated (strips out target-volume noise)")
        L.append("")
        L.append(f"beta fit directly on y/t residual (train): {beta_ypt:+.4f}")
        L.append(f"raw correlation(signal, actual_ypt - base_ypt): train {corr_train:+.4f}, "
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
