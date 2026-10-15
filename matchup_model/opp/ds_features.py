"""Weekly Data Suite tables -> as-of-week matchup features (research layer for matchup_study.py).

Everything is built strictly from games BEFORE the week being predicted:
  player side  : the player's own trailing usage / efficiency / alignment / man-zone-shell profile (EWMA half-life 4, last 10 games) plus the
                 latest game on its own (last 1 / last 3 vs the EWMA = "trend")
  defense side : the opponent's trailing coverage / box / pressure rates and what it has ALLOWED (per route, per carry), shrunk toward the league
  mismatch     : where they meet -- alignment mix x what the defense allows from each alignment, man/zone/shell edges x how often the
                 defense plays them, run-concept mix x zone/gap yards allowed, ...

Data caveat that shapes everything here: the weekly PLAYER exports are only trustworthy for seasons whose files were pulled unfiltered. The 2025
receiving exports were filtered on that week's outcome (0% of 0-target games present, 98% of 9+ target games), which would bias any trailing
feature built from them; `presence_by_usage` / `filtered_seasons` detect that so a season like it is excluded instead of silently used.
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd

from dfs.names import norm_team, normalize_name
from matchup_model.config import DATA

HL, WIN, DWIN = 4.0, 10, 8
K_ALLOWED = 300.0      # routes (or carries, for the run game) of pseudo-sample when shrinking a defense's allowed rate to the league
K_EDGE = 100.0         # routes when shrinking a player's man/zone/shell edge to 0


# ── readers ─────────────────────────────────────────────────────────────────────────────────────────────
def read_weekly(stem: str, seasons=None) -> pd.DataFrame:
    """Every `{stem}_{season}..._week.csv` export (two header rows, trailing footer) as one frame; League Avg rows dropped."""
    frames = []
    for p in sorted(glob.glob(str(DATA / f"{stem}_20*_week.csv"))):
        df = pd.read_csv(p, header=1)
        if "Name" not in df.columns:
            continue
        frames.append(df[df["Name"].notna() & (df["Name"] != "League Avg") & ~df["Name"].astype(str).str.contains("Exported", na=False)])
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True)
    d["Season"], d["WEEK"] = pd.to_numeric(d["Season"], errors="coerce"), pd.to_numeric(d["WEEK"], errors="coerce")
    d = d.dropna(subset=["Season", "WEEK"])
    return (d[d.Season.isin(seasons)] if seasons is not None else d).reset_index(drop=True)


def _num(d: pd.DataFrame, skip=("name", "team", "pos", "opp", "def", "off", "nk")) -> pd.DataFrame:
    for c in d.columns:
        if c not in skip:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def presence_by_usage(P: pd.DataFrame, played: pd.DataFrame, usage: str, low: tuple[int, int], high: tuple[int, int]) -> pd.DataFrame:
    """Per season, the share of players nflverse has on the field (`played`: season, week, nk and a `usage` column) that the Data Suite weekly
    export `P` lists, for low-usage weeks vs high-usage weeks. An unfiltered export lists a player whatever he did that week (WR/TE 2022-24: ~60-85% of
    0-target games, 100% of the rest); one filtered on the week's outcome drops the quiet games (2025: 0% / 10% / 50% / 86% / 98% from 0 to 9+ targets)."""
    have = set(zip(P.season, P.week, P.nk))
    d = played[played.season.isin(P.season.unique())].copy()
    d["present"] = [(s, w, n) in have for s, w, n in zip(d.season, d.week, d.nk)]
    u = d[usage].fillna(0)
    lo, hi = d[(u >= low[0]) & (u <= low[1])], d[(u >= high[0]) & (u <= high[1])]
    out = pd.DataFrame({"n_low": lo.groupby("season").size(), "low": lo.groupby("season").present.mean(),
                        "n_high": hi.groupby("season").size(), "high": hi.groupby("season").present.mean()})
    out["ratio"] = out.low / out.high
    return out


def filtered_seasons(presence: pd.DataFrame, max_ratio: float = 0.4, min_rows: int = 100) -> list[int]:
    """Seasons whose export drops quiet games: low-usage presence under `max_ratio` of high-usage presence (unfiltered seasons sit at 0.7-1.0, the
    filtered 2025 receiving export at 0.05). Seasons with too few comparison rows are not judged."""
    ok = presence[(presence.n_low >= min_rows) & (presence.n_high >= min_rows)]
    return [int(s) for s, r in ok.iterrows() if r.ratio < max_ratio]


RA_MAP = {  # receiving-advanced; 'RTE %.1' is route participation and 'TGT %.1' team target share (verified: corr 0.99 with nflverse target_share)
    "Season": "season", "WEEK": "week", "Name": "name", "Team": "team", "POS": "pos", "OPP": "opp", "RTE %.1": "rte_part", "TGT %.1": "tgt_share",
    "SNAP %": "snap_pct", "RTE": "rte", "TGT": "tgt", "YPRR": "yprr", "TPRR": "tprr", "aDOT": "adot", "AY": "ay", "AY Share": "ay_share", "YDS": "yds",
    "YPT": "ypt", "REC": "rec", "CR %": "cr", "YAC/REC": "yac_rec", "TD": "td", "XTD": "xtd", "EZTGT": "eztgt", "i20 TGT": "i20tgt", "1READ %": "read1",
    "DESIGN %": "design", "CTGT %": "ctgt", "DEEP": "deep", "EXP %": "exp", "DRP %": "drp", "THREAT %": "threat", "YPTOE": "yptoe", "EPA/TGT": "epa_tgt",
    "WIDE RTE %": "wide", "SLOT RTE %": "slot", "INLINE RTE %": "inline", "BACK RTE %": "back", "FP": "fp", "XFP": "xfp", "XFP/RR": "xfp_rr",
    "FP/RR": "fp_rr", "DIFF": "fp_diff"}
MZ_MAP = {"Season": "season", "WEEK": "week", "Name": "name", "Team": "team", "RTE": "z_rte", "TPRR": "z_tprr", "YPRR": "z_yprr", "RTE.1": "man_rte",
          "TPRR.1": "man_tprr", "YPRR.1": "man_yprr", "RTE.2": "zone_rte", "TPRR.2": "zone_tprr", "YPRR.2": "zone_yprr", "RTE.3": "sh_rte",
          "TPRR.3": "sh_tprr", "YPRR.3": "sh_yprr", "RTE.4": "th_rte", "TPRR.4": "th_tprr", "YPRR.4": "th_yprr"}
CM_MAP = {"Season": "season", "WEEK": "week", "Name": "def", "OPP": "off", "MAN %": "man", "ZONE %": "zone", "1-HI/MOF C %": "one_hi", "2-HI/MOF O %": "two_hi",
          "DISGUISE %": "disguise", "COVER 0 %": "c0", "COVER 1 %": "c1", "COVER 2 %": "c2", "COVER 2 MAN %": "c2m", "COVER 3 %": "c3", "COVER 4 %": "c4",
          "COVER 6 %": "c6", "BASE %": "base", "NICKEL %": "nickel", "DIME %": "dime", "PRESS % (ANY)": "press", "PRESS % (TGT)": "press_tgt"}


def receiving_players(seasons=None) -> pd.DataFrame:
    """WR/TE player-weeks: receiving-advanced + man/zone/shell splits, keyed (season, week, nk)."""
    ra = read_weekly("receiving-advanced", seasons).rename(columns=RA_MAP)
    ra = _num(ra[[c for c in RA_MAP.values() if c in ra.columns]])
    mz = read_weekly("receiving-manvszone", seasons).rename(columns=MZ_MAP)
    mz = _num(mz[[c for c in MZ_MAP.values() if c in mz.columns]])
    d = ra.merge(mz, on=["season", "week", "name", "team"], how="left")
    d["nk"] = d.name.map(normalize_name)
    d["team"], d["opp"] = d.team.map(norm_team), d.opp.map(norm_team)
    d["season"], d["week"] = d.season.astype(int), d.week.astype(int)
    return d.sort_values(["nk", "season", "week"]).reset_index(drop=True)


def coverage(seasons=None) -> pd.DataFrame:
    """Defense-weeks: man/zone, shells, personnel, press rates (coverage-matrix)."""
    d = read_weekly("coverage-matrix", seasons).rename(columns=CM_MAP)
    d = _num(d[[c for c in CM_MAP.values() if c in d.columns]])
    d["def"], d["off"] = d["def"].map(norm_team), d["off"].map(norm_team)
    d["season"], d["week"] = d.season.astype(int), d.week.astype(int)
    return d.sort_values(["def", "season", "week"]).reset_index(drop=True)


# ── as-of helpers ───────────────────────────────────────────────────────────────────────────────────────
def key(season, week) -> int:
    return int(season) * 100 + int(week)


def window(keys: np.ndarray, k: int, size: int) -> tuple[int, int]:
    """Slice [lo, hi) of the sorted game keys: the last `size` games strictly before key `k`, none older than the previous season (a season
    with no usable export -- 2025 WR/TE -- must not let two-year-old games stand in for last year's)."""
    hi = int(np.searchsorted(keys, k, side="left"))
    lo = int(np.searchsorted(keys, (k // 100 - 1) * 100, side="left"))
    return max(hi - size, lo), hi


def decay(n: int, base=None) -> np.ndarray:
    """Recency weights for `n` games, oldest first (half-life HL games), optionally times a usage weight such as routes."""
    w = 0.5 ** (np.arange(n)[::-1] / HL)
    return w if base is None else w * base


def ew(v, base=None) -> float:
    """Recency-weighted (half-life HL games), optionally also usage-weighted, mean of v (oldest -> newest)."""
    v = np.asarray(v, float)
    w = decay(len(v), base)
    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    return float((v[m] * w[m]).sum() / w[m].sum()) if m.any() else np.nan


def ratio(num, den) -> float:
    """Recency-weighted ratio of sums (sum(num)/sum(den))."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    w = decay(len(num))
    m = np.isfinite(num) & np.isfinite(den)
    d = (den[m] * w[m]).sum()
    return float((num[m] * w[m]).sum() / d) if d > 0 else np.nan


def _index(P: pd.DataFrame) -> dict:
    out = {}
    for nk, g in P.groupby("nk"):
        g = g.sort_values(["season", "week"])
        out[nk] = (g.season.to_numpy() * 100 + g.week.to_numpy(),
                   {c: g[c].to_numpy(float) for c in g.columns if c not in ("name", "team", "pos", "opp", "nk")})
    return out


# ── WR/TE ───────────────────────────────────────────────────────────────────────────────────────────────
G_METRICS = ["rte_part", "tgt_share", "snap_pct", "rte", "tgt", "xfp", "fp", "ay_share"]                 # per game
R_METRICS = ["tprr", "yprr", "adot", "fp_rr", "xfp_rr", "wide", "slot", "inline", "back"]                # per route
T_METRICS = ["cr", "read1", "design", "ctgt", "exp", "threat", "yptoe", "epa_tgt", "drp"]                  # per target


def player_features(P: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """One feature row per `rows` row from that player's rows strictly before (season, week). Columns prefixed `p_`."""
    idx = _index(P)
    out = []
    for r in rows.itertuples():
        f = {}
        hit = idx.get(r.nk)
        n = 0
        if hit is not None:
            keys, cols = hit
            lo, hi = window(keys, key(r.season, r.week), WIN)
            n = hi - lo
            if n:
                a = {c: v[lo:hi] for c, v in cols.items()}
                rte, tgt = np.nan_to_num(a["rte"]), np.nan_to_num(a["tgt"])
                for m in G_METRICS:
                    f[f"p_{m}"] = ew(a[m])
                for m in R_METRICS:
                    f[f"p_{m}"] = ew(a[m], rte)
                for m in T_METRICS:
                    f[f"p_{m}"] = ew(a[m], tgt)
                f["p_deep_rate"] = ratio(a["deep"], a["tgt"])
                for m in ("rte_part", "tgt_share", "tprr", "snap_pct"):
                    f[f"p_last1_{m}"] = float(a[m][-1])
                    f[f"p_last3_{m}"] = float(np.nanmean(a[m][-3:])) if np.isfinite(a[m][-3:]).any() else np.nan
                    f[f"p_trend_{m}"] = f[f"p_last3_{m}"] - f[f"p_{m}"] if np.isfinite(f[f"p_{m}"]) else np.nan
                f["p_routes_total"] = float(rte.sum())
                z_yprr, z_tprr = ratio(a["z_yprr"] * a["z_rte"], a["z_rte"]), ratio(a["z_tprr"] * a["z_rte"], a["z_rte"])
                tot_mz = np.nan_to_num(a["man_rte"]) + np.nan_to_num(a["zone_rte"])
                tot_sh = np.nan_to_num(a["sh_rte"]) + np.nan_to_num(a["th_rte"])
                for s, tot in (("man", tot_mz), ("zone", tot_mz), ("sh", tot_sh), ("th", tot_sh)):
                    rt = np.nan_to_num(a[f"{s}_rte"])
                    y = ratio(np.nan_to_num(a[f"{s}_yprr"]) * rt, rt)
                    t = ratio(np.nan_to_num(a[f"{s}_tprr"]) * rt, rt)
                    nr = float((rt * decay(len(rt))).sum())
                    f[f"p_{s}_share"] = ratio(rt, tot) if tot.sum() > 0 else np.nan
                    f[f"p_{s}_yprr_edge"] = nr * (y - z_yprr) / (nr + K_EDGE) if np.isfinite(y) and np.isfinite(z_yprr) else 0.0
                    f[f"p_{s}_tprr_edge"] = nr * (t - z_tprr) / (nr + K_EDGE) if np.isfinite(t) and np.isfinite(z_tprr) else 0.0
        f["p_n_prior"] = n
        out.append(f)
    return pd.DataFrame(out, index=rows.index)


COV_COLS = ["man", "zone", "one_hi", "two_hi", "disguise", "c0", "c1", "c2", "c2m", "c3", "c4", "c6", "base", "nickel", "dime", "press", "press_tgt"]
CLASSES = {"all": lambda d: np.ones(len(d), bool), "wr": lambda d: (d.pos == "WR").to_numpy(), "te": lambda d: (d.pos == "TE").to_numpy(),
           "slot": lambda d: (d.slot.fillna(0) >= 50).to_numpy(), "wide": lambda d: (d.wide.fillna(0) >= 60).to_numpy(),
           "inline": lambda d: (d.inline.fillna(0) >= 50).to_numpy()}
_ALLOWED = ["rte", "tgt", "yds", "ay", "rec", "xfp", "fp"]


def allowed_tables(P: pd.DataFrame) -> dict:
    """{class: frame of what each defense (the row's `opp`) gave up per week to that class of receiver}."""
    out = {}
    for name, sel in CLASSES.items():
        g = P[sel(P)].groupby(["opp", "season", "week"]).agg(rte=("rte", "sum"), tgt=("tgt", "sum"), yds=("yds", "sum"), ay=("ay", "sum"),
                                                              rec=("rec", "sum"), xfp=("xfp", "sum"), fp=("fp", "sum")).reset_index()
        g["key"] = g.season * 100 + g.week
        out[name] = g.sort_values(["opp", "key"])
    return out


def league_rates(tab: pd.DataFrame, k: int, nweeks: int = 6) -> dict:
    keys = np.sort(tab.key.unique())
    s = tab[tab.key.isin(keys[keys < k][-nweeks:])]
    if s.empty:
        return {}
    r = s[_ALLOWED].sum()
    return {"tprr": r.tgt / r.rte, "yprr": r.yds / r.rte, "adot": r.ay / max(r.tgt, 1), "cr": r.rec / max(r.tgt, 1), "xfp_rr": r.xfp / r.rte,
            "fin": (r.fp - r.xfp) / r.rte}


def coverage_features(C: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """The opponent's trailing coverage / personnel / press rates, raw (`d_man`) and against the league (`d_man_dev`)."""
    cov = {d: g.sort_values(["season", "week"]) for d, g in C.groupby("def")}
    cov_keys = {d: g.season.to_numpy() * 100 + g.week.to_numpy() for d, g in cov.items()}
    cm_week = C.assign(key=C.season * 100 + C.week).groupby("key")[COV_COLS].mean()
    cm_cache = {}
    out = []
    for r in rows.itertuples():
        k = key(r.season, r.week)
        f = {}
        d = r.opp
        if d in cov:
            lo, hi = window(cov_keys[d], k, DWIN)
            g = cov[d].iloc[lo:hi]
            if len(g):
                if k not in cm_cache:
                    prev = cm_week[cm_week.index < k].tail(6)
                    cm_cache[k] = prev.mean() if len(prev) else None
                lgm = cm_cache[k]
                for c in COV_COLS:
                    v = ew(g[c].to_numpy())
                    f[f"d_{c}"] = v
                    f[f"d_{c}_dev"] = v - float(lgm[c]) if lgm is not None else np.nan
        out.append(f)
    return pd.DataFrame(out, index=rows.index)


def defense_features(C: pd.DataFrame, A: dict, rows: pd.DataFrame) -> pd.DataFrame:
    """Opponent coverage rates (raw and vs the league) and allowed-by-class rates (vs the league). Columns prefixed `d_`."""
    al = {name: {d: (g.key.to_numpy(), g[_ALLOWED].to_numpy(float)) for d, g in tab.groupby("opp")} for name, tab in A.items()}
    lg_cache = {}
    out = []
    for r in rows.itertuples():
        k = key(r.season, r.week)
        f = {}
        d = r.opp
        f["d_n_prior"] = 0
        for name in ("all", "wr", "te", "slot", "wide", "inline"):
            hit = al[name].get(d)
            if hit is None:
                continue
            keys, arr = hit
            lo, hi = window(keys, k, DWIN)
            a = arr[lo:hi]
            if not len(a):
                continue
            if (name, k) not in lg_cache:
                lg_cache[(name, k)] = league_rates(A[name], k)
            lg = lg_cache[(name, k)]
            if not lg:
                continue
            S = (a * decay(len(a))[:, None]).sum(axis=0)          # decayed sums: rte, tgt, yds, ay, rec, xfp, fp
            rte, tgt = S[0], S[1]
            if name == "all":
                f["d_n_prior"] = len(a)

            def shr(x, n, L):
                return (n * x + K_ALLOWED * L) / (n + K_ALLOWED) if np.isfinite(x) else L
            f[f"d_{name}_tprr_rel"] = shr(tgt / rte if rte > 0 else np.nan, rte, lg["tprr"]) / lg["tprr"] - 1
            f[f"d_{name}_yprr_rel"] = shr(S[2] / rte if rte > 0 else np.nan, rte, lg["yprr"]) / lg["yprr"] - 1
            f[f"d_{name}_xfprr_rel"] = shr(S[5] / rte if rte > 0 else np.nan, rte, lg["xfp_rr"]) / lg["xfp_rr"] - 1
            f[f"d_{name}_adot_dev"] = shr(S[3] / tgt if tgt > 0 else np.nan, tgt * 3, lg["adot"]) - lg["adot"]
            f[f"d_{name}_cr_dev"] = shr(S[4] / tgt if tgt > 0 else np.nan, tgt * 3, lg["cr"]) - lg["cr"]
            f[f"d_{name}_fin_dev"] = shr((S[6] - S[5]) / rte if rte > 0 else np.nan, rte, lg["fin"]) - lg["fin"]
        out.append(f)
    return pd.concat([coverage_features(C, rows), pd.DataFrame(out, index=rows.index)], axis=1)


def _with(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """`df` with every column in `cols` present (NaN if absent): a batch where nobody has history has no such feature columns at all."""
    return df.reindex(columns=list(dict.fromkeys([*df.columns, *cols])))


def mismatch_features(pf: pd.DataFrame, df: pd.DataFrame, week_keys: pd.Series | None = None) -> pd.DataFrame:
    """Where the receiver and the defense meet (columns prefixed `mm_`; positive = favourable for the receiver). `week_keys` (season*100+week per row)
    lets the depth mismatch -- how deep he is used, z-scored among that week's receivers, x how much depth the defense gives up -- be built."""
    pf = _with(pf, ["p_wide", "p_slot", "p_inline", "p_adot", "p_deep_rate", *[f"p_{s}_{k}_edge" for s in ("man", "zone", "sh", "th") for k in ("yprr", "tprr")]])
    m = pd.DataFrame(index=pf.index)
    al = pf[["p_wide", "p_slot", "p_inline"]].fillna(0.0)
    sh = al.div(al.sum(axis=1).replace(0, np.nan), axis=0)
    for metric, label in (("yprr_rel", "yprr"), ("tprr_rel", "tprr"), ("xfprr_rel", "xfprr")):
        parts = [sh[f"p_{c}"] * df[f"d_{c}_{metric}"] for c in ("wide", "slot", "inline") if f"d_{c}_{metric}" in df.columns]
        m[f"mm_align_{label}"] = sum(p.fillna(0) for p in parts) if parts else np.nan
    for c in ("man", "zone"):
        if f"d_{c}_dev" in df.columns:
            m[f"mm_{c}_yprr"] = df[f"d_{c}_dev"] * pf[f"p_{c}_yprr_edge"]
            m[f"mm_{c}_tprr"] = df[f"d_{c}_dev"] * pf[f"p_{c}_tprr_edge"]
    for s, c in (("sh", "one_hi"), ("th", "two_hi")):
        if f"d_{c}_dev" in df.columns:
            m[f"mm_{s}_yprr"] = df[f"d_{c}_dev"] * pf[f"p_{s}_yprr_edge"]
            m[f"mm_{s}_tprr"] = df[f"d_{c}_dev"] * pf[f"p_{s}_tprr_edge"]
    if {"d_nickel_dev", "d_dime_dev", "d_base_dev"} <= set(df.columns):
        m["mm_nickel_slot"] = (df.d_nickel_dev + df.d_dime_dev) * sh["p_slot"]
        m["mm_base_inline"] = df.d_base_dev * sh["p_inline"]
    m["mm_scheme_sum"] = m[[c for c in m.columns if c.endswith("_yprr") and c != "mm_align_yprr"]].sum(axis=1, min_count=1)
    if week_keys is not None and "d_all_adot_dev" in df.columns:
        for src, label in (("p_adot", "adot"), ("p_deep_rate", "deep")):
            z = pf[src].groupby(week_keys).transform(lambda x: (x - x.mean()) / x.std())
            m[f"mm_depth_{label}"] = z * df.d_all_adot_dev
    return m


def build_receiving(rows: pd.DataFrame, P: pd.DataFrame, C: pd.DataFrame) -> pd.DataFrame:
    pf = player_features(P, rows)
    df = defense_features(C, allowed_tables(P), rows)
    return pd.concat([rows, pf, df, mismatch_features(pf, df, rows.season * 100 + rows.week)], axis=1)


# ── RB rushing ──────────────────────────────────────────────────────────────────────────────────────────
RB_MAP = {"Season": "season", "WEEK": "week", "Name": "name", "Team": "team", "OPP": "opp", "ATT %": "att_pct", "SNAP %": "snap_pct", "ATT": "att",
          "DCS %": "dcs", "YDS": "yds", "YPC": "ypc", "XTD": "xtd", "SUCC %": "succ", "STUFF %": "stuff", "RYOE": "ryoe", "YBC/ATT": "ybc", "HIT %": "hit",
          "YACO/ATT": "yaco", "EPA/A": "epa", "ZONE ATT": "z_att", "ZONE YDS": "z_yds", "MAN ATT": "g_att", "MAN YDS": "g_yds", "i5 %": "i5",
          "EXP RUN %": "exp", "FP": "fp", "XFP": "xfp"}
RB_BASIC_MAP = {"Season": "season", "WEEK": "week", "Name": "name", "Team": "team", "TGT": "tgt", "TGT %": "tgt_pct", "REC": "rec", "REC YDS": "rec_yds",
                "CR %": "cr", "YPT": "ypt"}
BOX_MAP = {"Name": "def", "OPP": "off", "BOX": "box", "RUSH SUCC %": "r_succ", "EPA/RU": "r_epa", "RUSH EXP %": "r_exp", "BASE %": "base", "SUB %": "sub",
           "MAN %": "man", "ZONE %": "zone", "1-HIGH %": "one_hi", "2-HIGH %": "two_hi", "PRESS %": "press", "BLITZ %": "blitz", "PASS %": "pass_faced",
           "PROE": "proe", "RB FPG": "rb_fpg", "Season": "season", "WEEK": "week"}
BOX_COLS = ["box", "r_succ", "r_epa", "r_exp", "base", "sub", "man", "zone", "one_hi", "two_hi", "press", "blitz", "pass_faced", "proe", "rb_fpg"]


def rushing_players(seasons=None) -> pd.DataFrame:
    ra = read_weekly("rushing-advanced", seasons).rename(columns=RB_MAP)
    ra = ra[[c for c in RB_MAP.values() if c in ra.columns]]
    rb = read_weekly("rushing-basic", seasons).rename(columns=RB_BASIC_MAP)
    rb = rb[[c for c in RB_BASIC_MAP.values() if c in rb.columns]]
    d = _num(ra.merge(rb, on=["season", "week", "name", "team"], how="left"))
    d["nk"] = d.name.map(normalize_name)
    d["team"], d["opp"] = d.team.map(norm_team), d.opp.map(norm_team)
    d["season"], d["week"] = d.season.astype(int), d.week.astype(int)
    d["ryoe_att"] = d.ryoe / d.att.replace(0, np.nan)
    d["z_share"] = d.z_att / (d.z_att + d.g_att).replace(0, np.nan)
    d["z_ypc"] = d.z_yds / d.z_att.replace(0, np.nan)
    d["g_ypc"] = d.g_yds / d.g_att.replace(0, np.nan)
    return d.sort_values(["nk", "season", "week"]).reset_index(drop=True)


def defense_box(seasons=None) -> pd.DataFrame:
    b = read_weekly("team-defense-box", seasons).rename(columns=BOX_MAP)
    b = _num(b[[c for c in BOX_MAP.values() if c in b.columns]])
    b["def"] = b["def"].map(norm_team)
    b["season"], b["week"] = b.season.astype(int), b.week.astype(int)
    return b.sort_values(["def", "season", "week"]).reset_index(drop=True)


RB_G = ["att_pct", "snap_pct", "att", "dcs", "xtd", "xfp", "fp", "tgt_pct", "tgt", "i5", "z_share"]
RB_A = ["ypc", "succ", "stuff", "ryoe_att", "ybc", "hit", "yaco", "epa", "exp", "z_ypc", "g_ypc"]


def rb_player_features(P: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    idx = _index(P)
    out = []
    for r in rows.itertuples():
        f = {}
        hit = idx.get(r.nk)
        n = 0
        if hit is not None:
            keys, cols = hit
            lo, hi = window(keys, key(r.season, r.week), WIN)
            n = hi - lo
            if n:
                a = {c: v[lo:hi] for c, v in cols.items()}
                att = np.nan_to_num(a["att"])
                for m in RB_G:
                    f[f"p_{m}"] = ew(a[m])
                for m in RB_A:
                    f[f"p_{m}"] = ew(a[m], att)
                f["p_yds_pg"] = ew(a["yds"])
                for m in ("att_pct", "snap_pct", "att", "tgt_pct"):
                    f[f"p_last1_{m}"] = float(a[m][-1])
                    f[f"p_last3_{m}"] = float(np.nanmean(a[m][-3:])) if np.isfinite(a[m][-3:]).any() else np.nan
                    f[f"p_trend_{m}"] = f[f"p_last3_{m}"] - f[f"p_{m}"] if np.isfinite(f[f"p_{m}"]) else np.nan
                f["p_att_total"] = float(att.sum())
        f["p_n_prior"] = n
        out.append(f)
    return pd.DataFrame(out, index=rows.index)


_RB_ALLOWED = ["att", "yds", "z_att", "z_yds", "g_att", "g_yds", "succ_n", "stuff_n", "ybc_n", "ryoe_n"]


def rb_allowed_table(P: pd.DataFrame) -> pd.DataFrame:
    g = P.groupby(["opp", "season", "week"]).agg(att=("att", "sum"), yds=("yds", "sum"), z_att=("z_att", "sum"), z_yds=("z_yds", "sum"),
                                                g_att=("g_att", "sum"), g_yds=("g_yds", "sum")).reset_index()
    t = P.assign(succ_n=P.succ * P.att, stuff_n=P.stuff * P.att, ybc_n=P.ybc * P.att, ryoe_n=P.ryoe_att * P.att)
    s = t.groupby(["opp", "season", "week"])[["succ_n", "stuff_n", "ybc_n", "ryoe_n"]].sum().reset_index()
    g = g.merge(s, on=["opp", "season", "week"])
    g["key"] = g.season * 100 + g.week
    return g.sort_values(["opp", "key"])


def rb_defense_features(box: pd.DataFrame, A: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    cov = {d: g.sort_values(["season", "week"]) for d, g in box.groupby("def")}
    ckeys = {d: g.season.to_numpy() * 100 + g.week.to_numpy() for d, g in cov.items()}
    bw = box.assign(key=box.season * 100 + box.week).groupby("key")[BOX_COLS].mean()
    al = {d: (g.key.to_numpy(), g[_RB_ALLOWED].to_numpy(float)) for d, g in A.groupby("opp")}
    lg_cache, bx_cache = {}, {}
    out = []
    for r in rows.itertuples():
        k = key(r.season, r.week)
        f = {}
        d = r.opp
        if d in cov:
            lo, hi = window(ckeys[d], k, DWIN)
            g = cov[d].iloc[lo:hi]
            if len(g):
                if k not in bx_cache:
                    bx_cache[k] = bw[bw.index < k].tail(6).mean()
                for c in BOX_COLS:
                    v = ew(g[c].to_numpy())
                    f[f"d_{c}"] = v
                    f[f"d_{c}_dev"] = v - float(bx_cache[k][c])
        f["d_n_prior"] = 0
        hit = al.get(d)
        if hit is not None:
            keys, arr = hit
            lo, hi = window(keys, k, DWIN)
            a = arr[lo:hi]
            if len(a):
                if k not in lg_cache:
                    kk = np.sort(A.key.unique())
                    lg_cache[k] = A[A.key.isin(kk[kk < k][-6:])][_RB_ALLOWED].sum()
                lg = lg_cache[k]
                if lg["att"] > 0:
                    f["d_n_prior"] = len(a)
                    S = (a * decay(len(a))[:, None]).sum(axis=0)
                    att = S[0]

                    def shr(x, n, L):
                        return (n * x + K_ALLOWED * L) / (n + K_ALLOWED) if np.isfinite(x) else L
                    L = {"ypc": lg["yds"] / lg["att"], "z": lg["z_yds"] / max(lg["z_att"], 1), "g": lg["g_yds"] / max(lg["g_att"], 1),
                         "succ": lg["succ_n"] / lg["att"], "stuff": lg["stuff_n"] / lg["att"], "ybc": lg["ybc_n"] / lg["att"], "ryoe": lg["ryoe_n"] / lg["att"]}
                    f["d_ypc_rel"] = shr(S[1] / att if att > 0 else np.nan, att, L["ypc"]) / L["ypc"] - 1
                    f["d_zypc_rel"] = shr(S[3] / S[2] if S[2] > 0 else np.nan, S[2], L["z"]) / L["z"] - 1
                    f["d_gypc_rel"] = shr(S[5] / S[4] if S[4] > 0 else np.nan, S[4], L["g"]) / L["g"] - 1
                    f["d_succ_dev"] = shr(S[6] / att if att > 0 else np.nan, att, L["succ"]) - L["succ"]
                    f["d_stuff_dev"] = shr(S[7] / att if att > 0 else np.nan, att, L["stuff"]) - L["stuff"]
                    f["d_ybc_dev"] = shr(S[8] / att if att > 0 else np.nan, att, L["ybc"]) - L["ybc"]
                    f["d_ryoe_dev"] = shr(S[9] / att if att > 0 else np.nan, att, L["ryoe"]) - L["ryoe"]
        out.append(f)
    return pd.DataFrame(out, index=rows.index)


def rb_mismatch_features(pf: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Where the back and the run defense meet (columns prefixed `mm_`; positive = favourable for the back)."""
    pf = _with(pf, ["p_z_share", "p_ypc", "p_hit", "p_ybc", "p_att_pct"])
    df = _with(df, ["d_zypc_rel", "d_gypc_rel", "d_ypc_rel", "d_box_dev", "d_succ_dev", "d_stuff_dev", "d_pass_faced_dev"])
    m = pd.DataFrame(index=pf.index)
    zs = pf.p_z_share.clip(0, 1).fillna(0.5)
    m["mm_concept_ypc"] = zs * df.d_zypc_rel + (1 - zs) * df.d_gypc_rel
    m["mm_ypc_stack"] = df.d_ypc_rel * pf.p_ypc.fillna(4.2)
    m["mm_box_hit"] = df.d_box_dev * pf.p_hit.fillna(0)
    m["mm_box_ybc"] = df.d_box_dev * pf.p_ybc.fillna(0)
    m["mm_succ_stuff"] = df.d_succ_dev - df.d_stuff_dev
    m["mm_run_plus"] = df.d_ypc_rel + df.d_succ_dev - df.d_stuff_dev
    m["mm_script_pass"] = df.d_pass_faced_dev * pf.p_att_pct.fillna(0)
    return m


def build_rushing(rows: pd.DataFrame, P: pd.DataFrame, box: pd.DataFrame) -> pd.DataFrame:
    pf = rb_player_features(P, rows)
    df = rb_defense_features(box, rb_allowed_table(P), rows)
    return pd.concat([rows, pf, df, rb_mismatch_features(pf, df)], axis=1)
