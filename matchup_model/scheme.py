"""Scheme-split data for the Matchup Machine screen.

Loads the FantasyPoints coverage-scheme / run-concept / alignment exports and exposes
tidy per-player and per-team frames plus the opponent-defense side, so the screen can lay
an offense's scheme profile next to what a defense runs and allows — the layout of the
user's "Matchup machine" spreadsheet.

Everything here is 2025-only for now (the files are `*_2025.csv`). Descriptive: the
coverage-matchup edge did not backtest as *predictive* (see backtest_report.md); this is
an eyeball tool.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from dfs.names import norm_team, normalize_name
from matchup_model.config import DATA

# Core buckets shown in the grids (the rest — Bracket / Prevent / Goal Line / Misc — are
# noise for matchup purposes and dropped).
COVERAGES = ["Cover 0", "Cover 1", "Cover 2", "Cover 2 Man", "Cover 3", "Cover 4", "Cover 6"]
CONCEPTS = ["Outside Zone", "Inside Zone", "Man/Duo", "Power", "Counter", "Pull Lead", "Draw"]

NFL_TEAMS = ["ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET",
             "GB", "HOU", "IND", "JAX", "KC", "LAC", "LAR", "LV", "MIA", "MIN", "NE", "NO",
             "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS"]

# man/zone/single-high/two-high rolled up from the Cover 0-6 rows (shown above the specifics).
BUCKETS = {
    "Man": ["Cover 0", "Cover 1", "Cover 2 Man"],
    "Zone": ["Cover 2", "Cover 3", "Cover 4", "Cover 6"],
    "Single-high": ["Cover 0", "Cover 1", "Cover 3"],
    "Two-high": ["Cover 2", "Cover 2 Man", "Cover 4", "Cover 6"],
}
BUCKET_NAMES = list(BUCKETS)

# Season blend: once 2026 scheme files land, ramp their weight in over the first ~8 weeks.
RAMP_WEEKS = 8


def _read(name: str) -> pd.DataFrame:
    p = DATA / name
    if not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(p, header=1, skiprows=[2])
    except Exception:
        return pd.DataFrame()
    df = df[df["Name"].notna() & ~df["Name"].astype(str).str.contains("Exported from", na=False)]
    for c in df.columns:
        if c not in ("Season", "Name", "Team", "POS", "CONCEPT", "COV", "ALIGN"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.reset_index(drop=True)


def _read_years(stem_fmt: str) -> pd.DataFrame:
    """Concat the 2025 (+2026 if present) versions of a file, tagging a `yr` column.
    `stem_fmt` has one `{}` for the year, e.g. 'receiving-coverage_wr_{}.csv'."""
    frames = []
    for yr in (2025, 2026):
        d = _read(stem_fmt.format(yr))
        if not d.empty:
            d = d.copy()
            d["yr"] = yr
            frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


@lru_cache(maxsize=1)
def blend_weight() -> float:
    """Weight on 2026 vs 2025 for the scheme tables. 0.0 until 2026 files exist, then
    ramps to 1.0 by RAMP_WEEKS. Uses the max games count found in any 2026 scheme file."""
    g = 0
    for stem in ("receiving-coverage_wr_{}.csv", "rushing-concept_player_{}.csv",
                 "receiving-coverage_defense_{}.csv"):
        d = _read(stem.format(2026))
        if not d.empty and "G" in d.columns:
            g = max(g, int(pd.to_numeric(d["G"], errors="coerce").max() or 0))
    return round(min(1.0, g / RAMP_WEEKS), 3)


def _year_blend(df: pd.DataFrame, keys: list[str], count_cols: list[str]) -> pd.DataFrame:
    """Collapse a two-season frame (col `yr`) to one row per `keys`, blending 2026 and
    2025 count columns by blend_weight() (rescaled so totals stay ~1 season's volume)."""
    if df.empty or "yr" not in df.columns or df["yr"].nunique() == 1:
        return df.drop(columns=["yr"], errors="ignore")
    w = blend_weight()
    parts = []
    for yr, wt in ((2026, w), (2025, 1.0 - w)):
        sub = df[df["yr"] == yr].copy()
        if sub.empty:
            continue
        for c in count_cols:
            if c in sub.columns:
                sub[c] = pd.to_numeric(sub[c], errors="coerce") * (2 * wt)  # 2x: two years -> one
        parts.append(sub)
    both = pd.concat(parts, ignore_index=True)
    agg = {c: "sum" for c in count_cols if c in both.columns}
    for c in both.columns:
        if c not in agg and c not in keys and c != "yr":
            agg[c] = "first"
    return both.groupby(keys, as_index=False).agg(agg)


# ── receiving by coverage ──────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _rec_cov_players() -> pd.DataFrame:
    frames = []
    for p in ("wr", "te"):
        d = _read_years(f"receiving-coverage_{p}_{{}}.csv")
        if not d.empty:
            frames.append(d)
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True)
    d["name_key"] = d["Name"].map(normalize_name)
    d["team"] = d["Team"].map(norm_team)
    return d


@lru_cache(maxsize=1)
def _rec_cov_defense() -> pd.DataFrame:
    d = _read_years("receiving-coverage_defense_{}.csv")
    if not d.empty:
        d["team"] = d["Name"].map(norm_team)
    return d


# raw count columns we roll up; rates get recomputed from these so buckets are consistent
_REC_COUNTS = ["RTE", "TGT", "REC", "YDS", "TD", "1READ"]
_PLAYER_COLS = ["look", "routes", "targets", "tgt/rt", "yds/rt", "yds/tgt", "catch%",
                "1st-read%", "TD"]
_DEF_COLS = ["look", "targets", "yds/tgt", "catch%", "yds/rec", "rating", "TD"]
_DEF_RANK_METRICS = ["yds/tgt", "catch%", "yds/rec", "rating", "TD"]


def _rec_rows_for(g: pd.DataFrame, key: str) -> pd.DataFrame:
    """From one entity's Cover 0-6 rows, build man/zone/single-high/two-high bucket rows
    then the specific Cover rows, all with recomputed rates. `key` = 'player' | 'defense'."""
    g = g[g["COV"].isin(COVERAGES)].copy()
    if g.empty:
        return pd.DataFrame()
    by_cov = {r["COV"]: r for _, r in g.iterrows()}

    def _agg(covs, label):
        present = [c for c in covs if c in by_cov]
        if not present:
            return None
        s = {c: float(np.nansum([by_cov[k].get(c, np.nan) for k in present])) for c in _REC_COUNTS}
        rte, tgt, rec, yds = s["RTE"], s["TGT"], s["REC"], s["YDS"]
        rating = np.nan
        if "RATE" in g.columns and tgt:
            rating = float(np.nansum([by_cov[k]["RATE"] * by_cov[k]["TGT"] for k in present]) / tgt)
        return {
            "look": label,
            "routes": round(rte),
            "targets": round(tgt),
            "tgt/rt": round(tgt / rte, 3) if rte else np.nan,
            "yds/rt": round(yds / rte, 2) if rte else np.nan,
            "yds/tgt": round(yds / tgt, 2) if tgt else np.nan,
            "catch%": round(100 * rec / tgt, 1) if tgt else np.nan,
            "yds/rec": round(yds / rec, 2) if rec else np.nan,
            "rating": round(rating, 1) if np.isfinite(rating) else np.nan,
            "TD": round(s["TD"]),
            "1st-read%": round(100 * s["1READ"] / rte, 1) if (rte and key == "player") else np.nan,
        }

    rows = [r for r in (_agg(c, n) for n, c in BUCKETS.items()) if r]
    rows += [r for r in (_agg([c], c) for c in COVERAGES) if r]
    cols = _PLAYER_COLS if key == "player" else _DEF_COLS
    return pd.DataFrame(rows)[cols]


def player_pass_by_coverage(name_key: str) -> pd.DataFrame:
    """A pass-catcher's efficiency by man/zone/1-high/2-high, then Cover 0-6:
    routes, targets, tgt/route, yds/route, yds/tgt, catch%, 1st-read%, TD."""
    d = _rec_cov_players()
    if d.empty:
        return pd.DataFrame()
    g = _year_blend(d[d["name_key"] == name_key], keys=["COV"], count_cols=_REC_COUNTS)
    return _rec_rows_for(g, "player").round(2).reset_index(drop=True)


def _rank_ascending(vals: dict) -> dict:
    """{key: rank}, 1 = lowest value, N = highest. For 'allowed efficiency' -> 1 = stingiest,
    N = softest."""
    return {k: i for i, k in enumerate(sorted(vals, key=lambda k: vals[k]), 1)}


def _rank_descending(vals: dict) -> dict:
    """{key: rank}, 1 = highest value. For 'usage / frequency' -> 1 = does it most."""
    return {k: i for i, k in enumerate(sorted(vals, key=lambda k: vals[k], reverse=True), 1)}


@lru_cache(maxsize=1)
def _defense_cov_ranks() -> dict:
    """{team: {look: {metric: rank}}}. rank runs 1..N with **N = allows the most =
    softest** for each metric+look, over all defenses on the same bucket+coverage rollup.
    Includes 'plays%' (N = plays that coverage most)."""
    d = _rec_cov_defense()
    if d.empty:
        return {}
    tables = {}
    for t, g in d.groupby("team"):
        gg = _year_blend(g, keys=["COV"], count_cols=_REC_COUNTS)
        tbl = _rec_rows_for(gg, "defense")
        if not tbl.empty:
            tables[t] = tbl.set_index("look")
    out: dict = {t: {} for t in tables}
    if not tables:
        return out
    looks = next(iter(tables.values())).index.tolist()
    plays = {t: {lk: v for lk, v in defense_coverage_rates(t).set_index("coverage")["plays%"].items()}
             for t in tables}
    for lk in looks:
        for m in _DEF_RANK_METRICS:
            vals = {t: tbl.loc[lk, m] for t, tbl in tables.items()
                    if lk in tbl.index and pd.notna(tbl.loc[lk, m])}
            for t, r in _rank_ascending(vals).items():
                out[t].setdefault(lk, {})[m] = r
        pvals = {t: plays[t].get(lk) for t in tables if pd.notna(plays[t].get(lk))}
        for t, r in _rank_descending(pvals).items():          # 1 = runs it most
            out[t].setdefault(lk, {})["plays%"] = r
    return out


def defense_pass_allowed_by_coverage(team: str) -> pd.DataFrame:
    """What a defense allows by man/zone/1-high/2-high then Cover 0-6: plays%, targets,
    yds/tgt, catch%, yds/rec, passer rating, TD — each with its league rank.
    `plays% rk`: **1 = runs that coverage most**. Allowed-efficiency ranks: **32 = softest**."""
    d = _rec_cov_defense()
    if d.empty:
        return pd.DataFrame()
    t = norm_team(team)
    g = _year_blend(d[d["team"] == t], keys=["COV"], count_cols=_REC_COUNTS)
    base = _rec_rows_for(g, "defense")
    if base.empty:
        return pd.DataFrame()
    rates = defense_coverage_rates(team).rename(columns={"coverage": "look"})
    out = base.merge(rates[["look", "plays%"]], on="look", how="left")
    rk = _defense_cov_ranks().get(t, {})
    for m in ["plays%"] + _DEF_RANK_METRICS:
        out[f"{m} rk"] = out["look"].map(lambda lk, _m=m: rk.get(lk, {}).get(_m))
    order = ["look", "plays%", "plays% rk", "targets", "yds/tgt", "yds/tgt rk",
             "catch%", "catch% rk", "yds/rec", "yds/rec rk", "rating", "rating rk",
             "TD", "TD rk"]
    return out[[c for c in order if c in out.columns]].round(2).reset_index(drop=True)


# ── defense coverage rates (from the weekly coverage matrix) ────────────────
@lru_cache(maxsize=1)
def _cov_matrix() -> pd.DataFrame:
    d = _read("coverage-matrix_2025_week.csv")
    if d.empty:
        return d
    d["team"] = d["Name"].map(norm_team)
    return d


_COV_RATE_COL = {"Cover 0": "COVER 0 %", "Cover 1": "COVER 1 %", "Cover 2": "COVER 2 %",
                 "Cover 2 Man": "COVER 2 MAN %", "Cover 3": "COVER 3 %",
                 "Cover 4": "COVER 4 %", "Cover 6": "COVER 6 %"}


def defense_coverage_rates(team: str) -> pd.DataFrame:
    """look -> plays% (season mean) + lean vs league average, for man/zone/1-high/2-high
    (summed from members) and Cover 0-6."""
    cm = _cov_matrix()
    order = BUCKET_NAMES + COVERAGES
    if cm.empty:
        return pd.DataFrame({"coverage": order, "plays%": np.nan, "vs lg": np.nan})
    t = norm_team(team)
    sub = cm[cm["team"] == t]

    def _rate(covs):
        cols = [_COV_RATE_COL[c] for c in covs if _COV_RATE_COL.get(c) in cm.columns]
        if not cols:
            return np.nan, np.nan
        tr = float(sub[cols].sum(axis=1).mean()) if not sub.empty else np.nan
        lg = float(cm[cols].sum(axis=1).mean())
        return tr, lg

    rows = []
    for look in order:
        covs = BUCKETS.get(look, [look])
        tr, lg = _rate(covs)
        rows.append({"coverage": look, "plays%": round(tr, 1) if np.isfinite(tr) else np.nan,
                     "vs lg": round(tr - lg, 1) if np.isfinite(tr) else np.nan})
    return pd.DataFrame(rows)


# ── rushing by concept ────────────────────────────────────────────────────
@lru_cache(maxsize=3)
def _rush_concept(kind: str) -> pd.DataFrame:
    d = _read(f"rushing-concept_{kind}_2025.csv")
    if d.empty:
        return d
    if kind == "player":
        d["name_key"] = d["Name"].map(normalize_name)
        d["team"] = d["Team"].map(norm_team)
    else:
        d["team"] = d["Name"].map(norm_team)
    return d


def player_run_by_concept(name_key: str) -> pd.DataFrame:
    d = _rush_concept("player")
    if d.empty:
        return pd.DataFrame()
    g = d[(d["name_key"] == name_key) & (d["CONCEPT"].isin(CONCEPTS))]
    if g.empty:
        return pd.DataFrame()
    out = (g.assign(concept=lambda x: pd.Categorical(x["CONCEPT"], CONCEPTS, ordered=True))
           .sort_values("concept")[["concept", "ATT %", "ATT", "YPC", "YDS", "TD",
                                    "SUCC %", "EXP RUN %", "RYOE"]]
           .rename(columns={"ATT %": "att%", "ATT": "att", "YDS": "yards",
                            "SUCC %": "success%", "EXP RUN %": "exp-run%"}))
    return out.round(2).reset_index(drop=True)


def team_run_by_concept(team: str, side: str) -> pd.DataFrame:
    """side='offense' (team's own run mix) or 'defense' (what the D allows)."""
    d = _rush_concept(side)
    if d.empty:
        return pd.DataFrame()
    t = norm_team(team)
    g = d[(d["team"] == t) & (d["CONCEPT"].isin(CONCEPTS))]
    if g.empty:
        return pd.DataFrame()
    out = (g.assign(concept=lambda x: pd.Categorical(x["CONCEPT"], CONCEPTS, ordered=True))
           .sort_values("concept")[["concept", "ATT %", "ATT", "YPC", "YDS", "TD",
                                    "SUCC %", "EXP RUN %"]]
           .rename(columns={"ATT %": "att%", "ATT": "att", "YDS": "yards",
                            "SUCC %": "success%", "EXP RUN %": "exp-run%"}))
    if side == "defense":
        rk = _defense_concept_ranks().get(t, {})
        for m in ("att%", "YPC", "success%", "exp-run%"):
            out[f"{m} rk"] = out["concept"].astype(str).map(
                lambda c, _m=m: rk.get(c, {}).get(_m)).astype("Int64")
    return out.round(2).reset_index(drop=True)


@lru_cache(maxsize=1)
def _defense_concept_ranks() -> dict:
    """{team: {concept: {metric: rank}}}, 1..N with N = the league extreme:
    att% -> faces that concept most; YPC / success% / exp-run% -> allows the most."""
    d = _rush_concept("defense")
    if d.empty:
        return {}
    d = d[d["CONCEPT"].isin(CONCEPTS)]
    ren = {"ATT %": "att%", "SUCC %": "success%", "EXP RUN %": "exp-run%"}
    d = d.rename(columns=ren)
    out: dict = {t: {} for t in d["team"].unique()}
    for c in CONCEPTS:
        sub = d[d["CONCEPT"] == c]
        for m in ("att%", "YPC", "success%", "exp-run%"):
            vals = {r["team"]: r[m] for _, r in sub.iterrows() if pd.notna(r[m])}
            ranker = _rank_descending if m == "att%" else _rank_ascending  # att% -> 1 = faces most
            for t, rank in ranker(vals).items():
                out[t].setdefault(c, {})[m] = rank
    return out


# ── personnel groupings (11 / 12 / 21 ...) ────────────────────────────────
PERSONNEL = ["11", "12", "21", "10", "13", "22"]


def _pers_label(x) -> str:
    try:
        return str(int(float(x)))
    except (TypeError, ValueError):
        return str(x)


@lru_cache(maxsize=4)
def _pers_frame(kind: str) -> pd.DataFrame:
    """kind: 'rec_player' | 'rec_defense' | 'rush_player' | 'rush_defense'."""
    unit, mode = kind.split("_")
    if unit == "rec" and mode == "player":
        frames = [_read(f"receiving-personnel_{p}_2025.csv") for p in ("wr", "te")]
        d = pd.concat([f for f in frames if not f.empty], ignore_index=True) if any(
            not f.empty for f in frames) else pd.DataFrame()
    else:
        d = _read(f"{'receiving' if unit == 'rec' else 'rushing'}-personnel_{mode}_2025.csv")
    if d.empty:
        return d
    d["pers"] = d["PERS"].map(_pers_label)
    if mode == "player":
        d["name_key"] = d["Name"].map(normalize_name)
    d["team"] = d["Name"].map(norm_team) if mode == "defense" else d.get("Team", d["Name"]).map(norm_team)
    return d


def player_pass_by_personnel(name_key: str) -> pd.DataFrame:
    d = _pers_frame("rec_player")
    if d.empty:
        return pd.DataFrame()
    g = d[(d["name_key"] == name_key) & (d["pers"].isin(PERSONNEL))]
    if g.empty:
        return pd.DataFrame()
    out = (g.assign(p=lambda x: pd.Categorical(x["pers"], PERSONNEL, ordered=True))
           .sort_values("p")[["pers", "RTE", "TPRR", "YPRR", "CR %", "TD"]]
           .rename(columns={"pers": "personnel", "RTE": "routes", "TPRR": "tgt/rt",
                            "YPRR": "yds/rt", "CR %": "catch%"}))
    return out.round(2).reset_index(drop=True)


def player_run_by_personnel(name_key: str) -> pd.DataFrame:
    d = _pers_frame("rush_player")
    if d.empty:
        return pd.DataFrame()
    g = d[(d["name_key"] == name_key) & (d["pers"].isin(PERSONNEL))]
    if g.empty:
        return pd.DataFrame()
    out = (g.assign(p=lambda x: pd.Categorical(x["pers"], PERSONNEL, ordered=True))
           .sort_values("p")[["pers", "ATT %", "ATT", "YPC", "TD", "SUCC %", "EXP RUN %"]]
           .rename(columns={"pers": "personnel", "ATT %": "att%", "ATT": "att",
                            "SUCC %": "success%", "EXP RUN %": "exp-run%"}))
    return out.round(2).reset_index(drop=True)


@lru_cache(maxsize=2)
def _defense_personnel_ranks(unit: str) -> dict:
    """unit: 'rec' or 'rush'. {team: {pers: {metric: rank}}}, N = league extreme
    (softest allowed / faces it most)."""
    d = _pers_frame(f"{unit}_defense")
    if d.empty:
        return {}
    d = d[d["pers"].isin(PERSONNEL)]
    freq, metrics = (("TGT %", ("TGT %", "YPT", "CR %", "RATE")) if unit == "rec"
                     else ("ATT %", ("ATT %", "YPC", "SUCC %", "EXP RUN %")))
    out: dict = {t: {} for t in d["team"].unique()}
    for p in PERSONNEL:
        sub = d[d["pers"] == p]
        for m in metrics:
            if m not in sub.columns:
                continue
            vals = {r["team"]: r[m] for _, r in sub.iterrows() if pd.notna(r[m])}
            ranker = _rank_descending if m == freq else _rank_ascending  # freq -> 1 = sees it most
            for t, rank in ranker(vals).items():
                out[t].setdefault(p, {})[m] = rank
    return out


# ── defense allowed by alignment (the "Defense Vs WR heat map") ─────────────
@lru_cache(maxsize=1)
def _align_defense() -> pd.DataFrame:
    d = _read("receiving-alignment_defense_2025.csv")
    if not d.empty:
        d["team"] = d["Name"].map(norm_team)
    return d


def defense_alignment_grid() -> pd.DataFrame:
    """All 32 defenses x alignment: yds/rt, tgt/rt, catch%, 1st-read% allowed. Wide/Slot/
    Inline/Backfield as columns of yds/rt so it reads like the sheet's heat map."""
    d = _align_defense()
    if d.empty:
        return pd.DataFrame()
    piv = (d[d["ALIGN"].isin(["Wide", "Slot", "Inline", "Backfield"])]
           .pivot_table(index="team", columns="ALIGN", values="YPRR", aggfunc="mean"))
    piv = piv.reindex(columns=[c for c in ["Wide", "Slot", "Inline", "Backfield"] if c in piv.columns])
    piv.columns = [f"{c} yds/rt" for c in piv.columns]
    return piv.round(2).reset_index().rename(columns={"team": "defense"})


def _ord(n: int) -> str:
    return f"{n}{'th' if 11 <= n % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def matchup_highlights(opp: str, pass_names: list[str], rb_names: list[str],
                       n: int = 12) -> pd.DataFrame:
    """The offense's best scheme edges against `opp`: for the coverages / run concepts
    this defense leans on (league rank) or is weak against, which of the listed players
    are the most efficient. One row per (player, look). Columns: kind, look, player, why, mark."""
    from dfs.names import normalize_name as _nk
    o = norm_team(opp)
    rows = []
    N = 32

    def _softmean(r, keys):
        v = [r[k] for k in keys if pd.notna(r.get(k))]
        return sum(v) / len(v) if v else float("nan")

    # ── pass: coverage ────────────────────────────────────────────────────
    covrk = _defense_cov_ranks().get(o, {})
    cov_rate = defense_coverage_rates(opp).set_index("coverage")["plays%"].to_dict()
    for c in BUCKET_NAMES + COVERAGES:
        r = covrk.get(c, {})
        usage = r.get("plays%")                       # 1 = runs it most
        soft = _softmean(r, ("yds/tgt", "catch%", "rating"))
        if not ((usage and usage <= 9) or (np.isfinite(soft) and soft >= 24)):
            continue
        why_bits = []
        if usage and usage <= 9:
            why_bits.append(f"runs it {_ord(usage)}-most ({cov_rate.get(c, 0):.0f}%)")
        if np.isfinite(soft) and soft >= 24:
            why_bits.append(f"{_ord(N - int(round(soft)) + 1)}-worst defending it")
        why = f"{o}: " + " · ".join(why_bits)
        cands = []
        for nm in pass_names:
            t = player_pass_by_coverage(_nk(nm))
            if t.empty or c not in set(t["look"]):
                continue
            pr = t.set_index("look").loc[c]
            eff, base = pr["yds/rt"], t["yds/rt"].mean()
            if not (pd.notna(eff) and eff > 0 and eff >= max(base * 1.05, 1.8) and pr["routes"] >= 20):
                continue
            cands.append((eff, nm, pr))
        freq_score = (N - usage + 1) / N if usage else 0
        for eff, nm, pr in sorted(cands, reverse=True, key=lambda x: x[0])[:2]:  # team's best 2
            rows.append({"kind": "pass", "look": c, "player": nm, "why": why,
                         "mark": f"{eff:.2f} yds/rt · {pr['catch%']:.0f}% catch ({int(pr['routes'])} rt)",
                         "_score": freq_score + min(eff / 4.0, 1.5)})

    # ── run: concept ─────────────────────────────────────────────────────
    conrk = _defense_concept_ranks().get(o, {})
    for c in CONCEPTS:
        r = conrk.get(c, {})
        faces = r.get("att%")                         # 1 = faces it most
        soft = _softmean(r, ("YPC", "success%", "exp-run%"))
        if not ((faces and faces <= 9) or (np.isfinite(soft) and soft >= 24)):
            continue
        why_bits = []
        if faces and faces <= 9:
            why_bits.append(f"faces it {_ord(faces)}-most")
        if np.isfinite(soft) and soft >= 24:
            why_bits.append(f"{_ord(N - int(round(soft)) + 1)}-worst vs it")
        why = f"{o}: " + " · ".join(why_bits)
        cands = []
        for nm in rb_names:
            t = player_run_by_concept(_nk(nm))
            if t.empty or c not in set(t["concept"]):
                continue
            pr = t.set_index("concept").loc[c]
            ypc, base = pr["YPC"], t["YPC"].mean()
            if not (pd.notna(ypc) and ypc > 0 and ypc >= max(base * 1.05, 4.2) and pr["att"] >= 12):
                continue
            cands.append((ypc, nm, pr))
        freq_score = (N - faces + 1) / N if faces else 0
        for ypc, nm, pr in sorted(cands, reverse=True, key=lambda x: x[0])[:2]:
            rows.append({"kind": "run", "look": c, "player": nm, "why": why,
                         "mark": f"{ypc:.2f} ypc · {pr['success%']:.0f}% success ({int(pr['att'])} att)",
                         "_score": freq_score + min(ypc / 5.0, 1.5)})

    # ── personnel (11 / 12 / 21) — pass ──────────────────────────────────
    prk_rec = _defense_personnel_ranks("rec").get(o, {})
    for p in PERSONNEL:
        r = prk_rec.get(p, {})
        used = r.get("TGT %")                          # 1 = faces this personnel most
        soft = _softmean(r, ("YPT", "CR %", "RATE"))
        if not ((used and used <= 9) or (np.isfinite(soft) and soft >= 24)):
            continue
        why_bits = []
        if used and used <= 9:
            why_bits.append(f"sees {p} pers {_ord(used)}-most")
        if np.isfinite(soft) and soft >= 24:
            why_bits.append(f"{_ord(N - int(round(soft)) + 1)}-worst vs it")
        why = f"{o}: " + " · ".join(why_bits)
        cands = []
        for nm in pass_names:
            t = player_pass_by_personnel(_nk(nm))
            if t.empty or p not in set(t["personnel"]):
                continue
            pr = t.set_index("personnel").loc[p]
            eff, base = pr["yds/rt"], t["yds/rt"].mean()
            if not (pd.notna(eff) and eff >= max(base * 1.05, 1.8) and pr["routes"] >= 20):
                continue
            cands.append((eff, nm, pr))
        freq_score = (N - used + 1) / N if used else 0
        for eff, nm, pr in sorted(cands, reverse=True, key=lambda x: x[0])[:2]:
            rows.append({"kind": "pass", "look": f"{p} personnel", "player": nm, "why": why,
                         "mark": f"{eff:.2f} yds/rt · {pr['catch%']:.0f}% catch ({int(pr['routes'])} rt)",
                         "_score": freq_score + min(eff / 4.0, 1.5)})

    # ── personnel — run ─────────────────────────────────────────────────
    prk_rush = _defense_personnel_ranks("rush").get(o, {})
    for p in PERSONNEL:
        r = prk_rush.get(p, {})
        faces = r.get("ATT %")
        soft = _softmean(r, ("YPC", "SUCC %", "EXP RUN %"))
        if not ((faces and faces <= 9) or (np.isfinite(soft) and soft >= 24)):
            continue
        why_bits = []
        if faces and faces <= 9:
            why_bits.append(f"sees {p} pers runs {_ord(faces)}-most")
        if np.isfinite(soft) and soft >= 24:
            why_bits.append(f"{_ord(N - int(round(soft)) + 1)}-worst vs it")
        why = f"{o}: " + " · ".join(why_bits)
        cands = []
        for nm in rb_names:
            t = player_run_by_personnel(_nk(nm))
            if t.empty or p not in set(t["personnel"]):
                continue
            pr = t.set_index("personnel").loc[p]
            ypc, base = pr["YPC"], t["YPC"].mean()
            if not (pd.notna(ypc) and ypc >= max(base * 1.05, 4.2) and pr["att"] >= 12):
                continue
            cands.append((ypc, nm, pr))
        freq_score = (N - faces + 1) / N if faces else 0
        for ypc, nm, pr in sorted(cands, reverse=True, key=lambda x: x[0])[:2]:
            rows.append({"kind": "run", "look": f"{p} personnel", "player": nm, "why": why,
                         "mark": f"{ypc:.2f} ypc · {pr['success%']:.0f}% success ({int(pr['att'])} att)",
                         "_score": freq_score + min(ypc / 5.0, 1.5)})

    if not rows:
        return pd.DataFrame(columns=["kind", "look", "player", "why", "mark"])
    df = pd.DataFrame(rows).sort_values("_score", ascending=False).head(n)
    return df[["kind", "look", "player", "why", "mark"]].reset_index(drop=True)


def _matches_team(raw_team, team: str) -> bool:
    """A scheme row's `Team` may be 'ATL, NYG' for a mid-season trade — match if `team`
    (normalized) appears anywhere in it."""
    t = norm_team(team)
    return any(norm_team(p.strip()) == t for p in str(raw_team).split(","))


def team_pass_catchers(team: str, n: int = 6) -> list[str]:
    """Top-n WR/TE names for a team by total 2025 routes (fallback when the team isn't on
    the DK slate, so there are no projections to rank by)."""
    d = _rec_cov_players()
    if d.empty:
        return []
    m = d[d["Team"].map(lambda x: _matches_team(x, team)) & d["COV"].isin(COVERAGES)]
    if m.empty:
        return []
    tot = m.groupby("Name")["RTE"].sum().sort_values(ascending=False)
    return list(tot.head(n).index)


def team_backs(team: str, n: int = 3) -> list[str]:
    """Top-n RB names for a team by total 2025 carries."""
    d = _rush_concept("player")
    if d.empty:
        return []
    m = d[d["Team"].map(lambda x: _matches_team(x, team)) & d["CONCEPT"].isin(CONCEPTS)]
    if m.empty:
        return []
    tot = m.groupby("Name")["ATT"].sum().sort_values(ascending=False)
    return list(tot.head(n).index)


def available() -> bool:
    return not _rec_cov_players().empty and not _rush_concept("player").empty
