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


@lru_cache(maxsize=1)
def _defense_cov_ranks() -> dict:
    """{team: {look: {metric: rank}}} — rank 1 = allows the most (softest) for that
    metric+look, over all 32 defenses on the same bucket+coverage rollup."""
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
    for m in _DEF_RANK_METRICS:
        for lk in looks:
            vals = {t: tbl.loc[lk, m] for t, tbl in tables.items()
                    if lk in tbl.index and pd.notna(tbl.loc[lk, m])}
            for i, t in enumerate(sorted(vals, key=vals.get, reverse=True), 1):
                out[t].setdefault(lk, {})[m] = i
    return out


def defense_pass_allowed_by_coverage(team: str) -> pd.DataFrame:
    """What a defense allows by man/zone/1-high/2-high then Cover 0-6: plays%, targets,
    yds/tgt, catch%, yds/rec, passer rating, TD — each with its league rank (1 = softest)."""
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
    for m in _DEF_RANK_METRICS:
        out[f"{m} rk"] = out["look"].map(lambda lk, _m=m: rk.get(lk, {}).get(_m))
    order = ["look", "plays%", "targets", "yds/tgt", "yds/tgt rk", "catch%", "catch% rk",
             "yds/rec", "yds/rec rk", "rating", "rating rk", "TD", "TD rk"]
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
    return out.round(2).reset_index(drop=True)


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


def available() -> bool:
    return not _rec_cov_players().empty and not _rush_concept("player").empty
