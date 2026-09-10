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


# ── receiving by coverage ──────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _rec_cov_players() -> pd.DataFrame:
    frames = [_read(f"receiving-coverage_{p}_2025.csv") for p in ("wr", "te")]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True)
    d["name_key"] = d["Name"].map(normalize_name)
    d["team"] = d["Team"].map(norm_team)
    return d


@lru_cache(maxsize=1)
def _rec_cov_defense() -> pd.DataFrame:
    d = _read("receiving-coverage_defense_2025.csv")
    if not d.empty:
        d["team"] = d["Name"].map(norm_team)
    return d


def player_pass_by_coverage(name_key: str) -> pd.DataFrame:
    """A pass-catcher's efficiency by coverage: routes%, tgt/rt, yds/rt, catch%, 1st-read%, TD."""
    d = _rec_cov_players()
    if d.empty:
        return pd.DataFrame()
    g = d[(d["name_key"] == name_key) & (d["COV"].isin(COVERAGES))]
    if g.empty:
        return pd.DataFrame()
    out = (g.assign(coverage=lambda x: pd.Categorical(x["COV"], COVERAGES, ordered=True))
           .sort_values("coverage")[["coverage", "RTE", "RTE %", "TPRR", "YPRR", "YDS",
                                     "CR %", "1READ %", "TD"]]
           .rename(columns={"RTE": "routes", "RTE %": "route%", "TPRR": "tgt/rt",
                            "YPRR": "yds/rt", "YDS": "yards", "CR %": "catch%",
                            "1READ %": "1st-read%"}))
    return out.round(2).reset_index(drop=True)


def defense_pass_allowed_by_coverage(team: str) -> pd.DataFrame:
    """What a defense gives up by coverage: yds/rt, tgt/rt, catch%, yards, TD, plus how
    often it plays that coverage (rate% from the coverage matrix)."""
    d = _rec_cov_defense()
    rates = defense_coverage_rates(team)
    if d.empty:
        return pd.DataFrame()
    t = norm_team(team)
    g = d[(d["team"] == t) & (d["COV"].isin(COVERAGES))]
    if g.empty:
        return pd.DataFrame()
    out = (g.assign(coverage=lambda x: pd.Categorical(x["COV"], COVERAGES, ordered=True))
           .sort_values("coverage")[["coverage", "YPT", "CR %", "YDS", "TD", "1READ %"]]
           .rename(columns={"YPT": "yds/tgt allowed", "CR %": "catch% allowed",
                            "YDS": "yards", "1READ %": "1st-read% allowed"}))
    out = out.merge(rates[["coverage", "plays%"]], on="coverage", how="left")
    cols = ["coverage", "plays%"] + [c for c in out.columns if c not in ("coverage", "plays%")]
    return out[cols].round(2).reset_index(drop=True)


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
    """coverage -> plays% (season mean) + lean vs league average."""
    cm = _cov_matrix()
    if cm.empty:
        return pd.DataFrame({"coverage": COVERAGES, "plays%": np.nan, "vs lg": np.nan})
    t = norm_team(team)
    sub = cm[cm["team"] == t]
    rows = []
    for cov, col in _COV_RATE_COL.items():
        if col not in cm.columns:
            continue
        team_rate = sub[col].mean() if not sub.empty else np.nan
        lg = cm[col].mean()
        rows.append({"coverage": cov, "plays%": round(float(team_rate), 1),
                     "vs lg": round(float(team_rate - lg), 1)})
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
