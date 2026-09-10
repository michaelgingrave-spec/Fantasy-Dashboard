"""Load the FantasyPoints Data Suite CSVs in data/dfs/matchup/ into tidy frames.

Every export has: row 0 = column-group banner, row 1 = real header (stat names repeat
per split group), rows 2..-1 = data, last row = "Exported from Data Suite 2.0 ..." footer,
and a first data row "League Avg". This module normalises all of that away.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from matchup_model.config import DATA, SEASONS, WEEK_BATCHES
from dfs.names import normalize_name, norm_team

_SLUG = re.compile(r"[^a-z0-9]+")
# Only these banner groups produce `<group>__<stat>` columns. Every other group
# (Receiving, Rushing, Passing, FP, Man/Zone, Coverages, ...) yields a flat stat name.
_SPLIT_BANNERS = {"overall", "man", "zone", "single_high", "two_high"}


def _slug(s: str) -> str:
    return _SLUG.sub("_", str(s).strip().lower()).strip("_")


def _colnames(banner: list[str], header: list[str]) -> list[str]:
    """Build unique column names. Split-group columns (Overall/Man/Zone/Single-High/
    Two-High) get `<group>__<stat>`; all other columns get the flat stat name,
    deduped with a numeric suffix on collision."""
    out, seen = [], {}
    for grp, stat in zip(banner, header):
        g = _slug(grp)
        name = f"{g}__{_slug(stat)}" if g in _SPLIT_BANNERS else _slug(stat)
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        out.append(name)
    return out


def _read_fp_csv(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, header=None, dtype=str, keep_default_na=False,
                      engine="python", on_bad_lines="skip")
    if len(raw) and str(raw.iloc[-1, 0]).startswith("Exported from"):
        raw = raw.iloc[:-1]
    banner, header = list(raw.iloc[0]), list(raw.iloc[1])
    df = raw.iloc[2:].reset_index(drop=True)
    df.columns = _colnames(banner, header)
    # numeric coercion for everything except the obvious id columns
    id_cols = {"season", "rank", "name", "team", "pos", "week", "opp", "team_dc",
               "team_hc", "opp_oc", "opp_pc", "opp_dc", "g"}
    for c in df.columns:
        if c in id_cols:
            continue
        df[c] = pd.to_numeric(
            df[c].astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False),
            errors="coerce",
        )
    df = df[df["name"].astype(str).str.strip().str.lower() != "league avg"].copy()
    df = df[df["name"].astype(str).str.strip() != ""].copy()
    for c in ("season", "week", "g"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.reset_index(drop=True)


def _files(prefix: str) -> list[Path]:
    return sorted(DATA.glob(f"{prefix}*.csv"))


def _concat(prefix: str) -> pd.DataFrame:
    frames = []
    for p in _files(prefix):
        try:
            d = _read_fp_csv(p)
        except Exception:
            continue
        if "season" not in d.columns or d["season"].isna().all():
            m = re.search(r"(20\d{2})", p.name)
            d["season"] = int(m.group(1)) if m else np.nan
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ── Player efficiency splits (WR/TE and QB) ─────────────────────────────────
_SPLIT_MAP = {"overall": "overall", "man": "man", "zone": "zone",
              "single_high": "single_high", "two_high": "two_high"}


def _melt_splits(df: pd.DataFrame, stats: list[str]) -> pd.DataFrame:
    """Wide `<split>__<stat>` columns -> long rows keyed by split."""
    keep = ["season", "week", "name", "team", "pos", "opp", "g"]
    keep = [c for c in keep if c in df.columns]
    out = []
    for split in _SPLIT_MAP:
        cols = {f"{split}__{s}": s for s in stats if f"{split}__{s}" in df.columns}
        if not cols:
            continue
        sub = df[keep + list(cols)].rename(columns=cols).copy()
        sub["split"] = split
        out.append(sub)
    long = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    if not long.empty:
        long["name_key"] = long["name"].map(normalize_name)
        long["team"] = long["team"].map(norm_team)
        long["opp"] = long["opp"].map(norm_team)
    return long


@lru_cache(maxsize=1)
def receiving_splits() -> pd.DataFrame:
    """Long: season, week, name_key, team, pos, opp, split, rte, tprr, yprr, fp_rr."""
    df = _concat("receiving-manvszone_")
    if df.empty:
        return df
    return _melt_splits(df, ["rte", "tprr", "yprr", "fp_rr"])


@lru_cache(maxsize=1)
def passing_splits() -> pd.DataFrame:
    """Long: season, week, name_key, team, pos, opp, split, db, cmp_pct, ypa, td, int, rate, epa_db, fp_db."""
    df = _concat("passing-situation_")
    if df.empty:
        return df
    return _melt_splits(df, ["db", "cmp", "ypa", "td", "int", "rate", "epa_db", "fp_db"])


# ── Weekly actuals + opportunity ───────────────────────────────────────────
def _finish(a: pd.DataFrame) -> pd.DataFrame:
    a = a.copy()
    a["name_key"] = a["name"].map(normalize_name)
    if "team" in a.columns:
        a["team"] = a["team"].map(norm_team)
    if "opp" in a.columns:
        a["opp"] = a["opp"].map(norm_team)
    return a.dropna(subset=["season", "week"]).reset_index(drop=True)


@lru_cache(maxsize=1)
def receiving_weekly() -> pd.DataFrame:
    """Per player-week (WR/TE): routes, route_pct, tgt, rec, rec_yds, rec_td, adot,
    yprr, air_yds, wide/slot/inline/back route share, fp."""
    adv = _concat("receiving-advanced_")
    if adv.empty:
        return adv
    # idx8/9 are the Splits route%/tgt%; rename them out of the way before the count renames
    a = adv.rename(columns={"rte": "rte_share_splits", "tgt": "tgt_share_splits"})
    a = a.rename(columns={"rte_1": "routes", "rte_2": "route_pct", "tgt_1": "tgt",
                          "yds": "rec_yds", "td": "rec_td", "ay": "air_yds",
                          "wide_rte": "wide_pct", "slot_rte": "slot_pct",
                          "inline_rte": "inline_pct", "back_rte": "back_pct"})
    cols = ["season", "week", "name", "team", "pos", "opp", "routes", "route_pct", "tgt",
            "rec", "rec_yds", "rec_td", "adot", "yprr", "tprr", "air_yds", "ay_share",
            "eztgt", "wide_pct", "slot_pct", "inline_pct", "back_pct", "fp", "xfp", "diff"]
    return _finish(a[[c for c in cols if c in a.columns]])


@lru_cache(maxsize=1)
def rushing_weekly() -> pd.DataFrame:
    """Per player-week (RB): att, att_pct, rush_yds, rush_td, ypc, succ_pct, ryoe,
    ybc_att, yaco_att, stuff_pct, epa_a, fp (+ receiving tgt/rec/rec_yds when present)."""
    adv = _concat("rushing-advanced_")
    bas = _concat("rushing-basic_")
    if adv.empty:
        return adv
    a = adv.rename(columns={"att_1": "att", "att": "att_pct", "yds": "rush_yds",
                            "td": "rush_td", "succ": "succ_pct", "stuff": "stuff_pct"})
    cols = ["season", "week", "name", "team", "pos", "opp", "att", "att_pct", "rush_yds",
            "rush_td", "ypc", "succ_pct", "ryoe", "ybc_att", "yaco_att", "stuff_pct",
            "epa_a", "fp", "xfp", "diff"]
    a = a[[c for c in cols if c in a.columns]]
    if not bas.empty:
        b = bas.rename(columns={"tgt_1": "rush_tgt", "rec": "rush_rec", "rec_yds": "rush_rec_yds"})
        keep = [c for c in ("season", "week", "name", "team", "rush_tgt", "rush_rec", "rush_rec_yds") if c in b.columns]
        a = a.merge(b[keep], on=[c for c in ("season", "week", "name", "team") if c in keep], how="left")
    return _finish(a)


@lru_cache(maxsize=1)
def passing_weekly() -> pd.DataFrame:
    """Per QB-week: dropbacks, att, cmp, pass_yds, pass_td, int, ypa, sack_pct, adot,
    press_pct, epa_db, fp, pass_fp, rush_fp."""
    adv = _concat("passing-advanced_")
    if adv.empty:
        return adv
    a = adv.rename(columns={"db_1": "dropbacks", "yds": "pass_yds", "td": "pass_td",
                            "sack_1": "sack_pct", "press": "press_pct"})
    cols = ["season", "week", "name", "team", "pos", "opp", "dropbacks", "att", "cmp",
            "pass_yds", "pass_td", "int", "ypa", "sack_pct", "adot", "press_pct",
            "epa_db", "fp", "xfp", "diff", "pass_fp", "rush_fp"]
    return _finish(a[[c for c in cols if c in a.columns]])


# ── Defense scheme profile (team-week) ─────────────────────────────────────
# CSV flat name -> our name. `name` col is the DEFENSE team.
_SCHEME_REN = {
    "man": "man", "zone": "zone", "1_hi_mof_c": "one_high", "2_hi_mof_o": "two_high",
    "disguise": "disguise", "to_2_hi": "to_two_high",
    "cover_0": "cover0", "cover_1": "cover1", "cover_2": "cover2", "cover_2_man": "cover2man",
    "cover_3": "cover3", "cover_4": "cover4", "cover_6": "cover6",
    "press_any": "press", "nickel": "nickel", "dime": "dime",
}
SCHEME_RATE_COLS = ["man", "zone", "one_high", "two_high", "disguise", "cover0", "cover1",
                    "cover2", "cover2man", "cover3", "cover4", "cover6", "press", "nickel", "dime"]


def _scheme_frame(prefix: str, extra_id: dict[str, str] | None = None) -> pd.DataFrame:
    df = _concat(prefix)
    if df.empty:
        return df
    ren = {"name": "team", **_SCHEME_REN, **(extra_id or {})}
    df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns})
    df["team"] = df["team"].map(norm_team)
    if "opp" in df.columns:
        df["opp"] = df["opp"].map(norm_team)
    keep = ["season", "week", "team"] + (["opp"] if "opp" in df.columns else []) \
        + list((extra_id or {}).values()) + [c for c in SCHEME_RATE_COLS if c in df.columns]
    return df[[c for c in keep if c in df.columns]].dropna(subset=["team", "season"]).reset_index(drop=True)


@lru_cache(maxsize=1)
def coverage_matrix() -> pd.DataFrame:
    """Per DEFENSE team-week: man/zone/one_high/two_high/press/cover0..6 rates + personnel."""
    return _scheme_frame("coverage-matrix_")


@lru_cache(maxsize=1)
def coordinator_profiles() -> pd.DataFrame:
    """Per DEFENSE team-week + DC name — for early-season backfill when the DC is unchanged."""
    return _scheme_frame("coordinator-split_team-defensive-coordinator_", {"team_dc": "dc"})


def available() -> dict[str, int]:
    """Quick inventory for sanity checks / the Data Check screen."""
    return {
        "receiving_splits_rows": len(receiving_splits()),
        "passing_splits_rows": len(passing_splits()),
        "receiving_weekly_rows": len(receiving_weekly()),
        "rushing_weekly_rows": len(rushing_weekly()),
        "passing_weekly_rows": len(passing_weekly()),
        "coverage_matrix_rows": len(coverage_matrix()),
        "coordinator_profiles_rows": len(coordinator_profiles()),
    }
