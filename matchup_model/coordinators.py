"""Defensive-coordinator history, from the Coach Records exports.

`coach-records_dc_<season>.csv` (Team ▸ Coach Records, role = DC, one season) has one
row per team's defensive coordinator that year: COACH, ROLE, TEAM, plus W-L / per-play
stats we don't need here. Stacking the per-season files gives, for every
(season, defense team), who coordinated that defense — which lets the Player Lookup
panel flag "you've faced this coordinator before" even when they've since changed teams.
"""
from __future__ import annotations

import glob
import re
from functools import lru_cache

import pandas as pd

from dfs.names import norm_team
from matchup_model.config import DATA

_GLOB = "coach-records_dc_*.csv"


@lru_cache(maxsize=1)
def dc_history() -> pd.DataFrame:
    """Unique (dc, season, team) rows — which coordinator ran each defense each year."""
    frames = []
    for f in sorted(glob.glob(str(DATA / _GLOB))):
        m = re.search(r"(20\d{2})", f)
        if not m:
            continue
        season = int(m.group(1))
        try:
            d = pd.read_csv(f, header=1)
        except Exception:
            continue
        if "COACH" not in d.columns or "TEAM" not in d.columns:
            continue
        d = d[d["COACH"].notna() & (d["COACH"].astype(str).str.strip() != "")]
        d = d[~d["COACH"].astype(str).str.contains("League Avg|Exported from", na=False)]
        for _, row in d.iterrows():
            dc = str(row["COACH"]).strip()
            for team in re.split(r"[,/]", str(row["TEAM"])):
                team = norm_team(team.strip())
                if team:
                    frames.append({"dc": dc, "season": season, "team": team})
    if not frames:
        return pd.DataFrame(columns=["dc", "season", "team"])
    return pd.DataFrame(frames).drop_duplicates().reset_index(drop=True)


def latest_coord_season() -> int | None:
    h = dc_history()
    return int(h["season"].max()) if not h.empty else None


def current_dc(team: str) -> dict:
    """Most recent coordinator on record for `team`. {'dc', 'season'} or {}."""
    h = dc_history()
    t = norm_team(team)
    sub = h[h["team"] == t]
    if sub.empty:
        return {}
    yr = int(sub["season"].max())
    mode = sub[sub["season"] == yr]["dc"].mode()
    return {"dc": mode.iat[0], "season": yr} if not mode.empty else {}


def dc_team_seasons(dc: str) -> list[tuple[int, str]]:
    """Every (season, team) a coordinator ran a defense, oldest first."""
    h = dc_history()
    rows = h[h["dc"] == dc][["season", "team"]].drop_duplicates()
    return sorted((int(s), t) for s, t in rows.itertuples(index=False, name=None))
