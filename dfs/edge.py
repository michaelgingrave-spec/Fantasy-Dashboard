"""Predictive Matchup Edge model.

Blends three signals into one score per player, each contributing 0 when its data is
absent so the model degrades gracefully:

  opportunity   — usage/efficiency from FantasyPoints Data Suite player tables
                  (route share, target share, aDOT, YPRR, air yards, red-zone/designed
                  looks, rush share, ...). Higher = better.
  defense_vuln  — how much the opposing defense gives up, from Data Suite team/defense
                  tables. Higher = softer matchup. A few "pressure/sack/EPA" style
                  columns are inverted.
  vegas         — opposing-adjusted implied team total from The Odds API. Higher = better.

edge = z(opportunity) + z(defense_vuln) + z(vegas), then re-z-scored within position.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dfs.datasuite.features import defense_features, player_features
from dfs.names import name_team_key
from dfs.odds import get_nfl_game_lines, team_environment

_OPPORTUNITY_HINTS = (
    "route", "target", "share", "yprr", "adot", "air", "redzone", "red_zone",
    "endzone", "end_zone", "designed", "carry", "carries", "rush_share", "touch",
    "opportunit", "wopr", "tprr", "yac", "sep", "separation", "first_down",
)
_DEFENSE_INVERT_HINTS = ("pressure", "sack", "hurr", "blitz", "epa", "stuff", "havoc", "rating")


def _z(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std(ddof=0)
    if not sd or np.isnan(sd):
        return pd.Series(0.0, index=s.index)
    return ((s - s.mean()) / sd).fillna(0.0)


def _z_by_pos(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby("pos")[col].transform(_z) if "pos" in df.columns else _z(df[col])


def _opportunity_score(feats: pd.DataFrame) -> pd.Series:
    cols = [c for c in feats.columns if c.startswith("ds__")]
    if not cols:
        return pd.Series(0.0, index=feats.index)
    picked = [c for c in cols if any(h in c for h in _OPPORTUNITY_HINTS)] or cols
    zs = pd.DataFrame({c: _z(feats[c]) for c in picked})
    return zs.mean(axis=1).fillna(0.0)


def _defense_vuln_score(dfn: pd.DataFrame) -> pd.Series:
    cols = [c for c in dfn.columns if c.startswith("ds__")]
    if not cols:
        return pd.Series(0.0, index=dfn.index)
    parts = []
    for c in cols:
        z = _z(dfn[c])
        if any(h in c for h in _DEFENSE_INVERT_HINTS):
            z = -z
        parts.append(z)
    return pd.concat(parts, axis=1).mean(axis=1).fillna(0.0)


def weekly_edges(slate: pd.DataFrame, datasuite_dir=None) -> pd.DataFrame:
    """One row per skill player with edge components + a plain-language `why`."""
    skill = slate[slate["pos"].isin(["QB", "RB", "WR", "TE"])].copy()
    if skill.empty:
        return skill
    skill["nt_key"] = [name_team_key(n, t) for n, t in zip(skill["name"], skill["team"])]
    skill["opp_team"] = skill["opp"].astype(str).str.lstrip("@")

    pfeat, p_src = player_features(datasuite_dir)
    dfeat, d_src = defense_features(datasuite_dir)

    if not pfeat.empty:
        merged = skill.merge(pfeat.drop(columns=["name", "team", "pos"], errors="ignore"),
                             on="nt_key", how="left")
    else:
        merged = skill
    merged["opportunity"] = _opportunity_score(merged) if not pfeat.empty else 0.0

    if not dfeat.empty:
        d = dfeat.rename(columns={"team_key": "opp_team"})
        d = d.rename(columns={c: f"def_{c}" for c in d.columns if c.startswith("ds__")})
        merged = merged.merge(d, on="opp_team", how="left")
        dcols = merged[[c for c in merged.columns if c.startswith("def_ds__")]]
        dcols = dcols.rename(columns=lambda c: c[4:])
        merged["defense_vuln"] = _defense_vuln_score(dcols)
    else:
        merged["defense_vuln"] = 0.0

    env = team_environment(get_nfl_game_lines())
    if not env.empty:
        merged = merged.merge(env[["team", "implied_total", "game_total", "pace_tier"]],
                              on="team", how="left")
        merged["vegas"] = _z(merged["implied_total"])
    else:
        merged["vegas"] = 0.0
        merged["implied_total"] = np.nan

    merged["opportunity_z"] = _z_by_pos(merged, "opportunity")
    merged["defense_z"] = _z_by_pos(merged, "defense_vuln")
    merged["vegas_z"] = _z_by_pos(merged, "vegas")
    merged["edge"] = merged[["opportunity_z", "defense_z", "vegas_z"]].sum(axis=1)
    merged["edge_z"] = _z_by_pos(merged, "edge")
    merged["why"] = merged.apply(_why, axis=1)

    keep = [
        "dk_id", "name", "pos", "team", "opp", "salary", "proj", "value",
        "implied_total", "opportunity_z", "defense_z", "vegas_z", "edge", "edge_z", "why",
    ]
    keep = [c for c in keep if c in merged.columns]
    return merged[keep].sort_values("edge", ascending=False).reset_index(drop=True)


def _why(r: pd.Series) -> str:
    bits = []
    comp = {
        "usage/efficiency": r.get("opportunity_z", 0.0),
        "soft matchup": r.get("defense_z", 0.0),
        "game total": r.get("vegas_z", 0.0),
    }
    for label, val in sorted(comp.items(), key=lambda kv: -abs(kv[1])):
        if abs(val) < 0.4:
            continue
        arrow = "＋" if val > 0 else "－"
        bits.append(f"{arrow}{label}")
    it = r.get("implied_total")
    if it == it and it:  # not NaN
        bits.append(f"{it:.1f} implied")
    return ", ".join(bits) if bits else "neutral"


def player_boosts(edges: pd.DataFrame, weight: float = 0.05) -> dict[str, float]:
    """dk_id -> objective multiplier in [1-2w, 1+2w]."""
    if edges is None or edges.empty or "edge_z" not in edges.columns:
        return {}
    clipped = edges["edge_z"].clip(-2, 2)
    return {str(i): 1.0 + weight * z for i, z in zip(edges["dk_id"], clipped)}
