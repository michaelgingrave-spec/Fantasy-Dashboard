"""Slate-level DFS suggestions derived from the joined slate + Matchup Edge table.

Pure functions over dataframes — no solver, no I/O.
"""
from __future__ import annotations

import pandas as pd

_POS = ("QB", "RB", "WR", "TE", "DST")


def value_plays(slate: pd.DataFrame, per_pos: int = 4, min_proj: float = 5.0) -> pd.DataFrame:
    """Best projected points per $1k salary, by position."""
    df = slate[slate["proj"] >= min_proj].copy()
    df["value"] = (df["proj"] / (df["salary"] / 1000.0)).round(2)
    out = (
        df.sort_values("value", ascending=False)
        .groupby("pos", group_keys=False)
        .head(per_pos)
    )
    return out.sort_values(["pos", "value"], ascending=[True, False])[
        ["name", "pos", "team", "opp", "salary", "proj", "value"]
    ].reset_index(drop=True)


def leverage_plays(slate: pd.DataFrame, edges: pd.DataFrame, n: int = 12,
                   salary_max: int = 5600) -> pd.DataFrame:
    """Cheap-ish players in strong spots: high edge_z, lower salary."""
    if edges is None or edges.empty:
        return pd.DataFrame()
    e = edges.merge(slate[["dk_id", "status"]], on="dk_id", how="left")
    e = e[(e["salary"] <= salary_max) & (e["edge_z"] > 0.5)]
    return (
        e.sort_values("edge_z", ascending=False)
        .head(n)[["name", "pos", "team", "opp", "salary", "proj", "edge_z", "why"]]
        .round({"edge_z": 2})
        .reset_index(drop=True)
    )


def punt_plays(slate: pd.DataFrame, per_pos: int = 3, salary_max: int = 4200) -> pd.DataFrame:
    """Minimum-salary bodies that still project for something — to free up cap for studs."""
    df = slate[(slate["salary"] <= salary_max) & (slate["proj"] > 0)].copy()
    out = (
        df.sort_values("proj", ascending=False)
        .groupby("pos", group_keys=False)
        .head(per_pos)
    )
    return out.sort_values(["pos", "proj"], ascending=[True, False])[
        ["name", "pos", "team", "opp", "salary", "proj"]
    ].reset_index(drop=True)


def stack_suggestions(slate: pd.DataFrame, edges: pd.DataFrame, n: int = 6) -> pd.DataFrame:
    """QB + best same-team pass-catcher, with a bring-back option, ranked by combined edge."""
    qbs = slate[slate["pos"] == "QB"].copy()
    pcs = slate[slate["pos"].isin(["WR", "TE"])].copy()
    if qbs.empty or pcs.empty:
        return pd.DataFrame()

    ez = (edges.set_index("dk_id")["edge_z"].to_dict() if edges is not None and not edges.empty else {})
    rows = []
    for _, qb in qbs.iterrows():
        mates = pcs[pcs["team"] == qb["team"]].sort_values("proj", ascending=False)
        if mates.empty:
            continue
        wr = mates.iloc[0]
        opp = str(qb["opp"]).lstrip("@")
        backs = pcs[pcs["team"] == opp].sort_values("proj", ascending=False)
        back = backs.iloc[0]["name"] if not backs.empty else "—"
        combined = (
            qb["proj"] + wr["proj"]
            + 2.0 * (ez.get(qb["dk_id"], 0.0) + ez.get(wr["dk_id"], 0.0))
        )
        rows.append(
            {
                "stack": f"{qb['name']} + {wr['name']}",
                "team": qb["team"],
                "opp": opp,
                "salary": int(qb["salary"] + wr["salary"]),
                "proj": round(float(qb["proj"] + wr["proj"]), 1),
                "bring_back": back,
                "score": round(float(combined), 2),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("score", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )
