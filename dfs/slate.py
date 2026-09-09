"""Join the DraftKings salary pool to the weekly projections into one 'slate' table."""
from __future__ import annotations

import pandas as pd

from dfs.names import fuzzy_match, normalize_name, player_key
from dfs.scoring import add_ceiling_floor


def build_slate(salaries: pd.DataFrame, projections: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (slate, unmatched).

    slate: one row per DK player that got a projection —
        dk_id, name, pos, team, opp, salary, game_time, status, playable,
        proj, ceiling, floor, value  (value = proj per $1000 of salary)
    unmatched: DK players with a salary but no projection (shown in Data Check).
    """
    sal = salaries.copy()
    proj = projections.copy()

    sal["key"] = [
        player_key(n, t, p) for n, t, p in zip(sal["name"], sal["team"], sal["pos"])
    ]
    proj_by_key = proj.set_index("key")["proj"].to_dict()

    merged_proj = []
    for _, row in sal.iterrows():
        val = proj_by_key.get(row["key"])
        if val is None:
            # fuzzy fallback within same team + position
            same = proj[(proj["team"] == row["team"]) & (proj["pos"] == row["pos"])]
            cand = fuzzy_match(normalize_name(row["name"]),
                              [normalize_name(x) for x in same["name"].tolist()])
            if cand is not None:
                hit = same[[normalize_name(x) == cand for x in same["name"]]]
                if not hit.empty:
                    val = float(hit.iloc[0]["proj"])
        merged_proj.append(val)

    sal["proj"] = merged_proj
    matched = sal[sal["proj"].notna()].copy()
    unmatched = sal[sal["proj"].isna()].copy()

    matched["proj"] = matched["proj"].astype(float)
    matched = add_ceiling_floor(matched)
    matched["value"] = (matched["proj"] / (matched["salary"] / 1000.0)).round(2)
    matched = matched.sort_values(["pos", "proj"], ascending=[True, False]).reset_index(drop=True)

    keep_un = [c for c in ("dk_id", "name", "pos", "team", "opp", "salary", "status") if c in unmatched.columns]
    return matched, unmatched[keep_un].reset_index(drop=True)
