"""Heuristic projected ownership — a "chalk score" for GPP leverage.

There's no historical actual-ownership data to fit against, so this is a formula, not a
model: within each position, rank players by the things that actually drive ownership —
value (pts per $1k), raw projection, the cheap-enabler pull, and game total — then
normalise so the numbers sum to realistic per-position totals. The *ordering* is the
point; treat the percentages as directional, not precise.

    from dfs.ownership import chalk_ownership
    pool = chalk_ownership(pool_df, week)   # adds an 'own_pct' column
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# DK Classic: 1 QB, 2 RB, 3 WR, 1 TE, 1 DST + 1 FLEX (~RB/WR/TE 45/45/10).
# Sum of a position's ownership across the field ~= (roster slots) * 100.
POS_OWN_SUM = {"QB": 100.0, "RB": 250.0, "WR": 350.0, "TE": 110.0, "DST": 100.0}

W_VALUE, W_PROJ, W_CHEAP, W_TOTAL = 1.00, 0.80, 0.45, 0.30
TEMP = 1.40            # softmax temperature — lower = more concentrated ownership
OWN_CLIP = (0.1, 45.0)
Z_CLIP = 2.5           # cap per-factor z-scores so a thin pool can't spike ownership


def _z(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std(ddof=0)
    if not sd or not np.isfinite(sd):
        return pd.Series(0.0, index=s.index)
    return ((s - s.mean()) / sd).clip(-Z_CLIP, Z_CLIP)


def _team_totals(week: int) -> dict:
    """{nflverse_team: implied points} for the week, from the cached Vegas lines. {} if
    the nflverse cache isn't available."""
    try:
        from matchup_model.opp import data as D
        from matchup_model.opp.blend import current_season
        g = D.games()
        g = g[(g.season == current_season()) & (g.week == int(week))]
        return {t: float(v) for t, v in zip(g["team"], g["team_total"]) if pd.notna(v)}
    except Exception:  # noqa: BLE001
        return {}


def chalk_ownership(pool: pd.DataFrame, week: int, salary_col: str = "salary",
                    proj_col: str = "proj", pos_col: str = "pos",
                    team_col: str = "team") -> pd.DataFrame:
    """Return `pool` with an `own_pct` column (NaN for players without a salary)."""
    out = pool.copy()
    out["own_pct"] = np.nan
    if salary_col not in out or proj_col not in out:
        return out

    sal = pd.to_numeric(out[salary_col], errors="coerce")
    proj = pd.to_numeric(out[proj_col], errors="coerce")
    priced = sal.notna() & (sal > 0) & proj.notna()
    if not priced.any():
        return out

    totals = _team_totals(week)
    try:
        from matchup_model.opp.data import canon_team
    except Exception:  # noqa: BLE001
        canon_team = lambda x: str(x).upper()  # noqa: E731

    d = out[priced].copy()
    d["_sal"] = sal[priced]
    d["_proj"] = proj[priced]
    d["_val"] = d["_proj"] / (d["_sal"] / 1000.0)
    d["_tt"] = (d[team_col].map(canon_team).map(totals) if team_col in d
                else np.nan) if totals else np.nan

    own = pd.Series(0.0, index=d.index)
    for pos, grp in d.groupby(d[pos_col].astype(str).str.upper()):
        if len(grp) == 0:
            continue
        vz, pz = _z(grp["_val"]), _z(grp["_proj"])
        # cheap-enabler pull: well-below-median salary AND a viable projection
        smed, s25 = grp["_sal"].median(), grp["_sal"].quantile(0.25)
        pmed = grp["_proj"].median()
        cheap = ((smed - grp["_sal"]) / max(smed - s25, 1.0)).clip(0, 1.6)
        cheap = cheap.where(grp["_proj"] >= 0.6 * pmed, 0.0)
        tz = _z(grp["_tt"]) if grp["_tt"].notna().any() else pd.Series(0.0, index=grp.index)

        raw = W_VALUE * vz + W_PROJ * pz + W_CHEAP * cheap + W_TOTAL * tz
        w = np.exp((raw - raw.max()) / TEMP)
        target = POS_OWN_SUM.get(pos, 100.0)
        o = ((w / w.sum()) * target).clip(*OWN_CLIP)
        own.loc[grp.index] = o

    out.loc[d.index, "own_pct"] = own.round(1)
    return out
