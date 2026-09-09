"""Load the user's weekly FantasyPoints projection export.

Expected file: data/projections.week{N}.csv, shaped like the FantasyPoints "Projections"
download — a group-label row first, then the real header:

    "","","Player Details","","","","","Projection"
    "#","NAME","POS","POS","SEASON","WEEK","OPP","FPTS"
    "1","Jahmyr Gibbs","RB","DET","2026","1","NO","23.1"

Only FPTS (a fantasy-point projection) is available this year — no stat components.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from dfs.config import projection_path
from dfs.names import norm_team, player_key

# FantasyPoints positions we keep (DK Classic has no kicker).
KEEP_POS = {"QB", "RB", "WR", "TE", "DST"}


class ProjectionError(RuntimeError):
    pass


def load_weekly_projections(week: int, path: str | Path | None = None) -> pd.DataFrame:
    """Return columns: name, pos, team, opp, week, proj, key."""
    p = Path(path) if path else projection_path(week)
    if not p.exists():
        raise ProjectionError(
            f"No projection file at {p}. Drop your FantasyPoints weekly export there "
            f"as 'projections.week{week}.csv'."
        )

    # The real header is the 2nd line; fall back to row 0 if the layout is flat.
    df = pd.read_csv(p, header=1, dtype=str, keep_default_na=False)
    if "NAME" not in df.columns:
        df = pd.read_csv(p, header=0, dtype=str, keep_default_na=False)
    if "NAME" not in df.columns or "FPTS" not in df.columns:
        raise ProjectionError(
            f"{p.name} is missing NAME/FPTS columns (got {list(df.columns)[:8]}...)."
        )

    # Second 'POS' column is the team; pandas renames the dupe to 'POS.1'.
    team_col = "POS.1" if "POS.1" in df.columns else ("Team" if "Team" in df.columns else None)
    out = pd.DataFrame(
        {
            "name": df["NAME"].str.replace("\n", " ", regex=False).str.strip(),
            "pos": df["POS"].str.strip().str.upper(),
            "team": (df[team_col] if team_col else "").map(norm_team) if team_col else "",
            "opp": df["OPP"].map(norm_team) if "OPP" in df.columns else "",
            "proj": pd.to_numeric(df["FPTS"], errors="coerce"),
        }
    )
    out = out[out["pos"].isin(KEEP_POS)].copy()
    out = out[out["name"].astype(bool)].copy()
    out["proj"] = out["proj"].fillna(0.0).clip(lower=0.0)
    out["week"] = int(week)
    out["key"] = [player_key(n, t, pos) for n, t, pos in zip(out["name"], out["team"], out["pos"])]
    out = out.drop_duplicates(subset="key", keep="first").reset_index(drop=True)
    if out.empty:
        raise ProjectionError(f"{p.name} parsed to zero usable rows.")
    return out
