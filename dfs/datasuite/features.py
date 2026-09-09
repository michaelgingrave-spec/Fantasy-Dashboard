"""Assemble per-player and per-defense feature frames from whatever Data Suite tables
are present. Missing tables simply mean missing columns; nothing raises.
"""
from __future__ import annotations

from functools import reduce
from pathlib import Path

import pandas as pd

from dfs.config import DATASUITE_DIR
from dfs.datasuite.registry import DataSuiteFile, discover, load_table


def _merge_all(frames: list[pd.DataFrame], on: str) -> pd.DataFrame:
    frames = [f for f in frames if not f.empty and on in f.columns]
    if not frames:
        return pd.DataFrame(columns=[on])
    def m(a, b):
        dupes = (set(a.columns) & set(b.columns)) - {on, "name", "team", "pos"}
        b = b.drop(columns=[c for c in dupes if c in b.columns])
        return a.merge(b, on=on, how="outer", suffixes=("", "_x"))
    return reduce(m, frames)


def player_features(folder: Path | None = None) -> tuple[pd.DataFrame, list[str]]:
    files = discover(Path(folder or DATASUITE_DIR))
    player_files = [f for f in files if f.scope == "player"]
    frames, sources = [], []
    for f in player_files:
        tbl = load_table(f)
        if not tbl.empty:
            frames.append(tbl)
            sources.append(f.path.name)
    merged = _merge_all(frames, on="nt_key")
    return merged, sources


def defense_features(folder: Path | None = None) -> tuple[pd.DataFrame, list[str]]:
    files = discover(Path(folder or DATASUITE_DIR))
    def_files = [f for f in files if f.scope in ("defense", "team")]
    frames, sources = [], []
    for f in def_files:
        tbl = load_table(f)
        if not tbl.empty:
            frames.append(tbl)
            sources.append(f.path.name)
    merged = _merge_all(frames, on="team_key")
    return merged, sources


def inventory(folder: Path | None = None) -> pd.DataFrame:
    """For the Data Check screen: what files were found and how they were classified."""
    files = discover(Path(folder or DATASUITE_DIR))
    rows = []
    for f in files:
        tbl = load_table(f)
        feat_cols = [c for c in tbl.columns if c.startswith("ds__")]
        rows.append(
            {
                "file": f.path.name,
                "table_type": f.table_type,
                "scope": f.scope,
                "rows": len(tbl),
                "features": len(feat_cols),
                "as_of": f.as_of.strftime("%Y-%m-%d %H:%M") if f.as_of else "",
                "usable": "yes" if len(tbl) and feat_cols else "no",
            }
        )
    return pd.DataFrame(rows)
