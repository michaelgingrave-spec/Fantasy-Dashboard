"""Turn a point projection into a ceiling / floor band and an optimizer objective value.

We only have a single mean projection per player, so the band is a crude position-level
coefficient-of-variation prior (see config.POSITION_CV). Good enough to bias lineups toward
higher-variance players for tournaments via the `lean` knob.
"""
from __future__ import annotations

import pandas as pd

from dfs.config import POSITION_CV


def add_ceiling_floor(slate: pd.DataFrame) -> pd.DataFrame:
    df = slate.copy()
    cv = df["pos"].map(POSITION_CV).fillna(0.40)
    df["ceiling"] = (df["proj"] * (1.0 + cv)).round(2)
    df["floor"] = (df["proj"] * (1.0 - cv)).clip(lower=0.0).round(2)
    return df


def objective_points(slate: pd.DataFrame, lean: float = 0.0) -> pd.Series:
    """lean 0 -> mean projection, lean 1 -> ceiling. Values used as MILP coefficients."""
    lean = max(0.0, min(1.0, float(lean)))
    if "ceiling" not in slate.columns:
        slate = add_ceiling_floor(slate)
    return (slate["proj"] + lean * (slate["ceiling"] - slate["proj"])).astype(float)
