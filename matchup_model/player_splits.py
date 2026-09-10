"""Shrunk per-coverage-split player efficiency.

Year-over-year stability check (see backtest_report.md): a player's *raw* efficiency vs a
coverage split (targets/route vs man: r≈0.6; yards/route: r≈0.4) is a real skill; the
*delta* vs their own overall is much noisier (fp/route delta: r≈0.13). So we estimate a
shrunk **raw** efficiency for each split and let `matchup.py` blend them by the coverage
rates the player is predicted to face.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from matchup_model.config import SEASON_DECAY, SHRINK_K
from matchup_model import ingest

SPLITS = ["overall", "man", "zone", "single_high", "two_high"]
# (position group -> per-unit stats, sample column, shrink-K bucket)
_SPEC = {
    "REC": dict(stats=["tprr", "yprr", "fp_rr"], n="rte", kbucket="routes"),
    "QB": dict(stats=["ypa", "fp_db"], n="db", kbucket="dropbacks"),
}
POS_TO_GROUP = {"WR": "REC", "TE": "REC", "QB": "QB"}
FP_STAT = {"REC": "fp_rr", "QB": "fp_db"}       # kept for the panel; nudge uses tprr+yprr


def _weighted(df: pd.DataFrame, val: str, wt: str) -> float:
    w = df[wt].to_numpy(dtype=float)
    v = df[val].to_numpy(dtype=float)
    m = np.isfinite(w) & np.isfinite(v) & (w > 0)
    return float(np.sum(w[m] * v[m]) / np.sum(w[m])) if m.any() else np.nan


@lru_cache(maxsize=1)
def _long() -> pd.DataFrame:
    rec = ingest.receiving_splits()
    qb = ingest.passing_splits()
    frames = []
    if not rec.empty:
        r = rec.copy(); r["source"] = "receiving"; frames.append(r)
    if not qb.empty:
        q = qb.copy(); q["source"] = "passing"; frames.append(q)
    long = pd.concat(frames, ignore_index=True)
    long["pos"] = long["pos"].fillna("").str.upper()
    long["grp"] = long["pos"].map(POS_TO_GROUP)
    long["rw"] = SEASON_DECAY ** (long["season"].max() - long["season"])
    return long


@lru_cache(maxsize=16)
def build_player_splits(through_season: int | None = None,
                        seasons: tuple[int, ...] | None = None) -> pd.DataFrame:
    """One row per (player, split, stat): shrunk raw efficiency `eff` + sample `n`,
    plus `eff_overall` and `delta` (eff - eff_overall) for display.

    `through_season` keeps seasons <= that year (walk-forward backtests).
    `seasons` restricts to an explicit set of years (the Player Lookup window picker)."""
    long = _long()
    if through_season is not None:
        long = long[long["season"] <= through_season]
    if seasons is not None:
        long = long[long["season"].isin(seasons)]
    rows = []
    for grp, spec in _SPEC.items():
        sub = long[long["grp"] == grp]
        if sub.empty:
            continue
        ncol, K = spec["n"], SHRINK_K[spec["kbucket"]]

        # position baseline per (split, stat) = sample-weighted mean across players
        base = {}
        for split in SPLITS:
            ss = sub[sub["split"] == split]
            for stat in spec["stats"]:
                pp = (ss.assign(w=ss["rw"] * ss[ncol])
                        .groupby("name_key")
                        .apply(lambda g: pd.Series({"e": _weighted(g, stat, "w"), "n": g[ncol].sum()})))
                base[(split, stat)] = (_weighted(pp.assign(w=pp["n"]).rename(columns={"e": stat}), stat, "w")
                                       if not pp.empty else np.nan)

        for name_key, pg in sub.groupby("name_key"):
            pos = pg["pos"].mode().iat[0] if not pg["pos"].mode().empty else ""
            for stat in spec["stats"]:
                # player's shrunk overall first (toward position baseline)
                o = pg[pg["split"] == "overall"]
                n_o = float(o[ncol].sum())
                raw_o = _weighted(o.assign(w=o["rw"] * o[ncol]), stat, "w")
                if not np.isfinite(raw_o):
                    continue
                b_o = base[("overall", stat)]
                eff_o = (n_o * raw_o + K * b_o) / (n_o + K) if np.isfinite(b_o) else raw_o
                for split in SPLITS:
                    s = pg[pg["split"] == split]
                    n_s = float(s[ncol].sum())
                    raw_s = _weighted(s.assign(w=s["rw"] * s[ncol]), stat, "w")
                    # shrink the split estimate toward this player's shrunk overall
                    eff_s = ((n_s * raw_s + K * eff_o) / (n_s + K)
                             if np.isfinite(raw_s) else eff_o)
                    rows.append(dict(
                        name_key=name_key, pos=pos, grp=grp, split=split, stat=stat,
                        eff=round(eff_s, 4), eff_overall=round(eff_o, 4),
                        delta=round(eff_s - eff_o, 4), n=int(n_s),
                    ))
    return pd.DataFrame(rows)


def player_eff(splits_df: pd.DataFrame, name_key: str, stat: str) -> dict:
    """{split: shrunk eff} for one player/stat, incl. 'overall'. Empty dict if unknown."""
    hit = splits_df[(splits_df.name_key == name_key) & (splits_df.stat == stat)]
    if hit.empty:
        return {}
    return dict(zip(hit["split"], hit["eff"]))


def best_spots(name_key: str, stat: str = "tprr",
               seasons: tuple[int, ...] | None = None) -> pd.DataFrame:
    df = build_player_splits(seasons=seasons)
    hit = df[(df.name_key == name_key) & (df.stat == stat)].copy()
    return hit.sort_values("delta", key=lambda s: s.abs(), ascending=False)[
        ["split", "eff_overall", "eff", "delta", "n"]
    ].reset_index(drop=True)
