"""The median correction (model v5): a projection is a MEAN, the book's line sits near the MEDIAN, and for right-skewed outcomes (receiving yards, receptions, rushing yards) the two differ.

On the 2026 graded lines the actual beat OUR projection only about 41% of the time and beat the LINE about 47%: a mean above the line is not an OVER edge by itself, which is why the screen's
OVER leans had no skill beyond the market's own tilt (matchup_model/restate_2026_report.md, fantasy-model-open-threads). This module learns, from graded lines of EARLIER weeks only, the
typical ratio actual / projection (its median) per market and per projection tercile, and rescales a projection to that median before it is compared with a line. Everything else
(`our proj`, the role check, the DFS tools, the simulator) keeps the mean.

    factors = learn_as_of(line_history, 2026, week)         # {market: (tercile cut points, median ratio overall, {tercile: median ratio or None})}
    our_med = apply(factors, "rec yds", our_proj)            # the number a line is judged against

Walk-forward test (2026 weeks 3-9, the rule learned only on weeks before each test week; the identical procedure is what `learn_as_of` does): at |z| >= 0.15 the lean as logged won
52.6% (+0.5% at -110) and the corrected lean 56.5% (+7.9%), ROI difference +7.4 points [+3.6, +11.5] (player-clustered bootstrap): it removes the losing OVERs (-3.3% -> +2.4%) and roughly
doubles the UNDER bets at the same ~+12%. The rule was proposed after looking at weeks 1-8, so week 9 (+11.8 [+1.3, +23.2]) was the first clean week; on 2025 closing lines it mostly recovered
the market's own UNDER tilt (a blind UNDER still did better).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_CELL = 25            # graded lines a market (or one projection tercile of it) needs before its median ratio is used
MIN_WEEKS = 2            # prior graded weeks needed before any correction applies (the test started at week 3, learned on weeks 1-2)
MIN_PROJ = 0.5           # projections below this are left out of the learning: a ratio to a near-zero projection is noise


def graded(lh: pd.DataFrame, season: int, before_week: int) -> pd.DataFrame:
    """The graded lines of `season` from weeks before `before_week`, one per (week, player, market), with numeric `our proj` and `actual`."""
    if lh is None or lh.empty or not {"season", "week", "player", "market", "our proj", "actual", "result"} <= set(lh.columns):
        return pd.DataFrame(columns=["week", "player", "market", "our proj", "actual"])
    g = lh[(lh["season"] == season) & (pd.to_numeric(lh["week"], errors="coerce") < before_week) & lh["result"].isin(["win", "loss", "push"])]
    g = g.drop_duplicates(["week", "player", "market"]).copy()
    for c in ("our proj", "actual"):
        g[c] = pd.to_numeric(g[c], errors="coerce")
    return g[g["our proj"].notna() & g["actual"].notna()]


def learn(g: pd.DataFrame, min_cell: int = MIN_CELL) -> dict:
    """{market: (cut points between projection terciles, median actual / projection over the market, {tercile: median ratio, None where it has fewer than `min_cell` lines})}."""
    out: dict = {}
    for mk, d in g.groupby("market"):
        d = d[d["our proj"] >= MIN_PROJ]
        if len(d) < min_cell:
            continue
        ratio = (d["actual"] / d["our proj"]).to_numpy()
        proj = d["our proj"].to_numpy()
        cuts = np.quantile(proj, [1 / 3, 2 / 3])
        bins = np.digitize(proj, cuts)
        out[mk] = (cuts, float(np.median(ratio)), {i: float(np.median(ratio[bins == i])) if int((bins == i).sum()) >= min_cell else None for i in range(3)})
    return out


def learn_as_of(lh: pd.DataFrame, season: int, week: int, min_cell: int = MIN_CELL, min_weeks: int = MIN_WEEKS) -> dict:
    """The factors for a pull in `week`: learned from the graded lines of the weeks before it. Empty (no correction) until `min_weeks` such weeks exist."""
    g = graded(lh, season, week)
    if g.empty or g["week"].nunique() < min_weeks:
        return {}
    return learn(g, min_cell)


def factor(factors: dict, market: str, proj: float) -> float:
    """The ratio a projection of `proj` in `market` is rescaled by; 1.0 for a market without factors."""
    f = factors.get(market)
    if f is None:
        return 1.0
    cuts, k_all, k_bin = f
    return float(k_bin.get(int(np.digitize(proj, cuts))) or k_all)


def apply(factors: dict, market: str, proj: float) -> float:
    return float(proj) * factor(factors, market, float(proj))
