"""Millionaire Maker study, part three: where the money goes, and whether the game around a player is already in his price.

Three tables, all from the same historical weeks as `dfs.mm_study`:

* `spend_table`        how the cap is split between QB, RB, WR, TE and DST in each week's best lineup (hindsight, so it describes the winners; `mm_study`'s forward test says
                       which splits pay before kickoff).
* `price_tier_table`   every position cut into price tiers: points per $1,000, how often a player returns 5x his price, how often he finishes among the week's top scorers at his
                       position. This is the market's own pricing curve, with no lineup in it.
* `environment_effects` regressions of the points a player scores *beyond what his salary implies* on the implied team total, the weakness of the defense he faces and the
                       strength of his own offense. If the salary already contains the game, these are zero.

Intervals resample whole weeks: a week's players and lineups share one slate, so they are not independent draws.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dfs import hindsight as HS

ERAS = (("2014-17", 2014, 2017), ("2018-21", 2018, 2021), ("2026", 2026, 2026))

# a lineup's cap split: the nine players' salaries by position, plus what was left unspent
SLOTS = (("QB", "sal_qb"), ("RBs (all)", "sal_rb"), ("WRs (all)", "sal_wr"), ("TEs (all)", "sal_te"), ("DST", "sal_dst"), ("Left unspent", "sal_left"),
         ("per RB", "avg_rb"), ("per WR", "avg_wr"), ("per TE", "avg_te"))

# price tiers by position (lower edge inclusive); absolute dollars, because that is what the builder's rules are written in
TIERS = {"QB": (0, 5500, 6500, 7500), "RB": (0, 4500, 6000, 7500), "WR": (0, 4500, 6000, 7500, 8500), "TE": (0, 3500, 4500, 5500), "DST": (0, 2600, 3000, 3400)}
TOP_K = {"QB": 3, "RB": 6, "WR": 9, "TE": 3, "DST": 3}              # "a top scorer at his position that week": roughly the finishes a winning lineup is built from
VALUE_TARGET = 5.0                                                   # points per $1,000 that counts as a ceiling game


def era_of(season: pd.Series | np.ndarray) -> np.ndarray:
    s = np.asarray(season)
    return np.select([s <= 2017, s <= 2021], ["2014-17", "2018-21"], default="2026")


def _slots(df: pd.DataFrame) -> pd.DataFrame:
    """The lineup table with each position's average price per player added."""
    d = df.copy()
    for pos, tot, n in (("rb", "sal_rb", "n_rb"), ("wr", "sal_wr", "n_wr"), ("te", "sal_te", "n_te")):
        d[f"avg_{pos}"] = d[tot] / d[n].replace(0, np.nan)
    return d


def _week_means(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df.groupby(["season", "week"])[cols].mean()


def spend_table(hind: pd.DataFrame, reps: int = 1000, seed: int = 5) -> list[dict]:
    """How the best lineups split the cap: for each slot, the mean spend of each week's single best lineup (with a 95% week-bootstrap interval), the mean over each week's best 25, the
    spread across weeks (10th and 90th percentile of the single best lineup), and the mean by era to see whether it is stable. `hind` is the hindsight feature table (rank 1 = the
    true optimum).

    There is no ordinary-lineup comparison on purpose. The lineups a salary-only optimizer builds have habits of their own (it fills the FLEX with a TE more than half the time, for
    one), so a gap against them measures the optimizer, not the winners. Whether a split *pays* is what the forward test is for."""
    hind = _slots(hind)
    cols = [c for _, c in SLOTS]
    best = _week_means(hind[hind["rank"] == 1], cols)
    top = _week_means(hind, cols)
    era = era_of(best.index.get_level_values("season"))
    idx = np.random.default_rng(seed).integers(0, len(best), size=(reps, len(best)))
    rows = []
    for label, col in SLOTS:
        a, t = best[col].to_numpy(float), top[col].to_numpy(float)
        boots = np.nanmean(a[idx], axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        eras = {name: {"best": float(np.nanmean(a[era == name])), "weeks": int((era == name).sum())} for name, _, _ in ERAS if (era == name).any()}
        rows.append({"slot": label, "col": col, "best1": float(np.nanmean(a)), "lo": float(lo), "hi": float(hi), "top25": float(np.nanmean(t)),
                     "p10": float(np.nanpercentile(a, 10)), "p90": float(np.nanpercentile(a, 90)), "eras": eras})
    return rows


# ── Price tiers ─────────────────────────────────────────────────────────────────────────────────────────
def load_pools(weeks: list[tuple[int, int]] | None = None) -> pd.DataFrame:
    """Every study week's pool (players with salary and DraftKings points and the pre-game environment columns) stacked, with `season` and `week` columns."""
    out = []
    for season, week in weeks or HS.study_weeks():
        p = HS.pool_context(season, week).copy()
        p["season"], p["week"] = season, week
        out.append(p)
    return pd.concat(out, ignore_index=True)


def _tier_label(lo: int, hi: float) -> str:
    if lo <= 0:
        return f"under ${hi:,.0f}"
    return f"${lo:,.0f} and up" if np.isinf(hi) else f"${lo:,.0f}-{hi - 1:,.0f}"


def _cluster_rate(codes: np.ndarray, n_weeks: int, mask: np.ndarray, hit: np.ndarray, idx: np.ndarray) -> tuple[float, float, float]:
    """Share of the masked rows that are hits, with a 95% interval from resampling weeks (`codes` = each row's week index, `idx` = the resampled week indices)."""
    n_w = np.bincount(codes[mask], minlength=n_weeks).astype(float)
    h_w = np.bincount(codes[mask], weights=hit[mask], minlength=n_weeks)
    est = h_w.sum() / max(n_w.sum(), 1.0)
    boots = h_w[idx].sum(axis=1) / np.maximum(n_w[idx].sum(axis=1), 1.0)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(est), float(lo), float(hi)


def price_tier_table(pools: pd.DataFrame, reps: int = 1000, seed: int = 11) -> list[dict]:
    """Per position and price tier: how many player-weeks, points per game and per $1,000, the share that returned `VALUE_TARGET`x their price, and the share that finished among
    the week's top `TOP_K` scorers at the position (with intervals). The share of all of a position's top finishers that came from the tier sits beside the tier's share of
    the pool, so a tier that supplies more winners than its size would predict stands out."""
    df = pools[pools["salary"] > 0].copy()
    df["wk"] = pd.factorize(df["season"] * 100 + df["week"])[0]
    n_weeks = int(df["wk"].max()) + 1
    df["x"] = df["pts"] / (df["salary"] / 1000.0)
    df["top"] = df.groupby(["wk", "pos"])["pts"].rank(ascending=False, method="first") <= df["pos"].map(TOP_K)
    codes = df["wk"].to_numpy()
    idx = np.random.default_rng(seed).integers(0, n_weeks, size=(reps, n_weeks))
    rows = []
    for pos, edges in TIERS.items():
        d = df["pos"] == pos
        bounds = list(edges) + [np.inf]
        n_pos, n_top = int(d.sum()), int((d & df["top"]).sum())
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            m = (d & (df["salary"] >= lo) & (df["salary"] < hi)).to_numpy()
            if not m.any():
                continue
            h5, h5_lo, h5_hi = _cluster_rate(codes, n_weeks, m, (df["x"] >= VALUE_TARGET).to_numpy(float), idx)
            tp, tp_lo, tp_hi = _cluster_rate(codes, n_weeks, m, df["top"].to_numpy(float), idx)
            rows.append({"pos": pos, "tier": _tier_label(lo, hi), "lo_price": int(lo), "players": int(m.sum()), "pool_share": float(m.sum() / n_pos),
                         "pts": float(df.loc[m, "pts"].mean()), "per_k": float(df.loc[m, "x"].mean()),
                         "hit": h5, "hit_lo": h5_lo, "hit_hi": h5_hi, "top": tp, "top_lo": tp_lo, "top_hi": tp_hi,
                         "top_share": float((m & df["top"].to_numpy()).sum() / max(n_top, 1))})
    return rows


# ── Is the game already in the price? ───────────────────────────────────────────────────────────────────
RELEVANT_PRICE = {"QB": 5000, "RB": 4000, "WR": 4000, "TE": 3000, "DST": 0}      # players priced below this are mostly backups: their surprise says nothing about the offense
ENV_SPECS = (
    ("Implied team total and the defense faced", ("tt_z", "weak_fpa"), None),
    ("Same, with the EPA-based defense measure (2021 and 2026 weeks only)", ("tt_z", "weak_epa"), (2021, 2026)),
    ("Own offense and the opponent's points allowed", ("own_off_z", "pa_weak"), None),
)
ENV_LABELS = {"tt_z": "Implied team total (+1 sd)", "weak_fpa": "Weaker defense for the position (+1 sd)", "weak_epa": "Weaker defense by EPA (+1 sd)",
              "own_off_z": "Better own offense (+1 sd)", "pa_weak": "Opponent allows more points (+1 sd)"}


def add_surprise(pools: pd.DataFrame) -> pd.DataFrame:
    """Each player's points minus what his salary and position average in *the other seasons* (a quadratic fit with the season held out), the standardised environment columns, and
    only the players priced like starters. Imported here, not at module level, so `mm_study` can import this module."""
    from dfs import mm_study as S
    P = pools.copy()
    P["exp"] = np.nan
    for season in sorted(P["season"].unique()):
        others = [g for s, g in P.groupby("season") if s != season] or [P]
        m = (P["season"] == season).to_numpy()
        P.loc[m, "exp"] = S.salary_projection(P[m], S.fit_pricing(others))
    P["surprise"] = P["pts"] - P["exp"]
    P = P[[sal >= RELEVANT_PRICE[pos] for pos, sal in zip(P["pos"], P["salary"])]].copy()
    wk = P.groupby(["season", "week"])
    P["tt_z"] = (P["team_total"] - wk["team_total"].transform("mean")) / wk["team_total"].transform("std")
    P["weak_fpa"] = -(P["env_fpa_rank"] - 16.5) / 9.2                       # rank 1 is the weakest opponent, so the sign flips: + = a weaker defense
    P["weak_epa"] = -(P["env_epa_rank"] - 16.5) / 9.2
    P["own_off_z"] = -(P["team_off_rank"] - 16.5) / 9.2                      # + = a better own offense
    P["pa_weak"] = -(P["env_pa_rank"] - 16.5) / 9.2
    return P


def clustered_ols(d: pd.DataFrame, xs: list[str], y: str = "surprise") -> tuple[np.ndarray, np.ndarray, int]:
    """OLS of `y` on `xs` with standard errors clustered by week. Returns (coefficients with the intercept first, standard errors, rows used)."""
    d = d.dropna(subset=list(xs) + [y])
    X = np.column_stack([np.ones(len(d))] + [d[x].to_numpy(float) for x in xs])
    Y = d[y].to_numpy(float)
    inv = np.linalg.inv(X.T @ X)
    b = inv @ X.T @ Y
    e = Y - X @ b
    cl = pd.factorize(d["season"].to_numpy() * 100 + d["week"].to_numpy())[0]
    scores = np.zeros((cl.max() + 1, X.shape[1]))
    np.add.at(scores, cl, X * e[:, None])
    V = inv @ (scores.T @ scores) @ inv
    return b, np.sqrt(np.diag(V)), len(d)


def environment_effects(pools: pd.DataFrame) -> list[dict]:
    """One row per (specification, position): the points a player scored beyond his salary's expectation per standard deviation of each environment factor, with the t-statistic
    (standard errors clustered by week). A positive number means the price did not fully include that factor."""
    P = add_surprise(pools)
    rows = []
    for spec, xs, seasons in ENV_SPECS:
        D = P if seasons is None else P[P["season"].isin(seasons)]
        for pos in ("QB", "RB", "WR", "TE", "DST"):
            d = D[D["pos"] == pos]
            if len(d.dropna(subset=list(xs))) < 100:
                continue
            b, se, n = clustered_ols(d, list(xs))
            rows.append({"spec": spec, "pos": pos, "n": n, "terms": [{"name": x, "label": ENV_LABELS[x], "coef": float(b[i + 1]), "t": float(b[i + 1] / se[i + 1])} for i, x in enumerate(xs)]})
    return rows
