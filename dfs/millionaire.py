"""Millionaire Maker lineup builder.

DraftKings' NFL Millionaire Maker pays its top prize to one lineup out of well over a hundred thousand, so what matters is the upper tail of a lineup's score, not its average.
`dfs.hindsight` finds the best lineup of every stored week and what those lineups have in common; `dfs.mm_study` asks which of those traits still pay when lineups are built
*before* kickoff (the rules that held up are the defaults in `MMConfig`). This module turns them into a builder: the same projections the Optimizer uses, a ceiling lean and a
random jitter so the lineups differ, and the rules enforced as hard constraints by `dfs.lineup_milp`.

    lineups = build_portfolio(slate, MMConfig(n_lineups=20))
    profile(lineups[0], slate)                    # stack text, teams, games, salary left, ownership
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from dfs.lineup_milp import PoolModel, Rules, solve_portfolio
from dfs.optimizer import Lineup, OptimizerError, _build_lineup


@dataclass
class MMConfig:
    n_lineups: int = 20
    # structure: the rules the study supports (see dfs/mm_report.md)
    qb_stack: int = 2                    # the QB's own WR/TE in the lineup (0 = no requirement)
    bring_back: int = 1                  # pass catchers from the QB's opponent (0 = no requirement)
    max_from_team: int = 3               # QB + RB/WR/TE from one team
    min_games: int = 0                   # players from at least this many games (0 = no requirement)
    min_teams: int = 0
    dst_vs: str = "lineup"               # 'any' | 'qb' | 'lineup': no offensive player in the lineup may face its own defense
    min_salary: int = 49000
    # objective
    lean: float = 0.5                    # 0 = projection, 1 = ceiling
    jitter: float = 0.25                 # +- this fraction of a player's value, drawn fresh for every lineup
    # portfolio
    max_exposure: float = 0.4            # no player in more than this share of the lineups (locks excepted)
    min_uniques: int = 3                 # every lineup differs from every other by at least this many players
    seed: int | None = None
    locks: list[str] = field(default_factory=list)
    excludes: list[str] = field(default_factory=list)
    # ownership (the heuristic estimate or a file the user supplies): both off by default
    own_max: float | None = None         # combined ownership % of the nine players at most this
    min_low_own: int = 0                 # at least this many players at or under `low_own` %
    low_own: float = 5.0

    def rules(self) -> Rules:
        return Rules(min_salary=self.min_salary, max_from_team=self.max_from_team, qb_stack=self.qb_stack, bring_back=self.bring_back, min_games=self.min_games,
                     min_teams=self.min_teams, dst_vs=self.dst_vs, own_max=self.own_max, min_low_own=self.min_low_own, low_own=self.low_own)


def _prepare(slate: pd.DataFrame, cfg: MMConfig) -> pd.DataFrame:
    pool = slate.copy()
    if "playable" in pool.columns:                                      # a player the news has ruled out stays out unless the user locked him
        pool = pool[pool["playable"].astype(bool) | pool["dk_id"].isin(cfg.locks)]
    pool = pool.drop_duplicates(subset="dk_id").reset_index(drop=True)
    for need in ("dk_id", "name", "pos", "team", "salary", "proj"):
        if need not in pool.columns:
            raise OptimizerError(f"Slate missing required column '{need}'.")
    if "ceiling" not in pool.columns:
        pool["ceiling"] = pool["proj"]
    if "opp" not in pool.columns:
        pool["opp"] = ""
    pool["opp"] = pool["opp"].fillna("").astype(str)
    if "own_pct" in pool.columns:
        pool["own"] = pool["own_pct"].astype(float)
    elif cfg.own_max is not None or cfg.min_low_own > 0:
        raise OptimizerError("The ownership rules need an ownership estimate on the slate (an 'own_pct' column).")
    return pool


def build_portfolio(slate: pd.DataFrame, cfg: MMConfig | None = None) -> list[Lineup]:
    """Up to `cfg.n_lineups` lineups under the Millionaire Maker rules, best projection first. Raises OptimizerError when the rules leave no lineup at all."""
    cfg = cfg or MMConfig()
    if slate.empty:
        raise OptimizerError("Slate is empty.")
    pool = _prepare(slate, cfg)
    base = (pool["proj"] + float(np.clip(cfg.lean, 0.0, 1.0)) * (pool["ceiling"] - pool["proj"])).to_numpy(dtype=float)
    rng = np.random.default_rng(cfg.seed)
    jit = float(np.clip(cfg.jitter, 0.0, 0.9))

    def objective(_k: int) -> np.ndarray:
        return base * (1.0 + rng.uniform(-jit, jit, size=len(base))) if jit > 0 else base

    model = PoolModel(pool, cfg.rules())
    sets = solve_portfolio(model, objective, n=cfg.n_lineups, min_uniques=cfg.min_uniques, max_exposure=cfg.max_exposure, locks=cfg.locks, excludes=cfg.excludes)
    if not sets:
        raise OptimizerError("No valid lineup under these rules (check the locks, the salary floor and the stacking rules).")
    opp = {t: o.lstrip("@") for t, o in zip(pool["team"], pool["opp"]) if t and o}
    lineups = [_build_lineup(pool.iloc[ix].copy(), opp) for ix in sets]
    lineups.sort(key=lambda lu: -lu.proj)
    return lineups


def profile(lu: Lineup, slate: pd.DataFrame | None = None) -> dict:
    """What a lineup looks like to a Millionaire Maker builder: its stack and bring-back, how many teams and games it spans, the salary left, the combined ownership."""
    p = lu.players
    qb = p[p["pos"] == "QB"].iloc[0]
    mates = p[(p["team"] == qb["team"]) & p["pos"].isin(["WR", "TE"])]
    opp = str(qb.get("opp") or "").lstrip("@")
    back = p[(p["team"] == opp) & p["pos"].isin(["WR", "TE"])] if opp else p.iloc[0:0]
    games = {"-".join(sorted((t, str(o).lstrip("@")))) for t, o in zip(p["team"], p["opp"])}
    out = {"stack": len(mates), "back": len(back), "teams": int(p["team"].nunique()), "games": len(games), "salary_left": 50000 - int(p["salary"].sum()),
           "max_from_team": int(p[p["pos"] != "DST"]["team"].value_counts().max())}
    if slate is not None and "own_pct" in slate.columns:
        own = slate.set_index("dk_id")["own_pct"]
        ids = p["dk_id"].astype(str)
        out["own_sum"] = float(own.reindex(ids).sum()) if own.reindex(ids).notna().all() else float("nan")
    return out


def portfolio_summary(lineups: list[Lineup], slate: pd.DataFrame | None = None) -> dict:
    """Portfolio-level facts: counts, the share of lineups with a stack / bring-back, projection range, QB mix."""
    prof = [profile(lu, slate) for lu in lineups]
    n = len(lineups)
    qbs = pd.Series([lu.players[lu.players["pos"] == "QB"].iloc[0]["name"] for lu in lineups]).value_counts()
    return {
        "lineups": n,
        "unique_players": len({d for lu in lineups for d in lu.players["dk_id"]}),
        "with_stack": sum(p["stack"] >= 1 for p in prof) / n, "with_back": sum(p["back"] >= 1 for p in prof) / n,
        "proj_min": min(lu.proj for lu in lineups), "proj_max": max(lu.proj for lu in lineups), "proj_mean": float(np.mean([lu.proj for lu in lineups])),
        "mean_teams": float(np.mean([p["teams"] for p in prof])), "mean_games": float(np.mean([p["games"] for p in prof])),
        "qbs": qbs.to_dict(),
    }


def read_ownership_file(df: pd.DataFrame, slate: pd.DataFrame) -> tuple[pd.Series, list[str]]:
    """Match a user-supplied ownership table (a name column and a percent column, any headers that look right; a team column makes the match name + team) to the slate.
    Returns (own % by dk_id, names of projected players it could not place). A spelling that is not exact is accepted only when exactly one name is that close."""
    from difflib import get_close_matches

    from dfs.names import norm_team, normalize_name
    cols = {c.lower().strip(): c for c in df.columns}
    name_col = next((cols[c] for c in cols if c in ("name", "player", "player name", "playername")), None)
    own_col = next((cols[c] for c in cols if c in ("own", "own%", "ownership", "ownership%", "pown", "pown%", "projown", "proj own", "projected ownership", "ownership %", "own %")), None)
    team_col = next((cols[c] for c in cols if c in ("team", "tm")), None)
    if name_col is None or own_col is None:
        raise ValueError(f"Could not find a name column and an ownership column in {list(df.columns)}")
    vals = pd.to_numeric(df[own_col].astype(str).str.replace("%", "", regex=False), errors="coerce")

    def key(name, team=None) -> str:
        return normalize_name(str(name)) + (f"|{norm_team(team)}" if team is not None else "")

    teams = df[team_col] if team_col else [None] * len(df)
    by_key = {key(n, t): float(v) for n, t, v in zip(df[name_col], teams, vals) if pd.notna(v)}
    keys = list(by_key)
    out = {}
    for r in slate.itertuples():
        k = key(r.name, r.team if team_col else None)
        if k not in by_key:
            close = get_close_matches(k, keys, n=2, cutoff=0.93)
            k = close[0] if len(close) == 1 else None
        if k is not None:
            out[r.dk_id] = by_key[k]
    missing = [r.name for r in slate.itertuples() if r.dk_id not in out and r.proj >= 8]
    return pd.Series(out, dtype=float), missing
