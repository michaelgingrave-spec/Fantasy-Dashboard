"""A small MILP layer for DraftKings Classic lineups: one model, two solvers.

`dfs/optimizer.py` builds each lineup with PuLP and the CBC executable, about half a second a solve. The Millionaire Maker study solves tens of thousands of lineups, so this module
describes the same contest rules (nine slots, $50,000, QB / 2-3 RB / 3-4 WR / 1-2 TE / DST with FLEX the extra RB / WR / TE) as sparse linear rows and solves them with HiGHS through
SciPy when SciPy is installed (milliseconds a solve), or with CBC through PuLP when it is not (the deployed app: SciPy is not in requirements.txt). Both solve the identical model;
`dfs/tests/test_lineup_milp.py` checks that the optimum matches the optimizer's.

The rules a Millionaire Maker lineup is built under live in `Rules`; every one of them is off by default, so `Rules()` is the plain DraftKings rule set.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from dfs.config import POSITION_LIMITS, ROSTER_SIZE, SALARY_CAP

try:  # SciPy is optional: the fast solver when present
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import csr_matrix, vstack as sp_vstack
    HAVE_SCIPY = True
except Exception:  # noqa: BLE001
    HAVE_SCIPY = False

OFFENSE = ("QB", "RB", "WR", "TE")
SKILL = ("RB", "WR", "TE")
GROUPS = {"QB": ("QB",), "RB": ("RB",), "WR": ("WR",), "TE": ("TE",), "DST": ("DST",), "SKILL": SKILL, "OFFENSE": OFFENSE, "ALL": OFFENSE + ("DST",)}


@dataclass(frozen=True)
class Rules:
    """Everything a lineup must satisfy beyond DraftKings' own roster rules. The defaults ask for nothing."""
    min_salary: int = 0
    max_from_team: int = 9               # QB + RB / WR / TE from one team (the defense is not counted)
    qb_stack: int = 0                    # at least this many of the QB's own pass catchers (stack_pos) in the lineup
    stack_pos: tuple[str, ...] = ("WR", "TE")
    bring_back: int = 0                  # at least this many pass catchers (bring_pos) from the QB's opponent
    bring_pos: tuple[str, ...] = ("WR", "TE")
    min_games: int = 0                   # players from at least this many different games
    max_per_game: int = 9                # at most this many players from one game
    min_teams: int = 0                   # players from at least this many different teams
    dst_vs: str = "any"                  # 'any' | 'qb' (defense must not face the QB) | 'lineup' (must not face any offensive player in the lineup)
    max_dst_salary: int | None = None
    max_qb_salary: int | None = None
    max_stars: int | None = None         # at most this many RB/WR/TE priced at `star_salary` or more
    star_salary: int = 8000
    max_te: int | None = None            # 1 keeps the FLEX for an RB or WR
    max_qb_total_rank: int | None = None  # the QB's game must be among the N highest game totals (needs a 'total_rank' column)
    min_top_games: int = 0               # at least this many players from the `top_games` highest-total games ...
    top_games: int = 0                   # ... (needs a 'total_rank' column)
    # -- how the cap is spent, and the game around each player -- all tuples so a Rules stays hashable; GROUP is QB | RB | WR | TE | DST | SKILL (RB/WR/TE) | OFFENSE | ALL
    spend: tuple = ()                    # ((group, lo, hi), ...): the group's players' salaries add up to between lo and hi
    counts: tuple = ()                   # ((pos, lo, hi), ...): between lo and hi players at that position (4 WRs = the FLEX is a WR)
    price_window: tuple = ()             # ((pos, lo, hi), ...): every player at that position in the lineup is priced between lo and hi
    price_counts: tuple = ()             # ((group, lo_price, hi_price, min_n, max_n), ...): how many of the group are priced in [lo_price, hi_price]
    env_bounds: tuple = ()               # ((group, column, lo, hi), ...): every player of the group has pool[column] in [lo, hi] (a missing value is outside)
    env_counts: tuple = ()               # ((group, column, lo, hi, min_n, max_n), ...): how many of the group have pool[column] in [lo, hi]
    own_max: float | None = None         # sum of the nine players' ownership % at most this (needs an 'own' column)
    own_min: float | None = None
    min_low_own: int = 0                 # at least this many players at or under `low_own` % ownership
    low_own: float = 5.0
    extra: tuple = field(default_factory=tuple)  # extra rows: ((dk_id, coefficient), ...), lo, hi) -- for experiments


class _Rows:
    """Sparse constraint rows (columns, coefficients, lo, hi) over `n` variables."""

    def __init__(self, n: int):
        self.n = n
        self.cols: list[np.ndarray] = []
        self.vals: list[np.ndarray] = []
        self.lo: list[float] = []
        self.hi: list[float] = []

    def add(self, cols, vals, lo: float = -math.inf, hi: float = math.inf) -> None:
        cols = np.asarray(cols, dtype=int)
        if np.isscalar(vals):
            vals = np.full(len(cols), float(vals))
        self.cols.append(cols)
        self.vals.append(np.asarray(vals, dtype=float))
        self.lo.append(lo)
        self.hi.append(hi)

    def matrix(self):
        indptr = np.cumsum([0] + [len(c) for c in self.cols])
        indices = np.concatenate(self.cols) if self.cols else np.array([], dtype=int)
        data = np.concatenate(self.vals) if self.vals else np.array([], dtype=float)
        return csr_matrix((data, indices, indptr), shape=(len(self.cols), self.n))


class PoolModel:
    """A slate's players as arrays, plus the model that every lineup of the slate must satisfy under one `Rules`."""

    def __init__(self, pool: pd.DataFrame, rules: Rules | None = None):
        pool = pool.reset_index(drop=True)
        self.pool = pool
        self.rules = rules or Rules()
        self.n_players = len(pool)
        self.ids = pool["dk_id"].astype(str).to_numpy()
        self.pos = pool["pos"].to_numpy()
        self.team = pool["team"].to_numpy()
        opp = pool["opp"] if "opp" in pool.columns else pd.Series([""] * len(pool))
        self.opp = opp.fillna("").astype(str).str.lstrip("@").to_numpy()
        self.salary = pool["salary"].to_numpy(dtype=float)
        if "game_id" in pool.columns and pool["game_id"].notna().all():
            self.game = pool["game_id"].astype(str).to_numpy()
        else:
            self.game = np.array(["-".join(sorted((t, o))) for t, o in zip(self.team, self.opp)])
        self._build()

    # -- model ---------------------------------------------------------------------------------------
    def _where(self, mask) -> np.ndarray:
        return np.flatnonzero(mask)

    def _build(self) -> None:
        r, n = self.rules, self.n_players
        pos, team, opp, game = self.pos, self.team, self.opp, self.game
        teams = sorted(set(team))
        games = sorted(set(game))
        self.n_aux = (len(games) if r.min_games > 0 else 0) + (len(teams) if r.min_teams > 0 else 0)
        self.n_vars = n + self.n_aux
        rows = _Rows(self.n_vars)
        allp = np.arange(n)
        rows.add(allp, 1.0, ROSTER_SIZE, ROSTER_SIZE)
        for p, (lo, hi) in POSITION_LIMITS.items():
            rows.add(self._where(pos == p), 1.0, lo, hi)
        rows.add(self._where(np.isin(pos, SKILL)), 1.0, 7, 7)
        rows.add(allp, self.salary, r.min_salary, SALARY_CAP)

        offense = np.isin(pos, OFFENSE)
        if r.max_from_team < 8:
            for t in teams:
                rows.add(self._where(offense & (team == t)), 1.0, -math.inf, r.max_from_team)
        qb_of = {t: self._where((pos == "QB") & (team == t)) for t in teams}
        if r.qb_stack > 0:
            for t in teams:
                if len(qb_of[t]) == 0:
                    continue
                mates = self._where(np.isin(pos, r.stack_pos) & (team == t))
                rows.add(np.concatenate([mates, qb_of[t]]), np.concatenate([np.ones(len(mates)), -float(r.qb_stack) * np.ones(len(qb_of[t]))]), 0.0, math.inf)
        if r.bring_back > 0:
            opp_of = {t: next((o for tt, o in zip(team, opp) if tt == t and o), "") for t in teams}
            for t in teams:
                o = opp_of[t]
                if len(qb_of[t]) == 0 or not o:
                    continue
                back = self._where(np.isin(pos, r.bring_pos) & (team == o))
                rows.add(np.concatenate([back, qb_of[t]]), np.concatenate([np.ones(len(back)), -float(r.bring_back) * np.ones(len(qb_of[t]))]), 0.0, math.inf)
        if r.dst_vs in ("qb", "lineup"):
            for d in self._where(pos == "DST"):
                o = opp[d]
                if not o:
                    continue
                if r.dst_vs == "qb":
                    against = self._where((pos == "QB") & (team == o))
                    rows.add(np.concatenate([[d], against]), 1.0, -math.inf, 1.0)
                else:
                    against = self._where(offense & (team == o))
                    rows.add(np.concatenate([[d], against]), np.concatenate([[8.0], np.ones(len(against))]), -math.inf, 8.0)
        if r.max_per_game < 9:
            for g in games:
                rows.add(self._where(game == g), 1.0, -math.inf, r.max_per_game)
        col = n
        if r.min_games > 0:
            gcols = []
            for g in games:
                members = self._where(game == g)
                rows.add(np.concatenate([[col], members]), np.concatenate([[1.0], -np.ones(len(members))]), -math.inf, 0.0)   # u_g <= sum of its players
                gcols.append(col)
                col += 1
            rows.add(gcols, 1.0, r.min_games, math.inf)
        if r.min_teams > 0:
            tcols = []
            for t in teams:
                members = self._where(team == t)
                rows.add(np.concatenate([[col], members]), np.concatenate([[1.0], -np.ones(len(members))]), -math.inf, 0.0)
                tcols.append(col)
                col += 1
            rows.add(tcols, 1.0, r.min_teams, math.inf)
        if r.max_stars is not None:
            rows.add(self._where(np.isin(pos, SKILL) & (self.salary >= r.star_salary)), 1.0, -math.inf, r.max_stars)
        if r.max_te is not None:
            rows.add(self._where(pos == "TE"), 1.0, -math.inf, r.max_te)
        for group, lo, hi in r.spend:
            members = self._where(self._group(group))
            rows.add(members, self.salary[members], lo, hi)
        for p, lo, hi in r.counts:
            rows.add(self._where(pos == p), 1.0, lo, hi)
        for group, lo_p, hi_p, min_n, max_n in r.price_counts:
            rows.add(self._where(self._group(group) & (self.salary >= lo_p) & (self.salary <= hi_p)), 1.0, min_n, max_n)
        for group, name, lo, hi, min_n, max_n in r.env_counts:
            v = self._col(name)
            rows.add(self._where(self._group(group) & (v >= lo) & (v <= hi)), 1.0, min_n, max_n)
        if r.min_top_games > 0 and r.top_games > 0:
            top = self._where(self._col("total_rank") <= r.top_games)
            rows.add(top, 1.0, r.min_top_games, math.inf)
        if r.own_max is not None or r.own_min is not None:
            own = self._col("own")
            rows.add(allp, own, -math.inf if r.own_min is None else r.own_min, math.inf if r.own_max is None else r.own_max)
        if r.min_low_own > 0:
            rows.add(self._where(self._col("own") <= r.low_own), 1.0, r.min_low_own, math.inf)
        for terms, lo, hi in r.extra:
            idx = {d: k for k, d in enumerate(self.ids)}
            rows.add([idx[d] for d, _ in terms], [c for _, c in terms], lo, hi)
        self.rows = rows
        self.lb = np.zeros(self.n_vars)
        self.ub = np.ones(self.n_vars)
        if r.max_dst_salary is not None:
            self.ub[self._where((pos == "DST") & (self.salary > r.max_dst_salary))] = 0
        if r.max_qb_salary is not None:
            self.ub[self._where((pos == "QB") & (self.salary > r.max_qb_salary))] = 0
        if r.max_qb_total_rank is not None:
            self.ub[self._where((pos == "QB") & (self._col("total_rank") > r.max_qb_total_rank))] = 0
        for p, lo, hi in r.price_window:
            self.ub[self._where((pos == p) & ~((self.salary >= lo) & (self.salary <= hi)))] = 0
        for group, name, lo, hi in r.env_bounds:
            v = self._col(name)
            self.ub[self._where(self._group(group) & ~((v >= lo) & (v <= hi)))] = 0
        self.A = rows.matrix() if HAVE_SCIPY else None

    def _group(self, group: str) -> np.ndarray:
        return np.isin(self.pos, GROUPS[group])

    def _col(self, name: str) -> np.ndarray:
        if name not in self.pool.columns:
            raise KeyError(f"the pool has no '{name}' column, which this rule needs")
        return self.pool[name].to_numpy(dtype=float)

    # -- solving -------------------------------------------------------------------------------------
    def solve(self, obj: np.ndarray, previous: list[np.ndarray] | None = None, min_uniques: int = 1, lb: np.ndarray | None = None,
              ub: np.ndarray | None = None, time_limit: float = 30.0) -> np.ndarray | None:
        """Indices of the nine players with the highest total `obj` under the rules, or None when there is none. `previous` lineups (index arrays) must each differ by at
        least `min_uniques` players."""
        lb = self.lb if lb is None else lb
        ub = self.ub if ub is None else ub
        extra = [(np.asarray(p, dtype=int), ROSTER_SIZE - max(1, min_uniques)) for p in (previous or [])]
        c = np.zeros(self.n_vars)
        c[: self.n_players] = obj
        if HAVE_SCIPY:
            return self._solve_scipy(c, lb, ub, extra, time_limit)
        return self._solve_pulp(c, lb, ub, extra)

    def _solve_scipy(self, c, lb, ub, extra, time_limit):
        A, lo, hi = self.A, np.array(self.rows.lo), np.array(self.rows.hi)
        if extra:
            ur = _Rows(self.n_vars)
            for cols, cap in extra:
                ur.add(cols, 1.0, -math.inf, cap)
            A = sp_vstack([A, ur.matrix()], format="csr")
            lo = np.concatenate([lo, ur.lo])
            hi = np.concatenate([hi, ur.hi])
        res = milp(-c, constraints=LinearConstraint(A, lo, hi), integrality=np.ones(self.n_vars), bounds=Bounds(lb, ub),
                   options={"disp": False, "time_limit": time_limit, "mip_rel_gap": 1e-9})
        if res.x is None or res.status not in (0, 1):
            return None
        return np.flatnonzero(np.round(res.x[: self.n_players]) == 1)

    def _solve_pulp(self, c, lb, ub, extra):
        import pulp
        prob = pulp.LpProblem("lineup", pulp.LpMaximize)
        x = [pulp.LpVariable(f"x{j}", lowBound=float(lb[j]), upBound=float(ub[j]), cat="Binary") for j in range(self.n_vars)]
        prob += pulp.lpSum(float(c[j]) * x[j] for j in range(self.n_players))
        for cols, vals, lo, hi in zip(self.rows.cols, self.rows.vals, self.rows.lo, self.rows.hi):
            expr = pulp.lpSum(float(v) * x[int(j)] for j, v in zip(cols, vals))
            if lo == hi:
                prob += expr == lo
                continue
            if lo > -math.inf:
                prob += expr >= lo
            if hi < math.inf:
                prob += expr <= hi
        for cols, cap in extra:
            prob += pulp.lpSum(x[int(j)] for j in cols) <= cap
        status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
        if pulp.LpStatus[status] != "Optimal":
            return None
        return np.array([j for j in range(self.n_players) if x[j].value() is not None and round(x[j].value()) == 1])


def solve_portfolio(model: PoolModel, objective, n: int = 1, min_uniques: int = 1, max_exposure: float = 1.0, locks: list[str] | None = None,
                    excludes: list[str] | None = None, time_limit: float = 30.0) -> list[np.ndarray]:
    """Up to `n` lineups, each an index array into `model.pool`, in the order they were built. `objective` is a vector of one value per player, or a function of the lineup number
    that returns one (so every lineup can carry its own random jitter). Each lineup differs from every earlier one by at least `min_uniques` players, and no player is used in more
    than `max_exposure` of the `n` lineups (locked players are exempt). Stops early when the rules leave no further lineup."""
    lock_ix = [int(np.flatnonzero(model.ids == str(d))[0]) for d in (locks or []) if str(d) in set(model.ids)]
    excl_ix = [int(np.flatnonzero(model.ids == str(d))[0]) for d in (excludes or []) if str(d) in set(model.ids)]
    cap = max(1, math.ceil(max_exposure * n))
    counts = np.zeros(model.n_players, dtype=int)
    out: list[np.ndarray] = []
    for k in range(n):
        obj = objective(k) if callable(objective) else np.asarray(objective, dtype=float)
        lb, ub = model.lb.copy(), model.ub.copy()
        lb[lock_ix] = 1
        ub[excl_ix] = 0
        capped = np.flatnonzero(counts >= cap)
        ub[[j for j in capped if j not in set(lock_ix)]] = 0
        sol = model.solve(obj, previous=out, min_uniques=min_uniques, lb=lb, ub=ub, time_limit=time_limit)
        if sol is None or len(sol) != ROSTER_SIZE:
            break
        out.append(sol)
        counts[sol] += 1
    return out


def best_lineups_by(model: PoolModel, score: np.ndarray, k: int = 1, min_uniques: int = 1) -> list[tuple[np.ndarray, float]]:
    """The `k` highest-`score` lineups the model allows, each at least `min_uniques` players apart: (indices, total score)."""
    sols = solve_portfolio(model, np.asarray(score, dtype=float), n=k, min_uniques=min_uniques)
    return [(s, float(np.asarray(score)[s].sum())) for s in sols]
