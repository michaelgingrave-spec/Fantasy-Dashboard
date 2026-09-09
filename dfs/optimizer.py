"""DraftKings NFL Classic lineup optimizer (MILP, via PuLP/CBC).

Supports: player locks (force-in) and excludes, min-salary floor, per-team caps,
QB stacking + optional bring-back, DST-vs-QB avoidance, and generating N diverse
lineups with a per-player exposure cap.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pulp

# PuLP 3.3 emits a DeprecationWarning per LpVariable / PULP_CBC_CMD call (thousands when
# generating many lineups). The bundled-CBC path is still the supported one for now.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pulp")

from dfs.config import (
    FLEX_POSITIONS,
    POSITION_LIMITS,
    ROSTER_SIZE,
    SALARY_CAP,
    OptimizerConfig,
)

SLOT_ORDER = ["QB", "RB1", "RB2", "WR1", "WR2", "WR3", "TE", "FLEX", "DST"]


class OptimizerError(RuntimeError):
    pass


@dataclass
class Lineup:
    players: pd.DataFrame            # 9 rows, has a 'slot' column in SLOT_ORDER order
    salary: int
    proj: float
    ceiling: float
    stack: str

    def dk_upload_cells(self) -> list[str]:
        """9 cells in DK template order: QB,RB,RB,WR,WR,WR,TE,FLEX,DST."""
        by_slot = {r["slot"]: f'{r["name"]} ({r["dk_id"]})' for _, r in self.players.iterrows()}
        return [by_slot[s] for s in SLOT_ORDER]


def _opp_map(pool: pd.DataFrame) -> dict[str, str]:
    """team -> opponent team abbrev (strip any '@')."""
    m: dict[str, str] = {}
    for _, r in pool.iterrows():
        o = str(r.get("opp") or "").lstrip("@")
        if r["team"] and o:
            m[r["team"]] = o
    return m


def _check_locks(pool: pd.DataFrame, locks: list[str]) -> None:
    locked = pool[pool["dk_id"].isin(locks)]
    missing = set(locks) - set(locked["dk_id"])
    if missing:
        raise OptimizerError(f"{len(missing)} locked player id(s) not on this slate: {sorted(missing)}")
    if len(locked) > ROSTER_SIZE:
        raise OptimizerError(f"{len(locked)} players locked — max is {ROSTER_SIZE}.")
    counts = locked["pos"].value_counts().to_dict()
    for pos, (_, hi) in POSITION_LIMITS.items():
        if counts.get(pos, 0) > hi:
            raise OptimizerError(f"{counts[pos]} {pos} locked — max {hi} in a Classic lineup.")
    flex_ct = sum(counts.get(p, 0) for p in FLEX_POSITIONS)
    if flex_ct > 7:
        raise OptimizerError(f"{flex_ct} RB/WR/TE locked — max 7 (incl. FLEX).")
    if int(locked["salary"].sum()) > SALARY_CAP:
        raise OptimizerError(
            f"Locked players cost ${int(locked['salary'].sum()):,} > ${SALARY_CAP:,} cap."
        )


def optimize(
    slate: pd.DataFrame,
    cfg: OptimizerConfig | None = None,
    boosts: dict[str, float] | None = None,
) -> list[Lineup]:
    cfg = cfg or OptimizerConfig()
    if slate.empty:
        raise OptimizerError("Slate is empty.")

    locks = [str(x) for x in (cfg.locks or [])]
    excludes = set(str(x) for x in (cfg.excludes or []))

    pool = slate[~slate["dk_id"].isin(excludes)].copy()
    pool = pool[pool.get("playable", True) | pool["dk_id"].isin(locks)].copy()
    pool = pool.drop_duplicates(subset="dk_id").reset_index(drop=True)
    for need in ("dk_id", "name", "pos", "team", "salary", "proj"):
        if need not in pool.columns:
            raise OptimizerError(f"Slate missing required column '{need}'.")
    if "ceiling" not in pool.columns:
        pool["ceiling"] = pool["proj"]
    if "opp" not in pool.columns:
        pool["opp"] = ""

    _check_locks(pool, locks)

    ids = pool["dk_id"].tolist()
    P = pool.set_index("dk_id")
    lean = max(0.0, min(1.0, cfg.lean))
    base_obj = P["proj"] + lean * (P["ceiling"] - P["proj"])
    if cfg.apply_matchup_boost and boosts:
        mult = pd.Series({i: float(boosts.get(i, 1.0)) for i in ids})
        base_obj = base_obj * mult

    opp = _opp_map(pool)
    teams = sorted(t for t in P["team"].unique() if t)
    qb_ids = [i for i in ids if P.at[i, "pos"] == "QB"]
    dst_ids = [i for i in ids if P.at[i, "pos"] == "DST"]
    pc_ids = [i for i in ids if P.at[i, "pos"] in ("WR", "TE")]  # pass catchers for stacks
    skill_ids = [i for i in ids if P.at[i, "pos"] in FLEX_POSITIONS]

    lineups: list[Lineup] = []
    used_sets: list[set[str]] = []
    exposure: dict[str, int] = {i: 0 for i in ids}
    max_count = max(1, math.ceil(cfg.max_exposure * cfg.n_lineups))
    banned: set[str] = set()
    variance = max(0.0, min(1.0, cfg.variance))
    rng = np.random.default_rng(cfg.seed)

    for n in range(cfg.n_lineups):
        prob = pulp.LpProblem(f"dk_classic_{n}", pulp.LpMaximize)
        x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in ids}

        # Variance: jitter each player's objective weight so the solver explores different
        # near-optimal player mixes. Reported lineup projections stay the true numbers.
        if variance > 0:
            noise = rng.uniform(1.0 - variance, 1.0 + variance, size=len(ids))
            obj = {i: float(base_obj[i]) * noise[k] for k, i in enumerate(ids)}
        else:
            obj = {i: float(base_obj[i]) for i in ids}
        prob += pulp.lpSum(obj[i] * x[i] for i in ids)

        prob += pulp.lpSum(x.values()) == ROSTER_SIZE
        for pos, (lo, hi) in POSITION_LIMITS.items():
            sel = [x[i] for i in ids if P.at[i, "pos"] == pos]
            prob += pulp.lpSum(sel) >= lo
            prob += pulp.lpSum(sel) <= hi
        prob += pulp.lpSum(x[i] for i in skill_ids) == 7  # RB+WR+TE incl. FLEX

        prob += pulp.lpSum(P.at[i, "salary"] * x[i] for i in ids) <= SALARY_CAP
        prob += pulp.lpSum(P.at[i, "salary"] * x[i] for i in ids) >= cfg.min_salary

        for i in locks:
            prob += x[i] == 1
        for i in banned:
            if i not in locks:
                prob += x[i] == 0

        for t in teams:
            skill_t = [x[i] for i in skill_ids if P.at[i, "team"] == t]
            if skill_t:
                prob += pulp.lpSum(skill_t) <= cfg.max_from_team

        if cfg.stack_qb and cfg.stack_min > 0:
            for t in teams:
                qb_t = [x[i] for i in qb_ids if P.at[i, "team"] == t]
                pc_t = [x[i] for i in pc_ids if P.at[i, "team"] == t]
                if qb_t:
                    prob += pulp.lpSum(pc_t) >= cfg.stack_min * pulp.lpSum(qb_t)

        if cfg.bring_back:
            for t in teams:
                o = opp.get(t)
                qb_t = [x[i] for i in qb_ids if P.at[i, "team"] == t]
                if not qb_t or not o:
                    continue
                pc_o = [x[i] for i in pc_ids if P.at[i, "team"] == o]
                prob += pulp.lpSum(pc_o) >= pulp.lpSum(qb_t)

        if cfg.avoid_dst_vs_qb:
            for d in dst_ids:
                dt = P.at[d, "team"]
                o = opp.get(dt)
                if not o:
                    continue
                opp_qb = [x[i] for i in qb_ids if P.at[i, "team"] == o]
                if opp_qb:
                    prob += x[d] + pulp.lpSum(opp_qb) <= 1

        for s in used_sets:
            prob += pulp.lpSum(x[i] for i in s if i in x) <= ROSTER_SIZE - cfg.min_uniques

        status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
        if pulp.LpStatus[status] != "Optimal":
            if not lineups:
                raise OptimizerError(
                    "No valid lineup under these constraints "
                    "(check locks, min-salary, and stacking options)."
                )
            break

        chosen = [i for i in ids if x[i].value() and round(x[i].value()) == 1]
        used_sets.append(set(chosen))
        for i in chosen:
            exposure[i] += 1
            if exposure[i] >= max_count and i not in locks:
                banned.add(i)
        lineups.append(_build_lineup(pool.set_index("dk_id").loc[chosen].reset_index(), opp))

    return lineups


def _build_lineup(rows: pd.DataFrame, opp: dict[str, str]) -> Lineup:
    rows = rows.copy()
    rows["proj"] = rows["proj"].astype(float)
    slots: dict[str, dict] = {}
    used = set()

    def take(pos: str, slot: str):
        cand = rows[(rows["pos"] == pos) & (~rows["dk_id"].isin(used))]
        cand = cand.sort_values("proj", ascending=False)
        r = cand.iloc[0]
        used.add(r["dk_id"])
        slots[slot] = r.to_dict()

    take("QB", "QB")
    take("RB", "RB1"); take("RB", "RB2")
    take("WR", "WR1"); take("WR", "WR2"); take("WR", "WR3")
    take("TE", "TE")
    flex = rows[(~rows["dk_id"].isin(used)) & (rows["pos"].isin(FLEX_POSITIONS))]
    fr = flex.sort_values("proj", ascending=False).iloc[0]
    used.add(fr["dk_id"]); slots["FLEX"] = fr.to_dict()
    take("DST", "DST")

    ordered = pd.DataFrame([{**slots[s], "slot": s} for s in SLOT_ORDER])
    qb = ordered[ordered["slot"] == "QB"].iloc[0]
    mates = ordered[(ordered["team"] == qb["team"]) & (ordered["pos"].isin(("WR", "TE")))]
    stack = f"{qb['team']} {qb['name'].split()[-1]}"
    if len(mates):
        stack += " + " + ", ".join(m["name"].split()[-1] for _, m in mates.iterrows())
    o = opp.get(qb["team"])
    if o is not None:
        bring = ordered[(ordered["team"] == o) & (ordered["pos"].isin(("WR", "TE")))]
        if len(bring):
            stack += " / back " + ", ".join(m["name"].split()[-1] for _, m in bring.iterrows())

    return Lineup(
        players=ordered[["slot", "name", "pos", "team", "opp", "salary", "proj", "dk_id"]],
        salary=int(ordered["salary"].sum()),
        proj=round(float(ordered["proj"].sum()), 2),
        ceiling=round(float(ordered["ceiling"].sum()), 2) if "ceiling" in ordered else 0.0,
        stack=stack,
    )


DK_UPLOAD_HEADER = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DST"]


def to_dk_upload(lineups: list[Lineup]) -> pd.DataFrame:
    """DataFrame with DraftKings' bulk-import layout (duplicate RB/WR headers preserved)."""
    return pd.DataFrame(
        [lu.dk_upload_cells() for lu in lineups], columns=DK_UPLOAD_HEADER
    )
