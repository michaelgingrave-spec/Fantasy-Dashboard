"""Diagnostics for the game simulator against past games: is it calibrated, and do its outcomes move together the way real ones did?

    python -m matchup_model.sim.calibrate 2023 2025        # simulate every cached game and print the report (build the cache first: matchup_model.sim.history)

Three checks, all on games the simulator is not told the result of:
  1. TEAM level: points, margin, total, pass / rush attempts and yards -- where do the actual numbers fall in the simulated distribution (probability integral transform,
     PIT; uniform if calibrated)?
  2. PLAYER level: the same for each player's box score, by role (QB, RB1, WR1, ...). Discrete stats use a randomised PIT.
  3. DEPENDENCE: the correlation of the PITs of two players' stats across real games, against the correlation the simulator itself implies for them (the same randomised-PIT
     transform on its own draws). This is what a parlay price depends on.
"""
from __future__ import annotations

import sys
from collections import defaultdict

import numpy as np
import pandas as pd

from dfs.names import normalize_name
from matchup_model.sim import engine as E
from matchup_model.sim import inputs as I

ROLE_STATS = {"QB": ("pass_yds", "pass_td", "comp", "pass_att", "int", "rush_yds"),
              "RB1": ("rush_yds", "rush_att", "rec", "rec_yds", "any_td"), "RB2": ("rush_yds", "rush_att", "rec", "rec_yds", "any_td"),
              "WR1": ("rec", "rec_yds", "tgt", "any_td"), "WR2": ("rec", "rec_yds", "tgt", "any_td"), "WR3": ("rec", "rec_yds", "tgt", "any_td"),
              "TE1": ("rec", "rec_yds", "tgt", "any_td")}
TEAM_STATS = ("points", "pass_att", "rush_att", "pass_yds", "rush_yds", "comp", "pass_td", "rush_td")
# (role.stat, role.stat, relation): 'same' = the same team, 'opp' = the first on the home team and the second on the away team
PAIRS = [("QB.pass_yds", "WR1.rec_yds", "same"), ("QB.pass_yds", "WR2.rec_yds", "same"), ("QB.pass_yds", "WR3.rec_yds", "same"), ("QB.pass_yds", "TE1.rec_yds", "same"),
         ("QB.pass_yds", "RB1.rec_yds", "same"), ("QB.pass_td", "WR1.any_td", "same"), ("QB.pass_td", "WR1.rec_yds", "same"), ("QB.pass_yds", "QB.pass_td", "same"),
         ("QB.comp", "WR1.rec", "same"), ("WR1.rec_yds", "WR2.rec_yds", "same"), ("WR1.rec_yds", "TE1.rec_yds", "same"), ("WR2.rec_yds", "WR3.rec_yds", "same"),
         ("WR1.rec", "WR2.rec", "same"), ("WR1.tgt", "WR2.tgt", "same"), ("WR1.any_td", "WR2.any_td", "same"), ("RB1.rush_yds", "QB.pass_yds", "same"),
         ("RB1.rush_yds", "WR1.rec_yds", "same"), ("RB1.rush_yds", "RB2.rush_yds", "same"), ("RB1.rush_yds", "RB1.rec_yds", "same"), ("RB1.rush_att", "QB.pass_att", "same"),
         ("RB1.any_td", "QB.pass_td", "same"), ("RB1.any_td", "WR1.any_td", "same"),
         ("QB.pass_yds", "QB.pass_yds", "opp"), ("WR1.rec_yds", "WR1.rec_yds", "opp"), ("RB1.rush_yds", "RB1.rush_yds", "opp"), ("QB.pass_yds", "WR1.rec_yds", "opp"),
         ("RB1.rush_yds", "QB.pass_yds", "opp"), ("WR1.any_td", "WR1.any_td", "opp")]
TEAM_PAIRS = [("points", "QB.pass_yds"), ("points", "WR1.rec_yds"), ("points", "RB1.rush_yds"), ("points", "QB.pass_td"), ("margin", "RB1.rush_yds"), ("margin", "QB.pass_yds"),
              ("total", "QB.pass_yds"), ("total", "WR1.rec_yds"), ("total", "RB1.rush_yds")]


def roles(spec: E.TeamSpec) -> dict[str, str]:
    """role -> player name for one team, by projected usage."""
    out: dict[str, str] = {}
    qbs = [p for p in spec.players if p.pos == "QB" and p.pass_att > 0]
    if qbs:
        out["QB"] = max(qbs, key=lambda p: p.pass_att).name
    for i, p in enumerate(sorted([p for p in spec.players if p.pos == "RB"], key=lambda p: -(p.rush_att + p.rec))[:2]):
        out[f"RB{i + 1}"] = p.name
    for i, p in enumerate(sorted([p for p in spec.players if p.pos == "WR"], key=lambda p: -p.tgt)[:3]):
        out[f"WR{i + 1}"] = p.name
    for i, p in enumerate(sorted([p for p in spec.players if p.pos == "TE"], key=lambda p: -p.tgt)[:1]):
        out[f"TE{i + 1}"] = p.name
    return out


def pit_of(draws: np.ndarray, actual: float, u: float) -> float:
    d = np.asarray(draws, float)
    return float(((d < actual).sum() + u * (d == actual).sum()) / len(d))


def pit_transform(x: np.ndarray, rng) -> np.ndarray:
    """The simulator's own draws on the uniform scale (randomised within ties), for correlating with the PITs of real outcomes."""
    s = np.sort(x)
    lt = np.searchsorted(s, x, side="left")
    le = np.searchsorted(s, x, side="right")
    return (lt + rng.random(len(x)) * (le - lt)) / len(x)


def _actual_stat(row: pd.Series, stat: str) -> float:
    if stat == "any_td":
        return float(row["rush_td"] + row["rec_td"] > 0)
    return float(row[stat])


def diagnose(recs: list[dict], params: E.SimParams | None = None, n: int = 3000, seed: int = 0, progress: bool = False) -> dict:
    """Simulate each record once and return {'player': DataFrame of PITs, 'team': DataFrame, 'pairs': DataFrame (empirical vs simulated), 'tpairs': DataFrame}."""
    rng = np.random.default_rng(seed + 991)
    prow, trow = [], []
    sim_corr: dict = defaultdict(list)
    tsim_corr: dict = defaultdict(list)
    emp: dict = defaultdict(lambda: ([], []))
    temp: dict = defaultdict(lambda: ([], []))
    for gi, rec in enumerate(recs):
        info = rec["info"]
        try:
            h, a = I.specs_from_table(rec["table"], rec["away"], rec["home"], info["away_implied"], info["home_implied"])
            res = E.simulate(h, a, n=n, seed=seed + gi, params=params)
        except Exception as exc:  # noqa: BLE001
            print("  skipped", rec["season"], rec["week"], rec["away"], rec["home"], exc, flush=True)
            continue
        act = rec["actual"].set_index("nk")
        sc = {rec["home"]: info["home_score"], rec["away"]: info["away_score"]}
        pit_cols: dict = {}                       # (side, 'role.stat') -> (pit of actual, pit-transformed draws)
        for side, spec in (("H", h), ("A", a)):
            tm = spec.abbr
            rl = roles(spec)
            ta = rec["team_actual"].get(tm, {})
            d = res.team[tm]
            for stat in TEAM_STATS:
                actual = sc[tm] if stat == "points" else ta.get(stat if stat != "comp" else "comp")
                if actual is None or (isinstance(actual, float) and np.isnan(actual)):
                    continue
                trow.append({"season": rec["season"], "week": rec["week"], "team": tm, "stat": stat, "simmean": float(d[stat].mean()), "simsd": float(d[stat].std()), "actual": float(actual), "pit": pit_of(d[stat], actual, rng.random())})
            for role, stats in ROLE_STATS.items():
                if role not in rl:
                    continue
                nm = rl[role]
                key = normalize_name(nm)
                played = key in act.index
                for stat in stats:
                    x = res.stats[nm][stat].astype(float)
                    col = f"{role}.{stat}"
                    pit_cols[(side, col)] = (pit_of(x, _actual_stat(act.loc[key], stat), rng.random()) if played else None, pit_transform(x, rng))
                    if played:
                        pr = [spec_p for spec_p in spec.players if spec_p.name == nm][0]
                        prow.append({"season": rec["season"], "week": rec["week"], "team": tm, "role": role, "name": nm, "stat": stat, "simmean": float(x.mean()), "simsd": float(x.std()),
                                     "actual": _actual_stat(act.loc[key], stat), "pit": pit_cols[(side, col)][0], "proj_tgt": pr.tgt, "proj_rec_yds": pr.rec_yds, "proj_rush_yds": pr.rush_yds,
                                     "proj_pass_yds": pr.pass_yds})
            # team-level latent series for the team-vs-player correlations
            for stat in ("points", "margin"):
                pit_cols[(side, stat)] = (pit_of(d[stat], sc[tm] if stat == "points" else sc[tm] - sc[a.abbr if side == "H" else h.abbr], rng.random()), pit_transform(d[stat].astype(float), rng))
        tot_sim = res.team[h.abbr]["points"] + res.team[a.abbr]["points"]
        tot_pit = (pit_of(tot_sim, sc[h.abbr] + sc[a.abbr], rng.random()), pit_transform(tot_sim.astype(float), rng))
        for (c1, c2, rel) in PAIRS:
            for side in ("H", "A"):
                s1 = side
                s2 = side if rel == "same" else ("A" if side == "H" else "H")
                if rel == "opp" and side == "A":
                    continue
                if rel == "same" and c1 == c2:
                    continue
                v1, v2 = pit_cols.get((s1, c1)), pit_cols.get((s2, c2))
                if v1 is None or v2 is None:
                    continue
                sim_corr[(c1, c2, rel)].append(float(np.corrcoef(v1[1], v2[1])[0, 1]))
                if v1[0] is not None and v2[0] is not None:
                    emp[(c1, c2, rel)][0].append(v1[0]); emp[(c1, c2, rel)][1].append(v2[0])
        for (tc, pc) in TEAM_PAIRS:
            for side in ("H", "A"):
                v2 = pit_cols.get((side, pc))
                v1 = tot_pit if tc == "total" else pit_cols.get((side, tc))
                if v1 is None or v2 is None:
                    continue
                tsim_corr[(tc, pc)].append(float(np.corrcoef(v1[1], v2[1])[0, 1]))
                if v1[0] is not None and v2[0] is not None:
                    temp[(tc, pc)][0].append(v1[0]); temp[(tc, pc)][1].append(v2[0])
        if progress and gi % 100 == 0:
            print("  simulated", gi, "of", len(recs), flush=True)

    def table(sim, empd, cols):
        rows = []
        for k, v in sim.items():
            a, b = empd[k]
            if len(a) < 30:
                continue
            ea, eb = np.array(a), np.array(b)
            r = float(np.corrcoef(ea, eb)[0, 1])
            se = (1 - r ** 2) / np.sqrt(len(a) - 1)
            rows.append({**dict(zip(cols, k)), "n": len(a), "actual": round(r, 3), "+/-": round(1.96 * se, 3), "simulated": round(float(np.mean(v)), 3), "gap": round(float(np.mean(v)) - r, 3)})
        return pd.DataFrame(rows)

    return {"player": pd.DataFrame(prow), "team": pd.DataFrame(trow), "pairs": table(sim_corr, emp, ("a", "b", "rel")), "tpairs": table(tsim_corr, temp, ("a", "b"))}


def summarize_pit(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Calibration of a PIT column: mean (0.5), variance (1/12 = 0.083), share in the outer 10% tails each side (0.10) and the bias of the simulated mean."""
    g = df.groupby(by)
    zz = df.assign(z=(df.actual - df.simmean) / df.simsd.replace(0, np.nan))
    gz = zz.groupby(by).z
    out = pd.DataFrame({"n": g.size(), "mean": g.pit.mean(), "var": g.pit.var(), "lo10": g.pit.apply(lambda s: (s < 0.1).mean()), "hi10": g.pit.apply(lambda s: (s > 0.9).mean()),
                        "z mean": gz.mean(), "z var": gz.var(), "sim mean": g.simmean.mean(), "actual mean": g.actual.mean()})
    out["bias %"] = 100 * (out["sim mean"] / out["actual mean"].replace(0, np.nan) - 1)
    return out.round(3)


KEY_ROWS = [("QB", "pass_yds"), ("QB", "pass_td"), ("QB", "pass_att"), ("QB", "comp"), ("RB1", "rush_yds"), ("RB1", "rush_att"), ("RB1", "rec"), ("RB2", "rush_yds"),
            ("WR1", "rec_yds"), ("WR1", "rec"), ("WR1", "tgt"), ("WR2", "rec_yds"), ("WR3", "rec_yds"), ("TE1", "rec_yds"), ("WR1", "any_td"), ("RB1", "any_td")]


def score(res: dict) -> float:
    """One number for tuning: the sum of squared standardised gaps between the simulated and the actual correlations, plus how far the standardised-residual variance is from 1."""
    p = res["pairs"]
    loss = float(((p["gap"] / (p["+/-"] / 1.96 + 0.01)) ** 2).sum() / max(len(p), 1))
    pl = summarize_pit(res["player"], ["role", "stat"])
    tm = summarize_pit(res["team"], ["stat"])
    loss += float(((np.log(pl["z var"].clip(0.05)) ** 2).mean() + (np.log(tm["z var"].clip(0.05)) ** 2).mean()))
    loss += float(((tm.loc[[k for k in ("pass_att", "rush_att", "pass_yds", "rush_yds") if k in tm.index], "bias %"] / 6.0) ** 2).mean())
    return loss


def report(res: dict, compact: bool = False) -> str:
    if compact:
        pl = summarize_pit(res["player"], ["role", "stat"])
        keep = pl.loc[[k for k in KEY_ROWS if k in pl.index]]
        return chr(10).join(["TEAM:", summarize_pit(res["team"], ["stat"]).to_string(), "", "PLAYER (key rows):", keep.to_string(), "", "PAIRS (PIT correlation, actual vs simulated):", res["pairs"].to_string(index=False), "",
                          "TEAM score vs player:", res["tpairs"].to_string(index=False), "", f"loss {score(res):.3f}"])
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 200)
    L = ["TEAM level (PIT of the actual in the simulated distribution; ideal mean .5, var .083, tails .10):", summarize_pit(res["team"], ["stat"]).to_string(), "",
         "PLAYER level by role and stat:", summarize_pit(res["player"], ["role", "stat"]).to_string(), "",
         "DEPENDENCE: correlation of PITs across real games vs what the simulator implies:", res["pairs"].to_string(index=False), "",
         "TEAM score vs player stats:", res["tpairs"].to_string(index=False)]
    return "\n".join(L)


def parse_params(argv: list[str]) -> E.SimParams:
    """`--set=a=1.5,b=0.2` overrides SimParams fields."""
    kv = {}
    for x in argv:
        if x.startswith("--set="):
            for pair in x[6:].split(","):
                k, v = pair.split("=")
                kv[k] = float(v)
    return E.SimParams().with_(**kv)


if __name__ == "__main__":
    from matchup_model.sim import history as H
    a = [int(x) for x in sys.argv[1:] if not x.startswith("--")] or [2023, 2025]
    recs = H.load(range(a[0], (a[1] if len(a) > 1 else a[0]) + 1))
    step = int(next((x.split("=")[1] for x in sys.argv if x.startswith("--every=")), 1))
    n = int(next((x.split("=")[1] for x in sys.argv if x.startswith("--n=")), 3000))
    recs = recs[::step]
    print(len(recs), "games")
    print(report(diagnose(recs, params=parse_params(sys.argv[1:]), n=n, progress=False), compact="--compact" in sys.argv))
