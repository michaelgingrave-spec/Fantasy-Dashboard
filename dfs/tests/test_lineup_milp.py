"""The MILP layer behind the Millionaire Maker builder: the same optimum as the PuLP optimizer, and every rule it offers is actually enforced."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dfs import lineup_milp as L
from dfs.lineup_milp import PoolModel, Rules, solve_portfolio
from dfs.optimizer import ideal_lineup

TEAMS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH"]
GAMES = {"AAA": ("AAA", "BBB"), "BBB": ("AAA", "BBB"), "CCC": ("CCC", "DDD"), "DDD": ("CCC", "DDD"),
         "EEE": ("EEE", "FFF"), "FFF": ("EEE", "FFF"), "GGG": ("GGG", "HHH"), "HHH": ("GGG", "HHH")}


def make_pool(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    n = 0

    def add(pos, team, lo, hi):
        nonlocal n
        sal = int(rng.integers(lo, hi) // 100 * 100)
        rows.append({"dk_id": str(1000 + n), "name": f"{pos}{n}", "pos": pos, "team": team, "opp": [t for t in GAMES[team] if t != team][0],
                     "game_id": "-".join(GAMES[team]), "salary": sal, "proj": float(sal / 1000 * 2.2 + rng.normal(0, 2.5)), "pts": float(max(0.0, sal / 1000 * 2.2 + rng.normal(0, 7)))})
        n += 1

    for t in TEAMS:
        add("QB", t, 5000, 8000)
        for _ in range(3): add("RB", t, 3000, 9000)
        for _ in range(4): add("WR", t, 3000, 9000)
        for _ in range(2): add("TE", t, 2500, 7000)
        add("DST", t, 2200, 4000)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def pool():
    return make_pool()


def lineup_frame(pool, ix):
    return pool.iloc[ix]


def check_basic(pool, ix):
    lu = lineup_frame(pool, ix)
    counts = lu["pos"].value_counts().to_dict()
    assert len(lu) == 9 and lu["salary"].sum() <= 50000
    assert counts["QB"] == 1 and counts["DST"] == 1
    assert 2 <= counts.get("RB", 0) <= 3 and 3 <= counts.get("WR", 0) <= 4 and 1 <= counts.get("TE", 0) <= 2
    assert counts.get("RB", 0) + counts.get("WR", 0) + counts.get("TE", 0) == 7


def test_matches_the_optimizers_optimum(pool):
    ix = solve_portfolio(PoolModel(pool, Rules()), pool["proj"].to_numpy(), n=1)[0]
    check_basic(pool, ix)
    ref = ideal_lineup(pool[["dk_id", "name", "pos", "team", "opp", "salary", "proj"]])
    assert pool["proj"].to_numpy()[ix].sum() == pytest.approx(ref.proj, abs=0.02)


def test_pulp_fallback_matches_highs(pool, monkeypatch):
    pytest.importorskip("scipy")
    obj = pool["proj"].to_numpy()
    rules = Rules(qb_stack=1, bring_back=1, max_from_team=3)
    fast = solve_portfolio(PoolModel(pool, rules), obj, n=1)[0]
    monkeypatch.setattr(L, "HAVE_SCIPY", False)
    slow = solve_portfolio(PoolModel(pool, rules), obj, n=1)[0]
    assert obj[fast].sum() == pytest.approx(obj[slow].sum(), abs=1e-6)
    check_basic(pool, slow)


def test_qb_stack_and_bring_back_are_enforced(pool):
    rng = np.random.default_rng(3)
    model = PoolModel(pool, Rules(qb_stack=2, bring_back=1))
    sets = solve_portfolio(model, lambda k: pool["proj"].to_numpy() * (1 + rng.uniform(-0.4, 0.4, len(pool))), n=12)
    assert len(sets) == 12
    for ix in sets:
        lu = lineup_frame(pool, ix)
        qb = lu[lu["pos"] == "QB"].iloc[0]
        mates = lu[(lu["team"] == qb["team"]) & lu["pos"].isin(["WR", "TE"])]
        back = lu[(lu["team"] == qb["opp"]) & lu["pos"].isin(["WR", "TE"])]
        assert len(mates) >= 2 and len(back) >= 1


def test_team_game_and_dst_rules(pool):
    rng = np.random.default_rng(4)
    rules = Rules(max_from_team=2, max_per_game=3, min_games=4, min_teams=7, dst_vs="lineup", min_salary=49000)
    sets = solve_portfolio(PoolModel(pool, rules), lambda k: pool["proj"].to_numpy() * (1 + rng.uniform(-0.4, 0.4, len(pool))), n=10)
    assert len(sets) >= 5
    for ix in sets:
        check_basic(pool, ix)
        lu = lineup_frame(pool, ix)
        off = lu[lu["pos"] != "DST"]
        assert off["team"].value_counts().max() <= 2
        assert lu["game_id"].value_counts().max() <= 3 and lu["game_id"].nunique() >= 4 and lu["team"].nunique() >= 7
        dst = lu[lu["pos"] == "DST"].iloc[0]
        assert not (off["team"] == dst["opp"]).any()
        assert lu["salary"].sum() >= 49000


def test_dst_may_not_face_the_qb_but_can_face_others(pool):
    sets = solve_portfolio(PoolModel(pool, Rules(dst_vs="qb")), pool["proj"].to_numpy(), n=8, min_uniques=2)
    for ix in sets:
        lu = lineup_frame(pool, ix)
        qb, dst = lu[lu["pos"] == "QB"].iloc[0], lu[lu["pos"] == "DST"].iloc[0]
        assert dst["opp"] != qb["team"]


def test_portfolio_uniqueness_exposure_locks_excludes(pool):
    rng = np.random.default_rng(5)
    obj = lambda k: pool["proj"].to_numpy() * (1 + rng.uniform(-0.3, 0.3, len(pool)))
    locked, banned = pool[pool["pos"] == "QB"]["dk_id"].iloc[0], pool[pool["pos"] == "QB"]["dk_id"].iloc[1]
    sets = solve_portfolio(PoolModel(pool, Rules()), obj, n=10, min_uniques=3, max_exposure=0.5, locks=[locked], excludes=[banned])
    assert len(sets) == 10
    ids = [set(pool["dk_id"].iloc[ix]) for ix in sets]
    assert all(locked in s and banned not in s for s in ids)
    for i in range(len(ids)):
        for j in range(i):
            assert len(ids[i] - ids[j]) >= 3                                   # each lineup is at least 3 players away from every earlier one
    counts = pd.Series([d for s in ids for d in s if d != locked]).value_counts()
    assert counts.max() <= 5                                                    # exposure cap 50% of 10


def test_impossible_rules_return_no_lineup(pool):
    assert solve_portfolio(PoolModel(pool, Rules(min_salary=60000)), pool["proj"].to_numpy(), n=3) == []


def test_ownership_rules(pool):
    p = pool.assign(own=np.random.default_rng(6).uniform(1, 30, len(pool)))
    sets = solve_portfolio(PoolModel(p, Rules(own_max=80, min_low_own=2, low_own=5)), p["proj"].to_numpy(), n=3, min_uniques=2)
    assert sets
    for ix in sets:
        lu = p.iloc[ix]
        assert lu["own"].sum() <= 80 + 1e-6 and (lu["own"] <= 5).sum() >= 2
    with pytest.raises(KeyError, match="own"):
        PoolModel(pool, Rules(own_max=80))


def test_salary_shape_rules(pool):
    rng = np.random.default_rng(8)
    rules = Rules(max_qb_salary=6500, max_stars=1, star_salary=7500, max_te=1, dst_vs="any")
    sets = solve_portfolio(PoolModel(pool, rules), lambda k: pool["proj"].to_numpy() * (1 + rng.uniform(-0.4, 0.4, len(pool))), n=8)
    assert len(sets) == 8
    for ix in sets:
        lu = pool.iloc[ix]
        assert lu[lu["pos"] == "QB"]["salary"].iloc[0] <= 6500
        assert (lu[lu["pos"].isin(["RB", "WR", "TE"])]["salary"] >= 7500).sum() <= 1
        assert (lu["pos"] == "TE").sum() == 1


def test_spend_count_and_price_rules(pool):
    rng = np.random.default_rng(9)
    obj = lambda k: pool["proj"].to_numpy() * (1 + rng.uniform(-0.4, 0.4, len(pool)))
    # the FLEX is a WR (4 WRs, 2 RBs, 1 TE), a cheap TE, a QB in a window, RBs and WRs each spend within a band
    rules = Rules(counts=(("WR", 4, 4), ("RB", 2, 2), ("TE", 1, 1)), price_window=(("TE", 0, 4500), ("QB", 5500, 7000)),
                  spend=(("RB", 9000, 16000), ("WR", 14000, 24000)), price_counts=(("SKILL", 7500, 99999, 0, 1),))
    sets = solve_portfolio(PoolModel(pool, rules), obj, n=8)
    assert len(sets) == 8
    for ix in sets:
        check_basic(pool, ix)
        lu = pool.iloc[ix]
        assert (lu["pos"] == "WR").sum() == 4 and (lu["pos"] == "RB").sum() == 2 and (lu["pos"] == "TE").sum() == 1
        assert lu[lu["pos"] == "TE"]["salary"].iloc[0] <= 4500 and 5500 <= lu[lu["pos"] == "QB"]["salary"].iloc[0] <= 7000
        assert 9000 <= lu[lu["pos"] == "RB"]["salary"].sum() <= 16000 and 14000 <= lu[lu["pos"] == "WR"]["salary"].sum() <= 24000
        assert (lu[lu["pos"].isin(["RB", "WR", "TE"])]["salary"] >= 7500).sum() <= 1


def test_environment_rules_use_pool_columns(pool):
    rng = np.random.default_rng(10)
    p = pool.assign(weak_rank=pool.groupby("pos")["salary"].rank(method="first").astype(float) % 8 + 1.0,        # a 1..8 number per player, whatever it stands for
                    tt=np.where(pool["team"].isin(["AAA", "BBB", "CCC"]), 27.0, 18.0))
    obj = lambda k: p["proj"].to_numpy() * (1 + rng.uniform(-0.4, 0.4, len(p)))
    rules = Rules(env_bounds=(("QB", "weak_rank", 1, 3),), env_counts=(("OFFENSE", "tt", 25, 99, 4, 9),))
    sets = solve_portfolio(PoolModel(p, rules), obj, n=6)
    assert len(sets) == 6
    for ix in sets:
        lu = p.iloc[ix]
        assert lu[lu["pos"] == "QB"]["weak_rank"].iloc[0] <= 3
        assert ((lu["pos"] != "DST") & (lu["tt"] >= 25)).sum() >= 4
    with pytest.raises(KeyError, match="nope"):
        PoolModel(p, Rules(env_bounds=(("QB", "nope", 1, 3),)))
    # a missing value is outside every window, so such a QB can never be picked
    q = p.assign(weak_rank=np.where(p["pos"] == "QB", np.nan, p["weak_rank"]))
    assert solve_portfolio(PoolModel(q, Rules(env_bounds=(("QB", "weak_rank", 1, 8),))), q["proj"].to_numpy(), n=1) == []
