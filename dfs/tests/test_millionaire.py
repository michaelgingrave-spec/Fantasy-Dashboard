"""The Millionaire Maker builder: every rule it advertises is true of every lineup it returns."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dfs.millionaire import MMConfig, build_portfolio, portfolio_summary, profile, read_ownership_file
from dfs.optimizer import OptimizerError

TEAMS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH", "III", "JJJ"]
FIRST = ["Aaron", "Brandon", "Calvin", "Derek", "Elijah", "Frank", "Gavin", "Hunter", "Ivan", "Jalen", "Kyle"]
LAST = ["Adams", "Bishop", "Carter", "Dixon", "Ellis", "Fisher", "Grant", "Hayes", "Irwin", "Jensen"]
OPPONENT = {t: TEAMS[i ^ 1] for i, t in enumerate(TEAMS)}


def make_slate(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows, n = [], 0

    def add(pos, team, lo, hi):
        nonlocal n
        sal = int(rng.integers(lo, hi) // 100 * 100)
        proj = float(sal / 1000 * 2.3 + rng.normal(0, 2.0))
        rows.append({"dk_id": str(5000 + n), "name": f"{FIRST[n % 11]} {LAST[n // 11]}", "pos": pos, "team": team, "opp": OPPONENT[team], "salary": sal, "proj": proj,
                     "ceiling": proj * 1.6, "floor": proj * 0.5, "playable": True, "own_pct": float(rng.uniform(1, 30))})
        n += 1

    for t in TEAMS:
        add("QB", t, 5000, 8000)
        for _ in range(3): add("RB", t, 3000, 9000)
        for _ in range(4): add("WR", t, 3000, 9000)
        for _ in range(2): add("TE", t, 2500, 7000)
        add("DST", t, 2200, 4000)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def slate():
    return make_slate()


def test_default_rules_hold_for_every_lineup(slate):
    cfg = MMConfig(n_lineups=12, seed=11)
    lineups = build_portfolio(slate, cfg)
    assert len(lineups) == 12
    assert [lu.proj for lu in lineups] == sorted((lu.proj for lu in lineups), reverse=True)           # best projection first
    sets = [set(lu.players["dk_id"]) for lu in lineups]
    for i, lu in enumerate(lineups):
        p = lu.players
        assert list(p["slot"]) == ["QB", "RB1", "RB2", "WR1", "WR2", "WR3", "TE", "FLEX", "DST"]
        assert 49000 <= lu.salary <= 50000
        qb, dst = p[p["pos"] == "QB"].iloc[0], p[p["pos"] == "DST"].iloc[0]
        off = p[p["pos"] != "DST"]
        assert len(p[(p["team"] == qb["team"]) & p["pos"].isin(["WR", "TE"])]) >= 2           # QB stack
        assert len(p[(p["team"] == qb["opp"]) & p["pos"].isin(["WR", "TE"])]) >= 1            # bring-back
        assert off["team"].value_counts().max() <= 3                                         # at most 3 from one team
        assert not (off["team"] == dst["opp"]).any()                                         # nobody faces the lineup's own defense
        for j in range(i):
            assert len(sets[i] - sets[j]) >= 3
    counts = pd.Series([d for s in sets for d in s]).value_counts()
    assert counts.max() <= int(np.ceil(0.4 * 12))                                            # exposure cap


def test_profile_matches_the_lineup(slate):
    lu = build_portfolio(slate, MMConfig(n_lineups=1, seed=2))[0]
    pr = profile(lu, slate)
    p = lu.players
    qb = p[p["pos"] == "QB"].iloc[0]
    assert pr["stack"] == len(p[(p["team"] == qb["team"]) & p["pos"].isin(["WR", "TE"])]) >= 2
    assert pr["back"] >= 1 and pr["teams"] == p["team"].nunique() and pr["max_from_team"] <= 3
    assert pr["salary_left"] == 50000 - lu.salary
    assert pr["own_sum"] == pytest.approx(slate.set_index("dk_id").loc[p["dk_id"].astype(str), "own_pct"].sum())


def test_locks_excludes_and_loosened_rules(slate):
    qbs = slate[slate["pos"] == "QB"]["dk_id"].tolist()
    lineups = build_portfolio(slate, MMConfig(n_lineups=5, seed=4, qb_stack=0, bring_back=0, max_from_team=9, dst_vs="any", locks=[qbs[0]], excludes=[qbs[1]], max_exposure=1.0))
    assert len(lineups) == 5
    for lu in lineups:
        ids = set(lu.players["dk_id"])
        assert qbs[0] in ids and qbs[1] not in ids


def test_portfolio_summary(slate):
    lineups = build_portfolio(slate, MMConfig(n_lineups=8, seed=5))
    s = portfolio_summary(lineups, slate)
    assert s["lineups"] == 8 and s["with_stack"] == 1.0 and s["with_back"] == 1.0
    assert s["proj_min"] <= s["proj_mean"] <= s["proj_max"] and sum(s["qbs"].values()) == 8


def test_ownership_rules_need_an_estimate_and_are_honoured(slate):
    with pytest.raises(OptimizerError, match="ownership"):
        build_portfolio(slate.drop(columns=["own_pct"]), MMConfig(n_lineups=1, own_max=100))
    lineups = build_portfolio(slate, MMConfig(n_lineups=4, seed=6, own_max=110, min_low_own=2, low_own=8, max_exposure=1.0))
    own = slate.set_index("dk_id")["own_pct"]
    for lu in lineups:
        o = own.loc[lu.players["dk_id"].astype(str)]
        assert o.sum() <= 110 + 1e-6 and (o <= 8).sum() >= 2


def test_errors(slate):
    with pytest.raises(OptimizerError, match="empty"):
        build_portfolio(slate.iloc[0:0])
    with pytest.raises(OptimizerError, match="proj"):
        build_portfolio(slate.drop(columns=["proj"]))
    with pytest.raises(OptimizerError, match="No valid lineup"):
        build_portfolio(slate, MMConfig(n_lineups=1, min_salary=70000))


def test_players_marked_not_playable_are_left_out_unless_locked(slate):
    out_qb = slate[slate["pos"] == "QB"]["dk_id"].iloc[0]
    s = slate.assign(playable=slate["dk_id"] != out_qb)
    lineups = build_portfolio(s, MMConfig(n_lineups=6, seed=7, max_exposure=1.0))
    assert all(out_qb not in set(lu.players["dk_id"]) for lu in lineups)
    locked = build_portfolio(s, MMConfig(n_lineups=2, seed=7, max_exposure=1.0, locks=[out_qb]))
    assert all(out_qb in set(lu.players["dk_id"]) for lu in locked)


def test_read_ownership_file(slate):
    names = slate["name"].tolist()
    df = pd.DataFrame({"Player": names[:40] + ["Somebody Else"], "Ownership %": [f"{i % 25 + 1}%" for i in range(41)]})
    own, missing = read_ownership_file(df, slate)
    assert len(own) == 40 and own.iloc[0] == 1.0
    assert all(slate.set_index("dk_id").loc[i, "name"] in names[:40] for i in own.index)
    assert any(slate.set_index("name").loc[m, "proj"] >= 8 for m in missing)                 # a real player it could not place is reported
    with pytest.raises(ValueError, match="ownership column"):
        read_ownership_file(pd.DataFrame({"foo": [1], "bar": [2]}), slate)
