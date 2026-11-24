"""The hindsight solver finds the true best lineup, and `lineup_features` reads a lineup's structure correctly."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from dfs import hindsight as HS

# three games: AAA@BBB, CCC@DDD, EEE@FFF
GAME = {"AAA": ("AAA", "BBB", "g1"), "BBB": ("BBB", "AAA", "g1"), "CCC": ("CCC", "DDD", "g2"), "DDD": ("DDD", "CCC", "g2"), "EEE": ("EEE", "FFF", "g3"), "FFF": ("FFF", "EEE", "g3")}
TEAMS = list(GAME)


def tiny_pool(seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows, n = [], 0
    for t in TEAMS:
        _, opp, g = GAME[t]
        spec = [("QB", 1, 5500, 8000), ("RB", 2, 3500, 9000), ("WR", 3, 3000, 9000), ("TE", 1, 2500, 6500), ("DST", 1, 2200, 3800)]
        for pos, count, lo, hi in spec:
            for _ in range(count):
                sal = int(rng.integers(lo, hi) // 100 * 100)
                rows.append({"dk_id": f"p{n}", "name": f"{pos}{n}", "pos": pos, "team": t, "opp": opp, "game_id": g, "salary": sal,
                             "pts": float(round(max(0.0, sal / 1000 * 2.0 + rng.normal(0, 7)), 1)), "tds": float(rng.integers(0, 3)),
                             "total": 45.0 + 2 * (g == "g1"), "spread": -3.0 if t in ("AAA", "CCC", "EEE") else 3.0, "is_home": int(t in ("BBB", "DDD", "FFF")),
                             "team_total": 24.0 if t in ("AAA", "CCC", "EEE") else 21.0, "opp_total": 21.0 if t in ("AAA", "CCC", "EEE") else 24.0, "played": True})
                n += 1
    p = pd.DataFrame(rows)
    p["proj"], p["own_pct"] = np.nan, np.nan
    p["sal_rank_pos"] = p.groupby("pos")["salary"].rank(ascending=False, method="min")
    p["proj_rank_pos"] = np.nan
    games = p.drop_duplicates("game_id")[["game_id", "total"]].assign(game_total_rank=lambda d: d["total"].rank(ascending=False, method="min"))
    p = p.merge(games[["game_id", "game_total_rank"]], on="game_id")
    teams = p.drop_duplicates("team")[["team", "team_total"]].assign(team_total_rank=lambda d: d["team_total"].rank(ascending=False, method="min"))
    return p.merge(teams[["team", "team_total_rank"]], on="team")


def brute_force_best(pool: pd.DataFrame, k: int = 1) -> list[float]:
    """Scores of the best `k` valid lineups by exhaustive search."""
    by = {pos: pool[pool["pos"] == pos] for pos in ("QB", "RB", "WR", "TE", "DST")}
    sal, pts = pool["salary"].to_numpy(), pool["pts"].to_numpy()
    idx = {pos: by[pos].index.to_numpy() for pos in by}
    scores = []
    for nrb, nwr, nte in ((2, 3, 2), (2, 4, 1), (3, 3, 1)):
        for rb in itertools.combinations(idx["RB"], nrb):
            for wr in itertools.combinations(idx["WR"], nwr):
                for te in itertools.combinations(idx["TE"], nte):
                    base = list(rb) + list(wr) + list(te)
                    bs, bp = sal[base].sum(), pts[base].sum()
                    for q in idx["QB"]:
                        for d in idx["DST"]:
                            if bs + sal[q] + sal[d] <= 50000:
                                scores.append(bp + pts[q] + pts[d])
    return sorted(scores, reverse=True)[:k]


@pytest.mark.parametrize("seed", [1, 2])
def test_best_lineup_equals_exhaustive_search(seed):
    pool = tiny_pool(seed)
    ix = HS.best_index_sets(pool, 1)[0]
    assert pool["pts"].to_numpy()[ix].sum() == pytest.approx(brute_force_best(pool)[0], abs=1e-6)


def test_runner_up_lineups_are_distinct_and_no_better():
    pool = tiny_pool(3)
    sets = HS.best_index_sets(pool, 4, min_uniques=3)
    scores = [pool["pts"].to_numpy()[ix].sum() for ix in sets]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == pytest.approx(brute_force_best(pool)[0], abs=1e-6)
    for i in range(len(sets)):
        for j in range(i):
            assert len(set(sets[i]) - set(sets[j])) >= 3


def test_features_of_a_hand_built_lineup():
    pool = tiny_pool(4)

    def pick(pos, team, n=1):
        return pool[(pool["pos"] == pos) & (pool["team"] == team)]["dk_id"].tolist()[:n]
    # QB AAA + two AAA WRs (stack of 2) + a BBB WR (bring-back) + an AAA RB; CCC RB; EEE WR; EEE TE; DST FFF (faces EEE, which is in the lineup)
    ids = pick("QB", "AAA") + pick("WR", "AAA", 2) + pick("WR", "BBB") + pick("RB", "AAA") + pick("RB", "CCC") + pick("WR", "EEE") + pick("TE", "EEE") + pick("DST", "FFF")
    assert len(ids) == 9
    f = HS.lineup_features(ids, pool)
    assert f["qb_team"] == "AAA" and f["stack_wr_te"] == 2 and f["stack_rb"] == 1 and f["stack_skill"] == 3
    assert f["bring_back"] == 1 and f["bring_back_wr_te"] == 1
    assert f["n_teams"] == 5 and f["n_games"] == 3 and f["max_from_team"] == 4
    assert f["qb_game_players"] == 5 and f["max_from_game"] == 5
    assert f["dst_faces_lineup"] == 2 and f["dst_faces_qb"] == 0                      # both EEE players face the FFF defense
    assert (f["n_rb"], f["n_wr"], f["n_te"]) == (2, 4, 1) and f["flex_pos"] == "WR"
    assert f["sal_left"] == 50000 - pool.set_index("dk_id").loc[ids, "salary"].sum()
    assert f["score"] == pytest.approx(pool.set_index("dk_id").loc[ids, "pts"].sum())
    assert f["qb_fav"] == 1 and f["n_mini_stacks"] == 1                                # EEE has two players besides the QB's team
    assert np.isnan(f["own_sum"])                                                      # no ownership estimate on this pool
