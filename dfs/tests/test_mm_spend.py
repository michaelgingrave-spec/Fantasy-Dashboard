"""The spend / price-tier / environment tables behind the Millionaire Maker evidence, on small synthetic frames whose answers are known."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dfs import mm_spend as M


def lineups(qb: float, rb: float, n_weeks: int = 24, k: int = 3, rank1_qb: float | None = None, seasons=(2015, 2019)) -> pd.DataFrame:
    rows = []
    for i in range(n_weeks):
        season = seasons[i % len(seasons)]
        for r in range(1, k + 1):
            q = rank1_qb if (rank1_qb is not None and r == 1) else qb
            rows.append({"season": season, "week": i // len(seasons) + 1, "rank": r, "sal_qb": q, "sal_rb": rb, "sal_wr": 20000.0, "sal_te": 4000.0, "sal_dst": 3000.0,
                         "sal_left": 50000 - q - rb - 27000.0, "n_rb": 2, "n_wr": 3, "n_te": 1})
    return pd.DataFrame(rows)


def test_spend_table_means_interval_and_eras():
    hind = lineups(qb=6000.0, rb=14000.0)
    rows = {r["slot"]: r for r in M.spend_table(hind, reps=200)}
    assert rows["QB"]["best1"] == pytest.approx(6000) and rows["QB"]["top25"] == pytest.approx(6000)
    assert rows["QB"]["lo"] == pytest.approx(6000) and rows["QB"]["hi"] == pytest.approx(6000) and rows["QB"]["p10"] == rows["QB"]["p90"] == 6000     # no week-to-week spread in this fixture
    assert rows["per RB"]["best1"] == pytest.approx(7000)                                                                                       # 14,000 over 2 RBs
    assert set(rows["QB"]["eras"]) == {"2014-17", "2018-21"} and rows["QB"]["eras"]["2014-17"]["weeks"] == 12


def test_spend_table_reads_the_single_best_lineup_apart_from_the_best_25():
    hind = lineups(qb=7000.0, rb=14000.0, rank1_qb=5000.0)                       # only rank 1 spends less at QB
    row = next(r for r in M.spend_table(hind, reps=100) if r["slot"] == "QB")
    assert row["best1"] == pytest.approx(5000) and row["top25"] == pytest.approx((5000 + 7000 + 7000) / 3)


def test_price_tiers_cover_the_position_once():
    rng = np.random.default_rng(4)
    rows = []
    for w in range(1, 31):
        for pos, sals in (("QB", [5000, 6000, 7000, 8000]), ("RB", [3500, 5000, 6500, 8000, 9000]), ("WR", [3500, 5000, 6500, 8000, 9000]), ("TE", [2800, 4000, 5000, 6000]), ("DST", [2500, 2800, 3200, 3600])):
            for i, sal in enumerate(sals * 3):
                rows.append({"season": 2018, "week": w, "pos": pos, "salary": sal, "pts": max(0.0, rng.normal(sal / 1000 * 2.5, 6))})
    out = M.price_tier_table(pd.DataFrame(rows), reps=100)
    for pos in M.TIERS:
        tiers = [r for r in out if r["pos"] == pos]
        assert len(tiers) == len(M.TIERS[pos])
        assert sum(r["pool_share"] for r in tiers) == pytest.approx(1.0)
        assert sum(r["top_share"] for r in tiers) == pytest.approx(1.0)
        assert all(r["hit_lo"] <= r["hit"] <= r["hit_hi"] and r["top_lo"] <= r["top"] <= r["top_hi"] for r in tiers)
    qb = [r for r in out if r["pos"] == "QB"]
    assert qb[0]["tier"] == "under $5,500" and qb[1]["tier"] == "$5,500-6,499" and qb[-1]["tier"] == "$7,500 and up"
    assert qb[-1]["pts"] > qb[0]["pts"]                                                                       # the generator pays more for a higher price


def test_price_tier_hit_definition_is_points_per_thousand():
    # one QB at $5,000 scoring 25 points is exactly 5x; one scoring 24.9 is not
    pool = pd.DataFrame({"season": 2018, "week": [1, 1, 2, 2], "pos": "QB", "salary": [5000, 5000, 5000, 5000], "pts": [25.0, 24.9, 10.0, 30.0]})
    row = M.price_tier_table(pool, reps=50)[0]
    assert row["players"] == 4 and row["hit"] == pytest.approx(0.5)


def test_clustered_ols_recovers_a_slope_and_a_null():
    rng = np.random.default_rng(1)
    n = 4000
    d = pd.DataFrame({"season": 2019, "week": rng.integers(1, 17, n), "x": rng.normal(size=n), "z": rng.normal(size=n)})
    d["surprise"] = 1.5 * d["x"] + rng.normal(0, 5, n)
    b, se, used = M.clustered_ols(d, ["x", "z"])
    assert used == n and b[1] == pytest.approx(1.5, abs=0.4) and abs(b[1] / se[1]) > 4 and abs(b[2] / se[2]) < 4


def test_environment_effects_find_a_planted_effect_and_leave_the_rest_alone():
    rng = np.random.default_rng(9)
    rows = []
    for season in (2015, 2016, 2021, 2026):
        for week in range(1, 15):
            for t in range(16):
                for pos, base in (("WR", 6000), ("RB", 6000), ("QB", 6500), ("TE", 4500), ("DST", 3000)):
                    sal = base + rng.integers(-1500, 1500)
                    rank = int(rng.integers(1, 33))
                    tt = rng.normal(22, 3)
                    pts = 0.002 * sal + 0.6 * (16.5 - rank) / 9.2 + rng.normal(0, 4)                            # a weaker defense (low rank number) adds points; the total adds nothing
                    rows.append({"season": season, "week": week, "pos": pos, "salary": float(sal), "pts": pts, "team_total": tt, "env_fpa_rank": rank, "env_epa_rank": rank,
                                 "team_off_rank": int(rng.integers(1, 33)), "env_pa_rank": int(rng.integers(1, 33))})
    out = M.environment_effects(pd.DataFrame(rows))
    assert {r["spec"] for r in out} == {s for s, _, _ in M.ENV_SPECS}
    wr = next(r for r in out if r["pos"] == "WR" and r["spec"] == M.ENV_SPECS[0][0])
    terms = {t["name"]: t for t in wr["terms"]}
    assert terms["weak_fpa"]["coef"] == pytest.approx(0.6, abs=0.25) and terms["weak_fpa"]["t"] > 3
    assert abs(terms["tt_z"]["t"]) < 3.5
    epa = [r for r in out if r["spec"] == M.ENV_SPECS[1][0]]
    assert epa and all(r["n"] > 100 for r in epa)                                                                   # only the 2021 and 2026 weeks feed the EPA specification
