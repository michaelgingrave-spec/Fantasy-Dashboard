"""Heuristic chalk-ownership: shape, normalisation, and graceful degradation."""
import numpy as np
import pandas as pd

from dfs import ownership as O


def _slate():
    rows = []
    # 3 studs, 3 mids, 4 punts per skill position — enough to exercise normalisation
    tiers = {"WR": [(8200, 22), (7600, 19), (7000, 17), (5800, 13), (5200, 11),
                    (4800, 10), (4000, 7.5), (3600, 6), (3200, 4.5), (3000, 3)],
             "RB": [(8400, 21), (7800, 19), (7200, 17), (5600, 13), (5000, 10),
                    (4400, 8), (3800, 6), (3400, 4)],
             "QB": [(7600, 22), (7200, 21), (6600, 19), (5400, 16), (4800, 14)],
             "TE": [(6400, 14), (5200, 11), (4200, 8), (3400, 5), (3000, 3.5)]}
    for pos, ps in tiers.items():
        for i, (sal, proj) in enumerate(ps):
            rows.append({"name": f"{pos}{i}", "pos": pos, "team": "KC",
                         "salary": sal, "proj": proj})
    return pd.DataFrame(rows)


def test_sums_near_position_targets():
    res = O.chalk_ownership(_slate(), week=1)
    for pos in ("QB", "RB", "WR", "TE"):
        s = res[res.pos == pos]["own_pct"].sum()
        target = O.POS_OWN_SUM[pos]
        assert 0.55 * target <= s <= 1.05 * target, f"{pos} sum {s} vs target {target}"


def test_higher_value_gets_more_ownership():
    res = O.chalk_ownership(_slate(), week=1).set_index("name")
    # WR0 (best salary+proj, strong value) should outrank a mid and a punt
    assert res.loc["WR0", "own_pct"] > res.loc["WR5", "own_pct"] > res.loc["WR9", "own_pct"]


def test_ownership_is_bounded():
    res = O.chalk_ownership(_slate(), week=1)
    o = res["own_pct"].dropna()
    assert o.min() >= O.OWN_CLIP[0] - 1e-6
    assert o.max() <= O.OWN_CLIP[1] + 1e-6


def test_unpriced_players_get_nan():
    slate = _slate()
    slate.loc[slate["name"] == "WR3", "salary"] = np.nan
    res = O.chalk_ownership(slate, week=1).set_index("name")
    assert pd.isna(res.loc["WR3", "own_pct"])
    assert res.loc["WR0", "own_pct"] > 0


def test_missing_columns_is_safe():
    res = O.chalk_ownership(pd.DataFrame({"name": ["x"], "pos": ["WR"]}), week=1)
    assert "own_pct" in res.columns and res["own_pct"].isna().all()
