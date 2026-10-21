"""matchup_model.opp.injury_boost_scan: the linear-in-the-multiplier construction, the leave-one-season-out scale, the surface and the report, on synthetic projections."""
import numpy as np
import pandas as pd
import pytest

from matchup_model.opp import injury_boost_scan as SC

TRUE_KN, TRUE_KO, LEAK = 0.5, 0.2, 0.9
STATS = {"tgt": "targets"}


def _frame(n=300, seed=1):
    """Three seasons of players whose actual stat is LEAK x (no-boost projection) x (1 + true boosts): the scan must find (0.5, 0.2) and a zero error there."""
    rng = np.random.default_rng(seed)
    m = 3 * n
    d = pd.DataFrame({"season": np.repeat([2023, 2024, 2025], n), "position": rng.choice(["WR", "TE"], m), "team": "AAA", "week": 5})
    d["p0_tgt"] = rng.uniform(1.0, 9.0, m)
    d["bn"] = rng.choice([0.0, 0.0, 0.2, 0.5], m)
    d["bo"] = rng.choice([0.0, 0.0, 0.15, 0.4], m)
    d["targets"] = LEAK * d.p0_tgt * (1 + TRUE_KN * d.bn + TRUE_KO * d.bo)
    return d


def test_a_boost_is_the_ratio_to_the_projection_without_it():
    p0 = pd.Series([2.0, 4.0, 0.01, 0.0])
    got = SC.boosts(p0, pd.Series([2.5, 4.0, 0.5, 0.3]))
    assert got.tolist() == pytest.approx([0.25, 0.0, 0.0, 0.0])            # projections at or under the floor cannot be scaled: no boost recorded


def test_the_scale_of_a_held_out_season_comes_from_the_other_seasons():
    d = _frame()
    d.loc[d.season == 2025, "targets"] *= 2.0                              # 2025 behaves differently: its own rows must not be used to learn its scale
    err, inv = SC.errors(d, STATS, TRUE_KN, TRUE_KO, "bn", "bo", "tgt", 3.0, by="position")
    e25 = err["tgt"][(d.season == 2025).to_numpy()]
    assert np.nanmean(e25) < -0.3 * d[d.season == 2025].targets.mean() and np.isfinite(e25).all()      # under-projects by about half: the other seasons' scale
    e23 = err["tgt"][(d.season == 2023).to_numpy()]
    assert np.nanmean(e23) > 0.3 * d[d.season == 2023].targets.mean()         # 2023 is scaled with the other two seasons, one of them doubled: it is over-projected
    assert inv.dtype == bool and inv.sum() < len(d)                             # 'involved' is the rows projected for >= 3


def test_the_surface_bottoms_out_at_the_true_fractions_and_the_old_rule_is_the_reference():
    d = _frame()
    t = SC.surface(d, STATS, "bn", "bo", "tgt", 3.0, by="position", kn_grid=[0.0, 0.25, 0.5, 0.75, 1.0], ko_grid=[0.0, 0.2, 0.4, 1.0])
    best = t.loc[t["all"].idxmin()]
    assert (best.kn, best.ko) == (0.5, 0.2) and best["all"] == pytest.approx(-100.0, abs=1e-6)    # zero error against the old rule's non-zero one
    ref = t[(t.kn == 1.0) & (t.ko == 1.0)].iloc[0]
    assert ref["all"] == pytest.approx(0.0) and ref["inv"] == pytest.approx(0.0)         # the old rule is the reference by construction
    off_grid = SC.surface(d, STATS, "bn", "bo", "tgt", 3.0, by="position", kn_grid=[0.5], ko_grid=[0.2])
    assert off_grid["all"].iloc[0] == pytest.approx(-100.0, abs=1e-6)                     # the reference does not need to be on the grid
    assert (t["all"] >= best["all"] - 1e-9).all()
    assert t[(t.kn == 0.0) & (t.ko == 0.0)].iloc[0]["all"] > -100.0           # no boost at all is worse than the truth


def test_the_context_table_breaks_the_error_down_by_injury_situation():
    d = _frame(n=120)
    ctx = pd.Series(np.where(d.bn > 0, "C new absence", "A none"), index=d.index)
    t = SC.context_table(d, ctx, STATS, [(1.0, 1.0), (TRUE_KN, TRUE_KO)], "bn", "bo", "tgt", 3.0, by="position")
    assert set(t.context) == {"A none", "C new absence"} and len(t) == 4
    right = t[t["kappa new / ongoing"] == f"{TRUE_KN:g} / {TRUE_KO:g}"]
    assert right.RMSE.max() < 1e-6                                              # exact data, exact setting
    old = t[(t["kappa new / ongoing"] == "1 / 1") & (t.context == "C new absence")].iloc[0]
    assert old["mean error"] > 0                                                # the old full boost over-projects the beneficiaries


def test_the_report_lists_every_table_and_the_shipped_fractions():
    d = _frame(n=60)
    t = SC.surface(d, STATS, "bn", "bo", "tgt", 3.0, by="position", kn_grid=[0.5, 1.0], ko_grid=[0.2, 1.0])
    ctx = pd.DataFrame({"context": ["A none"], "rows": [10], "kappa new / ongoing": ["1 / 1"], "RMSE": [2.0], "mean error": [0.1]})
    txt = SC.build_report({"WR/TE receiving": t}, ctx, {"rec": {"new": 0.46, "ongoing": 0.37}})
    for needle in ("## WR/TE receiving", "Everyone who played:", "Players the model expects to be involved", "new 0.5", "ongoing 0.2", "by injury context", "'new': 0.46"):
        assert needle in txt
    assert "by injury context" not in SC.build_report({"WR/TE receiving": t}, pd.DataFrame(), {"rec": {}})


def test_the_runs_switch_on_one_boost_at_a_time():
    assert SC.KINDS["zero"] == {"rec": {"new": 0.0, "ongoing": 0.0}, "rush": {"new": 0.0, "ongoing": 0.0}}
    assert SC.KINDS["rec_new"]["rec"] == {"new": 1.0, "ongoing": 0.0} and SC.KINDS["rec_old"]["rec"] == {"new": 0.0, "ongoing": 1.0}
    assert SC.KINDS["rush_new"]["rush"] == {"new": 1.0, "ongoing": 0.0} and SC.KINDS["rush_old"]["rec"] == {"new": 0.0, "ongoing": 0.0}
