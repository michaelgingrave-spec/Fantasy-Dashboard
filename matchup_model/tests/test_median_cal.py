"""matchup_model.opp.median_cal: the factors a projection is rescaled to its median by, learned only from the graded lines of earlier weeks."""
import numpy as np
import pandas as pd
import pytest

from matchup_model.opp import median_cal as MC


def _graded(market="rec yds", n=90, weeks=(1, 2, 3), ratios=(0.5, 0.8, 1.0), season=2026):
    """n lines with projections spread from 10 to 100 (so the terciles are clean), each actual = its projection x the ratio of its tercile."""
    proj = np.linspace(10, 100, n)
    tercile = np.digitize(proj, np.quantile(proj, [1 / 3, 2 / 3]))
    return pd.DataFrame({"season": season, "week": np.resize(list(weeks), n), "player": [f"P{i}" for i in range(n)], "market": market, "our proj": proj,
                         "actual": proj * np.array(ratios)[tercile], "result": "win"})


def test_each_projection_tercile_gets_its_own_median_ratio_and_a_thin_one_falls_back_to_the_market():
    g = _graded()
    f = MC.learn(g, min_cell=25)
    assert MC.factor(f, "rec yds", 20.0) == pytest.approx(0.5) and MC.factor(f, "rec yds", 60.0) == pytest.approx(0.8) and MC.factor(f, "rec yds", 95.0) == pytest.approx(1.0)
    assert MC.apply(f, "rec yds", 60.0) == pytest.approx(48.0)
    thin = MC.learn(g, min_cell=31)                                       # 30 lines a tercile: too few for their own ratio, so the market's overall median (0.8) applies everywhere
    assert MC.factor(thin, "rec yds", 20.0) == pytest.approx(0.8) and MC.factor(thin, "rec yds", 95.0) == pytest.approx(0.8)


def test_a_market_with_too_few_lines_is_left_alone():
    assert MC.learn(_graded(n=24), min_cell=25) == {}
    assert MC.factor({}, "rec yds", 55.0) == 1.0 and MC.apply({}, "rec yds", 55.0) == 55.0
    both = MC.learn(pd.concat([_graded("rec yds"), _graded("receptions", n=10)]))
    assert "rec yds" in both and "receptions" not in both and MC.factor(both, "receptions", 3.0) == 1.0


def test_projections_near_zero_are_not_learned_from():
    junk = _graded(n=6).assign(**{"our proj": 0.1, "actual": 5.0})        # a ratio of 50 to a near-zero projection is noise
    clean = _graded()
    a, b = MC.learn(clean), MC.learn(pd.concat([clean, junk], ignore_index=True))
    assert MC.factor(a, "rec yds", 60.0) == MC.factor(b, "rec yds", 60.0) and a["rec yds"][1] == b["rec yds"][1]


def test_only_graded_lines_of_earlier_weeks_of_the_season_count_one_per_line():
    g = _graded(n=60, weeks=(1, 2, 3, 4, 5))
    g.loc[g.index[:6], "result"] = ""                                      # not graded yet
    g.loc[g.index[6:9], "season"] = 2025                                   # last season
    dup = g.iloc[[10]].assign(**{"our proj": 999.0})                       # the same (week, player, market) logged twice: the first row stays
    g = pd.concat([g, dup], ignore_index=True).assign(actual=lambda d: d["actual"].astype(object))
    out = MC.graded(g, 2026, 4)
    assert set(out["week"]) <= {1, 2, 3} and len(out) == len(set(zip(out["week"], out["player"], out["market"])))
    assert not (out["our proj"] == 999.0).any() and out["our proj"].dtype == float and out["actual"].dtype == float
    assert MC.graded(g, 2026, 1).empty and MC.graded(pd.DataFrame(), 2026, 4).empty


def test_no_correction_until_two_graded_weeks_exist():
    g = _graded(n=90, weeks=(1, 2))
    assert MC.learn_as_of(g[g.week == 1], 2026, 2) == {}                  # one week behind it: not enough
    assert MC.learn_as_of(g, 2026, 2) == {}                               # week 2's own lines are not "before" week 2
    assert "rec yds" in MC.learn_as_of(g, 2026, 3)                        # weeks 1 and 2 are behind a week 3 pull
    assert MC.learn_as_of(g, 2026, 3, min_weeks=3) == {}


def test_the_factors_for_a_week_never_see_that_weeks_results():
    g = _graded(n=120, weeks=(1, 2, 3))
    wild = g.copy()
    wild.loc[wild.week == 3, "actual"] *= 10                              # week 3 turns out completely different
    a, b = MC.learn_as_of(g, 2026, 3), MC.learn_as_of(wild, 2026, 3)
    for p in (15.0, 55.0, 95.0):
        assert MC.factor(a, "rec yds", p) == MC.factor(b, "rec yds", p)
    assert MC.learn_as_of(g, 2026, 4)["rec yds"][1] != MC.learn_as_of(wild, 2026, 4)["rec yds"][1]      # a week 4 pull does see it (the market's median ratio moves)
