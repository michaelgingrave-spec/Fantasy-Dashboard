"""matchup_model.opp.role_rebase on the synthetic league of test_injury_absorption: W1 plays weeks 1-10 and is Out in 11 (new) and 12 (ongoing)."""
import numpy as np
import pandas as pd
import pytest

from matchup_model.opp import injury_absorption as IA
from matchup_model.opp import role_rebase as RR
from matchup_model.tests import test_injury_absorption as T


@pytest.fixture(scope="module")
def league():
    return T._league()


def _ongoing(league, late_team=None):
    base, inj, snaps = league
    if late_team:                                         # the key player joins that team in week 7: before that he was not 'missing' games, he was not there
        drop = (base["team"] == late_team) & (base["player_display_name"] == "W1") & (base["week"] < 7)
        base = base[~drop]
        snaps = snaps[~((snaps["team"] == late_team) & (snaps["name_key"] == "w1") & (snaps["week"] < 7))]
    obs = IA.observations("rec", base, inj, snaps)
    return RR.without_features("rec", IA.ongoing_absences(obs), base, snaps), base, inj, snaps


def test_a_teammate_has_one_game_without_the_key_player_in_an_ongoing_absence(league):
    d, *_ = _ongoing(league)
    w2 = d[d["name"] == "w2"].iloc[0]                                                 # names are the normalised keys
    assert w2["n_wo"] == 1 and w2["wo"] == pytest.approx(0.20 * 1.15) and w2["since"] == pytest.approx(0.20 * 1.15) and w2["n_since"] == 1
    assert w2["n_in"] == 9 and w2["w_in"] == pytest.approx(0.20)                     # weeks 2-10: the window is the ten team games before week 12, and he played nine of them
    assert w2["key_name"] == "w1" and set(d["name"]) == {"w2", "w3", "t1"}           # DEPTH was inactive: not a recipient


def test_games_before_the_key_player_arrived_are_not_games_he_missed(league):
    d, *_ = _ongoing(league, late_team="T00")
    for team, n in (("T00", 1), ("T01", 1)):
        assert (d[d["team"] == team]["n_wo"] == n).all()                              # only week 11; weeks 1-6 on T00 predate him


def test_a_new_absence_has_no_games_without_him_to_learn_from(league):
    base, inj, snaps = league
    obs = IA.new_absences(IA.observations("rec", base, inj, snaps))
    d = RR.without_features("rec", obs, base, snaps)
    assert (d["n_wo"] == 0).all() and d["wo"].isna().all() and d["n_in"].min() >= 4


def test_the_production_rule_is_exact_in_this_league_and_the_split_is_not(league):
    d, *_ = _ongoing(league)
    p = RR.with_predictions(d)
    assert np.allclose(p["P_K"], p["a"], atol=1e-9)                                  # an ongoing absence adds nothing here, and a == the trailing share
    r = RR.evaluate(p)
    assert r["rmse"]["K"] < 1e-6 < r["rmse"]["WO"] and r["rmse"]["K"] < r["rmse"]["A"] + 1e-9
    assert r["rows"] == len(d) and r["team_weeks"] == len(T.SEASONS) * len(T.TEAMS) and r["mean_n_wo"] == 1.0
    assert {s["season"] for s in r["by_season"]} == set(T.SEASONS) and r["since"]["rows"] == len(d)


def test_the_more_the_blend_leans_on_a_wrong_split_the_worse_it_does(league):
    d, *_ = _ongoing(league)
    rng = np.random.default_rng(5)
    d = d.assign(a=d["a"] + rng.normal(0, 0.01, len(d)))                             # a little noise, so the production rule is good but not exact
    r = RR.evaluate(RR.with_predictions(d))
    assert r["blend"][0.5][0] > r["blend"][2.0][0] > r["blend"][8.0][0] > 0 and r["wo_vs_k"][0] > r["blend"][0.5][0]


def test_the_blend_is_the_split_shrunk_toward_the_rule_by_its_sample_size():
    d = pd.DataFrame({"n_wo": [1, 3], "wo": [0.30, 0.20], "wo_ew": [0.30, 0.20], "P_K": [0.10, 0.10]})
    assert list(RR.blend(d, 2.0)) == pytest.approx([(1 * 0.30 + 2 * 0.10) / 3, (3 * 0.20 + 2 * 0.10) / 5])
    assert list(RR.blend(d, 0.0)) == pytest.approx([0.30, 0.20])                      # no shrinkage: the split alone


def test_the_clustered_interval_brackets_a_known_change():
    d = pd.DataFrame({"season": [2022] * 20, "week": list(range(1, 21)), "team": "AAA"})
    e0 = pd.Series(np.linspace(-0.05, 0.05, 20))
    p, (lo, hi) = RR.pct_change(d, 2 * e0, e0, reps=100)
    assert p == pytest.approx(100.0) and lo == pytest.approx(100.0) and hi == pytest.approx(100.0)       # twice the error everywhere: +100% in every resample
    p0, (lo0, hi0) = RR.pct_change(d, e0, e0, reps=100)
    assert p0 == pytest.approx(0.0) and lo0 == pytest.approx(0.0) and hi0 == pytest.approx(0.0)


def test_too_few_teammates_gives_no_evaluation(league):
    d, *_ = _ongoing(league)
    assert RR.evaluate(RR.with_predictions(d.iloc[:10])) is None


def test_run_covers_the_ongoing_absences_and_skips_what_has_no_data(league):
    base, inj, snaps = league
    res = RR.run(base, inj, snaps)
    assert res[("rec", "ONGOING")]["rows"] == len(T.SEASONS) * len(T.TEAMS) * 3
    assert res.get(("rec", "NEW")) is None and not any(k[0] == "rush" for k in res)       # no one to score on a new absence, and nobody carries in this league


def test_the_report_has_every_table_and_the_reading(league):
    base, inj, snaps = league
    txt = RR.build_report(RR.run(base, inj, snaps))
    for needle in ("## Receiving (target share, WR/TE/RB) - ONGOING absence", "## Rushing (carry share, RB) - NEW absence", "| shrinkage n |", "By season", "## Reading (2026-10-22)",
                   "Too few teammates", "BL(2)"):
        assert needle in txt
    assert RR.REPORT.name == "role_rebase_report.md" and RR.REPORT.parent.name == "matchup_model"
