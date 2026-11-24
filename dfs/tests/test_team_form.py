"""The as-of team form behind the Millionaire Maker environment rules: nothing from the game itself leaks in, the direction of the ranks is right, and a pool gets the right opponent."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dfs import team_form as T

TEAMS = ["AAA", "BBB", "CCC", "DDD"]
PAIRS = {1: [("AAA", "BBB"), ("CCC", "DDD")], 2: [("AAA", "CCC"), ("BBB", "DDD")], 3: [("AAA", "DDD"), ("BBB", "CCC")]}


def make_games(pa_bump: float = 0.0) -> pd.DataFrame:
    """Three weeks of a four-team league. Team k scores 10 + 5k points and allows 20 - 3k (so DDD has the best defense and the best offense); week 3's results can be shifted."""
    idx = {t: i for i, t in enumerate(TEAMS)}
    rows = []
    for week, games in PAIRS.items():
        for home, away in games:
            for team, opp in ((home, away), (away, home)):
                k = idx[team]
                bump = pa_bump if week == 3 else 0.0
                rows.append({"season": 2020, "week": week, "team": team, "opp": opp, "pf": 10.0 + 5 * k + bump, "pa": 20.0 - 3 * k + bump,
                             "fpa_qb": 15.0 + 2 * (3 - k) + bump, "fpa_rb": 20.0, "fpa_wr": 30.0, "fpa_te": 8.0,
                             "o_pass_sum": 0.1 * k * 35, "o_pass_n": 35.0, "o_rush_sum": 0.0, "o_rush_n": 25.0,
                             "d_pass_sum": 0.1 * (3 - k) * 35, "d_pass_n": 35.0, "d_rush_sum": 0.05 * (3 - k) * 25, "d_rush_n": 25.0})
    return pd.DataFrame(rows)


def test_as_of_never_includes_the_week_itself():
    a = T.build_as_of(make_games(), [2020])
    b = T.build_as_of(make_games(pa_bump=40.0), [2020])                      # week 3's results are wildly different ...
    key = ["season", "week", "team"]
    wk3 = a[a.week == 3].merge(b[b.week == 3], on=key, suffixes=("_a", "_b"))
    assert (wk3["def_pa_a"] == wk3["def_pa_b"]).all() and (wk3["def_fpa_qb_a"] == wk3["def_fpa_qb_b"]).all()       # ... and week 3's as-of values do not change
    assert (wk3["games_a"] == 2).all()


def test_first_week_has_nothing_and_second_week_is_the_first_game_with_a_prior():
    a = T.build_as_of(make_games(), [2020]).set_index(["week", "team"])
    assert a.loc[(1, "AAA"), "games"] == 0 and np.isnan(a.loc[(1, "AAA"), "def_pa"])                 # no earlier game, no last season: nothing to say
    # week 2: team AAA allowed 20 in week 1; the prior is the league's week-1 mean (20 - 3 * 1.5 = 15.5) worth PRIOR_GAMES games
    k = T.PRIOR_GAMES
    assert a.loc[(2, "AAA"), "def_pa"] == pytest.approx((20.0 + k * 15.5) / (1 + k))
    assert a.loc[(2, "DDD"), "def_pa"] == pytest.approx((11.0 + k * 15.5) / (1 + k))


def test_rank_one_is_the_weakest_defense_and_the_weakest_offense():
    a = T.build_as_of(make_games(), [2020])
    w3 = a[a.week == 3].set_index("team")
    assert w3.loc["AAA", "def_pa_rank"] == 1 and w3.loc["DDD", "def_pa_rank"] == 4                    # AAA allows the most
    assert w3.loc["AAA", "def_fpa_qb_rank"] == 1 and w3.loc["DDD", "def_fpa_qb_rank"] == 4
    assert w3.loc["AAA", "off_pf_weak_rank"] == 1 and w3.loc["DDD", "off_pf_weak_rank"] == 4         # AAA scores the least
    assert w3.loc["DDD", "off_pf_rank"] == 1                                                         # and DDD is the best offense
    assert w3.loc["AAA", "def_pass_epa_rank"] == 1 and w3.loc["DDD", "def_pass_epa_rank"] == 4        # the EPA ranks run the same way
    assert w3.loc["AAA", "def_pass_epa"] > w3.loc["DDD", "def_pass_epa"]                              # AAA's defense allowed more EPA


def test_attach_gives_each_player_the_unit_he_faces():
    a = T.build_as_of(make_games(), [2020])
    pool = pd.DataFrame({"team": ["AAA", "AAA", "DDD", "DDD"], "opp": ["DDD", "DDD", "AAA", "AAA"], "pos": ["QB", "RB", "QB", "DST"], "name": list("abcd")})
    out = T.attach(pool, 2020, 3, table=a)
    w3 = a[a.week == 3].set_index("team")
    # AAA's QB faces DDD's pass defense (strong -> a high rank number); DDD's QB faces AAA's (weakest -> rank 1)
    assert out.loc[0, "env_fpa"] == pytest.approx(w3.loc["DDD", "def_fpa_qb"]) and out.loc[0, "env_fpa_rank"] == 4
    assert out.loc[2, "env_fpa_rank"] == 1 and out.loc[2, "env_epa_rank"] == w3.loc["AAA", "def_pass_epa_rank"]
    assert out.loc[1, "env_epa"] == pytest.approx(w3.loc["DDD", "def_rush_epa"]) and out.loc[1, "env_epa"] != pytest.approx(w3.loc["DDD", "def_pass_epa"])   # an RB: the rush defense
    # DDD's DST faces AAA's offense, the weakest scoring offense -> rank 1; its env_fpa is that offense's points per game
    assert out.loc[3, "env_fpa_rank"] == 1 and out.loc[3, "env_fpa"] == pytest.approx(w3.loc["AAA", "off_pf"])
    assert out.loc[0, "team_off_rank"] == w3.loc["AAA", "off_pf_rank"]
    assert set(T.ENV_COLUMNS) <= set(out.columns)
    assert T.attach(pool, 2020, 9, table=a)["env_fpa"].isna().all()                                    # a week with no table rows leaves NaN, not an error


def test_last_seasons_finish_is_the_prior():
    g1 = make_games().assign(season=2019)
    g2 = make_games().assign(season=2020)
    a = T.build_as_of(pd.concat([g1, g2], ignore_index=True), [2019, 2020]).set_index(["season", "week", "team"])
    first = a.loc[(2020, 1, "AAA")]
    assert first["games"] == 0 and first["def_pa"] == pytest.approx(0.5 * 20.0 + 0.5 * np.mean([20.0, 17.0, 14.0, 11.0]) * 1)   # half of AAA's 2019 finish, half the league's


def test_the_week_after_the_last_game_is_the_form_for_the_slate_being_built():
    a = T.build_as_of(make_games(), [2020]).set_index(["week", "team"])
    assert a.index.get_level_values("week").max() == 4                                                # weeks 1-3 were played, so week 4 is the one being built
    assert (a.xs(4, level="week")["games"] == 3).all()                                                # it has seen all three
    assert a.loc[(4, "AAA"), "def_pa"] == pytest.approx((3 * 20.0 + T.PRIOR_GAMES * 15.5) / (3 + T.PRIOR_GAMES))
