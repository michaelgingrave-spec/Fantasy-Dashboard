"""matchup_model.sim.recommend: the lines that tend to hit with a parlay's legs, on the made-up game of the simulator tests."""
from __future__ import annotations

import pytest

from matchup_model.sim import engine as E
from matchup_model.sim import recommend as R
from matchup_model.sim import sgp as S
from matchup_model.tests.test_sim import _team

HOME = _team("HOM", 25.0, "Home QB", ("Home WR1", "Home WR2", "Home WR3"), "Home RB")
AWAY = _team("AWY", 20.0, "Away QB", ("Away WR1", "Away WR2", "Away WR3"), "Away RB")
GAME = {"home": "HOM", "away": "AWY", "spread_home": -3.0, "total": 45.0}           # the home team is a 3-point underdog
QB = S.Leg("pass_yds", "over", 250.5, player="Home QB", p=0.5)


@pytest.fixture(scope="module")
def res():
    return E.simulate(HOME, AWAY, n=20000, seed=5).compact()


def test_median_legs_are_about_coin_flips_and_only_a_quarterback_gets_a_passing_line(res):
    legs = R.median_legs(res, HOME, AWAY)
    assert legs and all(l.side == "over" for l in legs)
    for l in legs:
        if l.market not in ("any_td", "pass_td"):
            assert 0.25 < S.leg_prob(res, l) < 0.65, l.label()                  # a whole-number stat like receptions sits a little under a coin flip at its median + 0.5
    assert {l.player for l in legs if l.market == "pass_yds"} == {"Home QB", "Away QB"}


def test_candidates_have_both_sides_book_lines_replace_medians_and_the_game_adds_its_spreads_and_total(res):
    book = [S.Leg("rec_yds", "over", 66.5, player="Home WR1", p=0.52)]
    c = R.candidate_legs(res, HOME, AWAY, book, GAME)
    wr1 = [l for l in c if l.player == "Home WR1" and l.market == "rec_yds"]
    assert [(l.side, l.line, l.p) for l in wr1] == [("over", 66.5, 0.52), ("under", 66.5, pytest.approx(0.48))]       # the book's line, not a median line
    medians = [l for l in c if l.player == "Home WR2" and l.market == "rec_yds"]
    assert {l.side for l in medians} == {"over", "under"} and all(l.p is None for l in medians)
    assert [(l.side, l.line, l.p) for l in c if l.market == "total"] == [("over", 45.0, 0.5), ("under", 45.0, 0.5)]
    assert [(l.team, l.line) for l in c if l.market == "spread"] == [("HOM", 3.0), ("AWY", -3.0)]                    # the underdog's handicap is +3
    assert not [l for l in R.candidate_legs(res, HOME, AWAY) if l.market in S.TEAM_MARKETS]                          # no game given: player lines only
    stray = R.candidate_legs(res, HOME, AWAY, [S.Leg("rec_yds", "over", 50.5, player="Nobody", p=0.5)])
    assert all(l.player != "Nobody" for l in stray)


def test_the_quarterbacks_wideouts_come_first_and_players_already_in_the_parlay_are_left_out(res):
    cands = R.candidate_legs(res, HOME, AWAY, None, GAME)
    picks = R.recommend(res, [QB], cands)
    assert not picks.empty and picks["who"].iloc[0] in ("Home WR1", "Home WR2", "Home WR3", "HOM TE")
    assert "Home QB" not in set(picks["who"]) and picks["who"].is_unique                                            # one line per player, never the leg's own player
    assert list(picks["when_given"]) == sorted(picks["when_given"], reverse=True)
    assert (picks["lift"] >= R.MIN_LIFT).all() and picks["alone"].between(*R.BAND).all()
    top = picks.iloc[0]
    assert top["when_given"] == pytest.approx(top["joint"] / 0.5) and top["lift"] == pytest.approx(top["when_given"] / top["alone"])
    assert top["fair_odds"] == pytest.approx(S.to_american(top["joint"]))
    by_lift = R.recommend(res, [QB], cands, rank="lift")
    assert list(by_lift["lift"]) == sorted(by_lift["lift"], reverse=True)
    with pytest.raises(ValueError):
        R.recommend(res, [QB], cands, rank="best")


def test_a_book_line_keeps_the_books_probability_and_the_other_lines_say_where_they_come_from(res):
    book = [S.Leg("rec_yds", "over", 60.5, player="Home WR1", p=0.52)]
    cands = R.candidate_legs(res, HOME, AWAY, book, GAME)
    wr1 = R.recommend(res, [QB], [l for l in cands if l.player == "Home WR1" and l.market == "rec_yds"]).iloc[0]
    assert wr1["source"] == "book line" and wr1["alone"] == pytest.approx(0.52) and wr1["leg"].side == "over" and wr1["when_given"] > 0.52
    sources = set(R.recommend(res, [QB], cands, top=40)["source"])
    assert sources <= {"book line", "simulator median", "market line"} and {"simulator median", "market line"} <= sources
    assert set(R.recommend(res, [QB], R.candidate_legs(res, HOME, AWAY, None, None), top=40)["source"]) == {"simulator median"}


def test_a_spread_or_total_already_in_the_parlay_gets_no_second_line_and_the_game_total_can_be_suggested(res):
    cands = R.candidate_legs(res, HOME, AWAY, None, GAME)
    over = S.Leg("total", "over", 45.0, p=0.5)
    assert "game" not in set(R.recommend(res, [over], cands)["who"])
    cover = S.Leg("spread", "over", 3.0, team="HOM", p=0.5)
    assert not {"HOM", "AWY"} & set(R.recommend(res, [cover], cands)["who"])
    assert "game" in set(R.recommend(res, [QB], cands, top=40)["who"])                                              # more passing yards, more points: the total is a partner


def test_nothing_is_returned_when_no_line_clears_the_filters(res):
    picks = R.recommend(res, [QB], R.candidate_legs(res, HOME, AWAY, None, GAME), min_lift=5.0)
    assert picks.empty and list(picks.columns) == R.COLUMNS
