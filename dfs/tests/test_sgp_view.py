"""dfs.sgp_view, the Parlay Sim screen: turning the legs table into legs, the book's probability from prices, and the screen end to end on the synthetic league of the
Game Projections tests (no data download; the projection model is faked there too)."""
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from dfs import game_view as G
from dfs import sgp_view as V
from dfs.tests.test_game_view import FP, _line, league  # noqa: F401  (the synthetic league fixture)
from matchup_model.sim import sgp as S

SCRIPT = "from dfs import sgp_view\nsgp_view.render(7)\n"


# ------------------------------------------------------------------------------------------------------------------------------------------------ pure helpers
def test_leg_probability_uses_both_prices_when_given_and_assumes_a_vig_for_one():
    assert V.leg_probability(-110, -110) == pytest.approx(0.5)
    assert V.leg_probability(+150, -180) == pytest.approx(S.no_vig(+150, -180))
    assert V.leg_probability(-110, None) == pytest.approx(S.to_prob_american(-110) / V.ASSUMED_VIG)
    assert V.leg_probability(None, -110) is None and V.leg_probability(float("nan"), None) is None and V.leg_probability(0, -110) is None


def test_sides_map_to_over_and_under():
    assert [V.to_side(x) for x in ("Over", "Yes", "Under", "No")] == ["over", "over", "under", "under"]


def test_disagreements_flags_a_leg_the_simulator_thinks_impossible_or_far_from_the_book():
    legs = [S.Leg("rec_yds", "over", 15.5, player="Depth WR", p=0.50), S.Leg("rec_yds", "over", 60.5, player="Star WR", p=0.50), S.Leg("pass_yds", "over", 240.5, player="Backup QB", p=0.55),
            S.Leg("rush_yds", "over", 50.5, player="Back", p=None)]
    flagged = V.disagreements(legs, [0.0004, 0.46, 0.10, 0.0])
    assert len(flagged) == 2 and "Depth WR" in flagged[0] and "Backup QB" in flagged[1]       # a leg with no book price has nothing to disagree with


def test_legs_from_rows_reports_what_it_cannot_use():
    rows = pd.DataFrame([
        {"Who": "Alpha Passer", "Market": "Pass yds", "Side": "Over", "Line": 245.5, "Odds": -115, "Opposite odds": -105},
        {"Who": "AAA", "Market": "Team spread", "Side": "Yes", "Line": -3.5, "Odds": None, "Opposite odds": None},
        {"Who": "Game", "Market": "Game total", "Side": "Under", "Line": 47.5, "Odds": -110, "Opposite odds": None},
        {"Who": "Nobody", "Market": "Rec yds", "Side": "Over", "Line": 50.5, "Odds": None, "Opposite odds": None},
        {"Who": "Alpha Passer", "Market": "Pass TD", "Side": "Over", "Line": None, "Odds": None, "Opposite odds": None},
        {"Who": "Alpha Passer", "Market": "Team total", "Side": "Over", "Line": 24.5, "Odds": None, "Opposite odds": None},
    ])
    legs, errors = V.legs_from_rows(rows, {"Alpha Passer", "Golf Passer"}, {"AAA", "BBB"})
    assert [l.market for l in legs] == ["pass_yds", "spread", "total"]
    assert legs[0].p == pytest.approx(S.no_vig(-115, -105)) and legs[1].team == "AAA" and legs[1].p is None and legs[2].side == "under"
    assert len(errors) == 3 and any("not in this game" in e for e in errors) and any("needs a line" in e for e in errors) and any("needs a team" in e for e in errors)


# ------------------------------------------------------------------------------------------------------------------------------------------------ the screen
@pytest.fixture()
def screen(monkeypatch, league, tmp_path):  # noqa: F811
    from dfs import props
    monkeypatch.setattr(G, "_season", lambda: 2026)
    monkeypatch.setattr(G, "_games", lambda season, week: G.week_games(season, week))
    monkeypatch.setattr(G, "_load_fp", lambda week: FP)
    monkeypatch.setattr(G, "_fp_path", lambda week: tmp_path / f"projections.week{week}.csv")
    monkeypatch.setattr(G, "_table", lambda season, week, away, home, h: G.game_table(away, home, season, week, FP, line_fn=_line))
    monkeypatch.setattr(props, "load_last_pull", lambda: None)
    real_default = G.default_game_index
    monkeypatch.setattr(G, "default_game_index", lambda games, now=None: real_default(games, datetime(2026, 10, 21, 9, 0)))
    V._simulate.clear()
    from streamlit.testing.v1 import AppTest
    return lambda: AppTest.from_string(SCRIPT, default_timeout=120).run()


def _open_aaa_bbb(at):
    next(s for s in at.selectbox if s.key == "sgp_game").select("AAA @ BBB · Sun Oct 25 1:00 PM ET").run()
    return at


def test_the_screen_opens_and_asks_for_legs(screen):
    at = screen()
    assert not at.exception and at.header[0].value == "🎲 DFS Parlay Sim"
    _open_aaa_bbb(at)
    assert not at.exception and any("Add at least one leg" in i.value for i in at.info)
    assert any("the simulation averages" in c.value for c in at.caption)


def test_a_parlay_is_priced_from_the_legs_in_the_table(screen):
    at = screen()
    _open_aaa_bbb(at)
    at.session_state["sgp_rows"] = pd.DataFrame([
        {"Who": "Alpha Passer", "Market": "Pass yds", "Side": "Over", "Line": 225.5, "Odds": -110.0, "Opposite odds": -110.0},
        {"Who": "Bravo Wideout Jr.", "Market": "Rec yds", "Side": "Over", "Line": 45.5, "Odds": -110.0, "Opposite odds": -110.0}])
    at.run()
    assert not at.exception
    m = {x.label: x.value for x in at.metric}
    assert set(m) >= {"Chance all legs hit", "Fair odds", "If legs were independent", "Correlation lift"}
    assert m["If legs were independent"] == "25.0%" or m["If legs were independent"].startswith("25")
    assert float(m["Correlation lift"].rstrip("x")) > 1.0                       # a quarterback and his wideout move together
    offered = next(n for n in at.number_input if n.key == "sgp_offer")
    offered.set_value(400).run()
    assert any("break even" in x.value for x in at.markdown) and not at.exception


def test_a_leg_for_a_player_who_is_not_in_the_game_is_flagged_not_fatal(screen):
    at = screen()
    _open_aaa_bbb(at)
    at.session_state["sgp_rows"] = pd.DataFrame([{"Who": "Alpha Passer", "Market": "Pass yds", "Side": "Over", "Line": 225.5, "Odds": None, "Opposite odds": None},
                                                 {"Who": "Nobody Atall", "Market": "Rec yds", "Side": "Over", "Line": 40.5, "Odds": None, "Opposite odds": None}])
    at.run()
    assert not at.exception and any("not in this game" in w.value for w in at.warning)
    assert any("simulator's own probabilities" in c.value for c in at.caption)      # no odds entered: the simulator's own numbers


def test_the_correlation_map_puts_a_quarterback_pair_at_the_top_and_leaves_out_one_player_against_himself():
    from matchup_model.sim import engine as E
    from matchup_model.tests.test_sim import _team
    home = _team("HOM", 25.0, "Home QB", ("Home WR1", "Home WR2", "Home WR3"), "Home RB")
    away = _team("AWY", 20.0, "Away QB", ("Away WR1", "Away WR2", "Away WR3"), "Away RB")
    res = E.simulate(home, away, n=12000, seed=5)
    t = V.corr_map(res, home, away)
    assert not t.empty and t["Phi"].iloc[0] > 0.2 and "QB" in t["Pair"].iloc[0]
    assert t["Phi"].iloc[0] >= t["Phi"].iloc[-1]
    for pair in t["Pair"]:
        a, b = pair.split("  +  ")
        assert a.split(" over ")[0] != b.split(" over ")[0] or "anytime" in a + b       # never a player against himself
