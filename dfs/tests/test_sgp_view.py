"""dfs.sgp_view, the Parlay Sim screen: turning the legs table into legs, the book's probability from prices, and the screen end to end on the synthetic league of the
Game Projections tests (no data download; the projection model is faked there too)."""
import re
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from dfs import game_view as G
from dfs import sgp_view as V
from dfs.tests.test_game_view import FP, _line, league  # noqa: F401  (the synthetic league fixture)
from matchup_model.sim import past as P
from matchup_model.sim import sgp as S
from matchup_model.sim.history import ACTUAL_COLS

SCRIPT = "from dfs import sgp_view\nsgp_view.render(7)\n"


def _past_tables() -> P.Tables:
    """Four 2025 games of AAA: Alpha passes for 250 / 200 / 240 / 260 yards and Bravo catches for 60 / 50 / 30 / 20, so over 225.5 and over 45.5 hit 3 and 2 times, together once."""
    lines = [(w, n, v) for w, (a, b) in enumerate(zip((250, 200, 240, 260), (60, 50, 30, 20)), start=1) for n, v in (("alpha passer", {"passing_yards": a}), ("bravo wideout", {"receiving_yards": b}))]
    pw = pd.DataFrame([{"season": 2025, "week": w, "team": "AAA", "name_key": n, **v} for w, n, v in lines])
    pw = pw.reindex(columns=["season", "week", "team", "name_key", *ACTUAL_COLS]).fillna({c: 0.0 for c in ACTUAL_COLS})
    sched = pd.DataFrame([{"game_id": f"2025_{w:02d}_BBB_AAA", "season": 2025, "week": w, "away": "BBB", "home": "AAA", "away_score": 20, "home_score": 27} for w in range(1, 5)])
    return P.Tables(pw, None, sched)


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
    monkeypatch.setattr(P, "load_tables", _past_tables)
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


TWO_LEGS = pd.DataFrame([
    {"Who": "Alpha Passer", "Market": "Pass yds", "Side": "Over", "Line": 225.5, "Odds": -110.0, "Opposite odds": -110.0},
    {"Who": "Bravo Wideout Jr.", "Market": "Rec yds", "Side": "Over", "Line": 45.5, "Odds": -110.0, "Opposite odds": -110.0}])


def test_the_record_panel_shows_how_often_the_legs_hit_alone_and_together(screen):
    at = screen()
    _open_aaa_bbb(at)
    at.session_state["sgp_rows"] = TWO_LEGS
    at.run()
    assert not at.exception and any(h.value == "How often these legs have hit before" for h in at.subheader)
    tab = next(d.value for d in at.dataframe if "2026 so far" in d.value.columns)      # the pricing table also has a Leg column; this is the record
    assert list(tab.columns) == ["Leg", "2026 so far", "2025–26"]                      # the box scores in the fixture start in 2025
    by_leg = tab.set_index("Leg")
    assert by_leg.loc["Alpha Passer over 225.5 pass yds", "2025–26"] == "3 of 4 (75%)"
    assert by_leg.loc["Bravo Wideout Jr. over 45.5 rec yds", "2025–26"] == "2 of 4 (50%)"
    assert by_leg.loc["All legs together, in one game", "2025–26"] == "1 of 4 (25%)" and by_leg.loc["All legs together, in one game", "2026 so far"] == "no games"
    assert by_leg.loc["   if they had been independent in those games", "2025–26"] == "38%"         # 0.75 x 0.5
    assert any("Together since 2025: 1 of 4 games (25%)" in m.value for m in at.markdown)
    assert any(e.label == "The 4 games these players were all in" for e in at.expander)


ONE_LEG = pd.DataFrame([{"Who": "Alpha Passer", "Market": "Pass yds", "Side": "Over", "Line": 225.5, "Odds": -110.0, "Opposite odds": -110.0}])


def _picks_table(at) -> pd.DataFrame:
    return next(d.value for d in at.dataframe if "Hits when your legs hit" in d.value.columns)


def test_the_picks_panel_suggests_lines_that_hit_with_the_first_pick(screen):
    at = screen()
    _open_aaa_bbb(at)
    at.session_state["sgp_rows"] = ONE_LEG
    at.run()
    assert not at.exception and any(h.value == "Picks that tend to hit with your legs" for h in at.subheader)
    t = _picks_table(at)
    assert len(t) >= 2 and not any("Alpha Passer" in p for p in t["Pick"])                  # never a second line on the pick's own player
    assert (t["Lift"] >= 1.1).all() and t["Hits when your legs hit"].is_monotonic_decreasing
    assert any("Bravo Wideout Jr." in p for p in t["Pick"])                                  # the quarterback's wideout is a partner
    bravo = t[t["Pick"].str.contains("Bravo Wideout Jr.")].iloc[0]
    assert bravo["Line"] == "simulator median"
    assert re.fullmatch(r"\d of 4", bravo["Hit together before"]) and re.fullmatch(r"\d of 3 \(\d+%\)", bravo["When your legs hit, before"])    # four shared games in the made-up 2025 record, Alpha over in three
    next(r for r in at.radio if r.key == "sgp_rank").set_value("Lift").run()
    assert not at.exception and _picks_table(at)["Lift"].is_monotonic_decreasing


def test_picks_are_added_to_the_legs_table_with_the_edits_so_far_and_leave_the_suggestions(screen):
    at = screen()
    _open_aaa_bbb(at)
    at.session_state["sgp_rows"] = ONE_LEG
    at.run()
    first = _picks_table(at)["Pick"].iloc[0]
    next(m for m in at.multiselect if m.key == "sgp_picks_pick").select(first).run()
    next(b for b in at.button if b.key == "sgp_picks_add").click().run()
    assert not at.exception
    rows = at.session_state["sgp_rows"]
    assert len(rows) == 2 and rows.iloc[0]["Who"] == "Alpha Passer"                          # the first pick stays first, the suggestion follows
    assert first not in set(_picks_table(at)["Pick"])                                       # its player (or the game total) is now in the parlay
    assert len(next(d.value for d in at.dataframe if "Simulator prob" in d.value.columns)) == 2


def test_row_from_leg_gives_a_row_the_table_reads_back_as_the_same_leg():
    players, teams = {"Alpha Passer", "Bravo Wideout Jr."}, {"AAA", "BBB"}
    legs = [S.Leg("rec_yds", "under", 45.5, player="Bravo Wideout Jr.", p=0.55), S.Leg("any_td", "over", 0.0, player="Alpha Passer", p=0.30), S.Leg("spread", "over", -3.5, team="AAA", p=0.5),
            S.Leg("total", "under", 47.5, p=0.5), S.Leg("rush_yds", "over", 60.5, player="Alpha Passer")]
    back, errors = V.legs_from_rows(pd.DataFrame([V.row_from_leg(l) for l in legs]), players, teams)
    assert not errors and len(back) == len(legs)
    for want, got in zip(legs, back):
        assert (got.market, got.side, got.line, got.player, got.team) == (want.market, want.side, want.line, want.player, want.team)
        assert (got.p is None) == (want.p is None) and (want.p is None or abs(got.p - want.p) < 0.01)      # the prices give the probability back (to the rounding of the odds)
    assert V.row_from_leg(legs[2])["Odds"] == -110.0 and V.row_from_leg(legs[2])["Side"] == "Yes" and V.row_from_leg(legs[3])["Who"] == "Game"


def test_book_legs_turn_the_pulled_lines_into_over_legs_for_the_players_in_the_game():
    cand = pd.DataFrame({"player": ["Bravo Wideout", "Nobody Atall", "Alpha Passer", "Alpha Passer"], "market": ["rec yds", "rec yds", "pass TD", "sacks"],
                         "line": [45.5, 30.5, 1.5, 2.5], "p_over": [0.52, 0.5, 0.6, 0.5]})
    legs = V.book_legs(cand, {"Alpha Passer", "Bravo Wideout Jr."})
    assert [(l.player, l.market, l.line, l.p, l.side) for l in legs] == [("Bravo Wideout Jr.", "rec_yds", 45.5, 0.52, "over"), ("Alpha Passer", "pass_td", 1.5, 0.6, "over")]
    assert V.book_legs(pd.DataFrame(), {"Alpha Passer"}) == []


def test_the_record_panel_says_so_when_the_history_cannot_be_read_and_the_price_still_shows(screen, monkeypatch):
    def boom():
        raise RuntimeError("no box scores here")
    monkeypatch.setattr(P, "load_tables", boom)
    at = screen()
    _open_aaa_bbb(at)
    at.session_state["sgp_rows"] = TWO_LEGS
    at.run()
    assert not at.exception and any("past-game record is not available" in c.value and "no box scores here" in c.value for c in at.caption)
    assert "Chance all legs hit" in {m.label for m in at.metric}


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
