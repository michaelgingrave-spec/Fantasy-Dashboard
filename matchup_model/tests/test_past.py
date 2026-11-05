"""matchup_model.sim.past: how often a parlay's legs have hit in past games, on a small made-up league (three teams, two seasons) whose every number is checked by hand."""
import numpy as np
import pandas as pd
import pytest

from matchup_model.sim import past as P
from matchup_model.sim import sgp as S
from matchup_model.sim.history import ACTUAL_COLS

# (away, home, away score, home score) by (season, week); 2026 week 3 is "the game on screen", week 4 is not played yet, 2024 is before the box scores start
GAMES = {(2024, 1): ("BBB", "AAA", 10, 31),
         (2025, 1): ("BBB", "AAA", 20, 27), (2025, 2): ("AAA", "BBB", 17, 24), (2025, 3): ("CCC", "AAA", 10, 30), (2025, 4): ("AAA", "BBB", 21, 23),
         (2026, 1): ("BBB", "AAA", 17, 24), (2026, 2): ("AAA", "CCC", 13, 27), (2026, 3): ("BBB", "AAA", 10, 40), (2026, 4): ("AAA", "BBB", np.nan, np.nan)}
# (season, week, team, name_key, stats): who has a stat line; Bravo has none in 2025 week 3 (inactive) and a snap but no stat line in 2025 week 2
LINES = [(2025, 1, "AAA", "alpha passer", {"passing_yards": 260}), (2025, 2, "AAA", "alpha passer", {"passing_yards": 200}), (2025, 3, "AAA", "alpha passer", {"passing_yards": 300}),
         (2025, 4, "AAA", "alpha passer", {"passing_yards": 225}), (2026, 1, "AAA", "alpha passer", {"passing_yards": 250}), (2026, 2, "AAA", "alpha passer", {"passing_yards": 180}),
         (2026, 3, "AAA", "alpha passer", {"passing_yards": 400}),
         (2025, 1, "AAA", "bravo wideout", {"receiving_yards": 80}), (2025, 4, "AAA", "bravo wideout", {"receiving_yards": 60}), (2026, 1, "AAA", "bravo wideout", {"receiving_yards": 45}),
         (2026, 2, "AAA", "bravo wideout", {"receiving_yards": 90, "receiving_tds": 1}), (2026, 3, "AAA", "bravo wideout", {"receiving_yards": 100}),
         (2025, 1, "BBB", "charlie back", {"rushing_yards": 60, "rushing_tds": 1}), (2025, 2, "BBB", "charlie back", {"rushing_yards": 40}),
         (2025, 4, "BBB", "charlie back", {"rushing_yards": 80}), (2026, 1, "BBB", "charlie back", {"rushing_yards": 70, "rushing_tds": 1}), (2026, 3, "BBB", "charlie back", {"rushing_yards": 120})]
SNAPS = [(2025, 2, "AAA", "bravo wideout", 31.0), (2025, 3, "AAA", "bravo wideout", 0.0), (2025, 3, "AAA", "alpha passer", 60.0)]      # week 3: Bravo on the field for no offensive snap
SEASON, WEEK = 2026, 3
TEAMS = ("BBB", "AAA")                                                                                                          # (away, home) of the game on screen


def _tables(snaps: bool = True) -> P.Tables:
    pw = pd.DataFrame([{"season": s, "week": w, "team": t, "name_key": n, **v} for s, w, t, n, v in LINES])
    pw = pw.reindex(columns=["season", "week", "team", "name_key", *ACTUAL_COLS]).fillna({c: 0.0 for c in ACTUAL_COLS})
    sched = pd.DataFrame([{"game_id": f"{s}_{w:02d}_{a}_{h}", "season": s, "week": w, "away": a, "home": h, "away_score": asc, "home_score": hsc} for (s, w), (a, h, asc, hsc) in GAMES.items()])
    sn = pd.DataFrame(SNAPS, columns=["season", "week", "team", "name_key", "offense_snaps"])
    return P.Tables(pw, sn if snaps else None, sched)


def _leg(market: str, side: str, line: float, **kw) -> S.Leg:
    return S.Leg(market, side, line, **kw)


ALPHA = _leg("pass_yds", "over", 224.5, player="Alpha Passer")
BRAVO = _leg("rec_yds", "over", 54.5, player="Bravo Wideout Jr.")                      # the suffix is dropped when names are matched, as everywhere else
CHARLIE = _leg("rush_yds", "over", 50.5, player="Charlie Back")


def _past(legs, snaps: bool = True, week: int = WEEK) -> P.Past:
    return P.past_hits(legs, SEASON, week, TEAMS, _tables(snaps))


# ------------------------------------------------------------------------------------------------------------------------------------------------ one leg
def test_a_leg_alone_is_counted_over_every_past_game_and_not_the_game_on_screen():
    p = _past([ALPHA])
    assert p.windows == ["2026 so far", "2025–26"] and p.first == 2025                # the box scores start in 2025, so there is no third window
    assert p.legs[0]["cells"] == [(1, 2), (4, 6)]                                      # 260, 225 and 250 over 224.5; 200 and 180 not; the 400 of week 3 is the game on screen
    assert _past([ALPHA], week=4).legs[0]["cells"] == [(2, 3), (5, 7)]                 # a week later the 400 is history
    assert _past([ALPHA], week=1).legs[0]["cells"] == [(0, 0), (3, 4)]                 # in week 1 there is no "so far" yet, and the 2024 game (no box scores) is not read


def test_a_stat_exactly_on_a_whole_number_line_is_a_miss_on_both_sides():
    over, under = (_past([_leg("pass_yds", side, 250.0, player="Alpha Passer")]).legs[0]["cells"][0] for side in ("over", "under"))
    assert over == (0, 2) and under == (1, 2)                                          # 2026: 250 (a push) and 180


def test_anytime_td_and_its_no_side_read_the_touchdowns():
    yes = _past([_leg("any_td", "over", 0.0, player="Bravo Wideout")]).legs[0]["cells"]
    no = _past([_leg("any_td", "under", 0.0, player="Bravo Wideout")]).legs[0]["cells"]
    assert yes[1] == (1, 5) and no[1] == (4, 5)                                        # one receiving touchdown (2026 week 2) in five games


def test_a_game_he_played_without_a_stat_line_counts_and_an_inactive_game_does_not():
    assert _past([_leg("rec_yds", "under", 20.5, player="Bravo Wideout")]).legs[0]["cells"][1] == (1, 5)        # week 2 of 2025: a snap, no stat line, so 0 yards: the under hits
    assert _past([_leg("rec_yds", "under", 20.5, player="Bravo Wideout")], snaps=False).legs[0]["cells"][1] == (0, 4)   # without snap counts that game is not known to have been played
    assert _past([_leg("rec_yds", "over", 0.5, player="Bravo Wideout")]).legs[0]["games"] == 5                        # 2025 week 3: on the roster, no snap, no stat line


# ------------------------------------------------------------------------------------------------------------------------------------------------ legs together
def test_together_needs_every_player_in_the_game_and_every_leg_to_hit():
    p = _past([ALPHA, BRAVO])
    assert p.together == [(0, 2), (2, 5)]                                              # Bravo missed 2025 week 3, so five shared games; both hit in 2025 weeks 1 and 4
    assert p.last_given_rest == [(0, 1), (2, 3)]                                       # Alpha hit in 1 of the 2026 shared games and 3 of the five; Bravo went with him in 0 and 2
    assert _past([ALPHA]).last_given_rest == [(1, 2), (4, 6)]                          # one leg: nothing to condition on, so the second number is every game he played
    assert p.independent[1] == pytest.approx(0.6 * 0.6) and p.independent[0] == pytest.approx(0.5 * 0.5)
    assert list(p.games["All legs"]) == ["✗", "✗", "✓", "✗", "✓"] and list(p.games["Wk"]) == [2, 1, 4, 2, 1]
    assert p.games.iloc[0]["Alpha Passer over 224.5 pass yds"] == "180 ✗" and p.games.iloc[0]["Bravo Wideout Jr. over 54.5 rec yds"] == "90 ✓" and p.games.iloc[0]["Game"] == "AAA @ CCC"
    assert _past([ALPHA, BRAVO], snaps=False).together[1] == (2, 4)                    # the snap-only game (his 0 yards) is one fewer shared game without snap counts


def test_players_on_opposite_teams_are_together_only_in_the_games_their_teams_played():
    p = _past([ALPHA, CHARLIE])
    assert p.together == [(1, 1), (3, 4)]                                              # AAA-BBB games: 2025 weeks 1, 2, 4 and 2026 week 1, not the games against CCC


def test_team_legs_read_the_scores_and_a_game_total_belongs_to_both_teams():
    spread, team_total = _leg("spread", "over", -3.5, team="AAA"), _leg("team_total", "over", 22.5, team="AAA")
    total, win = _leg("total", "under", 44.5), _leg("moneyline", "over", 0.0, team="AAA")
    p = _past([spread, team_total, total, win])
    assert [l["cells"][1] for l in p.legs] == [(3, 6), (3, 6), (5, 6), (3, 6)]          # AAA margins +7 -7 +20 -2 +7 -14; points 27 17 30 21 24 13; totals 47 41 40 44 41 40
    assert p.together == [(1, 2), (2, 6)] and p.games.iloc[0]["AAA -3.5"] == "-14 ✗"
    assert _past([_leg("total", "under", 44.5)]).legs[0]["cells"][1] == (5, 6)         # the union of BBB's and AAA's games is the same six here


def test_a_player_who_is_not_in_the_box_scores_leaves_no_record_and_the_rest_still_reads():
    p = _past([ALPHA, _leg("rec_yds", "over", 40.5, player="Nobody Atall")])
    assert p.legs[1]["games"] == 0 and p.legs[0]["games"] == 6
    assert p.together == [(0, 0), (0, 0)] and p.independent == [None, None] and p.games.empty and list(p.games.columns)[-1] == "All legs"


# ------------------------------------------------------------------------------------------------------------------------------------------------ helpers
def test_on_side_is_the_rule_the_simulator_uses_and_a_number_on_the_line_misses_both_ways():
    x = np.array([40.0, 50.0, 60.0])
    assert list(S.on_side(_leg("rec_yds", "over", 50.0, player="X"), x)) == [False, False, True]
    assert list(S.on_side(_leg("rec_yds", "under", 50.0, player="X"), x)) == [True, False, False]
    assert list(S.on_side(_leg("spread", "over", -3.5, team="AAA"), np.array([3.0, 4.0]))) == [False, True]      # a 3.5-point favourite covers by winning by 4
    assert list(S.on_side(_leg("moneyline", "under", 0.0, team="AAA"), np.array([-3.0, 0.0, 3.0]))) == [True, False, False]
    assert list(S.on_side(_leg("any_td", "under", 0.0, player="X"), np.array([0, 1, 2]))) == [True, False, False]


def test_wilson_range_is_wide_for_a_few_games_and_stays_inside_zero_and_one():
    lo, hi = P.wilson(4, 8)
    assert lo == pytest.approx(0.249, abs=0.002) and hi == pytest.approx(0.751, abs=0.002)
    lo, hi = P.wilson(0, 3)
    assert lo == 0.0 and 0.3 < hi < 0.6
    assert P.wilson(0, 0) == (0.0, 1.0)
    assert P.wilson(200, 400)[1] - P.wilson(200, 400)[0] < 0.1
