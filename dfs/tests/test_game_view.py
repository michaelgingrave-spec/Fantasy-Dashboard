"""dfs.game_view, the Game Projections screen: the week's games, who is on a team's roster, the join to FantasyPoints, the table and its filters, and the screen itself.
nflverse (schedule, game logs, injuries) and the projection model are replaced by small synthetic frames, so nothing here needs a data download."""
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from dfs import game_view as G
from matchup_model.opp import data as D


# ------------------------------------------------------------------------------------------------------------------------------------------------ synthetic league
def _pw(rows):
    """Game-log rows: (player name, position, team, [weeks played])."""
    out = []
    for i, (name, pos, team, weeks) in enumerate(rows):
        for wk in weeks:
            out.append({"season": 2026, "week": wk, "player_id": f"id{i}", "player_display_name": name, "position": pos, "team": team,
                        "name_key": G.normalize_name(name)})
    return pd.DataFrame(out)


LOGS = _pw([
    ("Alpha Passer", "QB", "AAA", [3, 4, 5, 6]),
    ("Bravo Wideout", "WR", "AAA", [4, 5, 6]),
    ("Charlie Runner", "RB", "AAA", [3]),                    # not in AAA's last three games (4, 5, 6)
    ("Delta Traveler", "WR", "AAA", [3, 4, 5]),              # ...then played for BBB in week 6
    ("Echo Kicker", "K", "AAA", [3, 4, 5, 6]),               # not a skill position
    ("Foxtrot Tight", "TE", "AAA", [6]),
    ("Golf Passer", "QB", "BBB", [3, 4, 5, 6]),
    ("Hotel Wideout", "WR", "BBB", [3, 4, 5, 6]),
    ("Delta Traveler", "WR", "BBB", [6]),
])
LOGS = LOGS.assign(player_id=LOGS["player_display_name"].map(lambda n: "id-" + n))            # one id per human
SCHEDULE = pd.DataFrame({
    "game_id": ["2026_07_AAA_BBB", "2026_07_CCC_DDD", "2026_07_EEE_FFF"], "season": 2026, "week": 7,
    "kickoff": pd.to_datetime(["2026-10-25 13:00", "2026-10-22 20:15", "2026-10-26 20:15"]),
    "away": ["AAA", "CCC", "EEE"], "home": ["BBB", "DDD", "FFF"], "spread_home": [3.0, -2.5, 0.0], "total": [50.0, 44.5, 40.0], "roof": ["dome", None, "outdoors"],
    "away_qb": ["Alpha Passer", None, "X"], "home_qb": ["Golf Passer", None, "Y"], "away_score": np.nan, "home_score": np.nan})
INJ = pd.DataFrame([
    {"season": 2026, "week": 7, "team": "AAA", "gsis_id": "g1", "position": "WR", "full_name": "Bravo Wideout", "report_status": "questionable", "practice_status": "Limited Participation in Practice"},
    {"season": 2026, "week": 7, "team": "AAA", "gsis_id": "g2", "position": "TE", "full_name": "Foxtrot Tight", "report_status": None, "practice_status": "Did Not Participate In Practice"},
    {"season": 2026, "week": 7, "team": "BBB", "gsis_id": "g3", "position": "WR", "full_name": "Hotel Wideout", "report_status": "out", "practice_status": None},
    {"season": 2026, "week": 7, "team": "BBB", "gsis_id": "g4", "position": "QB", "full_name": "Golf Passer", "report_status": None, "practice_status": "Full Participation in Practice"},
    {"season": 2026, "week": 6, "team": "AAA", "gsis_id": "g5", "position": "QB", "full_name": "Alpha Passer", "report_status": "out", "practice_status": None},
])


def _fp(rows):
    return pd.DataFrame([{"name": n, "pos": p, "team": t, "opp": o, "week": 7, "proj": v, "key": f"{n}-{p}-{t}"} for n, p, t, o, v in rows])


FP = _fp([
    ("Alpha Passer", "QB", "AAA", "BBB", 20.5), ("Bravo Wideout Jr.", "WR", "AAA", "BBB", 12.0), ("Delta Traveler", "WR", "BBB", "AAA", 0.0),
    ("Hotel Wideout", "WR", "BBB", "AAA", 9.0), ("Rookie Receiver", "WR", "BBB", "AAA", 3.0), ("AAA", "DST", "AAA", "BBB", 8.0),
    ("Golf Passer", "QB", "BBB", "AAA", 18.0), ("Other Teamer", "WR", "CCC", "DDD", 11.0), ("Foxtrot Tight", "TE", "AAA", "BBB", 4.0)])


def _line(nk, pos, season, week):
    """The projection model, faked: known players get a line, anyone else (the rookie, the backups) gets nothing."""
    book = {"alpha passer": (22.0, {"pass_att": 33.0, "pass_yds": 250.0, "pass_td": 1.8, "int": 0.7, "rush_yds": 12.0, "rush_td": 0.2}),
            "bravo wideout": (11.0, {"tgt": 7.0, "rec": 4.5, "rec_yds": 55.0, "rec_td": 0.4}),
            "hotel wideout": (10.0, {"tgt": 6.0, "rec": 4.0, "rec_yds": 48.0, "rec_td": 0.3}),
            "golf passer": (17.0, {"pass_att": 31.0, "pass_yds": 230.0, "pass_td": 1.5, "int": 0.6, "rush_yds": 8.0, "rush_td": 0.1}),
            "delta traveler": (8.0, {"tgt": 5.0, "rec": 3.0, "rec_yds": 35.0, "rec_td": 0.2}),
            "foxtrot tight": (5.0, {"tgt": 3.0, "rec": 2.0, "rec_yds": 20.0, "rec_td": 0.15})}
    if nk not in book:
        return {"fp": None, "reason": "thin"}
    fp, ln = book[nk]
    return {"fp": fp, "line": ln, "opp_fp": fp - 1, "trailing_fp": fp + 1, "fp_role_ratio": 1.1, "source": "opp_blend", "games": 5}


@pytest.fixture()
def league(monkeypatch):
    monkeypatch.setattr(D, "schedule", lambda: SCHEDULE)
    monkeypatch.setattr(D, "player_weeks", lambda: LOGS)
    monkeypatch.setattr(D, "injuries", lambda: INJ)
    return None


# ------------------------------------------------------------------------------------------------------------------------------------------------ games
def test_the_weeks_games_come_in_kickoff_order_with_labels_spreads_and_implied_totals(league):
    g = G.week_games(2026, 7)
    assert list(g["home"]) == ["DDD", "BBB", "FFF"]                                               # Thursday night first
    assert g.loc[1, "label"] == "AAA @ BBB · Sun Oct 25 1:00 PM ET" and g.loc[0, "label"] == "CCC @ DDD · Thu Oct 22 8:15 PM ET"
    assert list(g["spread"]) == ["CCC -2.5", "BBB -3", "pick'em"]                                  # a negative home spread is the road team favoured
    assert g.loc[1, "home_implied"] == pytest.approx(26.5) and g.loc[1, "away_implied"] == pytest.approx(23.5)
    assert G.week_games(2026, 12).empty and list(G.week_games(2026, 12).columns).count("label") == 1


def test_spread_text_handles_both_favourites_and_unknowns():
    assert G.spread_text("NE", "CHI", 3.0) == "CHI -3" and G.spread_text("NE", "CHI", -2.5) == "NE -2.5"
    assert G.spread_text("NE", "CHI", 0.0) == "pick'em" and G.spread_text("NE", "CHI", None) == "pick'em" and G.spread_text("NE", "CHI", float("nan")) == "pick'em"


def test_the_default_game_is_the_next_one_to_kick_off(league):
    g = G.week_games(2026, 7)
    assert G.default_game_index(g, datetime(2026, 10, 21, 9, 0)) == 0                              # before Thursday night
    assert G.default_game_index(g, datetime(2026, 10, 25, 15, 0)) == 2                             # Sunday afternoon: Monday night is next
    assert G.default_game_index(g, datetime(2026, 10, 27, 9, 0)) == 2                              # all done: the last one
    assert G.default_game_index(g.iloc[:0]) == 0


# ------------------------------------------------------------------------------------------------------------------------------------------------ roster and candidates
def test_a_roster_is_skill_players_from_the_last_three_games_whose_latest_game_was_for_the_team(league):
    aaa = G.team_roster("AAA", 2026, 7)
    assert sorted(aaa["name"]) == ["Alpha Passer", "Bravo Wideout", "Foxtrot Tight"]
    #   Charlie's last game was week 3 (not in the last three), Delta's latest game was for BBB, Echo is a kicker
    bbb = G.team_roster("BBB", 2026, 7)
    assert sorted(bbb["name"]) == ["Delta Traveler", "Golf Passer", "Hotel Wideout"]
    assert set(aaa.columns) == {"nk", "name", "pos"} and G.team_roster("ZZZ", 2026, 7).empty


def test_a_roster_only_looks_at_games_before_the_week(league):
    early = G.team_roster("AAA", 2026, 5)                                                          # entering week 5 the games are weeks 3 and 4
    assert sorted(early["name"]) == ["Alpha Passer", "Bravo Wideout", "Charlie Runner", "Delta Traveler"]


def test_fantasypoints_rows_join_by_normalised_name_and_the_zero_and_the_stranger_are_marked(league):
    c = G.game_candidates("AAA", "BBB", 2026, 7, FP).set_index("name")
    assert c.loc["Bravo Wideout Jr.", "fp"] == 12.0 and c.loc["Bravo Wideout Jr.", "has_fp"] and "Bravo Wideout" not in c.index      # matched across the 'Jr.'; FantasyPoints' spelling is kept
    assert c.loc["Delta Traveler", "fp"] == 0.0 and not c.loc["Delta Traveler", "has_fp"]                                  # FantasyPoints has him at zero: no projection
    assert c.loc["Rookie Receiver", "has_fp"] and c.loc["Rookie Receiver", "team"] == "BBB" and c.loc["Rookie Receiver", "opp"] == "AAA"      # FantasyPoints only
    assert "Echo Kicker" not in c.index and c.loc["Foxtrot Tight", "fp"] == 4.0                                                 # a kicker is not a skill player
    assert "Other Teamer" not in c.index and "AAA" not in c.index                                                          # another team's player and the defense
    assert c.loc["Golf Passer", "pos"] == "QB"


def test_with_no_fantasypoints_file_every_roster_player_is_a_candidate_without_a_projection(league):
    c = G.game_candidates("AAA", "BBB", 2026, 7, None)
    assert len(c) == 6 and c["fp"].isna().all() and not c["has_fp"].any()


# ------------------------------------------------------------------------------------------------------------------------------------------------ the table
def test_injury_tags_use_the_official_status_then_the_practice_report(league):
    tags = G.injury_tags(("AAA", "BBB"), 2026, 7)
    assert tags[("AAA", "bravo wideout")] == "Questionable"                    # the official status wins over his limited practice
    assert tags[("AAA", "foxtrot tight")] == "DNP" and tags[("BBB", "hotel wideout")] == "Out"
    assert ("BBB", "golf passer") not in tags                                  # full participation is not worth a tag
    assert ("AAA", "alpha passer") not in tags                                 # last week's report is not this week's


def test_the_table_has_our_projection_the_fantasypoints_gap_and_the_derived_columns(league):
    t = G.game_table("AAA", "BBB", 2026, 7, FP, line_fn=_line).set_index("name")
    a = t.loc["Alpha Passer"]
    assert (a["ours"], a["fp"], a["diff"]) == (22.0, 20.5, pytest.approx(1.5)) and a["pass_yds"] == 250.0 and a["role_ratio"] == 1.1 and a["source"] == "opp_blend"
    assert np.isnan(a["any_td"])                                               # a quarterback has no rushing / receiving TD chance column
    b = t.loc["Bravo Wideout Jr."]
    assert b["any_td"] == pytest.approx(100 * (1 - np.exp(-0.4))) and b["inj"] == "Questionable"
    r = t.loc["Rookie Receiver"]
    assert np.isnan(r["ours"]) and r["fp"] == 3.0 and np.isnan(r["diff"])      # FantasyPoints projects him, we have no history to
    d = t.loc["Delta Traveler"]
    assert d["ours"] == 8.0 and np.isnan(d["diff"])                            # no gap against a zero
    order = list(G.game_table("AAA", "BBB", 2026, 7, FP, line_fn=_line)["name"])
    assert order[0] == "Alpha Passer" and order.index("Rookie Receiver") > order.index("Hotel Wideout")      # sorted by our number (FantasyPoints' where we have none)


def test_a_player_whose_projection_blows_up_does_not_take_the_table_down(league):
    def flaky(nk, pos, season, week):
        if nk == "bravo wideout":
            raise RuntimeError("thin data")
        return _line(nk, pos, season, week)
    t = G.game_table("AAA", "BBB", 2026, 7, FP, line_fn=flaky).set_index("name")
    assert np.isnan(t.loc["Bravo Wideout Jr.", "ours"]) and t.loc["Alpha Passer", "ours"] == 22.0


def test_an_empty_game_gives_an_empty_table(league):
    t = G.game_table("XXX", "YYY", 2026, 7, FP, line_fn=_line)
    assert t.empty and "ours" in t.columns


# ------------------------------------------------------------------------------------------------------------------------------------------------ filters
def test_the_default_filters_keep_only_players_fantasypoints_projects_and_drop_the_ones_ruled_out(league):
    t = G.game_table("AAA", "BBB", 2026, 7, FP, line_fn=_line)
    shown = G.filter_table(t)
    assert sorted(shown["name"]) == ["Alpha Passer", "Bravo Wideout Jr.", "Foxtrot Tight", "Golf Passer", "Rookie Receiver"]       # Hotel is Out, Delta is a FantasyPoints zero
    assert "Hotel Wideout" in set(G.filter_table(t, hide_out=False)["name"]) and "Delta Traveler" not in set(G.filter_table(t, hide_out=False)["name"])
    everyone = G.filter_table(t, only_fp=False, hide_out=False)
    assert len(everyone) == len(t) and "Delta Traveler" in set(everyone["name"])


def test_the_minimum_and_the_position_filters(league):
    t = G.game_table("AAA", "BBB", 2026, 7, FP, line_fn=_line)
    assert sorted(G.filter_table(t, min_fp=10.0)["name"]) == ["Alpha Passer", "Bravo Wideout Jr.", "Golf Passer"]
    assert set(G.filter_table(t, positions=["QB"])["pos"]) == {"QB"} and G.filter_table(t, positions=[]).empty
    assert G.filter_table(t.iloc[:0]).empty


def test_salaries_join_on_name_and_team_and_stay_blank_without_a_slate(league):
    t = G.game_table("AAA", "BBB", 2026, 7, FP, line_fn=_line)
    slate = pd.DataFrame({"name": ["Alpha Passer", "Hotel Wideout"], "team": ["AAA", "BBB"], "salary": [7000, 5400]})
    s = G.add_salary(t, slate).set_index("name")["salary"]
    assert s["Alpha Passer"] == 7000 and s["Hotel Wideout"] == 5400 and np.isnan(s["Golf Passer"])
    assert G.add_salary(t, None)["salary"].isna().all() and G.add_salary(t, slate.iloc[:0])["salary"].isna().all()


def test_the_team_strip_reports_volume_when_the_model_has_it_and_omits_it_when_not(monkeypatch):
    from matchup_model.opp import model as M
    monkeypatch.setattr(M, "team_volume", lambda team, season, week: {"pass_att": 34.6, "rush_att": 27.4})
    s = G.team_summary("CHI", "NE", 2026, 7, 26.75, "Caleb Williams")
    assert (s["pass_att"], s["rush_att"], s["implied"], s["qb"]) == (34.6, 27.4, 26.75, "Caleb Williams")
    monkeypatch.setattr(M, "team_volume", lambda *a: (_ for _ in ()).throw(KeyError("no row")))
    s2 = G.team_summary("CHI", "NE", 2026, 7, float("nan"), float("nan"))
    assert s2["pass_att"] is None and s2["implied"] is None and s2["qb"] is None


# ------------------------------------------------------------------------------------------------------------------------------------------------ the screen
SCRIPT = "from dfs import game_view\ngame_view.render(7, None)\n"


@pytest.fixture()
def screen(monkeypatch, league, tmp_path):
    """The screen with the projection model and the volume model faked (a plain function stands in for the cached table so tests do not share a cache)."""
    monkeypatch.setattr(G, "_season", lambda: 2026)
    monkeypatch.setattr(G, "_games", lambda season, week: G.week_games(season, week))
    monkeypatch.setattr(G, "_load_fp", lambda week: FP)
    monkeypatch.setattr(G, "_fp_path", lambda week: tmp_path / f"projections.week{week}.csv")
    monkeypatch.setattr(G, "_table", lambda season, week, away, home, h: G.game_table(away, home, season, week, FP, line_fn=_line))
    monkeypatch.setattr(G, "team_summary", lambda team, opp, season, week, implied, qb: {"team": team, "implied": implied, "pass_att": 33.0, "rush_att": 26.0, "qb": qb})
    real_default = G.default_game_index                                  # the screen opens on the next game to kick off: freeze 'now' to the Wednesday before the synthetic week
    monkeypatch.setattr(G, "default_game_index", lambda games, now=None: real_default(games, datetime(2026, 10, 21, 9, 0)))
    from streamlit.testing.v1 import AppTest
    return lambda: AppTest.from_string(SCRIPT, default_timeout=60).run()


def _tables(at):
    return {s.value.split(" (")[0]: d.value for s, d in zip(at.subheader, at.dataframe)}


def test_the_screen_opens_on_the_next_game_with_one_table_per_position(screen):
    at = screen()
    assert not at.exception and at.header[0].value == "🎮 DFS Game Projections"
    sel = next(s for s in at.selectbox if s.key == "gv_game")
    assert sel.value.startswith("CCC @ DDD") and len(sel.options) == 3
    assert list(_tables(at)) == ["WR"] and list(_tables(at)["WR"]["Player"]) == ["Other Teamer"]          # the Thursday game: only FantasyPoints knows this player
    next(s for s in at.selectbox if s.key == "gv_game").select("CCC @ DDD · Thu Oct 22 8:15 PM ET")
    next(n for n in at.number_input if n.key == "gv_minfp").set_value(30.0).run()
    assert any("No players match" in i.value for i in at.info) and not at.exception                          # filters that remove everyone say so instead of showing nothing


def test_picking_a_game_shows_the_fantasypoints_filtered_players_and_the_filters_work(screen):
    at = screen()
    next(s for s in at.selectbox if s.key == "gv_game").select("AAA @ BBB · Sun Oct 25 1:00 PM ET").run()
    t = _tables(at)
    assert set(t) == {"QB", "WR", "TE"} and sorted(t["QB"]["Player"]) == ["Alpha Passer", "Golf Passer"]
    assert sorted(t["WR"]["Player"]) == ["Bravo Wideout Jr.", "Rookie Receiver"]                  # Hotel is Out and Delta is a FantasyPoints zero
    assert list(t["QB"].columns) == ["Player", "Team", "Inj", "FP", "Ours", "Δ", "Pass att", "Pass yds", "Pass TD", "INT", "Rush yds", "Rush TD"]
    assert t["QB"].loc[t["QB"]["Player"] == "Alpha Passer", "Ours"].iloc[0] == 22.0
    next(c for c in at.checkbox if c.key == "gv_onlyfp").set_value(False).run()
    assert sorted(_tables(at)["WR"]["Player"]) == ["Bravo Wideout Jr.", "Delta Traveler", "Rookie Receiver"]   # the zero comes back; Hotel stays hidden (Out)
    next(c for c in at.checkbox if c.key == "gv_hideout").set_value(False).run()
    assert "Hotel Wideout" in set(_tables(at)["WR"]["Player"])
    next(c for c in at.checkbox if c.key == "gv_onlyfp").set_value(True).run()
    next(n for n in at.number_input if n.key == "gv_minfp").set_value(10.0).run()
    assert sorted(_tables(at)["WR"]["Player"]) == ["Bravo Wideout Jr."] and not at.exception


def test_the_position_picker_and_the_detail_toggle(screen):
    at = screen()
    next(s for s in at.selectbox if s.key == "gv_game").select("AAA @ BBB · Sun Oct 25 1:00 PM ET").run()
    next(m for m in at.multiselect if m.key == "gv_pos").set_value(["QB"]).run()
    assert set(_tables(at)) == {"QB"}
    next(c for c in at.checkbox if c.key == "gv_detail").set_value(True).run()
    assert {"Opp model", "Trailing", "FP ÷ ours"} <= set(_tables(at)["QB"].columns) and not at.exception


def test_without_a_fantasypoints_export_the_screen_says_so_and_shows_everyone(screen, monkeypatch):
    monkeypatch.setattr(G, "_load_fp", lambda week: None)
    monkeypatch.setattr(G, "_table", lambda season, week, away, home, h: G.game_table(away, home, season, week, None, line_fn=_line))
    at = screen()
    next(s for s in at.selectbox if s.key == "gv_game").select("AAA @ BBB · Sun Oct 25 1:00 PM ET").run()
    assert any("No FantasyPoints export for week 7" in w.value for w in at.warning)
    assert "Delta Traveler" in set(_tables(at)["WR"]["Player"]) and not at.exception


def test_a_week_with_no_games_gets_a_message(screen, monkeypatch):
    monkeypatch.setattr(G, "_games", lambda season, week: G.week_games(season, 12))
    at = screen()
    assert any("No week 7 games" in i.value for i in at.info) and not at.exception
