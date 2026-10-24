"""matchup_model.opp.data.schedule(): the regular-season schedule with kickoff times, canonical teams and the home spread."""
import pandas as pd
import pytest

from matchup_model.opp import data as D


@pytest.fixture()
def games_csv(tmp_path, monkeypatch):
    rows = [
        dict(game_id="2026_07_NE_CHI", season=2026, game_type="REG", week=7, gameday="2026-10-22", gametime="20:15", away_team="NE", home_team="CHI", spread_line=3.0,
             total_line=50.5, roof="outdoors", away_qb_name="Drake Maye", home_qb_name="Caleb Williams", away_score=None, home_score=None),
        dict(game_id="2026_07_JAC_LAR", season=2026, game_type="REG", week=7, gameday="2026-10-25", gametime=None, away_team="JAC", home_team="LAR", spread_line=-2.5,
             total_line=44.0, roof="dome", away_qb_name="A", home_qb_name="B", away_score=17, home_score=20),
        dict(game_id="2025_19_X_Y", season=2025, game_type="WC", week=19, gameday="2026-01-10", gametime="16:30", away_team="X", home_team="Y", spread_line=1.0,
             total_line=40.0, roof="dome", away_qb_name="C", home_qb_name="D", away_score=None, home_score=None),
    ]
    pd.DataFrame(rows).to_csv(tmp_path / "games.csv", index=False)
    monkeypatch.setattr(D, "CACHE", tmp_path)
    D.schedule.cache_clear()
    yield
    D.schedule.cache_clear()


def test_the_schedule_has_kickoffs_canonical_teams_and_only_regular_season_games(games_csv):
    s = D.schedule()
    assert list(s["game_id"]) == ["2026_07_NE_CHI", "2026_07_JAC_LAR"]                                 # the id is nflverse's own, untouched
    assert list(s["away"]) == ["NE", "JAX"] and list(s["home"]) == ["CHI", "LA"]                       # JAC -> JAX, LAR -> LA
    assert s.loc[0, "kickoff"] == pd.Timestamp("2026-10-22 20:15") and s.loc[1, "kickoff"] == pd.Timestamp("2026-10-25 13:00")      # a missing time defaults to 1 pm
    assert list(s["spread_home"]) == [3.0, -2.5] and list(s["total"]) == [50.5, 44.0] and s.loc[0, "home_qb"] == "Caleb Williams"
    assert pd.isna(s.loc[0, "home_score"]) and s.loc[1, "home_score"] == 20


def test_a_data_refresh_clears_the_cached_schedule(games_csv):
    assert len(D.schedule()) == 2
    assert D.schedule.cache_info().currsize == 1
    D.clear_caches()
    assert D.schedule.cache_info().currsize == 0
