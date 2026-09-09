import pandas as pd

from dfs.edge import player_boosts, weekly_edges
from dfs.odds.environment import team_environment


def test_implied_total_math():
    events = [
        {
            "home_team": "Cincinnati Bengals", "away_team": "Tampa Bay Buccaneers",
            "bookmakers": [{"key": "draftkings", "markets": [
                {"key": "totals", "outcomes": [{"name": "Over", "point": 47.5},
                                               {"name": "Under", "point": 47.5}]},
                {"key": "spreads", "outcomes": [{"name": "Cincinnati Bengals", "point": -2.5},
                                                {"name": "Tampa Bay Buccaneers", "point": 2.5}]},
            ]}],
        }
    ]
    env = team_environment(events).set_index("team")
    assert env.loc["CIN", "implied_total"] == 25.0
    assert env.loc["TB", "implied_total"] == 22.5
    assert bool(env.loc["CIN", "is_favorite"]) is True


def test_team_environment_empty_on_no_events():
    assert team_environment(None).empty
    assert team_environment([]).empty


def test_weekly_edges_runs_without_datasuite_or_odds(slate_df, tmp_path, monkeypatch):
    monkeypatch.setattr("dfs.edge.get_nfl_game_lines", lambda *a, **k: None)
    edges = weekly_edges(slate_df, datasuite_dir=tmp_path)
    assert not edges.empty
    assert {"dk_id", "edge", "edge_z", "why"}.issubset(edges.columns)
    # sorted by edge desc
    assert list(edges["edge"]) == sorted(edges["edge"], reverse=True)
    # no signal -> boosts near 1.0
    boosts = player_boosts(edges, weight=0.05)
    assert all(0.85 <= v <= 1.15 for v in boosts.values())


def test_weekly_edges_uses_datasuite_when_present(slate_df, tmp_path, monkeypatch):
    import shutil
    from pathlib import Path
    fx = Path(__file__).parent / "fixtures"
    shutil.copy(fx / "datasuite_receiving_advanced.sample.csv", tmp_path / "receiving_advanced_player.csv")
    shutil.copy(fx / "datasuite_defense_receiving.sample.csv", tmp_path / "defense_receiving_team.csv")
    monkeypatch.setattr("dfs.edge.get_nfl_game_lines", lambda *a, **k: None)
    edges = weekly_edges(slate_df, datasuite_dir=tmp_path)
    wr = edges[edges["pos"] == "WR"]
    assert wr["opportunity_z"].abs().sum() > 0  # usage term actually moved
