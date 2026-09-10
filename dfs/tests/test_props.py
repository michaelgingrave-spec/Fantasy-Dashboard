"""`dfs.props` edge-table math — hermetic, no network (a hand-built Odds API payload)."""
import pandas as pd

from dfs import props


def _raw():
    """Minimal Odds API /events/{id}/odds shape: two books, two markets, one clean edge."""
    return {
        "id": "evt1",
        "bookmakers": [
            {"key": "draftkings", "markets": [
                {"key": "player_reception_yds", "outcomes": [
                    {"description": "Test Wr", "name": "Over", "point": 50.5, "price": -110},
                    {"description": "Test Wr", "name": "Under", "point": 50.5, "price": -110},
                ]},
                {"key": "player_rush_yds", "outcomes": [
                    {"description": "Test Rb", "name": "Over", "point": 80.5, "price": -115},
                    {"description": "Test Rb", "name": "Under", "point": 80.5, "price": -105},
                ]},
            ]},
            {"key": "fanduel", "markets": [
                {"key": "player_reception_yds", "outcomes": [
                    {"description": "Test Wr", "name": "Over", "point": 49.5, "price": -108},
                    {"description": "Test Wr", "name": "Under", "point": 49.5, "price": -112},
                ]},
            ]},
        ],
    }


def test_edges_from_raw_shape(monkeypatch):
    # our projection: WR well above the line (edge, OVER), RB just under it
    import dfs.projections as pj
    import matchup_model.project_stats as ps

    def _no_file(*_a, **_k):
        raise RuntimeError("no projection file in test")

    monkeypatch.setattr(pj, "load_weekly_projections", _no_file)

    def fake_line(name_key, pos, *a, **k):
        if name_key == props.normalize_name("Test Wr"):
            return {"line": {"rec_yds": 62.0, "rec": 4.6}, "fp": 12.0, "games": 6}
        if name_key == props.normalize_name("Test Rb"):
            return {"line": {"rush_yds": 78.0}, "fp": 14.0, "games": 6}
        return {"fp": None, "reason": "x"}

    monkeypatch.setattr(ps, "projected_line", fake_line)

    df = props.edges_from_raw(_raw(), week=1)
    assert set(["player", "market", "line", "our proj", "edge", "lean", "best"]).issubset(df.columns)

    wr = df[df["player"] == "Test Wr"].iloc[0]
    assert wr["market"] == "rec yds"
    assert wr["line"] == 50.0                     # median of 50.5 / 49.5
    assert wr["our proj"] == 62.0
    assert wr["edge"] == 12.0 and wr["lean"] == "OVER"
    assert "fanduel" in wr["best"] or "draftkings" in wr["best"]

    rb = df[df["player"] == "Test Rb"].iloc[0]
    assert rb["lean"] == "UNDER"                  # 78 proj < 80.5 line


def test_confident_only_filters_ratio_and_role():
    df = pd.DataFrame({
        "player": ["A", "B", "C", "D"],
        "market": ["rec yds"] * 4,
        "line": [50.0, 50.0, 50.0, 50.0],
        "our proj": [58.0, 120.0, 55.0, 55.0],   # B is 2.4x the line -> dropped
        "edge": [8.0, 70.0, 5.0, 5.0],
        "role_ratio": [0.9, 0.9, 0.2, None],     # C role collapsed -> dropped; D unknown -> kept
    })
    out = props.confident_only(df)
    assert set(out["player"]) == {"A", "D"}


def test_american_odds_helpers():
    assert props._amer_to_prob(-110) > 0.5
    assert abs(props._amer_to_dec(100) - 2.0) < 1e-9
    assert abs(props._amer_to_dec(-200) - 1.5) < 1e-9
    assert abs(props._norm_cdf(0) - 0.5) < 1e-6
