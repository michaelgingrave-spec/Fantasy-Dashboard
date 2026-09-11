"""`dfs.props` edge-table math — hermetic, no network (a hand-built Odds API payload)."""
import pandas as pd
import pytest

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
    import matchup_model.opp.blend as blend
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

    # edges_from_raw imports blended_line as its projector; patch both it and the fallback
    monkeypatch.setattr(blend, "blended_line", fake_line)
    monkeypatch.setattr(ps, "projected_line", fake_line)

    df = props.edges_from_raw(_raw(), week=1)
    assert set(["player", "market", "line", "our proj", "edge", "conf", "z", "p edge",
                "lean", "best"]).issubset(df.columns)

    wr = df[df["player"] == "Test Wr"].iloc[0]
    assert wr["market"] == "rec yds"
    assert wr["line"] == 50.0                     # median of 50.5 / 49.5
    assert wr["our proj"] == 62.0
    assert wr["edge"] == 12.0 and wr["lean"] == "OVER"
    # edge 12 on a 50 rec-yd line: sigma ~ 16.28 + 0.332*50 ~ 33 -> z ~ 0.36 -> "solid"
    assert wr["z"] == pytest.approx(0.36, abs=0.05)
    assert wr["conf"] == "solid"
    assert "fanduel" in wr["best"] or "draftkings" in wr["best"]

    rb = df[df["player"] == "Test Rb"].iloc[0]
    assert rb["lean"] == "UNDER"                  # 78 proj < 80.5 line


def test_confident_only_filters_conf_ratio_and_role():
    df = pd.DataFrame({
        "player": ["A", "B", "C", "D", "E"],
        "market": ["rec yds"] * 5,
        "line": [50.0] * 5,
        "our proj": [58.0, 120.0, 55.0, 55.0, 55.0],
        "edge": [8.0, 70.0, 5.0, 5.0, 5.0],
        "conf": ["solid", "high", "solid", "solid", "—"],   # E has no edge tier -> dropped
        "role_ratio": [0.9, 0.9, 0.2, None, 0.9],           # C role collapsed -> dropped
    })
    out = props.confident_only(df)                          # B dropped: proj 2.4x line
    assert set(out["player"]) == {"A", "D"}


def test_conf_label_scale():
    assert props.conf_label(0.05) == "—"
    assert props.conf_label(0.20, "rec_yds") == "lean"
    assert props.conf_label(0.40, "rec_yds") == "solid"
    assert props.conf_label(0.90, "rec_yds") == "high"
    assert props.conf_label(0.50, "rush_yds") == "lean"     # rush needs more z
    assert props.conf_label(3.0, "pass_att") == "—"         # pass att never rates


def test_american_odds_helpers():
    assert props._amer_to_prob(-110) > 0.5
    assert abs(props._amer_to_dec(100) - 2.0) < 1e-9
    assert abs(props._amer_to_dec(-200) - 1.5) < 1e-9
    assert abs(props._norm_cdf(0) - 0.5) < 1e-6


def test_parlay_odds_two_leg_pickem():
    # two -110 legs: dec 1.909 each -> combined ~3.645 -> american +264ish
    p = props.parlay_odds([-110, -110])
    assert p["decimal"] == pytest.approx(3.645, abs=0.01)
    assert p["american"] == pytest.approx(264, abs=2)
    assert p["implied_from_price"] < 0.5           # parlay is a worse bet than either leg alone


def test_dec_amer_roundtrip():
    for a in (-250, -110, 120, 350):
        assert props._dec_to_amer(props._amer_to_dec(a)) == a


def test_save_load_clear_last_pull(tmp_path, monkeypatch):
    monkeypatch.setattr(props, "LAST_PULL_CSV", tmp_path / "last_pull.csv")
    monkeypatch.setattr(props, "LAST_PULL_META", tmp_path / "last_pull_meta.json")

    assert props.load_last_pull() is None          # nothing saved yet

    df = pd.DataFrame({"player": ["A", "B"], "market": ["rec yds"] * 2,
                       "conf": ["solid", "strong"], "role_ratio": [0.9, 1.1]})
    props.save_last_pull(df, "Game A @ B", "417", snap=2)

    restored = props.load_last_pull()
    assert restored is not None and restored["restored"] is True
    assert restored["game"] == "Game A @ B" and restored["rem"] == "417" and restored["snap"] == 2
    assert list(restored["df"]["player"]) == ["A", "B"]
    assert "role_ratio" in restored["df"].columns    # full frame round-trips, not a subset

    props.clear_last_pull()
    assert props.load_last_pull() is None
    assert not props.LAST_PULL_CSV.exists() and not props.LAST_PULL_META.exists()


def test_bulk_prop_edges_tags_game_and_sorts_by_conf(monkeypatch):
    import dfs.projections as pj
    import matchup_model.opp.blend as blend

    monkeypatch.setattr(pj, "load_weekly_projections",
                        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("no file")))

    def fake_line(name_key, pos, *a, **k):
        # small edge for the WR (line 50 -> proj 55), big edge for the RB (line 80.5 -> proj 130)
        if name_key == props.normalize_name("Test Wr"):
            return {"line": {"rec_yds": 55.0}, "fp": 12.0, "games": 6}
        if name_key == props.normalize_name("Test Rb"):
            return {"line": {"rush_yds": 130.0}, "fp": 20.0, "games": 6}
        return {"fp": None, "reason": "x"}

    monkeypatch.setattr(blend, "blended_line", fake_line)

    calls = []

    def fake_fetch(event_id, markets):
        calls.append(event_id)
        return _raw(), "999"

    monkeypatch.setattr(props, "fetch_event_odds", fake_fetch)
    events = [{"id": "e1", "label": "Game One"}, {"id": "e2", "label": "Game Two"}]
    df, rem = props.bulk_prop_edges(events, ["player_reception_yds", "player_rush_yds"], week=1)

    assert calls == ["e1", "e2"]              # one fetch per event, in order
    assert rem == "999"
    assert set(df["game"]) == {"Game One", "Game Two"}
    assert len(df) == 4                        # 2 players x 2 games
    # sorted by conf rank (desc) then p edge (desc) — never increasing in rank
    ranks = df["conf"].map(props._CONF_RANK).tolist()
    assert ranks == sorted(ranks, reverse=True)
    assert df.iloc[0]["player"] == "Test Rb"   # biggest edge (RB) sorts first
