"""The historical closing-line collector: right snapshot times, both prices side by side, and a credit guard that cannot be talked into overspending."""
from dfs import historical_lines as H


def test_week_windows_and_the_closing_snapshot_land_where_the_slate_is():
    assert H.iso(H.week_start(2025, 5)) == "2025-10-01T20:00:00Z"          # the Wednesday before week 5's Thursday game (Oct 2, 2025)
    assert H.iso(H.week_start(2025, 18)) == "2025-12-31T20:00:00Z"
    assert H.iso(H.week_start(2026, 1)) == "2026-09-09T20:00:00Z"
    assert H.closing_snapshot("2025-10-05T17:00:00Z") == "2025-10-05T16:55:00Z"


def _payload(book="draftkings", extra=()):
    out = lambda name, pt, price, who="Test Player": {"name": name, "description": who, "price": price, "point": pt}   # noqa: E731
    return {"bookmakers": [{"key": book, "markets": [
        {"key": "player_reception_yds", "outcomes": [out("Over", 61.5, -115), out("Under", 61.5, -105), out("Over", 40.5, -120, "Other Guy")]},
        {"key": "player_rush_yds", "outcomes": [out("Over", 55.5, -110, "Back Man"), out("Under", 55.5, -110, "Back Man")]},
        {"key": "player_something_else", "outcomes": [out("Over", 1.5, -110)]}, *extra]}]}


def test_both_prices_of_a_line_sit_on_one_row_and_unknown_markets_and_pointless_outcomes_are_skipped():
    rows = {(r["player"], r["market"], r["book"]): r for r in H.parse_event_books(_payload())}
    assert set(rows) == {("Test Player", "rec yds", "draftkings"), ("Other Guy", "rec yds", "draftkings"), ("Back Man", "rush yds", "draftkings")}
    r = rows[("Test Player", "rec yds", "draftkings")]
    assert (r["point"], r["over_price"], r["under_price"]) == (61.5, -115, -105)
    assert rows[("Other Guy", "rec yds", "draftkings")]["under_price"] is None                                   # a one-sided line stays one-sided
    assert H.parse_event_books({"bookmakers": [{"key": "fanduel", "markets": [{"key": "player_receptions", "outcomes": [{"name": "Over", "description": "X", "price": -110}]}]}]}) == []


class FakeApi:
    """Stands in for the Odds API: an event list (one game outside the week's window) and a flat 30-credit odds call."""
    def __init__(self, balance=1000, games=3):
        self.balance, self.calls = balance, []
        self.events = [{"id": f"g{i}", "commence_time": f"2025-10-0{5 + i % 2}T17:00:00Z", "home_team": f"H{i}", "away_team": f"A{i}"} for i in range(games)]
        self.events.append({"id": "later", "commence_time": "2025-10-20T17:00:00Z", "home_team": "H", "away_team": "A"})

    def __call__(self, path, **params):
        self.calls.append(path)
        cost = 1 if path == "/events" else 30
        self.balance -= cost
        hdr = {"remaining": str(self.balance), "last": str(cost)}
        return ({"data": self.events} if path == "/events" else {"timestamp": params["date"], "data": _payload()}), hdr


def test_a_dry_run_buys_no_odds_and_a_real_run_buys_each_game_once(tmp_path):
    api = FakeApi()
    dry = H.pull(2025, 5, 5, floor=0, execute=False, get=api, raw_dir=tmp_path, say=lambda *a, **k: None)
    assert [c for c in api.calls if c != "/events"] == [] and sum(len(e) for _, e in dry["plan"]) == 3          # the game outside the week is not planned
    real = H.pull(2025, 5, 5, floor=0, execute=True, get=api, raw_dir=tmp_path, say=lambda *a, **k: None)
    assert sum(1 for c in api.calls if c.endswith("/odds")) == 3 and len(real["rows"]) == 9 and api.balance == 1000 - 1 - 90      # event list cached from the dry run
    again = H.pull(2025, 5, 5, floor=0, execute=True, get=api, raw_dir=tmp_path, say=lambda *a, **k: None)
    assert sum(1 for c in api.calls if c.endswith("/odds")) == 3 and len(again["rows"]) == 9                       # nothing is bought twice


def test_the_balance_floor_stops_the_pull_before_it_overspends(tmp_path):
    api = FakeApi(balance=100, games=3)                          # 99 after the event list: three 30-credit games would leave 9
    out = H.pull(2025, 5, 5, floor=40, execute=True, get=api, raw_dir=tmp_path, say=lambda *a, **k: None)
    assert api.balance >= 40 and sum(1 for c in api.calls if c.endswith("/odds")) == 0                               # refused up front: the whole pull would cross the floor
    api2 = FakeApi(balance=100, games=3)
    out2 = H.pull(2025, 5, 5, floor=40, execute=True, get=api2, raw_dir=tmp_path / "b", say=lambda *a, **k: None)
    assert out["spent"] == 1 and api2.balance >= 40 and out2["spent"] == 1


def test_a_failed_call_is_skipped_not_retried_and_the_rest_continue(tmp_path):
    import requests
    api = FakeApi()
    real_call = api.__call__

    def flaky(path, **params):
        if path == "/events/g1/odds":
            raise requests.HTTPError("422")
        return real_call(path, **params)
    out = H.pull(2025, 5, 5, floor=0, execute=True, get=flaky, raw_dir=tmp_path, say=lambda *a, **k: None)
    assert len(out["rows"]) == 6 and not (tmp_path / "2025_wk5_g1.json").exists()
