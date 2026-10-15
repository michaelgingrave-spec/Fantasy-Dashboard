"""Opening-line collector: the parts that don't touch the network."""
from dfs import opening_lines as OL


def _payload():
    def mk(key, pts_prices):
        return {"key": key, "outcomes": [{"name": n, "description": p, "price": pr, "point": pt}
                                        for p, n, pr, pt in pts_prices]}
    return {"bookmakers": [
        {"key": "draftkings", "markets": [mk("player_receptions", [("A Player", "Over", -120, 4.5), ("A Player", "Under", 100, 4.5)])]},
        {"key": "fanduel", "markets": [mk("player_receptions", [("A Player", "Over", -105, 5.5), ("A Player", "Under", -115, 5.5)]),
                                       mk("player_reception_yds", [("A Player", "Over", -110, 61.5), ("A Player", "Under", -110, 61.5)])]},
    ]}


def test_parse_event_odds_uses_the_median_line_and_the_best_price_like_the_app():
    rows = {(r["player"], r["market"]): r for r in OL.parse_event_odds(_payload())}
    rec = rows[("A Player", "receptions")]
    assert rec["line_open"] == 5.0                       # median of 4.5, 4.5, 5.5, 5.5 -- same as edges_from_raw
    assert rec["n_books"] == 2
    assert (rec["over_price"], rec["over_book"]) == (-105, "fanduel")      # best (least negative) over
    assert (rec["under_price"], rec["under_book"]) == (100, "draftkings")
    assert rows[("A Player", "rec yds")]["line_open"] == 61.5
    assert OL.parse_event_odds({}) == []                 # a game with no props posted yet


def test_snapshot_is_the_wednesday_before_each_thursday_kickoff():
    assert OL.snapshot_time(2026, 1) == "2026-09-09T20:00:00Z"
    assert OL.snapshot_time(2026, 5) == "2026-10-07T20:00:00Z"
