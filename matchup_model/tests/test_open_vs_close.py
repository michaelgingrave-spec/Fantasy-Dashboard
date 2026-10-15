"""Open-vs-close analysis: the statistics and the join, on small constructed cases."""
import numpy as np
import pandas as pd
import pytest

from matchup_model.opp import open_vs_close as OC


def test_clustered_slope_recovers_a_planted_slope():
    rng = np.random.default_rng(0)
    x = np.linspace(-1, 1, 200)
    y = 0.3 * x + rng.normal(0, 0.05, 200)
    b, lo, hi = OC.clustered_slope(x, y, np.arange(200) // 2, np.random.default_rng(1), reps=300)
    assert b == pytest.approx(0.3, abs=0.03) and lo < 0.3 < hi


def test_clustered_slope_ci_spans_zero_for_unrelated_series():
    rng = np.random.default_rng(2)
    x, y = rng.normal(size=300), rng.normal(size=300)
    b, lo, hi = OC.clustered_slope(x, y, np.arange(300) // 3, np.random.default_rng(3), reps=300)
    assert lo < 0 < hi


def test_play_bets_the_side_the_projection_implies_and_flags_pushes():
    won, push, over = OC.play(proj=[60, 40, 50, 55], line=[50, 50, 50, 50], actual=[70, 30, 50, 40])
    assert list(over) == [True, False, False, True]                       # 50 vs 50 is not over: a tie leans under
    assert list(won) == [True, True, False, False]                        # over won at 70, under won at 30, 50 is a push, over lost at 40
    assert list(push) == [False, False, True, False]


def test_paired_gain_ignores_pushes_and_clusters_by_player():
    a = np.array([1, 1, 0, 1], float)
    b = np.array([0, 1, 0, 0], float)
    push = np.array([False, False, False, True])                          # last row pushed on one line: dropped
    est, lo, hi, n = OC.paired_gain(a, b, push, ["p", "q", "r", "s"], np.random.default_rng(4), reps=200)
    assert n == 3 and est == pytest.approx(1 / 3)                         # +1, 0, 0 over three rows
    assert lo <= est <= hi


def test_paired_gain_with_no_usable_rows_is_nan_not_a_crash():
    est, lo, hi, n = OC.paired_gain([], [], np.array([], bool), [], np.random.default_rng(0), reps=10)
    assert n == 0 and np.isnan(est) and np.isnan(lo) and np.isnan(hi)
    est, lo, hi, n = OC.paired_gain([1.0], [0.0], np.array([True]), ["a"], np.random.default_rng(0), reps=10)    # everything pushed
    assert n == 0 and np.isnan(est)


def test_load_joined_keeps_only_graded_receiving_rows_with_both_lines(tmp_path, monkeypatch):
    pd.DataFrame({
        "season": 2026, "week": [1, 1, 1, 1], "player": ["A A", "B B", "C C", "D D"],
        "market": ["rec yds", "receptions", "rush yds", "rec yds"], "line": [50.0, 4.5, 60.0, 30.0],
        "our proj": [58.0, 5.2, 70.0, 20.0], "actual": [61.0, 3.0, 80.0, np.nan],
        "result": ["win", "loss", "win", ""],
    }).to_csv(tmp_path / "lh.csv", index=False)
    pd.DataFrame({
        "season": 2026, "week": [1, 1, 1, 1], "player": ["A A", "B B", "C C", "D D"],
        "market": ["rec yds", "receptions", "rush yds", "rec yds"], "line_open": [48.5, 4.5, 58.5, 31.5],
        "over_price": -110, "under_price": -110,
    }).to_csv(tmp_path / "open.csv", index=False)
    monkeypatch.setattr(OC, "HISTORY", tmp_path / "lh.csv")
    monkeypatch.setattr(OC, "OPENING", tmp_path / "open.csv")
    d = OC.load_joined()
    assert sorted(d.player) == ["A A", "B B"]                              # rush yds is not a receiving market; D D is ungraded
    a = d[d.player == "A A"].iloc[0]
    assert a.move == pytest.approx(1.5)                                    # late 50.0 minus open 48.5
    assert a.edge_open_z > 0 and a.move_z > 0 and a.sigma > 0              # we were above the open and the line rose toward us
    assert d[d.player == "B B"].iloc[0].move == 0


def test_report_builds_on_a_small_frame():
    rng = np.random.default_rng(6)
    n = 60
    line_open = rng.uniform(30, 80, n).round(1)
    proj = line_open + rng.normal(0, 8, n)
    d = pd.DataFrame({
        "week": np.repeat([3, 4, 5], 20), "player": [f"P{i % 30}" for i in range(n)], "market": "rec yds",
        "line_open": line_open, "line": line_open + rng.normal(0, 1, n).round(1), "our proj": proj,
        "actual": rng.uniform(0, 120, n).round(), "over_price": -110, "under_price": -110})
    d["sigma"] = 16.28 + 0.3324 * d.line_open
    d["move"] = d.line - d.line_open
    d["move_z"] = d.move / d.sigma
    d["edge_open_z"] = (d["our proj"] - d.line_open) / d.sigma
    d["nk"] = d.player.str.lower()
    txt = OC.build_report(d)
    assert "## 2. Does the market move toward our number?" in txt and "Paired difference, open minus late" in txt
    assert "| receptions | 0 | n/a | n/a |" in txt                        # a slice with no rows is reported, not crashed on
    assert "| weeks 1-2 (mid-week lines) | 0 | n/a | n/a |" in txt


def test_clustered_slope_with_too_few_rows_is_nan():
    assert all(np.isnan(v) for v in OC.clustered_slope([], [], [], np.random.default_rng(0)))
    assert all(np.isnan(v) for v in OC.clustered_slope([1.0, 1.0, 1.0], [1.0, 2.0, 3.0], ["a", "b", "c"], np.random.default_rng(0)))
