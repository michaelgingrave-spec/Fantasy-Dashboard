"""Bet log: add, payout math, edge-bucket analysis. Grading (nflverse) is tested lightly."""
import numpy as np
import pytest

from dfs import bets as B


@pytest.fixture
def tmp_log(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "LOG_PATH", tmp_path / "bet_log.csv")
    return B.LOG_PATH


def test_add_and_edge_fields(tmp_log):
    bid = B.add_bet(season=2025, week=1, event="SF @ LAR", player="Puka Nacua",
                    market="rec yds", side="OVER", line=89.5, odds=-110, our_proj=96.1)
    df = B.load()
    assert len(df) == 1 and df.iloc[0]["bet_id"] == bid
    r = df.iloc[0]
    assert r["edge_toward"] == pytest.approx(6.6, abs=0.01)          # 96.1 - 89.5
    assert r["edge_pct_toward"] == pytest.approx(7.4, abs=0.1)
    # UNDER flips the edge direction
    B.add_bet(season=2025, week=1, event="x", player="CMC", market="rec yds",
              side="UNDER", line=39.5, odds=-115, our_proj=29.0)
    u = B.load().iloc[1]
    assert u["edge_toward"] == pytest.approx(10.5, abs=0.01)         # 39.5 - 29.0


def test_payout_math():
    assert B._payout(-110, "win", 1.0) == pytest.approx(0.909, abs=0.001)
    assert B._payout(+150, "win", 2.0) == pytest.approx(3.0, abs=0.001)
    assert B._payout(-110, "loss", 1.0) == -1.0
    assert B._payout(-110, "push", 1.0) == 0.0


def test_delete(tmp_log):
    a = B.add_bet(season=2025, week=1, event="x", player="A", market="rec yds",
                  side="OVER", line=50, odds=-110, our_proj=60)
    B.add_bet(season=2025, week=1, event="x", player="B", market="rec yds",
              side="OVER", line=50, odds=-110, our_proj=60)
    B.delete_bet(a)
    df = B.load()
    assert len(df) == 1 and df.iloc[0]["player"] == "B"


def test_edge_bucket_analysis(tmp_log):
    # hand-grade a few rows and check bucketing + ROI
    B.add_bet(season=2025, week=1, event="x", player="A", market="rec yds",
              side="OVER", line=100, odds=-110, our_proj=106)      # 6% edge
    B.add_bet(season=2025, week=1, event="x", player="B", market="rec yds",
              side="OVER", line=100, odds=-110, our_proj=130)      # 30% edge
    df = B.load()
    df.loc[0, ["result", "payout", "stake"]] = ["win", 0.909, 1.0]
    df.loc[1, ["result", "payout", "stake"]] = ["loss", -1.0, 1.0]
    B._save(df)
    eb = B.by_edge_bucket()
    assert set(eb["edge range"]) == {"4-8%", "20%+"}
    assert eb.set_index("edge range").loc["4-8%", "win%"] == 100.0
    assert eb.set_index("edge range").loc["20%+", "ROI%"] == -100.0


def test_load_missing_is_empty(tmp_log):
    assert B.load().empty
    assert B.overall()["bets"] == 0
