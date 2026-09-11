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
    # rec yds sigma at a 100 line ~= 49.5, so edge 6 -> z~0.12 ("—"), edge 30 -> z~0.61 ("strong")
    B.add_bet(season=2025, week=1, event="x", player="A", market="rec yds",
              side="OVER", line=100, odds=-110, our_proj=106)
    B.add_bet(season=2025, week=1, event="x", player="B", market="rec yds",
              side="OVER", line=100, odds=-110, our_proj=130)
    df = B.load()
    assert df.loc[0, "z"] < 0.15 and df.loc[1, "z"] > 0.5
    assert df.loc[0, "conf"] == "—" and df.loc[1, "conf"] == "strong"
    df.loc[0, ["result", "payout", "stake"]] = ["win", 0.909, 1.0]
    df.loc[1, ["result", "payout", "stake"]] = ["loss", -1.0, 1.0]
    B._save(df)
    eb = B.by_edge_bucket().set_index("conf")
    assert eb.loc["—", "win%"] == 100.0
    assert eb.loc["strong", "ROI%"] == -100.0


def test_load_missing_is_empty(tmp_log):
    assert B.load().empty
    assert B.overall()["bets"] == 0


def test_line_history_snapshot_and_buckets(tmp_path, monkeypatch):
    import pandas as pd
    from dfs import props

    monkeypatch.setattr(props, "LINE_HISTORY_PATH", tmp_path / "lh.csv")
    edf = pd.DataFrame({
        "player": [f"P{i}" for i in range(12)],
        "market": ["rec yds"] * 12,
        "line": [50.0] * 12,
        "our proj": [56.0] * 6 + [70.0] * 6,
        "edge": [6.0] * 6 + [20.0] * 6,
        "z": [0.18] * 6 + [0.61] * 6,                  # small edge x6, big edge x6
        "conf": ["lean"] * 6 + ["strong"] * 6,
        "p(hit)%": [55] * 12, "p edge": [3.0] * 12, "book %": [52] * 12,
        "edge %": [12.0] * 6 + [40.0] * 6,
        "lean": ["OVER"] * 12, "best": [""] * 12, "role_ratio": [1.0] * 12,
    })
    assert props.snapshot_lines(edf, 2025, 3, "X @ Y") == 12
    lh = B.load_line_history()
    assert len(lh) == 12 and "result" in lh.columns and "conf" in lh.columns
    lh.loc[lh["conf"] == "lean", "result"] = "loss"   # small edges lose
    lh.loc[lh["conf"] == "strong", "result"] = "win"  # big edges win
    lh.to_csv(props.LINE_HISTORY_PATH, index=False)
    bk = B.line_history_buckets().set_index("conf")
    lean = "lean"
    strong = "strong"
    assert bk.loc[strong, "win%"] == 100.0
    assert bk.loc[lean, "win%"] == 0.0
