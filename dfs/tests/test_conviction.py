"""Calibrated conviction ranking: bucket/price math, shrinkage, pick rules, and -- the one
that matters -- that a test week is calibrated only on strictly earlier weeks."""
import numpy as np
import pandas as pd
import pytest

from dfs import conviction as C


def _rows(n, week, market="receptions", lean="OVER", line=1.5, model_p=0.62, wins=0,
          price="draftkings -120", conf="solid", prefix="P"):
    out = []
    for i in range(n):
        out.append({"season": 2026, "week": week, "player": f"{prefix}{week}_{i}",
                    "market": market, "line": line, "our proj": line * 1.3, "conf": conf,
                    "z": 0.4, "p edge": 5.0, "lean": lean, "best": price,
                    "p(hit)%": model_p * 100, "role_ratio": 1.0,
                    "result": "win" if i < wins else "loss"})
    return out


def test_line_bucket_price_and_breakeven():
    assert C.line_bucket("receptions", 1.5) == "<=1.5"
    assert C.line_bucket("receptions", 2.5) == "2.5"
    assert C.line_bucket("receptions", 4.5) == "4.5+"
    assert C.line_bucket("rec yds", 20) == "<=20" and C.line_bucket("rec yds", 20.5) == "20-40"
    assert C.line_bucket("not a market", 3) is None
    assert C.parse_price("draftkings -112") == -112 and C.parse_price("fanduel +172") == 172
    assert np.isnan(C.parse_price("no price here"))
    assert C.breakeven(-150) == pytest.approx(0.6) and C.breakeven(100) == pytest.approx(0.5)
    assert C.win_profit(-200) == pytest.approx(0.5) and C.win_profit(150) == pytest.approx(1.5)


def test_whole_number_consensus_lines_dropped_on_count_markets_only():
    # books disagreeing 0.5 vs 1.5 catches -> consensus 1.0 + one book's price is not one bet
    rows = (_rows(3, 1, market="receptions", line=1.0, prefix="W")
            + _rows(3, 1, market="receptions", line=1.5, prefix="H")
            + _rows(3, 1, market="rec yds", line=62.0, prefix="Y"))
    kept = C.prep(pd.DataFrame(rows))
    assert set(kept["market"]) == {"receptions", "rec yds"}
    assert (kept[kept["market"] == "receptions"]["line"] == 1.5).all()      # the 1.0 rows are gone
    assert (kept["market"] == "rec yds").sum() == 3                          # yardage 62.0 stays


def test_thin_bucket_is_shrunk_toward_the_global_miss_not_its_own_lucky_streak():
    big = _rows(100, 1, line=3.5, model_p=0.62, wins=40)          # says 62%, wins 40%
    tiny = _rows(2, 1, line=1.5, model_p=0.62, wins=2, prefix="T")  # 2-for-2, says 62%
    cal = C.fit(pd.DataFrame(big + tiny))
    t = cal.table.set_index("bucket")
    assert cal.global_delta < -0.15
    # raw miss for the tiny bucket is positive (+38 pts) but it must stay near the global miss
    assert t.loc["<=1.5", "delta_raw"] > 0.3
    assert abs(t.loc["<=1.5", "delta"] - cal.global_delta) < abs(t.loc["<=1.5", "delta"] - t.loc["<=1.5", "delta_raw"])


def test_top_picks_one_per_player_floor_and_positive_ev():
    s = pd.DataFrame({
        "player": ["A", "A", "B", "C", "D"],
        "p_cal": [0.70, 0.66, 0.54, 0.62, 0.60],
        "ev": [0.30, 0.25, 0.10, -0.01, 0.20],
    })
    out = C.top_picks(s, k=8, p_min=0.55)
    assert list(out["player"]) == ["A", "D"]      # B under the floor, C negative EV, A deduped


def test_walk_forward_calibrates_only_on_earlier_weeks():
    # weeks 1-2: this bucket badly underdelivers (says 62%, wins 30%).
    # week 3: every one of them wins -- that must NOT leak into week 3's own calibration.
    lh = pd.DataFrame(_rows(40, 1, wins=12) + _rows(40, 2, wins=12) + _rows(20, 3, wins=20))
    wf = C.walk_forward(lh, [3], ks=(8,))
    rel = wf["reliab"]
    assert (rel["model_p"] > rel["p_cal"]).all()        # corrected DOWN from weeks 1-2 evidence
    assert rel["won"].mean() == 1.0                      # even though week 3 itself all won
