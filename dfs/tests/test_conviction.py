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


def test_annotate_keeps_every_row_and_blanks_what_it_cannot_price():
    hist = pd.DataFrame(_rows(60, 1, line=3.5, model_p=0.62, wins=26)      # says 62%, wins ~43%
                        + _rows(60, 2, line=3.5, model_p=0.62, wins=26))
    cal = C.fit(hist)
    live = pd.DataFrame(_rows(1, 3, line=3.5, model_p=0.62, prefix="A")
                        + _rows(1, 3, line=1.0, model_p=0.62, price="draftkings +194", prefix="B")
                        + _rows(1, 3, market="rec yds", line=40.5, model_p=0.60, prefix="C"))
    live = live.drop(columns=["result"])               # live pulls have no result column
    out = C.annotate(live, cal)
    assert len(out) == 3 and list(out["player"]) == list(live["player"])
    assert out.loc[0, "p cal %"] < out.loc[0, "p(hit)%"]        # overconfident bucket corrected down
    assert np.isnan(out.loc[1, "p cal %"]) and np.isnan(out.loc[1, "EV %"])   # whole-number count line
    assert out.loc[2, "p cal %"] < 60                            # unseen bucket inherits global miss


def test_annotate_shows_the_graded_record_of_each_kind_of_prop():
    hist = pd.DataFrame(_rows(60, 1, line=3.5, model_p=0.62, wins=26)
                        + _rows(60, 2, line=3.5, model_p=0.62, wins=26))     # 52-68 over two weeks
    cal = C.fit(hist)
    live = pd.DataFrame(_rows(1, 3, line=3.5, prefix="A")
                        + _rows(1, 3, line=1.0, price="draftkings +194", prefix="B")   # unpriceable
                        + _rows(1, 3, market="rec yds", line=40.5, model_p=0.60, prefix="C"))  # no history
    out = C.annotate(live.drop(columns=["result"]), cal)
    assert out.loc[0, "kind record"] == "52-68 (43%)"
    assert out.loc[1, "kind record"] == ""             # whole-number count line: not one bet
    assert out.loc[2, "kind record"] == ""             # a kind the history has never seen
    assert list(out["player"]) == list(live["player"])


def _tier_history():
    return pd.DataFrame(
        _rows(40, 1, conf="lean", price="draftkings -110", wins=20, prefix="L")      # 20-20
        + _rows(40, 2, conf="solid", price="fanduel -200", wins=28, prefix="S")      # 28-12, pays 0.5
        + _rows(20, 3, conf="—", price="draftkings -110", wins=8, prefix="N"))       # 8-12


def test_tier_help_text_quotes_live_results_against_the_break_even_their_prices_need():
    from dfs import bets
    hist = _tier_history()
    txt = C.tier_help_text(hist)
    assert "lean 50.0% vs 52.4% (n=40)" in txt                       # 20-20 at -110
    assert "solid 70.0% vs 66.7% (n=40)" in txt                      # 28-12 at -200: a heavier price
    assert "2026 through week 3" in txt
    assert "it has gone 40.0% vs 52.4% (n=20)" in txt                # the never-shown tier
    assert "strong " not in txt and "high " not in txt               # nothing graded there, nothing quoted
    assert "57%" not in txt and "62%" not in txt and "+19%" not in txt   # none of the old backtest claims
    table = bets.line_history_buckets(hist).set_index("conf")        # same win% and n as the screen's table
    assert all(f"{table.loc[t, 'win%']:.1f}% " in txt and f"(n={table.loc[t, 'n']})" in txt
               for t in ("lean", "solid"))


def test_tier_help_text_has_no_numbers_until_enough_is_graded():
    thin = pd.DataFrame(_rows(30, 1, conf="solid", wins=15))
    txt = C.tier_help_text(thin)
    assert "Not enough graded lines yet" in txt and "%" not in txt
    assert "Not enough graded lines yet" in C.tier_help_text(pd.DataFrame())


def test_line_groups_text_lists_every_market_group_from_the_bins():
    t = C.line_groups_text()
    assert "receptions ≤1.5 / 2.5 / 3.5 / 4.5+" in t
    assert "rec yds ≤20 / 20-40 / 40-60 / 60+" in t
    assert all(m in t for m in C.LINE_BINS)


def test_history_calibration_needs_enough_graded_history():
    small = pd.DataFrame(_rows(40, 1, wins=20))
    assert C.history_calibration(small) is None
    enough = pd.DataFrame(_rows(60, 1, wins=30) + _rows(60, 2, wins=30))
    assert C.history_calibration(enough) is not None


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
