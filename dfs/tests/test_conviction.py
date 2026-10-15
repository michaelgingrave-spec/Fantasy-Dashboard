"""Calibrated conviction ranking: bucket/price math, shrinkage, pick rules, and -- the one
that matters -- that a test week is calibrated only on strictly earlier weeks."""
import re

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


def test_tier_help_text_counts_only_recommended_bets_by_default():
    stable = pd.DataFrame(_rows(100, 1, conf="solid", price="draftkings -110", wins=56, prefix="S"))
    shaky = pd.DataFrame(_rows(60, 2, conf="solid", price="draftkings -110", wins=15, prefix="X"))
    shaky["role_ratio"] = 2.0                      # FantasyPoints disagrees with our role read
    hist = pd.concat([stable, shaky], ignore_index=True)
    txt = C.tier_help_text(hist)
    assert "recommended bets" in txt and "solid 56.0% vs 52.4% (n=100 of 160)" in txt
    assert "it has gone" not in txt                # the never-shown tier can't be recommended
    every = C.tier_help_text(hist, recommended_only=False)
    assert "pulled lines" in every and "solid 44.4% vs 52.4% (n=160)" in every


def test_tier_help_text_quotes_live_results_against_the_break_even_their_prices_need():
    from dfs import bets
    hist = _tier_history()
    txt = C.tier_help_text(hist, recommended_only=False)
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
    txt = C.tier_help_text(thin, recommended_only=False)
    assert "Not enough graded lines yet" in txt and "%" not in txt
    assert "Not enough graded lines yet" in C.tier_help_text(pd.DataFrame())


def test_calibrated_picks_reads_the_numbers_as_the_table_shows_them():
    shown = pd.DataFrame({
        "player": list("ABCDEF"),
        "p cal %": [55.0, 55.0, 54.0, np.nan, 70.0, 62.0],
        "EV %": [1.0, 0.0, 9.0, 12.0, -3.0, np.nan],
    })
    out = C.calibrated_picks(shown)
    # only A clears both: B has no +EV, C is under 55, D/F are blank, E is priced too steeply
    assert list(out["player"]) == ["A"]
    assert list(out.index) == [0]


def test_annotate_and_the_rule_agree_on_the_rounded_numbers():
    hist = pd.DataFrame(_rows(60, 1, line=1.5, model_p=0.62, wins=40)
                        + _rows(60, 2, line=1.5, model_p=0.62, wins=40))     # wins 67%, says 62%
    cal = C.fit(hist)
    live = pd.DataFrame(_rows(3, 3, line=1.5, model_p=0.62, prefix="R")).drop(columns=["result"])
    out = C.annotate(live, cal)
    assert (out["p cal %"] == out["p cal %"].round(0)).all()          # whole numbers, as displayed
    assert len(C.calibrated_picks(out)) == int(((out["p cal %"] >= 55) & (out["EV %"] > 0)).sum()) == 3


def test_style_picks_colors_only_the_two_columns_the_rule_reads():
    view = pd.DataFrame({
        "player": ["A", "B", "C", "D"],
        "z": [0.5, 0.4, 0.3, 0.2],
        "p edge": [9.0, 8.0, 7.0, 6.0],
        "p cal %": [55.0, 54.0, 60.0, np.nan],
        "EV %": [1.0, 9.0, -2.0, np.nan],
    })
    html = C.style_picks(view).to_html()
    good, bad = "rgba(46, 160, 67, 0.40)", "rgba(220, 70, 70, 0.32)"

    def cells(color):      # pandas groups cells with one style into a single CSS rule
        rules = re.findall(r"([^{}]+)\{([^{}]*)\}", html)
        return sorted(s.strip().split("_row")[1] for sel, body in rules if color in body
                      for s in sel.split(","))

    # columns: player 0, z 1, p edge 2, p cal % 3, EV % 4.  p cal: A and C clear 55, B does not;
    # EV: A and B are above 0, C is not; D's blanks stay plain
    assert cells(good) == ["0_col3", "0_col4", "1_col4", "2_col3"]
    assert cells(bad) == ["1_col3", "2_col4"]
    assert "+1<" in html and "-2<" in html and ">55<" in html and ">60<" in html   # as the table shows them
    assert ">0.50<" in html and ">9.00<" in html and "0.500000" not in html        # other columns: 2 decimals
    # z and p edge carry no color of their own
    plain = C.style_picks(view[["player", "z", "p edge"]]).to_html()
    assert good not in plain and bad not in plain


def test_lead_columns_put_the_decision_columns_first():
    assert C.LEAD_COLUMNS[:4] == ("player", "market", "lean", "line")
    assert C.LEAD_COLUMNS.index("p cal %") < C.LEAD_COLUMNS.index("best") < C.LEAD_COLUMNS.index("kind record")


def _rule_history():
    """Weeks 1-2: this kind of prop beats its stated 52% (70% wins) and pays -120. Week 3: every
    one loses. 10 more week-3 rows project 2.5x the line, which the screen never shows."""
    rows = (_rows(40, 1, model_p=0.52, wins=28) + _rows(40, 2, model_p=0.52, wins=28)
            + _rows(20, 3, model_p=0.52, wins=0, prefix="W3"))
    hist = pd.DataFrame(rows)
    blow = pd.DataFrame(_rows(10, 3, model_p=0.52, wins=0, prefix="BLOW"))
    blow["our proj"] = blow["line"] * 2.5
    return pd.concat([hist, blow], ignore_index=True)


def test_rule_record_calibrates_each_week_only_on_earlier_weeks_and_skips_hidden_rows():
    rec = C.rule_record(_rule_history())
    assert [r["week"] for r in rec["by_week"]] == [3]                # weeks 1-2 have no earlier history
    # corrected UP from weeks 1-2 (52% said, 70% won), so all 20 clear the rule even though they all
    # lose -- week 3's own losses must not leak into its calibration (they would pull p cal under 55)
    assert rec["n"] == 20 and rec["win"] == 0.0 and rec["roi"] == -100.0
    assert rec["base_n"] == 20                                        # the 10 blow-up rows are not shown


def test_calibrated_buckets_scores_the_rule_by_tier_out_of_sample():
    tbl = C.calibrated_buckets(_rule_history())
    assert list(tbl.columns) == ["conf", "n", "of all", "win%", "ROI@-110%"]
    row = tbl.set_index("conf").loc["solid"]
    # week 3 only (weeks 1-2 have no earlier history); all 20 screen-visible rows clear the rule
    # and lose, and the 10 blow-up rows the screen never shows are not counted
    assert (row["n"], row["of all"], row["win%"], row["ROI@-110%"]) == (20, 20, 0.0, -100.0)
    assert tbl.attrs["weeks"] == (3, 3)
    assert C.calibrated_buckets(pd.DataFrame()).empty
    assert C.calibrated_buckets(pd.DataFrame(_rows(30, 1, wins=15))).empty       # too little history


def test_rule_help_text_quotes_the_live_record_and_admits_when_it_cannot_test():
    txt = C.rule_help_text(_rule_history())
    assert "55 or more" in txt and "above 0" in txt
    assert "week 3" in txt and "20 picks won 0.0%" in txt and "wk 3 0% (n=20)" in txt
    thin = pd.DataFrame(_rows(30, 1, wins=15))
    assert "Not enough graded weeks yet" in C.rule_help_text(thin)
    assert "Not enough graded weeks yet" in C.rule_help_text(pd.DataFrame())


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
