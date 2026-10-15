"""The edge meter: preparing graded lines and the clustered logistic fit."""
import numpy as np
import pandas as pd
import pytest

from matchup_model.opp import market_edge as ME


def _lines(n=40, **over):
    rng = np.random.default_rng(0)
    base = {"player": [f"P{i % 20}" for i in range(n)], "market": ["rec yds"] * n, "line": [50.0] * n, "actual": rng.uniform(10, 90, n).round(),
            "our proj": rng.uniform(40, 60, n), "lean": rng.choice(["OVER", "UNDER"], n), "book %": [52.0] * n, "result": ["win"] * n}
    base.update(over)
    return pd.DataFrame(base)


def test_prepare_keeps_graded_non_push_rows_in_modelled_markets_and_converts_the_book_probability_to_the_over_side():
    lh = _lines(6, market=["rec yds", "receptions", "rush att", "rec yds", "rec yds", "rec yds"], line=[50.0] * 6,
                actual=[60.0, 5.0, 20.0, 50.0, 70.0, 30.0], result=["win", "loss", "win", "push", "win", "win"],
                lean=["OVER", "OVER", "OVER", "OVER", "UNDER", "UNDER"], **{"book %": [55.0, 55.0, 55.0, 55.0, 60.0, np.nan]})
    d = ME.prepare(lh)
    # kept: rec yds OVER win, receptions OVER loss, rec yds UNDER win (70 > 50 with an UNDER lean) -- not the rush att market, the push, or the missing probability
    assert list(d.market) == ["rec yds", "receptions", "rec yds"]
    assert list(d.p_mkt_over.round(2)) == [0.55, 0.55, 0.40]            # an UNDER lean at 60% means the market had the OVER at 40%
    assert list(d.over) == [1.0, 0.0, 1.0]


def test_prepare_scores_rows_by_model_version_with_older_rows_as_v1():
    lh = _lines(4, **{"model_ver": [np.nan, np.nan, 3, 3]})
    d = ME.prepare(lh)
    assert sorted(d.ver.unique()) == [1, 3]
    assert ME.prepare(_lines(4)).ver.unique().tolist() == [1]              # no column at all: all v1


def test_prepare_on_nothing_is_empty():
    assert ME.prepare(pd.DataFrame()).empty and ME.prepare(None).empty
    assert ME.prepare(pd.DataFrame({"player": ["a"]})).empty


def test_fit_logit_recovers_planted_coefficients_with_an_interval_that_covers_them():
    rng = np.random.default_rng(1)
    n = 3000
    x = rng.normal(size=n)
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(-0.2 + 0.8 * x)))).astype(float)
    b, lo, hi = ME.fit_logit(x[:, None], y, np.arange(n) // 3, reps=150)
    assert b[0] == pytest.approx(-0.2, abs=0.12) and b[1] == pytest.approx(0.8, abs=0.12)
    assert lo[1] < 0.8 < hi[1]


def test_fit_logit_interval_spans_zero_for_an_unrelated_regressor():
    rng = np.random.default_rng(2)
    n = 1500
    x, y = rng.normal(size=n), (rng.uniform(size=n) < 0.5).astype(float)
    b, lo, hi = ME.fit_logit(x[:, None], y, np.arange(n) // 2, reps=150)
    assert lo[1] < 0 < hi[1]


def test_edge_vs_market_finds_a_planted_edge_beyond_the_price():
    rng = np.random.default_rng(3)
    n = 2500
    z = rng.normal(size=n)
    p_mkt = np.clip(rng.normal(0.5, 0.05, n), 0.3, 0.7)
    over = (rng.uniform(size=n) < 1 / (1 + np.exp(-(logit_ := np.log(p_mkt / (1 - p_mkt))) - 0.4 * z))).astype(float)
    d = pd.DataFrame({"p_mkt_over": p_mkt, "z": z, "over": over, "player": [f"P{i % 400}" for i in range(n)]})
    r = ME.edge_vs_market(d, reps=150)
    assert r["z_coef"][0] == pytest.approx(0.4, abs=0.12) and r["z_coef"][1] > 0
    assert r["move"][0] == pytest.approx(100 * (1 / (1 + np.exp(-r["z_coef"][0])) - 0.5))


def test_report_builds_and_handles_small_versions_and_no_data():
    lh = _lines(400, **{"model_ver": [np.nan] * 360 + [3] * 40}, **{"book %": [52.0] * 400})
    txt = ME.build_report(lh)
    import re
    assert re.search(r"\| v1 \| 3\d\d \|", txt)                  # ~360 minus the odd synthetic push
    assert "too few lines yet" in txt and "## Win rate by the side we leaned" in txt
    assert "No graded lines" in ME.build_report(pd.DataFrame())
