"""Production-pipeline exam (matchup_model.opp.restate): payload construction, constants handling, and the report."""
import numpy as np
import pandas as pd
import pytest

from dfs import props
from matchup_model.opp import model as M
from matchup_model.opp import restate as RS


def _lh(n=24):
    rng = np.random.default_rng(3)
    line = rng.uniform(20, 80, n).round(1)
    return pd.DataFrame({
        "season": 2026, "week": np.repeat([1, 2], n // 2), "player": [f"P{i % 12}" for i in range(n)],
        "market": ["rec yds", "receptions"] * (n // 2), "line": line, "our proj": line + rng.normal(0, 6, n),
        "lean": rng.choice(["OVER", "UNDER"], n), "conf": rng.choice(["lean", "solid", "strong"], n),
        "role_ratio": rng.uniform(0.8, 1.25, n), "actual": rng.uniform(0, 100, n).round(), "result": "win"})


def test_payload_has_an_over_and_under_pair_at_each_stored_line():
    rows = pd.DataFrame({"player": ["A", "B"], "market": ["rec yds", "receptions"], "line": [55.5, 4.5]})
    raw = RS.raw_for(rows)
    mk = {m["key"]: m for m in raw["bookmakers"][0]["markets"]}
    assert set(mk) == {"player_reception_yds", "player_receptions"}
    outs = mk["player_reception_yds"]["outcomes"]
    assert [(o["name"], o["description"], o["point"]) for o in outs] == [("Over", "A", 55.5), ("Under", "A", 55.5)]


def test_restate_runs_each_pull_under_the_given_constants_and_restores_the_shipped_ones(monkeypatch):
    seen = []

    def fake_edges(raw, week):
        seen.append((M.K_SHARE_REC, M.K_YPT, M.DEF_TGT_B, week))
        rows = []
        for m in raw["bookmakers"][0]["markets"]:
            for o in m["outcomes"][::2]:
                rows.append({"player": o["description"], "market": props._NICE[props.MARKETS[m["key"]]],
                             "line": o["point"], "lean": "OVER", "conf": "solid", "role_ratio": 1.0, "our proj": o["point"] + 5})
        return pd.DataFrame(rows)

    monkeypatch.setattr(props, "edges_from_raw", fake_edges)
    keep = (M.K_SHARE_REC, M.K_YPT, M.DEF_TGT_B)
    lh = pd.DataFrame({"week": [1, 1, 2], "player": ["A", "B", "A"], "market": ["rec yds", "receptions", "rec yds"],
                       "line": [50.0, 4.5, 40.0], "actual": [60.0, 3.0, 40.0]})
    d = RS.restate(lh, RS.BEFORE)
    assert seen == [(3.0, 28.0, 0.0, 1), (3.0, 28.0, 0.0, 2)]               # one pull per week, under the previous constants
    assert (M.K_SHARE_REC, M.K_YPT, M.DEF_TGT_B) == keep                    # the shipped ones are back afterwards
    got = d.set_index(["week", "player"]).result
    assert got[(1, "A")] == "win" and got[(1, "B")] == "loss" and got[(2, "A")] == "push"   # OVER vs actual 60 / 3 / 40 on 50 / 4.5 / 40


def test_restate_restores_the_constants_even_if_a_pull_fails(monkeypatch):
    def boom(raw, week):
        raise RuntimeError("no projections")

    monkeypatch.setattr(props, "edges_from_raw", boom)
    keep = (M.K_SHARE_REC, M.K_YPT, M.DEF_TGT_B)
    lh = pd.DataFrame({"week": [1], "player": ["A"], "market": ["rec yds"], "line": [50.0], "actual": [60.0]})
    with pytest.raises(RuntimeError):
        RS.restate(lh, RS.BEFORE)
    assert (M.K_SHARE_REC, M.K_YPT, M.DEF_TGT_B) == keep


def test_report_builds_from_two_restated_frames():
    lh = _lh()
    before = lh.copy()
    after = lh.assign(lean=np.where(lh.lean == "OVER", "UNDER", "OVER"), result=np.where(lh.actual > lh.line, "loss", "win"))
    txt = RS.build_report(lh, before, after)
    for head in ("## Win rate of the leans", "## Paired difference", "## Tier tables", "## Filters and role agreement",
                 "**Harness check**"):
        assert head in txt
    assert "same lean 100%" in txt                                          # before == stored here, so the harness check is perfect
