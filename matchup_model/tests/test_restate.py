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
        seen.append((M.K_SHARE_REC, M.K_YPT, M.DEF_TGT_B, M.TEAM_VOLUME_MODEL, week))
        rows = []
        for m in raw["bookmakers"][0]["markets"]:
            for o in m["outcomes"][::2]:
                rows.append({"player": o["description"], "market": props._NICE[props.MARKETS[m["key"]]],
                             "line": o["point"], "lean": "OVER", "conf": "solid", "role_ratio": 1.0, "our proj": o["point"] + 5})
        return pd.DataFrame(rows)

    monkeypatch.setattr(props, "edges_from_raw", fake_edges)
    keep = RS.shipped()
    lh = pd.DataFrame({"week": [1, 1, 2], "player": ["A", "B", "A"], "market": ["rec yds", "receptions", "rec yds"],
                       "line": [50.0, 4.5, 40.0], "actual": [60.0, 3.0, 40.0]})
    d = RS.restate(lh, RS.BEFORE)
    assert seen == [(3.0, 28.0, 0.0, "legacy", 1), (3.0, 28.0, 0.0, "legacy", 2)]    # one pull per week, under the original model
    assert RS.shipped() == keep                                                      # every shipped constant (RB ones too) is back
    got = d.set_index(["week", "player"]).result
    assert got[(1, "A")] == "win" and got[(1, "B")] == "loss" and got[(2, "A")] == "push"   # OVER vs actual 60 / 3 / 40 on 50 / 4.5 / 40


def test_restate_restores_the_constants_even_if_a_pull_fails(monkeypatch):
    def boom(raw, week):
        raise RuntimeError("no projections")

    monkeypatch.setattr(props, "edges_from_raw", boom)
    keep = RS.shipped()
    lh = pd.DataFrame({"week": [1], "player": ["A"], "market": ["rec yds"], "line": [50.0], "actual": [60.0]})
    with pytest.raises(RuntimeError):
        RS.restate(lh, RS.BEFORE)
    assert RS.shipped() == keep


def test_report_builds_from_the_restated_frames_of_each_version():
    lh = _lh()
    v1 = lh.copy()
    v2 = lh.assign(lean=np.where(lh.lean == "OVER", "UNDER", "OVER"), result=np.where(lh.actual > lh.line, "loss", "win"))
    v3 = lh.assign(result=np.where(lh.actual > lh.line, "win", "loss"))
    txt = RS.build_report(lh, {"v1 (original)": v1, "v2 (recal)": v2, "v3 (current)": v3})
    for head in ("## Win rate of the leans", "## Paired difference (v3 (current) minus each earlier version)", "## Tier tables",
                 "## Filters and role agreement", "**Harness check**"):
        assert head in txt
    assert "- vs v1 (original), all markets:" in txt and "- vs v2 (recal), receiving markets:" in txt
    assert "same lean 100%" in txt                                          # v1 == stored here, so the harness check is perfect
    assert "v2 (recal), restated" in txt and "v1 (original), restated" not in txt   # tier tables: stored + each later version


def test_versions_carry_the_volume_model_and_rb_constants_they_ran_on():
    assert RS.BEFORE["TEAM_VOLUME_MODEL"] == "legacy" and RS.V2["TEAM_VOLUME_MODEL"] == "legacy"
    assert RS.shipped()["TEAM_VOLUME_MODEL"] == M.TEAM_VOLUME_MODEL
    for v in (RS.BEFORE, RS.V2):                                   # the RB model was not part of v2: both earlier versions had the original RB
        assert (v["K_CARRY"], v["RB_CARRY_PRIOR"], v["RB_YDS_CATCH_FACTOR"]) == (3.0, 0.42, True)
        assert all(x == 1.0 for x in v["RB_SCALE"].values())
    assert set(RS.RB_KEYS) <= set(RS.shipped()) and set(RS.RB_KEYS) <= set(RS.BEFORE)
