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
    assert "## Where v3 (current) differs from v2 (recal)" in txt
    flipped = [r for r in txt.splitlines() if r.startswith("| lean flipped by v3 (current)")][0]
    assert f"| {len(lh)} |" in flipped and "-> " in flipped                    # v2 took the other side of every line


def test_versions_carry_the_volume_model_and_rb_constants_they_ran_on():
    assert RS.BEFORE["TEAM_VOLUME_MODEL"] == "legacy" and RS.V2["TEAM_VOLUME_MODEL"] == "legacy"
    assert RS.shipped()["TEAM_VOLUME_MODEL"] == M.TEAM_VOLUME_MODEL
    for v in (RS.BEFORE, RS.V2):                                   # the RB model was not part of v2: both earlier versions had the original RB
        assert (v["K_CARRY"], v["RB_CARRY_PRIOR"], v["RB_YDS_CATCH_FACTOR"]) == (3.0, 0.42, True)
        assert all(x == 1.0 for x in v["RB_SCALE"].values())
    assert set(RS.RB_KEYS) <= set(RS.shipped()) and set(RS.RB_KEYS) <= set(RS.BEFORE)


def test_versions_carry_the_injury_rule_they_ran_on_and_restate_restores_it(monkeypatch):
    from matchup_model.opp.receiving_calibration import INJ_LEGACY
    assert INJ_LEGACY == {"rec": {"new": 1.0, "ongoing": 1.0}, "rush": {"new": 1.0, "ongoing": 1.0}}       # the v1-v3 rule: the whole vacated share
    assert RS.BEFORE["INJ_REDIST"] == INJ_LEGACY and RS.V2["INJ_REDIST"] == INJ_LEGACY
    assert RS.shipped()["INJ_REDIST"] == M.INJ_REDIST and RS.shipped()["INJ_REDIST"] is not M.INJ_REDIST     # a copy, so it cannot be changed through the dict
    seen = []

    def fake_edges(raw, week):
        seen.append(M.INJ_REDIST)
        return pd.DataFrame([{"player": "A", "market": "rec yds", "line": 50.0, "lean": "OVER", "conf": "solid", "role_ratio": 1.0, "our proj": 55.0}])

    monkeypatch.setattr(props, "edges_from_raw", fake_edges)
    keep = M.INJ_REDIST
    lh = pd.DataFrame({"week": [1], "player": ["A"], "market": ["rec yds"], "line": [50.0], "actual": [60.0]})
    RS.restate(lh, RS.BEFORE)
    assert seen == [INJ_LEGACY] and M.INJ_REDIST == keep                  # the old rule during the pull, the shipped one back afterwards


def test_the_differs_section_looks_only_at_the_lines_the_new_version_changed():
    n = 40
    base = pd.DataFrame({"week": 1, "player": [f"P{i}" for i in range(n)], "market": "rec yds", "line": 50.0, "actual": 60.0, "our proj": 70.0, "lean": "OVER",
                         "conf": "solid", "role_ratio": 1.0, "result": "win"})
    new = base.copy()
    new.loc[:9, "our proj"] = 40.0                                         # 10 lines: the projection falls 43% and the lean flips to UNDER, which loses (actual 60 > 50)
    new.loc[:9, "lean"], new.loc[:9, "result"] = "UNDER", "loss"
    new.loc[10:14, "our proj"] = 70.0 * 1.06                              # 5 lines: moved 6%, same lean
    old, lh = base, base.assign(season=2026)
    txt = RS.build_report(lh, {"vA (before)": old, "vB (now)": new})
    rows = {r.split("|")[1].strip(): [c.strip() for c in r.split("|")[2:-1]] for r in txt.splitlines() if r.startswith("| ") and "|--" not in r}
    assert rows["all lines"][0] == "40" and rows["projection moved >= 1%"][0] == "15" and rows["projection moved >= 5%"][0] == "15"
    assert rows["projection moved >= 10%"][0] == "10" and rows["lean flipped by vB (now)"][0] == "10" and rows["projection moved < 1%"][0] == "25"
    assert rows["lean flipped by vB (now)"][1:3] == ["100.0%", "0.0%"]            # vA wins all ten, vB loses all ten
    assert rows["lean flipped by vB (now)"][4] == "100% -> 0%"
    assert rows["projection moved < 1%"][1:3] == ["100.0%", "100.0%"]              # untouched lines are identical
    # one version alone has nothing to compare with
    assert "## Where" not in RS.build_report(lh, {"vA (before)": old})


def test_every_past_version_sets_every_constant_so_none_of_the_current_ones_leak_into_it():
    keys = set(RS.shipped())
    for name, v in (("v1", RS.BEFORE), ("v2", RS.V2), ("v3", RS.V3)):
        assert keys <= set(v), f"{name} leaves {sorted(keys - set(v))} at whatever the current model has"


def test_v3_is_the_model_as_shipped_before_the_injury_rule_changed():
    from matchup_model.opp.receiving_calibration import INJ_LEGACY
    v = RS.V3
    assert v["INJ_REDIST"] == INJ_LEGACY and v["TEAM_VOLUME_MODEL"] == "fitted" and (v["K_SHARE_REC"], v["K_YPT"], v["DEF_TGT_B"]) == (1.0, 60.0, 0.09)
    assert v["REC_SCALE"]["WR"] == {"tgt": 0.847, "rec": 0.840, "rec_yds": 0.848, "td": 0.933}
    assert v["REC_SCALE"]["TE"] == {"tgt": 0.867, "rec": 0.883, "rec_yds": 0.893, "td": 1.010}
    assert v["RB_SCALE"] == {"rush_att": 0.969, "rush_yds": 0.973, "rec": 0.911, "rec_yds": 0.872, "td": 1.017}
    assert (v["K_CARRY"], v["RB_CARRY_PRIOR"], v["RB_YDS_CATCH_FACTOR"]) == (1.0, 0.25, False)


def test_a_uniform_recalibration_is_not_counted_as_a_move():
    base = pd.DataFrame({"week": 1, "player": [f"P{i}" for i in range(20)], "market": "receptions", "line": 4.5, "actual": 5.0, "our proj": 5.0, "lean": "OVER",
                         "conf": "solid", "role_ratio": 1.0, "result": "win"})
    scaled = base.assign(**{"our proj": 5.0 * 1.06})                            # every line rises 6%: a new scale constant
    scaled.loc[:2, "our proj"] = 5.0 * 1.06 * 0.80                              # three lines fall 20% relative to the rest: the part v4 actually did
    txt = RS.build_report(base.assign(season=2026), {"vA": base, "vB": scaled})
    rows = {r.split("|")[1].strip(): [c.strip() for c in r.split("|")[2:-1]] for r in txt.splitlines() if r.startswith("| ") and "|--" not in r}
    assert rows["projection moved >= 5%"][0] == "3" and rows["projection moved < 1%"][0] == "17"
