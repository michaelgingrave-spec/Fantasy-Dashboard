"""RB recalibration: the model's RB constants and the calibration helpers."""
import numpy as np
import pandas as pd
import pytest

from matchup_model.opp import data as D
from matchup_model.opp import model as M
from matchup_model.opp import rb_calibration as RB

needs_cache = pytest.mark.skipif(
    not ((D.CACHE / "stats_player_week_2024.parquet").exists() and (D.CACHE / "games.csv").exists()),
    reason="nflverse cache not downloaded")


@pytest.fixture(autouse=True)
def _fresh_caches():
    M.opp_line.cache_clear(); M.team_volume.cache_clear()
    yield
    M.opp_line.cache_clear(); M.team_volume.cache_clear()


def _top_rb(season=2025):
    pw = D.player_weeks()
    d = pw[(pw.season == season) & (pw.position == "RB")]
    return d.groupby("name_key").carries.sum().idxmax()


def test_shipped_rb_constants_are_the_recalibrated_ones():
    assert M.K_CARRY < 3.0 and M.RB_CARRY_PRIOR < 0.42           # committee backs are no longer pulled up to a 0.42 share
    assert M.RB_YDS_CATCH_FACTOR is False                        # the catch rate is no longer counted twice
    assert set(M.RB_SCALE) == {"rush_att", "rush_yds", "rec", "rec_yds", "td"}
    assert all(0.7 < v < 1.2 for v in M.RB_SCALE.values())
    assert M.MODEL_VERSION >= 3 and M.TEAM_VOLUME_MODEL == "fitted"


@needs_cache
def test_rb_scale_multiplies_each_stat_and_the_td_multiplier_scales_both_td_kinds(monkeypatch):
    nk = _top_rb()
    unit = {"rush_att": 1.0, "rush_yds": 1.0, "rec": 1.0, "rec_yds": 1.0, "td": 1.0}
    monkeypatch.setattr(M, "RB_SCALE", unit)
    raw = M.opp_line(nk, "RB", 2025, 12)["line"]
    M.opp_line.cache_clear()
    monkeypatch.setattr(M, "RB_SCALE", {"rush_att": 0.9, "rush_yds": 0.8, "rec": 0.7, "rec_yds": 0.6, "td": 1.25})
    got = M.opp_line(nk, "RB", 2025, 12)["line"]
    for stat, k in (("rush_att", 0.9), ("rush_yds", 0.8), ("rec", 0.7), ("rec_yds", 0.6), ("rush_td", 1.25), ("rec_td", 1.25)):
        assert got[stat] == pytest.approx(k * raw[stat], rel=3e-3, abs=2e-3), stat


@needs_cache
def test_rb_receiving_yards_no_longer_count_the_catch_rate_twice(monkeypatch):
    nk = _top_rb()
    monkeypatch.setattr(M, "RB_SCALE", {"rush_att": 1.0, "rush_yds": 1.0, "rec": 1.0, "rec_yds": 1.0, "td": 1.0})
    monkeypatch.setattr(M, "RB_YDS_CATCH_FACTOR", True)
    old = M.opp_line(nk, "RB", 2025, 12)["line"]["rec_yds"]
    M.opp_line.cache_clear()
    monkeypatch.setattr(M, "RB_YDS_CATCH_FACTOR", False)
    new = M.opp_line(nk, "RB", 2025, 12)["line"]["rec_yds"]
    assert new == pytest.approx(old / M.RB_CATCH, rel=3e-3)


@needs_cache
def test_lighter_carry_shrinkage_projects_a_committee_back_below_the_old_prior(monkeypatch):
    pw = D.player_weeks()
    d = pw[(pw.season == 2025) & (pw.position == "RB") & (pw.carries > 0)]
    # a back who carried in a handful of games at a small share: the old 0.42 prior inflated him
    per = d.groupby("name_key").agg(g=("week", "size"), c=("carries", "sum"))
    nk = per[(per.g >= 5) & (per.c / per.g < 4)].index[0]
    monkeypatch.setattr(M, "RB_SCALE", {"rush_att": 1.0, "rush_yds": 1.0, "rec": 1.0, "rec_yds": 1.0, "td": 1.0})
    monkeypatch.setattr(M, "K_CARRY", 3.0); monkeypatch.setattr(M, "RB_CARRY_PRIOR", 0.42)
    old = M.opp_line(nk, "RB", 2025, 14)["line"]["rush_att"]
    M.opp_line.cache_clear()
    monkeypatch.setattr(M, "K_CARRY", 1.0); monkeypatch.setattr(M, "RB_CARRY_PRIOR", 0.25)
    new = M.opp_line(nk, "RB", 2025, 14)["line"]["rush_att"]
    assert new < old


# ── the calibration helpers ───────────────────────────────────────────────────────────────────────────────
def _frame():
    return pd.DataFrame({
        "p_rush_att": [10.0, 10.0, 2.0, 2.0], "p_rush_yds": [45.0, 45.0, 9.0, 9.0], "p_rec": [2.0, 2.0, 1.0, 1.0], "p_rec_yds": [14.0, 14.0, 7.0, 7.0],
        "carries": [9.0, 9.0, 0.0, 0.0], "rushing_yards": [36.0, 36.0, 0.0, 0.0], "receptions": [1.0, 1.0, 5.0, 5.0],
        "receiving_yards": [7.0, 7.0, 40.0, 40.0], "p_rush_td": [0.4, 0.4, 0.1, 0.1], "p_rec_td": [0.1, 0.1, 0.0, 0.0],
        "rushing_tds": [1.0, 0.0, 0.0, 0.0], "receiving_tds": [0.0, 0.0, 0.0, 1.0]})


def test_involvement_decides_who_each_scale_is_learned_on():
    d = _frame()
    assert list(RB._involved(d, "rush_att")) == [True, True, False, False]       # 6+ projected carries
    assert list(RB._involved(d, "rec")) == [True, True, False, False]            # 1.5+ projected receptions


def test_scales_use_only_the_involved_backs_and_td_pools_rushing_and_receiving():
    sc = RB._scales(_frame())
    assert sc["rush_att"] == pytest.approx(0.9) and sc["rush_yds"] == pytest.approx(0.8)     # the 0-carry depth backs don't dilute them
    assert sc["rec"] == pytest.approx(0.5) and sc["rec_yds"] == pytest.approx(0.5)
    assert sc["td"] == pytest.approx(1.0 / 1.0)        # involved rows: 1 rushing TD scored vs (0.4+0.1)*2 expected


def test_the_original_and_current_settings_differ_where_they_should():
    assert RB.ORIGINAL["TEAM_VOLUME_MODEL"] == "legacy" and RB.NEW["TEAM_VOLUME_MODEL"] == "fitted"
    assert RB.ORIGINAL["K_CARRY"] == 3.0 and RB.ORIGINAL["RB_CARRY_PRIOR"] == 0.42 and RB.ORIGINAL["RB_YDS_CATCH_FACTOR"] is True
    assert (RB.NEW["K_CARRY"], RB.NEW["RB_CARRY_PRIOR"], RB.NEW["RB_YDS_CATCH_FACTOR"]) == (M.K_CARRY, M.RB_CARRY_PRIOR, M.RB_YDS_CATCH_FACTOR)


@needs_cache
def test_project_leaves_the_rb_constants_as_it_found_them():
    before = {k: getattr(M, k) for k in RB.RB_KEYS}
    u = RB.RC.universe_rows(("RB",))
    got = RB.project(RB.ORIGINAL, u.query("no_row == False").head(4))
    assert {k: getattr(M, k) for k in RB.RB_KEYS} == before
    assert {"p_rush_att", "p_rush_yds", "p_rec", "p_rec_yds", "fp"} <= set(got.columns)
