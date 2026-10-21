"""Injury redistribution — skipped unless the nflverse cache exists."""
import pytest

from matchup_model.opp import data as D

_READY = (D.CACHE / "injuries_2024.parquet").exists() and (D.CACHE / "stats_player_week_2024.parquet").exists()
pytestmark = pytest.mark.skipif(not _READY, reason="nflverse cache not downloaded")


def test_play_prob_scale():
    from matchup_model.opp import injuries as I
    assert I.play_prob("Out") == 0.0
    assert I.play_prob("Questionable") == 0.70
    assert I.play_prob("") == 1.0 and I.play_prob(None) == 1.0


def test_kupp_out_redistributes_to_rams_group():
    from matchup_model.opp import injuries as I
    # Cooper Kupp was Out for LAR in weeks 3-5 of 2024: week 3 is NEW news (he played week 2), week 5 is an ONGOING absence
    gm = I.group_multipliers("LAR", "rec", 2024, 3)          # accepts LAR or LA
    assert gm, "expected a non-empty multiplier map when a starter is Out"
    meta = gm.pop("_meta", {})
    assert any("Kupp" in n for n in meta.get("hurt", {}))
    # the out player is zeroed, everyone else is bumped a little above 1 (only the real share of the vacated usage: model.INJ_REDIST)
    bumps = [v for v in gm.values()]
    assert min(bumps) == 0.0
    assert 1.0 < max(bumps) <= I._MULT_CAP
    ongoing = I.group_multipliers("LAR", "rec", 2024, 5)
    ongoing.pop("_meta", None)
    assert min(ongoing.values()) == 0.0 and 1.0 < max(ongoing.values()) < max(bumps)     # an absence that is already in the trailing shares earns a smaller boost than news


def test_healthy_week_no_adjustment():
    from matchup_model.opp import injuries as I
    # not literally guaranteed, but a mid-season week with a set rotation usually returns {}
    empties = sum(1 for wk in (10, 11, 12, 13)
                  if not I.group_multipliers("KC", "rush", 2024, wk))
    assert empties >= 1


def test_opp_line_injury_flag_moves_projection():
    from matchup_model.opp import model as M
    # week 3: Kupp's first game out (new news) -> a small boost for his teammate
    base = M.opp_line("demarcus robinson", "WR", 2024, 3, by="name", injury_adj=False)
    adj = M.opp_line("demarcus robinson", "WR", 2024, 3, by="name", injury_adj=True)
    if base.get("dk_fp") and adj.get("dk_fp"):
        assert adj["dk_fp"] >= base["dk_fp"]          # teammate (Kupp) out -> more work
        assert adj.get("inj_mult", 1.0) > 1.0


def _fake_components():
    """One rotation: a .30 receiver who just got hurt, three teammates at .20 / .15 / .15 (so .50 stays available)."""
    return {"sh": {"out": 0.30, "a": 0.20, "b": 0.15, "c": 0.15}, "avail": {"out": 0.0, "a": 1.0, "b": 1.0, "c": 1.0}, "names": {"out": "Out Guy", "a": "A", "b": "B", "c": "C"},
            "vac_new": 0.30, "vac_old": 0.0, "avail_share": 0.50}


def test_the_proportional_boost_is_scaled_by_the_kind_of_absence(monkeypatch):
    from matchup_model.opp import injuries as I
    from matchup_model.opp import model as M
    monkeypatch.setattr(I, "_group_components", lambda *a: _fake_components())
    monkeypatch.setattr(M, "INJ_REDIST", {"rec": {"new": 1.0, "ongoing": 1.0}, "rush": {"new": 1.0, "ongoing": 1.0}})
    full = I.group_multipliers("AAA", "rec", 2026, 7)
    assert full["out"] == 0.0 and full["a"] == pytest.approx(1 + 0.30 / 0.50)                           # the v1-v3 rule: the whole vacated share, proportionally
    assert full["_meta"] == {"vacated_share": 0.3, "hurt": {"Out Guy": 1.0}}
    monkeypatch.setattr(M, "INJ_REDIST", {"rec": {"new": 0.25, "ongoing": 0.10}, "rush": {"new": 0.50, "ongoing": 0.20}})       # distinct test values, one per path
    assert I.group_multipliers("AAA", "rec", 2026, 7)["b"] == pytest.approx(1 + 0.25 * 0.30 / 0.50)    # a NEW receiving absence: a quarter of the boost is real
    assert I.group_multipliers("AAA", "rush", 2026, 7)["b"] == pytest.approx(1 + 0.50 * 0.30 / 0.50)
    ongoing = dict(_fake_components(), vac_new=0.0, vac_old=0.30)
    monkeypatch.setattr(I, "_group_components", lambda *a: ongoing)
    assert I.group_multipliers("AAA", "rec", 2026, 8)["a"] == pytest.approx(1 + 0.10 * 0.30 / 0.50)    # ongoing: its own, smaller fraction
    assert I.group_multipliers("AAA", "rush", 2026, 8)["a"] == pytest.approx(1 + 0.20 * 0.30 / 0.50)
    mixed = dict(_fake_components(), vac_new=0.20, vac_old=0.10)                                         # a new and an ongoing absence at once add their own parts
    monkeypatch.setattr(I, "_group_components", lambda *a: mixed)
    assert I.group_multipliers("AAA", "rec", 2026, 9)["c"] == pytest.approx(1 + (0.25 * 0.20 + 0.10 * 0.10) / 0.50)
    monkeypatch.setattr(M, "INJ_REDIST", {"rec": {"new": 0.0, "ongoing": 0.0}, "rush": {"new": 0.0, "ongoing": 0.0}})
    assert I.group_multipliers("AAA", "rec", 2026, 9)["c"] == pytest.approx(1.0)                        # a zero fraction leaves only the own-status multiplier
    monkeypatch.setattr(I, "_group_components", lambda *a: None)
    assert I.group_multipliers("AAA", "rec", 2026, 8) == {}                                              # nobody hurt


def test_the_cap_still_applies_to_a_legacy_sized_boost(monkeypatch):
    from matchup_model.opp import injuries as I
    from matchup_model.opp import model as M
    comp = dict(_fake_components(), sh={"out": 0.60, "a": 0.05}, avail={"out": 0.0, "a": 1.0}, vac_new=0.60, avail_share=0.05)
    monkeypatch.setattr(I, "_group_components", lambda *a: comp)
    monkeypatch.setattr(M, "INJ_REDIST", {"rec": {"new": 1.0, "ongoing": 1.0}, "rush": {"new": 1.0, "ongoing": 1.0}})
    assert I.group_multipliers("AAA", "rec", 2026, 7)["a"] == I._MULT_CAP                                # 1 + 0.6/0.05 = 13, capped


def test_a_new_absence_is_told_from_an_ongoing_one():
    from matchup_model.opp import injuries as I
    first = I._group_components("LAR", "rec", 2024, 3)       # Kupp's first game out: he played week 2 (Davis Allen, out since week 2, is the small ongoing part)
    later = I._group_components("LAR", "rec", 2024, 5)       # Kupp had already missed week 4
    assert first["vac_new"] > 0.25 > first["vac_old"] > 0
    assert later["vac_old"] > 0.25 and later["vac_new"] == 0


def test_refreshing_the_data_clears_the_injury_report_and_the_snap_counts():
    from matchup_model.opp import injuries as I
    D.injuries(); D.snaps(); I._group_components("LAR", "rec", 2024, 3)
    assert D.injuries.cache_info().currsize and I._group_components.cache_info().currsize
    D.clear_caches()                                          # a running app that refreshes the nflverse files must not keep yesterday's injury report
    assert D.injuries.cache_info().currsize == 0 and D.snaps.cache_info().currsize == 0
    assert I._group_components.cache_info().currsize == 0 and I._shares_base.cache_info().currsize == 0
