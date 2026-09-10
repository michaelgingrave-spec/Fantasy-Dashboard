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
    # Cooper Kupp was Out for LAR in weeks 3-5 of 2024
    gm = I.group_multipliers("LAR", "rec", 2024, 5)          # accepts LAR or LA
    assert gm, "expected a non-empty multiplier map when a starter is Out"
    meta = gm.pop("_meta", {})
    assert any("Kupp" in n for n in meta.get("hurt", {}))
    # the out player is zeroed, everyone else is bumped above 1
    bumps = [v for v in gm.values()]
    assert min(bumps) == 0.0
    assert max(bumps) > 1.0 and max(bumps) <= I._MULT_CAP


def test_healthy_week_no_adjustment():
    from matchup_model.opp import injuries as I
    # not literally guaranteed, but a mid-season week with a set rotation usually returns {}
    empties = sum(1 for wk in (10, 11, 12, 13)
                  if not I.group_multipliers("KC", "rush", 2024, wk))
    assert empties >= 1


def test_opp_line_injury_flag_moves_projection():
    from matchup_model.opp import model as M
    base = M.opp_line("demarcus robinson", "WR", 2024, 5, by="name", injury_adj=False)
    adj = M.opp_line("demarcus robinson", "WR", 2024, 5, by="name", injury_adj=True)
    if base.get("dk_fp") and adj.get("dk_fp"):
        assert adj["dk_fp"] >= base["dk_fp"]          # teammate (Kupp) out -> more work
        assert adj.get("inj_mult", 1.0) > 1.0
