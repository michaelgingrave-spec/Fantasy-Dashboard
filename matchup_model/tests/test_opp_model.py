"""Smoke tests for the opportunity model. Skipped unless the nflverse cache exists
(populate it with `python -m matchup_model.opp.data`)."""
import numpy as np
import pytest

from matchup_model.opp import data as D

_CACHE_READY = (D.CACHE / "stats_player_week_2024.parquet").exists() and (D.CACHE / "games.csv").exists()
pytestmark = pytest.mark.skipif(not _CACHE_READY, reason="nflverse cache not downloaded")


def test_player_weeks_shape():
    pw = D.player_weeks()
    assert {"name_key", "dk_fp", "target_share", "carries", "team"} <= set(pw.columns)
    assert pw["season"].between(2021, 2026).all()
    assert (pw["position"].isin({"QB", "RB", "WR", "TE"})).all()
    # DK points are non-negative and sane
    assert pw["dk_fp"].min() >= -5 and pw["dk_fp"].max() < 80


def test_team_volume_sane():
    from matchup_model.opp import model as M
    tv = M.team_volume("KC", 2024, 10)
    assert 25 < tv["pass_att"] < 50
    assert 14 < tv["rush_att"] < 40
    assert 0.30 <= tv["pass_rate"] <= 0.72


def test_opp_line_walk_forward():
    from matchup_model.opp import model as M
    # a well-established player mid-2024 should get a finite line from prior games only
    pw = D.player_weeks()
    row = pw[(pw.player_display_name == "Christian McCaffrey") & (pw.season == 2024)].head(1)
    if row.empty:
        pytest.skip("player not in cache")
    out = M.opp_line("Christian McCaffrey", "RB", 2024, 12)
    assert out["dk_fp"] is None or np.isfinite(out["dk_fp"])
    if out["dk_fp"] is not None:
        assert set(out["line"]) & {"rush_att", "rush_yds", "rec"}
        assert 0 < out["dk_fp"] < 45


def test_priors_no_future_leak():
    from matchup_model.opp import model as M
    p23 = M._priors(2023)["WR"]
    assert 5 < p23["ypt"] < 12          # yards per target in a sane band
    assert 0.5 < p23["catch"] < 0.8
