"""The thin-history projection (`blend.blended_line(thin=True)`, `model.opp_line(min_games=1)`): a quarterback with one game, on the 2024 box scores (Bo Nix, week 2: 42 passes in week 1)."""
import numpy as np
import pytest

from matchup_model.opp import blend
from matchup_model.opp import model as M

NIX = "bo nix"


def test_the_normal_path_still_wants_three_games_and_the_thin_path_projects_from_one():
    assert M.opp_line(NIX, "QB", 2024, 2)["dk_fp"] is None and "1 prior games" in M.opp_line(NIX, "QB", 2024, 2)["reason"]
    thin = M.opp_line(NIX, "QB", 2024, 2, min_games=1)
    assert thin["games"] == 1 and thin["dk_fp"] > 5 and {"pass_att", "pass_yds", "pass_td", "int", "rush_yds", "rush_td"} <= set(thin["line"])
    assert M.opp_line(NIX, "QB", 2024, 1, min_games=1)["dk_fp"] is None                   # no games at all: still nothing to project


def test_a_player_with_enough_games_gets_the_same_line_either_way():
    a = M.opp_line("christian mccaffrey", "RB", 2024, 12)
    b = M.opp_line("christian mccaffrey", "RB", 2024, 12, min_games=1)
    assert a["line"] == b["line"] and a["dk_fp"] == b["dk_fp"] and a["games"] == b["games"] >= M.MIN_GAMES


def test_a_thin_quarterback_gets_the_leagues_rushing_not_one_games():
    line = M.opp_line(NIX, "QB", 2024, 2, min_games=1)["line"]
    prior = M._priors(2024)["QB"]
    assert line["rush_yds"] == pytest.approx(prior["rush_yds_g"], abs=1e-3) and line["rush_td"] == pytest.approx(prior["rush_td_g"], abs=1e-3)
    assert 8 < prior["rush_yds_g"] < 30 and 0.05 < prior["rush_td_g"] < 0.5                  # his own week 1 had a rushing touchdown: 1.0 a game would be absurd
    assert line["rush_td"] < 0.5
    full = M.opp_line("josh allen", "QB", 2024, 12)["line"]                                    # a quarterback with a full history keeps his own rushing
    assert full["rush_yds"] != pytest.approx(prior["rush_yds_g"], abs=1.0)


def test_blended_line_is_empty_for_a_thin_player_by_default_and_marked_opp_thin_when_asked():
    assert blend.blended_line(NIX, "QB", 2024, 2).get("fp") is None
    out = blend.blended_line(NIX, "QB", 2024, 2, thin=True)
    assert out["source"] == "opp_thin" and out["games"] == 1 and out["fp"] > 5 and out["line"]["pass_att"] > 20
    assert "trailing_fp" not in out and "opp_fp" not in out                                  # no trailing blend: a cameo is not a trailing level
    assert blend.blended_line("nobody atall", "QB", 2024, 2, thin=True).get("fp") is None and blend.blended_line("nobody atall", "QB", 2024, 2, thin=True) is not None
    assert not np.isnan(out["fp"])
