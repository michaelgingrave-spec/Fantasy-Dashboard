"""Fitted team volume: the feature builder (no look-ahead), applying a fit, picking the right season's fit, the spread sign, and the
calibration helpers."""
import numpy as np
import pandas as pd
import pytest

from matchup_model.opp import data as D
from matchup_model.opp import model as M
from matchup_model.opp import team_volume as TV
from matchup_model.opp import team_volume_calibration as TC

needs_cache = pytest.mark.skipif(
    not ((D.CACHE / "stats_player_week_2024.parquet").exists() and (D.CACHE / "games.csv").exists()),
    reason="nflverse cache not downloaded")


@pytest.fixture(autouse=True)
def _fresh_caches():
    TV.features.cache_clear(); M.team_volume.cache_clear(); M.opp_line.cache_clear()
    yield
    TV.features.cache_clear(); M.team_volume.cache_clear(); M.opp_line.cache_clear()


def _tables(n_weeks=6, team="AAA", opp="BBB"):
    """Tiny synthetic history: AAA passes 30,31,...; BBB's defence has faced 20,21,...; one game per week against each other."""
    rows_a, rows_b = [], []
    for w in range(1, n_weeks + 1):
        rows_a.append({"season": 2025, "week": w, "team": team, "opponent_team": opp, "attempts": 29 + w, "carries": 20 + w, "plays": 60 + w})
        rows_b.append({"season": 2025, "week": w, "team": opp, "opponent_team": team, "attempts": 19 + w, "carries": 30 + w, "plays": 70 + w})
    own = {team: pd.DataFrame(rows_a), opp: pd.DataFrame(rows_b)}
    faced = {opp: pd.DataFrame(rows_a), team: pd.DataFrame(rows_b)}      # what each defence has faced = the other offence's rows
    games = {(2025, w, team): {"season": 2025, "week": w, "team": team, "opp": opp, "is_home": 1, "spread": -3.0, "total_line": 47.0,
                               "roof": "dome", "wind": np.nan} for w in range(1, n_weeks + 2)}
    games.update({(2025, w, opp): {"season": 2025, "week": w, "team": opp, "opp": team, "is_home": 0, "spread": 3.0, "total_line": 47.0,
                                   "roof": "outdoors", "wind": 12.0} for w in range(1, n_weeks + 2)})
    return own, faced, games


def test_features_use_only_games_before_the_week(monkeypatch):
    monkeypatch.setattr(TV, "_tables", lambda: _tables())
    f4 = TV.features("AAA", 2025, 4)                      # weeks 1-3 only
    assert f4["n_prior"] == 3
    assert f4["tr_att"] == pytest.approx(TV.ewma([30, 31, 32]))
    # an outcome in week 4 or later must not move week-4 features
    own, faced, games = _tables()
    own["AAA"].loc[own["AAA"].week >= 4, "attempts"] = 99
    TV.features.cache_clear()
    monkeypatch.setattr(TV, "_tables", lambda: (own, faced, games))
    assert TV.features("AAA", 2025, 4)["tr_att"] == pytest.approx(TV.ewma([30, 31, 32]))


def test_features_read_the_opponents_defence_and_pace_and_the_game_environment(monkeypatch):
    monkeypatch.setattr(TV, "_tables", lambda: _tables())
    f = TV.features("AAA", 2025, 5)
    assert f["op_df_att"] == pytest.approx(TV.ewma([30, 31, 32, 33]))        # what BBB's defence has faced = AAA's attempts
    assert f["op_plays"] == pytest.approx(TV.ewma([71, 72, 73, 74]))         # BBB's own pace
    assert (f["spread"], f["total_line"], f["is_home"], f["dome"], f["wind"]) == (-3.0, 47.0, 1.0, 1.0, 0.0)   # NaN wind -> 0
    g = TV.features("BBB", 2025, 5)
    assert (g["spread"], g["is_home"], g["dome"], g["wind"]) == (3.0, 0.0, 0.0, 12.0)


def test_features_are_none_without_history_or_a_scheduled_game(monkeypatch):
    monkeypatch.setattr(TV, "_tables", lambda: _tables())
    assert TV.features("AAA", 2025, 2) is None            # one prior game: fewer than MIN_PRIOR
    assert TV.features("AAA", 2025, 99) is None           # not on the schedule
    assert TV.features("ZZZ", 2025, 5) is None


def _fit(intercept=30.0, beta=(2.0, -1.0), clip=True):
    m = {"features": ["tr_att", "spread"], "mean": [33.0, 0.0], "sd": [4.0, 6.0], "beta": list(beta), "intercept": intercept}
    return {"att": m, "car": {**m, "features": ["tr_car", "spread"]}}


def test_apply_fit_standardises_and_adds_the_intercept():
    got = TV.apply_fit(_fit(), {"tr_att": 37.0, "spread": 6.0}, "att")        # z = (+1, +1) -> 30 + 2 - 1
    assert got == pytest.approx(31.0)


def test_apply_fit_imputes_the_training_average_for_missing_inputs_and_clips():
    assert TV.apply_fit(_fit(), {"tr_att": 41.0, "spread": np.nan}, "att") == pytest.approx(34.0)    # spread missing -> z 0
    assert TV.apply_fit(_fit(), {"tr_att": 41.0}, "att") == pytest.approx(34.0)                      # absent entirely
    assert TV.apply_fit(_fit(intercept=500.0), {"tr_att": 33.0, "spread": 0.0}, "att") == TV.CLIP["att"][1]
    assert TV.apply_fit(_fit(intercept=-500.0), {"tr_att": 33.0, "spread": 0.0}, "att") == TV.CLIP["att"][0]


def test_the_fit_used_is_the_latest_one_trained_entirely_before_the_season(monkeypatch):
    fits = {"2022": {"id": "a"}, "2023": {"id": "b"}, "2024": {"id": "c"}}
    monkeypatch.setattr(TV, "_coefs", lambda: fits)
    assert TV.fit_for(2024)["id"] == "b"                  # trained through 2023 -- never through 2024 itself
    assert TV.fit_for(2026)["id"] == "c"
    assert TV.fit_for(2023)["id"] == "a"
    assert TV.fit_for(2019)["id"] == "a"                  # older than any fit: the earliest
    monkeypatch.setattr(TV, "_coefs", lambda: {})
    assert TV.fit_for(2024) is None


def test_predict_is_none_when_it_cannot_run(monkeypatch):
    monkeypatch.setattr(TV, "_coefs", lambda: {})
    monkeypatch.setattr(TV, "_tables", lambda: _tables())
    assert TV.predict("AAA", 2025, 5) is None             # no coefficients -> the model falls back to the formula


def test_shipped_coefficients_have_the_right_sign_on_the_spread():
    """The pre-v3 formula had favourites passing more and running less. Favourites (negative spread) must run more than
    otherwise identical underdogs, and the effect on rushing must be sizeable."""
    fit = TV.fit_for(2026)
    assert fit is not None and set(fit) >= {"att", "car"}
    base = {"tr_att": 33.0, "tr_car": 27.0, "tr_plays": 62.0, "op_df_att": 33.0, "op_df_car": 27.0, "op_plays": 62.0, "spread": 0.0,
            "total_line": 44.5, "is_home": 0.5, "dome": 0.0, "wind": 0.0}
    fav, dog = {**base, "spread": -10.0}, {**base, "spread": +10.0}
    assert TV.apply_fit(fit, fav, "car") - TV.apply_fit(fit, dog, "car") > 2.5
    assert abs(TV.apply_fit(fit, fav, "att") - TV.apply_fit(fit, dog, "att")) < 1.5   # attempts hardly move with the spread


def test_coefficient_files_cover_every_walk_forward_cutoff():
    fits = TV._coefs()
    assert set(fits) >= {"2022", "2023", "2024", "2025"}
    for k, fit in fits.items():
        assert fit["train_through"] == int(k)
        for tgt in ("att", "car"):
            m = fit[tgt]
            assert m["features"] == list(TV.FEATURES[tgt]) and len(m["beta"]) == len(m["mean"]) == len(m["sd"]) == len(m["features"])
            assert all(s > 0 for s in m["sd"]) and m["n"] > 500


def test_game_environment_feature_names_match_the_games_table_columns():
    # a rename in data.games() would silently turn these into NaN -> training average; fail loudly instead
    cols = set(D.games().columns) if (D.CACHE / "games.csv").exists() else {"spread", "total_line", "is_home", "roof", "wind", "opp"}
    assert {"spread", "total_line", "is_home", "roof", "wind", "opp"} <= cols


# ── the model ─────────────────────────────────────────────────────────────────────────────────────────────
def test_team_volume_uses_the_fitted_model_and_reports_a_consistent_plays_and_pass_rate(monkeypatch):
    monkeypatch.setattr(TV, "predict", lambda team, season, week: (31.0, 27.0))
    monkeypatch.setattr(M, "vegas_row", lambda team, season, week: {"team_total": 24.5})
    tv = M.team_volume("AAA", 2025, 5)
    assert tv["pass_att"] == 31.0 and tv["rush_att"] == 27.0 and tv["team_total"] == 24.5
    dropbacks = 31.0 / (1 - M.SACK_FRAC)
    assert tv["plays"] == pytest.approx(dropbacks + 27.0 + M.QB_SCRAMBLE)
    assert tv["pass_rate"] == pytest.approx(dropbacks / tv["plays"])


def test_team_volume_falls_back_to_the_corrected_formula_when_the_model_cannot_run(monkeypatch):
    monkeypatch.setattr(TV, "predict", lambda team, season, week: None)
    ctx = {"plays_pg": 62.0, "pass_rate": 0.56, "n": 8}
    monkeypatch.setattr(M, "team_context", lambda team, season, week: ctx)
    out = {}
    for spread in (-7.0, 7.0):
        monkeypatch.setattr(M, "vegas_row", lambda team, season, week, s=spread: {"spread": s, "total_line": 44.0, "team_total": 22.0})
        M.team_volume.cache_clear()
        out[spread] = M.team_volume("AAA", 2025, 5)
    assert out[7.0]["pass_rate"] > out[-7.0]["pass_rate"]          # the underdog passes more, the favourite runs more
    assert out[-7.0]["rush_att"] > out[7.0]["rush_att"]


def test_the_legacy_setting_reproduces_the_reversed_spread_effect(monkeypatch):
    monkeypatch.setattr(M, "team_context", lambda team, season, week: {"plays_pg": 62.0, "pass_rate": 0.56, "n": 8})
    monkeypatch.setattr(M, "TEAM_VOLUME_MODEL", "legacy")
    res = {}
    for spread in (-7.0, 7.0):
        monkeypatch.setattr(M, "vegas_row", lambda team, season, week, s=spread: {"spread": s, "total_line": 44.0, "team_total": 22.0})
        M.team_volume.cache_clear()
        res[spread] = M.team_volume("AAA", 2025, 5)
    assert res[-7.0]["pass_rate"] > res[7.0]["pass_rate"]          # the old behaviour: favourites projected to pass MORE


@needs_cache
def test_team_volume_is_sane_across_the_league_for_a_live_week():
    for team in ("KC", "BUF", "SEA", "DEN", "LAR", "BAL"):
        tv = M.team_volume(team, 2026, 6)
        assert 22 < tv["pass_att"] < 45 and 18 < tv["rush_att"] < 36, (team, tv)
        assert 0.40 < tv["pass_rate"] < 0.68


# ── the calibration helpers ───────────────────────────────────────────────────────────────────────────────
def test_ridge_recovers_planted_coefficients():
    rng = np.random.default_rng(1)
    n = 600
    df = pd.DataFrame({"a": rng.normal(10, 2, n), "b": rng.normal(0, 5, n)})
    za, zb = (df.a - df.a.mean()) / df.a.std(), (df.b - df.b.mean()) / df.b.std()          # the units the fit works in
    df["y"] = 20 + 1.5 * za - 0.5 * zb + rng.normal(0, 0.05, n)
    m = TC.ridge(df, ("a", "b"), "y", lam=1e-6)
    assert m["intercept"] == pytest.approx(20.0, abs=0.05)
    assert m["beta"][0] == pytest.approx(1.5, abs=0.05) and m["beta"][1] == pytest.approx(-0.5, abs=0.05)
    assert m["n"] == n


def test_ridge_drops_rows_with_missing_inputs_rather_than_imputing_them():
    df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0, 5.0], "y": [1.0, 2.0, 3.0, 4.0, 5.0]})
    assert TC.ridge(df, ("a",), "y")["n"] == 4


def test_the_legacy_helper_matches_the_model_formula_with_the_old_sign(monkeypatch):
    monkeypatch.setattr(M, "team_context", lambda team, season, week: {"plays_pg": 62.0, "pass_rate": 0.56, "n": 8})
    monkeypatch.setattr(M, "vegas_row", lambda team, season, week: {"spread": 5.0, "total_line": 47.0, "team_total": 22.0})
    got = TC.legacy("AAA", 2025, 5)
    ref = M._formula_volume("AAA", 2025, 5, legacy=True)
    assert got == pytest.approx((ref["pass_att"], ref["rush_att"]))


@needs_cache
def test_refreshing_the_data_clears_every_opportunity_model_cache():
    """data.clear_caches() is what the dashboard's refresh button calls; a cached function it forgets keeps serving pre-refresh numbers."""
    M.team_volume("KC", 2024, 10); M.def_target_rel("KC", "WR", 2024, 10); M.opp_line("justin jefferson", "WR", 2024, 10)
    TV.features("KC", 2024, 10); TV._coefs()
    held = [M.team_volume, M.def_target_rel, M._def_target_table, M.opp_line, TV.features, TV._tables, TV._coefs]
    assert all(f.cache_info().currsize > 0 for f in held)
    D.clear_caches()
    assert [f.__name__ for f in held if f.cache_info().currsize] == []
