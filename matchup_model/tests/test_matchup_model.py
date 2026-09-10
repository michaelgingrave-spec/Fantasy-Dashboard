"""Sanity tests for the matchup research module. The coverage-matchup projection edge did
not validate (see backtest_report.md) — these just guard the ingest / shrinkage / defense
predictor so the harness stays usable for a future retry with better inputs.
"""
import numpy as np
import pytest

from matchup_model import defense_model, ingest, player_splits


def test_ingest_loads_all_tables():
    inv = ingest.available()
    assert inv["receiving_splits_rows"] > 40_000       # 5 splits x ~12k player-weeks
    assert inv["coverage_matrix_rows"] > 1_000
    assert inv["passing_weekly_rows"] > 1_000


def test_receiving_splits_shape_and_sanity():
    rs = ingest.receiving_splits()
    assert set(rs["split"].unique()) == {"overall", "man", "zone", "single_high", "two_high"}
    la_yprr = (rs.assign(w=rs["rte"]).query("split=='overall'")
               .pipe(lambda d: np.average(d["yprr"].fillna(0), weights=d["w"].fillna(0))))
    assert 1.0 < la_yprr < 2.2          # league YPRR is ~1.5


def test_shrinkage_pulls_thin_samples_toward_baseline():
    df = player_splits.build_player_splits(through_season=2023)
    assert not df.empty
    # a split with tiny sample must land close to the player's shrunk overall
    thin = df[(df["n"] < 20) & (df["stat"] == "tprr")]
    assert (thin["delta"].abs() < 0.05).mean() > 0.8


def test_defense_predictor_recovers_known_tendencies():
    # DET 2024 was the most man-heavy defense in the league
    det = defense_model.predict("DET", 2024, 12)
    tb = defense_model.predict("TB", 2024, 12)
    assert det["man"] > 40 and det["man"] > tb["man"] + 15
    assert det["_source"] == "trend" and det["_n_games"] >= 5


def test_scheme_table_has_lean_column():
    t = defense_model.scheme_table("BAL", 2024, 99)
    assert {"look", "predicted_%", "league_%", "lean"}.issubset(t.columns)
    assert len(t) >= 8
