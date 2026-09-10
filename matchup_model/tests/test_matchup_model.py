"""Sanity tests for the matchup research module. The coverage-matchup projection edge did
not validate (see backtest_report.md) — these just guard the ingest / shrinkage / defense
predictor so the harness stays usable for a future retry with better inputs.
"""
import numpy as np
import pytest

from matchup_model import (coordinators, defense_model, ingest, player_splits,
                           project, project_stats, scheme)


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


def test_regression_lean_shape_and_bounds():
    r = project.projection_lean("josh allen", "QB", as_of_season=2024, as_of_week=19)
    assert set(["lean", "fp_trail", "xfp_trail", "n_games", "reason"]).issubset(r)
    assert r["n_games"] >= 4 and abs(r["lean"]) < 6          # small nudge, not a rewrite
    # unknown player -> zero lean with a note, never an error
    z = project.projection_lean("no such person xyz", "WR", 2024, 19)
    assert z["lean"] == 0.0 and "game logs" in z["reason"]


def test_coordinator_history_tracks_dc_across_teams():
    h = coordinators.dc_history()
    assert {2022, 2023, 2024, 2025}.issubset(set(h["season"]))
    assert h["team"].nunique() >= 30
    # Vic Fangio: Miami in 2023, Philadelphia 2024-25 — the cross-team case the panel needs
    fangio = coordinators.dc_team_seasons("Vic Fangio")
    assert (2023, "MIA") in fangio and (2024, "PHI") in fangio
    assert coordinators.current_dc("PHI")["dc"] == "Vic Fangio"


def test_player_splits_window_scopes_sample():
    wide = player_splits.build_player_splits(seasons=(2022, 2023, 2024, 2025))
    one = player_splits.build_player_splits(seasons=(2025,))
    a = wide[(wide.name_key == "jamarr chase") & (wide.stat == "tprr")]["n"].sum()
    b = one[(one.name_key == "jamarr chase") & (one.stat == "tprr")]["n"].sum()
    assert 0 < b < a          # a single year is a strict subset of the 4-year sample


def test_scheme_grids_load():
    assert scheme.available()
    # a WR's coverage grid: man/zone/1-high/2-high buckets first, then Cover 0-6
    p = scheme.player_pass_by_coverage("puka nacua")
    assert list(p["look"])[:4] == scheme.BUCKET_NAMES
    assert set(p["look"][4:]).issubset(set(scheme.COVERAGES))
    assert p["yds/rt"].between(0, 12).all() and p["catch%"].between(0, 100).all()
    # opponent defense: plays% + a league rank (1-32, 32 = softest) on every stat
    d = scheme.defense_pass_allowed_by_coverage("BAL")
    assert "plays%" in d.columns and list(d["look"])[:4] == scheme.BUCKET_NAMES
    for m in ("plays%", "yds/tgt", "catch%", "rating", "TD"):
        assert f"{m} rk" in d.columns and d[f"{m} rk"].dropna().between(1, 32).all()
    # rank 32 goes to the team that allows the most — check the direction on a full sweep
    ytg = {t: scheme.defense_pass_allowed_by_coverage(t).set_index("look").loc["Zone", "yds/tgt"]
           for t in scheme._rec_cov_defense()["team"].unique()}
    ytg_rk = {t: scheme.defense_pass_allowed_by_coverage(t).set_index("look").loc["Zone", "yds/tgt rk"]
              for t in ytg}
    softest = max(ytg, key=ytg.get)
    assert ytg_rk[softest] == max(ytg_rk.values())
    # 1st-read% was the broken column — must not be in the defense table
    assert "1st-read%" not in d.columns
    # season blend weight is 0 until 2026 files exist
    assert scheme.blend_weight() == 0.0
    # run concept grid for a bell-cow RB
    rc = scheme.player_run_by_concept("bijan robinson")
    assert set(rc["concept"]).issubset(set(scheme.CONCEPTS)) and rc["YPC"].max() > 2
    # 32-team alignment heat grid
    ag = scheme.defense_alignment_grid()
    assert len(ag) == 32 and "Wide yds/rt" in ag.columns
    # defense run-concept ranks (1..N, N = league extreme)
    dr = scheme.team_run_by_concept("BAL", "defense")
    assert "YPC rk" in dr.columns and dr["YPC rk"].dropna().between(1, 32).all()
    # highlight table: rows tie a real player to a look the opponent is extreme in
    hl = scheme.matchup_highlights("BAL", ["Ja'Marr Chase", "Tee Higgins"], ["Chase Brown"])
    assert not hl.empty and set(hl.columns) == {"kind", "look", "player", "why", "mark"}
    assert hl["kind"].isin(["pass", "run"]).all()
    assert hl["why"].str.startswith("BAL:").all()
    # personnel splits load and can produce a "NN personnel" edge somewhere in the league
    pp = scheme.player_pass_by_personnel("jamarr chase")
    assert not pp.empty and set(pp["personnel"]).issubset(set(scheme.PERSONNEL))
    assert scheme._defense_personnel_ranks("rush") and scheme._defense_personnel_ranks("rec")


def test_projected_stat_line():
    # a high-volume WR with 2022-25 logs -> a plausible line and DK points
    r = project_stats.projected_line("jamarr chase", "WR", as_of_season=2025, as_of_week=10)
    assert r["fp"] is not None and 5 < r["fp"] < 45
    assert 4 < r["line"]["tgt"] < 16 and r["line"]["rec"] <= r["line"]["tgt"]
    # QB path exercises the rushing-fp add-on
    q = project_stats.projected_line("josh allen", "QB", as_of_season=2025, as_of_week=10)
    assert q["fp"] is not None and 8 < q["fp"] < 45
    # unknown player -> no line, a reason, never an error
    z = project_stats.projected_line("no such person xyz", "RB")
    assert z["fp"] is None and "recent games" in z["reason"]
