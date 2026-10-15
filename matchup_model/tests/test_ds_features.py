"""The Data Suite as-of feature layer: nothing may see the game it predicts, and exports filtered on the outcome must be caught."""
import numpy as np
import pandas as pd
import pytest

from matchup_model.opp import ds_features as F

PLAYERS, DEFENSES, WEEKS = ["a", "b", "c", "d", "e", "f"], ["D1", "D2", "D3", "D4"], range(1, 11)
KEYS = {"season", "week"}


def _poison(df: pd.DataFrame, from_week: int = 0, only_week: int | None = None, factor: float = 100.0) -> pd.DataFrame:
    """Scale every numeric column of the rows at/after `from_week` (or only AT `only_week`), so any leak into an earlier row's features shows."""
    out = df.copy()
    hit = (out.week == only_week) if only_week is not None else (out.week >= from_week)
    for c in out.columns:
        if c not in KEYS and pd.api.types.is_numeric_dtype(out[c]):
            out.loc[hit, c] = out.loc[hit, c] * factor
    return out


def receiving_frame(seed: int = 0, weeks=WEEKS) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i, nk in enumerate(PLAYERS):
        for w in weeks:
            r = {"season": 2024, "week": w, "nk": nk, "name": nk, "team": f"T{i}", "pos": "TE" if i % 3 == 2 else "WR", "opp": DEFENSES[(i + w) % 4]}
            for c in ("rte_part", "tgt_share", "snap_pct", "xfp", "fp", "ay_share", "tprr", "yprr", "adot", "fp_rr", "xfp_rr", "cr", "read1", "design", "ctgt",
                      "exp", "threat", "yptoe", "epa_tgt", "drp", "z_yprr", "z_tprr", "man_yprr", "man_tprr", "zone_yprr", "zone_tprr", "sh_yprr", "sh_tprr",
                      "th_yprr", "th_tprr", "yds", "ay", "rec"):
                r[c] = rng.uniform(0.1, 3.0)
            r.update(rte=float(rng.integers(10, 40)), tgt=float(rng.integers(1, 12)), deep=float(rng.integers(0, 4)), z_rte=30.0, man_rte=8.0, zone_rte=20.0,
                     sh_rte=12.0, th_rte=14.0, wide=rng.uniform(10, 80), slot=rng.uniform(10, 80), inline=rng.uniform(0, 30), back=rng.uniform(0, 5))
            rows.append(r)
    return pd.DataFrame(rows)


def coverage_frame(seed: int = 1, weeks=WEEKS) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame([{"def": d, "season": 2024, "week": w, **{c: rng.uniform(5, 60) for c in F.COV_COLS}} for d in DEFENSES for w in weeks])


def rushing_frames(seed: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows = []
    for i, nk in enumerate(PLAYERS):
        for w in WEEKS:
            r = {"season": 2024, "week": w, "nk": nk, "name": nk, "team": f"T{i}", "opp": DEFENSES[(i + w) % 4]}
            for c in ("att_pct", "snap_pct", "dcs", "xtd", "xfp", "fp", "tgt_pct", "i5", "z_share", "ypc", "succ", "stuff", "ryoe_att", "ybc", "hit", "yaco", "epa",
                      "exp", "z_ypc", "g_ypc"):
                r[c] = rng.uniform(0.1, 6.0)
            r.update(att=float(rng.integers(3, 25)), tgt=float(rng.integers(0, 6)), yds=float(rng.integers(5, 120)), z_att=8.0, z_yds=40.0, g_att=6.0, g_yds=25.0)
            rows.append(r)
    box = pd.DataFrame([{"def": d, "season": 2024, "week": w, **{c: rng.uniform(1, 50) for c in F.BOX_COLS}} for d in DEFENSES for w in WEEKS])
    return pd.DataFrame(rows), box


def query(week: int = 7, nk: str = "a", opp: str = "D1") -> pd.DataFrame:
    return pd.DataFrame([{"season": 2024, "week": week, "nk": nk, "position": "WR", "opp": opp}])


def test_receiving_features_ignore_the_game_being_predicted_and_everything_after_it():
    P, C, q = receiving_frame(), coverage_frame(), query(week=7)
    base = F.build_receiving(q, P, C)
    pd.testing.assert_frame_equal(base, F.build_receiving(q, _poison(P, 7), _poison(C, 7)))
    # control: the same check CAN fail -- changing the game just before it moves the features
    assert not base.equals(F.build_receiving(q, _poison(P, only_week=6), _poison(C, only_week=6)))
    assert base.p_n_prior.iloc[0] == 6 and base.d_n_prior.iloc[0] == 6


def test_running_back_features_ignore_the_game_being_predicted_and_everything_after_it():
    R, B = rushing_frames()
    q = query(week=7)
    base = F.build_rushing(q, R, B)
    pd.testing.assert_frame_equal(base, F.build_rushing(q, _poison(R, 7), _poison(B, 7)))
    assert not base.equals(F.build_rushing(q, _poison(R, only_week=6), _poison(B, only_week=6)))
    assert base.p_n_prior.iloc[0] == 6 and base.d_n_prior.iloc[0] == 6


def test_history_is_capped_at_the_window():
    weeks = range(1, 18)
    late = F.build_receiving(query(week=18), receiving_frame(weeks=weeks), coverage_frame(weeks=weeks))
    assert late.p_n_prior.iloc[0] == F.WIN and late.d_n_prior.iloc[0] == F.DWIN


def test_games_older_than_the_previous_season_never_stand_in_for_recent_ones():
    keys = np.array([202201, 202301, 202401, 202402])
    assert F.window(keys, 202403, 10) == (1, 4)                      # 2022 is two seasons back: out
    assert F.window(keys, 202403, 2) == (2, 4)                       # the size cap still applies
    assert F.window(keys, 202201, 10) == (0, 0)                      # nothing before the first game
    old = receiving_frame().assign(season=2022)                      # a player's whole 2022, then only 2024 weeks 1-4 (no usable 2023: a gap like 2025's)
    P = pd.concat([old, receiving_frame(seed=5, weeks=range(1, 5))], ignore_index=True)
    assert F.build_receiving(query(week=5), P, coverage_frame()).p_n_prior.iloc[0] == 4
    P = pd.concat([old.assign(season=2023), receiving_frame(seed=5, weeks=range(1, 5))], ignore_index=True)
    assert F.build_receiving(query(week=5), P, coverage_frame()).p_n_prior.iloc[0] == F.WIN      # last season's games are fine


def test_a_batch_where_nobody_has_history_gets_empty_features_instead_of_an_error():
    P, C = receiving_frame(), coverage_frame()
    first = F.build_receiving(query(week=1), P, C)
    assert first.p_n_prior.iloc[0] == 0 and first.d_n_prior.iloc[0] == 0
    assert first.filter(regex="^mm_").isna().all(axis=None)
    R, B = rushing_frames()
    rb = F.build_rushing(query(week=1), R, B)
    assert rb.p_n_prior.iloc[0] == 0 and rb.filter(regex="^mm_").isna().all(axis=None)


def test_defense_allowed_rates_are_shrunk_toward_the_league_when_the_sample_is_small():
    def yprr_rel(routes_per_row: float) -> float:
        P = receiving_frame()
        P["rte"] = routes_per_row
        P["yds"] = P.rte * np.where(P.opp == "D1", 2.0, 1.0)      # D1 gives up twice the yards per route of everyone else
        return float(F.build_receiving(query(week=4), P, coverage_frame()).d_all_yprr_rel.iloc[0])
    few, many = yprr_rel(4.0), yprr_rel(400.0)
    assert 0 < few < many and many > 0.3


def test_mismatch_is_positive_when_a_slot_receiver_meets_a_defense_that_is_weak_in_the_slot():
    pf = pd.DataFrame({"p_wide": [0.0, 0.0], "p_slot": [90.0, 90.0], "p_inline": [0.0, 0.0]})
    d = pd.DataFrame({"d_slot_yprr_rel": [0.3, -0.3], "d_wide_yprr_rel": [-0.3, 0.3], "d_inline_yprr_rel": [0.0, 0.0]})
    mm = F.mismatch_features(pf, d)
    assert mm.mm_align_yprr.iloc[0] == pytest.approx(0.3) and mm.mm_align_yprr.iloc[1] == pytest.approx(-0.3)


def test_depth_mismatch_pairs_how_deep_he_is_used_with_how_much_depth_the_defense_gives_up():
    pf = pd.DataFrame({"p_adot": [14.0, 14.0, 6.0, 6.0], "p_deep_rate": [0.30, 0.30, 0.05, 0.05], "p_wide": 50.0, "p_slot": 20.0, "p_inline": 5.0})
    d = pd.DataFrame({"d_all_adot_dev": [1.0, -1.0, 1.0, -1.0]})
    mm = F.mismatch_features(pf, d, week_keys=pd.Series([202407] * 4))
    assert mm.mm_depth_adot.iloc[0] > 0 > mm.mm_depth_adot.iloc[1]            # deep receiver: a deep-allowing defense helps, a deep-suppressing one hurts
    assert mm.mm_depth_adot.iloc[2] < 0 < mm.mm_depth_adot.iloc[3]            # short receiver: the reverse
    assert mm.mm_depth_deep.iloc[0] > 0 > mm.mm_depth_deep.iloc[1]
    assert "mm_depth_adot" not in F.mismatch_features(pf, d).columns           # no week keys, no cross-section to z-score within


def test_ew_weights_recent_games_more_and_a_ratio_is_a_decayed_ratio_of_sums():
    assert F.ew([0.0, 10.0]) > 5.0 > F.ew([10.0, 0.0])
    old = 0.5 ** (1 / F.HL)
    assert F.ew([0.0, 1.0]) == pytest.approx(1 / (old + 1))
    assert F.ratio([1.0, 3.0], [2.0, 2.0]) == pytest.approx((1 * old + 3) / (2 * old + 2))
    assert np.isnan(F.ew([np.nan])) and np.isnan(F.ratio([1.0], [0.0]))


# ── outcome-filtered exports ────────────────────────────────────────────────────────────────────────────
def _played(n_per_season: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    rows = [{"season": s, "week": 5 + i % 10, "nk": f"{s}-{i}", "targets": float(rng.choice([0, 0, 1, 2, 4, 7, 10, 12]))} for s in (2023, 2025) for i in range(n_per_season)]
    return pd.DataFrame(rows)


def _presence(export: pd.DataFrame, played: pd.DataFrame) -> pd.DataFrame:
    return F.presence_by_usage(export[["season", "week", "nk"]], played, "targets", low=(0, 2), high=(9, 999))


def test_an_export_that_drops_quiet_games_is_flagged_and_a_complete_one_is_not():
    played = _played()
    # 2025 keeps only the games with 6+ targets -- the way the real 2025 export was cut
    export = pd.concat([played[played.season == 2023], played[(played.season == 2025) & (played.targets >= 6)]])
    presence = _presence(export, played)
    assert presence.loc[2023, "ratio"] == pytest.approx(1.0) and presence.loc[2025, "ratio"] == 0.0
    assert F.filtered_seasons(presence) == [2025]


def test_a_season_missing_a_week_evenly_is_not_mistaken_for_an_outcome_filter():
    played = _played()
    assert F.filtered_seasons(_presence(played[played.week != 7], played)) == []


def test_a_season_with_too_few_comparison_rows_is_not_judged():
    played = _played(n_per_season=40)
    assert F.filtered_seasons(_presence(played[played.targets >= 6], played)) == []


# ── reading the exports ─────────────────────────────────────────────────────────────────────────────────
def test_read_weekly_drops_league_and_footer_rows_and_skips_files_without_a_name_column(tmp_path, monkeypatch):
    (tmp_path / "x-demo_2024_week.csv").write_text("Banner,,,\nSeason,WEEK,Name,Team\n2024,1,A One,KC\n2024,2,B Two,LV\nLeague Avg,,League Avg,\n,,Exported by FP,\n", encoding="utf-8")
    (tmp_path / "x-demo_2026wk1_week.csv").write_text("Banner,,\nSeason,WEEK,Player\n2026,1,No Name Column\n", encoding="utf-8")
    monkeypatch.setattr(F, "DATA", tmp_path)
    d = F.read_weekly("x-demo")
    assert list(d.Name) == ["A One", "B Two"] and list(d.WEEK) == [1, 2]
    assert F.read_weekly("x-demo", seasons=[2023]).empty
    assert F.read_weekly("not-there").empty


# ── route-type exposure ─────────────────────────────────────────────────────────────────────────────────
def route_frame(seed: int = 7, weeks=WEEKS) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i, nk in enumerate(PLAYERS):
        for w in weeks:
            counts = rng.multinomial(int(rng.integers(20, 45)), rng.dirichlet(np.ones(12)))
            r = {"season": 2024, "week": w, "nk": nk, "name": nk, "team": f"T{i}", "opp": DEFENSES[(i + w) % 4], "rte": float(counts.sum())}
            for x, c in zip(F.ROUTES.values(), counts):
                r[f"rte_{x}"], r[f"yds_{x}"], r[f"tgt_{x}"] = float(c), float(c * rng.uniform(0.2, 2.0)), float(c * rng.uniform(0.02, 0.3))
            rows.append(r)
    return pd.DataFrame(rows)


def test_route_features_ignore_the_game_being_predicted_and_everything_after_it():
    R, q = route_frame(), query(week=7)
    base = F.build_routes(q, R)
    pd.testing.assert_frame_equal(base, F.build_routes(q, _poison(R, 7)))
    assert not base.equals(F.build_routes(q, _poison(R, only_week=6)))
    assert base.rp_n_prior.iloc[0] == 6 and base.rd_n_prior.iloc[0] == 6


def test_route_mix_partitions_his_routes_and_the_depth_buckets_add_up():
    out = F.build_routes(query(week=8), route_frame())
    assert out[[f"rp_mix_{x}" for x in F.ROUTES.values()]].sum(axis=1).iloc[0] == pytest.approx(1.0)
    assert out[[f"rp_mix_{b}" for b in F.BUCKETS]].sum(axis=1).iloc[0] == pytest.approx(1.0)
    assert sorted(m for ms in F.BUCKETS.values() for m in ms) == sorted(F.ROUTES.values())      # every route type sits in exactly one bucket


def _skewed(deep_share: float, deep_yds_mult: float) -> pd.DataFrame:
    """Everyone runs `deep_share` of his routes deep; defense D1 gives up `deep_yds_mult` x the league yards per route on deep routes."""
    R = route_frame()
    deep = F.BUCKETS["deep"]
    for x in F.ROUTES.values():
        share = deep_share / len(deep) if x in deep else (1 - deep_share) / (12 - len(deep))
        R[f"rte_{x}"] = 40.0 * share
        R[f"yds_{x}"] = R[f"rte_{x}"] * np.where((R.opp == "D1") & (x in deep), deep_yds_mult, 1.0)
        R[f"tgt_{x}"] = R[f"rte_{x}"] * 0.2
    R["rte"] = 40.0
    return R


def test_route_mismatch_is_favourable_for_a_deep_receiver_against_a_defense_that_gives_up_the_deep_ball_and_not_against_one_that_stops_it():
    q = query(week=4, opp="D1")
    soft, stingy = F.build_routes(q, _skewed(0.7, 2.0)), F.build_routes(q, _skewed(0.7, 0.5))
    assert soft.rm_deep_yprr.iloc[0] > 0.05 and soft.rm_yprr.iloc[0] > 0.02
    assert stingy.rm_deep_yprr.iloc[0] < -0.02 and stingy.rm_yprr.iloc[0] < 0
    # the same defense matters less to a receiver who works short: exposure, not just the defense, drives the mismatch
    assert abs(F.build_routes(q, _skewed(0.1, 2.0)).rm_deep_yprr.iloc[0]) < abs(soft.rm_deep_yprr.iloc[0])


def test_route_allowed_rates_are_shrunk_toward_the_league_when_a_cell_has_few_routes():
    def go_rel(scale: float) -> float:
        R = _skewed(0.5, 2.0)
        for c in [c for c in R.columns if c.startswith(("rte_", "yds_", "tgt_"))]:
            R[c] = R[c] * scale
        return float(F.build_routes(query(week=4, opp="D1"), R).rd_deep_yprr_rel.iloc[0])
    assert 0 < go_rel(0.1) < go_rel(10.0)


def test_route_players_reads_the_two_row_header_and_skips_the_league_row(tmp_path, monkeypatch):
    import csv
    groups = [""] + ["Player Details"] * 5 + ["Splits"] * 4 + ["Overall"] * 17 + [g for g in F.ROUTES for _ in range(7)]
    names = (["Season", "Rank", "Name", "Team", "POS", "G", "WEEK", "OPP", "RTE %", "TGT %"]
             + ["RTE", "SEP SCORE", "YPRR", "TPRR", "WIN RATE", "SEP MS", "ADOR", "TGT %", "+1 Rate", "+2 Rate", "+3 Rate", "Neg Rate", "TGT", "REC", "YDS", "TD", "AY"]
             + ["RTE", "RTE %", "SEP SCORE", "YPRR", "TPRR", "ADOR", "WIN RATE"] * 12)

    def row(name, team, opp, week, rte_each):
        v = {"Season": 2024, "Name": name, "Team": team, "POS": "WR", "WEEK": week, "OPP": opp}
        out, seen = [], {}
        for g, n in zip(groups, names):
            if g == "Overall" and n == "RTE":
                out.append(12 * rte_each)
            elif g in F.ROUTES and n == "RTE":
                out.append(rte_each)
            elif g in F.ROUTES and n == "YPRR":
                out.append(2.0)
            elif g in F.ROUTES and n == "TPRR":
                out.append(0.25)
            else:
                out.append(v.get(n, 0) if g in ("", "Player Details", "Splits") else 0)
        return out
    with open(tmp_path / "receiving-sepbyroutes_2024wk1-6_week.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(groups)
        w.writerow(names)
        w.writerow([2024, "", "League Avg"] + [""] * (len(names) - 3))
        w.writerow(row("Test Receiver", "KC", "LV", 3, 4))
        w.writerow(row("Other Receiver", "BLT", "CLV", 3, 2))
    monkeypatch.setattr(F, "DATA", tmp_path)
    d = F.route_players()
    assert list(d.nk) == [F.normalize_name("Other Receiver"), F.normalize_name("Test Receiver")] and (d.season == 2024).all() and (d.week == 3).all()
    t = d[d.nk == F.normalize_name("Test Receiver")].iloc[0]
    assert t.rte == 48 and t.rte_go == 4 and t.yds_go == pytest.approx(8.0) and t.tgt_go == pytest.approx(1.0)
    assert sum(t[f"rte_{x}"] for x in F.ROUTES.values()) == t.rte                       # the route types partition his routes
    assert t.team == F.norm_team("KC") and t.opp == F.norm_team("LV")
