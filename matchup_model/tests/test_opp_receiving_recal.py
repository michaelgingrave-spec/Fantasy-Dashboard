"""WR/TE receiving recalibration: the volume scale, the defense-targets nudge, and the calibration helpers."""
import numpy as np
import pandas as pd
import pytest

from matchup_model.opp import data as D
from matchup_model.opp import model as M
from matchup_model.opp import receiving_calibration as RC

needs_cache = pytest.mark.skipif(
    not ((D.CACHE / "stats_player_week_2024.parquet").exists() and (D.CACHE / "games.csv").exists()),
    reason="nflverse cache not downloaded")


@pytest.fixture(autouse=True)
def _fresh_caches():
    """opp_line / def_target_rel are cached: a test that changes a constant must not see (or leave) stale values."""
    M.opp_line.cache_clear(); M.def_target_rel.cache_clear()
    yield
    M.opp_line.cache_clear(); M.def_target_rel.cache_clear()


def _allowed(rows):
    """A defense-targets table like _def_target_table's: (defense, season, week, WR targets allowed)."""
    t = pd.DataFrame(rows, columns=["opp", "season", "week", "WR"])
    t["TE"], t["RB"] = 5.0, 4.0
    t["total"] = t.WR + t.TE + t.RB
    return t.sort_values(["opp", "season", "week"]).reset_index(drop=True)


def _season(opp, targets, season=2026):
    return [(opp, season, wk, t) for wk, t in enumerate(targets, start=1)]


def test_shipped_constants_are_the_recalibrated_ones():
    assert M.MODEL_VERSION >= 2
    assert (M.K_SHARE_REC, M.K_YPT) == (1.0, 60.0)
    for pos in ("WR", "TE"):
        assert set(M.REC_SCALE[pos]) == {"tgt", "rec", "rec_yds", "td"}
        assert all(0.7 < M.REC_SCALE[pos][k] < 1.0 for k in ("tgt", "rec", "rec_yds"))   # the old model over-projected, never under
        assert 0.8 < M.REC_SCALE[pos]["td"] < 1.2                                       # TDs are close to calibrated already
    assert 0.05 < M.DEF_TGT_B < 0.4


def test_defense_rate_is_relative_to_the_league_and_needs_three_games(monkeypatch):
    t = _allowed(_season("AAA", [12] * 5) + _season("BBB", [8] * 5) + _season("CCC", [8] * 5))
    monkeypatch.setattr(M, "_def_target_table", lambda: t)
    assert M.def_target_rel("AAA", "WR", 2026, 3) == 1.0                    # only 2 games: not enough to read
    assert M.def_target_rel("AAA", "WR", 2026, 6) == pytest.approx(12 / ((12 * 5 + 8 * 10) / 15))   # 12 vs a 9.33 league
    assert M.def_target_rel("NOPE", "WR", 2026, 6) == 1.0                   # a defense with no games


def test_defense_rate_uses_only_games_before_the_week(monkeypatch):
    base = _season("AAA", [12] * 5) + _season("BBB", [8] * 5) + _season("CCC", [8] * 5)
    asof = _allowed(base)
    later = _allowed(base + [("AAA", 2026, 6, 99)])                         # week 6 itself, and a freak one
    monkeypatch.setattr(M, "_def_target_table", lambda: asof)
    a = M.def_target_rel("AAA", "WR", 2026, 6)
    M.def_target_rel.cache_clear()
    monkeypatch.setattr(M, "_def_target_table", lambda: later)
    assert M.def_target_rel("AAA", "WR", 2026, 6) == a                      # the game being projected is never read


def test_defense_rate_trailing_window_and_clip(monkeypatch):
    t = _allowed(_season("AAA", [4, 4, 4, 10, 10]) + _season("BBB", [8] * 5) + _season("CCC", [8] * 5))
    monkeypatch.setattr(M, "_def_target_table", lambda: t)
    league = (4 * 3 + 10 * 2 + 8 * 10) / 15
    assert M.def_target_rel("AAA", "WR", 2026, 6) == pytest.approx(((4 * 3 + 10 * 2) / 5) / league)   # default: last 8 games
    M.def_target_rel.cache_clear()
    monkeypatch.setattr(M, "DEF_TGT_GAMES", 2)
    assert M.def_target_rel("AAA", "WR", 2026, 6) == pytest.approx(10 / league)                       # last 2 only
    M.def_target_rel.cache_clear()
    freak = _allowed(_season("AAA", [100] * 5) + _season("BBB", [8] * 5) + _season("CCC", [8] * 5))
    monkeypatch.setattr(M, "_def_target_table", lambda: freak)
    assert M.def_target_rel("AAA", "WR", 2026, 6) == M.DEF_TGT_CLIP[1]
    M.def_target_rel.cache_clear()
    stingy = _allowed(_season("AAA", [0.1] * 5) + _season("BBB", [8] * 5) + _season("CCC", [8] * 5))
    monkeypatch.setattr(M, "_def_target_table", lambda: stingy)
    assert M.def_target_rel("AAA", "WR", 2026, 6) == M.DEF_TGT_CLIP[0]


def test_defense_adjustment_is_the_weighted_rate_and_off_without_a_weight_or_a_game(monkeypatch):
    games = pd.DataFrame({"season": [2026], "week": [6], "team": ["XXX"], "opp": ["AAA"]})
    monkeypatch.setattr(M, "_games_idx", lambda: games)
    monkeypatch.setattr(M, "def_target_rel", lambda opp, pos, season, week: 1.25 if opp == "AAA" else 1.0)
    monkeypatch.setattr(M, "DEF_TGT_B", 0.2)
    assert M._def_target_adj("WR", "XXX", 2026, 6) == pytest.approx(1.05)   # 1 + 0.2 x (1.25 - 1)
    assert M._def_target_adj("WR", "XXX", 2026, 7) == 1.0                    # not on the schedule that week (bye)
    assert M._def_target_adj("WR", "YYY", 2026, 6) == 1.0
    monkeypatch.setattr(M, "_games_idx", lambda: games.assign(opp=[np.nan]))
    assert M._def_target_adj("WR", "XXX", 2026, 6) == 1.0                    # opponent unknown
    monkeypatch.setattr(M, "_games_idx", lambda: games)
    monkeypatch.setattr(M, "DEF_TGT_B", 0.0)
    assert M._def_target_adj("WR", "XXX", 2026, 6) == 1.0                    # weight off


def _top(pos, season=2025):
    pw = D.player_weeks()
    d = pw[(pw.season == season) & (pw.position == pos)]
    return d.groupby("name_key").targets.sum().idxmax()


@needs_cache
def test_wr_volume_scale_multiplies_targets_receptions_and_yards(monkeypatch):
    nk = _top("WR")
    monkeypatch.setattr(M, "DEF_TGT_B", 0.0)
    unit = {"tgt": 1.0, "rec": 1.0, "rec_yds": 1.0}
    monkeypatch.setattr(M, "REC_SCALE", {"WR": unit, "TE": unit})
    raw = M.opp_line(nk, "WR", 2025, 12)["line"]
    M.opp_line.cache_clear()
    monkeypatch.setattr(M, "REC_SCALE", {"WR": {"tgt": 0.8, "rec": 0.7, "rec_yds": 0.6}, "TE": unit})
    scaled = M.opp_line(nk, "WR", 2025, 12)["line"]
    assert scaled["tgt"] == pytest.approx(0.8 * raw["tgt"], rel=2e-3)
    assert scaled["rec"] == pytest.approx(0.7 * raw["rec"], rel=2e-3)
    assert scaled["rec_yds"] == pytest.approx(0.6 * raw["rec_yds"], rel=2e-3)


@needs_cache
def test_td_scale_multiplies_expected_tds_but_the_volume_scale_does_not(monkeypatch):
    nk = _top("WR")
    monkeypatch.setattr(M, "DEF_TGT_B", 0.0)
    base = {"tgt": 1.0, "rec": 1.0, "rec_yds": 1.0, "td": 1.0}
    unit = {"WR": base, "TE": base}
    monkeypatch.setattr(M, "REC_SCALE", unit)
    raw = M.opp_line(nk, "WR", 2025, 12)["line"]["rec_td"]
    M.opp_line.cache_clear()
    monkeypatch.setattr(M, "REC_SCALE", {"WR": {**base, "tgt": 0.5}, "TE": base})          # volume scale alone: TDs unchanged
    assert M.opp_line(nk, "WR", 2025, 12)["line"]["rec_td"] == pytest.approx(raw, rel=2e-3)
    M.opp_line.cache_clear()
    monkeypatch.setattr(M, "REC_SCALE", {"WR": {**base, "td": 1.25}, "TE": base})          # TD multiplier: scales it
    assert M.opp_line(nk, "WR", 2025, 12)["line"]["rec_td"] == pytest.approx(1.25 * raw, rel=2e-3)
    M.opp_line.cache_clear()
    monkeypatch.setattr(M, "REC_SCALE", {"WR": {"tgt": 1.0, "rec": 1.0, "rec_yds": 1.0}, "TE": base})   # no "td" key: defaults to 1
    assert M.opp_line(nk, "WR", 2025, 12)["line"]["rec_td"] == pytest.approx(raw, rel=2e-3)


def test_td_scales_are_scored_over_expected_by_position():
    d = pd.DataFrame({"position": ["WR", "WR", "TE", "TE"], "receiving_tds": [1.0, 0.0, 1.0, 1.0],
                      "p_rec_td": [0.4, 0.4, 0.5, np.nan]})
    got = RC._td_scales(d)
    assert got["WR"] == pytest.approx(1.0 / 0.8) and got["TE"] == pytest.approx(1.0 / 0.5)   # a missing projection drops out


@needs_cache
def test_te_uses_its_own_scale(monkeypatch):
    nk = _top("TE")
    monkeypatch.setattr(M, "DEF_TGT_B", 0.0)
    unit = {"tgt": 1.0, "rec": 1.0, "rec_yds": 1.0}
    monkeypatch.setattr(M, "REC_SCALE", {"WR": unit, "TE": unit})
    raw = M.opp_line(nk, "TE", 2025, 12)["line"]
    M.opp_line.cache_clear()
    monkeypatch.setattr(M, "REC_SCALE", {"WR": {"tgt": 0.1, "rec": 0.1, "rec_yds": 0.1},
                                        "TE": {"tgt": 0.9, "rec": 0.9, "rec_yds": 0.9}})
    scaled = M.opp_line(nk, "TE", 2025, 12)["line"]
    assert scaled["rec_yds"] == pytest.approx(0.9 * raw["rec_yds"], rel=2e-3)


@needs_cache
def test_defense_nudge_moves_targets_receptions_and_yards_together(monkeypatch):
    nk = _top("WR")
    monkeypatch.setattr(M, "DEF_TGT_B", 0.0)
    flat = M.opp_line(nk, "WR", 2025, 12)
    M.opp_line.cache_clear()
    monkeypatch.setattr(M, "DEF_TGT_B", 0.25)
    monkeypatch.setattr(M, "def_target_rel", lambda *a: 1.2)                # a defense that gives up 20% more
    soft = M.opp_line(nk, "WR", 2025, 12)
    for stat in ("tgt", "rec", "rec_yds"):
        assert soft["line"][stat] == pytest.approx(1.05 * flat["line"][stat], rel=2e-3)     # 1 + 0.25 x 0.2
    assert "defense targets x1.05" in soft["method"] and "defense targets" not in flat["method"]


@needs_cache
def test_rb_projection_ignores_the_receiver_recalibration(monkeypatch):
    nk = _top("RB")
    before = M.opp_line(nk, "RB", 2025, 12)["line"]
    M.opp_line.cache_clear()
    half = {"tgt": 0.5, "rec": 0.5, "rec_yds": 0.5}
    monkeypatch.setattr(M, "REC_SCALE", {"WR": half, "TE": half})
    monkeypatch.setattr(M, "DEF_TGT_B", 0.5)
    monkeypatch.setattr(M, "K_SHARE_REC", 9.0)
    monkeypatch.setattr(M, "K_YPT", 5.0)
    assert M.opp_line(nk, "RB", 2025, 12)["line"] == before


@needs_cache
def test_projected_wr_te_targets_are_no_longer_a_quarter_too_high():
    """The old model handed a team's WR+TE ~28% more targets than they get. With the shipped constants, among the
    players it expects to be involved, 2025 weeks 9-12 projected targets land within a few percent of the actual
    (0.95 -- a touch under 1 because players with no stat line are not rows here; the old constants gave 1.15-1.18)."""
    pw = D.player_weeks()
    d = pw[(pw.season == 2025) & pw.week.between(9, 12) & pw.position.isin(["WR", "TE"]) & (pw["season_type"] == "REG")]
    proj = act = 0.0
    for r in d.drop_duplicates(["season", "week", "name_key"]).itertuples():
        o = M.opp_line(r.name_key, r.position, r.season, r.week)
        if o.get("dk_fp") is None or o["line"]["tgt"] < 2.6:                  # ~3 targets before the scale
            continue
        proj += o["line"]["tgt"]
        act += r.targets
    assert act > 500
    assert 0.88 < proj / act < 1.08, proj / act


# ── the calibration helpers ───────────────────────────────────────────────────────────────────────────────
def test_scales_are_sum_actual_over_sum_projected_by_position():
    d = pd.DataFrame({"position": ["WR", "WR", "TE"], "p_tgt": [10., 10., 4.], "p_rec": [6., 6., 3.],
                      "p_rec_yds": [80., 80., 40.], "targets": [8., 8., 2.], "receptions": [5., 4., 1.5],
                      "receiving_yards": [60., 60., 30.]})
    sc = RC._scales(d)
    assert sc["WR"]["tgt"] == pytest.approx(0.8) and sc["WR"]["rec"] == pytest.approx(0.75)
    assert sc["TE"]["rec_yds"] == pytest.approx(0.75)


def test_blend_weight_fit_recovers_a_planted_weight_and_clips_to_0_1():
    naive = np.array([5.0, 8.0, 3.0, 10.0])
    opp = np.array([9.0, 4.0, 7.0, 6.0])
    tr = pd.DataFrame({"naive": naive, "opp": opp, "dk_fp": 0.3 * opp + 0.7 * naive})
    assert RC._w_fit(tr, "opp") == pytest.approx(0.3)
    assert RC._w_fit(tr.assign(dk_fp=2.0 * opp - naive), "opp") == 1.0       # wants more than all-opp: clipped
    assert RC._w_fit(tr.assign(dk_fp=-opp + 2 * naive), "opp") == 0.0        # wants the opposite sign: clipped


def test_before_constants_are_the_pre_recalibration_model():
    assert (RC.BEFORE["K_SHARE_REC"], RC.BEFORE["K_YPT"], RC.BEFORE["DEF_TGT_B"]) == (3.0, 28.0, 0.0)
    assert all(v == 1.0 for pos in ("WR", "TE") for v in RC.BEFORE["REC_SCALE"][pos].values())


def test_defense_slope_recovers_a_planted_weight(monkeypatch):
    rel = {"A": 1.25, "B": 0.80, "C": 1.0}
    monkeypatch.setattr(M, "def_target_rel", lambda opp, pos, season, week: rel[opp])
    opp = ["A", "B", "C", "A", "B"] * 40
    d = pd.DataFrame({"opp": opp, "position": "WR", "season": 2024, "week": 6, "p_tgt": 6.0})
    sc = {"WR": {"tgt": 0.8, "rec": 0.8, "rec_yds": 0.8}}
    d["targets"] = [0.8 * 6.0 * (1 + 0.3 * (rel[o] - 1)) for o in opp]
    assert RC._def_slope(d, sc) == pytest.approx(0.3)


def test_apply_scales_every_stat_and_nudges_by_the_defense(monkeypatch):
    monkeypatch.setattr(M, "def_target_rel", lambda opp, pos, season, week: 1.2)
    d = pd.DataFrame({"opp": ["A"], "position": ["TE"], "season": 2024, "week": 6,
                      "p_tgt": [5.0], "p_rec": [3.0], "p_rec_yds": [40.0]})
    sc = {"TE": {"tgt": 0.8, "rec": 0.9, "rec_yds": 0.7}}
    out = RC._apply(d, sc, 0.5)                                              # defense factor 1 + 0.5 x 0.2 = 1.1
    assert out["tgt"][0] == pytest.approx(5.0 * 0.8 * 1.1)
    assert out["rec"][0] == pytest.approx(3.0 * 0.9 * 1.1)
    assert out["rec_yds"][0] == pytest.approx(40.0 * 0.7 * 1.1)
    assert RC._apply(d, sc, 0.0)["tgt"][0] == pytest.approx(5.0 * 0.8)


def test_hist_targets_is_the_trailing_average_before_the_week(monkeypatch):
    pw = pd.DataFrame({"name_key": "a b", "position": "WR", "season": 2025, "week": range(1, 11),
                       "targets": [4, 6, 8, 10, 2, 2, 2, 2, 2, 2]})
    monkeypatch.setattr(RC.D, "player_weeks", lambda: pw)
    u = pd.DataFrame({"nk": ["a b", "a b", "a b", "zz"], "position": "WR", "season": 2025, "week": [3, 4, 10, 4]})
    out = RC._hist_targets(u)
    assert np.isnan(out[0])                                                  # two prior games: not enough to read
    assert out[1] == pytest.approx(6.0)                                      # (4 + 6 + 8) / 3
    assert out[2] == pytest.approx(np.mean([6, 8, 10, 2, 2, 2, 2, 2]))       # the last 8 of the 9 before week 10
    assert np.isnan(out[3])                                                  # a player with no history


def test_score_learns_each_seasons_scale_on_the_other_two():
    n = 4
    rows = []
    for season, actual in ((2023, 5.0), (2024, 5.0), (2025, 7.0)):
        rows.append(pd.DataFrame({"season": season, "week": 6, "opp": np.nan, "position": "WR",
                                  "p_tgt": 10.0, "p_rec": 6.0, "p_rec_yds": 80.0, "targets": actual,
                                  "receptions": 3.0, "receiving_yards": 40.0}, index=range(n)))
    got = RC._score(pd.concat(rows, ignore_index=True))
    # 2023 and 2024: scale (5 + 7) / 2 / 10 = 0.6 -> projects 6 vs 5 (miss 1); 2025: scale 0.5 -> 5 vs 7 (miss 2)
    assert got["tgt"] == pytest.approx(np.sqrt((1 + 1 + 4) / 3))
    assert got["rec"] == pytest.approx(0.0)                                  # a perfectly proportional stat has no error


@needs_cache
def test_universe_adds_players_who_played_without_a_stat_line():
    u = RC.universe_rows()
    assert not u.duplicated(RC.KEY).any()
    z = u[u.no_row]
    assert len(z) > 500                                                      # ~1,700 in 2023-25 weeks 5-18
    assert (z[["targets", "receptions", "receiving_yards"]] == 0).all().all()
    assert u[~u.no_row].targets.sum() > 0 and set(u.position) == {"WR", "TE"}


@needs_cache
def test_project_leaves_the_model_constants_as_it_found_them():
    before = (M.K_SHARE_REC, M.K_YPT, M.DEF_TGT_B, {p: dict(s) for p, s in M.REC_SCALE.items()})
    got = RC.project(RC.OLD, RC.universe_rows().query("no_row == False").head(5))
    assert (M.K_SHARE_REC, M.K_YPT, M.DEF_TGT_B, M.REC_SCALE) == before
    assert {"p_tgt", "p_rec", "p_rec_yds"} <= set(got.columns)
