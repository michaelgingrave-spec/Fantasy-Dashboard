"""The historical salary / scoring layer behind the Millionaire Maker study."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dfs import history as H

PAGE = """<html><body><pre>
Week;Year;GID;Name;Pos;Team;h/a;Oppt;DK points;DK salary
9;2019;1412;Wilson, Russell;QB;sea;h;tam;42.22;7100
9;2019;1452;Garoppolo, Jimmy;QB;sfo;a;ari;31.88;6500
9;2019;7032;Houston;Def;hou;a;jac;19;2800
9;2019;1500;Nobody, Nick;WR;kan;h;gnb;;
9;2019;1501;Bench, Ben;WR;kan;h;gnb;N/A;3000
</pre></body></html>"""


def test_offense_scoring_hand_checked():
    # QB: 300 pass yds (12 + 3 bonus) + 2 TD (8) - 1 INT + 20 rush yds (2) + 1 rush TD (6) = 30
    qb = pd.DataFrame([{"passing_yards": 300, "passing_tds": 2, "passing_interceptions": 1, "rushing_yards": 20, "rushing_tds": 1}])
    assert H.dk_offense_points(qb).iloc[0] == pytest.approx(30.0)
    # RB: 100 rush yds (10 + 3) + TD (6) + 3 rec (3) + 20 rec yds (2) - 1 fumble lost = 23
    rb = pd.DataFrame([{"rushing_yards": 100, "rushing_tds": 1, "receptions": 3, "receiving_yards": 20, "rushing_fumbles_lost": 1}])
    assert H.dk_offense_points(rb).iloc[0] == pytest.approx(23.0)
    # WR: 8 rec (8) + 100 rec yds (10 + 3) + TD (6) + a 2-point conversion (2) = 29; the 100-yard bonus is for 100, not 99
    wr = pd.DataFrame([{"receptions": 8, "receiving_yards": 100, "receiving_tds": 1, "receiving_2pt_conversions": 1}])
    assert H.dk_offense_points(wr).iloc[0] == pytest.approx(29.0)
    assert H.dk_offense_points(pd.DataFrame([{"receiving_yards": 99}])).iloc[0] == pytest.approx(9.9)


def test_dst_points_allowed_tiers():
    pa = pd.Series([0, 1, 6, 7, 13, 14, 20, 21, 27, 28, 34, 35, 50])
    assert H.dst_points_allowed_tier(pa).tolist() == [10, 7, 7, 4, 4, 1, 1, 0, 0, -1, -1, -4, -4]


def test_parse_archive_page():
    df = H.parse_archive_page(PAGE)
    assert len(df) == 4                                   # header line and the row without a salary are dropped; the 'N/A'-points row stays (0 points)
    qb = df[df["name"] == "Russell Wilson"].iloc[0]
    assert (qb["pos"], qb["team"], qb["salary"], qb["pts_recorded"]) == ("QB", "SEA", 7100, pytest.approx(42.22))
    d = df[df["pos"] == "DST"].iloc[0]
    assert d["name"] == "Houston DST" and d["team"] == "HOU" and d["pts_recorded"] == 19
    assert df[df["name"] == "Jimmy Garoppolo"].iloc[0]["team"] == "SF"       # sfo -> SF (DraftKings' code)


def test_a_zero_salary_player_is_not_draftable(monkeypatch):
    # Rotoguru lists 0 for a player DraftKings did not price that week: he counts toward the page being complete but must not become a free roster spot
    monkeypatch.setattr(H, "main_slate_teams", lambda s, w: {"SEA", "TB"})
    rows = []
    for t in ("SEA", "TB"):
        rows += [("QB", t)] + [("RB", t)] * 2 + [("WR", t)] * 3 + [("TE", t)] + [("DST", t)]
    page = pd.DataFrame([{"dk_id": f"rg{i}", "name": f"P{i}", "pos": p, "team": t, "salary": 5000, "pts_recorded": 1.0} for i, (p, t) in enumerate(rows)])
    page.loc[(page["team"] == "SEA") & (page["pos"] == "WR"), "salary"] = [0, 0, 3000]                  # two of SEA's three WRs have no price
    out = H.archive_rows(2019, 9, page)                                                                # complete page: no error
    assert len(out) == len(page) - 2 and (out["salary"] > 0).all()
    assert (H.load_salaries()["salary"] > 0).all()


def test_blank_points_read_as_zero():
    page = "9;2019;1501;Bench, Ben;WR;kan;h;gnb;N/A;3000\n"
    df = H.parse_archive_page(page)
    assert df.iloc[0]["pts_recorded"] == 0.0 and df.iloc[0]["team"] == "KC"


def test_archive_rows_rejects_a_week_with_a_missing_position(monkeypatch):
    monkeypatch.setattr(H, "main_slate_teams", lambda s, w: {"SEA", "TB"})
    rows = []
    for t in ("SEA", "TB"):
        rows += [("QB", t)] + [("RB", t)] * 2 + [("WR", t)] * 3 + [("TE", t)] + [("DST", t)]
    page = pd.DataFrame([{"dk_id": f"rg{i}", "name": f"P{i}", "pos": p, "team": t, "salary": 5000, "pts_recorded": 1.0} for i, (p, t) in enumerate(rows)])
    out = H.archive_rows(2019, 9, page)
    assert len(out) == 16 and set(out.columns) >= set(H.SALARY_COLUMNS) | {"pts_recorded"}
    with pytest.raises(ValueError, match="TB has no DST"):
        H.archive_rows(2019, 9, page[~((page["team"] == "TB") & (page["pos"] == "DST"))])
    monkeypatch.setattr(H, "main_slate_teams", lambda s, w: set())
    with pytest.raises(ValueError, match="no Sunday main-slate games"):
        H.archive_rows(2019, 9, page)


def test_main_slate_is_the_sunday_one_to_four_thirty_games():
    g = H.main_slate_games(2026, 11)
    assert len(g) >= 8
    assert (g["kickoff"].dt.weekday == 6).all()
    minutes = g["kickoff"].dt.hour * 60 + g["kickoff"].dt.minute
    assert minutes.between(13 * 60, 16 * 60 + 30).all()


def test_committed_salary_file_is_consistent():
    s = H.load_dk_salaries()
    assert len(s) > 5000 and s[["season", "week"]].drop_duplicates().shape[0] >= 11
    assert s["salary"].between(2000, 12000).all()
    for (season, week), g in s.groupby(["season", "week"]):
        teams = H.main_slate_teams(int(season), int(week))
        assert set(g["team"]) == teams, (season, week)                      # the pool is exactly the Sunday main slate's teams
        dst = g[g["pos"] == "DST"]
        assert len(dst) == len(teams) and dst["team"].is_unique
        assert (g["pos"] == "QB").sum() >= len(teams)


def test_week_pool_scores_and_context():
    p = H.week_pool(2026, 11)
    assert {"pts", "spread", "total", "team_total", "opp", "game_id", "played"} <= set(p.columns)
    assert p["pts"].ge(-5).all() and p["pts"].max() > 20
    assert p["team_total"].notna().all() and p["game_id"].notna().all()
    # a team's implied total plus its opponent's is the game total
    assert np.allclose(p["team_total"] + p["opp_total"], p["total"])
