"""Bet log: add, payout math, edge-bucket analysis. Grading (nflverse) is tested lightly."""
import numpy as np
import pytest

from dfs import bets as B


@pytest.fixture
def tmp_log(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "LOG_PATH", tmp_path / "bet_log.csv")
    return B.LOG_PATH


def test_add_and_edge_fields(tmp_log):
    bid = B.add_bet(season=2025, week=1, event="SF @ LAR", player="Puka Nacua",
                    market="rec yds", side="OVER", line=89.5, odds=-110, our_proj=96.1)
    df = B.load()
    assert len(df) == 1 and df.iloc[0]["bet_id"] == bid
    r = df.iloc[0]
    assert r["edge_toward"] == pytest.approx(6.6, abs=0.01)          # 96.1 - 89.5
    assert r["edge_pct_toward"] == pytest.approx(7.4, abs=0.1)
    # UNDER flips the edge direction
    B.add_bet(season=2025, week=1, event="x", player="CMC", market="rec yds",
              side="UNDER", line=39.5, odds=-115, our_proj=29.0)
    u = B.load().iloc[1]
    assert u["edge_toward"] == pytest.approx(10.5, abs=0.01)         # 39.5 - 29.0


def test_payout_math():
    assert B._payout(-110, "win", 1.0) == pytest.approx(0.909, abs=0.001)
    assert B._payout(+150, "win", 2.0) == pytest.approx(3.0, abs=0.001)
    assert B._payout(-110, "loss", 1.0) == -1.0
    assert B._payout(-110, "push", 1.0) == 0.0


def test_delete(tmp_log):
    a = B.add_bet(season=2025, week=1, event="x", player="A", market="rec yds",
                  side="OVER", line=50, odds=-110, our_proj=60)
    B.add_bet(season=2025, week=1, event="x", player="B", market="rec yds",
              side="OVER", line=50, odds=-110, our_proj=60)
    B.delete_bet(a)
    df = B.load()
    assert len(df) == 1 and df.iloc[0]["player"] == "B"


def test_edge_bucket_analysis(tmp_log):
    # rec yds sigma at a 100 line ~= 49.5, so edge 6 -> z~0.12 ("—"), edge 30 -> z~0.61 ("strong")
    B.add_bet(season=2025, week=1, event="x", player="A", market="rec yds",
              side="OVER", line=100, odds=-110, our_proj=106)
    B.add_bet(season=2025, week=1, event="x", player="B", market="rec yds",
              side="OVER", line=100, odds=-110, our_proj=130)
    df = B.load()
    assert df.loc[0, "z"] < 0.15 and df.loc[1, "z"] > 0.5
    assert df.loc[0, "conf"] == "—" and df.loc[1, "conf"] == "strong"
    df.loc[0, ["result", "payout", "stake"]] = ["win", 0.909, 1.0]
    df.loc[1, ["result", "payout", "stake"]] = ["loss", -1.0, 1.0]
    B._save(df)
    eb = B.by_edge_bucket().set_index("conf")
    assert eb.loc["—", "win%"] == 100.0
    assert eb.loc["strong", "ROI%"] == -100.0


def test_load_missing_is_empty(tmp_log):
    assert B.load().empty
    assert B.overall()["bets"] == 0


def test_line_history_snapshot_and_buckets(tmp_path, monkeypatch):
    import pandas as pd
    from dfs import props

    monkeypatch.setattr(props, "LINE_HISTORY_PATH", tmp_path / "lh.csv")
    edf = pd.DataFrame({
        "player": [f"P{i}" for i in range(12)],
        "market": ["rec yds"] * 12,
        "line": [50.0] * 12,
        "our proj": [56.0] * 6 + [70.0] * 6,
        "edge": [6.0] * 6 + [20.0] * 6,
        "z": [0.18] * 6 + [0.61] * 6,                  # small edge x6, big edge x6
        "conf": ["lean"] * 6 + ["strong"] * 6,
        "p(hit)%": [55] * 12, "p edge": [3.0] * 12, "book %": [52] * 12,
        "edge %": [12.0] * 6 + [40.0] * 6,
        "lean": ["OVER"] * 12, "best": [""] * 12, "role_ratio": [1.0] * 12,
    })
    assert props.snapshot_lines(edf, 2025, 3, "X @ Y") == 12
    lh = B.load_line_history()
    assert len(lh) == 12 and "result" in lh.columns and "conf" in lh.columns
    lh.loc[lh["conf"] == "lean", "result"] = "loss"   # small edges lose
    lh.loc[lh["conf"] == "strong", "result"] = "win"  # big edges win
    lh.to_csv(props.LINE_HISTORY_PATH, index=False)
    bk = B.line_history_buckets().set_index("conf")
    lean = "lean"
    strong = "strong"
    assert bk.loc[strong, "win%"] == 100.0
    assert bk.loc[lean, "win%"] == 0.0


def test_snapshot_stamps_the_model_version_and_leaves_older_rows_unstamped(tmp_path, monkeypatch):
    import pandas as pd
    from dfs import props

    monkeypatch.setattr(props, "LINE_HISTORY_PATH", tmp_path / "lh_ver.csv")
    monkeypatch.setattr(props, "_model_version", lambda: 2)
    # a history written before the stamp existed: no model_ver column at all
    pd.DataFrame({"season": [2026], "week": [1], "event": ["old"], "player": ["Old Guy"], "market": ["rec yds"],
                  "line": [50.0], "our proj": [55.0], "result": ["win"], "actual": [60.0]}
                 ).to_csv(props.LINE_HISTORY_PATH, index=False)
    edf = pd.DataFrame({"player": ["New Guy"], "market": ["rec yds"], "line": [40.0], "our proj": [45.0],
                        "lean": ["OVER"]})
    props.snapshot_lines(edf, 2026, 6, "X @ Y")
    lh = pd.read_csv(props.LINE_HISTORY_PATH).set_index("player")
    assert lh.loc["New Guy", "model_ver"] == 2
    assert pd.isna(lh.loc["Old Guy", "model_ver"])           # logged before versions existed: that is version 1


def test_version_note_names_the_models_only_when_the_history_spans_more_than_one():
    import pandas as pd

    base = {"season": 2026, "line": 50.0, "market": "rec yds", "our proj": 60.0, "conf": "solid", "role_ratio": 1.0}
    old = [{**base, "week": w, "result": "win"} for w in (1, 2, 3)]            # logged before the stamp: no model_ver
    assert B.version_note(pd.DataFrame(old)) == ""
    new = [{**base, "week": w, "result": "loss", "model_ver": 2} for w in (6, 7)]
    note = B.version_note(pd.DataFrame(old + new))
    assert "v1: 3 lines, weeks 1–3" in note and "v2: 2 lines, weeks 6–7" in note
    one_week = B.version_note(pd.DataFrame(old + new[:1]))
    assert "v2: 1 lines, week 6" in one_week
    assert B.version_note(pd.DataFrame(new + [{**base, "week": 8, "result": "", "model_ver": 3}])) == ""   # ungraded: not counted
    assert B.version_note(pd.DataFrame()) == ""


def test_model_version_is_the_opportunity_models_unless_the_blend_is_unavailable(monkeypatch):
    from dfs import props
    from matchup_model.opp import blend, model

    monkeypatch.setattr(blend, "available", lambda: True)
    assert props._model_version() == model.MODEL_VERSION
    monkeypatch.setattr(blend, "available", lambda: False)
    assert props._model_version() == 0                       # the project_stats fallback made these projections


def test_line_history_buckets_can_be_limited_to_the_recommended_rows():
    import pandas as pd

    base = {"season": 2026, "week": 1, "line": 50.0, "market": "rec yds", "our proj": 60.0,
            "conf": "solid"}
    # four stable-role rows (3-1) and four where FantasyPoints disagrees with our role read (0-4)
    rows = ([{**base, "role_ratio": 1.0, "result": r} for r in ("win", "win", "win", "loss")]
            + [{**base, "role_ratio": 2.0, "result": "loss"} for _ in range(4)])
    df = pd.DataFrame(rows)
    every = B.line_history_buckets(df).set_index("conf")
    rec = B.line_history_buckets(df, recommended_only=True).set_index("conf")
    assert every.loc["solid", "n"] == 8 and every.loc["solid", "win%"] == 37.5
    assert rec.loc["solid", "n"] == 4 and rec.loc["solid", "win%"] == 75.0
    assert rec.loc["solid", "of all"] == 8 and "of all" not in every.columns   # shows how many were screened out
    assert B.line_history_buckets(df.assign(role_ratio=2.0), recommended_only=True).empty


def test_line_history_by_market(tmp_path, monkeypatch):
    import pandas as pd
    from dfs import props

    monkeypatch.setattr(props, "LINE_HISTORY_PATH", tmp_path / "lh2.csv")
    edf = pd.DataFrame({
        "player": ["A", "B", "C"], "market": ["rec yds", "rec yds", "pass TD"],
        "line": [50.0] * 3, "our proj": [56.0] * 3, "edge": [6.0] * 3,
        "z": [0.3] * 3, "conf": ["solid"] * 3,
        "p(hit)%": [58] * 3, "p edge": [5.0] * 3, "book %": [50] * 3,
        "edge %": [12.0] * 3, "lean": ["OVER"] * 3, "best": [""] * 3, "role_ratio": [1.0] * 3,
    })
    props.snapshot_lines(edf, 2026, 1, "X @ Y")
    lh = B.load_line_history()
    lh.loc[lh.player.isin(["A", "B"]), "result"] = ["win", "loss"]
    lh.loc[lh.player == "C", "result"] = "win"
    lh.to_csv(props.LINE_HISTORY_PATH, index=False)

    bm = B.line_history_by_market().set_index("market")
    assert bm.loc["rec yds", "n"] == 2 and bm.loc["rec yds", "win%"] == 50.0
    assert bm.loc["pass TD", "win%"] == 100.0
    # sorted best-ROI first
    assert list(B.line_history_by_market()["market"])[0] == "pass TD"


def test_dnp_keys_needs_a_positive_zero_for_a_skill_player_on_that_team(monkeypatch):
    # week 5 2026: the Browns' LB Justin Jefferson (always 0 OFFENSE snaps) once marked the
    # Vikings' WR of the same name inactive even though he played 63 snaps.
    import pandas as pd

    snaps = pd.DataFrame([
        dict(season=2026, week=5, name_key="justin jefferson", team="CLE", position="LB",
             offense_snaps=0.0),
        dict(season=2026, week=5, name_key="justin jefferson", team="MIN", position="WR",
             offense_snaps=63.0),
        dict(season=2026, week=5, name_key="really inactive", team="KC", position="WR",
             offense_snaps=0.0),
        dict(season=2026, week=5, name_key="no snap value", team="KC", position="TE",
             offense_snaps=np.nan),
        dict(season=2026, week=4, name_key="other week", team="KC", position="RB",
             offense_snaps=0.0),
    ])
    monkeypatch.setattr("matchup_model.opp.data.snaps", lambda: snaps)
    assert B._dnp_keys(2026, 5) == {("really inactive", "KC")}


def _quiet_week(monkeypatch):
    """KC's box score is out (one player has a stat row); BUF's is not. nflverse lists a player
    only if he recorded a stat, so the quiet ones appear in the snap counts alone."""
    import pandas as pd

    pw = pd.DataFrame([dict(player_display_name="Stat Guy", nk="stat guy", season=2026, week=5,
                            team="KC", receiving_yards=48.0, receptions=4.0, rushing_yards=0.0,
                            passing_yards=0.0, passing_tds=0.0, carries=0.0, attempts=0.0)])
    snaps = pd.DataFrame([
        dict(season=2026, week=5, name_key="stat guy", team="KC", position="WR", offense_snaps=50.0),
        dict(season=2026, week=5, name_key="quiet wr", team="KC", position="WR", offense_snaps=22.0),
        dict(season=2026, week=5, name_key="inactive rb", team="KC", position="RB", offense_snaps=0.0),
        dict(season=2026, week=5, name_key="pending wr", team="BUF", position="WR", offense_snaps=30.0),
        dict(season=2026, week=5, name_key="two teams", team="KC", position="WR", offense_snaps=9.0),
        dict(season=2026, week=5, name_key="two teams", team="BUF", position="WR", offense_snaps=40.0),
        dict(season=2026, week=5, name_key="just a lb", team="KC", position="LB", offense_snaps=12.0),
    ])
    monkeypatch.setattr("matchup_model.opp.data.snaps", lambda: snaps)
    monkeypatch.setattr("matchup_model.opp.data.player_weeks", lambda: pw)
    return pw


def test_a_skill_player_on_the_field_with_no_stat_line_really_had_zero(monkeypatch):
    pw = _quiet_week(monkeypatch)
    caches = {"dnp": {}, "snaps": {}}
    get = lambda nk: B._actual_for(pw, nk, 2026, 5, "receiving_yards", caches)    # noqa: E731
    assert get("stat guy") == (48.0, False)                  # has a stat row: unchanged
    assert get("quiet wr") == (0.0, False)                   # 22 snaps, no stats: a real zero
    assert get("inactive rb") == (0.0, True)                 # 0 snaps: inactive, not a loss
    assert get("nobody") is None                             # no snap row either: can't tell
    assert get("pending wr") is None                         # his game's player stats aren't out yet
    assert get("two teams") is None                          # one name on two teams: ambiguous
    assert get("just a lb") is None                          # not a skill position


def test_grade_line_history_counts_the_quiet_players_it_used_to_skip(tmp_path, monkeypatch):
    import pandas as pd
    from dfs import props

    _quiet_week(monkeypatch)
    monkeypatch.setattr(props, "LINE_HISTORY_PATH", tmp_path / "lh.csv")
    rows = [("Stat Guy", "rec yds", "OVER", 40.5), ("Quiet WR", "rec yds", "OVER", 7.5),
            ("Quiet WR", "rush yds", "UNDER", 1.5), ("Inactive RB", "rec yds", "OVER", 12.5),
            ("Pending WR", "rec yds", "OVER", 20.5)]
    pd.DataFrame([dict(season=2026, week=5, event="X @ Y", player=p, market=m, line=l, lean=s,
                       conf="solid", actual=np.nan, result="") for p, m, s, l in rows]
                 ).to_csv(props.LINE_HISTORY_PATH, index=False)
    _, n = B.grade_line_history()
    got = B.load_line_history().set_index(["player", "market"])
    assert n == 4
    assert got.loc[("Stat Guy", "rec yds"), "result"] == "win"
    assert got.loc[("Quiet WR", "rec yds"), "result"] == "loss"       # the OVER that was never graded
    assert got.loc[("Quiet WR", "rec yds"), "actual"] == 0.0
    assert got.loc[("Quiet WR", "rush yds"), "result"] == "win"       # ...and an UNDER on him wins
    assert got.loc[("Inactive RB", "rec yds"), "result"] == "dnp"
    assert got.loc[("Pending WR", "rec yds"), "result"] == ""          # stays pending
