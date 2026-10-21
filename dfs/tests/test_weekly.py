"""dfs.weekly, the whole weekly job: credit-safe buying, logging that never overwrites a live pull, the week's summary, the FantasyPoints install, and the git step.
Everything outside the repo (the Odds API, nflverse, git, the slow reports) is replaced by fakes, so nothing here spends a credit or touches the working tree."""
import json
import time
from datetime import datetime, timezone

import pandas as pd
import pytest
import requests

from dfs import props
from dfs import weekly as W

TEAMS = "ARZ ATL BLT BUF CAR CHI CIN CLV DAL DEN DET GB HST IND JAX KC LA LAC LV MIA MIN NE NO NYG NYJ PHI PIT SEA SF TB TEN WAS".split()
quiet = lambda *a, **k: None   # noqa: E731


# ------------------------------------------------------------------------------------------------------------------------------------------------ fakes
class FakeApi:
    """The Odds API: a game listing (1 credit) and a flat 50-credit odds call; every call is recorded."""
    def __init__(self, balance=5000, commence=("2026-10-16T00:15:00Z", "2026-10-18T17:00:00Z", "2026-10-20T00:15:00Z")):
        self.balance, self.calls = balance, []
        self.events = [{"id": f"g{i}", "commence_time": t, "home_team": f"H{i}", "away_team": f"A{i}"} for i, t in enumerate(commence)]

    def __call__(self, path, **params):
        self.calls.append((path, params))
        cost = 1 if path == "/events" else 50
        self.balance -= cost
        hdr = {"remaining": str(self.balance), "last": str(cost)}
        return ({"data": self.events} if path == "/events" else {"timestamp": params["date"], "data": {"bookmakers": ["payload for " + path]}}), hdr

    def odds_calls(self):
        return [p for p, _ in self.calls if p.endswith("/odds")]


def fake_edges(raw, week):
    """One game's edges: two lines, named after the payload so each game is distinguishable."""
    tag = raw["bookmakers"][0].split("/events/")[1].split("/")[0]
    return pd.DataFrame([{"player": f"{tag} A", "market": "rec yds", "line": 40.5, "our proj": 50.0, "conf": "solid", "z": 0.5, "lean": "OVER", "role_ratio": 1.0},
                         {"player": f"{tag} B", "market": "receptions", "line": 3.5, "our proj": 3.0, "conf": "lean", "z": -0.3, "lean": "UNDER", "role_ratio": 1.0}])


@pytest.fixture()
def lh_path(tmp_path, monkeypatch):
    monkeypatch.setattr(props, "LINE_HISTORY_PATH", tmp_path / "line_history.csv")
    return props.LINE_HISTORY_PATH


# ------------------------------------------------------------------------------------------------------------------------------------------------ games and credits
def test_a_game_counts_as_finished_four_hours_after_kickoff():
    ev = [{"id": "a", "commence_time": "2026-10-18T17:00:00Z"}, {"id": "b", "commence_time": "2026-10-18T20:25:00Z"}]
    now = datetime(2026, 10, 18, 21, 5, tzinfo=timezone.utc)
    assert [e["id"] for e in W.finished(ev, now)] == ["a"]                       # 4h05 after the first, 40 minutes after the second
    assert W.finished(ev, datetime(2026, 10, 18, 12, 0, tzinfo=timezone.utc)) == []


def test_the_listing_is_bought_once_cached_and_never_fetched_in_a_dry_run(tmp_path):
    api = FakeApi()
    assert W.week_games(2026, 6, get=api, raw_dir=tmp_path, fetch=False) == (None, 0) and api.calls == []      # a dry run spends nothing
    ev, spent = W.week_games(2026, 6, get=api, raw_dir=tmp_path)
    assert [e["id"] for e in ev] == ["g0", "g1", "g2"] and spent == 1 and len(api.calls) == 1
    ev2, spent2 = W.week_games(2026, 6, get=api, raw_dir=tmp_path)
    assert [e["id"] for e in ev2] == ["g0", "g1", "g2"] and spent2 == 0 and len(api.calls) == 1               # from disk the second time


def test_each_game_is_bought_once_at_five_minutes_before_kickoff_with_the_five_core_markets(tmp_path):
    api = FakeApi()
    out = W.buy_lines(api.events, 2026, 6, floor=0, bal=api.balance, get=api, raw_dir=tmp_path, say=quiet)
    assert out["bought"] == 3 and out["spent"] == 150 and out["balance"] == 4850
    path, params = api.calls[0]
    assert path == "/events/g0/odds" and params["date"] == "2026-10-16T00:10:00Z" and params["markets"] == ",".join(props.CORE_MARKETS)
    assert params["bookmakers"] == "draftkings,fanduel" and len(props.CORE_MARKETS) == 5 and W.PER_GAME == 50
    again = W.buy_lines(api.events, 2026, 6, floor=0, bal=out["balance"], get=api, raw_dir=tmp_path, say=quiet)
    assert again["bought"] == 0 and again["cached"] == 3 and len(api.odds_calls()) == 3                       # nothing is bought twice


def test_the_floor_stops_the_buying_before_it_would_be_crossed(tmp_path):
    api = FakeApi(balance=140)
    out = W.buy_lines(api.events, 2026, 6, floor=60, bal=api.balance, get=api, raw_dir=tmp_path, say=quiet)
    assert out["bought"] == 1 and out["stopped"] and api.balance == 90 and len(api.odds_calls()) == 1          # 140 -> 90; a second game would leave 40 < 60


def test_a_failed_game_is_skipped_and_the_rest_are_still_bought(tmp_path):
    api = FakeApi()
    real = api.__call__

    def flaky(path, **params):
        if path == "/events/g1/odds":
            raise requests.HTTPError("422")
        return real(path, **params)
    out = W.buy_lines(api.events, 2026, 6, floor=0, bal=5000, get=flaky, raw_dir=tmp_path, say=quiet)
    assert out["bought"] == 2 and out["failed"] == ["A1 @ H1"] and not W.raw_path(2026, 6, "g1", tmp_path).exists()


# ------------------------------------------------------------------------------------------------------------------------------------------------ logging
def test_the_event_label_follows_the_us_eastern_weekday():
    assert W.label_for("2026-10-16T00:15:00Z", 6) == "Week 6 Thursday"       # Thursday night, already Friday in UTC
    assert W.label_for("2026-10-18T13:30:00Z", 6) == "Week 6 Sunday"
    assert W.label_for("2026-10-20T00:15:00Z", 6) == "Week 6 Monday"         # Monday night, already Tuesday in UTC


def _bought(tmp_path, api):
    W.buy_lines(api.events, 2026, 6, floor=0, bal=api.balance, get=api, raw_dir=tmp_path, say=quiet)


def test_lines_are_logged_by_game_and_a_rerun_adds_nothing(tmp_path, lh_path):
    api = FakeApi()
    _bought(tmp_path, api)
    out = W.log_lines(api.events, 2026, 6, edges_fn=fake_edges, raw_dir=tmp_path, say=quiet)
    lh = pd.read_csv(lh_path)
    assert out["rows"] == 6 and out["games"] == 3 and len(lh) == 6 and set(lh.week) == {6}
    assert set(lh.event) == {"Week 6 Thursday", "Week 6 Sunday", "Week 6 Monday"} and set(lh.model_ver) == {float(props._model_version())}
    again = W.log_lines(api.events, 2026, 6, edges_fn=fake_edges, raw_dir=tmp_path, say=quiet)
    assert again["rows"] == 0 and len(pd.read_csv(lh_path)) == 6                                              # nothing logged twice


def test_a_line_already_logged_live_is_not_overwritten(tmp_path, lh_path):
    api = FakeApi()
    _bought(tmp_path, api)
    live = pd.DataFrame([{"season": 2026, "week": 6, "event": "live pull", "player": "g0 A", "market": "rec yds", "line": 38.5, "our proj": 49.0, "lean": "OVER", "conf": "solid",
                          "result": "", "actual": None}])
    live.to_csv(lh_path, index=False)
    out = W.log_lines(api.events, 2026, 6, edges_fn=fake_edges, raw_dir=tmp_path, say=quiet)
    lh = pd.read_csv(lh_path)
    row = lh[(lh.player == "g0 A") & (lh.market == "rec yds")].iloc[0]
    assert out["rows"] == 5 and len(lh) == 6 and row.line == 38.5 and row.event == "live pull"                # the live line stays, the other five are added
    W.log_lines(api.events, 2026, 6, edges_fn=fake_edges, raw_dir=tmp_path, relog=True, say=quiet)
    assert pd.read_csv(lh_path).query("player == 'g0 A' and market == 'rec yds'").iloc[0].line == 40.5        # relog replaces it on request


def test_a_game_that_was_never_bought_is_reported_not_invented(tmp_path, lh_path):
    api = FakeApi()
    W.buy_lines(api.events[:2], 2026, 6, floor=0, bal=api.balance, get=api, raw_dir=tmp_path, say=quiet)
    out = W.log_lines(api.events, 2026, 6, edges_fn=fake_edges, raw_dir=tmp_path, say=quiet)
    assert out["games"] == 2 and out["missing"] == ["A2 @ H2"]


# ------------------------------------------------------------------------------------------------------------------------------------------------ the summary
def _graded(n_over=40, n_under=40, week=6):
    rows = []
    for i in range(n_over):
        rows.append({"season": 2026, "week": week, "player": f"O{i}", "market": "rec yds", "lean": "OVER", "conf": "solid", "role_ratio": 1.0, "line": 40.0, "our proj": 45.0,
                     "result": "win" if i < 20 else "loss", "bet price": -110.0, "model_ver": 4.0})
    for i in range(n_under):
        rows.append({"season": 2026, "week": week, "player": f"U{i}", "market": "receptions", "lean": "UNDER", "conf": "strong", "role_ratio": 1.0, "line": 4.5, "our proj": 4.0,
                     "result": "win" if i < 28 else "loss", "bet price": -110.0, "model_ver": 4.0})
    rows.append({"season": 2026, "week": week, "player": "P", "market": "rec yds", "lean": "OVER", "conf": "solid", "role_ratio": 1.0, "line": 40.0, "our proj": 45.0, "result": "push",
                 "bet price": -110.0, "model_ver": 4.0})
    rows.append({"season": 2026, "week": week, "player": "D", "market": "rec yds", "lean": "OVER", "conf": "solid", "role_ratio": 1.0, "line": 40.0, "our proj": 45.0, "result": "dnp",
                 "bet price": -110.0, "model_ver": 4.0})
    return pd.DataFrame(rows)


def test_the_week_report_states_counts_sides_and_the_honest_caveat():
    txt = "\n".join(W.week_report(_graded(), 2026, 6, reps=60))
    assert "82 lines logged, 80 graded win/loss, 1 push, 1 did not play, 0 ungraded; model version(s) [4.0]" in txt
    row = {l.split("  ")[0].strip(): l for l in txt.splitlines() if l.startswith(("all leans", "OVER leans", "UNDER leans", "screen default"))}
    assert " 80 " in row["all leans"] and "60.0%" in row["all leans"]                                       # 48 of 80
    assert " 40 " in row["OVER leans"] and "50.0%" in row["OVER leans"] and "70.0%" in row["UNDER leans"]
    assert "UNDER minus OVER win rate: +20.0 points" in txt and "rec yds" in txt and "receptions" in txt and "correlated lines" in txt


def test_the_report_survives_nothing_logged_and_nothing_graded():
    assert W.week_report(pd.DataFrame(), 2026, 6) == ["week 6: no lines logged"]
    ungraded = _graded().assign(result="")
    assert "nothing graded yet" in "\n".join(W.week_report(ungraded, 2026, 6, reps=20))


def test_the_weeks_so_far_table_appears_once_there_is_more_than_one_week():
    lh = pd.concat([_graded(week=5), _graded(week=6)], ignore_index=True)
    txt = "\n".join(W.week_report(lh, 2026, 6, reps=20))
    assert "the weeks so far" in txt and "week 5:" in txt and "week 6:" in txt


# ------------------------------------------------------------------------------------------------------------------------------------------------ FantasyPoints projections
def write_export(path, week=7, n=300, scale=1.0, drop=0, season=2026):
    """A FantasyPoints weekly export: a group-label row, the real header, then the players (n of them spread over all 32 teams)."""
    lines = ['"","","Player Details","","","","","Projection"', '"#","NAME","POS","POS","SEASON","WEEK","OPP","FPTS"']
    for i in range(drop, n):
        pos = ["QB", "RB", "WR", "TE", "WR", "RB"][i % 6]
        lines.append(f'"{i + 1}","Player {i}","{pos}","{TEAMS[i % 32]}","{season}","{week}","{TEAMS[(i + 7) % 32]}","{round((20 - i / 20) * scale, 1)}"')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_the_newest_recent_export_is_found_by_modification_time(tmp_path):
    a, b, c = (write_export(tmp_path / f"projections.weekly{s}.csv") for s in ("", " (1)", " (2)"))
    now = time.time()
    import os
    os.utime(a, (now - 3 * 86400, now - 3 * 86400)); os.utime(b, (now - 3600, now - 3600)); os.utime(c, (now - 20 * 86400, now - 20 * 86400))
    assert W.newest_export(tmp_path, now=now) == b                                                      # newest; the 20-day-old one is too stale to count
    assert W.newest_export(tmp_path / "empty", now=now) is None and W.newest_export(tmp_path, now=now + 40 * 86400) is None


def test_the_week_comes_from_the_file_itself(tmp_path):
    assert W.export_week(write_export(tmp_path / "a.csv", week=7)) == 7
    mixed = tmp_path / "mixed.csv"
    mixed.write_text(write_export(tmp_path / "x.csv", week=7).read_text().replace('"2026","7","', '"2026","8","', 1))
    assert W.export_week(mixed) is None and W.export_week(tmp_path / "missing.csv") is None


def test_a_good_export_installs_under_its_own_week_and_a_second_run_says_so(tmp_path):
    dl, data = tmp_path / "dl", tmp_path / "data"
    dl.mkdir(); data.mkdir()
    src = write_export(dl / "projections.weekly (15).csv", week=7)
    r = W.install_projections(downloads=dl, data_dir=data)
    assert r["status"] == "installed" and r["week"] == 7 and (data / "projections.week7.csv").read_bytes() == src.read_bytes()
    assert r["check"]["rows"] == 300 + 0 and r["check"]["teams"] == 32 and r["check"]["diff"] is None
    assert "installed projections.weekly (15).csv as projections.week7.csv (week 7): 300 rows, 32 teams (a new week)" in W._describe_install(r)
    assert W.install_projections(downloads=dl, data_dir=data)["status"] == "already installed"


def test_a_dry_run_checks_the_export_but_copies_nothing(tmp_path):
    dl, data = tmp_path / "dl", tmp_path / "data"
    dl.mkdir(); data.mkdir()
    write_export(dl / "projections.weekly.csv", week=7)
    r = W.install_projections(downloads=dl, data_dir=data, install=False)
    assert r["status"] == "dry run" and not (data / "projections.week7.csv").exists() and "would install" in W._describe_install(r)


def test_an_update_to_an_installed_week_reports_how_much_moved(tmp_path):
    dl, data = tmp_path / "dl", tmp_path / "data"
    dl.mkdir(); data.mkdir()
    write_export(data / "projections.week7.csv", week=7)
    write_export(dl / "projections.weekly (2).csv", week=7, scale=1.2)                                  # every projection 20% higher
    r = W.install_projections(downloads=dl, data_dir=data)
    d = r["check"]["diff"]
    assert r["status"] == "installed" and d["added"] == 0 and d["dropped"] == 0 and d["moved_1"] > 100 and d["mean_change"] > 0.5
    assert "against the file it replaces" in W._describe_install(r)


@pytest.mark.parametrize("make, needle", [
    (lambda p: write_export(p, week=7, n=60), "rows is outside the usual"),
    (lambda p: write_export(p, week=7, n=300).write_text(write_export(p, week=7, n=300).read_text().replace(',"FPTS"', ',"FPTS"').replace('"20.0"', '"abc"')), "not numbers"),
    (lambda p: write_export(p, week=7, n=300, season=2026).write_text(write_export(p, week=7, n=300).read_text().replace('"Player 5"', '"Player 4"')), "duplicate names"),
])
def test_a_bad_export_is_refused(tmp_path, make, needle):
    dl, data = tmp_path / "dl", tmp_path / "data"
    dl.mkdir(); data.mkdir()
    make(dl / "projections.weekly.csv")
    r = W.install_projections(downloads=dl, data_dir=data)
    assert r["status"] == "refused" and needle in "; ".join(r["check"]["problems"]) and not list(data.iterdir())


def test_an_export_for_the_wrong_week_or_the_wrong_size_is_refused(tmp_path):
    dl, data = tmp_path / "dl", tmp_path / "data"
    dl.mkdir(); data.mkdir()
    write_export(data / "projections.week7.csv", week=7, n=300)
    write_export(dl / "projections.weekly.csv", week=7, n=200)                                          # a third of the players are gone: a truncated download
    r = W.install_projections(downloads=dl, data_dir=data)
    assert r["status"] == "refused" and "far from the 300 in the file it replaces" in r["check"]["problems"][0]
    src = write_export(tmp_path / "other.csv", week=8)
    r2 = W.install_projections(src, 7, downloads=dl, data_dir=data)
    assert r2["status"] == "refused" and "WEEK column says 8, expected 7" in r2["check"]["problems"][0]


def test_with_no_export_in_downloads_it_says_what_to_do(tmp_path):
    r = W.install_projections(downloads=tmp_path, data_dir=tmp_path)
    assert r["status"] == "none found" and "Download CSV" in W._describe_install(r)


# ------------------------------------------------------------------------------------------------------------------------------------------------ git
class Git:
    """A fake subprocess runner: records the commands and answers `git diff --cached` from a script."""
    def __init__(self, staged="", clean_after_add=False):
        self.cmds, self.staged, self.clean = [], staged, clean_after_add

    def __call__(self, cmd, **kw):
        self.cmds.append(cmd)
        class R:  # noqa: N801
            stdout, returncode = "", 0
        r = R()
        if cmd[:4] == ["git", "diff", "--cached", "--name-only"]:
            r.stdout = self.staged
        if cmd[:4] == ["git", "diff", "--cached", "--quiet"]:
            r.returncode = 0 if self.clean else 1
        return r


def test_only_the_named_files_are_staged_committed_and_pushed(tmp_path):
    (tmp_path / "a.csv").write_text("x"); (tmp_path / "b.md").write_text("y")
    g = Git()
    msg = W.commit_and_push(["a.csv", "b.md", "gone.csv"], "Backfill and grade week 6 props", root=tmp_path, runner=g)
    assert ["git", "add", "--", "a.csv", "b.md"] in g.cmds and ["git", "commit", "-m", "Backfill and grade week 6 props"] in g.cmds and ["git", "push"] in g.cmds
    assert msg.startswith("committed + pushed")


def test_nothing_is_committed_or_pushed_when_nothing_changed(tmp_path):
    (tmp_path / "a.csv").write_text("x")
    g = Git(clean_after_add=True)
    assert W.commit_and_push(["a.csv"], "m", root=tmp_path, runner=g) == "nothing to commit"
    assert not any(c[:2] in (["git", "commit"], ["git", "push"]) for c in g.cmds)


def test_something_else_already_staged_stops_the_push_before_anything_is_added(tmp_path):
    g = Git(staged="dashboard.py\n")
    with pytest.raises(RuntimeError, match="already staged"):
        W.commit_and_push(["a.csv"], "m", root=tmp_path, runner=g)
    assert not any(c[:2] == ["git", "add"] for c in g.cmds)


def test_the_slow_reports_are_skipped_by_fast_and_a_failure_is_reported_not_raised(tmp_path):
    ran = []

    def runner(cmd, **kw):
        ran.append(cmd)
        class R:  # noqa: N801
            returncode, stdout, stderr = (1 if "restate" in " ".join(cmd) else 0), "", "boom"
        return R()
    assert W.run_reports(True, root=tmp_path, runner=runner, say=quiet) == [] and ran == []
    done = W.run_reports(False, root=tmp_path, runner=runner, say=quiet)
    assert done == [W.REPORTS[1]] and len(ran) == 2                                                     # the version exam failed, the forward tests still ran


# ------------------------------------------------------------------------------------------------------------------------------------------------ the whole job
def job(tmp_path, lh_path, api=None, **kw):
    """run_week with every outside dependency replaced; returns (result, api, calls)."""
    api = api or FakeApi()
    calls = {"refresh": 0, "grade": 0, "reports": [], "git": []}
    dl, data = tmp_path / "dl", tmp_path / "data"
    dl.mkdir(exist_ok=True); data.mkdir(exist_ok=True)
    kw.setdefault("now", datetime(2026, 10, 21, 12, 0, tzinfo=timezone.utc))

    def grade():
        calls["grade"] += 1
        df = pd.read_csv(lh_path)
        df["result"] = "win"
        df.to_csv(lh_path, index=False)
        return df, len(df)
    res = W.run_week(6, get=api, balance_fn=lambda: api.balance, refresh_fn=lambda: calls.__setitem__("refresh", calls["refresh"] + 1), edges_fn=fake_edges, grade_fn=grade,
                     reports_fn=lambda fast: calls["reports"].append(fast) or ["matchup_model/restate_2026_report.md"],
                     git_fn=lambda paths, msg: calls["git"].append((paths, msg)) or "committed + pushed", downloads=dl, data_dir=data, raw_dir=tmp_path / "raw", say=quiet, **kw)
    return res, api, calls


def test_a_dry_run_spends_nothing_writes_nothing_and_pushes_nothing(tmp_path, lh_path):
    (tmp_path / "dl").mkdir()
    write_export(tmp_path / "dl" / "projections.weekly.csv", week=7)
    res, api, calls = job(tmp_path, lh_path, push=True)
    assert api.calls == [] and calls == {"refresh": 0, "grade": 0, "reports": [], "git": []} and not lh_path.exists()
    assert res["touched"] == [] and not (tmp_path / "data" / "projections.week7.csv").exists() and not (tmp_path / "raw").exists()
    assert res["projections"]["status"] == "dry run"


def test_a_full_run_buys_logs_grades_installs_and_pushes_exactly_what_it_touched(tmp_path, lh_path):
    (tmp_path / "dl").mkdir()
    write_export(tmp_path / "dl" / "projections.weekly (3).csv", week=7)
    res, api, calls = job(tmp_path, lh_path, execute=True, push=True, floor=1000)
    assert [c for c in api.calls if c[0] == "/events"] and len(api.odds_calls()) == 3 and res["buy"]["spent"] == 151 and api.balance == 5000 - 151
    assert calls["refresh"] == 1 and calls["grade"] == 1 and calls["reports"] == [False]
    assert len(pd.read_csv(lh_path)) == 6 and (tmp_path / "data" / "projections.week7.csv").exists()
    paths, msg = calls["git"][0]
    assert paths == ["data/dfs/props/line_history.csv", "data/dfs/projections.week7.csv", "matchup_model/restate_2026_report.md"]
    assert msg == "Backfill and grade week 6 props; FantasyPoints week 7 projections"
    again, api2, calls2 = job(tmp_path, lh_path, api=api, execute=True)
    assert len(api.odds_calls()) == 3 and again["log"]["rows"] == 0                                    # a second run buys and logs nothing new


def test_a_game_still_to_be_played_is_left_for_the_next_run(tmp_path, lh_path):
    late = FakeApi(commence=("2026-10-16T00:15:00Z", "2026-10-18T17:00:00Z", "2026-10-21T00:15:00Z"))
    res, api, calls = job(tmp_path, lh_path, api=late, execute=True, do_projections=False, now=datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc))
    assert len(api.odds_calls()) == 2 and res["buy"]["bought"] == 2                                    # the third kicks off tomorrow


def test_the_floor_holds_for_a_whole_run(tmp_path, lh_path):
    poor = FakeApi(balance=190)
    res, api, calls = job(tmp_path, lh_path, api=poor, execute=True, floor=100, do_projections=False)
    assert api.balance >= 100 and res["buy"]["stopped"] and res["buy"]["bought"] == 1                  # 190 - 1 (listing) = 189; one game leaves 139, a second would leave 89 < 100


def test_lines_only_and_projections_only_each_leave_the_other_alone(tmp_path, lh_path):
    (tmp_path / "dl").mkdir()
    write_export(tmp_path / "dl" / "projections.weekly.csv", week=7)
    res, api, calls = job(tmp_path, lh_path, execute=True, do_lines=False)
    assert api.calls == [] and calls["grade"] == 0 and calls["reports"] == [] and res["touched"] == ["data/dfs/projections.week7.csv"]
    res2, api2, calls2 = job(tmp_path, lh_path, execute=True, do_projections=False)
    assert "projections" not in res2 and calls2["grade"] == 1


def test_the_command_line_runs_a_dry_run_unless_told_otherwise(monkeypatch):
    seen = {}
    monkeypatch.setattr(W, "run_week", lambda week, **kw: seen.update(week=week, **kw))
    import matchup_model.weekly_pull as WP
    monkeypatch.setattr(WP, "last_completed_week", lambda: 6)
    assert W.main([]) == 0 and seen["week"] == 6 and seen["execute"] is False and seen["push"] is False and seen["floor"] == W.DEFAULT_FLOOR
    assert W.main(["7", "--execute", "--push", "--floor=3000", "--fast", "--no-projections"]) == 0
    assert seen["week"] == 7 and seen["execute"] and seen["push"] and seen["floor"] == 3000 and seen["fast"] and seen["do_projections"] is False and seen["do_lines"] is True
    monkeypatch.setattr(WP, "last_completed_week", lambda: 0)
    assert W.main([]) == 1                                                                              # before the season there is nothing to grade
