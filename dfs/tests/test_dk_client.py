import json

import dfs.dk_client as dk
from dfs.dk_client import (
    SLATE_TYPES,
    parse_draftables,
    parse_slates,
    pick_main_slate,
    slate_types_present,
)


def test_parse_slates_classifies_classic_vs_showdown(getcontests_json):
    slates = parse_slates(getcontests_json)
    assert len(slates) >= 8
    classic = [s for s in slates if s.is_classic]
    assert classic, "expected at least one Classic slate"
    # single-game suffixes like ' (SF vs LAR)' must not count as classic
    assert all("vs" not in s.suffix.lower().split("(")[-1] or s.game_count >= 2 for s in classic)
    for s in classic:
        assert s.game_count >= 2


def test_pick_main_slate_prefers_sunday_main_type(getcontests_json):
    slates = parse_slates(getcontests_json)
    main = pick_main_slate(slates)
    assert main is not None
    assert main.is_classic
    if main.start:
        assert main.start.weekday() == 6  # Sunday
    # 153069 (Sunday-Monday, 14 games) outnumbers 153096/151307 (Sunday main, 12 games
    # each) in this fixture -- pick_main_slate must not be fooled by MNF into picking
    # the bigger slate over the 1pm/4pm-only one.
    assert main.slate_type == "Sunday main"
    assert main.draft_group_id in (153096, 151307)


def test_slate_type_classification(getcontests_json):
    slates = {s.draft_group_id: s for s in parse_slates(getcontests_json)}
    assert slates[153054].slate_type == "Full week"       # (Wed-Mon), starts Wed
    assert slates[153069].slate_type == "Sunday–Monday"   # (Sun-Mon)
    assert slates[153068].slate_type == "Sunday early"    # (Early Only)
    assert slates[153070].slate_type == "Sunday afternoon"  # (Afternoon Only)
    assert slates[153096].slate_type == "Sunday main"     # empty suffix, Sun 1pm
    assert slates[151307].slate_type == "Sunday main"     # None suffix, Sun 1pm


def test_slate_types_present_is_ordered_and_deduped(getcontests_json):
    classic = [s for s in parse_slates(getcontests_json) if s.is_classic]
    present = slate_types_present(classic)
    assert present == [t for t in SLATE_TYPES if t in present]  # SLATE_TYPES order
    assert len(present) == len(set(present))
    assert "Sunday main" in present and "Full week" in present


def test_snapshot_fallback_when_dk_unreachable(getcontests_json, draftables_json, tmp_path, monkeypatch):
    """When the live DK request fails, list_slates/get_salaries read the committed snapshot."""
    snap = tmp_path / "dk_snapshot"
    snap.mkdir()
    (snap / "lobby_NFL.json").write_text(json.dumps(getcontests_json))
    # snapshot a draftables file for one classic slate id present in the fixture lobby
    gid = next(s.draft_group_id for s in parse_slates(getcontests_json) if s.is_classic)
    (snap / f"draftables_{gid}.json").write_text(json.dumps(draftables_json))

    monkeypatch.setattr(dk, "DK_SNAPSHOT_DIR", snap)
    monkeypatch.setattr(dk, "_get_json", lambda *a, **k: (_ for _ in ()).throw(dk.DKError("blocked")))
    monkeypatch.setattr(dk, "_read_cache", lambda *a, **k: None)

    slates = dk.classic_slates(use_cache=False)
    assert slates, "should recover slates from the snapshot"
    sal = dk.get_salaries(gid, use_cache=False)
    assert not sal.empty and "DST" in set(sal["pos"])


def test_parse_draftables_shape(draftables_json):
    df = parse_draftables(draftables_json, 153069)
    assert set(["dk_id", "name", "pos", "team", "salary", "playable"]).issubset(df.columns)
    assert "DST" in set(df["pos"])
    assert df["salary"].dtype.kind in "iu" and df["salary"].min() > 0
    assert df["dk_id"].is_unique
    assert (df["team"].str.len() <= 3).all()


# ── the slates of the week being played ────────────────────────────────────────────────────────────────────
def _slate(gid, games, start, suffix=""):
    return dk.Slate(draft_group_id=gid, contest_type_id=21, game_count=games, start=start, suffix=suffix, tag="")


def _week6_lobby():
    """DraftKings' lobby on Fri 2026-10-16 as it really was: this week's slates (two 10-game 'Main's, because of byes) AND a 'Main' with MORE games for
    Nov 1 that DraftKings pre-created weeks ago (it has no salaries yet)."""
    from datetime import datetime, timezone
    u = timezone.utc
    return [_slate(154943, 12, datetime(2026, 10, 18, 17, tzinfo=u), "(Sun-Mon)"), _slate(154941, 10, datetime(2026, 10, 18, 17, tzinfo=u)),
            _slate(154968, 10, datetime(2026, 10, 18, 17, tzinfo=u)), _slate(154942, 7, datetime(2026, 10, 18, 17, tzinfo=u), "(Early Only)"),
            _slate(154944, 3, datetime(2026, 10, 18, 20, 5, tzinfo=u), "(Afternoon Only)"), _slate(154981, 2, datetime(2026, 10, 20, 0, 15, tzinfo=u), "(Mon-Thu)"),
            _slate(153728, 11, datetime(2026, 11, 1, 18, tzinfo=u))]


def test_the_default_slate_is_this_weeks_main_not_a_later_weeks_bigger_one():
    from datetime import datetime, timezone
    lobby = _week6_lobby()
    friday = datetime(2026, 10, 16, 21, tzinfo=timezone.utc)
    assert max((s for s in lobby if s.slate_type == "Sunday main"), key=lambda s: s.game_count).draft_group_id == 153728       # what the old rule picked
    assert pick_main_slate(lobby, friday).draft_group_id == 154941                                                           # this week's main (ties go to the first)
    assert pick_main_slate(lobby, datetime(2026, 10, 18, 22, tzinfo=timezone.utc)).draft_group_id == 154941                 # still it after the 1pm kickoff
    assert pick_main_slate([], friday) is None


def test_this_week_keeps_the_games_being_played_and_drops_later_weeks():
    from datetime import datetime, timezone
    u = timezone.utc
    lobby = _week6_lobby()
    ids = lambda now: {s.draft_group_id for s in dk.this_week(lobby, now)}  # noqa: E731
    this_wk = {154943, 154941, 154968, 154942, 154944, 154981}
    assert ids(datetime(2026, 10, 16, 21, tzinfo=u)) == this_wk                                                              # Friday: the whole week, not Nov 1
    assert ids(datetime(2026, 10, 18, 22, tzinfo=u)) == this_wk                                                              # Sunday evening: kicked off 5h ago, still listed
    assert ids(datetime(2026, 10, 19, 15, tzinfo=u)) == {154981}                                                             # Monday: only the night game is left
    assert ids(datetime(2026, 10, 20, 15, tzinfo=u)) == {153728}                                                             # nothing sooner is listed: the next one is all there is
    assert {s.draft_group_id for s in dk.this_week(lobby, datetime(2027, 1, 1, tzinfo=u))} == {s.draft_group_id for s in lobby}   # nothing upcoming: every slate


def test_this_week_keeps_undated_slates_and_tolerates_naive_start_times():
    from datetime import datetime, timezone
    lobby = _week6_lobby() + [_slate(1, 5, None), _slate(2, 6, datetime(2026, 10, 18, 17))]                                   # a naive start time is read as UTC
    got = {s.draft_group_id for s in dk.this_week(lobby, datetime(2026, 10, 16, 21, tzinfo=timezone.utc))}
    assert {1, 2} <= got and 153728 not in got
