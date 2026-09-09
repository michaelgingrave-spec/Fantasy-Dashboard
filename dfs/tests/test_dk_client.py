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


def test_pick_main_slate_prefers_sunday_biggest(getcontests_json):
    slates = parse_slates(getcontests_json)
    main = pick_main_slate(slates)
    assert main is not None
    assert main.is_classic
    if main.start:
        assert main.start.weekday() == 6  # Sunday


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
