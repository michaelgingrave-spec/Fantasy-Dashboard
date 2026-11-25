"""The Millionaire Maker evidence panel renders from the evidence file's structure (and degrades to a note when there is none), without needing the hosted slate."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from test_mm_study import _feature_frames

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest


def _evidence() -> dict:
    """A real `build_evidence` result on tiny frames, with price tiers and environment rows added."""
    from dfs import mm_study as S
    hind, fwd = _feature_frames()
    rng = np.random.default_rng(3)
    rows = []
    for w in range(1, 21):
        for pos, sals in (("QB", [5000, 6000, 7000]), ("RB", [3500, 5000, 7000]), ("WR", [3500, 5000, 7000, 8500]), ("TE", [2800, 4000, 5500]), ("DST", [2500, 3000, 3500])):
            for sal in sals * 4:
                rows.append({"season": 2018 + (w % 2), "week": w, "pos": pos, "salary": sal, "pts": max(0.0, rng.normal(sal / 400, 6)), "team_total": rng.normal(22, 3),
                             "env_fpa_rank": int(rng.integers(1, 33)), "env_epa_rank": int(rng.integers(1, 33)), "team_off_rank": int(rng.integers(1, 33)), "env_pa_rank": int(rng.integers(1, 33))})
    pools = pd.DataFrame(rows)
    ev = S.build_evidence(fwd, fp=fwd, hind=hind, pools=pools, reps=40, write=False)
    assert ev["spend"]["tiers"] and ev["environment"]["effects"]
    return json.loads(json.dumps(ev))                                                    # exactly what the screen reads back from disk


def _panel():
    import json as _json
    import os

    from dfs import mm_view as V
    V._evidence = lambda: _json.loads(os.environ["MM_EVIDENCE"]) if os.environ.get("MM_EVIDENCE") else None
    V._evidence_panel()


def test_panel_renders_every_tab_from_the_evidence_file(monkeypatch):
    monkeypatch.setenv("MM_EVIDENCE", json.dumps(_evidence()))
    at = AppTest.from_function(_panel, default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    assert [t.label for t in at.tabs] == ["Stacks and teams", "Spending by position", "Game totals and defenses", "Combinations and 2026"]
    frames = [df.value for df in at.dataframe]
    assert any("trait" in f.columns for f in frames)                                             # the hindsight table
    assert any("lift at 5% (95% interval)" in f.columns for f in frames)                               # the forward tables
    assert any("best lineup (95% interval)" in f.columns for f in frames)                         # the spend table
    assert any("5x rate (95%)" in f.columns for f in frames)                                     # the price tiers
    assert any("player-weeks" in f.columns for f in frames)                                      # the environment regression
    spend = next(f for f in frames if "best lineup (95% interval)" in f.columns)
    assert list(spend["slot"])[:5] == ["QB", "RBs (all)", "WRs (all)", "TEs (all)", "DST"]


def test_panel_without_an_evidence_file_says_so(monkeypatch):
    monkeypatch.delenv("MM_EVIDENCE", raising=False)
    at = AppTest.from_function(_panel, default_timeout=60).run()
    assert not at.exception
    assert any("not on this deploy" in c.value for c in at.caption)


def test_panel_tolerates_an_older_evidence_file_without_groups(monkeypatch):
    ev = _evidence()
    for r in ev["hindsight"]:
        r.pop("group", None)
    for key in ("forward", "forward_fp"):
        for r in ev[key]["rows"]:
            r.pop("group", None)
    for key in ("spend", "environment", "group_titles"):
        ev.pop(key, None)
    monkeypatch.setenv("MM_EVIDENCE", json.dumps(ev))
    at = AppTest.from_function(_panel, default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]


# ------------------------------------------------------------------------------------------------------------------------------------------------ the lineups the screen prints
SLOTS = ["QB", "RB1", "RB2", "WR1", "WR2", "WR3", "TE", "FLEX", "DST"]


def _cell_name(cell: str) -> str:
    return cell.rsplit(" (", 1)[0]


def test_lineup_table_prints_every_roster_slot_of_every_lineup():
    from test_millionaire import make_slate

    from dfs import mm_view as V
    from dfs.millionaire import MMConfig, build_portfolio
    slate = make_slate()
    lineups = build_portfolio(slate, MMConfig(n_lineups=4, seed=5))
    table = V._lineup_table(lineups, slate)
    assert list(table.columns[:10]) == ["#", *SLOTS]                                  # the whole lineup, left to right, in DraftKings' order
    assert {"salary", "left", "proj", "ceiling", "stack", "teams", "games", "own %"} <= set(table.columns)
    assert len(table) == len(lineups)
    for lu, row in zip(lineups, table.to_dict("records")):
        assert [_cell_name(row[s]) for s in SLOTS] == list(lu.players["name"])               # all nine players, none dropped or reordered
        assert all(row[s].endswith(f"({t})") for s, t in zip(SLOTS, lu.players["team"]))
        assert row["salary"] == lu.salary and row["proj"] == round(lu.proj, 1)
        assert row["stack"] == lu.stack


def _screen():
    from test_millionaire import make_slate

    from dfs import mm_view as V
    V._evidence = lambda: None                                                              # the evidence panel is tested above
    V.render(11, make_slate(), "Main · 5 games", V.MAIN_SLATE_TYPE)


def test_the_built_screen_shows_full_lineups_not_just_the_stack():
    at = AppTest.from_function(_screen, default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]
    at.number_input(key="mm_n").set_value(4).run()
    at.button(key="mm_build").click().run()
    assert not at.exception, [e.value for e in at.exception]
    lineups = at.session_state["mm_lineups"]
    assert len(lineups) == 4
    grids = [df.value for df in at.dataframe if set(SLOTS) <= set(df.value.columns)]
    assert len(grids) == 1, "the lineup table must carry one column per roster slot"
    grid = grids[0]
    assert len(grid) == 4
    for lu, row in zip(lineups, grid.to_dict("records")):
        assert {_cell_name(row[s]) for s in SLOTS} == set(lu.players["name"]) and len(set(row[s] for s in SLOTS)) == 9
    assert any("whole lineup" in c.value for c in at.caption)
