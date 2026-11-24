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
