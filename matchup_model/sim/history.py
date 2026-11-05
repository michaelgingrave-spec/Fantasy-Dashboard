"""Past games as the simulator would have seen them: the walk-forward projections for everyone in the game, the spread / total, and what actually happened.

    python -m matchup_model.sim.history 2022 2026        # build / refresh the cache (data/dfs/.cache/sim, not tracked), a few minutes per season

Each record holds the `game_table` the app would have shown (only data from before the game goes into it), the schedule row, and the box score of every player who played.
`calibrate.py` and `validate.py` read these; nothing here is fitted.
"""
from __future__ import annotations

import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from dfs.config import ROOT
from dfs.names import normalize_name

CACHE = ROOT / "data" / "dfs" / ".cache" / "sim"
ACTUAL_COLS = {"attempts": "pass_att", "completions": "comp", "passing_yards": "pass_yds", "passing_tds": "pass_td", "passing_interceptions": "int", "carries": "rush_att",
               "rushing_yards": "rush_yds", "rushing_tds": "rush_td", "targets": "tgt", "receptions": "rec", "receiving_yards": "rec_yds", "receiving_tds": "rec_td"}
TEAM_COLS = {"attempts": "pass_att", "carries": "rush_att", "passing_yards": "pass_yds", "rushing_yards": "rush_yds", "passing_tds": "pass_td", "rushing_tds": "rush_td"}


def _path(season: int, week: int) -> Path:
    return CACHE / f"games_{season}_{week:02d}.pkl"


def build_week(season: int, week: int) -> list[dict]:
    """The records for every completed game of one week (those with a spread, a total and a final score)."""
    from dfs import game_view as GV
    from matchup_model.opp import data as D
    from matchup_model.sim import inputs as I
    s = D.schedule()
    g = s[(s["season"] == season) & (s["week"] == week) & s["spread_home"].notna() & s["total"].notna() & s["home_score"].notna()]
    pw = D.player_weeks()
    pw = pw[(pw["season"] == season) & (pw["week"] == week)]
    tw = D.team_weeks()
    tw = tw[(tw["season"] == season) & (tw["week"] == week)]
    out = []
    for r in g.itertuples():
        info = I.game_row(season, week, r.away, r.home)
        if info is None:
            continue
        try:
            t = GV.game_table(r.away, r.home, season, week, None)
        except Exception as exc:  # noqa: BLE001 -- one game with thin data must not stop the build
            print(f"  skipped {season} wk {week} {r.away} @ {r.home}: {exc}", flush=True)
            continue
        act = pw[pw["team"].isin([r.away, r.home])]
        act = act.assign(nk=act["name_key"] if "name_key" in act else act["player_display_name"].map(normalize_name))
        act = act[["nk", "team", "position", *ACTUAL_COLS]].rename(columns=ACTUAL_COLS)
        team = {x.team: {v: float(getattr(x, k)) for k, v in TEAM_COLS.items()} for x in tw[tw["team"].isin([r.away, r.home])].itertuples()}
        out.append({"season": season, "week": week, "away": r.away, "home": r.home, "info": info, "table": t, "actual": act.reset_index(drop=True), "team_actual": team})
    return out


def _build_and_save(args) -> str:
    season, week = args
    recs = build_week(season, week)
    CACHE.mkdir(parents=True, exist_ok=True)
    with open(_path(season, week), "wb") as f:
        pickle.dump(recs, f)
    return f"{season} wk {week}: {len(recs)} games"


def build(seasons, weeks=range(3, 19), refresh: bool = False, workers: int = 4) -> None:
    jobs = [(s, w) for s in seasons for w in weeks if refresh or not _path(s, w).exists()]
    if not jobs:
        return
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for msg in ex.map(_build_and_save, jobs):
            print(msg, flush=True)


def load(seasons, weeks=range(3, 19)) -> list[dict]:
    """Cached records (build them first with `build`)."""
    recs: list[dict] = []
    for s in seasons:
        for w in weeks:
            p = _path(s, w)
            if p.exists():
                with open(p, "rb") as f:
                    recs += pickle.load(f)
    return recs


if __name__ == "__main__":
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    a = [int(x) for x in args] or [2022, 2025]
    wk = next((x.split("=")[1] for x in sys.argv[1:] if x.startswith("--weeks=")), "3-18")
    w0, w1 = (int(v) for v in wk.split("-"))
    build(range(a[0], (a[1] if len(a) > 1 else a[0]) + 1), weeks=range(w0, w1 + 1), refresh="--refresh" in sys.argv)
