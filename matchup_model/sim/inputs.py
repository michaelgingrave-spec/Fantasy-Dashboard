"""Build the simulator's inputs for a real game: the schedule row (spread, total) and the app's own per-player projections (`dfs.game_view.game_table`)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from matchup_model.sim.engine import PlayerSpec, TeamSpec

_FIELDS = ("pass_att", "pass_yds", "pass_td", "int", "rush_att", "rush_yds", "rush_td", "tgt", "rec", "rec_yds", "rec_td")


def game_row(season: int, week: int, away: str, home: str) -> dict | None:
    """{'spread_home', 'total', 'away_implied', 'home_implied', 'away_score', 'home_score', 'kickoff'} for the game, or None when the schedule has no such game."""
    from matchup_model.opp import data as D
    s = D.schedule()
    g = s[(s["season"] == season) & (s["week"] == week) & (s["away"] == away) & (s["home"] == home)]
    if g.empty:
        return None
    r = g.iloc[0]
    spread, total = float(r["spread_home"]), float(r["total"])
    if not (np.isfinite(spread) and np.isfinite(total)):
        return None
    return {"spread_home": spread, "total": total, "away_implied": total / 2 - spread / 2, "home_implied": total / 2 + spread / 2,
            "away_score": r.get("away_score"), "home_score": r.get("home_score"), "kickoff": r.get("kickoff")}


def _player(r) -> PlayerSpec:
    v = {k: (0.0 if (k not in r or pd.isna(r[k])) else float(r[k])) for k in _FIELDS}
    return PlayerSpec(name=r["name"], pos=r["pos"], tgt=v["tgt"], rec=v["rec"], rec_yds=v["rec_yds"], rec_td=v["rec_td"], rush_att=v["rush_att"], rush_yds=v["rush_yds"],
                      rush_td=v["rush_td"], pass_att=v["pass_att"], pass_yds=v["pass_yds"], pass_td=v["pass_td"], ints=v["int"])


def specs_from_table(table: pd.DataFrame, away: str, home: str, away_implied: float, home_implied: float, drop_out: bool = True) -> tuple[TeamSpec, TeamSpec]:
    """TeamSpecs from a `game_table`: players with a model line (`ours`), without those ruled Out, with their projected box-score line as the mean."""
    t = table[table["ours"].notna()]
    if drop_out and "inj" in t:
        t = t[t["inj"] != "Out"]
    out = {}
    for tm, imp in ((home, home_implied), (away, away_implied)):
        rows = t[t["team"] == tm]
        players = [_player(r) for _, r in rows.iterrows()]
        seen: dict = {}
        for p in players:                     # two players with one name in a game are rare; keep them apart
            seen[p.name] = seen.get(p.name, 0) + 1
            if seen[p.name] > 1:
                p.name = f"{p.name} ({tm})"
        out[tm] = TeamSpec(abbr=tm, implied=float(imp), players=players)
    return out[home], out[away]


def build_specs(season: int, week: int, away: str, home: str, fp: pd.DataFrame | None = None, table: pd.DataFrame | None = None) -> tuple[TeamSpec, TeamSpec, dict]:
    """(home spec, away spec, game info) for a scheduled game; `fp` is the FantasyPoints export for the week (None is fine for past weeks)."""
    from dfs import game_view as GV
    info = game_row(season, week, away, home)
    if info is None:
        raise ValueError(f"no scheduled game {away} @ {home} in {season} week {week}, or its spread / total is missing")
    t = table if table is not None else GV.game_table(away, home, season, week, fp)
    h, a = specs_from_table(t, away, home, info["away_implied"], info["home_implied"])
    return h, a, info
