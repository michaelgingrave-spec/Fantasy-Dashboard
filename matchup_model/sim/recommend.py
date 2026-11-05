"""Picks that go with a parlay: other lines that tend to hit when the legs already chosen hit.

`candidate_legs` lists the lines worth a look in a game: the lines pulled from the books where there are any, each main player stat at the simulator's own median where there is none (about a
coin flip in the simulation, so NOT a book's number), and the game's spread and total. `recommend` scores each against the parlay so far with `sgp.conditional`: every leg that has a book price is
pinned to it and only the association between the legs comes from the simulator, which has been validated on past lines for a quarterback and his receivers and for nothing else
(`matchup_model/sim_report.md`). A suggestion is a line that tends to hit with the legs, not a bet with an edge: the book's price for the parlay may already charge for the link.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from matchup_model.sim import engine as E
from matchup_model.sim import sgp as S

BAND = (0.25, 0.80)           # a suggestion's own probability: a near-certain or near-impossible line says nothing about the legs
MIN_LIFT = 1.10               # below this a line hits about as often with the legs as without them
SIM_BAND = (0.12, 0.88)       # the simulator's own probability: outside it a leg has too few paths on one side for its association to mean anything
COLUMNS = ["leg", "who", "alone", "when_given", "lift", "joint", "fair_odds", "source"]


def median_legs(res: E.SimResult, h: E.TeamSpec, a: E.TeamSpec, per_team: int = 7) -> list[S.Leg]:
    """The game's main player legs (over side) at the simulator's own medians, so each is about a coin flip in the simulation; anytime TD at zero, passing TDs at 1.5."""
    legs: list[S.Leg] = []
    for spec in (h, a):
        ranked = sorted(spec.players, key=lambda p: -(p.rec_yds + p.rush_yds + p.pass_yds / 4))[:per_team]
        for p in ranked:
            for market in (("pass_yds", "rec_yds", "rush_yds", "rec", "any_td") if p.pos != "QB" else ("pass_yds", "pass_td", "rush_yds")):
                if market not in res.stats[p.name]:
                    continue
                x = res.stats[p.name][market].astype(float)
                if market == "any_td":
                    line = 0.0
                elif market in ("pass_td",):
                    line = 1.5
                else:
                    med = float(np.median(x))
                    if med < 1.0 or x.mean() < (3 if market == "rec" else 8):
                        continue
                    line = np.floor(med) + 0.5
                if market in ("pass_yds",) and p.pos != "QB":
                    continue
                if market == "any_td" and float((x > 0.5).mean()) < 0.12:
                    continue
                legs.append(S.Leg(market, "over", float(line), player=p.name))
    return legs


def candidate_legs(res: E.SimResult, h: E.TeamSpec, a: E.TeamSpec, book: list[S.Leg] | None = None, game: dict | None = None) -> list[S.Leg]:
    """Both sides of every line worth a look. `book` = over-side legs carrying the book's probability, which replace the median line of the same player and stat; `game` =
    {'home', 'away', 'spread_home', 'total'} adds the two spreads (each team covering) and the total's two sides at the market's number, each taken as a 50/50 line."""
    book = [l for l in (book or []) if l.player in res.stats and S.PLAYER_MARKETS[l.market] in res.stats[l.player]]
    taken = {(l.player, l.market) for l in book}
    out: list[S.Leg] = []
    for l in book + [m for m in median_legs(res, h, a) if (m.player, m.market) not in taken]:
        out += [l, S.Leg(l.market, "under", l.line, player=l.player, p=None if l.p is None else 1.0 - l.p)]
    if game:
        out += [S.Leg("total", side, float(game["total"]), p=0.5) for side in ("over", "under")]
        out += [S.Leg("spread", "over", handicap, team=team, p=0.5) for team, handicap in ((game["home"], -float(game["spread_home"])), (game["away"], float(game["spread_home"])))]
    return out


def _source(leg: S.Leg) -> str:
    if leg.p is None:
        return "simulator median"
    return "market line" if leg.market in S.TEAM_MARKETS else "book line"


def recommend(res: E.SimResult, given: list[S.Leg], candidates: list[S.Leg], rank: str = "when_given", min_lift: float = MIN_LIFT, band: tuple[float, float] = BAND,
              top: int = 12) -> pd.DataFrame:
    """The best line per player (or team, or the game total) to add to `given`, ranked by `rank` ('when_given': how often it hits in the paths where every given leg does; or 'lift': that
    over its own probability). Players already in `given` are left out, and so is a second spread or total when one is there. Columns: leg, who, alone, when_given, lift, joint (the parlay
    with it) and its fair American odds, source of the line ('book line', 'market line' for a spread or total at the market's number, 'simulator median')."""
    if rank not in ("when_given", "lift"):
        raise ValueError(f"unknown rank {rank!r}")
    used = {l.player for l in given if l.player}
    used_markets = {l.market for l in given if not l.player}                  # a spread or a total already in the parlay: no second line on it
    keep = []
    for c in candidates:
        if c.player and (c.player in used or c.player not in res.stats):
            continue
        if not c.player and c.market in used_markets:
            continue
        if not SIM_BAND[0] < float(S.leg_events(res, c).mean()) < SIM_BAND[1]:
            continue
        if c.p is not None and not band[0] <= c.p <= band[1]:
            continue
        keep.append(c)
    rows = []
    for c, r in zip(keep, S.conditional(res, given, keep)):
        if band[0] <= r["alone"] <= band[1] and r["lift"] >= min_lift:
            rows.append({"leg": c, "who": c.player or c.team or "game", "alone": r["alone"], "when_given": r["when_given"], "lift": r["lift"], "joint": r["joint"],
                         "fair_odds": S.to_american(r["joint"]), "source": _source(c)})
    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    d = pd.DataFrame(rows).sort_values(rank, ascending=False).drop_duplicates("who")
    return d.head(top).reset_index(drop=True)
