"""How often a parlay's legs have hit before: the same lines, the same players, in the games they really played.

`past_hits` takes the legs the Parlay Sim prices and walks back through the regular-season box scores (`matchup_model.opp.data`, 2021 on): each leg alone over every past game
its player played, and all the legs together over the past games in which every one of the players (and teams) was in the same game. It is the record and nothing more: no
forecast, nothing that feeds the price, and the number of games sits beside every figure so a small sample reads as one.

A player played in a game when he has a stat line or an offensive snap there (a receiver who ran routes and was never targeted has snaps and no stat line, and an under on his
yards hit). A game is past when it is before the one on screen and has a final score. `sgp.on_side` decides a hit, so the record and the simulator share one rule.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dfs.names import normalize_name
from matchup_model.sim import sgp as S
from matchup_model.sim.history import ACTUAL_COLS

STATS = list(ACTUAL_COLS.values())
TEAM_VALUE = {"spread": "margin", "moneyline": "margin", "team_total": "pts", "total": "total"}


@dataclass(frozen=True)
class Tables:
    """What the record is read from: player-games (`D.player_weeks`), offensive snaps (`D.snaps`; None = only players with a stat line count as having played) and the schedule."""
    player_weeks: pd.DataFrame
    snaps: pd.DataFrame | None
    schedule: pd.DataFrame


@dataclass
class Past:
    first: int                               # the first season of the box scores the record reads from
    windows: list[str]                       # column labels, newest first: '2026 so far', '2025-26', '2021-26'
    legs: list[dict]                         # per leg: label, games (past games found at all) and cells = [(hits, games)] per window
    together: list[tuple[int, int]]          # per window: (games in which every leg hit, games all the players were in)
    independent: list[float | None]          # per window: the product of the legs' own hit rates over those same games (None with no games)
    last_given_rest: list[tuple[int, int]]   # per window: (games every leg hit, games every leg but the last hit): how often the last leg hit when the others did
    games: pd.DataFrame                      # the games all of them were in, newest first: each leg's number with a tick or a cross, then whether every leg hit


def load_tables() -> Tables:
    from matchup_model.opp import data as D
    try:
        snaps = D.snaps()
    except Exception:  # noqa: BLE001 -- without snap counts only a player with a stat line counts as having played
        snaps = None
    return Tables(D.player_weeks(), snaps, D.schedule())


def wilson(k: int, n: int, z: float = 1.645) -> tuple[float, float]:
    """Wilson score interval for k hits in n games (z = 1.645 gives a 90% range): the rates the record is consistent with."""
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, float(c - h)), min(1.0, float(c + h))


def _team_games(schedule: pd.DataFrame, season: int, week: int, first: int) -> pd.DataFrame:
    """One row per team per scored game from `first` up to (not including) this one: game_id, season, week, team, opp, pts, opp_pts, margin, total."""
    s = schedule[schedule["home_score"].notna() & schedule["away_score"].notna() & (schedule["season"] >= first)
                 & ((schedule["season"] < season) | ((schedule["season"] == season) & (schedule["week"] < week)))]
    cols = ["game_id", "season", "week", "team", "opp", "pts", "opp_pts"]
    home = s.rename(columns={"home": "team", "away": "opp", "home_score": "pts", "away_score": "opp_pts"})[cols]
    away = s.rename(columns={"away": "team", "home": "opp", "away_score": "pts", "home_score": "opp_pts"})[cols]
    tg = pd.concat([home, away], ignore_index=True)
    return tg.assign(margin=tg["pts"] - tg["opp_pts"], total=tg["pts"] + tg["opp_pts"])


def _player_games(t: Tables, tg: pd.DataFrame, nk: str) -> pd.DataFrame:
    """Every past game in `tg` the player played: game_id, season, week, team, opp and his line under the simulator's stat names (zeros for a game he played without a stat line)."""
    pw = t.player_weeks
    rows = pw.loc[pw["name_key"] == nk, ["season", "week", "team", *ACTUAL_COLS]].rename(columns=ACTUAL_COLS)
    if t.snaps is not None:
        sn = t.snaps
        on = sn.loc[(sn["name_key"] == nk) & (sn["offense_snaps"] > 0), ["season", "week", "team"]].drop_duplicates()
        extra = on.merge(rows[["season", "week", "team"]], on=["season", "week", "team"], how="left", indicator=True)
        extra = extra[extra["_merge"] == "left_only"].drop(columns="_merge")
        rows = pd.concat([rows, extra], ignore_index=True).fillna({c: 0.0 for c in STATS})
    rows = rows.assign(any_td=rows["rush_td"] + rows["rec_td"])
    return rows.merge(tg[["game_id", "season", "week", "team", "opp"]], on=["season", "week", "team"]).drop_duplicates("game_id")


def _leg_frame(leg: S.Leg, tg: pd.DataFrame, t: Tables, teams: tuple[str, str]) -> pd.DataFrame:
    """One row per past game the leg can be judged on: game_id, season, week, team, opp, value (the stat, the team's margin or points, the game total) and hit."""
    if leg.market in S.PLAYER_MARKETS:
        g = _player_games(t, tg, normalize_name(leg.player or ""))
        x = g[S.PLAYER_MARKETS[leg.market]].astype(float)
    else:
        g = tg[tg["team"] == leg.team] if leg.team else tg[tg["team"].isin(teams)].drop_duplicates("game_id")      # a game total belongs to both teams
        x = g[TEAM_VALUE[leg.market]].astype(float)
    return g[["game_id", "season", "week", "team", "opp"]].assign(value=x, hit=np.asarray(S.on_side(leg, x), dtype=bool))


def _windows(season: int, first: int) -> list[tuple[str, int]]:
    """(label, first season in the window): this season so far, the last two seasons, everything since the data starts; a window the one before it already covers is left out."""
    yy = f"{season % 100:02d}"
    out = [(f"{season} so far", season)]
    for label, start in ((f"{season - 1}–{yy}", max(season - 1, first)), (f"{first}–{yy}", first)):
        if start < out[-1][1]:
            out.append((label, start))
    return out


def _count(f: pd.DataFrame, start: int) -> tuple[int, int]:
    m = f["season"] >= start
    return int(f.loc[m, "hit"].sum()), int(m.sum())


def past_hits(legs: list[S.Leg], season: int, week: int, teams: tuple[str, str], tables: Tables | None = None) -> Past:
    """The record of `legs` in the games before `season` week `week`; `teams` = (away, home) of that game, the two teams a game-total leg belongs to."""
    if not legs:
        raise ValueError("no legs to look up")
    t = tables if tables is not None else load_tables()
    first = int(t.player_weeks["season"].min())
    tg = _team_games(t.schedule, season, week, first)
    frames = [_leg_frame(l, tg, t, teams) for l in legs]
    wins = _windows(season, first)

    joined: pd.DataFrame | None = None
    for i, f in enumerate(frames):
        part = f[["game_id", "season", "week", "value", "hit"]].rename(columns={"value": f"v{i}", "hit": f"h{i}"})
        joined = part if joined is None else joined.merge(part[["game_id", f"v{i}", f"h{i}"]], on="game_id")
    hits = joined[[f"h{i}" for i in range(len(frames))]].astype(bool)
    every = hits.all(axis=1)
    rest = hits.iloc[:, :-1].all(axis=1)                                          # every leg but the last (all True for a single leg)

    together, independent, last_given_rest = [], [], []
    for _, start in wins:
        m = joined["season"] >= start
        together.append((int(every[m].sum()), int(m.sum())))
        independent.append(float(np.prod(hits[m].mean().to_numpy())) if m.any() else None)
        last_given_rest.append((int(every[m].sum()), int(rest[m].sum())))

    sched = t.schedule.set_index("game_id")
    newest = joined.sort_values(["season", "week"], ascending=False)
    table = pd.DataFrame({"Season": newest["season"].astype(int), "Wk": newest["week"].astype(int), "Game": [f"{sched.at[g, 'away']} @ {sched.at[g, 'home']}" for g in newest["game_id"]]})
    for i, leg in enumerate(legs):
        fmt = "{:+g}" if leg.market in ("spread", "moneyline") else "{:g}"
        table[leg.label()] = [f"{fmt.format(v)} {'✓' if h else '✗'}" for v, h in zip(newest[f"v{i}"], newest[f"h{i}"])]
    table["All legs"] = ["✓" if e else "✗" for e in every.loc[newest.index]]
    return Past(first=first, windows=[w for w, _ in wins],
                legs=[{"label": l.label(), "games": len(f), "cells": [_count(f, s) for _, s in wins]} for l, f in zip(legs, frames)],
                together=together, independent=independent, last_given_rest=last_given_rest, games=table.reset_index(drop=True))
