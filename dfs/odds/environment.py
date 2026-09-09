"""Turn Odds API events into a per-team game-environment table."""
from __future__ import annotations

import pandas as pd

from dfs.names import norm_team


def _dk_book(event: dict) -> dict | None:
    books = event.get("bookmakers") or []
    for b in books:
        if b.get("key") == "draftkings":
            return b
    return books[0] if books else None


def team_environment(events: list[dict] | None) -> pd.DataFrame:
    """Columns: team, opp, game_total, spread, implied_total, is_favorite, pace_tier.

    Empty frame if `events` is falsy.
    """
    if not events:
        return pd.DataFrame(
            columns=["team", "opp", "game_total", "spread", "implied_total", "is_favorite", "pace_tier"]
        )

    rows = []
    for ev in events:
        home = norm_team(ev.get("home_team", ""))
        away = norm_team(ev.get("away_team", ""))
        book = _dk_book(ev)
        if not book or not home or not away:
            continue
        total = None
        spread = {home: None, away: None}
        for mkt in book.get("markets", []):
            if mkt["key"] == "totals":
                for o in mkt["outcomes"]:
                    if o.get("name", "").lower() == "over":
                        total = o.get("point")
            elif mkt["key"] == "spreads":
                for o in mkt["outcomes"]:
                    t = norm_team(o.get("name", ""))
                    if t in spread:
                        spread[t] = o.get("point")
        for team, opp in ((home, away), (away, home)):
            sp = spread.get(team)
            imp = None
            if total is not None and sp is not None:
                imp = round(total / 2.0 - sp / 2.0, 2)
            rows.append(
                {
                    "team": team,
                    "opp": opp,
                    "game_total": total,
                    "spread": sp,
                    "implied_total": imp,
                    "is_favorite": (sp is not None and sp < 0),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    gt = df["game_total"].dropna()
    if len(gt) >= 4:
        df["pace_tier"] = pd.qcut(df["game_total"], 3, labels=["slow", "avg", "fast"], duplicates="drop")
    else:
        df["pace_tier"] = "avg"
    return df
