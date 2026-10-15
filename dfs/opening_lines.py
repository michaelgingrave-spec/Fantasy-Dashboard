"""Early-week ("opening") prop lines, pulled from the Odds API's historical snapshots.

Why: every graded line in line_history.csv is a late line (closing or pull-time), the sharpest the market gets.
If our projections lead the market, the line should MOVE toward our number between early week and kickoff, and
betting early would beat the close. That needs the early line for the same player-markets.

    python -m dfs.opening_lines 2026 1 5            # dry run: events per week and the credit cost, nothing charged
                                                    # beyond the event lists (~1 credit each)
    python -m dfs.opening_lines 2026 1 5 --execute  # pull, with a hard credit guard

Cost: 10 credits per market per game (DraftKings + FanDuel count as one region). Writes
data/dfs/props/opening_lines.csv, one row per (season, week, player, market), consensus line = median of the book
points exactly as dfs.props.edges_from_raw computes it.
"""
from __future__ import annotations

import json
import statistics as _st
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from dfs.config import ODDS_API_KEY
from dfs.props import _NICE, BOOKS, DATA, MARKETS

OPENING_PATH = DATA / "props" / "opening_lines.csv"
_HBASE = "https://api.the-odds-api.com/v4/historical/sports/americanfootball_nfl"
MARKET_KEYS = ["player_reception_yds", "player_receptions"]     # the markets whose projections we found biased
CREDITS_PER_MARKET_GAME = 10
WINDOW_DAYS = 6.4          # a week's slate runs Thursday night to Monday night


def snapshot_time(season: int, week: int) -> str:
    """Wednesday 20:00 UTC (4pm ET) of that NFL week: after the books have posted Sunday props, well before
    they sharpen. Week 1 of 2026 opens Thursday Sep 10."""
    thursday = datetime(season, 9, 10, tzinfo=timezone.utc) + timedelta(weeks=week - 1)
    wed = thursday - timedelta(days=1)
    return wed.replace(hour=20, minute=0, second=0, microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_event_odds(raw: dict) -> list[dict]:
    """One row per (player, market) from one event's raw odds payload: consensus line (median of every book's
    points, as edges_from_raw does), number of books, and the best over / under price and book."""
    acc: dict = {}
    for bk in raw.get("bookmakers", []):
        for mkt in bk.get("markets", []):
            mk = mkt["key"]
            if mk not in MARKETS:
                continue
            for o in mkt.get("outcomes", []):
                d = acc.setdefault((o.get("description", ""), mk),
                                   {"points": [], "over": [], "under": [], "books": set()})
                if o.get("point") is not None:
                    d["points"].append(float(o["point"]))
                d["books"].add(bk["key"])
                side = {"Over": "over", "Under": "under"}.get(o.get("name"))
                if side:
                    d[side].append((o["price"], bk["key"]))
    rows = []
    for (player, mk), d in acc.items():
        if not d["points"]:
            continue
        best_o = max(d["over"], key=lambda x: x[0]) if d["over"] else (None, "")
        best_u = max(d["under"], key=lambda x: x[0]) if d["under"] else (None, "")
        rows.append(dict(player=player, market=_NICE[MARKETS[mk]], line_open=float(_st.median(d["points"])),
                         n_books=len(d["books"]), over_price=best_o[0], over_book=best_o[1],
                         under_price=best_u[0], under_book=best_u[1]))
    return rows


def _get(path: str, **params) -> tuple[dict, dict]:
    r = requests.get(f"{_HBASE}{path}", params={"apiKey": ODDS_API_KEY, **params}, timeout=30)
    r.raise_for_status()
    return r.json(), {"remaining": r.headers.get("x-requests-remaining"), "last": r.headers.get("x-requests-last")}


def week_events(season: int, week: int) -> tuple[list[dict], str, dict]:
    """The week's games as listed at the snapshot (commence_time inside the week's window)."""
    snap = snapshot_time(season, week)
    payload, hdr = _get("/events", date=snap)
    t0 = datetime.strptime(snap, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    t1 = t0 + timedelta(days=WINDOW_DAYS)
    out = []
    for e in payload.get("data", []):
        ct = datetime.strptime(e["commence_time"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        if t0 <= ct <= t1:
            out.append(e)
    return out, snap, hdr


def main(argv: list[str]) -> int:
    season, first, last = int(argv[0]), int(argv[1]), int(argv[2])
    execute = "--execute" in argv
    budget = int(next((a.split("=")[1] for a in argv if a.startswith("--budget=")), 1700))
    floor = int(next((a.split("=")[1] for a in argv if a.startswith("--floor=")), 16200))   # never go below this balance
    plan = []
    for wk in range(first, last + 1):
        events, snap, hdr = week_events(season, wk)
        plan.append((wk, snap, events))
        print(f"week {wk}: snapshot {snap} -> {len(events)} games listed "
              f"(event-list cost {hdr['last']}, balance {hdr['remaining']})")
    games = sum(len(e) for _, _, e in plan)
    est = games * len(MARKET_KEYS) * CREDITS_PER_MARKET_GAME
    print(f"\n{games} games x {len(MARKET_KEYS)} markets x {CREDITS_PER_MARKET_GAME} credits = {est:,} credits")
    if not execute:
        print("dry run: nothing pulled. Add --execute to pull.")
        return 0
    if est > budget:
        print(f"estimate {est:,} exceeds the budget of {budget:,}; stopping.")
        return 1
    raw_dir = DATA / ".cache" / "opening_raw"          # every paid payload is kept, so a parsing bug never costs credits twice
    raw_dir.mkdir(parents=True, exist_ok=True)
    spent, stop = 0, False
    for wk, snap, events in plan:
        rows = []
        for e in events:
            payload, hdr = _get(f"/events/{e['id']}/odds", date=snap, bookmakers=BOOKS, oddsFormat="american",
                                markets=",".join(MARKET_KEYS))
            spent += int(hdr["last"] or 0)
            (raw_dir / f"{season}_wk{wk}_{e['id']}.json").write_text(json.dumps(payload), encoding="utf-8")
            try:
                got = parse_event_odds(payload.get("data") or {})
            except Exception as exc:  # noqa: BLE001 -- keep going; the raw payload is on disk to reparse
                print(f"  ! parse failed for {e.get('away_team')} @ {e.get('home_team')}: {exc}")
                got = []
            rows += [dict(season=season, week=wk, event_id=e["id"], away=e.get("away_team"),
                          home=e.get("home_team"), commence_time=e["commence_time"], snapshot=snap, **r)
                     for r in got]
            print(f"  wk{wk} {e.get('away_team')} @ {e.get('home_team')}: cost {hdr['last']}, "
                  f"balance {hdr['remaining']}, props {len(got)}", flush=True)
            if int(hdr["remaining"] or 0) < floor:
                print("balance floor reached; stopping.")
                stop = True
                break
        _save(rows)                                    # after every week, so partial progress survives
        if stop:
            break
    print(f"\nspent {spent:,} credits; opening lines saved to {OPENING_PATH}")
    return 0


def _save(rows: list[dict]) -> None:
    new = pd.DataFrame(rows)
    if OPENING_PATH.exists():
        new = pd.concat([pd.read_csv(OPENING_PATH), new], ignore_index=True)
    new = new.drop_duplicates(["season", "week", "player", "market"], keep="last")
    OPENING_PATH.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(OPENING_PATH, index=False)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
