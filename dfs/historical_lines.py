"""Closing prop lines for a whole past season, from the Odds API's historical snapshots.

Why: the 2026 tests of "is this signal already in the book's line?" have a few hundred lines. A full past season of DraftKings and
FanDuel closing lines (a few thousand per market) settles it, because the signal and the outcome can then be set against the line
the market actually closed on.

    python -m dfs.historical_lines 2025 5 18                         # dry run: games per week and the credit estimate (event lists ~1 credit each)
    python -m dfs.historical_lines 2025 5 18 --execute --floor=8335  # pull; never lets the balance fall below --floor

Cost: 10 credits per market per game (DraftKings + FanDuel count as one region), so the three markets below are 30 per game. Every paid
payload (event lists too) is kept in data/dfs/.cache/historical_raw/ (git-ignored) and nothing already on disk is bought twice, so a run can
be resumed and a parsing bug never costs credits twice. Writes data/dfs/props/historical_lines_{season}.csv, one row per
(week, game, player, market, book, point) with both prices.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from dfs.config import DATA, ODDS_API_KEY
from dfs.props import _NICE, BOOKS, MARKETS

HBASE = "https://api.the-odds-api.com/v4/historical/sports/americanfootball_nfl"
MARKET_KEYS = ["player_reception_yds", "player_receptions", "player_rush_yds"]
CREDITS_PER_MARKET_GAME = 10
WINDOW_DAYS = 6.4                 # a week's slate runs Thursday night to Monday night
CLOSE_MINUTES = 5                 # snapshot this long before kickoff: the book's last word before it suspends the prop
RAW_DIR = DATA / ".cache" / "historical_raw"
OPENERS = {2023: (9, 7), 2024: (9, 5), 2025: (9, 4), 2026: (9, 10)}      # the Thursday of week 1


def out_path(season: int) -> Path:
    return DATA / "props" / f"historical_lines_{season}.csv"


def week_start(season: int, week: int) -> datetime:
    """Wednesday 20:00 UTC before that week's Thursday game: the week's slate is listed, none of it has kicked off."""
    month, day = OPENERS[season]
    thursday = datetime(season, month, day, tzinfo=timezone.utc) + timedelta(weeks=week - 1)
    return (thursday - timedelta(days=1)).replace(hour=20, minute=0, second=0, microsecond=0)


def iso(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def closing_snapshot(commence_time: str) -> str:
    t = datetime.strptime(commence_time, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return iso(t - timedelta(minutes=CLOSE_MINUTES))


def parse_event_books(raw: dict) -> list[dict]:
    """One row per (player, market, book, point) from one event's odds payload, with the Over and Under prices side by side."""
    acc: dict = {}
    for bk in raw.get("bookmakers", []):
        for mkt in bk.get("markets", []):
            key = mkt.get("key")
            if key not in MARKETS:
                continue
            for o in mkt.get("outcomes", []):
                if o.get("point") is None or o.get("name") not in ("Over", "Under"):
                    continue
                d = acc.setdefault((o.get("description", ""), key, bk["key"], float(o["point"])), {})
                d[o["name"].lower() + "_price"] = o.get("price")
    return [dict(player=p, market=_NICE[MARKETS[k]], book=b, point=pt, over_price=d.get("over_price"), under_price=d.get("under_price"))
            for (p, k, b, pt), d in acc.items()]


def _get(path: str, **params) -> tuple[dict, dict]:
    r = requests.get(f"{HBASE}{path}", params={"apiKey": ODDS_API_KEY, **params}, timeout=30)
    r.raise_for_status()
    return r.json(), {"remaining": r.headers.get("x-requests-remaining"), "last": r.headers.get("x-requests-last")}


def week_events(season: int, week: int, get=_get) -> tuple[list[dict], str, dict]:
    """The week's games as listed at the start of the week (commence_time inside the week's window)."""
    t0 = week_start(season, week)
    payload, hdr = get("/events", date=iso(t0))
    t1 = t0 + timedelta(days=WINDOW_DAYS)
    out = [e for e in payload.get("data", [])
           if t0 <= datetime.strptime(e["commence_time"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc) <= t1]
    return out, iso(t0), hdr


def pull(season: int, first: int, last: int, floor: int, execute: bool, get=_get, raw_dir: Path = RAW_DIR, say=print) -> dict:
    """Plan (always) and buy (only if `execute`) the closing odds of every game in weeks first..last, never letting the credit balance fall
    below `floor`. Event lists and game payloads already on disk are not bought again. Returns {'plan', 'spent', 'balance', 'rows'}."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    plan, spent, balance = [], 0, None
    for wk in range(first, last + 1):
        cache = raw_dir / f"{season}_wk{wk}_events.json"
        if cache.exists():
            events = json.loads(cache.read_text(encoding="utf-8"))
        else:
            events, snap, hdr = week_events(season, wk, get)
            cache.write_text(json.dumps(events), encoding="utf-8")
            spent += int(hdr["last"] or 0)
            balance = int(hdr["remaining"])
            say(f"week {wk}: {len(events)} games listed at {snap} (balance {hdr['remaining']})")
        plan.append((wk, events))
    games = sum(len(e) for _, e in plan)
    per_game = len(MARKET_KEYS) * CREDITS_PER_MARKET_GAME
    todo = sum(1 for wk, ev in plan for e in ev if not (raw_dir / f"{season}_wk{wk}_{e['id']}.json").exists())
    say(f"\n{games} games, {todo} not yet on disk x {per_game} credits = {todo * per_game:,} credits; floor {floor:,}"
        + (f", balance {balance:,} -> {balance - todo * per_game:,} after" if balance is not None else ""))
    if not execute:
        say("dry run: nothing bought. Add --execute to pull.")
        return {"plan": plan, "spent": spent, "balance": balance, "rows": pd.DataFrame()}
    if balance is not None and balance - todo * per_game < floor:
        say("the whole pull would take the balance below the floor; stopping before buying anything.")
        return {"plan": plan, "spent": spent, "balance": balance, "rows": pd.DataFrame()}
    for wk, events in plan:
        for e in events:
            f = raw_dir / f"{season}_wk{wk}_{e['id']}.json"
            if f.exists():
                continue
            if balance is not None and balance - per_game < floor:
                say("balance floor reached; stopping.")
                return {"plan": plan, "spent": spent, "balance": balance, "rows": _collect(season, plan, raw_dir)}
            try:
                payload, hdr = get(f"/events/{e['id']}/odds", date=closing_snapshot(e["commence_time"]), bookmakers=BOOKS, oddsFormat="american",
                                   markets=",".join(MARKET_KEYS))
            except requests.HTTPError as exc:
                say(f"  ! wk{wk} {e.get('away_team')} @ {e.get('home_team')}: {exc}")
                continue
            f.write_text(json.dumps(payload), encoding="utf-8")
            spent += int(hdr["last"] or 0)
            balance = int(hdr["remaining"])
            say(f"  wk{wk} {e.get('away_team')} @ {e.get('home_team')}: cost {hdr['last']}, balance {hdr['remaining']}", flush=True)
    return {"plan": plan, "spent": spent, "balance": balance, "rows": _collect(season, plan, raw_dir)}


def _collect(season: int, plan: list, raw_dir: Path) -> pd.DataFrame:
    rows = []
    for wk, events in plan:
        for e in events:
            f = raw_dir / f"{season}_wk{wk}_{e['id']}.json"
            if f.exists():
                payload = json.loads(f.read_text(encoding="utf-8"))
                rows += [dict(season=season, week=wk, event_id=e["id"], away=e.get("away_team"), home=e.get("home_team"), commence_time=e["commence_time"],
                              snapshot=payload.get("timestamp"), **r) for r in parse_event_books(payload.get("data") or {})]
    return pd.DataFrame(rows)


def load(season: int) -> pd.DataFrame:
    p = out_path(season)
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def main(argv: list[str]) -> int:
    season, first, last = int(argv[0]), int(argv[1]), int(argv[2])
    execute = "--execute" in argv
    floor = int(next((a.split("=")[1] for a in argv if a.startswith("--floor=")), -1))
    if execute and floor < 0:
        print("--execute needs an explicit --floor=<credit balance never to go below>.")
        return 2
    res = pull(season, first, last, floor, execute)
    if execute and len(res["rows"]):
        old = load(season)
        new = pd.concat([old, res["rows"]], ignore_index=True) if len(old) else res["rows"]
        new = new.drop_duplicates(["week", "event_id", "player", "market", "book", "point"], keep="last")
        out_path(season).parent.mkdir(parents=True, exist_ok=True)
        new.to_csv(out_path(season), index=False)
        print(f"\nspent {res['spent']:,} credits this run; balance {res['balance']}; {len(new):,} lines saved to {out_path(season)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
