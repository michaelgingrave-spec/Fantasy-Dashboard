"""The weekly job in one command: box scores -> the finished week's prop lines (bought, logged, graded) -> the newest FantasyPoints projections -> the forward-test reports -> commit + push.

    python -m dfs.weekly                    # DRY RUN for the week just played: shows what it would do and what it would cost. Nothing is bought, written or pushed.
    python -m dfs.weekly --execute          # do it. Credits are spent only here, and never below --floor (default 2,000).
    python -m dfs.weekly --execute --push   # ... and commit + push exactly the files it touched (nothing else in the working tree)
    weekly                                  # weekly.cmd in the repo root = python -m dfs.weekly --execute --push (the whole job; extra options are passed on)
    python -m dfs.weekly 7 ...              # a specific week instead of the one just played

    --floor=N               never let the Odds API balance fall below N credits (default 2,000)
    --fast                  skip the two slow reports (the version exam and the forward-test refresh, about 8 minutes)
    --projections=PATH      install this FantasyPoints export instead of the newest one in Downloads
    --no-lines / --no-projections   do only the other part

What each part does:
  box scores      nflverse player/team/snap/injury files refreshed (grading and the projections need the finished week).
  lines           the week's games are listed (1 credit), then each finished game's DraftKings + FanDuel prop lines are bought as they stood 5 minutes before
                  kickoff (5 markets x 10 credits = 50 a game). Payloads already on disk are never bought twice. They go through the screen's own edges_from_raw and are
                  appended to data/dfs/props/line_history.csv (only (player, market) lines not already logged for that week, so a live pull is never overwritten), then
                  every ungraded line is graded against the box scores. The rows carry the CURRENT model_ver: they are today's leans on closing lines, not what the app
                  showed live. A summary of the week prints at the end.
  projections     the newest projections.weekly*.csv in Downloads (the FantasyPoints "Download CSV" button) is checked (week, rows, teams, names, size against the file
                  it replaces) and installed as data/dfs/projections.week{N}.csv. The download itself needs your signed-in browser, so it is the one manual step:
                  https://www.fantasypoints.com/nfl/projections/weekly -> Download CSV. FantasyPoints posts the next week's file on Tuesday night.
  reports         matchup_model.opp.restate (v1..current on every graded line, with the "where the current version differs" table) and
                  matchup_study --forward (the registered forward tests), which read the lines graded above.
  push            git add of only: line_history.csv, the projections file, the two reports. Refuses if anything else is already staged.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

if __name__ == "__main__" and __package__ is None:  # allow `python dfs/weekly.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dfs import bets, props
from dfs import historical_lines as HL
from dfs.config import DATA, ODDS_API_KEY, ROOT
from dfs.projections import ProjectionError, load_weekly_projections

SEASON = 2026
DEFAULT_FLOOR = 2_000
PER_GAME = len(props.CORE_MARKETS) * HL.CREDITS_PER_MARKET_GAME          # credits to buy one game's lines
GAME_HOURS = 4.0                                                         # a game kicked off this long ago is treated as finished
DOWNLOADS = Path.home() / "Downloads"
EXPORT_GLOB = "projections.weekly*.csv"
EXPORT_DAYS = 14                                                         # an export older than this is not "the newest"
REPORTS = ["matchup_model/restate_2026_report.md", "matchup_model/matchup_study_report.md"]
EASTERN = ZoneInfo("America/New_York")
BOOT_REPS = 500


def say_(*a, **k):
    print(*a, **k, flush=True)


# ------------------------------------------------------------------------------------------------------------------------------------------------ credits and games
def balance(api_key: str = ODDS_API_KEY, get=requests.get) -> int | None:
    """The Odds API's /sports call is free and reports the remaining credits in a header."""
    try:
        r = get("https://api.the-odds-api.com/v4/sports", params={"apiKey": api_key}, timeout=20)
        return int(r.headers["x-requests-remaining"])
    except Exception:  # noqa: BLE001 -- no key, no network: the caller decides what that means
        return None


def raw_path(season: int, week: int, event_id: str, raw_dir: Path = HL.RAW_DIR) -> Path:
    return raw_dir / f"{season}_wk{week}_{event_id}_core.json"


def week_games(season: int, week: int, *, get=HL._get, raw_dir: Path = HL.RAW_DIR, fetch: bool = True) -> tuple[list[dict] | None, int]:
    """(games sorted by kickoff, credits spent). The listing is cached next to the game payloads; fetching it costs 1 credit, so with fetch=False a missing listing is None."""
    cache = raw_dir / f"{season}_wk{week}_events.json"
    if cache.exists():
        return sorted(json.loads(cache.read_text(encoding="utf-8")), key=lambda e: e["commence_time"]), 0
    if not fetch:
        return None, 0
    raw_dir.mkdir(parents=True, exist_ok=True)
    events, _, hdr = HL.week_events(season, week, get)
    cache.write_text(json.dumps(events), encoding="utf-8")
    return sorted(events, key=lambda e: e["commence_time"]), int(hdr.get("last") or 0)


def finished(events: list[dict], now: datetime | None = None) -> list[dict]:
    now = now or datetime.now(timezone.utc)
    cut = now - timedelta(hours=GAME_HOURS)
    return [e for e in events if datetime.strptime(e["commence_time"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc) <= cut]


def buy_lines(events: list[dict], season: int, week: int, *, floor: int, bal: int | None, get=HL._get, raw_dir: Path = HL.RAW_DIR, say=say_) -> dict:
    """Buy the closing lines of every game not already on disk, never letting the balance fall below `floor`. Returns {'spent', 'balance', 'bought', 'cached', 'failed', 'stopped'}."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = {"spent": 0, "balance": bal, "bought": 0, "cached": 0, "failed": [], "stopped": False}
    for e in events:
        f = raw_path(season, week, e["id"], raw_dir)
        name = f"{e.get('away_team')} @ {e.get('home_team')}"
        if f.exists():
            out["cached"] += 1
            continue
        if out["balance"] is not None and out["balance"] - PER_GAME < floor:
            say(f"  balance {out['balance']:,} would fall below the floor {floor:,}: stopping before {name}")
            out["stopped"] = True
            break
        try:
            payload, hdr = get(f"/events/{e['id']}/odds", date=HL.closing_snapshot(e["commence_time"]), bookmakers=props.BOOKS, oddsFormat="american",
                               markets=",".join(props.CORE_MARKETS))
        except requests.HTTPError as exc:
            say(f"  ! {name}: {exc}")
            out["failed"].append(name)
            continue
        f.write_text(json.dumps(payload), encoding="utf-8")
        out["spent"] += int(hdr.get("last") or 0)
        out["balance"] = int(hdr["remaining"]) if hdr.get("remaining") is not None else out["balance"]
        out["bought"] += 1
        say(f"  bought {name}: balance {out['balance']}")
    return out


# ------------------------------------------------------------------------------------------------------------------------------------------------ logging and grading
def label_for(commence: str, week: int) -> str:
    """'Week 6 Sunday' etc. from the kickoff time (US Eastern), the event label the earlier backfills used."""
    local = datetime.strptime(commence, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).astimezone(EASTERN)
    return f"Week {week} {local.strftime('%A')}"


def log_lines(events: list[dict], season: int, week: int, *, edges_fn=None, raw_dir: Path = HL.RAW_DIR, relog: bool = False, say=say_) -> dict:
    """Run each bought game through the screen's own edges_from_raw and append the lines not already logged for this week to line_history.csv
    (a live pull's lines are never overwritten unless relog=True). Returns {'rows', 'games', 'missing'}."""
    edges_fn = edges_fn or props.edges_from_raw
    have = bets.load_line_history()
    if len(have) and not relog:
        have = have[(have.season == season) & (have.week == week)]
        taken = set(zip(have.player, have.market))
    else:
        taken = set()
    by_label, missing, games = {}, [], 0
    for e in events:
        f = raw_path(season, week, e["id"], raw_dir)
        if not f.exists():
            missing.append(f"{e.get('away_team')} @ {e.get('home_team')}")
            continue
        raw = json.loads(f.read_text(encoding="utf-8")).get("data")
        if not raw:
            missing.append(f"{e.get('away_team')} @ {e.get('home_team')} (empty)")
            continue
        edges = edges_fn(raw, week)
        if edges is None or edges.empty:
            continue
        fresh = edges[[(p, m) not in taken for p, m in zip(edges.player, edges.market)]]
        games += 1
        if len(fresh):
            by_label.setdefault(label_for(e["commence_time"], week), []).append(fresh)
    rows = 0
    for label, frames in by_label.items():
        rows += props.snapshot_lines(pd.concat(frames, ignore_index=True), season, week, label)
    say(f"  logged {rows} new lines from {games} games" + (f"; not available: {missing}" if missing else ""))
    return {"rows": rows, "games": games, "missing": missing}


# ------------------------------------------------------------------------------------------------------------------------------------------------ the week's summary
def _roi110(d: pd.DataFrame) -> float:
    p = (d.result == "win").mean()
    return 100 * (p * 100 / 110 - (1 - p))


def _boot(df: pd.DataFrame, fn, reps: int = BOOT_REPS, seed: int = 7) -> tuple[float, float]:
    """Player-clustered 95% interval of fn(df)."""
    groups = [g for _, g in df.groupby("player")]
    if len(groups) < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals = [fn(pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])) for _ in range(reps)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _placed_roi(d: pd.DataFrame) -> float:
    """ROI per unit at the number and price the screen placed each lean at (`bet price`), where one is stored."""
    x = d[d["bet price"].notna()] if "bet price" in d else d.iloc[:0]
    if x.empty:
        return float("nan")
    pay = [(-1.0 if r == "loss" else (p / 100 if p > 0 else 100 / abs(p))) for p, r in zip(x["bet price"], x.result)]
    return 100 * float(np.mean(pay))


def week_report(lh: pd.DataFrame, season: int, week: int, reps: int = BOOT_REPS) -> list[str]:
    """Plain-text results of one week's graded lines: by side, the screen's default view, by market, by tier, and the weeks so far."""
    w = lh[(lh.season == season) & (lh.week == week)] if len(lh) else lh
    if w.empty:
        return [f"week {week}: no lines logged"]
    g = w[w.result.isin(["win", "loss"])]
    L = [f"week {week} ({season}): {len(w)} lines logged, {len(g)} graded win/loss, {int((w.result == 'push').sum())} push, {int((w.result == 'dnp').sum())} did not play, "
         f"{int((~w.result.isin(['win', 'loss', 'push', 'dnp'])).sum())} ungraded; model version(s) {sorted(w.model_ver.dropna().unique().tolist()) if 'model_ver' in w else 'n/a'}", ""]
    if g.empty:
        return L + ["nothing graded yet"]
    default = bets.recommended_rows(w)
    default = default[default.result.isin(["win", "loss"]) & (default.lean == "UNDER")]
    rows = [("all leans", g), ("OVER leans", g[g.lean == "OVER"]), ("UNDER leans", g[g.lean == "UNDER"]), ("screen default (Recommended, UNDER)", default)]
    L += [f"{'':38s}{'bets':>5s}{'win':>8s}{'ROI@-110':>10s}  {'95% interval':>17s}{'at the placed price':>22s}"]
    for label, d in rows:
        if d.empty:
            L.append(f"{label:38s}{0:>5d}")
            continue
        lo, hi = _boot(d, _roi110, reps)
        placed = f"{_placed_roi(d):+.1f}%" if d.lean.eq("UNDER").all() else ""
        L.append(f"{label:38s}{len(d):>5d}{100 * (d.result == 'win').mean():>7.1f}%{_roi110(d):>+9.1f}%  [{lo:+6.1f}, {hi:+6.1f}]{placed:>22s}")
    o, u = g[g.lean == "OVER"], g[g.lean == "UNDER"]
    if len(o) and len(u):
        L.append(f"UNDER minus OVER win rate: {100 * ((u.result == 'win').mean() - (o.result == 'win').mean()):+.1f} points (the registered side test, weeks 6-8)")
    L += ["", "by market and side (bets, win%):"]
    for mk, d in g.groupby("market"):
        L.append("  " + f"{mk:11s} " + "   ".join(f"{side} {len(x):>3d} {100 * (x.result == 'win').mean():5.1f}%" for side, x in d.groupby("lean")))
    tiers = bets.line_history_buckets(w)
    if len(tiers):
        L += ["", "by tier, all lines (break-even: lean 52.4 / solid 53.0 / strong 53.7 / high 55.4):"]
        L += ["  " + "  ".join(f"{c}={v}" for c, v in r.items()) for r in tiers.to_dict("records")]
    since = lh[(lh.season == season) & lh.result.isin(["win", "loss"])]
    if since.week.nunique() > 1:
        L += ["", "the weeks so far (all leans / UNDER leans / OVER leans win%):"]
        for wk, d in since.groupby("week"):
            L.append(f"  week {int(wk)}: {len(d):>4d} bets  {100 * (d.result == 'win').mean():5.1f}%  /  {100 * (d[d.lean == 'UNDER'].result == 'win').mean():5.1f}%  /  {100 * (d[d.lean == 'OVER'].result == 'win').mean():5.1f}%")
    L += ["", "one week is about 500 correlated lines: a win rate is good to about +/-2.3 points, and the ROI interval above is the honest size of that."]
    return L


# ------------------------------------------------------------------------------------------------------------------------------------------------ FantasyPoints projections
def newest_export(downloads: Path = DOWNLOADS, now: float | None = None, days: float = EXPORT_DAYS) -> Path | None:
    """The newest projections.weekly*.csv in Downloads that is no more than `days` old."""
    import time
    now = time.time() if now is None else now
    c = [p for p in Path(downloads).glob(EXPORT_GLOB) if now - p.stat().st_mtime <= days * 86400]
    return max(c, key=lambda p: p.stat().st_mtime) if c else None


def export_week(path: Path) -> int | None:
    """The single WEEK value inside a FantasyPoints export, or None."""
    try:
        raw = pd.read_csv(path, header=1, dtype=str, keep_default_na=False)
        weeks = {v.strip() for v in raw["WEEK"] if v.strip()}
        return int(next(iter(weeks))) if len(weeks) == 1 else None
    except Exception:  # noqa: BLE001
        return None


def check_export(path: Path, week: int, current: Path | None = None) -> dict:
    """Is this export fit to install as week `week`'s file? {'ok', 'problems', 'rows', 'teams', 'diff'}; diff compares with `current` (the file it would replace)."""
    problems: list[str] = []
    try:
        d = load_weekly_projections(week, path)
    except ProjectionError as e:
        return {"ok": False, "problems": [str(e)], "rows": 0, "teams": 0, "diff": None}
    wk = export_week(path)
    if wk != week:
        problems.append(f"the WEEK column says {wk}, expected {week}")
    raw = pd.read_csv(path, header=1, dtype=str, keep_default_na=False)
    if "FPTS" in raw and int(pd.to_numeric(raw["FPTS"], errors="coerce").isna().sum() - (raw["FPTS"].str.strip() == "").sum()) > 0:
        problems.append("some FPTS values are not numbers")
    diff, n = None, len(d)
    if current is not None and Path(current).exists():
        cur = load_weekly_projections(week, current)
        if not (0.85 * len(cur) <= n <= 1.15 * len(cur)):
            problems.append(f"{n} rows is far from the {len(cur)} in the file it replaces")
        m = cur[["key", "proj"]].merge(d[["key", "proj"]], on="key", how="outer", suffixes=("_old", "_new"), indicator=True)
        both = m[m._merge == "both"]
        change = (both.proj_new - both.proj_old).abs()
        diff = {"added": int((m._merge == "right_only").sum()), "dropped": int((m._merge == "left_only").sum()), "moved_1": int((change >= 1).sum()),
                "moved_3": int((change >= 3).sum()), "mean_change": float((both.proj_new - both.proj_old).mean()) if len(both) else 0.0}
    elif not 250 <= n <= 800:
        problems.append(f"{n} rows is outside the usual 250-800")
    if d.name.duplicated().any():
        problems.append(f"duplicate names: {d.name[d.name.duplicated()].head(3).tolist()}")
    if d.team.nunique() < 30:
        problems.append(f"only {d.team.nunique()} teams")
    return {"ok": not problems, "problems": problems, "rows": n, "teams": int(d.team.nunique()), "diff": diff}


def install_projections(src: Path | None = None, week: int | None = None, *, downloads: Path = DOWNLOADS, data_dir: Path = DATA, install: bool = True, say=say_) -> dict:
    """Install the newest FantasyPoints export (or `src`) as data/dfs/projections.week{N}.csv, N from the file's own WEEK column unless given.
    Returns {'status': 'installed' | 'dry run' | 'already installed' | 'refused' | 'none found', 'week', 'path', 'check'}."""
    src = Path(src) if src else newest_export(downloads)
    if src is None:
        return {"status": "none found", "week": week, "path": None, "check": None}
    week = week or export_week(src)
    if week is None:
        return {"status": "refused", "week": None, "path": src, "check": {"ok": False, "problems": ["cannot read a single WEEK value from the file"]}}
    dest = Path(data_dir) / f"projections.week{week}.csv"
    if dest.exists() and dest.read_bytes() == src.read_bytes():
        return {"status": "already installed", "week": week, "path": dest, "check": None}
    chk = check_export(src, week, dest if dest.exists() else None)
    if not chk["ok"]:
        return {"status": "refused", "week": week, "path": src, "check": chk}
    if not install:
        return {"status": "dry run", "week": week, "path": src, "check": chk, "src": src, "dest": dest}
    shutil.copyfile(src, dest)
    return {"status": "installed", "week": week, "path": dest, "check": chk, "src": src, "dest": dest}


def _describe_install(res: dict) -> str:
    st, chk = res["status"], res.get("check") or {}
    if st == "none found":
        return ("no projections.weekly*.csv from the last 14 days in Downloads. Open https://www.fantasypoints.com/nfl/projections/weekly (signed in), click Download CSV, "
                "and run this again (FantasyPoints posts the next week's file on Tuesday night)")
    if st == "already installed":
        return f"week {res['week']}: the newest export is already installed ({res['path'].name})"
    if st == "refused":
        return f"REFUSED: {'; '.join(chk.get('problems', []))}"
    d = chk.get("diff")
    extra = (f"; against the file it replaces {d['moved_1']} players moved >= 1 pt, {d['moved_3']} >= 3, {d['added']} added, {d['dropped']} dropped, mean change {d['mean_change']:+.2f}"
             if d else " (a new week)")
    return f"{'would install' if st == 'dry run' else 'installed'} {res['src'].name} as {res['dest'].name} (week {res['week']}): {chk['rows']} rows, {chk['teams']} teams{extra}"


# ------------------------------------------------------------------------------------------------------------------------------------------------ reports, git
def run_reports(fast: bool, root: Path = ROOT, runner=subprocess.run, say=say_) -> list[str]:
    """Refresh the version exam and the registered forward tests (they read the graded lines). Returns the report files written."""
    if fast:
        say("  --fast: skipping the version exam and the forward-test refresh")
        return []
    done = []
    for label, args, out in (("version exam", ["-m", "matchup_model.opp.restate"], REPORTS[0]),
                             ("forward tests", ["-m", "matchup_model.opp.matchup_study", "--forward"], REPORTS[1])):
        say(f"  {label} ...")
        r = runner([sys.executable, *args], cwd=root, capture_output=True, text=True)
        if r.returncode != 0:
            say(f"  ! {label} failed:\n{(r.stderr or r.stdout)[-600:]}")
        else:
            done.append(out)
    return done


def commit_and_push(paths: list[str], message: str, *, root: Path = ROOT, runner=subprocess.run) -> str:
    """git add of exactly `paths`, commit, push. Refuses if the index already holds something else."""
    staged = runner(["git", "diff", "--cached", "--name-only"], cwd=root, capture_output=True, text=True, check=True).stdout.split()
    if staged:
        raise RuntimeError(f"files are already staged ({', '.join(staged[:3])}...): commit or unstage them first")
    existing = [p for p in paths if (Path(root) / p).exists()]
    runner(["git", "add", "--", *existing], cwd=root, check=True)
    if runner(["git", "diff", "--cached", "--quiet"], cwd=root).returncode == 0:
        return "nothing to commit"
    runner(["git", "commit", "-m", message], cwd=root, check=True)
    runner(["git", "push"], cwd=root, check=True)
    return "committed + pushed (Streamlit Cloud redeploys in about 2 minutes)"


# ------------------------------------------------------------------------------------------------------------------------------------------------ the job
def run_week(week: int, *, execute: bool = False, push: bool = False, floor: int = DEFAULT_FLOOR, fast: bool = False, projections: Path | None = None,
        do_lines: bool = True, do_projections: bool = True, season: int = SEASON, get=None, balance_fn=None, refresh_fn=None, edges_fn=None, grade_fn=None,
        reports_fn=None, git_fn=None, now: datetime | None = None, downloads: Path = DOWNLOADS, data_dir: Path = DATA, raw_dir: Path = HL.RAW_DIR, say=say_) -> dict:
    """The whole job for `week` (the one just played). Without execute nothing is bought, written or pushed. Returns what happened, for the caller and the tests."""
    get = get or HL._get
    balance_fn = balance_fn or balance
    grade_fn = grade_fn or bets.grade_line_history
    reports_fn = reports_fn or run_reports
    git_fn = git_fn or commit_and_push
    res: dict = {"week": week, "execute": execute, "touched": []}
    say(f"Week {week} ({season}) -- " + ("EXECUTE" if execute else "DRY RUN: nothing is bought, written or pushed (add --execute to do it)"))

    if do_lines:
        say("\n[box scores]")
        if execute:
            if refresh_fn is None:
                from matchup_model.opp import data as D
                D.refresh(force=True)
                D.clear_caches()
            else:
                refresh_fn()
            say("  refreshed")
        else:
            say("  would refresh the nflverse files")

        say("\n[lines]")
        try:
            events, spent = week_games(season, week, get=get, raw_dir=raw_dir, fetch=execute)
        except requests.HTTPError as exc:
            raise SystemExit(f"the Odds API refused the game listing for week {week}: {exc}") from exc
        bal = balance_fn()
        if events is None:                                   # dry run, no listing on disk: the free nflverse schedule says how many games there are
            try:
                from matchup_model.opp import data as D
                n_games = int(len(D.games().query("season == @season and week == @week")) // 2)
            except Exception:  # noqa: BLE001
                n_games = 0
            say(f"  about {n_games} games; the listing costs 1 credit, then {PER_GAME} per game = about {n_games * PER_GAME + 1:,} credits"
                + (f"; balance {bal:,} -> {bal - n_games * PER_GAME - 1:,}; floor {floor:,}" if bal is not None else ""))
        else:
            done = finished(events, now)
            todo = [e for e in done if not raw_path(season, week, e["id"], raw_dir).exists()]
            say(f"  {len(events)} games listed, {len(done)} finished, {len(done) - len(todo)} already on disk, {len(todo)} to buy x {PER_GAME} = {len(todo) * PER_GAME:,} credits"
                + (f"; balance {bal:,} -> {bal - len(todo) * PER_GAME:,}; floor {floor:,}" if bal is not None else "") + (f" (the listing cost {spent})" if spent else ""))
            if len(done) < len(events):
                say(f"  {len(events) - len(done)} game(s) not finished yet: left for the next run")
            if execute:
                if todo and not ODDS_API_KEY and get is HL._get:
                    raise SystemExit("ODDS_API_KEY is not set (put it in .env): cannot buy the lines")
                if todo and bal is None and get is HL._get:
                    say("  could not read the balance: not buying (the floor cannot be checked)")
                else:
                    res["buy"] = buy_lines(done, season, week, floor=floor, bal=bal, get=get, raw_dir=raw_dir, say=say)
                    res["buy"]["spent"] += spent
                    res["log"] = log_lines(done, season, week, edges_fn=edges_fn, raw_dir=raw_dir, say=say)
                    df, n = grade_fn()
                    say(f"  graded {n} lines")
                    res["touched"].append("data/dfs/props/line_history.csv")
                    lh = bets.load_line_history()
                    say("\n[week " + str(week) + " results]")
                    for line in week_report(lh, season, week):
                        say("  " + line if line else "")

    if do_projections:
        say("\n[FantasyPoints projections]")
        r = install_projections(projections, downloads=downloads, data_dir=data_dir, install=execute, say=say)
        say("  " + _describe_install(r))
        res["projections"] = r
        if r["status"] == "installed":
            res["touched"].append(f"data/dfs/projections.week{r['week']}.csv")

    if do_lines:
        say("\n[reports]")
        if execute:
            res["touched"] += reports_fn(fast)
        else:
            say("  would refresh the version exam and the forward tests" + (" (skipped by --fast)" if fast else " (about 8 minutes)"))

    if push:
        say("\n[push]")
        if not execute:
            say(f"  would commit and push exactly: line_history.csv, the projections file, {len(REPORTS)} reports")
        else:
            wk = res["projections"].get("week") if res.get("projections") and res["projections"]["status"] == "installed" else None
            msg = f"Backfill and grade week {week} props" + (f"; FantasyPoints week {wk} projections" if wk else "")
            say("  " + git_fn(res["touched"] or ["data/dfs/props/line_history.csv"], msg))
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dfs.weekly", description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("week", nargs="?", type=int, help="the week to grade (default: the week just played)")
    ap.add_argument("--execute", action="store_true", help="actually buy, write and install (default is a dry run)")
    ap.add_argument("--push", action="store_true", help="commit + push the files this run touched")
    ap.add_argument("--floor", type=int, default=DEFAULT_FLOOR, help="never let the Odds API balance fall below this many credits")
    ap.add_argument("--fast", action="store_true", help="skip the two slow reports")
    ap.add_argument("--projections", type=Path, help="install this FantasyPoints export instead of the newest in Downloads")
    ap.add_argument("--no-lines", action="store_true")
    ap.add_argument("--no-projections", action="store_true")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from matchup_model.weekly_pull import last_completed_week
    week = a.week or last_completed_week()
    if week < 1:
        print("the season has not started: no finished week to grade")
        return 1
    run_week(week, execute=a.execute, push=a.push, floor=a.floor, fast=a.fast, projections=a.projections, do_lines=not a.no_lines, do_projections=not a.no_projections)
    return 0


if __name__ == "__main__":
    sys.exit(main())
