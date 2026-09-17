"""Recipe + bookkeeping for the weekly FantasyPoints Data Suite pull.

The CSV export is a client-side button, so the actual downloading has to be done by a
browser (the claude-in-chrome session in Chrome, logged into fantasypointsdata.com). This
module supplies the URL list and the file move / rename / verify / commit steps so the
browser part is "get to the right page + split, click Download CSV, run move_fresh".

IMPORTANT — learned the hard way, don't re-discover this: `tables()`'s ~10 weekly URLs
download fine from a direct navigate. `scheme_tables()`'s 13 URLs (all carry `splits=`)
mostly do NOT — the download click silently no-ops on most of them (confirmed: zero
network requests, zero console output, even with tracking armed before the click), while
the page renders completely normally and looks identical to a working one. The fix:
navigate to the BASE page (no `splits=` in the URL), set position/mode, then apply the
split through the UI itself — Open Filters (left sidebar) -> Split tab -> find the
checkbox (Personnel under OFFENSE, Coverage Scheme under DEFENSE, Alignment/Rush Concept
under RECEIVING/RUSHING) -> Apply Filters -> close the panel -> download. That reliably
works for all 13. Full step-by-step is in the `fantasypoints-weekly-pull` scheduled task's
SKILL.md (~/.claude/scheduled-tasks/fantasypoints-weekly-pull/SKILL.md) — read that before
doing a pull by hand, it has the exact click sequence and section names.

Also: the download-icon click itself sometimes needs 2-3 tries even at the exact right
spot — that's normal flakiness on this site, not a sign the approach is wrong. And
`find`-based ref clicks on "Download CSV" are less reliable than a precise coordinate
click read from a fresh screenshot; verify with move_fresh rather than trusting a ref
click succeeded.

CORRECTION, found on a later pull after the "fresh tab" theory below sent a long chase
in the wrong direction: the dominant cause of repeated download-click failures is that
**the download icon's pixel position moves** depending on the page's layout that load —
NOT a per-tab throttle. Specifically it sits at a standard spot (roughly y=212 in a
1568x765 screenshot) on a plain table, but shifts down ~70-120px (to roughly y=283, in a
shorter ~1568x726 frame) when the page shows an active-filter badge row ("MIB 7+",
"Split: Week", an over-cap row-count warning, etc.) above the table. Reusing a
remembered coordinate across different pages/filters means the click silently lands on
the wrong icon (often "Glossary" or empty space) and nothing downloads — that looked
exactly like random flakiness until confirmed via `getBoundingClientRect()` in
`javascript_tool` on the actual "Download CSV" button. **Take a fresh screenshot of the
specific page you're on and read the icon's real position before every click that isn't
immediately after a click that just worked on that exact page** — don't trust a
coordinate carried over from a different table/filter state, even one used successfully
minutes earlier.

(The "rotate to a fresh tab every 4-5 downloads" idea below is probably not a real fix —
it likely just happened to correlate with page layouts that matched the assumed
coordinate. Harmless to keep doing as a habit, but don't rely on it if downloads are
failing; check the actual icon position first.)

    from matchup_model.weekly_pull import tables, move_fresh, finalize
    for t in tables(2026, 3):
        # (browser) navigate t["url"], click Download CSV
        move_fresh(t["dest"])          # rename newest ~/Downloads csv -> data/dfs/matchup/<dest>
    finalize(2026, 3)                  # row-count report + git add/commit/push
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

from matchup_model.config import DATA, ROOT

BASE = "https://fantasypointsdata.com"
DOWNLOADS = Path.home() / "Downloads"


def tables(season: int, week: int) -> list[dict]:
    """All tables to pull for one completed NFL week. `dest` filenames match the ingest globs."""
    W = str(week)
    rec = f"seasons={season}&regWeeks={W}&positions=WR,TE&splits=week"
    rb = f"seasons={season}&regWeeks={W}&positions=RB&splits=week"
    qb = f"seasons={season}&regWeeks={W}&positions=QB&splits=week"
    tag = f"{season}wk{week}"
    t = [
        ("receiving/basic", rec, f"receiving-basic_{tag}_week.csv"),
        ("receiving/advanced", rec, f"receiving-advanced_{tag}_week.csv"),
        ("receiving/man-vs-zone", rec, f"receiving-manvszone_{tag}_week.csv"),
        ("rushing/basic", rb, f"rushing-basic_{tag}_week.csv"),
        ("rushing/advanced", rb, f"rushing-advanced_{tag}_week.csv"),
        ("passing/basic", qb, f"passing-basic_{tag}_week.csv"),
        ("passing/advanced", qb, f"passing-advanced_{tag}_week.csv"),
        ("passing/situation", qb, f"passing-situation_{tag}_week.csv"),
        ("team/coverage-matrix", f"seasons={season}&regWeeks={W}&mode=defense&splits=week",
         f"coverage-matrix_{tag}_week.csv"),
        # optional — needs downloads allow-listed; skip if it won't export
        ("receiving/separation-by-coverage", rec, f"receiving-sepbycoverage_{tag}_week.csv"),
    ]
    return [{"name": n, "url": f"{BASE}/{n}?{q}", "dest": d} for n, q, d in t]


# Full-season backfill (e.g. add the 2025 season). Receiving tables are capped at 1500
# rows for a whole season, so they get week-batched into wk1-6 / wk7-12 / wk13-18 the
# same way the committed 2022-24 files are; the rest fit in one file. Week filtering is
# URL-driven: &splitValues=week:1,2,3,4,5,6  (no button clicking).
_WEEK_BATCHES = {"wk1-6": "1,2,3,4,5,6", "wk7-12": "7,8,9,10,11,12", "wk13-18": "13,14,15,16,17,18"}


def backfill_tables(season: int) -> list[dict]:
    """Every table for a whole completed season. `dest` names match the ingest globs and
    the existing 2022-24 files. 15 downloads: 3 receiving tables x 3 week-batches + 6 singles."""
    rec = f"seasons={season}&positions=WR,TE&splits=week"
    rb = f"seasons={season}&positions=RB&splits=week"
    qb = f"seasons={season}&positions=QB&splits=week"
    out = []
    for page, pref in (("receiving/basic", "receiving-basic"),
                       ("receiving/advanced", "receiving-advanced"),
                       ("receiving/man-vs-zone", "receiving-manvszone")):
        for tag, weeks in _WEEK_BATCHES.items():
            out.append((page, f"{rec}&splitValues=week:{weeks}", f"{pref}_{season}{tag}_week.csv"))
    for page, pref, q in (("rushing/basic", "rushing-basic", rb),
                          ("rushing/advanced", "rushing-advanced", rb),
                          ("passing/basic", "passing-basic", qb),
                          ("passing/advanced", "passing-advanced", qb),
                          ("passing/situation", "passing-situation", qb),
                          ("team/coverage-matrix", "coverage-matrix",
                           f"seasons={season}&mode=defense&splits=week")):
        out.append((page, q, f"{pref}_{season}_week.csv"))
    return [{"name": n, "url": f"{BASE}/{n}?{q}", "dest": d} for n, q, d in out]


def scheme_tables(season: int) -> list[dict]:
    """Season-to-date scheme splits for the Matchup Machine (run concept / coverage /
    alignment). Re-pull weekly and OVERWRITE the `*_<season>.csv` files — the screen reads
    game counts from them to ramp 2026 weight over 2025. 8 downloads."""
    rb = f"seasons={season}&positions=RB&splits=rushConcept"
    rec = f"seasons={season}&positions=WR,TE&splits=coverageScheme"
    t = [
        ("rushing/advanced", rb, f"rushing-concept_player_{season}.csv"),
        ("rushing/advanced", f"seasons={season}&splits=rushConcept&mode=offense",
         f"rushing-concept_offense_{season}.csv"),
        ("rushing/advanced", f"seasons={season}&splits=rushConcept&mode=defense",
         f"rushing-concept_defense_{season}.csv"),
        ("receiving/advanced", f"seasons={season}&positions=WR&splits=coverageScheme",
         f"receiving-coverage_wr_{season}.csv"),
        ("receiving/advanced", f"seasons={season}&positions=TE&splits=coverageScheme",
         f"receiving-coverage_te_{season}.csv"),
        ("receiving/advanced", f"{rec}&mode=defense",
         f"receiving-coverage_defense_{season}.csv"),
        ("passing/advanced", f"seasons={season}&positions=QB&splits=coverageScheme&mode=offense",
         f"passing-coverage_offense_{season}.csv"),
        ("receiving/advanced", f"seasons={season}&positions=WR,TE&splits=alignmentPosition&mode=defense",
         f"receiving-alignment_defense_{season}.csv"),
        # personnel groupings (11 / 12 / 21) — split=personnel
        ("rushing/advanced", f"seasons={season}&positions=RB&splits=personnel",
         f"rushing-personnel_player_{season}.csv"),
        ("rushing/advanced", f"seasons={season}&splits=personnel&mode=defense",
         f"rushing-personnel_defense_{season}.csv"),
        ("receiving/advanced", f"seasons={season}&positions=WR&splits=personnel",
         f"receiving-personnel_wr_{season}.csv"),
        ("receiving/advanced", f"seasons={season}&positions=TE&splits=personnel",
         f"receiving-personnel_te_{season}.csv"),
        ("receiving/advanced", f"seasons={season}&positions=WR,TE&splits=personnel&mode=defense",
         f"receiving-personnel_defense_{season}.csv"),
    ]
    return [{"name": n, "url": f"{BASE}/{n}?{q}", "dest": d} for n, q, d in t]


def pending(dests: list[str]) -> list[str]:
    """Which of these dest filenames aren't in data/dfs/matchup/ yet (resume a partial run)."""
    return [d for d in dests if not (DATA / d).exists()]


def move_fresh(dest: str, timeout_s: int = 50) -> dict:
    """Poll ~/Downloads for the CSV that just landed and move it to data/dfs/matchup/<dest>.
    Returns {ok, rows, path} or {ok: False, reason}."""
    deadline = time.time() + timeout_s
    seen = None
    while time.time() < deadline:
        cands = [p for p in DOWNLOADS.glob("*.csv")
                 if time.time() - p.stat().st_mtime < 90 and not p.name.startswith("~")]
        if cands:
            seen = max(cands, key=lambda p: p.stat().st_mtime)
            # let a slow client-side export finish writing
            size1 = seen.stat().st_size
            time.sleep(2)
            if seen.exists() and seen.stat().st_size == size1 and size1 > 200:
                break
        time.sleep(3)
    if not seen or not seen.exists():
        return {"ok": False, "reason": "no fresh download appeared"}
    rows = max(0, sum(1 for _ in seen.open("r", encoding="utf-8", errors="ignore")) - 3)
    target = DATA / dest
    seen.replace(target)
    return {"ok": True, "rows": rows, "path": str(target),
            "flag": "near cap" if rows >= 1450 else ""}


def finalize(season: int, week: int, push: bool = True) -> str:
    """Row-count report for the week's files + git add/commit(/push)."""
    tag = f"{season}wk{week}"
    files = sorted(DATA.glob(f"*_{tag}_week.csv"))
    lines = [f"week {week} ({season}) — {len(files)} files"]
    for f in files:
        n = max(0, sum(1 for _ in f.open("r", encoding="utf-8", errors="ignore")) - 3)
        lines.append(f"  {f.name:44s} {n:>6} rows" + ("  <-- near cap" if n >= 1450 else ""))
    if push and files:
        rel = "data/dfs/matchup"
        subprocess.run(["git", "add", rel], cwd=ROOT, check=True)
        if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode != 0:
            subprocess.run(["git", "commit", "-m", f"FantasyPoints week {week} {season}"], cwd=ROOT, check=True)
            subprocess.run(["git", "push"], cwd=ROOT, check=True)
            lines.append("committed + pushed — Streamlit Cloud redeploys in ~2 min")
        else:
            lines.append("nothing new to commit")
    return "\n".join(lines)


def finalize_backfill(season: int, push: bool = True) -> str:
    """Row-count report for a whole-season backfill + git add/commit(/push)."""
    files = sorted(set(DATA.glob(f"*_{season}_week.csv")) | set(DATA.glob(f"*_{season}wk*_week.csv")))
    lines = [f"{season} backfill — {len(files)} files"]
    for f in files:
        n = max(0, sum(1 for _ in f.open("r", encoding="utf-8", errors="ignore")) - 3)
        lines.append(f"  {f.name:40s} {n:>6} rows" + ("  <-- near cap" if n >= 1450 else ""))
    if push and files:
        subprocess.run(["git", "add", "data/dfs/matchup"], cwd=ROOT, check=True)
        if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode != 0:
            subprocess.run(["git", "commit", "-m", f"FantasyPoints {season} season backfill"], cwd=ROOT, check=True)
            subprocess.run(["git", "push"], cwd=ROOT, check=True)
            lines.append("committed + pushed — Streamlit Cloud redeploys in ~2 min")
        else:
            lines.append("nothing new to commit")
    return "\n".join(lines)


def last_completed_week(today=None, kickoff: str = "2026-09-10") -> int:
    """Best-guess NFL week whose games have all finished, for a Wednesday pull.

    `kickoff` = the Thursday of Week 1. Week N's games run Thu..Mon of week N; from the
    Tuesday after through the following Wednesday is a "grace period" where week N counts
    as complete (MNF is done, no more football that calendar week). Returns 0 before the
    season starts. The scheduled task should still sanity-check this against the site
    before committing.

    `today` defaults to the current date in US/Eastern (the NFL's own reference
    timezone), NOT naive server-local time — Streamlit Cloud's container runs on UTC,
    which is 4-5 hours ahead of Eastern. Using naive `date.today()` there flips to the
    next calendar day while it's still evening in the US, which silently bumped this
    (and current_week() in opp/blend.py, which calls this) a week early — e.g. showing
    week 3 as "current" on a Wednesday night when every US clock still says week 2.
    Confirmed live on 2026-09-16: UTC was already 2026-09-17 by ~10pm ET.

    NOTE the separate bug this fixes, found live on 2026-09-17: `days // 7 + 1` jumps a
    full week early the moment `days` crosses a multiple of 7 -- which is week N+1's own
    *kickoff* day, not its completion. That showed "week 2 complete" the instant week 2's
    Thursday game started, 5 days before it actually finished. Decomposing into "which
    week's window are we in" (`days // 7`) x "are we past that week's games into its own
    grace period" (`days % 7 >= 5`) avoids the boundary crossing entirely.
    """
    from datetime import date, datetime
    from zoneinfo import ZoneInfo
    if today is None:
        today = datetime.now(ZoneInfo("America/New_York")).date()
    days = (today - date.fromisoformat(kickoff)).days
    if days < 5:                      # Week 1 not done yet (or preseason)
        return 0
    week_in_progress = days // 7 + 1  # 1-indexed week whose Thu..Wed window `days` falls in
    in_grace_period = (days % 7) >= 5  # Tue/Wed of that window -> that week now counts done
    last_done = week_in_progress if in_grace_period else week_in_progress - 1
    return max(1, min(18, last_done))


def week_of_date(d, kickoff: str = "2026-09-10") -> int:
    """Which NFL week's Thu..Wed window a given date falls in -- for deriving
    `as_of_week` from a SPECIFIC GAME's own commence_time, rather than from "today" (via
    last_completed_week/current_week) or a UI widget that may not match the game actually
    being pulled.

    That mismatch is a real bug this found live: the DFS Prop Edges screen passes the
    sidebar's global "NFL week" number into every props pull regardless of which game is
    selected (dfs/screens.py) -- so whenever that sidebar value is wrong (it has been,
    twice, from two separate bugs in last_completed_week/current_week) OR just hasn't
    been bumped yet for a new week, "our proj" gets computed with a walk-forward cutoff
    for the WRONG week while being priced against the RIGHT week's actual book line --
    silently comparing two different games' worth of trailing data. Deriving the week
    from the event's own commence_time instead makes a specific-game pull correct
    regardless of whatever the sidebar currently shows.

    `d` may be a `date`, `datetime`, or ISO string (e.g. the Odds API's `commence_time`).
    """
    from datetime import date, datetime
    from zoneinfo import ZoneInfo
    eastern = ZoneInfo("America/New_York")
    if isinstance(d, str):
        d = datetime.fromisoformat(d.replace("Z", "+00:00")).astimezone(eastern).date()
    elif isinstance(d, datetime):
        # naive datetimes are assumed already-Eastern (this codebase's convention
        # elsewhere), not system-local -- system-local is UTC on Streamlit Cloud, the
        # exact bug this function exists to avoid reintroducing.
        d = d.astimezone(eastern).date() if d.tzinfo else d.date()
    days = (d - date.fromisoformat(kickoff)).days
    return max(1, min(18, days // 7 + 1))
