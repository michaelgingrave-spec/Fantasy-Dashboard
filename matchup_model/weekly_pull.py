"""Recipe + bookkeeping for the weekly FantasyPoints Data Suite pull.

The CSV export is a client-side button, so the actual downloading has to be done by a
browser (the claude-in-chrome session in Chrome, logged into fantasypointsdata.com, with
automatic downloads allow-listed for that site). This module supplies the exact URL list
and the file move / rename / verify / commit steps so the browser part is just "click
Download CSV on each of these pages".

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


def last_completed_week(today=None, kickoff: str = "2026-09-10") -> int:
    """Best-guess NFL week whose games have all finished, for a Wednesday pull.

    `kickoff` = the Thursday of Week 1. Week N's games run Thu..Mon of week N, so by the
    following Wednesday week N is complete. Returns 0 before the season starts. The
    scheduled task should still sanity-check this against the site before committing.
    """
    from datetime import date
    today = today or date.today()
    days = (today - date.fromisoformat(kickoff)).days
    if days < 5:                      # Week 1 not done yet (or preseason)
        return 0
    return max(1, min(18, days // 7 + 1))
