"""Weekly DraftKings snapshot refresher.

DraftKings' JSON only answers from a residential IP, so the live pull in the DFS
screens works when you run the dashboard locally but NOT on Streamlit Cloud. Run
this on your PC once a week (Thursday morning is ideal) to save the current
slates + salaries into ``data/dfs/dk_snapshot/``. The DFS screens fall back to
that snapshot whenever the live request fails.

    python -m dfs.refresh_dk              # just refresh the local snapshot
    python -m dfs.refresh_dk --push      # refresh, then git commit + push so the
                                         # deployed dashboard picks it up

Automate it with Windows Task Scheduler:
    Program:   <path-to>\python.exe
    Arguments: -m dfs.refresh_dk --push
    Start in:  C:\\Users\\mjgin\\Fantasy Model
    Trigger:   Weekly, Thursday 8:00 AM
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__" and __package__ is None:  # allow `python dfs/refresh_dk.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dfs.config import DK_SNAPSHOT_DIR
from dfs.dk_client import (
    DRAFTABLES_URL,
    LOBBY_URL,
    DKError,
    _get_json,
    classic_slates,
)

LOBBY_FILE = "lobby_NFL.json"

# The raw DK responses are megabytes (images, prize tables, per-row attribute blobs).
# Keep only what dfs/dk_client.py actually parses so weekly commits stay tiny.
_DG_KEYS = ("DraftGroupId", "ContestTypeId", "GameCount", "StartDate",
            "ContestStartTimeSuffix", "DraftGroupTag", "Sport")
_DR_KEYS = ("playerDkId", "displayName", "position", "salary", "teamAbbreviation",
            "status", "isDisabled")


def _slim_lobby(data: dict) -> dict:
    return {"DraftGroups": [{k: g.get(k) for k in _DG_KEYS} for g in data.get("DraftGroups", [])]}


def _slim_draftables(data: dict) -> dict:
    rows, seen = [], set()
    for r in data.get("draftables", []):
        pid = r.get("playerDkId")
        if pid in seen:  # DK repeats each player once per roster-slot eligibility
            continue
        seen.add(pid)
        comp = r.get("competition") or {}
        rows.append({
            **{k: r.get(k) for k in _DR_KEYS},
            "competition": {"name": comp.get("name"), "startTime": comp.get("startTime")},
        })
    return {"draftables": rows}


def _save(name: str, data: dict) -> Path:
    p = DK_SNAPSHOT_DIR / name
    p.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    return p


def refresh(prune: bool = True) -> list[int]:
    """Write lobby + every classic slate's draftables into the snapshot dir.
    Returns the list of draft-group ids saved."""
    DK_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    lobby = _get_json(LOBBY_URL.format(sport="NFL"))
    _save(LOBBY_FILE, _slim_lobby(lobby))
    print(f"  {LOBBY_FILE}  ({len(lobby.get('DraftGroups', []))} draft groups)")

    slates = classic_slates(use_cache=False)
    saved: list[int] = []
    for s in slates:
        try:
            dr = _get_json(DRAFTABLES_URL.format(gid=s.draft_group_id))
        except DKError as e:
            print(f"  ! skip dg {s.draft_group_id}: {e}")
            continue
        n = len({r.get("playerDkId") for r in dr.get("draftables", [])})
        _save(f"draftables_{s.draft_group_id}.json", _slim_draftables(dr))
        saved.append(s.draft_group_id)
        print(f"  draftables_{s.draft_group_id}.json  [{s.slate_type}, {s.game_count} games, {n} players]")

    if prune:
        keep = {LOBBY_FILE} | {f"draftables_{i}.json" for i in saved}
        for f in DK_SNAPSHOT_DIR.glob("*.json"):
            if f.name not in keep:
                f.unlink()
                print(f"  pruned stale {f.name}")

    return saved


def git_push() -> None:
    root = Path(__file__).resolve().parent.parent
    rel = "data/dfs/dk_snapshot"
    subprocess.run(["git", "add", rel], cwd=root, check=True)
    diff = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=root).returncode
    if diff == 0:
        print("nothing changed — not committing")
        return
    from datetime import date
    subprocess.run(["git", "commit", "-m", f"DK snapshot {date.today():%Y-%m-%d}"], cwd=root, check=True)
    subprocess.run(["git", "push"], cwd=root, check=True)
    print("committed + pushed — Streamlit Cloud will redeploy in ~2 min")


def main() -> None:
    args = set(sys.argv[1:])
    print(f"Refreshing DK snapshot -> {DK_SNAPSHOT_DIR}")
    saved = refresh(prune="--no-prune" not in args)
    print(f"Saved {len(saved)} slate(s).")
    if "--push" in args:
        git_push()


if __name__ == "__main__":
    main()
