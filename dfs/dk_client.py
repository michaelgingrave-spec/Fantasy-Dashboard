"""Pull the current DraftKings NFL Classic slate (player pool + salaries) from DK's
public JSON endpoints. No login required.

Flow:
    getcontests?sport=NFL           -> list of DraftGroups (slates)
    pick_main_slate(...)            -> the Sunday main Classic slate (or user override)
    draftgroups/{id}/draftables     -> player pool with salaries

DK blocks datacenter IPs, so this must run from the user's machine. Responses are cached
to data/dk/ for a few minutes. On failure the caller can drop a manual
data/dk/draftables_<id>.json (the browser-Claude session can save one).
"""
from __future__ import annotations

import gzip
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

try:
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover
    _ET = timezone.utc

import pandas as pd

from dfs.config import DK_CACHE_DIR, DK_CACHE_TTL_MIN, DK_SNAPSHOT_DIR, DK_SPORT
from dfs.names import norm_team

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Encoding": "gzip",
    "Referer": "https://www.draftkings.com/",
}

LOBBY_URL = "https://www.draftkings.com/lobby/getcontests?sport={sport}"
DRAFTABLES_URL = "https://api.draftkings.com/draftgroups/v1/draftgroups/{gid}/draftables"

# ContestTypeIds that are salary-cap Classic (not showdown / snake / tiers / novelty).
CLASSIC_CONTEST_TYPE_IDS = {21, 189}
# roster-slot -> base position is not needed; `position` field is authoritative.
INJURY_OUT_STATUSES = {"O", "OUT", "IR", "PUP", "NFI", "SUSP", "D"}


class DKError(RuntimeError):
    pass


def _get_json(url: str, timeout: int = 25) -> dict:
    req = Request(url, headers=_HEADERS)
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            if resp.headers.get("Content-Encoding") == "gzip":
                raw = gzip.decompress(raw)
            return json.loads(raw)
    except Exception as e:  # noqa: BLE001 - surface a single clean error to the UI
        raise DKError(f"DraftKings request failed ({url}): {e}") from e


def _cache_path(name: str) -> Path:
    return DK_CACHE_DIR / name


def _read_cache(name: str, ttl_min: int) -> dict | None:
    p = _cache_path(name)
    if not p.exists():
        return None
    age_min = (time.time() - p.stat().st_mtime) / 60
    if age_min > ttl_min:
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_cache(name: str, data: dict) -> None:
    try:
        _cache_path(name).write_text(json.dumps(data), encoding="utf-8")
    except Exception:
        pass


# ── Slates ───────────────────────────────────────────────────────────────────
_SINGLE_GAME_SUFFIX = re.compile(r"\([A-Z]{2,3}\s*(?:@|vs\.?)\s*[A-Z]{2,3}", re.IGNORECASE)

# Slate types, in the order they should appear in a picker (most common first).
SLATE_TYPES = [
    "Sunday main",
    "Sunday–Monday",
    "Full week",
    "Sunday early",
    "Sunday afternoon",
    "Primetime",
    "Other",
]


@dataclass
class Slate:
    draft_group_id: int
    contest_type_id: int
    game_count: int
    start: datetime | None
    suffix: str
    tag: str

    @property
    def start_et(self) -> datetime | None:
        return self.start.astimezone(_ET) if self.start else None

    @property
    def slate_type(self) -> str:
        """Coarse contest-timing bucket, from the start day + DK's suffix label.

        'Full week' = spans a non-Sunday day (Thu/Sat games). The Sunday-only
        buckets split by how much of Sunday they cover.
        """
        suf = re.sub(r"[()]", "", self.suffix or "").strip().lower()
        day = self.start_et.weekday() if self.start_et else 6  # Mon=0 … Sun=6
        if day != 6 or "wed-" in suf or "thu-" in suf or "sat" in suf:
            return "Full week"
        if "sun-mon" in suf or ("sun" in suf and "mon" in suf):
            return "Sunday–Monday"
        if suf.startswith("sun-") or "sun-tue" in suf or "sun-wed" in suf or "sun-thu" in suf:
            return "Full week"  # Sunday start but runs into the following week
        if "early" in suf or "1pm" in suf or "1 pm" in suf:
            return "Sunday early"
        if "afternoon" in suf or "4pm" in suf or "late only" in suf:
            return "Sunday afternoon"
        if "night" in suf or "primetime" in suf or "prime time" in suf:
            return "Primetime"
        if suf in ("", "main", "featured", "full slate", "all day"):
            return "Sunday main"
        return "Other"

    @property
    def label(self) -> str:
        et = self.start_et
        when = et.strftime("%a %m/%d %I:%M%p ET") if et else "?"
        suf = self.suffix.strip() or "Main"
        return f"{suf} · {self.game_count} games · {when}  (dg {self.draft_group_id})"

    @property
    def is_classic(self) -> bool:
        return (
            self.contest_type_id in CLASSIC_CONTEST_TYPE_IDS
            and self.game_count >= 2
            and not _SINGLE_GAME_SUFFIX.search(self.suffix or "")
        )


def slate_types_present(slates: list[Slate]) -> list[str]:
    """SLATE_TYPES restricted to (and ordered by) what's in `slates`."""
    have = {s.slate_type for s in slates}
    return [t for t in SLATE_TYPES if t in have]


def _parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def parse_slates(data: dict, sport: str = DK_SPORT) -> list[Slate]:
    """Pure: getcontests JSON -> Slate list."""
    slates: list[Slate] = []
    for d in data.get("DraftGroups", []):
        if sport and (d.get("Sport") or "").upper() not in ("", sport.upper()):
            continue
        slates.append(
            Slate(
                draft_group_id=int(d["DraftGroupId"]),
                contest_type_id=int(d.get("ContestTypeId") or 0),
                game_count=int(d.get("GameCount") or 0),
                start=_parse_dt(d.get("StartDate")),
                suffix=str(d.get("ContestStartTimeSuffix") or ""),
                tag=str(d.get("DraftGroupTag") or ""),
            )
        )
    return slates


def _snapshot(name: str) -> dict | None:
    """Weekly committed snapshot written by dfs/refresh_dk.py — used when the live
    DK request fails (e.g. on Streamlit Cloud, whose IP DraftKings blocks)."""
    p = DK_SNAPSHOT_DIR / name
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def list_slates(sport: str = DK_SPORT, use_cache: bool = True) -> list[Slate]:
    """All DraftGroups for the sport, newest-first-ish. Filter with `.is_classic`."""
    cache_name = f"lobby_{sport}.json"
    data = _read_cache(cache_name, DK_CACHE_TTL_MIN) if use_cache else None
    if data is None:
        try:
            data = _get_json(LOBBY_URL.format(sport=sport))
            _write_cache(cache_name, data)
        except DKError:
            data = _snapshot(cache_name)
            if data is None:
                raise
    return parse_slates(data, sport)


def classic_slates(sport: str = DK_SPORT, use_cache: bool = True) -> list[Slate]:
    out = [s for s in list_slates(sport, use_cache) if s.is_classic]
    out.sort(key=lambda s: (s.start or datetime.max.replace(tzinfo=timezone.utc), -s.game_count))
    return out


def pick_main_slate(slates: list[Slate]) -> Slate | None:
    """Default choice: the earliest Sunday slate with the most games (the Sunday main).
    Falls back to the largest classic slate overall.
    """
    classic = [s for s in slates if s.is_classic]
    if not classic:
        return None
    sundays = [s for s in classic if s.start_et and s.start_et.weekday() == 6]
    pool = sundays or classic
    earliest = min(s.start_et for s in pool if s.start_et)
    same_day = [s for s in pool if s.start_et and s.start_et.date() == earliest.date()]
    return max(same_day or pool, key=lambda s: s.game_count)


# ── Draftables (salaries) ────────────────────────────────────────────────────
def _load_draftables_json(draft_group_id: int, use_cache: bool = True) -> dict:
    cache_name = f"draftables_{draft_group_id}.json"
    # A manually-dropped file (no timestamp check) always wins if present.
    manual = _cache_path(cache_name)
    if manual.exists() and not use_cache:
        return json.loads(manual.read_text(encoding="utf-8"))
    data = _read_cache(cache_name, DK_CACHE_TTL_MIN) if use_cache else None
    if data is None:
        try:
            data = _get_json(DRAFTABLES_URL.format(gid=draft_group_id))
            _write_cache(cache_name, data)
        except DKError:
            if manual.exists():
                return json.loads(manual.read_text(encoding="utf-8"))
            snap = _snapshot(cache_name)
            if snap is not None:
                return snap
            raise
    return data


def parse_draftables(data: dict, draft_group_id: int | str = "") -> pd.DataFrame:
    """Pure: draftables JSON -> salary DataFrame.

    Columns: dk_id, name, pos, team, opp, salary, game_time, status, playable.
    """
    rows = data.get("draftables", [])
    if not rows:
        raise DKError(f"Draft group {draft_group_id} has no draftables.")

    seen: set[int] = set()
    out = []
    for r in rows:
        dk_id = r.get("playerDkId")
        if dk_id in seen:
            continue
        seen.add(dk_id)
        salary = r.get("salary")
        if salary is None:
            continue  # snake/best-ball group — not a salary slate
        team = norm_team(r.get("teamAbbreviation") or "")
        comp = r.get("competition") or {}
        opp = _opponent_from_competition(comp.get("name"), team)
        status = str(r.get("status") or "None").upper()
        out.append(
            {
                "dk_id": str(dk_id),
                "name": r.get("displayName") or "",
                "pos": str(r.get("position") or "").upper(),
                "team": team,
                "opp": opp,
                "salary": int(salary),
                "game_time": comp.get("startTime") or "",
                "status": "None" if status == "NONE" else status,
                "playable": (not r.get("isDisabled", False)) and status not in INJURY_OUT_STATUSES,
            }
        )

    df = pd.DataFrame(out)
    if df.empty or "DST" not in set(df["pos"]):
        raise DKError(
            f"Draft group {draft_group_id} doesn't look like a Classic slate "
            f"(no salaries or no DST). Pick another slate."
        )
    return df.sort_values(["pos", "salary"], ascending=[True, False]).reset_index(drop=True)


def get_salaries(draft_group_id: int, use_cache: bool = True) -> pd.DataFrame:
    """Fetch (or read cached/manual) draftables for a draft group and parse to a frame."""
    data = _load_draftables_json(draft_group_id, use_cache)
    return parse_draftables(data, draft_group_id)


def _opponent_from_competition(name: str | None, team: str) -> str:
    """Opponent abbrev, prefixed '@' when `team` is the visitor.
    'TB @ CIN' -> team TB: '@CIN'; team CIN: 'TB'.
    """
    if not name:
        return ""
    m = re.match(r"\s*([A-Za-z]{2,3})\s*@\s*([A-Za-z]{2,3})\s*", name)
    if m:
        away, home = norm_team(m.group(1)), norm_team(m.group(2))
        if team == away:
            return f"@{home}"
        if team == home:
            return away
        return ""
    m = re.match(r"\s*([A-Za-z]{2,3})\s*vs\.?\s*([A-Za-z]{2,3})\s*", name, re.IGNORECASE)
    if m:
        a, b = norm_team(m.group(1)), norm_team(m.group(2))
        if team == a:
            return b
        if team == b:
            return a
    return ""
