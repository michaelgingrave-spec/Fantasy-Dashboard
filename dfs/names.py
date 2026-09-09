"""Player-name and team-abbreviation normalization.

DraftKings and FantasyPoints spell teams (and a few players) differently. Everything in
this app is normalized to **DraftKings-standard** team abbreviations and a canonical
lowercase player key so the salary pool and the projection sheet can be joined.
"""
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

# ── Team abbreviations ──────────────────────────────────────────────────────────
# Canonical = DraftKings. Map every alias (FantasyPoints, ESPN, PFR, full names) onto it.
DK_TEAMS = {
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET", "GB",
    "HOU", "IND", "JAX", "KC", "LAC", "LAR", "LV", "MIA", "MIN", "NE", "NO", "NYG",
    "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
}

_TEAM_ALIASES = {
    # FantasyPoints non-standard abbreviations
    "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU", "LA": "LAR", "JAC": "JAX",
    # other common variants
    "GNB": "GB", "GBP": "GB", "KAN": "KC", "KCC": "KC", "LVR": "LV", "OAK": "LV",
    "NWE": "NE", "NEP": "NE", "NOR": "NO", "NOS": "NO", "SFO": "SF", "SF49": "SF",
    "TAM": "TB", "TBB": "TB", "SD": "LAC", "SDG": "LAC", "STL": "LAR", "WFT": "WAS",
    "WSH": "WAS", "LAR ": "LAR", "NYG ": "NYG",
}

_FULL_TO_DK = {
    "arizona cardinals": "ARI", "atlanta falcons": "ATL", "baltimore ravens": "BAL",
    "buffalo bills": "BUF", "carolina panthers": "CAR", "chicago bears": "CHI",
    "cincinnati bengals": "CIN", "cleveland browns": "CLE", "dallas cowboys": "DAL",
    "denver broncos": "DEN", "detroit lions": "DET", "green bay packers": "GB",
    "houston texans": "HOU", "indianapolis colts": "IND", "jacksonville jaguars": "JAX",
    "kansas city chiefs": "KC", "los angeles chargers": "LAC", "los angeles rams": "LAR",
    "las vegas raiders": "LV", "miami dolphins": "MIA", "minnesota vikings": "MIN",
    "new england patriots": "NE", "new orleans saints": "NO", "new york giants": "NYG",
    "new york jets": "NYJ", "philadelphia eagles": "PHI", "pittsburgh steelers": "PIT",
    "seattle seahawks": "SEA", "san francisco 49ers": "SF", "tampa bay buccaneers": "TB",
    "tennessee titans": "TEN", "washington commanders": "WAS",
}

# team abbrev -> nickname, for matching DST rows ("Eagles" <-> PHI)
DK_TEAM_NICKNAME = {
    "ARI": "cardinals", "ATL": "falcons", "BAL": "ravens", "BUF": "bills",
    "CAR": "panthers", "CHI": "bears", "CIN": "bengals", "CLE": "browns",
    "DAL": "cowboys", "DEN": "broncos", "DET": "lions", "GB": "packers",
    "HOU": "texans", "IND": "colts", "JAX": "jaguars", "KC": "chiefs",
    "LAC": "chargers", "LAR": "rams", "LV": "raiders", "MIA": "dolphins",
    "MIN": "vikings", "NE": "patriots", "NO": "saints", "NYG": "giants",
    "NYJ": "jets", "PHI": "eagles", "PIT": "steelers", "SEA": "seahawks",
    "SF": "49ers", "TB": "buccaneers", "TEN": "titans", "WAS": "commanders",
}
_NICKNAME_TO_DK = {v: k for k, v in DK_TEAM_NICKNAME.items()}


def norm_team(value: str) -> str:
    """Return the DraftKings-standard abbreviation for any team spelling."""
    if value is None:
        return ""
    s = str(value).strip()
    if not s:
        return ""
    up = s.upper()
    if up in DK_TEAMS:
        return up
    if up in _TEAM_ALIASES:
        return _TEAM_ALIASES[up]
    low = s.lower()
    if low in _FULL_TO_DK:
        return _FULL_TO_DK[low]
    if low in _NICKNAME_TO_DK:
        return _NICKNAME_TO_DK[low]
    # "Philadelphia Eagles" style not in the exact map -> try last word as nickname
    last = low.split()[-1] if low.split() else ""
    if last in _NICKNAME_TO_DK:
        return _NICKNAME_TO_DK[last]
    return up  # unknown — return as-is, upper-cased


_SUFFIX_RE = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b\.?", re.IGNORECASE)
_NONALNUM_RE = re.compile(r"[^a-z0-9 ]+")
_WS_RE = re.compile(r"\s+")


def normalize_name(name: str) -> str:
    """Canonical player key: ascii, lowercase, no punctuation, no generational suffix.

    'Ja'Marr Chase' -> 'jamarr chase'; 'A.J. Brown' -> 'aj brown';
    'Patrick Mahomes II' -> 'patrick mahomes'; 'Michael Pittman Jr.' -> 'michael pittman'.
    """
    if name is None:
        return ""
    s = unicodedata.normalize("NFKD", str(name))
    s = s.encode("ascii", "ignore").decode("ascii")
    s = s.replace("\n", " ").replace("’", "'").lower()
    s = s.replace(".", "").replace("'", "")
    s = _SUFFIX_RE.sub(" ", s)
    s = _NONALNUM_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def is_dst(position: str) -> bool:
    return str(position).strip().upper() in {"DST", "D/ST", "DEF", "D"}


def player_key(name: str, team: str, position: str) -> str:
    """Join key used across salary pool, projections and Data Suite tables.

    DST rows collapse to 'dst <team>' so 'Eagles' / 'Philadelphia Eagles' / 'PHI DST' align.
    """
    team_n = norm_team(team)
    if is_dst(position):
        # name might be "Eagles", "Philadelphia Eagles", or a team abbrev
        nick = normalize_name(name).split()[-1] if normalize_name(name) else ""
        team_n = team_n or _NICKNAME_TO_DK.get(nick, "")
        return f"dst {team_n}".strip()
    return f"{normalize_name(name)} {team_n}".strip()


def name_team_key(name: str, team: str) -> str:
    """Looser join key (no position) for merging stat tables onto the slate."""
    return f"{normalize_name(name)} {norm_team(team)}".strip()


def fuzzy_match(target: str, candidates: list[str], cutoff: float = 0.9) -> str | None:
    """Best candidate for `target` by SequenceMatcher ratio, or None below cutoff.

    Caller is responsible for restricting `candidates` to the same team + position.
    """
    best, best_r = None, 0.0
    for c in candidates:
        r = SequenceMatcher(None, target, c).ratio()
        if r > best_r:
            best, best_r = c, r
    return best if best_r >= cutoff else None
