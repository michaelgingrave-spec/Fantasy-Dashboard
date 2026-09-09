"""Optional: pull NFL game lines from The Odds API (https://the-odds-api.com).

Only game lines (h2h / spreads / totals) — cheap on the free tier. Used to add an
implied-team-total term to the Matchup Edge model. If ODDS_API_KEY is unset this module
returns None and the rest of the app carries on.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

from dfs.config import ODDS_API_KEY, ODDS_CACHE_DIR, EDGE_ODDS_TTL_HOURS

_BASE = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"


class OddsError(RuntimeError):
    pass


def _cache_file() -> Path:
    return ODDS_CACHE_DIR / "nfl_game_lines.json"


def get_nfl_game_lines(use_cache: bool = True) -> list[dict] | None:
    """Raw Odds API events (list), or None when no key is configured."""
    if not ODDS_API_KEY:
        return None

    cf = _cache_file()
    if use_cache and cf.exists():
        age_h = (time.time() - cf.stat().st_mtime) / 3600
        if age_h < EDGE_ODDS_TTL_HOURS:
            try:
                return json.loads(cf.read_text(encoding="utf-8"))
            except Exception:
                pass

    qs = urlencode(
        {
            "apiKey": ODDS_API_KEY,
            "regions": "us",
            "markets": "h2h,spreads,totals",
            "oddsFormat": "american",
            "bookmakers": "draftkings",
        }
    )
    try:
        with urlopen(f"{_BASE}?{qs}", timeout=25) as resp:
            data = json.loads(resp.read())
    except Exception as e:  # noqa: BLE001
        raise OddsError(f"Odds API request failed: {e}") from e

    try:
        cf.write_text(json.dumps(data), encoding="utf-8")
    except Exception:
        pass
    return data
