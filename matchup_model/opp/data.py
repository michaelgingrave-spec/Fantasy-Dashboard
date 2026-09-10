"""Fetch + cache the nflverse release assets the opportunity model runs on.

All free, pulled straight from the nflverse-data GitHub releases (and nfldata for the
Vegas lines), cached as parquet under data/dfs/nflverse/ (git-ignored). Call
`refresh()` to (re-)download; the tidy accessors below read the cache.
"""
from __future__ import annotations

import io
import time
import urllib.request
from functools import lru_cache
from pathlib import Path

import pandas as pd

from matchup_model.config import ROOT

CACHE = ROOT / "data" / "dfs" / "nflverse"
CACHE.mkdir(parents=True, exist_ok=True)

_REL = "https://github.com/nflverse/nflverse-data/releases/download"
_GAMES_URL = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"

# season range we keep locally (backtest seasons + the live one)
SEASONS = [2021, 2022, 2023, 2024, 2025]

POS_KEEP = {"QB", "RB", "WR", "TE"}

# relocated / alt abbreviations -> nflverse canonical
_TEAM_FIX = {"OAK": "LV", "SD": "LAC", "SDG": "LAC", "STL": "LA", "LAR": "LA",
             "JAC": "JAX", "WSH": "WAS", "ARZ": "ARI", "CLV": "CLE", "BLT": "BAL",
             "HST": "HOU", "GNB": "GB", "KAN": "KC", "NWE": "NE", "NOR": "NO",
             "SFO": "SF", "TAM": "TB"}


def canon_team(t) -> str:
    t = str(t or "").upper().strip()
    return _TEAM_FIX.get(t, t)


def _download(url: str, dest: Path, tries: int = 2) -> bool:
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            raw = urllib.request.urlopen(req, timeout=45).read()
            dest.write_bytes(raw)
            return True
        except Exception as e:  # noqa: BLE001
            if k == tries - 1:
                print(f"  ! failed {url}\n    {type(e).__name__}: {e}")
                return False
            time.sleep(0.6)
    return False


@lru_cache(maxsize=1)
def _reachable() -> bool:
    """Cheap one-shot connectivity probe so a blocked host (e.g. some cloud egress)
    fails fast instead of retrying every asset."""
    try:
        req = urllib.request.Request(_GAMES_URL, method="HEAD",
                                     headers={"User-Agent": "Mozilla/5.0"})
        urllib.request.urlopen(req, timeout=10)
        return True
    except Exception:  # noqa: BLE001
        return False


def refresh(seasons: list[int] | None = None, force: bool = False) -> None:
    """Download the player/team weekly stats, snap counts, and the games/lines file."""
    seasons = seasons or SEASONS
    if not any((CACHE / f"stats_player_week_{y}.parquet").exists() for y in seasons) \
            and not _reachable():
        print("refresh: nflverse host unreachable — skipping (model will fall back)")
        return
    jobs = [(f"{_REL}/stats_player/stats_player_week_{y}.parquet",
             CACHE / f"stats_player_week_{y}.parquet") for y in seasons]
    jobs += [(f"{_REL}/stats_team/stats_team_week_{y}.parquet",
              CACHE / f"stats_team_week_{y}.parquet") for y in seasons]
    jobs += [(f"{_REL}/snap_counts/snap_counts_{y}.parquet",
              CACHE / f"snap_counts_{y}.parquet") for y in seasons]
    n = 0
    for url, dest in jobs:
        if dest.exists() and not force:
            continue
        print(f"  + {dest.name}")
        n += _download(url, dest)
    # games.csv is small and updates weekly — always refresh
    print("  + games.csv")
    n += _download(_GAMES_URL, CACHE / "games.csv")
    print(f"refresh: wrote {n} file(s) to {CACHE}")


def _ensure(name_fmt: str, seasons: list[int]) -> list[Path]:
    have = [CACHE / name_fmt.format(y) for y in seasons]
    missing = [p for p in have if not p.exists()]
    if missing:
        refresh(seasons)
    return [p for p in have if p.exists()]


@lru_cache(maxsize=1)
def player_weeks() -> pd.DataFrame:
    """One row per player-game: usage + box score + PPR/DK-ready fields, QB/RB/WR/TE."""
    paths = _ensure("stats_player_week_{}.parquet", SEASONS)
    keep = ["season", "week", "season_type", "player_id", "player_display_name",
            "position", "team", "opponent_team",
            "attempts", "completions", "passing_yards", "passing_tds",
            "passing_interceptions", "sacks_suffered", "carries", "rushing_yards",
            "rushing_tds", "targets", "receptions", "receiving_yards", "receiving_tds",
            "receiving_air_yards", "target_share", "air_yards_share", "wopr",
            "fantasy_points", "fantasy_points_ppr"]
    frames = []
    for p in paths:
        d = pd.read_parquet(p)
        d = d[[c for c in keep if c in d.columns]].copy()
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["season_type"] == "REG"]
    df["position"] = df["position"].astype(str).str.upper()
    df = df[df["position"].isin(POS_KEEP)]
    for c in ("team", "opponent_team"):
        df[c] = df[c].map(canon_team)
    num = df.select_dtypes("number").columns
    df[num] = df[num].fillna(0.0)
    df["name_key"] = df["player_display_name"].map(_nk)
    df["dk_fp"] = dk_points_from_row(df)
    return df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)


@lru_cache(maxsize=1)
def team_weeks() -> pd.DataFrame:
    """One row per team-game: pace + pass-rate building blocks."""
    paths = _ensure("stats_team_week_{}.parquet", SEASONS)
    keep = ["season", "week", "season_type", "team", "opponent_team", "attempts",
            "completions", "passing_yards", "passing_tds", "sacks_suffered", "carries",
            "rushing_yards", "rushing_tds", "passing_interceptions"]
    frames = []
    for p in paths:
        d = pd.read_parquet(p)
        frames.append(d[[c for c in keep if c in d.columns]].copy())
    df = pd.concat(frames, ignore_index=True)
    df = df[df["season_type"] == "REG"]
    for c in ("team", "opponent_team"):
        df[c] = df[c].map(canon_team)
    num = df.select_dtypes("number").columns
    df[num] = df[num].fillna(0.0)
    df["dropbacks"] = df["attempts"] + df["sacks_suffered"]
    df["plays"] = df["dropbacks"] + df["carries"]
    df["pass_rate"] = (df["dropbacks"] / df["plays"]).where(df["plays"] > 0)
    return df.sort_values(["team", "season", "week"]).reset_index(drop=True)


@lru_cache(maxsize=1)
def games() -> pd.DataFrame:
    """Schedule + closing lines + weather, long-form (one row per team-game)."""
    _ensure("stats_team_week_{}.parquet", SEASONS)  # cheap way to trigger refresh dir
    p = CACHE / "games.csv"
    if not p.exists():
        refresh()
    g = pd.read_csv(p)
    g = g[g["game_type"] == "REG"].copy()
    for c in ("home_team", "away_team"):
        g[c] = g[c].map(canon_team)
    rows = []
    for _, r in g.iterrows():
        # spread_line = points the HOME team is favored by
        home_tot = r["total_line"] / 2 + r["spread_line"] / 2 if pd.notna(r.get("total_line")) else None
        away_tot = r["total_line"] / 2 - r["spread_line"] / 2 if pd.notna(r.get("total_line")) else None
        base = dict(season=r["season"], week=r["week"], total_line=r.get("total_line"),
                    roof=r.get("roof"), temp=r.get("temp"), wind=r.get("wind"))
        rows.append({**base, "team": r["home_team"], "opp": r["away_team"],
                     "is_home": 1, "spread": -r["spread_line"], "team_total": home_tot,
                     "opp_total": away_tot})
        rows.append({**base, "team": r["away_team"], "opp": r["home_team"],
                     "is_home": 0, "spread": r["spread_line"], "team_total": away_tot,
                     "opp_total": home_tot})
    out = pd.DataFrame(rows)
    # spread here = points THIS team is favored by (positive = favored)
    return out.sort_values(["season", "week", "team"]).reset_index(drop=True)


@lru_cache(maxsize=1)
def snaps() -> pd.DataFrame:
    paths = _ensure("snap_counts_{}.parquet", SEASONS)
    frames = []
    for p in paths:
        d = pd.read_parquet(p)
        frames.append(d[["season", "week", "player", "team", "position",
                         "offense_snaps", "offense_pct"]].copy())
    df = pd.concat(frames, ignore_index=True)
    df["team"] = df["team"].map(canon_team)
    df["name_key"] = df["player"].map(_nk)
    return df


# ── scoring ────────────────────────────────────────────────────────────────
def dk_points_from_row(d: pd.DataFrame) -> pd.Series:
    """DraftKings NFL Classic scoring from box-score columns (full PPR + yardage bonuses)."""
    p = (
        d.get("passing_yards", 0) * 0.04 + d.get("passing_tds", 0) * 4
        - d.get("passing_interceptions", 0) * 1.0
        + (d.get("passing_yards", 0) >= 300).astype(float) * 3.0
        + d.get("rushing_yards", 0) * 0.1 + d.get("rushing_tds", 0) * 6
        + (d.get("rushing_yards", 0) >= 100).astype(float) * 3.0
        + d.get("receptions", 0) * 1.0 + d.get("receiving_yards", 0) * 0.1
        + d.get("receiving_tds", 0) * 6
        + (d.get("receiving_yards", 0) >= 100).astype(float) * 3.0
    )
    return p.astype(float)


def dk_points_from_line(line: dict) -> float:
    g = line.get
    p = 0.0
    p += g("pass_yds", 0) * 0.04 + g("pass_td", 0) * 4 - g("int", 0) * 1.0
    p += 3.0 if g("pass_yds", 0) >= 300 else 0.0
    p += g("rush_yds", 0) * 0.1 + g("rush_td", 0) * 6
    p += 3.0 if g("rush_yds", 0) >= 100 else 0.0
    p += g("rec", 0) * 1.0 + g("rec_yds", 0) * 0.1 + g("rec_td", 0) * 6
    p += 3.0 if g("rec_yds", 0) >= 100 else 0.0
    return round(p, 2)


@lru_cache(maxsize=1)
def _nk_fn():
    from dfs.names import normalize_name
    return normalize_name


def _nk(name) -> str:
    return _nk_fn()(str(name))


if __name__ == "__main__":  # `python -m matchup_model.opp.data` -> refresh the cache
    refresh(force=False)
    pw, tw, gm = player_weeks(), team_weeks(), games()
    print(f"player_weeks {pw.shape}  seasons {sorted(pw.season.unique())}")
    print(f"team_weeks   {tw.shape}")
    print(f"games        {gm.shape}  lines non-null: {gm.team_total.notna().mean():.0%}")
