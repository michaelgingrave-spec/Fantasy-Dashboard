"""Historical DraftKings Sunday-main-slate salaries joined to what every player actually scored.

The raw material for the hindsight ("what was the best lineup?") study behind the Millionaire Maker tool:

  * `data/dfs/history/main_salaries.csv` (committed): the DraftKings salary pool of each week's Sunday MAIN Classic slate, the one the Millionaire Maker runs on
    (Sunday 1:00 PM + the 4 PM games, no Thursday / London / Sunday-night / Monday games).
  * actual DK points per player from the nflverse box scores (cached under data/dfs/nflverse/), DraftKings' scoring applied to them, and DST points built
    from the team box scores and the final scores.
  * the schedule context of every game: closing spread, total, implied team totals, roof, wind.

`python -m dfs.history build` rebuilds the salary file from the local DK cache (and fetches any group DraftKings still serves); `python -m dfs.history add --season S
--week W --dg ID` adds one more week. DraftKings keeps serving a draft group's player list after the games ('Game Info' just reads 'Final'), so a past week can be
filled in as long as its draft group id is known.

Eleven weeks is a thin sample, so `python -m dfs.history archive` adds the 2014-2021 seasons from Rotoguru's DraftKings archive (a free page per week with each player's DraftKings
salary and the DraftKings points he scored). Those rows live in `archive_salaries.csv`, which stays on this machine (git-ignored): the archive is somebody else's compiled
data, so the repository keeps only what the study derives from it. Archive weeks carry `pts_recorded`, DraftKings' own points, and use it in place of the box-score scoring.
"""
from __future__ import annotations

import argparse
import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from dfs.config import DATA, DK_CACHE_DIR
from dfs.names import fuzzy_match, norm_team, normalize_name

HISTORY_DIR = DATA / "history"
SALARY_FILE = HISTORY_DIR / "main_salaries.csv"
ARCHIVE_FILE = HISTORY_DIR / "archive_salaries.csv"      # 2014-2021 from the Rotoguru archive; local only
ARCHIVE_RAW_DIR = HISTORY_DIR / "rotoguru_raw"           # the pages as fetched, so a rebuild does not ask the site again
ARCHIVE_URL = "http://rotoguru1.com/cgi-bin/fyday.pl?week={week}&year={season}&game=dk&scsv=1"
ARCHIVE_SEASONS = tuple(range(2014, 2022))

# (season, week) -> draft group of that week's Sunday main Classic slate. Weeks 4, 5 and 8 were not in the local cache; DraftKings still serves their pools.
MAIN_SLATES: dict[tuple[int, int], int] = {
    (2026, 1): 151307, (2026, 2): 153428, (2026, 3): 153769, (2026, 4): 154078, (2026, 5): 154468, (2026, 6): 154941,
    (2026, 7): 155198, (2026, 8): 153728, (2026, 9): 155994, (2026, 10): 156416, (2026, 11): 156561,
}

MAIN_START = (13, 0)          # a game belongs to the main slate when it kicks off Sunday between 1:00 PM and 4:30 PM Eastern
MAIN_END = (16, 30)


# ── DraftKings scoring ───────────────────────────────────────────────────────────────────────────────────
def _col(df: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(df[name], errors="coerce").fillna(0.0) if name in df.columns else pd.Series(0.0, index=df.index)


def dk_offense_points(df: pd.DataFrame) -> pd.Series:
    """DraftKings NFL Classic scoring for a box-score row (nflverse stat names): passing 0.04/yd, 4/TD, -1/INT, +3 at 300 yds; rushing and receiving 0.1/yd, 6/TD,
    +3 at 100 yds; 1 per reception; 6 for a return TD or a recovered-fumble TD; 2 per two-point conversion; -1 per fumble lost."""
    py, ry, cy = _col(df, "passing_yards"), _col(df, "rushing_yards"), _col(df, "receiving_yards")
    fumbles_lost = _col(df, "rushing_fumbles_lost") + _col(df, "receiving_fumbles_lost") + _col(df, "sack_fumbles_lost")
    two_pt = _col(df, "passing_2pt_conversions") + _col(df, "rushing_2pt_conversions") + _col(df, "receiving_2pt_conversions")
    pts = (0.04 * py + 4.0 * _col(df, "passing_tds") - _col(df, "passing_interceptions") + 3.0 * (py >= 300)
           + 0.1 * ry + 6.0 * _col(df, "rushing_tds") + 3.0 * (ry >= 100)
           + _col(df, "receptions") + 0.1 * cy + 6.0 * _col(df, "receiving_tds") + 3.0 * (cy >= 100)
           + 6.0 * (_col(df, "special_teams_tds") + _col(df, "fumble_recovery_tds"))
           + 2.0 * two_pt - fumbles_lost)
    return pts.astype(float)


def dst_points_allowed_tier(points_allowed: pd.Series) -> pd.Series:
    """DraftKings DST 'points allowed' bonus: 0 -> +10, 1-6 -> +7, 7-13 -> +4, 14-20 -> +1, 21-27 -> 0, 28-34 -> -1, 35+ -> -4."""
    pa = pd.to_numeric(points_allowed, errors="coerce")
    return pd.Series(np.select([pa <= 0, pa <= 6, pa <= 13, pa <= 20, pa <= 27, pa <= 34], [10.0, 7.0, 4.0, 1.0, 0.0, -1.0], default=-4.0), index=pa.index)


# ── Local data ──────────────────────────────────────────────────────────────────────────────────────────
def _nflverse_dir() -> Path:
    from matchup_model.opp import data as D
    return D.CACHE


def _read_nflverse(kind: str, season: int) -> pd.DataFrame:
    """The cached nflverse table (`stats_player_week` / `stats_team_week`) for one season, downloading it when the cache has not got it yet."""
    from matchup_model.opp import data as D
    name = f"{kind}_{season}.parquet"
    path = D.CACHE / name
    if not path.exists():
        D.refresh([season])
    return pd.read_parquet(path)


@lru_cache(maxsize=1)
def _schedule() -> pd.DataFrame:
    """One row per regular-season game, DK-standard team codes: season, week, kickoff (Eastern wall clock), away, home, scores, spread_home, total_line, roof, temp, wind."""
    from matchup_model.opp import data as D
    p = D.CACHE / "games.csv"
    if not p.exists():
        D.refresh()
    g = pd.read_csv(p)
    g = g[g["game_type"] == "REG"].copy()
    g["kickoff"] = pd.to_datetime(g["gameday"].astype(str) + " " + g["gametime"].fillna("13:00").astype(str), errors="coerce")
    g["away"] = g["away_team"].map(norm_team)
    g["home"] = g["home_team"].map(norm_team)
    keep = ["game_id", "season", "week", "kickoff", "away", "home", "away_score", "home_score", "spread_line", "total_line", "roof", "temp", "wind"]
    return g[keep].rename(columns={"spread_line": "spread_home"}).reset_index(drop=True)


def main_slate_games(season: int, week: int) -> pd.DataFrame:
    """The games of the week's Sunday main slate (Sunday kickoff between 1:00 and 4:30 PM Eastern)."""
    g = _schedule()
    g = g[(g.season == season) & (g.week == week)]
    t = g["kickoff"].dt.hour * 60 + g["kickoff"].dt.minute
    ok = (g["kickoff"].dt.weekday == 6) & (t >= MAIN_START[0] * 60 + MAIN_START[1]) & (t <= MAIN_END[0] * 60 + MAIN_END[1])
    return g[ok].reset_index(drop=True)


def main_slate_teams(season: int, week: int) -> set[str]:
    g = main_slate_games(season, week)
    return set(g["away"]) | set(g["home"])


def team_context(season: int, week: int) -> pd.DataFrame:
    """One row per team playing in the week: opp, is_home, spread (this team's, negative = favourite), total, team_total (implied), opp_total, roof, temp, wind, game_id."""
    g = _schedule()
    g = g[(g.season == season) & (g.week == week)]
    rows = []
    for r in g.itertuples():
        total = r.total_line
        sp = r.spread_home                                    # home team's spread: positive = home is favoured by that many
        home_tt = total / 2 + sp / 2 if pd.notna(total) and pd.notna(sp) else np.nan
        away_tt = total / 2 - sp / 2 if pd.notna(total) and pd.notna(sp) else np.nan
        base = dict(total=total, roof=r.roof, temp=r.temp, wind=r.wind, game_id=r.game_id, kickoff=r.kickoff)
        rows.append({**base, "team": r.home, "opp": r.away, "is_home": 1, "spread": -sp if pd.notna(sp) else np.nan, "team_total": home_tt, "opp_total": away_tt})
        rows.append({**base, "team": r.away, "opp": r.home, "is_home": 0, "spread": sp if pd.notna(sp) else np.nan, "team_total": away_tt, "opp_total": home_tt})
    return pd.DataFrame(rows)


# ── What each player and defense actually scored ────────────────────────────────────────────────────────
def player_actuals(season: int) -> pd.DataFrame:
    """Per player-game: key (name + team), season, week, dk_pts and the stats the analysis uses. Regular season only. Every position is scored (DraftKings' own position label is
    the one that counts on the salary side)."""
    d = _read_nflverse("stats_player_week", season)
    d = d[d["season_type"] == "REG"].copy()
    out = pd.DataFrame({
        "season": d["season"].astype(int), "week": d["week"].astype(int),
        "name": d["player_display_name"], "team": d["team"].map(norm_team), "nf_pos": d["position"],
        "dk_pts": dk_offense_points(d).round(2),
        "pass_yds": _col(d, "passing_yards"), "pass_td": _col(d, "passing_tds"), "rush_yds": _col(d, "rushing_yards"), "rush_td": _col(d, "rushing_tds"),
        "rec": _col(d, "receptions"), "rec_yds": _col(d, "receiving_yards"), "rec_td": _col(d, "receiving_tds"), "targets": _col(d, "targets"),
        "carries": _col(d, "carries"),
    })
    out["tds"] = out["pass_td"] + out["rush_td"] + out["rec_td"]
    out["key"] = [f"{normalize_name(n)} {t}" for n, t in zip(out["name"], out["team"])]
    return out


def dst_actuals(season: int) -> pd.DataFrame:
    """DraftKings DST points per team-game from the team box scores: 1 per sack, 2 per interception / fumble recovery / safety / blocked kick, 6 per defensive,
    special-teams or fumble-recovery TD, plus the points-allowed tier (the opponent's points, less 6 per TD its own defense / special teams scored and 2 per safety).
    Checked against DraftKings' own recorded DST points for every 2021 game (512 team-games): 93.4% exact, mean absolute difference 0.12; the variants that did not
    subtract the opponent's return TDs, or that subtracted 7, matched worse. Offense (QB / RB / WR / TE) matched 99-100% exactly on three 2021 weeks."""
    tw = _read_nflverse("stats_team_week", season)
    tw = tw[tw["season_type"] == "REG"].copy()
    tw["team"] = tw["team"].map(norm_team)
    tw["opponent_team"] = tw["opponent_team"].map(norm_team)
    g = _schedule()[["game_id", "home", "away", "home_score", "away_score"]]
    tw = tw.merge(g, on="game_id", how="left")
    tw["team_score"] = np.where(tw["team"] == tw["home"], tw["home_score"], tw["away_score"])
    tw["opp_score"] = np.where(tw["team"] == tw["home"], tw["away_score"], tw["home_score"])
    return_tds = _col(tw, "def_tds") + _col(tw, "special_teams_tds") + _col(tw, "fumble_recovery_tds")
    own = pd.DataFrame({"game_id": tw["game_id"], "team": tw["team"], "return_tds": return_tds, "safeties": _col(tw, "def_safeties")})
    opp = own.rename(columns={"team": "opponent_team", "return_tds": "opp_return_tds", "safeties": "opp_safeties"})
    tw = tw.merge(opp, on=["game_id", "opponent_team"], how="left")
    pa = tw["opp_score"] - 6.0 * tw["opp_return_tds"].fillna(0.0) - 2.0 * tw["opp_safeties"].fillna(0.0)
    blocks = _col(tw, "def_punt_blocks") + _col(tw, "def_fg_blocks") + _col(tw, "def_pat_blocks")
    pts = (_col(tw, "def_sacks") + 2.0 * _col(tw, "def_interceptions") + 2.0 * _col(tw, "fumble_recovery_opp") + 2.0 * _col(tw, "def_safeties")
           + 2.0 * blocks + 6.0 * return_tds.values + dst_points_allowed_tier(pa.clip(lower=0)))
    return pd.DataFrame({
        "season": tw["season"].astype(int), "week": tw["week"].astype(int), "team": tw["team"], "dk_pts": pts.round(2), "points_allowed": pa,
        "key": [f"dst {t}" for t in tw["team"]],
        "sacks": _col(tw, "def_sacks"), "ints": _col(tw, "def_interceptions"), "def_td": return_tds.values,
    })


# ── Salary history ──────────────────────────────────────────────────────────────────────────────────────
SALARY_COLUMNS = ["season", "week", "draft_group_id", "dk_id", "name", "pos", "team", "salary"]


def _load_group(draft_group_id: int) -> pd.DataFrame:
    """A draft group's player list: from the local DK cache when the file is there (any age), else straight from DraftKings."""
    from dfs import dk_client as K
    p = DK_CACHE_DIR / f"draftables_{draft_group_id}.json"
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
    else:
        data = K._csv_to_draftables(K._get_csv(K.CSV_URL.format(gid=draft_group_id)))
        try:
            p.write_text(json.dumps(data), encoding="utf-8")
        except OSError:
            pass
    return K.parse_draftables(data, draft_group_id)


def salary_rows(season: int, week: int, draft_group_id: int) -> pd.DataFrame:
    """The main-slate pool for one week: the draft group's players, restricted to the teams the schedule puts in the Sunday main slate."""
    df = _load_group(draft_group_id)
    teams = main_slate_teams(season, week)
    if not teams:
        raise ValueError(f"{season} week {week}: the schedule has no Sunday main-slate games (is games.csv current?)")
    pool = df[df["team"].isin(teams)].copy()
    have = set(pool["team"])
    if have != teams:
        raise ValueError(f"{season} week {week} (dg {draft_group_id}): the pool's teams are not the Sunday main slate's -- missing {sorted(teams - have)}, extra {sorted(set(df['team']) - teams)}")
    extra = set(df["team"]) - teams
    if extra:
        raise ValueError(f"{season} week {week} (dg {draft_group_id}) has teams outside the Sunday main slate: {sorted(extra)}. Not the main draft group.")
    pool.insert(0, "draft_group_id", draft_group_id)
    pool.insert(0, "week", week)
    pool.insert(0, "season", season)
    return pool[SALARY_COLUMNS].reset_index(drop=True)


def build_salary_history(slates: dict[tuple[int, int], int] | None = None, write: bool = True) -> pd.DataFrame:
    frames = [salary_rows(s, w, dg) for (s, w), dg in sorted((slates or MAIN_SLATES).items())]
    out = pd.concat(frames, ignore_index=True)
    if write:
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        out.to_csv(SALARY_FILE, index=False)
    return out


def add_week(season: int, week: int, draft_group_id: int) -> pd.DataFrame:
    """Add (or replace) one week in the committed salary file."""
    new = salary_rows(season, week, draft_group_id)
    old = load_dk_salaries()
    old = old[~((old.season == season) & (old.week == week))] if len(old) else old
    out = pd.concat([old, new], ignore_index=True).sort_values(["season", "week", "pos", "salary"], ascending=[True, True, True, False])
    out = out[SALARY_COLUMNS]                                  # the committed file never carries the archive's recorded-points column
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(SALARY_FILE, index=False)
    load_salaries.cache_clear()
    return new


def _read_salary_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=SALARY_COLUMNS + ["pts_recorded"])
    df = pd.read_csv(path, dtype={"dk_id": str})
    df["salary"] = df["salary"].astype(int)
    df = df[df["salary"] > 0].reset_index(drop=True)                      # a player with no price was not on the slate (see `parse_archive_page`)
    if "pts_recorded" not in df.columns:
        df["pts_recorded"] = np.nan
    return df


def load_dk_salaries() -> pd.DataFrame:
    """Only the committed file: the weeks whose salaries came straight from DraftKings (2026)."""
    return _read_salary_csv(SALARY_FILE)


@lru_cache(maxsize=1)
def load_salaries() -> pd.DataFrame:
    """Every stored week: the DraftKings-sourced weeks plus the archive's 2014-2021 weeks when `archive_salaries.csv` is on this machine."""
    frames = [f for f in (_read_salary_csv(SALARY_FILE), _read_salary_csv(ARCHIVE_FILE)) if len(f)]
    if not frames:
        return pd.DataFrame(columns=SALARY_COLUMNS + ["pts_recorded"])
    return pd.concat(frames, ignore_index=True).sort_values(["season", "week"], kind="stable").reset_index(drop=True)


def weeks_available() -> list[tuple[int, int]]:
    s = load_salaries()
    return sorted({(int(a), int(b)) for a, b in zip(s["season"], s["week"])})


# ── The 2014-2021 archive ────────────────────────────────────────────────────────────────────────────────
def _archive_page(season: int, week: int, delay: float = 1.0) -> str | None:
    """The archive's page for one week: from the local copy when there is one, else fetched (and kept), with a pause after each request."""
    import time
    import urllib.request
    path = ARCHIVE_RAW_DIR / f"{season}_{week:02d}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8", errors="replace")
    req = urllib.request.Request(ARCHIVE_URL.format(season=season, week=week), headers={"User-Agent": "Mozilla/5.0 (research; one request per second)"})
    try:
        text = urllib.request.urlopen(req, timeout=40).read().decode("latin-1")
    except Exception:  # noqa: BLE001 -- a week the site cannot serve is skipped, and the build reports it
        return None
    ARCHIVE_RAW_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    time.sleep(delay)
    return text


def parse_archive_page(text: str) -> pd.DataFrame:
    """Rows of one archive page (semicolon format: week;year;id;name;pos;team;h/a;opp;DK points;DK salary). `name` is turned into 'First Last'; the defense's position 'Def'
    becomes 'DST'; team codes follow DraftKings'. A row without a salary is dropped; a salary of 0 is kept here (`archive_rows` decides what to do with it); a blank points
    cell reads as 0 (the player did not play)."""
    rows = []
    for line in re.sub(r"<[^>]+>", "\n", text).splitlines():
        parts = [x.strip() for x in line.strip().split(";")]
        if len(parts) < 10 or not parts[0].isdigit():
            continue
        _wk, _yr, gid, name, pos, team, _ha, _opp, pts, sal = parts[:10]
        if not sal.replace(".", "", 1).isdigit():
            continue
        if pos.lower() == "def":
            pos, name = "DST", f"{name} DST"
        elif "," in name:
            last, first = [x.strip() for x in name.split(",", 1)]
            name = f"{first} {last}"
        try:
            points = float(pts)
        except ValueError:
            points = 0.0
        rows.append({"dk_id": f"rg{gid}", "name": name, "pos": pos.upper(), "team": norm_team(team.upper()), "salary": int(float(sal)), "pts_recorded": points})
    return pd.DataFrame(rows, columns=["dk_id", "name", "pos", "team", "salary", "pts_recorded"])


def archive_rows(season: int, week: int, page: pd.DataFrame) -> pd.DataFrame:
    """One archive week restricted to the Sunday main slate's teams, in the salary file's layout (+ `pts_recorded`). Raises when the page lacks a team the schedule puts in
    the slate, or a position that team needs (a QB, two RBs, three WRs, a TE, a defense). Priced players only: the archive lists 0 for a player DraftKings did not offer that week
    (about 270 skill players in 2014-2020), who could not be drafted and would be a free roster spot to an optimizer, so those rows count toward the page being complete but are
    not kept."""
    teams = main_slate_teams(season, week)
    if not teams:
        raise ValueError(f"{season} week {week}: no Sunday main-slate games on the schedule")
    pool = page[page["team"].isin(teams)].copy()
    need = {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "DST": 1}
    for t in sorted(teams):
        have = pool[pool["team"] == t]["pos"].value_counts()
        short = [p for p, n in need.items() if have.get(p, 0) < n]
        if short:
            raise ValueError(f"{season} week {week}: {t} has no {'/'.join(short)} in the archive page")
    pool = pool[pool["salary"] > 0].drop_duplicates("dk_id")
    pool.insert(0, "draft_group_id", np.nan)
    pool.insert(0, "week", week)
    pool.insert(0, "season", season)
    return pool[SALARY_COLUMNS + ["pts_recorded"]].reset_index(drop=True)


def build_archive(seasons: tuple[int, ...] | list[int] = ARCHIVE_SEASONS, delay: float = 1.0, write: bool = True, log=print) -> pd.DataFrame:
    """Every regular-season week of the given seasons that has a Sunday main slate, from the archive. Weeks the page or the schedule cannot support are listed, not guessed."""
    sch = _schedule()
    frames, skipped = [], []
    for season in seasons:
        n_before = len(frames)
        for week in sorted(sch[sch.season == season]["week"].unique()):
            if not main_slate_teams(season, int(week)):
                skipped.append((season, int(week), "no Sunday main-slate games"))
                continue
            text = _archive_page(season, int(week), delay)
            if not text:
                skipped.append((season, int(week), "page not available"))
                continue
            try:
                frames.append(archive_rows(season, int(week), parse_archive_page(text)))
            except ValueError as e:
                skipped.append((season, int(week), str(e)))
        log(f"{season}: {len(frames) - n_before} weeks")
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=SALARY_COLUMNS + ["pts_recorded"])
    for item in skipped:
        log(f"  skipped {item}")
    if write and len(out):
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        out.to_csv(ARCHIVE_FILE, index=False)
        load_salaries.cache_clear()
    return out


# ── A week's pool with the outcome ─────────────────────────────────────────────────────────────────────
def _attach_actuals(pool: pd.DataFrame, act: pd.DataFrame, dst: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """Add `pts` (DK points actually scored; 0 when the player did not play) and the stat columns to a salary pool. Exact name + team first, then the same team + position
    with a fuzzy name (DraftKings and nflverse spell a few players differently)."""
    a = act[(act.season == season) & (act.week == week)].drop_duplicates("key")
    d = dst[(dst.season == season) & (dst.week == week)].drop_duplicates("key")
    stat_cols = ["dk_pts", "tds", "pass_yds", "rush_yds", "rec", "rec_yds", "targets", "carries"]
    by_key = a.set_index("key")[stat_cols].to_dict("index")
    dst_by_key = d.set_index("key")[["dk_pts", "points_allowed", "sacks", "ints", "def_td"]].to_dict("index")
    a_by_team = {t: g for t, g in a.groupby("team")}
    rows, matched = [], []
    for r in pool.itertuples():
        if r.pos == "DST":
            hit = dst_by_key.get(f"dst {r.team}")
            rows.append({"pts": hit["dk_pts"] if hit else np.nan, "tds": hit["def_td"] if hit else 0.0, "points_allowed": hit["points_allowed"] if hit else np.nan})
            matched.append(hit is not None)
            continue
        key = f"{normalize_name(r.name)} {r.team}"
        hit = by_key.get(key)
        if hit is None and r.team in a_by_team:
            g = a_by_team[r.team]
            names = [normalize_name(n) for n in g["name"]]
            cand = fuzzy_match(normalize_name(r.name), names, cutoff=0.88)
            if cand is not None:
                hit = g.iloc[names.index(cand)][stat_cols].to_dict()
        if hit is None:
            rows.append({"pts": 0.0})
            matched.append(False)
        else:
            rows.append({"pts": float(hit["dk_pts"]), "tds": float(hit["tds"]), "pass_yds": float(hit["pass_yds"]), "rush_yds": float(hit["rush_yds"]), "rec": float(hit["rec"]),
                         "rec_yds": float(hit["rec_yds"]), "targets": float(hit["targets"]), "carries": float(hit["carries"])})
            matched.append(True)
    out = pd.concat([pool.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    out["played"] = matched
    for c in ("tds", "pass_yds", "rush_yds", "rec", "rec_yds", "targets", "carries"):
        out[c] = out[c].fillna(0.0) if c in out else 0.0
    out["pts"] = out["pts"].fillna(0.0)
    return out


def week_pool(season: int, week: int) -> pd.DataFrame:
    """The main-slate pool of one week with everything known before kickoff (salary, opponent, home/away, spread, total, implied team total, roof, wind) and what each
    player scored. One row per DraftKings player; players who did not play score 0."""
    sal = load_salaries()
    sal = sal[(sal.season == season) & (sal.week == week)]
    if sal.empty:
        raise ValueError(f"no salary pool stored for {season} week {week}")
    ctx = team_context(season, week)
    pool = sal.merge(ctx[["team", "opp", "is_home", "spread", "total", "team_total", "opp_total", "roof", "temp", "wind", "game_id"]], on="team", how="left")
    if pool["pts_recorded"].notna().any():
        # an archive week: DraftKings' own recorded points; the box-score columns are not part of the archive
        pool["pts"] = pool["pts_recorded"].fillna(0.0).astype(float)
        pool["played"] = pool["pts_recorded"].notna()
        for c in ("tds", "pass_yds", "rush_yds", "rec", "rec_yds", "targets", "carries"):
            pool[c] = np.nan
    else:
        pool = _attach_actuals(pool.drop(columns=["pts_recorded"]), player_actuals(season), dst_actuals(season), season, week)
    pool["dk_id"] = pool["dk_id"].astype(str)
    return pool.drop(columns=["pts_recorded"], errors="ignore").reset_index(drop=True)


def unmatched_with_salary(season: int, week: int, min_salary: int = 4000) -> pd.DataFrame:
    """Priced players the box scores could not be matched to: the QA list (a real inactive shows up here too)."""
    p = week_pool(season, week)
    return p[(~p["played"]) & (p["salary"] >= min_salary)][["name", "pos", "team", "salary"]]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Maintain the historical Sunday-main-slate salary file.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build", help="rebuild data/dfs/history/main_salaries.csv from MAIN_SLATES")
    r = sub.add_parser("archive", help="build the local 2014-2021 archive file (one request per second, pages kept locally)")
    r.add_argument("--seasons", type=int, nargs="*", default=list(ARCHIVE_SEASONS))
    r.add_argument("--delay", type=float, default=1.0)
    a = sub.add_parser("add", help="add one week")
    a.add_argument("--season", type=int, required=True)
    a.add_argument("--week", type=int, required=True)
    a.add_argument("--dg", type=int, required=True, help="DraftKings draft group id of that week's Sunday main Classic slate")
    args = ap.parse_args(argv)
    if args.cmd == "build":
        df = build_salary_history()
        print(f"wrote {len(df)} rows for {df.groupby(['season', 'week']).ngroups} weeks to {SALARY_FILE}")
    elif args.cmd == "archive":
        df = build_archive(args.seasons, args.delay)
        print(f"wrote {len(df)} rows for {df.groupby(['season', 'week']).ngroups} weeks to {ARCHIVE_FILE}")
    else:
        new = add_week(args.season, args.week, args.dg)
        print(f"added {len(new)} players for {args.season} week {args.week}")


if __name__ == "__main__":
    main()
