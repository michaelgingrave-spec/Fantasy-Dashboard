"""What each offense and defense had been doing BEFORE a given week: the game-environment side of the Millionaire Maker study.

Every number here is as of kickoff: only the team's earlier games of the same season count (never the game itself), pulled toward last season's finish when the season is young.

  * points for / against a game, from the schedule;
  * DraftKings points a defense has allowed to each position (QB, RB, WR, TE) -- the "fantasy points allowed" table the Data Suite also publishes -- from the archive pages (2014-2021)
    and the nflverse box scores (2022 on);
  * EPA per dropback and per rush each offense gained and each defense allowed (the Data Suite's defensive-efficiency columns), from the nflverse play-by-play wherever that is cached
    (2021 on). Rebuilt from play-by-play, it matches the Data Suite's own weekly team-defense table (`data/dfs/matchup/team-defense-box_*_week.csv`, 2022-2025): see `compare_to_data_suite`.

`attach(pool, season, week)` adds the columns a pool needs to apply environment rules (`env_fpa`, `env_fpa_rank`, `env_epa`, `env_epa_rank`, `env_pa_rank`, `team_off_rank`): for each player the
quality of the unit he faces, ranked 1 = the weakest opponent for him (for a defense: the weakest OFFENSE it faces).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from dfs import history as H
from dfs.names import norm_team

CACHE_FILE = H.HISTORY_DIR / "team_form_local.csv"          # the as-of table; rebuilt when the schedule file or a season's stats are newer
POS = ("QB", "RB", "WR", "TE")
PRIOR_GAMES = 4.0        # a team's earlier games count at least this many games' worth against last season's finish
PRIOR_WEIGHT = 0.5       # how much of last season's rate (against the league's) the starting prior keeps
PASS_PLAYS, RUSH_PLAYS = 35.0, 25.0     # plays per game, for turning "4 games' worth" into a play weight


# ── Raw material: points by position and EPA by team-game ────────────────────────────────────────────────
def _player_points(season: int) -> pd.DataFrame:
    """season, week, team, pos, pts for every QB / RB / WR / TE with a recorded score that season (all teams, all games)."""
    frames = []
    if season in H.ARCHIVE_SEASONS:
        for path in sorted(H.ARCHIVE_RAW_DIR.glob(f"{season}_*.txt")):
            week = int(path.stem.split("_")[1])
            page = H.parse_archive_page(path.read_text(encoding="utf-8", errors="replace"))
            page = page[page["pos"].isin(POS)]
            frames.append(pd.DataFrame({"season": season, "week": week, "team": page["team"].to_numpy(), "pos": page["pos"].to_numpy(), "pts": page["pts_recorded"].to_numpy()}))
    else:
        a = H.player_actuals(season)
        pos = a["nf_pos"].replace({"FB": "RB"})
        a = a[pos.isin(POS)].assign(pos=pos[pos.isin(POS)])
        frames.append(a[["season", "week", "team", "pos"]].assign(pts=a["dk_pts"].to_numpy()))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["season", "week", "team", "pos", "pts"])


def _pbp_epa(season: int) -> pd.DataFrame:
    """Per team-game offense EPA sums and play counts (pass = QB dropbacks incl. sacks and scrambles, rush = designed runs) -- empty when the season is not cached."""
    path = H._nflverse_dir() / f"play_by_play_{season}.parquet"
    if not path.exists():
        return pd.DataFrame()
    d = pd.read_parquet(path, columns=["week", "season_type", "posteam", "defteam", "play_type", "epa", "qb_dropback", "rush"])
    d = d[(d["season_type"] == "REG") & d["epa"].notna() & d["posteam"].notna() & d["play_type"].isin(["pass", "run"])]
    drop = d["qb_dropback"].fillna(0) == 1
    rush = (d["rush"].fillna(0) == 1) & ~drop
    out = []
    for kind, m in (("pass", drop), ("rush", rush)):
        g = d[m].groupby(["week", "posteam", "defteam"])["epa"].agg(["sum", "count"]).reset_index()
        g.columns = ["week", "off", "def", f"{kind}_sum", f"{kind}_n"]
        out.append(g.set_index(["week", "off", "def"]))
    e = out[0].join(out[1], how="outer").fillna(0.0).reset_index()
    e["off"], e["def"] = e["off"].map(norm_team), e["def"].map(norm_team)
    e.insert(0, "season", season)
    return e


def team_games(seasons: list[int]) -> pd.DataFrame:
    """One row per team-game: pf, pa, the DK points allowed to each position (`fpa_qb` ...), the EPA sums / counts of this team's offense (`o_pass_sum`, `o_pass_n`, `o_rush_*`) and of
    the offense it faced (`d_pass_sum` ... = what this defense allowed). EPA columns are NaN where play-by-play is not cached."""
    sch = H._schedule()
    sch = sch[sch["season"].isin(seasons) & sch["home_score"].notna()]
    home = pd.DataFrame({"season": sch["season"], "week": sch["week"], "team": sch["home"], "opp": sch["away"], "pf": sch["home_score"], "pa": sch["away_score"]})
    away = pd.DataFrame({"season": sch["season"], "week": sch["week"], "team": sch["away"], "opp": sch["home"], "pf": sch["away_score"], "pa": sch["home_score"]})
    tg = pd.concat([home, away], ignore_index=True)
    pts = pd.concat([_player_points(s) for s in seasons], ignore_index=True)
    allowed = pts.groupby(["season", "week", "team", "pos"])["pts"].sum().unstack("pos").reindex(columns=list(POS))
    allowed.columns = [f"fpa_{p.lower()}" for p in allowed.columns]
    allowed = allowed.reset_index().rename(columns={"team": "opp"})                               # the points the OPPONENT's players scored = what `team`'s defense allowed
    tg = tg.merge(allowed, on=["season", "week", "opp"], how="left")
    epa = pd.concat([_pbp_epa(s) for s in seasons], ignore_index=True)
    for c in ("o_pass_sum", "o_pass_n", "o_rush_sum", "o_rush_n", "d_pass_sum", "d_pass_n", "d_rush_sum", "d_rush_n"):
        tg[c] = np.nan
    if len(epa):
        o = epa.rename(columns={"off": "team", "def": "opp", "pass_sum": "o_pass_sum", "pass_n": "o_pass_n", "rush_sum": "o_rush_sum", "rush_n": "o_rush_n"})
        d = epa.rename(columns={"def": "team", "off": "opp", "pass_sum": "d_pass_sum", "pass_n": "d_pass_n", "rush_sum": "d_rush_sum", "rush_n": "d_rush_n"})
        tg = tg.drop(columns=[c for c in tg.columns if c.startswith(("o_", "d_"))])
        tg = tg.merge(o, on=["season", "week", "team", "opp"], how="left").merge(d, on=["season", "week", "team", "opp"], how="left")
    return tg.sort_values(["season", "week", "team"]).reset_index(drop=True)


# ── As-of table ──────────────────────────────────────────────────────────────────────────────────────────
RATE_COLS = {  # as-of column -> (sum column, count column, games' worth of plays for the prior)
    "def_pass_epa": ("d_pass_sum", "d_pass_n", PASS_PLAYS), "def_rush_epa": ("d_rush_sum", "d_rush_n", RUSH_PLAYS),
    "off_pass_epa": ("o_pass_sum", "o_pass_n", PASS_PLAYS), "off_rush_epa": ("o_rush_sum", "o_rush_n", RUSH_PLAYS),
}
GAME_COLS = {"off_pf": "pf", "def_pa": "pa", "def_fpa_qb": "fpa_qb", "def_fpa_rb": "fpa_rb", "def_fpa_wr": "fpa_wr", "def_fpa_te": "fpa_te"}


def _final_rates(tg: pd.DataFrame, season: int) -> dict[str, pd.Series]:
    """A season's end-of-year value per team for every as-of column (the starting prior of the next season), pulled halfway to the league mean."""
    g = tg[tg["season"] == season]
    out = {}
    if g.empty:
        return out
    for col, (s, n, _) in RATE_COLS.items():
        by = g.groupby("team")[[s, n]].sum()
        rate = by[s] / by[n].replace(0, np.nan)
        league = g[s].sum() / g[n].sum() if g[n].sum() > 0 else np.nan
        out[col] = PRIOR_WEIGHT * rate + (1 - PRIOR_WEIGHT) * league
    for col, src in GAME_COLS.items():
        by = g.groupby("team")[src].mean()
        out[col] = PRIOR_WEIGHT * by + (1 - PRIOR_WEIGHT) * g[src].mean()
    return out


def build_as_of(tg: pd.DataFrame | None = None, seasons: list[int] | None = None) -> pd.DataFrame:
    """For every (season, week, team): the as-of columns of `RATE_COLS` and `GAME_COLS`, their league ranks, and the number of earlier games. A team's earlier games of the season are
    averaged with a prior worth `PRIOR_GAMES` games: last season's finish (pulled halfway to the league), or the league's mean so far when there is no last season."""
    seasons = seasons or sorted(H._schedule().query("season >= 2014")["season"].unique().tolist())
    tg = team_games(seasons) if tg is None else tg
    rows = []
    prior_cache: dict[int, dict] = {}
    for season in sorted(tg["season"].unique()):
        if season - 1 not in prior_cache:
            prior_cache[season - 1] = _final_rates(tg, season - 1)
        prior = prior_cache[season - 1]
        g = tg[tg["season"] == season]
        weeks = sorted(g["week"].unique())
        teams = sorted(g["team"].unique())
        for week in range(1, max(weeks) + 2):                    # one week past the last game played: the as-of form for the slate being built
            prev = g[g["week"] < week]
            for team in teams:
                mine = prev[prev["team"] == team]
                row = {"season": season, "week": week, "team": team, "games": len(mine)}
                for col, (s, n, per_game) in RATE_COLS.items():
                    lg = prev[s].sum() / prev[n].sum() if prev[n].sum() > 0 else np.nan
                    p = prior.get(col, pd.Series(dtype=float)).get(team, np.nan)
                    p = lg if pd.isna(p) else p
                    k = PRIOR_GAMES * per_game
                    num, den = mine[s].sum(), mine[n].sum()
                    row[col] = (num + k * p) / (den + k) if pd.notna(p) else (num / den if den > 0 else np.nan)
                for col, src in GAME_COLS.items():
                    lg = prev[src].mean() if len(prev) else np.nan
                    p = prior.get(col, pd.Series(dtype=float)).get(team, np.nan)
                    p = lg if pd.isna(p) else p
                    vals = mine[src].dropna()
                    row[col] = (vals.sum() + PRIOR_GAMES * p) / (len(vals) + PRIOR_GAMES) if pd.notna(p) else (vals.mean() if len(vals) else np.nan)
                rows.append(row)
    t = pd.DataFrame(rows)
    t["off_epa"] = (t["off_pass_epa"] * PASS_PLAYS + t["off_rush_epa"] * RUSH_PLAYS) / (PASS_PLAYS + RUSH_PLAYS)
    t["def_epa"] = (t["def_pass_epa"] * PASS_PLAYS + t["def_rush_epa"] * RUSH_PLAYS) / (PASS_PLAYS + RUSH_PLAYS)
    by = t.groupby(["season", "week"])
    for col in ("def_pass_epa", "def_rush_epa", "def_epa", "def_pa", "def_fpa_qb", "def_fpa_rb", "def_fpa_wr", "def_fpa_te"):
        t[col + "_rank"] = by[col].rank(ascending=False, method="min")            # 1 = weakest defense (allows the most)
    for col in ("off_epa", "off_pf"):
        t[col + "_rank"] = by[col].rank(ascending=False, method="min")            # 1 = best offense
        t[col + "_weak_rank"] = by[col].rank(ascending=True, method="min")        # 1 = weakest offense
    return t


def _stamp() -> float:
    files = [H._nflverse_dir() / "games.csv"] + sorted(H._nflverse_dir().glob("stats_player_week_*.parquet")) + sorted(H._nflverse_dir().glob("play_by_play_*.parquet"))
    return max((f.stat().st_mtime for f in files if f.exists()), default=0.0)


@lru_cache(maxsize=1)
def as_of_table() -> pd.DataFrame:
    """The as-of table, from its cache file when that is newer than every input, else rebuilt (a minute or two) and saved."""
    if CACHE_FILE.exists() and CACHE_FILE.stat().st_mtime >= _stamp():
        return pd.read_csv(CACHE_FILE)
    t = build_as_of()
    H.HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    t.to_csv(CACHE_FILE, index=False)
    return t


@lru_cache(maxsize=4)
def live_table(season: int) -> pd.DataFrame:
    """The as-of table of one season on its own (last season only as the prior): what the hosted app needs for the slate it is building. No play-by-play is required (the EPA
    columns stay NaN without it); the points-allowed columns come from the nflverse box scores, downloaded on demand."""
    return build_as_of(team_games([season - 1, season]), [season - 1, season]).query("season == @season").reset_index(drop=True)


# ── Joining it to a pool ─────────────────────────────────────────────────────────────────────────────────
ENV_COLUMNS = ["env_fpa", "env_fpa_rank", "env_epa", "env_epa_rank", "env_pa_rank", "team_off_rank", "team_off_epa"]


def attach(pool: pd.DataFrame, season: int, week: int, table: pd.DataFrame | None = None) -> pd.DataFrame:
    """`pool` (needs team, opp, pos) plus the environment columns, all as of this week's kickoff. For every player they describe the unit he faces, rank 1 = the weakest for him:
    QB/RB/WR/TE: `env_fpa` = DK points his opponent allows per game to his position, `env_epa` = EPA its defense allows (per dropback for QB/WR/TE, per rush for RB);
    DST: `env_fpa` = points per game the opposing offense scores, `env_epa` = the opposing offense's EPA per play. `env_pa_rank` = the opponent's points-allowed rank (DST: points scored
    rank of the offense it faces); `team_off_rank` = the player's own offense by points scored per game (1 = best) and `team_off_epa` its EPA per play. EPA columns are NaN before 2021
    (no play-by-play cached); everything else covers every season. Missing values stay NaN."""
    t = (as_of_table() if table is None else table)
    t = t[(t["season"] == season) & (t["week"] == week)].set_index("team")
    out = pool.copy()
    for c in ENV_COLUMNS:
        out[c] = np.nan
    if t.empty:
        return out
    opp, pos = out["opp"].astype(str).str.lstrip("@"), out["pos"]

    def pick(col: str, who: pd.Series) -> np.ndarray:
        return who.map(t[col]).to_numpy(dtype=float) if col in t.columns else np.full(len(out), np.nan)

    for p in POS:
        m = (pos == p).to_numpy()
        if not m.any():
            continue
        out.loc[m, "env_fpa"] = pick(f"def_fpa_{p.lower()}", opp)[m]
        out.loc[m, "env_fpa_rank"] = pick(f"def_fpa_{p.lower()}_rank", opp)[m]
        col = "def_rush_epa" if p == "RB" else "def_pass_epa"
        out.loc[m, "env_epa"] = pick(col, opp)[m]
        out.loc[m, "env_epa_rank"] = pick(col + "_rank", opp)[m]
    d = (pos == "DST").to_numpy()
    if d.any():
        out.loc[d, "env_fpa"] = pick("off_pf", opp)[d]
        out.loc[d, "env_fpa_rank"] = pick("off_pf_weak_rank", opp)[d]
        out.loc[d, "env_epa"] = pick("off_epa", opp)[d]
        out.loc[d, "env_epa_rank"] = pick("off_epa_weak_rank", opp)[d]
    out["env_pa_rank"] = np.where(d, pick("off_pf_weak_rank", opp), pick("def_pa_rank", opp))
    out["team_off_rank"] = pick("off_pf_rank", out["team"])
    out["team_off_epa"] = pick("off_epa", out["team"])
    return out


# ── Does the rebuilt EPA match the Data Suite's own table? ──────────────────────────────────────────────
def compare_to_data_suite(season: int) -> pd.DataFrame:
    """Per team-game of a season, this module's defensive EPA per dropback / per rush allowed against the Data Suite's weekly team-defense table (EPA/DB and EPA/RU, the same quantities)."""
    from matchup_model.opp import ds_features as DS
    ds = DS.read_weekly("team-defense-box", [season])
    ds = ds.assign(team=ds["Name"].map(norm_team), week=pd.to_numeric(ds["WEEK"]).astype(int))
    tg = team_games([season])
    mine = tg.assign(pass_epa=tg["d_pass_sum"] / tg["d_pass_n"].replace(0, np.nan), rush_epa=tg["d_rush_sum"] / tg["d_rush_n"].replace(0, np.nan))
    theirs = ds[["team", "week"]].assign(**{c: pd.to_numeric(ds[c], errors="coerce") for c in ("EPA/DB", "EPA/RU")})
    m = mine.merge(theirs, on=["team", "week"], how="inner")
    return m[["season", "week", "team", "pass_epa", "EPA/DB", "rush_epa", "EPA/RU"]]


if __name__ == "__main__":
    t = build_as_of()
    t.to_csv(CACHE_FILE, index=False)
    print(f"wrote {len(t)} team-weeks to {CACHE_FILE}")
