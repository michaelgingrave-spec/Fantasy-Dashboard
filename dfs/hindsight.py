"""Hindsight study for the Millionaire Maker tool: the best lineup of each week, and what the best lineups have in common.

For every week of the historical salary file (`dfs.history`), the lineup that actually scored the most DraftKings points under the contest's own rules (nine slots,
$50,000, QB / 2-3 RB / 3-4 WR / 1-2 TE / DST, FLEX = the extra RB / WR / TE) is found exactly with the same MILP the optimizer uses, only with the real points as the
objective. A few more near-best lineups per week, each at least `min_uniques` players apart, are kept alongside it so the shared structure does not hang on one player.

`lineup_features` turns a lineup into the structural facts a builder can act on before kickoff: how it stacks the QB, whether it brings the other side's pass catcher
back, how many teams and games it spans, how the salary is spread, what kind of game environment the players come from.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dfs import history as H
from dfs import team_form
from dfs.config import SALARY_CAP, projection_path
from dfs.lineup_milp import PoolModel, Rules, solve_portfolio

FEATURES_FILE = H.HISTORY_DIR / "mm_hindsight_features.csv"          # structure only, no player names: committed
LINEUPS_FILE = H.HISTORY_DIR / "mm_hindsight_lineups_local.csv"      # the same rows with the players: stays on this machine (the archive weeks' names come from it)
MIN_MAIN_GAMES = 6                                                    # a week whose Sunday window holds fewer games than this is not a Millionaire Maker slate


# ── The pool of a week with everything known before kickoff ─────────────────────────────────────────────
def pool_context(season: int, week: int) -> pd.DataFrame:
    """`history.week_pool` plus the pre-game columns the features use: the FantasyPoints projection and a chalk-ownership estimate (2026 weeks, where the projection
    files exist), salary and projection ranks within the position, and the game-total / implied-total ranks across the slate (1 = highest)."""
    pool = H.week_pool(season, week)
    pool["proj"] = np.nan
    pool["own_pct"] = np.nan
    path = projection_path(week)
    if season == 2026 and path.exists():
        from dfs.projections import load_weekly_projections
        from dfs.slate import build_slate
        try:
            slate, _ = build_slate(pool[["dk_id", "name", "pos", "team", "opp", "salary"]], load_weekly_projections(week, path))
            pool["proj"] = pool["dk_id"].map(slate.set_index("dk_id")["proj"])
            from dfs.ownership import chalk_ownership
            own = chalk_ownership(pool[["dk_id", "pos", "team", "salary", "proj"]].dropna(subset=["proj"]), week)
            pool["own_pct"] = pool["dk_id"].map(own.set_index("dk_id")["own_pct"])
        except Exception:  # noqa: BLE001 -- a missing or odd projection file leaves the pre-game columns empty, the hindsight part still works
            pass
    pool["sal_rank_pos"] = pool.groupby("pos")["salary"].rank(ascending=False, method="min")
    pool["proj_rank_pos"] = pool.groupby("pos")["proj"].rank(ascending=False, method="min")
    games = pool.drop_duplicates("game_id")[["game_id", "total"]]
    games = games.assign(game_total_rank=games["total"].rank(ascending=False, method="min"))
    pool = pool.merge(games[["game_id", "game_total_rank"]], on="game_id", how="left")
    teams = pool.drop_duplicates("team")[["team", "team_total"]]
    teams = teams.assign(team_total_rank=teams["team_total"].rank(ascending=False, method="min"))
    pool = pool.merge(teams[["team", "team_total_rank"]], on="team", how="left")
    return team_form.attach(pool, season, week)                       # the opponent's / own offense's as-of form: env_fpa, env_fpa_rank, env_epa, ..., team_off_rank


# ── Best lineups ────────────────────────────────────────────────────────────────────────────────────────
def study_weeks() -> list[tuple[int, int]]:
    """Every stored week whose Sunday window holds a real main slate (at least `MIN_MAIN_GAMES` games)."""
    return [(s, w) for s, w in H.weeks_available() if len(H.main_slate_games(s, w)) >= MIN_MAIN_GAMES]


def best_index_sets(pool: pd.DataFrame, k: int = 1, min_uniques: int = 1) -> list[np.ndarray]:
    """The `k` highest-scoring lineups the pool allows under DraftKings' rules alone (row positions in `pool`), each at least `min_uniques` players away from every better one,
    best first. The score is what the players actually scored."""
    pool = pool.reset_index(drop=True)
    pts = pool["pts"].to_numpy(dtype=float)
    sols = solve_portfolio(PoolModel(pool, Rules()), pts, n=k, min_uniques=min_uniques)
    return sorted(sols, key=lambda ix: -float(pts[ix].sum())) if k > 1 else sols


# ── Features of a lineup ────────────────────────────────────────────────────────────────────────────────
def _env(frame: pd.DataFrame, name: str) -> pd.Series:
    """A pool column the pre-game environment adds (`team_form.attach`), all-NaN when the pool has not been given it."""
    return frame[name] if name in frame.columns else pd.Series(np.nan, index=frame.index)



def lineup_features(ids: list[str], pool: pd.DataFrame) -> dict:
    """Structural facts about one lineup (its nine dk_ids) read against the week's pool. Only the `score` group uses what happened in the games; every other number is
    known before kickoff (salaries, schedule, closing lines, and, for 2026, the FantasyPoints projection and the ownership estimate)."""
    P = pool.set_index("dk_id")
    lu = P.loc[ids]
    qb = lu[lu["pos"] == "QB"].iloc[0]
    skill = lu[lu["pos"].isin(["RB", "WR", "TE"])]
    off = lu[lu["pos"] != "DST"]
    dst = lu[lu["pos"] == "DST"].iloc[0]
    qb_team, qb_opp = qb["team"], qb["opp"]
    mates = lu[(lu["team"] == qb_team) & (lu.index != qb.name)]
    opp_side = lu[lu["team"] == qb_opp]
    rb_ids = lu.index[lu["pos"] == "RB"].tolist()
    wr_ids = lu.index[lu["pos"] == "WR"].tolist()
    te_ids = lu.index[lu["pos"] == "TE"].tolist()
    # FLEX = the RB / WR / TE beyond the position minimums
    flex_pos = "RB" if len(rb_ids) == 3 else "WR" if len(wr_ids) == 4 else "TE"
    sal = off["salary"].astype(float)
    sk_sal = skill["salary"].astype(float)
    games = lu["game_id"].nunique()
    per_team = lu["team"].value_counts()
    per_game = lu["game_id"].value_counts()
    top_proj = lu["proj_rank_pos"]
    f = {
        # composition
        "n_rb": len(rb_ids), "n_wr": len(wr_ids), "n_te": len(te_ids), "flex_pos": flex_pos,
        # how it correlates
        "qb_team": qb_team, "qb_name": qb["name"],
        "stack_wr_te": int(mates["pos"].isin(["WR", "TE"]).sum()), "stack_rb": int((mates["pos"] == "RB").sum()),
        "stack_skill": int(mates["pos"].isin(["WR", "TE", "RB"]).sum()), "stack_dst": int(dst["team"] == qb_team),
        "bring_back": int(opp_side["pos"].isin(["WR", "TE", "RB"]).sum()), "bring_back_wr_te": int(opp_side["pos"].isin(["WR", "TE"]).sum()),
        "qb_game_players": int(lu["team"].isin([qb_team, qb_opp]).sum()),
        "n_teams": int(lu["team"].nunique()), "n_games": int(games), "max_from_team": int(per_team.max()), "max_from_game": int(per_game.max()),
        "max_skill_from_team": int(skill["team"].value_counts().max()),
        "dst_faces_lineup": int((off["team"] == dst["opp"]).sum()), "dst_faces_qb": int(dst["opp"] == qb_team),
        "dst_team_skill": int((skill["team"] == dst["team"]).sum()),
        # salary
        "salary": int(lu["salary"].sum()), "sal_left": int(SALARY_CAP - lu["salary"].sum()),
        "sal_qb": int(qb["salary"]), "sal_rb": int(lu.loc[rb_ids, "salary"].sum()), "sal_wr": int(lu.loc[wr_ids, "salary"].sum()), "sal_te": int(lu.loc[te_ids, "salary"].sum()),
        "sal_dst": int(dst["salary"]),
        "n_8k": int((sk_sal >= 8000).sum()), "n_7k": int((sk_sal >= 7000).sum()), "n_le_4k": int((sk_sal <= 4000).sum()), "n_le_5k": int((sk_sal <= 5000).sum()),
        "min_skill_sal": int(sk_sal.min()), "max_skill_sal": int(sk_sal.max()), "sal_sd": float(sal.std(ddof=0)),
        "qb_sal_rank": int(qb["sal_rank_pos"]), "dst_sal_rank": int(dst["sal_rank_pos"]),
        # game environment
        "qb_total": float(qb["total"]), "qb_total_rank": int(qb["game_total_rank"]), "qb_team_total": float(qb["team_total"]), "qb_tt_rank": int(qb["team_total_rank"]),
        "qb_spread": float(qb["spread"]), "qb_fav": int(qb["spread"] < 0), "qb_home": int(qb["is_home"]),
        "mean_game_total": float(off["total"].mean()), "n_top3_games": int((off["game_total_rank"] <= 3).sum()),
        "n_top_half_games": int((off["game_total_rank"] <= max(1, pool["game_id"].nunique() // 2)).sum()),
        "mean_team_total": float(off["team_total"].mean()), "n_on_favorites": int((off["spread"] < 0).sum()),
        # pre-game price of the players (2026 weeks)
        "proj_sum": float(lu["proj"].sum()) if lu["proj"].notna().all() else np.nan,
        "own_sum": float(lu["own_pct"].sum()) if lu["own_pct"].notna().all() else np.nan,
        "n_low_own": int((lu["own_pct"] <= 5).sum()) if lu["own_pct"].notna().all() else np.nan,
        "n_chalk": int((lu["own_pct"] >= 15).sum()) if lu["own_pct"].notna().all() else np.nan,
        "qb_proj_rank": float(qb["proj_rank_pos"]) if pd.notna(qb["proj_rank_pos"]) else np.nan,
        "avg_proj_rank": float(top_proj[lu["pos"] != "DST"].mean()) if top_proj.notna().all() else np.nan,
        # more structure
        "n_mini_stacks": int(sum(1 for t, c in off["team"].value_counts().items() if t != qb_team and c >= 2)),
        "dst_spread": float(dst["spread"]), "dst_fav": int(dst["spread"] < 0),
        "n_top_sal_skill": int((skill["sal_rank_pos"] <= 10).sum()), "mean_sal_rank_skill": float(skill["sal_rank_pos"].mean()),
        # the units the players face (as of kickoff; rank 1 = the weakest opponent for the player's position) and the offenses they play for
        "qb_fpa_rank": float(qb.get("env_fpa_rank", np.nan)), "qb_epa_rank": float(qb.get("env_epa_rank", np.nan)), "qb_pa_rank": float(qb.get("env_pa_rank", np.nan)),
        "mean_opp_fpa_rank": float(_env(off, "env_fpa_rank").mean()), "n_weak_opp": int((_env(off, "env_fpa_rank") <= 8).sum()), "n_strong_opp": int((_env(off, "env_fpa_rank") >= 25).sum()),
        "dst_opp_off_rank": float(dst.get("env_fpa_rank", np.nan)), "mean_team_off_rank": float(_env(off, "team_off_rank").mean()), "n_top_offense": int((_env(off, "team_off_rank") <= 8).sum()),
        "n_high_team_total": int((off["team_total"] >= 25).sum()), "n_low_team_total": int((off["team_total"] <= 20).sum()),
        # what happened
        "score": float(lu["pts"].sum()), "n_20plus": int((lu["pts"] >= 20).sum()), "n_tds": float(lu["tds"].sum()) if lu["tds"].notna().all() else np.nan,
        "top_player_share": float(lu["pts"].max() / lu["pts"].sum()) if lu["pts"].sum() > 0 else np.nan,
    }
    return f


# ── The study ───────────────────────────────────────────────────────────────────────────────────────────
SLOT_POS = {"QB": 0, "RB": 1, "WR": 2, "TE": 3, "DST": 4}


def _players_text(lu: pd.DataFrame) -> str:
    lu = lu.assign(_o=lu["pos"].map(SLOT_POS)).sort_values(["_o", "pts"], ascending=[True, False])
    return "; ".join(f"{r.pos} {r.name} ({r.team}, ${int(r.salary):,}, {r.pts:.1f})" for r in lu.itertuples())


def study(weeks: list[tuple[int, int]] | None = None, k: int = 25, min_uniques: int = 3) -> pd.DataFrame:
    """One row per (week, rank): the best lineup of each week and the next best `k - 1` that differ by at least `min_uniques` players, with `lineup_features`, the nine dk_ids
    and the players. Rank 1 is the true optimum."""
    rows = []
    for season, week in (weeks or study_weeks()):
        pool = pool_context(season, week).reset_index(drop=True)
        for rank, ix in enumerate(best_index_sets(pool, k, min_uniques), start=1):
            ids = pool["dk_id"].iloc[ix].tolist()
            rows.append({"season": season, "week": week, "rank": rank, **lineup_features(ids, pool), "dk_ids": "|".join(ids), "players": _players_text(pool.iloc[ix])})
    return pd.DataFrame(rows)


def save_study(df: pd.DataFrame) -> None:
    H.HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(LINEUPS_FILE, index=False)
    df.drop(columns=["dk_ids", "players", "qb_name"]).to_csv(FEATURES_FILE, index=False)


def read_features() -> pd.DataFrame:
    """The committed structure-only table of every week's best lineups (what `save_study` wrote)."""
    return pd.read_csv(FEATURES_FILE)


def main() -> None:
    df = study()
    save_study(df)
    top = df[df["rank"] == 1]
    print(f"wrote {len(df)} lineups ({df.groupby(['season', 'week']).ngroups} weeks): features to {FEATURES_FILE}, players to {LINEUPS_FILE}")
    print(top[["season", "week", "score", "salary", "stack_wr_te", "bring_back", "n_teams", "n_games"]].tail(15).to_string(index=False))


if __name__ == "__main__":
    main()
