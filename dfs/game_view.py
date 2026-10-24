"""Game Projections: pick a game, see our projection for every player in it, and (by default) only the players FantasyPoints also projects.

"Our projection" is the number the rest of the dashboard uses: `matchup_model.opp.blend.blended_line` -- the opportunity model (team pace x pass rate x Vegas x usage
share, the weekly injury report folded in) blended per position with the player's trailing DK average and nudged toward FantasyPoints when the two disagree a lot --
together with its projected box-score line (targets, receptions, yards, TDs, ...). The pure functions here build the table; `render()` is the Streamlit part.
"""
from __future__ import annotations

import hashlib
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st

from dfs.names import normalize_name

POS = ("QB", "RB", "WR", "TE")
ROSTER_GAMES = 3                  # a player is on the team's roster here if he played in one of its last three games (and his latest game was for it)
STAT_KEYS = ("pass_att", "pass_yds", "pass_td", "int", "rush_att", "rush_yds", "rush_td", "tgt", "rec", "rec_yds", "rec_td")
# the stat columns each position's table shows: (key, label)
_CATCHES = [("rec", "Rec"), ("rec_yds", "Rec yds"), ("rec_td", "Rec TD"), ("any_td", "Any TD %")]
GROUP_COLS = {
    "QB": [("pass_att", "Pass att"), ("pass_yds", "Pass yds"), ("pass_td", "Pass TD"), ("int", "INT"), ("rush_yds", "Rush yds"), ("rush_td", "Rush TD")],
    "RB": [("rush_att", "Carries"), ("rush_yds", "Rush yds"), ("rush_td", "Rush TD")] + _CATCHES,          # the RB model projects catches, not targets
    "WR": [("tgt", "Targets")] + _CATCHES,
    "TE": [("tgt", "Targets")] + _CATCHES,
}
_PRACTICE = {"did not participate in practice": "DNP", "limited participation in practice": "Limited"}


# ---------------------------------------------------------------------------------------------------------------------------------------------------------- games
def _kickoff_text(t: pd.Timestamp) -> str:
    return pd.Timestamp(t).strftime("%a %b %d %I:%M %p").replace(" 0", " ") + " ET"


def spread_text(away: str, home: str, spread_home: float) -> str:
    """'CHI -3' for a home favourite (positive spread_home), 'NE -2.5' for an away favourite, "pick'em" at zero or when unknown."""
    if spread_home is None or not np.isfinite(spread_home) or spread_home == 0:
        return "pick'em"
    return f"{home} -{spread_home:g}" if spread_home > 0 else f"{away} -{abs(spread_home):g}"


def week_games(season: int, week: int) -> pd.DataFrame:
    """The week's games in kickoff order with a display `label`, the spread text and each team's implied total."""
    from matchup_model.opp import data as D
    s = D.schedule()
    g = s[(s["season"] == season) & (s["week"] == week)].sort_values(["kickoff", "away"]).reset_index(drop=True)
    if g.empty:
        return g.assign(label=[], spread="", away_implied=[], home_implied=[])
    g = g.assign(
        away_implied=g["total"] / 2 - g["spread_home"] / 2, home_implied=g["total"] / 2 + g["spread_home"] / 2,
        spread=[spread_text(r.away, r.home, r.spread_home) for r in g.itertuples()],
        label=[f"{r.away} @ {r.home} · {_kickoff_text(r.kickoff)}" for r in g.itertuples()])
    return g


def default_game_index(games: pd.DataFrame, now: datetime | None = None) -> int:
    """The next game to kick off (US Eastern); the last one of the week once they are all done, 0 for an empty list."""
    if games.empty:
        return 0
    now = now or datetime.now(ZoneInfo("America/New_York")).replace(tzinfo=None)
    upcoming = games.index[games["kickoff"] >= pd.Timestamp(now)]
    return int(upcoming[0]) if len(upcoming) else len(games) - 1


# ---------------------------------------------------------------------------------------------------------------------------------------------------------- players
def team_roster(team: str, season: int, week: int, games_back: int = ROSTER_GAMES) -> pd.DataFrame:
    """The QB/RB/WR/TE our model can project for `team` entering (season, week): he played in one of the team's last `games_back` games and his most recent game
    (for any team) was for this one, so players traded away or long gone do not appear. Columns: nk, name, pos."""
    from matchup_model.opp import data as D
    pw = D.player_weeks()
    pw = pw[pw["position"].isin(POS) & ((pw["season"] < season) | ((pw["season"] == season) & (pw["week"] < week)))]
    if pw.empty:
        return pd.DataFrame(columns=["nk", "name", "pos"])
    last = pw.sort_values(["season", "week"]).groupby("player_id").tail(1)
    mine = pw[pw["team"] == team]
    recent_weeks = mine[["season", "week"]].drop_duplicates().sort_values(["season", "week"]).tail(games_back)
    ids = set(mine.merge(recent_weeks, on=["season", "week"])["player_id"])
    on = last[(last["team"] == team) & last["player_id"].isin(ids)]
    return pd.DataFrame({"nk": on["name_key"].to_numpy(), "name": on["player_display_name"].to_numpy(), "pos": on["position"].to_numpy()}).drop_duplicates("nk")


def fp_for_team(fp: pd.DataFrame | None, team: str) -> pd.DataFrame:
    """FantasyPoints' QB/RB/WR/TE rows for one team (canonical nflverse abbreviations). Columns: nk, name, pos, fp."""
    from matchup_model.opp import data as D
    if fp is None or fp.empty:
        return pd.DataFrame(columns=["nk", "name", "pos", "fp"])
    d = fp[fp["pos"].isin(POS) & (fp["team"].map(D.canon_team) == team)]
    return pd.DataFrame({"nk": d["name"].map(normalize_name).to_numpy(), "name": d["name"].to_numpy(), "pos": d["pos"].to_numpy(), "fp": d["proj"].to_numpy(dtype=float)})


def game_candidates(away: str, home: str, season: int, week: int, fp: pd.DataFrame | None) -> pd.DataFrame:
    """Everyone who could be in the table: our roster for each team joined by normalised name to FantasyPoints' rows (a player on either side alone is kept).
    Columns: team, opp, nk, name, pos, fp (NaN if FantasyPoints has no row), has_fp (a row with a projection above 0)."""
    frames = []
    for team, opp in ((away, home), (home, away)):
        ours, theirs = team_roster(team, season, week), fp_for_team(fp, team)
        m = ours.merge(theirs, on="nk", how="outer", suffixes=("", "_fp"))
        m["name"] = m["name_fp"].where(m["name_fp"].notna(), m["name"]) if "name_fp" in m else m["name"]
        m["pos"] = m["pos"].where(m["pos"].notna(), m.get("pos_fp"))
        m = m[["nk", "name", "pos", "fp"]].assign(team=team, opp=opp)
        frames.append(m)
    c = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["nk", "name", "pos", "fp", "team", "opp"])
    c["has_fp"] = c["fp"].fillna(0.0) > 0
    return c[["team", "opp", "nk", "name", "pos", "fp", "has_fp"]]


def injury_tags(teams: tuple[str, ...], season: int, week: int) -> dict[tuple[str, str], str]:
    """(team, normalised name) -> 'Out' / 'Doubtful' / 'Questionable' once the official report is out, else 'DNP' / 'Limited' from the practice report."""
    from matchup_model.opp import data as D
    inj = D.injuries()
    if inj is None or inj.empty:
        return {}
    r = inj[(inj["season"] == season) & (inj["week"] == week) & inj["team"].isin(teams)]
    out: dict[tuple[str, str], str] = {}
    for x in r.itertuples():
        status = str(x.report_status or "").strip().lower()
        if status in ("out", "doubtful", "questionable"):
            tag = status.capitalize()
        else:
            tag = _PRACTICE.get(str(x.practice_status or "").strip().lower(), "")
        if tag:
            out[(x.team, normalize_name(x.full_name))] = tag
    return out


def _blended(nk: str, pos: str, season: int, week: int) -> dict:
    from matchup_model.opp import blend
    return blend.blended_line(nk, pos, season, week)


def project_players(cands: pd.DataFrame, season: int, week: int, line_fn=None) -> pd.DataFrame:
    """Our projection for each candidate: `ours` (blended DK points), the opportunity-model and trailing pieces, the role ratio, the stat line, and how it was made."""
    line_fn = line_fn or _blended
    rows = []
    for c in cands.itertuples():
        try:
            bl = line_fn(c.nk, c.pos, season, week) or {}
        except Exception:  # noqa: BLE001 -- one player with thin data must not take the table down
            bl = {}
        ln = bl.get("line") or {}
        fpv = bl.get("fp")
        rows.append({"ours": float(fpv) if fpv is not None and np.isfinite(fpv) else np.nan, "opp_fp": bl.get("opp_fp", np.nan), "trailing_fp": bl.get("trailing_fp", np.nan),
                     "role_ratio": bl.get("fp_role_ratio", np.nan), "source": bl.get("source", ""), "games": bl.get("games"),
                     **{k: float(ln[k]) if k in ln and ln[k] is not None else np.nan for k in STAT_KEYS}})
    return pd.concat([cands.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def game_table(away: str, home: str, season: int, week: int, fp: pd.DataFrame | None, line_fn=None) -> pd.DataFrame:
    """One row per player in the game with everything the screen shows (filters are applied afterwards, so toggling them is instant)."""
    cands = game_candidates(away, home, season, week, fp)
    if cands.empty:
        return cands.assign(ours=[], inj=[], diff=[], sort_key=[], any_td=[])
    t = project_players(cands, season, week, line_fn)
    tags = injury_tags((away, home), season, week)
    t["inj"] = [tags.get((tm, nk), "") for tm, nk in zip(t["team"], t["nk"])]
    td = t["rush_td"].fillna(0.0) + t["rec_td"].fillna(0.0)
    has_line = t[["rush_td", "rec_td"]].notna().any(axis=1) & (t["pos"] != "QB")
    t["any_td"] = np.where(has_line, 100.0 * (1.0 - np.exp(-td)), np.nan)
    t["diff"] = np.where(t["has_fp"] & t["ours"].notna(), t["ours"] - t["fp"], np.nan)
    t["sort_key"] = t["ours"].where(t["ours"].notna(), t["fp"])
    return t.sort_values("sort_key", ascending=False, na_position="last").reset_index(drop=True)


def filter_table(t: pd.DataFrame, *, only_fp: bool = True, min_fp: float = 0.0, hide_out: bool = True, positions=POS) -> pd.DataFrame:
    """Apply the screen's filters. only_fp keeps players FantasyPoints projects (above 0, and at least `min_fp`); hide_out drops players ruled Out."""
    if t.empty:
        return t
    keep = t["pos"].isin(list(positions))
    if only_fp:
        keep &= t["has_fp"] & (t["fp"] >= max(min_fp, 1e-9))
    if hide_out:
        keep &= t["inj"] != "Out"
    return t[keep]


def add_salary(t: pd.DataFrame, slate: pd.DataFrame | None) -> pd.DataFrame:
    """A DraftKings `salary` column from the slate table (dfs.slate.build_slate) matched on normalised name and team, blank where there is none."""
    if t.empty or slate is None or slate.empty or "salary" not in slate:
        return t.assign(salary=np.nan)
    from matchup_model.opp import data as D
    sal = {(D.canon_team(tm), normalize_name(n)): s for tm, n, s in zip(slate["team"], slate["name"], slate["salary"])}
    return t.assign(salary=[sal.get((tm, nk), np.nan) for tm, nk in zip(t["team"], t["nk"])])


def team_summary(team: str, opp: str, season: int, week: int, implied: float, qb: str | None) -> dict:
    """What the model expects for one team's offense: implied total and our pass / rush attempt volume. Missing pieces are None."""
    out = {"team": team, "implied": implied if implied is not None and np.isfinite(implied) else None, "pass_att": None, "rush_att": None, "qb": qb if isinstance(qb, str) else None}
    try:
        from matchup_model.opp import model as M
        tv = M.team_volume(team, season, week)
        out["pass_att"], out["rush_att"] = float(tv["pass_att"]), float(tv["rush_att"])
    except Exception:  # noqa: BLE001 -- the volume model needs the schedule row; without it the strip just omits the numbers
        pass
    return out


# ---------------------------------------------------------------------------------------------------------------------------------------------------------- screen
def _file_hash(p: Path) -> str:
    try:
        return hashlib.md5(Path(p).read_bytes()).hexdigest()[:10]
    except OSError:
        return "missing"


def _season() -> int:
    from matchup_model.opp.blend import current_season
    return current_season()


def _load_fp(week: int) -> pd.DataFrame | None:
    from dfs.projections import load_weekly_projections
    try:
        return load_weekly_projections(week)
    except Exception:  # noqa: BLE001 -- no export for this week yet
        return None


@st.cache_data(show_spinner=False)
def _games(season: int, week: int) -> pd.DataFrame:
    return week_games(season, week)


@st.cache_data(show_spinner="Projecting every player in the game…")
def _table(season: int, week: int, away: str, home: str, fp_hash: str) -> pd.DataFrame:
    return game_table(away, home, season, week, _load_fp(week))


def _col_config(cols: list[tuple[str, str]], detail: bool) -> dict:
    cc = {"Player": st.column_config.TextColumn("Player", width="medium"),
          "Inj": st.column_config.TextColumn("Inj", help="Official game status once the injury report is out (Out / Doubtful / Questionable); until then the practice report (DNP = did not practice)."),
          "FP": st.column_config.NumberColumn("FP", format="%.1f", help="FantasyPoints' own projection for the week."),
          "Ours": st.column_config.NumberColumn("Ours", format="%.1f", help="Our DK-point projection: opportunity model blended with the player's trailing average, nudged toward FantasyPoints when they disagree a lot."),
          "Δ": st.column_config.NumberColumn("Ours − FP", format="%+.1f"), "Salary": st.column_config.NumberColumn("Salary", format="$%d")}
    for _, label in cols:
        cc[label] = st.column_config.NumberColumn(label, format="%.0f" if label == "Any TD %" else "%.1f",
                                                  help="Chance of at least one rushing or receiving TD, from our expected TDs (1 - e^-x)." if label == "Any TD %" else None)
    if detail:
        cc.update({"Opp model": st.column_config.NumberColumn("Opp model", format="%.1f"), "Trailing": st.column_config.NumberColumn("Trailing", format="%.1f"),
                   "FP ÷ ours": st.column_config.NumberColumn("FP ÷ ours", format="%.2f", help="FantasyPoints' number over our pre-nudge number (1.0 = they agree).")})
    return cc


def _show(t: pd.DataFrame, pos: str, detail: bool, has_salary: bool) -> None:
    cols = GROUP_COLS[pos]
    g = t[t["pos"] == pos]
    if g.empty:
        return
    out = pd.DataFrame({"Player": g["name"], "Team": g["team"], "Inj": g["inj"], **({"Salary": g["salary"]} if has_salary else {}),
                        "FP": g["fp"].where(g["has_fp"]), "Ours": g["ours"], "Δ": g["diff"]})
    for k, label in cols:
        out[label] = g[k]
    if detail:
        out["Opp model"], out["Trailing"], out["FP ÷ ours"] = g["opp_fp"], g["trailing_fp"], g["role_ratio"]
    st.subheader(f"{pos} ({len(g)})")
    st.dataframe(out, hide_index=True, width="stretch", column_config=_col_config(cols, detail))


def render(week: int, slate_df: pd.DataFrame | None = None) -> None:
    st.header("🎮 DFS Game Projections")
    season = _season()
    games = _games(season, week)
    if games.empty:
        st.info(f"No week {week} games in the nflverse schedule yet (season {season}).")
        return
    pick = st.selectbox("Game", list(games["label"]), index=default_game_index(games), key="gv_game")
    g = games.iloc[list(games["label"]).index(pick)]

    c1, c2, c3, c4 = st.columns([2, 1, 2, 3])
    only_fp = c1.checkbox("Only players FantasyPoints projects", value=True, key="gv_onlyfp",
                          help="Keeps players with a FantasyPoints projection above 0, so depth players and anyone FantasyPoints has at zero stay out of the table.")
    min_fp = c2.number_input("Min FP pts", 0.0, 30.0, 0.0, 0.5, key="gv_minfp", disabled=not only_fp, help="Raise it to hide FantasyPoints' low-end projections too.")
    hide_out = c3.checkbox("Hide players ruled Out", value=True, key="gv_hideout")
    positions = c4.multiselect("Positions", list(POS), default=list(POS), key="gv_pos")
    detail = st.checkbox("Show model detail (opportunity model, trailing average, FP ÷ ours)", value=False, key="gv_detail")

    fp_path = _fp_path(week)
    table = _table(season, week, g["away"], g["home"], _file_hash(fp_path))
    have_fp = _load_fp(week) is not None
    if not have_fp:
        st.warning(f"No FantasyPoints export for week {week} (`{fp_path.name}`): showing every player our model can project.")
    view = add_salary(filter_table(table, only_fp=only_fp and have_fp, min_fp=min_fp, hide_out=hide_out, positions=positions), slate_df)

    a, b = st.columns(2)
    for col, team, opp, implied, qb in ((a, g["away"], g["home"], g["away_implied"], g["away_qb"]), (b, g["home"], g["away"], g["home_implied"], g["home_qb"])):
        s = team_summary(team, opp, season, week, implied, qb)
        bits = [f"**{team}**" + (" (away)" if team == g["away"] else " (home)")]
        if s["implied"] is not None:
            bits.append(f"implied {s['implied']:.1f}")
        if s["pass_att"] is not None:
            bits.append(f"our volume {s['pass_att']:.0f} pass / {s['rush_att']:.0f} rush attempts")
        if s["qb"]:
            bits.append(f"QB {s['qb']}")
        col.markdown(" · ".join(bits))
    total = g["total"]
    st.caption(f"{pick} · spread {g['spread']}" + (f" · total {total:g}" if pd.notna(total) else "") + (f" · {g['roof']}" if isinstance(g["roof"], str) else ""))

    if view.empty:
        st.info("No players match these filters.")
        return
    has_salary = bool(view["salary"].notna().any())
    for pos in POS:
        if pos in positions:
            _show(view, pos, detail, has_salary)
    n_hidden = len(table) - len(view)
    st.caption(f"{len(view)} players shown" + (f", {n_hidden} hidden by the filters" if n_hidden else "") + ". Stat columns are our opportunity-model line, scaled to the FantasyPoints "
               "nudge; `Ours` also blends in the trailing DK average. A blank `Ours` means too little game history to project (a rookie or a new starter), and a blank `FP` "
               "means FantasyPoints has no projection for him.")


def _fp_path(week: int) -> Path:
    from dfs.config import projection_path
    return projection_path(week)
