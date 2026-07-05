import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import json
import datetime
import re
from pathlib import Path

st.set_page_config(
    page_title="Fantasy Model | Best Ball Dashboard",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Style ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .block-container { padding-top: 1rem; }
  [data-testid="metric-container"] { background:#1e2130; border-radius:8px; padding:10px; }
  .tier-badge {
    display:inline-block; padding:2px 8px; border-radius:12px;
    font-size:11px; font-weight:700; color:#fff;
  }
  /* ── Mobile responsive ── */
  @media (max-width: 640px) {
    .block-container { padding-left: 0.5rem !important; padding-right: 0.5rem !important; }
    [data-testid="metric-container"] { padding: 5px !important; }
    h1 { font-size: 1.3rem !important; }
    h2 { font-size: 1.1rem !important; }
    h3 { font-size: 1.0rem !important; }
    p, li { font-size: 0.85rem !important; }
  }
  [data-testid="stDataFrame"] { overflow-x: auto; }
</style>
""", unsafe_allow_html=True)

# ── Team name ↔ abbreviation ───────────────────────────────────────────────────
FULL_TO_ABB = {
    "Arizona Cardinals": "ARZ", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BLT",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLV", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HST", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Los Angeles Rams": "LA", "Los Angeles Chargers": "LAC",
    "Las Vegas Raiders": "LV", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "Seattle Seahawks": "SEA", "San Francisco 49ers": "SF", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}
ABB_TO_FULL = {v: k for k, v in FULL_TO_ABB.items()}

TIER_COLORS = {
    1: "#FFD700", 2: "#C0C0C0", 3: "#CD7F32",
    4: "#4fc3f7", 5: "#81c784", 6: "#e57373",
    7: "#ba68c8", 8: "#ff8a65", 9: "#90a4ae",
    **{i: "#78909c" for i in range(10, 19)},
}

# ── Data loaders ───────────────────────────────────────────────────────────────
DATA      = Path(__file__).parent / "data"
PROJ_ROOT = Path(__file__).parent

@st.cache_data
def load_projections():
    df = pd.read_csv(DATA / "projections_2026.csv")
    return df

@st.cache_data
def load_all_projections():
    """Unified projection table for all positions from the single projection CSV."""
    path = PROJ_ROOT / "2026 NFL Fantasy Football Season Rankings  Projections  Fantasy Points (2).csv"
    df = pd.read_csv(path)[["Name", "POS", "Team", "Bye", "FPTS", "G", "FPTS/G"]].copy()
    df = df.rename(columns={"FPTS": "Proj_FP", "G": "Games", "FPTS/G": "Proj_PG"})
    df["Bye"]     = pd.to_numeric(df["Bye"],    errors="coerce")
    df["Proj_FP"] = pd.to_numeric(df["Proj_FP"], errors="coerce")
    df["Games"]   = pd.to_numeric(df["Games"],  errors="coerce").clip(lower=1)
    df["Proj_PG"] = pd.to_numeric(df["Proj_PG"], errors="coerce")
    return df


@st.cache_data
def load_historical_weekly():
    """Load all 2021-2025 weekly FP data from Rushing/Receiving/Passing Stats folders.
    Returns unified DataFrame with Name, Team, POS, Season, WEEK, FP.
    POS filter avoids double-counting: RB from Rushing, WR/TE from Receiving, QB from Passing.
    """
    BASE = Path(__file__).parent
    file_map = []
    for yr in range(2021, 2026):
        pass_name = f"{yr}Passiing.csv" if yr == 2021 else f"{yr}Passing.csv"
        file_map.extend([
            (BASE / "Rushing Stats"   / f"{yr}Rushing.csv",  ("RB",)),
            (BASE / "Receiving Stats" / f"{yr}Receiving.csv", ("WR", "TE")),
            (BASE / "Passing Stats"   / pass_name,            ("QB",)),
        ])

    frames = []
    for path, keep_pos in file_map:
        if not path.exists():
            continue
        try:
            raw  = pd.read_csv(path, header=None)
            cols = raw.iloc[1].tolist()
            df   = raw.iloc[2:].copy()
            df.columns = cols
            for c in ("Season", "WEEK", "FP"):
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=["Name", "FP"])
            df = df[df["POS"].isin(keep_pos)]
            keep = [c for c in ("Name", "Team", "POS", "Season", "WEEK", "FP") if c in df.columns]
            frames.append(df[keep].copy())
        except Exception:
            pass

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


@st.cache_data
def compute_historical_variance():
    """Per-player weekly FP std dev from 2022-2025 (4 seasons = sufficient sample).
    Returns dict: {name: {mean_fpg, std_fpg, n_games}}.
    """
    df = load_historical_weekly()
    if df.empty:
        return {}
    if "Season" in df.columns:
        df = df[df["Season"] >= 2022]
    result = {}
    for name, grp in df.groupby("Name"):
        fps = grp["FP"].dropna()
        n   = len(fps)
        if n >= 6:
            result[str(name)] = {
                "mean_fpg": round(float(fps.mean()), 2),
                "std_fpg":  round(float(fps.std()),  2),
                "n_games":  n,
            }
    return result


@st.cache_data
def load_defense_matchup_adj():
    """Build {team: {pos: {week: adj_factor}}} for weeks 1-14.
    adj_factor = opp_FP_per_game_allowed / league_avg_FP_per_game_allowed.
    >1.0 = easier matchup (defense allows more), <1.0 = tougher matchup.
    Covers QB, RB, WR, TE using existing fantasy points allowed data.
    """
    def_ranks = build_defense_ranks()
    sched_df  = load_schedule()

    result = {}
    for team in FULL_TO_ABB.values():
        team_sched = get_team_schedule(team, sched_df)
        team_sched = team_sched[team_sched["Week"] <= 17]
        result[team] = {}
        for pos_key in ("QB", "RB", "WR", "TE"):
            rdf = def_ranks.get(pos_key)
            if rdf is None:
                continue
            league_avg = rdf["FP_per_game"].mean()
            if league_avg <= 0:
                continue
            opp_adj = {r["Team"]: r["FP_per_game"] / league_avg
                       for _, r in rdf.iterrows()}
            result[team][pos_key] = {
                int(r["Week"]): opp_adj.get(r["Opponent"], 1.0)
                for _, r in team_sched.iterrows()
            }
    return result


@st.cache_data
def load_weekly_data():
    """Player-level weekly stats (WR + TE). Compute FP allowed by opponent+week+pos."""
    df = pd.read_csv(DATA / "2025byweek.csv", header=1)
    df["WEEK"] = pd.to_numeric(df["WEEK"], errors="coerce")
    df["FP"] = pd.to_numeric(df["FP"], errors="coerce")
    df["SLOT RTE %"] = pd.to_numeric(df["SLOT RTE %"], errors="coerce")
    df["WIDE RTE %"] = pd.to_numeric(df["WIDE RTE %"], errors="coerce")
    df["RTE"] = pd.to_numeric(df["RTE"], errors="coerce")
    return df.dropna(subset=["WEEK", "FP", "Opponent"])

# Nickname / legal-name aliases — maps any alternate spelling → the name used in the CSV.
# Add entries here whenever DraftKings and the rankings file disagree on a player's name.
_NAME_ALIASES = {
    "Kenneth Gainwell": "Kenny Gainwell",
    "Tre' Harris":      "Tre Harris",       # DK uses apostrophe, CSV does not
}
# Build a lowercase lookup so matching is case-insensitive
_NAME_ALIASES_LC = {k.lower(): v for k, v in _NAME_ALIASES.items()}

@st.cache_data
def load_fp_rankings():
    fp = pd.read_csv(DATA / "BestBallRankingsDraftKings (2).csv")
    fp = fp[fp["POS"].isin(["QB", "WR", "RB", "TE"])].copy()
    fp = fp.rename(columns={"NAME": "Name_clean", "OVERALL": "FP_Rank", "ADP": "FP_ADP"})
    # Parse positional rank string e.g. "RB1" → 1
    fp["FP_Pos_Rank"] = fp["POSITION"].str.extract(r"(\d+)$").astype(float)
    fp = fp.sort_values("FP_Rank")
    return fp[["Name_clean", "POS", "TEAM", "FP_Rank", "FP_Pos_Rank", "FP_ADP"]].rename(columns={"TEAM": "Team"})

@st.cache_data
def load_season_fp_allowed():
    """Season-level FP allowed by team per position."""
    pos_files = {
        "QB":       "fantasyPointsAllowedExport QB.csv",
        "RB":       "fantasyPointsAllowedExport RB.csv",
        "TE":       "fantasyPointsAllowedExport TE.csv",
        "WR":       "fantasyPointsAllowedExport WR All.csv",
        "WR_Slot":  "fantasyPointsAllowedExport WR SLot.csv",
        "WR_Wide":  "fantasyPointsAllowedExport WR WIde.csv",
    }
    frames = {}
    for pos, fname in pos_files.items():
        try:
            df = pd.read_csv(DATA / fname)
            df["abbr"] = df["Name"].map(FULL_TO_ABB)
            frames[pos] = df
        except FileNotFoundError:
            pass
    return frames

def load_schedule():
    df = pd.read_csv(DATA / "nfl_2026_schedule_with_coordinators.csv")
    df["home_abb"] = df["home_team"].map(FULL_TO_ABB)
    df["away_abb"] = df["away_team"].map(FULL_TO_ABB)
    return df

@st.cache_data
def build_defense_ranks():
    """
    Rank all 32 defenses 1-32 by position using 2025 data.
    Rank 1 = fewest FP allowed (hardest matchup).
    Rank 32 = most FP allowed (easiest matchup).
    Returns dict: pos -> DataFrame with columns [Team, FP_per_game, Rank].
    """
    weekly = compute_weekly_fp_allowed()
    ranks = {}

    # WR All, WR Slot, WR Wide, TE — from weekly data
    for pos_key in ("WR", "WR_Slot", "WR_Wide", "TE"):
        sub = (weekly[weekly["POS"] == pos_key]
               .groupby("Team")["FP_Allowed"]
               .agg(total="sum", games="count")
               .reset_index())
        sub["FP_per_game"] = sub["total"] / sub["games"]
        sub = sub.sort_values("FP_per_game", ascending=True).reset_index(drop=True)
        sub["Rank"] = sub.index + 1          # rank 32 = most FP allowed = easiest
        ranks[pos_key] = sub[["Team", "FP_per_game", "Rank"]]

    # RB — season-level file (includes rushing + receiving)
    rb_df = pd.read_csv(DATA / "fantasyPointsAllowedExport RB.csv")
    rb_df["Team"] = rb_df["Name"].map(FULL_TO_ABB)
    rb_df = rb_df.dropna(subset=["Team"])
    rb_df = rb_df.sort_values("FP/G", ascending=True).reset_index(drop=True)
    rb_df["Rank"] = rb_df.index + 1
    ranks["RB"] = rb_df[["Team", "FP/G", "Rank"]].rename(columns={"FP/G": "FP_per_game"})

    # QB — season-level file
    qb_df = pd.read_csv(DATA / "fantasyPointsAllowedExport QB.csv")
    qb_df["Team"] = qb_df["Name"].map(FULL_TO_ABB)
    qb_df = qb_df.dropna(subset=["Team"])
    qb_df = qb_df.sort_values("FP/G", ascending=True).reset_index(drop=True)
    qb_df["Rank"] = qb_df.index + 1
    ranks["QB"] = qb_df[["Team", "FP/G", "Rank"]].rename(columns={"FP/G": "FP_per_game"})

    return ranks

def get_team_schedule(team_abb, sched):
    """Return sorted DataFrame of [Week, Opponent] for a team's 2026 schedule."""
    home = sched[sched["home_abb"] == team_abb][["week", "away_abb"]].rename(
        columns={"week": "Week", "away_abb": "Opponent"})
    away = sched[sched["away_abb"] == team_abb][["week", "home_abb"]].rename(
        columns={"week": "Week", "home_abb": "Opponent"})
    return pd.concat([home, away]).dropna(subset=["Opponent"]).sort_values("Week").reset_index(drop=True)

@st.cache_data
def compute_weekly_fp_allowed():
    """Aggregate 2025byweek to defense-level FP allowed per week per position."""
    df = load_weekly_data()

    # All WR / TE
    all_wp = (df.groupby(["Opponent", "WEEK", "POS"], as_index=False)["FP"]
                .sum().rename(columns={"Opponent": "Team", "FP": "FP_Allowed"}))

    # WR Slot (players where slot route share ≥ 50% that game)
    slot_df = df[(df["POS"] == "WR") & (df["SLOT RTE %"] >= 50)].copy()
    slot_wp = (slot_df.groupby(["Opponent", "WEEK"], as_index=False)["FP"]
                      .sum()
                      .rename(columns={"Opponent": "Team", "FP": "FP_Allowed"}))
    slot_wp["POS"] = "WR_Slot"

    # WR Wide (players where wide route share ≥ 50% that game)
    wide_df = df[(df["POS"] == "WR") & (df["WIDE RTE %"] >= 50)].copy()
    wide_wp = (wide_df.groupby(["Opponent", "WEEK"], as_index=False)["FP"]
                      .sum()
                      .rename(columns={"Opponent": "Team", "FP": "FP_Allowed"}))
    wide_wp["POS"] = "WR_Wide"

    combined = pd.concat([all_wp, slot_wp, wide_wp], ignore_index=True)
    return combined

@st.cache_data
def compute_bye_weeks():
    """Derive each team's single bye week from the 2026 schedule."""
    sched_df = load_schedule()
    bye_map = {}
    for team in FULL_TO_ABB.values():
        team_sched = get_team_schedule(team, sched_df)
        weeks_played = set(team_sched["Week"].astype(int).tolist())
        byes = sorted(set(range(1, 19)) - weeks_played)
        bye_map[team] = byes[0] if byes else None
    return bye_map


_SEASON_WEEKS = 17  # full NFL regular season; FPTS / 17 = true per-week base rate

def _build_proj_lu(all_proj_df, bye_map, var_lu=None, matchup_adj=None):
    """Build {player_name: {pos, proj_pg, proj_ppw, weekly_ppw, bye, ...}} lookup.
    proj_ppw  = FPTS / 17  (per-week base across the full NFL season).
    weekly_ppw = {week: pts} adjusted by opponent defense FP/G allowed vs league avg.
    """
    lu = {}
    for _, row in all_proj_df.iterrows():
        bye = int(row["Bye"]) if pd.notna(row.get("Bye")) else (bye_map.get(row["Team"]) or 0)
        ppg = float(row["Proj_PG"]) if pd.notna(row.get("Proj_PG")) else 0.0
        if ppg > 0:
            name    = row["Name"]
            team    = row["Team"]
            pos     = row["POS"]
            proj_fp = float(row["Proj_FP"]) if pd.notna(row.get("Proj_FP")) else ppg * _SEASON_WEEKS
            proj_ppw = proj_fp / _SEASON_WEEKS  # base per-week rate over full 17-wk season
            entry = {"pos": pos, "proj_pg": ppg, "proj_ppw": proj_ppw,
                     "bye": bye, "team": team, "std_fpg": 0.0, "n_games": 0,
                     "weekly_ppw": None}
            if var_lu:
                v = var_lu.get(name) or var_lu.get(_NAME_ALIASES.get(name, ""))
                if v:
                    entry["std_fpg"] = v["std_fpg"]
                    entry["n_games"] = v["n_games"]
            if matchup_adj:
                team_adj = matchup_adj.get(team, {})
                pos_adj  = team_adj.get(pos, {})  # {week: adj_factor}
                if pos_adj:
                    entry["weekly_ppw"] = {wk: round(proj_ppw * f, 2)
                                           for wk, f in pos_adj.items()}
            lu[name] = entry
    return lu


def _project_bb_score(roster_names, proj_lu, num_weeks=14):
    """
    Estimate DraftKings Best Ball total for a 20-player roster.
    Lineup each week: QB + 2 RB + 3 WR + TE + FLEX (best remaining RB/WR/TE).
    Returns (total_pts, weekly_scores_list).
    """
    players = []
    for name in roster_names:
        if name in proj_lu:
            p = proj_lu[name]
            players.append((p["pos"], p.get("proj_ppw", p["proj_pg"]), p["bye"],
                            p.get("weekly_ppw")))  # weekly_ppw = {wk: pts} or None

    total = 0.0
    weekly = []
    for wk in range(1, num_weeks + 1):
        by_pos: dict = {"QB": [], "RB": [], "WR": [], "TE": []}
        for pos, ppw, bye, weekly_ppw in players:
            if bye != wk and pos in by_pos:
                pts = weekly_ppw.get(wk, ppw) if weekly_ppw else ppw
                by_pos[pos].append(pts)
        for pos in by_pos:
            by_pos[pos].sort(reverse=True)

        qb_pts   = by_pos["QB"][0] if by_pos["QB"] else 0.0
        rb_pts   = sum(by_pos["RB"][:2]);  rb_flex = by_pos["RB"][2:]
        wr_pts   = sum(by_pos["WR"][:3]);  wr_flex = by_pos["WR"][3:]
        te_pts   = by_pos["TE"][0] if by_pos["TE"] else 0.0
        te_flex  = by_pos["TE"][1:]

        flex_pool = sorted(rb_flex + wr_flex + te_flex, reverse=True)
        flex_pts  = flex_pool[0] if flex_pool else 0.0

        wk_score = qb_pts + rb_pts + wr_pts + te_pts + flex_pts
        total   += wk_score
        weekly.append(round(wk_score, 1))
    return round(total, 1), weekly


@st.cache_data
def compute_boom_rates():
    """Boom rate from 2025 weekly data: % of games above threshold. WR≥20, TE≥12."""
    df = load_weekly_data()
    thresholds = {"WR": 20.0, "TE": 12.0}
    result = {}
    for pos, thresh in thresholds.items():
        sub = df[df["POS"] == pos].copy()
        if sub.empty or "Name" not in sub.columns:
            continue
        sub["Name"] = sub["Name"].astype(str).str.strip()
        stats = sub.groupby("Name").agg(games=("FP", "count"), booms=("FP", lambda x: (x >= thresh).sum())).reset_index()
        for _, row in stats.iterrows():
            result[row["Name"]] = int(round(row["booms"] / row["games"] * 100))
    return result

@st.cache_data
def build_team_schedule_matrix():
    """Pre-build {(team, pos_key): {week: def_rank}} for instant complement scoring."""
    sched_df = load_schedule()
    def_rnks = build_defense_ranks()
    matrix = {}
    for team in FULL_TO_ABB.values():
        team_sched = get_team_schedule(team, sched_df)
        for pos_key in ["QB", "RB", "WR", "TE"]:
            rank_df = def_rnks.get(pos_key)
            if rank_df is None:
                continue
            rmap = dict(zip(rank_df["Team"], rank_df["Rank"]))
            matrix[(team, pos_key)] = {
                int(row["Week"]): rmap[row["Opponent"]]
                for _, row in team_sched.iterrows()
                if row["Opponent"] in rmap
            }
    return matrix

@st.cache_data
def build_team_opponent_lookup():
    """For each team return {week: opponent_team} — used for playoff game-stack detection."""
    sched_df = load_schedule()
    lookup = {}
    for team in FULL_TO_ABB.values():
        team_sched = get_team_schedule(team, sched_df)
        lookup[team] = {int(row["Week"]): row["Opponent"] for _, row in team_sched.iterrows()}
    return lookup

@st.cache_data
def compute_all_playoff_scores():
    """Pre-compute avg def rank in weeks 14-17 (32=easiest) for every ranked player."""
    sched_df = load_schedule()
    fp = load_fp_rankings().rename(columns={"Name_clean": "Name"})
    def_rnks = build_defense_ranks()
    pw_set = {14, 15, 16, 17}
    result = {}
    for _, p in fp.iterrows():
        pos_key = p["POS"] if p["POS"] != "WR" else "WR"
        rank_df = def_rnks.get(pos_key)
        if rank_df is None or rank_df.empty:
            result[p["Name"]] = None
            continue
        rmap = dict(zip(rank_df["Team"], rank_df["Rank"]))
        team_sched = get_team_schedule(p["Team"], sched_df)
        pw_sched = team_sched[team_sched["Week"].isin(pw_set)]
        week_ranks = [rmap[row["Opponent"]] for _, row in pw_sched.iterrows() if row["Opponent"] in rmap]
        result[p["Name"]] = round(float(np.mean(week_ranks)), 1) if week_ranks else None
    return result


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏈 Best Ball Dashboard")
    st.markdown("---")
    tab_choice = st.radio(
        "Screen",
        ["📊 Player Projections", "🛡️ Defense Matchups", "📈 Schedule Rankings", "📅 Schedule Viewer", "📉 Weekly Projections", "🎯 Draft Room"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("2025 defensive data · 2026 projections")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — PLAYER PROJECTIONS
# ══════════════════════════════════════════════════════════════════════════════
if tab_choice == "📊 Player Projections":
    st.header("2026 Player Projections")
    fp_ranks = load_fp_rankings()
    # merged is FP rankings as primary; used by opponent preview too
    merged = fp_ranks.rename(columns={"Name_clean": "Name"})

    col1, col2 = st.columns([2, 1])
    with col1:
        pos_filter = st.multiselect("Position", ["QB", "WR", "RB", "TE"], default=["QB", "WR", "RB", "TE"])
    with col2:
        search = st.text_input("Search player")

    df = merged[merged["POS"].isin(pos_filter)].copy()
    if search:
        df = df[df["Name"].str.contains(search, case=False, na=False)]

    df = df.sort_values("FP_Rank", ascending=True, na_position="last")
    df = df.reset_index(drop=True)
    df.index += 1

    display = df[["Name", "POS", "Team", "FP_Rank", "FP_Pos_Rank", "FP_ADP"]].copy()
    display.columns = ["Player", "POS", "Team", "Rank", "Pos Rank", "ADP"]
    display["ADP"] = display["ADP"].round(1)

    st.write(
        display.to_html(escape=False, index=True, classes="dataframe"),
        unsafe_allow_html=True,
    )

    # ── Opponent Offense Preview (weeks 15-17) ─────────────────────────────────
    if search and not df.empty:
        st.markdown("---")
        st.subheader("📋 Weeks 15–17 Opponent Offense Preview")
        st.caption("Players on the opposing offense weeks 15–17, ordered by FP rank. Use for best-ball stacks.")
        sched_data  = load_schedule()
        preview_wks = [15, 16, 17]

        for _, prow in df.head(3).iterrows():
            pname = prow["Name"]
            pteam = prow["Team"]
            ppos  = prow["POS"]
            team_s = get_team_schedule(pteam, sched_data)

            st.markdown(f"**{pname}** ({ppos} · {ABB_TO_FULL.get(pteam, pteam)})")
            wcols = st.columns(3)
            for ci, wk in enumerate(preview_wks):
                wk_row = team_s[team_s["Week"] == wk]
                with wcols[ci]:
                    if wk_row.empty:
                        st.markdown(f"**Wk {wk}:** BYE")
                        continue
                    opp_abb  = wk_row.iloc[0]["Opponent"]
                    opp_full = ABB_TO_FULL.get(opp_abb, opp_abb)
                    st.markdown(f"**Wk {wk} vs {opp_abb}** — {opp_full}")
                    opp_players = merged[
                        (merged["Team"] == opp_abb) &
                        (merged["POS"].isin(["QB", "WR", "RB", "TE"]))
                    ].sort_values("FP_Rank", ascending=True, na_position="last")

                    if opp_players.empty:
                        st.caption("No players found")
                    else:
                        rows_out = [
                            {"Rank": int(op["FP_Rank"]) if pd.notna(op["FP_Rank"]) else "—",
                             "Player": op["Name"], "POS": op["POS"],
                             "ADP": round(op["FP_ADP"], 1) if pd.notna(op["FP_ADP"]) else "—"}
                            for _, op in opp_players.iterrows()
                        ]
                        st.dataframe(
                            pd.DataFrame(rows_out),
                            width="stretch",
                            hide_index=True,
                            height=min(40 + len(rows_out) * 35, 400),
                        )
            st.markdown("")



# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — DEFENSE MATCHUP ANALYZER
# ══════════════════════════════════════════════════════════════════════════════
elif tab_choice == "🛡️ Defense Matchups":
    st.header("🛡️ Defense Matchup Analyzer")
    st.caption("2025 season data — FP allowed each week against each defense")

    weekly = compute_weekly_fp_allowed()
    season_fp = load_season_fp_allowed()

    # Controls
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        all_teams = sorted(weekly["Team"].unique())
        team_labels = {t: ABB_TO_FULL.get(t, t) for t in all_teams}
        selected_team = st.selectbox(
            "Select Defense",
            all_teams,
            format_func=lambda t: team_labels[t],
        )
    with c2:
        pos_options = {
            "WR (All)":  "WR",
            "WR (Slot)": "WR_Slot",
            "WR (Wide)": "WR_Wide",
            "TE":        "TE",
        }
        pos_label = st.selectbox("Position", list(pos_options.keys()))
        selected_pos = pos_options[pos_label]
    with c3:
        show_avg = st.checkbox("Show league avg", value=True)

    # Compute league averages
    avg_by_week = (weekly[weekly["POS"] == selected_pos]
                   .groupby("WEEK")["FP_Allowed"]
                   .mean()
                   .reset_index())

    # Selected team data
    team_data = weekly[(weekly["Team"] == selected_team) & (weekly["POS"] == selected_pos)].copy()
    team_data = team_data.sort_values("WEEK")

    # Season rank
    if selected_pos in ["WR", "WR_Slot", "WR_Wide", "TE"]:
        pos_key = selected_pos.split("_")[0] if "_" not in selected_pos else selected_pos
        # Compute season total FP allowed per team for ranking
        season_totals = (weekly[weekly["POS"] == selected_pos]
                         .groupby("Team")["FP_Allowed"]
                         .sum()
                         .reset_index()
                         .sort_values("FP_Allowed", ascending=False)
                         .reset_index(drop=True))
        season_totals["Rank"] = season_totals.index + 1
        team_rank_row = season_totals[season_totals["Team"] == selected_team]
        team_season_total = team_data["FP_Allowed"].sum()
        team_rank = int(team_rank_row["Rank"].values[0]) if not team_rank_row.empty else "N/A"
    else:
        team_season_total = team_data["FP_Allowed"].sum()
        team_rank = "N/A"

    league_avg_total = (weekly[weekly["POS"] == selected_pos]
                        .groupby("Team")["FP_Allowed"]
                        .sum()
                        .mean())

    # Summary metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Season FP Allowed", f"{team_season_total:.0f}")
    m2.metric("vs League Avg", f"{team_season_total - league_avg_total:+.0f}",
              delta_color="normal" if team_season_total >= league_avg_total else "inverse")
    m3.metric("Defense Rank", f"#{team_rank} / 32")
    m4.metric("Avg FP/Week", f"{team_data['FP_Allowed'].mean():.1f}")

    # ── Run chart ──────────────────────────────────────────────────────────────
    fig = go.Figure()

    # Color each bar by vs. league average
    if not team_data.empty:
        avg_vals = avg_by_week.set_index("WEEK")["FP_Allowed"]
        colors = []
        for _, row in team_data.iterrows():
            wk_avg = avg_vals.get(row["WEEK"], np.nan)
            if np.isnan(wk_avg):
                colors.append("#90a4ae")
            elif row["FP_Allowed"] >= wk_avg * 1.15:
                colors.append("#4caf50")   # easy (green)
            elif row["FP_Allowed"] <= wk_avg * 0.85:
                colors.append("#ef5350")   # tough (red)
            else:
                colors.append("#ffa726")   # average (orange)

        fig.add_trace(go.Bar(
            x=team_data["WEEK"],
            y=team_data["FP_Allowed"],
            marker_color=colors,
            name=f"{selected_team} FP Allowed",
            text=team_data["FP_Allowed"].round(1),
            textposition="outside",
            textfont=dict(size=10),
        ))

        # Trend line
        if len(team_data) >= 4:
            z = np.polyfit(team_data["WEEK"], team_data["FP_Allowed"], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(team_data["WEEK"].min(), team_data["WEEK"].max(), 50)
            fig.add_trace(go.Scatter(
                x=x_trend, y=p(x_trend),
                mode="lines",
                line=dict(color="#90a4ae", dash="dot", width=1.5),
                name="Trend",
            ))

    # League average line
    if show_avg and not avg_by_week.empty:
        fig.add_trace(go.Scatter(
            x=avg_by_week["WEEK"],
            y=avg_by_week["FP_Allowed"],
            mode="lines+markers",
            line=dict(color="#b0bec5", dash="dash", width=1.5),
            marker=dict(size=5),
            name="League Avg",
        ))

    team_full = ABB_TO_FULL.get(selected_team, selected_team)
    fig.update_layout(
        title=f"{team_full} — {pos_label} FP Allowed per Week (2025)",
        xaxis=dict(title="Week", tickmode="linear", dtick=1),
        yaxis=dict(title="Fantasy Points Allowed"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=480,
        hovermode="x unified",
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        bargap=0.25,
    )
    st.plotly_chart(fig, width="stretch")

    # ── All-team comparison bar ────────────────────────────────────────────────
    st.markdown("---")
    st.subheader(f"All 32 Teams — {pos_label} Season FP Allowed")

    season_totals_disp = (weekly[weekly["POS"] == selected_pos]
                          .groupby("Team")["FP_Allowed"]
                          .sum()
                          .reset_index()
                          .sort_values("FP_Allowed", ascending=False))
    season_totals_disp["Team_Full"] = season_totals_disp["Team"].map(
        lambda t: ABB_TO_FULL.get(t, t)
    )
    avg_line = season_totals_disp["FP_Allowed"].mean()

    bar_colors = [
        "#4caf50" if fp >= avg_line * 1.10 else
        "#ef5350" if fp <= avg_line * 0.90 else
        "#ffa726"
        for fp in season_totals_disp["FP_Allowed"]
    ]
    # Highlight selected team
    bar_colors = [
        "#1565c0" if row["Team"] == selected_team else bar_colors[i]
        for i, (_, row) in enumerate(season_totals_disp.iterrows())
    ]

    fig2 = go.Figure(go.Bar(
        x=season_totals_disp["Team"],
        y=season_totals_disp["FP_Allowed"],
        marker_color=bar_colors,
        text=season_totals_disp["FP_Allowed"].round(0).astype(int),
        textposition="outside",
        textfont=dict(size=9),
    ))
    fig2.add_hline(y=avg_line, line_dash="dash", line_color="#b0bec5",
                   annotation_text=f"Avg: {avg_line:.0f}", annotation_position="top right")
    fig2.update_layout(
        xaxis=dict(title="", tickfont=dict(size=10)),
        yaxis=dict(title="Total FP Allowed"),
        height=380,
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        margin=dict(t=20, b=10),
        showlegend=False,
    )
    st.plotly_chart(fig2, width="stretch")

    # Color legend
    st.markdown(
        "🟢 **Easy** (≥ +10% above avg)  &nbsp; 🟠 **Average**  &nbsp; 🔴 **Tough** (≥ −10% below avg)  &nbsp; 🔵 Selected team",
        unsafe_allow_html=True,
    )

    # ── WR slot/wide side-by-side (only for WR) ───────────────────────────────
    if selected_pos in ("WR", "WR_Slot", "WR_Wide"):
        st.markdown("---")
        st.subheader(f"{team_full} — Slot vs Wide WR Breakdown (Week by Week)")
        slot_d = weekly[(weekly["Team"] == selected_team) & (weekly["POS"] == "WR_Slot")].sort_values("WEEK")
        wide_d = weekly[(weekly["Team"] == selected_team) & (weekly["POS"] == "WR_Wide")].sort_values("WEEK")

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=slot_d["WEEK"], y=slot_d["FP_Allowed"],
            mode="lines+markers", name="Slot WR",
            line=dict(color="#42a5f5", width=2),
            marker=dict(size=7),
        ))
        fig3.add_trace(go.Scatter(
            x=wide_d["WEEK"], y=wide_d["FP_Allowed"],
            mode="lines+markers", name="Wide WR",
            line=dict(color="#ef5350", width=2),
            marker=dict(size=7),
        ))
        fig3.update_layout(
            xaxis=dict(title="Week", tickmode="linear", dtick=1),
            yaxis=dict(title="FP Allowed"),
            height=350,
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font=dict(color="#fafafa"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig3, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — SCHEDULE RANKINGS
# ══════════════════════════════════════════════════════════════════════════════
elif tab_choice == "📈 Schedule Rankings":
    st.header("📈 Schedule Rankings")
    st.caption(
        "Defense ranked 1–32 by 2025 FP/game allowed per position. "
        "**Rank 1 = easiest matchup** (most FP allowed). Rank 32 = toughest."
    )

    sched       = load_schedule()
    def_ranks   = build_defense_ranks()
    all_teams   = sorted(set(sched["home_abb"].dropna()) | set(sched["away_abb"].dropna()))

    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        sel_team = st.selectbox(
            "Team",
            all_teams,
            index=all_teams.index("DAL") if "DAL" in all_teams else 0,
            format_func=lambda t: ABB_TO_FULL.get(t, t),
        )
    with c2:
        pos_opts = {
            "QB":        "QB",
            "WR (All)":  "WR",
            "WR (Slot)": "WR_Slot",
            "WR (Wide)": "WR_Wide",
            "RB":        "RB",
            "TE":        "TE",
        }
        # Multi-position select for overlay
        sel_positions = st.multiselect(
            "Positions to show",
            list(pos_opts.keys()),
            default=["QB", "WR (All)", "RB", "TE"],
        )
    with c3:
        playoff_wks = st.multiselect("Playoff Weeks", list(range(1, 19)), default=[14, 15, 16, 17])

    team_sched = get_team_schedule(sel_team, sched)
    if team_sched.empty:
        st.warning(f"No 2026 schedule found for {sel_team}.")
        st.stop()

    # Build rank lookup: week → opponent → rank per pos
    rank_lookup = {}   # pos_label -> list of (week, opp, rank, fpg)
    for pos_label in sel_positions:
        pos_key = pos_opts[pos_label]
        ranks_df = def_ranks.get(pos_key)
        if ranks_df is None:
            continue
        rank_map  = dict(zip(ranks_df["Team"], ranks_df["Rank"]))
        fpg_map   = dict(zip(ranks_df["Team"], ranks_df["FP_per_game"]))
        rows = []
        for _, row in team_sched.iterrows():
            opp  = row["Opponent"]
            rnk  = rank_map.get(opp)
            fpg  = fpg_map.get(opp)
            rows.append({"Week": int(row["Week"]), "Opponent": opp,
                         "Rank": rnk, "FP_per_game": fpg})
        rank_lookup[pos_label] = rows

    # ── Summary metrics ────────────────────────────────────────────────────────
    if sel_positions:
        first_pos = sel_positions[0]
        first_key = pos_opts[first_pos]
        first_rows = rank_lookup.get(first_pos, [])
        if first_rows:
            all_ranks = [r["Rank"] for r in first_rows if r["Rank"] is not None]
            playoff_ranks = [r["Rank"] for r in first_rows
                             if r["Rank"] is not None and r["Week"] in playoff_wks]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Season Avg Rank", f"#{np.mean(all_ranks):.1f}" if all_ranks else "N/A",
                      help="Average opponent defense rank (32=easiest, 1=hardest)")
            m2.metric("Playoff Avg Rank", f"#{np.mean(playoff_ranks):.1f}" if playoff_ranks else "N/A")
            easy_wks  = sum(1 for r in all_ranks if r >= 23)
            tough_wks = sum(1 for r in all_ranks if r <= 10)
            m3.metric("Easy Weeks (rank 23-32)", easy_wks)
            m4.metric("Tough Weeks (rank 1-10)", tough_wks)

    st.markdown("---")

    # ── One subplot per position ───────────────────────────────────────────────
    POS_COLORS = {
        "QB":        "#ce93d8",
        "WR (All)":  "#42a5f5",
        "WR (Slot)": "#29b6f6",
        "WR (Wide)": "#00bcd4",
        "RB":        "#66bb6a",
        "TE":        "#ffa726",
    }

    for pos_label in sel_positions:
        pos_key = pos_opts[pos_label]
        rows    = rank_lookup.get(pos_label, [])
        if not rows:
            continue

        weeks    = [r["Week"] for r in rows]
        ranks    = [r["Rank"] for r in rows]
        opps     = [r["Opponent"] for r in rows]
        fpgs     = [r["FP_per_game"] for r in rows]

        # Bar colors based on rank (32=easiest, 1=hardest)
        bar_colors = []
        for rk in ranks:
            if rk is None:
                bar_colors.append("#555")
            elif rk >= 23:
                bar_colors.append("#4caf50")    # easy (rank 23-32 = most FP allowed)
            elif rk <= 10:
                bar_colors.append("#ef5350")    # tough (rank 1-10)
            else:
                bar_colors.append("#ffa726")    # average

        fig = go.Figure()

        # Playoff week shading
        for pw in playoff_wks:
            fig.add_vrect(x0=pw - 0.5, x1=pw + 0.5,
                          fillcolor="rgba(25,118,210,0.15)", line_width=0, layer="below")

        # Bars — height = FP/game so tall bar = easy (advantageous) matchup
        # Bar height = rank directly: rank 32 (easiest) → tallest bar
        bar_heights = [r if r is not None else 0 for r in ranks]
        valid_ranks = [r for r in ranks if r is not None]
        avg_rank    = np.mean(valid_ranks) if valid_ranks else 16.5
        avg_height  = avg_rank

        hover_text = [
            f"Week {w} vs {o}<br>Def Rank #{r} (32=easiest)<br>{f:.1f} FP/g allowed"
            if r is not None and f is not None else f"Week {w} vs {o}"
            for w, o, r, f in zip(weeks, opps, ranks, fpgs)
        ]

        fig.add_trace(go.Bar(
            x=weeks,
            y=bar_heights,
            marker_color=bar_colors,
            text=[
                f"#{r}<br>{o}" if r is not None else (o or "")
                for r, o in zip(ranks, opps)
            ],
            textposition="inside",
            textfont=dict(size=9, color="white"),
            hovertext=hover_text,
            hoverinfo="text",
            name=pos_label,
        ))

        # Average rank line
        fig.add_hline(y=avg_height, line_dash="dash", line_color="#90a4ae", line_width=1.5,
                      annotation_text=f"avg rank #{avg_rank:.1f}",
                      annotation_position="top right",
                      annotation_font=dict(color="#90a4ae", size=10))

        # Y-axis tick labels: rank 32 at top = easiest
        tick_ranks  = [1, 8, 16, 24, 32]
        tick_vals   = tick_ranks
        tick_labels = [f"#{r}" for r in tick_ranks]

        fig.update_layout(
            title=dict(
                text=f"{ABB_TO_FULL.get(sel_team, sel_team)} — {pos_label} Defense Rank (taller = easier matchup)",
                font=dict(size=14),
            ),
            xaxis=dict(
                title="Week", tickmode="linear", dtick=1,
                range=[0.5, max(weeks) + 0.5],
            ),
            yaxis=dict(
                title="Defense Rank (32 = easiest)",
                range=[0, 34],
                tickvals=tick_vals,
                ticktext=tick_labels,
            ),
            height=300,
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font=dict(color="#fafafa"),
            showlegend=False,
            margin=dict(t=40, b=30, l=50, r=20),
            bargap=0.3,
        )
        st.plotly_chart(fig, width="stretch")

    # ── Full schedule table (all positions side by side) ──────────────────────
    st.markdown("---")
    st.subheader("Full Schedule — All Positions")

    table_rows = []
    for _, row in team_sched.iterrows():
        wk  = int(row["Week"])
        opp = row["Opponent"]
        tr  = {"Week": wk, "Opponent": opp, "Opp Full": ABB_TO_FULL.get(opp, opp),
               "Playoff": "✅" if wk in playoff_wks else ""}
        for pos_label in ("QB", "WR (All)", "RB", "TE", "WR (Slot)", "WR (Wide)"):
            pos_key = pos_opts.get(pos_label)
            if pos_key and pos_key in def_ranks:
                rank_map = dict(zip(def_ranks[pos_key]["Team"], def_ranks[pos_key]["Rank"]))
                fpg_map  = dict(zip(def_ranks[pos_key]["Team"], def_ranks[pos_key]["FP_per_game"]))
                rk  = rank_map.get(opp)
                fpg = fpg_map.get(opp)
                short = pos_label.replace(" (All)", "").replace(" (Slot)", "-Slot").replace(" (Wide)", "-Wide")
                tr[f"{short} Rank"]  = f"#{rk}" if rk else "—"
                tr[f"{short} FP/G"]  = f"{fpg:.1f}" if fpg else "—"
        table_rows.append(tr)

    table_df = pd.DataFrame(table_rows)

    def highlight_rank(val):
        if not isinstance(val, str) or not val.startswith("#"):
            return ""
        try:
            n = int(val[1:])
            if n >= 23:  return "background-color: rgba(76,175,80,0.25)"   # easy (32=easiest)
            if n <= 10:  return "background-color: rgba(239,83,80,0.25)"   # tough (1=hardest)
            return "background-color: rgba(255,167,38,0.15)"
        except ValueError:
            return ""

    rank_cols = [c for c in table_df.columns if "Rank" in c]
    st.dataframe(
        table_df.style.map(highlight_rank, subset=rank_cols),
        width="stretch",
        hide_index=True,
        height=650,
    )
    st.markdown(
        "🟢 Rank 23–32 (easiest — most FP allowed) &nbsp; 🟠 11–22 (average) &nbsp; 🔴 Rank 1–10 (toughest)",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — SCHEDULE VIEWER
# ══════════════════════════════════════════════════════════════════════════════
elif tab_choice == "📅 Schedule Viewer":
    st.header("📅 Schedule Matchup Viewer")
    st.caption("See a player's 2026 weekly opponents and defensive strength at their position")

    proj = load_projections()
    fp_all = load_fp_rankings().rename(columns={"Name_clean": "Name"})
    sched = load_schedule()
    all_def_ranks = build_defense_ranks()   # covers QB, RB, WR, TE from correct sources

    # Build player list from FP rankings (includes QBs) merged with proj for team/pos info
    fp_players = fp_all[["Name", "POS", "Team"]].drop_duplicates("Name")
    proj_players = proj[["Name", "POS", "Team"]].drop_duplicates("Name")
    all_players = pd.concat([fp_players, proj_players]).drop_duplicates("Name").sort_values("Name")

    c1, c2 = st.columns([3, 1])
    with c1:
        player_list = sorted(all_players["Name"].tolist())
        player_name = st.selectbox("Select Player", player_list)
    with c2:
        playoff_weeks = st.multiselect("Playoff Weeks", list(range(1, 19)),
                                       default=[14, 15, 16, 17])

    # Look up player in FP rankings first, fall back to proj
    player_row_fp  = fp_all[fp_all["Name"] == player_name]
    player_row_prj = proj[proj["Name"] == player_name]

    if not player_row_fp.empty:
        pr = player_row_fp.iloc[0]
        team_abb = pr["Team"]
        pos      = pr["POS"]
        fp_rank  = int(pr["FP_Rank"]) if pd.notna(pr["FP_Rank"]) else None
        fp_adp   = round(pr["FP_ADP"], 1) if pd.notna(pr["FP_ADP"]) else None
        fp_pos   = f"{pr['POS']}{int(pr['FP_Pos_Rank'])}" if pd.notna(pr["FP_Pos_Rank"]) else None
    elif not player_row_prj.empty:
        pr = player_row_prj.iloc[0]
        team_abb = pr["Team"]
        pos      = pr["POS"]
        fp_rank = fp_adp = fp_pos = None
    else:
        st.error("Player not found")
        st.stop()

    pos_key = "WR" if pos == "WR" else pos

    pos_ranks_df   = all_def_ranks.get(pos_key, pd.DataFrame(columns=["Team", "FP_per_game", "Rank"]))
    fpg_map        = dict(zip(pos_ranks_df["Team"], pos_ranks_df["FP_per_game"]))
    rank_map       = dict(zip(pos_ranks_df["Team"], pos_ranks_df["Rank"]))
    league_avg_fpg = pos_ranks_df["FP_per_game"].mean() if not pos_ranks_df.empty else 0
    avg_rank_sv    = pos_ranks_df["Rank"].mean() if not pos_ranks_df.empty else 16.5

    schedule_df = get_team_schedule(team_abb, sched)
    schedule_df["Opp_Full"]    = schedule_df["Opponent"].map(lambda t: ABB_TO_FULL.get(t, t))
    schedule_df["FP_per_game"] = schedule_df["Opponent"].map(lambda t: fpg_map.get(t, np.nan))
    schedule_df["Def_Rank"]    = schedule_df["Opponent"].map(lambda t: rank_map.get(t))
    schedule_df["vs_avg"]      = schedule_df["FP_per_game"] - league_avg_fpg
    schedule_df["Is_Playoff"]  = schedule_df["Week"].isin(playoff_weeks)

    if schedule_df.empty:
        st.warning(f"No 2026 schedule found for {team_abb}.")
        st.stop()

    # Player summary
    st.markdown(f"### {player_name} — {pos} | {ABB_TO_FULL.get(team_abb, team_abb)}")
    pm1, pm2, pm3, pm4 = st.columns(4)
    pm1.metric("FP Rank", f"#{fp_rank}" if fp_rank else "N/A")
    pm2.metric("Pos Rank", fp_pos if fp_pos else "N/A")
    pm3.metric("ADP", f"{fp_adp}" if fp_adp else "N/A")
    playoff_avg_rank = schedule_df[schedule_df["Is_Playoff"]]["Def_Rank"].mean()
    pm4.metric("Playoff Avg Def Rank", f"#{playoff_avg_rank:.1f}" if pd.notna(playoff_avg_rank) else "N/A")

    # Schedule chart — Y axis = rank (32=easiest, tall bar = easy)
    fig = go.Figure()

    bar_heights = []
    bar_colors  = []
    hover_texts = []
    bar_texts   = []
    for _, row in schedule_df.iterrows():
        rk  = row["Def_Rank"]
        fpg = row["FP_per_game"]
        opp = row["Opponent"]
        if rk is None or (isinstance(fpg, float) and np.isnan(fpg)):
            bar_heights.append(0)
            bar_colors.append("#555")
            hover_texts.append(f"vs {opp} — no data")
            bar_texts.append(opp)
        else:
            bar_heights.append(rk)
            fpg_str = f"{fpg:.1f} FP/g"
            hover_texts.append(f"vs {ABB_TO_FULL.get(opp, opp)}<br>Def Rank #{rk} (32=easiest)<br>{fpg_str}")
            bar_texts.append(f"#{rk}<br>{opp}")
            if rk >= 23:
                bar_colors.append("#1565c0" if row["Is_Playoff"] else "#4caf50")
            elif rk <= 10:
                bar_colors.append("#b71c1c" if row["Is_Playoff"] else "#ef5350")
            else:
                bar_colors.append("#1976d2" if row["Is_Playoff"] else "#ffa726")

    avg_height = avg_rank_sv

    fig.add_trace(go.Bar(
        x=schedule_df["Week"],
        y=bar_heights,
        marker_color=bar_colors,
        text=bar_texts,
        textposition="inside",
        textfont=dict(size=9, color="white"),
        hovertext=hover_texts,
        hoverinfo="text",
    ))

    if playoff_weeks:
        for pw in playoff_weeks:
            fig.add_vrect(x0=pw - 0.5, x1=pw + 0.5,
                          fillcolor="rgba(25,118,210,0.12)", line_width=0, layer="below")

    fig.add_hline(y=avg_height, line_dash="dash", line_color="#90a4ae",
                  annotation_text=f"avg rank #{avg_rank_sv:.1f}",
                  annotation_position="top right",
                  annotation_font=dict(color="#90a4ae", size=10))

    tick_ranks  = [1, 8, 16, 24, 32]
    tick_vals   = tick_ranks
    tick_labels = [f"#{r}" for r in tick_ranks]

    fig.update_layout(
        title=f"{player_name} — 2026 Weekly {pos} Defense Rank (taller = easier matchup)",
        xaxis=dict(title="Week", tickmode="linear", dtick=1,
                   range=[0.5, schedule_df["Week"].max() + 0.5]),
        yaxis=dict(title="Defense Rank (32 = easiest)", range=[0, 34],
                   tickvals=tick_vals, ticktext=tick_labels),
        height=480,
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        bargap=0.2,
        margin=dict(t=50),
    )
    st.plotly_chart(fig, width="stretch")

    # Schedule table
    st.subheader("Full Schedule")
    sched_display = schedule_df[["Week", "Opponent", "Opp_Full", "Def_Rank", "FP_per_game", "Is_Playoff"]].copy()
    sched_display.columns = ["Week", "Opp", "Opponent", "Def Rank", "Def FP/G", "Playoff?"]
    sched_display["Def Rank"] = sched_display["Def Rank"].apply(lambda r: f"#{int(r)}" if pd.notna(r) else "—")
    sched_display["Def FP/G"] = sched_display["Def FP/G"].round(1)

    def color_row(row):
        if row["Playoff?"]:
            return ["background-color: rgba(25,118,210,0.2)"] * len(row)
        fpg = row["Def FP/G"]
        if isinstance(fpg, float) and not np.isnan(fpg):
            if fpg >= league_avg_fpg * 1.10:
                return ["background-color: rgba(76,175,80,0.15)"] * len(row)
            if fpg <= league_avg_fpg * 0.90:
                return ["background-color: rgba(239,83,80,0.15)"] * len(row)
        return [""] * len(row)

    st.dataframe(
        sched_display.style.apply(color_row, axis=1),
        width="stretch",
        hide_index=True,
        height=600,
    )

    st.markdown(
        "🔵 **Playoff week** &nbsp; 🟢 **Easy matchup** (≥ +10% above avg) &nbsp; 🟠 **Average** &nbsp; 🔴 **Tough** (≥ −10% below avg)",
        unsafe_allow_html=True,
    )

    # ── Schedule Complement Finder ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader(f"🔄 Best Ball Schedule Complements — {pos}")
    st.caption(
        f"Same-position players with the easiest schedules in **{player_name}'s tough weeks**. "
        "Pair these on your roster so someone always has a good matchup."
    )

    # Classify searched player's weeks
    HARD_THRESHOLD = 13   # def rank ≤ 13 = tough (bottom third, 32=easiest)
    EASY_THRESHOLD = 21   # def rank ≥ 21 = easy (top third)

    hard_weeks = sorted(schedule_df[
        schedule_df["Def_Rank"].notna() & (schedule_df["Def_Rank"] <= HARD_THRESHOLD)
    ]["Week"].tolist())

    if not hard_weeks:
        st.info(f"No clearly tough weeks found for {player_name} — their schedule is quite favorable all season.")
    else:
        st.markdown(f"**{player_name}'s tough weeks:** {', '.join(f'Wk {w}' for w in hard_weeks)}")

        # Score all same-position players by how easy their schedule is in those tough weeks
        pos_players = fp_all[fp_all["POS"] == pos].copy()
        complement_scores = []

        for _, p in pos_players.iterrows():
            pname_c  = p["Name"]
            pteam_c  = p["Team"]
            if pname_c == player_name or pd.isna(pteam_c):
                continue

            p_sched = get_team_schedule(pteam_c, sched)
            if p_sched.empty:
                continue

            week_details = []
            for wk in hard_weeks:
                wk_row = p_sched[p_sched["Week"] == wk]
                if wk_row.empty:
                    week_details.append({"week": wk, "rank": None, "bye": True})
                    continue
                opp = wk_row.iloc[0]["Opponent"]
                rk  = rank_map.get(opp)
                week_details.append({"week": wk, "rank": rk, "bye": False})

            valid_ranks = [d["rank"] for d in week_details if d["rank"] is not None]
            if not valid_ranks:
                continue

            avg_rank_in_hard  = np.mean(valid_ranks)
            easy_overlap_count = sum(1 for r in valid_ranks if r >= EASY_THRESHOLD)
            # Score: higher avg rank = easier matchups = better complement (32=easiest)
            complement_score  = avg_rank_in_hard + easy_overlap_count * 3

            week_label = " | ".join(
                f"Wk{d['week']} #{d['rank']}" if d["rank"] is not None
                else f"Wk{d['week']} BYE"
                for d in week_details
            )

            complement_scores.append({
                "Player":        pname_c,
                "Team":          pteam_c,
                "Pos Rank":      f"{pos}{int(p['FP_Pos_Rank'])}" if pd.notna(p["FP_Pos_Rank"]) else "—",
                "ADP":           round(p["FP_ADP"], 1) if pd.notna(p["FP_ADP"]) else None,
                "Avg Rank (tough wks)": round(avg_rank_in_hard, 1),
                "Easy Overlaps": easy_overlap_count,
                "_score":        complement_score,
                "_week_detail":  week_label,
            })

        if not complement_scores:
            st.info("Not enough schedule data to compute complements.")
        else:
            comp_df = (pd.DataFrame(complement_scores)
                       .sort_values("_score", ascending=False)
                       .reset_index(drop=True))
            comp_df.index += 1

            # Top complement table
            show_df = comp_df[["Player", "Team", "Pos Rank", "ADP",
                                "Easy Overlaps", "Avg Rank (tough wks)", "_week_detail"]].head(20).copy()
            show_df.columns = ["Player", "Team", "Pos Rank", "ADP",
                                f"Easy Wks (of {len(hard_weeks)})", "Avg Rank", "Matchups in Your Tough Weeks"]

            def color_complement(row):
                easy = row[f"Easy Wks (of {len(hard_weeks)})"]
                if easy >= max(1, len(hard_weeks) * 0.6):
                    return ["background-color: rgba(76,175,80,0.20)"] * len(row)
                if easy == 0:
                    return ["background-color: rgba(239,83,80,0.10)"] * len(row)
                return [""] * len(row)

            st.dataframe(
                show_df.style.apply(color_complement, axis=1),
                width="stretch",
                hide_index=False,
                height=min(60 + len(show_df) * 35, 650),
            )
            st.caption("🟢 Strong complement (easy in most tough weeks) · #rank shown per week · lower = easier")

    # ── Weeks 15-17 opponent offense preview ──────────────────────────────────
    st.markdown("---")
    st.subheader("📋 Weeks 15–17 Opponent Offense Preview")
    st.caption("Players on the opposing offense, ordered by FP rank. Use for best-ball stacks.")

    fp_data = load_fp_rankings().rename(columns={"Name_clean": "Name"})
    preview_wks = [15, 16, 17]
    late_sched = schedule_df[schedule_df["Week"].isin(preview_wks)].set_index("Week")

    wcols = st.columns(3)
    for ci, wk in enumerate(preview_wks):
        with wcols[ci]:
            if wk not in late_sched.index:
                st.markdown(f"**Wk {wk}:** BYE")
                continue
            opp_abb  = late_sched.loc[wk, "Opponent"]
            opp_full = ABB_TO_FULL.get(opp_abb, opp_abb)
            st.markdown(f"**Wk {wk} @ {opp_full}**")
            opp_players = fp_data[
                (fp_data["Team"] == opp_abb) &
                (fp_data["POS"].isin(["QB", "WR", "RB", "TE"]))
            ].sort_values("FP_Rank", ascending=True, na_position="last")

            if opp_players.empty:
                st.caption("No players found")
            else:
                rows_out = [
                    {"Rank": int(op["FP_Rank"]) if pd.notna(op["FP_Rank"]) else "—",
                     "Player": op["Name"], "POS": op["POS"],
                     "ADP": round(op["FP_ADP"], 1) if pd.notna(op["FP_ADP"]) else "—"}
                    for _, op in opp_players.iterrows()
                ]
                st.dataframe(
                    pd.DataFrame(rows_out),
                    width="stretch",
                    hide_index=True,
                    height=min(40 + len(rows_out) * 35, 420),
                )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 5 — WEEKLY PROJECTIONS COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif tab_choice == "📉 Weekly Projections":
    st.header("📉 Weekly Projection Comparison")
    st.caption("Compare week-by-week projected scores for any players. Projections use FPTS ÷ 17 as base rate, adjusted for opponent defense strength each week.")

    _all_proj    = load_all_projections()
    _bye_map     = compute_bye_weeks()
    _matchup_adj = load_defense_matchup_adj()
    _proj_lu_wp  = _build_proj_lu(_all_proj, _bye_map, matchup_adj=_matchup_adj)

    # Build sorted player list for the multiselect
    _all_names = sorted(_proj_lu_wp.keys())

    # Defaults: top RBs + WRs to start
    _default_players = [n for n in [
        "Jahmyr Gibbs", "Bijan Robinson", "Puka Nacua", "Jaxon Smith-Njigba", "Ja'Marr Chase"
    ] if n in _proj_lu_wp]

    _MAX_PLAYERS = 5
    wpc1, wpc2 = st.columns([3, 1])
    with wpc1:
        _selected = st.multiselect(
            f"Select up to {_MAX_PLAYERS} players to compare (type to search)",
            options=_all_names,
            default=_default_players[:_MAX_PLAYERS],
            max_selections=_MAX_PLAYERS,
            key="wp_players",
        )
    with wpc2:
        _show_bye = st.checkbox("Show bye week gaps", value=True, key="wp_bye_gaps")

    if not _selected:
        st.info("Select at least one player above.")
    else:
        # Distinct color palette — one per player slot regardless of position
        _PALETTE = ["#EF5350", "#42A5F5", "#66BB6A", "#FFA726", "#AB47BC"]

        import plotly.graph_objects as go

        fig = go.Figure()

        _table_rows = []
        for _pi, nm in enumerate(_selected):
            p = _proj_lu_wp[nm]
            pos      = p["pos"]
            bye      = p.get("bye", 0)
            proj_ppw = p["proj_ppw"]
            wppw     = p.get("weekly_ppw") or {}
            color    = _PALETTE[_pi % len(_PALETTE)]

            weeks = list(range(1, 18))
            pts   = []
            for wk in weeks:
                if wk == bye and _show_bye:
                    pts.append(None)   # gap in line
                else:
                    pts.append(wppw.get(wk, proj_ppw))

            fig.add_trace(go.Scatter(
                x=weeks,
                y=pts,
                mode="lines+markers",
                name=f"{nm} ({pos})",
                line=dict(color=color, width=2),
                marker=dict(size=6),
                connectgaps=not _show_bye,
                hovertemplate=f"<b>{nm}</b><br>Week %{{x}}: %{{y:.1f}} pts<extra></extra>",
            ))

            # Build table row
            row = {"Player": nm, "POS": pos, "Team": p["team"],
                   "Bye": bye or "—",
                   "Base/wk": round(proj_ppw, 1)}
            for wk in weeks:
                if wk == bye:
                    row[f"W{wk}"] = "BYE"
                else:
                    row[f"W{wk}"] = round(wppw.get(wk, proj_ppw), 1)
            _table_rows.append(row)

        fig.update_layout(
            xaxis=dict(title="Week", tickmode="linear", tick0=1, dtick=1,
                       tickvals=list(range(1, 18))),
            yaxis=dict(title="Projected Points"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            hovermode="x unified",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e0e0"),
            height=580,
            margin=dict(l=40, r=20, t=60, b=40),
        )
        fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)")
        fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)", zeroline=False)

        st.plotly_chart(fig, use_container_width=True)

        # Week-by-week table
        with st.expander("📋 Week-by-week numbers", expanded=False):
            if _table_rows:
                _tdf = pd.DataFrame(_table_rows)
                week_cols = [c for c in _tdf.columns if c.startswith("W")]

                def _color_wp(val):
                    if val == "BYE":
                        return "color: #888"
                    return ""

                st.dataframe(
                    _tdf.style.map(_color_wp, subset=week_cols),
                    width="stretch",
                    hide_index=True,
                    height=min(60 + len(_table_rows) * 38, 400),
                )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 6 — DRAFT ROOM
# ══════════════════════════════════════════════════════════════════════════════
elif tab_choice == "🎯 Draft Room":
    st.header("🎯 Live Draft Room")
    st.caption("Track picks in real time — add yours and others' to see who's available and who to target next.")

    # ── Session state ──────────────────────────────────────────────────────────
    if "draft_board" not in st.session_state or not isinstance(st.session_state.draft_board, pd.DataFrame):
        st.session_state.draft_board = pd.DataFrame({
            "Pick": list(range(1, 301)),
            "Player": [""] * 300,
            "My Pick": [False] * 300,
        })
    if "my_picks"          not in st.session_state: st.session_state.my_picks          = []
    if "other_picks"       not in st.session_state: st.session_state.other_picks       = []
    if "pick_key"          not in st.session_state: st.session_state.pick_key          = 0
    if "board_editor_key"  not in st.session_state: st.session_state.board_editor_key  = 0
    if "draft_num_teams"   not in st.session_state: st.session_state.draft_num_teams   = 12
    if "draft_my_slot"     not in st.session_state: st.session_state.draft_my_slot     = 6
    if "draft_total_rounds" not in st.session_state: st.session_state.draft_total_rounds = 20

    # ── Load data ──────────────────────────────────────────────────────────────
    fp_all         = load_fp_rankings().rename(columns={"Name_clean": "Name"})
    bye_map        = compute_bye_weeks()
    boom_rates     = compute_boom_rates()
    playoff_scores = compute_all_playoff_scores()
    _all_proj      = load_all_projections()
    _hist_var      = compute_historical_variance()
    _matchup_adj   = load_defense_matchup_adj()
    _proj_lu       = _build_proj_lu(_all_proj, bye_map, var_lu=_hist_var, matchup_adj=_matchup_adj)
    sched_matrix   = build_team_schedule_matrix()   # (team, pos_key) → {week: rank}

    HARD_THR = 13   # rank ≤ 13 = tough (32=easiest system)
    EASY_THR = 21   # rank ≥ 21 = easy

    # ── Draft settings ─────────────────────────────────────────────────────────
    _team_opts  = [6, 8, 10, 12]
    _round_opts = [15, 18, 20, 22, 24]

    # Apply saved-draft settings BEFORE widgets render (avoids "cannot be modified" error)
    if "pending_load" in st.session_state:
        _pl = st.session_state.pop("pending_load")
        _nt = _pl.get("num_teams", 12)
        _tr = _pl.get("total_rounds", 20)
        st.session_state.sb_num_teams    = _nt if _nt in _team_opts  else 12
        st.session_state.sb_my_slot      = _pl.get("my_slot", 6)
        st.session_state.sb_total_rounds = _tr if _tr in _round_opts else 20

    # Pre-set widget keys so selectboxes start at our defaults on first load
    if "sb_num_teams"    not in st.session_state: st.session_state.sb_num_teams    = 12
    if "sb_total_rounds" not in st.session_state: st.session_state.sb_total_rounds = 20
    if "sb_my_slot"      not in st.session_state: st.session_state.sb_my_slot      = 6

    ds1, ds2, ds3 = st.columns([1, 1, 1])
    with ds1:
        num_teams = st.selectbox("# Teams in Draft", _team_opts, key="sb_num_teams")
    with ds2:
        my_slot = st.number_input("Your Draft Slot", min_value=1, max_value=num_teams,
                                  value=min(st.session_state.sb_my_slot, num_teams),
                                  key="sb_my_slot")
    with ds3:
        total_rounds = st.selectbox("Total Rounds", _round_opts, key="sb_total_rounds")

    # ── Draft management (save / load / reset) ─────────────────────────────────
    dm1, dm2, dm3, dm4 = st.columns([3, 1, 1, 1])
    with dm1:
        draft_name = st.text_input("Draft name", placeholder="e.g. DK Slow Draft 1",
                                   label_visibility="collapsed", key="draft_name_input")
    with dm2:
        _save_data = {
            "name":         draft_name or "draft",
            "saved_at":     datetime.datetime.now().isoformat(timespec="seconds"),
            "num_teams":    num_teams,
            "my_slot":      my_slot,
            "total_rounds": total_rounds,
            "board":        st.session_state.draft_board.to_dict(orient="records"),
        }
        _fname = (draft_name or "draft").replace(" ", "_") + ".json"
        st.download_button("💾 Save", json.dumps(_save_data), file_name=_fname,
                           mime="application/json", width="stretch")
    with dm3:
        _uploaded = st.file_uploader("Load", type=["json"], label_visibility="collapsed",
                                     key="draft_upload")
        if _uploaded is not None:
            try:
                _loaded = json.loads(_uploaded.read())
                st.session_state.draft_board  = pd.DataFrame(_loaded["board"])
                # Store settings in pending_load — applied before widgets render on next rerun
                st.session_state.pending_load = {
                    "num_teams":    _loaded.get("num_teams", 12),
                    "my_slot":      _loaded.get("my_slot", 6),
                    "total_rounds": _loaded.get("total_rounds", 20),
                }
                st.session_state.pick_key         = st.session_state.get("pick_key", 0) + 1
                st.session_state.board_editor_key = st.session_state.get("board_editor_key", 0) + 1
                st.session_state.my_picks         = []
                st.session_state.other_picks      = []
                st.rerun()
            except Exception as e:
                st.error(f"Could not load draft: {e}")
    with dm4:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ Reset", width="stretch"):
            st.session_state.my_picks         = []
            st.session_state.other_picks      = []
            st.session_state.pick_key         = st.session_state.get("pick_key", 0) + 1
            st.session_state.board_editor_key = st.session_state.get("board_editor_key", 0) + 1
            st.session_state.draft_board      = pd.DataFrame({
                "Pick": list(range(1, 301)),
                "Player": [""] * 300,
                "My Pick": [False] * 300,
            })
            st.rerun()

    # ── DraftKings text-paste importer ────────────────────────────────────────
    if "dk_scan_results" not in st.session_state: st.session_state.dk_scan_results = None
    if "dk_paste_key"    not in st.session_state: st.session_state.dk_paste_key    = 0
    if "dk_exp_open"     not in st.session_state: st.session_state.dk_exp_open     = False

    # Keep expander open while there is content to review
    _dk_exp_default = st.session_state.dk_exp_open or bool(st.session_state.dk_scan_results)

    with st.expander("📋 Import from DraftKings — paste draft board text",
                     expanded=_dk_exp_default):
        st.caption(
            "On the DraftKings draft board page: select all (Ctrl+A), copy (Ctrl+C), "
            "then paste into the box below."
        )

        _dk_paste = st.text_area("Paste draft board text here", height=160,
                                  key=f"dk_paste_input_{st.session_state.dk_paste_key}",
                                  placeholder="Paste the copied DraftKings draft board text here…",
                                  label_visibility="collapsed")

        _pb1, _pb2 = st.columns([3, 1])
        with _pb1:
            _do_parse = st.button("Parse & Preview", key="dk_parse_btn",
                                  disabled=not _dk_paste.strip(), width="stretch")
        with _pb2:
            if st.button("🗑️ Clear", key="dk_clear_btn", width="stretch"):
                st.session_state.dk_paste_key   += 1
                st.session_state.dk_scan_results = None
                st.session_state.dk_exp_open     = True   # keep expander open after rerun
                st.rerun()

        if _do_parse:
            st.session_state.dk_scan_results = None
            _text     = _dk_paste.strip()
            _sections = re.split(r'User Avatar\s*\n', _text)
            _sections = [s for s in _sections if s.strip()]
            _pick_pat = re.compile(r'^(\d+)\n([^\n]+\s+icon)', re.MULTILINE)

            _raw_picks = []
            for _col_idx, _sec in enumerate(_sections, start=1):
                for _m in _pick_pat.finditer(_sec):
                    _overall = int(_m.group(1))
                    _name    = re.sub(r'\s*icon\s*$', '', _m.group(2), flags=re.I).strip()
                    if _name:
                        _raw_picks.append((_overall, _col_idx, _name))

            if not _raw_picks:
                st.error("No picks found — make sure you selected all and copied the full "
                         "DraftKings draft board page.")
            else:
                _raw_picks.sort(key=lambda x: x[0])
                _exact_lu = {n.lower(): n for n in fp_all["Name"].tolist()}
                _sfx_set  = {'jr', 'sr', 'ii', 'iii', 'iv', 'v', 'jr.', 'sr.'}

                def _canon_full(raw):
                    tl = raw.strip().lower()
                    # Check alias map first (e.g. "kenneth gainwell" → "Kenny Gainwell")
                    if tl in _NAME_ALIASES_LC: return _canon_full(_NAME_ALIASES_LC[tl])
                    if tl in _exact_lu: return _exact_lu[tl]
                    _words = tl.split()
                    _core  = [w for w in _words if w.rstrip('.') not in _sfx_set]
                    _core_str = ' '.join(_core)
                    if _core_str != tl and _core_str in _exact_lu: return _exact_lu[_core_str]
                    _ws = [w for w in _core if len(w) > 1]
                    if _ws:
                        _hits = [v for k, v in _exact_lu.items() if all(w in k for w in _ws)]
                        if len(_hits) == 1: return _hits[0]
                    return raw

                st.session_state.dk_scan_results = [
                    (_col, _canon_full(_nm)) for _, _col, _nm in _raw_picks
                ]
                st.session_state.dk_exp_open = True
                # no st.rerun() — results block below renders in this same pass

        if st.session_state.dk_scan_results:
            _owc = st.session_state.dk_scan_results
            st.success(f"Found {len(_owc)} picks — your picks are in column {my_slot}. Review then confirm.")
            st.dataframe(
                pd.DataFrame({
                    "#":       range(1, len(_owc) + 1),
                    "Player":  [nm for _, nm in _owc],
                    "My Pick": ["✓" if c == my_slot else "" for c, _ in _owc],
                }),
                width="stretch", height=220, hide_index=True,
            )
            _bc1, _bc2 = st.columns(2)
            with _bc1:
                if st.button("✅ Confirm & Import", key="dk_confirm_btn"):
                    _new_board = pd.DataFrame({"Pick": list(range(1, 301)),
                                               "Player": [""] * 300, "My Pick": [False] * 300})
                    for _i, (_c, _pl) in enumerate(_owc):
                        if _i >= 300: break
                        _new_board.loc[_i, "Player"]  = _pl
                        _new_board.loc[_i, "My Pick"] = (_c == my_slot)
                    st.session_state.draft_board      = _new_board
                    st.session_state.dk_scan_results  = None
                    st.session_state.dk_exp_open      = False
                    st.session_state.board_editor_key = st.session_state.get("board_editor_key", 0) + 1
                    st.session_state.pick_key         = st.session_state.get("pick_key", 0) + 1
                    st.session_state.my_picks         = []
                    st.session_state.other_picks      = []
                    st.rerun()
            with _bc2:
                if st.button("🔄 Clear & Re-paste", key="dk_rescan_btn"):
                    st.session_state.dk_scan_results = None
                    st.session_state.dk_exp_open     = True
                    # no st.rerun() — same pass hides results, expander stays open

    # ── Quick pick entry ───────────────────────────────────────────────────────
    n_picks = num_teams * total_rounds
    _already_logged = set(
        st.session_state.draft_board.loc[
            st.session_state.draft_board["Player"].notna() &
            (st.session_state.draft_board["Player"] != ""), "Player"
        ]
    )
    player_opts = [""] + [
        n for n in fp_all.sort_values("FP_ADP")["Name"].tolist()
        if n not in _already_logged
    ]

    logged = st.session_state.draft_board[
        st.session_state.draft_board["Player"].notna() &
        (st.session_state.draft_board["Player"] != "")
    ]
    next_pick_num = int(logged["Pick"].max()) + 1 if not logged.empty else 1
    next_pick_num = min(next_pick_num, n_picks)

    def _is_my_slot(pick_num):
        rnd  = (pick_num - 1) // num_teams + 1
        slot = (pick_num - 1) % num_teams + 1
        my_pos = my_slot if rnd % 2 == 1 else num_teams + 1 - my_slot
        return slot == my_pos

    is_my_turn_now = _is_my_slot(next_pick_num)
    pick_label = f"{'⭐ ' if is_my_turn_now else ''}Log Pick #{next_pick_num}"

    qc1, qc2, qc3, qc4 = st.columns([5, 1, 1, 1])
    with qc1:
        new_player = st.selectbox(pick_label, options=player_opts,
                                  key=f"quick_pick_{st.session_state.get('pick_key', 0)}")
    with qc2:
        new_is_mine = st.checkbox("My Pick", value=is_my_turn_now,
                                  key=f"quick_mine_{st.session_state.get('pick_key', 0)}")
    with qc3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("➕ Log", width="stretch") and new_player:
            idx = next_pick_num - 1
            st.session_state.draft_board.loc[idx, "Player"]  = new_player
            st.session_state.draft_board.loc[idx, "My Pick"] = new_is_mine
            st.session_state["pick_key"] = st.session_state.get("pick_key", 0) + 1
            st.rerun()
    with qc4:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("↩️ Undo", width="stretch") and not logged.empty:
            last_idx = int(logged["Pick"].max()) - 1
            st.session_state.draft_board.loc[last_idx, "Player"]  = ""
            st.session_state.draft_board.loc[last_idx, "My Pick"] = False
            st.session_state["pick_key"] = st.session_state.get("pick_key", 0) + 1
            st.rerun()

    # ── Draft Board display (editable for corrections) ─────────────────────────
    board = st.session_state.draft_board.head(n_picks).copy()
    board["Rnd"]   = ((board["Pick"] - 1) // num_teams + 1).astype(int)
    board["Yours"] = board["Pick"].apply(_is_my_slot)

    edited = st.data_editor(
        board[["Pick", "Rnd", "Yours", "Player", "My Pick"]],
        column_config={
            "Pick":    st.column_config.NumberColumn("Pick",   disabled=True, width="small"),
            "Rnd":     st.column_config.NumberColumn("Rnd",    disabled=True, width="small"),
            "Yours":   st.column_config.CheckboxColumn("Yours?", disabled=True, width="small"),
            "Player":  st.column_config.TextColumn("Player (edit to correct)", width="large"),
            "My Pick": st.column_config.CheckboxColumn("My Pick", width="small"),
        },
        hide_index=True,
        width="stretch",
        height=350,
        key=f"draft_board_editor_{st.session_state.board_editor_key}",
    )

    # Only write back when user actually changed something in the editor
    _cur_players = st.session_state.draft_board.loc[:n_picks - 1, "Player"].values
    _cur_mines   = st.session_state.draft_board.loc[:n_picks - 1, "My Pick"].values
    _ed_players  = edited["Player"].values
    _ed_mines    = edited["My Pick"].values
    if not ((_cur_players == _ed_players).all() and (_cur_mines == _ed_mines).all()):
        st.session_state.draft_board.loc[:n_picks - 1, "Player"]  = _ed_players
        st.session_state.draft_board.loc[:n_picks - 1, "My Pick"] = _ed_mines

    # Derive my_picks / other_picks for all analysis below
    filled      = edited[edited["Player"].notna() & (edited["Player"] != "")]
    my_names    = filled[filled["My Pick"]]["Player"].tolist()
    other_names = filled[~filled["My Pick"]]["Player"].tolist()

    st.session_state.my_picks = [
        {"Name": r["Name"], "POS": r["POS"], "Team": r["Team"],
         "FP_Rank": r["FP_Rank"], "FP_ADP": r["FP_ADP"]}
        for name in my_names
        for _, r in fp_all[fp_all["Name"] == name].head(1).iterrows()
    ]
    st.session_state.other_picks = other_names

    # ── Available pool (derived from board) ────────────────────────────────────
    all_drafted = set(my_names) | set(other_names)
    available   = fp_all[~fp_all["Name"].isin(all_drafted)].sort_values("FP_ADP").reset_index(drop=True)

    # ── Pre-compute marginal values (displayed above Available Players) ─────────
    _marginal:      list  = []
    _base_weekly:   list  = []
    _weak_weeks:    set   = set()
    _avg_weekly:    float = 0.0
    _base_total:    float = 0.0
    _weak_thresh:   float = 0.0
    _has_variation: bool  = False

    if my_names:
        _base_total, _base_weekly = _project_bb_score(my_names, _proj_lu)
        _my_bye_wks_pre = {_proj_lu.get(n, {}).get("bye") for n in my_names}
        _non_bye_pre    = [(wk, sc) for wk, sc in enumerate(_base_weekly, 1)
                           if wk not in _my_bye_wks_pre and sc > 0]
        if _non_bye_pre:
            _avg_weekly = sum(sc for _, sc in _non_bye_pre) / len(_non_bye_pre)
            _std_pre    = (sum((s - _avg_weekly) ** 2
                               for _, s in _non_bye_pre) / len(_non_bye_pre)) ** 0.5
            _has_variation = _std_pre >= 2.0
            if _has_variation:
                _sorted_nb  = sorted(_non_bye_pre, key=lambda x: x[1])
                _n_weak     = min(5, len(_sorted_nb))
                _weak_weeks = {wk for wk, _ in _sorted_nb[:_n_weak]}
                _weak_thresh = _sorted_nb[_n_weak - 1][1]

        for _, _ar in available.iterrows():
            _nm = _ar["Name"]
            if _nm not in _proj_lu:
                continue
            _new_total, _new_weekly = _project_bb_score(my_names + [_nm], _proj_lu)
            _delta = round(_new_total - _base_total, 1)
            if _delta > 0:
                _p   = _proj_lu[_nm]
                _std = _p.get("std_fpg", 0.0)
                _ww_gain = (round(sum(_new_weekly[wk - 1] - _base_weekly[wk - 1]
                                      for wk in _weak_weeks), 1)
                            if _weak_weeks else 0.0)
                _marginal.append({
                    "Player":  _nm,
                    "POS":     _p["pos"],
                    "Team":    _p["team"],
                    "Proj/G":  round(_p["proj_pg"], 1),
                    "Std/G":   round(_std, 1) if _std > 0 else pd.NA,
                    "Bye":     _p["bye"] or "—",
                    "ADP":     round(float(_ar["FP_ADP"]), 1) if pd.notna(_ar.get("FP_ADP")) else pd.NA,
                    "+Pts":    _delta,
                    "WW+Pts":  _ww_gain if _ww_gain > 0 else pd.NA,
                })

    st.markdown("---")

    # ── Pick tracker ───────────────────────────────────────────────────────────
    total_made    = len(st.session_state.my_picks) + len(st.session_state.other_picks)
    current_pick  = total_made + 1
    round_num     = (current_pick - 1) // num_teams + 1
    pick_in_round = (current_pick - 1) % num_teams + 1

    my_turn_this_round = my_slot if round_num % 2 == 1 else num_teams + 1 - my_slot
    picks_until = my_turn_this_round - pick_in_round
    if picks_until < 0:
        nr = round_num + 1
        next_my_turn = my_slot if nr % 2 == 1 else num_teams + 1 - my_slot
        picks_until  = (num_teams - pick_in_round) + (next_my_turn - 1)

    tm1, tm2, tm3, tm4, tm5 = st.columns(5)
    tm1.metric("Overall Pick",    f"#{current_pick}")
    tm2.metric("Round",           f"{round_num} of {total_rounds}")
    tm3.metric("Picks Until Yours", "NOW 🎯" if picks_until == 0 else picks_until)
    tm4.metric("My Picks",        f"{len(st.session_state.my_picks)} / {total_rounds}")
    tm5.metric("Available",       len(available))

    # ── Pre-compute all scores on full available pool ──────────────────────────
    opp_lookup = build_team_opponent_lookup()

    my_teams        = {p["Team"] for p in st.session_state.my_picks}
    my_team_players = {}
    for p in st.session_state.my_picks:
        my_team_players.setdefault(p["Team"], []).append(p["Name"])

    # Roster tough weeks per position (WR complement only checks WR tough weeks, etc.)
    pos_tough_weeks: dict = {}
    for p in st.session_state.my_picks:
        for wk, rk in sched_matrix.get((p["Team"], p["POS"]), {}).items():
            if rk <= HARD_THR:
                pos_tough_weeks.setdefault(p["POS"], set()).add(wk)
    roster_tough_weeks = set().union(*pos_tough_weeks.values()) if pos_tough_weeks else set()
    n_tough = len(roster_tough_weeks)

    avail_all = available.copy()
    avail_all["Bye Wk"]      = avail_all["Team"].map(bye_map)
    avail_all["Playoff Scr"] = avail_all["Name"].map(playoff_scores)
    avail_all["Boom%"]       = avail_all["Name"].map(boom_rates)
    avail_all["_fell"]       = (current_pick - avail_all["FP_ADP"])
    avail_all["Fell"]        = avail_all["_fell"].round(0).astype("Int64").where(
                                   avail_all["_fell"] > 0, other=pd.NA)

    # Complement scores — pair each available player against each individual roster player
    # at the same position. Best individual pairing wins.
    my_pick_data = [
        {"name": p["Name"], "pos": p["POS"],
         "tough_wks": {wk for wk, rk in sched_matrix.get((p["Team"], p["POS"]), {}).items()
                       if rk <= HARD_THR}}
        for p in st.session_state.my_picks
    ]

    def _cmpl_best(row):
        best_score, best_denom, best_name = 0, 0, ""
        for rp in my_pick_data:
            if rp["pos"] != row["POS"] or not rp["tough_wks"]:
                continue
            easy = sum(1 for wk in rp["tough_wks"]
                       if sched_matrix.get((row["Team"], row["POS"]), {}).get(wk, 0) >= EASY_THR)
            if easy > best_score or (easy == best_score and len(rp["tough_wks"]) > best_denom):
                best_score = easy
                best_denom = len(rp["tough_wks"])
                best_name  = rp["name"]
        return best_score, best_denom, best_name

    if st.session_state.my_picks:
        cmpl_result = avail_all.apply(
            lambda r: pd.Series(_cmpl_best(r), index=["_cmpl", "_cmpl_denom", "_cmpl_vs"]), axis=1)
        avail_all[["_cmpl", "_cmpl_denom", "_cmpl_vs"]] = cmpl_result
        avail_all["Cmpl"] = avail_all.apply(
            lambda r: f"{int(r['_cmpl'])}/{int(r['_cmpl_denom'])}" if r["_cmpl_vs"] else "—", axis=1)
        avail_all["Fits"] = avail_all["_cmpl_vs"].apply(
            lambda n: n.split()[-1] if n else "")   # last name of the roster player
    else:
        avail_all["_cmpl"]       = 0
        avail_all["_cmpl_denom"] = 0
        avail_all["_cmpl_vs"]    = ""
        avail_all["Cmpl"]        = "—"
        avail_all["Fits"]        = ""

    # Playoff opponent detection: available player faces one of my teams in wks 15-17
    def _faces_my(avail_team):
        parts = []
        for wk in [15, 16, 17]:
            opp = opp_lookup.get(avail_team, {}).get(wk)
            if opp and opp in my_teams:
                parts.append(f"W{wk} vs {opp}")
        return " · ".join(parts)
    avail_all["Faces My"] = avail_all["Team"].apply(_faces_my)

    # ── Recommendations (full width) ──────────────────────────────────────────
    st.markdown("---")
    st.subheader("🎯 Draft Recommendations")

    pos_targets = {"QB": 3, "RB": 6, "WR": 8, "TE": 3}
    pos_have    = {pos: sum(1 for p in st.session_state.my_picks if p["POS"] == pos)
                   for pos in pos_targets}
    remaining   = max(total_rounds - len(st.session_state.my_picks), 1)

    c_need, c_fits, c_val, c_opp = st.columns(4)

    with c_need:
        st.markdown("**📍 Positional Need**")
        need_order = sorted(pos_targets, key=lambda p: (pos_targets[p] - pos_have[p]) / remaining, reverse=True)
        shown = 0
        for pos in need_order:
            need = max(0, pos_targets[pos] - pos_have[pos])
            if need <= 0:
                continue
            pct  = pos_have[pos] / pos_targets[pos]
            icon = "🔴" if pct < 0.25 else "🟡" if pct < 0.6 else "🟢"
            st.caption(f"{icon} **{pos}**: {pos_have[pos]}/{pos_targets[pos]} — need {need}")
            # Show players with ADP near your next pick (±15 picks); fall back to best overall
            _near = avail_all[
                (avail_all["POS"] == pos) &
                (avail_all["FP_ADP"] >= current_pick - 5) &
                (avail_all["FP_ADP"] <= current_pick + 18)
            ].sort_values("FP_ADP")
            _pool = _near if not _near.empty else avail_all[avail_all["POS"] == pos]
            for _, r in _pool.head(3).iterrows():
                adp_s = f"ADP {r['FP_ADP']:.1f}" if pd.notna(r["FP_ADP"]) else ""
                fell_s = f" ↓{int(r['_fell'])}" if pd.notna(r.get("_fell")) and r["_fell"] > 0 else ""
                st.markdown(f"&nbsp;&nbsp;**{r['Name']}** ({r['Team']}) {adp_s}{fell_s}")
            shown += 1
            if shown >= 2:
                break

    with c_fits:
        st.markdown("**🔄 Schedule Fits**")
        if st.session_state.my_picks:
            fits = avail_all[avail_all["_cmpl"] > 0].sort_values(
                ["_cmpl", "_cmpl_denom", "FP_ADP"], ascending=[False, False, True])
            shown = 0
            for _, r in fits.iterrows():
                if shown >= 6:
                    break
                adp_s  = f"ADP {r['FP_ADP']:.1f}" if pd.notna(r["FP_ADP"]) else ""
                pct    = r["_cmpl"] / r["_cmpl_denom"] if r["_cmpl_denom"] > 0 else 0
                icon   = "🟢" if pct >= 0.6 else "🟡"
                vs_s   = r["_cmpl_vs"].split()[-1] if r["_cmpl_vs"] else ""
                st.markdown(f"{icon} **{r['Name']}** ({r['POS']}·{r['Team']}) {adp_s}")
                st.caption(f"&nbsp;&nbsp;&nbsp;covers {r['Cmpl']} of {vs_s}'s tough wks")
                shown += 1
            if shown == 0:
                st.caption("No clear fits yet.")
        else:
            st.caption("Add picks to see fits.")

    with c_val:
        st.markdown("**💰 Best Value**")
        value_pool = avail_all[avail_all["Fell"].notna()].sort_values("_fell", ascending=False)
        shown = 0
        for _, r in value_pool.iterrows():
            fell = int(r["_fell"])
            if fell <= 0:
                break
            adp_s = f"ADP {r['FP_ADP']:.1f}" if pd.notna(r["FP_ADP"]) else ""
            icon  = "🟡" if fell >= 8 else "⬜"
            st.markdown(f"{icon} **{r['Name']}** ({r['POS']}·{r['Team']}) {adp_s} ↓{fell}")
            shown += 1
            if shown >= 6:
                break
        if shown == 0:
            st.caption("No players past ADP yet.")

    with c_opp:
        st.markdown("**📅 Playoff Opponents (Wks 15–17)**")
        if not st.session_state.my_picks:
            st.caption("Add picks to see opponents.")
        else:
            # opp_details: opp_team → list of (my_player_name, my_team, wk)
            opp_details: dict = {}
            for p in st.session_state.my_picks:
                for wk in [15, 16, 17]:
                    opp = opp_lookup.get(p["Team"], {}).get(wk)
                    if opp:
                        opp_details.setdefault(opp, []).append((p["Name"], p["Team"], wk))

            if not opp_details:
                st.caption("No playoff schedule data.")
            else:
                opp_team_set = set(opp_details.keys())
                opp_players  = avail_all[avail_all["Team"].isin(opp_team_set)].sort_values("FP_ADP")
                if opp_players.empty:
                    st.caption("No ranked players on opposing teams.")
                else:
                    for _, r in opp_players.head(10).iterrows():
                        matchups = opp_details.get(r["Team"], [])
                        adp_s = f"ADP {r['FP_ADP']:.1f}" if pd.notna(r["FP_ADP"]) else ""
                        vs_parts = [f"W{wk} vs your {nm.split()[-1]} ({tm})"
                                    for nm, tm, wk in sorted(matchups, key=lambda x: x[2])]
                        st.markdown(f"**{r['Name']}** ({r['POS']}·{r['Team']}) {adp_s}")
                        st.caption(f"&nbsp;&nbsp;{' · '.join(vs_parts)}")

    # ── Coming up near your pick that faces your players in wks 15-17 ─────────
    if st.session_state.my_picks and my_teams:
        window = avail_all[
            (avail_all["Faces My"] != "") &
            (avail_all["FP_ADP"] >= current_pick - 3) &
            (avail_all["FP_ADP"] <= current_pick + 12)
        ].sort_values("FP_ADP")
        if not window.empty:
            st.markdown("---")
            st.markdown("**⚠️ Upcoming Near Your Pick — Plays Against Your Teams in Wks 15–17**")
            st.caption("Drafting these creates a game-stack correlation in the playoffs.")
            up_cols = st.columns(min(len(window), 4))
            for col, (_, r) in zip(up_cols, window.head(4).iterrows()):
                with col:
                    adp_s = f"ADP {r['FP_ADP']:.1f}" if pd.notna(r["FP_ADP"]) else ""
                    st.info(f"**{r['Name']}**  \n{r['POS']} · {r['Team']}  \n{adp_s}  \n{r['Faces My']}")

    st.markdown("---")

    # ── Main layout ────────────────────────────────────────────────────────────
    left_col, right_col = st.columns([1, 2])

    # ── LEFT: Stack targets + Roster + Bye weeks ──────────────────────────────
    with left_col:
        # ── Stack Targets ────────────────────────────────────────────────────
        my_picks_list = st.session_state.my_picks
        my_pick_teams = {}
        for p in my_picks_list:
            my_pick_teams.setdefault(p["Team"], []).append(p)

        my_next_pick = current_pick + picks_until   # overall pick# when it's my turn next
        _NEAR_WINDOW = 20                           # ADP within ±20 of my next pick

        if my_pick_teams:
            st.subheader("📡 Stack Targets")

            # ── Near-pick opportunities first ────────────────────────────────
            near_rows = []
            for team, drafted in my_pick_teams.items():
                team_avail = available[available["Team"] == team].sort_values("FP_ADP")
                for _, t in team_avail.iterrows():
                    if pd.notna(t["FP_ADP"]) and abs(t["FP_ADP"] - my_next_pick) <= _NEAR_WINDOW:
                        drafted_names = ", ".join(p["Name"].split()[-1] for p in drafted)
                        near_rows.append({
                            "Player": t["Name"],
                            "POS":    t["POS"],
                            "Team":   team,
                            "ADP":    round(t["FP_ADP"], 1),
                            "Stacks with": drafted_names,
                        })
            if near_rows:
                near_df = (pd.DataFrame(near_rows)
                           .sort_values("ADP")
                           .reset_index(drop=True))
                st.markdown(f"**📍 Available near pick #{my_next_pick} (±{_NEAR_WINDOW} ADP)**")

                def _color_near(row):
                    pos_bg = {"QB": "rgba(206,147,216,0.25)", "RB": "rgba(102,187,106,0.25)",
                              "WR": "rgba(66,165,245,0.25)",  "TE": "rgba(255,167,38,0.25)"}
                    return [f"background-color: {pos_bg.get(row['POS'],'')}"] * len(row)

                st.dataframe(
                    near_df.style.apply(_color_near, axis=1),
                    hide_index=True,
                    width="stretch",
                    height=min(60 + len(near_rows) * 35, 280),
                )
            else:
                st.caption(f"No teammates available near pick #{my_next_pick}.")

            # ── Full stack list by team ──────────────────────────────────────
            st.markdown("**All available teammates by team:**")
            for team in sorted(my_pick_teams.keys()):
                drafted   = my_pick_teams[team]
                team_avail = available[available["Team"] == team].sort_values("FP_ADP")
                my_pos_str = " + ".join(p["POS"] for p in drafted)
                st.markdown(f"**{team}** *(have: {my_pos_str})*")
                if team_avail.empty:
                    st.caption("&nbsp;&nbsp;All teammates drafted.")
                else:
                    for _, t in team_avail.iterrows():
                        adp_s  = f"ADP {t['FP_ADP']:.1f}" if pd.notna(t["FP_ADP"]) else ""
                        rank_s = f"#{int(t['FP_Rank'])}" if pd.notna(t["FP_Rank"]) else ""
                        near_s = " 📍" if pd.notna(t["FP_ADP"]) and abs(t["FP_ADP"] - my_next_pick) <= _NEAR_WINDOW else ""
                        st.markdown(f"&nbsp;&nbsp;{t['Name']} · {t['POS']} · {rank_s} {adp_s}{near_s}")
            st.markdown("---")

        st.subheader("My Roster")
        my_picks = st.session_state.my_picks

        if not my_picks:
            st.info("No picks yet — add your first pick above.")
        else:
            POS_ICONS = {"QB": "🟣", "RB": "🟢", "WR": "🔵", "TE": "🟠"}
            for pos in ["QB", "RB", "WR", "TE"]:
                pos_players = [p for p in my_picks if p["POS"] == pos]
                if not pos_players:
                    continue
                st.markdown(f"**{POS_ICONS[pos]} {pos}** ({len(pos_players)})")
                for p in pos_players:
                    bye_s  = f"Bye Wk {bye_map.get(p['Team'])}" if bye_map.get(p["Team"]) else ""
                    rank_s = f"#{int(p['FP_Rank'])}" if pd.notna(p["FP_Rank"]) else ""
                    st.markdown(f"&nbsp;&nbsp;&nbsp;{p['Name']} · {p['Team']} {rank_s} {bye_s}")

        if my_picks:
            st.markdown("---")
            # Compact bye week summary — one line per week
            bye_counts: dict = {}
            for p in my_picks:
                bye = bye_map.get(p["Team"])
                if bye is not None:
                    bye_counts.setdefault(bye, []).append(p["Name"].split()[-1])
            if bye_counts:
                st.markdown("**Bye Weeks**")
                for bye_wk in sorted(bye_counts):
                    names = bye_counts[bye_wk]
                    icon  = "🔴" if len(names) >= 3 else "🟡" if len(names) >= 2 else "🟢"
                    st.caption(f"{icon} Wk {bye_wk}: {', '.join(names)}")

    # ── RIGHT: Scarcity + Available players table ──────────────────────────────
    with right_col:
        st.subheader("Positional Scarcity")
        scarcity = {"QB": (15, 30), "RB": (30, 60), "WR": (40, 80), "TE": (15, 30)}
        sc1, sc2, sc3, sc4 = st.columns(4)
        for col, pos in zip([sc1, sc2, sc3, sc4], ["QB", "RB", "WR", "TE"]):
            ap = available[available["POS"] == pos]
            t1, t2 = scarcity[pos]
            col.metric(pos, f"{len(ap)} left",
                       delta=f"{len(ap[ap['FP_Pos_Rank'] <= t1])} top-{t1} | {len(ap[ap['FP_Pos_Rank'] <= t2])} top-{t2}",
                       delta_color="off")

        # ── Best Available by Points Added ────────────────────────────────────
        st.markdown("---")
        st.subheader("🎯 Best Available by Points Added")

        # Filters rendered unconditionally so session state persists across picks
        if my_names:
            _mf1, _mf2, _mf3, _mf4 = st.columns([2, 1, 1, 1])
            with _mf1:
                _pos_filter_m = st.multiselect("Filter position", ["QB","RB","WR","TE"],
                                                default=["QB","RB","WR","TE"],
                                                key="marg_pos_filter")
            with _mf2:
                _show_m = st.selectbox("Show top", [10, 25, 50], key="marg_show")
            with _mf3:
                _near_pick_only = st.checkbox("Near my pick (±20 ADP)",
                                               help="Filter to players likely available at your next pick",
                                               key="marg_near_pick")
            with _mf4:
                _max_per_pos = st.checkbox("Max 2 per position",
                                            value=True,
                                            help="Prevents one position from flooding the list",
                                            key="marg_max_per_pos")

        if not my_names:
            st.caption("Add your picks to see marginal value recommendations.")
        elif not _marginal:
            # Team is fully optimized — no remaining player improves the score.
            # Show available players ranked by raw 14-week projected total, with +Pts column.
            st.caption("Your roster is fully optimized — no available player increases your projected score. "
                       "Ranked by raw projected total (weeks 1-14).")
            _fallback_rows = []
            for _, _ar in available.iterrows():
                _nm = _ar["Name"]
                if _nm not in _proj_lu:
                    continue
                if _nm not in [r["Player"] for r in _fallback_rows[:999]] and _ar["POS"] not in _pos_filter_m:
                    continue
                _p = _proj_lu[_nm]
                _bye = _p.get("bye", 0)
                _wppw = _p.get("weekly_ppw")
                if _wppw:
                    _proj14 = round(sum(_wppw.get(wk, _p["proj_ppw"])
                                        for wk in range(1, 15) if wk != _bye), 1)
                else:
                    _active = sum(1 for wk in range(1, 15) if wk != _bye)
                    _proj14 = round(_p["proj_ppw"] * _active, 1)
                _new_total_fb, _ = _project_bb_score(my_names + [_nm], _proj_lu)
                _delta_fb = round(_new_total_fb - _base_total, 1)
                _fallback_rows.append({
                    "Player":    _nm,
                    "POS":       _p["pos"],
                    "Team":      _p["team"],
                    "Proj/G":    round(_p["proj_pg"], 1),
                    "Proj 1-14": _proj14,
                    "+Pts":      _delta_fb,
                    "Bye":       _bye or "—",
                    "ADP":       round(float(_ar["FP_ADP"]), 1) if pd.notna(_ar.get("FP_ADP")) else pd.NA,
                })
            if _fallback_rows:
                _fb_df = (pd.DataFrame(_fallback_rows)
                          .sort_values("Proj 1-14", ascending=False)
                          .reset_index(drop=True))
                if _max_per_pos:
                    _pos_counts: dict = {}
                    _keep = []
                    for _, _fr in _fb_df.iterrows():
                        _pc = _pos_counts.get(_fr["POS"], 0)
                        if _pc < 2:
                            _keep.append(True)
                            _pos_counts[_fr["POS"]] = _pc + 1
                        else:
                            _keep.append(False)
                    _fb_df = _fb_df[_keep].reset_index(drop=True)
                _fb_df.index = range(1, len(_fb_df) + 1)

                def _color_fb(row):
                    pos_bg = {"QB": "rgba(206,147,216,0.18)", "RB": "rgba(102,187,106,0.18)",
                              "WR": "rgba(66,165,245,0.18)",  "TE": "rgba(255,167,38,0.18)"}
                    return [f"background-color: {pos_bg.get(row['POS'],'')}"] * len(row)

                st.dataframe(
                    _fb_df.head(_show_m).style.apply(_color_fb, axis=1),
                    width="stretch", hide_index=False,
                    height=min(60 + _show_m * 35, 460),
                )
        else:
            _marg_df = (pd.DataFrame(_marginal)
                          .sort_values("+Pts", ascending=False)
                          .reset_index(drop=True))

            _show_marg = _marg_df[_marg_df["POS"].isin(_pos_filter_m)].copy()
            if _near_pick_only:
                _show_marg = _show_marg[
                    _show_marg["ADP"].notna() &
                    (_show_marg["ADP"] >= current_pick - 5) &
                    (_show_marg["ADP"] <= current_pick + 20)
                ]
            if _max_per_pos:
                _pos_counts2: dict = {}
                _keep2 = []
                for _, _mr in _show_marg.iterrows():
                    _pc2 = _pos_counts2.get(_mr["POS"], 0)
                    if _pc2 < 2:
                        _keep2.append(True)
                        _pos_counts2[_mr["POS"]] = _pc2 + 1
                    else:
                        _keep2.append(False)
                _show_marg = _show_marg[_keep2].reset_index(drop=True)
            _show_marg.index = range(1, len(_show_marg) + 1)

            def _color_marg(row):
                pos_bg = {"QB": "rgba(206,147,216,0.18)", "RB": "rgba(102,187,106,0.18)",
                          "WR": "rgba(66,165,245,0.18)",  "TE": "rgba(255,167,38,0.18)"}
                return [f"background-color: {pos_bg.get(row['POS'],'')}"] * len(row)

            st.dataframe(
                _show_marg.head(_show_m).style.apply(_color_marg, axis=1),
                width="stretch", hide_index=False,
                height=min(60 + _show_m * 35, 460),
            )
            _ww_note = (" · WW+Pts = gain in your 5 weakest weeks" if _weak_weeks else "")
            st.caption(f"+Pts = additional season pts added to your lineup{_ww_note}.")

            # Weak Week Targets (only when variation exists)
            if _weak_weeks and _marginal:
                _ww_df = (pd.DataFrame(_marginal)
                          .dropna(subset=["WW+Pts"])
                          .sort_values("WW+Pts", ascending=False)
                          .reset_index(drop=True))
                if not _ww_df.empty:
                    _ww_df.index = range(1, len(_ww_df) + 1)
                    st.markdown(f"**🔴 Weak Week Targets — Top Picks for Your {len(_weak_weeks)} Lowest Weeks**")
                    def _color_ww(row):
                        pos_bg = {"QB": "rgba(206,147,216,0.18)", "RB": "rgba(102,187,106,0.18)",
                                  "WR": "rgba(66,165,245,0.18)",  "TE": "rgba(255,167,38,0.18)"}
                        return [f"background-color: {pos_bg.get(row['POS'],'')}"] * len(row)
                    st.dataframe(
                        _ww_df[["Player","POS","Team","ADP","WW+Pts","+Pts","Proj/G","Bye"]]
                              .head(10).style.apply(_color_ww, axis=1),
                        width="stretch", hide_index=False,
                        height=min(60 + 10 * 35, 420),
                    )

        st.markdown("---")
        st.subheader("Available Players")

        af1, af2, af3 = st.columns([2, 1, 1])
        with af1:
            pos_filter = st.multiselect("Position", ["QB", "RB", "WR", "TE"],
                                        default=["QB", "RB", "WR", "TE"], key="draft_pos_filter")
        with af2:
            show_top = st.selectbox("Show", [25, 50, 100, "All"], index=1, key="draft_show")
        with af3:
            value_only = st.checkbox("Value only", value=False, help="Players past their ADP")

        avail_show = avail_all[avail_all["POS"].isin(pos_filter)].copy()
        if value_only:
            avail_show = avail_show[avail_show["Fell"].notna()]
        if show_top != "All":
            avail_show = avail_show.head(int(show_top))

        avail_out = avail_show[["Name", "POS", "Team", "FP_Rank", "FP_ADP",
                                 "Fell", "Cmpl", "Fits", "Faces My", "Bye Wk", "Playoff Scr", "Boom%"]].copy()
        avail_out.columns = ["Player", "POS", "Team", "Rank", "ADP",
                              "Fell", "Cmpl", "Fits", "Faces My", "Bye", "Playoff", "Boom%"]
        avail_out["ADP"] = avail_out["ADP"].round(1)

        POS_BG = {"QB": "rgba(206,147,216,0.18)", "RB": "rgba(102,187,106,0.18)",
                  "WR": "rgba(66,165,245,0.18)",  "TE": "rgba(255,167,38,0.18)"}

        def color_avail(row):
            bg = POS_BG.get(row["POS"], "")
            if row.get("Faces My", ""):                          # faces my player in playoffs
                bg = "rgba(186,104,200,0.30)"
            try:                                                  # strong schedule complement
                parts = str(row["Cmpl"]).split("/")
                if len(parts) == 2 and int(parts[1]) > 0 and int(parts[0]) / int(parts[1]) >= 0.6:
                    bg = "rgba(0,230,118,0.25)"
            except (ValueError, ZeroDivisionError):
                pass
            try:                                                  # clear value — highest priority
                if pd.notna(row["Fell"]) and int(row["Fell"]) >= 8:
                    bg = "rgba(255,235,59,0.28)"
            except (TypeError, ValueError):
                pass
            return [f"background-color: {bg}"] * len(row) if bg else [""] * len(row)

        st.dataframe(
            avail_out.style.apply(color_avail, axis=1),
            width="stretch", hide_index=True,
            height=min(60 + len(avail_out) * 35, 520),
        )
        st.caption(
            "🟣 QB · 🟢 RB · 🔵 WR · 🟠 TE &nbsp;|&nbsp; "
            "🟪 Faces your team wks 15-17 &nbsp; "
            "🟩 Schedule complement (≥60% tough-wk coverage) &nbsp; "
            "🟡 Value (fell ≥8 past ADP) &nbsp; "
            "**Playoff** = avg def rank wks 14-17 (32=easiest)",
            unsafe_allow_html=True,
        )

    # ── Top Complements — grouped by individual roster player ─────────────────
    roster_with_tough = [rp for rp in my_pick_data if rp["tough_wks"]]
    if roster_with_tough:
        st.markdown("---")
        st.subheader("🔄 Schedule Complements by Player")
        st.caption("Available players whose schedules best cover each of your players' tough weeks individually.")
        n_cols    = min(len(roster_with_tough), 4)
        cmpl_cols = st.columns(n_cols)
        for col, rp in zip(cmpl_cols, roster_with_tough[:4]):
            with col:
                tough_wk_str = ", ".join(f"Wk {w}" for w in sorted(rp["tough_wks"]))
                st.markdown(f"**{rp['name']}**")
                st.caption(f"Tough: {tough_wk_str}")
                best = (avail_all[
                            (avail_all["POS"] == rp["pos"]) &
                            (avail_all["_cmpl_vs"] == rp["name"]) &
                            (avail_all["_cmpl"] > 0)
                        ]
                        .sort_values(["_cmpl", "FP_ADP"], ascending=[False, True])
                        .head(7))
                if best.empty:
                    st.caption("No complements found.")
                else:
                    for _, r in best.iterrows():
                        adp_s = f"ADP {r['FP_ADP']:.1f}" if pd.notna(r["FP_ADP"]) else ""
                        pct   = r["_cmpl"] / r["_cmpl_denom"] if r["_cmpl_denom"] > 0 else 0
                        icon  = "🟢" if pct >= 0.6 else "🟡"
                        st.markdown(f"{icon} **{r['Name']}** ({r['Team']}) {adp_s} · {r['Cmpl']}")

    # ── Projected Points — league comparison ──────────────────────────────────
    st.markdown("---")
    st.subheader("📈 Projected Points — League Comparison")

    _filled_board = st.session_state.draft_board[
        st.session_state.draft_board["Player"].notna() &
        (st.session_state.draft_board["Player"] != "")
    ].copy().head(n_picks)

    if _filled_board.empty or not st.session_state.my_picks:
        st.info("Add picks — projected points update as the draft progresses.")
    else:
        # Map every pick to its draft slot (snake order)
        def _slot(pick_num):
            rnd = (pick_num - 1) // num_teams + 1
            pos = (pick_num - 1) % num_teams + 1
            return pos if rnd % 2 == 1 else num_teams + 1 - pos

        _filled_board["slot"] = _filled_board["Pick"].apply(_slot)

        # Project each team's season total
        _league_rows = []
        for _sl in range(1, num_teams + 1):
            _sdf  = _filled_board[_filled_board["slot"] == _sl]
            if _sdf.empty:
                continue
            _names = _sdf["Player"].tolist()
            _total, _weekly = _project_bb_score(_names, _proj_lu)
            _league_rows.append({
                "slot":     _sl,
                "is_mine":  (_sl == my_slot),
                "n":        len(_names),
                "proj_pts": _total,
                "weekly":   _weekly,
                "names":    _names,
            })

        if not _league_rows:
            st.info("Add picks to generate projections.")
        else:
            _ldf = pd.DataFrame(_league_rows).sort_values("proj_pts", ascending=False).reset_index(drop=True)
            _ldf["league_rank"] = range(1, len(_ldf) + 1)

            _my_row  = _ldf[_ldf["is_mine"]].iloc[0] if _ldf["is_mine"].any() else None
            _n_teams = len(_ldf)

            # ── Top-line metrics for my team
            if _my_row is not None:
                _my_proj = _my_row["proj_pts"]
                _my_rank = int(_my_row["league_rank"])
                _max_pts = _ldf["proj_pts"].max()
                _avg_pts = _ldf["proj_pts"].mean()

                def _grade(r, n):
                    pct = 1 - (r - 1) / max(n - 1, 1)
                    if pct >= 0.85: return "A", "🟢"
                    if pct >= 0.65: return "B", "🟡"
                    if pct >= 0.40: return "C", "🟠"
                    if pct >= 0.20: return "D", "🔴"
                    return "F", "⛔"

                _gl, _gi = _grade(_my_rank, _n_teams)
                pp1, pp2, pp3, pp4 = st.columns(4)
                pp1.metric("Your Projected Total", f"{_my_proj:.1f} pts",
                           f"#{_my_rank} of {_n_teams} teams")
                pp2.metric("League Leader",        f"{_max_pts:.1f} pts",
                           f"gap: {_my_proj - _max_pts:+.1f}")
                pp3.metric("League Average",       f"{_avg_pts:.1f} pts",
                           f"vs avg: {_my_proj - _avg_pts:+.1f}")
                pp4.metric("Grade",                f"{_gi} {_gl}",
                           f"based on {_my_row['n']} picks so far")

            # ── League standings table
            st.markdown("**League Standings — Projected Season Total**")
            _display = _ldf[["league_rank", "slot", "n", "proj_pts"]].copy()
            _display.columns = ["#", "Slot", "Picks", "Proj Pts"]
            _display["Slot"] = _display["Slot"].apply(
                lambda s: f"⭐ Slot {s} (You)" if s == my_slot else f"Slot {s}")
            _display["Proj Pts"] = _display["Proj Pts"].round(1)

            def _highlight_mine(row):
                return (["background-color: rgba(66,165,245,0.25)"] * len(row)
                        if "(You)" in str(row["Slot"]) else [""] * len(row))

            st.dataframe(
                _display.style.apply(_highlight_mine, axis=1),
                hide_index=True, width="stretch",
                height=min(60 + len(_display) * 35, 480),
            )
            st.caption(
                "Projected using DK Best Ball scoring: QB + 2 RB + 3 WR + TE + FLEX per week, "
                "auto-set from each team's 20-man roster. Bye weeks count as 0. "
                "Projections from uploaded season totals ÷ games played."
            )

            # ── My team's weekly score breakdown (uses pre-computed _base_weekly)
            if _my_row is not None and _base_weekly:
                _my_bye_wks  = {_proj_lu.get(n, {}).get("bye") for n in _my_row["names"]}
                st.markdown("**Your Projected Weekly Scores**")
                _wk_df = pd.DataFrame({
                    "Week":  list(range(1, len(_base_weekly) + 1)),
                    "Score": [round(s, 1) for s in _base_weekly],
                })
                if _has_variation:
                    _wk_df["Status"] = _wk_df.apply(
                        lambda r: "🔴 Weak" if r["Week"] in _weak_weeks
                                  else ("💤 Bye" if r["Week"] in _my_bye_wks else ""), axis=1)

                    def _wk_color(val):
                        if val <= _weak_thresh: return "color: #ff6b6b; font-weight: bold"
                        if val >= _avg_weekly * 1.1: return "color: #69db7c"
                        return ""
                    styled = _wk_df.style.map(_wk_color, subset=["Score"])
                else:
                    _wk_df["Status"] = _wk_df["Week"].apply(
                        lambda w: "💤 Bye" if w in _my_bye_wks else "")
                    styled = _wk_df.style

                st.dataframe(styled, hide_index=True, width="stretch", height=240)
                if _has_variation and _avg_weekly > 0:
                    st.caption(
                        f"Avg {_avg_weekly:.1f} pts/wk · "
                        f"🔴 = your 5 lowest weeks — see Weak Week Targets above"
                    )
                else:
                    st.caption(
                        f"Avg {_avg_weekly:.1f} pts/wk · "
                        "Draft more picks for week-by-week variation to appear"
                    )

            # ── CSV Export ────────────────────────────────────────────────────
            st.markdown("---")
            st.subheader("⬇️ Export")

            # Build full draft CSV
            _export_rows = []
            for _, _pick_row in _filled_board.iterrows():
                _pick_num = int(_pick_row["Pick"])
                _rnd      = (_pick_num - 1) // num_teams + 1
                _slot_num = _slot(_pick_num)
                _pn3      = _pick_row["Player"]
                _pdata3   = _proj_lu.get(_pn3, {})
                _fp_row   = fp_all[fp_all["Name"] == _pn3].head(1)
                _adp      = round(float(_fp_row["FP_ADP"].iloc[0]), 1) if not _fp_row.empty and pd.notna(_fp_row["FP_ADP"].iloc[0]) else ""
                _export_rows.append({
                    "Pick":       _pick_num,
                    "Round":      _rnd,
                    "Slot":       _slot_num,
                    "My Pick":    "Yes" if _slot_num == my_slot else "No",
                    "Player":     _pn3,
                    "POS":        _pdata3.get("pos", ""),
                    "NFL Team":   _pdata3.get("team", ""),
                    "ADP":        _adp,
                    "Proj/G":     round(_pdata3.get("proj_pg", 0), 1) if _pdata3 else "",
                    "Proj PPW":   round(_pdata3.get("proj_ppw", 0), 2) if _pdata3 else "",
                    "Team Proj Total": "",  # filled in next pass
                })

            # Fill Team Proj Total per slot
            _slot_totals = {
                int(r["slot"]): round(r["proj_pts"], 1)
                for _, r in _ldf.iterrows()
            }
            for _er in _export_rows:
                _er["Team Proj Total"] = _slot_totals.get(_er["Slot"], "")

            _export_df = pd.DataFrame(_export_rows)

            ec1, ec2 = st.columns(2)
            with ec1:
                st.download_button(
                    "⬇️ Download Full Draft CSV",
                    data=_export_df.to_csv(index=False),
                    file_name="draft_analysis.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with ec2:
                # League standings CSV
                _standings_csv = _ldf[["league_rank","slot","n","proj_pts"]].copy()
                _standings_csv.columns = ["Rank","Slot","Picks","Proj Pts"]
                _standings_csv["My Team"] = _standings_csv["Slot"].apply(
                    lambda s: "Yes" if s == my_slot else "No")
                st.download_button(
                    "⬇️ Download Standings CSV",
                    data=_standings_csv.to_csv(index=False),
                    file_name="league_standings.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
