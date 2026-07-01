import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
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
DATA = Path(__file__).parent / "data"

@st.cache_data
def load_projections():
    df = pd.read_csv(DATA / "projections_2026.csv")
    return df

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


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏈 Best Ball Dashboard")
    st.markdown("---")
    tab_choice = st.radio(
        "Screen",
        ["📊 Player Projections", "🛡️ Defense Matchups", "📈 Schedule Rankings", "📅 Schedule Viewer"],
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
                            use_container_width=True,
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
    st.plotly_chart(fig, use_container_width=True)

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
    st.plotly_chart(fig2, use_container_width=True)

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
        st.plotly_chart(fig3, use_container_width=True)


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
        st.plotly_chart(fig, use_container_width=True)

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
        use_container_width=True,
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
    st.plotly_chart(fig, use_container_width=True)

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
        use_container_width=True,
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
                use_container_width=True,
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
                    use_container_width=True,
                    hide_index=True,
                    height=min(40 + len(rows_out) * 35, 420),
                )
