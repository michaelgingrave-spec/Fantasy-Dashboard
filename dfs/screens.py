"""DFS screens for the Fantasy Model dashboard — Optimizer, Matchup Finder,
Suggestions, Data Check. Rendered by dashboard.py via ``render(screen)``.

The whole DFS feature set was built as a standalone app and folded in here so
everything lives in one repo. DraftKings salaries are pulled live from DK's
public JSON — that works when you run the dashboard locally, but Streamlit
Cloud's servers get IP-blocked, so on the deployed site drop a saved
``data/dfs/dk/draftables_<id>.json`` instead.
"""
from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from dfs.config import DATASUITE_DIR, ODDS_API_KEY, OptimizerConfig, projection_path
from dfs.dk_client import (
    DKError,
    classic_slates,
    get_salaries,
    pick_main_slate,
    slate_types_present,
)
from dfs.datasuite.features import inventory
from dfs.edge import player_boosts, weekly_edges
from dfs.optimizer import OptimizerError, optimize, to_dk_upload
from dfs.projections import ProjectionError, load_weekly_projections
from dfs.slate import build_slate
from dfs import suggestions as sug

SCREENS = ["Optimizer", "Matchup Finder", "Suggestions", "Data Check"]


# ── cached loaders ──────────────────────────────────────────────────────────
def _hash_path(p: Path) -> str:
    try:
        return hashlib.md5(p.read_bytes()).hexdigest()
    except Exception:
        return "missing"


@st.cache_data(show_spinner=False)
def _slates(_nonce: int):
    return classic_slates()


@st.cache_data(show_spinner=True)
def _salaries(draft_group_id: int, _nonce: int):
    return get_salaries(draft_group_id)


@st.cache_data(show_spinner=False)
def _projections(week: int, file_hash: str):
    return load_weekly_projections(week)


@st.cache_data(show_spinner=True)
def _edges(slate_key: str, _slate: pd.DataFrame, _ds_hash: str):
    return weekly_edges(_slate)


def _ds_hash() -> str:
    h = hashlib.md5()
    for f in sorted(DATASUITE_DIR.glob("*")):
        h.update(f.name.encode())
        try:
            h.update(str(f.stat().st_mtime).encode())
        except OSError:
            pass
    return h.hexdigest()


def _player_labels(df: pd.DataFrame) -> dict[str, str]:
    return {
        f'{r["name"]} · {r["pos"]} · {r["team"]} · ${r["salary"]:,} · {r["proj"]:.1f}': r["dk_id"]
        for _, r in df.sort_values(["pos", "salary"], ascending=[True, False]).iterrows()
    }


# ── entry point ────────────────────────────────────────────────────────────
def render(screen: str) -> None:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🏈 DFS controls")
    week = int(st.sidebar.number_input("NFL week", 1, 18, value=1, step=1, key="dfs_week"))
    if st.sidebar.button("↻ Refresh DK / data", key="dfs_refresh"):
        st.cache_data.clear()
        st.session_state.pop("dfs_lineups", None)
        st.rerun()

    nonce = st.session_state.setdefault("dfs_nonce", 0)

    slate_df = unmatched = None
    slate_err = None
    pick = ""
    try:
        all_slates = _slates(nonce)
        if not all_slates:
            slate_err = ("DraftKings returned no Classic slates. Try Refresh, check your "
                         "connection, or (on the deployed site) drop a saved draftables JSON "
                         "in data/dfs/dk/.")
        else:
            types = ["All"] + slate_types_present(all_slates)
            default_slate = pick_main_slate(all_slates)
            default_type = default_slate.slate_type if default_slate else "All"
            type_idx = types.index(default_type) if default_type in types else 0
            stype = st.sidebar.selectbox(
                "Slate type", types, index=type_idx, key="dfs_stype",
                help="DraftKings runs several classic slates per week — the whole week "
                     "(Thu–Mon), Sunday + Monday, Sunday 1pm only, and so on.",
            )
            slates = all_slates if stype == "All" else [s for s in all_slates if s.slate_type == stype]
            labels = [s.label for s in slates]
            d_idx = next((i for i, s in enumerate(slates)
                          if default_slate and s.draft_group_id == default_slate.draft_group_id), 0)
            pick = st.sidebar.selectbox("Slate", labels, index=d_idx, key="dfs_slate")
            chosen = slates[labels.index(pick)]
            st.sidebar.caption(
                f"{chosen.slate_type} · {chosen.game_count} games · dg {chosen.draft_group_id}"
            )
            salaries = _salaries(chosen.draft_group_id, nonce)
            proj_p = projection_path(week)
            projections = _projections(week, _hash_path(proj_p))
            slate_df, unmatched = build_slate(salaries, projections)
    except (DKError, ProjectionError) as e:
        slate_err = str(e)
    except Exception as e:  # noqa: BLE001
        slate_err = f"{type(e).__name__}: {e}"

    st.sidebar.caption("Odds API: key set ✔" if ODDS_API_KEY else "Odds API: no key (Vegas term off)")

    def _need_slate() -> bool:
        if slate_df is None or slate_df.empty:
            st.error(slate_err or "No slate loaded.")
            if slate_err and "projection" in slate_err.lower():
                st.info(f"Drop your weekly export at `{projection_path(week)}`")
            return True
        return False

    # ── Optimizer ──────────────────────────────────────────────────────────
    if screen == "Optimizer":
        st.header("🏈 DFS Lineup Optimizer")
        if _need_slate():
            st.stop()

        st.caption(f"Pool: {len(slate_df)} projected players · {pick.split('·')[0].strip()} · "
                   f"players from other games are not on this slate")

        labels_map = _player_labels(slate_df)
        c1, c2 = st.columns(2)
        lock_lbls = c1.multiselect("🔒 Lock (in every lineup)", list(labels_map), key="dfs_lock")
        excl_lbls = c2.multiselect("🚫 Exclude", list(labels_map), key="dfs_excl")
        locks = [labels_map[x] for x in lock_lbls]
        excludes = [labels_map[x] for x in excl_lbls]

        used = int(slate_df[slate_df["dk_id"].isin(locks)]["salary"].sum())
        st.caption(
            f"Locked {len(locks)} · \\${used:,} used · \\${50000 - used:,} left for "
            f"{9 - len(locks)} open slots"
        )

        o1, o2, o3 = st.columns(3)
        n_lineups = int(o1.number_input("Lineups", 1, 150, 5, key="dfs_n",
                                        help="How many lineups to generate"))
        variance = o2.slider(
            "Variance", 0, 100, 25, key="dfs_var",
            help="0 = the best lineups in projection order. Higher = more random player "
                 "mixes — and re-clicking Build gives a fresh set of combinations.",
        ) / 100.0
        lean = o3.slider("Ceiling lean (GPP)", 0.0, 1.0, 0.0, 0.05, key="dfs_lean",
                         help="Bias toward high-upside players for tournaments")

        with st.expander("Advanced constraints"):
            a1, a2, a3 = st.columns(3)
            min_salary = a1.slider("Min salary", 45000, 50000, 49000, 250, key="dfs_minsal")
            max_exp = a2.slider("Max exposure", 0.1, 1.0, 0.6, 0.05, key="dfs_maxexp",
                                help="Cap on the share of lineups any one player can appear in")
            min_uniques = a3.slider("Min differences between lineups", 1, 5, 2, key="dfs_uniq")

        s1, s2, s3, s4 = st.columns(4)
        stack_qb = s1.checkbox("QB stack", value=True, key="dfs_stack")
        stack_min = s2.selectbox("Stack size", [1, 2], index=0, key="dfs_stackmin")
        bring_back = s3.checkbox("Bring-back", value=False, key="dfs_bring")
        boost = s4.checkbox("Apply matchup boosts", value=False, key="dfs_boost")

        build = st.button("⚙️ Build lineups", type="primary", key="dfs_build")
        if variance > 0:
            st.caption("Variance is on — click Build again any time for a different set of combos.")
        if build:
            cfg = OptimizerConfig(
                n_lineups=n_lineups, lean=lean, variance=variance, min_salary=min_salary,
                max_exposure=max_exp, min_uniques=min_uniques, stack_qb=stack_qb,
                stack_min=stack_min, bring_back=bring_back, apply_matchup_boost=boost,
                locks=locks, excludes=excludes,
            )
            boosts = {}
            if boost:
                try:
                    boosts = player_boosts(_edges(str(week) + pick, slate_df, _ds_hash()))
                except Exception as e:  # noqa: BLE001
                    st.warning(f"Matchup boosts unavailable: {e}")
            try:
                st.session_state["dfs_lineups"] = optimize(slate_df, cfg, boosts)
            except OptimizerError as e:
                st.session_state.pop("dfs_lineups", None)
                st.error(str(e))

        lineups = st.session_state.get("dfs_lineups")
        if lineups:
            projs = [lu.proj for lu in lineups]
            exposure: dict[str, int] = {}
            for lu in lineups:
                for _, r in lu.players.iterrows():
                    exposure[r["name"]] = exposure.get(r["name"], 0) + 1
            st.success(
                f"{len(lineups)} lineups · {len(exposure)} unique players · "
                f"proj {min(projs):.1f}–{max(projs):.1f}"
            )
            for i, lu in enumerate(lineups, 1):
                with st.expander(f"#{i} · {lu.proj:.1f} pts · \\${lu.salary:,} · {lu.stack}",
                                 expanded=(i == 1)):
                    show = lu.players.copy()
                    show["salary"] = show["salary"].map("${:,.0f}".format)
                    show["proj"] = show["proj"].round(1)
                    st.dataframe(show, hide_index=True, width="stretch")

            if len(lineups) > 1:
                exp_df = (
                    pd.DataFrame({"player": exposure.keys(), "lineups": exposure.values()})
                    .assign(pct=lambda d: (d["lineups"] / len(lineups) * 100).round(0))
                    .sort_values("lineups", ascending=False)
                    .head(30)
                )
                st.plotly_chart(
                    px.bar(exp_df, x="pct", y="player", orientation="h", height=650,
                           labels={"pct": "% of lineups"}),
                    use_container_width=True,
                )

            csv = to_dk_upload(lineups).to_csv(index=False).encode()
            st.download_button("⬇️ Download DK upload CSV", csv,
                               file_name=f"dk_lineups_week{week}.csv", mime="text/csv",
                               key="dfs_dl")

    # ── Matchup Finder ─────────────────────────────────────────────────────
    elif screen == "Matchup Finder":
        st.header("🎯 DFS Matchup Edge Finder")
        st.caption("Usage + opposing-defense softness + Vegas game environment. Research signal, "
                   "not a prop-EV model. Add Data Suite files (Data Check) to power the usage term.")
        if _need_slate():
            st.stop()
        try:
            edges = _edges(str(week) + pick, slate_df, _ds_hash())
        except Exception as e:  # noqa: BLE001
            st.error(f"Edge model failed: {e}")
            st.stop()
        if edges.empty:
            st.info("No skill players on the slate.")
            st.stop()

        has_signal = float(edges["edge"].abs().max() or 0) > 1e-6
        if not has_signal:
            st.info("The edge model has no inputs yet — no Data Suite files in "
                    "`data/dfs/datasuite/` and no `ODDS_API_KEY`. Table below is just projection "
                    "order. Add either to activate usage / matchup / Vegas terms.")

        pos_filter = st.multiselect("Positions", ["QB", "RB", "WR", "TE"],
                                    default=["QB", "RB", "WR", "TE"], key="dfs_posf")
        view = edges[edges["pos"].isin(pos_filter)].copy()
        if not has_signal:
            view = view.sort_values("proj", ascending=False)
        show_cols = [c for c in ["name", "pos", "team", "opp", "salary", "proj", "value",
                                 "implied_total", "opportunity_z", "defense_z", "vegas_z",
                                 "edge", "why"] if c in view.columns]
        col_cfg = {}
        if has_signal:
            col_cfg["edge"] = st.column_config.ProgressColumn(
                "edge", min_value=float(view["edge"].min()), max_value=float(view["edge"].max()))
        st.dataframe(view[show_cols].round(2).head(80), hide_index=True, width="stretch",
                     column_config=col_cfg)

        if has_signal:
            heat = view.pivot_table(index="team", columns="pos", values="edge", aggfunc="max")
            if not heat.empty:
                st.plotly_chart(
                    px.imshow(heat, color_continuous_scale="RdBu", aspect="auto", height=700,
                              labels={"color": "best edge"}),
                    use_container_width=True,
                )

    # ── Suggestions ────────────────────────────────────────────────────────
    elif screen == "Suggestions":
        st.header("💡 DFS Suggestions")
        if _need_slate():
            st.stop()
        try:
            edges = _edges(str(week) + pick, slate_df, _ds_hash())
        except Exception:
            edges = pd.DataFrame()

        st.subheader("💰 Value plays (pts per $1k)")
        st.dataframe(sug.value_plays(slate_df), hide_index=True, width="stretch")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("⚡ Leverage (cheap, strong spot)")
            lev = sug.leverage_plays(slate_df, edges)
            st.dataframe(lev if not lev.empty else pd.DataFrame({"note": ["needs Data Suite / odds"]}),
                         hide_index=True, width="stretch")
        with col2:
            st.subheader("🪫 Punts")
            st.dataframe(sug.punt_plays(slate_df), hide_index=True, width="stretch")

        st.subheader("🔗 Stacks")
        stk = sug.stack_suggestions(slate_df, edges)
        st.dataframe(stk if not stk.empty else pd.DataFrame({"note": ["no QB/pass-catcher pairs"]}),
                     hide_index=True, width="stretch")

    # ── Data Check ─────────────────────────────────────────────────────────
    else:
        st.header("🔍 DFS Data Check")

        st.subheader("DraftKings slate")
        if slate_err:
            st.error(slate_err)
        else:
            st.write(f"**{pick}**")
            un = unmatched.copy()
            if "status" not in un.columns:
                un["status"] = "None"
            notable = un[(un["salary"] >= 4500) & (un["status"].isin(["None", "Q"]))]
            backups = len(un) - len(notable)

            m1, m2, m3 = st.columns(3)
            m1.metric("Projected players", len(slate_df))
            m2.metric("Notable gaps", len(notable), help="Salary ≥ $4,500, healthy, but no projection")
            m3.metric("Min-salary backups w/o proj", backups, help="Safe to ignore")

            pos_line = []
            for p in ("QB", "RB", "WR", "TE", "DST"):
                got = int((slate_df["pos"] == p).sum())
                miss = int((notable["pos"] == p).sum())
                pos_line.append(f"{p}: {got}" + (f" (+{miss} gap)" if miss else ""))
            st.caption(" · ".join(pos_line))

            if len(notable):
                st.write("**Notable unmatched — add to the projection CSV if you want them optimizable:**")
                st.dataframe(notable.sort_values("salary", ascending=False),
                             hide_index=True, width="stretch")
            else:
                st.success("Every healthy, priced-up player has a projection.")

        st.subheader("Weekly projections")
        pp = projection_path(week)
        if pp.exists():
            st.success(f"{pp.name} · {datetime.fromtimestamp(pp.stat().st_mtime):%Y-%m-%d %H:%M}")
        else:
            st.error(f"Missing — drop the FantasyPoints export at {pp}")

        st.subheader("FantasyPoints Data Suite  ·  data/dfs/datasuite/")
        inv = inventory()
        if inv.empty:
            st.info("Empty. Save Data Suite tables here as CSV/JSON (e.g. "
                    "receiving_advanced_player.csv, defense_advanced_receiving_team.csv). "
                    "The Matchup Finder runs on Vegas + projections until then.")
        else:
            st.dataframe(inv, hide_index=True, width="stretch")

        st.subheader("Odds API")
        st.write("key configured ✔" if ODDS_API_KEY
                 else "no key — set ODDS_API_KEY in .env to add implied team totals")
