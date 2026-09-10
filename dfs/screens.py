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

    # The slate-type / slate pickers only matter for the Optimizer (it builds one
    # specific lineup). Every other DFS screen uses the widest slate so it sees the whole
    # player pool regardless of what the Optimizer is set to.
    _slate_pickers_here = (screen == "Optimizer")

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
            default_slate = pick_main_slate(all_slates)
            widest = max(all_slates, key=lambda s: s.game_count)
            if _slate_pickers_here:
                types = ["All"] + slate_types_present(all_slates)
                default_type = default_slate.slate_type if default_slate else "All"
                type_idx = types.index(default_type) if default_type in types else 0
                stype = st.sidebar.selectbox(
                    "Slate type", types, index=type_idx, key="dfs_stype",
                    help="DraftKings runs several classic slates per week — the whole week "
                         "(Thu–Mon), Sunday + Monday, Sunday 1pm only, and so on. "
                         "Only affects the Optimizer.",
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
            else:
                # every DFS screen except the Optimizer sees the full week: union of
                # every classic slate's player pool, deduped by DK id.
                frames = []
                for s in all_slates:
                    try:
                        frames.append(_salaries(s.draft_group_id, nonce))
                    except Exception:
                        continue
                salaries = (pd.concat(frames, ignore_index=True)
                            .sort_values("salary", ascending=False)
                            .drop_duplicates(subset="dk_id")
                            if frames else _salaries(widest.draft_group_id, nonce))
                pick = "Full week (all slates)"
                st.sidebar.caption(f"Player pool: all {len(all_slates)} classic slates merged "
                                   "(Slate-type picker is on the Optimizer tab)")
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
        boost = s4.checkbox("Apply regression lean", value=False, key="dfs_boost",
                            help="Small tilt toward players whose recent expected FP > actual "
                                 "(backtested; needs current-season game logs in data/dfs/matchup/).")

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
                    from dfs import matchup_view as _mv
                    from dfs.names import normalize_name as _nn
                    import numpy as _np
                    n_lean = 0
                    for _, pr in slate_df.iterrows():
                        lv = _mv.regression_lean(_nn(pr["name"]), pr["pos"])["lean"]
                        if lv:
                            n_lean += 1
                        m = 1.0 + float(_np.clip(lv / max(pr["proj"], 6.0), -0.12, 0.12))
                        boosts[pr["dk_id"]] = m
                    if n_lean == 0:
                        st.info("Regression lean is 0 for everyone — no current-season weekly "
                                "FantasyPoints exports in data/dfs/matchup/ yet.")
                except Exception as e:  # noqa: BLE001
                    st.warning(f"Regression lean unavailable: {e}")
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

    # ── Matchup Machine ───────────────────────────────────────────────────
    elif screen == "Matchup Machine":
        st.header("🧬 DFS Matchup Machine")
        st.caption("Pick any team → its pass-catchers by coverage and backs by run concept, "
                   "next to a defense you choose. Not limited by the DFS slate filter. "
                   "Heat: green = more productive. Descriptive — see matchup_model/backtest_report.md.")
        from dfs import matchup_view as mv
        from dfs.names import normalize_name as _nn
        from dfs.names import norm_team as _nt
        from dfs.names import DK_TEAM_NICKNAME as _NICK

        if not mv.scheme_available():
            st.info("Scheme-split data not loaded (data/dfs/matchup/*coverage*/*concept*). "
                    "Run the Wednesday pull, or `matchup_model/weekly_pull.py`.")
            st.stop()

        all_teams = mv.scheme_teams()
        # label like "LAR — Rams" so the search box matches "rams" / "49ers" too
        labels = [f"{t} — {_NICK.get(t, t).capitalize()}" for t in all_teams]
        _lab2team = dict(zip(labels, all_teams))
        _has_slate = slate_df is not None and not slate_df.empty
        slate_teams = ({_nt(t) for t in slate_df["team"].dropna().astype(str)}
                       if _has_slate else set())
        default_i = next((i for i, t in enumerate(all_teams) if t in slate_teams), 0)
        c_team, c_opp = st.columns(2)
        team = _lab2team[c_team.selectbox("Team", labels, index=default_i, key="mm_team")]

        if _has_slate:
            tdf = slate_df[slate_df["team"].map(lambda x: _nt(str(x))) == team].copy()
        else:
            tdf = pd.DataFrame(columns=["name", "pos", "proj", "opp"])
        slate_opp = str(tdf["opp"].iloc[0]).lstrip("@") if not tdf.empty else ""
        opp_default = next((i for i, t in enumerate(all_teams) if t == _nt(slate_opp)), 0)
        opp = _lab2team[c_opp.selectbox("Opponent defense", labels, index=opp_default, key="mm_opp")]
        if not tdf.empty and _nt(slate_opp) != opp:
            st.caption(f"(slate opponent is {_nt(slate_opp)} — showing {opp} by choice)")
        elif tdf.empty:
            st.caption(f"{team} isn't on the current slate — players ranked by 2025 volume, "
                       "no projections.")

        bw = mv.scheme_blend_weight()
        src = "2025" if bw == 0 else f"2026×{bw:.0%} + 2025×{1 - bw:.0%}"
        st.caption(f"Scheme data: **{src}**.")

        # player list: slate projections when available, else 2025 scheme volume
        tdf["pos"] = tdf["pos"].astype(str).str.upper() if not tdf.empty else tdf.get("pos")
        if not tdf.empty:
            def _top(p, c):
                s = tdf[(tdf["pos"] == p) & (tdf["proj"].fillna(0) > 2)].sort_values(
                    "proj", ascending=False).head(c)
                return [(r["name"], float(r["proj"])) for _, r in s.iterrows()]
            pass_players = _top("WR", 4) + _top("TE", 2)
            rb_players = _top("RB", 3)
        else:
            pass_players = [(n, None) for n in mv.team_pass_catchers(team, 6)]
            rb_players = [(n, None) for n in mv.team_backs(team, 3)]

        # ── highlight table: best scheme edges vs this opponent ─────────────
        st.subheader(f"⭐ Highlights — {team} vs {opp}")
        hl = mv.matchup_highlights(opp, [n for n, _ in pass_players],
                                   [n for n, _ in rb_players])
        if hl.empty:
            st.caption("No standout scheme edges — the opponent isn't a league outlier in "
                       "any coverage/concept these players are efficient against.")
        else:
            hl = hl.assign(**{"": hl["kind"].map({"pass": "🎯", "run": "🏃"})}).drop(columns=["kind"])
            hl = hl[["", "player", "look", "why", "mark"]].rename(
                columns={"look": "scheme", "why": "why it's an edge", "mark": "player's 2025 mark"})
            st.dataframe(hl, hide_index=True, width="stretch")
            st.caption("The defense **leans on** (league rank in usage) or is **weak against** "
                       "that coverage / run concept / personnel grouping, and the player is "
                       "above their own baseline in it.")

        def _pass_block(name, proj):
            tag = f"  ·  FP proj {proj:.1f}" if proj is not None else ""
            st.markdown(f"**{name}**{tag}")
            t = mv.player_pass_by_coverage(_nn(name))
            if t.empty:
                st.caption("No 2025 coverage-split routes for this player.")
            else:
                st.dataframe(mv.heat(t, ["tgt/rt", "yds/rt", "yds/tgt", "catch%",
                                         "1st-read%", "TD"]),
                             hide_index=True, width="stretch")
            pt = mv.player_pass_by_personnel(_nn(name))
            if not pt.empty:
                st.caption("by personnel")
                st.dataframe(mv.heat(pt, ["tgt/rt", "yds/rt", "catch%", "TD"]),
                             hide_index=True, width="stretch")

        def _run_block(name, proj):
            tag = f"  ·  FP proj {proj:.1f}" if proj is not None else ""
            st.markdown(f"**{name}**{tag}")
            t = mv.player_run_by_concept(_nn(name))
            if t.empty:
                st.caption("No 2025 concept-split carries for this player.")
            else:
                st.dataframe(mv.heat(t, ["YPC", "yards", "TD", "success%", "exp-run%", "att%"]),
                             hide_index=True, width="stretch")
            pt = mv.player_run_by_personnel(_nn(name))
            if not pt.empty:
                st.caption("by personnel")
                st.dataframe(mv.heat(pt, ["YPC", "success%", "exp-run%", "att%"]),
                             hide_index=True, width="stretch")

        # ── passing ────────────────────────────────────────────────────────
        st.subheader(f"{team} pass-catchers — by coverage")
        if not pass_players:
            st.caption("No pass-catchers found for this team.")
        for nm, pj in pass_players:
            _pass_block(nm, pj)

        if opp:
            st.markdown(f"**{opp} defense — allowed by coverage**  ·  `rk` 1–32, "
                        "**32 = softest** (allows the most); `plays% rk` **1 = plays it most**")
            dpc = mv.defense_pass_by_coverage(opp)
            if dpc.empty:
                st.caption(f"No coverage data for {opp}.")
            else:
                eff = ["yds/tgt", "catch%", "yds/rec", "rating", "TD"]
                eff_rk = [f"{m} rk" for m in eff if f"{m} rk" in dpc.columns]
                st.dataframe(mv.heat(dpc, eff + eff_rk, good_high=True),
                             hide_index=True, width="stretch")
            dpp = mv.defense_pass_by_personnel(opp)
            if not dpp.empty:
                st.caption(f"**{opp} defense — allowed by personnel**  ·  `sees% rk` 1 = faces it most")
                st.dataframe(mv.heat(dpp, ["yds/tgt", "yds/tgt rk", "catch%", "catch% rk",
                                           "rating", "rating rk", "TD"], good_high=True),
                             hide_index=True, width="stretch")

        # ── rushing ────────────────────────────────────────────────────────
        st.subheader(f"{team} backs — by run concept")
        tm = mv.team_run_by_concept(team, "offense")
        if not tm.empty:
            st.markdown(f"**{team} offense** — run mix")
            st.dataframe(mv.heat(tm, ["YPC", "success%", "exp-run%", "att%"]),
                         hide_index=True, width="stretch")
        if not rb_players:
            st.caption("No backs found for this team.")
        for nm, pj in rb_players:
            _run_block(nm, pj)

        if opp:
            st.markdown(f"**{opp} defense — allowed by concept**")
            drc = mv.team_run_by_concept(opp, "defense")
            if drc.empty:
                st.caption(f"No concept data for {opp}.")
            else:
                st.dataframe(mv.heat(drc, ["YPC", "success%", "exp-run%"], good_high=True),
                             hide_index=True, width="stretch")
            drp = mv.defense_run_by_personnel(opp)
            if not drp.empty:
                st.caption(f"**{opp} defense — allowed by personnel**  ·  `sees% rk` 1 = faces it most")
                st.dataframe(mv.heat(drp, ["YPC", "YPC rk", "success%", "success% rk",
                                           "exp-run%", "exp-run% rk"], good_high=True),
                             hide_index=True, width="stretch")

    # ── Player Lookup ─────────────────────────────────────────────────────
    elif screen == "Player Lookup":
        st.header("🔎 DFS Player Lookup")
        st.caption("Any projected player: FantasyPoints projection vs the backtested Model "
                   "proj, a trailing-usage stat line, recent form, coverage splits, and "
                   "whether they've faced the opponent's coordinator before. Not limited to "
                   "the DK slate — salary/value shows only when the player is priced.")
        from dfs import matchup_view as mv
        from dfs.names import normalize_name

        # player list = the whole weekly projection file (not the DK-priced subset)
        try:
            plist = _projections(week, _hash_path(projection_path(week)))
        except ProjectionError as e:
            st.error(str(e))
            st.stop()
        # salary lookup (optional) from the slate, keyed by normalised name
        _sal = {}
        if slate_df is not None and not slate_df.empty:
            _sal = {normalize_name(n): s for n, s in zip(slate_df["name"], slate_df["salary"])}

        plist = plist.sort_values(["pos", "proj"], ascending=[True, False])
        labels = {f'{r["name"]} · {r["pos"]} · {r["team"]}': r for _, r in plist.iterrows()}
        choice = st.selectbox("Player", list(labels), key="pl_pick")
        r = labels[choice]
        nk = normalize_name(r["name"])
        opp = str(r.get("opp", "")).lstrip("@")
        pos = str(r["pos"]).upper()
        fp_proj = float(r["proj"])
        salary = _sal.get(nk)

        mp = mv.model_projection(fp_proj, nk, pos) if mv.available() else {"proj": fp_proj, "lean": 0.0, "reason": ""}
        pl = mv.projected_line(nk, pos) if mv.available() else {"fp": None, "reason": ""}
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("FP proj", f'{fp_proj:.1f}', help="FantasyPoints' own weekly projection.")
        m2.metric("Model proj", f'{mp["proj"]:.1f}', delta=f'{mp["lean"]:+.1f} lean',
                  delta_color="off", help="FP proj + a backtested regression-to-expected lean.")
        m3.metric("Proj line", f'{pl["fp"]:.1f}' if pl.get("fp") is not None else "—",
                  help="DK points implied by the player's own trailing usage × efficiency.")
        m4.metric("Salary", f'${int(salary):,}' if salary else "—")
        val = mp["proj"] / (salary / 1000) if salary else 0.0
        st.caption(f"**{r['team']}** vs **{opp or '—'}**"
                   + (f"  ·  value (Model proj / $1k): **{val:.2f}**" if salary else "")
                   + (f"  ·  Lean: {mp['reason']}" if mp.get("reason") else ""))

        if not mv.available():
            st.info("Historical Data Suite tables not loaded (data/dfs/matchup/).")
            st.stop()

        if pl.get("fp") is not None:
            ln = pl["line"]
            order = ["pass_att", "pass_yds", "pass_td", "int", "rush_att", "rush_yds",
                     "rush_td", "tgt", "rec", "rec_yds", "rec_td"]
            nice = {"pass_att": "pass att", "pass_yds": "pass yds", "pass_td": "pass TD",
                    "int": "INT", "rush_att": "carries", "rush_yds": "rush yds",
                    "rush_td": "rush TD", "tgt": "targets", "rec": "rec", "rec_yds": "rec yds",
                    "rec_td": "rec TD"}
            row_line = {nice[k]: ln[k] for k in order if k in ln}
            st.write("**Projected line** — " + " · ".join(f"{k} {v}" for k, v in row_line.items())
                     + f"  →  **{pl['fp']:.1f}** DK pts")
            st.caption(pl.get("method", ""))

        summ = mv.usage_summary(nk)
        if summ:
            flag = summ.pop("regression_flag", None)
            st.write("**Recent form** (" + str(summ.pop("games", "?")) + " most recent games in data): "
                     + " · ".join(f"{k} {v}" for k, v in summ.items()))
            if flag:
                st.caption(f"Regression: {flag}")
        ru = mv.recent_usage(nk)
        if not ru.empty:
            st.dataframe(ru, hide_index=True, width="stretch")

        win = st.radio("Timeframe (splits & scheme tables)", list(mv.WINDOWS),
                       index=0, horizontal=True, key="pl_window")
        yrs = mv.window_seasons(win)
        c1, c2 = st.columns(2)
        with c1:
            st.subheader(f"Coverage splits — {pos} ({win})")
            stat = "ypa" if pos == "QB" else "tprr"
            cs = mv.coverage_splits(nk, stat, seasons=yrs)
            st.dataframe(cs if not cs.empty else pd.DataFrame({"note": ["no split history in this window"]}),
                         hide_index=True, width="stretch")
            st.caption(f"{'yds/dropback' if stat == 'ypa' else 'targets/route'} vs each look. "
                       "`diff` vs the player's own baseline — historically small and unstable.")
        with c2:
            osch = mv.opponent_scheme(opp, seasons=yrs)
            shown = osch.attrs.get("years") if not osch.empty else None
            lbl = f" ({'-'.join(map(str, [shown[0], shown[-1]])) if shown and len(shown) > 1 else (shown[0] if shown else win)})"
            st.subheader(f"{opp} defense — scheme tendencies{lbl}")
            st.dataframe(osch if not osch.empty else pd.DataFrame({"note": ["no scheme data for opponent in this window"]}),
                         hide_index=True, width="stretch")
            st.caption("`team_%` vs `league_%` for the same years; `lean` = the gap; "
                       "`rank` 1–32, **1 = runs it most** in the league.")

        vc = mv.vs_coordinator(nk, opp)
        if vc:
            s = vc["summary"]
            if s.get("games"):
                line = (f"**Faced {vc['dc']}** ({opp}'s DC) **{s['games']}×** before: "
                        f"{s.get('fp_avg', '?')} DK pts/g")
                if s.get("xfp_avg") is not None:
                    line += f" (xFP {s['xfp_avg']})"
                st.write(line)
                g = vc["games"]
                if not g.empty:
                    st.dataframe(g.round(1), hide_index=True, width="stretch")
            else:
                st.caption(f"No games vs **{vc['dc']}** ({opp}'s DC since {vc['dc_since']}) "
                           "in the 2022–25 data.")

    # ── Data Check ─────────────────────────────────────────────────────────
    elif screen == "Data Check":
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

    # ── unknown key → the deployed code is behind the sidebar ─────────────
    else:
        st.header(f"🧬 {screen}")
        st.warning(
            f"This screen (`{screen}`) isn't in the running build — the deployed code is "
            "behind the sidebar menu. On Streamlit Cloud: **Manage app → ⋮ → Reboot** to "
            "pull the latest push. (Locally: restart `streamlit run`.)"
        )
