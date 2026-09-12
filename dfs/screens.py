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


@st.cache_data(ttl=1800, show_spinner=False)
def _prop_events():
    """Upcoming NFL games from the odds feed. Free — no credits."""
    from dfs.props import list_events
    return list_events()


@st.cache_data(ttl=900, show_spinner=False)
def _prop_edges(event_id: str, markets: tuple, week: int):
    """Player-prop lines vs our projection. Costs 1 credit/market — cached 15 min."""
    from dfs.props import event_prop_edges
    return event_prop_edges(event_id, list(markets), week)


@st.cache_data(ttl=900, show_spinner=False)
def _bulk_prop_edges(event_ids: tuple, event_labels: tuple, markets: tuple, week: int):
    """Same as _prop_edges but for several events at once. Costs
    len(event_ids)*len(markets) credits total — cached 15 min."""
    from dfs.props import bulk_prop_edges
    events = [{"id": i, "label": lbl} for i, lbl in zip(event_ids, event_labels)]
    return bulk_prop_edges(events, list(markets), week)


@st.cache_data(show_spinner="Building value board…")
def _value_table(week: int, proj_hash: str, sal_key: str, _plist, _sal: dict):
    """One row per projected player: FantasyPoints projection, value, chalk ownership."""
    from dfs.names import normalize_name
    from dfs.ownership import chalk_ownership

    rows = []
    for _, r in _plist.iterrows():
        nk = normalize_name(r["name"])
        fp = float(r["proj"])
        sal = _sal.get(nk)
        rows.append({
            "Player": r["name"], "Pos": str(r["pos"]).upper(), "Team": r["team"],
            "Opp": str(r.get("opp", "")).lstrip("@"),
            "FP": round(fp, 1),
            "Salary": int(sal) if sal else None,
            "Val": round(fp / (sal / 1000), 2) if sal else None,
            "_salary": sal,
        })
    df = pd.DataFrame(rows)
    df = chalk_ownership(df, week, salary_col="_salary", proj_col="FP",
                         pos_col="Pos", team_col="Team").rename(columns={"own_pct": "pOwn%"})
    return df.drop(columns=["_salary"])


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


def _cur_season() -> int:
    try:
        from matchup_model.opp.blend import current_season
        return current_season()
    except Exception:  # noqa: BLE001
        from datetime import date
        t = date.today()
        return t.year if t.month >= 3 else t.year - 1


def _parse_amer(best: str | None) -> int:
    """'bovada +150' / 'draftkings -113' -> 150 / -113; default -110."""
    import re
    m = re.search(r"([+-]\d{2,5})", str(best or ""))
    return int(m.group(1)) if m else -110


def _fmt_pulled_at(iso: str | None) -> str:
    if not iso:
        return "earlier"
    try:
        from datetime import datetime
        return datetime.fromisoformat(iso).strftime("%a %I:%M %p").lstrip("0").replace(" 0", " ")
    except Exception:  # noqa: BLE001
        return iso


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

        # ── player picker ─────────────────────────────────────────────────
        # candidates + a projection/volume for each; auto-pick the clear starters, then
        # let the user trim or add anyone.
        tdf["pos"] = tdf["pos"].astype(str).str.upper() if not tdf.empty else tdf.get("pos")
        proj_of, rt_of, att_of = {}, {}, {}
        if not tdf.empty:
            _pos = dict(zip(tdf["name"], tdf["pos"]))
            for _, rr in tdf[tdf["proj"].fillna(0) > 0].iterrows():
                proj_of[rr["name"]] = float(rr["proj"])
            pass_pool = sorted((n for n in proj_of if _pos.get(n) in ("WR", "TE")),
                               key=lambda n: -proj_of[n])
            rb_pool = sorted((n for n in proj_of if _pos.get(n) == "RB"),
                             key=lambda n: -proj_of[n])
            def_pass, def_rb = pass_pool[:3], rb_pool[:2]
        else:
            rt_of = mv.team_pass_catcher_volume(team)
            att_of = mv.team_back_volume(team)
            pass_pool, rb_pool = list(rt_of), list(att_of)
            pcut = max(rt_of.values()) * 0.35 if rt_of else 0     # real 2025 role only
            bcut = max(att_of.values()) * 0.35 if att_of else 0
            def_pass = [n for n in pass_pool if rt_of[n] >= pcut][:3]
            def_rb = [n for n in rb_pool if att_of[n] >= bcut][:2]

        def _lbl(n):
            if n in proj_of:
                return f"{n} · {proj_of[n]:.1f} proj"
            if n in rt_of:
                return f"{n} · {rt_of[n]} routes '25"
            if n in att_of:
                return f"{n} · {att_of[n]} carries '25"
            return n

        sel = st.multiselect("Players", pass_pool + rb_pool, default=def_pass + def_rb,
                             format_func=_lbl, key="mm_players",
                             help="Auto-picked from last year's role — add or remove anyone.")
        pass_set, rb_set = set(pass_pool), set(rb_pool)
        pass_players = [(n, proj_of.get(n)) for n in sel if n in pass_set]
        rb_players = [(n, proj_of.get(n)) for n in sel if n in rb_set]

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
        st.header("🔎 DFS Player Value")
        st.caption("Sortable board: FantasyPoints projection, value (FP / $1k) where a salary "
                   "exists, and a heuristic projected ownership. Click a column header to "
                   "sort; pick a player below for our model's full breakdown.")
        from dfs import matchup_view as mv
        from dfs.names import normalize_name

        try:
            plist = _projections(week, _hash_path(projection_path(week)))
        except ProjectionError as e:
            st.error(str(e))
            st.stop()
        _sal = {}
        if slate_df is not None and not slate_df.empty:
            _sal = {normalize_name(n): int(s) for n, s in zip(slate_df["name"], slate_df["salary"])}

        vt = _value_table(week, _hash_path(projection_path(week)),
                          hashlib.md5(repr(sorted(_sal.items())).encode()).hexdigest(),
                          plist, _sal)

        fc1, fc2, fc3 = st.columns([3, 1, 1])
        poss = fc1.multiselect("Position", ["QB", "RB", "WR", "TE", "DST"],
                               default=["QB", "RB", "WR", "TE"], key="pv_pos")
        priced_only = fc2.checkbox("Priced only", value=bool(_sal), key="pv_priced")
        sort_by = fc3.selectbox("Sort by", ["Val", "FP", "pOwn%", "Salary"], key="pv_sort")
        view = vt[vt["Pos"].isin(poss)] if poss else vt
        if priced_only:
            view = view[view["Salary"].notna()]
        view = view.sort_values(sort_by, ascending=False, na_position="last").reset_index(drop=True)

        cfg = {}
        if hasattr(st, "column_config"):
            cfg = {"Salary": st.column_config.NumberColumn(format="$%d"),
                   "pOwn%": st.column_config.NumberColumn("pOwn%", format="%.1f"),
                   "Val": st.column_config.NumberColumn(format="%.2f")}
        st.dataframe(view, hide_index=True, width="stretch", column_config=cfg)
        st.caption("`Val` = FP projection / $1k salary · `pOwn%` = heuristic chalk score "
                   "(value, raw projection, cheap-enabler pull, game total) — it ranks how "
                   "popular a player will be, not exact ownership. Our opportunity-model "
                   "projection is in the per-player detail below.")

        st.divider()
        st.subheader("🔬 Player detail")
        _names = list(view["Player"]) or list(vt["Player"])
        _plook = plist.set_index(plist["name"])
        choice = st.selectbox("Player", _names, key="pl_pick")
        r = _plook.loc[choice]
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        nk = normalize_name(str(r["name"]))
        opp = str(r.get("opp", "")).lstrip("@")
        pos = str(r["pos"]).upper()
        fp_proj = float(r["proj"])
        salary = _sal.get(nk)

        # NOT gated on mv.available() (that only checks the OLD Data Suite CSVs) — both
        # functions have their own fallback to the nflverse opportunity/trailing blend,
        # same as Prop Edges uses, so they work independently of the Data Suite tables.
        mp = mv.model_projection(fp_proj, nk, pos)
        pl = mv.projected_line(nk, pos, as_of_week=week)
        _src = pl.get("source", "")
        _src_lbl = {"opp_blend": "opp+trailing blend", "opp": "opportunity model",
                    "project_stats": "trailing usage×eff (fallback)"}.get(_src, "")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("FP proj", f'{fp_proj:.1f}', help="FantasyPoints' own weekly projection.")
        m2.metric("Model proj", f'{mp["proj"]:.1f}', delta=f'{mp["lean"]:+.1f} lean',
                  delta_color="off", help="FP proj + a backtested regression-to-expected lean.")
        m3.metric("Our proj", f'{pl["fp"]:.1f}' if pl.get("fp") is not None else "—",
                  help="Backtested per-position blend of an opportunity model (team pace × "
                       "pass rate × Vegas × usage share) and the player's trailing DK "
                       "average. Falls back to trailing usage×efficiency if nflverse data "
                       "is unavailable. 2025 holdout RMSE 7.45 vs 7.65 trailing-average.")
        m4.metric("Salary", f'${int(salary):,}' if salary else "—")
        val = mp["proj"] / (salary / 1000) if salary else 0.0
        _bits = []
        if _src == "opp_blend" and pl.get("opp_fp") is not None:
            _bits.append(f"opp {pl['opp_fp']:.1f} / trailing {pl['trailing_fp']:.1f} → blend {pl['fp']:.1f}")
        elif _src_lbl:
            _bits.append(f"source: {_src_lbl}")
        if salary:
            _bits.append(f"value (Model proj / $1k): **{val:.2f}**")
        if mp.get("reason"):
            _bits.append(f"Lean: {mp['reason']}")
        st.caption(f"**{r['team']}** vs **{opp or '—'}**" + ("  ·  " + "  ·  ".join(_bits) if _bits else ""))

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
            _line_total = pl.get("line_fp", pl["fp"])
            st.write("**Projected line** — " + " · ".join(f"{k} {v}" for k, v in row_line.items())
                     + f"  →  **{_line_total:.1f}** DK pts")
            st.caption((pl.get("method", "") or "")
                       + ("  ·  the *Our proj* number above blends this with the trailing "
                          "DK average" if pl.get("source") == "opp_blend" else ""))

        inj = mv.team_injuries(r["team"], week=week) if hasattr(mv, "team_injuries") else pd.DataFrame()
        if not inj.empty:
            st.write(f"**{r['team']} injury report (wk {week})** — usage redistributes to the group:")
            st.dataframe(inj, hide_index=True, width="stretch")
            st.caption("`p(play)`: Out 0 · Doubtful .25 · Questionable .70. `rec ×` / `rush ×` "
                       "= the usage-share multiplier this creates for each rotation player "
                       "(already baked into *Our proj*).")

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

    # ── Prop Edges ────────────────────────────────────────────────────────
    elif screen == "Prop Edges":
        st.header("🎰 DFS Prop Edges")
        st.caption(
            "Player-prop lines from US sportsbooks (The Odds API) next to our projection — "
            "a backtested blend of an opportunity model (team pace × pass rate × Vegas × "
            "usage share) and the player's trailing average. **Edge = our proj − line**; "
            "`lean` is the side that implies. ⚠️ Early season the model leans on 2025 usage, "
            "so a changed 2026 role can still show a false edge — the confidence filter "
            "(on by default) hides the worst of those."
        )
        from dfs import bets as _bets0
        conf_perf = _bets0.line_history_buckets()
        if not conf_perf.empty:
            st.caption(f"**2026 so far, by confidence tier** ({int(conf_perf['n'].sum())} "
                       "graded props pulled this season — small samples this early, read "
                       "directionally)")
            _small_col, _ = st.columns([7, 3])   # ~30% narrower than full width
            _small_col.dataframe(conf_perf, hide_index=True, width="content",
                                 height=35 * (len(conf_perf) + 1))

        if not ODDS_API_KEY:
            st.error("No Odds API key. Add a free key from the-odds-api.com as "
                     "`ODDS_API_KEY` in `.env` (don't paste it into code).")
            st.stop()

        from dfs.props import (CORE_MARKETS, MARKETS, PropsError, confident_only,
                               parlay_odds, role_stable)

        try:
            events, ev_rem = _prop_events()
        except PropsError as e:
            st.error(str(e)); st.stop()
        except Exception as e:  # noqa: BLE001
            st.error(f"Odds feed error: {type(e).__name__}: {e}"); st.stop()
        if not events:
            st.info("No NFL games in the next ~8 days from the odds feed.")
            st.stop()

        ev_labels = {e["label"]: e["id"] for e in events}
        sunday = [e for e in events if e.get("is_sunday")]

        ev_pick = st.selectbox("Game (single-game pull)", list(ev_labels), key="pe_game")
        event_id = ev_labels[ev_pick]

        _mk_nice = {"player_reception_yds": "rec yds", "player_receptions": "receptions",
                    "player_rush_yds": "rush yds", "player_rush_attempts": "rush att",
                    "player_pass_yds": "pass yds", "player_pass_tds": "pass TD",
                    "player_pass_attempts": "pass att"}
        mk_sel = st.multiselect(
            "Markets", list(MARKETS), default=CORE_MARKETS,
            format_func=lambda k: _mk_nice.get(k, k), key="pe_markets",
            help="Each market costs 1 API credit per pull, per game (free tier = 500/month).",
        )
        bulk_cost = len(sunday) * len(mk_sel)

        c1, c2 = st.columns(2)
        go = c1.button("💰 Pull this game", type="primary", key="pe_go",
                       help=f"Spends {len(mk_sel)} credit(s) — 1 per market.")
        go_bulk = c2.button(f"🏈 Pull ALL {len(sunday)} Sunday games (~{bulk_cost} credits)",
                            key="pe_go_bulk", disabled=not sunday,
                            help="One request per game — pulls every Sunday game's odds in "
                                 "one go so you can build parlays across the whole slate.")
        if ev_rem is not None:
            st.caption(f"Odds API credits left this month: **{ev_rem}**")

        if go or go_bulk:
            if not mk_sel:
                st.warning("Pick at least one market.")
            elif go_bulk and not sunday:
                st.warning("No Sunday games in the odds feed right now.")
            else:
                try:
                    if go_bulk:
                        ids = tuple(e["id"] for e in sunday)
                        labels = tuple(e["label"] for e in sunday)
                        with st.spinner(f"Pulling {len(sunday)} Sunday games ({bulk_cost} credits)…"):
                            _df, _rem = _bulk_prop_edges(ids, labels, tuple(mk_sel), week)
                        game_lbl = f"All {len(sunday)} Sunday games"
                        event_lbl_for_snapshot = game_lbl
                    else:
                        with st.spinner("Pulling prop lines…"):
                            _df, _rem = _prop_edges(event_id, tuple(mk_sel), week)
                        if not _df.empty and "game" not in _df.columns:
                            _df.insert(0, "game", ev_pick)
                        game_lbl = ev_pick
                        event_lbl_for_snapshot = ev_pick.split("  ·")[0]
                    nsnap = None
                    try:
                        from dfs.props import snapshot_lines
                        nsnap = snapshot_lines(_df, _cur_season(), int(week), event_lbl_for_snapshot)
                    except Exception:  # noqa: BLE001
                        pass
                    st.session_state["pe_data"] = {"df": _df, "rem": _rem, "game": game_lbl,
                                                   "snap": nsnap}
                    try:
                        from dfs.props import save_last_pull
                        save_last_pull(_df, game_lbl, _rem, nsnap)
                    except Exception:  # noqa: BLE001
                        pass
                except PropsError as e:
                    st.error(str(e))
                except Exception as e:  # noqa: BLE001
                    st.error(f"{type(e).__name__}: {e}")

        if "pe_data" not in st.session_state:
            try:
                from dfs.props import load_last_pull
                restored = load_last_pull()
                if restored is not None:
                    st.session_state["pe_data"] = restored
            except Exception:  # noqa: BLE001
                pass

        cache = st.session_state.get("pe_data")
        if not cache:
            st.info("Pick a game and markets, then **Pull this game** — or **Pull ALL Sunday "
                    "games** to build parlays across the whole slate.")
            st.stop()

        df = cache["df"]
        cc1, cc2 = st.columns([5, 1])
        cc1.caption(
            f"Lines for **{cache['game']}**"
            + (f"  ·  credits left: **{cache['rem']}**" if cache.get("rem") else "")
            + (f"  ·  {cache['snap']} lines saved for calibration → grade them on "
               "**Bet Log** after the game" if cache.get("snap") else "")
            + (f"  ·  📦 restored from a pull saved {_fmt_pulled_at(cache.get('pulled_at'))} "
               "— no credits spent" if cache.get("restored") else "")
        )
        if cc2.button("🗑️ Clear", key="pe_clear",
                      help="Forget the saved pull so the next Pull click is a fresh one."):
            from dfs.props import clear_last_pull
            clear_last_pull()
            st.session_state.pop("pe_data", None)
            st.rerun()
        if df is not None and df.attrs.get("errors"):
            st.caption(f"⚠️ {len(df.attrs['errors'])} game(s) failed to pull and were skipped.")
        if df is None or df.empty:
            st.warning("No comparable props came back — either the book hasn't posted this "
                       "game yet, or we have no recent game logs for those players.")
            st.stop()

        tc1, tc2 = st.columns([3, 2])
        tiers = tc1.multiselect(
            "Show tiers", ["lean", "solid", "strong", "high"], default=["solid", "strong"],
            key="pe_tiers",
            help="Confidence tier from |z| (standardized edge). Backtest: lean ~54% hit, "
                 "solid ~57% / +9% ROI, strong ~57% / +10% ROI, high ~62% / +19% ROI "
                 "(break-even 52.4%). `—` (under ~0.15 SD) is never shown — it's a coin flip.",
        )
        rec_only = tc2.toggle(
            "✓ Recommended only", value=True, key="pe_recommended",
            help="Also requires role_ratio between 0.85-1.20 (this week's FantasyPoints "
                 "projection agrees our number isn't running on a stale 2025 role) AND our "
                 "proj under 1.4x the line. This is the extra screen on top of the tier — "
                 "off shows everything in the selected tier(s), including rows a changed "
                 "2026 role could be inflating.",
        )
        show = confident_only(df)
        show = show[show["conf"].isin(tiers)] if tiers else show.iloc[0:0]
        if rec_only:
            show = role_stable(show)
        hidden = len(df) - len(show)
        if show.empty:
            st.warning("Nothing cleared the filters. Widen the tiers, or turn off "
                       "\"Recommended only\" to see rows with a role-ratio flag.")
            st.stop()

        view = show.drop(columns=["role_ratio", "game"])
        try:
            from dfs import matchup_view as _mvh
            table = _mvh.heat(view, ["p edge", "z"], good_high=True)
        except Exception:  # noqa: BLE001
            table = view
        st.dataframe(table, hide_index=True, width="stretch")
        st.caption(
            "`conf` = confidence tier from |z| (standardized edge = SDs past the line, "
            "comparable across markets). `z` = the raw number · `p(hit)%` = our shrunk "
            "probability the bet lands · `book %` = de-vigged book prob · `p edge` = "
            "p(hit) − book% (EV proxy) · `best` = best price across DK/FD."
            + (f"  ·  {hidden} row(s) hidden (no real edge, a role-ratio red flag, or "
               "outside the selected filters)." if hidden else "")
        )
        st.caption(
            "**role_ratio** (hidden from the table, used by the Recommended filter) = this "
            "week's FantasyPoints projection ÷ our model's own number. Near 1.0 = they agree "
            "— role looks stable. Far from 1.0 = FantasyPoints' current-season info disagrees "
            "with our still mostly-2025-trained read, i.e. a real role change our model "
            "doesn't know about yet."
        )
        if not df["role_ratio"].notna().any():
            st.caption("⚠️ No weekly projection file for this week — couldn't run the role "
                       f"check. Drop the export at `{projection_path(week)}`.")

        # ── parlay builder ──────────────────────────────────────────────
        with st.expander("🎰 Build a parlay", expanded=len(show) > 0):
            recs = show.to_dict("records")
            leg_opts = {f'{d["player"]} · {d["market"]} · {d["lean"]} {d["line"]} '
                        f'· {d["conf"]} · {d.get("game", cache["game"])} '
                        f'({d.get("best") or "?"})': d for d in recs}
            legs_pick = st.multiselect("Legs (from the table above)", list(leg_opts), key="pe_legs")
            if len(legs_pick) >= 2:
                legs = [leg_opts[k] for k in legs_pick]
                prices = [_parse_amer(g.get("best")) for g in legs]
                combo = parlay_odds(prices)
                p_our = 1.0
                for g in legs:
                    p_our *= max(min((g.get("p(hit)%") or 55) / 100.0, 0.97), 0.03)
                stake = st.number_input("Stake ($)", 0.0, 100000.0, 10.0, 5.0, key="pe_parstake")
                payout = stake * (combo["decimal"] - 1)
                ev = p_our * payout - (1 - p_our) * stake
                pc1, pc2, pc3, pc4 = st.columns(4)
                pc1.metric("Combined odds", f"{combo['american']:+d}")
                pc2.metric("Our win prob", f"{100*p_our:.1f}%")
                pc3.metric("Payout on win", f"${payout:,.2f}")
                pc4.metric("Est. EV", f"${ev:+,.2f}", delta_color="off")
                games_in_parlay = {g.get("game", cache["game"]) for g in legs}
                if len(games_in_parlay) < len(legs):
                    st.caption("⚠️ Two or more legs share a game — they're **not independent**. "
                               "A QB's pass yds and his own WR1's rec yds tend to hit together "
                               "(correlated up); two RBs' rush yds on the same team tend to "
                               "trade off (correlated down). The win-prob above assumes "
                               "independence and will be off for those legs specifically.")
                else:
                    st.caption("All legs are from different games — the independence "
                               "assumption for the combined probability is reasonable.")
                st.caption("Uses each leg's best DK/FD price · the parlay itself isn't logged "
                           "to the Bet Log yet — log the individual legs below if you want to "
                           "track them.")
            elif legs_pick:
                st.caption("Pick at least 2 legs to price a parlay.")

        # log a single bet straight from the table
        from dfs import bets as _bets
        with st.expander("🧾 Log a single bet from this table"):
            recs = show.to_dict("records")
            opts = {f'{d["player"]} · {d["market"]} · {d["lean"]} {d["line"]} '
                    f'({d.get("best") or "?"})': d for d in recs}
            pick_bet = st.selectbox("Row", list(opts), key="pe_logrow")
            br = opts[pick_bet]
            lc1, lc2, lc3 = st.columns(3)
            odds_in = lc1.number_input("Odds (american)", -100000, 100000,
                                       _parse_amer(br.get("best")), step=5, key="pe_logodds")
            stake_in = lc2.number_input("Stake (units)", 0.0, 100.0, 1.0, 0.5, key="pe_logstake")
            book_in = lc3.text_input("Book", str(br.get("best") or "").split()[0] if br.get("best") else "",
                                     key="pe_logbook")
            if st.button("Log it", key="pe_logbtn"):
                bid = _bets.add_bet(
                    season=int(_cur_season()), week=int(week),
                    event=str(br.get("game", cache["game"])).split("  ·")[0],
                    player=br["player"], market=br["market"],
                    side=br["lean"], line=float(br["line"]), odds=int(odds_in), book=book_in,
                    stake=float(stake_in), our_proj=float(br["our proj"]),
                    ev_pct=(float(br["p edge"]) if br.get("p edge") is not None else None),
                )
                st.success(f"Logged ({bid}). Grade it on the **Bet Log** screen after the game.")

    # ── Bet Log ──────────────────────────────────────────────────────────
    elif screen == "Bet Log":
        st.header("🧾 DFS Bet Log")
        st.caption("Log prop bets as you place them, grade them from box scores after the "
                   "games, and see which **standardized edge (z)** actually cashes. Stored "
                   "in `data/dfs/bets/bet_log.csv` (commit it to sync across machines).")
        from dfs import bets as _bets

        log = _bets.load()
        top = st.columns(4)
        ov = _bets.overall(log)
        top[0].metric("Bets", ov["bets"])
        top[1].metric("Record", f'{int((log["result"]=="win").sum())}-'
                                f'{int((log["result"]=="loss").sum())}-'
                                f'{int((log["result"]=="push").sum())}')
        top[2].metric("Units", f'{ov["units"]:+.2f}' if ov["units"] is not None else "—")
        top[3].metric("ROI", f'{ov["ROI%"]:+.1f}%' if ov["ROI%"] is not None else "—")

        gc1, gc2 = st.columns([1, 3])
        if gc1.button("⚖️ Grade ungraded", key="bl_grade"):
            _, n = _bets.grade()
            _, m = _bets.grade_line_history()
            st.success(f"Graded {n} bet(s) and {m} pulled line(s) from nflverse box scores.")
            st.rerun()
        gc2.caption("Grading needs the nflverse cache and only works once the game has been "
                    "played and stats posted.")

        # ── calibration: which standardized edge actually wins ──────────
        st.subheader("📈 Which edge wins — by z (standardized edge)")
        st.caption("`z` = how many outcome-SDs our projection sits past the line — comparable "
                   "across markets (a +33% rec-yd edge and a +7% pass-yd edge can be the same "
                   "z). Every **Prop Edges** pull auto-saves its lines here; *Grade ungraded* "
                   "fills them after games.")
        lh = _bets.load_line_history()
        graded_lh = int(lh["result"].isin(["win", "loss", "push"]).sum()) if not lh.empty else 0
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown(f"**From your pulled lines** — real book lines · {graded_lh} graded "
                        f"of {len(lh)}")
            lb = _bets.line_history_buckets(lh)
            if lb.empty:
                st.info("No graded pulled lines yet. Pull odds on **Prop Edges**, then grade "
                        "after the games.")
            else:
                st.dataframe(lb, hide_index=True, width="stretch")
        with cc2:
            st.markdown("**From the 2023–25 backtest** — our proj vs a trailing-form "
                        "stand-in line (no real odds needed)")
            bb = _bets.backtest_buckets()
            if bb.empty:
                st.info("Run `python -m matchup_model.opp.edge_calib` once to populate this.")
            else:
                st.dataframe(bb, hide_index=True, width="stretch")
        st.caption("Break-even at -110 is **52.4%**. Backtest read: **|z| ≥ 0.30 SD** is the "
                   "threshold (≈57% / +9% ROI); below 0.15 SD is a coin flip. rush yds / rush "
                   "att need z ≥ 0.60; skip pass att. The backtest is optimistic vs a real "
                   "book line — trust the threshold, not the absolute win%.")

        with st.expander("➕ Add a bet manually"):
            f = st.columns(3)
            p_name = f[0].text_input("Player", key="bl_p")
            p_mkt = f[1].selectbox("Market", list(_bets._STAT_COL), key="bl_m")
            p_side = f[2].selectbox("Side", ["OVER", "UNDER"], key="bl_s")
            g = st.columns(4)
            p_line = g[0].number_input("Line", 0.0, 1000.0, 0.5, 0.5, key="bl_l")
            p_odds = g[1].number_input("Odds", -100000, 100000, -110, 5, key="bl_o")
            p_book = g[2].text_input("Book", key="bl_b")
            p_stake = g[3].number_input("Stake (u)", 0.0, 100.0, 1.0, 0.5, key="bl_st")
            h = st.columns(3)
            p_wk = h[0].number_input("Week", 1, 18, int(week), key="bl_wk")
            p_proj = h[1].number_input("Our proj (optional)", 0.0, 1000.0, 0.0, 0.1, key="bl_pr")
            p_note = h[2].text_input("Note", key="bl_nt")
            if st.button("Add", key="bl_add") and p_name:
                _bets.add_bet(season=int(_cur_season()), week=int(p_wk),
                              event="", player=p_name, market=p_mkt, side=p_side,
                              line=float(p_line), odds=int(p_odds), book=p_book,
                              stake=float(p_stake),
                              our_proj=float(p_proj) or None, note=p_note)
                st.rerun()

        if log.empty:
            st.info("No bets logged yet.")
        else:
            show_cols = ["logged_at", "week", "player", "market", "side", "line", "our_proj",
                         "edge_pct_toward", "odds", "book", "stake", "result", "actual", "payout"]
            st.dataframe(log[show_cols].sort_values("logged_at", ascending=False),
                         hide_index=True, width="stretch")
            dc1, dc2 = st.columns([2, 1])
            del_id = dc1.text_input("Delete a bet by id", key="bl_del")
            if dc2.button("Delete", key="bl_delbtn") and del_id:
                _bets.delete_bet(del_id.strip())
                st.rerun()

            st.subheader("📊 Your logged bets, by confidence tier")
            eb = _bets.by_edge_bucket(log)
            if eb.empty:
                st.info("Grade some logged bets to see your own results by confidence tier.")
            else:
                st.dataframe(eb, hide_index=True, width="stretch")
                st.caption("`conf` = the same tier shown on Prop Edges (— / lean / solid / "
                           "strong / high), from |z| with the per-market adjustment applied. "
                           "Compare against the calibration tables above — if your live "
                           "results diverge from the backtest shape, that's information.")
            b1, b2 = st.columns(2)
            with b1:
                st.caption("By market")
                mk = _bets.by_key(log, "market")
                st.dataframe(mk if not mk.empty else pd.DataFrame({"note": ["—"]}),
                             hide_index=True, width="stretch")
            with b2:
                st.caption("By side")
                sd = _bets.by_key(log, "side")
                st.dataframe(sd if not sd.empty else pd.DataFrame({"note": ["—"]}),
                             hide_index=True, width="stretch")

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
