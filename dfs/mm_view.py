"""The 🏆 Millionaire Maker screen: build a portfolio of lineups under the structure the study supports, with the evidence beside the controls.

Rendered by `dfs.screens.render("Millionaire Maker")`, which has already loaded the week's DraftKings pool joined to the projections (`slate_df`).
"""
from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from dfs.config import DATA
from dfs.millionaire import MMConfig, build_portfolio, portfolio_summary, profile, read_ownership_file
from dfs.optimizer import SLOT_ORDER, OptimizerError, to_dk_upload

EVIDENCE_FILE = DATA / "history" / "mm_evidence.json"

MAIN_SLATE_TYPE = "Sunday main"                      # dk_client's name for DraftKings' main slate: Sunday 1 PM + 4 PM games only

STACK_OPTIONS = {"QB + 2 pass catchers": 2, "QB + 1 pass catcher": 1, "No stack requirement": 0}
TEAM_CAP_OPTIONS = {"3 (QB + 2 more)": 3, "4": 4, "No cap": 9}
DST_OPTIONS = {"Nobody in the lineup faces it": "lineup", "Not facing my QB": "qb", "No rule": "any"}


@st.cache_data(show_spinner=False)
def load_evidence(_mtime: float) -> dict | None:
    try:
        return json.loads(EVIDENCE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _evidence() -> dict | None:
    try:
        return load_evidence(EVIDENCE_FILE.stat().st_mtime)
    except OSError:
        return None


def _pct(x) -> str:
    return "–" if x is None or pd.isna(x) else f"{100 * x:.0f}%"


VERDICT_TEXT = {"better": "✅ better", "worse": "⚠️ worse", "unclear": "· no clear difference", "baseline": "baseline"}

# one tab per question: which hindsight trait groups, which forward-test group
EVIDENCE_TABS = (
    ("Stacks and teams", ("Stacking and teams", "Defense pairing"), "structure"),
    ("Spending by position", ("Spend by position", "Salary shape"), "spend"),
    ("Game totals and defenses", ("Game environment", "Opponent quality (as of kickoff)"), "environment"),
)


def _lift_text(lift, lo, hi, digits: int = 2) -> str:
    if lift is None or lo is None or hi is None or any(pd.isna(v) for v in (lift, lo, hi)):
        return "–"
    return f"{lift:.{digits}f}x ({lo:.{digits}f}-{hi:.{digits}f})"


def _forward_frame(fw: dict, group: str | None = None) -> pd.DataFrame:
    """The forward-test rows of one group (all rows when `group` is None), the baseline first and the rest best lift first, formatted for display."""
    rows = pd.DataFrame(fw.get("rows", []))
    if rows.empty:
        return rows
    if group is not None and "group" in rows.columns:
        rows = rows[(rows["group"] == group) | (rows["key"] == "base")]
    rows = rows.assign(_order=(rows["key"] != "base").astype(int), _lift=rows["lift"].fillna(0.0)).sort_values(["_order", "_lift"], ascending=[True, False])
    no1 = pd.Series(None, index=rows.index, dtype=object)
    lift1 = [_lift_text(a, b, c, 1) for a, b, c in zip(rows.get("lift1", no1), rows.get("lo1", no1), rows.get("hi1", no1))]
    return rows.assign(**{"top 5%": rows["top5"].map(_pct), "top 1%": rows["top1"].map(_pct), "mean": rows["mean"].round(1),
                          "lift at 5% (95% interval)": [_lift_text(a, b, c) for a, b, c in zip(rows["lift"], rows["lo"], rows["hi"])],
                          "lift at 1% (95% interval)": lift1,
                          "reads": rows.get("verdict", pd.Series("", index=rows.index)).map(lambda x: VERDICT_TEXT.get(x, x))})


def _show_forward(fw: dict, group: str | None, title: str) -> None:
    show = _forward_frame(fw, group)
    if len(show) <= 1:
        return
    st.markdown(f"**{title}** {fw.get('n_weeks', '?')} weeks, {fw.get('lineups_per_rule', '?')} lineups per rule set, each built from pre-game information only and scored on "
                "what really happened. A lineup is a top-5% (top-1%) lineup when it beats the 95th (99th) percentile of every lineup built that week by every rule set; "
                "`lift` is a rule set's rate over the unconstrained optimizer's, with an interval from resampling whole weeks, and `reads` says whether the 5% lift cleared 1.0.")
    st.dataframe(show[["rules", "mean", "top 5%", "top 1%", "lift at 5% (95% interval)", "lift at 1% (95% interval)", "reads"]], hide_index=True, width="stretch")


def _hindsight_frame(ev: dict, groups: tuple[str, ...]) -> pd.DataFrame:
    hs = pd.DataFrame(ev.get("hindsight", []))
    if hs.empty:
        return hs
    if "group" in hs.columns:
        hs = hs[hs["group"].isin(groups)]
    hs = hs.sort_values("rank1", ascending=False)
    return hs.assign(best=hs["rank1"].map(_pct), **{"top 25": hs["top25"].map(_pct), "typical": hs["typical"].map(_pct)}, lift=hs["lift"].map(lambda v: "–" if pd.isna(v) else f"{v:.1f}x"))


def _show_hindsight(ev: dict, groups: tuple[str, ...]) -> None:
    show = _hindsight_frame(ev, groups)
    if show.empty:
        return
    st.markdown(f"**What the best lineups of {ev.get('weeks', {}).get('hindsight', '?')} past weeks looked like.** `best` is the share of weeks the single best lineup had the trait; "
                "`top 25` averages each week's 25 best; `typical` is the same trait among the lineups a salary-only optimizer builds (its stacks and game picks are accidents, so it is a fair "
                "comparison there; where it is not, as for how the cap is split, the column is blank); `lift` = top 25 ÷ typical. These are picked after the scores are known, so the forward test below is the check.")
    st.dataframe(show[["trait", "best", "top 25", "typical", "lift"]], hide_index=True, width="stretch")


def _money(x) -> str:
    return "–" if x is None or pd.isna(x) else f"${x:,.0f}"


def _show_spend(ev: dict) -> None:
    sp = ev.get("spend") or {}
    slots = pd.DataFrame(sp.get("slots", []))
    if len(slots):
        st.markdown("**How the best lineups split the $50,000.** The mean spend at each slot of the single best lineup of every week (95% interval from resampling weeks), the mean over each week's "
                    "best 25, the range across weeks, and the mean by era to see whether it is stable. There is no ordinary-lineup comparison on purpose: the lineups a salary-only optimizer builds "
                    "have habits of their own (it puts a TE in the FLEX more than half the time), so a gap against them would measure the optimizer. Whether a split pays is the forward test's job.")
        eras = ["2014-17", "2018-21", "2026"]
        show = slots.assign(**{"best lineup (95% interval)": slots.apply(lambda r: f"{_money(r['best1'])} ({_money(r['lo'])}-{_money(r['hi'])})", axis=1),
                               "best 25": slots["top25"].map(_money), "weeks 10th-90th pct": slots.apply(lambda r: f"{_money(r['p10'])}-{_money(r['p90'])}", axis=1)})
        for e in eras:
            show[e] = slots["eras"].map(lambda d, e=e: _money(d[e]["best"]) if e in d else "–")
        st.dataframe(show[["slot", "best lineup (95% interval)", "best 25", "weeks 10th-90th pct"] + eras], hide_index=True, width="stretch")
    tiers = pd.DataFrame(sp.get("tiers", []))
    if len(tiers):
        st.markdown("**What each price tier has returned** (every main-slate player of every study week, no lineup involved). `5x rate` is the share who scored at least five points per $1,000 of salary; "
                    "`top finish` the share who ranked among the week's top 3 QBs / 6 RBs / 9 WRs / 3 TEs / 3 DSTs. A tier's share of the top finishes beside its share of the pool shows where the winners come from.")
        show = tiers.assign(**{"pts": tiers["pts"].round(1), "per $1k": tiers["per_k"].round(2),
                               "5x rate (95%)": tiers.apply(lambda r: f"{_pct(r['hit'])} ({_pct(r['hit_lo'])}-{_pct(r['hit_hi'])})", axis=1),
                               "top finish (95%)": tiers.apply(lambda r: f"{_pct(r['top'])} ({_pct(r['top_lo'])}-{_pct(r['top_hi'])})", axis=1),
                               "share of pool": tiers["pool_share"].map(_pct), "share of top finishes": tiers["top_share"].map(_pct)})
        st.dataframe(show[["pos", "tier", "players", "pts", "per $1k", "5x rate (95%)", "top finish (95%)", "share of pool", "share of top finishes"]], hide_index=True, width="stretch")


def _show_environment(ev: dict) -> None:
    eff = (ev.get("environment") or {}).get("effects", [])
    if not eff:
        return
    st.markdown("**Is the game already in the salary?** For every starter-priced player, the points he scored beyond what his salary averages in other seasons, regressed on the game around him "
                "(per one standard deviation of the factor, t-statistics clustered by week; positive = the price did not fully include it). The effects are small next to a lineup's "
                "spread of about 25 points, which the forward test below measures.")
    for spec in dict.fromkeys(r["spec"] for r in eff):
        rows = [r for r in eff if r["spec"] == spec]
        table = pd.DataFrame([{"position": r["pos"], "player-weeks": r["n"], **{t["label"]: f"{t['coef']:+.2f} (t {t['t']:+.1f})" for t in r["terms"]}} for r in rows])
        st.caption(spec)
        st.dataframe(table, hide_index=True, width="stretch")


def _evidence_panel() -> None:
    ev = _evidence()
    with st.expander("📊 Why these rules: what the best lineups of every week had in common", expanded=False):
        if not ev:
            st.caption("The study file (`data/dfs/history/mm_evidence.json`) is not on this deploy, so the evidence tables are not shown.")
            return
        for line in ev.get("summary", []):
            st.markdown(f"- {line}")
        titles = ev.get("group_titles", {})
        tabs = st.tabs([t for t, _, _ in EVIDENCE_TABS] + ["Combinations and 2026"])
        for tab, (name, hind_groups, fwd_group) in zip(tabs, EVIDENCE_TABS):
            with tab:
                _show_hindsight(ev, hind_groups)
                if fwd_group == "spend":
                    _show_spend(ev)
                if fwd_group == "environment":
                    _show_environment(ev)
                _show_forward(ev.get("forward") or {}, fwd_group, f"Does building to these rules pay before kickoff? ({titles.get(fwd_group, name)})")
        with tabs[-1]:
            _show_forward(ev.get("forward") or {}, "combined", "Rules combined.")
            _show_forward(ev.get("forward_fp") or {}, None, "The same test on 2026 with the FantasyPoints projections, every rule set.")
        for line in ev.get("caveats", []):
            st.caption("• " + line)


def _lineup_table(lineups, slate_df: pd.DataFrame) -> pd.DataFrame:
    """One row per lineup, whole: all nine roster slots in DraftKings' order (player and team), then the salary, projection, stack and build profile."""
    rows = []
    for i, lu in enumerate(lineups, 1):
        pr = profile(lu, slate_df)
        by_slot = {r["slot"]: f"{r['name']} ({r['team']})" for _, r in lu.players.iterrows()}
        rows.append({"#": i, **{slot: by_slot.get(slot, "") for slot in SLOT_ORDER},
                     "salary": lu.salary, "left": pr["salary_left"], "proj": round(lu.proj, 1), "ceiling": round(lu.ceiling, 1), "stack": lu.stack,
                     "teams": pr["teams"], "games": pr["games"], "own %": None if pd.isna(pr.get("own_sum", float("nan"))) else round(pr["own_sum"], 0)})
    return pd.DataFrame(rows)


def _show_lineup(lu) -> None:
    show = lu.players.copy()
    show["salary"] = show["salary"].map("${:,.0f}".format)
    show["proj"] = show["proj"].round(1)
    st.dataframe(show, hide_index=True, width="stretch")


def render(week: int, slate_df: pd.DataFrame, pick: str, pick_type: str) -> None:
    st.header("🏆 Millionaire Maker Lineup Builder")
    st.caption(f"Pool: {len(slate_df)} projected players · {pick.split('·')[0].strip()}. The NFL Millionaire Maker runs on DraftKings' Sunday main slate "
               "(the 1 PM and 4 PM games) and pays its top prize to one lineup in over a hundred thousand, so the builder aims at the top of the score range, not the average.")
    if pick_type and pick_type != MAIN_SLATE_TYPE:
        st.warning(f"The slate picked in the sidebar is '{pick_type}', not DraftKings' '{MAIN_SLATE_TYPE}' slate. The Millionaire Maker is entered on the Sunday main slate: "
                   "change the Slate type in the sidebar if you are building for the contest.")
    _evidence_panel()

    # ownership: a heuristic ordering unless the user supplies real numbers
    slate = slate_df.copy()
    own_note = "Ownership is estimated from value, projection and the game total: an ordering to lean on, not a measurement."
    try:
        from dfs.ownership import chalk_ownership
        slate = chalk_ownership(slate, week)
    except Exception:  # noqa: BLE001 -- a display extra must never take the screen down
        slate["own_pct"] = float("nan")
        own_note = "No ownership estimate is available for this slate."

    labels = {f"{r.name} ({r.pos}, {r.team}, ${r.salary:,.0f})": r.dk_id for r in slate.sort_values("proj", ascending=False).itertuples()}
    c1, c2 = st.columns(2)
    lock_lbls = c1.multiselect("🔒 Lock (in every lineup)", list(labels), key="mm_lock")
    excl_lbls = c2.multiselect("🚫 Exclude", list(labels), key="mm_excl")

    a1, a2, a3 = st.columns(3)
    n_lineups = int(a1.number_input("Lineups", 1, 150, 20, key="mm_n", help="The contest allows up to 150 entries. On the hosted app a large portfolio takes a minute or more to build."))
    lean = a2.slider("Ceiling lean", 0.0, 1.0, 0.5, 0.05, key="mm_lean",
                     help="0 builds to the median projection, 1 to the ceiling. A tournament this size rewards the high end, so the default leans halfway.")
    jitter = a3.slider("Randomness", 0, 60, 25, key="mm_jit", help="Each lineup is built from player values nudged up or down by up to this many percent, so the portfolio holds different combinations.") / 100.0

    b1, b2, b3 = st.columns(3)
    stack_label = b1.selectbox("QB stack", list(STACK_OPTIONS), index=0, key="mm_stack",
                               help="How many of the QB's own WR/TE the lineup must hold. 60% of the single best lineups of 138 past weeks held at least one (22% of ordinary optimizer lineups did), and built before kickoff, "
                                    "QB + 2 put about 1.2x as many lineups in a week's top 5% as no requirement (QB + 1: 1.1x). See the evidence panel above.")
    bring = b2.checkbox("Bring-back (an opposing WR/TE)", value=True, key="mm_bring",
                        help="Require a pass catcher from the QB's opponent. It adds little to a one-man stack, but on top of QB + 2 it raised the top-5% rate from 1.2x to 1.3x and the top-1% rate from 1.5x to 2.3x of ordinary lineups.")
    team_label = b3.selectbox("Most players from one team", list(TEAM_CAP_OPTIONS), index=0, key="mm_teamcap",
                              help="Counts the QB and every RB/WR/TE (the defense is separate). 99% of the best lineups held 3 or fewer from one team; the cap on its own made no measurable difference.")

    d1, d2 = st.columns(2)
    max_exp = d1.slider("Max exposure", 0.1, 1.0, 0.4, 0.05, key="mm_exp", help="No player in more than this share of the lineups (locked players excepted).")
    min_uniq = d2.slider("Min players different between lineups", 1, 6, 3, key="mm_uniq")

    with st.expander("More rules"):
        e1, e2, e3 = st.columns(3)
        min_salary = e1.slider("Min salary", 45000, 50000, 49000, 250, key="mm_minsal")
        dst_label = e2.selectbox("Defense", list(DST_OPTIONS), index=0, key="mm_dst",
                                 help="Rostering a defense against your own offensive players works against both. 'Nobody in the lineup faces it' forbids it outright: 91% of the best lineups of past weeks obeyed it "
                                      "(70% of ordinary lineups), and with a QB + 2 stack and a bring-back it gave the best rate of any rule set tested (1.36x vs 1.30x without it, a difference within noise).")
        e4, e5 = st.columns(2)
        min_games = int(e4.slider("Min games represented", 0, 8, 0, key="mm_mingames", help="0 = no rule."))
        min_teams = int(e5.slider("Min teams represented", 0, 9, 0, key="mm_minteams", help="0 = no rule."))
        st.markdown("**Ownership leverage** (optional)")
        st.caption(own_note)
        up = st.file_uploader("Your own projected-ownership file (CSV with a name column and an ownership % column; a team column helps)", type=["csv"], key="mm_ownfile")
        if up is not None:
            try:
                own, missing = read_ownership_file(pd.read_csv(up), slate)
                slate.loc[slate["dk_id"].isin(own.index), "own_pct"] = slate["dk_id"].map(own)
                st.caption(f"Matched {len(own)} players from your file." + (f" Not placed (projected 8+): {', '.join(missing[:12])}" if missing else ""))
            except Exception as ex:  # noqa: BLE001 -- a bad upload is reported, not fatal
                st.warning(f"Could not read that file: {ex}")
        use_own = st.checkbox("Apply ownership rules", value=False, key="mm_useown")
        f1, f2 = st.columns(2)
        own_max = f1.slider("Combined ownership of the nine players at most", 40, 250, 110, 5, key="mm_ownmax", disabled=not use_own)
        min_low = int(f2.slider("At least this many players at 5% ownership or less", 0, 5, 2, key="mm_lowown", disabled=not use_own))

    build = st.button("⚙️ Build Millionaire Maker lineups", type="primary", key="mm_build")
    if jitter > 0:
        st.caption("Randomness is on: click Build again for a different set of combinations.")
    if build:
        cfg = MMConfig(n_lineups=n_lineups, qb_stack=STACK_OPTIONS[stack_label], bring_back=1 if bring else 0, max_from_team=TEAM_CAP_OPTIONS[team_label],
                       min_games=min_games, min_teams=min_teams, dst_vs=DST_OPTIONS[dst_label], min_salary=min_salary, lean=lean, jitter=jitter, max_exposure=max_exp,
                       min_uniques=min_uniq, locks=[labels[x] for x in lock_lbls], excludes=[labels[x] for x in excl_lbls],
                       own_max=float(own_max) if use_own else None, min_low_own=min_low if use_own else 0)
        try:
            with st.spinner("Solving…"):
                st.session_state["mm_lineups"] = build_portfolio(slate, cfg)
            st.session_state["mm_slate"] = slate
        except OptimizerError as ex:
            st.session_state.pop("mm_lineups", None)
            st.error(str(ex))

    lineups = st.session_state.get("mm_lineups")
    if not lineups:
        return
    shown_slate = st.session_state.get("mm_slate", slate)
    s = portfolio_summary(lineups, shown_slate)
    st.success(f"{s['lineups']} lineups · {s['unique_players']} unique players · proj {s['proj_min']:.1f}–{s['proj_max']:.1f} (mean {s['proj_mean']:.1f}) · "
               f"{100 * s['with_stack']:.0f}% stacked · {100 * s['with_back']:.0f}% with a bring-back · {s['mean_teams']:.1f} teams / {s['mean_games']:.1f} games per lineup")
    st.caption("Each row is a whole lineup in DraftKings' roster order (the table scrolls sideways). Open Lineup detail below for every player's opponent, salary and projection.")
    st.dataframe(_lineup_table(lineups, shown_slate), hide_index=True, width="stretch")
    with st.expander("Lineup detail (every player's opponent, salary and projection)"):
        for i, lu in enumerate(lineups, 1):
            st.markdown(f"**#{i} · {lu.proj:.1f} pts · ${lu.salary:,} · {lu.stack}**")
            _show_lineup(lu)

    exposure: dict[str, int] = {}
    for lu in lineups:
        for _, r in lu.players.iterrows():
            exposure[r["name"]] = exposure.get(r["name"], 0) + 1
    if len(lineups) > 1:
        exp_df = (pd.DataFrame({"player": list(exposure), "lineups": list(exposure.values())}).assign(pct=lambda d: (d["lineups"] / len(lineups) * 100).round(0))
                  .sort_values("lineups", ascending=False).head(30))
        st.plotly_chart(px.bar(exp_df, x="pct", y="player", orientation="h", height=650, labels={"pct": "% of lineups"}), use_container_width=True)
        qb_mix = pd.DataFrame({"QB": list(s["qbs"]), "lineups": list(s["qbs"].values())})
        st.caption("QBs in the portfolio: " + ", ".join(f"{q} ×{n}" for q, n in zip(qb_mix["QB"], qb_mix["lineups"])))

    csv = to_dk_upload(lineups).to_csv(index=False).encode()
    st.download_button("⬇️ Download DK upload CSV", csv, file_name=f"dk_millionaire_maker_week{week}.csv", mime="text/csv", key="mm_dl")
