"""Parlay Sim screen: price a same-game parlay from the game simulator and compare it with what the book is offering.

Pick a game, take (or type) the legs with the book's odds, and the screen shows the parlay's fair price. Each leg is pinned to the book's no-vig probability, so the single
legs are the market's and only the way they move together comes from the simulator (`matchup_model.sim`). Books price a same-game parlay with their own correlation and
margin; the fair price is what to compare their offer with. The simulator is validated against past lines in `matchup_model/sim_report.md`. Under the price sits the legs'
record: how often these players have hit these lines before, each alone and all together in one game (`matchup_model.sim.past`); under that, lines that tend to hit when the legs do,
with a button to add them (`matchup_model.sim.recommend`).
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import streamlit as st

from dfs import game_view as GV
from dfs.names import normalize_name
from matchup_model.sim import engine as E
from matchup_model.sim import inputs as I
from matchup_model.sim import past as P
from matchup_model.sim import recommend as R
from matchup_model.sim import sgp as S

N_SIMS = 20000
ASSUMED_VIG = 1.045                       # total implied probability of a two-way market, used only when a leg has one price
MARKETS = {"Pass yds": "pass_yds", "Pass TD": "pass_td", "Rec yds": "rec_yds", "Receptions": "rec", "Rush yds": "rush_yds", "Rush attempts": "rush_att", "Anytime TD": "any_td",
           "Team spread": "spread", "Team moneyline": "moneyline", "Team total": "team_total", "Game total": "total"}
PULL_MARKETS = {"rec yds": "rec_yds", "receptions": "rec", "rush yds": "rush_yds", "pass yds": "pass_yds", "pass TD": "pass_td"}
SIDES = ["Over", "Under", "Yes", "No"]
COLS = ["Who", "Market", "Side", "Line", "Odds", "Opposite odds"]


def to_side(label: str) -> str:
    return "over" if label in ("Over", "Yes") else "under"


def leg_probability(odds, opposite) -> float | None:
    """The book's no-vig probability that the bet side hits, from its price (and the other side's price when given); None when no price was entered."""
    if odds is None or pd.isna(odds) or float(odds) == 0:
        return None
    p = S.to_prob_american(float(odds))
    if opposite is not None and not pd.isna(opposite) and float(opposite) != 0:
        return p / (p + S.to_prob_american(float(opposite)))
    return min(p / ASSUMED_VIG, 0.99)


def disagreements(legs: list[S.Leg], sim_probs) -> list[str]:
    """The legs on which the simulator and the book disagree sharply (the simulator's own probability under 2% or over 98%, or 30 points from the book's), worded for a warning."""
    return [f"{l.label()} (simulator {o:.0%}, book {l.p:.0%})" for l, o in zip(legs, sim_probs) if l.p is not None and (o < 0.02 or o > 0.98 or abs(o - l.p) > 0.30)]


def legs_from_rows(rows: pd.DataFrame, players: set[str], teams: set[str]) -> tuple[list[S.Leg], list[str]]:
    legs: list[S.Leg] = []
    errors: list[str] = []
    for i, r in rows.iterrows():
        if pd.isna(r.get("Who")) or pd.isna(r.get("Market")):
            continue
        key = MARKETS.get(r["Market"])
        if key is None:
            errors.append(f"row {i + 1}: unknown market")
            continue
        side = to_side("Over" if pd.isna(r.get("Side")) else r["Side"])
        p = leg_probability(r.get("Odds"), r.get("Opposite odds"))
        line = 0.0 if pd.isna(r.get("Line")) else float(r["Line"])
        who = r["Who"]
        if key in S.TEAM_MARKETS:
            if key == "total":
                legs.append(S.Leg("total", side, line, p=p))
            elif who in teams:
                legs.append(S.Leg(key, side, line, team=who, p=p))
            else:
                errors.append(f"row {i + 1}: {r['Market']} needs a team in Who")
        elif who in players:
            if key in ("any_td",) or line > 0:
                legs.append(S.Leg(key, side, line, player=who, p=p))
            else:
                errors.append(f"row {i + 1}: {who} {r['Market']} needs a line")
        else:
            errors.append(f"row {i + 1}: {who} is not in this game's simulation")
    return legs, errors


@st.cache_resource(show_spinner="Simulating the game…", max_entries=3)
def _simulate(season: int, week: int, away: str, home: str, fp_hash: str, removed: tuple, n: int):
    from dfs.projections import load_weekly_projections  # noqa: F401  (the table is already joined to FantasyPoints)
    table = GV._table(season, week, away, home, fp_hash)
    info = I.game_row(season, week, away, home)
    t = table[table["ours"].notna()]
    if "has_fp" in t and t["has_fp"].any():
        t = t[t["has_fp"]]
    t = t[~t["name"].isin(removed)]
    h, a = I.specs_from_table(t, away, home, info["away_implied"], info["home_implied"])
    return E.simulate(h, a, n=n, seed=2026).compact(), h, a


def pull_candidates(table: pd.DataFrame) -> pd.DataFrame:
    """Lines from the last Prop Edges pull when it is for this game (matched by the players in it): player, market, line, p_over."""
    from dfs import props
    pull = props.load_last_pull()
    if not pull or pull.get("df") is None or pull["df"].empty:
        return pd.DataFrame()
    df = pull["df"]
    mine = {normalize_name(n) for n in table["name"]}
    theirs = df["player"].map(normalize_name)
    if len(mine & set(theirs)) < max(3, 0.5 * theirs.nunique()):
        return pd.DataFrame()
    d = df[df["market"].isin(PULL_MARKETS) & df["book %"].notna()].copy()
    d["p_over"] = np.where(d["lean"] == "OVER", d["book %"] / 100.0, 1 - d["book %"] / 100.0)
    return d[["player", "market", "line", "p_over"]].reset_index(drop=True)


def _idea_table(res: E.SimResult, cand: pd.DataFrame, players: set[str], min_joint: float = 0.08, top: int = 15) -> pd.DataFrame:
    """The most positively associated two-leg parlays among the pulled lines (both sides of every line): lift = simulator joint / product of the books' probabilities."""
    names = {normalize_name(p): p for p in players}
    legs = []
    for r in cand.itertuples():
        nm = names.get(normalize_name(r.player))
        stat = PULL_MARKETS.get(r.market)
        if nm is None or stat is None or nm not in res.stats:
            continue
        for side, p in (("over", r.p_over), ("under", 1 - r.p_over)):
            if 0.30 <= p <= 0.72:
                legs.append(S.Leg(stat, side, float(r.line), player=nm, p=float(p)))
    ev = [S.leg_events(res, l) for l in legs]
    rows = []
    for i, j in itertools.combinations(range(len(legs)), 2):
        if legs[i].player == legs[j].player:
            continue
        a, b = legs[i].p, legs[j].p
        pa, pb = float(ev[i].mean()), float(ev[j].mean())
        p11 = float((ev[i] & ev[j]).mean())
        q = S.odds_ratio_fit(p11, pa - p11, pb - p11, 1 - pa - pb + p11, a, b, n=len(ev[i]))
        if q >= min_joint:
            rows.append({"Parlay": f"{legs[i].label()}  +  {legs[j].label()}", "Joint": q, "If independent": a * b, "Lift": q / (a * b), "Fair odds": S.to_american(q)})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Lift", ascending=False).head(top)


def corr_map(res: E.SimResult, h: E.TeamSpec, a: E.TeamSpec, top: int = 12) -> pd.DataFrame:
    """The game's main player legs at the simulator's own medians (each about a coin flip) and how they move together: phi correlation and the chance both hit."""
    legs = R.median_legs(res, h, a)
    labels = [l.label() for l in legs]
    ev = [S.leg_events(res, l) for l in legs]
    rows = []
    for i, j in itertools.combinations(range(len(legs)), 2):
        if legs[i].player == legs[j].player:
            continue
        pa, pb = float(ev[i].mean()), float(ev[j].mean())
        if not (0.12 < pa < 0.88 and 0.12 < pb < 0.88):
            continue
        p11 = float((ev[i] & ev[j]).mean())
        phi = (p11 - pa * pb) / np.sqrt(pa * (1 - pa) * pb * (1 - pb))
        rows.append({"Pair": f"{labels[i]}  +  {labels[j]}", "Phi": phi, "Both hit": p11, "If independent": pa * pb})
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows).sort_values("Phi", ascending=False)
    return pd.concat([d.head(top), d.tail(max(top // 2, 1))]).drop_duplicates("Pair")


def _fmt_odds(x: float) -> str:
    return "n/a" if not np.isfinite(x) else f"{x:+.0f}"


def _cell(k: int, n: int) -> str:
    return "no games" if n == 0 else f"{k} of {n} ({k / n:.0%})"


def _render_past(legs: list[S.Leg], joint: float, season: int, week: int, away: str, home: str) -> None:
    """The legs' record in past games, each alone and all together, with the number of games behind every figure. A description: it does not enter the price."""
    st.subheader("How often these legs have hit before")
    try:
        past = P.past_hits(legs, season, week, (away, home))
    except Exception as exc:  # noqa: BLE001 -- the price above stands without the record
        st.caption(f"The past-game record is not available right now ({exc}).")
        return

    def cells(pairs) -> dict:
        return {w: _cell(hits, games) for w, (hits, games) in zip(past.windows, pairs)}

    rows = [{"Leg": l["label"] + ("" if l["games"] else "  (no past games found)"), **cells(l["cells"])} for l in past.legs]
    k, n = past.together[-1]                                                      # the widest window holds every game there is
    if len(legs) > 1:
        rows.append({"Leg": "All legs together, in one game", **cells(past.together)})
        rows.append({"Leg": "   if they had been independent in those games", **{w: "" if p is None else f"{p:.0%}" for w, p in zip(past.windows, past.independent)}})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if len(legs) > 1:
        if n:
            lo, hi = P.wilson(k, n)
            st.markdown(f"**Together since {past.first}: {k} of {n} games ({k / n:.0%}).** With {n} games the real rate could be anywhere from about {lo:.0%} to {hi:.0%} (90% range). "
                        f"The price above is {joint:.1%}.")
        else:
            st.caption(f"These players have not all been in a game together since {past.first} (or one of them is missing from the box scores), so there is no record for the parlay as a whole.")
    st.caption(f"Regular-season games before this one, {past.first} on, judged on the lines above (not the lines those games were set at). The leg rows count every game that player played; the "
               "last two rows count only the games in which all of the players were in the same game, with what independence would give from each leg's own rate in those games. A player counts "
               "as having played with a stat line or an offensive snap, and a stat exactly on a whole-number line is a miss. This is the record, not a forecast, and it does not change the price: "
               "lines, roles, quarterbacks and opponents differ from game to game.")
    if n:
        with st.expander(f"The {n} games these players were all in"):
            st.dataframe(past.games.head(60), hide_index=True, width="stretch")


def book_legs(cand: pd.DataFrame, players: set[str]) -> list[S.Leg]:
    """The lines pulled from the books for this game, as over-side legs carrying the book's probability of the over."""
    names = {normalize_name(p): p for p in players}
    out = []
    for r in cand.itertuples():
        nm, stat = names.get(normalize_name(r.player)), PULL_MARKETS.get(r.market)
        if nm is not None and stat is not None:
            out.append(S.Leg(stat, "over", float(r.line), player=nm, p=float(r.p_over)))
    return out


def row_from_leg(leg: S.Leg) -> dict:
    """The legs-table row for a leg. A leg with a probability gets the prices that give it back: a main spread or total -110 each way, anything else the no-vig prices of its two sides."""
    label = next(k for k, v in MARKETS.items() if v == leg.market)
    side = ("Yes" if leg.side == "over" else "No") if leg.market in ("any_td", "spread", "moneyline") else ("Over" if leg.side == "over" else "Under")
    if leg.p is None:
        odds = opposite = None
    elif leg.market in S.TEAM_MARKETS:
        odds = opposite = -110.0
    else:
        odds, opposite = (float(round(S.to_american(q))) for q in (leg.p, 1.0 - leg.p))
    return {"Who": leg.player or leg.team or "Game", "Market": label, "Side": side, "Line": None if leg.market == "any_td" else float(leg.line), "Odds": odds, "Opposite odds": opposite}


def _past_cells(legs: list[S.Leg], picks: pd.DataFrame, season: int, week: int, away: str, home: str) -> list[tuple[str, str]]:
    """Per pick: how often it hit together with the legs in past games, and how often it hit in the games where the legs did (n/a for a spread or total: those games had other lines)."""
    cells = []
    for leg in picks["leg"]:
        if leg.market in S.TEAM_MARKETS:
            cells.append(("n/a", "n/a"))
            continue
        past = P.past_hits(legs + [leg], season, week, (away, home))
        (k, n), (x, y) = past.together[-1], past.last_given_rest[-1]
        cells.append((f"{k} of {n}" if n else "never together", f"{x} of {y} ({x / y:.0%})" if y else "n/a"))
    return cells


def _render_picks(res: E.SimResult, h: E.TeamSpec, a: E.TeamSpec, legs: list[S.Leg], cand: pd.DataFrame, table_now: pd.DataFrame, season: int, week: int, away: str, home: str, rows_key: str) -> None:
    """Lines that tend to hit when all of the legs hit (`matchup_model.sim.recommend`), with their past record and a button that adds the ones picked to the legs table."""
    st.subheader("Picks that tend to hit with your legs")
    st.caption("Other lines that tend to hit when all of your legs above hit, the best one per player (players already in your parlay are left out). The chances come from the simulator's link between legs, "
               "with every leg you gave odds for held at the book's number.")
    rank = st.radio("Rank by", ["Hits when your legs hit", "Lift"], horizontal=True, key="sgp_rank",
                    help="Hits when your legs hit: the share of the simulated games where all your legs hit in which the line also hits. Lift: that over the line's chance on its own, which favours lines that are less likely alone.")
    info = I.game_row(season, week, away, home)
    game = None if info is None else {"home": home, "away": away, "spread_home": info["spread_home"], "total": info["total"]}
    options = R.candidate_legs(res, h, a, book_legs(cand, {p.name for p in h.players + a.players}), game)
    picks = R.recommend(res, legs, options, rank="lift" if rank == "Lift" else "when_given")
    if picks.empty:
        st.caption(f"No line clears the filters: at least {R.MIN_LIFT - 1:.0%} likelier to hit with your legs than alone, and between {R.BAND[0]:.0%} and {R.BAND[1]:.0%} on its own.")
        return
    try:
        past = _past_cells(legs, picks, season, week, away, home)
    except Exception:  # noqa: BLE001 -- the suggestions stand without their record
        past = [("n/a", "n/a")] * len(picks)
    shown = pd.DataFrame({"Pick": [l.label() for l in picks["leg"]], "Chance alone": picks["alone"], "Hits when your legs hit": picks["when_given"], "Lift": picks["lift"],
                          "Parlay with it": picks["joint"], "Fair odds": picks["fair_odds"], "Line": picks["source"], "Hit together before": [c[0] for c in past],
                          "When your legs hit, before": [c[1] for c in past]})
    st.dataframe(shown, hide_index=True, width="stretch", column_config={
        "Chance alone": st.column_config.NumberColumn(format="percent"), "Hits when your legs hit": st.column_config.NumberColumn(format="percent"),
        "Lift": st.column_config.NumberColumn(format="%.2fx"), "Parlay with it": st.column_config.NumberColumn(format="percent"), "Fair odds": st.column_config.NumberColumn(format="%+d")})
    st.caption("Lines marked 'simulator median' are not book lines (each is set where the simulator thinks the stat is a coin flip): look up the book's number and enter it above. The last two columns are "
               "the record since 2021 in the games where all the players played, judged on these lines (not shown for a spread or total, whose lines differ from game to game). These are lines that "
               "go together, not bets with an edge: the book's price for the parlay may already charge for the link, so compare its offer with the fair odds. The simulator's link was confirmed on "
               "past lines for a quarterback and his receivers only (`matchup_model/sim_report.md`); for other pairs it is weak evidence, and spread, total and touchdown links were checked only against correlations.")
    chosen = st.multiselect("Add to my legs", list(shown["Pick"]), key="sgp_picks_pick")
    if st.button("Add these legs", key="sgp_picks_add") and chosen:
        new = pd.DataFrame([row_from_leg(l) for l in picks["leg"] if l.label() in chosen], columns=COLS)
        now = table_now[table_now[["Who", "Market", "Line"]].notna().any(axis=1)]                    # the table as it shows now, edits included
        st.session_state[rows_key] = pd.concat([now, new], ignore_index=True)
        st.session_state["sgp_editor_v"] = st.session_state.get("sgp_editor_v", 0) + 1                # a fresh editor: the old one's edits are already in the table
        st.rerun()


def render(week: int) -> None:
    st.header("🎲 DFS Parlay Sim")
    st.caption("Price a same-game parlay. The game is simulated 20,000 times from our own projections; each leg you add is pinned to the book's no-vig probability, so the "
               "single legs are the market's and the simulator only supplies how they move together. Compare the fair odds with the price the book offers. "
               "Validation against past lines: `matchup_model/sim_report.md`.")
    season = GV._season()
    games = GV._games(season, week)
    if games.empty:
        st.info(f"No week {week} games in the nflverse schedule yet (season {season}).")
        return
    pick = st.selectbox("Game", list(games["label"]), index=GV.default_game_index(games), key="sgp_game")
    g = games.iloc[list(games["label"]).index(pick)]
    away, home = g["away"], g["home"]
    fp_path = GV._fp_path(week)
    fp_hash = GV._file_hash(fp_path)
    table = GV._table(season, week, away, home, fp_hash)
    base = table[table["ours"].notna()]
    if GV._load_fp(week) is not None and "has_fp" in base:
        base = base[base["has_fp"]]
    base = base[base["inj"] != "Out"]
    removed = st.multiselect("Players who are out or not playing (removed from the simulation)", sorted(base["name"]), key="sgp_removed",
                             help="The simulation starts from the players our model and FantasyPoints both expect to play (players ruled Out are already left out, and their teammates' projections already reflect that). "
                                  "Remove anyone you learn is inactive later: his line leaves the game, but his targets and carries are NOT handed to teammates, who keep their projections, so their legs run a little low. "
                                  "If a team lists two quarterbacks (a doubtful starter and his backup), remove the one who is not playing: only the one projected for more attempts gets the team's passing line.")
    try:
        res, h, a = _simulate(season, week, away, home, fp_hash, tuple(sorted(removed)), N_SIMS)
    except Exception as exc:  # noqa: BLE001
        st.error(f"The simulator could not run for this game: {exc}")
        return
    players = {p.name for p in h.players + a.players}
    teams = {home, away}
    pts_h, pts_a = res.team[home]["points"], res.team[away]["points"]
    st.caption(f"{pick} · spread {g['spread']} · total {g['total']:g} · the simulation averages {home} {pts_h.mean():.1f}, {away} {pts_a.mean():.1f} "
               f"(P({home} wins) {np.mean(pts_h > pts_a):.0%}).")

    cand = pull_candidates(table)
    rows_key = "sgp_rows"
    if rows_key not in st.session_state:
        st.session_state[rows_key] = pd.DataFrame([{"Who": None, "Market": None, "Side": "Over", "Line": None, "Odds": None, "Opposite odds": None}], columns=COLS)
    if not cand.empty:
        with st.expander(f"Add legs from the lines you pulled ({len(cand)} lines)", expanded=False):
            opts = {f"{r.player} {r.market} {r.line:g} (book over {r.p_over:.0%})": r for r in cand.itertuples()}
            chosen = st.multiselect("Lines", list(opts), key="sgp_pick")
            side = st.radio("Side", ["Over", "Under"], horizontal=True, key="sgp_pick_side")
            if st.button("Add these legs", key="sgp_add") and chosen:
                new = []
                names = {normalize_name(p): p for p in players}
                for c in chosen:
                    r = opts[c]
                    nm = names.get(normalize_name(r.player))
                    if nm is None:
                        continue
                    p = r.p_over if side == "Over" else 1 - r.p_over
                    odds = round(S.to_american(p))                      # the no-vig price of the side; the pull does not keep the book's own two prices
                    new.append({"Who": nm, "Market": [k for k, v in MARKETS.items() if v == PULL_MARKETS[r.market]][0], "Side": side, "Line": float(r.line), "Odds": float(odds), "Opposite odds": None})
                cur = st.session_state[rows_key]
                cur = cur[cur["Who"].notna()]
                st.session_state[rows_key] = pd.concat([cur, pd.DataFrame(new, columns=COLS)], ignore_index=True)
                st.rerun()

    st.subheader("Legs")
    st.caption("One row per leg. Odds = the American price of the side you are betting; Opposite odds = the other side's price (both give the book's no-vig probability; one alone assumes "
               f"{(ASSUMED_VIG - 1):.1%} vig; none falls back on the simulator's own number, which runs low on overs, so give odds for every leg). For spread: Who = the team, Line = its handicap (-3.5 or +7). Game total: Who can be blank.")
    who_opts = sorted(players) + sorted(teams) + ["Game"]
    edited = st.data_editor(
        st.session_state[rows_key], num_rows="dynamic", hide_index=True, width="stretch", key=f"sgp_editor_{st.session_state.get('sgp_editor_v', 0)}",
        column_config={"Who": st.column_config.SelectboxColumn("Who", options=who_opts), "Market": st.column_config.SelectboxColumn("Market", options=list(MARKETS)),
                       "Side": st.column_config.SelectboxColumn("Side", options=SIDES), "Line": st.column_config.NumberColumn("Line", format="%g"),
                       "Odds": st.column_config.NumberColumn("Odds", format="%+d", help="American odds of the side you bet, e.g. -115 or +120"),
                       "Opposite odds": st.column_config.NumberColumn("Opposite odds", format="%+d")})
    rows = edited[edited["Market"].notna()].reset_index(drop=True)
    rows.loc[rows["Market"] == "Game total", "Who"] = rows.loc[rows["Market"] == "Game total", "Who"].fillna("Game")
    legs, errors = legs_from_rows(rows, players, teams)
    for e in errors:
        st.warning(e)
    if not legs:
        st.info("Add at least one leg to price a parlay.")
    else:
        mode = "market" if any(l.p is not None for l in legs) else "own"
        out = S.price(res, legs, mode=mode)
        missing = [l.label() for l in legs if l.p is None]
        if mode == "market" and missing:
            st.warning("No odds for: " + "; ".join(missing) + ". Those legs use the simulator's own probability, which ran low on overs in testing (it said 43-45% where 47-49% went over): "
                       "enter their odds for a clean price.")
        sharp = disagreements(legs, out["sim_leg_prob"])
        if sharp:
            st.warning("The simulator and the book disagree sharply on: " + "; ".join(sharp) + ". The simulator's picture of that player is off (a projection far from the line, a player who is not playing, "
                       "or a second quarterback listed for the team), so how that leg moves with the others is unreliable: check his projection or remove the player who is not playing.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Chance all legs hit", f"{out['joint']:.1%}", help=f"With each leg pinned to the book's probability where you gave odds; the simulator's own number for the rest. Simulation noise is about +/-{100 * out['se']:.1f} points.")
        c2.metric("Fair odds", _fmt_odds(out["fair_american"]), help="What the parlay should pay with no margin: 1 / the chance above.")
        c3.metric("If legs were independent", f"{out['independent']:.1%}", help="The product of the legs' probabilities: what a book that ignores correlation would use.")
        c4.metric("Correlation lift", f"{out['lift']:.2f}x", help="Chance all hit over the independent product. Above 1 the legs help each other; below 1 they work against each other.")
        offered = st.number_input("The book's price for this parlay (American, optional)", value=0, step=5, key="sgp_offer",
                                  help="Type the odds DraftKings or FanDuel shows for the same legs, e.g. +650.")
        if offered:
            ev = S.ev_at(out["joint"], float(offered))
            be = S.to_prob_american(float(offered))
            st.markdown(f"**At {offered:+d}**: it needs a {be:.1%} chance to break even; the simulator says {out['joint']:.1%}, so the expected return is **{ev:+.1%}** per $1. "
                        + ("The book is paying more than the fair price." if ev > 0 else "The book is paying less than the fair price."))
        tab = pd.DataFrame({"Leg": out["legs"], "Book prob": [None if l.p is None else l.p for l in legs], "Simulator prob": out["sim_leg_prob"]})
        st.dataframe(tab, hide_index=True, width="stretch", column_config={"Book prob": st.column_config.NumberColumn(format="percent"), "Simulator prob": st.column_config.NumberColumn(format="percent")})
        if len(legs) > 1:
            cm = pd.DataFrame(out["corr"], index=out["legs"], columns=out["legs"])
            st.caption("Correlation between the legs' outcomes in the simulation (phi):")
            st.dataframe(cm.round(2), width="stretch")
        if mode == "own":
            st.caption("No odds were entered, so these are the simulator's own probabilities. They ran about 4 points low on overs in testing (`matchup_model/sim_report.md`): enter the book's odds to pin each leg to the market.")
        if any(float(l.line).is_integer() for l in legs if l.market not in ("any_td", "moneyline")):
            st.caption("A whole-number line can push; a push counts as a miss here, so the true chance is slightly higher.")
        _render_past(legs, out["joint"], season, week, away, home)
        _render_picks(res, h, a, legs, cand, edited, season, week, away, home, rows_key)

    with st.expander("Correlation map for this game"):
        cm_tab = corr_map(res, h, a)
        if cm_tab.empty:
            st.caption("Not enough projected players to map.")
        else:
            st.dataframe(cm_tab, hide_index=True, width="stretch", column_config={"Phi": st.column_config.NumberColumn(format="%+.2f"), "Both hit": st.column_config.NumberColumn(format="percent"),
                                                                                  "If independent": st.column_config.NumberColumn(format="percent")})
            st.caption("The game's main player legs, each set at the simulator's own median so it is about a coin flip: the most positively and most negatively associated pairs. Phi is the correlation "
                       "between the two outcomes (0 = independent). These are not book lines: use them to see which legs travel together, then price the real lines above.")

    if not cand.empty:
        with st.expander("Most correlated pairs among the pulled lines"):
            ideas = _idea_table(res, cand, players)
            if ideas.empty:
                st.caption("No pair clears the filters.")
            else:
                st.dataframe(ideas, hide_index=True, width="stretch", column_config={"Joint": st.column_config.NumberColumn(format="percent"),
                                                                                    "If independent": st.column_config.NumberColumn(format="percent"),
                                                                                    "Lift": st.column_config.NumberColumn(format="%.2fx"), "Fair odds": st.column_config.NumberColumn(format="%+d")})
                st.caption("Two-leg parlays among the lines you pulled, each leg at the book's probability (30%-72% legs only, joint at least 8%), ranked by how much more often the "
                           "pair hits than independence says. A high lift is only worth something if the book's parlay price does not already charge for it: compare their offer with the fair odds.")
