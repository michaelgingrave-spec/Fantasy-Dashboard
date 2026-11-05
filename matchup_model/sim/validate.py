"""Does the simulator price parlays better than treating the legs as independent? Tested on lines it was not fitted to.

    python -m matchup_model.sim.validate 2025 2026        # writes matchup_model/sim_report.md

For every past game with posted prop lines the simulator is run from the walk-forward projections (nothing from the game itself), and then:

  1. MARGINAL: its probability that each leg goes over, against the book's no-vig probability and what happened (Brier score, calibration by side).
  2. JOINT, the number that matters for a same-game parlay: every pair of lines in the game is priced two ways -- independently from the books' no-vig probabilities, and with
     the simulator's association (single legs pinned to the books' numbers, only the odds ratio between them taken from the simulator; see `sgp`) -- and each is scored on
     what actually happened to the pair (log score of the realised 2x2 cell). A positive gain means the simulator's dependence carries information the books' single-leg
     prices do not. Results are by pair type and clustered by game for the confidence intervals.
  3. TRIPLES: three-leg parlays hitting together, predicted against realised frequency.
"""
from __future__ import annotations

import itertools
import sys

import numpy as np
import pandas as pd

from dfs.names import normalize_name
from matchup_model.sim import engine as E
from matchup_model.sim import inputs as I
from matchup_model.sim.calibrate import roles
from matchup_model.sim.sgp import odds_ratio_fit  # noqa: F401  (also re-exported here for the tests)

MARKET_STAT = {"rec yds": "rec_yds", "receptions": "rec", "rush yds": "rush_yds", "pass yds": "pass_yds", "pass TD": "pass_td"}


def lines_2025() -> pd.DataFrame:
    """DraftKings closing lines (weeks 5-18): week, away, home, player nk, market, line, p_over (no vig)."""
    from dfs import historical_lines as HL
    from matchup_model.sim.sgp import no_vig
    L = HL.load(2025)
    L = L[(L.book == "draftkings") & L.over_price.notna() & L.under_price.notna()].copy()
    L["p_over"] = [no_vig(o, u) for o, u in zip(L.over_price, L.under_price)]
    L["nk"] = L.player.map(normalize_name)
    L["dist"] = (L.p_over - 0.5).abs()
    L = L.sort_values("dist").drop_duplicates(["season", "week", "event_id", "nk", "market"])
    return L.rename(columns={"point": "line"})[["season", "week", "nk", "market", "line", "p_over"]]


def lines_2026() -> pd.DataFrame:
    """The 2026 lines as logged (consensus number, the book's no-vig probability of the side we leaned, turned into P(over))."""
    from dfs import bets
    from matchup_model.opp import data as D
    lh = bets.load_line_history()
    lh = lh[(lh.season == 2026) & lh.market.isin(MARKET_STAT) & lh["book %"].notna()].drop_duplicates(["week", "player", "market"]).copy()
    lh["p_over"] = np.where(lh.lean == "OVER", lh["book %"] / 100.0, 1 - lh["book %"] / 100.0)
    lh["nk"] = lh.player.map(normalize_name)
    return lh[["season", "week", "nk", "market", "line", "p_over"]]


RECV_MARKETS = ("rec yds", "receptions")
CLASS_PIT_PAIRS = {
    "QB-recv": [("QB.pass_yds", f"{r}.rec_yds") for r in ("WR1", "WR2", "WR3", "TE1", "RB1")],
    "QBTD-recv": [("QB.pass_td", f"{r}.rec_yds") for r in ("WR1", "WR2", "WR3", "TE1", "RB1")],
    "QB-rush": [("QB.pass_yds", "RB1.rush_yds")],
    "recv-recv": [("WR1.rec_yds", "WR2.rec_yds"), ("WR1.rec_yds", "TE1.rec_yds"), ("WR2.rec_yds", "WR3.rec_yds"), ("WR1.rec_yds", "WR3.rec_yds"), ("WR2.rec_yds", "TE1.rec_yds")],
    "rush-recv": [("RB1.rush_yds", f"{r}.rec_yds") for r in ("WR1", "WR2", "TE1")],
    "rush-rush": [("RB1.rush_yds", "RB2.rush_yds")],
}


def pair_class(li: dict, lj: dict, same: bool) -> str | None:
    """The class of a two-leg pair for the simple constant-correlation control: one correlation for every pair of the same kind (None = treated as independent)."""
    if not same:
        return None
    qa, qb = li["role"] == "QB", lj["role"] == "QB"
    ma, mb = li["market"], lj["market"]
    if qa != qb:
        q, o = (li, lj) if qa else (lj, li)
        if o["market"] in RECV_MARKETS:
            return "QB-recv" if q["market"] == "pass yds" else "QBTD-recv" if q["market"] == "pass TD" else None
        if o["market"] == "rush yds" and q["market"] == "pass yds":
            return "QB-rush"
        return None
    if ma in RECV_MARKETS and mb in RECV_MARKETS:
        return "recv-recv"
    if (ma == "rush yds") != (mb == "rush yds") and (ma in RECV_MARKETS or mb in RECV_MARKETS):
        return "rush-recv"
    if ma == "rush yds" and mb == "rush yds":
        return "rush-rush"
    return None


def class_rhos(player_pit: pd.DataFrame) -> dict:
    """Gaussian-copula correlation for each pair class, from the real PITs of past games (calibrate.diagnose's player frame): pooled rank correlation, converted to a normal one."""
    wide = player_pit.assign(col=player_pit.role + "." + player_pit.stat).pivot_table(index=["season", "week", "team"], columns="col", values="pit")
    out = {}
    for cls, prs in CLASS_PIT_PAIRS.items():
        xs, ys = [], []
        for a, b in prs:
            if a in wide and b in wide:
                d = wide[[a, b]].dropna()
                xs += list(d[a]); ys += list(d[b])
        rs = float(np.corrcoef(xs, ys)[0, 1]) if len(xs) > 50 else 0.0
        out[cls] = 2 * np.sin(np.pi * rs / 6)
    return out


def add_control(P: pd.DataFrame, rhos: dict) -> pd.DataFrame:
    """Scores every pair under the control (a Gaussian copula with one correlation per pair class, the books' single-leg numbers as margins): column ll_cls."""
    from scipy.stats import multivariate_normal, norm
    P = P.copy()
    q11 = (P.a * P.b).to_numpy().copy()
    for cls, rho in rhos.items():
        m = (P.cls == cls).to_numpy()
        if m.any():
            pts = np.column_stack([norm.ppf(P.a[m]), norm.ppf(P.b[m])])
            q11[m] = multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]]).cdf(pts)
    a, b = P.a.to_numpy().copy(), P.b.to_numpy().copy()
    q11 = np.clip(q11, np.maximum(0, a + b - 1) + 1e-9, np.minimum(a, b) - 1e-9)
    cell = np.select([(P.ri == 1) & (P.rj == 1), (P.ri == 1) & (P.rj == 0), (P.ri == 0) & (P.rj == 1)], [q11, a - q11, b - q11], default=1 - a - b + q11)
    P["ll_cls"] = -np.log(np.clip(cell, 1e-9, None))
    return P


def run(recs: list[dict], lines: pd.DataFrame, params: E.SimParams | None = None, n: int = 8000, seed: int = 7, max_pairs_per_game: int = 700) -> dict:
    """Marginal and pairwise results for every record that has lines. Returns {'legs': DataFrame, 'pairs': DataFrame, 'triples': DataFrame}."""
    rng = np.random.default_rng(seed)
    leg_rows, pair_rows, tri_rows = [], [], []
    by_week = {k: g for k, g in lines.groupby(["season", "week"])}
    for gi, rec in enumerate(recs):
        gw = by_week.get((rec["season"], rec["week"]))
        if gw is None:
            continue
        info = rec["info"]
        in_game = set(rec["actual"]["nk"]) | {normalize_name(n) for n in rec["table"]["name"]}
        g = gw[gw["nk"].isin(in_game)]
        if g.empty:
            continue
        h, a = I.specs_from_table(rec["table"], rec["away"], rec["home"], info["away_implied"], info["home_implied"])
        res = E.simulate(h, a, n=n, seed=seed + gi, params=params)
        name_of = {normalize_name(p.name): p.name for p in h.players + a.players}
        rl = {}
        for spec in (h, a):
            for role, nm in roles(spec).items():
                rl[nm] = (role, spec.abbr)
        act = rec["actual"].set_index("nk")
        legs = []
        for r in g.itertuples():
            nm = name_of.get(r.nk)
            if nm is None or r.nk not in act.index:
                continue
            stat = MARKET_STAT[r.market]
            if stat not in res.stats[nm]:
                continue
            x = res.stats[nm][stat].astype(float)
            actual = float(act.loc[r.nk, {"rec_yds": "rec_yds", "rec": "rec", "rush_yds": "rush_yds", "pass_yds": "pass_yds", "pass_td": "pass_td"}[stat]])
            ev = x > r.line
            role, tm = rl.get(nm, ("other", res.team_of[nm]))
            legs.append({"name": nm, "market": r.market, "line": r.line, "p_mkt": float(np.clip(r.p_over, 0.02, 0.98)), "p_sim": float(ev.mean()), "p_push": float((x == r.line).mean()),
                         "over": float(actual > r.line), "push": actual == r.line, "ev": ev, "role": role, "team": tm})
        if not legs:
            continue
        for k, l in enumerate(legs):
            leg_rows.append({"season": rec["season"], "week": rec["week"], "game": gi, "market": l["market"], "role": l["role"], "p_mkt": l["p_mkt"], "p_sim": l["p_sim"], "over": l["over"], "push": l["push"]})
        A = np.array([l["ev"] for l in legs], dtype=np.float64)                  # (L, n)
        J = A @ A.T / n                                                          # P(both over)
        pm = A.mean(1)
        idx = [(i, j) for i, j in itertools.combinations(range(len(legs)), 2) if legs[i]["name"] != legs[j]["name"] and not (legs[i]["push"] or legs[j]["push"])]
        if len(idx) > max_pairs_per_game:
            idx = [idx[k] for k in rng.choice(len(idx), max_pairs_per_game, replace=False)]
        for i, j in idx:
            li, lj = legs[i], legs[j]
            p11, p10, p01 = J[i, j], pm[i] - J[i, j], pm[j] - J[i, j]
            p00 = 1 - p11 - p10 - p01
            a_, b_ = li["p_mkt"], lj["p_mkt"]
            q11 = odds_ratio_fit(p11, p10, p01, p00, a_, b_, n=n)
            cells_sim = {(1, 1): q11, (1, 0): a_ - q11, (0, 1): b_ - q11, (0, 0): 1 - a_ - b_ + q11}
            cells_ind = {(1, 1): a_ * b_, (1, 0): a_ * (1 - b_), (0, 1): (1 - a_) * b_, (0, 0): (1 - a_) * (1 - b_)}
            real = (int(li["over"]), int(lj["over"]))
            same = li["team"] == lj["team"]
            pair_rows.append({"game": gi, "season": rec["season"], "week": rec["week"], "rel": "same" if same else "opp", "ra": li["role"], "rb": lj["role"], "ma": li["market"], "mb": lj["market"],
                              "a": a_, "b": b_, "ri": int(li["over"]), "rj": int(lj["over"]), "cls": pair_class(li, lj, same),
                              "ll_ind": -np.log(max(cells_ind[real], 1e-9)), "ll_sim": -np.log(max(cells_sim[real], 1e-9)), "p_oo_ind": cells_ind[(1, 1)], "p_oo_sim": cells_sim[(1, 1)],
                              "real_oo": float(real == (1, 1)), "real_uu": float(real == (0, 0)), "p_uu_ind": cells_ind[(0, 0)], "p_uu_sim": cells_sim[(0, 0)]})
        # triples: a random sample of three different players' lines
        trip = [t for t in itertools.combinations(range(len(legs)), 3) if len({legs[x]["name"] for x in t}) == 3 and not any(legs[x]["push"] for x in t)]
        if trip:
            for t in [trip[k] for k in rng.choice(len(trip), min(len(trip), 60), replace=False)]:
                ev = [legs[x]["ev"] for x in t]
                pm_ = [legs[x]["p_mkt"] for x in t]
                from matchup_model.sim.sgp import cell_table, ipf_joint
                tab = cell_table(ev, np.full(n, 1.0 / n), eps=0.5 / n)
                fit = ipf_joint(tab, pm_)
                tri_rows.append({"game": gi, "p_ooo_ind": float(np.prod(pm_)), "p_ooo_sim": float(fit[(1, 1, 1)]), "real_ooo": float(all(legs[x]["over"] for x in t)),
                                 "p_uuu_ind": float(np.prod([1 - p for p in pm_])), "p_uuu_sim": float(fit[(0, 0, 0)]), "real_uuu": float(not any(legs[x]["over"] for x in t))})
    return {"legs": pd.DataFrame(leg_rows), "pairs": pd.DataFrame(pair_rows), "triples": pd.DataFrame(tri_rows)}


def boot_mean(x: np.ndarray, groups: np.ndarray, reps: int = 600, seed: int = 3) -> tuple[float, float, float]:
    """Mean of x with a 95% interval from resampling whole games."""
    rng = np.random.default_rng(seed)
    u, inv = np.unique(groups, return_inverse=True)
    s = np.bincount(inv, weights=x, minlength=len(u))
    c = np.bincount(inv, minlength=len(u)).astype(float)
    vals = []
    for _ in range(reps):
        i = rng.integers(0, len(u), len(u))
        vals.append(s[i].sum() / c[i].sum())
    return float(x.mean()), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def pair_summary(P: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Log-score gain of the simulator's joint over independence (positive = better), per pair, with a game-clustered interval; plus how the predicted and realised co-hit rates compare."""
    rows = []
    groups = P.groupby(by) if by else [("all", P)]
    for key, d in groups:
        if len(d) < 60:
            continue
        gain = (d.ll_ind - d.ll_sim).to_numpy()
        m, lo, hi = boot_mean(gain, d.game.to_numpy())
        rows.append({"pairs": key if not isinstance(key, tuple) else " / ".join(map(str, key)), "n": len(d), "gain (log score)": round(m, 4), "95% lo": round(lo, 4), "95% hi": round(hi, 4),
                     "over+over: indep / sim / real": f"{d.p_oo_ind.mean():.3f} / {d.p_oo_sim.mean():.3f} / {d.real_oo.mean():.3f}",
                     "under+under: indep / sim / real": f"{d.p_uu_ind.mean():.3f} / {d.p_uu_sim.mean():.3f} / {d.real_uu.mean():.3f}"})
    return pd.DataFrame(rows)


def control_summary(P: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Gain over independence for the simulator and for the constant-correlation control, and the simulator's gain over the control (positive = the simulator is better)."""
    rows = []
    groups = P.groupby(by) if by else [("all", P)]
    for key, d in groups:
        if len(d) < 60 or "ll_cls" not in d:
            continue
        row = {"pairs": key if not isinstance(key, tuple) else " / ".join(map(str, key)), "n": len(d)}
        for lab, x in (("simulator vs independent", d.ll_ind - d.ll_sim), ("control vs independent", d.ll_ind - d.ll_cls), ("simulator vs control", d.ll_cls - d.ll_sim)):
            m, lo, hi = boot_mean(x.to_numpy(), d.game.to_numpy())
            row[lab] = f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}]"
        rows.append(row)
    return pd.DataFrame(rows)


def marginal_summary(L: pd.DataFrame) -> pd.DataFrame:
    """Brier score of P(over) from the market and from the simulator, and the bias of each by side of the market's number."""
    rows = []
    for key, d in [("all", L)] + list(L.groupby("market")):
        d = d[~d.push]
        if len(d) < 50:
            continue
        rows.append({"market": key, "n": len(d), "went over": round(d.over.mean(), 3), "market said": round(d.p_mkt.mean(), 3), "sim said": round(d.p_sim.mean(), 3),
                     "Brier market": round(((d.p_mkt - d.over) ** 2).mean(), 4), "Brier sim": round(((d.p_sim - d.over) ** 2).mean(), 4),
                     "Brier market/sim 50-50": round((((d.p_mkt + d.p_sim) / 2 - d.over) ** 2).mean(), 4)})
    return pd.DataFrame(rows)


def triple_summary(T: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lab, a, b, r in (("OVER x3", "p_ooo_ind", "p_ooo_sim", "real_ooo"), ("UNDER x3", "p_uuu_ind", "p_uuu_sim", "real_uuu")):
        d = T
        rows.append({"parlay": lab, "n": len(d), "independent": round(d[a].mean(), 4), "simulated": round(d[b].mean(), 4), "realised": round(d[r].mean(), 4),
                     "ratio real/indep": round(d[r].mean() / d[a].mean(), 2), "ratio sim/indep": round(d[b].mean() / d[a].mean(), 2)})
    return pd.DataFrame(rows)


def _md(df: pd.DataFrame) -> list[str]:
    if df is None or df.empty:
        return ["(nothing to show)"]
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r.tolist()) + " |")
    return out


INTRO = [
    "# Game simulator: how well does it price parlays?", "",
    "`matchup_model.sim` plays a game 20,000 times from the model's own projections (means) and a small set of noise parameters, and `sgp.price` turns the paths into a parlay's "
    "chance and fair odds. This report is the test of it on games and lines it was **not** fitted to. The noise parameters were tuned on 2023-24 box scores only "
    "(`calibrate.py`, a coordinate search on the match to the real correlations between players' stats and to how spread out outcomes are); 2025 and 2026 are untouched by "
    "that tuning, **except** that the two input corrections of 2026-11-05 (finding 1) were chosen after comparing them with the first setting on these same seasons, so read their gain "
    "as partly selected. Every game is simulated from the projections the app would have shown beforehand (walk-forward; `history.py`).", "",
    "**How a parlay is priced.** Each leg is pinned to the book's no-vig probability (the books' own single-leg prices), and only the *association* between legs, the odds "
    "ratio of the simulator's 2x2 table, is taken from the simulator (`sgp.price`, `odds_ratio_fit`). The test below asks whether that association tells us anything the "
    "single-leg prices do not: every pair of lines in each game is priced independently and with the simulator's association, and each is scored on what actually "
    "happened to the pair (log score of the realised cell). A positive gain means the simulator's dependence is informative. Intervals resample whole games.", ""]


FINDINGS = ['## Findings (the run of 2026-11-05; regenerate and re-read after any change to the parameters)',
    '', "1. **Two input corrections since the first run (2026-11-04), and what they changed.** The first run scaled every player down to the volume a team actually has (`vol_pass` .935, `vol_rush` .9175), which put each role player 6-9% under his own projection, and gave the quarterback the sum of his receivers' lines, which put his passing yards about 10% over his own projection. Now every player's mean is his projection, and a starting quarterback's line is set to his (a quarterback projected for under 60% of the team's passing is a backup, weighted by his chance to play: if he is in the simulation he throws the team's passes). On the same held-out lines, for role players, the simulator's own chance of the over improved in both seasons and in every market: Brier .2625 -> .2586 in 2025 (-.0038 [-.0051, -.0027]) and .2558 -> .2526 in 2026 (-.0032 [-.0049, -.0015]); the average chance it gives an over (all lines) rose from 39% to 43% (2025) and from 42% to 45% (2026), against 47.0% and 48.6% that happened. The association between legs did not change (same-team pairs with a quarterback, 2026: +0.0102 -> +0.0105 per pair; every old-minus-new difference in the pair gain is within +/-0.0004 with intervals including 0). Quarterback passing-yards lines did not improve (206 lines in 2026, +0.0040 [-0.0183, +0.0281]): the pin makes his average his projection (it ran 10% over), it is not shown to help prices. Caveats: the pinned-means setting was picked after comparing it with the first setting on these same 2025-26 lines; the 2023-24 seasons, which did not pick it, agree (bias of the simulated mean against what happened: WR1 receiving yards -16.6% -> -10.8%, lead-back carries -9.9% -> -1.8%, quarterback yards +9.7% -> +1.7%; the spread of outcomes is mixed: the standardised-residual variance (ideal 1) is closer to 1 in 8 of the 16 key rows, unchanged in 4 and further in 4 (quarterback yards 1.16 -> 1.31, so his simulated spread is now about 13% too narrow, it was 7%; lead-back rushing yards .93 -> .81, tight end receiving yards .94 -> .87 and WR1 yards 1.03 -> .94, spreads a few per cent too wide), and the variance term of the tuning loss as a whole improved, .149 -> .101); the noise parameters were NOT re-tuned. Cost: the simulated TEAM volumes (team pass attempts, yards, carries) now run 6-13% above history on 2023-24, since they are the sum of the players' projections; the screen does not use them.",
    "2. **Where it helps: a quarterback and his receivers.** On 2026 lines (121 games) the 7,906 same-team pairs that include a quarterback gain +0.0105 log-score per pair over independence [+0.0074, +0.0141]. For pass yards with a receiver's yards the gain is +0.0163 [+0.0108, +0.0219] (2,408 pairs) and with receptions +0.0116 [+0.0069, +0.0162]; both legs went over together 30% of the time against 25% if independent (the simulator said 28%).",
    "3. **A simple correlation table does almost as well for those pairs.** One fixed correlation per kind of pair (estimated on the same 2023-24 games) gains +0.0265 per quarterback-receiver pair against the simulator's +0.0322 (simulator better by +0.0058 [+0.0013, +0.0103]); the two tie elsewhere (running-back pairs: simulator +0.0017 [-0.0005, +0.0041] over the table, 209 pairs). The simulator's edge over the table is consistency across any set of legs (touchdowns, totals, three-leg combinations) and context (a high total, a heavy favourite), which these tests cannot measure.",
    "4. **Nowhere else is there anything to gain, and it costs almost nothing.** Teammate receivers' yards and receptions are close to uncorrelated in real games (about 0 either way), and so are opposing players': the simulator's gain over independence there is -0.0001 to -0.0006 a pair.",
    '5. **2025 cannot test the quarterback**: no pass yards or pass TD lines were pulled for it, so its null result is the skill-player pairs, whose true correlation is about zero.',
    "6. **The simulator's own single-leg probabilities are better than they were but still not good enough to use alone, so pin the legs to the book's odds.** It says the over hits 43% (2025) and 45% (2026) when the books said 49.6% and 49.5% and 47.0% and 48.6% went over (before the corrections 39% and 42%); Brier .2592 and .2550 against the market's .2476 and .2484 on all lines (before: .2625 and .2580), and on role players .2586 and .2526 against .2473 and .2483 (the book is ahead by +.0113 [+.0063, +.0161] in 2025 and +.0043 [+.0001, +.0083] in 2026; on 2026 receiving-yards lines they tie, .2496 against .2500). It is still low on overs because the model's level sits under the books' lines. The screen uses the simulator's number only for a leg with no odds, and says so.",
    "7. **The books' no-vig prices lean towards the under.** Overs hit less often than priced (47.0% against 49.6% in 2025; 48.6% against 49.5% in 2026), so three-over parlays hit 0.82x as often as the product of their no-vig probabilities in 2025 and 1.02x in 2026, three-under parlays 1.17x and 1.08x. A market-pinned price inherits the books' numbers and therefore this lean; it is a reason to prefer under legs, and not something the simulator corrects.",
    "8. **Touchdown, spread and total legs have no historical lines here**, so they are checked only through the correlation tests above (team points with a quarterback's yards .30 real against .38 simulated in 2025 and .35 against .38 in 2026; his yards with his touchdowns .44 against .45 in 2025 and .36 against .45 in 2026).",
    "9. **Edge against the books' own same-game-parlay prices is untested**: those prices are not available through the Odds API. The screen takes the price the book offers and shows the expected return against the fair price; logging offers and results is the way to test it.",
    "10. **A leg the simulator thinks impossible has no association to give, and the pricing now says so.** One 2026 game had a depth receiver projected for 1 yard against a 15-yard line (0.04% in the simulator); an odds ratio built from three paths is noise, and it alone moved the 2026 all-pairs gain from +0.0013 to 0.0000 a pair. `odds_ratio_fit` and `price` now give every cell half a path first (the usual continuity correction), and the screen warns when the simulator and the book are 30 points apart or the simulator gives a leg under 2% or over 98%.",
    "11. **Limits the screen carries.** If a team lists two quarterbacks (a doubtful starter and a backup, both projected for a full game) only the one with more projected attempts gets the passing line, so remove the other. A late scratch's targets and carries are not handed to his teammates, who keep projections that already reflect the official injury report. A starter who leaves the game early is not modelled.",
    '']


def main(argv: list[str]) -> None:
    from matchup_model.sim import calibrate as C
    from matchup_model.sim import history as H
    seasons = [int(x) for x in argv if not x.startswith("--")] or [2025, 2026]
    write = "--report" in argv
    pd.set_option("display.width", 230)
    pd.set_option("display.max_columns", 30)
    P = E.SimParams()
    train = H.load([2023, 2024])[::2]
    rhos = class_rhos(C.diagnose(train, params=P, n=1500)["player"])
    print("control correlations (from 2023-24 PITs):", {k: round(v, 3) for k, v in rhos.items()}, flush=True)
    rep = list(INTRO) + list(FINDINGS)
    rep += ["## Parameters", "", ", ".join(f"`{k}` {v:g}" for k, v in P.__dict__.items()), ""]
    for s in seasons:
        recs = H.load([s], weeks=range(1, 19))
        L = lines_2025() if s == 2025 else lines_2026()
        L = L[L.season == s]
        print(s, len(recs), "games", len(L), "lines", flush=True)
        r = run(recs, L)
        r["pairs"]["kind"] = [("with a QB" if "QB" in (a, b) else "skill players") for a, b in zip(r["pairs"].ra, r["pairs"].rb)]
        r["pairs"]["mm"] = [" + ".join(sorted([a, b])) for a, b in zip(r["pairs"].ma, r["pairs"].mb)]
        r["pairs"] = add_control(r["pairs"], rhos)
        tabs = {"control": control_summary(r["pairs"], ["cls"]), "control_all": control_summary(r["pairs"]), "marginal": marginal_summary(r["legs"]), "all": pair_summary(r["pairs"]), "rel": pair_summary(r["pairs"], ["rel"]), "kind": pair_summary(r["pairs"], ["kind", "rel"]),
                "mm": pair_summary(r["pairs"], ["mm"]), "triples": triple_summary(r["triples"])}
        d = C.diagnose(recs, params=P, n=3000)
        print(f"\n=== {s}: marginal ===\n", tabs["marginal"].to_string(index=False))
        print(f"\n=== {s}: pairs, all ===\n", tabs["all"].to_string(index=False))
        print(f"\n=== {s}: pairs by relation ===\n", tabs["rel"].to_string(index=False))
        print(f"\n=== {s}: pairs by kind and relation ===\n", tabs["kind"].to_string(index=False))
        print(f"\n=== {s}: pairs by market pair ===\n", tabs["mm"].to_string(index=False))
        print(f"\n=== {s}: against a simple correlation-table control ===\n", tabs["control_all"].to_string(index=False), "\n", tabs["control"].to_string(index=False))
        print(f"\n=== {s}: triples ===\n", tabs["triples"].to_string(index=False), flush=True)
        rep += [f"## {s} ({len(recs)} games, {len(L)} lines)", "", "### Single legs: the simulator's chance of the over against the book's and against what happened", ""] + _md(tabs["marginal"]) + [""]
        rep += ["### Pairs: log-score gain of the simulator's association over independence (positive = better)", ""] + _md(tabs["all"]) + [""]
        rep += ["By teammates / opponents:", ""] + _md(tabs["rel"]) + ["", "By pair kind:", ""] + _md(tabs["kind"]) + ["", "By market pair:", ""] + _md(tabs["mm"]) + [""]
        rep += ["### Against a simple control: one correlation per pair class", "",
                f"The control is a Gaussian copula with a single correlation for each kind of pair ({', '.join(f'{k} {v:.2f}' for k, v in rhos.items())}), estimated from the same 2023-24 games, "
                "with the books' single-leg numbers as margins. Pairs outside these classes (and opponents) are treated as independent.", ""] + _md(tabs["control_all"]) + [""] + _md(tabs["control"]) + [""]
        rep += ["### Three-leg parlays: the chance they all hit together", ""] + _md(tabs["triples"]) + [""]
        pairs_tab = d["pairs"].copy()
        rep += ["### The correlations the simulator implies against real ones (PIT correlation across games)", ""] + _md(pairs_tab) + [""]
        rep += ["Team score against player stats:", ""] + _md(d["tpairs"]) + [""]
        pl = C.summarize_pit(d["player"], ["role", "stat"])
        rep += ["### Calibration of each player's stat in the simulated distribution (PIT: mean .5, variance .083)", ""] + _md(pl.reset_index()) + [""]
    if write:
        from pathlib import Path
        out = Path(__file__).resolve().parents[1] / "sim_report.md"
        out.write_text(chr(10).join(rep) + chr(10), encoding="utf-8")
        print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1:])
