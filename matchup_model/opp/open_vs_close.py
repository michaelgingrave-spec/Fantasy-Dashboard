"""Do our projections lead the market?  Early-week lines vs the late lines we graded, on the same player-markets.

    python -m matchup_model.opp.open_vs_close        # writes matchup_model/open_vs_close_2026_report.md

Inputs: data/dfs/props/opening_lines.csv (Wednesday 4pm ET snapshots from `python -m dfs.opening_lines`) joined to the graded
rows of data/dfs/props/line_history.csv (weeks 3+ are true closing lines, weeks 1-2 were pulled during the week). Receiving
markets only (rec yds, receptions) -- the only ones the opening pull covered.

Questions: (1) how far do lines move between the two snapshots; (2) does the move go toward OUR projection (slope of the move on
our disagreement with the open line, in outcome-SD units, player-clustered bootstrap); (3) would betting our side at the open
have won more than at the late line (paired, player-clustered); (4) which line is the better forecast of the result.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from dfs.props import DATA, _sigma
from matchup_model.opp.error_signal import MARKET_TO_STAT

OPENING = DATA / "props" / "opening_lines.csv"
HISTORY = DATA / "props" / "line_history.csv"
REPORT = Path(__file__).resolve().parents[1] / "open_vs_close_2026_report.md"
MARKETS = ["rec yds", "receptions"]
BIG = 0.4          # "large disagreement": |our projection - open line| in outcome SDs
B = 2000


def load_joined(season: int = 2026) -> pd.DataFrame:
    """One row per graded (week, player, market) that has both an opening and a late line."""
    o = pd.read_csv(OPENING)
    lh = pd.read_csv(HISTORY)
    lh = lh[(lh.season == season) & lh.market.isin(MARKETS) & lh.actual.notna()
            & lh.result.isin(["win", "loss", "push"])].copy()
    d = lh.merge(o[["week", "player", "market", "line_open", "over_price", "under_price"]],
                 on=["week", "player", "market"], how="inner").drop_duplicates(["week", "player", "market"])
    d["sigma"] = [_sigma(MARKET_TO_STAT[m], x) for m, x in zip(d.market, d.line_open)]
    d["move"] = d.line - d.line_open                               # late minus open, stat units
    d["move_z"] = d.move / d.sigma
    d["edge_open_z"] = (d["our proj"] - d.line_open) / d.sigma     # our disagreement with the open, in SDs
    d["nk"] = d.player.str.lower()
    return d.reset_index(drop=True)


def clustered_slope(x, y, groups, rng, reps: int = B) -> tuple[float, float, float]:
    """OLS slope of y on x with a player-clustered bootstrap 95% CI (all NaN when there are too few rows to fit)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.ptp(x) == 0:
        return float("nan"), float("nan"), float("nan")
    xc, yc = x - x.mean(), y - y.mean()
    b = float((xc * yc).sum() / (xc * xc).sum())
    s = pd.DataFrame({"xy": xc * yc, "xx": xc * xc, "g": np.asarray(groups)}).groupby("g").sum().to_numpy()
    bs = [s[(i := rng.integers(0, len(s), len(s))), 0].sum() / s[i, 1].sum() for _ in range(reps)]
    return b, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def play(proj, line, actual):
    """Bet the side our projection implies against `line`: (won, pushed, bet_over)."""
    proj, line, actual = (np.asarray(a, float) for a in (proj, line, actual))
    over = proj > line
    return np.where(over, actual > line, actual < line), actual == line, over


def paired_gain(won_a, won_b, push_either, groups, rng, reps: int = B) -> tuple[float, float, float, int]:
    """Mean of (won_a - won_b) over rows where neither pushed, player-clustered 95% CI, n."""
    m = ~np.asarray(push_either)
    diff = (np.asarray(won_a, float) - np.asarray(won_b, float))[m]
    g = pd.DataFrame({"d": diff, "g": np.asarray(groups)[m]}).groupby("g").d.agg(["sum", "count"]).to_numpy(float)
    est = g[:, 0].sum() / g[:, 1].sum()
    bs = [g[(i := rng.integers(0, len(g), len(g))), 0].sum() / g[i, 1].sum() for _ in range(reps)]
    return float(est), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), int(m.sum())


def build_report(d: pd.DataFrame) -> str:
    rng = np.random.default_rng(5)
    roi = lambda p: 100 * (p * 100 / 110 - (1 - p))  # noqa: E731 -- ROI at -110 for win rate p
    L = ["# Do our projections lead the market? Open vs late lines, 2026", "",
         f"{len(d)} graded receiving props with both an opening line (Wednesday 4pm ET snapshot) and a late line "
         f"(weeks 3+ closing, weeks 1-2 pulled mid-week); weeks {int(d.week.min())}-{int(d.week.max())}, "
         f"{d.groupby('week').size().to_dict()} rows per week. `our proj` is what the dashboard showed when the line was pulled.", ""]

    L += ["## 1. How far do lines move?", "", "| market | n | moved | mean move | mean abs move | in SD |", "|---|--:|--:|--:|--:|--:|"]
    for mk, g in d.groupby("market"):
        L.append(f"| {mk} | {len(g)} | {100 * (g.move != 0).mean():.0f}% | {g.move.mean():+.2f} "
                 f"(up {100 * (g.move > 0).mean():.0f}% / down {100 * (g.move < 0).mean():.0f}%) | "
                 f"{g.move.abs().mean():.2f} | {g.move_z.abs().mean():.3f} |")

    L += ["", "## 2. Does the market move toward our number?", "",
          "Slope of the open-to-late move on (our projection - open line), both in outcome SDs. 0 = no relation, "
          "1 = the market fully follows us. 95% CI from a player-clustered bootstrap.", "",
          "| rows | n | slope | 95% CI |", "|---|--:|--:|---|"]
    for label, sub in [("all", d), ("rec yds", d[d.market == "rec yds"]), ("receptions", d[d.market == "receptions"]),
                       ("weeks 3+ (true closing lines)", d[d.week >= 3]), ("weeks 1-2 (mid-week lines)", d[d.week <= 2])]:
        b, lo, hi = clustered_slope(sub.edge_open_z, sub.move_z, sub.nk, rng)
        L.append(f"| {label} | {len(sub)} | " + (f"{b:+.3f} | [{lo:+.3f}, {hi:+.3f}] |" if np.isfinite(b) else "n/a | n/a |"))
    big = d[d.edge_open_z.abs() >= BIG]
    toward = (np.sign(big.move) == np.sign(big.edge_open_z)).mean()
    away = (np.sign(big.move) == -np.sign(big.edge_open_z)).mean()
    L += ["", f"Large disagreements (|our projection - open| >= {BIG:g} SD, n={len(big)}): the line moved toward us "
          f"{100 * toward:.0f}% of the time, away from us {100 * away:.0f}%, unchanged {100 * (big.move == 0).mean():.0f}%.", ""]

    L += ["## 3. Our side at the open vs at the late line", "",
          "Bet the side our projection implies against each line, same rows, same projection (win rate, ROI at -110).", "",
          "| line used | bets | win % | OVER share | ROI at -110 |", "|---|--:|--:|--:|--:|"]
    res = {}
    for label, line in (("open", d.line_open), ("late", d.line)):
        w, p, ov = play(d["our proj"], line, d.actual)
        res[label] = (w, p)
        m = ~p
        L.append(f"| {label} | {int(m.sum())} | {100 * w[m].mean():.1f} | {100 * ov[m].mean():.0f}% | {roi(w[m].mean()):+.1f}% |")
    est, lo, hi, n = paired_gain(res["open"][0], res["late"][0], res["open"][1] | res["late"][1], d.nk, rng)
    L += ["", f"Paired difference, open minus late: **{100 * est:+.1f} pts** (95% CI {100 * lo:+.1f} to {100 * hi:+.1f}, n={n}). "
          "A CI that spans zero means betting early is not distinguishable from betting late on this sample.", ""]
    L += ["| week | win % at open | win % at late | n |", "|--:|--:|--:|--:|"]
    for wk, g in d.groupby("week"):
        r = []
        for line in (g.line_open, g.line):
            w, p, _ = play(g["our proj"], line, g.actual)
            r.append(f"{100 * w[~p].mean():.1f}")
        L.append(f"| {int(wk)} | {r[0]} | {r[1]} | {len(g)} |")

    L += ["", "## 4. Which line forecasts the result better?", "",
          "RMSE of (actual - line) in outcome SDs (lower is better); our projection for comparison.", "",
          "| forecast | RMSE | mean error |", "|---|--:|--:|"]
    for label, f in (("open line", d.line_open), ("late line", d.line), ("our projection", d["our proj"])):
        err = (d.actual - f) / d.sigma
        L.append(f"| {label} | {np.sqrt((err ** 2).mean()):.4f} | {err.mean():+.3f} |")
    side = np.where(d["our proj"] > d.line_open, d.over_price, d.under_price)
    L += ["", f"Price check: median best open price on our side {np.nanmedian(side):.0f}; "
          f"{100 * np.nanmean(side >= -110):.0f}% of open prices are -110 or better."]
    return "\n".join(L) + "\n"


def main() -> None:
    d = load_joined()
    txt = build_report(d)
    REPORT.write_text(txt, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(txt + f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
