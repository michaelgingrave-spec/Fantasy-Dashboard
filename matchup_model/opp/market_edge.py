"""The edge meter: does our projection predict the result BEYOND what the market's own price already says?

    python -m matchup_model.opp.market_edge        # writes matchup_model/market_edge_report.md from the graded lines in line_history.csv

For every graded line the book's no-vig probability of the OVER (stored with the line as `book %` for the side we leaned) is the market's forecast.
Logistic regression of 'went over' on logit(market P(over)) and our standardised edge z = (our proj - line) / sigma tells how much of the result
our edge explains once the price is accounted for: a z coefficient of 0 means the model adds nothing to the price, and +0.1 per SD moves a
coin-flip bet's win probability by ~2.5 points -- about what the vig costs at -110 (2.4). Confidence intervals resample whole players (a player
appears in several markets and weeks). Rows are scored by the model version that logged them (`model_ver`; rows from before it existed are v1).

It reads the lines AS LOGGED, so live leans from a new model version build up their own row as weeks are graded; matchup_model.opp.restate
re-scores the same lines under earlier versions when you want a like-for-like comparison.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from dfs.props import _NICE, _sigma

REPORT = Path(__file__).resolve().parents[1] / "market_edge_report.md"
COMP = {label: comp for comp, label in _NICE.items() if comp not in ("pass_att", "rush_att")}     # "rec yds" -> "rec_yds"
MIN_ROWS = 300            # a model version needs this many graded lines before it gets its own row
VIG_POINTS = 2.4          # win-rate points a -110 bet must beat to break even (52.4% vs 50%)


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def prepare(lh: pd.DataFrame) -> pd.DataFrame:
    """One row per graded, non-push line in a market we model, with the market's P(over), our z, and whether it went over."""
    need = {"player", "market", "line", "actual", "our proj", "lean", "book %", "result"}
    if lh is None or lh.empty or not need <= set(lh.columns):
        return pd.DataFrame()
    d = lh[lh.result.isin(["win", "loss"]) & lh.market.isin(COMP) & lh["book %"].notna() & lh.actual.notna()].copy()
    d = d[d.actual != d.line]
    if d.empty:
        return d
    p_lean = d["book %"].astype(float) / 100.0
    d["p_mkt_over"] = np.where(d.lean == "OVER", p_lean, 1 - p_lean)
    d["sigma"] = [_sigma(COMP[m], x) for m, x in zip(d.market, d.line)]
    d["z"] = (d["our proj"] - d.line) / d.sigma
    d["over"] = (d.actual > d.line).astype(float)
    d["ver"] = pd.to_numeric(d["model_ver"], errors="coerce").fillna(1).astype(int) if "model_ver" in d.columns else 1
    return d.reset_index(drop=True)


def _newton(X1: np.ndarray, y: np.ndarray, iters: int = 50) -> np.ndarray:
    beta = np.zeros(X1.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X1 @ beta))
        H = X1.T @ (X1 * (p * (1 - p))[:, None]) + 1e-6 * np.eye(len(beta))
        step = np.linalg.solve(H, X1.T @ (y - p))
        beta += step
        if np.abs(step).max() < 1e-8:
            break
    return beta


def fit_logit(X: np.ndarray, y: np.ndarray, groups, reps: int = 600, seed: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Logistic regression with an intercept; (coefficients, 2.5th, 97.5th percentile) from a bootstrap over whole `groups` (players)."""
    X1 = np.column_stack([np.ones(len(X)), X])
    beta = _newton(X1, y)
    gid = pd.factorize(pd.Series(groups))[0]
    rows = [np.where(gid == g)[0] for g in range(gid.max() + 1)]
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(reps):
        take = np.concatenate([rows[i] for i in rng.integers(0, len(rows), len(rows))])
        try:
            bs.append(_newton(X1[take], y[take], iters=25))
        except np.linalg.LinAlgError:
            continue
    bs = np.array(bs)
    return beta, np.percentile(bs, 2.5, axis=0), np.percentile(bs, 97.5, axis=0)


def edge_vs_market(d: pd.DataFrame, reps: int = 600) -> dict:
    """z coefficient beyond the market (per SD), its interval, and the implied move in P(over) at a coin-flip price (points)."""
    X = np.column_stack([logit(d.p_mkt_over), d.z])
    b, lo, hi = fit_logit(X, d.over.to_numpy(), d.player.to_numpy(), reps=reps)
    move = lambda c: 100 * (1 / (1 + np.exp(-c)) - 0.5)  # noqa: E731
    return {"n": len(d), "over_rate": 100 * float(d.over.mean()), "mkt_over": 100 * float(d.p_mkt_over.mean()),
            "mkt_coef": (b[1], lo[1], hi[1]), "z_coef": (b[2], lo[2], hi[2]), "move": (move(b[2]), move(lo[2]), move(hi[2]))}


def build_report(lh: pd.DataFrame) -> str:
    d = prepare(lh)
    L = ["# The edge meter: does our projection add anything to the market's own price?", "",
         "Logistic regression of 'went over' on the book's no-vig P(over) and our edge in SDs (`z`); player-clustered 95% intervals. "
         f"A +1 SD edge has to move a coin-flip bet's win probability by about {VIG_POINTS:g} points just to pay the vig at -110.", ""]
    if d.empty:
        return "\n".join(L + ["No graded lines with a stored book probability yet."]) + "\n"
    L += ["| model version | graded lines | went over | market said | market-price coefficient | z coefficient (per SD) | +1 SD moves P(over) by |",
          "|---|--:|--:|--:|--:|--:|--:|"]
    for ver, g in d.groupby("ver"):
        if len(g) < MIN_ROWS:
            L.append(f"| v{ver} | {len(g)} | | | | too few lines yet (needs {MIN_ROWS}) | |")
            continue
        r = edge_vs_market(g)
        L.append(f"| v{ver} | {r['n']} | {r['over_rate']:.1f}% | {r['mkt_over']:.1f}% | {r['mkt_coef'][0]:+.2f} [{r['mkt_coef'][1]:+.2f}, {r['mkt_coef'][2]:+.2f}] | "
                 f"{r['z_coef'][0]:+.3f} [{r['z_coef'][1]:+.3f}, {r['z_coef'][2]:+.3f}] | "
                 f"{r['move'][0]:+.1f} pts [{r['move'][1]:+.1f}, {r['move'][2]:+.1f}] |")
    L += ["", "A market-price coefficient below 1 means prices are more extreme than results (the books' favourites win less often than priced); "
          "a z coefficient whose interval spans 0 means the sample cannot yet tell our edge from none.", "",
          "## By market (all versions pooled)", "", "| market | graded lines | went over | z coefficient (per SD) |", "|---|--:|--:|--:|"]
    for mk, g in d.groupby("market"):
        if len(g) < 100:
            continue
        r = edge_vs_market(g, reps=300)
        L.append(f"| {mk} | {len(g)} | {r['over_rate']:.1f}% | {r['z_coef'][0]:+.3f} [{r['z_coef'][1]:+.3f}, {r['z_coef'][2]:+.3f}] |")
    side = d.assign(win=np.where(d.lean == "OVER", d.over, 1 - d.over))
    L += ["", "## Win rate by the side we leaned (as logged)", "", "| side | lines | win | ROI at -110 |", "|---|--:|--:|--:|"]
    for s, g in side.groupby("lean"):
        w = float(g.win.mean())
        L.append(f"| {s} | {len(g)} | {100 * w:.1f}% | {100 * (w * 100 / 110 - (1 - w)):+.1f}% |")
    return "\n".join(L) + "\n"


def main() -> None:
    from dfs import bets
    txt = build_report(bets.load_line_history())
    REPORT.write_text(txt, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(txt + f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
