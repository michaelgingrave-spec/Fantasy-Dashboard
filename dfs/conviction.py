"""Calibrated conviction ranking for prop picks.

The confidence tiers (lean/solid/strong/high) are one z-score scale applied to every
market, and graded results show the model's stated hit probability is not equally
trustworthy everywhere: weeks 1-5 of 2026, solid+ receptions OVERs at a 3.5 line were
claimed at ~62% and won 44%, while the same tier at a <=1.5 line was claimed at ~64% and
won 66%. OVERs are overconfident by ~7 points, UNDERs are close to calibrated.

This corrects the model's own hit probability by how that exact kind of prop (market x
side x line bucket) has actually done, using only previously graded weeks, then ranks by
expected value at the real price. The correction is empirical-Bayes shrunk toward the
model's GLOBAL miss, so thin buckets inherit the overall overconfidence instead of
getting a free pass from the model's own number.

    python -m dfs.conviction 2026      # walk-forward test + report
"""
from __future__ import annotations

import re
import sys
from typing import NamedTuple

import numpy as np
import pandas as pd

K_SHRINK = 25.0              # props of bucket evidence before it outweighs the global miss
P_MIN = 0.55                 # a pick has to be calibrated to at least this to count as a pick
P_CLIP = (0.30, 0.80)
CORE = ("rec yds", "receptions", "rush yds", "pass yds", "pass TD")
TIER_RANK = {"lean": 1, "solid": 2, "strong": 3, "high": 4}

# fixed up front (informed by a look at weeks 1-5 distributions, not tuned on results)
LINE_BINS = {
    "receptions": ([0, 1.75, 2.75, 3.75, 1e9], ["<=1.5", "2.5", "3.5", "4.5+"]),
    "rec yds": ([0, 20, 40, 60, 1e9], ["<=20", "20-40", "40-60", "60+"]),
    "rush yds": ([0, 25, 50, 1e9], ["<=25", "25-50", "50+"]),
    "pass yds": ([0, 225, 1e9], ["<=225", "225+"]),
    "pass TD": ([0, 1.75, 1e9], ["<=1.5", "2.5+"]),
}

_PRICE_RE = re.compile(r"([+-]\d+)\s*$")


class Calibration(NamedTuple):
    table: pd.DataFrame
    global_delta: float


def line_bucket(market: str, line) -> str | None:
    spec = LINE_BINS.get(market)
    try:
        line = float(line)
    except (TypeError, ValueError):
        return None
    if spec is None or not np.isfinite(line):
        return None
    edges, labels = spec
    for lo, hi, lab in zip(edges[:-1], edges[1:], labels):
        if lo < line <= hi:
            return lab
    return None


def parse_price(best) -> float:
    m = _PRICE_RE.search(str(best))
    return float(m.group(1)) if m else np.nan


def breakeven(odds: float) -> float:
    return abs(odds) / (abs(odds) + 100) if odds < 0 else 100 / (odds + 100)


def win_profit(odds: float) -> float:
    return 100 / abs(odds) if odds < 0 else odds / 100


COUNT_MARKETS = ("receptions", "pass TD")


def pairs_cleanly(market: str, line) -> bool:
    """line_history stores the CONSENSUS line (average across books) but the best price from
    ONE book. A whole-number consensus on a count market means the books disagree by a full
    unit (0.5 vs 1.5 catches), so that price belongs to a different bet than the one graded
    and modeled -- it showed up as 'OVER 1.0 at +194' ranking as the best EV on the board and
    then pushing. Yardage markets are fine: a 1-yard line gap barely moves the odds."""
    return not (market in COUNT_MARKETS and (float(line) * 2) % 2 == 0)


def prep(df: pd.DataFrame) -> pd.DataFrame:
    """Core-market rows with a line bucket, a parsed price, and the model's own p(hit)."""
    d = df[df["market"].isin(CORE)].copy()
    d = d[[pairs_cleanly(m, l) for m, l in zip(d["market"], d["line"])]].copy()
    d["bucket"] = [line_bucket(m, l) for m, l in zip(d["market"], d["line"])]
    d["odds"] = d["best"].map(parse_price)
    d["model_p"] = pd.to_numeric(d["p(hit)%"], errors="coerce") / 100.0
    return d.dropna(subset=["bucket", "odds", "model_p"])


def fit(train: pd.DataFrame, k: float = K_SHRINK) -> Calibration:
    """Per (market, side, line bucket) miss = actual win rate - model's p, shrunk toward
    the global miss. Trained only on labeled-tier, decided (win/loss) rows."""
    d = prep(train)
    d = d[(d["conf"] != "—") & d["result"].isin(["win", "loss"])].copy()
    if d.empty:
        return Calibration(pd.DataFrame(columns=["market", "lean", "bucket", "n", "win",
                                                 "model_p", "delta_raw", "delta"]), 0.0)
    d["won"] = (d["result"] == "win").astype(float)
    d["miss"] = d["won"] - d["model_p"]
    g = float(d["miss"].mean())
    rows = []
    for (mk, side, bk), x in d.groupby(["market", "lean", "bucket"]):
        n, raw = len(x), float(x["miss"].mean())
        rows.append(dict(market=mk, lean=side, bucket=bk, n=n, win=float(x["won"].mean()),
                         model_p=float(x["model_p"].mean()), delta_raw=raw,
                         delta=(n * raw + k * g) / (n + k)))
    return Calibration(pd.DataFrame(rows), g)


def score(cands: pd.DataFrame, cal: Calibration) -> pd.DataFrame:
    """Adds p_cal (corrected hit probability), breakeven, edge_pts and ev (profit per $1)."""
    d = prep(cands)
    d = d[d["conf"] != "—"].copy()
    if d.empty:
        return d.assign(p_cal=np.nan, breakeven=np.nan, edge_pts=np.nan, ev=np.nan)
    lookup = (cal.table.set_index(["market", "lean", "bucket"])["delta"].to_dict()
              if not cal.table.empty else {})
    delta = np.array([lookup.get((m, s, b), cal.global_delta)
                      for m, s, b in zip(d["market"], d["lean"], d["bucket"])], float)
    d["p_cal"] = np.clip(d["model_p"].to_numpy() + delta, *P_CLIP)
    d["breakeven"] = d["odds"].map(breakeven)
    d["edge_pts"] = (d["p_cal"] - d["breakeven"]) * 100
    d["ev"] = d["p_cal"] * d["odds"].map(win_profit) - (1 - d["p_cal"])
    return d


MIN_CAL_ROWS = 100   # under this much graded history a "calibrated" number is just the model's own


def history_calibration(lh: pd.DataFrame | None = None) -> Calibration | None:
    """Calibration from every graded week on file, or None while there isn't enough history
    for it to mean anything (the screen then shows no calibrated columns at all)."""
    if lh is None:
        from dfs import bets
        lh = bets.load_line_history()
    cal = fit(lh)
    return cal if (len(cal.table) and cal.table["n"].sum() >= MIN_CAL_ROWS) else None


def annotate(edges: pd.DataFrame, cal: Calibration) -> pd.DataFrame:
    """`edges` (live Prop Edges rows) with `p cal %` and `EV %` appended -- same rows, same
    order, NaN where a row can't be priced cleanly (whole-number count line, unknown line
    bucket, no parsed price). Display only: tiers and filters are untouched."""
    out = edges.copy()
    out["_k"] = np.arange(len(out))
    sc = score(out, cal).set_index("_k")
    out["p cal %"] = (100 * sc["p_cal"]).reindex(out["_k"]).to_numpy().round(0)
    out["EV %"] = (100 * sc["ev"]).reindex(out["_k"]).to_numpy().round(0)
    return out.drop(columns="_k")


def top_picks(scored: pd.DataFrame, k: int = 8, p_min: float = P_MIN) -> pd.DataFrame:
    """Best calibrated-EV props, one per player, only if calibrated to p_min+ and +EV.
    Can return fewer than k -- on a thin week that is the honest answer."""
    s = scored[(scored["p_cal"] >= p_min) & (scored["ev"] > 0)]
    return s.sort_values("ev", ascending=False).drop_duplicates(subset="player").head(k)


# ── baselines the tool effectively offers today ─────────────────────────────────

def tier_picks(scored: pd.DataFrame, k: int = 8, recommended: bool = False) -> pd.DataFrame:
    d = scored[scored["conf"].isin(TIER_RANK)].copy()
    if recommended:
        from dfs import props
        d = props.role_stable(props.confident_only(d))
    d["_tr"] = d["conf"].map(TIER_RANK)
    d["_az"] = d["z"].abs()
    return d.sort_values(["_tr", "_az"], ascending=False).drop_duplicates(subset="player").head(k)


def edge_picks(scored: pd.DataFrame, k: int = 8) -> pd.DataFrame:
    d = scored.copy()
    d["_pe"] = pd.to_numeric(d["p edge"], errors="coerce")
    return d.sort_values("_pe", ascending=False).drop_duplicates(subset="player").head(k)


# ── walk-forward evaluation ─────────────────────────────────────────────────────

def _grade(picks: pd.DataFrame) -> dict:
    d = picks[picks["result"].isin(["win", "loss"])]
    if d.empty:
        return dict(n=0, win=np.nan, roi=np.nan)
    won = d["result"] == "win"
    profit = np.where(won, d["odds"].map(win_profit), -1.0)
    return dict(n=len(d), win=100 * won.mean(), roi=100 * profit.mean())


def walk_forward(lh: pd.DataFrame, test_weeks: list[int], ks=(8, 16), k_shrink=K_SHRINK,
                 p_min=P_MIN) -> dict:
    """For each test week, calibrate on strictly earlier weeks only, then pick."""
    rows, brier, reliab, picks_log = [], [], [], []
    for w in test_weeks:
        cal = fit(lh[lh["week"] < w], k_shrink)
        cands = lh[lh["week"] == w]
        sc = score(cands, cal)
        sc_dec = sc[sc["result"].isin(["win", "loss"])].copy()
        for K in ks:
            methods = {
                "calibrated EV": top_picks(sc, K, p_min),
                "tier (high>strong>solid, then |z|)": tier_picks(sc, K),
                "tier, Recommended only": tier_picks(sc, K, recommended=True),
                "model p-edge": edge_picks(sc, K),
            }
            for name, pk in methods.items():
                rows.append(dict(method=name, K=K, week=w, picked=len(pk), **_grade(pk)))
        rows.append(dict(method="ALL labeled props (reference)", K=0, week=w,
                         picked=len(sc_dec), **_grade(sc_dec)))
        rows.append(dict(method="every calibrated pick (p>=p_min, +EV)", K=0, week=w,
                         picked=0, **_grade(sc[(sc["p_cal"] >= p_min) & (sc["ev"] > 0)])))
        if len(sc_dec):
            won = (sc_dec["result"] == "win").astype(float)
            t = cal.table
            base = float((t["win"] * t["n"]).sum() / t["n"].sum()) if len(t) else 0.5
            brier.append(dict(week=w, n=len(sc_dec),
                              constant=((base - won) ** 2).mean(),
                              model=((sc_dec["model_p"] - won) ** 2).mean(),
                              calibrated=((sc_dec["p_cal"] - won) ** 2).mean()))
            sc_dec["won"] = won
            reliab.append(sc_dec[["model_p", "p_cal", "won"]].assign(week=w))
        picks_log.append(top_picks(sc, 8, p_min).assign(test_week=w))
    return dict(grid=pd.DataFrame(rows), brier=pd.DataFrame(brier),
                reliab=pd.concat(reliab) if reliab else pd.DataFrame(),
                picks=pd.concat(picks_log) if picks_log else pd.DataFrame())


def _pooled(grid: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (m, K), g in grid.groupby(["method", "K"], sort=False):
        n = g["n"].sum()
        if n == 0:
            continue
        win = (g["win"] * g["n"]).sum() / n
        roi = (g["roi"] * g["n"]).sum() / n
        out.append(dict(method=m, K=K, n=int(n), win=round(win, 1), roi=round(roi, 1)))
    return pd.DataFrame(out)


def report(lh: pd.DataFrame, test_weeks: list[int]) -> str:
    wf = walk_forward(lh, test_weeks)
    L = [f"# Calibrated-conviction walk-forward — 2026 test weeks {test_weeks}", "",
         "Each test week is calibrated ONLY on earlier weeks; price-aware ROI at the best "
         f"available price; shrink k={K_SHRINK:g}, min calibrated p={P_MIN}.", ""]
    L.append("## Pooled across test weeks (n = decided picks)")
    L.append("")
    pooled = _pooled(wf["grid"])
    L.append("| method | K | n | win% | ROI |")
    L.append("|---|--:|--:|--:|--:|")
    for _, r in pooled.iterrows():
        L.append(f"| {r.method} | {r.K or 'all'} | {r.n} | {r.win} | {r.roi:+.1f}% |")
    L.append("")
    L.append("## Per week, top 8")
    L.append("")
    L.append("| method | week | n | win% | ROI |")
    L.append("|---|--:|--:|--:|--:|")
    g8 = wf["grid"][wf["grid"]["K"] == 8]
    for _, r in g8.iterrows():
        L.append(f"| {r.method} | {r.week} | {int(r.n)} | "
                 f"{'' if pd.isna(r.win) else round(r.win, 1)} | "
                 f"{'' if pd.isna(r.roi) else format(r.roi, '+.1f') + '%'} |")
    L.append("")
    if len(wf["brier"]):
        b = wf["brier"]
        tot = b["n"].sum()
        L.append("## Probability quality (Brier, lower is better) — every labeled prop in the test weeks")
        L.append("")
        L.append(f"n={tot}; constant (training win rate)={(b['constant']*b['n']).sum()/tot:.4f}, "
                 f"model p(hit)={(b['model']*b['n']).sum()/tot:.4f}, "
                 f"calibrated p={(b['calibrated']*b['n']).sum()/tot:.4f}")
        L.append("")
        r = wf["reliab"].copy()
        r["bin"] = pd.cut(r["p_cal"], [0, .50, .55, .60, .65, 1.0],
                          labels=["<50", "50-55", "55-60", "60-65", "65+"])
        L.append("| calibrated p | n | model said | calibrated said | actually won |")
        L.append("|---|--:|--:|--:|--:|")
        for lab, x in r.groupby("bin", observed=True):
            L.append(f"| {lab} | {len(x)} | {100*x.model_p.mean():.1f}% | "
                     f"{100*x.p_cal.mean():.1f}% | {100*x.won.mean():.1f}% |")
        L.append("")
    return "\n".join(L)


def main():
    from dfs import bets
    season = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    lh = bets.load_line_history()
    lh = lh[lh["season"] == season]
    weeks = sorted(int(w) for w in lh["week"].unique())
    test_weeks = [w for w in weeks if w >= 3]
    txt = report(lh, test_weeks)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(txt)
    with open(f"matchup_model/conviction_{season}_report.md", "w", encoding="utf-8") as f:
        f.write(txt)


if __name__ == "__main__":
    main()
