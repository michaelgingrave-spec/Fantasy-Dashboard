"""Which edge range actually wins — computed from history, no logged bets needed.

For every player-week 2023-25 we already have our walk-forward projection and the real
box score. We don't have historical sportsbook prop lines (those are a paid feed), so we
stand in the player's **trailing form** (EWMA of the stat over its last ~8 games) as the
line — that's what books anchor a prop to. Then:

    edge   = our_projection - trailing_line
    result = did the actual land on our side of the trailing line?

Bucket by |edge| %, look at hit rate + ROI at -110. The point is the *shape*: which edge
band is most profitable. Hit rates here are an upper bound (a real book line is sharper
than trailing form); the ranking of buckets is what transfers.

    python -m matchup_model.opp.edge_calib
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from matchup_model.opp import data as D
from matchup_model.opp import model as M

TEST_SEASONS = [2023, 2024, 2025]
TEST_WEEKS = range(4, 19)
POS = ["QB", "RB", "WR", "TE"]
_HL, _LB = 4, 8

# our line component -> (nflverse column, min trailing line to count it a real prop)
MARKETS = {
    "rec_yds": ("receiving_yards", 20.0), "rec": ("receptions", 2.5),
    "rush_yds": ("rushing_yards", 25.0), "rush_att": ("carries", 7.0),
    "pass_yds": ("passing_yards", 185.0), "pass_td": ("passing_tds", 0.9),
    "pass_att": ("attempts", 26.0),
}
NICE = {"rec_yds": "rec yds", "rec": "receptions", "rush_yds": "rush yds",
        "rush_att": "rush att", "pass_yds": "pass yds", "pass_td": "pass TD",
        "pass_att": "pass att"}

# |edge| % buckets (same as dfs.bets so the two analyses line up)
BUCKETS = [(0, 4, "0-4%"), (4, 8, "4-8%"), (8, 12, "8-12%"),
           (12, 20, "12-20%"), (20, 1e9, "20%+")]
PRICE = -110  # assumed juice for the ROI column

ROWS_CSV = Path(__file__).resolve().parents[1] / "opp_edge_calibration.csv"
REPORT = Path(__file__).resolve().parents[1] / "opp_edge_calibration.md"


def _ewma(v: np.ndarray) -> float:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)][-_LB:]
    if v.size == 0:
        return np.nan
    w = 0.5 ** (np.arange(v.size)[::-1] / _HL)
    return float((w * v).sum() / w.sum())


def _roi(wins: int, losses: int, price: int = PRICE) -> float:
    dec = price / 100.0 if price > 0 else 100.0 / abs(price)
    n = wins + losses
    return (wins * dec - losses) / n if n else np.nan


def run() -> pd.DataFrame:
    pw = M._pw_with_team()
    hist_by_id = {pid: g.sort_values(["season", "week"]) for pid, g in pw.groupby("player_id")}
    rows, t0 = [], time.time()
    for season in TEST_SEASONS:
        sl = pw[(pw.season == season) & (pw.week.isin(TEST_WEEKS)) & (pw.position.isin(POS))]
        sl = sl[(sl.targets + sl.carries + sl.attempts) >= 3]
        for _, r in sl.iterrows():
            pid, pos, wk = r["player_id"], r["position"], int(r["week"])
            g = hist_by_id.get(pid)
            if g is None:
                continue
            h = g[(g.season < season) | ((g.season == season) & (g.week < wk))].tail(_LB)
            if len(h) < 4:
                continue
            o = M.opp_line(pid, pos, season, wk, by="id", injury_adj=True)
            ln = o.get("line") or {}
            if not ln:
                continue
            for comp, (col, floor) in MARKETS.items():
                if comp not in ln:
                    continue
                proxy = _ewma(h[col].to_numpy())
                if not np.isfinite(proxy) or proxy < floor:
                    continue
                our = float(ln[comp])
                actual = float(r[col])
                edge = our - proxy
                side = "OVER" if edge > 0 else "UNDER"
                res = ("push" if actual == proxy else
                       "win" if ((actual > proxy) == (side == "OVER")) else "loss")
                rows.append(dict(
                    season=season, week=wk, player=r["player_display_name"], pos=pos,
                    market=NICE[comp], line=round(proxy, 2), our=round(our, 2),
                    actual=round(actual, 2), edge=round(edge, 2),
                    edge_pct=round(100 * edge / proxy, 1), side=side, result=res,
                    realized_pct=round(100 * (actual - proxy) / proxy, 1),
                ))
        print(f"  {season}: {len(rows)} rows ({time.time()-t0:.0f}s)")
    return pd.DataFrame(rows)


def _bucket_stats(df: pd.DataFrame) -> pd.DataFrame:
    e = df["edge_pct"].abs()
    out = []
    for lo, hi, lbl in BUCKETS:
        s = df[(e >= lo) & (e < hi)]
        dec = s[s.result.isin(["win", "loss"])]
        w, l = int((dec.result == "win").sum()), int((dec.result == "loss").sum())
        if w + l < 15:
            continue
        # directional: did the actual move our way at all
        dirhit = ((np.sign(s["edge"]) == np.sign(s["actual"] - s["line"]))
                  [s["actual"] != s["line"]].mean())
        out.append({"edge range": lbl, "n": len(s), "win%": round(100 * w / (w + l), 1),
                    "ROI@-110": round(100 * _roi(w, l), 1),
                    "dir hit%": round(100 * dirhit, 1),
                    "mean edge%": round(s["edge_pct"].abs().mean(), 1),
                    "mean realized%": round((s["realized_pct"] * np.sign(s["edge"])).mean(), 1)})
    return pd.DataFrame(out)


def bucket_table() -> pd.DataFrame:
    """Cached read for the app. Empty frame (with a hint) if the calc hasn't been run."""
    if not ROWS_CSV.exists():
        return pd.DataFrame()
    return _bucket_stats(pd.read_csv(ROWS_CSV))


def by_market() -> pd.DataFrame:
    if not ROWS_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(ROWS_CSV)
    out = []
    for mk, s in df.groupby("market"):
        dec = s[s.result.isin(["win", "loss"])]
        w, l = int((dec.result == "win").sum()), int((dec.result == "loss").sum())
        out.append({"market": mk, "n": len(s), "win%": round(100 * w / max(w + l, 1), 1),
                    "ROI@-110": round(100 * _roi(w, l), 1)})
    return pd.DataFrame(out).sort_values("ROI@-110", ascending=False)


def report(df: pd.DataFrame) -> str:
    L = ["# Model edge calibration — vs a trailing-form stand-in line", ""]
    L.append(f"- {len(df)} player-week-market rows, {TEST_SEASONS} wk {TEST_WEEKS.start}-"
             f"{TEST_WEEKS.stop-1}. Line = EWMA of the stat over the player's last {_LB} games.")
    L.append("- **Hit rates are an upper bound** — a real sportsbook line already prices "
             "matchup/pace/injuries, so a real edge is smaller. Trust the *ranking* of buckets.")
    L.append("")
    dec = df[df.result.isin(["win", "loss"])]
    w, l = int((dec.result == "win").sum()), int((dec.result == "loss").sum())
    L.append(f"Overall: **{w}-{l}** ({100*w/(w+l):.1f}%) · ROI@-110 **{100*_roi(w,l):+.1f}%** "
             f"· break-even is 52.4%")
    L.append("")
    L.append("## By |edge| bucket")
    L.append("")
    bs = _bucket_stats(df)
    L.append("| " + " | ".join(bs.columns) + " |")
    L.append("|" + "|".join(["---"] * len(bs.columns)) + "|")
    for _, r in bs.iterrows():
        L.append("| " + " | ".join(str(x) for x in r.values) + " |")
    L.append("")
    best = bs.loc[bs["ROI@-110"].idxmax()] if not bs.empty else None
    if best is not None:
        L.append(f"**Best bucket: {best['edge range']}** — {best['win%']}% hit, "
                 f"{best['ROI@-110']:+}% ROI on n={best['n']}. "
                 + ("Bigger edges do NOT do better — likely model error."
                    if bs.iloc[-1]["ROI@-110"] < best["ROI@-110"] else
                    "ROI keeps climbing with edge here."))
    L.append("")
    L.append("## By market")
    L.append("")
    for mk, s in df.groupby("market"):
        d = s[s.result.isin(["win", "loss"])]
        ww, ll = int((d.result == "win").sum()), int((d.result == "loss").sum())
        L.append(f"- **{mk}**: {ww}-{ll} ({100*ww/max(ww+ll,1):.1f}%), "
                 f"ROI {100*_roi(ww,ll):+.1f}%  (n={len(s)})")
    return "\n".join(L)


def main():
    df = run()
    if df.empty:
        print("no rows — populate the nflverse cache first (python -m matchup_model.opp.data)")
        return
    df.to_csv(ROWS_CSV, index=False)
    txt = report(df)
    REPORT.write_text(txt, encoding="utf-8")
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("\n" + txt + f"\n\nwrote {REPORT}")


if __name__ == "__main__":
    main()
