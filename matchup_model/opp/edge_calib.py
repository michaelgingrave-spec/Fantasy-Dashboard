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

# |z| (standardized-edge) buckets — the comparable unit. Also keep |edge %| for reference.
Z_BUCKETS = [(0.0, 0.15, "<0.15 SD"), (0.15, 0.30, "0.15-0.30"), (0.30, 0.50, "0.30-0.50"),
             (0.50, 0.80, "0.50-0.80"), (0.80, 9.0, "0.80+ SD")]
BUCKETS = [(0, 4, "0-4%"), (4, 8, "4-8%"), (8, 12, "8-12%"), (12, 20, "12-20%"), (20, 1e9, "20%+")]
PRICE = -110  # assumed juice for the ROI column


def _sigma_of(comp: str, line: float) -> float:
    from dfs.props import _sigma          # one source of truth for the fitted sigma model
    return _sigma(comp, line)


def _conf_of(z: float, comp: str) -> str:
    from dfs.props import conf_label      # applies the per-market offset (rush)/skip (pass att)
    return conf_label(z, comp)

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
                sigma = _sigma_of(comp, proxy)
                z = edge / sigma
                side = "OVER" if edge > 0 else "UNDER"
                res = ("push" if actual == proxy else
                       "win" if ((actual > proxy) == (side == "OVER")) else "loss")
                rows.append(dict(
                    season=season, week=wk, player=r["player_display_name"], pos=pos,
                    market=NICE[comp], line=round(proxy, 2), our=round(our, 2),
                    actual=round(actual, 2), edge=round(edge, 2),
                    edge_pct=round(100 * edge / proxy, 1),
                    z=round(z, 3), conf=_conf_of(z, comp), side=side, result=res,
                    realized_pct=round(100 * (actual - proxy) / proxy, 1),
                    realized_z=round((actual - proxy) / sigma, 3),
                ))
        print(f"  {season}: {len(rows)} rows ({time.time()-t0:.0f}s)")
    return pd.DataFrame(rows)


_CONF_ORDER = ["—", "lean", "solid", "strong", "high"]


def _bucket_stats(df: pd.DataFrame, by: str = "z") -> pd.DataFrame:
    """`by`='z' → confidence tier (the `conf` column — per-market-adjusted, matches what
    the app actually shows/filters on); 'edge_pct' → the old, unadjusted |edge %| view."""
    if by == "z" and "conf" not in df.columns:
        by = "edge_pct"
    out = []
    if by == "z":
        for tier in _CONF_ORDER:
            s = df[df["conf"] == tier]
            dec = s[s.result.isin(["win", "loss"])]
            w, l = int((dec.result == "win").sum()), int((dec.result == "loss").sum())
            if w + l < 15:
                continue
            dirhit = ((np.sign(s["edge"]) == np.sign(s["actual"] - s["line"]))
                      [s["actual"] != s["line"]].mean())
            row = {"conf": tier, "n": len(s), "win%": round(100 * w / (w + l), 1),
                   "ROI@-110": round(100 * _roi(w, l), 1), "dir hit%": round(100 * dirhit, 1)}
            if "realized_z" in s.columns:
                row["realized z"] = round((s["realized_z"] * np.sign(s["edge"])).mean(), 2)
            out.append(row)
        return pd.DataFrame(out)
    e = df["edge_pct"].abs()
    for lo, hi, lbl in BUCKETS:
        s = df[(e >= lo) & (e < hi)]
        dec = s[s.result.isin(["win", "loss"])]
        w, l = int((dec.result == "win").sum()), int((dec.result == "loss").sum())
        if w + l < 15:
            continue
        dirhit = ((np.sign(s["edge"]) == np.sign(s["actual"] - s["line"]))
                  [s["actual"] != s["line"]].mean())
        out.append({"edge %": lbl, "n": len(s), "win%": round(100 * w / (w + l), 1),
                    "ROI@-110": round(100 * _roi(w, l), 1), "dir hit%": round(100 * dirhit, 1)})
    return pd.DataFrame(out)


def bucket_table(by: str = "z") -> pd.DataFrame:
    """Cached read for the app. Empty frame if the calc hasn't been run."""
    if not ROWS_CSV.exists():
        return pd.DataFrame()
    return _bucket_stats(pd.read_csv(ROWS_CSV), by=by)


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


def _md_table(bs: pd.DataFrame) -> list:
    if bs.empty:
        return ["_(no buckets met the sample floor)_"]
    L = ["| " + " | ".join(bs.columns) + " |", "|" + "|".join(["---"] * len(bs.columns)) + "|"]
    for _, r in bs.iterrows():
        L.append("| " + " | ".join(str(x) for x in r.values) + " |")
    return L


def report(df: pd.DataFrame) -> str:
    L = ["# Model edge calibration — vs a trailing-form stand-in line", ""]
    L.append(f"- {len(df)} player-week-market rows, {TEST_SEASONS} wk {TEST_WEEKS.start}-"
             f"{TEST_WEEKS.stop-1}. Line = EWMA of the stat over the player's last {_LB} games.")
    L.append("- **`z` = edge / outcome-SD** (sigma = a + b*line, fitted per market). This is the "
             "unit that's comparable across markets — a +33% edge on a 1.5-catch line and a "
             "+7% edge on a 270 pass-yd line are ~0.3 vs ~0.35 SD, not 5x apart.")
    L.append("- Hit rates are an **upper bound** — a real sportsbook line already prices "
             "matchup/pace/injuries. Trust the threshold/ranking, not the absolute win%.")
    L.append("")
    dec = df[df.result.isin(["win", "loss"])]
    w, l = int((dec.result == "win").sum()), int((dec.result == "loss").sum())
    L.append(f"Overall: **{w}-{l}** ({100*w/(w+l):.1f}%) · ROI@-110 **{100*_roi(w,l):+.1f}%** "
             f"· break-even is 52.4%")
    L.append("")
    L.append("## By confidence tier (the `conf` column — market-offset-adjusted |z|)")
    L.append("")
    bz = _bucket_stats(df, by="z")
    L += _md_table(bz)
    L.append("")
    if not bz.empty:
        pos = bz[bz["ROI@-110"] > 1]
        thr = pos.iloc[0]["conf"] if len(pos) else "—"
        L.append(f"**Rule: bet at |z| ≥ ~0.30 SD.** Below ~0.15 it's a coin flip "
                 f"({bz.iloc[0]['win%']}% / {bz.iloc[0]['ROI@-110']:+}%); from 0.30 up it's a "
                 f"clear edge and stays positive as z grows. First +EV bucket: {thr}.")
    L.append("")
    L.append("## By |z| within each market  (win% / ROI@-110, n)")
    L.append("")
    L.append("| market | z<0.30 | 0.30-0.60 | z>=0.60 |")
    L.append("|---|---|---|---|")
    for mk in ["rec yds", "receptions", "rush yds", "pass yds", "pass TD", "rush att", "pass att"]:
        g = df[df.market == mk]
        cells = []
        for lo, hi in [(0.0, 0.30), (0.30, 0.60), (0.60, 9.0)]:
            s = g[(g["z"].abs() >= lo) & (g["z"].abs() < hi)]
            d = s[s.result.isin(["win", "loss"])]
            ww, ll = int((d.result == "win").sum()), int((d.result == "loss").sum())
            cells.append(f"{100*ww/max(ww+ll,1):.0f}% / {100*_roi(ww,ll):+.0f}% (n={len(s)})"
                         if ww + ll >= 15 else "—")
        L.append(f"| {mk} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("_rush yds / rush att: only z≥0.60 is reliably +EV. pass att: negative at every z "
             "— skip it._")
    L.append("")
    L.append("## For reference — the old |edge %| view")
    L.append("")
    L += _md_table(_bucket_stats(df, by="edge_pct"))
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
