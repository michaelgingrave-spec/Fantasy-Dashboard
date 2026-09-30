"""Stat-line-level accuracy check across three levels of deference to FantasyPoints'
own weekly projection: how much should their number move OUR projected stat line?

`week_accuracy.py` only ever compared total DK fantasy points. That can hide a real
answer -- FantasyPoints could nail the total while being off on the mix (right on
points, wrong on yards vs. TDs) -- and total-fp accuracy isn't actually what a prop
bets on. This grades the individual stat categories instead (rec_yds, receptions,
rush_yds, pass_yds, pass_td), walk-forward, for three variants:

  no_shift  -- pure trailing-usage model, FantasyPoints ignored entirely
  current   -- today's capped-shift blend (see matchup_model.opp.blend._fp_shift)
  anchored  -- our own stat-line SHAPE, rescaled 100% to FantasyPoints' total

All three reuse the same zero-floor guard (a FantasyPoints number under
FP_PROJ_FLOOR is untrustworthy when our own read is FP_PROJ_TRUST_MIN+ -- almost
always an unresolved-status placeholder, not a real "will score zero" call).

Part A -- broad accuracy: every player with real usage that week, RMSE/MAE per
(variant, stat), same player pool as week_accuracy.py.

Part B -- would-it-have-won: restricted to props we actually pulled real book lines
for (data/dfs/props/line_history.csv), re-leaning each variant's number against the
SAME real line and grading against the SAME real result. This is the number that
actually matters -- would switching variants have won more bets, not just lowered
error on paper.

    python -m matchup_model.opp.stat_accuracy 2026
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from dfs.names import normalize_name
from matchup_model.opp import data as D
from matchup_model.opp import model as M
from matchup_model.opp.blend import (
    BLEND_W, FP_DEV_FLOOR, FP_DEV_SPAN, FP_PROJ_FLOOR, FP_PROJ_TRUST_MIN,
    FP_SHIFT_CAP, INJURY_ADJ, _fp_projections, _naive_fp,
)
from matchup_model.weekly_pull import last_completed_week

STAT_COLS = {
    "rec_yds": "receiving_yards", "rec": "receptions", "rush_yds": "rushing_yards",
    "pass_yds": "passing_yards", "pass_td": "passing_tds",
}
# line_history.csv's human-readable market label -> the stat key above
MARKET_TO_STAT = {"rec yds": "rec_yds", "receptions": "rec", "rush yds": "rush_yds",
                  "pass yds": "pass_yds", "pass TD": "pass_td"}
VARIANTS = ["no_shift", "current", "anchored"]


def _shift(base_fp: float, fp_proj: float | None, variant: str) -> float:
    """Same gating as blend._fp_shift; only the shift AMOUNT differs by variant."""
    if variant == "no_shift" or fp_proj is None or base_fp < 1.0:
        return base_fp
    if fp_proj < FP_PROJ_FLOOR and base_fp >= FP_PROJ_TRUST_MIN:
        return base_fp
    role_ratio = fp_proj / base_fp
    if variant == "anchored":
        shift = 1.0
    else:
        dev = abs(role_ratio - 1.0)
        shift = min(max(dev - FP_DEV_FLOOR, 0.0) / FP_DEV_SPAN, FP_SHIFT_CAP)
    if shift <= 0:
        return base_fp
    return (1 - shift) * base_fp + shift * fp_proj


def variant_line(name_key: str, pos: str, season: int, week: int,
                 variant: str) -> dict | None:
    """Replays blended_line()'s math for one variant; returns the scaled stat-line
    dict (same keys as opp_line's "line"), or None if we have no base projection."""
    o = M.opp_line(name_key, pos, season, week, by="name", injury_adj=INJURY_ADJ)
    opp_fp = o.get("dk_fp")
    if opp_fp is None or not np.isfinite(opp_fp) or opp_fp <= 0.5:
        return None
    line = {k: float(v) for k, v in o["line"].items()}
    naive = _naive_fp(name_key, pos, season, week)
    if np.isfinite(naive) and pos in BLEND_W:
        w = BLEND_W[pos]
        base_fp = w * float(opp_fp) + (1 - w) * float(naive)
    else:
        base_fp = float(opp_fp)
    fp_proj = _fp_projections(week).get(name_key)
    adj_fp = _shift(base_fp, fp_proj, variant)
    scale = adj_fp / base_fp if base_fp > 0 else 1.0
    return {k: v * scale for k, v in line.items()}


# ── Part A: broad stat-category accuracy ────────────────────────────────────────

def run_broad(season: int, weeks: list[int]) -> pd.DataFrame:
    pw = D.player_weeks()
    rows = []
    for week in weeks:
        actual = pw[(pw.season == season) & (pw.week == week) &
                   (pw.position.isin(["QB", "RB", "WR", "TE"]))].copy()
        actual = actual[(actual.targets + actual.carries + actual.attempts) >= 1]
        actual["name_key"] = actual["player_display_name"].map(normalize_name)
        for _, r in actual.iterrows():
            nk, pos = r["name_key"], r["position"]
            for variant in VARIANTS:
                line = variant_line(nk, pos, season, week, variant)
                if line is None:
                    continue
                for stat, col in STAT_COLS.items():
                    if stat not in line:
                        continue
                    rows.append(dict(week=week, player=r["player_display_name"], pos=pos,
                                     variant=variant, stat=stat,
                                     proj=line[stat], actual=float(r[col])))
    return pd.DataFrame(rows)


def _rmse(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2))) if m.any() else np.nan


def _mae(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.mean(np.abs(a[m] - b[m]))) if m.any() else np.nan


def broad_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stat in STAT_COLS:
        for variant in VARIANTS:
            g = df[(df.stat == stat) & (df.variant == variant)]
            if len(g) < 5:
                continue
            rows.append({"stat": stat, "variant": variant, "n": len(g),
                        "rmse": round(_rmse(g.proj, g.actual), 2),
                        "mae": round(_mae(g.proj, g.actual), 2)})
    return pd.DataFrame(rows)


# ── Part B: would-it-have-won, on real pulled lines ─────────────────────────────

def run_vs_lines(season: int) -> pd.DataFrame:
    from dfs.bets import load_line_history
    lh = load_line_history()
    lh = lh[(lh.season == season) & (lh.result.isin(["win", "loss", "push"])) &
            (lh.market.isin(MARKET_TO_STAT))].copy()
    if lh.empty:
        return lh

    pw = D.player_weeks()
    pos_lookup = (pw.assign(nk=pw["player_display_name"].map(normalize_name))
                    .sort_values(["season", "week"]).groupby("nk")["position"].last())

    rows = []
    for _, r in lh.iterrows():
        nk = normalize_name(str(r["player"]))
        pos = pos_lookup.get(nk)
        if pos is None:
            continue
        stat = MARKET_TO_STAT[r["market"]]
        book_line = float(r["line"])
        actual = float(r["actual"])
        for variant in VARIANTS:
            line = variant_line(nk, pos, int(r["season"]), int(r["week"]), variant)
            if line is None or stat not in line:
                continue
            proj = line[stat]
            if proj == book_line:
                continue
            side = "OVER" if proj > book_line else "UNDER"
            result = "push" if actual == book_line else (
                "win" if ((actual > book_line) == (side == "OVER")) else "loss")
            rows.append(dict(season=r["season"], week=r["week"], player=r["player"],
                             market=r["market"], variant=variant, proj=round(proj, 2),
                             line=book_line, side=side, actual=actual, result=result))
    return pd.DataFrame(rows)


_PRICE = -110


def _roi_at(price: int, w: int, l: int) -> float:
    stake = w + l
    if stake == 0:
        return 0.0
    profit = w * (100 / abs(price)) - l
    return profit / stake


def vs_lines_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant in VARIANTS:
        g = df[df.variant == variant]
        dec = g[g.result.isin(["win", "loss"])]
        w = int((dec.result == "win").sum()); l = int((dec.result == "loss").sum())
        if w + l < 5:
            continue
        rows.append({"variant": variant, "n": len(g), "win%": round(100 * w / (w + l), 1),
                    "ROI@-110%": round(100 * _roi_at(_PRICE, w, l), 1)})
    return pd.DataFrame(rows)


# ── report / CLI ─────────────────────────────────────────────────────────────

def report(season: int, weeks: list[int]) -> str:
    L = [f"# Stat-line accuracy by FantasyPoints-deference variant — "
         f"{season} weeks {weeks[0]}-{weeks[-1]}", ""]

    broad = run_broad(season, weeks)
    L.append("## Part A — broad accuracy (every player with real usage, RMSE/MAE)")
    L.append("")
    if broad.empty:
        L.append("no rows")
    else:
        bt = broad_table(broad)
        for stat in STAT_COLS:
            sub = bt[bt.stat == stat].sort_values("rmse")
            if sub.empty:
                continue
            L.append(f"**{stat}**")
            L.append("")
            L.append("| variant | n | RMSE | MAE |")
            L.append("|---|--:|--:|--:|")
            for _, r in sub.iterrows():
                L.append(f"| {r.variant} | {r.n} | {r.rmse} | {r.mae} |")
            L.append("")

    L.append("## Part B — would it have won (real pulled lines, real results)")
    L.append("")
    vl = run_vs_lines(season)
    if vl.empty:
        L.append("no rows (no graded line_history for this season yet)")
    else:
        vt = vs_lines_table(vl)
        L.append(f"- {len(vl[vl.variant == VARIANTS[0]])} props with a real book line "
                 f"per variant")
        L.append("")
        L.append("| variant | n | win% | ROI@-110% |")
        L.append("|---|--:|--:|--:|")
        for _, r in vt.iterrows():
            L.append(f"| {r.variant} | {r.n} | {r['win%']} | {r['ROI@-110%']} |")
    return "\n".join(L)


def main():
    season = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    last = last_completed_week()
    weeks = list(range(1, last + 1)) if last else [1]
    txt = report(season, weeks)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(txt)
    with open(f"matchup_model/stat_accuracy_{season}_report.md", "w", encoding="utf-8") as f:
        f.write(txt)


if __name__ == "__main__":
    main()
