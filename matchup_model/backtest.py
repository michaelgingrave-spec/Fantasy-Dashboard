"""Walk-forward backtest: does base_fp + matchup Δ beat the naive base_fp out of sample?

Test seasons 2023-24, weeks 4-18. Player-split training is limited to seasons strictly
before the test season; defense predictions and the base projection use only prior weeks.

Run:  python -m matchup_model.backtest
Writes matchup_model/backtest_report.md
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from matchup_model.config import MAX_DELTA_FRAC, ROOT
from matchup_model import actuals as _actuals
from matchup_model import matchup
from matchup_model.base_projection import base_row

TEST_SEASONS = [2023, 2024]
TEST_WEEKS = range(4, 19)
MIN_PRIOR_GAMES = 3
POS_KEEP = {"WR", "TE", "QB", "RB"}
REPORT = Path(__file__).parent / "backtest_report.md"


def _rmse(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2))) if m.any() else np.nan


def _mae(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.mean(np.abs(a[m] - b[m]))) if m.any() else np.nan


def run() -> pd.DataFrame:
    act = _actuals.actuals()
    act = act[act["pos"].isin(POS_KEEP)]
    rows = []
    t0 = time.time()
    for season in TEST_SEASONS:
        thru = season - 1
        wk_slice = act[act.season == season]
        for _, r in wk_slice.iterrows():
            wk = int(r["week"])
            if wk not in TEST_WEEKS:
                continue
            nk, pos, opp = r["name_key"], r["pos"], r["opp"]
            if not isinstance(opp, str) or not opp:
                continue
            b = base_row(nk, pos, season, wk)
            if b["n_games"] < MIN_PRIOR_GAMES or not np.isfinite(b["base_fp"]):
                continue
            d = matchup.player_week_delta(nk, pos, opp, season, wk, through_season=thru)
            base_fp = float(b["base_fp"])
            dfp = float(d["delta_fp"])
            cap = MAX_DELTA_FRAC * base_fp
            dfp = float(np.clip(dfp, -cap, cap))
            rows.append(dict(
                season=season, week=wk, name_key=nk, pos=pos, opp=opp,
                actual=float(r["fp"]), base=base_fp, delta=dfp, model=base_fp + dfp,
            ))
        print(f"  {season}: {len(rows)} rows so far ({time.time()-t0:.0f}s)")
    return pd.DataFrame(rows)


def report(df: pd.DataFrame) -> str:
    L = ["# Matchup model — walk-forward backtest", ""]
    L.append(f"- rows: **{len(df)}**  ·  seasons {TEST_SEASONS}  ·  weeks {TEST_WEEKS.start}–{TEST_WEEKS.stop-1}")
    L.append(f"- players moved (|Δ| ≥ 0.25 fp): **{(df['delta'].abs() >= 0.25).mean()*100:.0f}%**  ·  "
             f"mean |Δ| = {df['delta'].abs().mean():.2f}  ·  max |Δ| = {df['delta'].abs().max():.1f}")
    L.append("")
    L.append("## Error vs the naive base (lower = better)")
    L.append("")
    L.append("| slice | n | RMSE base | RMSE model | ΔRMSE | MAE base | MAE model |")
    L.append("|---|--:|--:|--:|--:|--:|--:|")
    for label, sub in [("ALL", df)] + [(p, df[df.pos == p]) for p in ("WR", "TE", "QB", "RB")]:
        if sub.empty:
            continue
        rb, rm = _rmse(sub.base, sub.actual), _rmse(sub.model, sub.actual)
        L.append(f"| {label} | {len(sub)} | {rb:.3f} | {rm:.3f} | **{rb-rm:+.3f}** | "
                 f"{_mae(sub.base, sub.actual):.3f} | {_mae(sub.model, sub.actual):.3f} |")
    # on the subset the model actually moves
    moved = df[df["delta"].abs() >= 0.5]
    if not moved.empty:
        rb, rm = _rmse(moved.base, moved.actual), _rmse(moved.model, moved.actual)
        L.append(f"| **moved ≥0.5** | {len(moved)} | {rb:.3f} | {rm:.3f} | **{rb-rm:+.3f}** | "
                 f"{_mae(moved.base, moved.actual):.3f} | {_mae(moved.model, moved.actual):.3f} |")
    L.append("")
    L.append("## Calibration — predicted Δ vs realised (actual − base), by Δ quintile")
    L.append("")
    L.append("| Δ quintile | n | mean pred Δ | mean realised | slope-ish |")
    L.append("|---|--:|--:|--:|--:|")
    d2 = df[df["delta"].abs() > 1e-6].copy()
    if len(d2) > 25:
        d2["q"] = pd.qcut(d2["delta"], 5, duplicates="drop")
        for q, g in d2.groupby("q"):
            realised = (g["actual"] - g["base"]).mean()
            L.append(f"| {q} | {len(g)} | {g['delta'].mean():+.2f} | {realised:+.2f} | "
                     f"{realised / g['delta'].mean():+.2f} |" if g['delta'].mean() else "|")
    L.append("")
    L.append("## Directional test — does Δ point the right way?")
    L.append("")
    d3 = df[df["delta"].abs() > 1e-6].copy()
    d3["resid"] = d3["actual"] - d3["base"]
    d3["hit"] = np.sign(d3["delta"]) == np.sign(d3["resid"])
    corr = np.corrcoef(d3["delta"], d3["resid"])[0, 1] if len(d3) > 5 else np.nan
    L.append(f"- corr(predicted Δ, realised residual): **{corr:+.3f}**  (n={len(d3)})")
    L.append(f"- directional hit rate (sign match): **{d3['hit'].mean()*100:.1f}%**  (50% = coin flip)")
    for thr in (0.5, 1.0):
        big = d3[d3["delta"].abs() >= thr]
        if len(big) > 20:
            L.append(f"- |Δ| ≥ {thr}: hit rate **{big['hit'].mean()*100:.1f}%**, "
                     f"mean resid when Δ>0 = {big[big.delta>0]['resid'].mean():+.2f}, "
                     f"when Δ<0 = {big[big.delta<0]['resid'].mean():+.2f}  (n={len(big)})")
    L.append("")
    L.append("## Weekly tail test — top vs bottom decile / ventile by predicted Δ (WR+TE)")
    L.append("")
    rec = df[df.pos.isin(["WR", "TE"])]
    for frac, name in [(0.1, "decile"), (0.05, "ventile")]:
        tops, bots = [], []
        for (s, w), g in rec.groupby(["season", "week"]):
            if len(g) < 25:
                continue
            g = g.sort_values("delta")
            k = max(2, int(len(g) * frac))
            bots.append((g.head(k)["actual"] - g.head(k)["base"]).mean())
            tops.append((g.tail(k)["actual"] - g.tail(k)["base"]).mean())
        if tops:
            L.append(f"- top {name}: realised resid **{np.nanmean(tops):+.2f}** · "
                     f"bottom {name}: **{np.nanmean(bots):+.2f}** · "
                     f"spread **{np.nanmean(tops)-np.nanmean(bots):+.2f}** (want > 0)")
    L.append("")
    d_rmse = _rmse(df.base, df.actual) - _rmse(df.model, df.actual)
    ok = (d_rmse > 0) and (corr > 0.02) and (d3["hit"].mean() > 0.505)
    L.append(f"## Verdict: **{'promising — worth calibrating β and wiring in' if ok else 'not beating base — do not wire into the app'}**")
    L.append(f"(ΔRMSE={d_rmse:+.3f}, corr={corr:+.3f}, hit={d3['hit'].mean()*100:.1f}%)")
    return "\n".join(L)


def main():
    df = run()
    df.to_csv(Path(__file__).parent / "backtest_rows.csv", index=False)
    txt = report(df)
    REPORT.write_text(txt, encoding="utf-8")
    print("\n" + txt)
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
