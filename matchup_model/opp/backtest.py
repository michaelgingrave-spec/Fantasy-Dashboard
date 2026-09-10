"""Walk-forward: does the opportunity model beat (a) a naive trailing-FP average and
(b) the current trailing-usage x efficiency line (`project_stats`)?

Same player-weeks for all three, DK scoring throughout. Prior = strictly earlier weeks
(+ all earlier seasons). Test seasons 2023-2025, weeks 4-18.

    python -m matchup_model.opp.backtest
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from matchup_model.opp import data as D
from matchup_model.opp import model as M

TEST_SEASONS = [2023, 2024, 2025]
TRAIN_SEASONS = [2023, 2024]      # blend weights fit here
HOLDOUT_SEASON = 2025            # ...and judged here
TEST_WEEKS = range(4, 19)
POS = ["QB", "RB", "WR", "TE"]
TOPK = {"QB": 12, "RB": 24, "WR": 30, "TE": 12}
REPORT = Path(__file__).resolve().parents[1] / "opp_backtest_report.md"
ROWS_CSV = Path(__file__).resolve().parents[1] / "opp_backtest_rows.csv"

_HL, _LB = 4, 10


def _ewma(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)][-_LB:]
    if v.size == 0:
        return np.nan
    w = 0.5 ** (np.arange(v.size)[::-1] / _HL)
    return float((w * v).sum() / w.sum())


def _current_line(h: pd.DataFrame, pos: str) -> float:
    """Port of matchup_model.project_stats: EWMA(usage) x EWMA(efficiency), no shrink,
    no team/vegas context. `h` = trailing rows, oldest->newest."""
    if len(h) < 3:
        return np.nan
    g = h.replace(0, np.nan)
    if pos in ("WR", "TE"):
        tgt = _ewma(h["targets"].to_numpy())
        catch = _ewma((h.receptions / g.targets).to_numpy())
        ypt = _ewma((h.receiving_yards / g.targets).to_numpy())
        tdpt = _ewma((h.receiving_tds / g.targets).to_numpy())
        catch = min(max(catch, 0.3), 0.95) if np.isfinite(catch) else 0.65
        line = {"rec": tgt * catch, "rec_yds": tgt * ypt if np.isfinite(ypt) else 0.0,
                "rec_td": tgt * tdpt if np.isfinite(tdpt) else 0.0}
    elif pos == "RB":
        att = _ewma(h["carries"].to_numpy())
        ypc = _ewma((h.rushing_yards / g.carries).to_numpy())
        tdpc = _ewma((h.rushing_tds / g.carries).to_numpy())
        rec = _ewma(h["receptions"].to_numpy())
        recyd = _ewma(h["receiving_yards"].to_numpy())
        line = {"rush_yds": att * ypc if np.isfinite(ypc) else 0.0,
                "rush_td": att * tdpc if np.isfinite(tdpc) else 0.0,
                "rec": rec if np.isfinite(rec) else 0.0,
                "rec_yds": recyd if np.isfinite(recyd) else 0.0,
                "rec_td": 0.03 * (rec if np.isfinite(rec) else 0.0)}
    else:  # QB
        att = _ewma(h["attempts"].to_numpy())
        ypa = _ewma((h.passing_yards / g.attempts).to_numpy())
        tdpa = _ewma((h.passing_tds / g.attempts).to_numpy())
        intpa = _ewma((h.passing_interceptions / g.attempts).to_numpy())
        rushy = _ewma(h["rushing_yards"].to_numpy())
        rusht = _ewma(h["rushing_tds"].to_numpy())
        line = {"pass_yds": att * ypa if np.isfinite(ypa) else 0.0,
                "pass_td": att * tdpa if np.isfinite(tdpa) else 0.0,
                "int": att * intpa if np.isfinite(intpa) else 0.0,
                "rush_yds": rushy if np.isfinite(rushy) else 0.0,
                "rush_td": rusht if np.isfinite(rusht) else 0.0}
    return D.dk_points_from_line(line)


def run() -> pd.DataFrame:
    pw = M._pw_with_team()
    hist_by_id = {pid: g.sort_values(["season", "week"]) for pid, g in pw.groupby("player_id")}
    t0 = time.time()
    rows = []
    for season in TEST_SEASONS:
        sl = pw[(pw.season == season) & (pw.week.isin(TEST_WEEKS)) & (pw.position.isin(POS))]
        # require some involvement so we're scoring real fantasy options
        sl = sl[(sl.targets + sl.carries + sl.attempts) >= 3]
        for _, r in sl.iterrows():
            pid, pos, wk = r["player_id"], r["position"], int(r["week"])
            g = hist_by_id.get(pid)
            if g is None:
                continue
            h = g[(g.season < season) | ((g.season == season) & (g.week < wk))].tail(_LB)
            if len(h) < 3:
                continue
            naive = _ewma(h["dk_fp"].to_numpy())
            cur = _current_line(h, pos)
            o = M.opp_line(pid, pos, season, wk, by="id")
            opp = o.get("dk_fp")
            if not (np.isfinite(naive) and np.isfinite(cur) and opp is not None and np.isfinite(opp)):
                continue
            rows.append(dict(season=season, week=wk, player=r["player_display_name"],
                             pos=pos, team=r["team"], actual=float(r["dk_fp"]),
                             naive=float(naive), current=float(cur), opp=float(opp)))
        print(f"  {season}: {len(rows)} rows ({time.time()-t0:.0f}s)")
    return pd.DataFrame(rows)


def _fit_blend(train: pd.DataFrame) -> dict:
    """Per-position w in [0,1] minimising RMSE of w*opp + (1-w)*naive on the training rows."""
    w = {}
    for p in POS:
        g = train[train.pos == p]
        if len(g) < 50:
            w[p] = 0.5
            continue
        d = (g.opp - g.naive).to_numpy()
        y = (g.actual - g.naive).to_numpy()
        denom = float(np.dot(d, d))
        w[p] = float(np.clip(np.dot(d, y) / denom, 0.0, 1.0)) if denom > 0 else 0.5
    return w


def _apply_blend(df: pd.DataFrame, w: dict) -> pd.Series:
    ww = df.pos.map(w).astype(float)
    return ww * df.opp + (1 - ww) * df.naive


def _rmse(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2))) if m.any() else np.nan


def _mae(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.mean(np.abs(a[m] - b[m]))) if m.any() else np.nan


def _spearman(a, b):
    s = pd.DataFrame({"a": a, "b": b}).dropna()
    return float(s["a"].corr(s["b"], method="spearman")) if len(s) > 5 else np.nan


def _weekly_topk(df: pd.DataFrame, col: str, pos: str) -> tuple[float, float]:
    """Mean actual DK pts of the model's weekly top-K, and overlap with the actual top-K."""
    k = TOPK[pos]
    means, hits = [], []
    for (_, _), g in df[df.pos == pos].groupby(["season", "week"]):
        if len(g) < k:
            continue
        pick = g.nlargest(k, col)
        best = set(g.nlargest(k, "actual").index)
        means.append(pick["actual"].mean())
        hits.append(len(set(pick.index) & best) / k)
    return (float(np.mean(means)) if means else np.nan,
            float(np.mean(hits)) if hits else np.nan)


def _acc_table(df: pd.DataFrame, cols: list[str]) -> list[str]:
    head = "| slice | n | " + " | ".join(f"RMSE {c}" for c in cols) + " | " + \
           " | ".join(f"rho {c}" for c in cols) + " |"
    L = [head, "|---|--:|" + "--:|" * (2 * len(cols))]
    for label, sub in [("ALL", df)] + [(p, df[df.pos == p]) for p in POS]:
        if sub.empty:
            continue
        rmses = " | ".join(f"{_rmse(sub[c], sub.actual):.2f}" for c in cols)
        rhos = " | ".join(f"{_spearman(sub[c], sub.actual):.3f}" for c in cols)
        L.append(f"| {label} | {len(sub)} | {rmses} | {rhos} |")
    return L


def report(df: pd.DataFrame) -> str:
    train = df[df.season.isin(TRAIN_SEASONS)]
    hold = df[df.season == HOLDOUT_SEASON]
    w = _fit_blend(train)
    df = df.copy()
    df["blend"] = _apply_blend(df, w)
    hold = df[df.season == HOLDOUT_SEASON]

    L = ["# Opportunity model - walk-forward backtest", ""]
    L.append(f"- rows: **{len(df)}**  ({len(train)} train {TRAIN_SEASONS} / {len(hold)} "
             f"holdout {HOLDOUT_SEASON})  ·  weeks {TEST_WEEKS.start}-{TEST_WEEKS.stop-1}  ·  DK scoring")
    L.append("- **naive** = EWMA trailing DK pts · **current** = trailing usage x efficiency "
             "(`project_stats` port) · **opp** = opportunity model · **blend** = "
             "per-position naive/opp mix, weights fit on train only")
    L.append(f"- fitted opp weight in blend: " + ", ".join(f"{p} {w[p]:.2f}" for p in POS))
    L.append("")
    L.append("## Accuracy - all test rows (lower RMSE, higher rho better)")
    L += _acc_table(df, ["naive", "current", "opp", "blend"])
    L.append("")
    L.append(f"## Accuracy - {HOLDOUT_SEASON} holdout only (blend weights were NOT fit on this)")
    L += _acc_table(hold, ["naive", "current", "opp", "blend"])
    L.append("")
    L.append("## Weekly top-K by projection - mean actual DK pts of each model's weekly top-K")
    L.append("(K: " + ", ".join(f"{k}={v}" for k, v in TOPK.items()) + "; higher = better)")
    L.append("")
    L.append("| pos | naive | current | opp | blend | overlap naive | opp | blend |")
    L.append("|---|--:|--:|--:|--:|--:|--:|--:|")
    for p in POS:
        mn, hn = _weekly_topk(df, "naive", p)
        mc, _ = _weekly_topk(df, "current", p)
        mo, ho = _weekly_topk(df, "opp", p)
        mb, hb = _weekly_topk(df, "blend", p)
        L.append(f"| {p} | {mn:.2f} | {mc:.2f} | {mo:.2f} | **{mb:.2f}** | {hn:.0%} | {ho:.0%} | **{hb:.0%}** |")
    L.append("")
    # verdict on the holdout
    r = {c: _rmse(hold[c], hold.actual) for c in ("naive", "current", "opp", "blend")}
    rho = {c: _spearman(hold[c], hold.actual) for c in ("naive", "current", "opp", "blend")}
    best_base = min(r["naive"], r["current"])
    if r["blend"] < best_base - 0.03 and rho["blend"] >= rho["naive"] - 0.003:
        v = ("**blend beats the baselines on the holdout** - wire the opportunity model in as "
             "a per-position blend with the trailing average.")
    elif r["opp"] < best_base - 0.03:
        v = "**opp alone beats the baselines on the holdout** - worth wiring in."
    else:
        v = ("**marginal** - the opportunity model helps RB/TE but only ties overall. Bigger "
             "lever next: injuries/inactives + depth-chart redistribution (opportunity spikes).")
    L.append(f"## Verdict ({HOLDOUT_SEASON} holdout): {v}")
    L.append("")
    L.append("RMSE   " + " · ".join(f"{k} {v:.2f}" for k, v in r.items()))
    L.append("rho    " + " · ".join(f"{k} {v:.3f}" for k, v in rho.items()))
    return "\n".join(L)


def main():
    df = run()
    if df.empty:
        print("no rows — check the nflverse cache (`python -m matchup_model.opp.data`)")
        return
    df.to_csv(ROWS_CSV, index=False)
    txt = report(df)
    REPORT.write_text(txt, encoding="utf-8")
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("\n" + txt + f"\n\nwrote {REPORT}")


if __name__ == "__main__":
    main()
