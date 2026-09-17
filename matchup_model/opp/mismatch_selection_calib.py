"""Does picking the biggest-LOOKING situational mismatch -- the way the Data Suite's
Matchups page surfaces a handful of "Top Matchup Edges" per game -- actually flag the
player-games where the baseline projection is most wrong? Tests the SELECTION mechanism
itself, pooling the three opponent-conditioning axes already built and independently
found null on their own (scheme_calib.py man/zone, box_calib.py box count,
shell_calib.py 1-high/2-high shell) -- each is a real, walk-forward, EB-shrunk
defense-tendency x player-edge signal for a different position:

    scheme  WR/TE  man-vs-zone YPRR split      x  defense MAN % dev
    box     RB     light-vs-stacked-box YPC    x  defense BOX dev
    shell   QB     1-high-vs-2-high YPA split  x  defense 2-HIGH % dev

Each axis's OWN report already found no aggregate improvement (fitting one beta across
its whole population). The open question here is narrower and different: even if no
single axis moves the needle everywhere, does the subset of player-games where a signal
looks unusually large -- pooled across axes and positions, exactly like the Matchups
page mixing "run game" and "deep passing" edges on the same scorecard -- actually
correspond to bigger baseline misses? If mismatch size doesn't track error size at all,
"build a system that hunts for the biggest mismatch each week" has nothing to stand on,
regardless of which axis eventually looks most extreme in a given game.

Method: normalize each axis's |signal| to a percentile WITHIN that axis's own holdout
distribution (axes are on incommensurable scales -- shell signal maxes out ~0.4, box
~1.2 -- so raw magnitude comparison across axes would just reflect scaling, not real
mismatch size). Pool all three axes' 2025 holdout rows. Then:

  1. correlation(mismatch percentile, |base error|) -- pooled and per-axis. This is the
     foundational check: does a bigger cross-axis-normalized mismatch predict a bigger
     baseline miss at all?
  2. Take the top-K% of rows by mismatch percentile each week (the "scorecard" subset --
     whichever unit matchup is most extreme that week, regardless of position) and
     compare RMSE improvement (each row's own axis-fitted beta) in that subset against
     1,000 random same-size subsets drawn from the same pool, to see whether the real
     selection beats chance rather than just eyeballing one number.

Reuses the three existing calibration scripts' already-saved, already walk-forward-safe
rows CSVs (opp_scheme/box/shell_calibration_rows.csv) -- no new data pull, no re-fit of
the underlying shrinkage models. This script only adds the cross-axis selection layer.
Deliberately NOT wired into blend.py / the live model -- build-and-test only.

    python -m matchup_model.opp.mismatch_selection_calib
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HOLDOUT_SEASON = 2025
N_BOOT = 1000
TOPK_PCTS = [0.10, 0.20, 0.30]
RNG_SEED = 0

REPORT = ROOT / "opp_mismatch_selection.md"


def _rmse(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2))) if m.any() else np.nan


def _fit_beta(base, actual, signal) -> float:
    d = (np.asarray(base, float) * np.asarray(signal, float))
    y = (np.asarray(actual, float) - np.asarray(base, float))
    denom = float(np.dot(d, d))
    return float(np.dot(d, y) / denom) if denom > 0 else 0.0


def _load_axis(name: str, path: Path, train_seasons: list[int], signal_col: str,
               pos_fixed: str | None) -> pd.DataFrame:
    d = pd.read_csv(path)
    if signal_col != "signal":
        d = d.rename(columns={signal_col: "signal"})
    if pos_fixed is not None:
        d["pos"] = pos_fixed
    d = d[["season", "week", "player", "pos", "defense", "base", "signal", "actual"]].dropna(
        subset=["base", "signal", "actual"])
    train = d[d.season.isin(train_seasons)]
    beta = _fit_beta(train.base, train.actual, train.signal)
    hold = d[d.season == HOLDOUT_SEASON].copy()
    hold["axis"] = name
    hold["beta"] = beta
    hold["adjusted"] = hold.base * (1 + beta * hold.signal)
    return hold


def load_pooled() -> pd.DataFrame:
    axes = [
        _load_axis("scheme", ROOT / "opp_scheme_calibration_rows.csv",
                  [2022, 2023, 2024], "signal_v2", None),
        _load_axis("box", ROOT / "opp_box_calibration_rows.csv",
                  [2023, 2024], "signal", "RB"),
        _load_axis("shell", ROOT / "opp_shell_calibration_rows.csv",
                  [2023, 2024], "signal", "QB"),
    ]
    df = pd.concat(axes, ignore_index=True)
    df["abs_signal"] = df["signal"].abs()
    df["mismatch_pctile"] = df.groupby("axis")["abs_signal"].rank(pct=True)
    df["abs_err_base"] = (df.actual - df.base).abs()
    df["abs_err_adjusted"] = (df.actual - df.adjusted).abs()
    return df


def _topk_rmse_delta(df: pd.DataFrame, mask: pd.Series) -> float:
    sub = df[mask]
    if sub.empty:
        return np.nan
    return _rmse(sub.base, sub.actual) - _rmse(sub.adjusted, sub.actual)


def report(df: pd.DataFrame) -> str:
    L = ["# Does the biggest cross-axis mismatch flag the biggest baseline misses?", ""]
    L.append(f"- Pooled 2025 holdout: {len(df)} player-games across 3 axes -- "
             f"{(df.axis=='scheme').sum()} WR/TE (man/zone), "
             f"{(df.axis=='box').sum()} RB (box count), "
             f"{(df.axis=='shell').sum()} QB (1-high/2-high shell).")
    L.append("- `mismatch_pctile` = |signal| ranked to a 0-1 percentile WITHIN each "
             "axis's own holdout distribution, so a 'big' shell mismatch and a 'big' box "
             "mismatch are comparable -- pooling raw |signal| across axes would just "
             "reflect each axis's arbitrary scale, not real mismatch size.")
    L.append("- Each axis keeps its own beta, fit on ONLY that axis's designated train "
             "seasons (matches scheme_calib.py/box_calib.py/shell_calib.py exactly) -- "
             "nothing here re-fits or leaks holdout into training.")
    L.append("")

    L.append("## 1. Does mismatch size predict baseline error at all?")
    L.append("")
    L.append("| axis | n | corr(mismatch_pctile, |base error|) |")
    L.append("|---|--:|--:|")
    for ax in ["scheme", "box", "shell"]:
        sub = df[df.axis == ax]
        c = sub[["mismatch_pctile", "abs_err_base"]].corr().iloc[0, 1]
        L.append(f"| {ax} | {len(sub)} | {c:+.4f} |")
    pooled_c = df[["mismatch_pctile", "abs_err_base"]].corr().iloc[0, 1]
    L.append(f"| **pooled** | {len(df)} | **{pooled_c:+.4f}** |")
    L.append("")
    L.append("A meaningfully positive pooled correlation is the minimum bar for the "
             "whole idea to have anywhere to stand -- if bigger mismatches don't even "
             "correlate with bigger baseline misses, no selection rule built on top of "
             "these signals can help, independent of which axis wins in a given week.")
    L.append("")

    L.append("## 2. Selecting the top-K% biggest mismatch each week, vs. random selection")
    L.append("")
    L.append("Real DFS use only ever looks at a handful of marquee edges per week, not "
             "the whole slate -- this simulates that: take the top-K% of ALL pooled rows "
             "by mismatch_pctile *within each week* (mixing positions/axes, same as the "
             "Matchups page mixing 'run game' and 'deep passing' on one scorecard), and "
             "check the RMSE improvement (base vs. each row's own axis-beta-adjusted "
             "projection) inside that subset against 1,000 random same-size draws from "
             "the same week's pool.")
    L.append("")
    L.append("| top-K% | n selected | RMSE Δ (selected) | random draws: mean Δ | "
             "random draws: p95 Δ | percentile of real vs random |")
    L.append("|---|--:|--:|--:|--:|--:|")

    rng = np.random.default_rng(RNG_SEED)
    weeks = df["week"].unique()
    for pct in TOPK_PCTS:
        sel_mask = pd.Series(False, index=df.index)
        for wk in weeks:
            wk_idx = df.index[df.week == wk]
            n_take = max(1, int(round(len(wk_idx) * pct)))
            top_idx = df.loc[wk_idx].nlargest(n_take, "mismatch_pctile").index
            sel_mask.loc[top_idx] = True
        real_delta = _topk_rmse_delta(df, sel_mask)
        n_sel = int(sel_mask.sum())

        boot_deltas = np.empty(N_BOOT)
        for b in range(N_BOOT):
            rand_mask = pd.Series(False, index=df.index)
            for wk in weeks:
                wk_idx = df.index[df.week == wk]
                n_take = max(1, int(round(len(wk_idx) * pct)))
                pick = rng.choice(wk_idx, size=min(n_take, len(wk_idx)), replace=False)
                rand_mask.loc[pick] = True
            boot_deltas[b] = _topk_rmse_delta(df, rand_mask)
        boot_deltas = boot_deltas[np.isfinite(boot_deltas)]
        pctile_of_real = float((boot_deltas < real_delta).mean()) if len(boot_deltas) else np.nan
        L.append(f"| top {pct:.0%} | {n_sel} | **{real_delta:+.3f}** | "
                 f"{boot_deltas.mean():+.3f} | {np.percentile(boot_deltas, 95):+.3f} | "
                 f"{pctile_of_real:.0%} |")
    L.append("")
    L.append("'Percentile of real vs random' near 50% means picking the biggest-looking "
             "mismatch each week did no better than picking a random same-size handful "
             "of player-games -- the selection itself carries no information. It would "
             "need to sit up near 90-95%+ to call the selection mechanism real (matching "
             "the standard for the random draws' own p95 line).")
    L.append("")

    real_20 = _topk_rmse_delta(df, df.groupby("week")["mismatch_pctile"]
                               .rank(pct=True, ascending=False) <= 0.20)
    if pooled_c > 0.05 and real_20 > 0.3:
        v = "**real: mismatch size predicts error, and selecting on it helps**"
    elif pooled_c > 0.05:
        v = "**mixed: mismatch size weakly predicts error, but selection doesn't beat random**"
    else:
        v = "**no: mismatch size doesn't predict baseline error, selection is noise**"
    L.append(f"Verdict: {v}")
    return "\n".join(L)


def main():
    df = load_pooled()
    if df.empty:
        print("no rows"); return
    txt = report(df)
    REPORT.write_text(txt, encoding="utf-8")
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("\n" + txt + f"\n\nwrote {REPORT}")


if __name__ == "__main__":
    main()
