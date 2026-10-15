"""Production-pipeline exam of a projection change on the real graded lines.

    python -m matchup_model.opp.restate          # writes matchup_model/restate_2026_report.md

Every graded 2026 line in line_history.csv is pushed through `dfs.props.edges_from_raw` -- the exact code the Prop Edges screen
runs (blend with the trailing average, FantasyPoints nudge, z, tier, role_ratio) -- twice: with the projection constants from
BEFORE the WR/TE receiving recalibration and with the ones now in matchup_model/opp/model.py. The leans are then scored against
the real results and the tier tables rebuilt. The shipped constants were fixed from 2023-25 box scores, so nothing here was
fitted on 2026; the BEFORE run reproducing what was stored when the lines were pulled is the check that the harness is sound.
Prices are not stored with each line, so ROI is the flat -110 one the tier tables use.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from dfs import bets, props
from matchup_model.opp import model as M
from matchup_model.opp.open_vs_close import paired_gain
from matchup_model.opp.receiving_calibration import BEFORE      # every constant the recalibration touched, as it was

REPORT = Path(__file__).resolve().parents[1] / "restate_2026_report.md"
KEY_OF = {props._NICE[comp]: key for key, comp in props.MARKETS.items()}      # "rec yds" -> "player_reception_yds"
RECEIVING = ["rec yds", "receptions"]
BE = {"lean": 52.4, "solid": 53.0, "strong": 53.7, "high": 55.4}                 # break-even win % per tier at the typical price


def shipped() -> dict:
    return dict(K_SHARE_REC=M.K_SHARE_REC, K_YPT=M.K_YPT, DEF_TGT_B=M.DEF_TGT_B, REC_SCALE=M.REC_SCALE)


def _set(params: dict) -> None:
    for k, v in params.items():
        setattr(M, k, v)
    M.opp_line.cache_clear(); M.def_target_rel.cache_clear()


def graded_lines(season: int = 2026) -> pd.DataFrame:
    lh = bets.load_line_history()
    lh = lh[(lh.season == season) & lh.result.isin(["win", "loss", "push"]) & lh.actual.notna()]
    return lh.drop_duplicates(["week", "player", "market"]).reset_index(drop=True)


def raw_for(rows: pd.DataFrame) -> dict:
    """A minimal Odds API payload with one Over/Under pair per (player, market) at that row's stored line."""
    mk: dict = {}
    for r in rows.itertuples():
        m = mk.setdefault(KEY_OF[r.market], {"key": KEY_OF[r.market], "outcomes": []})
        m["outcomes"] += [{"name": "Over", "description": r.player, "price": -110, "point": float(r.line)},
                          {"name": "Under", "description": r.player, "price": -110, "point": float(r.line)}]
    return {"bookmakers": [{"key": "draftkings", "markets": list(mk.values())}]}


def restate(lh: pd.DataFrame, params: dict) -> pd.DataFrame:
    """The screen's edge table for every graded line under `params`, with the result of each lean; constants restored after."""
    keep = shipped()
    try:
        _set(params)
        frames = []
        for wk, g in lh.groupby("week"):
            e = props.edges_from_raw(raw_for(g), int(wk))
            frames.append(e.assign(week=int(wk)))
    finally:
        _set(keep)
    e = pd.concat(frames, ignore_index=True)
    d = e.merge(lh[["week", "player", "market", "line", "actual"]].rename(columns={"line": "line_h"}),
                on=["week", "player", "market"], how="inner")
    d = d.assign(line=d.line_h).drop(columns="line_h")
    d["result"] = np.where(d.actual == d.line, "push", np.where((d.actual > d.line) == (d.lean == "OVER"), "win", "loss"))
    return d


def _roi(p: float) -> float:
    return 100 * (p * 100 / 110 - (1 - p))


def _md(df: pd.DataFrame) -> list[str]:
    if df is None or df.empty:
        return ["(none)"]
    out = ["| " + " | ".join(df.columns) + " |", "|" + "|".join("---" for _ in df.columns) + "|"]
    out += ["| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return out


def _summary(d: pd.DataFrame, label: str) -> str:
    g = d[d.result != "push"]
    if g.empty:
        return f"| {label} | 0 | | | | | |"
    w, ov = (g.result == "win").mean(), g.lean == "OVER"
    ow = 100 * (g[ov].result == "win").mean() if ov.any() else float("nan")
    uw = 100 * (g[~ov].result == "win").mean() if (~ov).any() else float("nan")
    return (f"| {label} | {len(g)} | {100 * w:.1f}% | {_roi(w):+.1f}% | {100 * ov.mean():.0f}% | "
            f"{ow:.1f}% | {uw:.1f}% |")


def build_report(lh: pd.DataFrame, before: pd.DataFrame, after: pd.DataFrame) -> str:
    rng = np.random.default_rng(11)
    chk = before.merge(lh[["week", "player", "market", "our proj", "lean", "conf", "role_ratio"]],
                       on=["week", "player", "market"], suffixes=("", "_stored"))
    L = ["# Production-pipeline exam: previous projection constants vs the recalibrated ones, 2026 graded lines", "",
         f"{len(lh)} graded lines, weeks {int(lh.week.min())}-{int(lh.week.max())}, run through `dfs.props.edges_from_raw` under each set "
         "of constants. The recalibrated constants were fixed from 2023-25 box scores; nothing was fitted on 2026. ROI is at a flat "
         "-110 (prices are not stored per line).", "",
         "**Harness check** -- the previous constants reproduce what the screen stored when the lines were pulled: "
         f"our proj within 0.15 on {100 * ((chk['our proj'] - chk['our proj_stored']).abs() <= 0.15).mean():.0f}% of rows, same lean "
         f"{100 * (chk.lean == chk.lean_stored).mean():.0f}%, same tier {100 * (chk.conf == chk.conf_stored).mean():.0f}%, "
         f"role_ratio within .02 {100 * ((chk.role_ratio - chk.role_ratio_stored).abs().fillna(0) <= 0.02).mean():.0f}% "
         "(the rest is injury / data drift since the pull).", "",
         "## Win rate of the leans", "",
         "| lines | bets | win | ROI at -110 | OVER share | OVER win | UNDER win |", "|---|--:|--:|--:|--:|--:|--:|"]
    for label, mask in (("all markets", slice(None)), ("receiving (rec yds + receptions)", None)):
        for name, d in (("previous", before), ("recalibrated", after)):
            sub = d if mask is not None else d[d.market.isin(RECEIVING)]
            L.append(_summary(sub, f"{label}, {name}"))
    for mk in ["rec yds", "receptions", "rush yds", "pass yds", "pass TD"]:
        for name, d in (("previous", before), ("recalibrated", after)):
            L.append(_summary(d[d.market == mk], f"{mk}, {name}"))

    m = before.merge(after, on=["week", "player", "market"], suffixes=("_o", "_n"))
    L += ["", "## Paired difference (recalibrated minus previous), same lines, player-clustered 95% CI", ""]
    for label, sub in (("all markets", m), ("receiving markets", m[m.market.isin(RECEIVING)])):
        push = (sub.result_o == "push") | (sub.result_n == "push")
        est, lo, hi, n = paired_gain(sub.result_n == "win", sub.result_o == "win", push, sub.player, rng)
        L.append(f"- {label}: **{100 * est:+.2f} pts** (CI {100 * lo:+.2f} to {100 * hi:+.2f}, n={n})")
    mr = m[m.market.isin(RECEIVING)]
    L += ["", "| week | receiving: previous | recalibrated | lines |", "|--:|--:|--:|--:|"]
    for wk, g in mr.groupby("week"):
        g = g[(g.result_o != "push") & (g.result_n != "push")]
        L.append(f"| {int(wk)} | {100 * (g.result_o == 'win').mean():.1f}% | {100 * (g.result_n == 'win').mean():.1f}% | {len(g)} |")

    L += ["", "## Tier tables (all graded lines, win % and ROI at -110)", "",
          f"Break-even win % by tier at the usual price: " + ", ".join(f"{k} {v}" for k, v in BE.items()) + ".", ""]
    for label, d in (("As stored (what the screen showed)", lh), ("Previous constants, restated", before),
                     ("Recalibrated constants", after)):
        L += [f"**{label}**", "", "All bets:", ""] + _md(bets.line_history_buckets(d)) + ["", "Recommended only:", ""]
        L += _md(bets.line_history_buckets(d, recommended_only=True)) + [""]

    L += ["## Filters and role agreement", ""]
    for name, d in (("previous", before), ("recalibrated", after)):
        vis = props.confident_only(d)
        L.append(f"- {name}: {len(d)} rows -> {len(vis)} pass the always-on guard -> {len(props.role_stable(vis))} Recommended; "
                 f"role_ratio median {d.role_ratio.median():.2f}, {100 * d.role_ratio.between(.85, 1.20).mean():.0f}% inside the "
                 "0.85-1.20 band")
    L.append("")
    for name, d in (("previous", before), ("recalibrated", after)):
        L.append(f"- OVER share by market, {name}: " + ", ".join(
            f"{mk} {100 * (g.lean == 'OVER').mean():.0f}%" for mk, g in d.groupby("market") if not mk.endswith("att")))
    return "\n".join(L) + "\n"


def main() -> None:
    lh = graded_lines()
    before, after = restate(lh, BEFORE), restate(lh, shipped())
    txt = build_report(lh, before, after)
    REPORT.write_text(txt, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(txt + f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
