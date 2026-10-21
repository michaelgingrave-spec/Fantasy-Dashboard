"""Production-pipeline exam of projection changes on the real graded lines.

    python -m matchup_model.opp.restate          # writes matchup_model/restate_2026_report.md

Every graded 2026 line in line_history.csv is pushed through `dfs.props.edges_from_raw` -- the exact code the Prop Edges screen
runs (blend with the trailing average, FantasyPoints nudge, z, tier, role_ratio) -- once per model version: v1 (the original
constants and hand-set team volume), v2 (the 2026-10-15 WR/TE recalibration, still on the hand-set volume) and the current model in
matchup_model/opp/model.py. The leans are then scored against the real results and the tier tables rebuilt. Every constant of v1-v3 was fixed
from 2023-25 box scores before it met a 2026 line, so nothing there was fitted on 2026; v4's injury scales (model.INJ_REDIST) were fitted on
2022-26 box scores, a small share of which is 2026 (injury_absorption_report.md shows the fit with 2026 left out), so v4's lift here is not a
clean out-of-sample test. v1 reproducing what the screen stored when the lines were pulled is the check that the harness is sound. Prices are
not stored with each line, so ROI is the flat -110 one the tier tables use.
"""
from __future__ import annotations

import copy
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from dfs import bets, props
from matchup_model.opp import model as M
from matchup_model.opp.open_vs_close import paired_gain
from matchup_model.opp.receiving_calibration import BEFORE, RB_KEYS, V2, V3     # the constants of v1, v2 and v3, exactly as they were

REPORT = Path(__file__).resolve().parents[1] / "restate_2026_report.md"
KEY_OF = {props._NICE[comp]: key for key, comp in props.MARKETS.items()}      # "rec yds" -> "player_reception_yds"
RECEIVING = ["rec yds", "receptions"]
BE = {"lean": 52.4, "solid": 53.0, "strong": 53.7, "high": 55.4}                 # break-even win % per tier at the typical price


def shipped() -> dict:
    return dict(K_SHARE_REC=M.K_SHARE_REC, K_YPT=M.K_YPT, DEF_TGT_B=M.DEF_TGT_B, REC_SCALE=M.REC_SCALE,
                TEAM_VOLUME_MODEL=M.TEAM_VOLUME_MODEL, INJ_REDIST=copy.deepcopy(M.INJ_REDIST), **{k: getattr(M, k) for k in RB_KEYS})


def _set(params: dict) -> None:
    for k, v in params.items():
        setattr(M, k, v)
    M.opp_line.cache_clear(); M.def_target_rel.cache_clear(); M.team_volume.cache_clear()


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


def _restate_job(lh: pd.DataFrame, params: dict) -> pd.DataFrame:
    return restate(lh, params)


def restate_versions(lh: pd.DataFrame, versions: dict[str, dict]) -> dict[str, pd.DataFrame]:
    """`restate` for every version, one worker process each (the model's constants are module-global)."""
    with ProcessPoolExecutor(max_workers=len(versions)) as ex:
        futs = {name: ex.submit(_restate_job, lh, p) for name, p in versions.items()}
        return {name: f.result() for name, f in futs.items()}


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


def _differs(frames: dict[str, pd.DataFrame], rng: np.random.Generator) -> list[str]:
    """Where the current version takes a different view from the one before it, chosen without looking at results: the lines whose projection moved, or whose lean
    flipped. Everywhere else the two versions bet identically, so the headline paired difference is diluted by lines that cannot differ."""
    names = list(frames)
    prev, current = names[-2], names[-1]
    m = frames[prev].merge(frames[current], on=["week", "player", "market"], suffixes=("_o", "_n"))
    ratio = m["our proj_n"] / m["our proj_o"].where(m["our proj_o"].abs() > 1e-9)
    m = m.assign(moved=(ratio / ratio.groupby(m.market).transform("median") - 1).abs().fillna(0.0))     # against the typical change in that market: a recalibration moves every line alike
    subsets = (("all lines", m), ("projection moved >= 1%", m[m.moved >= 0.01]), ("projection moved >= 5%", m[m.moved >= 0.05]),
               ("projection moved >= 10%", m[m.moved >= 0.10]), (f"lean flipped by {current}", m[m.lean_o != m.lean_n]), ("projection moved < 1%", m[m.moved < 0.01]))
    L = ["", f"## Where {current} differs from {prev}", "",
         "Lines picked by what the model did, not by how they turned out. `moved` is the change in our projection relative to the typical change for that market (so a "
         "recalibration that shifts every line alike is not a move); where the lean flips one version wins and the other loses.", "",
         f"| lines | n | {prev} win | {current} win | paired difference (player-clustered 95% CI) | OVER share {prev} -> {current} |", "|---|--:|--:|--:|---|---|"]
    for label, sub in subsets:
        if sub.empty:
            L.append(f"| {label} | 0 | | | | |"); continue
        old, new = sub[sub.result_o != "push"], sub[sub.result_n != "push"]
        push = (sub.result_o == "push") | (sub.result_n == "push")
        est, lo, hi, n = paired_gain(sub.result_n == "win", sub.result_o == "win", push, sub.player, rng)
        L.append(f"| {label} | {len(sub)} | {100 * (old.result_o == 'win').mean():.1f}% | {100 * (new.result_n == 'win').mean():.1f}% | "
                 + (f"{100 * est:+.2f} pts [{100 * lo:+.2f}, {100 * hi:+.2f}] (n={n})" if n else "") + f" | {100 * (sub.lean_o == 'OVER').mean():.0f}% -> {100 * (sub.lean_n == 'OVER').mean():.0f}% |")
    return L


def build_report(lh: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> str:
    """`frames` maps a version label to its restated frame, oldest first; the last one is the current model."""
    rng = np.random.default_rng(11)
    names = list(frames)
    first, current = names[0], names[-1]
    chk = frames[first].merge(lh[["week", "player", "market", "our proj", "lean", "conf", "role_ratio"]],
                              on=["week", "player", "market"], suffixes=("", "_stored"))
    L = ["# Production-pipeline exam: model versions on the 2026 graded lines", "",
         f"{len(lh)} graded lines, weeks {int(lh.week.min())}-{int(lh.week.max())}, run through `dfs.props.edges_from_raw` under each version: "
         + "; ".join(names) + ". Every constant of v1-v3 was fixed from 2023-25 box scores; nothing there was fitted on 2026. v4's injury scales (`INJ_REDIST`) "
         "were fitted on 2022-26 box scores, of which 2026 is a small share (`injury_absorption_report.md` lists the fit with each season left out, 2026 included), so v4's "
         "lift on these lines is not a clean out-of-sample test. ROI is at a flat -110 (prices are not stored per line).", "",
         f"**Harness check** -- {first} reproduces what the screen stored when the lines were pulled: "
         f"our proj within 0.15 on {100 * ((chk['our proj'] - chk['our proj_stored']).abs() <= 0.15).mean():.0f}% of rows, same lean "
         f"{100 * (chk.lean == chk.lean_stored).mean():.0f}%, same tier {100 * (chk.conf == chk.conf_stored).mean():.0f}%, "
         f"role_ratio within .02 {100 * ((chk.role_ratio - chk.role_ratio_stored).abs().fillna(0) <= 0.02).mean():.0f}% "
         "(the rest is injury / data drift since the pull).", "",
         "## Win rate of the leans", "",
         "| lines | bets | win | ROI at -110 | OVER share | OVER win | UNDER win |", "|---|--:|--:|--:|--:|--:|--:|"]
    for label, mask in (("all markets", slice(None)), ("receiving (rec yds + receptions)", None)):
        for name in names:
            d = frames[name]
            sub = d if mask is not None else d[d.market.isin(RECEIVING)]
            L.append(_summary(sub, f"{label}, {name}"))
    for mk in ["rec yds", "receptions", "rush yds", "pass yds", "pass TD"]:
        for name in names:
            L.append(_summary(frames[name][frames[name].market == mk], f"{mk}, {name}"))

    L += ["", f"## Paired difference ({current} minus each earlier version), same lines, player-clustered 95% CI", ""]
    for other in names[:-1]:
        m = frames[other].merge(frames[current], on=["week", "player", "market"], suffixes=("_o", "_n"))
        for label, sub in (("all markets", m), ("receiving markets", m[m.market.isin(RECEIVING)]),
                           ("rush yds", m[m.market == "rush yds"])):
            push = (sub.result_o == "push") | (sub.result_n == "push")
            est, lo, hi, n = paired_gain(sub.result_n == "win", sub.result_o == "win", push, sub.player, rng)
            L.append(f"- vs {other}, {label}: " + (f"**{100 * est:+.2f} pts** (CI {100 * lo:+.2f} to {100 * hi:+.2f}, n={n})" if n else "no lines"))
    L += ["", "Win % on receiving markets by week:", "", "| week | " + " | ".join(names) + " | lines |", "|--:|" + "--:|" * (len(names) + 1)]
    wk_rows = {n: frames[n][frames[n].market.isin(RECEIVING)] for n in names}
    for wk in sorted(wk_rows[current].week.unique()):
        cells = []
        for n in names:
            g = wk_rows[n][(wk_rows[n].week == wk) & (wk_rows[n].result != "push")]
            cells.append(f"{100 * (g.result == 'win').mean():.1f}%")
        L.append(f"| {int(wk)} | " + " | ".join(cells) + f" | {int((wk_rows[current].week == wk).sum())} |")

    if len(names) > 1:
        L += _differs(frames, rng)

    L += ["", "## Tier tables (all graded lines, win % and ROI at -110)", "",
          "Break-even win % by tier at the usual price: " + ", ".join(f"{k} {v}" for k, v in BE.items()) + ".", ""]
    tables = [("As stored (what the screen showed)", lh)] + [(f"{n}, restated", frames[n]) for n in names[1:]]
    for label, d in tables:
        L += [f"**{label}**", "", "All bets:", ""] + _md(bets.line_history_buckets(d)) + ["", "Recommended only:", ""]
        L += _md(bets.line_history_buckets(d, recommended_only=True)) + [""]

    L += ["## Filters and role agreement", ""]
    for name in names:
        d = frames[name]
        vis = props.confident_only(d)
        L.append(f"- {name}: {len(d)} rows -> {len(vis)} pass the always-on guard -> {len(props.role_stable(vis))} Recommended; "
                 f"role_ratio median {d.role_ratio.median():.2f}, {100 * d.role_ratio.between(.85, 1.20).mean():.0f}% inside the "
                 "0.85-1.20 band")
    L.append("")
    for name in names:
        L.append(f"- OVER share by market, {name}: " + ", ".join(
            f"{mk} {100 * (g.lean == 'OVER').mean():.0f}%" for mk, g in frames[name].groupby("market") if not mk.endswith("att")))
    return "\n".join(L) + "\n"


def main() -> None:
    lh = graded_lines()
    versions = {"v1 (original)": BEFORE, "v2 (WR/TE recalibration)": V2, "v3 (fitted team volume, full injury boost)": V3, f"v{M.MODEL_VERSION} (current)": shipped()}
    frames = restate_versions(lh, versions)
    txt = build_report(lh, frames)
    REPORT.write_text(txt, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(txt + f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
