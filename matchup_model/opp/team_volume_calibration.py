"""Fit and audit the team-volume regression (see team_volume.py for what it is and why).

    python -m matchup_model.opp.team_volume_calibration     # writes team_volume_coefs.json and matchup_model/team_volume_report.md

For each cutoff season (2022-2025) the coefficients are fit on every REG team-game up to that season. The report scores each test season
(2023-2025, weeks 4-18) with the fit whose training stops the season before -- the same code path production uses -- against: the league
average of the training seasons, the team's own trailing average, and the pre-v3 hand-set formula (reproduced here with its original
reversed spread sign) so the gain is measured on identical games.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from matchup_model.opp import model as M
from matchup_model.opp import team_volume as TV

CUTOFFS = (2022, 2023, 2024, 2025)
TEST_SEASONS = (2023, 2024, 2025)
TEST_FROM_WEEK = 4
REPORT = Path(__file__).resolve().parents[1] / "team_volume_report.md"


def legacy(team: str, season: int, week: int) -> tuple[float, float]:
    """The hand-set formula as it was before v3, reversed spread sign included (positive spread = underdog in `games()`, but the
    formula assumed positive = favourite): (pass attempts, rush attempts)."""
    ctx, v = M.team_context(team, season, week), M.vegas_row(team, season, week)
    plays, pr = ctx["plays_pg"], ctx["pass_rate"]
    if not np.isfinite(plays):
        plays = 63.0
    if not np.isfinite(pr):
        pr = 0.57
    if v.get("total_line") and np.isfinite(v["total_line"]):
        plays *= 1.0 + M.PLAYS_TOTAL_BETA * (v["total_line"] - 44.0) / 44.0
    if v.get("spread") and np.isfinite(v["spread"]):
        pr = float(np.clip(pr + M.PR_SPREAD_BETA * (-v["spread"]), ctx["pass_rate"] - M.PR_CLAMP, ctx["pass_rate"] + M.PR_CLAMP))
    pr = float(np.clip(pr, 0.30, 0.72))
    return plays * pr * (1.0 - M.SACK_FRAC), max(12.0, plays * (1.0 - pr) - M.QB_SCRAMBLE)


def training_table() -> pd.DataFrame:
    """One row per REG team-game that has features and an outcome: the model inputs, actual attempts / carries, and the legacy forecast."""
    tw = TV.D.team_weeks()
    tw = tw[tw["season_type"] == "REG"][["season", "week", "team", "attempts", "carries"]]
    rows = []
    for r in tw.itertuples(index=False):
        f = TV.features(r.team, int(r.season), int(r.week))
        if f is None:
            continue
        la, lc = legacy(r.team, int(r.season), int(r.week))
        rows.append({**f, "season": int(r.season), "week": int(r.week), "team": r.team, "att_actual": float(r.attempts),
                     "car_actual": float(r.carries), "legacy_att": la, "legacy_car": lc})
    return pd.DataFrame(rows)


def ridge(df: pd.DataFrame, cols: tuple[str, ...], target: str, lam: float = TV.RIDGE) -> dict:
    """Ridge on standardised features (missing odds rows are dropped from the fit, not imputed)."""
    d = df.dropna(subset=list(cols) + [target])
    mu, sd = d[list(cols)].mean(), d[list(cols)].std().replace(0, 1.0)
    X = ((d[list(cols)] - mu) / sd).to_numpy()
    X1 = np.column_stack([np.ones(len(X)), X])
    b = np.linalg.solve(X1.T @ X1 + lam * np.diag([0.0] + [1.0] * len(cols)), X1.T @ d[target].to_numpy())
    return {"features": list(cols), "mean": [float(x) for x in mu], "sd": [float(x) for x in sd],
            "beta": [float(x) for x in b[1:]], "intercept": float(b[0]), "n": int(len(d))}


def fit_cutoff(df: pd.DataFrame, cutoff: int) -> dict:
    tr = df[df.season <= cutoff]
    return {"train_through": cutoff, "att": ridge(tr, TV.FEATURES["att"], "att_actual"), "car": ridge(tr, TV.FEATURES["car"], "car_actual")}


def _rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def _corr(a, b) -> float:
    return float(np.corrcoef(np.asarray(a, float), np.asarray(b, float))[0, 1])


def walk_forward(df: pd.DataFrame, fits: dict) -> pd.DataFrame:
    """Test-season games with every forecast attached; each season is scored with the fit that stops the season before."""
    out = []
    for test in TEST_SEASONS:
        te = df[(df.season == test) & (df.week >= TEST_FROM_WEEK)].copy()
        fit = fits[str(test - 1)]
        tr = df[df.season < test]
        for tgt, key in (("att", "att_actual"), ("car", "car_actual")):
            te[f"fit_{tgt}"] = [TV.apply_fit(fit, row, tgt) for row in te.to_dict("records")]
            te[f"const_{tgt}"] = tr[key].mean()
            te[f"trail_{tgt}"] = te[f"tr_{tgt}"]
        out.append(te)
    return pd.concat(out, ignore_index=True)


def build_report(df: pd.DataFrame, fits: dict, wf: pd.DataFrame) -> str:
    L = ["# Team pass / rush attempts: the fitted volume model vs the pre-v3 hand-set formula", "",
         f"{len(wf)} team-games, 2023-25 weeks {TEST_FROM_WEEK}-18, each season scored with coefficients fit on the seasons before it "
         f"(training {int(df.season.min())} on; {len(df)} team-games in all).", "",
         "| target | model | RMSE | correlation | mean error |", "|---|---|--:|--:|--:|"]
    names = {"const": "league average", "trail": "team's own trailing average", "legacy": "pre-v3 hand-set formula", "fit": "fitted regression (v3)"}
    for tgt, label in (("att", "pass attempts"), ("car", "rush attempts")):
        key = f"{tgt}_actual"
        for m in ("const", "trail", "legacy", "fit"):
            col = f"{m}_{tgt}"
            corr = "n/a" if m == "const" else f"{_corr(wf[col], wf[key]):.3f}"
            L.append(f"| {label} | {names[m]} | {_rmse(wf[col], wf[key]):.3f} | {corr} | {np.mean(wf[col] - wf[key]):+.2f} |")
    for tgt, label in (("att", "pass attempts"), ("car", "rush attempts")):
        key = f"{tgt}_actual"
        gain = 100 * (_rmse(wf[f"fit_{tgt}"], wf[key]) / _rmse(wf[f"legacy_{tgt}"], wf[key]) - 1)
        vs_const = 100 * (_rmse(wf[f"fit_{tgt}"], wf[key]) / _rmse(wf[f"const_{tgt}"], wf[key]) - 1)
        slope = np.polyfit(wf[f"legacy_{tgt}"], wf[key], 1)[0]
        L.append("")
        L.append(f"- {label}: fitted vs hand-set {gain:+.1f}% RMSE, vs league average {vs_const:+.1f}%; the hand-set projections were "
                 f"too spread out (actual moved {slope:.2f} as much as projected).")
    L += ["", "By season (RMSE, hand-set formula -> fitted):", "", "| season | pass attempts | rush attempts |", "|---|--:|--:|"]
    for s, g in wf.groupby("season"):
        L.append(f"| {s} | {_rmse(g.legacy_att, g.att_actual):.2f} -> {_rmse(g.fit_att, g.att_actual):.2f} | "
                 f"{_rmse(g.legacy_car, g.car_actual):.2f} -> {_rmse(g.fit_car, g.car_actual):.2f} |")

    wf = wf.assign(bucket=pd.cut(wf.spread, [-30, -7, -3, 0, 3, 7, 30], labels=["fav 7+", "fav 3-7", "fav 0-3", "dog 0-3", "dog 3-7", "dog 7+"]))
    L += ["", "## The spread sign", "",
          "`games()` stores the betting spread (negative = favourite). The hand-set formula assumed positive = favourite, so it projected "
          "favourites to pass more and run less than underdogs; real games go the other way for rushing and are flat for attempts. "
          "Mean attempts / carries by spread bucket:", "",
          "| spread | games | pass att actual | hand-set | fitted | rush att actual | hand-set | fitted |", "|---|--:|--:|--:|--:|--:|--:|--:|"]
    for b, g in wf.groupby("bucket", observed=True):
        L.append(f"| {b} | {len(g)} | {g.att_actual.mean():.1f} | {g.legacy_att.mean():.1f} | {g.fit_att.mean():.1f} | "
                 f"{g.car_actual.mean():.1f} | {g.legacy_car.mean():.1f} | {g.fit_car.mean():.1f} |")

    last = fits[str(max(CUTOFFS))]
    L += ["", f"## Coefficients of the fit through {last['train_through']} (per 1 SD of the feature; used for 2026)", "",
          "| feature | pass attempts | rush attempts |", "|---|--:|--:|"]
    names_f = sorted(set(last["att"]["features"]) | set(last["car"]["features"]))
    for n in names_f:
        cells = []
        for tgt in ("att", "car"):
            m = last[tgt]
            cells.append(f"{m['beta'][m['features'].index(n)]:+.2f}" if n in m["features"] else "")
        L.append(f"| {n} | {cells[0]} | {cells[1]} |")
    L.append(f"| (intercept) | {last['att']['intercept']:.2f} | {last['car']['intercept']:.2f} |")
    return "\n".join(L) + "\n"


def main() -> None:
    df = training_table()
    fits = {str(c): fit_cutoff(df, c) for c in CUTOFFS}
    TV.COEFS_PATH.write_text(json.dumps({"meta": {"halflife": TV.HALFLIFE, "window": TV.WINDOW, "ridge": TV.RIDGE,
                                                  "note": "regenerate with python -m matchup_model.opp.team_volume_calibration"},
                                         "fits": fits}, indent=1), encoding="utf-8")
    TV._coefs.cache_clear()
    wf = walk_forward(df, fits)
    txt = build_report(df, fits, wf)
    REPORT.write_text(txt, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(txt + f"\nwrote {TV.COEFS_PATH} and {REPORT}")


if __name__ == "__main__":
    main()
