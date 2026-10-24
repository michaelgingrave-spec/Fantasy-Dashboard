"""Does a teammate's own share in the games a key teammate missed (the Data Suite's on/off split) beat the model's trailing window when that teammate is Out again?

    python -m matchup_model.opp.role_rebase        # writes matchup_model/role_rebase_report.md (about a minute)

Same observations as injury_absorption.py: every healthy teammate who played in a team-week where an Out starter (the KEY player) vacates >= 12% of the unit's trailing
share, receiving (target share, WR/TE/RB) and rushing (carry share, RB), split into NEW absences (the key player appeared in the team's last game) and ONGOING ones (he
had already missed it). Predictions of the teammate's actual share, scored leave-one-season-out:

  A      beta x trailing share                               no injury information
  K      beta x s + kappa x (the proportional boost)          what model v4 does (kappa fit on the other seasons)
  WO     his mean share over the last L team games the key player did not play (he played)   the on/off split
  SINCE  his mean share over the consecutive most recent games the key player has missed       a 'since the injury' split (the play-split analogue)
  BL(n)  (n_wo x WO + n x K) / (n_wo + n)                      the on/off split shrunk toward K by its own sample size

Only games from the key player's first appearance with the team inside the window count, so games before he arrived are not 'games he missed'.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from matchup_model.opp import data as D
from matchup_model.opp import injuries as I
from matchup_model.opp import injury_absorption as IA

REPORT = Path(__file__).resolve().parents[1] / "role_rebase_report.md"
WINDOW = 10                         # recent team games considered
SHRINK = (0.5, 1.0, 2.0, 3.0, 5.0, 8.0)
REF_K = 2.0                         # the shrinkage shown in the by-season table
BOOT = 600


def without_features(unit: str, obs: pd.DataFrame, base: pd.DataFrame, snaps: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """Add n_wo / wo / wo_ew (games the key player missed, mean and recency-weighted share), n_since / since (the unbroken run of his most recent misses) and n_in / w_in
    (games he played) to every observation. `obs` needs team, season, week, pid, name, key_name (injury_absorption.observations)."""
    _, by_team, actual, played = IA._context(unit, base, snaps)
    team_games = {t: sorted(set(zip(g["season"], g["week"]))) for t, g in by_team.items()}
    rows = []
    for r in obs.itertuples():
        games = [g for g in team_games.get(r.team, []) if g < (r.season, r.week)][-window:]
        first = next((i for i, (s_, w_) in enumerate(games) if (s_, w_, r.team, r.key_name) in played), None)
        games = games[first:] if first is not None else games
        wo, inn, since, run = [], [], [], True
        for s_, w_ in reversed(games):                                   # newest first
            key_played = (s_, w_, r.team, r.key_name) in played
            me_played = (s_, w_, r.team, r.name) in played
            share = actual.get((s_, w_, r.team, r.pid), 0.0 if me_played else np.nan)
            if not key_played:
                if me_played:
                    wo.append(share)
                    if run:
                        since.append(share)
            else:
                run = False
                if me_played:
                    inn.append(share)
        rows.append({"n_wo": len(wo), "wo": float(np.mean(wo)) if wo else np.nan, "wo_ew": float(I._ewma(np.array(wo[::-1]))) if wo else np.nan,
                     "n_since": len(since), "since": float(np.mean(since)) if since else np.nan, "n_in": len(inn), "w_in": float(np.mean(inn)) if inn else np.nan})
    return pd.concat([obs.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def with_predictions(d: pd.DataFrame) -> pd.DataFrame:
    """P_A (no injury information) and P_K (the v4 rule) for every row, kappa learned on the other seasons."""
    beta = d["beta"] if "beta" in d else pd.Series(1.0, index=d.index)
    d = d.assign(beta=beta, adj=beta * (d["B"] - d["s"]), net=d["a"] - beta * d["s"])
    kappa = {}
    for s_ in sorted(d["season"].unique()):
        tr = d[d["season"] != s_]
        den = float((tr["adj"] ** 2).sum())
        kappa[s_] = float((tr["adj"] * tr["net"]).sum() / den) if den > 0 else 0.0
    return d.assign(P_A=d["beta"] * d["s"], P_K=d["beta"] * d["s"] + d["season"].map(kappa) * d["adj"])


def blend(d: pd.DataFrame, k: float, col: str = "wo") -> pd.Series:
    """The with-out share shrunk toward the production rule by its own sample size: (n x WO + k x K) / (n + k)."""
    return (d["n_wo"] * d[col] + k * d["P_K"]) / (d["n_wo"] + k)


def rmse(x) -> float:
    """Root mean square in share points."""
    return 100 * float(np.sqrt(np.mean(np.asarray(x, float) ** 2)))


def pct_change(d: pd.DataFrame, e1: pd.Series, e0: pd.Series, reps: int = BOOT, seed: int = 3) -> tuple[float, tuple[float, float]]:
    """RMSE of error e1 against error e0 as a % change, with a team-week-clustered bootstrap 95% interval. NaN where e0 is exactly zero (nothing to compare with)."""
    g = d.assign(_a=np.asarray(e1) ** 2, _b=np.asarray(e0) ** 2).groupby(["season", "week", "team"])[["_a", "_b"]].sum().to_numpy()

    def change(x):
        return 100 * (np.sqrt(x[:, 0].sum() / x[:, 1].sum()) - 1) if x[:, 1].sum() > 0 else float("nan")
    rng = np.random.default_rng(seed)
    draws = [change(g[rng.integers(0, len(g), len(g))]) for _ in range(reps)]
    draws = [x for x in draws if np.isfinite(x)]
    return float(change(g)), ((float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))) if draws else (float("nan"), float("nan")))


def evaluate(d: pd.DataFrame) -> dict | None:
    """Scores on the rows that have at least one game the key player missed. None when there are too few."""
    cov = d[d["n_wo"] >= 1]
    if len(cov) < 30:
        return None
    e_k = cov["a"] - cov["P_K"]
    out = {"rows": len(cov), "team_weeks": int(cov[["season", "week", "team"]].drop_duplicates().shape[0]), "all_rows": len(d), "mean_n_wo": float(cov["n_wo"].mean()),
           "rmse": {"A": rmse(cov["a"] - cov["P_A"]), "K": rmse(e_k), "WO": rmse(cov["a"] - cov["wo"]), "WO ewma": rmse(cov["a"] - cov["wo_ew"])}, "blend": {}}
    sd = cov[cov["n_since"] >= 1]
    out["since"] = {"rows": len(sd), "SINCE": rmse(sd["a"] - sd["since"]), "K": rmse(sd["a"] - sd["P_K"])} if len(sd) else None
    out["wo_vs_k"] = pct_change(cov, cov["a"] - cov["wo_ew"], e_k)
    for k in SHRINK:
        out["blend"][k] = pct_change(cov, cov["a"] - blend(cov, k, "wo_ew"), e_k)
    by_season = []
    for s_, g in cov.groupby("season"):
        by_season.append({"season": int(s_), "team_weeks": int(g[["week", "team"]].drop_duplicates().shape[0]), "rows": len(g), "K": rmse(g["a"] - g["P_K"]),
                          "WO": rmse(g["a"] - g["wo_ew"]), f"BL({REF_K:g})": rmse(g["a"] - blend(g, REF_K, "wo_ew"))})
    out["by_season"] = by_season
    return out


VERDICT = """## Reading (2026-10-22)

The on/off split is a small-sample statistic: a teammate has a median of about two games without the key player in the window, and one game is mostly noise (the with-out
share alone is worse than the production rule wherever he has only one such game). Receiving: no gain at any shrinkage worth shipping (the split alone is ~10% worse than K,
the shrunk blends are within 1% of K with intervals through zero). Rushing, ongoing absences only: the split shrunk toward K is about 5% better in share RMSE with the
interval clear of zero and the sign right in all five seasons, but it covers about 70 team-weeks in five seasons, roughly one team-week per NFL week (one or two backs), so
it cannot move a betting record. Not shipped. The unmeasured part of the Data Suite's play splits (the shares after a player leaves mid-game, for a NEW absence next week)
needs play-by-play participation data."""


def build_report(res: dict) -> str:
    L = ["# Does the with / without-teammate split beat the trailing window when a starter is Out?", "",
         "2022-26 nflverse box scores, snap counts and injury reports; teammates who played in a team-week where an Out starter vacates >= 12% of the unit's trailing share; "
         f"leave-one-season-out. `K` is model v4's rule, `WO` the teammate's share in the last {WINDOW} team games the key player missed, `SINCE` his share over the unbroken run "
         "of games the key player has missed, `BL(n)` the split shrunk toward K by `n` pseudo-games. RMSE is in points of the team's target share (receiving) or carry share "
         "(rushing), on the teammates who have at least one game without the key player.", ""]
    for unit, label in (("rec", "Receiving (target share, WR/TE/RB)"), ("rush", "Rushing (carry share, RB)")):
        for kind in ("ONGOING", "NEW"):
            r = res.get((unit, kind))
            L += [f"## {label} - {kind} absence", ""]
            if r is None:
                L += ["Too few teammates with a game without the key player.", ""]
                continue
            L += [f"{r['rows']} teammates in {r['team_weeks']} team-weeks (of {r['all_rows']} rows); on average {r['mean_n_wo']:.1f} games without the key player.", "",
                  "| prediction | RMSE (points) |", "|---|--:|"]
            L += [f"| {k} | {v:.2f} |" for k, v in r["rmse"].items()]
            if r["since"]:
                L += [f"| SINCE (on {r['since']['rows']} rows with an unbroken run) | {r['since']['SINCE']:.2f} (K on the same rows {r['since']['K']:.2f}) |"]
            p, (lo, hi) = r["wo_vs_k"]
            L += ["", f"WO (recency-weighted) against K: {p:+.1f}% [{lo:+.1f}, {hi:+.1f}].", "", "| shrinkage n | BL(n) against K, RMSE change | 95% interval (team-week clustered) |", "|--:|--:|---|"]
            L += [f"| {k:g} | {v[0]:+.1f}% | [{v[1][0]:+.1f}, {v[1][1]:+.1f}] |" for k, v in r["blend"].items()]
            L += ["", "By season (RMSE in points):", "", "| season | team-weeks | rows | K | WO | BL(2) |", "|---|--:|--:|--:|--:|--:|"]
            L += [f"| {s['season']} | {s['team_weeks']} | {s['rows']} | {s['K']:.2f} | {s['WO']:.2f} | {s[f'BL({REF_K:g})']:.2f} |" for s in r["by_season"]] + [""]
    return "\n".join(L + [VERDICT, ""]) + "\n"


def run(base: pd.DataFrame | None = None, inj: pd.DataFrame | None = None, snaps: pd.DataFrame | None = None) -> dict:
    base = I._shares_base() if base is None else base
    inj = D.injuries() if inj is None else inj
    inj = inj.assign(report_status=inj["report_status"].fillna("").str.lower())
    snaps = D.snaps() if snaps is None else snaps
    res = {}
    for unit in ("rec", "rush"):
        obs = IA.observations(unit, base, inj, snaps)
        if obs.empty:
            continue
        obs = obs[obs["season"] >= IA.FIRST_SEASON]
        clean = IA.clean_weeks(unit, base, inj, snaps)
        clean = clean[clean["season"] >= IA.FIRST_SEASON] if len(clean) else clean
        obs = obs.assign(beta=obs["season"].map(IA.leakage(clean, sorted(obs["season"].unique()))))
        for kind, sel in (("ONGOING", IA.ongoing_absences(obs)), ("NEW", IA.new_absences(obs))):
            if len(sel):
                res[(unit, kind)] = evaluate(with_predictions(without_features(unit, sel, base, snaps)))
    return res


def main() -> None:
    txt = build_report(run())
    REPORT.write_text(txt, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(txt + f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
