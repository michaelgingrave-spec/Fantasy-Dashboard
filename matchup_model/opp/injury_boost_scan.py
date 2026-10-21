"""Player-level check of model.INJ_REDIST: which fraction of the injury boost gives the best out-of-season projections?

    python -m matchup_model.opp.injury_boost_scan        # ~8 min on 8 cores; writes matchup_model/injury_boost_scan_report.md

The share-level fit (injury_absorption.py) says what fraction of the old proportional boost the healthy teammates really absorb. This asks the question the model is for:
projected targets / receptions / receiving yards (WR/TE) and carries / rushing yards (RB) for every player who played in 2023-25 weeks 5-18, scored out of season with the
scale constants re-learned for every setting (leave-one-season-out, on the players the model expects to be involved, as receiving_calibration / rb_calibration do).

The projection is linear in the injury multiplier (target share x inj_rec, carry share x inj_rush), so three runs per unit are enough: with no boost at all (p0), with only the
NEW-absence boost at scale 1, and with only the ONGOING one. Every (kappa_new, kappa_ongoing) is then  p0 x (1 + kappa_new b_new + kappa_ongoing b_old)  with b = p_run / p0 - 1.
The 2.2 cap on the multiplier is lifted for those runs (it binds on very few rows).

RMSE differences are small (the boost touches a minority of rows and a few tenths of a target), so this is a consistency check on the share-level fit, not a second fit:
the shipped fractions should sit where the surface is flat at its bottom, and the old full boost (1.0) should not.
"""
from __future__ import annotations

import itertools
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPORT = Path(__file__).resolve().parents[1] / "injury_boost_scan_report.md"
KN = [0.0, 0.14, 0.3, 0.46, 0.6, 0.75, 1.0]
KO = [0.0, 0.2, 0.37, 0.5, 0.75, 1.0]
FLOOR = 0.05                      # projections at or below this carry no ratio (a zero projection cannot be scaled)
_ZERO = {"new": 0.0, "ongoing": 0.0}
_ONE_NEW, _ONE_OLD = {"new": 1.0, "ongoing": 0.0}, {"new": 0.0, "ongoing": 1.0}
KINDS = {"zero": {"rec": _ZERO, "rush": _ZERO}, "rec_new": {"rec": _ONE_NEW, "rush": _ZERO}, "rec_old": {"rec": _ONE_OLD, "rush": _ZERO},
         "rush_new": {"rec": _ZERO, "rush": _ONE_NEW}, "rush_old": {"rec": _ZERO, "rush": _ONE_OLD}}


def boosts(p0: pd.Series, p_run: pd.Series) -> pd.Series:
    """b = p_run / p0 - 1 where p0 is large enough to divide by, 0 elsewhere (a row the boost cannot move)."""
    return (p_run / p0.where(p0 > FLOOR) - 1.0).fillna(0.0)


def errors(d: pd.DataFrame, stats: dict[str, str], kn: float, ko: float, bn: str, bo: str, involved: str, thr: float, by: str | None = None):
    """Out-of-season error (scaled projection - actual) of every row for each stat at (kn, ko), and the mask of 'involved' rows under that setting.
    `d` needs season, p0_<stat> for each stat, the actual columns named in `stats`, and the boost columns bn / bo; the scale of a held-out season is
    actual / projected summed over the involved rows of the other seasons (per `by` group when given)."""
    fac = 1.0 + kn * d[bn] + ko * d[bo]
    p = {s: d[f"p0_{s}"] * fac for s in stats}
    inv = (p[involved] >= thr).to_numpy()
    grp = d[by].to_numpy() if by else np.zeros(len(d), dtype=int)
    err = {}
    for s, col in stats.items():
        e = np.full(len(d), np.nan)
        for test in sorted(d.season.unique()):
            te = (d.season == test).to_numpy()
            tr = ~te & inv
            for g in np.unique(grp[te]):
                m_tr, m_te = tr & (grp == g), te & (grp == g)
                if not m_tr.any() or p[s][m_tr].sum() <= 0:
                    continue
                e[m_te] = (d.loc[m_tr, col].sum() / p[s][m_tr].sum()) * p[s][m_te] - d.loc[m_te, col]
        err[s] = e
    return err, inv


def _rmse(x) -> float:
    x = np.asarray(x, float)
    return float(np.sqrt(np.nanmean(x ** 2)))


def _score(d: pd.DataFrame, stats: dict[str, str], kn: float, ko: float, bn: str, bo: str, involved: str, thr: float, by: str | None) -> dict:
    err, inv = errors(d, stats, kn, ko, bn, bo, involved, thr, by)
    return {**{f"all_{s}": _rmse(e) for s, e in err.items()}, **{f"inv_{s}": _rmse(e[inv]) for s, e in err.items()}}


def surface(d: pd.DataFrame, stats: dict[str, str], bn: str, bo: str, involved: str, thr: float, by: str | None = None,
            kn_grid=KN, ko_grid=KO) -> pd.DataFrame:
    """Pooled out-of-season RMSE per (kappa_new, kappa_ongoing), as a % change against the old full boost (1, 1): the mean over the stats, for everyone and for the involved rows."""
    base = _score(d, stats, 1.0, 1.0, bn, bo, involved, thr, by)
    rows = [{"kn": kn, "ko": ko, **_score(d, stats, kn, ko, bn, bo, involved, thr, by)} for kn, ko in itertools.product(kn_grid, ko_grid)]
    t = pd.DataFrame(rows)
    for pre in ("all", "inv"):
        t[pre] = np.mean([100 * (t[f"{pre}_{s}"] / base[f"{pre}_{s}"] - 1) for s in stats], axis=0)
    return t[["kn", "ko", "all", "inv"]]


def context_table(d: pd.DataFrame, ctx: pd.Series, stats: dict[str, str], settings: list[tuple[float, float]], bn: str, bo: str, involved: str, thr: float,
                  by: str | None = None) -> pd.DataFrame:
    """RMSE and mean error (projected - actual) of the first stat, by injury context, for a few settings."""
    s0 = next(iter(stats))
    rows = []
    for kn, ko in settings:
        err, _ = errors(d, stats, kn, ko, bn, bo, involved, thr, by)
        for c in sorted(ctx.unique()):
            m = (ctx == c).to_numpy()
            rows.append({"context": c, "rows": int(m.sum()), "kappa new / ongoing": f"{kn:g} / {ko:g}", "RMSE": round(_rmse(err[s0][m]), 3), "mean error": round(float(np.nanmean(err[s0][m])), 2)})
    return pd.DataFrame(rows)


def context(team: str, season: int, week: int) -> str:
    """The injury context of a team's receiving rotation entering a week."""
    from matchup_model.opp import injuries as I
    c = I._group_components(team, "rec", season, week)
    if c is None:
        return "A no hurt player"
    if not any(a == 0.0 for a in c["avail"].values()):
        return "B Questionable / Doubtful only"
    if c["vac_new"] >= 0.12:
        return "C Out player newly absent (>= 12% vacated)"
    if c["vac_new"] < 0.03 and c["vac_old"] >= 0.12:
        return "D Out player already absent (>= 12%)"
    return "E Out player, small or mixed"


# ----------------------------------------------------------------------------------------------------------------- the projection runs (module level: Windows spawns workers)
def _job(family: str, kind: str, chunk: pd.DataFrame) -> pd.DataFrame:
    from matchup_model.opp import injuries as I
    from matchup_model.opp import rb_calibration as RB
    from matchup_model.opp import receiving_calibration as RC
    cap = I._MULT_CAP
    I._MULT_CAP = 1e9
    try:
        return (RC if family == "wrte" else RB).project({**(RC.NEW if family == "wrte" else RB.NEW), "INJ_REDIST": KINDS[kind]}, chunk)
    finally:
        I._MULT_CAP = cap


def _runs(family: str, kinds: list[str], universe: pd.DataFrame, seasons) -> dict[str, pd.DataFrame]:
    chunks = [universe[universe.season == s] for s in seasons]
    with ProcessPoolExecutor(max_workers=os.cpu_count() or 2) as ex:
        futs = {k: [ex.submit(_job, family, k, c) for c in chunks] for k in kinds}
        return {k: pd.concat([f.result() for f in fs], ignore_index=True) for k, fs in futs.items()}


def _attach(base: pd.DataFrame, runs: dict[str, pd.DataFrame], key: list[str], stat: str, pairs: dict[str, str]) -> pd.DataFrame:
    """Add boost columns: for each (name, run kind) in `pairs`, b = run[stat] / zero[stat] - 1, aligned on `key`."""
    z = runs["zero"].set_index(key)
    for name, kind in pairs.items():
        k = runs[kind].set_index(key)
        base[name] = boosts(z[stat], k[stat].reindex(z.index)).reindex(pd.MultiIndex.from_frame(base[key])).fillna(0.0).to_numpy()
    return base


def _fmt(t: pd.DataFrame, col: str) -> list[str]:
    p = t.pivot(index="kn", columns="ko", values=col).round(2)
    p.index = [f"new {i:g}" for i in p.index]
    p.columns = [f"ongoing {c:g}" for c in p.columns]
    return ["| |" + "|".join(p.columns) + "|", "|---|" + "|".join("--:" for _ in p.columns) + "|"] + [f"| {i} | " + " | ".join(f"{v:+.2f}" for v in row) + " |" for i, row in zip(p.index, p.to_numpy())]


def build_report(tables: dict[str, pd.DataFrame], ctx: pd.DataFrame, shipped: dict) -> str:
    L = ["# Does the shipped injury boost also predict better at the player level?", "",
         "2023-25 weeks 5-18, every player who played; each cell is the pooled out-of-season RMSE of the stats listed, as a % change against the OLD full boost (new 1, ongoing 1), "
         "with the scale constants re-learned for the setting on the other two seasons (negative = better than the old rule). Projections are linear in the multiplier, so "
         "every setting is built from three runs per unit (no boost / new only / ongoing only). The shipped fractions are "
         f"`{shipped}`.", ""]
    for title, t in tables.items():
        L += [f"## {title}", "", "Everyone who played:", ""] + _fmt(t, "all") + ["", "Players the model expects to be involved under each setting:", ""] + _fmt(t, "inv") + [""]
    if ctx is not None and len(ctx):
        L += ["## WR/TE receiving targets by injury context (each setting with its own re-learned scale)", "",
              "Mean error is projected minus actual targets. The boost only matters where a teammate is hurt: C and D are an Out player newly / already absent, B has only "
              "Questionable or Doubtful players.", ""]
        out = ["| " + " | ".join(ctx.columns) + " |", "|" + "|".join("---" for _ in ctx.columns) + "|"]
        out += ["| " + " | ".join(str(v) for v in row) + " |" for row in ctx.itertuples(index=False)]
        L += out + [""]
    return "\n".join(L) + "\n"


def main() -> None:
    from matchup_model.opp import model as M
    from matchup_model.opp import receiving_calibration as RC
    key = RC.KEY
    tables = {}
    # WR / TE
    u = RC.universe_rows()
    runs = _runs("wrte", ["zero", "rec_new", "rec_old"], u, RC.SEASONS)
    z = runs["zero"]
    d = z[key + ["position", "team", "targets", "receptions", "receiving_yards"]].copy()
    for s in RC.STATS:
        d[f"p0_{s}"] = z[f"p_{s}"].to_numpy()
    d = _attach(d, runs, key, "p_tgt", {"bn": "rec_new", "bo": "rec_old"}).reset_index(drop=True)
    stats = {s: RC.STATS[s] for s in RC.STATS}
    tables["WR/TE receiving (targets, receptions, receiving yards)"] = surface(d, stats, "bn", "bo", "tgt", RC.MIN_PROJ_TGT, by="position")
    keys = d[["team", "season", "week"]].drop_duplicates()
    cx = {k: context(*k) for k in keys.itertuples(index=False, name=None)}
    ctx = pd.Series([cx[k] for k in zip(d.team, d.season, d.week)], index=d.index)
    settings = [(1.0, 1.0), tuple(M.INJ_REDIST["rec"].values()), (0.14, 0.0), (0.0, 0.0)]
    ctx_t = context_table(d, ctx, stats, settings, "bn", "bo", "tgt", RC.MIN_PROJ_TGT, by="position")
    # RB
    from matchup_model.opp import rb_calibration as RB
    ub = RC.universe_rows(("RB",))
    runs = _runs("rb", list(KINDS), ub, RC.SEASONS)
    z = runs["zero"]
    r = z[key + ["team", "carries", "receptions", "receiving_yards", "rushing_yards"]].copy()
    for s in RB.STATS:
        r[f"p0_{s}"] = z[f"p_{s}"].to_numpy()
    r = _attach(r, runs, key, "p_rush_att", {"bn_rush": "rush_new", "bo_rush": "rush_old"})
    r = _attach(r, runs, key, "p_rec", {"bn_rec": "rec_new", "bo_rec": "rec_old"}).reset_index(drop=True)
    tables["RB rushing (carries, rushing yards)"] = surface(r, {"rush_att": "carries", "rush_yds": "rushing_yards"}, "bn_rush", "bo_rush", "rush_att", 6.0)
    tables["RB receiving (receptions, receiving yards), the receiving fractions"] = surface(r, {"rec": "receptions", "rec_yds": "receiving_yards"}, "bn_rec", "bo_rec", "rec", 1.5)
    txt = build_report(tables, ctx_t, M.INJ_REDIST)
    REPORT.write_text(txt, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(txt + f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
