"""Calibrate and audit the RB projection (carry-share shrinkage, a constant scale per stat, a TD multiplier).

    python -m matchup_model.opp.rb_calibration        # writes matchup_model/rb_calibration_report.md; ~3 min on 8 cores

Same recipe as the WR/TE recalibration (receiving_calibration.py): the universe is every RB who played in 2023-25 weeks 5-18 with
history -- a box-score row, or on the field per the snap counts with no stat line (a true zero). Each is projected under the
original RB model on the original team volume, the original RB model on the fitted team volume, and the new RB constants on the fitted
volume; scales are learned on the backs the model expects to be involved (>= 6 projected carries for the rushing stats, >= 1.5
projected receptions for the receiving ones -- the backs with prop lines) and every season is scored with scales learned on the
other two, so every RMSE below is out of sample for the constants.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from matchup_model.opp import model as M
from matchup_model.opp import receiving_calibration as RC
from matchup_model.opp.blend import INJURY_ADJ

SEASONS, KEY = RC.SEASONS, RC.KEY
STATS = {"rush_att": "carries", "rush_yds": "rushing_yards", "rec": "receptions", "rec_yds": "receiving_yards"}
UNIT = {"rush_att": 1.0, "rush_yds": 1.0, "rec": 1.0, "rec_yds": 1.0, "td": 1.0}
INVOLVED = {"rush_att": ("p_rush_att", 6.0), "rush_yds": ("p_rush_att", 6.0), "rec": ("p_rec", 1.5), "rec_yds": ("p_rec", 1.5)}
RB_KEYS = RC.RB_KEYS + ("TEAM_VOLUME_MODEL",)
ORIGINAL = {**RC.RB_ORIGINAL, "TEAM_VOLUME_MODEL": "legacy"}
MID = dict(K_CARRY=3.0, RB_CARRY_PRIOR=0.42, RB_YDS_CATCH_FACTOR=False, TEAM_VOLUME_MODEL="fitted")
NEW = dict(K_CARRY=1.0, RB_CARRY_PRIOR=0.25, RB_YDS_CATCH_FACTOR=False, TEAM_VOLUME_MODEL="fitted")
REPORT = Path(__file__).resolve().parents[1] / "rb_calibration_report.md"


def _set(params: dict) -> None:
    for k, v in params.items():
        setattr(M, k, v)
    M.opp_line.cache_clear(); M.team_volume.cache_clear()


def project(params: dict, universe: pd.DataFrame) -> pd.DataFrame:
    """Raw (unscaled) RB projections under `params`, plus DK points and expected TDs; model state is restored after."""
    saved = {k: getattr(M, k) for k in RB_KEYS}
    try:
        _set({**params, "RB_SCALE": dict(UNIT)})
        out = []
        for r in universe.itertuples():
            o = M.opp_line(r.nk, "RB", r.season, r.week, by="name", injury_adj=INJURY_ADJ)
            ok = o.get("dk_fp") is not None
            ln = o.get("line", {}) if ok else {}
            out.append({**{f"p_{k}": ln.get(k, np.nan) for k in STATS}, "p_rush_td": ln.get("rush_td", np.nan),
                        "p_rec_td": ln.get("rec_td", np.nan), "fp": o["dk_fp"] if ok else np.nan})
    finally:
        _set(saved)
    return pd.concat([universe.reset_index(drop=True), pd.DataFrame(out)], axis=1).dropna(subset=["p_rush_att", "p_rec"]).reset_index(drop=True)


def _chunk(params: dict, chunk: pd.DataFrame) -> pd.DataFrame:
    return project(params, chunk)


def project_many(jobs: list[dict], universe: pd.DataFrame) -> list[pd.DataFrame]:
    from concurrent.futures import ProcessPoolExecutor
    chunks = [universe[universe.season == s] for s in SEASONS]
    with ProcessPoolExecutor(max_workers=min(os.cpu_count() or 2, len(jobs) * len(chunks))) as ex:
        futs = [[ex.submit(_chunk, p, c) for c in chunks] for p in jobs]
        return [pd.concat([f.result() for f in fs], ignore_index=True) for fs in futs]


def _involved(d: pd.DataFrame, stat: str) -> pd.Series:
    col, thr = INVOLVED[stat]
    return d[col] >= thr


def _scales(d: pd.DataFrame) -> dict:
    """Actual / projected per stat on the backs involved in that stat, plus the TD multiplier (rush + receiving TDs)."""
    out = {s: float(d[_involved(d, s)][c].sum() / d[_involved(d, s)][f"p_{s}"].sum()) for s, c in STATS.items()}
    g = d[_involved(d, "rush_att")].dropna(subset=["p_rush_td", "p_rec_td"])
    out["td"] = float((g.rushing_tds + g.receiving_tds).sum() / (g.p_rush_td + g.p_rec_td).sum())
    return out


def _view(d: pd.DataFrame, tag: str) -> pd.DataFrame:
    """The same rows with another variant's projections (suffix `tag`) in the p_* columns."""
    return d.assign(**{f"p_{k}": d[f"p_{k}_{tag}"] for k in STATS})


def _rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def _scored(d: pd.DataFrame, tags: tuple[str, ...]) -> pd.DataFrame:
    """Pooled out-of-season RMSE per stat for each variant tag ('' = the current columns), scale learned on the other two seasons and
    population = the backs involved in that stat (variant-independent: the current model's projection decides who is involved)."""
    rows = []
    for stat, col in STATS.items():
        pop = d[_involved(d, stat)]
        for tag in tags:
            sq = 0.0
            for test in SEASONS:
                tr, te = pop[pop.season != test], pop[pop.season == test]
                v = (lambda x: x) if tag == "" else (lambda x, t=tag: _view(x, t))
                c = float(tr[col].sum() / v(tr)[f"p_{stat}"].sum())
                sq += float(((c * v(te)[f"p_{stat}"] - te[col]) ** 2).sum())
            raw = _rmse((pop[f"p_{stat}"] if tag == "" else pop[f"p_{stat}_{tag}"]), pop[col])
            rows.append(dict(stat=stat, variant=tag or "new", n=len(pop), scaled=float(np.sqrt(sq / len(pop))), raw=raw))
    return pd.DataFrame(rows)


def _tier_table(d: pd.DataFrame) -> list[str]:
    L = ["| projected | backs | with a stat row | actual / projected |", "|---|--:|--:|--:|"]
    for stat, bins, label in (("rush_att", [0, 2, 5, 8, 12, 16, 99], "carries"), ("rec", [0, 0.5, 1, 2, 3, 99], "receptions")):
        t = d.assign(tier=pd.cut(d[f"p_{stat}"], bins, right=False))
        for tier, g in t.groupby("tier", observed=True):
            L.append(f"| {label} {tier} | {len(g)} | {100 * (1 - g.no_row.mean()):.0f}% | {g[STATS[stat]].sum() / g[f'p_{stat}'].sum():.2f} |")
    return L


def _shipped() -> dict:
    return {k: getattr(M, k) for k in RB_KEYS}


def _audit_chunk(params: dict, chunk: pd.DataFrame) -> pd.DataFrame:
    """Worker: the production RB projection (DK points, expected rush + receiving TDs) and the trailing-average baseline per row."""
    from matchup_model.opp import blend as BL
    _set(params)
    fp, lam, naive = [], [], []
    for r in chunk.itertuples():
        o = M.opp_line(r.nk, "RB", r.season, r.week, by="name", injury_adj=INJURY_ADJ)
        ok = o.get("dk_fp") is not None
        fp.append(o["dk_fp"] if ok else np.nan)
        lam.append(o["line"].get("rush_td", 0.0) + o["line"].get("rec_td", 0.0) if ok else np.nan)
        naive.append(BL._naive_fp(r.nk, "RB", r.season, r.week))
    return chunk.assign(fp=fp, lam=lam, naive=naive)


def audit(u: pd.DataFrame) -> list[str]:
    """DK points and anytime-TD odds for the original RB model vs the shipped one, on everyone who played."""
    from concurrent.futures import ProcessPoolExecutor
    chunks = [u[u.season == s] for s in SEASONS]
    with ProcessPoolExecutor(max_workers=min(os.cpu_count() or 2, 6)) as ex:
        futs = {w: [ex.submit(_audit_chunk, p, c) for c in chunks] for w, p in (("old", ORIGINAL), ("new", _shipped()))}
        res = {w: pd.concat([f.result() for f in fs], ignore_index=True) for w, fs in futs.items()}
    d = res["old"][KEY + ["dk_fp", "rushing_tds", "receiving_tds", "carries", "naive", "fp", "lam"]].merge(
        res["new"][KEY + ["fp", "lam"]], on=KEY, suffixes=("_old", "_new")).dropna(subset=["fp_old", "fp_new", "naive"])
    d["td"] = ((d.rushing_tds + d.receiving_tds) >= 1).astype(int)
    d["p_old"], d["p_new"] = 1 - np.exp(-d.lam_old), 1 - np.exp(-d.lam_new)
    L = ["", "## Downstream check: DK fantasy points and anytime-TD probability", "",
         f"The shipped RB model against the original one, {len(d)} RB player-weeks (2023-25 weeks 5-18, everyone who played). Nothing is "
         "learned here.", "",
         "| backs | n | mean actual DK pts | projected, original | current | RMSE trailing avg | original | current |", "|---|--:|--:|--:|--:|--:|--:|--:|"]
    for label, sub in (("everyone who played", d), ("trailing carries: trailing-avg DK pts >= 8", d[d.naive >= 8])):
        L.append(f"| {label} | {len(sub)} | {sub.dk_fp.mean():.2f} | {sub.fp_old.mean():.2f} | {sub.fp_new.mean():.2f} | "
                 f"{_rmse(sub.naive, sub.dk_fp):.3f} | {_rmse(sub.fp_old, sub.dk_fp):.3f} | {_rmse(sub.fp_new, sub.dk_fp):.3f} |")
    L += ["", "P(at least one TD) = 1 - exp(-(expected rushing + receiving TDs)):", "",
          "| backs | n | actual rate | mean P, original | current | Brier, original | current | flat average |", "|---|--:|--:|--:|--:|--:|--:|--:|"]
    for label, sub in (("everyone who played", d), ("trailing-avg DK pts >= 8", d[d.naive >= 8])):
        base = sub.td.mean()
        L.append(f"| {label} | {len(sub)} | {100 * base:.1f}% | {100 * sub.p_old.mean():.1f}% | {100 * sub.p_new.mean():.1f}% | "
                 f"{np.mean((sub.p_old - sub.td) ** 2):.4f} | {np.mean((sub.p_new - sub.td) ** 2):.4f} | {np.mean((base - sub.td) ** 2):.4f} |")
    return L


def main(argv: list[str] | None = None) -> None:
    u = RC.universe_rows(("RB",))
    print(f"RB universe: {len(u)} player-weeks ({int(u.no_row.sum())} on the field with no stat line)", flush=True)
    orig, mid, new = project_many([ORIGINAL, MID, NEW], u)
    keys = set(zip(new.season, new.week, new.nk)) & set(zip(orig.season, orig.week, orig.nk)) & set(zip(mid.season, mid.week, mid.nk))
    sub = lambda f: f[[k in keys for k in zip(f.season, f.week, f.nk)]].reset_index(drop=True)  # noqa: E731
    orig, mid, new = sub(orig), sub(mid), sub(new)
    d = new.copy()
    for tag, f in (("orig", orig), ("mid", mid)):
        for k in STATS:
            d[f"p_{k}_{tag}"] = f[f"p_{k}"].to_numpy()
    print(f"projected on identical rows: {len(d)}", flush=True)

    res = _scored(d, ("orig", "mid", ""))
    final = _scales(d)
    L = ["# RB projection: carry-share shrinkage and constant scales (2023-25; scales learned on the other two seasons)", "",
         f"{len(d)} RB player-weeks (weeks 5-18; {int(d.no_row.sum())} are backs on the field with no stat line, true zeros). Scales are learned on the "
         f"backs the model expects to be involved: >= {INVOLVED['rush_att'][1]:g} projected carries for the rushing stats, >= "
         f"{INVOLVED['rec'][1]:g} projected receptions for the receiving ones. Pooled out-of-season RMSE on those backs:", "",
         "| stat | backs | original | + scale | + fitted team volume | + carry shrinkage (current) | total change |", "|---|--:|--:|--:|--:|--:|--:|"]
    for stat in STATS:
        r = res[res.stat == stat].set_index("variant")
        L.append(f"| {stat} | {int(r.n.iloc[0])} | {r.loc['orig', 'raw']:.3f} | {r.loc['orig', 'scaled']:.3f} | {r.loc['mid', 'scaled']:.3f} | "
                 f"{r.loc['new', 'scaled']:.3f} | {100 * (r.loc['new', 'scaled'] / r.loc['orig', 'raw'] - 1):+.1f}% |")
    L += ["", "(`+ scale` is a constant learned the same way on the ORIGINAL model's projections, so each step to its right is one change.)", "",
          "## Why a constant, and who it is learned on", "",
          "Actual / projected by projected usage under the current constants, before any scale. Lead backs are close to calibrated; committee "
          "and depth backs are heavily over-projected (their carry shares were shrunk up toward a 0.42 prior), and a constant is learned on "
          "the backs with prop lines.", ""] + _tier_table(d)

    sc = final
    L += audit(u)
    L += ["", "## Constants for model.py (learned on all three seasons)", "", "```python",
          f"K_CARRY = {NEW['K_CARRY']}", f"RB_CARRY_PRIOR = {NEW['RB_CARRY_PRIOR']}", f"RB_YDS_CATCH_FACTOR = {NEW['RB_YDS_CATCH_FACTOR']}",
          "RB_SCALE = {" + ", ".join(f'"{k}": {sc[k]:.3f}' for k in ("rush_att", "rush_yds", "rec", "rec_yds", "td")) + "}", "```"]
    text = "\n".join(L)
    print(text)
    REPORT.write_text(text + "\n", encoding="utf-8")
    d.to_pickle(Path(REPORT).with_suffix(".pkl")) if "--keep" in (argv or sys.argv[1:]) else None


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
