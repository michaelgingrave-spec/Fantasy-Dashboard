"""Derive (and audit) the WR/TE receiving recalibration constants in model.py.

    python -m matchup_model.opp.receiving_calibration            # ~3 min on 8 cores; writes matchup_model/receiving_calibration_report.md
    python -m matchup_model.opp.receiving_calibration --grid     # also re-checks the shrinkage settings (~5 min more)

Universe: every WR/TE who played in 2023-2025 weeks 5-18 and has >= MIN_GAMES of prior history -- a box-score row, OR on the
field per the snap counts with no stat line (nflverse lists a player only if he recorded a stat, so those are true zeros;
leaving them out conditions the sample on the outcome and inflates actual/projected). Each is projected under two
parameter sets: the old model (K_SHARE_REC 3, K_YPT 28, no scale, no defense nudge) and the recalibrated shrinkage
(K_SHARE_REC / K_YPT below). Then:

  * the constant scale per (position, stat) = sum(actual) / sum(projected), the defense weight = the regression-through-
    the-origin slope of the scaled target miss on projected x (defense rate - 1) -- both learned on players the model
    expects to be involved (>= MIN_PROJ_TGT projected targets before scaling): they are the ones with prop lines and DFS
    salaries, and one constant per position cannot also fit depth players (see the usage-tier table in the report);
  * each season is scored with constants learned ONLY on the other two seasons, as RMSE of targets, receptions and
    receiving yards, for those players and for everyone.

It prints a block to paste into model.py.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

from matchup_model.opp import data as D
from matchup_model.opp import model as M
from matchup_model.opp.blend import INJURY_ADJ

SEASONS = (2023, 2024, 2025)
WEEKS = range(5, 19)
NEW = dict(K_SHARE_REC=1.0, K_YPT=60.0)        # chosen by the shrinkage tuning on 2024-25, validated on 2023 (--grid re-checks)
OLD = dict(K_SHARE_REC=3.0, K_YPT=28.0)
UNIT = {"tgt": 1.0, "rec": 1.0, "rec_yds": 1.0, "td": 1.0}
BEFORE = {**OLD, "DEF_TGT_B": 0.0, "REC_SCALE": {"WR": dict(UNIT), "TE": dict(UNIT)}}   # every constant as it was before the change
STATS = {"tgt": "targets", "rec": "receptions", "rec_yds": "receiving_yards"}
MIN_PROJ_TGT = 3.0
KEY = ["season", "week", "nk"]
REPORT = "matchup_model/receiving_calibration_report.md"


def _set(params: dict) -> None:
    for k, v in params.items():
        setattr(M, k, v)
    M.opp_line.cache_clear(); M.def_target_rel.cache_clear()
    M.team_context.cache_clear(); M.team_volume.cache_clear()


def project(params: dict, universe: pd.DataFrame) -> pd.DataFrame:
    """Raw (unscaled, no defense nudge) projections for the universe under `params`; model state is restored after."""
    saved = (dict(M.REC_SCALE), M.DEF_TGT_B, {k: getattr(M, k) for k in params})
    neutral = {"WR": dict(UNIT), "TE": dict(UNIT)}
    try:
        M.REC_SCALE, M.DEF_TGT_B = neutral, 0.0
        _set(params)
        out = []
        for r in universe.itertuples():
            o = M.opp_line(r.nk, r.position, r.season, r.week, by="name", injury_adj=INJURY_ADJ)
            ln = o.get("line", {}) if o.get("dk_fp") is not None else {}
            out.append((ln.get("tgt", np.nan), ln.get("rec", np.nan), ln.get("rec_yds", np.nan), ln.get("rec_td", np.nan)))
    finally:
        M.REC_SCALE, M.DEF_TGT_B = saved[0], saved[1]
        _set(saved[2])
    return universe.assign(**dict(zip(("p_tgt", "p_rec", "p_rec_yds", "p_rec_td"), np.array(out, float).T))).dropna(
        subset=["p_tgt", "p_rec", "p_rec_yds"]).reset_index(drop=True)


def _project_chunk(params: dict, chunk: pd.DataFrame) -> pd.DataFrame:
    return project(params, chunk)


def project_many(jobs: list[dict], universe: pd.DataFrame) -> list[pd.DataFrame]:
    """`project` for each parameter set, spread over worker processes (one task per set x season): the model state is
    module-global, so settings can't share a process, but they parallelise cleanly across processes."""
    from concurrent.futures import ProcessPoolExecutor
    chunks = [universe[universe.season == s] for s in SEASONS]
    with ProcessPoolExecutor(max_workers=min(os.cpu_count() or 2, len(jobs) * len(chunks))) as ex:
        futs = [[ex.submit(_project_chunk, p, c) for c in chunks] for p in jobs]
        return [pd.concat([f.result() for f in fs], ignore_index=True) for fs in futs]


def universe_rows() -> pd.DataFrame:
    """Box-score rows plus the players on the field with no stat line (actual 0)."""
    pw = D.player_weeks()
    rows = pw[pw.season.isin(SEASONS) & pw.week.isin(WEEKS) & pw.position.isin(["WR", "TE"]) & (pw["season_type"] == "REG")]
    rows = (rows.rename(columns={"name_key": "nk"})
            .drop_duplicates(KEY)[KEY + ["position", "team", "targets", "receptions", "receiving_yards", "receiving_tds",
                                         "rushing_tds", "carries", "attempts", "dk_fp"]])
    have = set(zip(rows.season, rows.week, rows.nk))
    sn = D.snaps()
    sn = sn[sn.season.isin(SEASONS) & sn.week.isin(WEEKS) & sn.position.isin(["WR", "TE"]) & (sn.offense_snaps > 0)]
    sn = sn[sn.groupby(["season", "week", "name_key"]).team.transform("nunique") == 1]   # a name on two teams: can't tell who
    sn = sn.drop_duplicates(["season", "week", "name_key"]).rename(columns={"name_key": "nk"})
    sn = sn[[(s, w, n) not in have for s, w, n in zip(sn.season, sn.week, sn.nk)]]
    zeros = sn[KEY + ["position", "team"]].assign(team=lambda x: x.team.map(D.canon_team))
    for c in ("targets", "receptions", "receiving_yards", "receiving_tds", "rushing_tds", "carries", "attempts", "dk_fp"):
        zeros[c] = 0.0
    u = pd.concat([rows.assign(no_row=False), zeros.assign(no_row=True)], ignore_index=True)
    g = M._games_idx()[["season", "week", "team", "opp"]]
    return u.merge(g, on=["season", "week", "team"], how="left")


def _rel(d: pd.DataFrame) -> np.ndarray:
    return np.array([M.def_target_rel(o, p, s, w) if isinstance(o, str) else 1.0
                     for o, p, s, w in zip(d.opp, d.position, d.season, d.week)])


def _scales(d: pd.DataFrame) -> dict:
    return {pos: {s: float(g[STATS[s]].sum() / g[f"p_{s}"].sum()) for s in STATS} for pos, g in d.groupby("position")}


def _td_scales(d: pd.DataFrame) -> dict:
    """Receiving TDs scored / expected (unscaled projection), per position. Learned on the same rows as the other scales."""
    g = d.dropna(subset=["p_rec_td"])
    return {pos: float(x.receiving_tds.sum() / x.p_rec_td.sum()) for pos, x in g.groupby("position")}


def _def_slope(d: pd.DataFrame, scales: dict) -> float:
    base = np.array([scales[p]["tgt"] for p in d.position]) * d.p_tgt.to_numpy()
    x = base * (_rel(d) - 1.0)
    y = d.targets.to_numpy() - base
    return float((x * y).sum() / (x * x).sum())


def _rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def _old_view(d: pd.DataFrame) -> pd.DataFrame:
    """The same rows with the OLD model's projections in the p_* columns (so _scales/_apply can be reused on them)."""
    return d.assign(p_tgt=d.p_tgt_old, p_rec=d.p_rec_old, p_rec_yds=d.p_rec_yds_old)


def _apply(d: pd.DataFrame, scales: dict, b: float) -> dict:
    """Scaled projections (and the defense nudge) for rows `d`, per stat."""
    adj = 1.0 + b * (_rel(d) - 1.0)
    return {s: np.array([scales[p][s] for p in d.position]) * d[f"p_{s}"].to_numpy() * adj for s in STATS}


def _tier_table(new: pd.DataFrame) -> list[str]:
    bins = [0, 1.5, 3, 4.5, 6, 8, 99]
    lab = ["< 1.5", "1.5-3", "3-4.5", "4.5-6", "6-8", "8+"]
    t = new.assign(tier=pd.cut(new.p_tgt, bins, right=False, labels=lab))
    L = ["| position | projected targets | players | with a stat row | actual / projected targets | receptions | rec yards |",
         "|---|---|--:|--:|--:|--:|--:|"]
    for pos in ("WR", "TE"):
        for tier, g in t[t.position == pos].groupby("tier", observed=True):
            L.append(f"| {pos} | {tier} | {len(g)} | {100 * (1 - g.no_row.mean()):.0f}% | {g.targets.sum() / g.p_tgt.sum():.2f} | "
                     f"{g.receptions.sum() / g.p_rec.sum():.2f} | {g.receiving_yards.sum() / g.p_rec_yds.sum():.2f} |")
    return L


GRID = [("K_SHARE_REC 0.5", dict(K_SHARE_REC=0.5)), ("K_SHARE_REC 1 (shipped)", {}),
        ("K_SHARE_REC 2", dict(K_SHARE_REC=2.0)), ("K_SHARE_REC 3 (the old value)", dict(K_SHARE_REC=3.0)),
        ("K_SHARE_REC 5", dict(K_SHARE_REC=5.0)),
        ("K_YPT 28 (the old value)", dict(K_YPT=28.0)), ("K_YPT 100", dict(K_YPT=100.0))]
INVOLVED_HIST_TGT = 3.5      # the shrinkage grid's K-independent "involved": average targets over the player's last 8 games


def _hist_targets(u: pd.DataFrame, games: int = 8) -> np.ndarray:
    """Each row's average targets over the player's previous `games` games (NaN under 3): a usage measure that does not
    depend on any shrinkage setting, so every setting can be scored on the same rows."""
    pw = D.player_weeks()
    pw = pw[pw.position.isin(["WR", "TE"])].sort_values(["season", "week"])
    by_player = {k: (g.season.to_numpy() * 100 + g.week.to_numpy(), g.targets.to_numpy(float))
                 for k, g in pw.groupby(["name_key", "position"])}
    empty = (np.array([]), np.array([]))
    out = []
    for r in u.itertuples():
        when, tg = by_player.get((r.nk, r.position), empty)
        before = tg[when < r.season * 100 + r.week][-games:]
        out.append(before.mean() if len(before) >= 3 else np.nan)
    return np.array(out, float)


def _score(p: pd.DataFrame) -> dict:
    """Pooled out-of-season RMSE per stat: each season scored with the scale learned on the other two."""
    sq = {s: 0.0 for s in STATS}
    for test in SEASONS:
        tr, te = p[p.season != test], p[p.season == test]
        pr = _apply(te, _scales(tr), 0.0)
        for s, col in STATS.items():
            sq[s] += float(((pr[s] - te[col].to_numpy()) ** 2).sum())
    return {s: float(np.sqrt(v / len(p))) for s, v in sq.items()}


def _grid(u: pd.DataFrame) -> list[str]:
    """Is the shipped shrinkage near the best of its neighbours? Every setting is scored on the SAME rows, chosen by a usage
    measure that does not depend on any shrinkage (trailing 8-game average targets), each with its own scale learned on the
    other two seasons -- so this isolates the shrinkage. (Choosing rows by each setting's own projection would compare
    different players under different settings.)"""
    frames = project_many([{**NEW, **over} for _, over in GRID], u)
    keys = set.intersection(*[set(zip(f.season, f.week, f.nk)) for f in frames])
    hist = pd.Series(_hist_targets(u), index=pd.MultiIndex.from_frame(u[KEY]))
    L = ["", "## Shrinkage check (same rows for every setting; each scale learned on the other two seasons)", "",
         f"Rows: {len(keys)} WR/TE player-weeks; \"involved\" = averaged >= {INVOLVED_HIST_TGT:g} targets over their last 8 games "
         f"({int((hist.reindex(list(keys)) >= INVOLVED_HIST_TGT).sum())} rows). Pooled out-of-season RMSE.", "",
         "| setting | involved: targets | receptions | rec yards | everyone: targets | receptions | rec yards |",
         "|---|--:|--:|--:|--:|--:|--:|"]
    for (label, _), f in zip(GRID, frames):
        f = f[[k in keys for k in zip(f.season, f.week, f.nk)]]
        inv = f[(hist.reindex(pd.MultiIndex.from_frame(f[KEY])) >= INVOLVED_HIST_TGT).to_numpy()]
        a, b = _score(inv), _score(f)
        L.append(f"| {label} | " + " | ".join(f"{a[s]:.3f}" for s in STATS) + " | " + " | ".join(f"{b[s]:.3f}" for s in STATS) + " |")
    return L


def _shipped() -> dict:
    return dict(K_SHARE_REC=M.K_SHARE_REC, K_YPT=M.K_YPT, DEF_TGT_B=M.DEF_TGT_B, REC_SCALE=M.REC_SCALE)


def _audit_chunk(params: dict, chunk: pd.DataFrame) -> pd.DataFrame:
    """Worker: the production projection (DK points, expected receiving TDs) and the trailing-average baseline for each row."""
    from matchup_model.opp import blend as BL
    _set(params)
    fp, lam, naive = [], [], []
    for r in chunk.itertuples():
        o = M.opp_line(r.nk, r.position, r.season, r.week, by="name", injury_adj=INJURY_ADJ)
        ok = o.get("dk_fp") is not None
        fp.append(o["dk_fp"] if ok else np.nan)
        lam.append(o["line"].get("rec_td", np.nan) if ok else np.nan)
        naive.append(BL._naive_fp(r.nk, r.position, r.season, r.week))
    return chunk.assign(fp=fp, lam=lam, naive=naive)


def _w_fit(tr: pd.DataFrame, col: str) -> float:
    """Weight on the opportunity model that minimises squared error of w*opp + (1-w)*trailing average (as backtest._fit_blend)."""
    x, y = (tr[col] - tr.naive).to_numpy(), (tr.dk_fp - tr.naive).to_numpy()
    return float(np.clip((x * y).sum() / (x * x).sum(), 0.0, 1.0))


def audit(u: pd.DataFrame) -> list[str]:
    """What the change does to the quantities downstream of the receiving line -- DK fantasy points and P(at least one TD) --
    for the previous constants vs the shipped ones, on everyone who played (no outcome-based selection)."""
    from concurrent.futures import ProcessPoolExecutor
    from matchup_model.opp.blend import BLEND_W
    chunks = [u[u.season == s] for s in SEASONS]
    with ProcessPoolExecutor(max_workers=min(os.cpu_count() or 2, 2 * len(chunks))) as ex:
        futs = {w: [ex.submit(_audit_chunk, p, c) for c in chunks] for w, p in (("old", BEFORE), ("new", _shipped()))}
        res = {w: pd.concat([f.result() for f in fs], ignore_index=True) for w, fs in futs.items()}
    keep = KEY + ["position", "dk_fp", "receiving_tds", "rushing_tds", "targets", "carries", "attempts", "no_row", "naive"]
    d = res["old"][keep + ["fp", "lam"]].merge(res["new"][KEY + ["fp", "lam"]], on=KEY, suffixes=("_old", "_new"))
    d = d.dropna(subset=["fp_old", "fp_new", "naive"])
    d["td"] = ((d.receiving_tds + d.rushing_tds) >= 1).astype(int)
    d["inv3"] = (d.targets + d.carries + d.attempts) >= 3
    d["p_old"], d["p_new"] = 1 - np.exp(-d.lam_old), 1 - np.exp(-d.lam_new)

    L = ["", "## Downstream check: DK fantasy points and anytime-TD probability", "",
         f"The shipped model against the previous one, {len(d)} WR/TE player-weeks (2023-25 weeks 5-18). Nothing is learned here, so "
         "these are not out-of-season numbers for the scale -- they show what the change does to DK points and TD odds.", "",
         "| players | pos | n | mean actual DK pts | mean projected, previous | recalibrated | RMSE trailing avg | previous | recalibrated |",
         "|---|---|--:|--:|--:|--:|--:|--:|--:|"]
    for pop, sub in (("everyone who played", d), ("only those who ended up with 3+ targets/carries/attempts", d[d.inv3])):
        for pos in ("WR", "TE"):
            g = sub[sub.position == pos]
            L.append(f"| {pop} | {pos} | {len(g)} | {g.dk_fp.mean():.2f} | {g.fp_old.mean():.2f} | {g.fp_new.mean():.2f} | "
                     f"{_rmse(g.naive, g.dk_fp):.3f} | {_rmse(g.fp_old, g.dk_fp):.3f} | {_rmse(g.fp_new, g.dk_fp):.3f} |")
    L += ["", "The second population is the one `matchup_model.opp.backtest` and `td_calib` score (they keep only players who ended up "
          "with 3+ / 2+ touches). Selecting on the outcome inflates the actuals, so a model that over-projects looks best there; "
          "that is why those two reports cannot be used to compare the old and new models, and why they were not regenerated.", "",
          "Blend weight on the opportunity model (the rest is the player's trailing DK average); fitted on 2023-24, scored on 2025, "
          "everyone who played:", "",
          "| pos | production weight | best weight, previous model | best, recalibrated | 2025 RMSE: trailing avg | recalibrated at production weight | at its best weight |",
          "|---|--:|--:|--:|--:|--:|--:|"]
    for pos in ("WR", "TE"):
        g = d[d.position == pos]
        tr, te = g[g.season <= 2024], g[g.season == 2025]
        w_old, w_new, w_prod = _w_fit(tr.assign(opp=tr.fp_old), "opp"), _w_fit(tr.assign(opp=tr.fp_new), "opp"), BLEND_W[pos]
        at = lambda w: _rmse(w * te.fp_new + (1 - w) * te.naive, te.dk_fp)  # noqa: E731
        L.append(f"| {pos} | {w_prod:.2f} | {w_old:.2f} | {w_new:.2f} | {_rmse(te.naive, te.dk_fp):.3f} | {at(w_prod):.3f} | {at(w_new):.3f} |")
    L += ["", "The production weights were fit on the outcome-selected population with the old model; with the recalibrated model the "
          "blend is nearly flat between them and the best weight, so they are left alone (the blend only moves `fp` and `role_ratio`; "
          "the prop lines come straight from the opportunity model).", "",
          "P(at least one TD) = 1 - exp(-expected receiving TDs), against whether the player scored (receiving or rushing):", "",
          "| players | n | actual rate | mean P, previous | recalibrated | Brier, previous | recalibrated | flat average |",
          "|---|--:|--:|--:|--:|--:|--:|--:|"]
    for pop, sub in (("everyone who played", d), ("trailing 8-game average >= 3.5 targets",
                                                   d[_hist_targets(d) >= INVOLVED_HIST_TGT])):
        base = sub.td.mean()
        L.append(f"| {pop} | {len(sub)} | {100 * base:.1f}% | {100 * sub.p_old.mean():.1f}% | {100 * sub.p_new.mean():.1f}% | "
                 f"{np.mean((sub.p_old - sub.td) ** 2):.4f} | {np.mean((sub.p_new - sub.td) ** 2):.4f} | {np.mean((base - sub.td) ** 2):.4f} |")
    return L


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    u = universe_rows()
    print(f"universe: {len(u)} WR/TE player-weeks ({int(u.no_row.sum())} on the field with no stat line)", flush=True)
    old, new = project_many([OLD, NEW], u)
    # score both models on identical rows; who counts as 'involved' is the NEW model's call (its projected targets)
    new = new.merge(old[KEY + ["p_tgt", "p_rec", "p_rec_yds"]], on=KEY, suffixes=("", "_old"))
    new["involved"] = new.p_tgt >= MIN_PROJ_TGT
    print(f"projected: {len(new)} rows, {int(new.involved.sum())} with >= {MIN_PROJ_TGT:g} projected targets", flush=True)

    rows = []
    for test in SEASONS:
        tr = new[(new.season != test) & new.involved]
        sc, sc_old = _scales(tr), _scales(_old_view(tr))
        b = _def_slope(tr, sc)
        for pop in ("involved", "everyone"):
            te = new[(new.season == test) & (new.involved if pop == "involved" else True)]
            pr_scale, pr_def, pr_old_scaled = _apply(te, sc, 0.0), _apply(te, sc, b), _apply(_old_view(te), sc_old, 0.0)
            for s, col in STATS.items():
                rows.append(dict(season=test, pop=pop, stat=s, n=len(te),
                                 old=_rmse(te[f"p_{s}_old"], te[col]), old_scaled=_rmse(pr_old_scaled[s], te[col]),
                                 scaled=_rmse(pr_scale[s], te[col]), final=_rmse(pr_def[s], te[col]),
                                 old_bias=float(np.mean(te[f"p_{s}_old"] - te[col])), new_bias=float(np.mean(pr_def[s] - te[col]))))
    R = pd.DataFrame(rows)

    def pooled(g, c):
        return float(np.sqrt((g[c] ** 2 * g.n).sum() / g.n.sum()))

    final_tr = new[new.involved]
    final_sc = _scales(final_tr)
    final_td = _td_scales(final_tr)
    final_b = _def_slope(final_tr, final_sc)

    L = ["# WR/TE receiving recalibration (2023-25; every number scored with constants learned on the other two seasons)", "",
         f"{len(new)} WR/TE player-weeks (weeks 5-18; {int(new.no_row.sum())} are players on the field with no stat line, true zeros). "
         f"New shrinkage: K_SHARE_REC {NEW['K_SHARE_REC']:g}, K_YPT {NEW['K_YPT']:g} (old 3 / 28). Constants are learned on the "
         f"{len(final_tr)} rows projected for >= {MIN_PROJ_TGT:g} targets (\"involved\"): the players that get prop lines and DFS salaries. "
         "`td` is the multiplier on expected receiving TDs, learned the same way; the TD rate runs on the unscaled targets "
         "(see the downstream check: scaling TDs with the volume scale leaves involved players ~10% short on TDs).", "",
         "## Out-of-season RMSE", "",
         "| held-out season | players | stat | old | new shrinkage + scale | + defense nudge | old mean error | new mean error |",
         "|---|---|---|--:|--:|--:|--:|--:|"]
    for r in R.sort_values(["pop", "season", "stat"], ascending=[False, True, True]).itertuples():
        L.append(f"| {r.season} | {r.pop} ({r.n}) | {r.stat} | {r.old:.3f} | {r.scaled:.3f} | {r.final:.3f} | "
                 f"{r.old_bias:+.3f} | {r.new_bias:+.3f} |")
    L += ["", "## Where the gain comes from (pooled over the three held-out seasons)", "",
          "`old + scale only` re-scales the OLD model's projections by a constant learned the same way, so the step from there "
          "to `new shrinkage + scale` is the shrinkage change alone, and the last column is the defense nudge.", "",
          "| players | stat | old | old + scale only | new shrinkage + scale | + defense nudge | total change |",
          "|---|---|--:|--:|--:|--:|--:|"]
    for pop in ("involved", "everyone"):
        for s in STATS:
            g = R[(R["pop"] == pop) & (R.stat == s)]
            v = {c: pooled(g, c) for c in ("old", "old_scaled", "scaled", "final")}
            L.append(f"| {pop} | {s} | {v['old']:.3f} | {v['old_scaled']:.3f} | {v['scaled']:.3f} | {v['final']:.3f} | "
                     f"{100 * (v['final'] / v['old'] - 1):+.1f}% |")
    L += ["", "## Why one constant per position, and why it is learned on the involved players", "",
          "Actual / projected by projected-usage tier under the new shrinkage, before any scale. Over-projection is steepest for "
          "depth players (and many of them play without a single target); from 4.5 projected targets up it is a fairly steady "
          "0.82-0.89. In a one-off cross-season test when the constants were chosen (not re-run here), a usage-dependent "
          "scale beat the constant by 0.5-1.8% RMSE across everyone but by < 0.2% among players projected for 3+ targets, "
          "so the constant is kept.", ""]
    L += _tier_table(new)
    if "--grid" in argv:
        L += _grid(u)
    L += audit(u)
    L += ["", "## Constants for model.py (learned on all three seasons)", "", "```python",
          f"K_SHARE_REC = {NEW['K_SHARE_REC']}", f"K_YPT = {NEW['K_YPT']}", "REC_SCALE = {"]
    for pos in ("WR", "TE"):
        s = final_sc[pos]
        L.append(f'    "{pos}": {{"tgt": {s["tgt"]:.3f}, "rec": {s["rec"]:.3f}, "rec_yds": {s["rec_yds"]:.3f}, '
                 f'"td": {final_td[pos]:.3f}}},')
    L += ["}", f"DEF_TGT_B = {final_b:.2f}", "```"]
    text = "\n".join(L)
    print(text)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
