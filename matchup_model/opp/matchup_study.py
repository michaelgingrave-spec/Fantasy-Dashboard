"""Matchup mismatches and usage trends: can they say who will beat his normal -- and does the market already know?

    python -m matchup_model.opp.matchup_study            # the whole study -> matchup_model/matchup_study_report.md (~5 min; the first run also projects
                                                         #   ~20k player-weeks through the shipped model, ~15 min, cached in data/dfs/matchup/_model by its constants)
    python -m matchup_model.opp.matchup_study --forward  # after a new week is graded: refresh only the 2026-lines section (a few minutes)
    python -m matchup_model.opp.matchup_study --render   # re-render the last run (e.g. after editing matchup_study_verdict.md)
    python -m matchup_model.opp.matchup_study --quick    # same pipeline, far fewer bootstrap reps, writes matchup_study_report_quick.md (a smoke test)

The question: from the Data Suite tables (alignments, man/zone, shells, personnel, box, rush success ...), plus each player's own usage trend,
can we call -- as a yes/no, not a line -- that a WR/TE/RB will beat his normal usage or efficiency this week?

Built to be hard to fool:
  * as-of features only (ds_features.py): every number comes from games strictly before the one being predicted;
  * "normal" is the shipped model's own projection (the walk-forward `opp_line`), and the label is "actual > the CONDITIONAL MEDIAN of that stat
    given his projected level", the median learned on the training seasons only -- so level, skew and ceilings cannot help, every label is ~50/50,
    and any AUC above 0.5 is information beyond the level;
  * leave-one-season-out, nested feature sets (B0 level, B1 + his own profile, B2 + the defense, B3 + the mismatch), XGBoost and a heavily
    regularised logistic, with player-clustered bootstrap intervals on every AUC increment;
  * the most favourable test of a mismatch -- the top vs the bottom decile of each feature -- with Benjamini-Hochberg over every test run;
  * seasons whose weekly export is filtered on the week's outcome are detected and dropped (the 2025 WR/TE export) instead of trusted;
  * and the question a bettor cares about: on the real 2026 DK/FD lines, does the signal predict OVER beyond the book's own no-vig
    probability, and do the lines already move with it?
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from math import erfc, sqrt
from pathlib import Path
from typing import Callable, NamedTuple

import numpy as np
import pandas as pd

from dfs.names import normalize_name
from dfs.props import DATA, _amer_to_dec, _sigma, place_bet
from dfs import historical_lines as HL
from matchup_model.config import OUT
from matchup_model.opp import data as D
from matchup_model.opp import ds_features as F
from matchup_model.opp import market_edge as ME
from matchup_model.opp import model as M
from matchup_model.opp import receiving_calibration as RC
from matchup_model.opp import restate
from matchup_model.opp import team_volume as TV
from matchup_model.opp.blend import INJURY_ADJ
from matchup_model.opp.restate import _md

REPORT = Path(__file__).resolve().parents[1] / "matchup_study_report.md"
VERDICT = Path(__file__).resolve().parents[1] / "matchup_study_verdict.md"      # hand-written, dated; the report quotes it above the generated tables
RESULTS = OUT / "matchup_study_results.pkl"
HISTORY = (2022, 2023, 2024, 2025)     # seasons with weekly Data Suite exports; the outcome-filtered ones are dropped at run time
FIRST_WEEK, MIN_PRIOR = 5, 3           # a row needs >= 3 prior games on both sides
FORWARD_FROM = 6                       # 2026 weeks nobody had seen when the hypotheses below were fixed
LINE_KEYS = {"b_tgt": "tgt", "b_rec": "rec", "b_rec_yds": "rec_yds", "b_rush_att": "rush_att", "b_rush_yds": "rush_yds"}
REPS = {"auc": 300, "tails": 500, "slope": 800, "logit": 800}

WRTE_BASE = ["b_tgt", "b_rec", "b_rec_yds", "is_te"]
RB_BASE = ["b_rush_att", "b_rush_yds", "b_rec", "b_rec_yds"]
SET_LABELS = {"B0": "his projected level only", "B1": "+ his own usage / efficiency profile and trends", "B2": "+ the opposing defense",
              "B3": "+ the mismatch (his profile x their weakness)", "B4": "+ ONLY a short pre-specified list of mismatch features (so noise cannot drown them)",
              "B5": "+ ONLY the short route-type mismatch list (his route mix x what the defense allows on each route type)",
              "B6": "+ every route-type feature (route mix, allowed by route type and depth bucket, mismatch)"}
STEPS = (("B0", "B1"), ("B1", "B2"), ("B1", "B3"), ("B1", "B4"), ("B1", "B5"), ("B1", "B6"))
# the handful of features the question is really about, fixed in advance: alignment, man/zone, shell and depth mismatches plus what the defense gives up
WRTE_CORE = ["mm_align_yprr", "mm_align_tprr", "mm_man_yprr", "mm_zone_yprr", "mm_sh_yprr", "mm_th_yprr", "mm_depth_adot", "d_all_yprr_rel", "d_all_tprr_rel", "d_all_adot_dev"]
RB_CORE = ["mm_concept_ypc", "mm_run_plus", "mm_box_hit", "d_ypc_rel", "d_succ_dev", "d_stuff_dev", "d_zone_dev"]
ROUTE_CORE = ["rm_yprr", "rm_tprr", "rm_deep_yprr", "rm_int_yprr", "rm_short_yprr"]          # route-type mismatch, fixed before any outcome was looked at
WRTE_COMPACT = ["p_last1_rte_part", "p_last1_snap_pct", "p_last3_rte_part", "p_last3_snap_pct", "p_rte_part", "p_snap_pct", "p_trend_rte_part",
                "p_trend_snap_pct", "p_tgt_share", "p_last1_tgt_share", "p_trend_tgt_share", "p_tprr", "p_yprr", "p_xfp", "p_fp", "p_rte", "p_wide",
                "p_slot", "p_inline", "p_back"] + WRTE_BASE
RB_COMPACT = ["p_last1_att_pct", "p_last3_att_pct", "p_att_pct", "p_trend_att_pct", "p_last1_snap_pct", "p_last3_snap_pct", "p_snap_pct",
              "p_trend_snap_pct", "p_att", "p_tgt_pct", "p_last1_tgt_pct", "p_trend_tgt_pct", "p_ypc", "p_dcs", "p_xfp", "p_fp"] + RB_BASE


class Target(NamedTuple):
    actual: str                 # the outcome column
    level: list                 # the columns that define "his projected level" the median is conditioned on
    min_col: str | None = None  # a rate is only worth labelling off a minimum sample (routes / carries that week)
    min_val: float = 0.0


WRTE_TARGETS = {
    "targets": Target("targets", WRTE_BASE), "receptions": Target("receptions", WRTE_BASE), "rec yards": Target("receiving_yards", WRTE_BASE),
    "route participation": Target("now_rte_part", ["p_rte_part", "p_snap_pct", "is_te"]),
    "yards per route": Target("now_yprr", ["p_yprr", "p_tprr", "is_te"], "now_rte", 15),
    "targets per route": Target("now_tprr", ["p_tprr", "p_yprr", "is_te"], "now_rte", 15)}
RB_TARGETS = {
    "carries": Target("carries", RB_BASE), "rush yards": Target("rushing_yards", RB_BASE),
    "yards per carry": Target("now_ypc", ["p_ypc", "b_rush_yds"], "now_att", 8),
    "receptions": Target("receptions", RB_BASE), "rec yards": Target("receiving_yards", RB_BASE)}
USAGE_TARGETS = {"WR/TE": ("targets", "rec yards", "route participation"), "RB": ("carries", "rush yards")}


# ── the shipped model's own projections ("normal") ────────────────────────────────────────────────────
def _fingerprint() -> str:
    """Cache key of the baseline projections: the shipped constants and the fitted team-volume coefficients."""
    blob = json.dumps(restate.shipped(), sort_keys=True, default=str) + TV.COEFS_PATH.read_text(encoding="utf-8")
    return hashlib.md5(blob.encode()).hexdigest()[:10]


def _project_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """The production walk-forward line (`opp_line`, injury-adjusted, shipped constants) for each row."""
    out = []
    for r in chunk.itertuples():
        o = M.opp_line(r.nk, r.position, r.season, r.week, by="name", injury_adj=INJURY_ADJ)
        ln = o.get("line", {}) if o.get("dk_fp") is not None else {}
        out.append({b: ln.get(k, np.nan) for b, k in LINE_KEYS.items()})
    return pd.concat([chunk.reset_index(drop=True), pd.DataFrame(out)], axis=1)


def _run_chunks(chunks: list) -> pd.DataFrame:
    """Project the chunks (one per season x position) in parallel -- the model's constants are module-global, so one process per chunk."""
    with ProcessPoolExecutor(max_workers=min(os.cpu_count() or 2, len(chunks))) as ex:
        return pd.concat(list(ex.map(_project_chunk, chunks)), ignore_index=True)


def baselines(group: str, seasons, refresh: bool = False) -> pd.DataFrame:
    """Every WR/TE ('receiving') or RB ('rushing') player-week of weeks 5-18 -- box-score rows plus players on the field with no stat line --
    with the shipped model's projection `b_*`. One cache file per (group, season, model fingerprint)."""
    positions = ("RB",) if group == "rushing" else ("WR", "TE")
    key, parts, todo = _fingerprint(), [], []
    for s in seasons:
        path = OUT / f"matchup_study_{group}_{s}_{key}.csv"
        if path.exists() and not refresh:
            parts.append(pd.read_csv(path))
        else:
            todo.append((s, path))
    if todo:
        u = RC.universe_rows(positions, seasons=[s for s, _ in todo])
        res = _run_chunks([u[(u.season == s) & (u.position == p)] for s, _ in todo for p in positions])
        for s, path in todo:
            x = res[res.season == s]
            x.to_csv(path, index=False)
            parts.append(x)
    return pd.concat(parts, ignore_index=True)


# ── data ────────────────────────────────────────────────────────────────────────────────────────────────
def load_receiving() -> dict:
    """WR/TE Data Suite player-weeks minus any outcome-filtered season, the coverage table, and the evidence for what was dropped."""
    P, C, RT = F.receiving_players(), F.coverage(), F.route_players()
    played = RC.universe_rows(("WR", "TE"), seasons=sorted(P.season.unique()), weeks=range(1, 19))
    presence = F.presence_by_usage(P, played, "targets", low=(0, 2), high=(9, 999))
    dropped = F.filtered_seasons(presence)
    return {"P": P[~P.season.isin(dropped)].reset_index(drop=True), "C": C, "RT": RT[~RT.season.isin(dropped)].reset_index(drop=True), "presence": presence, "dropped": dropped}


def load_rushing() -> dict:
    R, B = F.rushing_players(), F.defense_box()
    played = RC.universe_rows(("RB",), seasons=sorted(R.season.unique()), weeks=range(1, 19))
    presence = F.presence_by_usage(R, played, "carries", low=(1, 2), high=(11, 999))
    dropped = F.filtered_seasons(presence)
    return {"R": R[~R.season.isin(dropped)].reset_index(drop=True), "B": B, "presence": presence, "dropped": dropped}


def _now(table: pd.DataFrame, cols: dict) -> pd.DataFrame:
    """This week's own Data Suite row, for LABELS only (never a feature)."""
    return table[["season", "week", "nk", *cols]].rename(columns=cols).drop_duplicates(["season", "week", "nk"])


def receiving_dataset(d: dict, seasons, refresh: bool = False) -> pd.DataFrame:
    u = baselines("receiving", seasons, refresh)
    u = u[u.b_tgt.notna() & u.opp.notna()].reset_index(drop=True)
    df = pd.concat([F.build_receiving(u, d["P"], d["C"]), F.build_routes(u, d["RT"])], axis=1)
    df = df.merge(_now(d["P"], {"rte": "now_rte", "rte_part": "now_rte_part", "yprr": "now_yprr", "tprr": "now_tprr"}), on=["season", "week", "nk"], how="left")
    df["is_te"] = (df.position == "TE").astype(float)
    keep = (df.p_n_prior >= MIN_PRIOR) & (df.d_n_prior >= MIN_PRIOR) & (df.week >= FIRST_WEEK) & (df.b_tgt >= 2.0)
    return df[keep].reset_index(drop=True)


def rushing_dataset(d: dict, seasons, refresh: bool = False) -> pd.DataFrame:
    u = baselines("rushing", seasons, refresh)
    u = u[u.b_rush_att.notna() & u.opp.notna()].reset_index(drop=True)
    df = F.build_rushing(u, d["R"], d["B"])
    df = df.merge(_now(d["R"], {"att": "now_att", "ypc": "now_ypc"}), on=["season", "week", "nk"], how="left")
    keep = (df.p_n_prior >= MIN_PRIOR) & (df.d_n_prior >= MIN_PRIOR) & (df.week >= FIRST_WEEK) & (df.b_rush_att >= 4.0)
    return df[keep].reset_index(drop=True)


def feature_sets(df: pd.DataFrame, level: list, base: list, core: list = (), route_core: list = ()) -> dict:
    """Nested sets. B0 is exactly what the label's median was conditioned on, so its AUC has to sit at 0.5 -- a check on the labels themselves.
    B4 (only when `core` is given) is B1 plus just that short list, the fair test for a small real signal that ~100 noise features would bury.
    B5 / B6 (only when `route_core` is given) do the same for the route-type features: the short list, and all of them."""
    p = [c for c in df if c.startswith("p_")]
    d = [c for c in df if c.startswith("d_") and c != "d_n_prior"]
    m = [c for c in df if c.startswith("mm_")]
    b1 = list(dict.fromkeys([*level, *base, *p]))
    sets = {"B0": list(level), "B1": b1, "B2": b1 + d, "B3": b1 + d + m}
    if core:
        sets["B4"] = b1 + [c for c in core if c in df.columns]
    if route_core:
        sets["B5"] = b1 + [c for c in route_core if c in df.columns]
        sets["B6"] = b1 + [c for c in df if c.startswith(("rp_", "rd_", "rm_")) and not c.endswith("n_prior")]
    return sets


# ── labels, models, intervals ───────────────────────────────────────────────────────────────────────────
def _jobs() -> int:
    return os.cpu_count() or 2


def median_model():
    import xgboost as xgb
    return xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, n_estimators=120, max_depth=2, learning_rate=0.05, min_child_weight=30,
                            subsample=0.8, tree_method="hist", n_jobs=_jobs(), verbosity=0, random_state=0)


def xgb_clf():
    import xgboost as xgb
    return xgb.XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.5, min_child_weight=25, reg_lambda=8.0,
                             reg_alpha=1.0, tree_method="hist", n_jobs=_jobs(), verbosity=0, eval_metric="logloss", random_state=0)


def logit_clf():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.02, max_iter=2000))


def eligible(df: pd.DataFrame, spec: Target) -> pd.DataFrame:
    f = df[df[spec.actual].notna() & df[spec.level].notna().all(axis=1)]
    return f[f[spec.min_col] >= spec.min_val] if spec.min_col else f


def centred_labels(f: pd.DataFrame, spec: Target, seasons, model=median_model) -> tuple[pd.Series, dict]:
    """'Beats his normal' = actual above the conditional median given his projected level. For each held-out season the median is fit on the OTHER
    seasons; the held-out rows get their label from it (`y`), and so do that fold's training rows (`ytr[season]`, for the classifier)."""
    y = pd.Series(0, index=f.index)
    ytr = {}
    for s in seasons:
        tr, te = f[f.season != s], f[f.season == s]
        mm = model().fit(tr[spec.level].astype(float), tr[spec.actual].astype(float))
        ytr[s] = pd.Series((tr[spec.actual].to_numpy() > mm.predict(tr[spec.level].astype(float))).astype(int), index=tr.index)
        y.loc[te.index] = (te[spec.actual].to_numpy() > mm.predict(te[spec.level].astype(float))).astype(int)
    return y, ytr


def out_of_fold(f: pd.DataFrame, cols: list, kind: str, ytr: dict, seasons) -> pd.Series:
    """P(beats his normal) for every row from a model that never saw that row's season."""
    pred = pd.Series(np.nan, index=f.index)
    for s in seasons:
        tr, te = f[f.season != s], f[f.season == s]
        m = xgb_clf() if kind == "xgb" else logit_clf()
        m.fit(tr[cols].astype(float), ytr[s].loc[tr.index])
        pred.loc[te.index] = m.predict_proba(te[cols].astype(float))[:, 1]
    return pred


def cluster_weights(groups, reps: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap multiplicities from resampling whole clusters (a player-season appears in several rows): (cluster id per row, reps x clusters)."""
    gid, uniq = pd.factorize(pd.Series(np.asarray(groups)))
    g = len(uniq)
    rng = np.random.default_rng(seed)
    return gid, np.stack([np.bincount(rng.integers(0, g, g), minlength=g) for _ in range(reps)]).astype(float)


def auc(y, p, w=None) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p, sample_weight=w))


def delta_auc(y, pa, pb, groups, reps: int, seed: int = 1) -> tuple[float, float, float]:
    """AUC(pb) - AUC(pa) with a 95% interval from a player-clustered bootstrap."""
    y, pa, pb = np.asarray(y), np.asarray(pa), np.asarray(pb)
    gid, W = cluster_weights(groups, reps, seed)
    d = [auc(y, pb, w[gid]) - auc(y, pa, w[gid]) for w in W]
    return auc(y, pb) - auc(y, pa), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def clustered_slope(x, y, groups, reps: int, seed: int = 9) -> tuple[float, float, float]:
    """OLS slope of y on x, 95% interval from a bootstrap over whole clusters (vectorised through per-cluster sums)."""
    x, y, groups = np.asarray(x, float), np.asarray(y, float), np.asarray(groups)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, groups = x[ok], y[ok], groups[ok]
    gid, W = cluster_weights(groups, reps, seed)
    g = W.shape[1]
    s = [np.bincount(gid, weights=v, minlength=g) for v in (np.ones_like(x), x, y, x * x, x * y)]

    def slope(w):
        n, sx, sy, sxx, sxy = (w @ v for v in s)
        return (n * sxy - sx * sy) / (n * sxx - sx ** 2)
    return float(slope(np.ones(g))), *(float(v) for v in np.percentile(slope(W), [2.5, 97.5]))


def benjamini_hochberg(p) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values)."""
    p = np.asarray(p, float)
    m, q, prev = len(p), np.empty(len(p)), 1.0
    for rank, i in zip(range(m, 0, -1), np.argsort(p)[::-1]):
        prev = min(prev, p[i] * m / rank)
        q[i] = prev
    return q


def tail_gaps(f: pd.DataFrame, y, features: list, by: str | None, reps: int, seed: int = 5, min_n: int = 60) -> pd.DataFrame:
    """Rate of 'beats his normal' in the top decile of each feature minus the bottom decile (ranked within `by`, e.g. position), with a
    player-clustered standard error. Rows with a missing feature are in neither tail."""
    y = np.asarray(y, float)
    gid, W = cluster_weights(f.nk + "_" + f.season.astype(str), reps, seed)
    g = W.shape[1]
    rows = []
    for feat in features:
        r = f[feat].groupby(f[by]).rank(pct=True) if by else f[feat].rank(pct=True)
        top, bot = (r >= 0.9).to_numpy(), (r <= 0.1).to_numpy()
        if top.sum() < min_n or bot.sum() < min_n:
            continue

        def rate(m, w=None):
            num, den = np.bincount(gid[m], weights=y[m], minlength=g), np.bincount(gid[m], minlength=g).astype(float)
            return (num.sum() / den.sum()) if w is None else (w @ num) / (w @ den)
        diff = rate(top) - rate(bot)
        se = float(np.std(rate(top, W) - rate(bot, W)))
        z = diff / se if se > 0 else np.nan
        rows.append({"feature": feat, "top": 100 * rate(top), "bottom": 100 * rate(bot), "gap": 100 * diff, "se": 100 * se, "z": z,
                     "p": erfc(abs(z) / sqrt(2)) if np.isfinite(z) else np.nan})
    return pd.DataFrame(rows)


# ── nested out-of-sample study ──────────────────────────────────────────────────────────────────────────
def nested_study(df: pd.DataFrame, targets: dict, base: list, seasons, kinds=("xgb", "logit"), tails_by: str | None = None, median=median_model,
                 core: list = (), route_core: list = ()) -> dict:
    """Everything for one position group: per target and model the AUC of each feature set, the AUC increments with intervals, and the extreme-decile
    tail tests (run on the same labels) with Benjamini-Hochberg over all of them."""
    auc_rows, step_rows, tail_rows, frames = [], [], [], {}
    # the defense and mismatch features, each once (the raw coverage rate and its deviation from the league would rank alike)
    mismatch_feats = [c for c in df if (c.startswith(("d_", "rd_")) and c.endswith(("_dev", "_rel"))) or c.startswith(("mm_", "rm_"))]
    mismatch_feats = [c for c in mismatch_feats if df[c].notna().mean() > 0.9]
    for name, spec in targets.items():
        f = eligible(df, spec)
        sets = feature_sets(df, spec.level, base, core, route_core)
        y, ytr = centred_labels(f, spec, seasons, median)
        frames[name] = (f, y, ytr)
        groups = f.nk + "_" + f.season.astype(str)
        for kind in kinds:
            preds = {s: out_of_fold(f, cols, kind, ytr, seasons) for s, cols in sets.items()}
            auc_rows.append({"target": name, "model": kind, "rows": len(f), "beat rate": f"{100 * y.mean():.1f}%",
                             **{s: f"{auc(y, p):.3f}" for s, p in preds.items()}})
            for a, b in (st for st in STEPS if st[1] in sets):
                d, lo, hi = delta_auc(y, preds[a], preds[b], groups, REPS["auc"])
                step_rows.append({"target": name, "model": kind, "step": f"{a}->{b}", "dAUC": d, "lo": lo, "hi": hi})
        t = tail_gaps(f, y, mismatch_feats, tails_by, REPS["tails"])
        tail_rows.append(t.assign(target=name))
    tails = pd.concat(tail_rows, ignore_index=True)
    tails["q"] = np.nan
    tails.loc[tails.p.notna(), "q"] = benjamini_hochberg(tails.p.dropna().to_numpy())          # a test with no usable standard error is not part of the family
    return {"auc": pd.DataFrame(auc_rows), "steps": pd.DataFrame(step_rows), "tails": tails, "frames": frames}


def usage_study(frames: dict, compact: list, which: tuple, seasons) -> pd.DataFrame:
    """The compact own-usage-trend model alone: out-of-fold AUC, by how many prior games he has, and the hit-rate by score decile."""
    rows = []
    for name in which:
        f, y, ytr = frames[name]
        p = out_of_fold(f, compact, "xgb", ytr, seasons)
        hit = y.groupby(pd.qcut(p, 10, labels=False, duplicates="drop")).mean()
        by_hist = []
        for lo, hi in ((3, 4), (5, 6), (7, 10)):
            m = (f.p_n_prior >= lo) & (f.p_n_prior <= hi)
            if m.sum() > 100:
                by_hist.append(f"{lo}-{hi} games: {auc(y[m], p[m]):.3f}")
        rows.append({"target": name, "AUC": f"{auc(y, p):.3f}", "AUC by prior games": ", ".join(by_hist),
                     "beats-his-normal rate, lowest -> highest score decile": " ".join(f"{100 * v:.0f}" for v in hit)})
    return pd.DataFrame(rows)


def profile_gain(df: pd.DataFrame, cols: list, stats: dict, seasons) -> pd.DataFrame:
    """Does his profile improve the raw stat itself? Leave-one-season-out XGB on (actual - baseline), RMSE vs the baseline, by season."""
    import xgboost as xgb
    rows = []
    for name, (actual, base) in stats.items():
        f = df[df[actual].notna() & df[base].notna()]
        pred = pd.Series(np.nan, index=f.index)
        for s in seasons:
            tr, te = f[f.season != s], f[f.season == s]
            m = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.5, min_child_weight=25, reg_lambda=8.0,
                                 tree_method="hist", n_jobs=_jobs(), verbosity=0, random_state=0)
            m.fit(tr[cols].astype(float), (tr[actual] - tr[base]).astype(float))
            pred.loc[te.index] = te[base] + m.predict(te[cols].astype(float))

        def rmse(a, b):
            return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))
        per = {s: (rmse(f[f.season == s][base], f[f.season == s][actual]), rmse(pred[f.season == s], f[f.season == s][actual])) for s in seasons}
        rows.append({"stat": name, "RMSE baseline -> with profile": f"{rmse(f[base], f[actual]):.3f} -> {rmse(pred, f[actual]):.3f}",
                     "change": f"{100 * (rmse(pred, f[actual]) / rmse(f[base], f[actual]) - 1):+.1f}%",
                     "by season": ", ".join(f"{s}: {100 * (b / a - 1):+.1f}%" for s, (a, b) in per.items())})
    return pd.DataFrame(rows)


# ── RB: the pre-specified run-defense / zone-rate hypotheses ────────────────────────────────────────────
RUN_FEATS = (("d_ypc_rel", 1.0), ("d_succ_dev", 1.0), ("d_stuff_dev", -1.0))


def fit_run_index(ref: pd.DataFrame) -> dict:
    """Run-defense weakness index: mean of the z-scores of (yards per carry allowed vs league, success rate allowed, minus stuff rate) -- fixed
    here, standardised on `ref` so history and 2026 share one unit (1 = one SD of the historical index)."""
    stats = {c: (float(ref[c].mean()), float(ref[c].std())) for c, _ in RUN_FEATS}
    z = sum(s * (ref[c] - stats[c][0]) / stats[c][1] for c, s in RUN_FEATS) / len(RUN_FEATS)
    return {"stats": stats, "sd": float(z.std())}


def run_index(df: pd.DataFrame, fit: dict) -> pd.Series:
    z = sum(s * (df[c] - fit["stats"][c][0]) / fit["stats"][c][1] for c, s in RUN_FEATS) / len(RUN_FEATS)
    return z / fit["sd"]


def zone_index(df: pd.DataFrame, C: pd.DataFrame) -> pd.Series:
    """How much more zone than the league the defense has been playing (trailing, as-of), in SDs of that across the league's defenses."""
    dev = F.coverage_features(C, df)["d_zone_dev"]
    return dev / float(C.groupby("def").zone.mean().std())


def nflverse_run_defense(rows: pd.DataFrame, games: int = 8, k: float = 150.0) -> pd.Series:
    """The same idea from nflverse alone (usable the day the games are played): the opponent's trailing yards per carry allowed to RBs, relative to
    the league, shrunk by `k` carries -- as-of, so only games before the row's week."""
    pw = D.player_weeks()
    g = pw[(pw.season_type == "REG") & (pw.position == "RB")].groupby(["opponent_team", "season", "week"]).agg(car=("carries", "sum"), yds=("rushing_yards", "sum")).reset_index()
    g["opp"], g["key"] = g.opponent_team.map(D.canon_team), g.season * 100 + g.week
    g = g.sort_values(["opp", "key"])
    tab = {d: (x.key.to_numpy(), x[["car", "yds"]].to_numpy(float)) for d, x in g.groupby("opp")}
    keys = np.sort(g.key.unique())
    cache = {}

    def one(d, kk):
        if (d, kk) not in cache:
            ks, arr = tab.get(d, (np.array([]), np.zeros((0, 2))))
            a = arr[max(0, int(np.searchsorted(ks, kk, side="left")) - games):int(np.searchsorted(ks, kk, side="left"))]
            val = np.nan
            if len(a) >= 3:
                S = (a * F.decay(len(a))[:, None]).sum(axis=0)
                lg = g[g.key.isin(keys[keys < kk][-6:])][["car", "yds"]].sum()
                lg_ypc = lg.yds / lg.car
                val = ((S[0] * (S[1] / S[0]) + k * lg_ypc) / (S[0] + k) if S[0] > 0 else lg_ypc) / lg_ypc - 1
            cache[(d, kk)] = val
        return cache[(d, kk)]
    return pd.Series([one(d, F.key(s, w)) for d, s, w in zip(rows.opp, rows.season, rows.week)], index=rows.index)


def rb_slopes(df: pd.DataFrame, C: pd.DataFrame, seasons) -> tuple[pd.DataFrame, dict]:
    """The pre-specified RB hypotheses, season by season, as slopes with player-clustered intervals (and the index fit, to reuse on 2026):
       (1) rush yards above the baseline per SD of run-defense weakness (also with the nflverse-only version of the index);
       (2) yards per carry above his own trailing YPC per SD;  (3) receptions above the baseline per SD of the defense's zone rate."""
    d = df.copy()
    fit = fit_run_index(d)
    d["run_weak"], d["zone_z"] = run_index(d, fit), zone_index(d, C)
    d["nfl_ypc_rel"] = nflverse_run_defense(d)
    d["nfl_z"] = (d.nfl_ypc_rel - d.nfl_ypc_rel.mean()) / d.nfl_ypc_rel.std()
    d["yds_resid"], d["rec_resid"] = d.rushing_yards - d.b_rush_yds, d.receptions - d.b_rec
    d["ypc_resid"] = np.where(d.now_att >= 8, d.now_ypc - d.p_ypc, np.nan)
    tests = [("rush yards vs v3 baseline", "run_weak", "yds_resid", "yds"), ("rush yards vs v3 baseline (nflverse-only index)", "nfl_z", "yds_resid", "yds"),
             ("yards per carry vs his trailing YPC", "run_weak", "ypc_resid", "ypc"), ("receptions vs v3 baseline, zone rate", "zone_z", "rec_resid", "rec")]
    rows = []
    for label, x, y, unit in tests:
        sub = d[d[x].notna() & d[y].notna()]
        for scope, g in [("all seasons", sub)] + [(str(s), sub[sub.season == s]) for s in seasons]:
            if len(g) < 100:
                continue
            grp = g.nk + "_" + g.season.astype(str)
            b, lo, hi = clustered_slope(g[x], g[y], grp, REPS["slope"] if scope == "all seasons" else REPS["slope"] // 2)
            rows.append({"test": label, "scope": scope, "n": len(g), "per +1 SD": b, "lo": lo, "hi": hi, "unit": unit})
    return pd.DataFrame(rows), fit


# ── 2026: does the market already know? ─────────────────────────────────────────────────────────────────
def lines_2026() -> pd.DataFrame:
    """Graded 2026 lines: the book's no-vig P(over), our v3 edge z, whether it went over, plus the player's position, team and opponent."""
    lh = restate.graded_lines()
    if lh.empty:
        return lh
    v3 = restate.restate(lh, restate.shipped())[["week", "player", "market", "our proj"]]
    d = ME.prepare(lh.drop(columns=["our proj"]).merge(v3, on=["week", "player", "market"]))
    if d.empty:
        return d
    pw = D.player_weeks()
    pw = pw[pw.season == 2026]
    d["nk"] = d.player.map(normalize_name)
    d["position"] = d.nk.map(pw.sort_values("week").groupby("name_key").position.last())
    team = pw[["week", "name_key", "team"]].drop_duplicates(["week", "name_key"]).rename(columns={"name_key": "nk"})
    d = d.merge(team, on=["week", "nk"], how="left").merge(M._games_idx()[["season", "week", "team", "opp"]], on=["season", "week", "team"], how="left")
    return d


def _weeks(w: pd.Series) -> str:
    w = sorted(set(int(x) for x in w))
    return f"{w[0]}-{w[-1]}" if len(w) > 1 else str(w[0])


def signal_vs_market(d: pd.DataFrame, signal: str, reps: int) -> dict:
    """What one signal is worth on the real lines, as a decomposition (all per SD of signal, in SDs of the stat, player-clustered): the result minus OUR raw
    projection (`replication` -- does the history's effect show up in 2026?) = how far the book moved its line from our projection with the signal
    (`book_moves`) + the result minus the line (`result_vs_line`, what is LEFT: 0 if the line already contains the signal). Plus the yes/no version:
    logistic 'went over' ~ logit(market P(over)) + our v3 edge z + signal."""
    s = d[d[signal].notna()]
    if len(s) < 60:
        return {"n": len(s)}
    X = np.column_stack([ME.logit(s.p_mkt_over), s.z, s[signal]])
    b, lo, hi = ME.fit_logit(X, s.over.to_numpy(), s.player.to_numpy(), reps=reps)
    move = lambda c: 100 * (1 / (1 + np.exp(-c)) - 0.5)  # noqa: E731

    def slope(y):
        return clustered_slope(s[signal], y, s.player, REPS["slope"] // 2)
    return {"n": len(s), "weeks": _weeks(s.week), "coef": (b[3], lo[3], hi[3]), "pts": (move(b[3]), move(lo[3]), move(hi[3])),
            "replication": slope((s.actual - s.raw) / s.sigma), "book_moves": slope((s.line - s.raw) / s.sigma), "result_vs_line": slope((s.actual - s.line) / s.sigma),
            "corr": float(np.corrcoef(s[signal], (s.line - s.raw) / s.sigma)[0, 1])}


class Signal(NamedTuple):
    """One hypothesis: its history (the signal for every row of `hist`, out of fold for the modelled scores) and a way to score new rows."""
    label: str                      # the row label in the report
    group: str                      # "wrte" or "rb": which feature frame new rows need
    market: str                     # the prop market it is tested on
    hist: pd.DataFrame
    hist_z: pd.Series
    rows_z: Callable                # feature frame of new rows -> the signal for them
    actual: str                     # outcome column in `hist`
    base: str                       # our raw projection column in `hist`
    comp: str                       # dfs.props component name (for the stat's SD)
    needs_history: bool = False     # a modelled score needs >= MIN_PRIOR prior games to be defined


def fit_usage(hist: pd.DataFrame, compact: list, spec: Target) -> tuple[pd.Series, Callable]:
    """Own-usage-trend score: a compact model trained on all the history, standardised on its own out-of-fold scores. Returns (the out-of-fold score of
    every eligible history row, a function that scores new rows with the model fit on everything)."""
    f = eligible(hist, spec)
    seasons = sorted(f.season.unique())
    y, ytr = centred_labels(f, spec, seasons)
    oof = out_of_fold(f, compact, "xgb", ytr, seasons)
    mm = median_model().fit(f[spec.level].astype(float), f[spec.actual].astype(float))
    clf = xgb_clf().fit(f[compact].astype(float), (f[spec.actual].to_numpy() > mm.predict(f[spec.level].astype(float))).astype(int))
    mu, sd = oof.mean(), oof.std()

    def rows_z(rows: pd.DataFrame) -> pd.Series:
        return pd.Series((clf.predict_proba(rows[compact].astype(float))[:, 1] - mu) / sd, index=rows.index)
    return (oof - mu) / sd, rows_z


def build_signals(wr_df: pd.DataFrame, rb_df: pd.DataFrame, rb_fit: dict, C: pd.DataFrame) -> list[Signal]:
    """The four hypotheses, five rows (the WR/TE usage score is tested on two markets, each with its own target)."""
    sigs = []
    for market, target, actual, base, comp in (("rec yds", "rec yards", "receiving_yards", "b_rec_yds", "rec_yds"), ("receptions", "targets", "receptions", "b_rec", "rec")):
        z_hist, rows_z = fit_usage(wr_df, WRTE_COMPACT, WRTE_TARGETS[target])
        sigs.append(Signal(f"WR/TE own-usage-trend score, {market} lines", "wrte", market, wr_df, z_hist, rows_z, actual, base, comp, True))
    z_hist, rows_z = fit_usage(rb_df, RB_COMPACT, RB_TARGETS["rush yards"])
    sigs += [Signal("RB run-defense weakness index, rush yds lines", "rb", "rush yds", rb_df, run_index(rb_df, rb_fit), lambda rows: run_index(rows, rb_fit),
                    "rushing_yards", "b_rush_yds", "rush_yds"),
             Signal("RB opponent zone rate, receptions lines", "rb", "receptions", rb_df, zone_index(rb_df, C), lambda rows: zone_index(rows, C), "receptions", "b_rec", "rec"),
             Signal("RB own-usage-trend score, rush yds lines", "rb", "rush yds", rb_df, z_hist, rows_z, "rushing_yards", "b_rush_yds", "rush_yds", True)]
    return sigs


def history_effect(hist: pd.DataFrame, signal: pd.Series, actual: str, base: str, comp: str) -> tuple[float, float, float]:
    """What 2022-25 says the signal is worth: the result minus OUR projection, in SDs of the stat, per SD of signal -- the number the 2026 'result minus
    line' column should reach if the book's line contains none of the signal, and 0 if it contains all of it."""
    signal = signal.dropna()
    h = hist.loc[signal.index]
    sd = np.array([_sigma(comp, b) for b in h[base]])
    return clustered_slope(signal, (h[actual] - h[base]) / sd, h.nk + "_" + h.season.astype(str), REPS["slope"] // 2)


def frames_2026(d26: pd.DataFrame, rec: dict, rush: dict, sigs: list[Signal]) -> dict:
    """For each signal, the 2026 graded lines of its market with our raw projection and the signal attached (label -> frame)."""
    feats = {}
    for group, pos in (("wrte", ["WR", "TE"]), ("rb", ["RB"])):
        mk = ["rec yds", "receptions"] if group == "wrte" else ["rush yds", "receptions"]
        r = d26[d26.position.isin(pos) & d26.market.isin(mk) & d26.opp.notna()]
        rows = r.drop_duplicates(["week", "nk"])[["season", "week", "nk", "position", "team", "opp"]].reset_index(drop=True)
        if group == "wrte":
            X = pd.concat([_project_chunk(rows), F.player_features(rec["P"], rows)], axis=1)
            X["is_te"] = (X.position == "TE").astype(float)
        else:
            X = pd.concat([_project_chunk(rows), F.build_rushing(rows, rush["R"], rush["B"]).drop(columns=rows.columns)], axis=1)
        feats[group] = (r, X)
    out = {}
    for sg in sigs:
        r, X = feats[sg.group]
        ok = X.p_n_prior >= MIN_PRIOR if sg.needs_history else pd.Series(True, index=X.index)
        z = pd.Series(np.nan, index=X.index)
        z[ok] = sg.rows_z(X[ok])
        out[sg.label] = (r[r.market == sg.market].merge(X[["week", "nk", sg.base]].assign(signal=z), on=["week", "nk"], how="left")
                         .rename(columns={sg.base: "raw"}))
    return out


def devig(over_price, under_price) -> np.ndarray:
    """No-vig P(over) from the two American prices."""
    def imp(a):
        a = np.asarray(a, float)
        return np.where(a > 0, 100 / (a + 100), -a / (-a + 100))
    po, pu = imp(over_price), imp(under_price)
    return po / (po + pu)


def closing_frame(L: pd.DataFrame, hist: pd.DataFrame, market: str, actual: str, base: str, comp: str, book: str, signal: pd.Series | None = None) -> pd.DataFrame:
    """One book's closing lines in one market, set against our raw projection and the result for every row of `hist` that has a line (one line per
    player-week: the one nearest a coin flip if a book lists several), with the signal attached when there is one. Keeps both prices for ROI."""
    cols = ["season", "week", "player", "line", "raw", "actual", "sigma", "z", "p_mkt_over", "over", "signal", "over_price", "under_price"]
    d = L[(L.market == market) & (L.book == book) & L.over_price.notna() & L.under_price.notna()].copy()
    if d.empty:
        return pd.DataFrame(columns=cols)
    d["nk"] = d.player.map(normalize_name)
    d["p_mkt_over"] = devig(d.over_price, d.under_price)
    d = d.assign(dist=(d.p_mkt_over - 0.5).abs()).sort_values("dist").drop_duplicates(["season", "week", "nk"], keep="first")
    h = hist[["season", "week", "nk", actual, base]].assign(signal=(signal.reindex(hist.index) if signal is not None else 0.0))
    m = d.merge(h, on=["season", "week", "nk"], how="inner").rename(columns={"point": "line", actual: "actual", base: "raw"})
    m = m[m.actual.notna() & m.raw.notna() & (m.actual != m.line)].copy()
    m["sigma"] = [_sigma(comp, x) for x in m.line]
    m["z"] = (m.raw - m.line) / m.sigma
    m["over"] = (m.actual > m.line).astype(float)
    m["player"] = m.nk
    return m[cols].reset_index(drop=True)


def historical_frame(L: pd.DataFrame, sg: Signal, book: str) -> pd.DataFrame:
    return closing_frame(L, sg.hist, sg.market, sg.actual, sg.base, sg.comp, book, sg.hist_z)


def profit(price, won) -> np.ndarray:
    """Profit on a 1-unit bet at an American price."""
    price = np.asarray(price, float)
    return np.where(np.asarray(won, bool), np.where(price < 0, 100 / np.abs(price), price / 100), -1.0)


def clustered_mean(x, groups, reps: int, seed: int = 11) -> tuple[float, float, float]:
    """Mean of x with a 95% interval from a bootstrap over whole clusters."""
    x = np.asarray(x, float)
    gid, W = cluster_weights(groups, reps, seed)
    num, den = np.bincount(gid, weights=x, minlength=W.shape[1]), np.bincount(gid, minlength=W.shape[1]).astype(float)
    return float(x.mean()), *(float(v) for v in np.percentile((W @ num) / (W @ den), [2.5, 97.5]))


def follow_table(L: pd.DataFrame, sg: Signal, ks=(0.5, 1.0, 1.5), books=("draftkings", "fanduel")) -> pd.DataFrame:
    """The bet-level version: OVER whenever the signal is >= k SDs, UNDER whenever it is <= -k, at the posted closing price (unit stakes, player-clustered 95%
    intervals), for all of them and for each side. The break-even column is the win rate those prices need."""
    rows = []
    for book in books:
        d = historical_frame(L, sg, book)
        for k in ks:
            for side in ("both", "over", "under"):
                sel = d[(d.signal >= k) | (d.signal <= -k)] if side == "both" else d[d.signal >= k] if side == "over" else d[d.signal <= -k]
                if len(sel) < 30:
                    continue
                over = (sel.signal > 0).to_numpy()
                won = np.where(over, sel.over == 1, sel.over == 0)
                price = np.where(over, sel.over_price, sel.under_price)
                pr = profit(price, won)
                be = float(np.mean(np.where(price < 0, -price / (-price + 100), 100 / (price + 100))))
                rows.append({"book": book, "k": k, "side": side, "bets": len(sel), "win": 100 * float(won.mean()), "break-even": 100 * be,
                             "roi": tuple(100 * v for v in clustered_mean(pr, sel.player, REPS["logit"]))})
    return pd.DataFrame(rows)


STAT_COL = {"rec yds": "receiving_yards", "receptions": "receptions", "rush yds": "rushing_yards"}


def book_pairs(L: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, player, market) listed at BOTH DraftKings and FanDuel, each book's number and prices (the line nearest a coin flip if a book lists
    several) and the result. `results` has season, week, nk and the stat columns, with players who played but recorded nothing as zeros."""
    d = L[L.over_price.notna() & L.under_price.notna() & L.book.isin(["draftkings", "fanduel"])].copy()
    d["nk"] = d.player.map(normalize_name)
    d = d.assign(dist=np.abs(devig(d.over_price, d.under_price) - 0.5)).sort_values("dist").drop_duplicates(["season", "week", "event_id", "nk", "market", "book"], keep="first")
    wide = d.pivot_table(index=["season", "week", "event_id", "nk", "market"], columns="book", values=["point", "over_price", "under_price"], aggfunc="first")
    wide.columns = [f"{a}_{'dk' if b == 'draftkings' else 'fd'}" for a, b in wide.columns]
    res = results[["season", "week", "nk", *STAT_COL.values()]].drop_duplicates(["season", "week", "nk"])
    out = wide.dropna().reset_index().merge(res, on=["season", "week", "nk"], how="left")
    out["actual"] = [row[STAT_COL[m]] for row, m in zip(out.to_dict("records"), out.market)]
    return out[out.actual.notna()].reset_index(drop=True)


def _leg(price, won, push) -> float:
    return 0.0 if push else (float(profit(price, True)) if won else -1.0)


def cross_book(P: pd.DataFrame, reps: int) -> dict:
    """Do the two books disagree on the same prop, and does it pay? (1) how often the numbers differ; (2) single bets: an OVER / UNDER at the lower / higher of
    the two numbers, each at its own posted price; (3) the 'middle' (OVER at the lower number AND UNDER at the higher, one unit each: both win if the result lands
    between); (4) arbitrage at the same number (best over price + best under price whose implied probabilities sum to less than 1)."""
    gaps, legs, mids, arbs = [], [], [], 0
    for m, g in P.groupby("market"):
        gap = (g.point_dk - g.point_fd).abs()
        gaps.append({"market": m, "props": len(g), "same": 100 * float((gap == 0).mean()), "by 1": 100 * float((gap == 1).mean()), "by 2-3": 100 * float(((gap >= 2) & (gap <= 3)).mean()),
                     "by 4+": 100 * float((gap >= 4).mean())})
        rows, mrows = [], []
        for r in g[gap > 0].itertuples():
            lo_b, hi_b = ("dk", "fd") if r.point_dk < r.point_fd else ("fd", "dk")
            lo, hi, x = getattr(r, f"point_{lo_b}"), getattr(r, f"point_{hi_b}"), r.actual
            for leg, book, pt, col in (("OVER at the lower number", lo_b, lo, "over"), ("OVER at the higher number", hi_b, hi, "over"),
                                       ("UNDER at the higher number", hi_b, hi, "under"), ("UNDER at the lower number", lo_b, lo, "under")):
                rows.append((leg, _leg(getattr(r, f"{col}_price_{book}"), (x > pt) if col == "over" else (x < pt), x == pt), r.nk))
            mrows.append((_leg(getattr(r, f"over_price_{lo_b}"), x > lo, x == lo) + _leg(getattr(r, f"under_price_{hi_b}"), x < hi, x == hi), float(lo < x < hi), r.nk))
        for leg in dict.fromkeys(r[0] for r in rows):
            sub_ = [r for r in rows if r[0] == leg]
            legs.append({"market": m, "leg": leg, "bets": len(sub_), "roi": tuple(100 * v for v in clustered_mean([r[1] for r in sub_], [r[2] for r in sub_], reps))})
        if mrows:
            mids.append({"market": m, "pairs": len(mrows), "middle hit": 100 * float(np.mean([r[1] for r in mrows])),
                         "roi": tuple(100 * v / 2 for v in clustered_mean([r[0] for r in mrows], [r[2] for r in mrows], reps))})
        same = g[g.point_dk == g.point_fd]
        imp = lambda a: np.where(a > 0, 100 / (a + 100), -a / (-a + 100))  # noqa: E731
        arbs += int(((np.minimum(imp(same.over_price_dk), imp(same.over_price_fd)) + np.minimum(imp(same.under_price_dk), imp(same.under_price_fd))) < 1).sum())
    return {"gaps": pd.DataFrame(gaps), "legs": pd.DataFrame(legs), "middle": pd.DataFrame(mids), "arb": arbs, "same_number": int((P.point_dk == P.point_fd).sum())}


def load_openers() -> pd.DataFrame:
    """The Wednesday-afternoon opening lines (data/dfs/props/opening_lines.csv): one row per week, player key and market."""
    path = DATA / "props" / "opening_lines.csv"
    if not path.exists():
        return pd.DataFrame(columns=["week", "nk", "market", "line_open"])
    o = pd.read_csv(path)
    o["nk"] = o.player.map(normalize_name)
    return o.drop_duplicates(["week", "nk", "market"])[["week", "nk", "market", "line_open"]]


def timing_table(frames: dict, openers: pd.DataFrame, reps: int) -> pd.DataFrame:
    """Is a signal priced BETWEEN the Wednesday opening line and the late line? The usage scores use only games already played, so they are the same on Wednesday
    as on Sunday. Per SD of signal, in SDs of the stat, player-clustered: what the opening line already held (opening line minus OUR projection), what the late line
    held, what the market ADDED during the week (late minus opening), and what was left at the open and at the late line (the result minus each)."""
    rows = []
    for label, f in frames.items():
        d = f[f.signal.notna()].merge(openers, on=["week", "nk", "market"], how="inner")
        if len(d) < 60:
            continue
        y = {"open_moved": (d.line_open - d.raw) / d.sigma, "close_moved": (d.line - d.raw) / d.sigma, "added": (d.line - d.line_open) / d.sigma,
             "left_open": (d.actual - d.line_open) / d.sigma, "left_close": (d.actual - d.line) / d.sigma}
        rows.append({"signal": label, "n": len(d), "weeks": _weeks(d.week), **{k: clustered_slope(d.signal, v, d.player, reps) for k, v in y.items()}})
    return pd.DataFrame(rows)


RAW_COL = {"rec yds": "b_rec_yds", "receptions": "b_rec", "rush yds": "b_rush_yds"}      # our baseline projection for each market


def execution_frame(L: pd.DataFrame, results: pd.DataFrame, wr_df: pd.DataFrame, rb_df: pd.DataFrame) -> pd.DataFrame:
    """Props listed at both books (`book_pairs`) with our baseline projection for the market (`raw`) and the two books' mean number (`ref`, the app's consensus
    line for two books); props with no projection of ours are dropped."""
    P = book_pairs(L, results)
    proj = pd.concat([wr_df[["season", "week", "nk", "b_rec_yds", "b_rec"]], rb_df[["season", "week", "nk", "b_rec_yds", "b_rec", "b_rush_yds"]]]).drop_duplicates(["season", "week", "nk"])
    P = P.merge(proj, on=["season", "week", "nk"], how="inner")
    P["raw"] = [row[RAW_COL[m]] for row, m in zip(P.to_dict("records"), P.market)]
    P = P[P.raw.notna()].copy()
    P["ref"] = (P.point_dk + P.point_fd) / 2
    return P.reset_index(drop=True)


def _placed(P: pd.DataFrame, over: np.ndarray, how: str) -> tuple[np.ndarray, np.ndarray]:
    """(number, price) of the bet on the side given by `over` for every prop under placement `how`: 'A' = the best PRICE at that book's own number (what the
    app did before 2026-10-15); 'B' = dfs.props.place_bet, the best NUMBER then the best price (what it does now)."""
    nums, prices = [], []
    for r, o in zip(P.itertuples(), over):
        side = "over" if o else "under"
        q = [("draftkings", side, r.point_dk, getattr(r, f"{side}_price_dk")), ("fanduel", side, r.point_fd, getattr(r, f"{side}_price_fd"))]
        pick = place_bet(q, side) if how == "B" else max(q, key=lambda x: _amer_to_dec(x[3]))
        nums.append(pick["point"] if how == "B" else pick[2])
        prices.append(pick["price"] if how == "B" else pick[3])
    return np.array(nums, float), np.array(prices, float)


def _bet_profit(over: np.ndarray, number: np.ndarray, price: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Profit of a 1-unit bet: a win pays decimal-1, a loss costs 1, a result equal to the number is a push."""
    won = np.where(over, x > number, x < number)
    return np.where(x == number, 0.0, np.where(won, np.where(price < 0, 100 / np.abs(price), price / 100), -1.0))


def execution_table(P: pd.DataFrame, reps: int) -> dict:
    """What EXECUTION alone earns on a season of two-book closing lines. Our projection's lean (OVER if our baseline is above the two books' mean number) is held
    fixed and only where and on which side we bet changes: the app's old rule (A, best price) against its new one (B, best number), each for every lean and for
    the UNDER leans only, then two baselines that ignore our projection (always UNDER, always OVER). ROI per unit at the posted prices, pushes refunded,
    player-clustered 95% intervals; `paired` is B minus A on the same bets."""
    over_lean = (P.raw > P.ref).to_numpy()
    keep = (P.raw != P.ref).to_numpy()
    x = P.actual.to_numpy(float)
    cuts = {"all yardage": P.market.isin(["rec yds", "rush yds"]).to_numpy(), "rec yds": (P.market == "rec yds").to_numpy(), "rush yds": (P.market == "rush yds").to_numpy(),
            "receptions": (P.market == "receptions").to_numpy()}

    def one(sel: np.ndarray, over: np.ndarray, how: str):
        n, pr = _placed(P[sel], over[sel], how)
        pnl = _bet_profit(over[sel], n, pr, x[sel])
        won = np.where(over[sel], x[sel] > n, x[sel] < n)[x[sel] != n]
        return pnl, float(won.mean()) if len(won) else float("nan")

    pol, blind = [], []
    for label, how, only_under in (("A  best price (the app before)", "A", False), ("B  best number (the app now)", "B", False),
                                   ("A  UNDER leans only", "A", True), ("B  UNDER leans only", "B", True)):
        for m, cut in cuts.items():
            sel = keep & cut & (~over_lean if only_under else True)
            if sel.sum() >= 30:
                pnl, win = one(sel, over_lean, how)
                pol.append({"policy": label, "market": m, "bets": int(sel.sum()), "over share": 100 * float(over_lean[sel].mean()), "win": 100 * win,
                            "roi": tuple(100 * v for v in clustered_mean(pnl, P.nk[sel], reps))})
    for side, over in (("always UNDER", False), ("always OVER", True)):
        for how, label in (("A", "best price"), ("B", "best number")):
            for m, cut in cuts.items():
                if cut.sum() < 30:
                    continue
                ov = np.full(len(P), over)
                pnl, win = one(cut, ov, how)
                blind.append({"bet": side, "placed": label, "market": m, "bets": int(cut.sum()), "win": 100 * win,
                              "roi": tuple(100 * v for v in clustered_mean(pnl, P.nk[cut], reps))})
    paired = {}
    for label, only_under in (("every lean", False), ("UNDER leans only", True)):
        sel = keep & cuts["all yardage"] & (~over_lean if only_under else True)
        a, _ = one(sel, over_lean, "A")
        b, _ = one(sel, over_lean, "B")
        paired[label] = (int(sel.sum()),) + tuple(100 * v for v in clustered_mean(b - a, P.nk[sel], reps))
    return {"policies": pd.DataFrame(pol), "blind": pd.DataFrame(blind), "paired": paired, "props": len(P), "over_lean": 100 * float(over_lean.mean())}


# H5, frozen 2026-10-15 before any week-6 line was graded: FantasyPoints' own weekly projection against our (already nudged) number.
FP_GAP_CAP = 0.693      # |log(FantasyPoints / ours)| beyond this (a 2x disagreement) is mostly an unresolved-status placeholder, not a view: clipped
FP_GAP_SD = 0.203       # SD of the clipped, within-week-centred gap on weeks 2-5 (the scale the registry's 'per SD' refers to; never re-fit)
FP_MARKETS = ("rec yds", "receptions", "rush yds", "pass yds")      # the history is also shown per market (descriptive)


def fp_gap_frame(lh: pd.DataFrame) -> pd.DataFrame:
    """Graded lines (as the app logged them) with FantasyPoints' gap signal: log(its weekly projection / our nudged number), which is the stored `role_ratio`;
    clipped at FP_GAP_CAP, centred within the week (the level of the gap moves with the model version, a within-week deviation does not) and scaled by FP_GAP_SD."""
    d = ME.prepare(lh)
    if d.empty or "role_ratio" not in d.columns:
        return pd.DataFrame()
    d = d[d.role_ratio.notna() & (d.role_ratio > 0)].copy()
    g = np.log(d.role_ratio.astype(float)).clip(-FP_GAP_CAP, FP_GAP_CAP)
    d["fp_dev"] = (g - g.groupby(d.week).transform("mean")) / FP_GAP_SD
    d["raw"] = d["our proj"].astype(float)
    return d


def fp_gap_table(lh: pd.DataFrame, reps: int) -> pd.DataFrame:
    """Does FantasyPoints' own projection know more than our number and the book's price? `signal_vs_market` on the gap signal for the weeks already seen
    (history, with a per-market breakdown that is descriptive only) and for the forward weeks (the registered test)."""
    d = fp_gap_frame(lh)
    if d.empty:
        return pd.DataFrame()
    hist, fwd = d[d.week < FORWARD_FROM], d[d.week >= FORWARD_FROM]
    cuts = []
    if len(hist):
        cuts += [(f"weeks {_weeks(hist.week)} as logged, all markets", hist)] + [(f"... {m} only (descriptive)", hist[hist.market == m]) for m in FP_MARKETS]
    cuts += [(f"forward (week {FORWARD_FROM}+), all markets", fwd)]
    return pd.DataFrame([{"lines": label, **signal_vs_market(sub, "fp_dev", reps)} for label, sub in cuts])


# the markets we model, as (label, history frame key, prop market, outcome column, our raw projection column, component for the stat's SD)
MARKET_SETS = (("WR/TE rec yds", "wr", "rec yds", "receiving_yards", "b_rec_yds", "rec_yds"), ("WR/TE receptions", "wr", "receptions", "receptions", "b_rec", "rec"),
               ("RB rush yds", "rb", "rush yds", "rushing_yards", "b_rush_yds", "rush_yds"), ("RB receptions", "rb", "receptions", "receptions", "b_rec", "rec"),
               ("RB rec yds", "rb", "rec yds", "receiving_yards", "b_rec_yds", "rec_yds"))


def lean_and_edge(L: pd.DataFrame, wr_df: pd.DataFrame, rb_df: pd.DataFrame, reps: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """On a season of closing lines, per market and book: (1) do the lines LEAN -- the share that went over vs what the no-vig price said, and what a blind UNDER
    (and OVER) bet at the posted price earned; (2) does OUR projection add anything to the closing line -- 'went over' on logit(market) + our edge z."""
    lean, edge = [], []
    for label, which, market, actual, base, comp in MARKET_SETS:
        hist = wr_df if which == "wr" else rb_df
        for book in ("draftkings", "fanduel"):
            d = closing_frame(L, hist, market, actual, base, comp, book)
            if len(d) < 100:
                continue
            gap = clustered_mean(d.over - d.p_mkt_over, d.player, reps)
            under = clustered_mean(profit(d.under_price, d.over == 0), d.player, reps)
            over = clustered_mean(profit(d.over_price, d.over == 1), d.player, reps)
            half = d.week <= d.week.median()
            lean.append({"market": label, "book": book, "n": len(d), "went over": 100 * float(d.over.mean()), "market said": 100 * float(d.p_mkt_over.mean()),
                         "gap": tuple(100 * v for v in gap), "first half over": 100 * float(d.over[half].mean()), "second half over": 100 * float(d.over[~half].mean()),
                         "under roi": tuple(100 * v for v in under), "over roi": tuple(100 * v for v in over)})
            X = np.column_stack([ME.logit(d.p_mkt_over), d.z])
            b, lo, hi = ME.fit_logit(X, d.over.to_numpy(), d.player.to_numpy(), reps=reps)
            move = lambda c: 100 * (1 / (1 + np.exp(-c)) - 0.5)  # noqa: E731
            edge.append({"market": label, "book": book, "n": len(d), "price coef": (b[1], lo[1], hi[1]), "z coef": (b[2], lo[2], hi[2]),
                         "pts": (move(b[2]), move(lo[2]), move(hi[2])), "price sd": float(np.std(ME.logit(d.p_mkt_over)))})
    return pd.DataFrame(lean), pd.DataFrame(edge)


def market_tables(sigs: list[Signal], f26: dict, L: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every signal against every set of lines: the 2026 weeks (all, and the forward weeks nobody had seen), a full season of 2025 closing lines at each
    book, and the two pooled; plus each signal's historical worth season by season (is the effect stable, or fading?)."""
    out, seasons = [], []
    for sg in sigs:
        history = history_effect(sg.hist, sg.hist_z, sg.actual, sg.base, sg.comp)
        sets = []
        if sg.label in f26:
            g = f26[sg.label]
            sets += [(f"2026 weeks 1-{int(g.week.max()) if len(g) else FORWARD_FROM - 1} (all)", g), (f"2026 forward (week {FORWARD_FROM}+)", g[g.week >= FORWARD_FROM])]
        if len(L):
            f25 = {bk: historical_frame(L, sg, bk) for bk in ("draftkings", "fanduel")}
            sets += [("2025 closing, DraftKings", f25["draftkings"]), ("2025 closing, FanDuel", f25["fanduel"])]
            if sg.label in f26:
                sets.append(("2025 DK closing + 2026", pd.concat([f25["draftkings"], f26[sg.label][f25["draftkings"].columns.intersection(f26[sg.label].columns)]], ignore_index=True)))
        for scope, g in sets:
            r = signal_vs_market(g, "signal", REPS["logit"])
            out.append({"signal": sg.label, "lines": scope, "n": r["n"], "history": history, **r})
        for sn in sorted(sg.hist.season.unique()):
            idx = sg.hist_z.dropna().index.intersection(sg.hist.index[sg.hist.season == sn])
            if len(idx) >= 100:
                seasons.append({"signal": sg.label, "season": int(sn), "n": len(idx), "history": history_effect(sg.hist, sg.hist_z[idx], sg.actual, sg.base, sg.comp)})
    return pd.DataFrame(out), pd.DataFrame(seasons)


# ── report ──────────────────────────────────────────────────────────────────────────────────────────────
def _ci(b, lo, hi, nd=3) -> str:
    return f"{b:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}]"


def _auc_md(res: dict) -> list[str]:
    a = res["auc"]
    return _md(a.rename(columns={s: f"AUC {s}" for s in SET_LABELS if s in a.columns}))


def _steps_md(res: dict) -> list[str]:
    s = res["steps"].assign(**{"dAUC [95% interval]": lambda x: [_ci(*v, 4) for v in zip(x.dAUC, x.lo, x.hi)],
                               "interval clears 0": lambda x: np.where(x.lo > 0, "YES", np.where(x.hi < 0, "worse", "no"))})
    return _md(s[["target", "model", "step", "dAUC [95% interval]", "interval clears 0"]])


def _tails_md(res: dict, n: int = 10) -> list[str]:
    t = res["tails"]
    big = t.reindex(t.gap.abs().sort_values(ascending=False).index).head(n)
    show = big.assign(**{"top decile": big.top.round(1).astype(str) + "%", "bottom decile": big.bottom.round(1).astype(str) + "%",
                         "gap (pts)": big.gap.round(1), "SE": big.se.round(1), "p": big.p.round(3), "BH q": big.q.round(3)})
    return _md(show[["target", "feature", "top decile", "bottom decile", "gap (pts)", "SE", "p", "BH q"]])


def _mismatch_md(res: dict) -> list[str]:
    """Every mismatch feature (his profile x their weakness) against every target: top-decile minus bottom-decile gap in points, with its BH q."""
    t = res["tails"]
    t = t[t.feature.str.startswith(("mm_", "rm_"))]
    if t.empty:
        return ["(none)"]
    cell = t.assign(c=t.gap.round(1).map("{:+.1f}".format) + " (q " + t.q.round(2).astype(str) + ")")
    return _md(cell.pivot(index="feature", columns="target", values="c")[list(dict.fromkeys(t.target))].reset_index())


def _tail_summary(res: dict) -> str:
    t = res["tails"]
    return (f"{len(t)} extreme-decile tests; {100 * (t.p < 0.05).mean():.1f}% have p < 0.05 (5% expected by chance); smallest Benjamini-Hochberg q = "
            f"{t.q.min():.3f}; median standard error of a gap {t.se.median():.1f} points, so a true top-vs-bottom gap of about {2.8 * t.se.median():.0f} points "
            "would have been detected about 80% of the time.")


def _market_md(m: pd.DataFrame) -> list[str]:
    cols = ["signal", "lines", "n", "weeks", "history: result minus OUR projection", "on these lines: result minus OUR projection", "on these lines: the book moved its line by",
            "on these lines: result minus the LINE (what is left)", "beyond-the-price coefficient (logit)", "moves P(over) by"]
    rows = []
    for r in m.itertuples():
        base = {"signal": r.signal, "lines": r.lines, "n": r.n, "history: result minus OUR projection": _ci(*r.history)}
        if "coef" not in m.columns or not isinstance(r.coef, tuple):
            rows.append({**base, "weeks": "", "on these lines: result minus the LINE (what is left)": "too few lines"})
            continue
        rows.append({**base, "weeks": r.weeks, "on these lines: result minus OUR projection": _ci(*r.replication), "on these lines: the book moved its line by": _ci(*r.book_moves),
                     "on these lines: result minus the LINE (what is left)": _ci(*r.result_vs_line), "beyond-the-price coefficient (logit)": _ci(*r.coef),
                     "moves P(over) by": f"{r.pts[0]:+.1f} pts [{r.pts[1]:+.1f}, {r.pts[2]:+.1f}]"})
    return _md(pd.DataFrame(rows).reindex(columns=cols))


def _lean_md(t: pd.DataFrame | None) -> list[str]:
    if t is None or t.empty:
        return ["(no historical lines pulled)"]
    r = lambda v: f"{v[0]:+.1f}% [{v[1]:+.1f}, {v[2]:+.1f}]"  # noqa: E731
    out = pd.DataFrame({"market": t.market, "book": t.book, "lines": t.n, "went over": t["went over"].round(1).astype(str) + "%", "market said": t["market said"].round(1).astype(str) + "%",
                        "gap (pts)": [f"{g[0]:+.1f} [{g[1]:+.1f}, {g[2]:+.1f}]" for g in t.gap], "over, first / second half": t["first half over"].round(1).astype(str) + " / " + t["second half over"].round(1).astype(str),
                        "blind UNDER roi": [r(v) for v in t["under roi"]], "blind OVER roi": [r(v) for v in t["over roi"]]})
    return _md(out)


def _edge_md(t: pd.DataFrame | None) -> list[str]:
    if t is None or t.empty:
        return []
    out = pd.DataFrame({"market": t.market, "book": t.book, "lines": t.n,
                        "market-price coefficient": [_ci(*v, 2) if sd >= 0.05 else "n/a: prices are flat (~-110/-110)" for v, sd in zip(t["price coef"], t["price sd"])],
                        "our edge z coefficient (per SD)": [_ci(*v) for v in t["z coef"]], "+1 SD of edge moves P(over) by": [f"{v[0]:+.1f} pts [{v[1]:+.1f}, {v[2]:+.1f}]" for v in t.pts]})
    return ["Does our raw projection add anything to the closing line?", ""] + _md(out)


def _cross_md(c: dict | None) -> list[str]:
    if not c:
        return ["(no historical lines pulled)"]
    g, lg, mid = c["gaps"], c["legs"], c["middle"]
    out = ["| market | props at both books | same number | differ by 1 | by 2-3 | by 4+ |", "|---|--:|--:|--:|--:|--:|"]
    out += [f"| {r.market} | {r.props} | {r.same:.0f}% | {r['by 1']:.0f}% | {r['by 2-3']:.0f}% | {r['by 4+']:.0f}% |" for _, r in g.iterrows()] + [""]
    t = pd.DataFrame({"market": lg.market, "bet (only where the numbers differ)": lg.leg, "bets": lg.bets, "roi per bet": [f"{v[0]:+.1f}% [{v[1]:+.1f}, {v[2]:+.1f}]" for v in lg.roi]})
    m = pd.DataFrame({"market": mid.market, "pairs": mid.pairs, "result lands between the numbers": mid["middle hit"].round(1).astype(str) + "%",
                      "middle roi per $ staked": [f"{v[0]:+.1f}% [{v[1]:+.1f}, {v[2]:+.1f}]" for v in mid.roi]}) if len(mid) else pd.DataFrame()
    return out + _md(t) + [""] + (_md(m) + [""] if len(m) else []) + [f"Same-number arbitrage: {c['arb']} of {c['same_number']} same-number props had a best over + best under price summing to under 100%."]


def _timing_md(t: pd.DataFrame | None) -> list[str]:
    if t is None or t.empty:
        return ["(no opening lines with a signal)"]
    return _md(pd.DataFrame({"signal": t.signal, "lines": t.n, "weeks": t.weeks, "the opening line already moved by": [_ci(*v) for v in t.open_moved],
                             "the late line moved by": [_ci(*v) for v in t.close_moved], "added during the week": [_ci(*v) for v in t.added],
                             "left at the OPEN (result minus opening line)": [_ci(*v) for v in t.left_open], "left at the late line": [_ci(*v) for v in t.left_close]}))


def _fp_md(t: pd.DataFrame | None) -> list[str]:
    if t is None or t.empty:
        return ["(no graded lines with a FantasyPoints number)"]
    rows = []
    for r in t.itertuples():
        if not isinstance(getattr(r, "coef", None), tuple):
            rows.append({"lines": r.lines, "n": r.n, "weeks": "", "beyond the price and our edge (logit per SD)": "too few lines"})
            continue
        rows.append({"lines": r.lines, "n": r.n, "weeks": r.weeks, "beyond the price and our edge (logit per SD)": _ci(*r.coef),
                     "moves P(over) by": f"{r.pts[0]:+.1f} pts [{r.pts[1]:+.1f}, {r.pts[2]:+.1f}]", "result minus OUR projection": _ci(*r.replication),
                     "the book moved its line by": _ci(*r.book_moves), "result minus the LINE (what is left)": _ci(*r.result_vs_line)})
    return _md(pd.DataFrame(rows))


def _execution_md(e: dict | None) -> list[str]:
    if not e:
        return ["(no historical lines pulled)"]
    r = lambda v: f"{v[0]:+.1f}% [{v[1]:+.1f}, {v[2]:+.1f}]"  # noqa: E731
    pol, bl = e["policies"], e["blind"]
    t = pd.DataFrame({"policy": pol.policy, "market": pol.market, "bets": pol.bets, "OVER share": pol["over share"].round(0).astype(int).astype(str) + "%",
                      "win": pol.win.round(1).astype(str) + "%", "roi per bet": [r(v) for v in pol.roi]})
    b = pd.DataFrame({"bet": bl.bet, "placed at": bl.placed, "market": bl.market, "bets": bl.bets, "win": bl.win.round(1).astype(str) + "%", "roi per bet": [r(v) for v in bl.roi]})
    pr = " ; ".join(f"{k}: {v[1]:+.2f} points of ROI per bet [{v[2]:+.2f}, {v[3]:+.2f}] over {v[0]} bets" for k, v in e["paired"].items())
    return ([f"{e['props']} props listed at both books with a projection of ours; our baseline leans OVER on {e['over_lean']:.0f}% of them.", ""] + _md(t) + ["",
            "Baselines that ignore our projection:", ""] + _md(b) + ["", f"Taking the best number instead of the best price, on the same yardage bets (B minus A): {pr}."])


def _follow_md(t: pd.DataFrame | None) -> list[str]:
    if t is None or t.empty:
        return ["(no historical lines pulled)"]
    out = pd.DataFrame({"book": t.book, "k (SD)": t.k, "side": t.side, "bets": t.bets, "win": t.win.round(1).astype(str) + "%", "break-even": t["break-even"].round(1).astype(str) + "%",
                        "roi per bet": [f"{v[0]:+.1f}% [{v[1]:+.1f}, {v[2]:+.1f}]" for v in t.roi]})
    return _md(out)


def _history_md(h: pd.DataFrame) -> list[str]:
    if h is None or h.empty:
        return ["(none)"]
    t = h.assign(c=[_ci(*v) for v in h.history], n_=h.n.astype(str))
    t["cell"] = t.c + " (n " + t.n_ + ")"
    pv = t.pivot(index="signal", columns="season", values="cell").reset_index()
    pv.columns = [str(c) for c in pv.columns]
    return _md(pv)


H1_LABEL = "RB own-usage-trend score, rush yds lines"        # the one candidate still alive: the bet-level table is for this signal

REGISTRY = [
    "## Forward-test registry (frozen 2026-10-15, rule tightened the same day before any forward week was seen; H5 added the same day, also before any week-6 line was graded)", "",
    f"One test per hypothesis, no re-fitting, on 2026 lines from week {FORWARD_FROM} on. After each week is graded run `python -m matchup_model.opp.matchup_study --forward` "
    "(a few minutes) and the 'forward' rows above fill in. 'Result minus the LINE' is 0 when the book already contains the signal, but it is measured in MEAN yards, and rushing "
    "yards are right-skewed: a mean shift of +0.4 SD can be only +3 points of P(over) (the 2025 RB result: the top score quintile beat the line by +0.45 SD on average but went "
    "over only 52.9% of the time). So the rule is in bet terms: a signal is acted on only if, over at least 300 forward bets at the posted price following it with k = 1, "
    "the 95% interval of ROI per bet is entirely above 0 AND the 'beyond-the-price' coefficient moves P(over) by more than the vig (2.4 points per SD). Until then it is tracked, "
    "not bet. Status as of 2026-10-15:", "",
    "| id | signal | market | where it stands | expectation |", "|---|---|---|---|---|",
    "| H1 | RB own-usage-trend score | RB rush yds | +0.19 to +0.31 SD of the stat per SD of score in every season 2022-25; on 612 DraftKings closing lines of 2025 the book moved only +0.07 of the +0.21 and +0.14 [+0.07, +0.21] was left (both halves of the season, FanDuel the same), almost all of it on RISING usage; but that is only about +3 points of P(over) per SD, and following the score at the posted price returned +0.5% [-11, +12] per bet at k = 1 (211 bets); 2026 weeks 1-5: +0.06 left (interval -0.07 to +0.19) | real but about the size of the vig: tracked, not bet |",
    "| H2 | RB run-defense weakness index | RB rush yds | +0.08 to +0.12 in 2022-24, +0.01 in 2025; on the 2025 closing lines the book moved +0.07 of the +0.09 (priced), -0.05 [-0.14, +0.03] left | dead |",
    "| H3 | RB opponent zone rate | RB receptions | +0.05 per SD in 2022-24; 2025 closing lines: book moved +0.02, about 0 left | dead |",
    "| H4 | WR/TE own-usage-trend score | rec yds, receptions | 2025 closing lines (1,629 / 1,564 DraftKings lines): the book moved +0.09 / +0.13 of the +0.10 / +0.17, +0.02 / +0.04 left (intervals about +/-0.06); 2026: +0.00 / +0.01 left | priced, no edge |",
    f"| H5 | FantasyPoints' own weekly projection vs our nudged number: log(role_ratio), clipped at +/-{FP_GAP_CAP}, centred within the week, in units of {FP_GAP_SD} | rec yds, receptions, rush yds, pass yds, pooled | weeks 2-5 as logged (2,016 lines): FantasyPoints is closer to the truth than our number (result minus OUR projection +0.156 [+0.103, +0.216] SD per SD) but the book's line had moved +0.189 [+0.160, +0.226] toward it, leaving -0.033 [-0.081, +0.025]; beyond the price and our edge +2.1 points of P(over) per SD [-1.0, +5.5] | priced so far: tracked, rule below |", "",
    f"**H5 rule (added 2026-10-15 before any week-{FORWARD_FROM} line was graded; the weeks-2-5 row is a description, it was run once before the rule was frozen).** The one test is the 'forward' row of the "
    "FantasyPoints table above (all markets pooled, the signal exactly as defined, nothing re-fitted). It is read at two fixed checkpoints only, after week 12 (about 3,000 forward lines) and "
    "after week 18, never from the weekly refreshes: FantasyPoints' weight (`blend.FP_SHIFT_CAP`, 0.50 now) is raised only if the 95% interval of the beyond-the-price coefficient is "
    "entirely above 0 AND its point estimate is at least 2.4 points of P(over) per SD; the new weight is then set from the 'result minus OUR projection' slope on the forward lines alone, "
    "and a pass at week 12 is re-checked at week 18 before it stays. Everything else in that table (per market, per week) is descriptive.", ""]

CAVEATS = [
    "## Caveats", "",
    "- **The 2025 WR/TE tables were re-pulled on 2026-10-15.** The first export had been cut by the Data Suite's 1,500-row cap to the top rows by receiving yards (no quiet games; it "
    "overstated 2025 YPRR by ~30%). `weekly_pull.backfill_tables` now filters weeks on the server (`regWeeks=`) and `filtered_seasons` would catch a repeat. Every trailing window is "
    "capped at the previous season, so a score needs >= 3 prior games inside that window.",
    "- **'Our projection beats the close' is not clean for 2025.** The RB and WR/TE constants of the shipped model were fit on 2023-25 box scores, so the 2025 rows of that "
    "table are partly in-sample (the signal tests are not affected: they compare the result with the LINE, and the scores are out of fold). Treat 2025 there as an upper "
    "bound; 2026 is the clean test.",
    "- **'Normal' is our own model.** The labels are centred on the shipped projection's level, whose constants were fit on 2023-25: that can tilt levels but cannot create "
    "a label imbalance (the median is re-fit on the other seasons), and B0 sitting at 0.5 checks it.",
    "- **Which lines.** 2025 is DraftKings and FanDuel CLOSING lines (a snapshot 5 minutes before kickoff, weeks 5-18; 205 of 208 games had props) with each book's own two prices; "
    "2026 is the lines as logged by the app (a few hundred per market, weeks 1-5) with the stored no-vig probability. A bettor who bets earlier than the close faces a softer line, "
    "so the close is the hardest test.",
    "- **What the bet-level tests can and cannot rule out.** At 200-500 bets per signal an ROI interval is about +/-8 to 13 points wide (the payoff of one bet has a standard "
    "deviation of about 0.95 units), so a true +5% edge would show up as 'entirely above zero' about 1 time in 5 at 500 bets (a true +3%: 1 in 10), and confirming +3% with 80% "
    "power takes about 8,000 bets. 'No improvement' therefore means no edge worth more than roughly +8 to 13% ROI; smaller ones are invisible in either direction. The "
    "line-movement tests are far tighter (+/-0.02 SD) because a line move is not a coin flip, which is why they are worth running first.",
    "- **FantasyPoints history.** FantasyPoints did not archive past weekly projections, so H5 exists only for 2026 weeks 2-5 as the app logged them (about 2,000 lines of about 900 "
    "player-weeks; one player's lines in a week share one gap, so the intervals are clustered by player). Those weeks ran on the original (v1) projections, which over-projected "
    "WR/TE, so part of FantasyPoints' edge over our number there was bias our later recalibration removes: the forward weeks (v3 projections) are the clean test. The raw log "
    "ratio, unclipped and uncentred, gives the same picture (+0.45 per unit beyond the price and our edge [-0.04, +0.96]).",
    "- **Many tests.** The WR/TE and RB tails alone are several hundred; Benjamini-Hochberg is applied within each group, and the pre-specified RB hypotheses are the only "
    "ones that were fixed before the outcomes were looked at.",
    "- **Not covered:** QBs, anytime-TD and passing markets, injuries/teammate-out usage shifts, game script, and the defender-level (cornerback / safety) Data Suite tables, "
    "which would be needed for receiver-vs-defender matchups.", ""]


def build_report(rec: dict, rush: dict, wr: dict, rb: dict, gain: pd.DataFrame, usage_wr: pd.DataFrame, usage_rb: pd.DataFrame, slopes: pd.DataFrame,
                 market: pd.DataFrame, history: pd.DataFrame, n_rows: dict, lean: pd.DataFrame | None = None, edge: pd.DataFrame | None = None,
                 follow: pd.DataFrame | None = None, cross: dict | None = None, timing: pd.DataFrame | None = None, fp: pd.DataFrame | None = None,
                 execution: dict | None = None) -> str:
    pres = pd.concat([rec["presence"].assign(table="WR/TE receiving"), rush["presence"].assign(table="RB rushing")]).reset_index().rename(columns={"index": "season"})
    pres = pres.assign(**{"low-usage presence": (100 * pres.low).round(0).astype(int).astype(str) + "%", "high-usage presence": (100 * pres.high).round(0).astype(int).astype(str) + "%",
                          "ratio": pres.ratio.round(2), "verdict": np.where(pres.ratio < 0.4, "FILTERED - dropped", "complete")})
    L = ["# Matchup mismatches and usage trends: can they tell us who will beat his normal?", "",
         "Generated by `python -m matchup_model.opp.matchup_study`. Everything below is out-of-sample (leave-one-season-out) with player-clustered bootstrap "
         "intervals unless it says otherwise; nothing here changes the shipped projections.", ""]
    if VERDICT.exists():
        L += [VERDICT.read_text(encoding="utf-8").strip(), ""]
    L += ["## How to read it", "",
         "'Beats his normal' means: actual > the conditional MEDIAN of that stat given his projected level (the shipped model's own walk-forward projection), "
         "the median learned on the other seasons only. That makes every label ~50/50 at every level, so projection level, skew and ceilings cannot help "
         "and any AUC above 0.5 is information the level does not contain. Feature sets are nested: "
         + "; ".join(f"**{k}** {v}" for k, v in SET_LABELS.items()) + ". B0 is exactly what the label was centred on, so its AUC should sit at 0.5: that is the "
         "check on the labels. A feature set earns its place only if the AUC increment's interval clears 0. B2 and B3 add ~100 defense features, so they can lose to "
         "noise even if a few carry a little signal; B4 is the fair version of that test: only the short list fixed in advance "
         f"(WR/TE: {', '.join(WRTE_CORE)}; RB: {', '.join(RB_CORE)}). B5 and B6 are the same two tests for the route-type features (WR/TE only; short list: "
         f"{', '.join(ROUTE_CORE)}).", "",
         "## Which Data Suite exports can be trusted", "",
         "The weekly player exports are only usable when they list a player whatever he did that week. The check compares each export with nflverse's "
         "players-on-the-field: an unfiltered export lists most quiet games; one filtered on the week's outcome drops them.", ""]
    L += _md(pres[["table", "season", "n_low", "low-usage presence", "n_high", "high-usage presence", "ratio", "verdict"]]) + [""]
    L += [f"Dropped from every player-level feature: {rec['dropped'] or 'nothing'} (WR/TE), {rush['dropped'] or 'nothing'} (RB). 2025 defense-level tables "
          "(coverage, box) and the 2025 RB tables are complete and used.", "",
          f"Modelling rows (>= {MIN_PRIOR} prior games on both sides, week >= {FIRST_WEEK}): "
          + ", ".join(f"{k} {v}" for k, v in n_rows.items()) + ".", ""]

    L += ["## WR/TE: does the matchup, or his own profile, predict beating his normal?", "",
          "Out-of-fold AUC (0.5 = no information) by feature set:", ""] + _auc_md(wr) + ["", "AUC increments with player-clustered 95% intervals:", ""] + _steps_md(wr) + [
          "", "### The most favourable test of a mismatch: the extremes", "",
          "For every defense and mismatch feature (alignment mix x what the defense allows from each alignment, man/zone/shell edges x how often they play them, "
          "personnel, press, ...), the beats-his-normal rate of the top decile minus the bottom decile (ranked within position).", "",
          _tail_summary(wr), "", "Largest gaps:", ""] + _tails_md(wr) + [
          "", "The mismatch features themselves (alignment, man/zone, shell, personnel, depth), gap in points (BH q):", ""] + _mismatch_md(wr) + [""]
    L += ["### His own usage trend, on its own", "", "A compact model on route participation, snap share, target share (last game, last three, trailing, trend), "
          "alignment mix and efficiency:", ""] + _md(usage_wr) + ["",
          "Does it improve the raw stat? Leave-one-season-out gradient boosting of (actual - baseline) on his profile; RMSE vs the baseline:", ""] + _md(gain) + [""]

    L += ["## RB: the same study", "", "Out-of-fold AUC by feature set:", ""] + _auc_md(rb) + ["", "AUC increments:", ""] + _steps_md(rb) + [
          "", _tail_summary(rb), "", "Largest gaps:", ""] + _tails_md(rb) + ["", "The mismatch features themselves, gap in points (BH q):", ""] + _mismatch_md(rb) + [
          "", "His own usage trend, compact model:", ""] + _md(usage_rb) + [""]

    L += ["## RB: the pre-specified run-defense and zone-rate hypotheses", "",
          "Fixed before looking at outcomes by feature: **run-defense weakness index** = mean of z(yards per carry the defense allows vs league), z(success rate "
          "allowed), -z(stuff rate), all trailing and as-of, standardised on 2022-25 (1 = one SD); **zone rate** = how much more zone than the league the defense has "
          "been playing. Slopes are per +1 SD with player-clustered 95% intervals:", ""]
    sl = slopes.assign(**{"slope [95% interval]": [_ci(*v, 2) for v in zip(slopes["per +1 SD"], slopes.lo, slopes.hi)]})
    L += _md(sl[["test", "scope", "n", "slope [95% interval]", "unit"]]) + [""]

    L += ["## Is the signal already in the book's line?", "",
          "All effects are per +1 SD of the signal, in SDs of the stat (rush yards, receptions, ...). **History** is what 2022-25 says the signal is worth: the result "
          "minus OUR projection, out of fold. On real lines that splits in two: (result minus our projection) = (how far the book moved its line from our projection with "
          "the signal) + (result minus the line). The last term is what is LEFT for a bettor: 0 if the line already contains the signal, equal to the history if it contains "
          "none of it. Line sets: the 2026 lines as logged (weeks 1-5, then the forward weeks) and a full season of 2025 DraftKings / FanDuel closing lines (5 minutes before "
          "kickoff, weeks 5-18, bought from the Odds API's historical snapshots), and the two pooled. The logistic columns ask the same as a yes/no beyond the book's own "
          "no-vig probability and our edge. "
          f"Weeks before {FORWARD_FROM} were seen while the hypotheses were formed; week {FORWARD_FROM} onward is the forward test. A +1 SD signal has to move a coin-flip "
          f"bet's win probability by about {ME.VIG_POINTS:g} points to pay the vig.", ""]
    L += _market_md(market) + [""]
    L += ["### Do the closing lines lean, and does our projection beat them? (2025 closing lines, DraftKings and FanDuel)", "",
          "Not about any signal: just the lines. **Lean**: the share of lines that went over vs what the no-vig price said, and what a blind UNDER / OVER bet earned at the "
          "posted price (unit stakes, pushes excluded, player-clustered 95% intervals). **Projection**: logistic 'went over' on the book's no-vig probability and our raw "
          "projection's edge in SDs (z); a z coefficient of 0 means our number adds nothing to the close, and about +0.1 per SD would move a coin-flip bet by the 2.4 points "
          "the vig costs.", ""] + _lean_md(lean) + [""] + _edge_md(edge) + [""]
    L += ["### Is it priced by Wednesday, or during the week? (2026 opening lines)", "",
          "The usage scores use only games already played, so they are the same on Wednesday as on Sunday. If the market learned them during the week the opening line would "
          "hold less of the signal than the late line, and betting early would capture the difference ('added during the week' above 0, 'left at the OPEN' above 'left at the "
          "late line'). Per +1 SD of signal, in SDs of the stat, player-clustered 95% intervals. The Wednesday 4pm ET snapshots exist for receiving markets only (weeks 1-5), so "
          "the RB rush yards signal cannot be tested this way. One look, exploratory.", ""] + _timing_md(timing) + [""]
    L += ["### What execution alone is worth: where and on which side the same leans are placed (2025 closing lines, DraftKings and FanDuel)", "",
          "Not a signal and not a forecast: our model's lean is held fixed and only the placement changes. A is what the Prop Edges screen did until 2026-10-15 (the book with the best "
          "PRICE on the lean, at that book's own number); B is what it does now (`dfs.props.place_bet`: the best NUMBER, best price on a tie; lowest for an OVER, highest for an UNDER). "
          "The UNDER-only rows drop the OVER leans, which is the screen's new default for rec yds, receptions and rush yds. In-sample for the idea (the 2025 lines are where "
          "the OVER/UNDER split was found); the 2026 graded leans, an independent season, show the same split.", ""] + _execution_md(execution) + [""]
    L += ["### Does FantasyPoints' own projection know more than our number? (2026 lines as logged)", "",
          "FantasyPoints' weekly export is the one proprietary input that is not a function of games already played. The app already nudges our number toward it (up to half "
          "the gap once they differ by more than 12%), and stores the remaining gap as `role_ratio` = FantasyPoints / our nudged number. The signal is log(role_ratio), clipped "
          f"at +/-{FP_GAP_CAP}, centred within the week and divided by {FP_GAP_SD} (the SD on weeks 2-5, fixed). 'Beyond the price and our edge' is the logistic coefficient "
          "of 'went over' on the book's no-vig probability, our edge z and the signal; a positive 'result minus OUR projection' means FantasyPoints is closer to the truth than "
          "our nudged number; 'what is left' is what is still there after the book's own line. Week 1 had no FantasyPoints number when its lines were pulled. The forward row "
          "is the registered test (H5 below); the per-market rows are descriptive.", ""] + _fp_md(fp) + [""]
    L += ["### Do DraftKings and FanDuel disagree, and does shopping the number pay? (2025 closing lines)", "",
          "The two books post different numbers on most yardage props. For every prop listed at both: how often the numbers differ; what a single bet earns at the lower or the "
          "higher of the two numbers (each at its own posted price, unit stakes, pushes refunded, player-clustered 95% intervals); the 'middle' (OVER at the lower number and "
          "UNDER at the higher, one unit each: both win when the result lands between); and same-number arbitrage.", ""] + _cross_md(cross) + [""]
    L += ["### H1 at the posted price: following the RB usage-trend score (2025 closing lines)", "",
          "OVER when the score is at least k SDs above his norm, UNDER when at least k below, at the book's closing price, one unit per bet. The thresholds are shown "
          "together, not chosen: bigger k means fewer, stronger-signal bets. `break-even` is the win rate those prices need. In-sample for the rule's existence "
          "(the 2025 lines are where it was found); the forward weeks are the test.", ""] + _follow_md(follow) + [""]
    L += ["### Is each signal stable across seasons, or fading?", "",
          "The 'history' column above, season by season (result minus OUR projection per SD of signal, out of fold):", ""] + _history_md(history) + [""]
    L += REGISTRY + CAVEATS
    return "\n".join(L) + "\n"


def datasets(refresh: bool = False) -> dict:
    """The Data Suite tables with the outcome-filtered seasons dropped, and the modelling frames built from them."""
    rec, rush = load_receiving(), load_rushing()
    print("WR/TE seasons dropped as outcome-filtered:", rec["dropped"] or "none", "| RB:", rush["dropped"] or "none", flush=True)
    wr_seasons = tuple(s for s in HISTORY if s not in rec["dropped"])
    rb_seasons = tuple(s for s in HISTORY if s not in rush["dropped"])
    return {"rec": rec, "rush": rush, "wr_seasons": wr_seasons, "rb_seasons": rb_seasons,
            "wr_df": receiving_dataset(rec, wr_seasons, refresh), "rb_df": rushing_dataset(rush, rb_seasons, refresh)}


def market_section(ds: dict) -> dict:
    """Everything that needs real lines: each signal against the 2026 and 2025 lines, the season-by-season history, whether the 2025 closing lines lean, and
    whether our projection adds anything to them."""
    sigs = build_signals(ds["wr_df"], ds["rb_df"], fit_run_index(ds["rb_df"]), ds["rec"]["C"])
    d26 = lines_2026()
    f26 = frames_2026(d26, ds["rec"], ds["rush"], sigs) if not d26.empty else {}
    L = HL.load(2025)
    market, history = market_tables(sigs, f26, L) if (f26 or len(L)) else (pd.DataFrame(), pd.DataFrame())
    lean, edge = lean_and_edge(L, ds["wr_df"], ds["rb_df"], REPS["logit"]) if len(L) else (pd.DataFrame(), pd.DataFrame())
    h1 = next(sg for sg in sigs if sg.label == H1_LABEL)
    results25 = RC.universe_rows(("WR", "TE", "RB"), seasons=[2025], weeks=range(5, 19)) if len(L) else None
    cross = cross_book(book_pairs(L, results25), REPS["logit"]) if len(L) else None
    execution = execution_table(execution_frame(L, results25, ds["wr_df"], ds["rb_df"]), REPS["logit"]) if len(L) else None
    timing = timing_table(f26, load_openers(), REPS["slope"] // 2) if f26 else pd.DataFrame()
    fp = fp_gap_table(restate.graded_lines(), REPS["logit"])
    return {"market": market, "history": history, "lean": lean, "edge": edge, "follow": follow_table(L, h1) if len(L) else pd.DataFrame(), "cross": cross, "timing": timing, "fp": fp, "execution": execution}


def compute(refresh: bool = False) -> dict:
    """Run the whole study; returns what `build_report` renders (the heavy frames stay out so the result can be cached and re-rendered)."""
    ds = datasets(refresh)
    rec, rush, wr_df, rb_df, wr_seasons, rb_seasons = ds["rec"], ds["rush"], ds["wr_df"], ds["rb_df"], ds["wr_seasons"], ds["rb_seasons"]
    n_rows = {"WR/TE": f"{len(wr_df)} ({', '.join(f'{s} {n}' for s, n in wr_df.groupby('season').size().items())})",
              "RB": f"{len(rb_df)} ({', '.join(f'{s} {n}' for s, n in rb_df.groupby('season').size().items())})"}
    print(n_rows, flush=True)
    wr = nested_study(wr_df, WRTE_TARGETS, WRTE_BASE, wr_seasons, tails_by="position", core=WRTE_CORE, route_core=ROUTE_CORE)
    print("WR/TE nested study done", flush=True)
    rb = nested_study(rb_df, RB_TARGETS, RB_BASE, rb_seasons, core=RB_CORE)
    print("RB nested study done", flush=True)
    usage_wr = usage_study(wr["frames"], WRTE_COMPACT, USAGE_TARGETS["WR/TE"], wr_seasons)
    usage_rb = usage_study(rb["frames"], RB_COMPACT, USAGE_TARGETS["RB"], rb_seasons)
    stats = {"targets": ("targets", "b_tgt"), "receptions": ("receptions", "b_rec"), "rec yards": ("receiving_yards", "b_rec_yds")}
    gain = profile_gain(wr_df, feature_sets(wr_df, WRTE_BASE, WRTE_BASE)["B1"], stats, wr_seasons)
    slopes, _ = rb_slopes(rb_df, rec["C"], rb_seasons)
    print("RB slopes done", flush=True)
    slim = {k: {"presence": v["presence"], "dropped": v["dropped"]} for k, v in (("rec", rec), ("rush", rush))}
    return {"rec": slim["rec"], "rush": slim["rush"], "wr": {k: v for k, v in wr.items() if k != "frames"}, "rb": {k: v for k, v in rb.items() if k != "frames"},
            "gain": gain, "usage_wr": usage_wr, "usage_rb": usage_rb, "slopes": slopes, "n_rows": n_rows, **market_section(ds)}


def main(argv: list[str] | None = None) -> None:
    """Modes: (none) the whole study; --forward  refresh only the 2026-lines section after a new week is graded (the history comes from the last full run);
    --render  re-render the last run (e.g. after editing the verdict); --quick  a smoke test with crude intervals (writes no cache); --refresh  re-project the baselines."""
    argv = sys.argv[1:] if argv is None else argv
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if "--quick" in argv:
        REPS.update({k: max(20, v // 8) for k, v in REPS.items()})
    if "--render" in argv or "--forward" in argv:
        if not RESULTS.exists():
            raise SystemExit(f"no cached run at {RESULTS}: run the full study first")
        results = pd.read_pickle(RESULTS)
        if "--forward" in argv:
            results.update(market_section(datasets()))
            if "--quick" not in argv:
                pd.to_pickle(results, RESULTS)
    else:
        results = compute("--refresh" in argv)
        if "--quick" not in argv:
            pd.to_pickle(results, RESULTS)
    txt = build_report(**results)
    out = REPORT.with_name("matchup_study_report_quick.md") if "--quick" in argv else REPORT       # a smoke test must not overwrite the real report
    out.write_text(txt, encoding="utf-8")
    print(txt)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
