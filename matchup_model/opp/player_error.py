"""Are we more right or wrong on certain TYPES of players -- or on specific players?

Two questions (plus a follow-up on the one type effect that stood out), on prop-level
projection error in z-units (actual - our projection, divided by the market's fitted
outcome SD, same units as error_signal.py so results pool across markets):

1. PLAYER-SPECIFIC: if we missed a player in one direction before, do we miss him the
   same way again? (persistence of error, plus a split-half check). If yes there is a
   player-level bias to correct; if no, "certain players" is just noise.
2. PLAYER-TYPE: error by position, role volume, volatility (trailing CV), history depth,
   rookie/new, game environment (Vegas team total) and how far our number sat from the
   line.
3. FOLLOW-UP (section 5): TEs stood out in (2), so their low-line OVER picks are compared with
   everyone else's and a TE-aware calibration is walk-forwarded against the current one.

Every feature is built only from games strictly before the prop's week. Significance is
by a player-clustered bootstrap (the same player across weeks/markets is not independent,
so naive p-values would overstate it). Every segment is reported, not just the
interesting ones.

    python -m matchup_model.opp.player_error 2026
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from dfs import conviction as C
from dfs.names import normalize_name
from matchup_model.opp import data as D
from dfs.props import _sigma
from matchup_model.opp.error_signal import MARKET_TO_STAT, _z_error

STAT_COL = {"rec yds": "receiving_yards", "receptions": "receptions", "rush yds": "rushing_yards",
            "pass yds": "passing_yards", "pass TD": "passing_tds"}
LOOKBACK = 10
B_BOOT = 500

# Section 5: the TE hypothesis. The pocket (low-line receiving OVERs) was picked AFTER seeing
# TEs stand out in the segment table, so it is reported as a hypothesis and walk-forwarded;
# weeks from FORWARD_FROM_WEEK on are the out-of-sample test that was fixed before they existed.
LOW_POCKET = {("receptions", "<=1.5"), ("rec yds", "<=20"), ("rec yds", "20-40")}
FORWARD_FROM_WEEK = 6


def build(season: int = 2026) -> pd.DataFrame:
    from dfs.bets import load_line_history
    lh = load_line_history()
    lh = lh[(lh["season"] == season) & lh["market"].isin(MARKET_TO_STAT) &
            lh["actual"].notna() & lh["our proj"].notna() &
            lh["result"].isin(["win", "loss", "push"])].copy()
    lh["z_err"] = lh.apply(_z_error, axis=1)
    lh = lh[np.isfinite(lh["z_err"])]

    pw = D.player_weeks()
    pw = pw.assign(nk=pw["player_display_name"].map(normalize_name)).sort_values(["season", "week"])
    by_player = {k: g for k, g in pw.groupby("nk")}
    first_season = pw.groupby("nk")["season"].min()
    team_total = {(r.season, r.week, r.team): r.team_total for r in D.games().itertuples()}

    rows = []
    for r in lh.to_dict("records"):
        nk = normalize_name(str(r["player"]))
        g = by_player.get(nk)
        if g is None:
            continue
        hist = g[(g["season"] < r["season"]) |
                 ((g["season"] == r["season"]) & (g["week"] < r["week"]))].tail(LOOKBACK)
        if hist.empty:
            continue
        cur = g[(g["season"] == r["season"]) & (g["week"] == r["week"])]
        team = cur.iloc[-1]["team"] if len(cur) else hist.iloc[-1]["team"]
        v = hist[STAT_COL[r["market"]]].to_numpy(float)
        mean = float(v.mean())
        cv = float(v.std(ddof=1) / mean) if len(v) >= 3 and mean > 0 else np.nan
        line = float(r["line"])
        rows.append(dict(
            nk=nk, player=r["player"], market=r["market"], week=int(r["week"]),
            position=hist.iloc[-1]["position"], z_err=float(r["z_err"]),
            conf=r["conf"], lean=r["lean"], result=r["result"], line=line, best=r.get("best"),
            n_prior=len(hist), n_this_season=int((hist["season"] == r["season"]).sum()),
            mean_stat=mean, cv=cv,
            new_player=bool(first_season.get(nk, 0) >= r["season"] - 1),
            team_total=team_total.get((r["season"], r["week"], team), np.nan),
            ext=abs(float(r["our proj"]) / line - 1) if line > 0 else np.nan,
            edge_z=(float(r["our proj"]) - line) / _sigma(MARKET_TO_STAT[r["market"]], line),
            real_z=(float(r["actual"]) - line) / _sigma(MARKET_TO_STAT[r["market"]], line)))
    df = pd.DataFrame(rows).sort_values(["nk", "market", "week"]).reset_index(drop=True)
    df["abs_z"] = df["z_err"].abs()
    grp = df.groupby(["nk", "market"])
    df["prior_z"] = grp["z_err"].transform(lambda s: s.shift().expanding().mean())
    df["prior_abs_z"] = grp["abs_z"].transform(lambda s: s.shift().expanding().mean())
    return df


def _terciles(df: pd.DataFrame, col: str) -> pd.Series:
    out = pd.Series("n/a", index=df.index, dtype=object)
    for _, g in df.groupby("market"):
        v = g[col].dropna()
        if len(v) < 6:
            continue
        q = pd.qcut(v.rank(method="first"), 3, labels=["low", "mid", "high"])
        out.loc[q.index] = q.astype(str)
    return out


def add_buckets(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["volume"] = _terciles(d, "mean_stat")
    d["volatility (CV)"] = _terciles(d, "cv")
    d["games of history"] = pd.cut(d["n_prior"], [0, 2, 5, 10], labels=["1-2", "3-5", "6-10"]).astype(str)
    d["rookie/2nd-yr"] = np.where(d["new_player"], "yes", "no")
    d["team total"] = pd.qcut(d["team_total"].rank(method="first"), 3,
                              labels=["low", "mid", "high"]).astype(str).where(d["team_total"].notna(), "n/a")
    d["proj vs line gap"] = pd.cut(d["ext"], [-0.001, .10, .30, .60, 99],
                                   labels=["<10%", "10-30%", "30-60%", "60%+"]).astype(str)
    return d


def _boot_mean(values, clusters, B=B_BOOT, seed=0):
    t = pd.DataFrame({"v": values, "c": clusters}).groupby("c")["v"].agg(["sum", "count"])
    s, n = t["sum"].to_numpy(float), t["count"].to_numpy(float)
    idx = np.random.default_rng(seed).integers(0, len(s), size=(B, len(s)))
    means = s[idx].sum(1) / n[idx].sum(1)
    return s.sum() / n.sum(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def _boot_slope_r(x, y, clusters, B=B_BOOT, seed=0):
    d = pd.DataFrame({"x": x, "y": y, "c": clusters}).dropna()
    d["xx"], d["xy"], d["yy"], d["n"] = d.x * d.x, d.x * d.y, d.y * d.y, 1.0
    t = d.groupby("c")[["x", "y", "xx", "xy", "yy", "n"]].sum().to_numpy(float)

    def stats(tot):
        sx, sy, sxx, sxy, syy, n = tot
        cov, vx, vy = sxy - sx * sy / n, sxx - sx * sx / n, syy - sy * sy / n
        return cov / vx, cov / np.sqrt(vx * vy)

    slope, r = stats(t.sum(0))
    idx = np.random.default_rng(seed).integers(0, len(t), size=(B, len(t)))
    boots = np.array([stats(t[i].sum(0)) for i in idx])
    return (slope, np.percentile(boots[:, 0], 2.5), np.percentile(boots[:, 0], 97.5),
            r, np.percentile(boots[:, 1], 2.5), np.percentile(boots[:, 1], 97.5), len(d))


def persistence(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sets = [("all markets", df)] + [(m, df[df["market"] == m]) for m in
                                     ("receptions", "rec yds", "rush yds")]
    for label, d in sets:
        d = d.dropna(subset=["prior_z"])
        if len(d) < 30:
            continue
        for what, x, y in [("signed error (bias)", d["prior_z"], d["z_err"]),
                           ("unsigned error (volatility)", d["prior_abs_z"], d["abs_z"])]:
            sl, sl_lo, sl_hi, r, r_lo, r_hi, n = _boot_slope_r(x, y, d["nk"])
            rows.append(dict(scope=label, measure=what, n=n, slope=sl, slope_ci=f"{sl_lo:+.2f} to {sl_hi:+.2f}",
                             r=r, r_ci=f"{r_lo:+.2f} to {r_hi:+.2f}"))
    return pd.DataFrame(rows)


def split_half(df: pd.DataFrame, cut: int = 3) -> tuple[float, float, float, int]:
    a = df[df["week"] <= cut].groupby("nk")["z_err"].agg(["mean", "count"])
    b = df[df["week"] > cut].groupby("nk")["z_err"].agg(["mean", "count"])
    j = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
    j = j[(j["count_a"] >= 2) & (j["count_b"] >= 2)]
    if len(j) < 15:
        return np.nan, np.nan, np.nan, len(j)
    ids = np.arange(len(j))
    sl, lo, hi, r, rlo, rhi, n = _boot_slope_r(j["mean_a"].to_numpy(), j["mean_b"].to_numpy(), ids)
    return r, rlo, rhi, n


def segments(df: pd.DataFrame) -> pd.DataFrame:
    d = add_buckets(df)
    rows = []
    for feat in ["position", "volume", "volatility (CV)", "games of history",
                 "rookie/2nd-yr", "team total", "proj vs line gap"]:
        for lvl, g in d.groupby(feat):
            if lvl == "n/a" or len(g) < 25:
                continue
            m, lo, hi = _boot_mean(g["z_err"].to_numpy(), g["nk"].to_numpy())
            picks = g[(g["conf"] != "—") & g["result"].isin(["win", "loss"])]
            rows.append(dict(feature=feat, level=lvl, n=len(g), players=g["nk"].nunique(),
                             bias=m, bias_ci=f"{lo:+.2f} to {hi:+.2f}",
                             sig="*" if (lo > 0 or hi < 0) else "",
                             rmse=float(np.sqrt((g["z_err"] ** 2).mean())),
                             pick_n=len(picks),
                             pick_win=100 * (picks["result"] == "win").mean() if len(picks) else np.nan))
    return pd.DataFrame(rows)


def edge_reality(df: pd.DataFrame) -> pd.DataFrame:
    """Regress (actual - line) on (our projection - line), both in z-units. The slope is
    the share of our disagreement with the market that turned out to be real: 1.0 means
    our number beats the line one-for-one, 0 means the line was right and we added noise."""
    scopes = [("all props", df), ("labeled picks only (not '—')", df[df["conf"] != "—"]),
              ("we lean OVER", df[df["edge_z"] > 0]), ("we lean UNDER", df[df["edge_z"] < 0])]
    scopes += [(f"position {p}", df[df["position"] == p]) for p in ("QB", "RB", "WR", "TE")]
    scopes += [(m, df[df["market"] == m]) for m in ("receptions", "rec yds", "rush yds")]
    rows = []
    for label, d in scopes:
        if len(d) < 40:
            continue
        sl, lo, hi, r, rlo, rhi, n = _boot_slope_r(d["edge_z"], d["real_z"], d["nk"])
        rows.append(dict(scope=label, n=n, real_share=sl, ci=f"{lo:+.2f} to {hi:+.2f}", r=r))
    return pd.DataFrame(rows)


def pick_crosstab(df: pd.DataFrame) -> pd.DataFrame:
    d = df[(df["conf"] != "—") & df["result"].isin(["win", "loss"])]
    rows = []
    for (pos, side), g in d.groupby(["position", "lean"]):
        if len(g) >= 20:
            rows.append(dict(position=pos, lean=side, n=len(g),
                             win=100 * (g["result"] == "win").mean()))
    return pd.DataFrame(rows)


def _boot_gap(a, a_clusters, b, b_clusters, B=B_BOOT, seed=0):
    """mean(a) - mean(b), bootstrapping players within each group (a player's props move together)."""
    def totals(v, c):
        return pd.DataFrame({"v": np.asarray(v, float), "c": np.asarray(c)}).groupby("c")["v"] \
            .agg(["sum", "count"]).to_numpy(float)

    ta, tb = totals(a, a_clusters), totals(b, b_clusters)
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, len(ta), size=(B, len(ta)))
    ib = rng.integers(0, len(tb), size=(B, len(tb)))
    gaps = ta[ia, 0].sum(1) / ta[ia, 1].sum(1) - tb[ib, 0].sum(1) / tb[ib, 1].sum(1)
    est = ta[:, 0].sum() / ta[:, 1].sum() - tb[:, 0].sum() / tb[:, 1].sum()
    return est, np.percentile(gaps, 2.5), np.percentile(gaps, 97.5)


def _profit(won: pd.Series, best: pd.Series) -> pd.Series:
    """Profit per $1 at the row's own best price; NaN where no price can be parsed."""
    odds = best.map(C.parse_price)
    win = odds.map(lambda o: C.win_profit(o) if pd.notna(o) else np.nan)
    return pd.Series(np.where(odds.isna(), np.nan, np.where(won == 1, win, -1.0)), index=won.index)


def receiving_overs(df: pd.DataFrame) -> pd.DataFrame:
    """Labeled, decided receptions / rec-yds OVER picks, flagged TE and low-line pocket."""
    d = df[(df["conf"] != "—") & (df["lean"] == "OVER") & df["market"].isin(["receptions", "rec yds"]) &
           df["result"].isin(["win", "loss"])]
    d = d[[C.pairs_cleanly(m, l) for m, l in zip(d["market"], d["line"])]].copy()
    d["won"] = (d["result"] == "win").astype(float)
    d["pocket"] = [(m, C.line_bucket(m, l)) in LOW_POCKET for m, l in zip(d["market"], d["line"])]
    d["is_te"] = d["position"].eq("TE")
    d["profit"] = _profit(d["won"], d["best"])
    return d


def te_pocket(df: pd.DataFrame) -> pd.DataFrame:
    """TE vs every other position on the same OVER picks, by line pocket and by period."""
    d = receiving_overs(df)
    slices = [("all receiving OVER picks", d),
              ("low-line pocket", d[d["pocket"]]),
              ("  weeks 1-3", d[d["pocket"] & (d["week"] <= 3)]),
              ("  weeks 4+", d[d["pocket"] & (d["week"] >= 4)]),
              ("  receptions only", d[d["pocket"] & (d["market"] == "receptions")]),
              ("  rec yds only", d[d["pocket"] & (d["market"] == "rec yds")]),
              (f"  weeks {FORWARD_FROM_WEEK}+ (forward test)", d[d["pocket"] & (d["week"] >= FORWARD_FROM_WEEK)]),
              ("mid/high lines (not pocket)", d[~d["pocket"]])]
    rows = []
    for label, g in slices:
        te, other = g[g["is_te"]], g[~g["is_te"]]
        if len(te) < 10 or len(other) < 10:
            continue
        gap, lo, hi = _boot_gap(te["won"], te["nk"], other["won"], other["nk"])
        rows.append(dict(slice=label, te_n=len(te), other_n=len(other),
                         te_win=100 * te["won"].mean(), other_win=100 * other["won"].mean(),
                         gap=100 * gap, ci=f"{100 * lo:+.1f} to {100 * hi:+.1f}",
                         te_roi=100 * te["profit"].mean(), other_roi=100 * other["profit"].mean()))
    return pd.DataFrame(rows)


def _fit_te(train: pd.DataFrame, k: float = C.K_SHRINK):
    """The current calibration plus a TE / non-TE delta per (market, side, bucket), shrunk toward
    that bucket's current delta (so a thin TE cell inherits the bucket, not its own streak)."""
    base = C.fit(train, k)
    d = C.prep(train)
    d = d[(d["conf"] != "—") & d["result"].isin(["win", "loss"])]
    miss = (d["result"] == "win").astype(float) - d["model_p"]
    bucket_delta = (base.table.set_index(["market", "lean", "bucket"])["delta"].to_dict()
                    if len(base.table) else {})
    te_delta = {}
    for key, x in miss.groupby([d["market"], d["lean"], d["bucket"], d["is_te"]]):
        b = bucket_delta.get(key[:3], base.global_delta)
        te_delta[key] = (len(x) * x.mean() + k * b) / (len(x) + k)
    return base, bucket_delta, te_delta


def te_walk_forward(season: int, df: pd.DataFrame) -> pd.DataFrame:
    """Decided, labeled props from every test week (3+), each scored by the current calibration
    and by the TE-aware one, both fit ONLY on strictly earlier weeks."""
    from dfs.bets import load_line_history
    lh = load_line_history()
    lh = lh[lh["season"] == season].copy()
    lh["nk"] = lh["player"].map(normalize_name)
    pos = df[["nk", "market", "week", "position"]].drop_duplicates(["nk", "market", "week"])
    lh = lh.merge(pos, on=["nk", "market", "week"], how="left")
    lh["is_te"] = lh["position"].eq("TE")          # no position (no prior games) counts as not-TE
    parts = []
    for w in sorted(int(x) for x in lh["week"].unique() if x >= 3):
        base, bucket_delta, te_delta = _fit_te(lh[lh["week"] < w])
        s = C.score(lh[lh["week"] == w], base)
        delta = [te_delta.get((m, side, b, bool(t)), bucket_delta.get((m, side, b), base.global_delta))
                 for m, side, b, t in zip(s["market"], s["lean"], s["bucket"], s["is_te"])]
        s["p_te"] = np.clip(s["model_p"].to_numpy() + np.array(delta, float), *C.P_CLIP)
        s["ev_te"] = s["p_te"] * s["odds"].map(C.win_profit) - (1 - s["p_te"])
        parts.append(s.assign(test_week=w))
    out = pd.concat(parts)
    out = out[out["result"].isin(["win", "loss"])].copy()
    out["won"] = (out["result"] == "win").astype(float)
    for col, p in (("b_model", "model_p"), ("b_base", "p_cal"), ("b_te", "p_te")):
        out[col] = (out[p] - out["won"]) ** 2
    return out


def te_walk_forward_tables(wf: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(Brier by slice, picks) from te_walk_forward. gain = base Brier - TE-aware Brier,
    so positive means the TE-aware calibration was better on weeks it had not seen."""
    recv = wf["market"].isin(["receptions", "rec yds"])
    pocket = (wf["lean"] == "OVER").to_numpy() & np.array(
        [(m, b) in LOW_POCKET for m, b in zip(wf["market"], wf["bucket"])], dtype=bool)
    slices = [("all labeled props", wf), ("receiving markets", wf[recv]), ("TE props", wf[wf["is_te"]]),
              ("non-TE receiving props", wf[recv & ~wf["is_te"]]), ("low-line receiving OVERs", wf[pocket])]
    rows = []
    for label, g in slices:
        if len(g) < 20:
            continue
        gain, lo, hi = _boot_mean((g["b_base"] - g["b_te"]).to_numpy(), g["nk"].to_numpy())
        rows.append(dict(slice=label, n=len(g), model=g["b_model"].mean(), base=g["b_base"].mean(),
                         te_aware=g["b_te"].mean(), gain=1e4 * gain, ci=f"{1e4 * lo:+.0f} to {1e4 * hi:+.0f}"))
    picks = []
    for label, sel in (("current calibration", (wf["p_cal"] >= C.P_MIN) & (wf["ev"] > 0)),
                       ("TE-aware calibration", (wf["p_te"] >= C.P_MIN) & (wf["ev_te"] > 0))):
        picks.append(dict(method=label, **C._grade(wf[sel])))
    return pd.DataFrame(rows), pd.DataFrame(picks)


def report(season: int) -> str:
    df = build(season)
    L = [f"# Player-error audit — {season}, {len(df)} graded props, {df['nk'].nunique()} players", "",
         "z-error = (actual - our projection) / market SD. Positive bias = we UNDER-project. "
         "CI = 95% player-clustered bootstrap; `*` = CI excludes zero.", ""]
    L.append("## 1. Do we miss the SAME players the same way?")
    L.append("")
    pr = persistence(df)
    L.append("| scope | measure | n | slope | 95% CI | r | 95% CI |")
    L.append("|---|---|--:|--:|---|--:|---|")
    for _, r in pr.iterrows():
        L.append(f"| {r.scope} | {r.measure} | {r.n} | {r.slope:+.2f} | {r.slope_ci} | {r.r:+.2f} | {r.r_ci} |")
    r, lo, hi, n = split_half(df)
    L.append("")
    L.append(f"Split-half (each player's mean error in weeks 1-3 vs weeks 4-5, players with 2+ props "
             f"in both): r = {r:+.2f} (95% CI {lo:+.2f} to {hi:+.2f}), n = {n} players.")
    L.append("")
    L.append("## 2. Error by player type")
    L.append("")
    sg = segments(df)
    L.append("| feature | level | n | players | bias | 95% CI | | RMSE | picks | pick win% |")
    L.append("|---|---|--:|--:|--:|---|---|--:|--:|--:|")
    for _, r in sg.iterrows():
        pw_ = "" if pd.isna(r.pick_win) else f"{r.pick_win:.0f}%"
        L.append(f"| {r.feature} | {r.level} | {r.n} | {r.players} | {r.bias:+.2f} | {r.bias_ci} | {r.sig} "
                 f"| {r.rmse:.2f} | {r.pick_n} | {pw_} |")
    L.append("")
    L.append("## 3. How much of our disagreement with the market is real?")
    L.append("")
    L.append("Slope of (actual - line) on (our projection - line), z-units. 1.0 = our number beats "
             "the line one-for-one; 0 = the line was right and our difference was noise.")
    L.append("")
    L.append("| scope | n | real share | 95% CI | r |")
    L.append("|---|--:|--:|---|--:|")
    for _, r in edge_reality(df).iterrows():
        L.append(f"| {r.scope} | {r.n} | {r.real_share:+.2f} | {r.ci} | {r.r:+.2f} |")
    L.append("")
    L.append("## 4. Pick win rate by position and side (labeled picks, n>=20)")
    L.append("")
    L.append("| position | lean | n | win% |")
    L.append("|---|---|--:|--:|")
    for _, r in pick_crosstab(df).iterrows():
        L.append(f"| {r.position} | {r.lean} | {r.n} | {r.win:.0f}% |")
    L.append("")
    L.append("## 5. Is the TE gap real? (a hypothesis found AFTER the scans above)")
    L.append("")
    L.append("Pocket = receptions at <=1.5 plus rec yds at <=40; labeled OVER picks only, whole-number "
             "consensus catch lines dropped (their price belongs to a different bet). The pocket was "
             "chosen after TEs stood out in section 2, so the in-sample gap in 5a overstates it. 5b is "
             "the fair test: each week is fit on earlier weeks only.")
    L.append("")
    L.append("**5a. TE vs every other position on the same kind of pick** "
             "(gap = TE win% minus the rest, 95% player-clustered CI; ROI at each pick's best price)")
    L.append("")
    L.append("| slice | TE n | other n | TE win% | other win% | gap (pts) | 95% CI | TE ROI | other ROI |")
    L.append("|---|--:|--:|--:|--:|--:|---|--:|--:|")
    for _, r in te_pocket(df).iterrows():
        L.append(f"| {r.slice} | {r.te_n} | {r.other_n} | {r.te_win:.1f}% | {r.other_win:.1f}% | "
                 f"{r.gap:+.1f} | {r.ci} | {r.te_roi:+.1f}% | {r.other_roi:+.1f}% |")
    L.append("")
    wf = te_walk_forward(season, df)
    brier, picks = te_walk_forward_tables(wf)
    L.append(f"**5b. Walk-forward** (test weeks {sorted(wf['test_week'].unique().tolist())}): does adding a "
             "TE / non-TE split to the calibration improve probabilities on weeks it never saw? "
             "Brier, lower is better; gain = current minus TE-aware (x1e4), positive favours TE-aware.")
    L.append("")
    L.append("| slice | n | model p | current | TE-aware | gain (x1e4) | 95% CI |")
    L.append("|---|--:|--:|--:|--:|--:|---|")
    for _, r in brier.iterrows():
        L.append(f"| {r.slice} | {r.n} | {r.model:.4f} | {r.base:.4f} | {r.te_aware:.4f} | "
                 f"{r.gain:+.0f} | {r.ci} |")
    L.append("")
    L.append("Every pick each calibration would make (calibrated p >= "
             f"{C.P_MIN:g} and +EV), test weeks pooled:")
    L.append("")
    L.append("| calibration | picks | win% | ROI |")
    L.append("|---|--:|--:|--:|")
    for _, r in picks.iterrows():
        L.append(f"| {r.method} | {int(r.n)} | {r.win:.1f}% | {r.roi:+.1f}% |")
    L.append("")
    L.append(f"**Forward test, fixed after week 5:** weeks {FORWARD_FROM_WEEK}+ are out of sample for this "
             f"hypothesis (the 'weeks {FORWARD_FROM_WEEK}+' row in 5a fills in as they are graded). A TE "
             "factor goes into `dfs/conviction.py` only if that row shows a gap whose CI excludes zero on "
             "100+ TE pocket picks, or the 5b gain turns positive with a CI excluding zero.")
    return "\n".join(L)


def main():
    season = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    txt = report(season)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(txt)
    with open(f"matchup_model/player_error_{season}_report.md", "w", encoding="utf-8") as f:
        f.write(txt)


if __name__ == "__main__":
    main()
