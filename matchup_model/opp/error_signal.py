"""Is there a signal, in the granular FantasyPoints Data Suite data, for WHICH props
we're more/less accurate on -- not just which market or tier, but which specific
matchup conditions or player profiles?

Three angles, ALL walk-forward-safe -- every player-profile feature is built from
strictly-prior-week per-week exports only, never the season-cumulative files. That
distinction matters: an earlier version of Part B used the season-cumulative
receiving-coverage file and found a striking YPRR correlation (r=0.601, p<1e-6) that
looked like real signal -- it wasn't. Backtesting a shrinkage nudge built on it
(matchup_model.opp.model.opp_line(..., yprr_adj=True)) made projections WORSE,
because a season-cumulative stat partly explains a week's error WITH that week's own
data baked in. Every feature here is now rebuilt from prior-weeks-only aggregates so
that mistake can't repeat silently.

Part A -- defense faced that week: for every graded WR/TE/QB prop, the OPPONENT
defense's real per-week coverage-scheme stats (man%, zone%, blitz/press%, disguise%,
Cover 0-6 mix) from coverage-matrix_2026wkN_week.csv.

Part B -- player's own trailing receiving profile: YPRR, aDOT, and man-coverage
target share, built from strictly-prior weeks of receiving-advanced /
receiving-manvszone weekly exports.

Part C -- player's own trailing rushing-efficiency profile: success rate, stuff
rate, yards-before-contact/att, yards-after-contact/att, EPA/att, missed tackles
forced/att, from strictly-prior weeks of rushing-advanced weekly exports.

NOT included (would need a new Data Suite pull, not just new analysis code):
receiver ALIGNMENT (wide/slot/inline) and PLAY-CALLER / coordinator splits -- both
exist for 2022-2024 (the historical backtest pull) but nothing for 2026 is pulled on
a per-week basis, only alignment's season-cumulative file, which is exactly the kind
of leaky source this rewrite is trying to stop using.

Error is expressed in z-units (error / the market's fitted outcome SD, same sigma
model dfs.props uses for confidence tiers) so results pool cleanly across markets
with very different natural scales.

Every candidate feature is reported, not just the ones that look interesting -- with
n, r, and p-value -- plus an explicit multiple-comparison note.

    python -m matchup_model.opp.error_signal 2026
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from dfs.names import normalize_name
from dfs.props import _sigma
from matchup_model.config import DATA
from matchup_model.opp import data as D
from matchup_model.scheme import _read

MARKET_TO_STAT = {"rec yds": "rec_yds", "receptions": "rec", "rush yds": "rush_yds",
                  "pass yds": "pass_yds", "pass TD": "pass_td"}

FULL_TO_ABBR = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS"}
ABBR_TO_FULL = {v: k for k, v in FULL_TO_ABBR.items()}

DEF_FEATURES = ["MAN %", "ZONE %", "1-HI/MOF C %", "2-HI/MOF O %", "DISGUISE %",
                "TO 1-HI %", "TO 2-HI %", "COVER 0 %", "COVER 1 %", "COVER 2 %",
                "COVER 2 MAN %", "COVER 3 %", "COVER 4 %", "COVER 6 %", "BASE %",
                "NICKEL %", "DIME %", "PRESS % (ANY)", "PRESS % (TGT)"]


def _z_error(row) -> float:
    stat = MARKET_TO_STAT.get(row["market"])
    if stat is None:
        return np.nan
    sig = _sigma(stat, float(row["line"]))
    return (float(row["actual"]) - float(row["our proj"])) / sig if sig else np.nan


def _opponent_map(season: int, week: int) -> dict:
    g = D.games()
    wk = g[(g.season == season) & (g.week == week)]
    return dict(zip(wk["team"], wk["opp"]))


def _graded(season: int, weeks: list[int], markets: list[str]) -> pd.DataFrame:
    from dfs.bets import load_line_history
    lh = load_line_history()
    return lh[(lh.season == season) & (lh.week.isin(weeks)) &
             (lh.result.isin(["win", "loss", "push"])) & (lh.market.isin(markets))].copy()


# ── Part A: defense faced that week ──────────────────────────────────────────────

def build_defense_dataset(season: int, weeks: list[int]) -> pd.DataFrame:
    pw = D.player_weeks()
    pos_lookup = (pw.assign(nk=pw["player_display_name"].map(normalize_name))
                    .sort_values(["season", "week"]).groupby("nk")["position"].last())
    team_lookup = pw.assign(nk=pw["player_display_name"].map(normalize_name))

    lh = _graded(season, weeks, list(MARKET_TO_STAT))
    if lh.empty:
        return lh

    rows = []
    cm_cache: dict[int, pd.DataFrame] = {}
    for _, r in lh.iterrows():
        week = int(r["week"])
        nk = normalize_name(str(r["player"]))
        pos = pos_lookup.get(nk)
        if pos is None:
            continue
        trow = team_lookup[(team_lookup.nk == nk) & (team_lookup.season == season) &
                           (team_lookup.week < week)]
        if trow.empty:
            continue
        team = trow.sort_values("week").iloc[-1]["team"]
        opp = _opponent_map(season, week).get(team)
        if not opp or opp not in ABBR_TO_FULL:
            continue
        if week not in cm_cache:
            cm_cache[week] = _read(f"coverage-matrix_{season}wk{week}_week.csv")
        cm = cm_cache[week]
        if cm.empty:
            continue
        drow = cm[cm["Name"] == ABBR_TO_FULL[opp]]
        if drow.empty:
            continue
        z = _z_error(r)
        if not np.isfinite(z):
            continue
        rec = dict(week=week, player=r["player"], market=r["market"], pos=pos,
                  opp=opp, z_error=z, abs_z_error=abs(z))
        for feat in DEF_FEATURES:
            rec[feat] = drow.iloc[0].get(feat, np.nan)
        rows.append(rec)
    return pd.DataFrame(rows)


# ── shared: walk-forward trailing weekly aggregation ─────────────────────────────

def _trailing_weekly(stem: str, season: int, week: int, pos_filter: tuple[str, ...],
                     rename: dict[str, str] | None = None) -> pd.DataFrame:
    """Concats `{stem}_{season}wk{w}_week.csv` for every w STRICTLY BEFORE `week`,
    filtered to `pos_filter`, with a normalized-name `nk` column. Empty if no prior
    week's file exists yet (week 1, or the Data Suite pull hasn't run for week N-1)."""
    frames = []
    for w in range(1, week):
        p = DATA / f"{stem}_{season}wk{w}_week.csv"
        if not p.exists():
            continue
        d = _read(p.name)
        if not d.empty:
            frames.append(d)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df[df["POS"].isin(pos_filter)].copy()
    df["nk"] = df["Name"].map(normalize_name)
    if rename:
        df = df.rename(columns=rename)
    return df


def _wavg(df: pd.DataFrame, col: str, weight: pd.Series) -> float:
    v = pd.to_numeric(df[col], errors="coerce")
    w = pd.to_numeric(weight, errors="coerce").clip(lower=0.01)
    m = v.notna() & w.notna()
    return float(np.average(v[m], weights=w[m])) if m.any() else np.nan


# ── Part B: player's own trailing receiving profile ──────────────────────────────

REC_FEATURES = ["YPRR", "aDOT", "man_tgt_share"]


def build_receiving_profile_dataset(season: int, weeks: list[int]) -> pd.DataFrame:
    lh = _graded(season, weeks, ["rec yds", "receptions"])
    if lh.empty:
        return lh
    lh["z_error"] = lh.apply(_z_error, axis=1)
    lh = lh[np.isfinite(lh.z_error)].copy()
    lh["nk"] = lh["player"].map(normalize_name)
    lh["week"] = lh["week"].astype(int)

    rows = []
    for (nk, week), g in lh.groupby(["nk", "week"]):
        adv = _trailing_weekly("receiving-advanced", season, week, ("WR", "TE"))
        mvz = _trailing_weekly("receiving-manvszone", season, week, ("WR", "TE"))
        if adv.empty:
            continue
        mine_adv = adv[adv.nk == nk]
        if mine_adv.empty:
            continue
        rec = {"nk": nk, "week": week, "z_error": g["z_error"].mean(),
              "abs_z_error": g["z_error"].abs().mean()}
        rec["YPRR"] = _wavg(mine_adv, "YPRR", mine_adv["RTE"])
        rec["aDOT"] = _wavg(mine_adv, "aDOT", mine_adv["TGT"]) if "aDOT" in mine_adv.columns else np.nan
        if not mvz.empty and "RTE.1" in mvz.columns and "RTE.2" in mvz.columns:
            mine_mvz = mvz[mvz.nk == nk]
            if not mine_mvz.empty:
                man_rte = pd.to_numeric(mine_mvz["RTE.1"], errors="coerce").sum()
                zone_rte = pd.to_numeric(mine_mvz["RTE.2"], errors="coerce").sum()
                tot = man_rte + zone_rte
                rec["man_tgt_share"] = 100 * man_rte / tot if tot else np.nan
        rows.append(rec)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.groupby("nk", as_index=False).agg(
        n=("z_error", "size"), mean_z_error=("z_error", "mean"),
        mean_abs_z_error=("abs_z_error", "mean"),
        YPRR=("YPRR", "mean"), aDOT=("aDOT", "mean"), man_tgt_share=("man_tgt_share", "mean"))


# ── Part C: player's own trailing rushing-efficiency profile ─────────────────────

RUSH_FEATURES = ["SUCC %", "STUFF %", "YBC/ATT", "YACO/ATT", "EPA/A", "MTF/A", "HIT %"]


def build_rushing_profile_dataset(season: int, weeks: list[int]) -> pd.DataFrame:
    lh = _graded(season, weeks, ["rush yds"])
    if lh.empty:
        return lh
    lh["z_error"] = lh.apply(_z_error, axis=1)
    lh = lh[np.isfinite(lh.z_error)].copy()
    lh["nk"] = lh["player"].map(normalize_name)
    lh["week"] = lh["week"].astype(int)

    rows = []
    for (nk, week), g in lh.groupby(["nk", "week"]):
        adv = _trailing_weekly("rushing-advanced", season, week, ("RB",))
        if adv.empty:
            continue
        mine = adv[adv.nk == nk]
        if mine.empty:
            continue
        rec = {"nk": nk, "week": week, "z_error": g["z_error"].mean(),
              "abs_z_error": g["z_error"].abs().mean()}
        for feat in RUSH_FEATURES:
            if feat in mine.columns:
                rec[feat] = _wavg(mine, feat, mine["ATT"])
        rows.append(rec)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    agg = {"n": ("z_error", "size"), "mean_z_error": ("z_error", "mean"),
          "mean_abs_z_error": ("abs_z_error", "mean")}
    for feat in RUSH_FEATURES:
        if feat in df.columns:
            agg[feat] = (feat, "mean")
    return df.groupby("nk", as_index=False).agg(**agg)


# ── correlation + report ─────────────────────────────────────────────────────────

def correlate(df: pd.DataFrame, features: list[str], targets: list[str]) -> pd.DataFrame:
    rows = []
    for feat in features:
        if feat not in df.columns:
            continue
        for tgt in targets:
            sub = df[[feat, tgt]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(sub) < 10 or sub[feat].std() == 0:
                continue
            r, p = pearsonr(sub[feat], sub[tgt])
            rows.append({"feature": feat, "target": tgt, "n": len(sub),
                        "r": round(r, 3), "p": round(p, 4)})
    return pd.DataFrame(rows).sort_values("r", key=lambda s: s.abs(), ascending=False)


def _table_block(title: str, df: pd.DataFrame, note: str) -> list[str]:
    L = [f"## {title}", "", note, ""]
    if df.empty:
        L.append("no rows")
    else:
        L.append("| feature | vs | n | r | p |")
        L.append("|---|---|--:|--:|--:|")
        for _, r in df.iterrows():
            L.append(f"| {r.feature} | {r.target} | {r.n} | {r.r} | {r.p} |")
    L.append("")
    return L


def report(season: int, weeks: list[int]) -> str:
    L = [f"# Error-signal audit — {season} weeks {weeks[0]}-{weeks[-1]}",
        "(all player-profile features built walk-forward from strictly-prior-week "
        "exports only -- see module docstring)", ""]

    dd = build_defense_dataset(season, weeks)
    n_feat = len(DEF_FEATURES) * 2
    ct = correlate(dd, DEF_FEATURES, ["z_error", "abs_z_error"]) if not dd.empty else pd.DataFrame()
    L += _table_block("Part A — defense faced that week", ct,
                      f"n={len(dd)} props. {n_feat} tests — expect ~{n_feat*0.05:.1f} "
                      f"false positives at p<0.05 by chance; treat p<0.01 as a lead, "
                      f"not a finding.")

    rd = build_receiving_profile_dataset(season, weeks)
    ct = correlate(rd, REC_FEATURES, ["mean_z_error", "mean_abs_z_error"]) if not rd.empty else pd.DataFrame()
    L += _table_block("Part B — trailing receiving profile (YPRR, aDOT, man-coverage share)",
                      ct, f"n={len(rd)} player-weeks. {len(REC_FEATURES)*2} tests.")

    ru = build_rushing_profile_dataset(season, weeks)
    ct = correlate(ru, RUSH_FEATURES, ["mean_z_error", "mean_abs_z_error"]) if not ru.empty else pd.DataFrame()
    L += _table_block("Part C — trailing rushing-efficiency profile (success rate, "
                      "stuff rate, YBC/att, YACO/att, EPA/att, MTF/att, hit rate)",
                      ct, f"n={len(ru)} player-weeks. {len(RUSH_FEATURES)*2} tests.")

    L.append("## Not tested — needs a new Data Suite pull, not just new code")
    L.append("")
    L.append("- **Receiver alignment** (wide/slot/inline): only a season-cumulative "
             "2026 file exists (`receiving-alignment_defense_2026.csv`), no per-week "
             "pull -- same leaky shape that broke the YPRR shrinkage test, so it's "
             "left out rather than reported on a source already shown unreliable.")
    L.append("- **Play-callers / coordinators**: zero 2026 data pulled at all -- the "
             "5 coordinator-split tables (head coach, OC, DC, playcaller) only exist "
             "for the 2022-2024 historical backtest seasons.")
    return "\n".join(L)


def main():
    season = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    from matchup_model.weekly_pull import last_completed_week
    last = last_completed_week()
    weeks = list(range(1, last + 1)) if last else [1]
    txt = report(season, weeks)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(txt)
    with open(f"matchup_model/error_signal_{season}_report.md", "w", encoding="utf-8") as f:
        f.write(txt)


if __name__ == "__main__":
    main()
