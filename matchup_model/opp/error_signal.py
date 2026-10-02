"""Is there a signal, in the granular FantasyPoints Data Suite data, for WHICH props
we're more/less accurate on -- not just which market or tier, but which specific
matchup conditions or player profiles?

Two angles, both walk-forward-safe (only ever uses data for weeks already graded):

Part A -- defense-side: for every graded WR/TE/QB prop, pull the OPPONENT defense's
coverage-scheme stats for that specific week (man%, zone%, blitz/press%, disguise%,
Cover 0-6 mix, nickel/dime%) from coverage-matrix_2026wkN_week.csv, and correlate
each stat against our projection error.

Part B -- player-side: for every player with enough graded props, pull their season-
to-date route profile (man-coverage target share, alignment mix, personnel-package
mix, aDOT, YPRR) from the coverage/personnel Data Suite files, and correlate each
against that PLAYER's average error across all his graded props.

Error is expressed in z-units (error / the market's fitted outcome SD, same sigma
model dfs.props uses for confidence tiers) so results pool cleanly across markets
with very different natural scales (a 20-yard rec-yds miss and a 1-catch
receptions miss are not the same size error).

Every candidate feature is reported, not just the ones that look interesting --
with n, r, and p-value -- plus an explicit multiple-comparison note, since testing
many features against a few hundred rows WILL throw up some noise by chance alone.

    python -m matchup_model.opp.error_signal 2026
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from dfs.names import normalize_name
from dfs.props import _sigma
from matchup_model.opp import data as D
from matchup_model.scheme import _read

MARKET_TO_STAT = {"rec yds": "rec_yds", "receptions": "rec", "rush yds": "rush_yds",
                  "pass yds": "pass_yds", "pass TD": "pass_td"}
# which Data Suite position-file each market's player belongs in
MARKET_POS = {"rec yds": ("WR", "TE"), "receptions": ("WR", "TE"), "pass yds": ("QB",),
             "pass TD": ("QB",)}

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

MAN_COVS = {"Cover 0", "Cover 1", "Cover 2 Man", "Bracket"}
ZONE_COVS = {"Cover 2", "Cover 3", "Cover 4", "Cover 6"}


def _z_error(row) -> float:
    stat = MARKET_TO_STAT.get(row["market"])
    if stat is None:
        return np.nan
    sig = _sigma(stat, float(row["line"]))
    return (float(row["actual"]) - float(row["our proj"])) / sig if sig else np.nan


def _opponent_map(season: int, week: int) -> dict:
    """team abbrev -> opponent abbrev, for one week, from nflverse's games.csv."""
    g = D.games()
    wk = g[(g.season == season) & (g.week == week)]
    return dict(zip(wk["team"], wk["opp"]))


# ── Part A: defense-side, per-week matchup ──────────────────────────────────────

def build_defense_dataset(season: int, weeks: list[int]) -> pd.DataFrame:
    pw = D.player_weeks()
    pos_lookup = (pw.assign(nk=pw["player_display_name"].map(normalize_name))
                    .sort_values(["season", "week"]).groupby("nk")["position"].last())
    team_lookup = (pw.assign(nk=pw["player_display_name"].map(normalize_name)))

    from dfs.bets import load_line_history
    lh = load_line_history()
    lh = lh[(lh.season == season) & (lh.week.isin(weeks)) &
            (lh.result.isin(["win", "loss", "push"])) &
            (lh.market.isin(MARKET_TO_STAT))].copy()
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
        opp_map = _opponent_map(season, week)
        opp = opp_map.get(team)
        if not opp or opp not in ABBR_TO_FULL:
            continue
        if week not in cm_cache:
            cm_cache[week] = _read(f"coverage-matrix_2026wk{week}_week.csv")
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


# ── Part B: player-side, season-to-date profile ─────────────────────────────────

def _player_profile(name_key: str, pos_files: tuple[str, ...]) -> dict | None:
    for suffix in pos_files:
        rc = _read(f"receiving-coverage_{suffix}_2026.csv")
        if rc.empty:
            continue
        rows = rc[rc["Name"].map(normalize_name) == name_key]
        if rows.empty:
            continue
        man = rows[rows["COV"].isin(MAN_COVS)]["TGT"].sum()
        zone = rows[rows["COV"].isin(ZONE_COVS)]["TGT"].sum()
        tot = man + zone
        prof = {
            "man_tgt_share": 100 * man / tot if tot else np.nan,
            "wide_rte_pct": np.average(rows["WIDE RTE %"], weights=rows["RTE"].clip(lower=0.01)),
            "slot_rte_pct": np.average(rows["SLOT RTE %"], weights=rows["RTE"].clip(lower=0.01)),
            "inline_rte_pct": np.average(rows["INLINE RTE %"], weights=rows["RTE"].clip(lower=0.01)),
            "aDOT": np.average(rows["aDOT"], weights=rows["TGT"].clip(lower=0.01)),
            "YPRR": np.average(rows["YPRR"], weights=rows["RTE"].clip(lower=0.01)),
        }
        rp = _read(f"receiving-personnel_{suffix}_2026.csv")
        if not rp.empty:
            prows = rp[rp["Name"].map(normalize_name) == name_key]
            if not prows.empty:
                tgt11 = prows[prows["PERS"] == 11.0]["TGT"].sum()
                tgttot = prows["TGT"].sum()
                prof["pers11_tgt_share"] = 100 * tgt11 / tgttot if tgttot else np.nan
        return prof
    return None


def build_player_dataset(season: int, weeks: list[int]) -> pd.DataFrame:
    from dfs.bets import load_line_history
    lh = load_line_history()
    lh = lh[(lh.season == season) & (lh.week.isin(weeks)) &
            (lh.result.isin(["win", "loss", "push"])) &
            (lh.market.isin(["rec yds", "receptions"]))].copy()
    if lh.empty:
        return lh
    lh["z_error"] = lh.apply(_z_error, axis=1)
    lh = lh[np.isfinite(lh.z_error)]
    lh["nk"] = lh["player"].map(normalize_name)

    rows = []
    for nk, g in lh.groupby("nk"):
        if len(g) < 3:
            continue
        prof = _player_profile(nk, ("wr", "te"))
        if prof is None:
            continue
        rec = dict(player=g["player"].iloc[0], n=len(g),
                  mean_z_error=g["z_error"].mean(), mean_abs_z_error=g["z_error"].abs().mean())
        rec.update(prof)
        rows.append(rec)
    return pd.DataFrame(rows)


# ── correlation + report ─────────────────────────────────────────────────────

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


def report(season: int, weeks: list[int]) -> str:
    L = [f"# Error-signal audit — {season} weeks {weeks[0]}-{weeks[-1]}", ""]

    L.append("## Part A — defense faced that week (per-matchup, rec/receptions/pass props)")
    L.append("")
    dd = build_defense_dataset(season, weeks)
    if dd.empty:
        L.append("no rows")
    else:
        n_feat = len(DEF_FEATURES) * 2
        L.append(f"n={len(dd)} graded props with a confirmed opponent-defense row. "
                 f"{n_feat} feature x target tests run — at p<0.05 alone, expect "
                 f"~{n_feat*0.05:.1f} false positives by chance; treat anything "
                 f"above that bar, or without p<0.01, as a lead to re-check next "
                 f"week, not a finding.")
        L.append("")
        ct = correlate(dd, DEF_FEATURES, ["z_error", "abs_z_error"])
        L.append("| feature | vs | n | r | p |")
        L.append("|---|---|--:|--:|--:|")
        for _, r in ct.iterrows():
            L.append(f"| {r.feature} | {r.target} | {r.n} | {r.r} | {r.p} |")
        L.append("")

    L.append("## Part B — player's own season-to-date profile (rec/receptions props, "
             "n>=3 graded each)")
    L.append("")
    pd_ = build_player_dataset(season, weeks)
    if pd_.empty:
        L.append("no rows")
    else:
        feats = ["man_tgt_share", "wide_rte_pct", "slot_rte_pct", "inline_rte_pct",
                "aDOT", "YPRR", "pers11_tgt_share"]
        n_feat = len(feats) * 2
        L.append(f"n={len(pd_)} players. {n_feat} feature x target tests — same "
                 f"chance-noise caveat as Part A.")
        L.append("")
        ct = correlate(pd_, feats, ["mean_z_error", "mean_abs_z_error"])
        L.append("| feature | vs | n | r | p |")
        L.append("|---|---|--:|--:|--:|")
        for _, r in ct.iterrows():
            L.append(f"| {r.feature} | {r.target} | {r.n} | {r.r} | {r.p} |")
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
