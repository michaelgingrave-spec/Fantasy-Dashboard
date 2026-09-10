"""Opportunity-based projected stat line (walk-forward safe).

`opp_line(name_key, pos, season, week)` projects a player's box score for that game from:

  team pace + pass rate  (trailing, blended prior season early)
    x  Vegas game environment  (spread -> pass-rate lean, total -> plays, team total -> TDs)
    x  the player's trailing usage *share*  (target share / carry share, EB-shrunk)
    x  lightly-shrunk trailing efficiency  (yds/target, yds/carry, yds/att, TD rates)

Everything reads only games strictly before (season, week). Returns
`{"line": {...}, "dk_fp": float, "vol": {...}, "method": str}` or `{"dk_fp": None, ...}`.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from matchup_model.opp import data as D

MIN_GAMES = 3
LOOKBACK = 10
HALFLIFE = 4

# hand-set coefficients (v1 — not fit; see backtest report for whether they earn their keep)
PR_SPREAD_BETA = 0.0045    # +pass-rate per point of pregame underdog
PR_CLAMP = 0.09            # max shift from trailing pass rate
PLAYS_TOTAL_BETA = 0.06    # +plays fraction per (total-44)/44
SACK_FRAC = 0.055          # dropbacks -> attempts
QB_SCRAMBLE = 1.5          # plays skimmed off team rush att for QB scrambles/kneels

# empirical-Bayes pseudo-counts (event units, not games)
K_SHARE = 3.0             # games, for target/carry share
K_YDS = 28.0             # targets or carries, for yds/opportunity
K_CATCH = 26.0
K_TD = 55.0
K_YPA = 90.0             # dropbacks, for QB yds/att
K_PASS_TD = 130.0
K_INT = 150.0

# team offensive TDs (by unit) per implied point -> distributed by the player's usage share
TD_FROM_TOTAL_PASS = 0.062   # ~1.5 team passing TDs at a 24-pt implied total
TD_FROM_TOTAL_RUSH = 0.037   # ~0.9 team rushing TDs
TD_VEGAS_BLEND = 0.5         # weight on the Vegas-implied TD vs the trailing-rate TD


def _ewma(v: np.ndarray) -> float:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)][-LOOKBACK:]
    if v.size == 0:
        return np.nan
    w = 0.5 ** (np.arange(v.size)[::-1] / HALFLIFE)
    return float(np.sum(w * v) / np.sum(w))


def _shrink(obs: float, n: float, prior: float, k: float) -> float:
    if not np.isfinite(obs):
        return prior
    return (n * obs + k * prior) / (n + k)


# ── league priors (computed from seasons strictly before the test season) ──────
@lru_cache(maxsize=8)
def _priors(before_season: int) -> dict:
    pw = D.player_weeks()
    pw = pw[pw.season < before_season]
    if pw.empty:                       # fall back to all data (first test season)
        pw = D.player_weeks()
    out = {}
    for pos in ("WR", "TE", "RB", "QB"):
        d = pw[(pw.position == pos) & (pw.dk_fp.notna())]
        # only rows with real involvement, so priors aren't dragged by cameos
        rec = d[d.targets >= 2]
        rush = d[d.carries >= 3]
        pas = d[d.attempts >= 10]
        out[pos] = {
            "ypt": _wmean(rec.receiving_yards, rec.targets),
            "catch": _wmean(rec.receptions, rec.targets),
            "rec_td": _wmean(rec.receiving_tds, rec.targets),
            "ypc": _wmean(rush.rushing_yards, rush.carries),
            "rush_td": _wmean(rush.rushing_tds, rush.carries),
            "ypa": _wmean(pas.passing_yards, pas.attempts),
            "pass_td": _wmean(pas.passing_tds, pas.attempts),
            "int": _wmean(pas.passing_interceptions, pas.attempts),
        }
    return out


def _wmean(num: pd.Series, den: pd.Series) -> float:
    n, d = float(np.nansum(num)), float(np.nansum(den))
    return n / d if d > 0 else np.nan


# ── team context ─────────────────────────────────────────────────────────────
@lru_cache(maxsize=4)
def _team_weeks_idx():
    tw = D.team_weeks()
    return tw


@lru_cache(maxsize=4096)
def team_context(team: str, season: int, week: int) -> dict:
    """Trailing plays/game and pass rate for `team` entering (season, week),
    blended toward last season's mean while the current sample is thin."""
    tw = _team_weeks_idx()
    prior = tw[(tw.team == team) & ((tw.season < season) |
                                   ((tw.season == season) & (tw.week < week)))]
    if prior.empty:
        return {"plays_pg": 63.0, "pass_rate": 0.57, "n": 0}
    cur = prior[prior.season == season].tail(8)
    prev = prior[prior.season == season - 1]
    w = min(1.0, len(cur) / 4.0) if len(cur) else 0.0

    def _blend(col):
        c = _ewma(cur[col].to_numpy()) if len(cur) else np.nan
        p = float(prev[col].mean()) if len(prev) else np.nan
        if not np.isfinite(c):
            return p if np.isfinite(p) else float(prior[col].tail(8).mean())
        if not np.isfinite(p):
            return c
        return w * c + (1 - w) * p

    return {"plays_pg": _blend("plays"), "pass_rate": _blend("pass_rate"),
            "n": int(len(cur))}


@lru_cache(maxsize=4)
def _games_idx():
    return D.games()


@lru_cache(maxsize=4096)
def vegas_row(team: str, season: int, week: int) -> dict:
    g = _games_idx()
    hit = g[(g.season == season) & (g.week == week) & (g.team == team)]
    if hit.empty:
        return {}
    r = hit.iloc[0]
    return {"spread": float(r["spread"]) if pd.notna(r["spread"]) else np.nan,
            "team_total": float(r["team_total"]) if pd.notna(r["team_total"]) else np.nan,
            "total_line": float(r["total_line"]) if pd.notna(r["total_line"]) else np.nan,
            "wind": float(r["wind"]) if pd.notna(r["wind"]) else np.nan,
            "roof": str(r["roof"]) if pd.notna(r["roof"]) else ""}


@lru_cache(maxsize=4096)
def team_volume(team: str, season: int, week: int) -> dict:
    """Projected team pass attempts / rush attempts / implied points for the game."""
    ctx = team_context(team, season, week)
    v = vegas_row(team, season, week)
    plays = ctx["plays_pg"]
    pr = ctx["pass_rate"]
    if not np.isfinite(plays):
        plays = 63.0
    if not np.isfinite(pr):
        pr = 0.57
    if v.get("total_line") and np.isfinite(v["total_line"]):
        plays *= 1.0 + PLAYS_TOTAL_BETA * (v["total_line"] - 44.0) / 44.0
    if v.get("spread") and np.isfinite(v["spread"]):
        pr = pr + PR_SPREAD_BETA * (-v["spread"])          # spread>0 = favored -> pass less
        pr = float(np.clip(pr, ctx["pass_rate"] - PR_CLAMP, ctx["pass_rate"] + PR_CLAMP))
    pr = float(np.clip(pr, 0.30, 0.72))
    dropbacks = plays * pr
    pass_att = dropbacks * (1.0 - SACK_FRAC)
    rush_att = max(12.0, plays * (1.0 - pr) - QB_SCRAMBLE)
    return {"pass_att": pass_att, "rush_att": rush_att,
            "team_total": v.get("team_total", np.nan), "plays": plays, "pass_rate": pr}


# ── per-player line ──────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _pw_with_team():
    pw = D.player_weeks()
    tw = D.team_weeks()[["season", "week", "team", "carries", "attempts", "sacks_suffered"]]
    tw = tw.rename(columns={"carries": "team_carries", "attempts": "team_att",
                            "sacks_suffered": "team_sacks"})
    m = pw.merge(tw, on=["season", "week", "team"], how="left")
    m["carry_share"] = (m["carries"] / m["team_carries"]).where(m["team_carries"] > 0)
    return m


def _hist(key: str, pos: str, season: int, week: int, by: str = "name") -> pd.DataFrame:
    d = _pw_with_team()
    col = "player_id" if by == "id" else "name_key"
    h = d[(d[col] == key) & (d.position == pos) &
          ((d.season < season) | ((d.season == season) & (d.week < week)))]
    return h.sort_values(["season", "week"]).tail(LOOKBACK)


def opp_line(key: str, pos: str, season: int, week: int, by: str = "name",
             injury_adj: bool = False) -> dict:
    pos = (pos or "").upper()
    if pos not in ("WR", "TE", "RB", "QB"):
        return {"dk_fp": None, "reason": f"pos {pos}"}
    h = _hist(key, pos, season, week, by=by)
    if len(h) < MIN_GAMES:
        return {"dk_fp": None, "reason": f"{len(h)} prior games"}
    pri = _priors(season)[pos]
    team = h.iloc[-1]["team"]
    gsis = h.iloc[-1]["player_id"]
    tv = team_volume(team, season, week)
    n_g = len(h)
    inj_rec = inj_rush = 1.0
    if injury_adj:
        from matchup_model.opp import injuries as _inj
        inj_rec = _inj.multiplier(gsis, team, pos, "rec", season, week)
        if pos == "RB":
            inj_rush = _inj.multiplier(gsis, team, pos, "rush", season, week)

    if pos in ("WR", "TE"):
        ts = _shrink(_ewma(h["target_share"].to_numpy()), n_g,
                     _pos_share_prior(pos), K_SHARE) * inj_rec
        tgt = max(0.0, ts) * tv["pass_att"]
        n_t = float(h["targets"].sum())
        ypt = _shrink(_ewma((h.receiving_yards / h.targets.replace(0, np.nan)).to_numpy()),
                      n_t, pri["ypt"], K_YDS)
        catch = _shrink(_ewma((h.receptions / h.targets.replace(0, np.nan)).to_numpy()),
                        n_t, pri["catch"], K_CATCH)
        catch = min(max(catch, 0.35), 0.95)
        td_rt = _shrink(_ewma((h.receiving_tds / h.targets.replace(0, np.nan)).to_numpy()),
                        n_t, pri["rec_td"], K_TD)
        # team passing TDs (from the implied total) x this player's target share
        vegas_td = TD_FROM_TOTAL_PASS * tv["team_total"] * max(0.0, ts) \
            if np.isfinite(tv["team_total"]) else np.nan
        rec_td = _td_blend(tgt * td_rt, vegas_td)
        line = {"tgt": tgt, "rec": tgt * catch, "rec_yds": tgt * ypt, "rec_td": rec_td}
        method = f"{ts:.1%} tgt share x {tv['pass_att']:.0f} att -> {tgt:.1f} tgt; {ypt:.1f} y/t"

    elif pos == "RB":
        cs = _shrink(_ewma(h["carry_share"].to_numpy()), n_g, 0.42, K_SHARE) * inj_rush
        car = max(0.0, cs) * tv["rush_att"]
        n_c = float(h["carries"].sum())
        ypc = _shrink(_ewma((h.rushing_yards / h.carries.replace(0, np.nan)).to_numpy()),
                      n_c, pri["ypc"], K_YDS)
        rtd = _shrink(_ewma((h.rushing_tds / h.carries.replace(0, np.nan)).to_numpy()),
                      n_c, pri["rush_td"], K_TD)
        # team rushing TDs (from the implied total) x this back's carry share (goal-line
        # backs skew higher, but carry share is the honest walk-forward proxy)
        vegas_rtd = TD_FROM_TOTAL_RUSH * tv["team_total"] * max(0.0, cs) \
            if np.isfinite(tv["team_total"]) else np.nan
        rush_td = _td_blend(car * rtd, vegas_rtd)
        ts = _shrink(_ewma(h["target_share"].to_numpy()), n_g, 0.09, K_SHARE) * inj_rec
        tgt = max(0.0, ts) * tv["pass_att"]
        n_t = float(h["targets"].sum())
        rypt = _shrink(_ewma((h.receiving_yards / h.targets.replace(0, np.nan)).to_numpy()),
                       n_t, pri["ypt"], K_YDS) if n_t else pri["ypt"]
        rcatch = 0.75
        line = {"rush_att": car, "rush_yds": car * ypc, "rush_td": rush_td,
                "rec": tgt * rcatch, "rec_yds": tgt * rcatch * rypt,
                "rec_td": 0.04 * tgt * rcatch}
        method = f"{cs:.1%} carry share x {tv['rush_att']:.0f} -> {car:.1f} car; {ypc:.1f} ypc"

    else:  # QB
        recent = h.tail(4)
        if float(recent["attempts"].mean()) < 12:
            return {"dk_fp": None, "reason": "not a recent starter"}
        # the QB's own trailing attempt level matters as much as the team projection
        own_att = _ewma(h["attempts"].to_numpy())
        att = 0.5 * tv["pass_att"] + 0.5 * (own_att if np.isfinite(own_att) else tv["pass_att"])
        n_a = float(h["attempts"].sum())
        ypa = _shrink(_ewma((h.passing_yards / h.attempts.replace(0, np.nan)).to_numpy()),
                      n_a, pri["ypa"], K_YPA)
        ptd = _shrink(_ewma((h.passing_tds / h.attempts.replace(0, np.nan)).to_numpy()),
                      n_a, pri["pass_td"], K_PASS_TD)
        irt = _shrink(_ewma((h.passing_interceptions / h.attempts.replace(0, np.nan)).to_numpy()),
                      n_a, pri["int"], K_INT)
        # rate-only TD (kneel-downs make Vegas-blend hurt elite passers), light nudge on total
        pass_td = att * ptd
        if np.isfinite(tv["team_total"]):
            pass_td *= float(np.clip(tv["team_total"] / 23.5, 0.85, 1.18))
        rush_yds = _ewma(h["rushing_yards"].to_numpy())
        rush_td = _ewma(h["rushing_tds"].to_numpy())
        line = {"pass_att": att, "pass_yds": att * ypa, "pass_td": pass_td,
                "int": att * irt, "rush_yds": rush_yds if np.isfinite(rush_yds) else 0.0,
                "rush_td": rush_td if np.isfinite(rush_td) else 0.0}
        method = f"{att:.0f} att x {ypa:.1f} ypa; +{rush_yds:.0f} rush yd"

    line = {k: float(round(v, 3)) for k, v in line.items() if np.isfinite(v)}
    if injury_adj and (abs(inj_rec - 1) > 1e-3 or abs(inj_rush - 1) > 1e-3):
        method += f"  ·  injury usage x{inj_rec:.2f}" + (
            f"/{inj_rush:.2f} rush" if pos == "RB" and abs(inj_rush - 1) > 1e-3 else "")
    return {"line": line, "dk_fp": D.dk_points_from_line(line), "games": n_g,
            "team": team, "vol": tv, "method": method,
            "inj_mult": round(inj_rec if pos != "RB" else min(inj_rec, inj_rush), 3)}


@lru_cache(maxsize=1)
def _share_priors():
    """Mean target share among rotation WR/TE (target_share>=0.08) — a shrink target that
    keeps low-usage guys from being pulled up to the roster-wide mean."""
    pw = D.player_weeks()
    out = {}
    for pos in ("WR", "TE"):
        d = pw[(pw.position == pos) & (pw.target_share >= 0.08)]
        out[pos] = float(d["target_share"].mean()) if len(d) else (0.16 if pos == "WR" else 0.13)
    return out


def _pos_share_prior(pos: str) -> float:
    return _share_priors().get(pos, 0.15)


def _td_blend(rate_td: float, vegas_td: float) -> float:
    if not np.isfinite(vegas_td):
        return max(0.0, rate_td)
    if not np.isfinite(rate_td):
        return max(0.0, vegas_td)
    return max(0.0, TD_VEGAS_BLEND * vegas_td + (1 - TD_VEGAS_BLEND) * rate_td)
