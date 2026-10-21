"""Injury-driven opportunity redistribution.

When a team's target-share or carry-share contributor is Out/Doubtful for the upcoming
game, that usage doesn't vanish — it flows to the healthy players in the same group. This
turns the weekly injury report into a multiplier on each remaining player's projected
usage share.

Walk-forward safe: week W's injury report is published Wed-Fri, before week W's game.

How much of the vacated usage teammates really absorb is not 100%: measured on 2022-26 (matchup_model/opp/injury_absorption.py, report
matchup_model/injury_absorption_report.md) the healthy teammates who play pick up a bit under half of a vacated TARGET share (the rest goes to depth players
outside the rotation) and about three quarters of a vacated CARRY share when the absence is new, and less when the player had already missed games (his absence is
partly in teammates' trailing shares). model.INJ_REDIST holds those fractions; 1.0 everywhere reproduces the older rule.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from matchup_model.opp import data as D

# P(plays) by report status. Questionable players historically suit up ~70% of the time;
# Doubtful ~a quarter. Blank status = treated as available.
PLAY_PROB = {"out": 0.0, "doubtful": 0.25, "questionable": 0.70, "": 1.0}

_TRAIL = 6            # team game-weeks that define the current rotation
_MIN_SHARE = 0.04     # ignore players below this trailing share when redistributing
_MULT_CAP = 2.2       # never more than ~double a share from injuries alone
_UNIT_POS = {"rec": ("WR", "TE", "RB"), "rush": ("RB",)}
_UNIT_COL = {"rec": "target_share", "rush": "carry_share"}


def play_prob(status: str) -> float:
    return PLAY_PROB.get((status or "").strip().lower(), 1.0)


@lru_cache(maxsize=1)
def _shares_base() -> pd.DataFrame:
    """Per player-game share of team targets / team carries, so we can trail them."""
    pw = D.player_weeks()[["season", "week", "player_id", "player_display_name",
                           "position", "team", "targets", "carries", "target_share"]].copy()
    tw = D.team_weeks()[["season", "week", "team", "carries"]].rename(
        columns={"carries": "team_carries"})
    m = pw.merge(tw, on=["season", "week", "team"], how="left")
    m["carry_share"] = (m["carries"] / m["team_carries"]).where(m["team_carries"] > 0).fillna(0.0)
    m["target_share"] = m["target_share"].fillna(0.0)
    return m.sort_values(["player_id", "season", "week"])


def _ewma(v: np.ndarray, hl: float = 3.0) -> float:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)][-_TRAIL:]
    if v.size == 0:
        return 0.0
    w = 0.5 ** (np.arange(v.size)[::-1] / hl)
    return float((w * v).sum() / w.sum())


@lru_cache(maxsize=8192)
def _group_components(team: str, unit: str, season: int, week: int) -> dict | None:
    """What the multipliers are made of, independent of the redistribution scales (so it can be cached): each rotation player's trailing share and P(plays),
    and the vacated share split into what is NEW news (the hurt player appeared in the team's last game) and what is ONGOING (he had already missed it, so
    teammates' trailing shares already reflect his absence). None when nobody in the rotation is hurt."""
    if unit not in _UNIT_COL:
        return None
    team = D.canon_team(team)
    col = _UNIT_COL[unit]
    base = _shares_base()
    prior = base[(base.team == team) & (base.position.isin(_UNIT_POS[unit])) &
                 ((base.season < season) | ((base.season == season) & (base.week < week)))]
    if prior.empty:
        return None

    # restrict to the team's most recent ~6 game-weeks -> departed players drop out
    gw = prior[["season", "week"]].drop_duplicates().sort_values(["season", "week"]).tail(_TRAIL)
    recent = prior.merge(gw, on=["season", "week"])
    if recent.empty:
        return None
    last = gw.iloc[-1]
    in_last = set(recent[(recent.season == last.season) & (recent.week == last.week)].player_id)

    sh: dict[str, float] = {}
    names: dict[str, str] = {}
    for pid, g in recent.groupby("player_id"):
        if len(g) < 2 and float(g[col].mean()) < 0.10:
            continue                                # one-week cameo, low usage — skip
        s = _ewma(g[col].to_numpy())
        if s >= _MIN_SHARE:
            sh[pid] = s
            names[pid] = g["player_display_name"].iloc[-1]
    if not sh:
        return None

    # availability from this week's report
    inj = D.injuries()
    rep = inj[(inj.season == season) & (inj.week == week) & (inj.team == team)]
    status = {r.gsis_id: r.report_status for r in rep.itertuples()
              if r.report_status in ("out", "doubtful", "questionable")}
    avail = {pid: play_prob(status.get(pid, "")) for pid in sh}
    if all(a >= 0.999 for a in avail.values()):
        return None                                # nobody hurt — no adjustment
    return {"sh": sh, "avail": avail, "names": names,
            "vac_new": sum(sh[p] * (1.0 - avail[p]) for p in sh if p in in_last),
            "vac_old": sum(sh[p] * (1.0 - avail[p]) for p in sh if p not in in_last),
            "avail_share": sum(sh[p] * avail[p] for p in sh)}


def group_multipliers(team: str, unit: str, season: int, week: int) -> dict:
    """{gsis_id: usage-share multiplier} for the team's `unit` group ('rec' | 'rush')
    entering (season, week). 1.0 = unchanged, 0.0 = out, >1 = absorbs vacated share.
    Only players in the team's last ~6 game-weeks count as the current rotation.

    The vacated share is redistributed to the healthy teammates in proportion to their own share, but only the part that is real: model.INJ_REDIST gives,
    per unit, the fraction of the proportional boost that teammates really absorbed when the absence was NEW news and when it was ONGOING (measured on
    2022-26: receiving 0.46 / 0.37, rushing 0.74 / 0.42; 1.0 everywhere is the v1-v3 rule that handed teammates the whole share)."""
    comp = _group_components(team, unit, season, week)
    if not comp:
        return {}
    from matchup_model.opp import model as _m
    k = _m.INJ_REDIST[unit]
    sh, avail, names, avail_share = comp["sh"], comp["avail"], comp["names"], comp["avail_share"]
    boost = k["new"] * comp["vac_new"] + k["ongoing"] * comp["vac_old"]
    out = {}
    for p in sh:
        if avail_share <= 1e-9:
            new = avail[p] * sh[p]
        else:
            new = avail[p] * sh[p] * (1.0 + boost / avail_share)
        mult = new / sh[p] if sh[p] > 1e-9 else 0.0
        out[p] = round(min(mult, _MULT_CAP), 4)
    out["_meta"] = {"vacated_share": round(comp["vac_new"] + comp["vac_old"], 3),
                    "hurt": {names[p]: round(1 - avail[p], 2) for p in sh if avail[p] < 0.999}}
    return out


def multiplier(gsis_id: str, team: str, pos: str, unit: str, season: int, week: int) -> float:
    m = group_multipliers(team, unit, season, week)
    return float(m.get(gsis_id, 1.0)) if gsis_id in m else 1.0


def team_report(team: str, season: int, week: int) -> pd.DataFrame:
    """Human-readable: who on `team` is dinged this week and the resulting usage multiplier."""
    team = D.canon_team(team)
    inj = D.injuries()
    rep = inj[(inj.season == season) & (inj.week == week) & (inj.team == team) &
              inj.report_status.isin(["out", "doubtful", "questionable"])]
    if rep.empty:
        return pd.DataFrame()
    rec = group_multipliers(team, "rec", season, week)
    rush = group_multipliers(team, "rush", season, week)
    rows = []
    for r in rep.itertuples():
        rows.append({"player": r.full_name, "pos": r.position,
                     "status": r.report_status.capitalize(),
                     "p(play)": play_prob(r.report_status),
                     "rec ×": rec.get(r.gsis_id), "rush ×": rush.get(r.gsis_id)})
    return pd.DataFrame(rows)
