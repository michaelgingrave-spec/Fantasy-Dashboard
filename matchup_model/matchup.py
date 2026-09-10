"""Project a player's efficiency as a coverage-weighted blend of their (shrunk, stable)
raw per-split efficiency, then convert the gap vs their coverage-neutral efficiency into a
fantasy-point adjustment on top of a base projection.

    face   = predicted coverage rates the player will see (from defense_model)
    eff_c  = player's shrunk efficiency vs coverage c (from player_splits)
    proj   = Σ_c face_c · eff_c        (man/zone view and 1-hi/2-hi view, averaged)
    Δstat  = opportunity · (proj − eff_overall) · β
    Δfp    = Δyards·0.1  +  Δtargets·0.63   (WR/TE)     |     Δpass_yards·0.04  (QB)
"""
from __future__ import annotations

import numpy as np

from matchup_model.config import BETA, MAX_DELTA_FRAC
from matchup_model import defense_model, player_splits
from matchup_model.base_projection import base_row

_CATCH_PT = 0.63          # PPR points per extra target (≈ catch rate)
_REC_YDS_PT = 0.1
_PASS_YDS_PT = 0.04


def _norm(a: float, b: float) -> tuple[float, float]:
    s = a + b
    return (a / s, b / s) if s else (0.5, 0.5)


def _blend(eff: dict, face_mz: tuple, face_mof: tuple) -> float:
    """Average of the man/zone blend and the 1-hi/2-hi blend."""
    wm, wz = face_mz
    w1, w2 = face_mof
    mz = wm * eff.get("man", eff.get("overall", np.nan)) + wz * eff.get("zone", eff.get("overall", np.nan))
    mof = w1 * eff.get("single_high", eff.get("overall", np.nan)) + w2 * eff.get("two_high", eff.get("overall", np.nan))
    parts = [p for p in (mz, mof) if np.isfinite(p)]
    return float(np.mean(parts)) if parts else np.nan


def player_week_delta(name_key: str, pos: str, def_team: str, season: int, week: int,
                      through_season: int | None = None) -> dict:
    pos = (pos or "").upper()
    grp = player_splits.POS_TO_GROUP.get(pos)
    base = base_row(name_key, pos, season, week)
    opp_unit = base.get("opp_unit", np.nan)
    out = {"delta_fp": 0.0, "delta_yds": 0.0, "delta_tgt": 0.0, "why": "",
           "base_fp": base.get("base_fp"), "opp_unit": opp_unit}
    if grp is None or not np.isfinite(opp_unit) or opp_unit <= 0:
        return out

    p = defense_model.predict(def_team, season, week)
    face_mz = _norm(p["man"] / 100.0, p["zone"] / 100.0)
    face_mof = _norm(p["one_high"] / 100.0, p["two_high"] / 100.0)

    S = player_splits.build_player_splits(through_season)
    Sp = S[S.name_key == name_key]
    if Sp.empty:
        return out

    def gap(stat: str) -> tuple[float, float]:
        eff = dict(zip(Sp[Sp.stat == stat]["split"], Sp[Sp.stat == stat]["eff"]))
        if "overall" not in eff:
            return 0.0, np.nan
        proj = _blend(eff, face_mz, face_mof)
        if not np.isfinite(proj):
            return 0.0, eff["overall"]
        return (proj - eff["overall"]) * BETA.get("blend", 1.0), eff["overall"]

    if grp == "REC":
        g_tprr, _ = gap("tprr")
        g_yprr, _ = gap("yprr")
        d_tgt = opp_unit * g_tprr
        d_yds = opp_unit * g_yprr
        d_fp = d_yds * _REC_YDS_PT + d_tgt * _CATCH_PT
        drv, dstat = ("targets/route", g_tprr) if abs(g_tprr) * 30 > abs(g_yprr) * 2 else ("yds/route", g_yprr)
    else:  # QB
        g_ypa, _ = gap("ypa")
        d_tgt = 0.0
        d_yds = opp_unit * g_ypa
        d_fp = d_yds * _PASS_YDS_PT
        drv, dstat = "yds/dropback", g_ypa

    b = base.get("base_fp")
    if np.isfinite(b) and b > 0:
        cap = MAX_DELTA_FRAC * b
        d_fp = float(np.clip(d_fp, -cap, cap))

    lean = p["man"] - defense_model.league_rate(season, "man")
    tilt = "man-heavy" if lean > 4 else ("zone-heavy" if lean < -4 else "balanced")
    out.update(delta_fp=round(float(d_fp), 2), delta_yds=round(float(d_yds), 1),
               delta_tgt=round(float(d_tgt), 2),
               why=(f"{def_team} projects {tilt} ({p['man']:.0f}% man); "
                    f"player's coverage-weighted {drv} {dstat:+.3f} vs their baseline "
                    f"-> {d_fp:+.1f} pts"))
    return out


def model_projection(name_key: str, pos: str, def_team: str, season: int, week: int,
                     fp_base: float | None = None, through_season: int | None = None) -> dict:
    d = player_week_delta(name_key, pos, def_team, season, week, through_season)
    base = fp_base if (fp_base is not None and np.isfinite(fp_base)) else d["base_fp"]
    model = (base + d["delta_fp"]) if np.isfinite(base) else np.nan
    return {"base_fp": round(base, 2) if np.isfinite(base) else None,
            "delta_fp": d["delta_fp"], "model_fp": round(model, 2) if np.isfinite(model) else None,
            "delta_yds": d["delta_yds"], "delta_tgt": d["delta_tgt"], "why": d["why"]}
