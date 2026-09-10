"""The projection the dashboard actually uses: a per-position blend of the opportunity
model and a naive trailing-fantasy-points average, with `project_stats` as the fallback
when the nflverse cache can't be built.

`blended_line()` is a drop-in for `matchup_model.project_stats.projected_line()` — same
return shape `{"line": {...}, "fp": float, "games": int, "method": str}` (plus `"source"`).

Blend weights are the opp-model weight fit on 2023-24 and judged on a 2025 holdout
(see matchup_model/opp_backtest_report.md): 2025 holdout RMSE 7.48 blend vs 7.65 naive
vs 7.70 project_stats.
"""
from __future__ import annotations

from datetime import date
from functools import lru_cache

import numpy as np

# opp-model weight in the blend, per position (rest is the trailing-FP average).
# Fitted on 2023-24, judged on a 2025 holdout — see matchup_model/opp_backtest_report.md.
# Re-pin from that report's "fitted opp weight" line whenever the model changes materially.
BLEND_W = {"QB": 0.64, "RB": 0.55, "WR": 0.90, "TE": 1.00}
INJURY_ADJ = True         # fold the weekly injury report into projected usage share

_HL, _LB = 4, 10          # trailing-FP EWMA — matches the backtest


def current_season() -> int:
    """NFL season year for 'right now' (season spans Sep-Feb)."""
    t = date.today()
    return t.year if t.month >= 3 else t.year - 1


@lru_cache(maxsize=1)
def _ready() -> bool:
    try:
        from matchup_model.opp import data as D
        return not D.player_weeks().empty
    except Exception:
        return False


def available() -> bool:
    return _ready()


def _ewma(v: np.ndarray) -> float:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)][-_LB:]
    if v.size == 0:
        return np.nan
    w = 0.5 ** (np.arange(v.size)[::-1] / _HL)
    return float((w * v).sum() / w.sum())


def _naive_fp(name_key: str, pos: str, season: int, week: int) -> float:
    from matchup_model.opp import data as D
    d = D.player_weeks()
    h = d[(d.name_key == name_key) & (d.position == pos) &
          ((d.season < season) | ((d.season == season) & (d.week < week)))]
    if h.empty:
        return np.nan
    h = h.sort_values(["season", "week"]).tail(_LB)
    return _ewma(h["dk_fp"].to_numpy())


def _fallback(name_key: str, pos: str, as_of_season, as_of_week) -> dict:
    from matchup_model import project_stats as _pjs
    out = _pjs.projected_line(name_key, pos, as_of_season, as_of_week)
    if out.get("fp") is not None:
        out = {**out, "source": "project_stats"}
    return out


def blended_line(name_key: str, pos: str, as_of_season: int | None = None,
                 as_of_week: int | None = None) -> dict:
    """Blended projected stat line + DK points. Falls back to `project_stats` when the
    opportunity model has no line or the nflverse cache is unavailable."""
    pos = (pos or "").upper()
    season = int(as_of_season) if as_of_season else current_season()
    week = int(as_of_week) if as_of_week else 1

    if pos not in BLEND_W or not _ready():
        return _fallback(name_key, pos, as_of_season, as_of_week)

    from dfs.names import normalize_name
    name_key = normalize_name(name_key)          # idempotent — callers usually pre-normalize

    try:
        from matchup_model.opp import model as M
        from matchup_model.opp.data import dk_points_from_line

        o = M.opp_line(name_key, pos, season, week, by="name", injury_adj=INJURY_ADJ)
        opp_fp = o.get("dk_fp")
        if opp_fp is None or not np.isfinite(opp_fp) or opp_fp <= 0.5:
            return _fallback(name_key, pos, as_of_season, as_of_week)

        line = {k: round(v, 2) for k, v in o["line"].items()}
        line_fp = dk_points_from_line(line)     # the opp line's own DK total
        naive = _naive_fp(name_key, pos, season, week)
        if not np.isfinite(naive):
            return {"line": line, "fp": round(float(opp_fp), 1), "games": o.get("games"),
                    "method": o.get("method", ""), "source": "opp", "line_fp": round(line_fp, 1)}

        w = BLEND_W[pos]
        blend_fp = w * float(opp_fp) + (1.0 - w) * float(naive)
        # the stat line stays the opportunity model's (its yardage is the trustworthy part);
        # `fp` is the blended total, which the 2025 holdout says is the better point estimate.
        return {"line": line, "fp": round(blend_fp, 1), "games": o.get("games"),
                "method": f"opp x{w:.2f} + trailing avg x{1 - w:.2f}  ({o.get('method', '')})",
                "source": "opp_blend", "opp_fp": round(float(opp_fp), 1),
                "trailing_fp": round(float(naive), 1), "line_fp": round(line_fp, 1)}
    except Exception as e:  # noqa: BLE001 — never let the projection layer crash a screen
        fb = _fallback(name_key, pos, as_of_season, as_of_week)
        fb.setdefault("reason", f"blend error: {e}")
        return fb
