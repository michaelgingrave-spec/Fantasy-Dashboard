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

from datetime import datetime
from functools import lru_cache
from zoneinfo import ZoneInfo

import numpy as np

# opp-model weight in the blend, per position (rest is the trailing-FP average).
# QB and RB are fit on the position's primary YARDAGE stat (pass_yds / rush_yds)
# directly, not on composite DK-fp -- fp mixes in TD variance, which distorted the
# weight without actually costing holdout accuracy for QB (stat-fit RMSE was 0.4 yards
# *better*) and left RB a toss-up either way (stat-fit is the more defensible choice in
# principle even though the two were statistically indistinguishable on this holdout).
# WR/TE showed no meaningful fp-vs-stat difference, so those stayed fp-fit.
# Fitted on 2023-24, judged on a 2025 holdout — see matchup_model/opp_backtest_report.md.
# Re-pin from that report's "fitted opp weight" lines whenever the model changes materially.
BLEND_W = {"QB": 0.50, "RB": 0.33, "WR": 0.90, "TE": 1.00}
INJURY_ADJ = True         # fold the weekly injury report into projected usage share

_HL, _LB = 4, 10          # trailing-FP EWMA — matches the backtest

# ── FantasyPoints projection nudge (v1, hand-set — NOT backtested the way BLEND_W was) ──
# matchup_model/opp/week_accuracy.py, week 1 2026 (n=267, the only real data point we
# have — FantasyPoints history isn't archived before this season): when our own blend
# and FantasyPoints' weekly export already agree (role_ratio 0.88-1.12), we're ~tied
# (RMSE 8.84 vs 8.89) — trailing-usage read is fine on its own. Once they disagree, they
# pull ahead (7.16 vs 8.13 RMSE at mid disagreement; 4.83 vs 5.54 at the widest) — exactly
# the stale-2025-role cases role_ratio already exists to flag. So: leave the blend alone
# when they agree, nudge toward FantasyPoints' number in proportion to the disagreement,
# capped well short of fully replacing our own read (FP was better even at the widest
# mismatches, not dominant). Revisit these three constants once a few more weeks of 2026
# data land — rerun week_accuracy.py and re-fit rather than trusting a single week.
FP_DEV_FLOOR = 0.12   # |role_ratio-1| below this: no nudge, trust our own blend fully
FP_DEV_SPAN = 0.40    # dev range the nudge ramps over, from 0 to FP_SHIFT_CAP
FP_SHIFT_CAP = 0.50   # never move more than half the distance to FantasyPoints' number


def current_season() -> int:
    """NFL season year for 'right now' (season spans Sep-Feb).

    Uses US/Eastern, not naive server-local time — Streamlit Cloud's container runs on
    UTC (4-5h ahead of Eastern), so late evening in the US can already be "tomorrow" on
    the server. That flip caused current_week() to jump a week early in practice
    (confirmed live 2026-09-16: UTC hit 2026-09-17 by ~10pm ET) — same fix applied here
    for the season-year boundary, even though it's a much rarer edge case (Feb/Mar)."""
    t = datetime.now(ZoneInfo("America/New_York")).date()
    return t.year if t.month >= 3 else t.year - 1


def current_week() -> int:
    """NFL week to project FOR right now — the week whose games haven't kicked off yet.
    Drives the DFS screens' 'NFL week' sidebar default so a fresh session doesn't
    silently sit on week 1 (2025-only trailing data, stale role reads) all season.
    One past matchup_model.weekly_pull.last_completed_week(), which owns the season's
    kickoff date (update that each year) — reused here rather than re-pinning a second
    copy of the same constant. Falls back to 1 if that's unavailable for any reason."""
    try:
        from matchup_model.weekly_pull import last_completed_week
        return max(1, min(18, last_completed_week() + 1))
    except Exception:  # noqa: BLE001 — a sidebar default is never worth crashing a screen
        return 1


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


@lru_cache(maxsize=4096)
def _naive_fp(name_key: str, pos: str, season: int, week: int) -> float:
    from matchup_model.opp import data as D
    d = D.player_weeks()
    h = d[(d.name_key == name_key) & (d.position == pos) &
          ((d.season < season) | ((d.season == season) & (d.week < week)))]
    if h.empty:
        return np.nan
    h = h.sort_values(["season", "week"]).tail(_LB)
    return _ewma(h["dk_fp"].to_numpy())


@lru_cache(maxsize=4)
def _fp_projections(week: int) -> dict:
    """FantasyPoints' own weekly projection for every player in that week's export,
    name_key -> FPTS. Empty dict if the file hasn't been dropped for this week yet —
    callers treat that as 'nothing to compare', not an error."""
    try:
        from dfs.names import normalize_name
        from dfs.projections import load_weekly_projections
        df = load_weekly_projections(week)
        return {normalize_name(n): float(p) for n, p in zip(df["name"], df["proj"])}
    except Exception:  # noqa: BLE001 — no file yet, bad file, whatever — just means no nudge
        return {}


def _fp_shift(base_fp: float, fp_proj: float | None) -> tuple[float, float | None]:
    """Nudge base_fp toward fp_proj in proportion to how much they disagree (see the
    FP_* constants above). Returns (adjusted_fp, role_ratio); role_ratio is None when
    there's no FantasyPoints number for this player to compare against."""
    if fp_proj is None or base_fp < 1.0:
        return base_fp, None
    role_ratio = fp_proj / base_fp
    dev = abs(role_ratio - 1.0)
    shift = min(max(dev - FP_DEV_FLOOR, 0.0) / FP_DEV_SPAN, FP_SHIFT_CAP)
    if shift <= 0:
        return base_fp, role_ratio
    return (1 - shift) * base_fp + shift * fp_proj, role_ratio


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
            base_fp, method, source = float(opp_fp), o.get("method", ""), "opp"
        else:
            w = BLEND_W[pos]
            base_fp = w * float(opp_fp) + (1.0 - w) * float(naive)
            method = f"opp x{w:.2f} + trailing avg x{1 - w:.2f}  ({o.get('method', '')})"
            source = "opp_blend"

        fp_proj = _fp_projections(week).get(name_key)
        adj_fp, fp_role_ratio = _fp_shift(base_fp, fp_proj)
        if fp_role_ratio is not None and abs(adj_fp - base_fp) > 1e-6:
            # nudged — scale every stat in the line by the same factor, not just the
            # total, so the individual prop numbers (rec_yds, rush_yds, ...) move too;
            # that's the whole point, this is exactly what feeds the props screen's
            # "our proj" column for the stale-2025-role cases role_ratio flags there.
            scale = adj_fp / base_fp if base_fp > 0 else 1.0
            line = {k: round(v * scale, 2) for k, v in line.items()}
            method += f"  ·  FP-nudged x{scale:.2f} (role_ratio {fp_role_ratio:.2f})"

        out = {"line": line, "fp": round(adj_fp, 1), "games": o.get("games"),
              "method": method, "source": source, "line_fp": round(line_fp, 1),
              "pre_fp_shift": round(base_fp, 1)}
        if source == "opp_blend":
            out["opp_fp"] = round(float(opp_fp), 1)
            out["trailing_fp"] = round(float(naive), 1)
        if fp_role_ratio is not None:
            out["fp_role_ratio"] = round(fp_role_ratio, 3)
        return out
    except Exception as e:  # noqa: BLE001 — never let the projection layer crash a screen
        fb = _fallback(name_key, pos, as_of_season, as_of_week)
        fb.setdefault("reason", f"blend error: {e}")
        return fb
