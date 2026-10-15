"""Team pass and rush attempts for a game, from trailing form and the game environment (runtime half; the fit lives in
team_volume_calibration.py and its coefficients in team_volume_coefs.json).

Why a fitted model. The earlier hand-set formula (trailing pace x pass rate, nudged by the game total and spread) scored WORSE
than a league-average constant on 2023-25 team games (pass attempts RMSE 8.0 vs 7.7, rush attempts 8.1 vs 7.4) because its
projections were far more spread out than the outcomes (actual moved 0.47 / 0.26 as much as projected), and its spread term had
the wrong sign: `games()` stores the betting spread (negative = favourite) but the formula treated positive as favoured, so
favourites were projected to pass more and run less when they do the opposite (2021-25 favourites of 7+ averaged 29.1 carries,
underdogs of 7+ 24.4; the formula had 23.6 and 26.7).

The model is a ridge regression per target on standardised features, every one known before kickoff:
  tr_att / tr_car / tr_plays   the team's own trailing attempts / carries / plays (EWMA, half-life 4 games, last 10, across seasons)
  op_df_att / op_df_car        attempts / carries the OPPONENT's defence has faced, trailing
  op_plays                     the opponent offence's trailing plays (pace)
  spread, total_line           betting spread (negative = favourite) and game total
  is_home, dome, wind          home flag, dome / closed roof, wind mph
Missing values (a line not posted yet, no wind reading) fall back to the training average. Coefficients are fit per cutoff season
(training = every season up to it) and a game is projected with the fit whose training seasons all precede it, so backtests stay
walk-forward and a 2026 game uses the fit through 2025.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from matchup_model.opp import data as D

COEFS_PATH = Path(__file__).with_name("team_volume_coefs.json")
HALFLIFE = 4.0
WINDOW = 10
MIN_PRIOR = 3                       # games of history a team needs before it is projected
RIDGE = 60.0                        # penalty on standardised coefficients (the fit is insensitive: 1-200 differ by < 0.2%)
CLIP = {"att": (18.0, 55.0), "car": (12.0, 45.0)}
FEATURES = {
    "att": ("tr_att", "tr_plays", "op_df_att", "op_plays", "spread", "total_line", "is_home", "dome", "wind"),
    "car": ("tr_car", "tr_plays", "op_df_car", "op_plays", "spread", "total_line", "is_home", "dome", "wind"),
}
TARGETS = {"att": "attempts", "car": "carries"}


def ewma(values, halflife: float = HALFLIFE, n: int = WINDOW) -> float:
    v = np.asarray(values, float)
    v = v[np.isfinite(v)][-n:]
    if v.size == 0:
        return np.nan
    w = 0.5 ** (np.arange(v.size)[::-1] / halflife)
    return float((w * v).sum() / w.sum())


@lru_cache(maxsize=1)
def _tables() -> tuple[dict, dict, dict]:
    """(own offence by team, offence faced by each defence, game-environment row by (season, week, team)). REG season only."""
    tw = D.team_weeks()
    tw = tw[tw["season_type"] == "REG"].sort_values(["season", "week"])
    own = {t: g.reset_index(drop=True) for t, g in tw.groupby("team")}
    faced = {t: g.reset_index(drop=True) for t, g in tw.groupby("opponent_team")}
    games = {(r.season, r.week, r.team): r._asdict() for r in D.games().itertuples(index=False)}
    return own, faced, games


def _before(frame: pd.DataFrame | None, season: int, week: int) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["attempts", "carries", "plays"])
    return frame[(frame.season < season) | ((frame.season == season) & (frame.week < week))]


@lru_cache(maxsize=8192)
def features(team: str, season: int, week: int) -> dict | None:
    """The model's inputs for `team` in (season, week), from games strictly before it; None if the schedule has no such game or
    the team has fewer than MIN_PRIOR games of history."""
    own, faced, games = _tables()
    g = games.get((season, week, team))
    if g is None:
        return None
    mine = _before(own.get(team), season, week)
    if len(mine) < MIN_PRIOR:
        return None
    opp = g["opp"]
    opp_off = _before(own.get(opp), season, week)
    opp_def = _before(faced.get(opp), season, week)
    roof = str(g.get("roof") or "")
    wind = g.get("wind")
    return {
        "tr_att": ewma(mine.attempts), "tr_car": ewma(mine.carries), "tr_plays": ewma(mine.plays),
        "op_df_att": ewma(opp_def.attempts), "op_df_car": ewma(opp_def.carries), "op_plays": ewma(opp_off.plays),
        "spread": float(g["spread"]) if pd.notna(g.get("spread")) else np.nan,
        "total_line": float(g["total_line"]) if pd.notna(g.get("total_line")) else np.nan,
        "is_home": float(g["is_home"]), "dome": 1.0 if roof in ("dome", "closed") else 0.0,
        "wind": float(wind) if pd.notna(wind) else 0.0,
        "n_prior": len(mine),
    }


@lru_cache(maxsize=1)
def _coefs() -> dict:
    if not COEFS_PATH.exists():
        return {}
    return json.loads(COEFS_PATH.read_text(encoding="utf-8")).get("fits", {})


def fit_for(season: int) -> dict | None:
    """The fit whose training seasons all precede `season` (the latest such); the earliest fit for older seasons."""
    fits = _coefs()
    if not fits:
        return None
    usable = [k for k in fits if int(k) < season]
    return fits[max(usable, key=int)] if usable else fits[min(fits, key=int)]


def apply_fit(model: dict, f: dict, target: str) -> float:
    """One target's prediction from a fit and a feature dict (missing values -> the training average)."""
    m = model[target]
    x = np.array([f.get(name, np.nan) for name in m["features"]], float)
    mu, sd, beta = np.array(m["mean"], float), np.array(m["sd"], float), np.array(m["beta"], float)
    z = np.where(np.isfinite(x), (x - mu) / sd, 0.0)
    lo, hi = CLIP[target]
    return float(np.clip(m["intercept"] + float(z @ beta), lo, hi))


def predict(team: str, season: int, week: int) -> tuple[float, float] | None:
    """(pass attempts, rush attempts) for the game, or None when the model can't run (no coefficients, no history, no such game)."""
    fit = fit_for(season)
    f = features(team, season, week)
    if fit is None or f is None:
        return None
    return apply_fit(fit, f, "att"), apply_fit(fit, f, "car")
