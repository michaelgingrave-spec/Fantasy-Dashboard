"""Descriptive projected stat line for the upcoming week.

Trailing usage x trailing efficiency -> a reconstructed box-score line and its DK
fantasy points. This is NOT a market-beating model (the coverage-matchup edge did not
validate, see backtest_report.md) — it's a "what does this player's recent role imply"
sanity check to sit next to the FantasyPoints projection in the Player Lookup panel.

    from matchup_model.project_stats import projected_line
    projected_line("jamarr chase", "WR")
    # -> {"line": {"tgt": 9.1, "rec": 6.4, "rec_yds": 82.0, "rec_td": 0.55}, "fp": 17.6, ...}
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from matchup_model.ingest import receiving_weekly, rushing_weekly, passing_weekly

LOOKBACK = 10          # trailing games to consider
HALFLIFE = 4           # EWMA half-life (games) — recent weeks weigh more
MIN_GAMES = 3


def _ewma(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    w = 0.5 ** (np.arange(x.size)[::-1] / HALFLIFE)   # oldest ... newest
    return float(np.sum(w * x) / np.sum(w))


def _hist(d: pd.DataFrame, name_key: str, as_of_season, as_of_week) -> pd.DataFrame:
    h = d[d["name_key"] == name_key].copy()
    if as_of_season is not None:
        h = h[(h["season"] < as_of_season) |
              ((h["season"] == as_of_season) & (h["week"] < (as_of_week or 99)))]
    return h.sort_values(["season", "week"]).tail(LOOKBACK)


def _dk_points(line: dict) -> float:
    p = 0.0
    p += line.get("pass_yds", 0) * 0.04 + line.get("pass_td", 0) * 4 - line.get("int", 0)
    p += 3.0 if line.get("pass_yds", 0) >= 300 else 0.0
    p += line.get("rush_yds", 0) * 0.1 + line.get("rush_td", 0) * 6
    p += 3.0 if line.get("rush_yds", 0) >= 100 else 0.0
    p += line.get("rec", 0) * 1.0 + line.get("rec_yds", 0) * 0.1 + line.get("rec_td", 0) * 6
    p += 3.0 if line.get("rec_yds", 0) >= 100 else 0.0
    return round(p, 1)


def _rec_line(name_key, as_of_season, as_of_week) -> dict | None:
    h = _hist(receiving_weekly(), name_key, as_of_season, as_of_week)
    if len(h) < MIN_GAMES:
        return None
    tgt = _ewma(h["tgt"].to_numpy())
    catch = _ewma((h["rec"] / h["tgt"].replace(0, np.nan)).to_numpy())
    ypt = _ewma((h["rec_yds"] / h["tgt"].replace(0, np.nan)).to_numpy())
    tdpt = _ewma((h["rec_td"] / h["tgt"].replace(0, np.nan)).to_numpy())
    catch = min(max(catch, 0.3), 0.95) if np.isfinite(catch) else 0.65
    rec = tgt * catch
    line = {"tgt": round(tgt, 1), "rec": round(rec, 1),
            "rec_yds": round(tgt * ypt, 0) if np.isfinite(ypt) else 0.0,
            "rec_td": round(tgt * tdpt, 2) if np.isfinite(tdpt) else 0.0}
    return {"line": line, "fp": _dk_points(line), "games": int(len(h)),
            "method": f"trailing {len(h)} g: {tgt:.1f} tgt x {catch:.0%} catch x {ypt:.1f} yds/tgt"}


def _rush_line(name_key, as_of_season, as_of_week) -> dict | None:
    h = _hist(rushing_weekly(), name_key, as_of_season, as_of_week)
    if len(h) < MIN_GAMES:
        return None
    att = _ewma(h["att"].to_numpy())
    ypc = _ewma(h["ypc"].to_numpy())
    tdpc = _ewma((h["rush_td"] / h["att"].replace(0, np.nan)).to_numpy())
    rec = _ewma(h["rush_rec"].to_numpy()) if "rush_rec" in h else float("nan")
    recyd = _ewma(h["rush_rec_yds"].to_numpy()) if "rush_rec_yds" in h else float("nan")
    line = {"rush_att": round(att, 1),
            "rush_yds": round(att * ypc, 0) if np.isfinite(ypc) else 0.0,
            "rush_td": round(att * tdpc, 2) if np.isfinite(tdpc) else 0.0,
            "rec": round(rec, 1) if np.isfinite(rec) else 0.0,
            "rec_yds": round(recyd, 0) if np.isfinite(recyd) else 0.0}
    line["rec_td"] = round(0.03 * line["rec"], 2)      # ~3% of RB catches score
    return {"line": line, "fp": _dk_points(line), "games": int(len(h)),
            "method": f"trailing {len(h)} g: {att:.1f} att x {ypc:.1f} ypc + {line['rec']:.1f} rec"}


def _pass_line(name_key, as_of_season, as_of_week) -> dict | None:
    h = _hist(passing_weekly(), name_key, as_of_season, as_of_week)
    if len(h) < MIN_GAMES:
        return None
    att = _ewma(h["att"].to_numpy())
    ypa = _ewma(h["ypa"].to_numpy())
    tdpa = _ewma((h["pass_td"] / h["att"].replace(0, np.nan)).to_numpy())
    intpa = _ewma((h["int"] / h["att"].replace(0, np.nan)).to_numpy())
    rush_fp = _ewma(h["rush_fp"].to_numpy()) if "rush_fp" in h else 0.0
    line = {"pass_att": round(att, 1),
            "pass_yds": round(att * ypa, 0) if np.isfinite(ypa) else 0.0,
            "pass_td": round(att * tdpa, 2) if np.isfinite(tdpa) else 0.0,
            "int": round(att * intpa, 2) if np.isfinite(intpa) else 0.0}
    fp = _dk_points(line) + (round(rush_fp, 1) if np.isfinite(rush_fp) else 0.0)
    return {"line": line, "fp": round(fp, 1), "games": int(len(h)),
            "method": f"trailing {len(h)} g: {att:.1f} att x {ypa:.1f} ypa, +{rush_fp:.1f} rush fp"}


_DISPATCH = {"WR": _rec_line, "TE": _rec_line, "RB": _rush_line, "QB": _pass_line}


def projected_line(name_key: str, pos: str, as_of_season=None, as_of_week=None) -> dict:
    """Reconstructed stat line + DK points from trailing usage/efficiency.
    Returns {"line": {...}, "fp": float, "games": int, "method": str} or
    {"fp": None, "reason": ...} when there isn't enough game-log history."""
    fn = _DISPATCH.get((pos or "").upper())
    if fn is None:
        return {"fp": None, "reason": f"no stat-line model for {pos}"}
    try:
        out = fn(name_key, as_of_season, as_of_week)
    except Exception as e:                       # pragma: no cover - defensive
        return {"fp": None, "reason": f"error: {e}"}
    if out is None:
        return {"fp": None, "reason": f"need {MIN_GAMES}+ recent games in data/dfs/matchup/"}
    return out
