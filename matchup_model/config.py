"""Paths, season list, and tunable knobs for the coverage-matchup projection model.

Research module: reads the FantasyPoints Data Suite CSVs in data/dfs/matchup/, produces
per-player weekly projection adjustments, and (once backtested) feeds the DFS dashboard.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # Fantasy Model/
if str(ROOT) not in sys.path:                            # so `from dfs.names import ...` works
    sys.path.insert(0, str(ROOT))

DATA = ROOT / "data" / "dfs" / "matchup"                 # the 60+ FantasyPoints CSVs
OUT = ROOT / "data" / "dfs" / "matchup" / "_model"       # model outputs (git-ignored)
OUT.mkdir(parents=True, exist_ok=True)

SEASONS = [2022, 2023, 2024]
WEEK_BATCHES = {"wk1-6": range(1, 7), "wk7-12": range(7, 13), "wk13-18": range(13, 19)}

# Player-efficiency split-group labels as they appear in the CSV banner row.
SPLIT_GROUPS = ["Overall", "Man", "Zone", "Single-High", "Two-High"]

# ── Empirical-Bayes shrinkage pseudo-counts (tuned by backtest) ──────────────
# eff_split_shrunk = (n*eff_split + K*eff_overall) / (n + K)
SHRINK_K = {
    "routes": 130,      # WR/TE per-route stats
    "dropbacks": 160,   # QB per-dropback stats
    "carries": 45,      # RB per-carry stats
}
SEASON_DECAY = 0.72     # weight for prior seasons when aggregating player splits

# ── Defense scheme predictor ────────────────────────────────────────────────
EWMA_HALFLIFE_GAMES = 3
EWMA_LOOKBACK_GAMES = 8
MIN_GAMES_FOR_TREND = 2     # below this, lean on coordinator / league prior

# ── Matchup -> projection delta ────────────────────────────────────────────
# Global shrink on the coverage-blend gap (fit by backtest; <1 expected).
BETA = {"blend": 1.0}
MAX_DELTA_FRAC = 0.30      # clamp: never move a projection more than +/-30%
