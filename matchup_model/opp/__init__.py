"""Opportunity-based weekly projection (nflverse data).

A research prototype that projects a player's *volume* (targets / carries / dropbacks)
from team pace + pass rate + Vegas game environment + the player's trailing usage share,
then applies lightly-shrunk efficiency. Compare against the current trailing-usage-x-
efficiency line (`matchup_model.project_stats`) with:

    python -m matchup_model.opp.backtest
"""
