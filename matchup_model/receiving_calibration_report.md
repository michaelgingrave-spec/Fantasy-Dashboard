# WR/TE receiving recalibration (2023-25; every number scored with constants learned on the other two seasons)

9598 WR/TE player-weeks (weeks 5-18; 1341 are players on the field with no stat line, true zeros). Current model: K_SHARE_REC 1, K_YPT 60 (original 3 / 28) on the fitted team-volume model (team_volume.py; the original used a hand-set formula). Constants are learned on the 5292 rows projected for >= 3 targets ("involved"): the players that get prop lines and DFS salaries. `td` is the multiplier on expected receiving TDs, learned the same way; the TD rate runs on the unscaled targets (see the downstream check: scaling TDs with the volume scale leaves involved players ~10% short on TDs).

## Out-of-season RMSE

| held-out season | players | stat | original | current model (scaled) | + defense nudge | original mean error | current mean error |
|---|---|---|--:|--:|--:|--:|--:|
| 2023 | involved (1694) | rec | 2.312 | 2.141 | 2.141 | +0.762 | -0.121 |
| 2023 | involved (1694) | rec_yds | 34.908 | 33.104 | 33.116 | +8.817 | -1.687 |
| 2023 | involved (1694) | tgt | 3.119 | 2.818 | 2.819 | +1.178 | -0.184 |
| 2024 | involved (1809) | rec | 2.301 | 2.206 | 2.206 | +0.573 | -0.143 |
| 2024 | involved (1809) | rec_yds | 33.065 | 31.517 | 31.520 | +7.322 | -1.000 |
| 2024 | involved (1809) | tgt | 3.070 | 2.891 | 2.890 | +0.911 | -0.169 |
| 2025 | involved (1789) | rec | 2.259 | 2.054 | 2.054 | +0.817 | +0.262 |
| 2025 | involved (1789) | rec_yds | 32.349 | 30.314 | 30.310 | +9.182 | +2.637 |
| 2025 | involved (1789) | tgt | 3.089 | 2.726 | 2.724 | +1.227 | +0.349 |
| 2023 | everyone (3201) | rec | 1.981 | 1.741 | 1.741 | +0.905 | +0.096 |
| 2023 | everyone (3201) | rec_yds | 28.535 | 26.111 | 26.115 | +10.516 | +1.191 |
| 2023 | everyone (3201) | tgt | 2.727 | 2.314 | 2.314 | +1.380 | +0.131 |
| 2024 | everyone (3161) | rec | 2.005 | 1.829 | 1.829 | +0.782 | +0.095 |
| 2024 | everyone (3161) | rec_yds | 27.870 | 25.767 | 25.771 | +9.409 | +1.735 |
| 2024 | everyone (3161) | tgt | 2.725 | 2.413 | 2.412 | +1.212 | +0.172 |
| 2025 | everyone (3236) | rec | 1.949 | 1.714 | 1.715 | +0.886 | +0.316 |
| 2025 | everyone (3236) | rec_yds | 27.061 | 24.718 | 24.716 | +10.095 | +3.558 |
| 2025 | everyone (3236) | tgt | 2.687 | 2.287 | 2.287 | +1.322 | +0.433 |

## Where the gain comes from (pooled over the three held-out seasons)

Each column adds one change to the one before it: a constant scale on the original model, then the new shrinkage, then the fitted team-volume model (replacing the hand-set formula whose spread term was backwards), then the defense nudge.

| players | stat | original | + scale | + new shrinkage | + fitted team volume | + defense nudge | total change |
|---|---|--:|--:|--:|--:|--:|--:|
| involved | tgt | 3.092 | 2.906 | 2.853 | 2.813 | 2.812 | -9.1% |
| involved | rec | 2.291 | 2.186 | 2.157 | 2.135 | 2.135 | -6.8% |
| involved | rec_yds | 33.430 | 32.274 | 31.900 | 31.639 | 31.642 | -5.3% |
| everyone | tgt | 2.713 | 2.484 | 2.360 | 2.338 | 2.338 | -13.8% |
| everyone | rec | 1.978 | 1.846 | 1.773 | 1.762 | 1.762 | -11.0% |
| everyone | rec_yds | 27.826 | 26.458 | 25.685 | 25.535 | 25.537 | -8.2% |

## Why one constant per position, and why it is learned on the involved players

Actual / projected by projected-usage tier under the new shrinkage, before any scale. Over-projection is steepest for depth players (and many of them play without a single target); from 4.5 projected targets up it is a fairly steady 0.82-0.89. In a one-off cross-season test when the constants were chosen (not re-run here), a usage-dependent scale beat the constant by 0.5-1.8% RMSE across everyone but by < 0.2% among players projected for 3+ targets, so the constant is kept.

| position | projected targets | players | with a stat row | actual / projected targets | receptions | rec yards |
|---|---|--:|--:|--:|--:|--:|
| WR | < 1.5 | 701 | 88% | 0.53 | 0.53 | 0.55 |
| WR | 1.5-3 | 1513 | 81% | 0.67 | 0.65 | 0.61 |
| WR | 3-4.5 | 1279 | 89% | 0.77 | 0.75 | 0.72 |
| WR | 4.5-6 | 924 | 97% | 0.89 | 0.88 | 0.89 |
| WR | 6-8 | 915 | 99% | 0.93 | 0.93 | 0.94 |
| WR | 8+ | 730 | 100% | 0.91 | 0.91 | 0.92 |
| TE | < 1.5 | 559 | 58% | 0.50 | 0.47 | 0.37 |
| TE | 1.5-3 | 1533 | 68% | 0.58 | 0.59 | 0.56 |
| TE | 3-4.5 | 603 | 90% | 0.84 | 0.89 | 0.88 |
| TE | 4.5-6 | 444 | 98% | 0.93 | 0.92 | 0.95 |
| TE | 6-8 | 314 | 100% | 0.95 | 0.97 | 0.98 |
| TE | 8+ | 83 | 100% | 0.94 | 0.94 | 0.91 |

## Downstream check: DK fantasy points and anytime-TD probability

The shipped model against the previous one, 9598 WR/TE player-weeks (2023-25 weeks 5-18). Nothing is learned here, so these are not out-of-season numbers for the scale -- they show what the change does to DK points and TD odds.

| players | pos | n | mean actual DK pts | mean projected, previous | recalibrated | RMSE trailing avg | previous | recalibrated |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| everyone who played | WR | 6062 | 6.73 | 8.66 | 6.77 | 6.361 | 6.690 | 6.325 |
| everyone who played | TE | 3536 | 4.60 | 6.52 | 5.27 | 4.885 | 5.259 | 4.867 |
| only those who ended up with 3+ targets/carries/attempts | WR | 3363 | 11.13 | 10.99 | 9.00 | 7.766 | 7.580 | 7.736 |
| only those who ended up with 3+ targets/carries/attempts | TE | 1350 | 10.02 | 9.09 | 7.91 | 6.745 | 6.391 | 6.586 |

The second population is the one `matchup_model.opp.backtest` and `td_calib` score (they keep only players who ended up with 3+ / 2+ touches). Selecting on the outcome inflates the actuals, so a model that over-projects looks best there; that is why those two reports cannot be used to compare the old and new models, and why they were not regenerated.

Blend weight on the opportunity model (the rest is the player's trailing DK average); fitted on 2023-24, scored on 2025, everyone who played:

| pos | production weight | best weight, previous model | best, recalibrated | 2025 RMSE: trailing avg | recalibrated at production weight | at its best weight |
|---|--:|--:|--:|--:|--:|--:|
| WR | 0.90 | 0.17 | 0.49 | 6.072 | 5.926 | 5.937 |
| TE | 1.00 | 0.05 | 0.40 | 5.097 | 4.980 | 5.006 |

The production weights were fit on the outcome-selected population with the old model; with the recalibrated model the blend is nearly flat between them and the best weight, so they are left alone (the blend only moves `fp` and `role_ratio`; the prop lines come straight from the opportunity model).

P(at least one TD) = 1 - exp(-expected receiving TDs), against whether the player scored (receiving or rushing):

| players | n | actual rate | mean P, previous | recalibrated | Brier, previous | recalibrated | flat average |
|---|--:|--:|--:|--:|--:|--:|--:|
| everyone who played | 9598 | 15.0% | 18.2% | 16.0% | 0.1163 | 0.1148 | 0.1277 |
| trailing 8-game average >= 3.5 targets | 4085 | 26.3% | 25.8% | 24.6% | 0.1867 | 0.1868 | 0.1940 |

## Constants for model.py (learned on all three seasons)

```python
K_SHARE_REC = 1.0
K_YPT = 60.0
# (learned under TEAM_VOLUME_MODEL = "fitted")
REC_SCALE = {
    "WR": {"tgt": 0.881, "rec": 0.873, "rec_yds": 0.882, "td": 0.970},
    "TE": {"tgt": 0.909, "rec": 0.925, "rec_yds": 0.935, "td": 1.060},
}
DEF_TGT_B = 0.09
```
