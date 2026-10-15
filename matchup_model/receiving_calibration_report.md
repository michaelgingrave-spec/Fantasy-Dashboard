# WR/TE receiving recalibration (2023-25; every number scored with constants learned on the other two seasons)

9598 WR/TE player-weeks (weeks 5-18; 1341 are players on the field with no stat line, true zeros). Current model: K_SHARE_REC 1, K_YPT 60 (original 3 / 28) on the fitted team-volume model (team_volume.py; the original used a hand-set formula). Constants are learned on the 5454 rows projected for >= 3 targets ("involved"): the players that get prop lines and DFS salaries. `td` is the multiplier on expected receiving TDs, learned the same way; the TD rate runs on the unscaled targets (see the downstream check: scaling TDs with the volume scale leaves involved players ~10% short on TDs).

## Out-of-season RMSE

| held-out season | players | stat | original | current model (scaled) | + defense nudge | original mean error | current mean error |
|---|---|---|--:|--:|--:|--:|--:|
| 2023 | involved (1755) | rec | 2.296 | 2.130 | 2.130 | +0.769 | -0.131 |
| 2023 | involved (1755) | rec_yds | 34.630 | 32.891 | 32.904 | +8.864 | -1.833 |
| 2023 | involved (1755) | tgt | 3.103 | 2.804 | 2.804 | +1.198 | -0.187 |
| 2024 | involved (1856) | rec | 2.292 | 2.194 | 2.193 | +0.585 | -0.145 |
| 2024 | involved (1856) | rec_yds | 32.889 | 31.346 | 31.349 | +7.457 | -0.984 |
| 2024 | involved (1856) | tgt | 3.063 | 2.873 | 2.873 | +0.931 | -0.173 |
| 2025 | involved (1843) | rec | 2.250 | 2.052 | 2.052 | +0.830 | +0.275 |
| 2025 | involved (1843) | rec_yds | 32.165 | 30.189 | 30.185 | +9.285 | +2.774 |
| 2025 | involved (1843) | tgt | 3.079 | 2.730 | 2.728 | +1.240 | +0.358 |
| 2023 | everyone (3201) | rec | 1.981 | 1.742 | 1.742 | +0.905 | +0.075 |
| 2023 | everyone (3201) | rec_yds | 28.535 | 26.132 | 26.137 | +10.516 | +0.955 |
| 2023 | everyone (3201) | tgt | 2.727 | 2.315 | 2.315 | +1.380 | +0.102 |
| 2024 | everyone (3161) | rec | 2.005 | 1.824 | 1.824 | +0.782 | +0.076 |
| 2024 | everyone (3161) | rec_yds | 27.870 | 25.751 | 25.754 | +9.409 | +1.536 |
| 2024 | everyone (3161) | tgt | 2.725 | 2.402 | 2.401 | +1.212 | +0.143 |
| 2025 | everyone (3236) | rec | 1.949 | 1.718 | 1.718 | +0.886 | +0.308 |
| 2025 | everyone (3236) | rec_yds | 27.061 | 24.736 | 24.734 | +10.095 | +3.474 |
| 2025 | everyone (3236) | tgt | 2.687 | 2.291 | 2.290 | +1.322 | +0.418 |

## Where the gain comes from (pooled over the three held-out seasons)

Each column adds one change to the one before it: a constant scale on the original model, then the new shrinkage, then the fitted team-volume model (replacing the hand-set formula whose spread term was backwards), then the defense nudge.

| players | stat | original | + scale | + new shrinkage | + fitted team volume | + defense nudge | total change |
|---|---|--:|--:|--:|--:|--:|--:|
| involved | tgt | 3.082 | 2.892 | 2.835 | 2.803 | 2.802 | -9.1% |
| involved | rec | 2.279 | 2.172 | 2.142 | 2.126 | 2.126 | -6.7% |
| involved | rec_yds | 33.220 | 32.048 | 31.654 | 31.472 | 31.476 | -5.3% |
| everyone | tgt | 2.713 | 2.482 | 2.359 | 2.336 | 2.336 | -13.9% |
| everyone | rec | 1.978 | 1.845 | 1.773 | 1.761 | 1.761 | -11.0% |
| everyone | rec_yds | 27.826 | 26.444 | 25.681 | 25.542 | 25.545 | -8.2% |

## Why one constant per position, and why it is learned on the involved players

Actual / projected by projected-usage tier under the new shrinkage, before any scale. Over-projection is steepest for depth players (and many of them play without a single target); from 4.5 projected targets up it is a fairly steady 0.82-0.89. In a one-off cross-season test when the constants were chosen (not re-run here), a usage-dependent scale beat the constant by 0.5-1.8% RMSE across everyone but by < 0.2% among players projected for 3+ targets, so the constant is kept.

| position | projected targets | players | with a stat row | actual / projected targets | receptions | rec yards |
|---|---|--:|--:|--:|--:|--:|
| WR | < 1.5 | 698 | 88% | 0.53 | 0.54 | 0.56 |
| WR | 1.5-3 | 1416 | 81% | 0.64 | 0.62 | 0.57 |
| WR | 3-4.5 | 1269 | 88% | 0.73 | 0.72 | 0.70 |
| WR | 4.5-6 | 908 | 96% | 0.83 | 0.82 | 0.83 |
| WR | 6-8 | 931 | 100% | 0.91 | 0.90 | 0.91 |
| WR | 8+ | 840 | 100% | 0.87 | 0.87 | 0.89 |
| TE | < 1.5 | 553 | 58% | 0.48 | 0.47 | 0.36 |
| TE | 1.5-3 | 1477 | 68% | 0.56 | 0.57 | 0.54 |
| TE | 3-4.5 | 604 | 88% | 0.77 | 0.81 | 0.81 |
| TE | 4.5-6 | 455 | 97% | 0.90 | 0.89 | 0.92 |
| TE | 6-8 | 324 | 100% | 0.93 | 0.95 | 0.97 |
| TE | 8+ | 123 | 100% | 0.87 | 0.87 | 0.84 |

## Shrinkage check (same rows for every setting; each scale learned on the other two seasons)

Rows: 9598 WR/TE player-weeks; "involved" = averaged >= 3.5 targets over their last 8 games (4085 rows). Pooled out-of-season RMSE.

| setting | involved: targets | receptions | rec yards | everyone: targets | receptions | rec yards |
|---|--:|--:|--:|--:|--:|--:|
| K_SHARE_REC 0.5 | 2.943 | 2.251 | 33.708 | 2.316 | 1.749 | 25.417 |
| K_SHARE_REC 1 (shipped) | 2.948 | 2.253 | 33.729 | 2.343 | 1.764 | 25.588 |
| K_SHARE_REC 2 | 2.961 | 2.259 | 33.796 | 2.408 | 1.803 | 25.990 |
| K_SHARE_REC 3 (the old value) | 2.976 | 2.267 | 33.881 | 2.475 | 1.842 | 26.395 |
| K_SHARE_REC 5 | 3.010 | 2.285 | 34.069 | 2.590 | 1.910 | 27.108 |
| K_YPT 28 (the old value) | 2.948 | 2.253 | 33.879 | 2.343 | 1.764 | 25.635 |
| K_YPT 100 | 2.948 | 2.253 | 33.695 | 2.343 | 1.764 | 25.597 |

## Downstream check: DK fantasy points and anytime-TD probability

The shipped model against the previous one, 9598 WR/TE player-weeks (2023-25 weeks 5-18). Nothing is learned here, so these are not out-of-season numbers for the scale -- they show what the change does to DK points and TD odds.

| players | pos | n | mean actual DK pts | mean projected, previous | recalibrated | RMSE trailing avg | previous | recalibrated |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| everyone who played | WR | 6062 | 6.73 | 8.66 | 6.74 | 6.361 | 6.690 | 6.333 |
| everyone who played | TE | 3536 | 4.60 | 6.52 | 5.20 | 4.885 | 5.259 | 4.854 |
| only those who ended up with 3+ targets/carries/attempts | WR | 3363 | 11.13 | 10.99 | 9.00 | 7.766 | 7.580 | 7.758 |
| only those who ended up with 3+ targets/carries/attempts | TE | 1350 | 10.02 | 9.09 | 7.88 | 6.745 | 6.391 | 6.613 |

The second population is the one `matchup_model.opp.backtest` and `td_calib` score (they keep only players who ended up with 3+ / 2+ touches). Selecting on the outcome inflates the actuals, so a model that over-projects looks best there; that is why those two reports cannot be used to compare the old and new models, and why they were not regenerated.

Blend weight on the opportunity model (the rest is the player's trailing DK average); fitted on 2023-24, scored on 2025, everyone who played:

| pos | production weight | best weight, previous model | best, recalibrated | 2025 RMSE: trailing avg | recalibrated at production weight | at its best weight |
|---|--:|--:|--:|--:|--:|--:|
| WR | 0.90 | 0.17 | 0.48 | 6.072 | 5.928 | 5.931 |
| TE | 1.00 | 0.05 | 0.46 | 5.097 | 4.980 | 4.988 |

The production weights were fit on the outcome-selected population with the old model; with the recalibrated model the blend is nearly flat between them and the best weight, so they are left alone (the blend only moves `fp` and `role_ratio`; the prop lines come straight from the opportunity model).

P(at least one TD) = 1 - exp(-expected receiving TDs), against whether the player scored (receiving or rushing):

| players | n | actual rate | mean P, previous | recalibrated | Brier, previous | recalibrated | flat average |
|---|--:|--:|--:|--:|--:|--:|--:|
| everyone who played | 9598 | 15.0% | 18.2% | 15.9% | 0.1163 | 0.1147 | 0.1277 |
| trailing 8-game average >= 3.5 targets | 4085 | 26.3% | 25.8% | 24.5% | 0.1867 | 0.1868 | 0.1940 |

## Constants for model.py (learned on all three seasons)

```python
K_SHARE_REC = 1.0
K_YPT = 60.0
# (learned under TEAM_VOLUME_MODEL = "fitted")
REC_SCALE = {
    "WR": {"tgt": 0.847, "rec": 0.840, "rec_yds": 0.848, "td": 0.933},
    "TE": {"tgt": 0.867, "rec": 0.883, "rec_yds": 0.893, "td": 1.010},
}
DEF_TGT_B = 0.09
```
