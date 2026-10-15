# WR/TE receiving recalibration (2023-25; every number scored with constants learned on the other two seasons)

9598 WR/TE player-weeks (weeks 5-18; 1341 are players on the field with no stat line, true zeros). New shrinkage: K_SHARE_REC 1, K_YPT 60 (old 3 / 28). Constants are learned on the 5579 rows projected for >= 3 targets ("involved"): the players that get prop lines and DFS salaries. `td` is the multiplier on expected receiving TDs, learned the same way; the TD rate runs on the unscaled targets (see the downstream check: scaling TDs with the volume scale leaves involved players ~10% short on TDs).

## Out-of-season RMSE

| held-out season | players | stat | old | new shrinkage + scale | + defense nudge | old mean error | new mean error |
|---|---|---|--:|--:|--:|--:|--:|
| 2023 | involved (1859) | rec | 2.265 | 2.105 | 2.103 | +0.775 | +0.021 |
| 2023 | involved (1859) | rec_yds | 34.028 | 32.294 | 32.295 | +9.050 | +0.151 |
| 2023 | involved (1859) | tgt | 3.071 | 2.780 | 2.778 | +1.210 | +0.046 |
| 2024 | involved (1876) | rec | 2.290 | 2.202 | 2.200 | +0.603 | -0.209 |
| 2024 | involved (1876) | rec_yds | 32.773 | 31.466 | 31.463 | +7.636 | -1.900 |
| 2024 | involved (1876) | tgt | 3.055 | 2.879 | 2.875 | +0.969 | -0.260 |
| 2025 | involved (1844) | rec | 2.247 | 2.059 | 2.057 | +0.857 | +0.182 |
| 2025 | involved (1844) | rec_yds | 32.180 | 30.307 | 30.277 | +9.652 | +1.643 |
| 2025 | involved (1844) | tgt | 3.077 | 2.762 | 2.754 | +1.273 | +0.203 |
| 2023 | everyone (3201) | rec | 1.981 | 1.753 | 1.751 | +0.905 | +0.179 |
| 2023 | everyone (3201) | rec_yds | 28.535 | 26.221 | 26.220 | +10.516 | +2.191 |
| 2023 | everyone (3201) | tgt | 2.727 | 2.333 | 2.331 | +1.380 | +0.260 |
| 2024 | everyone (3161) | rec | 2.005 | 1.836 | 1.834 | +0.782 | +0.021 |
| 2024 | everyone (3161) | rec_yds | 27.870 | 25.933 | 25.930 | +9.409 | +0.768 |
| 2024 | everyone (3161) | tgt | 2.725 | 2.423 | 2.420 | +1.212 | +0.058 |
| 2025 | everyone (3236) | rec | 1.949 | 1.728 | 1.727 | +0.886 | +0.227 |
| 2025 | everyone (3236) | rec_yds | 27.061 | 24.869 | 24.850 | +10.095 | +2.481 |
| 2025 | everyone (3236) | tgt | 2.687 | 2.319 | 2.314 | +1.322 | +0.291 |

## Where the gain comes from (pooled over the three held-out seasons)

`old + scale only` re-scales the OLD model's projections by a constant learned the same way, so the step from there to `new shrinkage + scale` is the shrinkage change alone, and the last column is the defense nudge.

| players | stat | old | old + scale only | new shrinkage + scale | + defense nudge | total change |
|---|---|--:|--:|--:|--:|--:|
| involved | tgt | 3.067 | 2.870 | 2.808 | 2.803 | -8.6% |
| involved | rec | 2.268 | 2.157 | 2.123 | 2.121 | -6.5% |
| involved | rec_yds | 33.004 | 31.793 | 31.369 | 31.359 | -5.0% |
| everyone | tgt | 2.713 | 2.479 | 2.358 | 2.355 | -13.2% |
| everyone | rec | 1.978 | 1.843 | 1.772 | 1.771 | -10.5% |
| everyone | rec_yds | 27.826 | 26.427 | 25.677 | 25.669 | -7.7% |

## Why one constant per position, and why it is learned on the involved players

Actual / projected by projected-usage tier under the new shrinkage, before any scale. Over-projection is steepest for depth players (and many of them play without a single target); from 4.5 projected targets up it is a fairly steady 0.82-0.89. In a one-off cross-season test when the constants were chosen (not re-run here), a usage-dependent scale beat the constant by 0.5-1.8% RMSE across everyone but by < 0.2% among players projected for 3+ targets, so the constant is kept.

| position | projected targets | players | with a stat row | actual / projected targets | receptions | rec yards |
|---|---|--:|--:|--:|--:|--:|
| WR | < 1.5 | 677 | 88% | 0.51 | 0.51 | 0.52 |
| WR | 1.5-3 | 1373 | 80% | 0.64 | 0.61 | 0.58 |
| WR | 3-4.5 | 1303 | 89% | 0.72 | 0.71 | 0.68 |
| WR | 4.5-6 | 915 | 96% | 0.84 | 0.81 | 0.83 |
| WR | 6-8 | 883 | 99% | 0.88 | 0.88 | 0.91 |
| WR | 8+ | 911 | 100% | 0.84 | 0.84 | 0.85 |
| TE | < 1.5 | 544 | 58% | 0.47 | 0.45 | 0.37 |
| TE | 1.5-3 | 1425 | 67% | 0.54 | 0.55 | 0.52 |
| TE | 3-4.5 | 653 | 88% | 0.77 | 0.82 | 0.80 |
| TE | 4.5-6 | 445 | 97% | 0.89 | 0.88 | 0.93 |
| TE | 6-8 | 319 | 99% | 0.89 | 0.92 | 0.92 |
| TE | 8+ | 150 | 100% | 0.82 | 0.82 | 0.80 |

## Shrinkage check (same rows for every setting; each scale learned on the other two seasons)

Rows: 9598 WR/TE player-weeks; "involved" = averaged >= 3.5 targets over their last 8 games (4085 rows). Pooled out-of-season RMSE.

| setting | involved: targets | receptions | rec yards | everyone: targets | receptions | rec yards |
|---|--:|--:|--:|--:|--:|--:|
| K_SHARE_REC 0.5 | 2.984 | 2.271 | 33.952 | 2.337 | 1.759 | 25.548 |
| K_SHARE_REC 1 (shipped) | 2.988 | 2.273 | 33.972 | 2.363 | 1.774 | 25.712 |
| K_SHARE_REC 2 | 3.001 | 2.279 | 34.037 | 2.426 | 1.811 | 26.105 |
| K_SHARE_REC 3 (the old value) | 3.016 | 2.287 | 34.122 | 2.491 | 1.850 | 26.503 |
| K_SHARE_REC 5 | 3.050 | 2.306 | 34.310 | 2.605 | 1.917 | 27.208 |
| K_YPT 28 (the old value) | 2.988 | 2.273 | 34.088 | 2.363 | 1.774 | 25.744 |
| K_YPT 100 | 2.988 | 2.273 | 33.959 | 2.363 | 1.774 | 25.731 |

## Downstream check: DK fantasy points and anytime-TD probability

The shipped model against the previous one, 9598 WR/TE player-weeks (2023-25 weeks 5-18). Nothing is learned here, so these are not out-of-season numbers for the scale -- they show what the change does to DK points and TD odds.

| players | pos | n | mean actual DK pts | mean projected, previous | recalibrated | RMSE trailing avg | previous | recalibrated |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| everyone who played | WR | 6062 | 6.73 | 8.66 | 6.70 | 6.361 | 6.690 | 6.356 |
| everyone who played | TE | 3536 | 4.60 | 6.52 | 5.18 | 4.885 | 5.259 | 4.881 |
| only those who ended up with 3+ targets/carries/attempts | WR | 3363 | 11.13 | 10.99 | 8.96 | 7.766 | 7.580 | 7.801 |
| only those who ended up with 3+ targets/carries/attempts | TE | 1350 | 10.02 | 9.09 | 7.86 | 6.745 | 6.391 | 6.664 |

The second population is the one `matchup_model.opp.backtest` and `td_calib` score (they keep only players who ended up with 3+ / 2+ touches). Selecting on the outcome inflates the actuals, so a model that over-projects looks best there; that is why those two reports cannot be used to compare the old and new models, and why they were not regenerated.

Blend weight on the opportunity model (the rest is the player's trailing DK average); fitted on 2023-24, scored on 2025, everyone who played:

| pos | production weight | best weight, previous model | best, recalibrated | 2025 RMSE: trailing avg | recalibrated at production weight | at its best weight |
|---|--:|--:|--:|--:|--:|--:|
| WR | 0.90 | 0.17 | 0.46 | 6.072 | 5.967 | 5.950 |
| TE | 1.00 | 0.05 | 0.40 | 5.097 | 5.015 | 5.006 |

The production weights were fit on the outcome-selected population with the old model; with the recalibrated model the blend is nearly flat between them and the best weight, so they are left alone (the blend only moves `fp` and `role_ratio`; the prop lines come straight from the opportunity model).

P(at least one TD) = 1 - exp(-expected receiving TDs), against whether the player scored (receiving or rushing):

| players | n | actual rate | mean P, previous | recalibrated | Brier, previous | recalibrated | flat average |
|---|--:|--:|--:|--:|--:|--:|--:|
| everyone who played | 9598 | 15.0% | 18.2% | 15.8% | 0.1163 | 0.1147 | 0.1277 |
| trailing 8-game average >= 3.5 targets | 4085 | 26.3% | 25.8% | 24.4% | 0.1867 | 0.1868 | 0.1940 |

## Constants for model.py (learned on all three seasons)

```python
K_SHARE_REC = 1.0
K_YPT = 60.0
REC_SCALE = {
    "WR": {"tgt": 0.826, "rec": 0.820, "rec_yds": 0.827, "td": 0.922},
    "TE": {"tgt": 0.844, "rec": 0.860, "rec_yds": 0.869, "td": 0.992},
}
DEF_TGT_B = 0.22
```
