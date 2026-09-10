# Opportunity model - walk-forward backtest

- rows: **9198**  (6137 train [2023, 2024] / 3061 holdout 2025)  ·  weeks 4-18  ·  DK scoring
- **naive** = EWMA trailing DK pts · **current** = trailing usage x efficiency (`project_stats` port) · **opp** = opportunity model · **blend** = per-position naive/opp mix, weights fit on train only
- fitted opp weight in blend: QB 0.64, RB 0.70, WR 0.46, TE 0.60

## Accuracy - all test rows (lower RMSE, higher rho better)
| slice | n | RMSE naive | RMSE current | RMSE opp | RMSE blend | rho naive | rho current | rho opp | rho blend |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ALL | 9198 | 7.58 | 7.66 | 7.54 | 7.41 | 0.524 | 0.518 | 0.513 | 0.526 |
| QB | 1355 | 8.45 | 8.59 | 8.44 | 8.40 | 0.358 | 0.340 | 0.353 | 0.360 |
| RB | 2785 | 7.35 | 7.37 | 7.18 | 7.15 | 0.571 | 0.568 | 0.569 | 0.573 |
| WR | 3619 | 7.75 | 7.84 | 7.85 | 7.61 | 0.454 | 0.446 | 0.453 | 0.458 |
| TE | 1439 | 6.70 | 6.75 | 6.46 | 6.34 | 0.358 | 0.358 | 0.385 | 0.380 |

## Accuracy - 2025 holdout only (blend weights were NOT fit on this)
| slice | n | RMSE naive | RMSE current | RMSE opp | RMSE blend | rho naive | rho current | rho opp | rho blend |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ALL | 3061 | 7.65 | 7.70 | 7.62 | 7.48 | 0.507 | 0.505 | 0.499 | 0.511 |
| QB | 445 | 8.57 | 8.73 | 8.64 | 8.58 | 0.332 | 0.295 | 0.311 | 0.327 |
| RB | 924 | 7.89 | 7.90 | 7.65 | 7.65 | 0.550 | 0.552 | 0.555 | 0.557 |
| WR | 1169 | 7.41 | 7.49 | 7.64 | 7.31 | 0.430 | 0.425 | 0.435 | 0.436 |
| TE | 523 | 6.86 | 6.82 | 6.52 | 6.47 | 0.334 | 0.365 | 0.404 | 0.379 |

## Weekly top-K by projection - mean actual DK pts of each model's weekly top-K
(K: QB=12, RB=24, WR=30, TE=12; higher = better)

| pos | naive | current | opp | blend | overlap naive | opp | blend |
|---|--:|--:|--:|--:|--:|--:|--:|
| QB | 20.15 | 19.90 | 19.96 | **20.19** | 56% | 56% | **56%** |
| RB | 15.45 | 15.40 | 15.46 | **15.46** | 63% | 64% | **64%** |
| WR | 15.05 | 15.02 | 15.06 | **15.08** | 57% | 58% | **58%** |
| TE | 12.42 | 12.35 | 12.67 | **12.62** | 53% | 54% | **54%** |

## Verdict (2025 holdout): **blend beats the baselines on the holdout** - wire the opportunity model in as a per-position blend with the trailing average.

RMSE   naive 7.65 · current 7.70 · opp 7.62 · blend 7.48
rho    naive 0.507 · current 0.505 · opp 0.499 · blend 0.511