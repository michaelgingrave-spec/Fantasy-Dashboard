# Opportunity model - walk-forward backtest

- rows: **9198**  (6137 train [2023, 2024] / 3061 holdout 2025)  ·  weeks 4-18  ·  DK scoring
- **naive** = EWMA trailing DK pts · **current** = trailing usage x efficiency (`project_stats` port) · **opp** = opportunity model · **blend** = per-position naive/opp mix, weights fit on train only
- fitted opp weight in blend: QB 0.64, RB 0.54, WR 0.91, TE 1.00

## Accuracy - all test rows (lower RMSE, higher rho better)
| slice | n | RMSE naive | RMSE current | RMSE opp | RMSE blend | rho naive | rho current | rho opp | rho blend |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ALL | 9198 | 7.58 | 7.66 | 7.43 | 7.40 | 0.524 | 0.518 | 0.519 | 0.526 |
| QB | 1355 | 8.45 | 8.59 | 8.44 | 8.40 | 0.358 | 0.340 | 0.353 | 0.360 |
| RB | 2785 | 7.35 | 7.37 | 7.29 | 7.21 | 0.571 | 0.568 | 0.566 | 0.572 |
| WR | 3619 | 7.75 | 7.84 | 7.53 | 7.53 | 0.454 | 0.446 | 0.446 | 0.449 |
| TE | 1439 | 6.70 | 6.75 | 6.37 | 6.37 | 0.358 | 0.358 | 0.375 | 0.375 |

## Accuracy - 2025 holdout only (blend weights were NOT fit on this)
| slice | n | RMSE naive | RMSE current | RMSE opp | RMSE blend | rho naive | rho current | rho opp | rho blend |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ALL | 3061 | 7.65 | 7.70 | 7.48 | 7.45 | 0.507 | 0.505 | 0.505 | 0.512 |
| QB | 445 | 8.57 | 8.73 | 8.64 | 8.58 | 0.332 | 0.295 | 0.311 | 0.327 |
| RB | 924 | 7.89 | 7.90 | 7.79 | 7.72 | 0.550 | 0.552 | 0.552 | 0.554 |
| WR | 1169 | 7.41 | 7.49 | 7.18 | 7.18 | 0.430 | 0.425 | 0.428 | 0.430 |
| TE | 523 | 6.86 | 6.82 | 6.45 | 6.45 | 0.334 | 0.365 | 0.395 | 0.395 |

## Weekly top-K by projection - mean actual DK pts of each model's weekly top-K
(K: QB=12, RB=24, WR=30, TE=12; higher = better)

| pos | naive | current | opp | blend | overlap naive | opp | blend |
|---|--:|--:|--:|--:|--:|--:|--:|
| QB | 20.15 | 19.90 | 19.96 | **20.19** | 56% | 56% | **56%** |
| RB | 15.45 | 15.40 | 15.43 | **15.45** | 63% | 63% | **63%** |
| WR | 15.05 | 15.02 | 15.05 | **15.12** | 57% | 58% | **58%** |
| TE | 12.42 | 12.35 | 12.72 | **12.72** | 53% | 54% | **54%** |

## Injury redistribution — does folding the weekly injury report into usage help?
- rows where a teammate's Out/Doubtful/Questionable moved this player's share: **2991** (32.5% of all) · holdout 928

| slice | n | RMSE opp | RMSE opp+inj | ΔRMSE | RMSE blend | RMSE blend+inj | ΔRMSE |
|---|--:|--:|--:|--:|--:|--:|--:|
| ALL test | 9198 | 7.43 | 7.41 | **+0.02** | 7.40 | 7.38 | **+0.02** |
|   disrupted only | 2991 | 7.54 | 7.51 | **+0.03** | 7.53 | 7.49 | **+0.04** |
| 2025 holdout | 3061 | 7.48 | 7.44 | **+0.04** | 7.45 | 7.42 | **+0.03** |
|   disrupted (holdout) | 928 | 7.46 | 7.40 | **+0.06** | 7.47 | 7.40 | **+0.07** |

- on disrupted holdout rows, MAE improves by **-0.14** DK pts with the injury layer

## Verdict (2025 holdout): **blend+injury beats the baselines on the holdout** — it's the projection source. Injury layer helps.

RMSE   naive 7.65 · current 7.70 · opp 7.48 · blend 7.45 · opp_inj 7.44 · blend_inj 7.42
rho    naive 0.507 · current 0.505 · opp 0.505 · blend 0.512 · blend_inj 0.518