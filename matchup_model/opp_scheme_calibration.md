# Scheme-conditioned efficiency (man vs zone) vs opponent tendency — v1 vs v2

- 8043 WR/TE player-games total, wk 4-18, seasons [2022, 2023, 2024]+2025.
- **v1** = each player's own trailing man/zone YPRR minus his own trailing overall YPRR — no pooling across players. Only scored where he has 4+ prior games with split data (7528/8043 rows).
- **v2** = the same per-game deltas, empirical-Bayes shrunk toward a POPULATION prior (pooled across every WR/TE, years of data) with pseudo-count K=100 routes — the same shrinkage style `opp_line()` already uses for share/efficiency, extended to the coverage split. Scored on every row — a player with no man-route history on file just gets the population prior instead of being dropped.
- `adjusted` = base*(1+beta*signal), beta fit by OLS on train, judged on a 2025 holdout it never saw.

## v1 vs v2 on the SAME rows (only where v1 had enough personal data — fair fight)

### v1 — own-player only  (n=7528: 5674 train, 1854 holdout)

fitted beta: **+0.126**

| slice | n | RMSE base | RMSE adjusted | Δ | MAE base | MAE adjusted | Δ |
|---|--:|--:|--:|--:|--:|--:|--:|
| train [2022, 2023, 2024] | 5674 | 31.25 | 31.24 | **+0.01** | 24.08 | 24.07 | **+0.01** |
| HOLDOUT 2025 | 1854 | 30.33 | 30.34 | **-0.02** | 23.47 | 23.48 | **-0.00** |

Verdict: **no improvement**

### v2 — population-shrunk, same rows  (n=7528: 5674 train, 1854 holdout)

fitted beta: **+0.369**

| slice | n | RMSE base | RMSE adjusted | Δ | MAE base | MAE adjusted | Δ |
|---|--:|--:|--:|--:|--:|--:|--:|
| train [2022, 2023, 2024] | 5674 | 31.25 | 31.24 | **+0.01** | 24.08 | 24.07 | **+0.01** |
| HOLDOUT 2025 | 1854 | 30.33 | 30.37 | **-0.04** | 23.47 | 23.50 | **-0.03** |

Verdict: **no improvement**

## v2 on its full reach (every row, including thin/zero personal history)

### v2 — population-shrunk, full data  (n=8043: 6015 train, 2028 holdout)

fitted beta: **+0.344**

| slice | n | RMSE base | RMSE adjusted | Δ | MAE base | MAE adjusted | Δ |
|---|--:|--:|--:|--:|--:|--:|--:|
| train [2022, 2023, 2024] | 6015 | 31.15 | 31.15 | **+0.01** | 24.02 | 24.01 | **+0.01** |
| HOLDOUT 2025 | 2028 | 29.87 | 29.90 | **-0.04** | 23.05 | 23.07 | **-0.02** |

Verdict: **no improvement**
