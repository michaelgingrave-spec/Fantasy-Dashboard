# Opponent-efficiency-allowed calibration — cheap check before scheme splits

- 9213 player-games, seasons [2023, 2024]+2025, wk 4-18.
- `base` = opp_line's own rec_yds/rush_yds (trailing usage x efficiency, ZERO opponent conditioning today). `factor` = this defense's trailing allowed-yards-per-target(carry) / league average (>1 = soft matchup, <1 = tough), strictly walk-forward. `adjusted` = base*(1+beta*(factor-1)), beta fit by OLS on [2023, 2024] only, judged on a 2025 holdout it never saw.

## WR/TE receiving yards  (n=6200: 4131 train, 2069 holdout)

**fitted beta: +0.059**  (defense signal moved the fit toward using it)

| slice | n | RMSE base | RMSE adjusted | Δ | MAE base | MAE adjusted | Δ |
|---|--:|--:|--:|--:|--:|--:|--:|
| train [2023, 2024] | 4131 | 31.22 | 31.22 | **+0.00** | 23.97 | 23.98 | **-0.00** |
| HOLDOUT 2025 | 2069 | 29.86 | 29.87 | **-0.01** | 23.09 | 23.09 | **+0.00** |

By how extreme the matchup is (holdout only):

| tercile | n | factor range | RMSE base | RMSE adjusted | Δ |
|---|--:|---|--:|--:|--:|
| soft/tough closest to avg | 691 | 0.96-1.04 | 30.38 | 30.38 | **+0.00** |
| mid | 688 | 0.90-1.10 | 29.76 | 29.77 | **-0.01** |
| biggest mismatch | 690 | 0.74-1.34 | 29.43 | 29.44 | **-0.01** |

Verdict: **no improvement on the holdout — opponent-efficiency-allowed alone isn't enough signal here**

## RB rushing yards  (n=3013: 2013 train, 1000 holdout)

**fitted beta: +0.774**  (defense signal moved the fit toward using it)

| slice | n | RMSE base | RMSE adjusted | Δ | MAE base | MAE adjusted | Δ |
|---|--:|--:|--:|--:|--:|--:|--:|
| train [2023, 2024] | 2013 | 29.88 | 29.59 | **+0.29** | 22.67 | 22.33 | **+0.34** |
| HOLDOUT 2025 | 1000 | 32.04 | 32.11 | **-0.07** | 23.37 | 23.59 | **-0.22** |

By how extreme the matchup is (holdout only):

| tercile | n | factor range | RMSE base | RMSE adjusted | Δ |
|---|--:|---|--:|--:|--:|
| soft/tough closest to avg | 336 | 0.95-1.05 | 30.28 | 30.28 | **+0.01** |
| mid | 332 | 0.89-1.11 | 33.90 | 34.02 | **-0.12** |
| biggest mismatch | 332 | 0.76-1.61 | 31.86 | 31.94 | **-0.08** |

Verdict: **no improvement on the holdout — opponent-efficiency-allowed alone isn't enough signal here**
