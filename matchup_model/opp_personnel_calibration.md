# Defensive personnel (nickel/dime rate) vs receiving yards

- 8043 WR/TE player-games (6015 train [2022, 2023, 2024], 2028 holdout 2025), wk 4-18.
- `factor` = defense's trailing (100-BASE%) [nickel+dime rate] / league average, walk-forward. `adjusted` = base*(1+beta*(factor-1)), beta fit by OLS on [2022, 2023, 2024], judged on a 2025 holdout it never saw. Press-rate is NOT included — 0% populated in 2022-2024, no walk-forward holdout possible with one season of real data.

**fitted beta: -0.035**

| slice | n | RMSE base | RMSE adjusted | Δ | MAE base | MAE adjusted | Δ |
|---|--:|--:|--:|--:|--:|--:|--:|
| train [2022, 2023, 2024] | 6015 | 31.15 | 31.15 | **+0.00** | 24.02 | 24.02 | **+0.00** |
| HOLDOUT 2025 | 2028 | 29.87 | 29.89 | **-0.02** | 23.05 | 23.07 | **-0.02** |

By how far from league-average personnel usage (holdout only):

| tercile | n | RMSE base | RMSE adjusted | Δ |
|---|--:|--:|--:|--:|
| closest to avg | 677 | 29.71 | 29.71 | **+0.00** |
| mid | 677 | 29.61 | 29.61 | **-0.01** |
| biggest mismatch | 674 | 30.28 | 30.35 | **-0.07** |

Verdict: **no improvement**