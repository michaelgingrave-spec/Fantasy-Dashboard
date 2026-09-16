# Box-count-conditioned RB efficiency vs defensive box tendency

- 2799 RB player-games, wk 4-18, seasons [2023, 2024]+2025.
- `signal` = defense's trailing avg-men-in-box deviation from league avg x (RB's shrunk stacked-box edge minus light-box edge) -- positive when a defense that stacks the box more than usual faces a back who's relatively better against stacked fronts (or the mirror: light-box defense, light-box-dependent back).
- Player edges are EB-shrunk toward a population prior (K=100 attempts), same style opp_line() already uses -- and this session's other scheme calibrations used.

## Whole-population RMSE (n=1881 train, 918 holdout)

fitted beta (train): **+0.1060**

| slice | n | RMSE base | RMSE adjusted | Δ | MAE base | MAE adjusted | Δ |
|---|--:|--:|--:|--:|--:|--:|--:|
| train [2023, 2024] | 1881 | 30.41 | 30.40 | **+0.01** | 22.98 | 22.98 | **-0.00** |
| HOLDOUT 2025 | 918 | 33.01 | 33.12 | **-0.12** | 24.00 | 23.96 | **+0.04** |

## Holdout RMSE by |signal| tercile

| tercile | n | signal range | RMSE base | RMSE adjusted | Δ |
|---|--:|---|--:|--:|--:|
| low | 306 | 0.0000-0.1172 | 34.38 | 34.37 | **+0.011** |
| mid | 306 | 0.1174-0.2668 | 32.37 | 32.52 | **-0.151** |
| high | 306 | 0.2672-1.1890 | 32.23 | 32.45 | **-0.216** |

Direction agreement (n=889): **54.3%** (50% = coin flip)

## Yards/carry isolated (strips out attempt-volume noise)

beta fit directly on y/c residual (train): -0.4944
raw correlation(signal, actual_ypc - base_ypc): train +0.0092, holdout -0.0556

Verdict: **no improvement**