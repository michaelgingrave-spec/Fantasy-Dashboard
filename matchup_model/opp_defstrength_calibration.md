# Schedule-adjusted, reliability-shrunk RB defense strength

- 2799 RB player-games, wk 4-18, seasons [2023, 2024]+2025.
- `dev` = defense's trailing (walk-forward) schedule-adjusted rush-fp-allowed delta -- each game compared to the OPPOSING OFFENSE'S OWN trailing pace entering that game (SAFPA-style, made walk-forward safe), then EB-shrunk toward 0 with K=24 games, derived from measured split-half reliability (~0.265) via Tango's method, not hand-picked.

## Whole-population RMSE (n=1881 train, 918 holdout)

fitted beta (train): **+0.0824**

| slice | n | RMSE base | RMSE adjusted | Δ | MAE base | MAE adjusted | Δ |
|---|--:|--:|--:|--:|--:|--:|--:|
| train [2023, 2024] | 1881 | 30.41 | 30.03 | **+0.38** | 22.98 | 22.65 | **+0.33** |
| HOLDOUT 2025 | 918 | 33.01 | 33.10 | **-0.09** | 24.00 | 24.13 | **-0.13** |

## Holdout RMSE by |dev| tercile

| tercile | n | dev range | RMSE base | RMSE adjusted | Δ |
|---|--:|---|--:|--:|--:|
| low | 308 | 0.00-0.50 | 33.62 | 33.63 | **-0.012** |
| mid | 305 | 0.51-1.21 | 30.45 | 30.66 | **-0.217** |
| high | 305 | 1.21-4.19 | 34.80 | 34.86 | **-0.057** |

Direction agreement (n=918): **50.5%** (50% = coin flip)

Verdict: **no improvement**