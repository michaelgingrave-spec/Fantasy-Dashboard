# Slot vs wide alignment-conditioned efficiency vs opponent tendency

- 6080 WR/TE player-games, wk 4-18, seasons [2023, 2024]+2025.
- `signal` = defense's trailing slot-allowed-YPT deviation from league avg x (player's shrunk slot-edge minus wide-edge) -- positive when a relatively-better-from-the-slot player faces a defense that's relatively softer against slot-targeted throws.
- Player edges are EB-shrunk toward a population prior (K=100 routes), same style as opp_line() already uses -- and scheme_calib.py used for man/zone.

## Whole-population RMSE (n=4052 train, 2028 holdout)

fitted beta (train): **-0.0406**

| slice | n | RMSE base | RMSE adjusted | Δ | MAE base | MAE adjusted | Δ |
|---|--:|--:|--:|--:|--:|--:|--:|
| train [2023, 2024] | 4052 | 31.31 | 31.31 | **+0.00** | 24.01 | 24.00 | **+0.00** |
| HOLDOUT 2025 | 2028 | 29.87 | 29.89 | **-0.02** | 23.05 | 23.06 | **-0.01** |

## Holdout RMSE by |signal| tercile (does it concentrate in the biggest mismatches?)

| tercile | n | signal range | RMSE base | RMSE adjusted | Δ |
|---|--:|---|--:|--:|--:|
| low | 677 | 0.0000-0.0226 | 28.96 | 28.97 | **-0.001** |
| mid | 675 | 0.0227-0.0908 | 28.94 | 28.94 | **-0.003** |
| high | 676 | 0.0910-1.5622 | 31.62 | 31.68 | **-0.062** |

Direction agreement (nudge sign vs actual-vs-base sign, n=1960): **49.3%** (50% = coin flip)

## Yards/target isolated (strips out target-volume noise)

beta fit directly on y/t residual (train): +0.0051
raw correlation(signal, actual_ypt - base_ypt): train -0.0006, holdout +0.0196

Verdict: **no improvement**