# Shell-conditioned (1-high/2-high) QB passing-yard calibration

- 1291 QB player-games, wk 4-18, seasons [2023, 2024]+2025, min 10 attempts.
- `signal` = defense's trailing 2-HIGH % deviation from league avg x (QB's shrunk two-high-shell edge minus single-high-shell edge) -- positive when a defense that plays 2-high more than usual faces a QB who's relatively better against two-high shells than his own single-high number would predict.
- Player edges are EB-shrunk toward a population prior (K=100 dropbacks), same style opp_line()/box_calib.py already use.
- Distinct from scheme_calib.py's man-vs-zone axis (coverage TYPE, found null): this is safety-shell alignment (1-high vs 2-high), which cuts across man and zone alike.

## Whole-population RMSE (n=873 train, 418 holdout)

fitted beta (train): **-0.0294**

| slice | n | RMSE base | RMSE adjusted | Δ | MAE base | MAE adjusted | Δ |
|---|--:|--:|--:|--:|--:|--:|--:|
| train [2023, 2024] | 873 | 78.51 | 78.51 | **+0.00** | 62.77 | 62.75 | **+0.02** |
| HOLDOUT 2025 | 418 | 76.21 | 76.29 | **-0.08** | 60.73 | 60.78 | **-0.05** |

## Holdout RMSE by |signal| tercile

| tercile | n | signal range | RMSE base | RMSE adjusted | Δ |
|---|--:|---|--:|--:|--:|
| low | 140 | 0.0000-0.0166 | 77.62 | 77.62 | **-0.007** |
| mid | 139 | 0.0168-0.0561 | 79.35 | 79.34 | **+0.011** |
| high | 139 | 0.0562-0.3932 | 71.42 | 71.68 | **-0.262** |

Direction agreement (n=403): **46.4%** (50% = coin flip)

## Yards/attempt isolated (strips out attempt-volume noise)

beta fit directly on y/a residual (train): +0.1431
raw correlation(signal, actual_ypa - base_ypa): train +0.0217, holdout +0.1192

Verdict: **no improvement**