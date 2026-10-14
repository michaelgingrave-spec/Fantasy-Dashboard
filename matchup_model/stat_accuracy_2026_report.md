# Stat-line accuracy by FantasyPoints-deference variant — 2026 weeks 1-5

## Part A — broad accuracy (every player with real usage, RMSE/MAE)

**rec_yds**

| variant | n | RMSE | MAE |
|---|--:|--:|--:|
| anchored | 1220 | 25.64 | 18.09 |
| current | 1220 | 25.79 | 18.58 |
| no_shift | 1220 | 26.85 | 19.85 |

**rec**

| variant | n | RMSE | MAE |
|---|--:|--:|--:|
| current | 1220 | 1.89 | 1.41 |
| anchored | 1220 | 1.92 | 1.42 |
| no_shift | 1220 | 1.98 | 1.51 |

**rush_yds**

| variant | n | RMSE | MAE |
|---|--:|--:|--:|
| current | 521 | 26.07 | 18.62 |
| anchored | 521 | 26.25 | 18.24 |
| no_shift | 521 | 26.83 | 19.43 |

**pass_yds**

| variant | n | RMSE | MAE |
|---|--:|--:|--:|
| current | 163 | 83.77 | 66.45 |
| no_shift | 163 | 84.01 | 66.72 |
| anchored | 163 | 86.53 | 67.94 |

**pass_td**

| variant | n | RMSE | MAE |
|---|--:|--:|--:|
| current | 163 | 1.12 | 0.89 |
| anchored | 163 | 1.12 | 0.89 |
| no_shift | 163 | 1.13 | 0.89 |

## Part B — would it have won (real pulled lines, real results)

- 2448 props with a real book line per variant

| variant | n | win% | ROI@-110% |
|---|--:|--:|--:|
| no_shift | 2448 | 50.5 | -3.6 |
| current | 2448 | 50.4 | -3.8 |
| anchored | 2448 | 51.2 | -2.2 |