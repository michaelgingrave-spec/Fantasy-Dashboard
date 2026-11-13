# Stat-line accuracy by FantasyPoints-deference variant — 2026 weeks 1-9

## Part A — broad accuracy (every player with real usage, RMSE/MAE)

**rec_yds**

| variant | n | RMSE | MAE |
|---|--:|--:|--:|
| current | 2130 | 25.12 | 17.7 |
| anchored | 2130 | 25.2 | 17.8 |
| no_shift | 2130 | 25.62 | 18.23 |

**rec**

| variant | n | RMSE | MAE |
|---|--:|--:|--:|
| current | 2130 | 1.8 | 1.32 |
| anchored | 2130 | 1.82 | 1.36 |
| no_shift | 2130 | 1.84 | 1.35 |

**rush_yds**

| variant | n | RMSE | MAE |
|---|--:|--:|--:|
| current | 908 | 25.29 | 17.83 |
| anchored | 908 | 25.59 | 17.82 |
| no_shift | 908 | 25.75 | 18.32 |

**pass_yds**

| variant | n | RMSE | MAE |
|---|--:|--:|--:|
| current | 285 | 83.82 | 66.08 |
| no_shift | 285 | 84.2 | 66.43 |
| anchored | 285 | 86.81 | 68.85 |

**pass_td**

| variant | n | RMSE | MAE |
|---|--:|--:|--:|
| no_shift | 285 | 1.09 | 0.87 |
| current | 285 | 1.09 | 0.87 |
| anchored | 285 | 1.09 | 0.88 |

## Part B — would it have won (real pulled lines, real results)

- 4441 props with a real book line per variant

| variant | n | win% | ROI@-110% |
|---|--:|--:|--:|
| no_shift | 4441 | 52.7 | 0.7 |
| current | 4441 | 52.7 | 0.5 |
| anchored | 4444 | 51.3 | -2.1 |