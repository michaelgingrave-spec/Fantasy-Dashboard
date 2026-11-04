# Stat-line accuracy by FantasyPoints-deference variant — 2026 weeks 1-8

## Part A — broad accuracy (every player with real usage, RMSE/MAE)

**rec_yds**

| variant | n | RMSE | MAE |
|---|--:|--:|--:|
| current | 1888 | 25.15 | 17.8 |
| anchored | 1888 | 25.23 | 17.89 |
| no_shift | 1888 | 25.7 | 18.38 |

**rec**

| variant | n | RMSE | MAE |
|---|--:|--:|--:|
| current | 1888 | 1.83 | 1.35 |
| anchored | 1888 | 1.86 | 1.38 |
| no_shift | 1888 | 1.87 | 1.38 |

**rush_yds**

| variant | n | RMSE | MAE |
|---|--:|--:|--:|
| current | 808 | 25.7 | 18.0 |
| anchored | 808 | 25.88 | 17.9 |
| no_shift | 808 | 26.24 | 18.54 |

**pass_yds**

| variant | n | RMSE | MAE |
|---|--:|--:|--:|
| current | 253 | 84.08 | 65.88 |
| no_shift | 253 | 84.27 | 66.11 |
| anchored | 253 | 86.74 | 68.13 |

**pass_td**

| variant | n | RMSE | MAE |
|---|--:|--:|--:|
| no_shift | 253 | 1.09 | 0.88 |
| current | 253 | 1.09 | 0.88 |
| anchored | 253 | 1.09 | 0.88 |

## Part B — would it have won (real pulled lines, real results)

- 3931 props with a real book line per variant

| variant | n | win% | ROI@-110% |
|---|--:|--:|--:|
| no_shift | 3931 | 52.4 | 0.1 |
| current | 3931 | 52.4 | 0.1 |
| anchored | 3933 | 51.2 | -2.2 |