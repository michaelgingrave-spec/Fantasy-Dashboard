# Stat-line accuracy by FantasyPoints-deference variant — 2026 weeks 1-7

## Part A — broad accuracy (every player with real usage, RMSE/MAE)

**rec_yds**

| variant | n | RMSE | MAE |
|---|--:|--:|--:|
| current | 1671 | 24.94 | 17.76 |
| anchored | 1671 | 25.04 | 17.86 |
| no_shift | 1671 | 25.47 | 18.31 |

**rec**

| variant | n | RMSE | MAE |
|---|--:|--:|--:|
| current | 1671 | 1.83 | 1.34 |
| anchored | 1671 | 1.86 | 1.38 |
| no_shift | 1671 | 1.87 | 1.37 |

**rush_yds**

| variant | n | RMSE | MAE |
|---|--:|--:|--:|
| current | 709 | 24.93 | 17.65 |
| anchored | 709 | 25.17 | 17.6 |
| no_shift | 709 | 25.48 | 18.2 |

**pass_yds**

| variant | n | RMSE | MAE |
|---|--:|--:|--:|
| current | 221 | 80.17 | 63.17 |
| no_shift | 221 | 80.25 | 63.27 |
| anchored | 221 | 83.05 | 65.72 |

**pass_td**

| variant | n | RMSE | MAE |
|---|--:|--:|--:|
| no_shift | 221 | 1.09 | 0.86 |
| current | 221 | 1.09 | 0.86 |
| anchored | 221 | 1.09 | 0.87 |

## Part B — would it have won (real pulled lines, real results)

- 3432 props with a real book line per variant

| variant | n | win% | ROI@-110% |
|---|--:|--:|--:|
| no_shift | 3432 | 52.6 | 0.4 |
| current | 3432 | 52.8 | 0.7 |
| anchored | 3434 | 51.3 | -2.0 |