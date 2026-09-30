# Stat-line accuracy by FantasyPoints-deference variant — 2026 weeks 1-3

## Part A — broad accuracy (every player with real usage, RMSE/MAE)

**rec_yds**

| variant | n | RMSE | MAE |
|---|--:|--:|--:|
| anchored | 718 | 25.09 | 17.71 |
| current | 718 | 25.3 | 18.24 |
| no_shift | 718 | 26.48 | 19.56 |

**rec**

| variant | n | RMSE | MAE |
|---|--:|--:|--:|
| current | 718 | 1.81 | 1.41 |
| anchored | 718 | 1.83 | 1.4 |
| no_shift | 718 | 1.9 | 1.51 |

**rush_yds**

| variant | n | RMSE | MAE |
|---|--:|--:|--:|
| anchored | 307 | 24.96 | 17.98 |
| current | 307 | 25.02 | 18.31 |
| no_shift | 307 | 26.03 | 19.24 |

**pass_yds**

| variant | n | RMSE | MAE |
|---|--:|--:|--:|
| no_shift | 100 | 82.59 | 65.26 |
| current | 100 | 82.6 | 65.26 |
| anchored | 100 | 86.2 | 66.61 |

**pass_td**

| variant | n | RMSE | MAE |
|---|--:|--:|--:|
| no_shift | 100 | 1.2 | 0.96 |
| current | 100 | 1.2 | 0.96 |
| anchored | 100 | 1.2 | 0.96 |

## Part B — would it have won (real pulled lines, real results)

- 1371 props with a real book line per variant

| variant | n | win% | ROI@-110% |
|---|--:|--:|--:|
| no_shift | 1371 | 50.1 | -4.3 |
| current | 1371 | 50.3 | -3.9 |
| anchored | 1371 | 50.8 | -3.1 |