# RB projection: carry-share shrinkage and constant scales (2023-25; scales learned on the other two seasons)

3523 RB player-weeks (weeks 5-18; 118 are backs on the field with no stat line, true zeros). Scales are learned on the backs the model expects to be involved: >= 6 projected carries for the rushing stats, >= 1.5 projected receptions for the receiving ones. Pooled out-of-season RMSE on those backs:

| stat | backs | original | + scale | + fitted team volume | + carry shrinkage (current) | total change |
|---|--:|--:|--:|--:|--:|--:|
| rush_att | 1883 | 5.738 | 5.739 | 5.511 | 5.385 | -6.2% |
| rush_yds | 1883 | 34.139 | 34.155 | 33.244 | 32.914 | -3.6% |
| rec | 1782 | 1.807 | 1.779 | 1.765 | 1.763 | -2.4% |
| rec_yds | 1782 | 18.054 | 17.976 | 17.921 | 17.906 | -0.8% |

(`+ scale` is a constant learned the same way on the ORIGINAL model's projections, so each step to its right is one change.)

## Why a constant, and who it is learned on

Actual / projected by projected usage under the current constants, before any scale. Lead backs are close to calibrated; committee and depth backs are heavily over-projected (their carry shares were shrunk up toward a 0.42 prior), and a constant is learned on the backs with prop lines.

| projected | backs | with a stat row | actual / projected |
|---|--:|--:|--:|
| carries [0, 2) | 498 | 92% | 0.75 |
| carries [2, 5) | 901 | 93% | 0.89 |
| carries [5, 8) | 624 | 97% | 0.92 |
| carries [8, 12) | 543 | 99% | 0.99 |
| carries [12, 16) | 647 | 100% | 1.02 |
| carries [16, 99) | 310 | 100% | 0.99 |
| receptions [0.0, 0.5) | 41 | 100% | 0.51 |
| receptions [0.5, 1.0) | 826 | 92% | 0.55 |
| receptions [1.0, 2.0) | 1471 | 97% | 0.79 |
| receptions [2.0, 3.0) | 849 | 100% | 0.97 |
| receptions [3.0, 99.0) | 336 | 100% | 0.97 |

## Downstream check: DK fantasy points and anytime-TD probability

The shipped RB model against the original one, 3523 RB player-weeks (2023-25 weeks 5-18, everyone who played). Nothing is learned here.

| backs | n | mean actual DK pts | projected, original | current | RMSE trailing avg | original | current |
|---|--:|--:|--:|--:|--:|--:|--:|
| everyone who played | 3523 | 7.86 | 8.52 | 8.12 | 6.460 | 6.525 | 6.345 |
| trailing carries: trailing-avg DK pts >= 8 | 1483 | 13.53 | 12.71 | 12.93 | 8.297 | 8.210 | 8.090 |

P(at least one TD) = 1 - exp(-(expected rushing + receiving TDs)):

| backs | n | actual rate | mean P, original | current | Brier, original | current | flat average |
|---|--:|--:|--:|--:|--:|--:|--:|
| everyone who played | 3523 | 23.6% | 27.4% | 25.8% | 0.1554 | 0.1522 | 0.1801 |
| trailing-avg DK pts >= 8 | 1483 | 40.9% | 39.2% | 40.2% | 0.2323 | 0.2313 | 0.2417 |

## Constants for model.py (learned on all three seasons)

```python
K_CARRY = 1.0
RB_CARRY_PRIOR = 0.25
RB_YDS_CATCH_FACTOR = False
RB_SCALE = {"rush_att": 0.989, "rush_yds": 0.993, "rec": 0.946, "rec_yds": 0.902, "td": 1.047}
```
