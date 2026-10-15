# RB projection: carry-share shrinkage and constant scales (2023-25; scales learned on the other two seasons)

3523 RB player-weeks (weeks 5-18; 118 are backs on the field with no stat line, true zeros). Scales are learned on the backs the model expects to be involved: >= 6 projected carries for the rushing stats, >= 1.5 projected receptions for the receiving ones. Pooled out-of-season RMSE on those backs:

| stat | backs | original | + scale | + fitted team volume | + carry shrinkage (current) | total change |
|---|--:|--:|--:|--:|--:|--:|
| rush_att | 1915 | 5.738 | 5.739 | 5.511 | 5.418 | -5.6% |
| rush_yds | 1915 | 34.093 | 34.109 | 33.199 | 32.921 | -3.4% |
| rec | 1835 | 1.805 | 1.779 | 1.766 | 1.766 | -2.2% |
| rec_yds | 1835 | 18.152 | 18.069 | 18.009 | 18.009 | -0.8% |

(`+ scale` is a constant learned the same way on the ORIGINAL model's projections, so each step to its right is one change.)

## Why a constant, and who it is learned on

Actual / projected by projected usage under the current constants, before any scale. Lead backs are close to calibrated; committee and depth backs are heavily over-projected (their carry shares were shrunk up toward a 0.42 prior), and a constant is learned on the backs with prop lines.

| projected | backs | with a stat row | actual / projected |
|---|--:|--:|--:|
| carries [0, 2) | 494 | 92% | 0.73 |
| carries [2, 5) | 882 | 93% | 0.88 |
| carries [5, 8) | 617 | 97% | 0.89 |
| carries [8, 12) | 546 | 99% | 0.98 |
| carries [12, 16) | 643 | 100% | 0.99 |
| carries [16, 99) | 341 | 100% | 0.96 |
| receptions [0.0, 0.5) | 41 | 100% | 0.51 |
| receptions [0.5, 1.0) | 823 | 92% | 0.55 |
| receptions [1.0, 2.0) | 1381 | 97% | 0.75 |
| receptions [2.0, 3.0) | 859 | 99% | 0.94 |
| receptions [3.0, 99.0) | 419 | 100% | 0.92 |

## Downstream check: DK fantasy points and anytime-TD probability

The shipped RB model against the original one, 3523 RB player-weeks (2023-25 weeks 5-18, everyone who played). Nothing is learned here.

| backs | n | mean actual DK pts | projected, original | current | RMSE trailing avg | original | current |
|---|--:|--:|--:|--:|--:|--:|--:|
| everyone who played | 3523 | 7.86 | 8.52 | 8.10 | 6.460 | 6.525 | 6.334 |
| trailing carries: trailing-avg DK pts >= 8 | 1483 | 13.53 | 12.71 | 12.92 | 8.297 | 8.210 | 8.091 |

P(at least one TD) = 1 - exp(-(expected rushing + receiving TDs)):

| backs | n | actual rate | mean P, original | current | Brier, original | current | flat average |
|---|--:|--:|--:|--:|--:|--:|--:|
| everyone who played | 3523 | 23.6% | 27.4% | 25.7% | 0.1554 | 0.1520 | 0.1801 |
| trailing-avg DK pts >= 8 | 1483 | 40.9% | 39.2% | 39.9% | 0.2323 | 0.2314 | 0.2417 |

## Constants for model.py (learned on all three seasons)

```python
K_CARRY = 1.0
RB_CARRY_PRIOR = 0.25
RB_YDS_CATCH_FACTOR = False
RB_SCALE = {"rush_att": 0.969, "rush_yds": 0.973, "rec": 0.911, "rec_yds": 0.872, "td": 1.017}
```
