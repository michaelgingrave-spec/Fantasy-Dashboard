# The edge meter: does our projection add anything to the market's own price?

Logistic regression of 'went over' on the book's no-vig P(over) and our edge in SDs (`z`); player-clustered 95% intervals. A +1 SD edge has to move a coin-flip bet's win probability by about 2.4 points just to pay the vig at -110.

| model version | graded lines | went over | market said | market-price coefficient | z coefficient (per SD) | +1 SD moves P(over) by |
|---|--:|--:|--:|--:|--:|--:|
| v1 | 2439 | 48.1% | 49.5% | +0.64 [+0.21, +1.13] | +0.055 [-0.211, +0.348] | +1.4 pts [-5.3, +8.6] |
| v4 | 978 | 46.3% | 49.6% | +1.15 [+0.39, +2.05] | +0.366 [-0.121, +0.968] | +9.0 pts [-3.0, +22.5] |

A market-price coefficient below 1 means prices are more extreme than results (the books' favourites win less often than priced); a z coefficient whose interval spans 0 means the sample cannot yet tell our edge from none.

## By market (all versions pooled)

| market | graded lines | went over | z coefficient (per SD) |
|---|--:|--:|--:|
| pass TD | 205 | 48.8% | +0.712 [-0.477, +1.769] |
| pass yds | 204 | 55.9% | -0.302 [-1.371, +0.390] |
| rec yds | 1230 | 47.4% | +0.207 [-0.125, +0.569] |
| receptions | 1197 | 47.6% | +0.207 [-0.140, +0.615] |
| rush yds | 581 | 44.8% | -0.161 [-0.672, +0.323] |

## Win rate by the side we leaned (as logged)

| side | lines | win | ROI at -110 |
|---|--:|--:|--:|
| OVER | 2366 | 49.1% | -6.3% |
| UNDER | 1051 | 55.7% | +6.3% |
