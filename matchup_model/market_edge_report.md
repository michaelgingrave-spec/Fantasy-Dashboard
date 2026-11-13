# The edge meter: does our projection add anything to the market's own price?

Logistic regression of 'went over' on the book's no-vig P(over) and our edge in SDs (`z`); player-clustered 95% intervals. A +1 SD edge has to move a coin-flip bet's win probability by about 2.4 points just to pay the vig at -110.

| model version | graded lines | went over | market said | market-price coefficient | z coefficient (per SD) | +1 SD moves P(over) by |
|---|--:|--:|--:|--:|--:|--:|
| v1 | 2439 | 48.1% | 49.5% | +0.64 [+0.21, +1.13] | +0.055 [-0.211, +0.348] | +1.4 pts [-5.3, +8.6] |
| v4 | 1992 | 46.2% | 49.4% | +0.99 [+0.44, +1.59] | +0.358 [-0.069, +0.867] | +8.9 pts [-1.7, +20.4] |

A market-price coefficient below 1 means prices are more extreme than results (the books' favourites win less often than priced); a z coefficient whose interval spans 0 means the sample cannot yet tell our edge from none.

## By market (all versions pooled)

| market | graded lines | went over | z coefficient (per SD) |
|---|--:|--:|--:|
| pass TD | 262 | 47.3% | +0.672 [-0.338, +1.385] |
| pass yds | 261 | 54.0% | -0.137 [-0.992, +0.493] |
| rec yds | 1600 | 47.3% | +0.275 [-0.044, +0.710] |
| receptions | 1559 | 47.3% | +0.299 [-0.077, +0.660] |
| rush yds | 749 | 44.7% | -0.329 [-0.712, +0.128] |

## Win rate by the side we leaned (as logged)

| side | lines | win | ROI at -110 |
|---|--:|--:|--:|
| OVER | 3032 | 49.0% | -6.5% |
| UNDER | 1399 | 56.4% | +7.7% |
