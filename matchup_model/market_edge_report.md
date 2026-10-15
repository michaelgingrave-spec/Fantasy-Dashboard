# The edge meter: does our projection add anything to the market's own price?

Logistic regression of 'went over' on the book's no-vig P(over) and our edge in SDs (`z`); player-clustered 95% intervals. A +1 SD edge has to move a coin-flip bet's win probability by about 2.4 points just to pay the vig at -110.

| model version | graded lines | went over | market said | market-price coefficient | z coefficient (per SD) | +1 SD moves P(over) by |
|---|--:|--:|--:|--:|--:|--:|
| v1 | 2439 | 48.1% | 49.5% | +0.64 [+0.21, +1.13] | +0.055 [-0.211, +0.348] | +1.4 pts [-5.3, +8.6] |

A market-price coefficient below 1 means prices are more extreme than results (the books' favourites win less often than priced); a z coefficient whose interval spans 0 means the sample cannot yet tell our edge from none.

## By market (all versions pooled)

| market | graded lines | went over | z coefficient (per SD) |
|---|--:|--:|--:|
| pass TD | 150 | 50.7% | +0.501 [-1.026, +1.683] |
| pass yds | 150 | 55.3% | -0.114 [-1.267, +0.740] |
| rec yds | 874 | 48.2% | +0.133 [-0.183, +0.516] |
| receptions | 851 | 47.8% | +0.114 [-0.256, +0.505] |
| rush yds | 414 | 45.2% | -0.212 [-0.718, +0.223] |

## Win rate by the side we leaned (as logged)

| side | lines | win | ROI at -110 |
|---|--:|--:|--:|
| OVER | 1710 | 48.7% | -7.0% |
| UNDER | 729 | 53.2% | +1.6% |
