# The edge meter: does our projection add anything to the market's own price?

Logistic regression of 'went over' on the book's no-vig P(over) and our edge in SDs (`z`); player-clustered 95% intervals. A +1 SD edge has to move a coin-flip bet's win probability by about 2.4 points just to pay the vig at -110.

| model version | graded lines | went over | market said | market-price coefficient | z coefficient (per SD) | +1 SD moves P(over) by |
|---|--:|--:|--:|--:|--:|--:|
| v1 | 2439 | 48.1% | 49.5% | +0.64 [+0.21, +1.13] | +0.055 [-0.211, +0.348] | +1.4 pts [-5.3, +8.6] |
| v4 | 1479 | 46.2% | 49.5% | +1.19 [+0.51, +1.81] | +0.201 [-0.216, +0.688] | +5.0 pts [-5.4, +16.6] |

A market-price coefficient below 1 means prices are more extreme than results (the books' favourites win less often than priced); a z coefficient whose interval spans 0 means the sample cannot yet tell our edge from none.

## By market (all versions pooled)

| market | graded lines | went over | z coefficient (per SD) |
|---|--:|--:|--:|
| pass TD | 233 | 48.9% | +0.591 [-0.501, +1.446] |
| pass yds | 232 | 55.2% | -0.250 [-1.170, +0.443] |
| rec yds | 1410 | 47.3% | +0.225 [-0.048, +0.622] |
| receptions | 1374 | 47.4% | +0.222 [-0.122, +0.602] |
| rush yds | 669 | 44.5% | -0.314 [-0.811, +0.124] |

## Win rate by the side we leaned (as logged)

| side | lines | win | ROI at -110 |
|---|--:|--:|--:|
| OVER | 2691 | 48.8% | -6.9% |
| UNDER | 1227 | 55.6% | +6.1% |
