# Do our projections lead the market? Open vs late lines, 2026

933 graded receiving props with both an opening line (Wednesday 4pm ET snapshot) and a late line (weeks 3+ closing, weeks 1-2 pulled mid-week); weeks 1-5, {1: 253, 2: 152, 3: 124, 4: 224, 5: 180} rows per week. `our proj` is what the dashboard showed when the line was pulled.

## 1. How far do lines move?

| market | n | moved | mean move | mean abs move | in SD |
|---|--:|--:|--:|--:|--:|
| rec yds | 468 | 86% | +0.75 (up 53% / down 33%) | 2.00 | 0.070 |
| receptions | 465 | 21% | +0.05 (up 13% / down 8%) | 0.15 | 0.073 |

## 2. Does the market move toward our number?

Slope of the open-to-late move on (our projection - open line), both in outcome SDs. 0 = no relation, 1 = the market fully follows us. 95% CI from a player-clustered bootstrap.

| rows | n | slope | 95% CI |
|---|--:|--:|---|
| all | 933 | +0.096 | [+0.051, +0.148] |
| rec yds | 468 | +0.066 | [+0.025, +0.113] |
| receptions | 465 | +0.124 | [+0.072, +0.179] |
| weeks 3+ (true closing lines) | 528 | +0.110 | [+0.037, +0.182] |
| weeks 1-2 (mid-week lines) | 405 | +0.085 | [+0.033, +0.146] |

Large disagreements (|our projection - open| >= 0.4 SD, n=294): the line moved toward us 39% of the time, away from us 17%, unchanged 44%.

## 3. Our side at the open vs at the late line

Bet the side our projection implies against each line, same rows, same projection (win rate, ROI at -110).

| line used | bets | win % | OVER share | ROI at -110 |
|---|--:|--:|--:|--:|
| open | 924 | 51.8 | 70% | -1.0% |
| late | 922 | 51.2 | 66% | -2.3% |

Paired difference, open minus late: **+0.7 pts** (95% CI -1.4 to +2.7, n=915). A CI that spans zero means betting early is not distinguishable from betting late on this sample.

| week | win % at open | win % at late | n |
|--:|--:|--:|--:|
| 1 | 51.8 | 51.8 | 253 |
| 2 | 53.3 | 50.0 | 152 |
| 3 | 48.4 | 43.9 | 124 |
| 4 | 56.1 | 57.9 | 224 |
| 5 | 47.7 | 48.0 | 180 |

## 4. Which line forecasts the result better?

RMSE of (actual - line) in outcome SDs (lower is better); our projection for comparison.

| forecast | RMSE | mean error |
|---|--:|--:|
| open line | 1.1207 | +0.171 |
| late line | 1.1082 | +0.145 |
| our projection | 1.1240 | -0.037 |

Price check: median best open price on our side -113; 33% of open prices are -110 or better.
