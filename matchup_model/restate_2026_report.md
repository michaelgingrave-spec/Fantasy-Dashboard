# Production-pipeline exam: previous projection constants vs the recalibrated ones, 2026 graded lines

2468 graded lines, weeks 1-5, run through `dfs.props.edges_from_raw` under each set of constants. The recalibrated constants were fixed from 2023-25 box scores; nothing was fitted on 2026. ROI is at a flat -110 (prices are not stored per line).

**Harness check** -- the previous constants reproduce what the screen stored when the lines were pulled: our proj within 0.15 on 89% of rows, same lean 98%, same tier 94%, role_ratio within .02 98% (the rest is injury / data drift since the pull).

## Win rate of the leans

| lines | bets | win | ROI at -110 | OVER share | OVER win | UNDER win |
|---|--:|--:|--:|--:|--:|--:|
| all markets, previous | 2443 | 50.3% | -4.0% | 71% | 48.9% | 53.7% |
| all markets, recalibrated | 2443 | 51.8% | -1.1% | 59% | 50.0% | 54.5% |
| receiving (rec yds + receptions), previous | 1725 | 50.3% | -3.9% | 75% | 48.9% | 54.7% |
| receiving (rec yds + receptions), recalibrated | 1725 | 52.5% | +0.3% | 58% | 50.4% | 55.4% |
| rec yds, previous | 874 | 51.1% | -2.4% | 77% | 49.6% | 56.6% |
| rec yds, recalibrated | 874 | 51.8% | -1.1% | 66% | 50.0% | 55.3% |
| receptions, previous | 851 | 49.5% | -5.6% | 73% | 48.2% | 53.1% |
| receptions, recalibrated | 851 | 53.2% | +1.6% | 50% | 51.0% | 55.5% |
| rush yds, previous | 414 | 45.4% | -13.3% | 67% | 43.0% | 50.4% |
| rush yds, recalibrated | 414 | 45.4% | -13.3% | 67% | 43.0% | 50.4% |
| pass yds, previous | 150 | 51.3% | -2.0% | 59% | 55.7% | 45.2% |
| pass yds, recalibrated | 150 | 51.3% | -2.0% | 59% | 55.7% | 45.2% |
| pass TD, previous | 150 | 62.0% | +18.4% | 53% | 62.0% | 62.0% |
| pass TD, recalibrated | 150 | 62.0% | +18.4% | 53% | 62.0% | 62.0% |

## Paired difference (recalibrated minus previous), same lines, player-clustered 95% CI

- all markets: **+1.56 pts** (CI +0.08 to +2.98, n=2443)
- receiving markets: **+2.20 pts** (CI +0.18 to +4.21, n=1725)

| week | receiving: previous | recalibrated | lines |
|--:|--:|--:|--:|
| 1 | 52.6% | 55.4% | 285 |
| 2 | 47.2% | 46.9% | 339 |
| 3 | 48.2% | 53.2% | 359 |
| 4 | 55.8% | 56.5% | 391 |
| 5 | 47.6% | 50.4% | 351 |

## Tier tables (all graded lines, win % and ROI at -110)

Break-even win % by tier at the usual price: lean 52.4, solid 53.0, strong 53.7, high 55.4.

**As stored (what the screen showed)**

All bets:

| conf | n | win% | ROI@-110% |
|---|---|---|---|
| — | 922 | 47.0 | -10.3 |
| lean | 549 | 51.7 | -1.2 |
| solid | 510 | 51.6 | -1.5 |
| strong | 370 | 52.2 | -0.4 |
| high | 117 | 53.0 | 1.2 |

Recommended only:

| conf | n | of all | win% | ROI@-110% |
|---|---|---|---|---|
| lean | 322 | 549 | 50.5 | -3.6 |
| solid | 218 | 510 | 53.0 | 1.2 |
| strong | 46 | 370 | 50.0 | -4.5 |
| high | 1 | 117 | 100.0 | 90.9 |

**Previous constants, restated**

All bets:

| conf | n | win% | ROI@-110% |
|---|---|---|---|
| — | 946 | 47.7 | -8.9 |
| lean | 547 | 51.4 | -1.9 |
| solid | 522 | 51.8 | -1.2 |
| strong | 358 | 52.5 | 0.3 |
| high | 95 | 52.6 | 0.5 |

Recommended only:

| conf | n | of all | win% | ROI@-110% |
|---|---|---|---|---|
| lean | 377 | 547 | 50.8 | -3.0 |
| solid | 267 | 522 | 53.0 | 1.2 |
| strong | 59 | 358 | 57.6 | 10.0 |
| high | 1 | 95 | 100.0 | 90.9 |

**Recalibrated constants**

All bets:

| conf | n | win% | ROI@-110% |
|---|---|---|---|
| — | 1131 | 50.4 | -3.8 |
| lean | 664 | 52.2 | -0.3 |
| solid | 455 | 53.7 | 2.4 |
| strong | 175 | 53.7 | 2.5 |
| high | 43 | 55.8 | 6.6 |

Recommended only:

| conf | n | of all | win% | ROI@-110% |
|---|---|---|---|---|
| lean | 466 | 664 | 51.5 | -1.7 |
| solid | 244 | 455 | 56.1 | 7.2 |
| strong | 47 | 175 | 44.7 | -14.7 |
| high | 1 | 43 | 100.0 | 90.9 |

## Filters and role agreement

- previous: 2468 rows -> 1300 pass the always-on guard -> 704 Recommended; role_ratio median 0.97, 68% inside the 0.85-1.20 band
- recalibrated: 2468 rows -> 1180 pass the always-on guard -> 758 Recommended; role_ratio median 1.04, 75% inside the 0.85-1.20 band

- OVER share by market, previous: pass TD 52%, pass yds 58%, rec yds 77%, receptions 73%, rush yds 67%
- OVER share by market, recalibrated: pass TD 52%, pass yds 58%, rec yds 66%, receptions 51%, rush yds 67%
