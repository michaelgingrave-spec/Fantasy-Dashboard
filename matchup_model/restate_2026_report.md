# Production-pipeline exam: model versions on the 2026 graded lines

5446 graded lines, weeks 1-11, run through `dfs.props.edges_from_raw` under each version: v1 (original); v2 (WR/TE recalibration); v3 (fitted team volume, full injury boost); v4 (injury redistribution scaled); v5 (current: median correction). Every constant of v1-v3 was fixed from 2023-25 box scores; nothing there was fitted on 2026. v4's injury scales (`INJ_REDIST`) were fitted on 2022-26 box scores, of which 2026 is a small share (`injury_absorption_report.md` lists the fit with each season left out, 2026 included), so v4's lift on these lines is not a clean out-of-sample test. v5's median factors (`median_cal.py`) are learned for each week from the graded lines of the weeks before it, so they never see the week they are applied to, but the rule was chosen after looking at weeks 1-8: week 9 onward is the clean test. ROI is at a flat -110 (prices are not stored per line).

**Harness check** -- v1 (original) reproduces what the screen stored when the lines were pulled: our proj within 0.15 on 50% of rows, same lean 84%, same tier 64%, role_ratio within .02 63% (the rest is injury / data drift since the pull).

## Win rate of the leans

| lines | bets | win | ROI at -110 | OVER share | OVER win | UNDER win |
|---|--:|--:|--:|--:|--:|--:|
| all markets, v1 (original) | 5398 | 49.5% | -5.5% | 75% | 47.7% | 54.8% |
| all markets, v2 (WR/TE recalibration) | 5398 | 51.4% | -1.9% | 63% | 48.8% | 55.9% |
| all markets, v3 (fitted team volume, full injury boost) | 5398 | 51.5% | -1.7% | 63% | 48.9% | 56.0% |
| all markets, v4 (injury redistribution scaled) | 5398 | 51.8% | -1.1% | 65% | 49.1% | 56.7% |
| all markets, v5 (current: median correction) | 5398 | 54.3% | +3.7% | 39% | 51.8% | 55.9% |
| receiving (rec yds + receptions), v1 (original) | 3870 | 49.2% | -6.0% | 79% | 47.6% | 55.7% |
| receiving (rec yds + receptions), v2 (WR/TE recalibration) | 3870 | 51.9% | -0.9% | 63% | 49.0% | 56.7% |
| receiving (rec yds + receptions), v3 (fitted team volume, full injury boost) | 3870 | 51.6% | -1.4% | 63% | 48.8% | 56.4% |
| receiving (rec yds + receptions), v4 (injury redistribution scaled) | 3870 | 52.0% | -0.7% | 65% | 49.2% | 57.3% |
| receiving (rec yds + receptions), v5 (current: median correction) | 3870 | 54.4% | +3.8% | 37% | 51.8% | 55.9% |
| rec yds, v1 (original) | 1957 | 49.7% | -5.1% | 82% | 48.1% | 56.9% |
| rec yds, v2 (WR/TE recalibration) | 1957 | 50.7% | -3.1% | 71% | 48.6% | 56.0% |
| rec yds, v3 (fitted team volume, full injury boost) | 1957 | 50.0% | -4.6% | 75% | 48.1% | 55.5% |
| rec yds, v4 (injury redistribution scaled) | 1957 | 50.3% | -3.9% | 77% | 48.4% | 56.6% |
| rec yds, v5 (current: median correction) | 1957 | 53.4% | +1.9% | 44% | 50.7% | 55.5% |
| receptions, v1 (original) | 1913 | 48.7% | -7.0% | 77% | 46.9% | 54.7% |
| receptions, v2 (WR/TE recalibration) | 1913 | 53.1% | +1.3% | 55% | 49.7% | 57.1% |
| receptions, v3 (fitted team volume, full injury boost) | 1913 | 53.3% | +1.8% | 51% | 49.9% | 56.9% |
| receptions, v4 (injury redistribution scaled) | 1913 | 53.7% | +2.6% | 53% | 50.3% | 57.6% |
| receptions, v5 (current: median correction) | 1913 | 55.4% | +5.8% | 29% | 53.4% | 56.2% |
| rush yds, v1 (original) | 903 | 46.2% | -11.8% | 68% | 43.6% | 51.7% |
| rush yds, v2 (WR/TE recalibration) | 903 | 46.2% | -11.8% | 68% | 43.6% | 51.7% |
| rush yds, v3 (fitted team volume, full injury boost) | 903 | 48.0% | -8.5% | 67% | 44.8% | 54.4% |
| rush yds, v4 (injury redistribution scaled) | 903 | 48.1% | -8.2% | 68% | 44.9% | 54.6% |
| rush yds, v5 (current: median correction) | 903 | 52.8% | +0.8% | 41% | 47.4% | 56.6% |
| pass yds, v1 (original) | 310 | 50.0% | -4.5% | 62% | 53.4% | 44.4% |
| pass yds, v2 (WR/TE recalibration) | 310 | 50.0% | -4.5% | 62% | 53.4% | 44.4% |
| pass yds, v3 (fitted team volume, full injury boost) | 310 | 49.7% | -5.2% | 63% | 53.1% | 44.0% |
| pass yds, v4 (injury redistribution scaled) | 310 | 49.7% | -5.2% | 63% | 53.1% | 44.0% |
| pass yds, v5 (current: median correction) | 310 | 51.9% | -0.9% | 60% | 55.1% | 47.2% |
| pass TD, v1 (original) | 311 | 61.7% | +17.9% | 55% | 58.7% | 65.5% |
| pass TD, v2 (WR/TE recalibration) | 311 | 61.7% | +17.9% | 55% | 58.7% | 65.5% |
| pass TD, v3 (fitted team volume, full injury boost) | 311 | 61.7% | +17.9% | 53% | 59.1% | 64.6% |
| pass TD, v4 (injury redistribution scaled) | 311 | 61.7% | +17.9% | 53% | 59.1% | 64.6% |
| pass TD, v5 (current: median correction) | 311 | 59.5% | +13.6% | 43% | 58.5% | 60.2% |

## Paired difference (v5 (current: median correction) minus each earlier version), same lines, player-clustered 95% CI

- vs v1 (original), all markets: **+4.82 pts** (CI +3.03 to +6.64, n=5398)
- vs v1 (original), receiving markets: **+5.17 pts** (CI +2.89 to +7.51, n=3870)
- vs v1 (original), rush yds: **+6.64 pts** (CI +2.91 to +10.44, n=903)
- vs v2 (WR/TE recalibration), all markets: **+2.91 pts** (CI +1.44 to +4.45, n=5398)
- vs v2 (WR/TE recalibration), receiving markets: **+2.51 pts** (CI +0.51 to +4.42, n=3870)
- vs v2 (WR/TE recalibration), rush yds: **+6.64 pts** (CI +3.12 to +10.34, n=903)
- vs v3 (fitted team volume, full injury boost), all markets: **+2.80 pts** (CI +1.39 to +4.19, n=5398)
- vs v3 (fitted team volume, full injury boost), receiving markets: **+2.76 pts** (CI +0.93 to +4.61, n=3870)
- vs v3 (fitted team volume, full injury boost), rush yds: **+4.87 pts** (CI +1.80 to +8.09, n=903)
- vs v4 (injury redistribution scaled), all markets: **+2.50 pts** (CI +1.12 to +3.98, n=5398)
- vs v4 (injury redistribution scaled), receiving markets: **+2.38 pts** (CI +0.58 to +4.09, n=3870)
- vs v4 (injury redistribution scaled), rush yds: **+4.76 pts** (CI +1.67 to +7.79, n=903)

Win % on receiving markets by week:

| week | v1 (original) | v2 (WR/TE recalibration) | v3 (fitted team volume, full injury boost) | v4 (injury redistribution scaled) | v5 (current: median correction) | lines |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 52.6% | 55.4% | 51.2% | 53.0% | 53.0% | 285 |
| 2 | 47.2% | 46.9% | 49.9% | 50.1% | 50.1% | 346 |
| 3 | 48.2% | 53.2% | 52.6% | 55.2% | 53.8% | 360 |
| 4 | 55.8% | 56.5% | 56.0% | 56.5% | 54.5% | 397 |
| 5 | 47.6% | 50.4% | 50.4% | 50.1% | 53.0% | 359 |
| 6 | 51.0% | 53.2% | 52.4% | 52.7% | 52.7% | 363 |
| 7 | 44.7% | 49.3% | 52.4% | 53.9% | 59.7% | 351 |
| 8 | 48.7% | 51.0% | 51.0% | 49.9% | 57.7% | 359 |
| 9 | 54.1% | 56.3% | 55.2% | 55.7% | 55.7% | 379 |
| 10 | 41.6% | 45.9% | 46.5% | 45.7% | 54.6% | 368 |
| 11 | 49.9% | 52.5% | 49.6% | 49.0% | 53.1% | 345 |

## Where v5 (current: median correction) differs from v4 (injury redistribution scaled)

Lines picked by what the model did, not by how they turned out. `moved` is the change in our projection relative to the typical change for that market (so a recalibration that shifts every line alike is not a move); where the lean flips one version wins and the other loses.

| lines | n | v4 (injury redistribution scaled) win | v5 (current: median correction) win | paired difference (player-clustered 95% CI) | OVER share v4 (injury redistribution scaled) -> v5 (current: median correction) |
|---|--:|--:|--:|---|---|
| all lines | 5446 | 51.8% | 54.3% | +2.50 pts [+1.08, +3.89] (n=5398) | 65% -> 39% |
| projection moved >= 1% | 0 | | | | |
| projection moved >= 5% | 0 | | | | |
| projection moved >= 10% | 0 | | | | |
| lean flipped by v5 (current: median correction) | 1434 | 45.2% | 54.8% | +9.50 pts [+4.13, +14.76] (n=1421) | 99% -> 1% |
| projection moved < 1% | 5446 | 51.8% | 54.3% | +2.50 pts [+1.12, +3.89] (n=5398) | 65% -> 39% |

## Tier tables (all graded lines, win % and ROI at -110)

Break-even win % by tier at the usual price: lean 52.4, solid 53.0, strong 53.7, high 55.4.

**As stored (what the screen showed)**

All bets:

| conf | n | win% | ROI@-110% |
|---|---|---|---|
| — | 2341 | 50.0 | -4.5 |
| lean | 1335 | 53.5 | 2.1 |
| solid | 1053 | 53.1 | 1.4 |
| strong | 563 | 54.2 | 3.5 |
| high | 154 | 50.6 | -3.3 |

Recommended only:

| conf | n | of all | win% | ROI@-110% |
|---|---|---|---|---|
| lean | 880 | 1335 | 53.0 | 1.3 |
| solid | 495 | 1053 | 54.9 | 4.8 |
| strong | 96 | 563 | 51.6 | -1.5 |
| high | 1 | 154 | 100.0 | 90.9 |

**v2 (WR/TE recalibration), restated**

All bets:

| conf | n | win% | ROI@-110% |
|---|---|---|---|
| — | 2433 | 50.5 | -3.6 |
| lean | 1474 | 50.8 | -3.0 |
| solid | 1015 | 53.9 | 2.9 |
| strong | 426 | 52.3 | -0.1 |
| high | 98 | 51.0 | -2.6 |

Recommended only:

| conf | n | of all | win% | ROI@-110% |
|---|---|---|---|---|
| lean | 1053 | 1474 | 50.3 | -4.0 |
| solid | 503 | 1015 | 54.6 | 4.2 |
| strong | 87 | 426 | 43.7 | -16.6 |
| high | 2 | 98 | 50.0 | -4.5 |

**v3 (fitted team volume, full injury boost), restated**

All bets:

| conf | n | win% | ROI@-110% |
|---|---|---|---|
| — | 2556 | 49.9 | -4.8 |
| lean | 1429 | 52.5 | 0.3 |
| solid | 1009 | 51.5 | -1.6 |
| strong | 370 | 59.0 | 12.6 |
| high | 82 | 50.0 | -4.5 |

Recommended only:

| conf | n | of all | win% | ROI@-110% |
|---|---|---|---|---|
| lean | 1047 | 1429 | 53.2 | 1.6 |
| solid | 483 | 1009 | 51.1 | -2.4 |
| strong | 55 | 370 | 56.4 | 7.6 |
| high | 2 | 82 | 100.0 | 90.9 |

**v4 (injury redistribution scaled), restated**

All bets:

| conf | n | win% | ROI@-110% |
|---|---|---|---|
| — | 2582 | 50.2 | -4.2 |
| lean | 1416 | 52.9 | 0.9 |
| solid | 988 | 52.2 | -0.3 |
| strong | 386 | 58.4 | 11.6 |
| high | 74 | 47.3 | -9.7 |

Recommended only:

| conf | n | of all | win% | ROI@-110% |
|---|---|---|---|---|
| lean | 1021 | 1416 | 52.8 | 0.9 |
| solid | 453 | 988 | 51.7 | -1.4 |
| strong | 58 | 386 | 55.2 | 5.3 |
| high | 1 | 74 | 100.0 | 90.9 |

**v5 (current: median correction), restated**

All bets:

| conf | n | win% | ROI@-110% |
|---|---|---|---|
| — | 2426 | 53.1 | 1.4 |
| lean | 1477 | 55.0 | 5.0 |
| solid | 1083 | 56.2 | 7.3 |
| strong | 389 | 54.1 | 3.4 |
| high | 71 | 52.1 | -0.5 |

Recommended only:

| conf | n | of all | win% | ROI@-110% |
|---|---|---|---|---|
| lean | 993 | 1477 | 54.5 | 4.1 |
| solid | 748 | 1083 | 56.9 | 8.6 |
| strong | 230 | 389 | 55.3 | 5.5 |
| high | 15 | 71 | 73.3 | 40.0 |

## Filters and role agreement

- v1 (original): 5446 rows -> 2892 pass the always-on guard -> 1485 Recommended; role_ratio median 0.95, 67% inside the 0.85-1.20 band
- v2 (WR/TE recalibration): 5446 rows -> 2607 pass the always-on guard -> 1645 Recommended; role_ratio median 1.03, 76% inside the 0.85-1.20 band
- v3 (fitted team volume, full injury boost): 5446 rows -> 2495 pass the always-on guard -> 1587 Recommended; role_ratio median 1.02, 77% inside the 0.85-1.20 band
- v4 (injury redistribution scaled): 5446 rows -> 2466 pass the always-on guard -> 1533 Recommended; role_ratio median 1.02, 76% inside the 0.85-1.20 band
- v5 (current: median correction): 5446 rows -> 2685 pass the always-on guard -> 1986 Recommended; role_ratio median 1.02, 76% inside the 0.85-1.20 band

- OVER share by market, v1 (original): pass TD 55%, pass yds 62%, rec yds 82%, receptions 77%, rush yds 68%
- OVER share by market, v2 (WR/TE recalibration): pass TD 55%, pass yds 62%, rec yds 71%, receptions 55%, rush yds 68%
- OVER share by market, v3 (fitted team volume, full injury boost): pass TD 52%, pass yds 62%, rec yds 75%, receptions 51%, rush yds 67%
- OVER share by market, v4 (injury redistribution scaled): pass TD 52%, pass yds 62%, rec yds 77%, receptions 53%, rush yds 68%
- OVER share by market, v5 (current: median correction): pass TD 43%, pass yds 60%, rec yds 44%, receptions 29%, rush yds 41%
