# Production-pipeline exam: model versions on the 2026 graded lines

4979 graded lines, weeks 1-10, run through `dfs.props.edges_from_raw` under each version: v1 (original); v2 (WR/TE recalibration); v3 (fitted team volume, full injury boost); v4 (injury redistribution scaled); v5 (current: median correction). Every constant of v1-v3 was fixed from 2023-25 box scores; nothing there was fitted on 2026. v4's injury scales (`INJ_REDIST`) were fitted on 2022-26 box scores, of which 2026 is a small share (`injury_absorption_report.md` lists the fit with each season left out, 2026 included), so v4's lift on these lines is not a clean out-of-sample test. v5's median factors (`median_cal.py`) are learned for each week from the graded lines of the weeks before it, so they never see the week they are applied to, but the rule was chosen after looking at weeks 1-8: week 9 onward is the clean test. ROI is at a flat -110 (prices are not stored per line).

**Harness check** -- v1 (original) reproduces what the screen stored when the lines were pulled: our proj within 0.15 on 53% of rows, same lean 87%, same tier 67%, role_ratio within .02 65% (the rest is injury / data drift since the pull).

## Win rate of the leans

| lines | bets | win | ROI at -110 | OVER share | OVER win | UNDER win |
|---|--:|--:|--:|--:|--:|--:|
| all markets, v1 (original) | 4933 | 49.3% | -6.0% | 75% | 47.5% | 54.3% |
| all markets, v2 (WR/TE recalibration) | 4933 | 51.2% | -2.3% | 63% | 48.6% | 55.5% |
| all markets, v3 (fitted team volume, full injury boost) | 4933 | 51.6% | -1.5% | 63% | 48.9% | 56.1% |
| all markets, v4 (injury redistribution scaled) | 4933 | 52.0% | -0.8% | 64% | 49.3% | 56.8% |
| all markets, v5 (current: median correction) | 4933 | 54.3% | +3.8% | 39% | 51.8% | 56.0% |
| receiving (rec yds + receptions), v1 (original) | 3527 | 49.2% | -6.1% | 79% | 47.6% | 55.2% |
| receiving (rec yds + receptions), v2 (WR/TE recalibration) | 3527 | 51.8% | -1.1% | 62% | 49.0% | 56.4% |
| receiving (rec yds + receptions), v3 (fitted team volume, full injury boost) | 3527 | 51.8% | -1.1% | 63% | 49.0% | 56.5% |
| receiving (rec yds + receptions), v4 (injury redistribution scaled) | 3527 | 52.3% | -0.1% | 64% | 49.5% | 57.5% |
| receiving (rec yds + receptions), v5 (current: median correction) | 3527 | 54.6% | +4.2% | 37% | 52.1% | 56.0% |
| rec yds, v1 (original) | 1784 | 49.6% | -5.3% | 81% | 48.1% | 56.2% |
| rec yds, v2 (WR/TE recalibration) | 1784 | 50.7% | -3.2% | 70% | 48.6% | 55.8% |
| rec yds, v3 (fitted team volume, full injury boost) | 1784 | 50.1% | -4.3% | 74% | 48.2% | 55.5% |
| rec yds, v4 (injury redistribution scaled) | 1784 | 50.7% | -3.2% | 76% | 48.7% | 57.1% |
| rec yds, v5 (current: median correction) | 1784 | 53.6% | +2.3% | 44% | 51.0% | 55.7% |
| receptions, v1 (original) | 1743 | 48.7% | -7.0% | 77% | 47.0% | 54.3% |
| receptions, v2 (WR/TE recalibration) | 1743 | 53.0% | +1.1% | 54% | 49.7% | 56.8% |
| receptions, v3 (fitted team volume, full injury boost) | 1743 | 53.6% | +2.3% | 51% | 50.3% | 57.0% |
| receptions, v4 (injury redistribution scaled) | 1743 | 54.0% | +3.1% | 53% | 50.7% | 57.7% |
| receptions, v5 (current: median correction) | 1743 | 55.6% | +6.1% | 29% | 53.9% | 56.3% |
| rush yds, v1 (original) | 829 | 45.1% | -13.9% | 68% | 42.6% | 50.6% |
| rush yds, v2 (WR/TE recalibration) | 829 | 45.1% | -13.9% | 68% | 42.6% | 50.6% |
| rush yds, v3 (fitted team volume, full injury boost) | 829 | 47.4% | -9.5% | 67% | 44.1% | 54.0% |
| rush yds, v4 (injury redistribution scaled) | 829 | 47.5% | -9.3% | 67% | 44.3% | 54.2% |
| rush yds, v5 (current: median correction) | 829 | 52.4% | -0.1% | 41% | 46.5% | 56.5% |
| pass yds, v1 (original) | 286 | 49.7% | -5.2% | 62% | 53.1% | 44.0% |
| pass yds, v2 (WR/TE recalibration) | 286 | 49.7% | -5.2% | 62% | 53.1% | 44.0% |
| pass yds, v3 (fitted team volume, full injury boost) | 286 | 49.7% | -5.2% | 62% | 53.1% | 44.0% |
| pass yds, v4 (injury redistribution scaled) | 286 | 49.7% | -5.2% | 62% | 53.1% | 44.0% |
| pass yds, v5 (current: median correction) | 286 | 52.1% | -0.5% | 59% | 55.3% | 47.4% |
| pass TD, v1 (original) | 287 | 62.0% | +18.4% | 55% | 58.9% | 65.9% |
| pass TD, v2 (WR/TE recalibration) | 287 | 62.0% | +18.4% | 55% | 58.9% | 65.9% |
| pass TD, v3 (fitted team volume, full injury boost) | 287 | 62.4% | +19.1% | 53% | 59.6% | 65.4% |
| pass TD, v4 (injury redistribution scaled) | 287 | 62.4% | +19.1% | 53% | 59.6% | 65.4% |
| pass TD, v5 (current: median correction) | 287 | 59.2% | +13.1% | 43% | 58.1% | 60.1% |

## Paired difference (v5 (current: median correction) minus each earlier version), same lines, player-clustered 95% CI

- vs v1 (original), all markets: **+5.09 pts** (CI +3.21 to +6.97, n=4933)
- vs v1 (original), receiving markets: **+5.42 pts** (CI +3.10 to +8.03, n=3527)
- vs v1 (original), rush yds: **+7.24 pts** (CI +3.18 to +11.08, n=829)
- vs v2 (WR/TE recalibration), all markets: **+3.18 pts** (CI +1.66 to +4.81, n=4933)
- vs v2 (WR/TE recalibration), receiving markets: **+2.75 pts** (CI +0.86 to +4.76, n=3527)
- vs v2 (WR/TE recalibration), rush yds: **+7.24 pts** (CI +3.46 to +11.16, n=829)
- vs v3 (fitted team volume, full injury boost), all markets: **+2.76 pts** (CI +1.38 to +4.25, n=4933)
- vs v3 (fitted team volume, full injury boost), receiving markets: **+2.75 pts** (CI +0.85 to +4.65, n=3527)
- vs v3 (fitted team volume, full injury boost), rush yds: **+4.95 pts** (CI +1.92 to +8.06, n=829)
- vs v4 (injury redistribution scaled), all markets: **+2.37 pts** (CI +0.97 to +3.85, n=4933)
- vs v4 (injury redistribution scaled), receiving markets: **+2.24 pts** (CI +0.36 to +4.15, n=3527)
- vs v4 (injury redistribution scaled), rush yds: **+4.83 pts** (CI +1.68 to +7.83, n=829)

Win % on receiving markets by week:

| week | v1 (original) | v2 (WR/TE recalibration) | v3 (fitted team volume, full injury boost) | v4 (injury redistribution scaled) | v5 (current: median correction) | lines |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 52.6% | 55.4% | 51.2% | 53.0% | 53.0% | 285 |
| 2 | 47.2% | 46.9% | 49.9% | 50.4% | 50.4% | 346 |
| 3 | 48.2% | 53.2% | 52.6% | 55.2% | 53.8% | 360 |
| 4 | 55.8% | 56.5% | 56.0% | 56.5% | 54.5% | 397 |
| 5 | 47.6% | 50.4% | 50.4% | 50.1% | 53.0% | 359 |
| 6 | 51.0% | 53.2% | 52.4% | 52.7% | 53.0% | 363 |
| 7 | 44.7% | 49.3% | 52.4% | 53.9% | 59.7% | 351 |
| 8 | 48.7% | 51.0% | 51.0% | 49.9% | 57.7% | 359 |
| 9 | 54.1% | 56.3% | 55.2% | 55.7% | 55.7% | 379 |
| 10 | 41.6% | 45.9% | 46.5% | 45.7% | 54.6% | 368 |

## Where v5 (current: median correction) differs from v4 (injury redistribution scaled)

Lines picked by what the model did, not by how they turned out. `moved` is the change in our projection relative to the typical change for that market (so a recalibration that shifts every line alike is not a move); where the lean flips one version wins and the other loses.

| lines | n | v4 (injury redistribution scaled) win | v5 (current: median correction) win | paired difference (player-clustered 95% CI) | OVER share v4 (injury redistribution scaled) -> v5 (current: median correction) |
|---|--:|--:|--:|---|---|
| all lines | 4979 | 52.0% | 54.3% | +2.37 pts [+0.95, +3.81] (n=4933) | 64% -> 39% |
| projection moved >= 1% | 0 | | | | |
| projection moved >= 5% | 0 | | | | |
| projection moved >= 10% | 0 | | | | |
| lean flipped by v5 (current: median correction) | 1272 | 45.4% | 54.6% | +9.29 pts [+3.58, +15.00] (n=1259) | 99% -> 1% |
| projection moved < 1% | 4979 | 52.0% | 54.3% | +2.37 pts [+0.98, +3.72] (n=4933) | 64% -> 39% |

## Tier tables (all graded lines, win % and ROI at -110)

Break-even win % by tier at the usual price: lean 52.4, solid 53.0, strong 53.7, high 55.4.

**As stored (what the screen showed)**

All bets:

| conf | n | win% | ROI@-110% |
|---|---|---|---|
| — | 2114 | 49.5 | -5.5 |
| lean | 1214 | 53.5 | 2.1 |
| solid | 962 | 52.8 | 0.9 |
| strong | 538 | 54.4 | 3.8 |
| high | 151 | 51.7 | -1.4 |

Recommended only:

| conf | n | of all | win% | ROI@-110% |
|---|---|---|---|---|
| lean | 794 | 1214 | 53.3 | 1.7 |
| solid | 434 | 962 | 55.2 | 5.4 |
| strong | 84 | 538 | 50.0 | -4.5 |
| high | 1 | 151 | 100.0 | 90.9 |

**v2 (WR/TE recalibration), restated**

All bets:

| conf | n | win% | ROI@-110% |
|---|---|---|---|
| — | 2240 | 50.2 | -4.1 |
| lean | 1343 | 50.5 | -3.5 |
| solid | 923 | 53.7 | 2.4 |
| strong | 381 | 52.5 | 0.2 |
| high | 92 | 52.2 | -0.3 |

Recommended only:

| conf | n | of all | win% | ROI@-110% |
|---|---|---|---|---|
| lean | 958 | 1343 | 50.2 | -4.1 |
| solid | 460 | 923 | 54.2 | 3.6 |
| strong | 79 | 381 | 43.0 | -17.8 |
| high | 2 | 92 | 50.0 | -4.5 |

**v3 (fitted team volume, full injury boost), restated**

All bets:

| conf | n | win% | ROI@-110% |
|---|---|---|---|
| — | 2336 | 49.9 | -4.8 |
| lean | 1306 | 52.6 | 0.3 |
| solid | 929 | 52.1 | -0.5 |
| strong | 332 | 58.2 | 11.1 |
| high | 76 | 52.6 | 0.5 |

Recommended only:

| conf | n | of all | win% | ROI@-110% |
|---|---|---|---|---|
| lean | 953 | 1306 | 53.7 | 2.5 |
| solid | 450 | 929 | 51.6 | -1.6 |
| strong | 50 | 332 | 58.0 | 10.7 |
| high | 2 | 76 | 100.0 | 90.9 |

**v4 (injury redistribution scaled), restated**

All bets:

| conf | n | win% | ROI@-110% |
|---|---|---|---|
| — | 2358 | 50.2 | -4.2 |
| lean | 1298 | 52.9 | 1.0 |
| solid | 909 | 53.4 | 1.9 |
| strong | 346 | 57.4 | 9.6 |
| high | 68 | 50.0 | -4.5 |

Recommended only:

| conf | n | of all | win% | ROI@-110% |
|---|---|---|---|---|
| lean | 934 | 1298 | 53.1 | 1.5 |
| solid | 419 | 909 | 52.7 | 0.7 |
| strong | 53 | 346 | 54.7 | 4.5 |
| high | 1 | 68 | 100.0 | 90.9 |

**v5 (current: median correction), restated**

All bets:

| conf | n | win% | ROI@-110% |
|---|---|---|---|
| — | 2199 | 53.0 | 1.2 |
| lean | 1356 | 55.1 | 5.3 |
| solid | 993 | 56.3 | 7.4 |
| strong | 363 | 54.3 | 3.7 |
| high | 68 | 54.4 | 3.9 |

Recommended only:

| conf | n | of all | win% | ROI@-110% |
|---|---|---|---|---|
| lean | 907 | 1356 | 54.8 | 4.7 |
| solid | 688 | 993 | 57.4 | 9.5 |
| strong | 217 | 363 | 54.6 | 4.3 |
| high | 15 | 68 | 73.3 | 40.0 |

## Filters and role agreement

- v1 (original): 4979 rows -> 2635 pass the always-on guard -> 1354 Recommended; role_ratio median 0.95, 67% inside the 0.85-1.20 band
- v2 (WR/TE recalibration): 4979 rows -> 2374 pass the always-on guard -> 1499 Recommended; role_ratio median 1.03, 76% inside the 0.85-1.20 band
- v3 (fitted team volume, full injury boost): 4979 rows -> 2289 pass the always-on guard -> 1455 Recommended; role_ratio median 1.03, 77% inside the 0.85-1.20 band
- v4 (injury redistribution scaled): 4979 rows -> 2263 pass the always-on guard -> 1407 Recommended; role_ratio median 1.02, 76% inside the 0.85-1.20 band
- v5 (current: median correction): 4979 rows -> 2477 pass the always-on guard -> 1827 Recommended; role_ratio median 1.02, 76% inside the 0.85-1.20 band

- OVER share by market, v1 (original): pass TD 55%, pass yds 62%, rec yds 81%, receptions 77%, rush yds 68%
- OVER share by market, v2 (WR/TE recalibration): pass TD 55%, pass yds 62%, rec yds 70%, receptions 54%, rush yds 68%
- OVER share by market, v3 (fitted team volume, full injury boost): pass TD 52%, pass yds 62%, rec yds 74%, receptions 51%, rush yds 67%
- OVER share by market, v4 (injury redistribution scaled): pass TD 52%, pass yds 62%, rec yds 76%, receptions 53%, rush yds 67%
- OVER share by market, v5 (current: median correction): pass TD 43%, pass yds 59%, rec yds 44%, receptions 29%, rush yds 41%
