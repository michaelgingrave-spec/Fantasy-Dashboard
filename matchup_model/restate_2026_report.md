# Production-pipeline exam: model versions on the 2026 graded lines

3461 graded lines, weeks 1-7, run through `dfs.props.edges_from_raw` under each version: v1 (original); v2 (WR/TE recalibration); v3 (fitted team volume, full injury boost); v4 (current). Every constant of v1-v3 was fixed from 2023-25 box scores; nothing there was fitted on 2026. v4's injury scales (`INJ_REDIST`) were fitted on 2022-26 box scores, of which 2026 is a small share (`injury_absorption_report.md` lists the fit with each season left out, 2026 included), so v4's lift on these lines is not a clean out-of-sample test. ROI is at a flat -110 (prices are not stored per line).

**Harness check** -- v1 (original) reproduces what the screen stored when the lines were pulled: our proj within 0.15 on 69% of rows, same lean 94%, same tier 80%, role_ratio within .02 79% (the rest is injury / data drift since the pull).

## Win rate of the leans

| lines | bets | win | ROI at -110 | OVER share | OVER win | UNDER win |
|---|--:|--:|--:|--:|--:|--:|
| all markets, v1 (original) | 3421 | 49.7% | -5.1% | 73% | 48.2% | 53.9% |
| all markets, v2 (WR/TE recalibration) | 3421 | 51.5% | -1.6% | 61% | 49.3% | 55.1% |
| all markets, v3 (fitted team volume, full injury boost) | 3421 | 52.1% | -0.6% | 61% | 49.8% | 55.8% |
| all markets, v4 (current) | 3421 | 52.8% | +0.8% | 63% | 50.3% | 57.0% |
| receiving (rec yds + receptions), v1 (original) | 2427 | 49.6% | -5.3% | 77% | 48.1% | 54.7% |
| receiving (rec yds + receptions), v2 (WR/TE recalibration) | 2427 | 52.2% | -0.4% | 61% | 49.7% | 55.9% |
| receiving (rec yds + receptions), v3 (fitted team volume, full injury boost) | 2427 | 52.2% | -0.3% | 61% | 49.8% | 56.1% |
| receiving (rec yds + receptions), v4 (current) | 2427 | 53.2% | +1.5% | 63% | 50.5% | 57.7% |
| rec yds, v1 (original) | 1230 | 50.0% | -4.5% | 80% | 48.4% | 56.4% |
| rec yds, v2 (WR/TE recalibration) | 1230 | 50.9% | -2.8% | 68% | 48.8% | 55.5% |
| rec yds, v3 (fitted team volume, full injury boost) | 1230 | 50.4% | -3.8% | 73% | 48.5% | 55.5% |
| rec yds, v4 (current) | 1230 | 51.1% | -2.4% | 75% | 49.0% | 57.4% |
| receptions, v1 (original) | 1197 | 49.2% | -6.1% | 75% | 47.9% | 53.2% |
| receptions, v2 (WR/TE recalibration) | 1197 | 53.5% | +2.1% | 53% | 51.0% | 56.2% |
| receptions, v3 (fitted team volume, full injury boost) | 1197 | 54.1% | +3.3% | 49% | 51.8% | 56.4% |
| receptions, v4 (current) | 1197 | 55.2% | +5.4% | 52% | 52.8% | 57.9% |
| rush yds, v1 (original) | 581 | 45.6% | -12.9% | 67% | 42.9% | 51.3% |
| rush yds, v2 (WR/TE recalibration) | 581 | 45.6% | -12.9% | 67% | 42.9% | 51.3% |
| rush yds, v3 (fitted team volume, full injury boost) | 581 | 48.5% | -7.3% | 66% | 44.9% | 55.5% |
| rush yds, v4 (current) | 581 | 48.9% | -6.7% | 66% | 45.2% | 56.1% |
| pass yds, v1 (original) | 204 | 51.0% | -2.7% | 60% | 55.7% | 43.9% |
| pass yds, v2 (WR/TE recalibration) | 204 | 51.0% | -2.7% | 60% | 55.7% | 43.9% |
| pass yds, v3 (fitted team volume, full injury boost) | 204 | 50.5% | -3.6% | 61% | 55.2% | 43.0% |
| pass yds, v4 (current) | 204 | 50.5% | -3.6% | 61% | 55.2% | 43.0% |
| pass TD, v1 (original) | 205 | 61.5% | +17.3% | 54% | 59.5% | 63.8% |
| pass TD, v2 (WR/TE recalibration) | 205 | 61.5% | +17.3% | 54% | 59.5% | 63.8% |
| pass TD, v3 (fitted team volume, full injury boost) | 205 | 61.5% | +17.3% | 51% | 60.0% | 63.0% |
| pass TD, v4 (current) | 205 | 61.5% | +17.3% | 51% | 60.0% | 63.0% |

## Paired difference (v4 (current) minus each earlier version), same lines, player-clustered 95% CI

- vs v1 (original), all markets: **+3.07 pts** (CI +1.66 to +4.41, n=3421)
- vs v1 (original), receiving markets: **+3.54 pts** (CI +1.72 to +5.30, n=2427)
- vs v1 (original), rush yds: **+3.27 pts** (CI +0.00 to +6.36, n=581)
- vs v2 (WR/TE recalibration), all markets: **+1.26 pts** (CI +0.06 to +2.45, n=3421)
- vs v2 (WR/TE recalibration), receiving markets: **+0.99 pts** (CI -0.41 to +2.39, n=2427)
- vs v2 (WR/TE recalibration), rush yds: **+3.27 pts** (CI +0.18 to +6.45, n=581)
- vs v3 (fitted team volume, full injury boost), all markets: **+0.70 pts** (CI -0.03 to +1.47, n=3421)
- vs v3 (fitted team volume, full injury boost), receiving markets: **+0.91 pts** (CI -0.08 to +1.93, n=2427)
- vs v3 (fitted team volume, full injury boost), rush yds: **+0.34 pts** (CI -0.62 to +1.34, n=581)

Win % on receiving markets by week:

| week | v1 (original) | v2 (WR/TE recalibration) | v3 (fitted team volume, full injury boost) | v4 (current) | lines |
|--:|--:|--:|--:|--:|--:|
| 1 | 52.6% | 55.4% | 51.2% | 53.0% | 285 |
| 2 | 47.2% | 46.9% | 49.9% | 50.4% | 346 |
| 3 | 48.2% | 53.2% | 52.6% | 54.9% | 360 |
| 4 | 55.8% | 56.5% | 56.0% | 56.5% | 397 |
| 5 | 47.6% | 50.4% | 50.4% | 50.1% | 359 |
| 6 | 51.0% | 53.2% | 52.4% | 52.7% | 363 |
| 7 | 44.7% | 49.3% | 52.4% | 53.9% | 351 |

## Where v4 (current) differs from v3 (fitted team volume, full injury boost)

Lines picked by what the model did, not by how they turned out. `moved` is the change in our projection relative to the typical change for that market (so a recalibration that shifts every line alike is not a move); where the lean flips one version wins and the other loses.

| lines | n | v3 (fitted team volume, full injury boost) win | v4 (current) win | paired difference (player-clustered 95% CI) | OVER share v3 (fitted team volume, full injury boost) -> v4 (current) |
|---|--:|--:|--:|---|---|
| all lines | 3461 | 52.1% | 52.8% | +0.70 pts [+0.03, +1.47] (n=3421) | 61% -> 63% |
| projection moved >= 1% | 1847 | 51.3% | 52.4% | +1.10 pts [-0.11, +2.29] (n=1823) | 65% -> 67% |
| projection moved >= 5% | 407 | 55.8% | 55.5% | -0.25 pts [-3.49, +2.93] (n=400) | 68% -> 62% |
| projection moved >= 10% | 76 | 49.3% | 53.3% | +4.00 pts [-6.76, +13.75] (n=75) | 86% -> 66% |
| lean flipped by v4 (current) | 145 | 41.4% | 58.6% | +17.14 pts [-0.69, +33.33] (n=140) | 31% -> 69% |
| projection moved < 1% | 1614 | 53.0% | 53.3% | +0.25 pts [-0.50, +1.05] (n=1598) | 57% -> 58% |

## Tier tables (all graded lines, win % and ROI at -110)

Break-even win % by tier at the usual price: lean 52.4, solid 53.0, strong 53.7, high 55.4.

**As stored (what the screen showed)**

All bets:

| conf | n | win% | ROI@-110% |
|---|---|---|---|
| — | 1399 | 48.6 | -7.2 |
| lean | 819 | 53.2 | 1.6 |
| solid | 671 | 51.4 | -2.0 |
| strong | 440 | 53.8 | 2.7 |
| high | 132 | 53.8 | 2.7 |

Recommended only:

| conf | n | of all | win% | ROI@-110% |
|---|---|---|---|---|
| lean | 525 | 819 | 51.8 | -1.0 |
| solid | 291 | 671 | 52.8 | 0.8 |
| strong | 55 | 440 | 52.7 | 0.7 |
| high | 1 | 132 | 100.0 | 90.9 |

**v2 (WR/TE recalibration), restated**

All bets:

| conf | n | win% | ROI@-110% |
|---|---|---|---|
| — | 1559 | 50.4 | -3.7 |
| lean | 940 | 51.9 | -0.8 |
| solid | 640 | 52.7 | 0.6 |
| strong | 257 | 52.9 | 1.0 |
| high | 65 | 55.6 | 6.1 |

Recommended only:

| conf | n | of all | win% | ROI@-110% |
|---|---|---|---|---|
| lean | 672 | 940 | 51.0 | -2.7 |
| solid | 329 | 640 | 55.5 | 5.9 |
| strong | 61 | 257 | 42.6 | -18.6 |
| high | 2 | 65 | 50.0 | -4.5 |

**v3 (fitted team volume, full injury boost), restated**

All bets:

| conf | n | win% | ROI@-110% |
|---|---|---|---|
| — | 1647 | 50.6 | -3.4 |
| lean | 917 | 53.7 | 2.5 |
| solid | 625 | 51.2 | -2.2 |
| strong | 221 | 58.4 | 11.6 |
| high | 51 | 54.9 | 4.8 |

Recommended only:

| conf | n | of all | win% | ROI@-110% |
|---|---|---|---|---|
| lean | 674 | 917 | 53.5 | 2.1 |
| solid | 312 | 625 | 52.1 | -0.6 |
| strong | 38 | 221 | 52.6 | 0.5 |
| high | 2 | 51 | 100.0 | 90.9 |

**v4 (current), restated**

All bets:

| conf | n | win% | ROI@-110% |
|---|---|---|---|
| — | 1644 | 51.5 | -1.7 |
| lean | 911 | 54.2 | 3.5 |
| solid | 621 | 52.4 | 0.1 |
| strong | 237 | 57.2 | 9.2 |
| high | 48 | 54.2 | 3.4 |

Recommended only:

| conf | n | of all | win% | ROI@-110% |
|---|---|---|---|---|
| lean | 663 | 911 | 53.1 | 1.3 |
| solid | 302 | 621 | 52.3 | -0.1 |
| strong | 41 | 237 | 51.2 | -2.2 |
| high | 1 | 48 | 100.0 | 90.9 |

## Filters and role agreement

- v1 (original): 3461 rows -> 1822 pass the always-on guard -> 964 Recommended; role_ratio median 0.95, 68% inside the 0.85-1.20 band
- v2 (WR/TE recalibration): 3461 rows -> 1655 pass the always-on guard -> 1064 Recommended; role_ratio median 1.04, 76% inside the 0.85-1.20 band
- v3 (fitted team volume, full injury boost): 3461 rows -> 1583 pass the always-on guard -> 1026 Recommended; role_ratio median 1.03, 77% inside the 0.85-1.20 band
- v4 (current): 3461 rows -> 1581 pass the always-on guard -> 1007 Recommended; role_ratio median 1.03, 76% inside the 0.85-1.20 band

- OVER share by market, v1 (original): pass TD 54%, pass yds 59%, rec yds 80%, receptions 75%, rush yds 68%
- OVER share by market, v2 (WR/TE recalibration): pass TD 54%, pass yds 59%, rec yds 68%, receptions 53%, rush yds 68%
- OVER share by market, v3 (fitted team volume, full injury boost): pass TD 51%, pass yds 61%, rec yds 73%, receptions 50%, rush yds 66%
- OVER share by market, v4 (current): pass TD 51%, pass yds 61%, rec yds 75%, receptions 52%, rush yds 66%
