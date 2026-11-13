# Production-pipeline exam: model versions on the 2026 graded lines

4481 graded lines, weeks 1-9, run through `dfs.props.edges_from_raw` under each version: v1 (original); v2 (WR/TE recalibration); v3 (fitted team volume, full injury boost); v4 (injury redistribution scaled); v5 (current: median correction). Every constant of v1-v3 was fixed from 2023-25 box scores; nothing there was fitted on 2026. v4's injury scales (`INJ_REDIST`) were fitted on 2022-26 box scores, of which 2026 is a small share (`injury_absorption_report.md` lists the fit with each season left out, 2026 included), so v4's lift on these lines is not a clean out-of-sample test. v5's median factors (`median_cal.py`) are learned for each week from the graded lines of the weeks before it, so they never see the week they are applied to, but the rule was chosen after looking at weeks 1-8: week 9 onward is the clean test. ROI is at a flat -110 (prices are not stored per line).

**Harness check** -- v1 (original) reproduces what the screen stored when the lines were pulled: our proj within 0.15 on 57% of rows, same lean 91%, same tier 71%, role_ratio within .02 69% (the rest is injury / data drift since the pull).

## Win rate of the leans

| lines | bets | win | ROI at -110 | OVER share | OVER win | UNDER win |
|---|--:|--:|--:|--:|--:|--:|
| all markets, v1 (original) | 4435 | 49.9% | -4.7% | 74% | 48.1% | 55.0% |
| all markets, v2 (WR/TE recalibration) | 4435 | 51.7% | -1.4% | 62% | 49.1% | 55.8% |
| all markets, v3 (fitted team volume, full injury boost) | 4435 | 52.2% | -0.4% | 62% | 49.5% | 56.4% |
| all markets, v4 (injury redistribution scaled) | 4435 | 52.6% | +0.5% | 64% | 49.9% | 57.3% |
| all markets, v5 (current: median correction) | 4435 | 54.3% | +3.7% | 40% | 52.0% | 55.8% |
| receiving (rec yds + receptions), v1 (original) | 3159 | 50.0% | -4.5% | 78% | 48.3% | 56.3% |
| receiving (rec yds + receptions), v2 (WR/TE recalibration) | 3159 | 52.5% | +0.3% | 62% | 49.9% | 56.7% |
| receiving (rec yds + receptions), v3 (fitted team volume, full injury boost) | 3159 | 52.5% | +0.1% | 62% | 49.8% | 56.8% |
| receiving (rec yds + receptions), v4 (injury redistribution scaled) | 3159 | 53.1% | +1.3% | 64% | 50.3% | 58.0% |
| receiving (rec yds + receptions), v5 (current: median correction) | 3159 | 54.6% | +4.2% | 37% | 52.6% | 55.8% |
| rec yds, v1 (original) | 1600 | 50.2% | -4.2% | 80% | 48.4% | 57.4% |
| rec yds, v2 (WR/TE recalibration) | 1600 | 51.3% | -2.0% | 69% | 49.0% | 56.5% |
| rec yds, v3 (fitted team volume, full injury boost) | 1600 | 50.6% | -3.5% | 74% | 48.6% | 56.2% |
| rec yds, v4 (injury redistribution scaled) | 1600 | 51.2% | -2.2% | 75% | 49.0% | 58.0% |
| rec yds, v5 (current: median correction) | 1600 | 53.8% | +2.6% | 45% | 51.2% | 55.9% |
| receptions, v1 (original) | 1559 | 49.9% | -4.7% | 76% | 48.2% | 55.3% |
| receptions, v2 (WR/TE recalibration) | 1559 | 53.8% | +2.6% | 53% | 51.0% | 56.9% |
| receptions, v3 (fitted team volume, full injury boost) | 1559 | 54.4% | +3.8% | 50% | 51.7% | 57.1% |
| receptions, v4 (injury redistribution scaled) | 1559 | 55.0% | +4.9% | 52% | 52.2% | 58.0% |
| receptions, v5 (current: median correction) | 1559 | 55.5% | +5.9% | 30% | 54.8% | 55.8% |
| rush yds, v1 (original) | 749 | 45.1% | -13.8% | 68% | 42.5% | 50.6% |
| rush yds, v2 (WR/TE recalibration) | 749 | 45.1% | -13.8% | 68% | 42.5% | 50.6% |
| rush yds, v3 (fitted team volume, full injury boost) | 749 | 47.9% | -8.5% | 66% | 44.5% | 54.8% |
| rush yds, v4 (injury redistribution scaled) | 749 | 48.1% | -8.2% | 67% | 44.6% | 55.0% |
| rush yds, v5 (current: median correction) | 749 | 51.9% | -0.8% | 42% | 46.0% | 56.2% |
| pass yds, v1 (original) | 261 | 49.8% | -4.9% | 61% | 53.1% | 44.6% |
| pass yds, v2 (WR/TE recalibration) | 261 | 49.8% | -4.9% | 61% | 53.1% | 44.6% |
| pass yds, v3 (fitted team volume, full injury boost) | 261 | 50.2% | -4.2% | 62% | 53.4% | 45.0% |
| pass yds, v4 (injury redistribution scaled) | 261 | 50.2% | -4.2% | 62% | 53.4% | 45.0% |
| pass yds, v5 (current: median correction) | 261 | 52.1% | -0.5% | 59% | 55.2% | 47.7% |
| pass TD, v1 (original) | 262 | 61.8% | +18.0% | 54% | 58.5% | 65.8% |
| pass TD, v2 (WR/TE recalibration) | 262 | 61.8% | +18.0% | 54% | 58.5% | 65.8% |
| pass TD, v3 (fitted team volume, full injury boost) | 262 | 62.2% | +18.8% | 52% | 59.3% | 65.4% |
| pass TD, v4 (injury redistribution scaled) | 262 | 62.2% | +18.8% | 52% | 59.3% | 65.4% |
| pass TD, v5 (current: median correction) | 262 | 59.2% | +12.9% | 43% | 57.5% | 60.4% |

## Paired difference (v5 (current: median correction) minus each earlier version), same lines, player-clustered 95% CI

- vs v1 (original), all markets: **+4.40 pts** (CI +2.39 to +6.41, n=4435)
- vs v1 (original), receiving markets: **+4.56 pts** (CI +2.13 to +6.95, n=3159)
- vs v1 (original), rush yds: **+6.81 pts** (CI +2.52 to +10.72, n=749)
- vs v2 (WR/TE recalibration), all markets: **+2.64 pts** (CI +0.97 to +4.25, n=4435)
- vs v2 (WR/TE recalibration), receiving markets: **+2.09 pts** (CI +0.25 to +3.96, n=3159)
- vs v2 (WR/TE recalibration), rush yds: **+6.81 pts** (CI +2.84 to +11.02, n=749)
- vs v3 (fitted team volume, full injury boost), all markets: **+2.14 pts** (CI +0.69 to +3.60, n=4435)
- vs v3 (fitted team volume, full injury boost), receiving markets: **+2.15 pts** (CI +0.35 to +3.98, n=3159)
- vs v3 (fitted team volume, full injury boost), rush yds: **+4.01 pts** (CI +0.69 to +7.19, n=749)
- vs v4 (injury redistribution scaled), all markets: **+1.67 pts** (CI +0.25 to +3.13, n=4435)
- vs v4 (injury redistribution scaled), receiving markets: **+1.52 pts** (CI -0.34 to +3.24, n=3159)
- vs v4 (injury redistribution scaled), rush yds: **+3.87 pts** (CI +0.68 to +7.04, n=749)

Win % on receiving markets by week:

| week | v1 (original) | v2 (WR/TE recalibration) | v3 (fitted team volume, full injury boost) | v4 (injury redistribution scaled) | v5 (current: median correction) | lines |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 52.6% | 55.4% | 51.2% | 53.0% | 53.0% | 285 |
| 2 | 47.2% | 46.9% | 49.9% | 50.4% | 50.4% | 346 |
| 3 | 48.2% | 53.2% | 52.6% | 54.9% | 53.8% | 360 |
| 4 | 55.8% | 56.5% | 56.0% | 56.5% | 54.5% | 397 |
| 5 | 47.6% | 50.4% | 50.4% | 50.1% | 53.0% | 359 |
| 6 | 51.0% | 53.2% | 52.4% | 52.7% | 53.2% | 363 |
| 7 | 44.7% | 49.3% | 52.4% | 53.9% | 59.7% | 351 |
| 8 | 48.7% | 51.0% | 51.0% | 49.9% | 57.7% | 359 |
| 9 | 54.1% | 56.3% | 55.2% | 55.7% | 55.7% | 379 |

## Where v5 (current: median correction) differs from v4 (injury redistribution scaled)

Lines picked by what the model did, not by how they turned out. `moved` is the change in our projection relative to the typical change for that market (so a recalibration that shifts every line alike is not a move); where the lean flips one version wins and the other loses.

| lines | n | v4 (injury redistribution scaled) win | v5 (current: median correction) win | paired difference (player-clustered 95% CI) | OVER share v4 (injury redistribution scaled) -> v5 (current: median correction) |
|---|--:|--:|--:|---|---|
| all lines | 4481 | 52.6% | 54.3% | +1.67 pts [+0.24, +3.09] (n=4435) | 63% -> 40% |
| projection moved >= 1% | 0 | | | | |
| projection moved >= 5% | 0 | | | | |
| projection moved >= 10% | 0 | | | | |
| lean flipped by v5 (current: median correction) | 1101 | 46.6% | 53.4% | +6.80 pts [+0.55, +12.66] (n=1088) | 98% -> 2% |
| projection moved < 1% | 4481 | 52.6% | 54.3% | +1.67 pts [+0.14, +3.04] (n=4435) | 63% -> 40% |

## Tier tables (all graded lines, win % and ROI at -110)

Break-even win % by tier at the usual price: lean 52.4, solid 53.0, strong 53.7, high 55.4.

**As stored (what the screen showed)**

All bets:

| conf | n | win% | ROI@-110% |
|---|---|---|---|
| — | 1889 | 48.8 | -6.7 |
| lean | 1077 | 53.0 | 1.2 |
| solid | 867 | 52.3 | -0.2 |
| strong | 503 | 54.8 | 4.6 |
| high | 145 | 53.1 | 1.4 |

Recommended only:

| conf | n | of all | win% | ROI@-110% |
|---|---|---|---|---|
| lean | 707 | 1077 | 52.7 | 0.7 |
| solid | 372 | 867 | 53.7 | 2.4 |
| strong | 62 | 503 | 53.2 | 1.6 |
| high | 1 | 145 | 100.0 | 90.9 |

**v2 (WR/TE recalibration), restated**

All bets:

| conf | n | win% | ROI@-110% |
|---|---|---|---|
| — | 2018 | 50.2 | -4.1 |
| lean | 1216 | 52.0 | -0.8 |
| solid | 828 | 54.4 | 3.8 |
| strong | 335 | 51.3 | -2.0 |
| high | 84 | 56.1 | 7.1 |

Recommended only:

| conf | n | of all | win% | ROI@-110% |
|---|---|---|---|---|
| lean | 863 | 1216 | 51.9 | -0.9 |
| solid | 422 | 828 | 55.8 | 6.6 |
| strong | 76 | 335 | 40.8 | -22.1 |
| high | 2 | 84 | 50.0 | -4.5 |

**v3 (fitted team volume, full injury boost), restated**

All bets:

| conf | n | win% | ROI@-110% |
|---|---|---|---|
| — | 2116 | 50.5 | -3.6 |
| lean | 1170 | 53.5 | 2.2 |
| solid | 844 | 51.8 | -1.1 |
| strong | 282 | 59.3 | 13.2 |
| high | 69 | 55.1 | 5.1 |

Recommended only:

| conf | n | of all | win% | ROI@-110% |
|---|---|---|---|---|
| lean | 859 | 1170 | 54.1 | 3.2 |
| solid | 416 | 844 | 51.7 | -1.3 |
| strong | 46 | 282 | 54.3 | 3.8 |
| high | 2 | 69 | 100.0 | 90.9 |

**v4 (injury redistribution scaled), restated**

All bets:

| conf | n | win% | ROI@-110% |
|---|---|---|---|
| — | 2134 | 51.0 | -2.6 |
| lean | 1169 | 53.8 | 2.7 |
| solid | 817 | 53.2 | 1.5 |
| strong | 300 | 58.2 | 11.1 |
| high | 61 | 52.5 | 0.1 |

Recommended only:

| conf | n | of all | win% | ROI@-110% |
|---|---|---|---|---|
| lean | 845 | 1169 | 53.5 | 2.2 |
| solid | 383 | 817 | 53.3 | 1.7 |
| strong | 48 | 300 | 52.1 | -0.6 |
| high | 1 | 61 | 100.0 | 90.9 |

**v5 (current: median correction), restated**

All bets:

| conf | n | win% | ROI@-110% |
|---|---|---|---|
| — | 1974 | 52.9 | 1.0 |
| lean | 1219 | 54.8 | 4.7 |
| solid | 897 | 56.0 | 7.0 |
| strong | 329 | 55.0 | 5.1 |
| high | 62 | 58.1 | 10.9 |

Recommended only:

| conf | n | of all | win% | ROI@-110% |
|---|---|---|---|---|
| lean | 820 | 1219 | 54.4 | 3.9 |
| solid | 626 | 897 | 56.7 | 8.2 |
| strong | 195 | 329 | 56.2 | 7.3 |
| high | 15 | 62 | 73.3 | 40.0 |

## Filters and role agreement

- v1 (original): 4481 rows -> 2368 pass the always-on guard -> 1221 Recommended; role_ratio median 0.95, 67% inside the 0.85-1.20 band
- v2 (WR/TE recalibration): 4481 rows -> 2139 pass the always-on guard -> 1363 Recommended; role_ratio median 1.03, 76% inside the 0.85-1.20 band
- v3 (fitted team volume, full injury boost): 4481 rows -> 2056 pass the always-on guard -> 1323 Recommended; role_ratio median 1.03, 77% inside the 0.85-1.20 band
- v4 (injury redistribution scaled): 4481 rows -> 2031 pass the always-on guard -> 1277 Recommended; role_ratio median 1.02, 76% inside the 0.85-1.20 band
- v5 (current: median correction): 4481 rows -> 2241 pass the always-on guard -> 1656 Recommended; role_ratio median 1.02, 76% inside the 0.85-1.20 band

- OVER share by market, v1 (original): pass TD 54%, pass yds 61%, rec yds 81%, receptions 76%, rush yds 68%
- OVER share by market, v2 (WR/TE recalibration): pass TD 54%, pass yds 61%, rec yds 69%, receptions 54%, rush yds 68%
- OVER share by market, v3 (fitted team volume, full injury boost): pass TD 51%, pass yds 61%, rec yds 74%, receptions 50%, rush yds 66%
- OVER share by market, v4 (injury redistribution scaled): pass TD 51%, pass yds 61%, rec yds 76%, receptions 52%, rush yds 67%
- OVER share by market, v5 (current: median correction): pass TD 43%, pass yds 59%, rec yds 45%, receptions 29%, rush yds 42%
