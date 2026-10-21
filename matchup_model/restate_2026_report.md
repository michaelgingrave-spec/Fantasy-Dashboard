# Production-pipeline exam: model versions on the 2026 graded lines

2970 graded lines, weeks 1-6, run through `dfs.props.edges_from_raw` under each version: v1 (original); v2 (WR/TE recalibration); v3 (fitted team volume, full injury boost); v4 (current). Every constant of v1-v3 was fixed from 2023-25 box scores; nothing there was fitted on 2026. v4's injury scales (`INJ_REDIST`) were fitted on 2022-26 box scores, of which 2026 is a small share (`injury_absorption_report.md` lists the fit with each season left out, 2026 included), so v4's lift on these lines is not a clean out-of-sample test. ROI is at a flat -110 (prices are not stored per line).

**Harness check** -- v1 (original) reproduces what the screen stored when the lines were pulled: our proj within 0.15 on 77% of rows, same lean 96%, same tier 86%, role_ratio within .02 87% (the rest is injury / data drift since the pull).

## Win rate of the leans

| lines | bets | win | ROI at -110 | OVER share | OVER win | UNDER win |
|---|--:|--:|--:|--:|--:|--:|
| all markets, v1 (original) | 2935 | 50.4% | -3.7% | 72% | 49.0% | 54.0% |
| all markets, v2 (WR/TE recalibration) | 2935 | 52.0% | -0.7% | 60% | 50.1% | 54.8% |
| all markets, v3 (fitted team volume, full injury boost) | 2935 | 52.2% | -0.4% | 61% | 50.3% | 55.1% |
| all markets, v4 (current) | 2935 | 52.8% | +0.9% | 62% | 50.8% | 56.2% |
| receiving (rec yds + receptions), v1 (original) | 2080 | 50.4% | -3.7% | 76% | 49.1% | 54.7% |
| receiving (rec yds + receptions), v2 (WR/TE recalibration) | 2080 | 52.6% | +0.5% | 60% | 50.7% | 55.5% |
| receiving (rec yds + receptions), v3 (fitted team volume, full injury boost) | 2080 | 52.2% | -0.3% | 61% | 50.4% | 55.1% |
| receiving (rec yds + receptions), v4 (current) | 2080 | 53.1% | +1.3% | 63% | 51.0% | 56.5% |
| rec yds, v1 (original) | 1055 | 50.7% | -3.2% | 79% | 49.3% | 56.0% |
| rec yds, v2 (WR/TE recalibration) | 1055 | 51.3% | -2.1% | 67% | 49.6% | 54.8% |
| rec yds, v3 (fitted team volume, full injury boost) | 1055 | 50.4% | -3.7% | 72% | 49.0% | 54.1% |
| rec yds, v4 (current) | 1055 | 51.0% | -2.6% | 74% | 49.4% | 55.5% |
| receptions, v1 (original) | 1025 | 50.1% | -4.3% | 74% | 48.9% | 53.5% |
| receptions, v2 (WR/TE recalibration) | 1025 | 54.0% | +3.2% | 52% | 52.3% | 56.0% |
| receptions, v3 (fitted team volume, full injury boost) | 1025 | 54.0% | +3.2% | 49% | 52.4% | 55.6% |
| receptions, v4 (current) | 1025 | 55.2% | +5.4% | 51% | 53.5% | 57.0% |
| rush yds, v1 (original) | 497 | 46.3% | -11.7% | 68% | 43.3% | 52.5% |
| rush yds, v2 (WR/TE recalibration) | 497 | 46.3% | -11.7% | 68% | 43.3% | 52.5% |
| rush yds, v3 (fitted team volume, full injury boost) | 497 | 49.1% | -6.3% | 66% | 45.3% | 56.5% |
| rush yds, v4 (current) | 497 | 49.3% | -5.9% | 67% | 45.5% | 57.0% |
| pass yds, v1 (original) | 177 | 51.4% | -1.8% | 58% | 55.9% | 45.3% |
| pass yds, v2 (WR/TE recalibration) | 177 | 51.4% | -1.8% | 58% | 55.9% | 45.3% |
| pass yds, v3 (fitted team volume, full injury boost) | 177 | 52.0% | -0.8% | 60% | 56.1% | 45.7% |
| pass yds, v4 (current) | 177 | 52.0% | -0.8% | 60% | 56.1% | 45.7% |
| pass TD, v1 (original) | 177 | 61.0% | +16.5% | 53% | 60.6% | 61.4% |
| pass TD, v2 (WR/TE recalibration) | 177 | 61.0% | +16.5% | 53% | 60.6% | 61.4% |
| pass TD, v3 (fitted team volume, full injury boost) | 177 | 60.5% | +15.4% | 50% | 60.7% | 60.2% |
| pass TD, v4 (current) | 177 | 60.5% | +15.4% | 50% | 60.7% | 60.2% |

## Paired difference (v4 (current) minus each earlier version), same lines, player-clustered 95% CI

- vs v1 (original), all markets: **+2.42 pts** (CI +0.86 to +3.95, n=2935)
- vs v1 (original), receiving markets: **+2.64 pts** (CI +0.75 to +4.56, n=2080)
- vs v1 (original), rush yds: **+3.02 pts** (CI -0.57 to +6.72, n=497)
- vs v2 (WR/TE recalibration), all markets: **+0.85 pts** (CI -0.41 to +2.14, n=2935)
- vs v2 (WR/TE recalibration), receiving markets: **+0.43 pts** (CI -1.01 to +2.04, n=2080)
- vs v2 (WR/TE recalibration), rush yds: **+3.02 pts** (CI -0.60 to +6.54, n=497)
- vs v3 (fitted team volume, full injury boost), all markets: **+0.65 pts** (CI -0.10 to +1.39, n=2935)
- vs v3 (fitted team volume, full injury boost), receiving markets: **+0.87 pts** (CI -0.05 to +1.78, n=2080)
- vs v3 (fitted team volume, full injury boost), rush yds: **+0.20 pts** (CI -0.99 to +1.42, n=497)

Win % on receiving markets by week:

| week | v1 (original) | v2 (WR/TE recalibration) | v3 (fitted team volume, full injury boost) | v4 (current) | lines |
|--:|--:|--:|--:|--:|--:|
| 1 | 52.6% | 55.4% | 51.2% | 53.0% | 285 |
| 2 | 47.2% | 46.9% | 49.9% | 50.4% | 346 |
| 3 | 48.2% | 53.2% | 52.6% | 55.2% | 360 |
| 4 | 55.8% | 56.5% | 56.0% | 56.5% | 397 |
| 5 | 47.6% | 50.4% | 50.4% | 50.1% | 359 |
| 6 | 51.0% | 53.2% | 52.4% | 52.7% | 363 |

## Where v4 (current) differs from v3 (fitted team volume, full injury boost)

Lines picked by what the model did, not by how they turned out. `moved` is the change in our projection relative to the typical change for that market (so a recalibration that shifts every line alike is not a move); where the lean flips one version wins and the other loses.

| lines | n | v3 (fitted team volume, full injury boost) win | v4 (current) win | paired difference (player-clustered 95% CI) | OVER share v3 (fitted team volume, full injury boost) -> v4 (current) |
|---|--:|--:|--:|---|---|
| all lines | 2970 | 52.2% | 52.8% | +0.65 pts [-0.03, +1.41] (n=2935) | 61% -> 62% |
| projection moved >= 1% | 1573 | 51.6% | 52.8% | +1.16 pts [+0.00, +2.36] (n=1551) | 65% -> 67% |
| projection moved >= 5% | 333 | 58.3% | 57.4% | -0.92 pts [-4.31, +2.37] (n=326) | 66% -> 60% |
| projection moved >= 10% | 45 | 54.5% | 54.5% | +0.00 pts [-13.51, +11.54] (n=44) | 80% -> 62% |
| lean flipped by v4 (current) | 120 | 41.7% | 58.3% | +16.52 pts [-1.75, +33.35] (n=115) | 31% -> 69% |
| projection moved < 1% | 1397 | 52.8% | 52.9% | +0.07 pts [-0.79, +0.94] (n=1384) | 56% -> 58% |

## Tier tables (all graded lines, win % and ROI at -110)

Break-even win % by tier at the usual price: lean 52.4, solid 53.0, strong 53.7, high 55.4.

**As stored (what the screen showed)**

All bets:

| conf | n | win% | ROI@-110% |
|---|---|---|---|
| — | 1143 | 48.4 | -7.6 |
| lean | 683 | 52.5 | 0.3 |
| solid | 601 | 51.4 | -2.0 |
| strong | 414 | 53.2 | 1.5 |
| high | 129 | 53.5 | 2.1 |

Recommended only:

| conf | n | of all | win% | ROI@-110% |
|---|---|---|---|---|
| lean | 424 | 683 | 50.5 | -3.6 |
| solid | 260 | 601 | 52.9 | 1.0 |
| strong | 53 | 414 | 50.9 | -2.7 |
| high | 1 | 129 | 100.0 | 90.9 |

**v2 (WR/TE recalibration), restated**

All bets:

| conf | n | win% | ROI@-110% |
|---|---|---|---|
| — | 1329 | 51.0 | -2.7 |
| lean | 810 | 52.4 | 0.1 |
| solid | 554 | 52.7 | 0.7 |
| strong | 217 | 53.9 | 2.9 |
| high | 60 | 54.2 | 3.5 |

Recommended only:

| conf | n | of all | win% | ROI@-110% |
|---|---|---|---|---|
| lean | 573 | 810 | 51.0 | -2.7 |
| solid | 292 | 554 | 55.3 | 5.6 |
| strong | 52 | 217 | 44.2 | -15.6 |
| high | 2 | 60 | 50.0 | -4.5 |

**v3 (fitted team volume, full injury boost), restated**

All bets:

| conf | n | win% | ROI@-110% |
|---|---|---|---|
| — | 1404 | 51.0 | -2.6 |
| lean | 781 | 53.4 | 2.0 |
| solid | 551 | 51.7 | -1.2 |
| strong | 188 | 57.2 | 9.2 |
| high | 46 | 52.2 | -0.4 |

Recommended only:

| conf | n | of all | win% | ROI@-110% |
|---|---|---|---|---|
| lean | 571 | 781 | 52.8 | 0.7 |
| solid | 278 | 551 | 52.3 | -0.1 |
| strong | 33 | 188 | 51.5 | -1.7 |
| high | 2 | 46 | 100.0 | 90.9 |

**v4 (current), restated**

All bets:

| conf | n | win% | ROI@-110% |
|---|---|---|---|
| — | 1388 | 51.9 | -0.9 |
| lean | 775 | 53.7 | 2.6 |
| solid | 551 | 52.6 | 0.4 |
| strong | 211 | 56.4 | 7.7 |
| high | 45 | 53.3 | 1.8 |

Recommended only:

| conf | n | of all | win% | ROI@-110% |
|---|---|---|---|---|
| lean | 562 | 775 | 52.3 | -0.2 |
| solid | 271 | 551 | 52.4 | 0.0 |
| strong | 39 | 211 | 48.7 | -7.0 |
| high | 1 | 45 | 100.0 | 90.9 |

## Filters and role agreement

- v1 (original): 2970 rows -> 1556 pass the always-on guard -> 826 Recommended; role_ratio median 0.96, 68% inside the 0.85-1.20 band
- v2 (WR/TE recalibration): 2970 rows -> 1433 pass the always-on guard -> 919 Recommended; role_ratio median 1.04, 75% inside the 0.85-1.20 band
- v3 (fitted team volume, full injury boost): 2970 rows -> 1368 pass the always-on guard -> 884 Recommended; role_ratio median 1.03, 76% inside the 0.85-1.20 band
- v4 (current): 2970 rows -> 1379 pass the always-on guard -> 873 Recommended; role_ratio median 1.03, 75% inside the 0.85-1.20 band

- OVER share by market, v1 (original): pass TD 53%, pass yds 57%, rec yds 79%, receptions 74%, rush yds 68%
- OVER share by market, v2 (WR/TE recalibration): pass TD 53%, pass yds 57%, rec yds 67%, receptions 52%, rush yds 68%
- OVER share by market, v3 (fitted team volume, full injury boost): pass TD 50%, pass yds 60%, rec yds 72%, receptions 49%, rush yds 66%
- OVER share by market, v4 (current): pass TD 50%, pass yds 60%, rec yds 74%, receptions 51%, rush yds 67%
