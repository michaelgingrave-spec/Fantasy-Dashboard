# Production-pipeline exam: model versions on the 2026 graded lines

2468 graded lines, weeks 1-5, run through `dfs.props.edges_from_raw` under each version: v1 (original); v2 (WR/TE recalibration); v3 (current). Every constant was fixed from 2023-25 box scores; nothing was fitted on 2026. ROI is at a flat -110 (prices are not stored per line).

**Harness check** -- v1 (original) reproduces what the screen stored when the lines were pulled: our proj within 0.15 on 89% of rows, same lean 98%, same tier 94%, role_ratio within .02 98% (the rest is injury / data drift since the pull).

## Win rate of the leans

| lines | bets | win | ROI at -110 | OVER share | OVER win | UNDER win |
|---|--:|--:|--:|--:|--:|--:|
| all markets, v1 (original) | 2443 | 50.3% | -4.0% | 71% | 48.9% | 53.7% |
| all markets, v2 (WR/TE recalibration) | 2443 | 51.8% | -1.1% | 59% | 50.0% | 54.5% |
| all markets, v3 (current) | 2443 | 51.8% | -1.1% | 60% | 49.9% | 54.5% |
| receiving (rec yds + receptions), v1 (original) | 1725 | 50.3% | -3.9% | 75% | 48.9% | 54.7% |
| receiving (rec yds + receptions), v2 (WR/TE recalibration) | 1725 | 52.5% | +0.3% | 58% | 50.4% | 55.4% |
| receiving (rec yds + receptions), v3 (current) | 1725 | 52.2% | -0.4% | 59% | 50.1% | 55.1% |
| rec yds, v1 (original) | 874 | 51.1% | -2.4% | 77% | 49.6% | 56.6% |
| rec yds, v2 (WR/TE recalibration) | 874 | 51.8% | -1.1% | 66% | 50.0% | 55.3% |
| rec yds, v3 (current) | 874 | 50.9% | -2.8% | 71% | 49.4% | 54.7% |
| receptions, v1 (original) | 851 | 49.5% | -5.6% | 73% | 48.2% | 53.1% |
| receptions, v2 (WR/TE recalibration) | 851 | 53.2% | +1.6% | 50% | 51.0% | 55.5% |
| receptions, v3 (current) | 851 | 53.5% | +2.1% | 47% | 51.4% | 55.3% |
| rush yds, v1 (original) | 414 | 45.4% | -13.3% | 67% | 43.0% | 50.4% |
| rush yds, v2 (WR/TE recalibration) | 414 | 45.4% | -13.3% | 67% | 43.0% | 50.4% |
| rush yds, v3 (current) | 414 | 47.3% | -9.6% | 66% | 44.4% | 53.2% |
| pass yds, v1 (original) | 150 | 51.3% | -2.0% | 59% | 55.7% | 45.2% |
| pass yds, v2 (WR/TE recalibration) | 150 | 51.3% | -2.0% | 59% | 55.7% | 45.2% |
| pass yds, v3 (current) | 150 | 50.0% | -4.5% | 59% | 54.5% | 43.5% |
| pass TD, v1 (original) | 150 | 62.0% | +18.4% | 53% | 62.0% | 62.0% |
| pass TD, v2 (WR/TE recalibration) | 150 | 62.0% | +18.4% | 53% | 62.0% | 62.0% |
| pass TD, v3 (current) | 150 | 60.7% | +15.8% | 49% | 61.6% | 59.7% |

## Paired difference (v3 (current) minus each earlier version), same lines, player-clustered 95% CI

- vs v1 (original), all markets: **+1.51 pts** (CI -0.12 to +3.23, n=2443)
- vs v1 (original), receiving markets: **+1.86 pts** (CI -0.29 to +4.11, n=1725)
- vs v1 (original), rush yds: **+1.93 pts** (CI -2.10 to +5.84, n=414)
- vs v2 (WR/TE recalibration), all markets: **-0.04 pts** (CI -1.46 to +1.36, n=2443)
- vs v2 (WR/TE recalibration), receiving markets: **-0.35 pts** (CI -1.97 to +1.33, n=1725)
- vs v2 (WR/TE recalibration), rush yds: **+1.93 pts** (CI -2.07 to +5.60, n=414)

Win % on receiving markets by week:

| week | v1 (original) | v2 (WR/TE recalibration) | v3 (current) | lines |
|--:|--:|--:|--:|--:|
| 1 | 52.6% | 55.4% | 51.2% | 285 |
| 2 | 47.2% | 46.9% | 49.9% | 346 |
| 3 | 48.2% | 53.2% | 52.6% | 360 |
| 4 | 55.8% | 56.5% | 56.0% | 397 |
| 5 | 47.6% | 50.4% | 50.4% | 359 |

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

**v2 (WR/TE recalibration), restated**

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

**v3 (current), restated**

All bets:

| conf | n | win% | ROI@-110% |
|---|---|---|---|
| — | 1184 | 50.1 | -4.4 |
| lean | 639 | 53.5 | 2.1 |
| solid | 462 | 52.3 | -0.2 |
| strong | 149 | 56.8 | 8.4 |
| high | 34 | 50.0 | -4.5 |

Recommended only:

| conf | n | of all | win% | ROI@-110% |
|---|---|---|---|---|
| lean | 463 | 639 | 53.7 | 2.5 |
| solid | 238 | 462 | 51.9 | -0.9 |
| strong | 26 | 149 | 50.0 | -4.5 |
| high | 2 | 34 | 100.0 | 90.9 |

## Filters and role agreement

- v1 (original): 2468 rows -> 1300 pass the always-on guard -> 704 Recommended; role_ratio median 0.97, 68% inside the 0.85-1.20 band
- v2 (WR/TE recalibration): 2468 rows -> 1180 pass the always-on guard -> 758 Recommended; role_ratio median 1.04, 75% inside the 0.85-1.20 band
- v3 (current): 2468 rows -> 1136 pass the always-on guard -> 729 Recommended; role_ratio median 1.04, 75% inside the 0.85-1.20 band

- OVER share by market, v1 (original): pass TD 52%, pass yds 58%, rec yds 77%, receptions 73%, rush yds 67%
- OVER share by market, v2 (WR/TE recalibration): pass TD 52%, pass yds 58%, rec yds 66%, receptions 51%, rush yds 67%
- OVER share by market, v3 (current): pass TD 48%, pass yds 58%, rec yds 71%, receptions 47%, rush yds 67%
