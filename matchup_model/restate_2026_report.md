# Production-pipeline exam: model versions on the 2026 graded lines

3964 graded lines, weeks 1-8, run through `dfs.props.edges_from_raw` under each version: v1 (original); v2 (WR/TE recalibration); v3 (fitted team volume, full injury boost); v4 (current). Every constant of v1-v3 was fixed from 2023-25 box scores; nothing there was fitted on 2026. v4's injury scales (`INJ_REDIST`) were fitted on 2022-26 box scores, of which 2026 is a small share (`injury_absorption_report.md` lists the fit with each season left out, 2026 included), so v4's lift on these lines is not a clean out-of-sample test. ROI is at a flat -110 (prices are not stored per line).

**Harness check** -- v1 (original) reproduces what the screen stored when the lines were pulled: our proj within 0.15 on 62% of rows, same lean 92%, same tier 75%, role_ratio within .02 73% (the rest is injury / data drift since the pull).

## Win rate of the leans

| lines | bets | win | ROI at -110 | OVER share | OVER win | UNDER win |
|---|--:|--:|--:|--:|--:|--:|
| all markets, v1 (original) | 3922 | 49.5% | -5.5% | 74% | 47.9% | 54.0% |
| all markets, v2 (WR/TE recalibration) | 3922 | 51.3% | -2.1% | 62% | 48.9% | 55.0% |
| all markets, v3 (fitted team volume, full injury boost) | 3922 | 51.9% | -0.9% | 62% | 49.5% | 55.9% |
| all markets, v4 (current) | 3922 | 52.4% | +0.0% | 63% | 49.9% | 56.7% |
| receiving (rec yds + receptions), v1 (original) | 2784 | 49.5% | -5.5% | 78% | 48.0% | 54.9% |
| receiving (rec yds + receptions), v2 (WR/TE recalibration) | 2784 | 52.0% | -0.7% | 61% | 49.5% | 56.0% |
| receiving (rec yds + receptions), v3 (fitted team volume, full injury boost) | 2784 | 52.1% | -0.6% | 61% | 49.5% | 56.1% |
| receiving (rec yds + receptions), v4 (current) | 2784 | 52.7% | +0.7% | 63% | 50.1% | 57.3% |
| rec yds, v1 (original) | 1410 | 49.8% | -5.0% | 80% | 48.2% | 56.3% |
| rec yds, v2 (WR/TE recalibration) | 1410 | 51.0% | -2.6% | 69% | 48.8% | 55.9% |
| rec yds, v3 (fitted team volume, full injury boost) | 1410 | 50.4% | -3.9% | 73% | 48.4% | 55.7% |
| rec yds, v4 (current) | 1410 | 51.1% | -2.5% | 75% | 48.9% | 57.5% |
| receptions, v1 (original) | 1374 | 49.2% | -6.1% | 76% | 47.7% | 53.8% |
| receptions, v2 (WR/TE recalibration) | 1374 | 53.1% | +1.3% | 53% | 50.4% | 56.0% |
| receptions, v3 (fitted team volume, full injury boost) | 1374 | 53.9% | +2.8% | 49% | 51.3% | 56.4% |
| receptions, v4 (current) | 1374 | 54.4% | +3.9% | 51% | 51.8% | 57.2% |
| rush yds, v1 (original) | 669 | 45.3% | -13.5% | 68% | 42.5% | 51.2% |
| rush yds, v2 (WR/TE recalibration) | 669 | 45.3% | -13.5% | 68% | 42.5% | 51.2% |
| rush yds, v3 (fitted team volume, full injury boost) | 669 | 47.8% | -8.7% | 67% | 44.3% | 55.0% |
| rush yds, v4 (current) | 669 | 48.0% | -8.4% | 67% | 44.4% | 55.3% |
| pass yds, v1 (original) | 232 | 50.0% | -4.5% | 61% | 54.2% | 43.3% |
| pass yds, v2 (WR/TE recalibration) | 232 | 50.0% | -4.5% | 61% | 54.2% | 43.3% |
| pass yds, v3 (fitted team volume, full injury boost) | 232 | 51.7% | -1.3% | 62% | 55.6% | 45.5% |
| pass yds, v4 (current) | 232 | 51.7% | -1.3% | 62% | 55.6% | 45.5% |
| pass TD, v1 (original) | 233 | 60.9% | +16.3% | 55% | 58.9% | 63.5% |
| pass TD, v2 (WR/TE recalibration) | 233 | 60.9% | +16.3% | 55% | 58.9% | 63.5% |
| pass TD, v3 (fitted team volume, full injury boost) | 233 | 61.4% | +17.2% | 52% | 59.8% | 63.1% |
| pass TD, v4 (current) | 233 | 61.4% | +17.2% | 52% | 59.8% | 63.1% |

## Paired difference (v4 (current) minus each earlier version), same lines, player-clustered 95% CI

- vs v1 (original), all markets: **+2.91 pts** (CI +1.48 to +4.28, n=3922)
- vs v1 (original), receiving markets: **+3.23 pts** (CI +1.43 to +5.04, n=2784)
- vs v1 (original), rush yds: **+2.69 pts** (CI -0.00 to +5.54, n=669)
- vs v2 (WR/TE recalibration), all markets: **+1.12 pts** (CI +0.03 to +2.20, n=3922)
- vs v2 (WR/TE recalibration), receiving markets: **+0.72 pts** (CI -0.59 to +2.11, n=2784)
- vs v2 (WR/TE recalibration), rush yds: **+2.69 pts** (CI +0.00 to +5.49, n=669)
- vs v3 (fitted team volume, full injury boost), all markets: **+0.48 pts** (CI -0.20 to +1.20, n=3922)
- vs v3 (fitted team volume, full injury boost), receiving markets: **+0.65 pts** (CI -0.32 to +1.61, n=2784)
- vs v3 (fitted team volume, full injury boost), rush yds: **+0.15 pts** (CI -0.68 to +1.10, n=669)

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
| 8 | 48.7% | 51.0% | 51.0% | 49.9% | 359 |

## Where v4 (current) differs from v3 (fitted team volume, full injury boost)

Lines picked by what the model did, not by how they turned out. `moved` is the change in our projection relative to the typical change for that market (so a recalibration that shifts every line alike is not a move); where the lean flips one version wins and the other loses.

| lines | n | v3 (fitted team volume, full injury boost) win | v4 (current) win | paired difference (player-clustered 95% CI) | OVER share v3 (fitted team volume, full injury boost) -> v4 (current) |
|---|--:|--:|--:|---|---|
| all lines | 3964 | 51.9% | 52.4% | +0.48 pts [-0.18, +1.17] (n=3922) | 62% -> 63% |
| projection moved >= 1% | 2122 | 50.9% | 51.7% | +0.76 pts [-0.39, +1.87] (n=2096) | 66% -> 67% |
| projection moved >= 5% | 472 | 54.8% | 54.8% | +0.00 pts [-3.19, +3.15] (n=465) | 69% -> 62% |
| projection moved >= 10% | 105 | 53.8% | 57.7% | +3.85 pts [-7.41, +12.90] (n=104) | 87% -> 64% |
| lean flipped by v4 (current) | 164 | 44.0% | 56.0% | +11.95 pts [-5.13, +28.66] (n=159) | 33% -> 67% |
| projection moved < 1% | 1842 | 53.1% | 53.2% | +0.16 pts [-0.56, +0.91] (n=1826) | 57% -> 59% |

## Tier tables (all graded lines, win % and ROI at -110)

Break-even win % by tier at the usual price: lean 52.4, solid 53.0, strong 53.7, high 55.4.

**As stored (what the screen showed)**

All bets:

| conf | n | win% | ROI@-110% |
|---|---|---|---|
| — | 1653 | 48.4 | -7.6 |
| lean | 951 | 52.7 | 0.5 |
| solid | 760 | 51.7 | -1.4 |
| strong | 463 | 54.3 | 3.8 |
| high | 137 | 53.3 | 1.7 |

Recommended only:

| conf | n | of all | win% | ROI@-110% |
|---|---|---|---|---|
| lean | 619 | 951 | 52.3 | -0.1 |
| solid | 326 | 760 | 52.6 | 0.5 |
| strong | 56 | 463 | 51.8 | -1.1 |
| high | 1 | 137 | 100.0 | 90.9 |

**v2 (WR/TE recalibration), restated**

All bets:

| conf | n | win% | ROI@-110% |
|---|---|---|---|
| — | 1798 | 50.4 | -3.8 |
| lean | 1084 | 51.1 | -2.5 |
| solid | 716 | 53.7 | 2.4 |
| strong | 293 | 50.9 | -2.9 |
| high | 73 | 54.9 | 4.9 |

Recommended only:

| conf | n | of all | win% | ROI@-110% |
|---|---|---|---|---|
| lean | 772 | 1084 | 50.9 | -2.8 |
| solid | 364 | 716 | 56.2 | 7.3 |
| strong | 69 | 293 | 40.6 | -22.5 |
| high | 2 | 73 | 50.0 | -4.5 |

**v3 (fitted team volume, full injury boost), restated**

All bets:

| conf | n | win% | ROI@-110% |
|---|---|---|---|
| — | 1892 | 50.3 | -3.9 |
| lean | 1045 | 53.2 | 1.7 |
| solid | 723 | 51.5 | -1.6 |
| strong | 247 | 59.2 | 13.0 |
| high | 57 | 52.6 | 0.5 |

Recommended only:

| conf | n | of all | win% | ROI@-110% |
|---|---|---|---|---|
| lean | 773 | 1045 | 53.7 | 2.6 |
| solid | 357 | 723 | 51.8 | -1.0 |
| strong | 42 | 247 | 52.4 | 0.0 |
| high | 2 | 57 | 100.0 | 90.9 |

**v4 (current), restated**

All bets:

| conf | n | win% | ROI@-110% |
|---|---|---|---|
| — | 1898 | 50.9 | -2.8 |
| lean | 1043 | 53.6 | 2.3 |
| solid | 710 | 52.6 | 0.5 |
| strong | 260 | 57.9 | 10.6 |
| high | 53 | 52.8 | 0.9 |

Recommended only:

| conf | n | of all | win% | ROI@-110% |
|---|---|---|---|---|
| lean | 757 | 1043 | 53.3 | 1.7 |
| solid | 337 | 710 | 52.2 | -0.3 |
| strong | 42 | 260 | 50.0 | -4.5 |
| high | 1 | 53 | 100.0 | 90.9 |

## Filters and role agreement

- v1 (original): 3964 rows -> 2092 pass the always-on guard -> 1097 Recommended; role_ratio median 0.95, 68% inside the 0.85-1.20 band
- v2 (WR/TE recalibration): 3964 rows -> 1881 pass the always-on guard -> 1207 Recommended; role_ratio median 1.03, 76% inside the 0.85-1.20 band
- v3 (fitted team volume, full injury boost): 3964 rows -> 1806 pass the always-on guard -> 1174 Recommended; role_ratio median 1.03, 77% inside the 0.85-1.20 band
- v4 (current): 3964 rows -> 1794 pass the always-on guard -> 1137 Recommended; role_ratio median 1.02, 76% inside the 0.85-1.20 band

- OVER share by market, v1 (original): pass TD 55%, pass yds 61%, rec yds 80%, receptions 76%, rush yds 68%
- OVER share by market, v2 (WR/TE recalibration): pass TD 55%, pass yds 61%, rec yds 69%, receptions 53%, rush yds 68%
- OVER share by market, v3 (fitted team volume, full injury boost): pass TD 52%, pass yds 62%, rec yds 73%, receptions 49%, rush yds 67%
- OVER share by market, v4 (current): pass TD 52%, pass yds 62%, rec yds 75%, receptions 51%, rush yds 67%
