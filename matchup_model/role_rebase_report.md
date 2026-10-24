# Does the with / without-teammate split beat the trailing window when a starter is Out?

2022-26 nflverse box scores, snap counts and injury reports; teammates who played in a team-week where an Out starter vacates >= 12% of the unit's trailing share; leave-one-season-out. `K` is model v4's rule, `WO` the teammate's share in the last 10 team games the key player missed, `SINCE` his share over the unbroken run of games the key player has missed, `BL(n)` the split shrunk toward K by `n` pseudo-games. RMSE is in points of the team's target share (receiving) or carry share (rushing), on the teammates who have at least one game without the key player.

## Receiving (target share, WR/TE/RB) - ONGOING absence

935 teammates in 152 team-weeks (of 1002 rows); on average 1.8 games without the key player.

| prediction | RMSE (points) |
|---|--:|
| A | 7.78 |
| K | 7.69 |
| WO | 8.52 |
| WO ewma | 8.51 |
| SINCE (on 913 rows with an unbroken run) | 8.68 (K on the same rows 7.72) |

WO (recency-weighted) against K: +10.6% [+6.2, +14.8].

| shrinkage n | BL(n) against K, RMSE change | 95% interval (team-week clustered) |
|--:|--:|---|
| 0.5 | +4.0% | [+0.8, +6.8] |
| 1 | +1.5% | [-0.9, +3.9] |
| 2 | -0.1% | [-2.0, +1.6] |
| 3 | -0.6% | [-2.1, +0.7] |
| 5 | -0.8% | [-1.9, +0.1] |
| 8 | -0.8% | [-1.5, -0.1] |

By season (RMSE in points):

| season | team-weeks | rows | K | WO | BL(2) |
|---|--:|--:|--:|--:|--:|
| 2022 | 39 | 254 | 7.68 | 8.60 | 7.67 |
| 2023 | 36 | 210 | 7.30 | 8.61 | 7.50 |
| 2024 | 31 | 193 | 8.60 | 9.09 | 8.48 |
| 2025 | 36 | 218 | 7.21 | 7.90 | 7.12 |
| 2026 | 10 | 60 | 7.66 | 7.93 | 7.62 |

## Receiving (target share, WR/TE/RB) - NEW absence

435 teammates in 84 team-weeks (of 1412 rows); on average 2.0 games without the key player.

| prediction | RMSE (points) |
|---|--:|
| A | 8.46 |
| K | 8.30 |
| WO | 9.98 |
| WO ewma | 9.99 |
| SINCE (on 87 rows with an unbroken run) | 9.30 (K on the same rows 9.00) |

WO (recency-weighted) against K: +20.4% [+12.7, +29.4].

| shrinkage n | BL(n) against K, RMSE change | 95% interval (team-week clustered) |
|--:|--:|---|
| 0.5 | +9.1% | [+4.0, +15.1] |
| 1 | +4.8% | [+0.7, +9.3] |
| 2 | +1.6% | [-1.4, +4.6] |
| 3 | +0.5% | [-1.9, +2.7] |
| 5 | -0.2% | [-1.9, +1.3] |
| 8 | -0.4% | [-1.6, +0.7] |

By season (RMSE in points):

| season | team-weeks | rows | K | WO | BL(2) |
|---|--:|--:|--:|--:|--:|
| 2022 | 19 | 106 | 8.49 | 10.74 | 9.07 |
| 2023 | 15 | 72 | 8.92 | 10.27 | 8.99 |
| 2024 | 21 | 106 | 8.25 | 10.28 | 8.25 |
| 2025 | 24 | 134 | 8.08 | 9.05 | 7.93 |
| 2026 | 5 | 17 | 5.93 | 9.17 | 6.62 |

## Rushing (carry share, RB) - ONGOING absence

130 teammates in 72 team-weeks (of 134 rows); on average 1.7 games without the key player.

| prediction | RMSE (points) |
|---|--:|
| A | 21.05 |
| K | 19.11 |
| WO | 18.03 |
| WO ewma | 17.79 |
| SINCE (on 129 rows with an unbroken run) | 18.36 (K on the same rows 19.17) |

WO (recency-weighted) against K: -6.9% [-16.3, +0.8].

| shrinkage n | BL(n) against K, RMSE change | 95% interval (team-week clustered) |
|--:|--:|---|
| 0.5 | -7.8% | [-14.7, -2.0] |
| 1 | -7.2% | [-12.7, -2.5] |
| 2 | -5.8% | [-9.6, -2.3] |
| 3 | -4.8% | [-7.8, -2.0] |
| 5 | -3.5% | [-5.7, -1.5] |
| 8 | -2.5% | [-4.0, -1.1] |

By season (RMSE in points):

| season | team-weeks | rows | K | WO | BL(2) |
|---|--:|--:|--:|--:|--:|
| 2022 | 10 | 16 | 14.60 | 16.24 | 13.80 |
| 2023 | 19 | 35 | 17.76 | 17.53 | 17.10 |
| 2024 | 17 | 34 | 20.25 | 17.86 | 19.02 |
| 2025 | 18 | 33 | 22.55 | 19.48 | 20.98 |
| 2026 | 8 | 12 | 13.78 | 15.27 | 13.03 |

## Rushing (carry share, RB) - NEW absence

44 teammates in 29 team-weeks (of 169 rows); on average 2.4 games without the key player.

| prediction | RMSE (points) |
|---|--:|
| A | 20.47 |
| K | 19.38 |
| WO | 19.69 |
| WO ewma | 19.49 |
| SINCE (on 7 rows with an unbroken run) | 18.69 (K on the same rows 14.80) |

WO (recency-weighted) against K: +0.6% [-21.9, +43.8].

| shrinkage n | BL(n) against K, RMSE change | 95% interval (team-week clustered) |
|--:|--:|---|
| 0.5 | -5.1% | [-18.0, +23.2] |
| 1 | -6.4% | [-16.1, +15.1] |
| 2 | -6.6% | [-14.0, +8.4] |
| 3 | -6.1% | [-11.7, +4.9] |
| 5 | -5.0% | [-9.3, +2.4] |
| 8 | -4.0% | [-7.2, +0.9] |

By season (RMSE in points):

| season | team-weeks | rows | K | WO | BL(2) |
|---|--:|--:|--:|--:|--:|
| 2022 | 7 | 10 | 28.61 | 28.52 | 27.53 |
| 2023 | 5 | 8 | 19.99 | 15.16 | 16.37 |
| 2024 | 9 | 15 | 15.29 | 16.49 | 14.45 |
| 2025 | 5 | 7 | 14.25 | 9.55 | 11.38 |
| 2026 | 3 | 4 | 7.26 | 22.54 | 12.81 |

## Reading (2026-10-22)

The on/off split is a small-sample statistic: a teammate has a median of about two games without the key player in the window, and one game is mostly noise (the with-out
share alone is worse than the production rule wherever he has only one such game). Receiving: no gain at any shrinkage worth shipping (the split alone is ~10% worse than K,
the shrunk blends are within 1% of K with intervals through zero). Rushing, ongoing absences only: the split shrunk toward K is about 5% better in share RMSE with the
interval clear of zero and the sign right in all five seasons, but it covers about 70 team-weeks in five seasons, roughly one team-week per NFL week (one or two backs), so
it cannot move a betting record. Not shipped. The unmeasured part of the Data Suite's play splits (the shares after a player leaves mid-game, for a NEW absence next week)
needs play-by-play participation data.

