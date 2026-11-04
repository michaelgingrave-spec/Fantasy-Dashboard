# Error-signal audit — 2026 weeks 1-8
(all player-profile features built walk-forward from strictly-prior-week exports only -- see module docstring)

## Part A — defense faced that week

n=2521 props. 38 tests — expect ~1.9 false positives at p<0.05 by chance; treat p<0.01 as a lead, not a finding.

| feature | vs | n | r | p |
|---|---|--:|--:|--:|
| PRESS % (TGT) | abs_z_error | 2521 | 0.046 | 0.0199 |
| PRESS % (ANY) | abs_z_error | 2521 | 0.044 | 0.028 |
| COVER 2 % | abs_z_error | 2521 | 0.041 | 0.0373 |
| TO 2-HI % | z_error | 2521 | -0.033 | 0.1008 |
| PRESS % (ANY) | z_error | 2521 | -0.033 | 0.1025 |
| COVER 4 % | abs_z_error | 2521 | -0.029 | 0.1406 |
| COVER 1 % | z_error | 2521 | -0.027 | 0.1692 |
| DISGUISE % | z_error | 2521 | -0.026 | 0.1888 |
| BASE % | z_error | 2521 | 0.026 | 0.1959 |
| COVER 2 % | z_error | 2521 | -0.023 | 0.2587 |
| PRESS % (TGT) | z_error | 2521 | -0.023 | 0.254 |
| MAN % | z_error | 2521 | -0.02 | 0.3164 |
| DISGUISE % | abs_z_error | 2521 | 0.02 | 0.3267 |
| COVER 3 % | z_error | 2521 | 0.02 | 0.3106 |
| 1-HI/MOF C % | abs_z_error | 2521 | -0.018 | 0.3748 |
| NICKEL % | z_error | 2521 | -0.018 | 0.3682 |
| COVER 2 MAN % | abs_z_error | 2521 | 0.018 | 0.363 |
| 2-HI/MOF O % | abs_z_error | 2521 | 0.018 | 0.3762 |
| COVER 2 MAN % | z_error | 2521 | -0.017 | 0.4054 |
| COVER 4 % | z_error | 2521 | 0.016 | 0.4322 |
| TO 1-HI % | abs_z_error | 2521 | 0.016 | 0.4231 |
| COVER 0 % | abs_z_error | 2521 | 0.015 | 0.4661 |
| COVER 3 % | abs_z_error | 2521 | -0.014 | 0.4875 |
| COVER 6 % | z_error | 2521 | -0.013 | 0.5188 |
| COVER 0 % | z_error | 2521 | 0.01 | 0.6256 |
| ZONE % | abs_z_error | 2521 | -0.009 | 0.6681 |
| BASE % | abs_z_error | 2521 | -0.009 | 0.652 |
| COVER 1 % | abs_z_error | 2521 | -0.009 | 0.6419 |
| ZONE % | z_error | 2521 | 0.008 | 0.7005 |
| DIME % | abs_z_error | 2521 | 0.008 | 0.6819 |
| DIME % | z_error | 2521 | -0.007 | 0.7192 |
| MAN % | abs_z_error | 2521 | 0.005 | 0.7976 |
| NICKEL % | abs_z_error | 2521 | -0.003 | 0.8938 |
| TO 1-HI % | z_error | 2521 | -0.003 | 0.8951 |
| TO 2-HI % | abs_z_error | 2521 | 0.003 | 0.8843 |
| 1-HI/MOF C % | z_error | 2521 | -0.001 | 0.9551 |
| 2-HI/MOF O % | z_error | 2521 | 0.001 | 0.9473 |
| COVER 6 % | abs_z_error | 2521 | 0.001 | 0.9622 |

## Part B — trailing receiving profile (YPRR, aDOT, man-coverage share)

n=220 player-weeks. 6 tests.

| feature | vs | n | r | p |
|---|---|--:|--:|--:|
| YPRR | mean_abs_z_error | 220 | 0.125 | 0.0632 |
| aDOT | mean_z_error | 217 | 0.117 | 0.0848 |
| man_tgt_share | mean_z_error | 220 | 0.115 | 0.0877 |
| YPRR | mean_z_error | 220 | 0.065 | 0.3392 |
| aDOT | mean_abs_z_error | 217 | 0.06 | 0.3768 |
| man_tgt_share | mean_abs_z_error | 220 | 0.03 | 0.6556 |

## Part C — trailing rushing-efficiency profile (success rate, stuff rate, YBC/att, YACO/att, EPA/att, MTF/att, hit rate)

n=89 player-weeks. 14 tests.

| feature | vs | n | r | p |
|---|---|--:|--:|--:|
| SUCC % | mean_z_error | 89 | 0.241 | 0.0232 |
| EPA/A | mean_z_error | 89 | 0.238 | 0.0248 |
| STUFF % | mean_z_error | 89 | -0.17 | 0.1109 |
| YACO/ATT | mean_z_error | 89 | 0.124 | 0.2468 |
| MTF/A | mean_abs_z_error | 89 | -0.115 | 0.2814 |
| YBC/ATT | mean_z_error | 89 | 0.075 | 0.4867 |
| HIT % | mean_abs_z_error | 89 | -0.073 | 0.4963 |
| YACO/ATT | mean_abs_z_error | 89 | -0.06 | 0.5735 |
| EPA/A | mean_abs_z_error | 89 | 0.043 | 0.6905 |
| YBC/ATT | mean_abs_z_error | 89 | -0.026 | 0.8058 |
| STUFF % | mean_abs_z_error | 89 | -0.025 | 0.8162 |
| HIT % | mean_z_error | 89 | -0.023 | 0.833 |
| MTF/A | mean_z_error | 89 | 0.022 | 0.8374 |
| SUCC % | mean_abs_z_error | 89 | -0.007 | 0.9465 |

## Not tested — needs a new Data Suite pull, not just new code

- **Receiver alignment** (wide/slot/inline): only a season-cumulative 2026 file exists (`receiving-alignment_defense_2026.csv`), no per-week pull -- same leaky shape that broke the YPRR shrinkage test, so it's left out rather than reported on a source already shown unreliable.
- **Play-callers / coordinators**: zero 2026 data pulled at all -- the 5 coordinator-split tables (head coach, OC, DC, playcaller) only exist for the 2022-2024 historical backtest seasons.