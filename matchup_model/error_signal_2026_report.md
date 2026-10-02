# Error-signal audit — 2026 weeks 1-3
(all player-profile features built walk-forward from strictly-prior-week exports only -- see module docstring)

## Part A — defense faced that week

n=950 props. 38 tests — expect ~1.9 false positives at p<0.05 by chance; treat p<0.01 as a lead, not a finding.

| feature | vs | n | r | p |
|---|---|--:|--:|--:|
| PRESS % (ANY) | z_error | 950 | -0.061 | 0.0601 |
| COVER 2 % | abs_z_error | 950 | 0.056 | 0.0821 |
| PRESS % (ANY) | abs_z_error | 950 | 0.053 | 0.1045 |
| NICKEL % | abs_z_error | 950 | -0.052 | 0.1099 |
| COVER 0 % | z_error | 950 | 0.05 | 0.1244 |
| COVER 3 % | abs_z_error | 950 | -0.049 | 0.1301 |
| BASE % | z_error | 950 | 0.049 | 0.1341 |
| PRESS % (TGT) | z_error | 950 | -0.047 | 0.1484 |
| DIME % | z_error | 950 | -0.046 | 0.1601 |
| COVER 6 % | abs_z_error | 950 | 0.042 | 0.1992 |
| DISGUISE % | z_error | 950 | -0.042 | 0.1951 |
| 1-HI/MOF C % | abs_z_error | 950 | -0.041 | 0.2022 |
| BASE % | abs_z_error | 950 | 0.041 | 0.2097 |
| 2-HI/MOF O % | abs_z_error | 950 | 0.041 | 0.2022 |
| COVER 1 % | z_error | 950 | -0.039 | 0.2312 |
| TO 1-HI % | z_error | 950 | -0.039 | 0.2261 |
| COVER 4 % | abs_z_error | 950 | -0.033 | 0.3072 |
| COVER 2 MAN % | z_error | 950 | -0.028 | 0.3912 |
| MAN % | z_error | 950 | -0.025 | 0.4502 |
| PRESS % (TGT) | abs_z_error | 950 | 0.025 | 0.4406 |
| COVER 2 MAN % | abs_z_error | 950 | 0.024 | 0.4522 |
| 1-HI/MOF C % | z_error | 950 | -0.022 | 0.5046 |
| 2-HI/MOF O % | z_error | 950 | 0.022 | 0.5046 |
| COVER 4 % | z_error | 950 | 0.018 | 0.5745 |
| DIME % | abs_z_error | 950 | 0.017 | 0.5924 |
| ZONE % | abs_z_error | 950 | -0.016 | 0.6216 |
| COVER 0 % | abs_z_error | 950 | 0.016 | 0.6296 |
| COVER 6 % | z_error | 950 | -0.014 | 0.6631 |
| COVER 3 % | z_error | 950 | 0.011 | 0.7452 |
| ZONE % | z_error | 950 | 0.011 | 0.7339 |
| MAN % | abs_z_error | 950 | 0.01 | 0.759 |
| COVER 2 % | z_error | 950 | -0.008 | 0.7995 |
| TO 2-HI % | z_error | 950 | 0.008 | 0.8134 |
| NICKEL % | z_error | 950 | -0.004 | 0.8926 |
| TO 1-HI % | abs_z_error | 950 | 0.003 | 0.9374 |
| DISGUISE % | abs_z_error | 950 | -0.003 | 0.9153 |
| COVER 1 % | abs_z_error | 950 | -0.002 | 0.9584 |
| TO 2-HI % | abs_z_error | 950 | 0.0 | 0.9929 |

## Part B — trailing receiving profile (YPRR, aDOT, man-coverage share)

n=142 player-weeks. 6 tests.

| feature | vs | n | r | p |
|---|---|--:|--:|--:|
| YPRR | mean_z_error | 142 | 0.201 | 0.0167 |
| man_tgt_share | mean_z_error | 142 | 0.193 | 0.0214 |
| YPRR | mean_abs_z_error | 142 | 0.094 | 0.2655 |
| man_tgt_share | mean_abs_z_error | 142 | -0.076 | 0.3656 |
| aDOT | mean_z_error | 140 | 0.048 | 0.5768 |
| aDOT | mean_abs_z_error | 140 | -0.007 | 0.937 |

## Part C — trailing rushing-efficiency profile (success rate, stuff rate, YBC/att, YACO/att, EPA/att, MTF/att, hit rate)

n=61 player-weeks. 14 tests.

| feature | vs | n | r | p |
|---|---|--:|--:|--:|
| HIT % | mean_abs_z_error | 61 | -0.192 | 0.1391 |
| YACO/ATT | mean_abs_z_error | 61 | -0.147 | 0.2593 |
| HIT % | mean_z_error | 61 | 0.146 | 0.2606 |
| YACO/ATT | mean_z_error | 61 | 0.14 | 0.2829 |
| EPA/A | mean_abs_z_error | 61 | 0.136 | 0.2973 |
| MTF/A | mean_z_error | 61 | 0.131 | 0.3153 |
| YBC/ATT | mean_abs_z_error | 61 | 0.126 | 0.3323 |
| EPA/A | mean_z_error | 61 | 0.099 | 0.4487 |
| SUCC % | mean_abs_z_error | 61 | 0.074 | 0.5721 |
| SUCC % | mean_z_error | 61 | 0.05 | 0.7015 |
| MTF/A | mean_abs_z_error | 61 | 0.032 | 0.8046 |
| STUFF % | mean_abs_z_error | 61 | -0.028 | 0.8279 |
| STUFF % | mean_z_error | 61 | -0.027 | 0.836 |
| YBC/ATT | mean_z_error | 61 | -0.004 | 0.975 |

## Not tested — needs a new Data Suite pull, not just new code

- **Receiver alignment** (wide/slot/inline): only a season-cumulative 2026 file exists (`receiving-alignment_defense_2026.csv`), no per-week pull -- same leaky shape that broke the YPRR shrinkage test, so it's left out rather than reported on a source already shown unreliable.
- **Play-callers / coordinators**: zero 2026 data pulled at all -- the 5 coordinator-split tables (head coach, OC, DC, playcaller) only exist for the 2022-2024 historical backtest seasons.