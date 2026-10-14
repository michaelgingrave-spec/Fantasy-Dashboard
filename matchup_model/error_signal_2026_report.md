# Error-signal audit — 2026 weeks 1-5
(all player-profile features built walk-forward from strictly-prior-week exports only -- see module docstring)

## Part A — defense faced that week

n=978 props. 38 tests — expect ~1.9 false positives at p<0.05 by chance; treat p<0.01 as a lead, not a finding.

| feature | vs | n | r | p |
|---|---|--:|--:|--:|
| BASE % | z_error | 978 | 0.062 | 0.052 |
| DIME % | z_error | 978 | -0.057 | 0.0761 |
| PRESS % (ANY) | z_error | 978 | -0.056 | 0.0821 |
| COVER 2 % | abs_z_error | 978 | 0.054 | 0.0886 |
| DISGUISE % | z_error | 978 | -0.05 | 0.1213 |
| NICKEL % | abs_z_error | 978 | -0.05 | 0.1159 |
| COVER 0 % | z_error | 978 | 0.05 | 0.1147 |
| PRESS % (ANY) | abs_z_error | 978 | 0.047 | 0.1391 |
| COVER 6 % | abs_z_error | 978 | 0.045 | 0.1635 |
| TO 1-HI % | z_error | 978 | -0.045 | 0.1559 |
| COVER 3 % | abs_z_error | 978 | -0.044 | 0.1699 |
| 1-HI/MOF C % | abs_z_error | 978 | -0.041 | 0.2011 |
| 2-HI/MOF O % | abs_z_error | 978 | 0.041 | 0.2011 |
| PRESS % (TGT) | z_error | 978 | -0.036 | 0.255 |
| COVER 4 % | abs_z_error | 978 | -0.034 | 0.2946 |
| BASE % | abs_z_error | 978 | 0.031 | 0.3308 |
| COVER 1 % | z_error | 978 | -0.028 | 0.3851 |
| COVER 2 MAN % | z_error | 978 | -0.027 | 0.3918 |
| DIME % | abs_z_error | 978 | 0.024 | 0.4547 |
| COVER 2 MAN % | abs_z_error | 978 | 0.022 | 0.4925 |
| COVER 6 % | z_error | 978 | -0.02 | 0.533 |
| 1-HI/MOF C % | z_error | 978 | -0.019 | 0.5633 |
| COVER 4 % | z_error | 978 | 0.019 | 0.5425 |
| 2-HI/MOF O % | z_error | 978 | 0.019 | 0.5633 |
| PRESS % (TGT) | abs_z_error | 978 | 0.018 | 0.5751 |
| MAN % | z_error | 978 | -0.015 | 0.6457 |
| COVER 0 % | abs_z_error | 978 | 0.012 | 0.7018 |
| ZONE % | abs_z_error | 978 | -0.009 | 0.7671 |
| COVER 2 % | z_error | 978 | -0.008 | 0.7915 |
| TO 1-HI % | abs_z_error | 978 | 0.007 | 0.8216 |
| COVER 1 % | abs_z_error | 978 | -0.007 | 0.8173 |
| TO 2-HI % | abs_z_error | 978 | 0.005 | 0.8766 |
| COVER 3 % | z_error | 978 | 0.005 | 0.8786 |
| MAN % | abs_z_error | 978 | 0.003 | 0.915 |
| NICKEL % | z_error | 978 | -0.003 | 0.9149 |
| DISGUISE % | abs_z_error | 978 | 0.003 | 0.9221 |
| ZONE % | z_error | 978 | 0.002 | 0.9546 |
| TO 2-HI % | z_error | 978 | 0.001 | 0.9657 |

## Part B — trailing receiving profile (YPRR, aDOT, man-coverage share)

n=193 player-weeks. 6 tests.

| feature | vs | n | r | p |
|---|---|--:|--:|--:|
| aDOT | mean_z_error | 189 | 0.198 | 0.0062 |
| man_tgt_share | mean_z_error | 193 | 0.179 | 0.0129 |
| YPRR | mean_z_error | 193 | 0.118 | 0.1015 |
| YPRR | mean_abs_z_error | 193 | 0.042 | 0.5605 |
| aDOT | mean_abs_z_error | 189 | 0.034 | 0.647 |
| man_tgt_share | mean_abs_z_error | 193 | -0.001 | 0.9896 |

## Part C — trailing rushing-efficiency profile (success rate, stuff rate, YBC/att, YACO/att, EPA/att, MTF/att, hit rate)

n=71 player-weeks. 14 tests.

| feature | vs | n | r | p |
|---|---|--:|--:|--:|
| EPA/A | mean_abs_z_error | 71 | 0.273 | 0.0211 |
| YACO/ATT | mean_abs_z_error | 71 | 0.231 | 0.0523 |
| YACO/ATT | mean_z_error | 71 | 0.214 | 0.0732 |
| EPA/A | mean_z_error | 71 | 0.185 | 0.1234 |
| MTF/A | mean_z_error | 71 | 0.172 | 0.1506 |
| MTF/A | mean_abs_z_error | 71 | 0.127 | 0.2904 |
| SUCC % | mean_abs_z_error | 71 | -0.111 | 0.3557 |
| SUCC % | mean_z_error | 71 | 0.111 | 0.3584 |
| HIT % | mean_abs_z_error | 71 | -0.11 | 0.3626 |
| STUFF % | mean_z_error | 71 | -0.087 | 0.4725 |
| STUFF % | mean_abs_z_error | 71 | -0.072 | 0.5515 |
| YBC/ATT | mean_abs_z_error | 71 | 0.069 | 0.5648 |
| YBC/ATT | mean_z_error | 71 | 0.029 | 0.8087 |
| HIT % | mean_z_error | 71 | -0.007 | 0.9552 |

## Not tested — needs a new Data Suite pull, not just new code

- **Receiver alignment** (wide/slot/inline): only a season-cumulative 2026 file exists (`receiving-alignment_defense_2026.csv`), no per-week pull -- same leaky shape that broke the YPRR shrinkage test, so it's left out rather than reported on a source already shown unreliable.
- **Play-callers / coordinators**: zero 2026 data pulled at all -- the 5 coordinator-split tables (head coach, OC, DC, playcaller) only exist for the 2022-2024 historical backtest seasons.