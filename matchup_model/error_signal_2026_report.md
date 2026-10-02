# Error-signal audit — 2026 weeks 1-3

## Part A — defense faced that week (per-matchup, rec/receptions/pass props)

n=950 graded props with a confirmed opponent-defense row. 38 feature x target tests run — at p<0.05 alone, expect ~1.9 false positives by chance; treat anything above that bar, or without p<0.01, as a lead to re-check next week, not a finding.

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

## Part B — player's own season-to-date profile (rec/receptions props, n>=3 graded each)

n=122 players. 14 feature x target tests — same chance-noise caveat as Part A.

| feature | vs | n | r | p |
|---|---|--:|--:|--:|
| YPRR | mean_z_error | 122 | 0.601 | 0.0 |
| inline_rte_pct | mean_z_error | 122 | -0.23 | 0.0109 |
| wide_rte_pct | mean_z_error | 122 | 0.181 | 0.0461 |
| YPRR | mean_abs_z_error | 122 | 0.141 | 0.1203 |
| pers11_tgt_share | mean_z_error | 122 | 0.133 | 0.143 |
| slot_rte_pct | mean_abs_z_error | 122 | -0.096 | 0.2945 |
| man_tgt_share | mean_z_error | 122 | 0.087 | 0.3415 |
| wide_rte_pct | mean_abs_z_error | 122 | 0.04 | 0.6595 |
| inline_rte_pct | mean_abs_z_error | 122 | 0.032 | 0.7262 |
| slot_rte_pct | mean_z_error | 122 | 0.032 | 0.7297 |
| man_tgt_share | mean_abs_z_error | 122 | -0.021 | 0.8225 |
| pers11_tgt_share | mean_abs_z_error | 122 | 0.015 | 0.8671 |