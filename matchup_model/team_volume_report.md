# Team pass / rush attempts: the fitted volume model vs the pre-v3 hand-set formula

1344 team-games, 2023-25 weeks 4-18, each season scored with coefficients fit on the seasons before it (training 2021 on; 2780 team-games in all).

| target | model | RMSE | correlation | mean error |
|---|---|--:|--:|--:|
| pass attempts | league average | 7.703 | n/a | +0.83 |
| pass attempts | team's own trailing average | 7.719 | 0.220 | +0.14 |
| pass attempts | pre-v3 hand-set formula | 8.015 | 0.177 | +0.71 |
| pass attempts | fitted regression (v3) | 7.451 | 0.241 | -0.00 |
| rush attempts | league average | 7.455 | n/a | +0.05 |
| rush attempts | team's own trailing average | 7.498 | 0.219 | -0.09 |
| rush attempts | pre-v3 hand-set formula | 8.114 | 0.108 | -1.67 |
| rush attempts | fitted regression (v3) | 7.124 | 0.298 | +0.29 |

- pass attempts: fitted vs hand-set -7.0% RMSE, vs league average -3.3%; the hand-set projections were too spread out (actual moved 0.34 as much as projected).

- rush attempts: fitted vs hand-set -12.2% RMSE, vs league average -4.4%; the hand-set projections were too spread out (actual moved 0.22 as much as projected).

By season (RMSE, hand-set formula -> fitted):

| season | pass attempts | rush attempts |
|---|--:|--:|
| 2023 | 7.76 -> 7.28 | 8.00 -> 7.13 |
| 2024 | 8.18 -> 7.61 | 7.90 -> 6.80 |
| 2025 | 8.10 -> 7.46 | 8.43 -> 7.43 |

## The spread sign

`games()` stores the betting spread (negative = favourite). The hand-set formula assumed positive = favourite, so it projected favourites to pass more and run less than underdogs; real games go the other way for rushing and are flat for attempts. Mean attempts / carries by spread bucket:

| spread | games | pass att actual | hand-set | fitted | rush att actual | hand-set | fitted |
|---|--:|--:|--:|--:|--:|--:|--:|
| fav 7+ | 188 | 31.9 | 35.2 | 32.9 | 29.6 | 24.2 | 29.4 |
| fav 3-7 | 320 | 32.2 | 34.4 | 32.9 | 28.3 | 24.9 | 28.2 |
| fav 0-3 | 164 | 33.0 | 33.8 | 32.9 | 27.0 | 25.0 | 27.6 |
| dog 0-3 | 261 | 33.0 | 32.8 | 32.5 | 26.9 | 25.9 | 27.0 |
| dog 3-7 | 263 | 34.6 | 32.8 | 32.9 | 25.2 | 25.8 | 26.0 |
| dog 7+ | 148 | 31.7 | 31.8 | 32.8 | 23.9 | 26.0 | 24.7 |

## Coefficients of the fit through 2025 (per 1 SD of the feature; used for 2026)

| feature | pass attempts | rush attempts |
|---|--:|--:|
| dome | -0.12 | +0.14 |
| is_home | +0.06 | +0.08 |
| op_df_att | +0.73 |  |
| op_df_car |  | +0.73 |
| op_plays | +0.18 | +0.24 |
| spread | -0.23 | -0.94 |
| total_line | +0.84 | -0.42 |
| tr_att | +2.25 |  |
| tr_car |  | +1.56 |
| tr_plays | -0.45 | -0.66 |
| wind | -0.46 | +0.22 |
| (intercept) | 33.17 | 26.96 |
