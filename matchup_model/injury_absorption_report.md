# How much of a vacated share do the healthy teammates really absorb?

Evidence behind `model.INJ_REDIST` (model v4). 2022-26 nflverse box scores, snap counts and injury reports; teammates who played only; leave-one-season-out. `injuries.py` used to hand the whole vacated share to the healthy teammates in proportion to their own share (a scale of 1.0). The unit is points of the team's target share (rec) or carry share (rush); RMSE of the teammates' shares.

The no-injury expectation is `beta x` the trailing share, with `beta` the usual fall-back of an actual share from its trailing value on weeks with nobody Out or Doubtful on the team's report (learned on the other seasons); kappa is the fraction of the model's boost, `beta (B - s)`, that shows up beyond that. `kappa raw` is the same fit against the plain trailing share (beta = 1), which counts the usual fall-back against the boost and reads too low.

## Receiving (target share, WR/TE/RB)

| absence | team-weeks | teammates | mean vacated | beta | teammates' gain beyond beta / vacated | kappa (95% CI) | kappa raw | kappa by held-out season | RMSE no boost | RMSE the old full boost | RMSE damped by kappa |
|---|---|---|---|---|---|---|---|---|---|---|---|
| NEW (played the team's last game) | 239 | 1412 | 0.22 | 0.91 | +0.46 | 0.46 (+0.36 to +0.57) | 0.14 | 2022 +0.48, 2023 +0.52, 2024 +0.41, 2025 +0.47, 2026 +0.44 | 8.30 | 8.36 | 8.14 |
| ONGOING (already missed it) | 159 | 1002 | 0.20 | 0.91 | +0.31 | 0.37 (+0.21 to +0.54) | -0.04 | 2022 +0.36, 2023 +0.35, 2024 +0.38, 2025 +0.37, 2026 +0.39 | 7.88 | 8.03 | 7.80 |

Who picked up a NEW absence's vacated share beyond the usual fall-back, by position of the recipient: WR +0.36, TE +0.03, RB +0.07 (the rest went to depth players outside the rotation).

## Rushing (carry share, RB)

| absence | team-weeks | teammates | mean vacated | beta | teammates' gain beyond beta / vacated | kappa (95% CI) | kappa raw | kappa by held-out season | RMSE no boost | RMSE the old full boost | RMSE damped by kappa |
|---|---|---|---|---|---|---|---|---|---|---|---|
| NEW (played the team's last game) | 107 | 169 | 0.39 | 0.98 | +0.46 | 0.74 (+0.54 to +0.96) | 0.69 | 2022 +0.82, 2023 +0.61, 2024 +0.88, 2025 +0.70, 2026 +0.72 | 23.75 | 20.60 | 20.65 |
| ONGOING (already missed it) | 72 | 134 | 0.40 | 0.97 | +0.40 | 0.42 (+0.25 to +0.60) | 0.37 | 2022 +0.44, 2023 +0.41, 2024 +0.44, 2025 +0.41, 2026 +0.40 | 20.86 | 22.44 | 18.96 |

Who picked up a NEW absence's vacated share beyond the usual fall-back, by position of the recipient: WR +0.00, TE +0.00, RB +0.46 (the rest went to depth players outside the rotation).

## What was shipped

`model.INJ_REDIST` = `{'rec': {'new': 0.46, 'ongoing': 0.37}, 'rush': {'new': 0.74, 'ongoing': 0.42}}`: the kappa fitted above, rounded to two places. Receiving 0.46 for a new absence and 0.37 for an ongoing one, rushing 0.74 / 0.42. An ongoing absence is already partly in the teammates' trailing shares, so it earns less than a new one. The scale constants (REC_SCALE, RB_SCALE) were re-learned under it, because the old full boost had inflated the average involved player's projection and the old scales had absorbed that. Re-run this module after more weeks and compare.

## Not changed in v4: the hurt player's OWN projection

The model multiplies his own usage by P(plays) (Questionable 0.70, Doubtful 0.25). A prop line only exists if he is expected to play, and a bet on a player who does not is void, so the comparison that matters is his share GIVEN that he plays:

| unit | designation | players | played | actual / trailing share when he played | the model's own multiplier |
|---|---|---|---|---|---|
| rec | doubtful | 116 | 1% | 0.71 | 0.25 |
| rec | questionable | 1161 | 70% | 0.86 | 0.70 |
| rush | doubtful | 54 | 0% |  | 0.25 |
| rush | questionable | 351 | 64% | 0.88 | 0.70 |

Against the usual fall-back (beta, about 0.9) a ratio near 0.86-0.88 is about what a healthy player gets, so the model's 0.70 under-projects Questionable players who play (a candidate for the next version; not changed here because it also feeds the DFS screens, where P(plays) matters).

