# Game simulator: how well does it price parlays?

`matchup_model.sim` plays a game 20,000 times from the model's own projections (means) and a small set of noise parameters, and `sgp.price` turns the paths into a parlay's chance and fair odds. This report is the test of it on games and lines it was **not** fitted to. The noise parameters were tuned on 2023-24 box scores only (`calibrate.py`, a coordinate search on the match to the real correlations between players' stats and to how spread out outcomes are); 2025 and 2026 are untouched by that tuning, **except** that the two input corrections of 2026-11-05 (finding 1) were chosen after comparing them with the first setting on these same seasons, so read their gain as partly selected. Every game is simulated from the projections the app would have shown beforehand (walk-forward; `history.py`).

**How a parlay is priced.** Each leg is pinned to the book's no-vig probability (the books' own single-leg prices), and only the *association* between legs, the odds ratio of the simulator's 2x2 table, is taken from the simulator (`sgp.price`, `odds_ratio_fit`). The test below asks whether that association tells us anything the single-leg prices do not: every pair of lines in each game is priced independently and with the simulator's association, and each is scored on what actually happened to the pair (log score of the realised cell). A positive gain means the simulator's dependence is informative. Intervals resample whole games.

## Findings (the run of 2026-11-05; regenerate and re-read after any change to the parameters)

1. **Two input corrections since the first run (2026-11-04), and what they changed.** The first run scaled every player down to the volume a team actually has (`vol_pass` .935, `vol_rush` .9175), which put each role player 6-9% under his own projection, and gave the quarterback the sum of his receivers' lines, which put his passing yards about 10% over his own projection. Now every player's mean is his projection, and a starting quarterback's line is set to his (a quarterback projected for under 60% of the team's passing is a backup, weighted by his chance to play: if he is in the simulation he throws the team's passes). On the same held-out lines, for role players, the simulator's own chance of the over improved in both seasons and in every market: Brier .2625 -> .2586 in 2025 (-.0038 [-.0051, -.0027]) and .2558 -> .2526 in 2026 (-.0032 [-.0049, -.0015]); the average chance it gives an over (all lines) rose from 39% to 43% (2025) and from 42% to 45% (2026), against 47.0% and 48.6% that happened. The association between legs did not change (same-team pairs with a quarterback, 2026: +0.0102 -> +0.0105 per pair; every old-minus-new difference in the pair gain is within +/-0.0004 with intervals including 0). Quarterback passing-yards lines did not improve (206 lines in 2026, +0.0040 [-0.0183, +0.0281]): the pin makes his average his projection (it ran 10% over), it is not shown to help prices. Caveats: the pinned-means setting was picked after comparing it with the first setting on these same 2025-26 lines; the 2023-24 seasons, which did not pick it, agree (bias of the simulated mean against what happened: WR1 receiving yards -16.6% -> -10.8%, lead-back carries -9.9% -> -1.8%, quarterback yards +9.7% -> +1.7%; the spread of outcomes is mixed: the standardised-residual variance (ideal 1) is closer to 1 in 8 of the 16 key rows, unchanged in 4 and further in 4 (quarterback yards 1.16 -> 1.31, so his simulated spread is now about 13% too narrow, it was 7%; lead-back rushing yards .93 -> .81, tight end receiving yards .94 -> .87 and WR1 yards 1.03 -> .94, spreads a few per cent too wide), and the variance term of the tuning loss as a whole improved, .149 -> .101); the noise parameters were NOT re-tuned. Cost: the simulated TEAM volumes (team pass attempts, yards, carries) now run 6-13% above history on 2023-24, since they are the sum of the players' projections; the screen does not use them.
2. **Where it helps: a quarterback and his receivers.** On 2026 lines (121 games) the 7,906 same-team pairs that include a quarterback gain +0.0105 log-score per pair over independence [+0.0074, +0.0141]. For pass yards with a receiver's yards the gain is +0.0163 [+0.0108, +0.0219] (2,408 pairs) and with receptions +0.0116 [+0.0069, +0.0162]; both legs went over together 30% of the time against 25% if independent (the simulator said 28%).
3. **A simple correlation table does almost as well for those pairs.** One fixed correlation per kind of pair (estimated on the same 2023-24 games) gains +0.0265 per quarterback-receiver pair against the simulator's +0.0322 (simulator better by +0.0058 [+0.0013, +0.0103]); the two tie elsewhere (running-back pairs: simulator +0.0017 [-0.0005, +0.0041] over the table, 209 pairs). The simulator's edge over the table is consistency across any set of legs (touchdowns, totals, three-leg combinations) and context (a high total, a heavy favourite), which these tests cannot measure.
4. **Nowhere else is there anything to gain, and it costs almost nothing.** Teammate receivers' yards and receptions are close to uncorrelated in real games (about 0 either way), and so are opposing players': the simulator's gain over independence there is -0.0001 to -0.0006 a pair.
5. **2025 cannot test the quarterback**: no pass yards or pass TD lines were pulled for it, so its null result is the skill-player pairs, whose true correlation is about zero.
6. **The simulator's own single-leg probabilities are better than they were but still not good enough to use alone, so pin the legs to the book's odds.** It says the over hits 43% (2025) and 45% (2026) when the books said 49.6% and 49.5% and 47.0% and 48.6% went over (before the corrections 39% and 42%); Brier .2592 and .2550 against the market's .2476 and .2484 on all lines (before: .2625 and .2580), and on role players .2586 and .2526 against .2473 and .2483 (the book is ahead by +.0113 [+.0063, +.0161] in 2025 and +.0043 [+.0001, +.0083] in 2026; on 2026 receiving-yards lines they tie, .2496 against .2500). It is still low on overs because the model's level sits under the books' lines. The screen uses the simulator's number only for a leg with no odds, and says so.
7. **The books' no-vig prices lean towards the under.** Overs hit less often than priced (47.0% against 49.6% in 2025; 48.6% against 49.5% in 2026), so three-over parlays hit 0.82x as often as the product of their no-vig probabilities in 2025 and 1.02x in 2026, three-under parlays 1.17x and 1.08x. A market-pinned price inherits the books' numbers and therefore this lean; it is a reason to prefer under legs, and not something the simulator corrects.
8. **Touchdown, spread and total legs have no historical lines here**, so they are checked only through the correlation tests above (team points with a quarterback's yards .30 real against .38 simulated in 2025 and .35 against .38 in 2026; his yards with his touchdowns .44 against .45 in 2025 and .36 against .45 in 2026).
9. **Edge against the books' own same-game-parlay prices is untested**: those prices are not available through the Odds API. The screen takes the price the book offers and shows the expected return against the fair price; logging offers and results is the way to test it.
10. **A leg the simulator thinks impossible has no association to give, and the pricing now says so.** One 2026 game had a depth receiver projected for 1 yard against a 15-yard line (0.04% in the simulator); an odds ratio built from three paths is noise, and it alone moved the 2026 all-pairs gain from +0.0013 to 0.0000 a pair. `odds_ratio_fit` and `price` now give every cell half a path first (the usual continuity correction), and the screen warns when the simulator and the book are 30 points apart or the simulator gives a leg under 2% or over 98%.
11. **Limits the screen carries.** If a team lists two quarterbacks (a doubtful starter and a backup, both projected for a full game) only the one with more projected attempts gets the passing line, so remove the other. A late scratch's targets and carries are not handed to his teammates, who keep projections that already reflect the official injury report. A starter who leaves the game early is not modelled.

## Parameters

`fg_mean` 1.35, `fg_slope` 0.03, `misc_pts` 0.35, `sd_td` 1.278, `sd_fg` 1, `fg_td_corr` 0.3, `a_yards` 0.63, `a_script` 0.3, `def_td_p` 0.05, `vol_pass` 1, `vol_rush` 1, `pass_scale` 1.0355, `qb_pin` 1, `qb_pin_min` 0.6, `pass_G` 0.075, `rush_G` 0.03, `pass_script` 0.09, `rush_script` 0.189, `pass_sd` 0.156, `rush_sd` 0.2366, `kappa_tgt` 60, `kappa_car` 26.67, `cr_eff` 0.15, `cr_sd` 0.1, `rec_shape` 1.7, `ypc_eff` 0.0765, `ypc_sd` 0.22, `rush_shape` 1.231, `ypc_r_eff` 0.07, `ypc_r_sd` 0.338, `rush_rho` 0, `td_pass_link` 0.98, `int_eff` 0.25, `int_script` 0.12

## 2025 (240 games, 5566 lines)

### Single legs: the simulator's chance of the over against the book's and against what happened

| market | n | went over | market said | sim said | Brier market | Brier sim | Brier market/sim 50-50 |
|---|---|---|---|---|---|---|---|
| all | 5224 | 0.47 | 0.496 | 0.427 | 0.2476 | 0.2592 | 0.2482 |
| rec yds | 2136 | 0.475 | 0.5 | 0.447 | 0.25 | 0.2569 | 0.2498 |
| receptions | 2036 | 0.457 | 0.491 | 0.426 | 0.2441 | 0.2529 | 0.2437 |
| rush yds | 1052 | 0.483 | 0.497 | 0.388 | 0.2496 | 0.2762 | 0.2537 |

### Pairs: log-score gain of the simulator's association over independence (positive = better)

| pairs | n | gain (log score) | 95% lo | 95% hi | over+over: indep / sim / real | under+under: indep / sim / real |
|---|---|---|---|---|---|---|
| all | 63345 | -0.0006 | -0.001 | -0.0002 | 0.246 / 0.246 / 0.221 | 0.255 / 0.255 / 0.280 |

By teammates / opponents:

| pairs | n | gain (log score) | 95% lo | 95% hi | over+over: indep / sim / real | under+under: indep / sim / real |
|---|---|---|---|---|---|---|
| opp | 33973 | -0.0006 | -0.0012 | -0.0001 | 0.246 / 0.247 / 0.219 | 0.254 / 0.256 / 0.283 |
| same | 29372 | -0.0005 | -0.0012 | 0.0001 | 0.245 / 0.245 / 0.223 | 0.255 / 0.254 / 0.276 |

By pair kind:

| pairs | n | gain (log score) | 95% lo | 95% hi | over+over: indep / sim / real | under+under: indep / sim / real |
|---|---|---|---|---|---|---|
| skill players / opp | 29985 | -0.0006 | -0.0013 | -0.0001 | 0.246 / 0.247 / 0.218 | 0.254 / 0.255 / 0.285 |
| skill players / same | 25491 | -0.0003 | -0.0007 | 0.0002 | 0.246 / 0.245 / 0.220 | 0.254 / 0.254 / 0.277 |
| with a QB / opp | 3988 | -0.0004 | -0.0017 | 0.001 | 0.244 / 0.246 / 0.230 | 0.256 / 0.259 / 0.269 |
| with a QB / same | 3881 | -0.0023 | -0.0066 | 0.0002 | 0.244 / 0.243 / 0.242 | 0.256 / 0.255 / 0.270 |

By market pair:

| pairs | n | gain (log score) | 95% lo | 95% hi | over+over: indep / sim / real | under+under: indep / sim / real |
|---|---|---|---|---|---|---|
| rec yds + rec yds | 10525 | -0.0003 | -0.0005 | 0.0 | 0.250 / 0.250 / 0.228 | 0.250 / 0.250 / 0.276 |
| rec yds + receptions | 20009 | -0.0001 | -0.0003 | 0.0001 | 0.245 / 0.245 / 0.218 | 0.255 / 0.255 / 0.285 |
| rec yds + rush yds | 10742 | -0.0005 | -0.0012 | 0.0002 | 0.248 / 0.250 / 0.230 | 0.252 / 0.253 / 0.269 |
| receptions + receptions | 9518 | -0.0001 | -0.0004 | 0.0002 | 0.240 / 0.239 / 0.211 | 0.260 / 0.259 / 0.297 |
| receptions + rush yds | 10231 | -0.0008 | -0.002 | 0.0002 | 0.243 / 0.245 / 0.219 | 0.257 / 0.258 / 0.274 |
| rush yds + rush yds | 2320 | -0.0072 | -0.0173 | 0.0009 | 0.247 / 0.246 / 0.225 | 0.253 / 0.253 / 0.252 |

### Against a simple control: one correlation per pair class

The control is a Gaussian copula with a single correlation for each kind of pair (QB-recv 0.31, QBTD-recv 0.12, QB-rush -0.13, recv-recv -0.01, rush-recv -0.02, rush-rush 0.05), estimated from the same 2023-24 games, with the books' single-leg numbers as margins. Pairs outside these classes (and opponents) are treated as independent.

| pairs | n | simulator vs independent | control vs independent | simulator vs control |
|---|---|---|---|---|
| all | 63345 | -0.0006 [-0.0010, -0.0002] | +0.0000 [-0.0000, +0.0001] | -0.0006 [-0.0010, -0.0002] |

| pairs | n | simulator vs independent | control vs independent | simulator vs control |
|---|---|---|---|---|
| recv-recv | 18337 | -0.0001 [-0.0003, +0.0002] | -0.0000 [-0.0001, +0.0001] | -0.0001 [-0.0003, +0.0002] |
| rush-recv | 6761 | -0.0006 [-0.0023, +0.0012] | +0.0003 [-0.0000, +0.0006] | -0.0009 [-0.0026, +0.0008] |
| rush-rush | 393 | -0.0027 [-0.0101, +0.0052] | +0.0005 [-0.0025, +0.0040] | -0.0033 [-0.0095, +0.0034] |

### Three-leg parlays: the chance they all hit together

| parlay | n | independent | simulated | realised | ratio real/indep | ratio sim/indep |
|---|---|---|---|---|---|---|
| OVER x3 | 12091 | 0.122 | 0.1222 | 0.0996 | 0.82 | 1.0 |
| UNDER x3 | 12091 | 0.1281 | 0.1283 | 0.1499 | 1.17 | 1.0 |

### The correlations the simulator implies against real ones (PIT correlation across games)

| a | b | rel | n | actual | +/- | simulated | gap |
|---|---|---|---|---|---|---|---|
| QB.pass_yds | WR1.rec_yds | same | 390 | 0.361 | 0.086 | 0.468 | 0.107 |
| QB.pass_yds | WR2.rec_yds | same | 379 | 0.291 | 0.092 | 0.375 | 0.085 |
| QB.pass_yds | WR3.rec_yds | same | 335 | 0.211 | 0.102 | 0.287 | 0.076 |
| QB.pass_yds | TE1.rec_yds | same | 373 | 0.335 | 0.09 | 0.341 | 0.006 |
| QB.pass_yds | RB1.rec_yds | same | 387 | 0.253 | 0.093 | 0.197 | -0.055 |
| QB.pass_td | WR1.any_td | same | 390 | 0.339 | 0.088 | 0.27 | -0.069 |
| QB.pass_td | WR1.rec_yds | same | 390 | 0.12 | 0.098 | 0.219 | 0.099 |
| QB.pass_yds | QB.pass_td | same | 410 | 0.439 | 0.078 | 0.448 | 0.009 |
| QB.comp | WR1.rec | same | 390 | 0.334 | 0.088 | 0.387 | 0.053 |
| WR1.rec_yds | WR2.rec_yds | same | 411 | -0.048 | 0.097 | 0.012 | 0.06 |
| WR1.rec_yds | TE1.rec_yds | same | 408 | 0.064 | 0.097 | 0.01 | -0.054 |
| WR2.rec_yds | WR3.rec_yds | same | 353 | -0.024 | 0.104 | 0.007 | 0.031 |
| WR1.rec | WR2.rec | same | 411 | -0.059 | 0.096 | -0.009 | 0.05 |
| WR1.tgt | WR2.tgt | same | 411 | -0.022 | 0.097 | -0.024 | -0.001 |
| WR1.any_td | WR2.any_td | same | 411 | -0.081 | 0.096 | -0.016 | 0.065 |
| RB1.rush_yds | QB.pass_yds | same | 387 | -0.095 | 0.099 | -0.074 | 0.021 |
| RB1.rush_yds | WR1.rec_yds | same | 420 | 0.072 | 0.095 | -0.035 | -0.107 |
| RB1.rush_yds | RB2.rush_yds | same | 385 | 0.005 | 0.1 | 0.038 | 0.033 |
| RB1.rush_yds | RB1.rec_yds | same | 449 | 0.028 | 0.093 | -0.024 | -0.052 |
| RB1.rush_att | QB.pass_att | same | 387 | -0.156 | 0.097 | -0.171 | -0.015 |
| RB1.any_td | QB.pass_td | same | 387 | -0.078 | 0.099 | -0.032 | 0.047 |
| RB1.any_td | WR1.any_td | same | 420 | -0.036 | 0.096 | -0.037 | -0.001 |
| QB.pass_yds | QB.pass_yds | opp | 172 | 0.196 | 0.144 | -0.022 | -0.219 |
| WR1.rec_yds | WR1.rec_yds | opp | 209 | 0.051 | 0.136 | -0.005 | -0.056 |
| RB1.rush_yds | RB1.rush_yds | opp | 211 | -0.194 | 0.13 | -0.095 | 0.099 |
| QB.pass_yds | WR1.rec_yds | opp | 193 | 0.098 | 0.14 | -0.01 | -0.109 |
| RB1.rush_yds | QB.pass_yds | opp | 188 | 0.06 | 0.143 | 0.096 | 0.036 |
| WR1.any_td | WR1.any_td | opp | 209 | 0.036 | 0.136 | 0.001 | -0.035 |

Team score against player stats:

| a | b | n | actual | +/- | simulated | gap |
|---|---|---|---|---|---|---|
| points | QB.pass_yds | 410 | 0.301 | 0.088 | 0.377 | 0.076 |
| points | WR1.rec_yds | 448 | 0.191 | 0.089 | 0.185 | -0.006 |
| points | RB1.rush_yds | 449 | 0.251 | 0.087 | 0.25 | -0.001 |
| points | QB.pass_td | 410 | 0.53 | 0.07 | 0.613 | 0.083 |
| margin | RB1.rush_yds | 449 | 0.295 | 0.085 | 0.223 | -0.072 |
| margin | QB.pass_yds | 410 | 0.087 | 0.096 | 0.187 | 0.1 |
| total | QB.pass_yds | 410 | 0.343 | 0.085 | 0.346 | 0.003 |
| total | WR1.rec_yds | 448 | 0.166 | 0.09 | 0.167 | 0.0 |
| total | RB1.rush_yds | 449 | 0.074 | 0.092 | 0.126 | 0.052 |

### Calibration of each player's stat in the simulated distribution (PIT: mean .5, variance .083)

| role | stat | n | mean | var | lo10 | hi10 | z mean | z var | sim mean | actual mean | bias % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| QB | comp | 410 | 0.419 | 0.103 | 0.232 | 0.098 | -0.364 | 1.731 | 21.487 | 19.563 | 9.835 |
| QB | int | 410 | 0.498 | 0.08 | 0.083 | 0.098 | -0.018 | 1.112 | 0.696 | 0.641 | 8.525 |
| QB | pass_att | 410 | 0.452 | 0.115 | 0.232 | 0.124 | -0.26 | 2.398 | 31.98 | 30.2 | 5.894 |
| QB | pass_td | 410 | 0.462 | 0.082 | 0.124 | 0.073 | -0.168 | 0.96 | 1.668 | 1.471 | 13.389 |
| QB | pass_yds | 410 | 0.487 | 0.091 | 0.149 | 0.093 | -0.108 | 1.29 | 227.571 | 217.805 | 4.484 |
| QB | rush_yds | 410 | 0.538 | 0.099 | 0.141 | 0.141 | 0.05 | 0.832 | 16.099 | 16.093 | 0.041 |
| RB1 | any_td | 449 | 0.497 | 0.088 | 0.118 | 0.118 | -0.023 | 0.988 | 0.419 | 0.408 | 2.886 |
| RB1 | rec | 449 | 0.51 | 0.078 | 0.091 | 0.085 | 0.017 | 0.877 | 2.436 | 2.47 | -1.366 |
| RB1 | rec_yds | 449 | 0.495 | 0.094 | 0.149 | 0.12 | 0.034 | 1.161 | 18.647 | 19.047 | -2.097 |
| RB1 | rush_att | 449 | 0.543 | 0.082 | 0.096 | 0.098 | 0.127 | 1.095 | 13.948 | 14.47 | -3.606 |
| RB1 | rush_yds | 449 | 0.533 | 0.085 | 0.102 | 0.12 | 0.107 | 1.194 | 61.649 | 64.283 | -4.097 |
| RB2 | any_td | 414 | 0.5 | 0.083 | 0.092 | 0.111 | 0.05 | 1.055 | 0.235 | 0.256 | -8.344 |
| RB2 | rec | 414 | 0.475 | 0.082 | 0.126 | 0.087 | -0.091 | 0.839 | 1.48 | 1.36 | 8.852 |
| RB2 | rec_yds | 414 | 0.455 | 0.088 | 0.128 | 0.092 | -0.094 | 0.942 | 10.811 | 9.585 | 12.799 |
| RB2 | rush_att | 414 | 0.498 | 0.103 | 0.152 | 0.147 | 0.068 | 1.522 | 6.537 | 6.681 | -2.151 |
| RB2 | rush_yds | 414 | 0.51 | 0.101 | 0.145 | 0.14 | 0.105 | 1.383 | 27.466 | 29.258 | -6.128 |
| TE1 | any_td | 438 | 0.532 | 0.086 | 0.082 | 0.144 | 0.128 | 1.2 | 0.234 | 0.283 | -17.428 |
| TE1 | rec | 438 | 0.522 | 0.076 | 0.068 | 0.091 | 0.048 | 0.848 | 3.649 | 3.731 | -2.184 |
| TE1 | rec_yds | 438 | 0.531 | 0.07 | 0.055 | 0.087 | 0.031 | 0.724 | 38.529 | 39.205 | -1.725 |
| TE1 | tgt | 438 | 0.523 | 0.072 | 0.062 | 0.089 | 0.045 | 0.826 | 5.04 | 5.121 | -1.585 |
| WR1 | any_td | 448 | 0.537 | 0.081 | 0.08 | 0.116 | 0.128 | 1.153 | 0.277 | 0.33 | -16.248 |
| WR1 | rec | 448 | 0.513 | 0.073 | 0.067 | 0.089 | 0.024 | 0.92 | 4.623 | 4.663 | -0.86 |
| WR1 | rec_yds | 448 | 0.527 | 0.074 | 0.071 | 0.092 | 0.034 | 0.794 | 58.74 | 59.725 | -1.65 |
| WR1 | tgt | 448 | 0.513 | 0.075 | 0.065 | 0.074 | 0.026 | 0.861 | 7.202 | 7.268 | -0.906 |
| WR2 | any_td | 438 | 0.516 | 0.086 | 0.107 | 0.107 | 0.09 | 1.114 | 0.2 | 0.237 | -15.766 |
| WR2 | rec | 438 | 0.51 | 0.076 | 0.075 | 0.073 | -0.02 | 0.807 | 3.165 | 3.132 | 1.054 |
| WR2 | rec_yds | 438 | 0.508 | 0.076 | 0.08 | 0.078 | -0.023 | 0.757 | 40.988 | 40.578 | 1.012 |
| WR2 | tgt | 438 | 0.503 | 0.082 | 0.11 | 0.1 | -0.008 | 0.9 | 5.122 | 5.087 | 0.7 |
| WR3 | any_td | 387 | 0.527 | 0.087 | 0.093 | 0.114 | 0.103 | 1.259 | 0.134 | 0.165 | -19.011 |
| WR3 | rec | 387 | 0.506 | 0.089 | 0.096 | 0.111 | 0.052 | 1.168 | 2.092 | 2.134 | -2.002 |
| WR3 | rec_yds | 387 | 0.492 | 0.09 | 0.119 | 0.106 | 0.011 | 1.095 | 26.82 | 26.351 | 1.777 |
| WR3 | tgt | 387 | 0.519 | 0.084 | 0.065 | 0.127 | 0.1 | 1.216 | 3.381 | 3.558 | -4.971 |

## 2026 (121 games, 3974 lines)

### Single legs: the simulator's chance of the over against the book's and against what happened

| market | n | went over | market said | sim said | Brier market | Brier sim | Brier market/sim 50-50 |
|---|---|---|---|---|---|---|---|
| all | 3646 | 0.486 | 0.495 | 0.446 | 0.2484 | 0.255 | 0.2481 |
| pass TD | 221 | 0.493 | 0.484 | 0.504 | 0.2376 | 0.2386 | 0.2326 |
| pass yds | 219 | 0.553 | 0.5 | 0.443 | 0.25 | 0.2893 | 0.2639 |
| rec yds | 1303 | 0.487 | 0.5 | 0.456 | 0.2501 | 0.2506 | 0.248 |
| receptions | 1273 | 0.488 | 0.489 | 0.436 | 0.2474 | 0.2513 | 0.2462 |
| rush yds | 630 | 0.452 | 0.498 | 0.423 | 0.2501 | 0.2657 | 0.2524 |

### Pairs: log-score gain of the simulator's association over independence (positive = better)

| pairs | n | gain (log score) | 95% lo | 95% hi | over+over: indep / sim / real | under+under: indep / sim / real |
|---|---|---|---|---|---|---|
| all | 52899 | 0.0014 | 0.0008 | 0.0019 | 0.245 / 0.248 / 0.240 | 0.256 / 0.259 / 0.268 |

By teammates / opponents:

| pairs | n | gain (log score) | 95% lo | 95% hi | over+over: indep / sim / real | under+under: indep / sim / real |
|---|---|---|---|---|---|---|
| opp | 28279 | -0.0003 | -0.0007 | 0.0001 | 0.245 / 0.246 / 0.240 | 0.256 / 0.257 / 0.268 |
| same | 24620 | 0.0033 | 0.0022 | 0.0044 | 0.245 / 0.250 / 0.239 | 0.256 / 0.261 / 0.268 |

By pair kind:

| pairs | n | gain (log score) | 95% lo | 95% hi | over+over: indep / sim / real | under+under: indep / sim / real |
|---|---|---|---|---|---|---|
| skill players / opp | 19566 | -0.0003 | -0.0007 | 0.0001 | 0.245 / 0.246 / 0.239 | 0.255 / 0.256 / 0.265 |
| skill players / same | 16714 | -0.0001 | -0.0005 | 0.0003 | 0.245 / 0.244 / 0.234 | 0.255 / 0.254 / 0.259 |
| with a QB / opp | 8713 | -0.0001 | -0.0009 | 0.0006 | 0.244 / 0.246 / 0.241 | 0.256 / 0.258 / 0.275 |
| with a QB / same | 7906 | 0.0105 | 0.0074 | 0.0141 | 0.244 / 0.264 / 0.251 | 0.256 / 0.275 / 0.289 |

By market pair:

| pairs | n | gain (log score) | 95% lo | 95% hi | over+over: indep / sim / real | under+under: indep / sim / real |
|---|---|---|---|---|---|---|
| pass TD + pass TD | 103 | 0.0076 | 0.0007 | 0.0149 | 0.237 / 0.236 / 0.301 | 0.265 / 0.264 / 0.320 |
| pass TD + pass yds | 203 | 0.0002 | -0.0056 | 0.0059 | 0.243 / 0.249 / 0.325 | 0.257 / 0.263 / 0.281 |
| pass TD + rec yds | 2434 | 0.0026 | -0.0009 | 0.0066 | 0.243 / 0.257 / 0.257 | 0.257 / 0.272 / 0.281 |
| pass TD + receptions | 2373 | 0.0026 | -0.0004 | 0.0053 | 0.237 / 0.248 / 0.256 | 0.263 / 0.274 / 0.281 |
| pass TD + rush yds | 972 | -0.0033 | -0.0062 | -0.0003 | 0.243 / 0.244 / 0.229 | 0.257 / 0.258 / 0.288 |
| pass yds + pass yds | 101 | -0.0008 | -0.0076 | 0.0052 | 0.250 / 0.248 / 0.317 | 0.250 / 0.248 / 0.208 |
| pass yds + rec yds | 2408 | 0.0163 | 0.0108 | 0.0219 | 0.250 / 0.276 / 0.304 | 0.250 / 0.277 / 0.260 |
| pass yds + receptions | 2350 | 0.0116 | 0.0069 | 0.0162 | 0.244 / 0.264 / 0.302 | 0.256 / 0.276 / 0.254 |
| pass yds + rush yds | 963 | 0.0017 | -0.0023 | 0.0051 | 0.249 / 0.253 / 0.244 | 0.251 / 0.254 / 0.242 |
| rec yds + rec yds | 6734 | -0.0008 | -0.0017 | -0.0001 | 0.249 / 0.250 / 0.244 | 0.251 / 0.251 / 0.257 |
| rec yds + receptions | 13156 | -0.0002 | -0.0006 | 0.0 | 0.244 / 0.243 / 0.241 | 0.256 / 0.256 / 0.260 |
| rec yds + rush yds | 6712 | -0.0004 | -0.0011 | 0.0002 | 0.249 / 0.250 / 0.212 | 0.251 / 0.252 / 0.283 |
| receptions + receptions | 6426 | 0.0 | -0.0003 | 0.0004 | 0.238 / 0.237 / 0.240 | 0.262 / 0.261 / 0.265 |
| receptions + rush yds | 6553 | 0.0001 | -0.0009 | 0.0009 | 0.243 / 0.244 / 0.209 | 0.257 / 0.258 / 0.280 |
| rush yds + rush yds | 1411 | 0.0037 | 0.001 | 0.0062 | 0.248 / 0.245 / 0.184 | 0.252 / 0.249 / 0.286 |

### Against a simple control: one correlation per pair class

The control is a Gaussian copula with a single correlation for each kind of pair (QB-recv 0.31, QBTD-recv 0.12, QB-rush -0.13, recv-recv -0.01, rush-recv -0.02, rush-rush 0.05), estimated from the same 2023-24 games, with the books' single-leg numbers as margins. Pairs outside these classes (and opponents) are treated as independent.

| pairs | n | simulator vs independent | control vs independent | simulator vs control |
|---|---|---|---|---|
| all | 52899 | +0.0014 [+0.0008, +0.0019] | +0.0014 [+0.0009, +0.0018] | +0.0000 [-0.0003, +0.0004] |

| pairs | n | simulator vs independent | control vs independent | simulator vs control |
|---|---|---|---|---|
| QB-recv | 2263 | +0.0322 [+0.0244, +0.0410] | +0.0265 [+0.0184, +0.0351] | +0.0058 [+0.0013, +0.0103] |
| QB-rush | 359 | +0.0036 [-0.0010, +0.0078] | +0.0052 [-0.0030, +0.0132] | -0.0015 [-0.0056, +0.0025] |
| QBTD-recv | 2289 | +0.0044 [-0.0010, +0.0103] | +0.0032 [-0.0009, +0.0076] | +0.0012 [-0.0016, +0.0037] |
| recv-recv | 12135 | -0.0001 [-0.0005, +0.0004] | -0.0000 [-0.0002, +0.0001] | -0.0001 [-0.0006, +0.0004] |
| rush-recv | 4084 | +0.0004 [-0.0006, +0.0013] | +0.0006 [+0.0002, +0.0010] | -0.0002 [-0.0009, +0.0005] |
| rush-rush | 209 | +0.0020 [-0.0025, +0.0064] | +0.0002 [-0.0039, +0.0047] | +0.0017 [-0.0005, +0.0041] |

### Three-leg parlays: the chance they all hit together

| parlay | n | independent | simulated | realised | ratio real/indep | ratio sim/indep |
|---|---|---|---|---|---|---|
| OVER x3 | 7140 | 0.1215 | 0.1262 | 0.1234 | 1.02 | 1.04 |
| UNDER x3 | 7140 | 0.1286 | 0.1335 | 0.1384 | 1.08 | 1.04 |

### The correlations the simulator implies against real ones (PIT correlation across games)

| a | b | rel | n | actual | +/- | simulated | gap |
|---|---|---|---|---|---|---|---|
| QB.pass_yds | WR1.rec_yds | same | 204 | 0.394 | 0.116 | 0.477 | 0.084 |
| QB.pass_yds | WR2.rec_yds | same | 184 | 0.331 | 0.129 | 0.389 | 0.059 |
| QB.pass_yds | WR3.rec_yds | same | 159 | 0.198 | 0.15 | 0.29 | 0.091 |
| QB.pass_yds | TE1.rec_yds | same | 204 | 0.432 | 0.112 | 0.347 | -0.085 |
| QB.pass_yds | RB1.rec_yds | same | 195 | 0.287 | 0.129 | 0.188 | -0.099 |
| QB.pass_td | WR1.any_td | same | 204 | 0.199 | 0.132 | 0.275 | 0.076 |
| QB.pass_td | WR1.rec_yds | same | 204 | 0.159 | 0.134 | 0.223 | 0.064 |
| QB.pass_yds | QB.pass_td | same | 215 | 0.358 | 0.117 | 0.449 | 0.091 |
| QB.comp | WR1.rec | same | 204 | 0.388 | 0.117 | 0.39 | 0.002 |
| WR1.rec_yds | WR2.rec_yds | same | 192 | -0.01 | 0.142 | 0.012 | 0.022 |
| WR1.rec_yds | TE1.rec_yds | same | 214 | 0.05 | 0.134 | 0.009 | -0.041 |
| WR2.rec_yds | WR3.rec_yds | same | 149 | -0.047 | 0.161 | 0.008 | 0.055 |
| WR1.rec | WR2.rec | same | 192 | -0.0 | 0.142 | -0.01 | -0.01 |
| WR1.tgt | WR2.tgt | same | 192 | 0.112 | 0.14 | -0.025 | -0.137 |
| WR1.any_td | WR2.any_td | same | 192 | -0.091 | 0.141 | -0.013 | 0.078 |
| RB1.rush_yds | QB.pass_yds | same | 195 | -0.082 | 0.14 | -0.072 | 0.01 |
| RB1.rush_yds | WR1.rec_yds | same | 207 | 0.019 | 0.137 | -0.032 | -0.051 |
| RB1.rush_yds | RB2.rush_yds | same | 167 | 0.118 | 0.15 | 0.038 | -0.081 |
| RB1.rush_yds | RB1.rec_yds | same | 221 | -0.151 | 0.129 | -0.026 | 0.125 |
| RB1.rush_att | QB.pass_att | same | 195 | -0.101 | 0.139 | -0.177 | -0.076 |
| RB1.any_td | QB.pass_td | same | 195 | -0.061 | 0.14 | -0.04 | 0.021 |
| RB1.any_td | WR1.any_td | same | 207 | 0.017 | 0.137 | -0.037 | -0.054 |
| QB.pass_yds | QB.pass_yds | opp | 95 | 0.087 | 0.201 | -0.023 | -0.11 |
| WR1.rec_yds | WR1.rec_yds | opp | 107 | -0.05 | 0.19 | -0.007 | 0.043 |
| RB1.rush_yds | RB1.rush_yds | opp | 102 | -0.267 | 0.181 | -0.097 | 0.17 |
| QB.pass_yds | WR1.rec_yds | opp | 102 | -0.055 | 0.194 | -0.011 | 0.044 |
| RB1.rush_yds | QB.pass_yds | opp | 98 | 0.131 | 0.196 | 0.095 | -0.036 |
| WR1.any_td | WR1.any_td | opp | 107 | 0.084 | 0.189 | 0.001 | -0.083 |

Team score against player stats:

| a | b | n | actual | +/- | simulated | gap |
|---|---|---|---|---|---|---|
| points | QB.pass_yds | 215 | 0.354 | 0.117 | 0.376 | 0.021 |
| points | WR1.rec_yds | 228 | 0.21 | 0.124 | 0.19 | -0.021 |
| points | RB1.rush_yds | 221 | 0.32 | 0.119 | 0.257 | -0.063 |
| points | QB.pass_td | 215 | 0.561 | 0.092 | 0.607 | 0.046 |
| margin | RB1.rush_yds | 221 | 0.339 | 0.117 | 0.228 | -0.11 |
| margin | QB.pass_yds | 215 | 0.107 | 0.132 | 0.186 | 0.079 |
| total | QB.pass_yds | 215 | 0.319 | 0.12 | 0.345 | 0.026 |
| total | WR1.rec_yds | 228 | 0.206 | 0.125 | 0.171 | -0.035 |
| total | RB1.rush_yds | 221 | 0.134 | 0.13 | 0.13 | -0.004 |

### Calibration of each player's stat in the simulated distribution (PIT: mean .5, variance .083)

| role | stat | n | mean | var | lo10 | hi10 | z mean | z var | sim mean | actual mean | bias % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| QB | comp | 215 | 0.474 | 0.102 | 0.181 | 0.112 | -0.171 | 1.733 | 21.527 | 20.623 | 4.382 |
| QB | int | 215 | 0.487 | 0.08 | 0.121 | 0.084 | -0.028 | 1.056 | 0.706 | 0.651 | 8.458 |
| QB | pass_att | 215 | 0.51 | 0.115 | 0.181 | 0.167 | 0.005 | 2.397 | 32.176 | 32.121 | 0.17 |
| QB | pass_td | 215 | 0.47 | 0.084 | 0.126 | 0.112 | -0.115 | 0.915 | 1.654 | 1.521 | 8.733 |
| QB | pass_yds | 215 | 0.517 | 0.09 | 0.135 | 0.102 | -0.012 | 1.168 | 230.062 | 228.572 | 0.652 |
| QB | rush_yds | 215 | 0.526 | 0.093 | 0.13 | 0.107 | 0.144 | 7.478 | 16.131 | 14.972 | 7.737 |
| RB1 | any_td | 221 | 0.493 | 0.091 | 0.109 | 0.118 | 0.047 | 1.023 | 0.442 | 0.462 | -4.3 |
| RB1 | rec | 221 | 0.533 | 0.082 | 0.086 | 0.14 | 0.121 | 1.011 | 2.435 | 2.624 | -7.212 |
| RB1 | rec_yds | 221 | 0.519 | 0.098 | 0.127 | 0.122 | 0.091 | 1.17 | 17.842 | 19.335 | -7.721 |
| RB1 | rush_att | 221 | 0.544 | 0.081 | 0.095 | 0.104 | 0.087 | 0.873 | 14.718 | 15.163 | -2.935 |
| RB1 | rush_yds | 221 | 0.516 | 0.081 | 0.09 | 0.095 | 0.015 | 0.855 | 63.618 | 64.068 | -0.702 |
| RB2 | any_td | 183 | 0.498 | 0.078 | 0.082 | 0.087 | -0.054 | 0.856 | 0.224 | 0.208 | 7.785 |
| RB2 | rec | 183 | 0.505 | 0.082 | 0.115 | 0.104 | 0.026 | 0.974 | 1.492 | 1.546 | -3.53 |
| RB2 | rec_yds | 183 | 0.498 | 0.094 | 0.158 | 0.104 | 0.028 | 1.155 | 10.937 | 11.749 | -6.91 |
| RB2 | rush_att | 183 | 0.481 | 0.105 | 0.158 | 0.153 | 0.091 | 2.036 | 6.079 | 6.23 | -2.419 |
| RB2 | rush_yds | 183 | 0.465 | 0.111 | 0.169 | 0.158 | 0.074 | 2.111 | 25.795 | 26.098 | -1.162 |
| TE1 | any_td | 228 | 0.518 | 0.076 | 0.057 | 0.092 | 0.035 | 1.013 | 0.254 | 0.272 | -6.452 |
| TE1 | rec | 228 | 0.497 | 0.083 | 0.105 | 0.07 | -0.023 | 0.908 | 3.785 | 3.728 | 1.523 |
| TE1 | rec_yds | 228 | 0.495 | 0.081 | 0.105 | 0.07 | -0.048 | 0.814 | 39.987 | 38.311 | 4.374 |
| TE1 | tgt | 228 | 0.518 | 0.074 | 0.048 | 0.075 | 0.037 | 0.804 | 5.195 | 5.285 | -1.706 |
| WR1 | any_td | 228 | 0.554 | 0.086 | 0.096 | 0.149 | 0.158 | 1.199 | 0.277 | 0.342 | -18.906 |
| WR1 | rec | 228 | 0.553 | 0.084 | 0.092 | 0.123 | 0.157 | 1.052 | 4.576 | 4.961 | -7.743 |
| WR1 | rec_yds | 228 | 0.549 | 0.079 | 0.083 | 0.101 | 0.101 | 0.819 | 59.284 | 63.61 | -6.801 |
| WR1 | tgt | 228 | 0.53 | 0.084 | 0.092 | 0.11 | 0.102 | 1.09 | 7.267 | 7.61 | -4.505 |
| WR2 | any_td | 202 | 0.498 | 0.09 | 0.119 | 0.119 | 0.031 | 1.081 | 0.198 | 0.208 | -4.78 |
| WR2 | rec | 202 | 0.517 | 0.077 | 0.079 | 0.124 | 0.06 | 0.927 | 3.196 | 3.351 | -4.633 |
| WR2 | rec_yds | 202 | 0.54 | 0.073 | 0.069 | 0.114 | 0.097 | 0.88 | 42.304 | 46.015 | -8.065 |
| WR2 | tgt | 202 | 0.527 | 0.076 | 0.059 | 0.104 | 0.087 | 0.908 | 5.202 | 5.48 | -5.071 |
| WR3 | any_td | 173 | 0.513 | 0.082 | 0.098 | 0.121 | 0.044 | 1.11 | 0.137 | 0.15 | -9.145 |
| WR3 | rec | 173 | 0.526 | 0.082 | 0.11 | 0.11 | 0.061 | 0.903 | 2.107 | 2.179 | -3.311 |
| WR3 | rec_yds | 173 | 0.513 | 0.079 | 0.098 | 0.092 | -0.024 | 0.757 | 27.055 | 25.971 | 4.172 |
| WR3 | tgt | 173 | 0.536 | 0.081 | 0.052 | 0.145 | 0.136 | 1.029 | 3.38 | 3.642 | -7.181 |

