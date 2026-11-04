# Player-error audit — 2026, 3952 graded props, 361 players

z-error = (actual - our projection) / market SD. Positive bias = we UNDER-project. CI = 95% player-clustered bootstrap; `*` = CI excludes zero.

## 1. Do we miss the SAME players the same way?

| scope | measure | n | slope | 95% CI | r | 95% CI |
|---|---|--:|--:|---|--:|---|
| all markets | signed error (bias) | 3137 | -0.02 | -0.08 to +0.04 | -0.01 | -0.05 to +0.03 |
| all markets | unsigned error (volatility) | 3137 | +0.06 | +0.00 to +0.10 | +0.04 | +0.00 to +0.07 |
| receptions | signed error (bias) | 1106 | -0.03 | -0.11 to +0.03 | -0.02 | -0.07 to +0.02 |
| receptions | unsigned error (volatility) | 1106 | -0.00 | -0.09 to +0.07 | -0.00 | -0.06 to +0.05 |
| rec yds | signed error (bias) | 1117 | +0.00 | -0.09 to +0.08 | +0.00 | -0.06 to +0.06 |
| rec yds | unsigned error (volatility) | 1117 | +0.05 | -0.03 to +0.13 | +0.04 | -0.02 to +0.09 |
| rush yds | signed error (bias) | 530 | +0.04 | -0.13 to +0.19 | +0.02 | -0.07 to +0.12 |
| rush yds | unsigned error (volatility) | 530 | +0.24 | +0.08 to +0.39 | +0.13 | +0.04 to +0.21 |

Split-half (each player's mean error in weeks 1-3 vs weeks 4-5, players with 2+ props in both): r = -0.02 (95% CI -0.17 to +0.12), n = 222 players.

## 2. Error by player type

| feature | level | n | players | bias | 95% CI | | RMSE | picks | pick win% |
|---|---|--:|--:|--:|---|---|--:|--:|--:|
| position | QB | 700 | 43 | -0.00 | -0.07 to +0.06 |  | 0.90 | 315 | 58% |
| position | RB | 1115 | 97 | +0.06 | -0.00 to +0.12 |  | 1.03 | 459 | 55% |
| position | TE | 732 | 74 | -0.19 | -0.27 to -0.10 | * | 1.04 | 522 | 50% |
| position | WR | 1405 | 147 | -0.07 | -0.15 to +0.01 |  | 1.09 | 982 | 52% |
| volume | high | 1317 | 169 | -0.07 | -0.13 to -0.00 | * | 1.10 | 760 | 53% |
| volume | low | 1320 | 228 | +0.00 | -0.05 to +0.06 |  | 0.91 | 737 | 55% |
| volume | mid | 1315 | 220 | -0.06 | -0.12 to -0.00 | * | 1.08 | 781 | 51% |
| volatility (CV) | high | 1312 | 241 | +0.03 | -0.03 to +0.10 |  | 0.96 | 732 | 55% |
| volatility (CV) | low | 1316 | 199 | -0.09 | -0.15 to -0.03 | * | 1.07 | 794 | 53% |
| volatility (CV) | mid | 1311 | 254 | -0.06 | -0.13 to +0.01 |  | 1.07 | 744 | 50% |
| games of history | 3-5 | 179 | 40 | -0.17 | -0.36 to +0.03 |  | 1.04 | 115 | 59% |
| games of history | 6-10 | 3766 | 342 | -0.04 | -0.07 to +0.00 |  | 1.03 | 2157 | 53% |
| rookie/2nd-yr | no | 3155 | 272 | -0.06 | -0.10 to -0.01 | * | 1.01 | 1799 | 52% |
| rookie/2nd-yr | yes | 797 | 89 | +0.01 | -0.07 to +0.11 |  | 1.11 | 479 | 56% |
| team total | high | 1317 | 209 | -0.03 | -0.09 to +0.03 |  | 1.02 | 762 | 51% |
| team total | low | 1318 | 223 | -0.07 | -0.14 to -0.01 | * | 1.03 | 755 | 54% |
| team total | mid | 1317 | 249 | -0.02 | -0.09 to +0.05 |  | 1.05 | 761 | 54% |
| proj vs line gap | 10-30% | 1412 | 305 | -0.05 | -0.12 to -0.01 | * | 0.99 | 973 | 51% |
| proj vs line gap | 30-60% | 702 | 246 | -0.07 | -0.15 to +0.02 |  | 1.03 | 614 | 55% |
| proj vs line gap | 60%+ | 600 | 216 | -0.25 | -0.34 to -0.14 | * | 1.10 | 537 | 52% |
| proj vs line gap | <10% | 1238 | 272 | +0.09 | +0.03 to +0.14 | * | 1.05 | 154 | 58% |

## 3. How much of our disagreement with the market is real?

Slope of (actual - line) on (our projection - line), z-units. 1.0 = our number beats the line one-for-one; 0 = the line was right and our difference was noise.

| scope | n | real share | 95% CI | r |
|---|--:|--:|---|--:|
| all props | 3952 | +0.29 | +0.18 to +0.41 | +0.09 |
| labeled picks only (not '—') | 2303 | +0.32 | +0.21 to +0.44 | +0.12 |
| we lean OVER | 2667 | +0.21 | +0.04 to +0.42 | +0.05 |
| we lean UNDER | 1180 | +0.26 | -0.15 to +0.55 | +0.05 |
| position QB | 700 | +0.27 | +0.02 to +0.53 | +0.09 |
| position RB | 1115 | +0.17 | +0.01 to +0.37 | +0.05 |
| position WR | 1405 | +0.38 | +0.17 to +0.58 | +0.12 |
| position TE | 732 | +0.36 | +0.08 to +0.66 | +0.11 |
| receptions | 1399 | +0.36 | +0.22 to +0.52 | +0.12 |
| rec yds | 1415 | +0.33 | +0.18 to +0.51 | +0.10 |
| rush yds | 668 | -0.09 | -0.29 to +0.15 | -0.03 |

## 4. Pick win rate by position and side (labeled picks, n>=20)

| position | lean | n | win% |
|---|---|--:|--:|
| QB | OVER | 191 | 57% |
| QB | UNDER | 124 | 60% |
| RB | OVER | 317 | 52% |
| RB | UNDER | 142 | 60% |
| TE | OVER | 466 | 48% |
| TE | UNDER | 56 | 66% |
| WR | OVER | 776 | 51% |
| WR | UNDER | 206 | 56% |

## 5. Is the TE gap real? (a hypothesis found AFTER the scans above)

Pocket = receptions at <=1.5 plus rec yds at <=40; labeled OVER picks only, whole-number consensus catch lines dropped (their price belongs to a different bet). The pocket was chosen after TEs stood out in section 2, so the in-sample gap in 5a overstates it. 5b is the fair test: each week is fit on earlier weeks only.

**5a. TE vs every other position on the same kind of pick** (gap = TE win% minus the rest, 95% player-clustered CI; ROI at each pick's best price)

| slice | TE n | other n | TE win% | other win% | gap (pts) | 95% CI | TE ROI | other ROI |
|---|--:|--:|--:|--:|--:|---|--:|--:|
| all receiving OVER picks | 454 | 982 | 48.2% | 51.7% | -3.5 | -10.6 to +2.6 | -9.0% | -1.6% |
| low-line pocket | 313 | 657 | 47.3% | 55.6% | -8.3 | -17.4 to +0.7 | -10.6% | +5.5% |
|   weeks 1-3 | 120 | 207 | 40.0% | 55.6% | -15.6 | -28.0 to -3.3 | -23.9% | +7.0% |
|   weeks 4+ | 193 | 450 | 51.8% | 55.6% | -3.7 | -15.2 to +5.5 | -2.3% | +4.9% |
|   receptions only | 87 | 208 | 49.4% | 62.5% | -13.1 | -26.6 to +1.5 | -8.2% | +17.9% |
|   rec yds only | 226 | 449 | 46.5% | 52.3% | -5.9 | -15.2 to +1.8 | -11.5% | -0.2% |
|   weeks 6+ (forward test) | 107 | 261 | 57.9% | 53.3% | +4.7 | -9.3 to +18.5 | +7.4% | +0.4% |
| mid/high lines (not pocket) | 141 | 325 | 50.4% | 44.0% | +6.4 | -2.3 to +13.7 | -5.4% | -16.2% |

**5b. Walk-forward** (test weeks [3, 4, 5, 6, 7, 8]): does adding a TE / non-TE split to the calibration improve probabilities on weeks it never saw? Brier, lower is better; gain = current minus TE-aware (x1e4), positive favours TE-aware.

| slice | n | model p | current | TE-aware | gain (x1e4) | 95% CI |
|---|--:|--:|--:|--:|--:|---|
| all labeled props | 1671 | 0.2542 | 0.2504 | 0.2515 | -12 | -35 to +12 |
| receiving markets | 1359 | 0.2545 | 0.2507 | 0.2518 | -11 | -44 to +22 |
| TE props | 377 | 0.2537 | 0.2527 | 0.2576 | -48 | -135 to +40 |
| non-TE receiving props | 982 | 0.2549 | 0.2500 | 0.2496 | +4 | -23 to +31 |
| low-line receiving OVERs | 780 | 0.2551 | 0.2518 | 0.2534 | -16 | -72 to +34 |

Every pick each calibration would make (calibrated p >= 0.55 and +EV), test weeks pooled:

| calibration | picks | win% | ROI |
|---|--:|--:|--:|
| current calibration | 226 | 50.0% | -6.9% |
| TE-aware calibration | 361 | 53.5% | +0.2% |

**Forward test, fixed after week 5:** weeks 6+ are out of sample for this hypothesis (the 'weeks 6+' row in 5a fills in as they are graded). A TE factor goes into `dfs/conviction.py` only if that row shows a gap whose CI excludes zero on 100+ TE pocket picks, or the 5b gain turns positive with a CI excluding zero.