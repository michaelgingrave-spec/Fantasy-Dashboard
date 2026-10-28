# Player-error audit — 2026, 3452 graded props, 345 players

z-error = (actual - our projection) / market SD. Positive bias = we UNDER-project. CI = 95% player-clustered bootstrap; `*` = CI excludes zero.

## 1. Do we miss the SAME players the same way?

| scope | measure | n | slope | 95% CI | r | 95% CI |
|---|---|--:|--:|---|--:|---|
| all markets | signed error (bias) | 2672 | -0.03 | -0.10 to +0.03 | -0.02 | -0.07 to +0.03 |
| all markets | unsigned error (volatility) | 2672 | +0.04 | -0.02 to +0.09 | +0.03 | -0.01 to +0.07 |
| receptions | signed error (bias) | 941 | -0.05 | -0.14 to +0.02 | -0.04 | -0.09 to +0.02 |
| receptions | unsigned error (volatility) | 941 | -0.02 | -0.12 to +0.05 | -0.02 | -0.08 to +0.04 |
| rec yds | signed error (bias) | 952 | -0.01 | -0.09 to +0.07 | -0.01 | -0.07 to +0.05 |
| rec yds | unsigned error (volatility) | 952 | +0.04 | -0.04 to +0.12 | +0.03 | -0.03 to +0.09 |
| rush yds | signed error (bias) | 451 | +0.03 | -0.14 to +0.18 | +0.02 | -0.08 to +0.12 |
| rush yds | unsigned error (volatility) | 451 | +0.18 | +0.01 to +0.35 | +0.11 | +0.00 to +0.21 |

Split-half (each player's mean error in weeks 1-3 vs weeks 4-5, players with 2+ props in both): r = -0.10 (95% CI -0.26 to +0.06), n = 221 players.

## 2. Error by player type

| feature | level | n | players | bias | 95% CI | | RMSE | picks | pick win% |
|---|---|--:|--:|--:|---|---|--:|--:|--:|
| position | QB | 616 | 43 | +0.03 | -0.04 to +0.09 |  | 0.88 | 280 | 59% |
| position | RB | 969 | 90 | +0.05 | -0.03 to +0.12 |  | 0.99 | 396 | 54% |
| position | TE | 637 | 70 | -0.21 | -0.30 to -0.12 | * | 1.04 | 475 | 50% |
| position | WR | 1230 | 142 | -0.07 | -0.14 to +0.01 |  | 1.10 | 883 | 52% |
| volume | high | 1151 | 161 | -0.07 | -0.14 to +0.00 |  | 1.10 | 682 | 53% |
| volume | low | 1152 | 212 | -0.00 | -0.06 to +0.06 |  | 0.89 | 650 | 55% |
| volume | mid | 1149 | 208 | -0.06 | -0.13 to +0.01 |  | 1.06 | 702 | 52% |
| volatility (CV) | high | 1146 | 230 | +0.03 | -0.04 to +0.09 |  | 0.96 | 648 | 55% |
| volatility (CV) | low | 1148 | 188 | -0.09 | -0.16 to -0.03 | * | 1.07 | 712 | 54% |
| volatility (CV) | mid | 1146 | 244 | -0.08 | -0.15 to -0.01 | * | 1.04 | 666 | 49% |
| games of history | 3-5 | 154 | 34 | -0.15 | -0.35 to +0.07 |  | 1.06 | 100 | 60% |
| games of history | 6-10 | 3292 | 325 | -0.04 | -0.08 to -0.00 | * | 1.02 | 1928 | 52% |
| rookie/2nd-yr | no | 2778 | 265 | -0.05 | -0.10 to -0.01 | * | 1.01 | 1616 | 52% |
| rookie/2nd-yr | yes | 674 | 80 | -0.01 | -0.11 to +0.08 |  | 1.08 | 418 | 56% |
| team total | high | 1151 | 202 | -0.01 | -0.07 to +0.05 |  | 1.04 | 690 | 52% |
| team total | low | 1151 | 212 | -0.11 | -0.17 to -0.03 | * | 1.00 | 667 | 53% |
| team total | mid | 1150 | 239 | -0.02 | -0.10 to +0.06 |  | 1.03 | 677 | 54% |
| proj vs line gap | 10-30% | 1235 | 290 | -0.05 | -0.12 to -0.00 | * | 0.99 | 859 | 51% |
| proj vs line gap | 30-60% | 636 | 231 | -0.09 | -0.18 to -0.00 | * | 1.01 | 558 | 55% |
| proj vs line gap | 60%+ | 536 | 202 | -0.26 | -0.37 to -0.14 | * | 1.10 | 483 | 52% |
| proj vs line gap | <10% | 1045 | 251 | +0.10 | +0.04 to +0.16 | * | 1.02 | 134 | 59% |

## 3. How much of our disagreement with the market is real?

Slope of (actual - line) on (our projection - line), z-units. 1.0 = our number beats the line one-for-one; 0 = the line was right and our difference was noise.

| scope | n | real share | 95% CI | r |
|---|--:|--:|---|--:|
| all props | 3452 | +0.26 | +0.14 to +0.39 | +0.09 |
| labeled picks only (not '—') | 2057 | +0.28 | +0.15 to +0.41 | +0.11 |
| we lean OVER | 2353 | +0.22 | +0.03 to +0.42 | +0.05 |
| we lean UNDER | 1020 | +0.26 | -0.14 to +0.59 | +0.05 |
| position QB | 616 | +0.24 | -0.04 to +0.50 | +0.09 |
| position RB | 969 | +0.13 | -0.04 to +0.32 | +0.04 |
| position WR | 1230 | +0.36 | +0.16 to +0.60 | +0.12 |
| position TE | 637 | +0.35 | +0.06 to +0.65 | +0.11 |
| receptions | 1222 | +0.32 | +0.16 to +0.48 | +0.11 |
| rec yds | 1235 | +0.32 | +0.14 to +0.49 | +0.10 |
| rush yds | 581 | -0.08 | -0.29 to +0.16 | -0.03 |

## 4. Pick win rate by position and side (labeled picks, n>=20)

| position | lean | n | win% |
|---|---|--:|--:|
| QB | OVER | 168 | 57% |
| QB | UNDER | 112 | 61% |
| RB | OVER | 269 | 52% |
| RB | UNDER | 127 | 59% |
| TE | OVER | 428 | 48% |
| TE | UNDER | 47 | 66% |
| WR | OVER | 707 | 51% |
| WR | UNDER | 176 | 56% |

## 5. Is the TE gap real? (a hypothesis found AFTER the scans above)

Pocket = receptions at <=1.5 plus rec yds at <=40; labeled OVER picks only, whole-number consensus catch lines dropped (their price belongs to a different bet). The pocket was chosen after TEs stood out in section 2, so the in-sample gap in 5a overstates it. 5b is the fair test: each week is fit on earlier weeks only.

**5a. TE vs every other position on the same kind of pick** (gap = TE win% minus the rest, 95% player-clustered CI; ROI at each pick's best price)

| slice | TE n | other n | TE win% | other win% | gap (pts) | 95% CI | TE ROI | other ROI |
|---|--:|--:|--:|--:|--:|---|--:|--:|
| all receiving OVER picks | 416 | 876 | 47.8% | 51.7% | -3.9 | -10.2 to +1.8 | -9.5% | -1.7% |
| low-line pocket | 283 | 572 | 46.6% | 55.6% | -9.0 | -17.7 to +0.1 | -11.6% | +5.7% |
|   weeks 1-3 | 120 | 207 | 40.0% | 55.6% | -15.6 | -28.0 to -3.3 | -23.9% | +7.0% |
|   weeks 4+ | 163 | 365 | 51.5% | 55.6% | -4.1 | -17.8 to +7.8 | -2.5% | +4.9% |
|   receptions only | 80 | 186 | 48.8% | 60.2% | -11.5 | -27.7 to +2.7 | -8.5% | +13.9% |
|   rec yds only | 203 | 386 | 45.8% | 53.4% | -7.6 | -16.3 to +0.4 | -12.8% | +1.7% |
|   weeks 6+ (forward test) | 77 | 176 | 59.7% | 52.3% | +7.5 | -8.4 to +22.5 | +10.7% | -1.6% |
| mid/high lines (not pocket) | 133 | 304 | 50.4% | 44.4% | +6.0 | -4.0 to +14.5 | -5.1% | -15.4% |

**5b. Walk-forward** (test weeks [3, 4, 5, 6, 7]): does adding a TE / non-TE split to the calibration improve probabilities on weeks it never saw? Brier, lower is better; gain = current minus TE-aware (x1e4), positive favours TE-aware.

| slice | n | model p | current | TE-aware | gain (x1e4) | 95% CI |
|---|--:|--:|--:|--:|--:|---|
| all labeled props | 1428 | 0.2545 | 0.2512 | 0.2527 | -15 | -45 to +13 |
| receiving markets | 1164 | 0.2555 | 0.2520 | 0.2536 | -15 | -53 to +24 |
| TE props | 331 | 0.2554 | 0.2543 | 0.2592 | -48 | -130 to +46 |
| non-TE receiving props | 833 | 0.2555 | 0.2511 | 0.2513 | -2 | -30 to +28 |
| low-line receiving OVERs | 663 | 0.2560 | 0.2529 | 0.2548 | -19 | -72 to +33 |

Every pick each calibration would make (calibrated p >= 0.55 and +EV), test weeks pooled:

| calibration | picks | win% | ROI |
|---|--:|--:|--:|
| current calibration | 197 | 49.7% | -6.9% |
| TE-aware calibration | 313 | 52.7% | -0.5% |

**Forward test, fixed after week 5:** weeks 6+ are out of sample for this hypothesis (the 'weeks 6+' row in 5a fills in as they are graded). A TE factor goes into `dfs/conviction.py` only if that row shows a gap whose CI excludes zero on 100+ TE pocket picks, or the 5b gain turns positive with a CI excluding zero.