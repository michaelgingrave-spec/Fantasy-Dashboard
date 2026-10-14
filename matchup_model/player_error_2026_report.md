# Player-error audit — 2026, 2462 graded props, 312 players

z-error = (actual - our projection) / market SD. Positive bias = we UNDER-project. CI = 95% player-clustered bootstrap; `*` = CI excludes zero.

## 1. Do we miss the SAME players the same way?

| scope | measure | n | slope | 95% CI | r | 95% CI |
|---|---|--:|--:|---|--:|---|
| all markets | signed error (bias) | 1759 | -0.01 | -0.08 to +0.06 | -0.01 | -0.06 to +0.05 |
| all markets | unsigned error (volatility) | 1759 | +0.05 | -0.01 to +0.11 | +0.04 | -0.01 to +0.08 |
| receptions | signed error (bias) | 614 | -0.02 | -0.12 to +0.06 | -0.02 | -0.09 to +0.05 |
| receptions | unsigned error (volatility) | 614 | -0.01 | -0.11 to +0.08 | -0.01 | -0.08 to +0.07 |
| rec yds | signed error (bias) | 622 | -0.01 | -0.11 to +0.07 | -0.01 | -0.08 to +0.06 |
| rec yds | unsigned error (volatility) | 622 | +0.05 | -0.04 to +0.13 | +0.04 | -0.03 to +0.11 |
| rush yds | signed error (bias) | 301 | +0.12 | -0.04 to +0.26 | +0.08 | -0.03 to +0.18 |
| rush yds | unsigned error (volatility) | 301 | +0.14 | -0.06 to +0.33 | +0.09 | -0.03 to +0.20 |

Split-half (each player's mean error in weeks 1-3 vs weeks 4-5, players with 2+ props in both): r = -0.02 (95% CI -0.16 to +0.14), n = 212 players.

## 2. Error by player type

| feature | level | n | players | bias | 95% CI | | RMSE | picks | pick win% |
|---|---|--:|--:|--:|---|---|--:|--:|--:|
| position | QB | 448 | 40 | +0.06 | -0.02 to +0.13 |  | 0.92 | 208 | 60% |
| position | RB | 684 | 77 | +0.03 | -0.07 to +0.12 |  | 0.99 | 290 | 55% |
| position | TE | 454 | 67 | -0.30 | -0.42 to -0.17 | * | 1.07 | 355 | 46% |
| position | WR | 876 | 128 | -0.08 | -0.18 to +0.01 |  | 1.14 | 676 | 52% |
| volume | high | 820 | 142 | -0.09 | -0.17 to -0.01 | * | 1.14 | 531 | 51% |
| volume | low | 823 | 185 | -0.01 | -0.08 to +0.06 |  | 0.92 | 477 | 53% |
| volume | mid | 819 | 185 | -0.09 | -0.19 to -0.01 | * | 1.07 | 521 | 52% |
| volatility (CV) | high | 818 | 188 | +0.04 | -0.03 to +0.12 |  | 0.99 | 472 | 54% |
| volatility (CV) | low | 821 | 171 | -0.12 | -0.20 to -0.04 | * | 1.09 | 556 | 53% |
| volatility (CV) | mid | 816 | 201 | -0.12 | -0.19 to -0.02 | * | 1.07 | 496 | 48% |
| games of history | 3-5 | 92 | 27 | -0.29 | -0.56 to -0.04 | * | 1.15 | 60 | 55% |
| games of history | 6-10 | 2366 | 286 | -0.05 | -0.11 to -0.00 | * | 1.05 | 1465 | 52% |
| rookie/2nd-yr | no | 2001 | 244 | -0.06 | -0.11 to +0.00 |  | 1.04 | 1231 | 52% |
| rookie/2nd-yr | yes | 461 | 68 | -0.08 | -0.20 to +0.05 |  | 1.11 | 298 | 54% |
| team total | high | 821 | 166 | -0.02 | -0.11 to +0.06 |  | 1.08 | 522 | 53% |
| team total | low | 821 | 169 | -0.12 | -0.22 to -0.04 | * | 1.03 | 505 | 50% |
| team total | mid | 820 | 208 | -0.05 | -0.14 to +0.05 |  | 1.03 | 502 | 52% |
| proj vs line gap | 10-30% | 885 | 250 | -0.06 | -0.13 to +0.01 |  | 1.03 | 637 | 50% |
| proj vs line gap | 30-60% | 489 | 194 | -0.12 | -0.22 to -0.02 | * | 1.01 | 433 | 53% |
| proj vs line gap | 60%+ | 395 | 163 | -0.33 | -0.44 to -0.18 | * | 1.13 | 363 | 52% |
| proj vs line gap | <10% | 693 | 206 | +0.13 | +0.04 to +0.21 | * | 1.05 | 96 | 60% |

## 3. How much of our disagreement with the market is real?

Slope of (actual - line) on (our projection - line), z-units. 1.0 = our number beats the line one-for-one; 0 = the line was right and our difference was noise.

| scope | n | real share | 95% CI | r |
|---|--:|--:|---|--:|
| all props | 2462 | +0.19 | +0.07 to +0.34 | +0.07 |
| labeled picks only (not '—') | 1543 | +0.20 | +0.07 to +0.35 | +0.08 |
| we lean OVER | 1702 | +0.13 | -0.04 to +0.33 | +0.03 |
| we lean UNDER | 710 | +0.31 | -0.11 to +0.72 | +0.06 |
| position QB | 448 | +0.21 | -0.13 to +0.56 | +0.08 |
| position RB | 684 | +0.12 | -0.06 to +0.35 | +0.04 |
| position WR | 876 | +0.26 | +0.01 to +0.51 | +0.09 |
| position TE | 454 | +0.22 | -0.07 to +0.54 | +0.07 |
| receptions | 867 | +0.21 | +0.01 to +0.40 | +0.07 |
| rec yds | 878 | +0.26 | +0.08 to +0.47 | +0.09 |
| rush yds | 415 | -0.07 | -0.31 to +0.19 | -0.03 |

## 4. Pick win rate by position and side (labeled picks, n>=20)

| position | lean | n | win% |
|---|---|--:|--:|
| QB | OVER | 121 | 60% |
| QB | UNDER | 87 | 59% |
| RB | OVER | 192 | 52% |
| RB | UNDER | 98 | 60% |
| TE | OVER | 326 | 44% |
| TE | UNDER | 29 | 62% |
| WR | OVER | 560 | 52% |
| WR | UNDER | 116 | 50% |

## 5. Is the TE gap real? (a hypothesis found AFTER the scans above)

Pocket = receptions at <=1.5 plus rec yds at <=40; labeled OVER picks only, whole-number consensus catch lines dropped (their price belongs to a different bet). The pocket was chosen after TEs stood out in section 2, so the in-sample gap in 5a overstates it. 5b is the fair test: each week is fit on earlier weeks only.

**5a. TE vs every other position on the same kind of pick** (gap = TE win% minus the rest, 95% player-clustered CI; ROI at each pick's best price)

| slice | TE n | other n | TE win% | other win% | gap (pts) | 95% CI | TE ROI | other ROI |
|---|--:|--:|--:|--:|--:|---|--:|--:|
| all receiving OVER picks | 317 | 671 | 44.5% | 52.3% | -7.8 | -15.5 to -0.9 | -15.3% | -0.2% |
| low-line pocket | 206 | 396 | 41.7% | 57.1% | -15.3 | -24.9 to -5.5 | -19.9% | +8.9% |
|   weeks 1-3 | 120 | 207 | 40.0% | 55.6% | -15.6 | -28.0 to -3.3 | -23.9% | +7.0% |
|   weeks 4+ | 86 | 189 | 44.2% | 58.7% | -14.5 | -27.7 to -0.8 | -14.3% | +11.0% |
|   receptions only | 57 | 131 | 40.4% | 60.3% | -20.0 | -35.1 to -6.3 | -20.6% | +15.4% |
|   rec yds only | 149 | 265 | 42.3% | 55.5% | -13.2 | -22.5 to -3.6 | -19.6% | +5.7% |
| mid/high lines (not pocket) | 111 | 275 | 49.5% | 45.5% | +4.1 | -7.6 to +14.3 | -6.8% | -13.4% |

**5b. Walk-forward** (test weeks [3, 4, 5]): does adding a TE / non-TE split to the calibration improve probabilities on weeks it never saw? Brier, lower is better; gain = current minus TE-aware (x1e4), positive favours TE-aware.

| slice | n | model p | current | TE-aware | gain (x1e4) | 95% CI |
|---|--:|--:|--:|--:|--:|---|
| all labeled props | 933 | 0.2570 | 0.2530 | 0.2539 | -10 | -46 to +21 |
| receiving markets | 759 | 0.2584 | 0.2535 | 0.2544 | -9 | -47 to +31 |
| TE props | 214 | 0.2649 | 0.2567 | 0.2575 | -8 | -102 to +99 |
| non-TE receiving props | 545 | 0.2558 | 0.2522 | 0.2531 | -10 | -42 to +26 |
| low-line receiving OVERs | 408 | 0.2584 | 0.2539 | 0.2531 | +8 | -59 to +77 |

Every pick each calibration would make (calibrated p >= 0.55 and +EV), test weeks pooled:

| calibration | picks | win% | ROI |
|---|--:|--:|--:|
| current calibration | 145 | 49.7% | -6.1% |
| TE-aware calibration | 230 | 53.0% | +1.2% |

**Forward test, fixed after week 5:** weeks 6+ are out of sample for this hypothesis (the 'weeks 6+' row in 5a fills in as they are graded). A TE factor goes into `dfs/conviction.py` only if that row shows a gap whose CI excludes zero on 100+ TE pocket picks, or the 5b gain turns positive with a CI excluding zero.