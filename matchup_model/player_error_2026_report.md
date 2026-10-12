# Player-error audit — 2026, 2359 graded props, 303 players

z-error = (actual - our projection) / market SD. Positive bias = we UNDER-project. CI = 95% player-clustered bootstrap; `*` = CI excludes zero.

## 1. Do we miss the SAME players the same way?

| scope | measure | n | slope | 95% CI | r | 95% CI |
|---|---|--:|--:|---|--:|---|
| all markets | signed error (bias) | 1671 | -0.03 | -0.11 to +0.03 | -0.03 | -0.08 to +0.02 |
| all markets | unsigned error (volatility) | 1671 | +0.04 | -0.03 to +0.10 | +0.03 | -0.02 to +0.08 |
| receptions | signed error (bias) | 576 | -0.05 | -0.14 to +0.04 | -0.04 | -0.11 to +0.03 |
| receptions | unsigned error (volatility) | 576 | -0.01 | -0.13 to +0.09 | -0.01 | -0.09 to +0.07 |
| rec yds | signed error (bias) | 583 | -0.03 | -0.13 to +0.07 | -0.03 | -0.10 to +0.05 |
| rec yds | unsigned error (volatility) | 583 | +0.05 | -0.06 to +0.14 | +0.04 | -0.04 to +0.11 |
| rush yds | signed error (bias) | 294 | +0.12 | -0.03 to +0.29 | +0.08 | -0.02 to +0.20 |
| rush yds | unsigned error (volatility) | 294 | +0.12 | -0.06 to +0.32 | +0.08 | -0.04 to +0.21 |

Split-half (each player's mean error in weeks 1-3 vs weeks 4-5, players with 2+ props in both): r = +0.00 (95% CI -0.16 to +0.15), n = 202 players.

## 2. Error by player type

| feature | level | n | players | bias | 95% CI | | RMSE | picks | pick win% |
|---|---|--:|--:|--:|---|---|--:|--:|--:|
| position | QB | 442 | 40 | +0.05 | -0.03 to +0.12 |  | 0.91 | 206 | 59% |
| position | RB | 670 | 76 | +0.02 | -0.07 to +0.12 |  | 0.99 | 283 | 55% |
| position | TE | 409 | 61 | -0.23 | -0.35 to -0.11 | * | 1.06 | 312 | 50% |
| position | WR | 838 | 126 | -0.03 | -0.14 to +0.06 |  | 1.12 | 642 | 54% |
| volume | high | 787 | 139 | -0.07 | -0.17 to +0.02 |  | 1.13 | 505 | 52% |
| volume | low | 788 | 179 | +0.03 | -0.05 to +0.11 |  | 0.91 | 445 | 58% |
| volume | mid | 784 | 183 | -0.07 | -0.15 to +0.02 |  | 1.06 | 493 | 54% |
| volatility (CV) | high | 785 | 184 | +0.08 | -0.01 to +0.16 |  | 0.98 | 445 | 57% |
| volatility (CV) | low | 785 | 168 | -0.12 | -0.19 to -0.04 | * | 1.07 | 528 | 54% |
| volatility (CV) | mid | 782 | 196 | -0.07 | -0.16 to +0.03 |  | 1.07 | 465 | 51% |
| games of history | 3-5 | 88 | 26 | -0.24 | -0.54 to +0.02 |  | 1.14 | 56 | 59% |
| games of history | 6-10 | 2267 | 278 | -0.03 | -0.08 to +0.02 |  | 1.03 | 1383 | 54% |
| rookie/2nd-yr | no | 1912 | 237 | -0.03 | -0.09 to +0.02 |  | 1.02 | 1158 | 54% |
| rookie/2nd-yr | yes | 447 | 66 | -0.04 | -0.17 to +0.08 |  | 1.10 | 285 | 56% |
| team total | high | 786 | 167 | +0.01 | -0.08 to +0.10 |  | 1.08 | 494 | 54% |
| team total | low | 787 | 162 | -0.09 | -0.18 to -0.00 | * | 1.01 | 475 | 53% |
| team total | mid | 786 | 206 | -0.02 | -0.11 to +0.08 |  | 1.02 | 474 | 55% |
| proj vs line gap | 10-30% | 859 | 241 | -0.04 | -0.11 to +0.02 |  | 1.01 | 615 | 50% |
| proj vs line gap | 30-60% | 472 | 188 | -0.09 | -0.18 to +0.01 |  | 1.00 | 416 | 55% |
| proj vs line gap | 60%+ | 349 | 155 | -0.26 | -0.39 to -0.12 | * | 1.13 | 318 | 58% |
| proj vs line gap | <10% | 679 | 203 | +0.12 | +0.05 to +0.20 | * | 1.05 | 94 | 60% |

## 3. How much of our disagreement with the market is real?

Slope of (actual - line) on (our projection - line), z-units. 1.0 = our number beats the line one-for-one; 0 = the line was right and our difference was noise.

| scope | n | real share | 95% CI | r |
|---|--:|--:|---|--:|
| all props | 2359 | +0.24 | +0.10 to +0.38 | +0.09 |
| labeled picks only (not '—') | 1457 | +0.24 | +0.10 to +0.37 | +0.10 |
| we lean OVER | 1609 | +0.18 | +0.00 to +0.41 | +0.05 |
| we lean UNDER | 701 | +0.30 | -0.20 to +0.67 | +0.06 |
| position QB | 442 | +0.22 | -0.11 to +0.56 | +0.08 |
| position RB | 670 | +0.11 | -0.09 to +0.33 | +0.04 |
| position WR | 838 | +0.35 | +0.11 to +0.61 | +0.12 |
| position TE | 409 | +0.18 | -0.14 to +0.52 | +0.06 |
| receptions | 823 | +0.25 | +0.07 to +0.42 | +0.09 |
| rec yds | 831 | +0.33 | +0.12 to +0.54 | +0.11 |
| rush yds | 407 | -0.05 | -0.25 to +0.21 | -0.02 |

## 4. Pick win rate by position and side (labeled picks, n>=20)

| position | lean | n | win% |
|---|---|--:|--:|
| QB | OVER | 120 | 60% |
| QB | UNDER | 86 | 58% |
| RB | OVER | 186 | 52% |
| RB | UNDER | 97 | 61% |
| TE | OVER | 285 | 49% |
| TE | UNDER | 27 | 59% |
| WR | OVER | 526 | 55% |
| WR | UNDER | 116 | 50% |

## 5. Is the TE gap real? (a hypothesis found AFTER the scans above)

Pocket = receptions at <=1.5 plus rec yds at <=40; labeled OVER picks only, whole-number consensus catch lines dropped (their price belongs to a different bet). The pocket was chosen after TEs stood out in section 2, so the in-sample gap in 5a overstates it. 5b is the fair test: each week is fit on earlier weeks only.

**5a. TE vs every other position on the same kind of pick** (gap = TE win% minus the rest, 95% player-clustered CI; ROI at each pick's best price)

| slice | TE n | other n | TE win% | other win% | gap (pts) | 95% CI | TE ROI | other ROI |
|---|--:|--:|--:|--:|--:|---|--:|--:|
| all receiving OVER picks | 278 | 633 | 49.3% | 54.5% | -5.2 | -12.4 to +2.5 | -6.1% | +3.8% |
| low-line pocket | 171 | 368 | 48.5% | 60.9% | -12.3 | -21.7 to -3.3 | -6.7% | +16.2% |
|   weeks 1-3 | 101 | 196 | 47.5% | 58.7% | -11.1 | -26.2 to +1.5 | -9.6% | +13.0% |
|   weeks 4+ | 70 | 172 | 50.0% | 63.4% | -13.4 | -28.5 to +2.3 | -2.4% | +19.9% |
|   receptions only | 44 | 120 | 50.0% | 65.0% | -15.0 | -32.9 to -2.4 | -0.7% | +24.6% |
|   rec yds only | 127 | 248 | 48.0% | 58.9% | -10.8 | -21.0 to -1.0 | -8.7% | +12.1% |
| mid/high lines (not pocket) | 107 | 265 | 50.5% | 45.7% | +4.8 | -5.1 to +15.4 | -5.1% | -13.3% |

**5b. Walk-forward** (test weeks [3, 4, 5]): does adding a TE / non-TE split to the calibration improve probabilities on weeks it never saw? Brier, lower is better; gain = current minus TE-aware (x1e4), positive favours TE-aware.

| slice | n | model p | current | TE-aware | gain (x1e4) | 95% CI |
|---|--:|--:|--:|--:|--:|---|
| all labeled props | 866 | 0.2503 | 0.2492 | 0.2508 | -16 | -42 to +11 |
| receiving markets | 696 | 0.2501 | 0.2487 | 0.2502 | -16 | -46 to +17 |
| TE props | 186 | 0.2546 | 0.2553 | 0.2595 | -42 | -121 to +47 |
| non-TE receiving props | 510 | 0.2485 | 0.2462 | 0.2468 | -6 | -37 to +27 |
| low-line receiving OVERs | 359 | 0.2426 | 0.2454 | 0.2458 | -4 | -64 to +50 |

Every pick each calibration would make (calibrated p >= 0.55 and +EV), test weeks pooled:

| calibration | picks | win% | ROI |
|---|--:|--:|--:|
| current calibration | 235 | 57.4% | +10.1% |
| TE-aware calibration | 289 | 56.1% | +6.8% |

**Forward test, fixed after week 5:** weeks 6+ are out of sample for this hypothesis (the 'weeks 6+' row in 5a fills in as they are graded). A TE factor goes into `dfs/conviction.py` only if that row shows a gap whose CI excludes zero on 100+ TE pocket picks, or the 5b gain turns positive with a CI excluding zero.