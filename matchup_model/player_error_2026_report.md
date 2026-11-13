# Player-error audit — 2026, 4467 graded props, 367 players

z-error = (actual - our projection) / market SD. Positive bias = we UNDER-project. CI = 95% player-clustered bootstrap; `*` = CI excludes zero.

## 1. Do we miss the SAME players the same way?

| scope | measure | n | slope | 95% CI | r | 95% CI |
|---|---|--:|--:|---|--:|---|
| all markets | signed error (bias) | 3636 | -0.02 | -0.08 to +0.04 | -0.01 | -0.05 to +0.03 |
| all markets | unsigned error (volatility) | 3636 | +0.05 | +0.00 to +0.10 | +0.03 | +0.00 to +0.07 |
| receptions | signed error (bias) | 1290 | -0.04 | -0.11 to +0.03 | -0.03 | -0.08 to +0.02 |
| receptions | unsigned error (volatility) | 1290 | +0.00 | -0.07 to +0.07 | +0.00 | -0.05 to +0.05 |
| rec yds | signed error (bias) | 1302 | -0.00 | -0.09 to +0.08 | -0.00 | -0.05 to +0.05 |
| rec yds | unsigned error (volatility) | 1302 | +0.05 | -0.03 to +0.12 | +0.03 | -0.02 to +0.08 |
| rush yds | signed error (bias) | 608 | +0.04 | -0.13 to +0.19 | +0.02 | -0.07 to +0.12 |
| rush yds | unsigned error (volatility) | 608 | +0.21 | +0.04 to +0.35 | +0.12 | +0.02 to +0.19 |

Split-half (each player's mean error in weeks 1-3 vs weeks 4-5, players with 2+ props in both): r = +0.00 (95% CI -0.15 to +0.15), n = 226 players.

## 2. Error by player type

| feature | level | n | players | bias | 95% CI | | RMSE | picks | pick win% |
|---|---|--:|--:|--:|---|---|--:|--:|--:|
| position | QB | 783 | 46 | -0.00 | -0.08 to +0.06 |  | 0.90 | 346 | 58% |
| position | RB | 1258 | 97 | +0.05 | -0.01 to +0.10 |  | 1.01 | 522 | 54% |
| position | TE | 836 | 75 | -0.18 | -0.27 to -0.10 | * | 1.02 | 595 | 51% |
| position | WR | 1590 | 149 | -0.05 | -0.13 to +0.02 |  | 1.09 | 1093 | 53% |
| volume | high | 1489 | 171 | -0.06 | -0.13 to -0.01 | * | 1.08 | 838 | 53% |
| volume | low | 1490 | 241 | +0.00 | -0.05 to +0.06 |  | 0.90 | 837 | 55% |
| volume | mid | 1488 | 227 | -0.05 | -0.11 to +0.01 |  | 1.07 | 881 | 52% |
| volatility (CV) | high | 1485 | 251 | +0.02 | -0.03 to +0.08 |  | 0.95 | 828 | 55% |
| volatility (CV) | low | 1485 | 206 | -0.09 | -0.15 to -0.03 | * | 1.06 | 877 | 54% |
| volatility (CV) | mid | 1484 | 264 | -0.05 | -0.11 to +0.01 |  | 1.05 | 843 | 51% |
| games of history | 3-5 | 201 | 41 | -0.11 | -0.29 to +0.10 |  | 1.04 | 132 | 60% |
| games of history | 6-10 | 4259 | 349 | -0.03 | -0.07 to -0.00 | * | 1.02 | 2418 | 53% |
| rookie/2nd-yr | no | 3540 | 276 | -0.05 | -0.09 to -0.01 | * | 1.01 | 2009 | 52% |
| rookie/2nd-yr | yes | 927 | 91 | +0.02 | -0.06 to +0.10 |  | 1.09 | 547 | 56% |
| team total | high | 1489 | 214 | -0.03 | -0.09 to +0.03 |  | 1.00 | 853 | 51% |
| team total | low | 1489 | 241 | -0.06 | -0.12 to -0.00 | * | 1.02 | 849 | 54% |
| team total | mid | 1489 | 265 | -0.02 | -0.09 to +0.04 |  | 1.04 | 854 | 55% |
| proj vs line gap | 10-30% | 1588 | 311 | -0.05 | -0.10 to +0.00 |  | 0.98 | 1091 | 52% |
| proj vs line gap | 30-60% | 807 | 267 | -0.07 | -0.14 to +0.01 |  | 1.02 | 702 | 55% |
| proj vs line gap | 60%+ | 657 | 224 | -0.22 | -0.31 to -0.11 | * | 1.10 | 592 | 52% |
| proj vs line gap | <10% | 1415 | 290 | +0.08 | +0.02 to +0.13 | * | 1.03 | 171 | 58% |

## 3. How much of our disagreement with the market is real?

Slope of (actual - line) on (our projection - line), z-units. 1.0 = our number beats the line one-for-one; 0 = the line was right and our difference was noise.

| scope | n | real share | 95% CI | r |
|---|--:|--:|---|--:|
| all props | 4467 | +0.32 | +0.23 to +0.42 | +0.10 |
| labeled picks only (not '—') | 2582 | +0.34 | +0.23 to +0.45 | +0.13 |
| we lean OVER | 3001 | +0.24 | +0.09 to +0.42 | +0.06 |
| we lean UNDER | 1348 | +0.25 | -0.09 to +0.50 | +0.05 |
| position QB | 783 | +0.27 | +0.07 to +0.47 | +0.10 |
| position RB | 1258 | +0.22 | +0.04 to +0.41 | +0.07 |
| position WR | 1590 | +0.40 | +0.19 to +0.60 | +0.12 |
| position TE | 836 | +0.40 | +0.16 to +0.66 | +0.13 |
| receptions | 1586 | +0.39 | +0.25 to +0.54 | +0.13 |
| rec yds | 1605 | +0.37 | +0.20 to +0.53 | +0.11 |
| rush yds | 748 | -0.08 | -0.27 to +0.16 | -0.03 |

## 4. Pick win rate by position and side (labeled picks, n>=20)

| position | lean | n | win% |
|---|---|--:|--:|
| QB | OVER | 207 | 56% |
| QB | UNDER | 139 | 62% |
| RB | OVER | 365 | 52% |
| RB | UNDER | 157 | 60% |
| TE | OVER | 525 | 49% |
| TE | UNDER | 70 | 69% |
| WR | OVER | 857 | 51% |
| WR | UNDER | 236 | 57% |

## 5. Is the TE gap real? (a hypothesis found AFTER the scans above)

Pocket = receptions at <=1.5 plus rec yds at <=40; labeled OVER picks only, whole-number consensus catch lines dropped (their price belongs to a different bet). The pocket was chosen after TEs stood out in section 2, so the in-sample gap in 5a overstates it. 5b is the fair test: each week is fit on earlier weeks only.

**5a. TE vs every other position on the same kind of pick** (gap = TE win% minus the rest, 95% player-clustered CI; ROI at each pick's best price)

| slice | TE n | other n | TE win% | other win% | gap (pts) | 95% CI | TE ROI | other ROI |
|---|--:|--:|--:|--:|--:|---|--:|--:|
| all receiving OVER picks | 510 | 1103 | 48.8% | 52.0% | -3.2 | -9.6 to +3.2 | -8.1% | -1.2% |
| low-line pocket | 357 | 758 | 48.7% | 55.0% | -6.3 | -14.7 to +1.9 | -8.1% | +4.4% |
|   weeks 1-3 | 120 | 207 | 40.0% | 55.6% | -15.6 | -28.0 to -3.3 | -23.9% | +7.0% |
|   weeks 4+ | 237 | 551 | 53.2% | 54.8% | -1.6 | -11.2 to +6.6 | -0.1% | +3.5% |
|   receptions only | 100 | 238 | 51.0% | 61.3% | -10.3 | -23.8 to +2.4 | -6.8% | +15.4% |
|   rec yds only | 257 | 520 | 47.9% | 52.1% | -4.3 | -12.6 to +3.9 | -8.7% | -0.6% |
|   weeks 6+ (forward test) | 151 | 362 | 58.3% | 52.8% | +5.5 | -6.0 to +16.2 | +7.9% | -0.4% |
| mid/high lines (not pocket) | 153 | 345 | 49.0% | 45.5% | +3.5 | -5.0 to +11.1 | -7.9% | -13.5% |

**5b. Walk-forward** (test weeks [3, 4, 5, 6, 7, 8, 9]): does adding a TE / non-TE split to the calibration improve probabilities on weeks it never saw? Brier, lower is better; gain = current minus TE-aware (x1e4), positive favours TE-aware.

| slice | n | model p | current | TE-aware | gain (x1e4) | 95% CI |
|---|--:|--:|--:|--:|--:|---|
| all labeled props | 1943 | 0.2533 | 0.2498 | 0.2514 | -16 | -39 to +9 |
| receiving markets | 1594 | 0.2534 | 0.2504 | 0.2521 | -17 | -42 to +10 |
| TE props | 446 | 0.2521 | 0.2512 | 0.2563 | -51 | -127 to +24 |
| non-TE receiving props | 1148 | 0.2539 | 0.2500 | 0.2504 | -4 | -29 to +21 |
| low-line receiving OVERs | 927 | 0.2547 | 0.2515 | 0.2534 | -19 | -63 to +29 |

Every pick each calibration would make (calibrated p >= 0.55 and +EV), test weeks pooled:

| calibration | picks | win% | ROI |
|---|--:|--:|--:|
| current calibration | 278 | 51.8% | -3.5% |
| TE-aware calibration | 433 | 54.3% | +1.8% |

**Forward test, fixed after week 5:** weeks 6+ are out of sample for this hypothesis (the 'weeks 6+' row in 5a fills in as they are graded). A TE factor goes into `dfs/conviction.py` only if that row shows a gap whose CI excludes zero on 100+ TE pocket picks, or the 5b gain turns positive with a CI excluding zero.