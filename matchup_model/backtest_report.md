# Matchup model — audit + backtest

**Bottom line: the player coverage-matchup thesis does not produce a projectable
weekly fantasy edge in this data. Tested six ways; every cut is null. Not wired into the app.**

## 1. Data integrity — clean
- Player name-key join to splits/weekly: **100%** (0 missing by FP).
- Opponent-team join to coverage-matrix: **100%** (all 32 teams).
- No duplicate rows. Value ranges sane (overall YPRR 1.55, man 1.54, zone 1.60).
- One dead column: `press` is 100% zero in these exports (not used).

## 2. Signal-existence audit — the exploitable part isn't stable
Year-over-year correlation (players with 100+ man routes both years):

| trait (vs man) | 2022→23 | 2023→24 |
|---|--:|--:|
| **raw** targets/route | 0.57 | 0.64 |
| **raw** yards/route | 0.53 | 0.32 |
| Δ yards/route (man − overall) | 0.37 | 0.32 |
| **Δ fp/route (man − overall)** | **0.13** | **0.14** |

The *raw* split stats look stable — but only because a player's **overall** target/yards
rate is stable (a high-target WR is high-target in every split). The **differential** by
coverage — the only thing a matchup can move — is small and barely persists (r ≈ 0.06–0.14).

## 3. Efficiency-level test — coverage adjustment adds nothing
Predict each WR/TE's **actual weekly targets/route** (2024, wk5+, n=1,825), walk-forward:

| predictor | RMSE |
|---|--:|
| player's prior **overall** targets/route | 0.1010 |
| coverage-weighted blend (opp man/zone/MOF × player splits) | 0.1009 |

The blend does **not** beat "just use their season rate." corr(blend−overall gap,
actual−overall) = +0.06.

## 4. Fantasy-point backtest — walk-forward, 2023–24, 9,186 player-weeks
Two model formulations tried (raw-delta × predicted lean; and coverage-weighted efficiency
blend), three driver stats (fp/route, yards/route, targets/route):

| | ΔRMSE vs naive base | corr(Δ, residual) | directional hit rate |
|---|--:|--:|--:|
| delta × lean, fp/route | −0.003 | −0.005 | — |
| efficiency blend, tprr+yprr | −0.001 | −0.006 | 46.2% |

Naive base = recency-weighted trailing fantasy points. Model never beats it; directional
accuracy is below a coin flip.

## 5. Alternative scheme angles — also null
| angle | result |
|---|---|
| deep-aDOT WR vs 2-high shells | fp **+0.83** vs high-2-high D (wrong sign / confounded), yprr −0.13 |
| slot WR vs nickel/dime rate | no separation |
| QB dropbacks vs opp 2-high% | r = +0.14 (weak) |
| QB fp vs opp zone% | r = −0.11 (weak) |

## Why this is a real result, not a bug
1. The exploitable quantity (a player's *differential* efficiency by coverage) is small and
   doesn't carry year to year.
2. The stable, predictive quantity (overall skill) is already in the base projection and
   priced by the market/DK salaries.
3. Weekly opponent-coverage variation is only ±15pp — even a real differential moves a
   projection < 0.5 fp, inside a 6.3-fp weekly RMSE.

## The one remaining shot (modest odds)
`receiving-separation-by-coverage` — FantasyPoints' own **SEP SCORE** and **WIN RATE** by
Man / Zone / Cover-N — was never pulled (Chrome auto-download block this session). It is a
purpose-built stable metric; there's maybe a 1-in-4 chance its *differential* persists
better than tprr/yprr. Procedure if pulled: run the §3 efficiency-level test on it FIRST;
only build downstream if the blend beats "overall" there.

## What's reusable regardless
`matchup_model/` — ingest for every FantasyPoints table, the (working, r=0.75 half-to-half)
defense scheme predictor, empirical-Bayes shrinkage, and this walk-forward backtest harness.
Swapping the driver stat is a one-line change in `player_splits.FP_STAT` / `_SPEC`.

---

## Regression-to-expected lean — the one thing that DID work (`project.py`)

A player's recent **expected** fantasy points (xFP, from the advanced tables) predicts next
week slightly better than their recent **actual** FP. `lean = 0.5·(trailing_xFP − trailing_FP)`,
shrunk toward each player's own long-run FP−xFP gap (so rushing QBs aren't perpetually faded).
Walk-forward, train 2023 / test 2024:

| position | RMSE naive | RMSE +lean | Spearman naive | Spearman +lean | top−bottom decile of lean |
|---|--:|--:|--:|--:|--:|
| WR/TE | 5.802 | 5.785 | 0.641 | 0.642 | **+1.1 fp** |
| RB | 6.444 | 6.427 | 0.618 | 0.617 | **+1.2 fp** |
| QB | 7.862 | 7.792 | **0.447** | **0.460** | **+3.2 fp** |

Small for WR/RB, genuinely useful for QB (ranking + ~1% RMSE). The lean magnitude is small
(std ~0.7 fp WR, ~1.5 fp QB; clamped ±). It's a **tiebreaker**, not a projection rewrite.

Wired into the app as:
- `dfs/screens.py` **DFS Player Lookup** → a "Regression lean" metric + reason line.
- **DFS Optimizer** → "Apply regression lean" checkbox tilts the objective by
  `1 + clip(lean / max(proj, 6), ±0.12)`.

Needs current-season weekly FantasyPoints exports in `data/dfs/matchup/` (the ingest globs
by year, so newer files just work). Without them the lean is 0 everywhere and the UI says so.
