# Does the shipped injury boost also predict better at the player level?

2023-25 weeks 5-18, every player who played; each cell is the pooled out-of-season RMSE of the stats listed, as a % change against the OLD full boost (new 1, ongoing 1), with the scale constants re-learned for the setting on the other two seasons (negative = better than the old rule). Projections are linear in the multiplier, so every setting is built from three runs per unit (no boost / new only / ongoing only). The shipped fractions are `{'rec': {'new': 0.46, 'ongoing': 0.37}, 'rush': {'new': 0.74, 'ongoing': 0.42}}`.

## WR/TE receiving (targets, receptions, receiving yards)

Everyone who played:

| |ongoing 0|ongoing 0.2|ongoing 0.37|ongoing 0.5|ongoing 0.75|ongoing 1|
|---|--:|--:|--:|--:|--:|--:|
| new 0 | +0.53 | +0.39 | +0.33 | +0.30 | +0.27 | +0.30 |
| new 0.14 | +0.40 | +0.28 | +0.22 | +0.19 | +0.16 | +0.20 |
| new 0.3 | +0.29 | +0.17 | +0.11 | +0.08 | +0.05 | +0.09 |
| new 0.46 | +0.20 | +0.08 | +0.02 | -0.01 | -0.03 | +0.01 |
| new 0.6 | +0.15 | +0.04 | -0.02 | -0.05 | -0.07 | -0.02 |
| new 0.75 | +0.14 | +0.03 | -0.03 | -0.06 | -0.08 | -0.03 |
| new 1 | +0.16 | +0.05 | -0.00 | -0.03 | -0.04 | +0.00 |

Players the model expects to be involved under each setting:

| |ongoing 0|ongoing 0.2|ongoing 0.37|ongoing 0.5|ongoing 0.75|ongoing 1|
|---|--:|--:|--:|--:|--:|--:|
| new 0 | +1.44 | +1.16 | +1.03 | +0.96 | +0.86 | +0.86 |
| new 0.14 | +1.23 | +0.97 | +0.88 | +0.76 | +0.71 | +0.71 |
| new 0.3 | +0.95 | +0.69 | +0.60 | +0.54 | +0.46 | +0.43 |
| new 0.46 | +0.73 | +0.48 | +0.44 | +0.35 | +0.29 | +0.24 |
| new 0.6 | +0.64 | +0.42 | +0.35 | +0.29 | +0.20 | +0.18 |
| new 0.75 | +0.54 | +0.35 | +0.23 | +0.19 | +0.08 | +0.10 |
| new 1 | +0.43 | +0.24 | +0.18 | +0.10 | -0.01 | +0.00 |

## RB rushing (carries, rushing yards)

Everyone who played:

| |ongoing 0|ongoing 0.2|ongoing 0.37|ongoing 0.5|ongoing 0.75|ongoing 1|
|---|--:|--:|--:|--:|--:|--:|
| new 0 | +0.76 | +0.41 | +0.24 | +0.18 | +0.23 | +0.50 |
| new 0.14 | +0.31 | -0.04 | -0.21 | -0.27 | -0.22 | +0.05 |
| new 0.3 | -0.04 | -0.39 | -0.56 | -0.62 | -0.57 | -0.30 |
| new 0.46 | -0.23 | -0.57 | -0.74 | -0.81 | -0.76 | -0.50 |
| new 0.6 | -0.25 | -0.59 | -0.77 | -0.83 | -0.79 | -0.53 |
| new 0.75 | -0.14 | -0.48 | -0.66 | -0.72 | -0.69 | -0.44 |
| new 1 | +0.35 | -0.00 | -0.18 | -0.25 | -0.23 | +0.00 |

Players the model expects to be involved under each setting:

| |ongoing 0|ongoing 0.2|ongoing 0.37|ongoing 0.5|ongoing 0.75|ongoing 1|
|---|--:|--:|--:|--:|--:|--:|
| new 0 | +0.51 | +0.18 | +0.10 | +0.01 | +0.25 | +0.54 |
| new 0.14 | +0.15 | -0.19 | -0.27 | -0.36 | -0.11 | +0.21 |
| new 0.3 | -0.41 | -0.71 | -0.80 | -0.88 | -0.64 | -0.31 |
| new 0.46 | -0.61 | -0.92 | -1.00 | -1.08 | -0.84 | -0.55 |
| new 0.6 | -0.46 | -0.76 | -0.84 | -0.93 | -0.69 | -0.41 |
| new 0.75 | -0.38 | -0.69 | -0.78 | -0.87 | -0.67 | -0.37 |
| new 1 | +0.08 | -0.23 | -0.37 | -0.47 | -0.27 | +0.00 |

## RB receiving (receptions, receiving yards), the receiving fractions

Everyone who played:

| |ongoing 0|ongoing 0.2|ongoing 0.37|ongoing 0.5|ongoing 0.75|ongoing 1|
|---|--:|--:|--:|--:|--:|--:|
| new 0 | +0.55 | +0.44 | +0.36 | +0.32 | +0.26 | +0.25 |
| new 0.14 | +0.46 | +0.35 | +0.27 | +0.23 | +0.18 | +0.16 |
| new 0.3 | +0.37 | +0.26 | +0.19 | +0.15 | +0.10 | +0.08 |
| new 0.46 | +0.30 | +0.20 | +0.12 | +0.09 | +0.04 | +0.03 |
| new 0.6 | +0.26 | +0.16 | +0.09 | +0.05 | +0.01 | -0.00 |
| new 0.75 | +0.25 | +0.14 | +0.08 | +0.05 | -0.00 | -0.01 |
| new 1 | +0.25 | +0.15 | +0.08 | +0.05 | +0.01 | +0.00 |

Players the model expects to be involved under each setting:

| |ongoing 0|ongoing 0.2|ongoing 0.37|ongoing 0.5|ongoing 0.75|ongoing 1|
|---|--:|--:|--:|--:|--:|--:|
| new 0 | +0.45 | +0.31 | +0.18 | +0.29 | +0.07 | +0.55 |
| new 0.14 | +0.27 | +0.12 | -0.03 | +0.09 | -0.10 | +0.33 |
| new 0.3 | +0.05 | -0.10 | -0.25 | -0.15 | -0.33 | +0.11 |
| new 0.46 | -0.06 | -0.20 | -0.36 | -0.26 | -0.41 | +0.00 |
| new 0.6 | -0.15 | -0.30 | -0.46 | -0.38 | -0.55 | -0.14 |
| new 0.75 | -0.09 | -0.24 | -0.38 | -0.25 | -0.48 | -0.06 |
| new 1 | +0.01 | -0.22 | -0.39 | -0.23 | -0.41 | +0.00 |

## WR/TE receiving targets by injury context (each setting with its own re-learned scale)

Mean error is projected minus actual targets. The boost only matters where a teammate is hurt: C and D are an Out player newly / already absent, B has only Questionable or Doubtful players.

| context | rows | kappa new / ongoing | RMSE | mean error |
|---|---|---|---|---|
| A no hurt player | 3976 | 1 / 1 | 2.256 | 0.23 |
| B Questionable / Doubtful only | 2714 | 1 / 1 | 2.369 | 0.12 |
| C Out player newly absent (>= 12% vacated) | 1124 | 1 / 1 | 2.509 | 0.25 |
| D Out player already absent (>= 12%) | 579 | 1 / 1 | 2.294 | 0.58 |
| E Out player, small or mixed | 1205 | 1 / 1 | 2.375 | 0.23 |
| A no hurt player | 3976 | 0.46 / 0.37 | 2.26 | 0.38 |
| B Questionable / Doubtful only | 2714 | 0.46 / 0.37 | 2.376 | 0.15 |
| C Out player newly absent (>= 12% vacated) | 1124 | 0.46 / 0.37 | 2.487 | 0.02 |
| D Out player already absent (>= 12%) | 579 | 0.46 / 0.37 | 2.273 | 0.39 |
| E Out player, small or mixed | 1205 | 0.46 / 0.37 | 2.395 | 0.17 |
| A no hurt player | 3976 | 0.14 / 0 | 2.268 | 0.47 |
| B Questionable / Doubtful only | 2714 | 0.14 / 0 | 2.384 | 0.17 |
| C Out player newly absent (>= 12% vacated) | 1124 | 0.14 / 0 | 2.513 | -0.14 |
| D Out player already absent (>= 12%) | 579 | 0.14 / 0 | 2.285 | 0.26 |
| E Out player, small or mixed | 1205 | 0.14 / 0 | 2.417 | 0.12 |
| A no hurt player | 3976 | 0 / 0 | 2.27 | 0.48 |
| B Questionable / Doubtful only | 2714 | 0 / 0 | 2.387 | 0.17 |
| C Out player newly absent (>= 12% vacated) | 1124 | 0 / 0 | 2.529 | -0.2 |
| D Out player already absent (>= 12%) | 579 | 0 / 0 | 2.285 | 0.28 |
| E Out player, small or mixed | 1205 | 0 / 0 | 2.42 | 0.12 |

