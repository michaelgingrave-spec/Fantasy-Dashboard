# Does the biggest cross-axis mismatch flag the biggest baseline misses?

- Pooled 2025 holdout: 3364 player-games across 3 axes -- 2028 WR/TE (man/zone), 918 RB (box count), 418 QB (1-high/2-high shell).
- `mismatch_pctile` = |signal| ranked to a 0-1 percentile WITHIN each axis's own holdout distribution, so a 'big' shell mismatch and a 'big' box mismatch are comparable -- pooling raw |signal| across axes would just reflect each axis's arbitrary scale, not real mismatch size.
- Each axis keeps its own beta, fit on ONLY that axis's designated train seasons (matches scheme_calib.py/box_calib.py/shell_calib.py exactly) -- nothing here re-fits or leaks holdout into training.

## 1. Does mismatch size predict baseline error at all?

| axis | n | corr(mismatch_pctile, |base error|) |
|---|--:|--:|
| scheme | 2028 | +0.0815 |
| box | 918 | -0.0242 |
| shell | 418 | -0.0123 |
| **pooled** | 3364 | **+0.0261** |

A meaningfully positive pooled correlation is the minimum bar for the whole idea to have anywhere to stand -- if bigger mismatches don't even correlate with bigger baseline misses, no selection rule built on top of these signals can help, independent of which axis wins in a given week.

## 2. Selecting the top-K% biggest mismatch each week, vs. random selection

Real DFS use only ever looks at a handful of marquee edges per week, not the whole slate -- this simulates that: take the top-K% of ALL pooled rows by mismatch_pctile *within each week* (mixing positions/axes, same as the Matchups page mixing 'run game' and 'deep passing' on one scorecard), and check the RMSE improvement (base vs. each row's own axis-beta-adjusted projection) inside that subset against 1,000 random same-size draws from the same week's pool.

| top-K% | n selected | RMSE Δ (selected) | random draws: mean Δ | random draws: p95 Δ | percentile of real vs random |
|---|--:|--:|--:|--:|--:|
| top 10% | 335 | **-0.185** | -0.059 | +0.020 | 1% |
| top 20% | 672 | **-0.178** | -0.062 | -0.015 | 0% |
| top 30% | 1011 | **-0.164** | -0.062 | -0.026 | 0% |

'Percentile of real vs random' near 50% means picking the biggest-looking mismatch each week did no better than picking a random same-size handful of player-games -- the selection itself carries no information. It would need to sit up near 90-95%+ to call the selection mechanism real (matching the standard for the random draws' own p95 line).

Verdict: **no: mismatch size doesn't predict baseline error, selection is noise**