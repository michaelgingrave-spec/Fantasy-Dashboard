# Model edge calibration — vs a trailing-form stand-in line

- 15788 player-week-market rows, [2023, 2024, 2025] wk 4-18. Line = EWMA of the stat over the player's last 8 games.
- **`z` = edge / outcome-SD** (sigma = a + b*line, fitted per market). This is the unit that's comparable across markets — a +33% edge on a 1.5-catch line and a +7% edge on a 270 pass-yd line are ~0.3 vs ~0.35 SD, not 5x apart.
- Hit rates are an **upper bound** — a real sportsbook line already prices matchup/pace/injuries. Trust the threshold/ranking, not the absolute win%.

Overall: **8581-7207** (54.4%) · ROI@-110 **+3.8%** · break-even is 52.4%

## By confidence tier (the `conf` column — market-offset-adjusted |z|)

| conf | n | win% | ROI@-110 | dir hit% | realized z |
|---|---|---|---|---|---|
| — | 8514 | 51.8 | -1.1 | 51.3 | 0.04 |
| lean | 3567 | 55.6 | 6.2 | 55.6 | 0.16 |
| solid | 2379 | 58.8 | 12.2 | 58.8 | 0.26 |
| strong | 1012 | 59.0 | 12.6 | 59.0 | 0.34 |
| high | 316 | 59.8 | 14.2 | 60.0 | 0.41 |

**Rule: bet at |z| ≥ ~0.30 SD.** Below ~0.15 it's a coin flip (51.8% / -1.1%); from 0.30 up it's a clear edge and stays positive as z grows. First +EV bucket: lean.

## By |z| within each market  (win% / ROI@-110, n)

| market | z<0.30 | 0.30-0.60 | z>=0.60 |
|---|---|---|---|
| rec yds | 54% / +4% (n=2945) | 59% / +12% (n=1407) | 59% / +13% (n=332) |
| receptions | 52% / -1% (n=2709) | 58% / +11% (n=1078) | 57% / +8% (n=337) |
| rush yds | 51% / -3% (n=1575) | 56% / +6% (n=486) | 66% / +27% (n=95) |
| pass yds | 54% / +3% (n=954) | 50% / -5% (n=62) | — |
| pass TD | 56% / +7% (n=908) | 65% / +24% (n=139) | — |
| rush att | 53% / +2% (n=925) | 52% / -1% (n=536) | 66% / +26% (n=218) |
| pass att | 49% / -6% (n=963) | 49% / -7% (n=105) | — |

_rush yds / rush att: only z≥0.60 is reliably +EV. pass att: negative at every z — skip it._

## For reference — the old |edge %| view

| edge % | n | win% | ROI@-110 | dir hit% |
|---|---|---|---|---|
| 0-4% | 3714 | 51.6 | -1.6 | 50.3 |
| 4-8% | 2759 | 52.2 | -0.4 | 52.2 |
| 8-12% | 2124 | 53.4 | 1.9 | 53.4 |
| 12-20% | 3128 | 54.5 | 4.1 | 54.5 |
| 20%+ | 4063 | 58.8 | 12.2 | 58.8 |