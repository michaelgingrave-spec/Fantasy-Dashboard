# Model edge calibration — vs a trailing-form stand-in line

- 15788 player-week-market rows, [2023, 2024, 2025] wk 4-18. Line = EWMA of the stat over the player's last 8 games.
- **Hit rates are an upper bound** — a real sportsbook line already prices matchup/pace/injuries, so a real edge is smaller. Trust the *ranking* of buckets.

Overall: **8581-7207** (54.4%) · ROI@-110 **+3.8%** · break-even is 52.4%

## By |edge| bucket

| edge range | n | win% | ROI@-110 | dir hit% | mean edge% | mean realized% |
|---|---|---|---|---|---|---|
| 0-4% | 3714 | 51.6 | -1.6 | 50.3 | 1.9 | 1.8 |
| 4-8% | 2759 | 52.2 | -0.4 | 52.2 | 5.9 | 2.3 |
| 8-12% | 2124 | 53.4 | 1.9 | 53.4 | 9.9 | 5.6 |
| 12-20% | 3128 | 54.5 | 4.1 | 54.5 | 15.7 | 8.8 |
| 20%+ | 4063 | 58.8 | 12.2 | 58.8 | 35.5 | 21.6 |

**Best bucket: 20%+** — 58.8% hit, +12.2% ROI on n=4063. ROI keeps climbing with edge here.

## By market

- **pass TD**: 601-449 (57.2%), ROI +9.3%  (n=1050)
- **pass att**: 531-544 (49.4%), ROI -5.7%  (n=1075)
- **pass yds**: 551-469 (54.0%), ROI +3.1%  (n=1020)
- **rec yds**: 2623-2061 (56.0%), ROI +6.9%  (n=4684)
- **receptions**: 2227-1897 (54.0%), ROI +3.1%  (n=4124)
- **rush att**: 916-763 (54.6%), ROI +4.2%  (n=1679)
- **rush yds**: 1132-1024 (52.5%), ROI +0.2%  (n=2156)