# Anytime-TD calibration — expected-TD-count -> P(>=1 TD)

- 9213 player-games, [2023, 2024, 2025] wk 4-18.
- `lam` = our model's expected TD count for the game (rec_td+rush_td from opp_line, already blends trailing rate with the Vegas-implied team total). `p_model` = 1-e^-lam. `p_base` = flat league rate for the position group (seasons strictly before the test season) — the "just guess the average" comparator, no player-specific info at all.

## WR/TE  (n=6200)

Brier score (lower = better): **model 0.1711** vs **flat-average baseline 0.1799**  (model wins, +4.9% improvement)
Actual score rate this sample: 23.5%  ·  mean model p: 22.1%  ·  mean baseline p: 22.6%

### Calibration — does the predicted probability match reality?

| pred range | n | mean predicted | actual rate |
|---|--:|--:|--:|
| 5%-10% | 344 | 8.6% | 8.7% |
| 10%-15% | 1209 | 12.8% | 14.7% |
| 15%-20% | 1366 | 17.5% | 17.6% |
| 20%-30% | 2092 | 24.6% | 26.1% |
| 30%-45% | 1113 | 35.4% | 38.0% |
| 45%-101% | 75 | 48.9% | 54.7% |

Weekly top-12 by our model: **41.3%** actually scored vs **24.6%** for a random 12 (n=45 weeks)

## RB  (n=3013)

Brier score (lower = better): **model 0.1867** vs **flat-average baseline 0.2071**  (model wins, +9.9% improvement)
Actual score rate this sample: 29.3%  ·  mean model p: 30.5%  ·  mean baseline p: 28.7%

### Calibration — does the predicted probability match reality?

| pred range | n | mean predicted | actual rate |
|---|--:|--:|--:|
| 10%-15% | 241 | 13.0% | 8.7% |
| 15%-20% | 423 | 17.6% | 13.2% |
| 20%-30% | 875 | 24.7% | 20.8% |
| 30%-45% | 1069 | 37.0% | 38.5% |
| 45%-101% | 392 | 50.8% | 53.6% |

Weekly top-12 by our model: **52.2%** actually scored vs **29.6%** for a random 12 (n=45 weeks)

## Where does the model's edge concentrate?

Same Brier-score check as above, cut into terciles of each axis. `dev` = |model p - baseline p|, i.e. how far the model strays from 'just guess the position average' — the cut that matters most for spotting a bettable edge, since a near-average lam prices about like the market already does.

### WR/TE

**By Vegas implied team total (scoring environment):**

| tercile | n | range | Brier model | Brier baseline | improvement | actual rate |
|---|--:|---|--:|--:|--:|--:|
| low | 2086 | 10.50 to 20.25 | 0.1473 | 0.1530 | +3.7% | 18.6% |
| mid | 2206 | 20.50 to 24.00 | 0.1706 | 0.1763 | +3.2% | 22.8% |
| high | 1908 | 24.25 to 32.25 | 0.1977 | 0.2134 | +7.4% | 29.6% |

**By spread (positive = this team favored):**

| tercile | n | range | Brier model | Brier baseline | improvement | actual rate |
|---|--:|---|--:|--:|--:|--:|
| low | 2341 | -19.50 to -3.00 | 0.1883 | 0.2012 | +6.4% | 27.4% |
| mid | 1961 | -2.50 to 3.00 | 0.1660 | 0.1721 | +3.5% | 22.1% |
| high | 1898 | 3.50 to 19.50 | 0.1551 | 0.1616 | +4.0% | 20.2% |

**By this game's actual role share (target share WR/TE, carry share RB):**

| tercile | n | range | Brier model | Brier baseline | improvement | actual rate |
|---|--:|---|--:|--:|--:|--:|
| low | 2185 | 0.00 to 0.11 | 0.1107 | 0.1182 | +6.3% | 12.3% |
| mid | 1985 | 0.11 to 0.19 | 0.1719 | 0.1712 | -0.4% | 21.9% |
| high | 2030 | 0.20 to 0.60 | 0.2352 | 0.2547 | +7.7% | 37.1% |

**By |model p - baseline p| (model's disagreement with a flat rate):**

| tercile | n | range | Brier model | Brier baseline | improvement | actual rate |
|---|--:|---|--:|--:|--:|--:|
| low | 2069 | 0.00 to 0.04 | 0.1802 | 0.1814 | +0.7% | 23.8% |
| mid | 2069 | 0.04 to 0.09 | 0.1662 | 0.1700 | +2.3% | 21.7% |
| high | 2062 | 0.09 to 0.36 | 0.1669 | 0.1882 | +11.4% | 25.0% |

### RB

**By Vegas implied team total (scoring environment):**

| tercile | n | range | Brier model | Brier baseline | improvement | actual rate |
|---|--:|---|--:|--:|--:|--:|
| low | 1007 | 10.50 to 20.25 | 0.1672 | 0.1783 | +6.2% | 22.5% |
| mid | 1063 | 20.50 to 24.00 | 0.1862 | 0.2032 | +8.4% | 28.3% |
| high | 943 | 24.25 to 32.25 | 0.2081 | 0.2424 | +14.2% | 37.5% |

**By spread (positive = this team favored):**

| tercile | n | range | Brier model | Brier baseline | improvement | actual rate |
|---|--:|---|--:|--:|--:|--:|
| low | 1178 | -19.50 to -3.00 | 0.2079 | 0.2376 | +12.5% | 36.4% |
| mid | 945 | -2.50 to 3.00 | 0.1773 | 0.1916 | +7.4% | 25.6% |
| high | 890 | 3.50 to 19.50 | 0.1685 | 0.1833 | +8.1% | 23.7% |

**By this game's actual role share (target share WR/TE, carry share RB):**

| tercile | n | range | Brier model | Brier baseline | improvement | actual rate |
|---|--:|---|--:|--:|--:|--:|
| low | 1030 | 0.00 to 0.20 | 0.0962 | 0.1188 | +19.1% | 8.5% |
| mid | 986 | 0.20 to 0.47 | 0.2079 | 0.2116 | +1.7% | 30.3% |
| high | 997 | 0.47 to 1.00 | 0.2592 | 0.2939 | +11.8% | 49.6% |

**By |model p - baseline p| (model's disagreement with a flat rate):**

| tercile | n | range | Brier model | Brier baseline | improvement | actual rate |
|---|--:|---|--:|--:|--:|--:|
| low | 1005 | 0.00 to 0.06 | 0.1981 | 0.2002 | +1.1% | 27.7% |
| mid | 1004 | 0.06 to 0.12 | 0.1696 | 0.1842 | +7.9% | 23.9% |
| high | 1004 | 0.12 to 0.41 | 0.1923 | 0.2369 | +18.8% | 36.3% |
