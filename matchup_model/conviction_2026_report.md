# Calibrated-conviction walk-forward — 2026 test weeks [3, 4, 5, 6, 7]

Each test week is calibrated ONLY on earlier weeks; price-aware ROI at the best available price; shrink k=25, min calibrated p=0.55.

## Pooled across test weeks (n = decided picks)

| method | K | n | win% | ROI |
|---|--:|--:|--:|--:|
| calibrated EV | 8 | 39 | 41.0 | -18.9% |
| tier (high>strong>solid, then |z|) | 8 | 39 | 53.8 | -2.7% |
| tier, Recommended only | 8 | 40 | 52.5 | -7.6% |
| model p-edge | 8 | 39 | 53.8 | +25.2% |
| calibrated EV | 16 | 79 | 48.1 | -7.5% |
| tier (high>strong>solid, then |z|) | 16 | 79 | 59.5 | +5.1% |
| tier, Recommended only | 16 | 80 | 56.2 | +0.9% |
| model p-edge | 16 | 79 | 51.9 | +12.4% |
| ALL labeled props (reference) | all | 1428 | 53.2 | -0.9% |
| every calibrated pick (p>=p_min, +EV) | all | 197 | 49.7 | -6.9% |

## Per week, top 8

| method | week | n | win% | ROI |
|---|--:|--:|--:|--:|
| calibrated EV | 3 | 8 | 12.5 | -66.8% |
| tier (high>strong>solid, then |z|) | 3 | 8 | 37.5 | -28.7% |
| tier, Recommended only | 3 | 8 | 25.0 | -55.3% |
| model p-edge | 3 | 8 | 25.0 | -31.5% |
| calibrated EV | 4 | 7 | 57.1 | +9.0% |
| tier (high>strong>solid, then |z|) | 4 | 7 | 71.4 | +31.6% |
| tier, Recommended only | 4 | 8 | 75.0 | +39.5% |
| model p-edge | 4 | 7 | 71.4 | +75.3% |
| calibrated EV | 5 | 8 | 37.5 | -22.3% |
| tier (high>strong>solid, then |z|) | 5 | 8 | 50.0 | -5.9% |
| tier, Recommended only | 5 | 8 | 37.5 | -32.3% |
| model p-edge | 5 | 8 | 62.5 | +39.1% |
| calibrated EV | 6 | 8 | 37.5 | -23.1% |
| tier (high>strong>solid, then |z|) | 6 | 8 | 50.0 | -12.3% |
| tier, Recommended only | 6 | 8 | 50.0 | -16.1% |
| model p-edge | 6 | 8 | 62.5 | +36.8% |
| calibrated EV | 7 | 8 | 62.5 | +12.0% |
| tier (high>strong>solid, then |z|) | 7 | 8 | 62.5 | +6.2% |
| tier, Recommended only | 7 | 8 | 75.0 | +26.1% |
| model p-edge | 7 | 8 | 50.0 | +12.4% |

## Probability quality (Brier, lower is better) — every labeled prop in the test weeks

n=1428; constant (training win rate)=0.2497, model p(hit)=0.2545, calibrated p=0.2512

| calibrated p | n | model said | calibrated said | actually won |
|---|--:|--:|--:|--:|
| <50 | 760 | 57.1% | 45.6% | 50.4% |
| 50-55 | 366 | 59.7% | 52.3% | 57.9% |
| 55-60 | 209 | 62.4% | 57.2% | 57.4% |
| 60-65 | 72 | 65.3% | 61.8% | 44.4% |
| 65+ | 21 | 72.8% | 69.0% | 57.1% |
