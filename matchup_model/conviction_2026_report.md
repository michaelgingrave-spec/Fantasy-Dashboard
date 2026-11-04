# Calibrated-conviction walk-forward — 2026 test weeks [3, 4, 5, 6, 7, 8]

Each test week is calibrated ONLY on earlier weeks; price-aware ROI at the best available price; shrink k=25, min calibrated p=0.55.

## Pooled across test weeks (n = decided picks)

| method | K | n | win% | ROI |
|---|--:|--:|--:|--:|
| calibrated EV | 8 | 47 | 42.6 | -16.6% |
| tier (high>strong>solid, then |z|) | 8 | 47 | 53.2 | -4.9% |
| tier, Recommended only | 8 | 48 | 52.1 | -9.1% |
| model p-edge | 8 | 47 | 53.2 | +22.4% |
| calibrated EV | 16 | 95 | 48.4 | -7.6% |
| tier (high>strong>solid, then |z|) | 16 | 95 | 58.9 | +3.1% |
| tier, Recommended only | 16 | 96 | 55.2 | -1.1% |
| model p-edge | 16 | 95 | 51.6 | +11.2% |
| ALL labeled props (reference) | all | 1671 | 53.1 | -1.3% |
| every calibrated pick (p>=p_min, +EV) | all | 226 | 50.0 | -6.9% |

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
| calibrated EV | 8 | 8 | 50.0 | -5.3% |
| tier (high>strong>solid, then |z|) | 8 | 8 | 50.0 | -15.9% |
| tier, Recommended only | 8 | 8 | 50.0 | -16.5% |
| model p-edge | 8 | 8 | 50.0 | +9.0% |

## Probability quality (Brier, lower is better) — every labeled prop in the test weeks

n=1671; constant (training win rate)=0.2496, model p(hit)=0.2542, calibrated p=0.2504

| calibrated p | n | model said | calibrated said | actually won |
|---|--:|--:|--:|--:|
| <50 | 861 | 56.9% | 45.6% | 50.1% |
| 50-55 | 443 | 59.5% | 52.2% | 56.7% |
| 55-60 | 254 | 61.6% | 57.2% | 59.8% |
| 60-65 | 86 | 64.5% | 61.9% | 45.3% |
| 65+ | 27 | 72.6% | 69.0% | 51.9% |
