# Calibrated-conviction walk-forward — 2026 test weeks [3, 4, 5]

Each test week is calibrated ONLY on earlier weeks; price-aware ROI at the best available price; shrink k=25, min calibrated p=0.55.

## Pooled across test weeks (n = decided picks)

| method | K | n | win% | ROI |
|---|--:|--:|--:|--:|
| calibrated EV | 8 | 23 | 56.5 | +35.0% |
| tier (high>strong>solid, then |z|) | 8 | 22 | 54.5 | +2.0% |
| tier, Recommended only | 8 | 23 | 47.8 | -12.4% |
| model p-edge | 8 | 22 | 54.5 | +31.3% |
| calibrated EV | 16 | 44 | 59.1 | +28.0% |
| tier (high>strong>solid, then |z|) | 16 | 43 | 65.1 | +17.1% |
| tier, Recommended only | 16 | 47 | 55.3 | +2.5% |
| model p-edge | 16 | 44 | 54.5 | +19.9% |
| ALL labeled props (reference) | all | 866 | 54.5 | +2.5% |
| every calibrated pick (p>=p_min, +EV) | all | 235 | 57.4 | +10.1% |

## Per week, top 8

| method | week | n | win% | ROI |
|---|--:|--:|--:|--:|
| calibrated EV | 3 | 8 | 37.5 | -2.7% |
| tier (high>strong>solid, then |z|) | 3 | 7 | 42.9 | -18.5% |
| tier, Recommended only | 3 | 8 | 25.0 | -55.3% |
| model p-edge | 3 | 7 | 28.6 | -21.7% |
| calibrated EV | 4 | 7 | 71.4 | +75.3% |
| tier (high>strong>solid, then |z|) | 4 | 7 | 71.4 | +31.6% |
| tier, Recommended only | 4 | 7 | 85.7 | +59.4% |
| model p-edge | 4 | 7 | 71.4 | +75.3% |
| calibrated EV | 5 | 8 | 62.5 | +37.4% |
| tier (high>strong>solid, then |z|) | 5 | 8 | 50.0 | -5.9% |
| tier, Recommended only | 5 | 8 | 37.5 | -32.3% |
| model p-edge | 5 | 8 | 62.5 | +39.1% |

## Probability quality (Brier, lower is better) — every labeled prop in the test weeks

n=866; constant (training win rate)=0.2489, model p(hit)=0.2503, calibrated p=0.2492

| calibrated p | n | model said | calibrated said | actually won |
|---|--:|--:|--:|--:|
| <50 | 311 | 56.7% | 45.9% | 52.7% |
| 50-55 | 261 | 58.6% | 52.5% | 52.1% |
| 55-60 | 178 | 61.6% | 57.1% | 60.1% |
| 60-65 | 86 | 65.1% | 62.2% | 53.5% |
| 65+ | 30 | 70.8% | 68.4% | 63.3% |
