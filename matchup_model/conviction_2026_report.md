# Calibrated-conviction walk-forward — 2026 test weeks [3, 4, 5]

Each test week is calibrated ONLY on earlier weeks; price-aware ROI at the best available price; shrink k=25, min calibrated p=0.55.

## Pooled across test weeks (n = decided picks)

| method | K | n | win% | ROI |
|---|--:|--:|--:|--:|
| calibrated EV | 8 | 23 | 34.8 | -28.2% |
| tier (high>strong>solid, then |z|) | 8 | 23 | 52.2 | -2.4% |
| tier, Recommended only | 8 | 24 | 45.8 | -16.0% |
| model p-edge | 8 | 23 | 52.2 | +25.6% |
| calibrated EV | 16 | 47 | 44.7 | -11.1% |
| tier (high>strong>solid, then |z|) | 16 | 47 | 55.3 | -1.0% |
| tier, Recommended only | 16 | 48 | 56.2 | +4.7% |
| model p-edge | 16 | 47 | 51.1 | +12.3% |
| ALL labeled props (reference) | all | 933 | 52.1 | -2.0% |
| every calibrated pick (p>=p_min, +EV) | all | 145 | 49.7 | -6.1% |

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

## Probability quality (Brier, lower is better) — every labeled prop in the test weeks

n=933; constant (training win rate)=0.2504, model p(hit)=0.2570, calibrated p=0.2530

| calibrated p | n | model said | calibrated said | actually won |
|---|--:|--:|--:|--:|
| <50 | 483 | 57.4% | 45.2% | 49.9% |
| 50-55 | 242 | 60.3% | 52.3% | 56.6% |
| 55-60 | 142 | 63.4% | 57.1% | 57.0% |
| 60-65 | 52 | 66.1% | 61.6% | 36.5% |
| 65+ | 14 | 73.5% | 68.6% | 57.1% |
