# Calibrated-conviction walk-forward — 2026 test weeks [3, 4, 5, 6, 7, 8, 9]

Each test week is calibrated ONLY on earlier weeks; price-aware ROI at the best available price; shrink k=25, min calibrated p=0.55.

## Pooled across test weeks (n = decided picks)

| method | K | n | win% | ROI |
|---|--:|--:|--:|--:|
| calibrated EV | 8 | 55 | 43.6 | -13.4% |
| tier (high>strong>solid, then |z|) | 8 | 55 | 52.7 | -4.9% |
| tier, Recommended only | 8 | 56 | 53.6 | -7.6% |
| model p-edge | 8 | 55 | 54.5 | +23.2% |
| calibrated EV | 16 | 111 | 48.6 | -6.4% |
| tier (high>strong>solid, then |z|) | 16 | 111 | 57.7 | +1.3% |
| tier, Recommended only | 16 | 112 | 56.2 | +0.1% |
| model p-edge | 16 | 111 | 53.2 | +14.4% |
| ALL labeled props (reference) | all | 1943 | 53.5 | -0.5% |
| every calibrated pick (p>=p_min, +EV) | all | 278 | 51.8 | -3.5% |

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
| calibrated EV | 9 | 8 | 50.0 | +5.4% |
| tier (high>strong>solid, then |z|) | 9 | 8 | 50.0 | -4.8% |
| tier, Recommended only | 9 | 8 | 62.5 | +1.1% |
| model p-edge | 9 | 8 | 62.5 | +27.8% |

## Probability quality (Brier, lower is better) — every labeled prop in the test weeks

n=1943; constant (training win rate)=0.2493, model p(hit)=0.2533, calibrated p=0.2498

| calibrated p | n | model said | calibrated said | actually won |
|---|--:|--:|--:|--:|
| <50 | 967 | 56.9% | 45.6% | 50.2% |
| 50-55 | 526 | 59.3% | 52.3% | 56.8% |
| 55-60 | 304 | 61.1% | 57.2% | 59.9% |
| 60-65 | 112 | 64.0% | 61.9% | 52.7% |
| 65+ | 34 | 71.9% | 68.9% | 44.1% |
