# Benchmark (2026-10-16 05:56 UTC)

Dataset: synthetic Google/Meta/Bing daily, 2023-01-01 to 2025-12-31. Rolling origin backtest: 12 folds x 28-day horizon, planned spend known.

| Model | Total MAPE | Total P10-P90 coverage | Channel MAPE | Channel coverage | Scaled pinball | Fit+forecast (s) |
|---|---|---|---|---|---|---|
| ROAS-driven quantile (ours) | 6.00% | 81.8% | 9.42% | 80.7% | 0.0266 | 1.741 |
| Prophet (univariate) | 12.74% | 84.5% | 17.39% | 77.7% | 0.0530 | 0.873 |
| Prophet + spend regressor | 9.09% | 86.6% | 12.56% | 80.8% | 0.0380 | 0.848 |
| SARIMA(1,1,1)(1,0,1)[7] | 15.97% | 79.5% | 18.82% | 80.3% | 0.0641 | 2.133 |

Per channel (ROAS model):

| Series | MAPE | Coverage |
|---|---|---|
| bing | 12.02% | 80.4% |
| google | 6.88% | 81.8% |
| meta | 9.36% | 79.8% |
| total | 6.00% | 81.8% |
| channel_avg | 9.42% | 80.7% |

Caching: 25 scenario forecasts took 41.989s uncached vs 2.788s cached from cold (15.1x) and 1.065s warm (39.4x). Backtest re-run: 6.975s -> 0.129s (54.1x).

Optimiser: mean predicted uplift 2.403% vs true uplift under the data-generating process 1.466% (same total budget).
