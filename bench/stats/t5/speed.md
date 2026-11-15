Arrays of 100000 points (k spread over the distribution by its quantiles); times in ms.

| distribution | method | sd 1T | sd 10T | scipy | scipy / sd 1T | scipy / sd 10T |
|---|---|---:|---:|---:|---:|---:|
| Binomial | `pmf` | 11.23 | 1.22 | 4.20 | 0.37 | 3.4 |
| Binomial | `logpmf` | 10.20 | 1.19 | 3.15 | 0.31 | 2.6 |
| Binomial | `cdf` | 15.36 | 1.72 | 4.43 | 0.29 | 2.6 |
| Binomial | `logcdf` | 21.18 | 1.89 | 4.64 | 0.22 | 2.5 |
| Binomial | `ccdf` | 15.28 | 1.72 | 4.45 | 0.29 | 2.6 |
| Binomial | `logccdf` | 19.96 | 1.81 | 4.63 | 0.23 | 2.6 |
| Binomial | `icdf` | 79.12 | 6.86 | 23.45 | 0.30 | 3.4 |
| Binomial | `iccdf` | 78.87 | 6.87 | 23.47 | 0.30 | 3.4 |
| Poisson | `pmf` | 3.02 | 0.37 | 2.01 | 0.67 | 5.4 |
| Poisson | `logpmf` | 2.48 | 0.34 | 1.77 | 0.71 | 5.3 |
| Poisson | `cdf` | 3.96 | 0.49 | 3.85 | 0.97 | 7.9 |
| Poisson | `logcdf` | 4.21 | 0.52 | 4.01 | 0.95 | 7.7 |
| Poisson | `ccdf` | 4.15 | 0.49 | 3.83 | 0.92 | 7.8 |
| Poisson | `logccdf` | 4.33 | 0.53 | 4.05 | 0.94 | 7.7 |
| Poisson | `icdf` | 13.72 | 1.50 | 96.80 | 7.06 | 64.3 |
| Poisson | `iccdf` | 13.29 | 1.51 | 95.51 | 7.19 | 63.4 |
| Geometric | `pmf` | 1.41 | 0.20 | 0.84 | 0.60 | 4.2 |
| Geometric | `logpmf` | 1.06 | 0.22 | 0.94 | 0.88 | 4.2 |
| Geometric | `cdf` | 1.12 | 0.22 | 1.24 | 1.11 | 5.7 |
| Geometric | `logcdf` | 1.99 | 0.27 | 1.48 | 0.74 | 5.5 |
| Geometric | `ccdf` | 0.23 | 0.18 | 0.87 | 3.73 | 4.8 |
| Geometric | `logccdf` | 0.13 | 0.18 | 0.58 | 4.54 | 3.2 |
| Geometric | `icdf` | 2.04 | 0.29 | 2.07 | 1.01 | 7.2 |
| Geometric | `iccdf` | 2.07 | 0.27 | 2.09 | 1.01 | 7.8 |
| NegativeBinomial | `pmf` | 11.07 | 1.28 | 3.30 | 0.30 | 2.6 |
| NegativeBinomial | `logpmf` | 10.55 | 1.24 | 4.27 | 0.41 | 3.4 |
| NegativeBinomial | `cdf` | 13.37 | 1.51 | 14.09 | 1.05 | 9.3 |
| NegativeBinomial | `logcdf` | 15.89 | 1.84 | 25.29 | 1.59 | 13.7 |
| NegativeBinomial | `ccdf` | 13.56 | 1.52 | 13.75 | 1.01 | 9.1 |
| NegativeBinomial | `logccdf` | 16.82 | 1.84 | 14.22 | 0.85 | 7.7 |
| NegativeBinomial | `icdf` | 75.71 | 7.41 | 89.10 | 1.18 | 12.0 |
| NegativeBinomial | `iccdf` | 76.36 | 7.39 | 86.45 | 1.13 | 11.7 |
| Hypergeometric | `pmf` | 27.05 | 2.98 | 52.87 | 1.95 | 17.7 |
| Hypergeometric | `logpmf` | 26.21 | 2.95 | 12.99 | 0.50 | 4.4 |
| Hypergeometric | `cdf` | 181.99 | 21.32 | 64.45 | 0.35 | 3.0 |
| Hypergeometric | `logcdf` | 183.36 | 20.47 | 5396.14 | 29.43 | 263.6 |
| Hypergeometric | `ccdf` | 181.70 | 20.07 | 63.41 | 0.35 | 3.2 |
| Hypergeometric | `logccdf` | 183.27 | 20.13 | 5398.44 | 29.46 | 268.1 |
| Hypergeometric | `icdf` | 929.11 | 106.86 | 1663.66 | 1.79 | 15.6 |
| Hypergeometric | `iccdf` | 933.61 | 100.18 | 1637.53 | 1.75 | 16.3 |
| Bernoulli | `pmf` | 0.08 | 0.18 | 2.82 | 33.80 | 15.8 |
| Bernoulli | `logpmf` | 0.40 | 0.20 | 2.55 | 6.33 | 13.0 |
| Bernoulli | `cdf` | 0.10 | 0.18 | 1.21 | 12.21 | 6.7 |
| Bernoulli | `logcdf` | 0.35 | 0.21 | 1.24 | 3.58 | 6.1 |
| Bernoulli | `ccdf` | 0.09 | 0.18 | 0.97 | 10.83 | 5.3 |
| Bernoulli | `logccdf` | 0.32 | 0.21 | 1.03 | 3.24 | 5.0 |
| Bernoulli | `icdf` | 1.42 | 0.25 | 4.31 | 3.03 | 17.0 |
| Bernoulli | `iccdf` | 1.43 | 0.22 | 4.21 | 2.95 | 18.9 |
| DiscreteUniform | `pmf` | 0.13 | 0.18 | 0.53 | 4.21 | 2.9 |
| DiscreteUniform | `logpmf` | 0.28 | 0.19 | 0.66 | 2.38 | 3.5 |
| DiscreteUniform | `cdf` | 0.12 | 0.18 | 0.66 | 5.54 | 3.7 |
| DiscreteUniform | `logcdf` | 0.88 | 0.20 | 0.88 | 0.99 | 4.5 |
| DiscreteUniform | `ccdf` | 0.12 | 0.18 | 0.61 | 5.29 | 3.4 |
| DiscreteUniform | `logccdf` | 0.91 | 0.23 | 0.89 | 0.98 | 3.8 |
| DiscreteUniform | `icdf` | 0.75 | 0.25 | 0.88 | 1.16 | 3.5 |
| DiscreteUniform | `iccdf` | 0.79 | 0.21 | 0.86 | 1.08 | 4.0 |
| BetaBinomial | `pmf` | 33.08 | 3.70 | 11.82 | 0.36 | 3.2 |
| BetaBinomial | `logpmf` | 31.70 | 3.67 | 11.50 | 0.36 | 3.1 |
| BetaBinomial | `cdf` | 106.53 | 11.89 | 708.90 | 6.65 | 59.6 |
| BetaBinomial | `logcdf` | 106.53 | 11.86 | 709.17 | 6.66 | 59.8 |
| BetaBinomial | `ccdf` | 105.77 | 11.90 | 709.63 | 6.71 | 59.6 |
| BetaBinomial | `logccdf` | 106.83 | 11.94 | 708.67 | 6.63 | 59.3 |
| BetaBinomial | `icdf` | 744.64 | 82.84 | 6771.57 | 9.09 | 81.7 |
| BetaBinomial | `iccdf` | 740.62 | 82.58 | 6752.48 | 9.12 | 81.8 |
| LogSeries | `pmf` | 2.30 | 0.30 | 1.07 | 0.46 | 3.5 |
| LogSeries | `logpmf` | 1.91 | 0.27 | 1.20 | 0.63 | 4.4 |
| LogSeries | `cdf` | 11.80 | 1.37 | 229.69 | 19.47 | 167.9 |
| LogSeries | `logcdf` | 13.38 | 1.50 | 230.13 | 17.20 | 153.6 |
| LogSeries | `ccdf` | 13.34 | 1.45 | 18.64 | 1.40 | 12.9 |
| LogSeries | `logccdf` | 11.33 | 1.33 | 18.83 | 1.66 | 14.2 |
| LogSeries | `icdf` | 24.37 | 2.79 | 3964.69 | 162.70 | 1420.2 |
| LogSeries | `iccdf` | 24.41 | 2.78 | 3978.05 | 162.99 | 1431.6 |
| Zipf | `pmf` | 1.54 | 0.24 | 8.47 | 5.52 | 35.6 |
| Zipf | `logpmf` | 1.13 | 0.22 | 8.80 | 7.79 | 40.0 |
| Zipf | `cdf` | 1.03 | 0.22 | 240.12 | 232.27 | 1114.4 |
| Zipf | `logcdf` | 9.39 | 1.06 | 239.48 | 25.50 | 225.3 |
| Zipf | `ccdf` | 8.51 | 0.97 | 239.86 | 28.19 | 247.7 |
| Zipf | `logccdf` | 11.43 | 1.37 | 241.79 | 21.15 | 177.0 |
| Zipf | `icdf` | 12.84 | 1.49 | — | — | — |
| Zipf | `iccdf` | 12.37 | 1.39 | — | — | — |

One Python float (cdf), in microseconds:

| distribution | sd | scipy | scipy / sd |
|---|---:|---:|---:|
| Binomial | 0.38 | 16.47 | 43.8 |
| Poisson | 0.22 | 13.99 | 64.8 |
| Geometric | 0.18 | 15.29 | 85.4 |
| NegativeBinomial | 0.24 | 15.70 | 66.3 |
| Hypergeometric | 2.05 | 21.62 | 10.5 |
| Bernoulli | 0.17 | 15.18 | 89.1 |
| DiscreteUniform | 0.21 | 18.26 | 87.4 |
| BetaBinomial | 1.23 | 27.57 | 22.4 |
| LogSeries | 0.30 | 20.27 | 67.1 |
| Zipf | 0.17 | 19.92 | 113.9 |
