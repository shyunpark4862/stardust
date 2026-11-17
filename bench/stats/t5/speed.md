Arrays of 100000 points (k spread over the distribution by its quantiles); times in ms.

| distribution | method | sd 1T | sd 10T | scipy | scipy / sd 1T | scipy / sd 10T |
|---|---|---:|---:|---:|---:|---:|
| Binomial | `pmf` | 0.18 | 0.06 | 4.76 | 26.70 | 85.7 |
| Binomial | `logpmf` | 0.13 | 0.05 | 3.20 | 23.99 | 62.0 |
| Binomial | `cdf` | 0.13 | 0.10 | 4.25 | 32.44 | 42.8 |
| Binomial | `logcdf` | 0.13 | 0.11 | 4.50 | 35.83 | 41.0 |
| Binomial | `ccdf` | 0.13 | 0.10 | 4.08 | 31.47 | 40.2 |
| Binomial | `logccdf` | 0.12 | 0.10 | 4.47 | 36.18 | 46.5 |
| Binomial | `icdf` | 1.80 | 0.45 | 22.47 | 12.48 | 49.6 |
| Binomial | `iccdf` | 1.94 | 0.48 | 22.42 | 11.54 | 46.9 |
| Poisson | `pmf` | 0.10 | 0.03 | 1.90 | 19.58 | 56.4 |
| Poisson | `logpmf` | 0.09 | 0.12 | 1.69 | 18.26 | 13.7 |
| Poisson | `cdf` | 0.11 | 0.08 | 3.63 | 32.06 | 44.3 |
| Poisson | `logcdf` | 0.10 | 0.18 | 3.87 | 39.06 | 21.6 |
| Poisson | `ccdf` | 0.12 | 0.07 | 3.50 | 30.13 | 50.4 |
| Poisson | `logccdf` | 0.10 | 0.19 | 3.83 | 39.65 | 20.4 |
| Poisson | `icdf` | 1.61 | 0.48 | 92.44 | 57.28 | 194.1 |
| Poisson | `iccdf` | 1.71 | 0.45 | 92.65 | 54.14 | 205.7 |
| Geometric | `pmf` | 0.10 | 0.04 | 0.81 | 8.26 | 21.5 |
| Geometric | `logpmf` | 0.10 | 0.02 | 0.90 | 9.30 | 43.5 |
| Geometric | `cdf` | 0.12 | 0.06 | 1.22 | 10.37 | 21.1 |
| Geometric | `logcdf` | 0.10 | 0.03 | 1.40 | 14.15 | 51.6 |
| Geometric | `ccdf` | 0.26 | 0.04 | 0.86 | 3.35 | 23.4 |
| Geometric | `logccdf` | 0.11 | 0.02 | 0.58 | 5.13 | 29.7 |
| Geometric | `icdf` | 0.49 | 0.07 | 1.96 | 3.99 | 27.1 |
| Geometric | `iccdf` | 0.41 | 0.06 | 2.01 | 4.95 | 33.0 |
| NegativeBinomial | `pmf` | 0.13 | 0.04 | 3.37 | 26.79 | 81.3 |
| NegativeBinomial | `logpmf` | 0.11 | 0.13 | 4.13 | 36.41 | 31.0 |
| NegativeBinomial | `cdf` | 0.12 | 0.11 | 13.48 | 116.56 | 119.0 |
| NegativeBinomial | `logcdf` | 0.11 | 0.11 | 24.52 | 219.79 | 229.1 |
| NegativeBinomial | `ccdf` | 0.12 | 0.12 | 13.40 | 108.40 | 113.4 |
| NegativeBinomial | `logccdf` | 0.12 | 0.11 | 13.41 | 115.38 | 121.3 |
| NegativeBinomial | `icdf` | 2.22 | 0.46 | 86.71 | 39.03 | 187.8 |
| NegativeBinomial | `iccdf` | 2.28 | 0.58 | 84.38 | 36.97 | 146.5 |
| Hypergeometric | `pmf` | 0.13 | 0.05 | 50.21 | 386.99 | 1058.0 |
| Hypergeometric | `logpmf` | 0.12 | 0.05 | 12.49 | 100.47 | 231.0 |
| Hypergeometric | `cdf` | 0.14 | 0.41 | 60.48 | 441.31 | 146.0 |
| Hypergeometric | `logcdf` | 0.13 | 0.40 | 5066.77 | 40053.48 | 12711.9 |
| Hypergeometric | `ccdf` | 0.13 | 0.42 | 59.27 | 445.05 | 141.4 |
| Hypergeometric | `logccdf` | 0.13 | 0.43 | 4941.11 | 39358.25 | 11524.5 |
| Hypergeometric | `icdf` | 1.95 | 0.45 | 1503.18 | 771.77 | 3326.2 |
| Hypergeometric | `iccdf` | 2.06 | 0.45 | 1500.50 | 729.17 | 3315.4 |
| Bernoulli | `pmf` | 0.08 | 0.01 | 2.59 | 33.05 | 176.9 |
| Bernoulli | `logpmf` | 0.38 | 0.06 | 2.33 | 6.07 | 39.8 |
| Bernoulli | `cdf` | 0.08 | 0.13 | 1.06 | 13.59 | 8.0 |
| Bernoulli | `logcdf` | 0.35 | 0.19 | 1.13 | 3.23 | 6.0 |
| Bernoulli | `ccdf` | 0.08 | 0.14 | 0.89 | 11.35 | 6.4 |
| Bernoulli | `logccdf` | 0.29 | 0.05 | 0.99 | 3.41 | 20.6 |
| Bernoulli | `icdf` | 1.58 | 0.24 | 3.98 | 2.52 | 16.7 |
| Bernoulli | `iccdf` | 1.72 | 0.23 | 4.06 | 2.36 | 17.4 |
| DiscreteUniform | `pmf` | 0.10 | 0.04 | 0.48 | 4.64 | 12.9 |
| DiscreteUniform | `logpmf` | 0.25 | 0.04 | 0.64 | 2.52 | 16.9 |
| DiscreteUniform | `cdf` | 0.10 | 0.14 | 0.64 | 6.25 | 4.5 |
| DiscreteUniform | `logcdf` | 0.12 | 0.03 | 0.84 | 7.01 | 29.2 |
| DiscreteUniform | `ccdf` | 0.10 | 0.12 | 0.61 | 6.23 | 5.0 |
| DiscreteUniform | `logccdf` | 0.11 | 0.03 | 0.83 | 7.57 | 33.1 |
| DiscreteUniform | `icdf` | 0.70 | 0.10 | 0.82 | 1.17 | 8.1 |
| DiscreteUniform | `iccdf` | 0.72 | 0.26 | 0.83 | 1.16 | 3.2 |
| BetaBinomial | `pmf` | 0.13 | 0.10 | 11.05 | 85.75 | 114.0 |
| BetaBinomial | `logpmf` | 0.12 | 0.09 | 10.54 | 85.81 | 113.0 |
| BetaBinomial | `cdf` | 0.13 | 0.38 | 662.60 | 5030.86 | 1735.3 |
| BetaBinomial | `logcdf` | 0.14 | 0.41 | 668.35 | 4827.09 | 1612.9 |
| BetaBinomial | `ccdf` | 0.14 | 0.42 | 658.43 | 4639.55 | 1572.7 |
| BetaBinomial | `logccdf` | 0.14 | 0.41 | 662.11 | 4684.73 | 1610.3 |
| BetaBinomial | `icdf` | 70.07 | 8.23 | 6407.49 | 91.45 | 778.9 |
| BetaBinomial | `iccdf` | 57.42 | 6.65 | 6351.99 | 110.62 | 954.9 |
| LogSeries | `pmf` | 0.10 | 0.02 | 1.01 | 10.34 | 45.4 |
| LogSeries | `logpmf` | 0.10 | 0.15 | 1.16 | 12.11 | 7.7 |
| LogSeries | `cdf` | 0.10 | 0.29 | 217.61 | 2186.09 | 742.5 |
| LogSeries | `logcdf` | 0.10 | 0.32 | 210.27 | 2038.16 | 660.5 |
| LogSeries | `ccdf` | 0.10 | 0.10 | 17.21 | 173.04 | 169.6 |
| LogSeries | `logccdf` | 0.09 | 0.09 | 17.44 | 187.39 | 186.2 |
| LogSeries | `icdf` | 1.04 | 0.45 | 3651.01 | 3494.07 | 8094.6 |
| LogSeries | `iccdf` | 1.13 | 0.45 | 3682.83 | 3272.78 | 8168.9 |
| Zipf | `pmf` | 0.10 | 0.04 | 7.87 | 79.89 | 195.2 |
| Zipf | `logpmf` | 0.09 | 0.16 | 8.04 | 86.53 | 49.5 |
| Zipf | `cdf` | 0.11 | 0.41 | 222.83 | 1943.28 | 540.1 |
| Zipf | `logcdf` | 0.10 | 0.29 | 226.36 | 2232.87 | 781.2 |
| Zipf | `ccdf` | 0.12 | 0.09 | 226.89 | 1901.29 | 2470.7 |
| Zipf | `logccdf` | 0.11 | 0.32 | 225.93 | 2128.07 | 714.7 |
| Zipf | `icdf` | 1.15 | 0.45 | — | — | — |
| Zipf | `iccdf` | 1.04 | 0.45 | — | — | — |

One Python float (cdf), in microseconds:

| distribution | sd | scipy | scipy / sd |
|---|---:|---:|---:|
| Binomial | 0.38 | 16.09 | 41.8 |
| Poisson | 0.23 | 13.71 | 60.6 |
| Geometric | 0.19 | 15.14 | 80.9 |
| NegativeBinomial | 0.25 | 14.86 | 60.6 |
| Hypergeometric | 0.64 | 20.16 | 31.3 |
| Bernoulli | 0.17 | 14.27 | 84.5 |
| DiscreteUniform | 0.20 | 16.83 | 83.9 |
| BetaBinomial | 0.53 | 25.87 | 48.5 |
| LogSeries | 0.30 | 18.63 | 62.2 |
| Zipf | 0.18 | 19.06 | 105.5 |

Wide parameters (nearly every k different), 20000 points; ns per element.

| distribution | method | sd 1T | sd 10T | scipy | scipy / sd 1T | scipy / sd 10T |
|---|---|---:|---:|---:|---:|---:|
| Binomial | `pmf` | 15.3 | 2.2 | 33.9 | 2.22 | 15.5 |
| Binomial | `logpmf` | 15.0 | 2.2 | 22.8 | 1.52 | 10.3 |
| Binomial | `cdf` | 90.2 | 10.9 | 4648.7 | 51.52 | 426.8 |
| Binomial | `logcdf` | 91.0 | 11.1 | 4649.5 | 51.09 | 418.2 |
| Binomial | `ccdf` | 94.8 | 11.2 | 4647.0 | 49.02 | 416.5 |
| Binomial | `logccdf` | 96.4 | 11.8 | 4645.7 | 48.17 | 393.6 |
| Binomial | `icdf` | 209.5 | 25.9 | 20925.6 | 99.88 | 807.6 |
| Binomial | `iccdf` | 207.8 | 25.8 | 20896.6 | 100.54 | 808.8 |
| Poisson | `pmf` | 9.2 | 1.5 | 11.6 | 1.26 | 8.0 |
| Poisson | `logpmf` | 9.0 | 1.5 | 9.5 | 1.06 | 6.5 |
| Poisson | `cdf` | 44.3 | 6.5 | 43.8 | 0.99 | 6.7 |
| Poisson | `logcdf` | 46.0 | 5.8 | 45.4 | 0.99 | 7.9 |
| Poisson | `ccdf` | 44.5 | 5.7 | 41.3 | 0.93 | 7.3 |
| Poisson | `logccdf` | 46.1 | 5.9 | 43.8 | 0.95 | 7.4 |
| Poisson | `icdf` | 141.4 | 17.4 | 309544.6 | 2188.85 | 17751.7 |
| Poisson | `iccdf` | 143.0 | 17.4 | 289117.5 | 2021.65 | 16612.0 |
| Geometric | `pmf` | 4.1 | 0.9 | 8.1 | 1.98 | 9.2 |
| Geometric | `logpmf` | 2.8 | 0.7 | 8.2 | 2.97 | 12.5 |
| Geometric | `cdf` | 7.9 | 1.4 | 10.3 | 1.30 | 7.3 |
| Geometric | `logcdf` | 8.0 | 7.8 | 12.6 | 1.57 | 1.6 |
| Geometric | `ccdf` | 2.4 | 0.5 | 8.7 | 3.62 | 17.0 |
| Geometric | `logccdf` | 1.1 | 0.3 | 6.2 | 5.76 | 18.4 |
| Geometric | `icdf` | 4.3 | 6.9 | 17.6 | 4.10 | 2.6 |
| Geometric | `iccdf` | 3.9 | 0.7 | 18.7 | 4.76 | 25.5 |
| NegativeBinomial | `pmf` | 16.1 | 2.3 | 33.7 | 2.09 | 14.8 |
| NegativeBinomial | `logpmf` | 16.0 | 2.3 | 20.4 | 1.28 | 9.0 |
| NegativeBinomial | `cdf` | 133.4 | 15.5 | 1920.2 | 14.39 | 124.0 |
| NegativeBinomial | `logcdf` | 133.2 | 15.8 | 2978.2 | 22.35 | 188.8 |
| NegativeBinomial | `ccdf` | 127.3 | 15.7 | 1887.0 | 14.82 | 120.4 |
| NegativeBinomial | `logccdf` | 129.7 | 15.7 | 1888.1 | 14.56 | 120.3 |
| NegativeBinomial | `icdf` | 294.7 | 36.0 | 8537.6 | 28.97 | 236.9 |
| NegativeBinomial | `iccdf` | 298.1 | 36.0 | 8560.5 | 28.72 | 237.8 |
| Hypergeometric | `pmf` | 17.5 | 6.0 | 281.6 | 16.05 | 47.1 |
| Hypergeometric | `logpmf` | 13.3 | 4.9 | 74.6 | 5.62 | 15.4 |
| Hypergeometric | `cdf` | 4431.4 | 765.4 | 10388.5 | 2.34 | 13.6 |
| Hypergeometric | `logcdf` | 4423.3 | 743.0 | 26737810.4 | 6044.78 | 35987.3 |
| Hypergeometric | `ccdf` | 4435.9 | 819.0 | 10412.4 | 2.35 | 12.7 |
| Hypergeometric | `logccdf` | 4457.8 | 805.5 | 26594253.1 | 5965.85 | 33015.0 |
| Hypergeometric | `icdf` | 1263.4 | 921.9 | 227240.4 | 179.86 | 246.5 |
| Hypergeometric | `iccdf` | 1240.6 | 868.7 | 223970.4 | 180.54 | 257.8 |
| DiscreteUniform | `pmf` | 1.0 | 3.5 | 5.8 | 5.51 | 1.7 |
| DiscreteUniform | `logpmf` | 2.5 | 0.6 | 7.2 | 2.92 | 11.9 |
| DiscreteUniform | `cdf` | 1.0 | 0.4 | 6.3 | 6.47 | 17.1 |
| DiscreteUniform | `logcdf` | 5.0 | 1.5 | 7.2 | 1.43 | 4.9 |
| DiscreteUniform | `ccdf` | 0.9 | 6.0 | 5.7 | 6.19 | 0.9 |
| DiscreteUniform | `logccdf` | 3.6 | 1.3 | 7.6 | 2.11 | 5.9 |
| DiscreteUniform | `icdf` | 6.4 | 1.1 | 8.8 | 1.37 | 7.7 |
| DiscreteUniform | `iccdf` | 6.5 | 7.1 | 8.8 | 1.35 | 1.2 |
| BetaBinomial | `pmf` | 86.7 | 10.5 | 59.3 | 0.68 | 5.7 |
| BetaBinomial | `logpmf` | 84.1 | 11.1 | 56.1 | 0.67 | 5.0 |
| BetaBinomial | `cdf` | 24236.0 | 2680.7 | 360568.9 | 14.88 | 134.5 |
| BetaBinomial | `logcdf` | 24245.3 | 2588.6 | 360863.3 | 14.88 | 139.4 |
| BetaBinomial | `ccdf` | 24231.6 | 2709.2 | 361045.3 | 14.90 | 133.3 |
| BetaBinomial | `logccdf` | 24269.4 | 2521.8 | 360982.6 | 14.87 | 143.1 |
| BetaBinomial | `icdf` | 48089.5 | 5216.9 | 5254476.0 | 109.26 | 1007.2 |
| BetaBinomial | `iccdf` | 46693.7 | 5183.2 | 5201033.4 | 111.39 | 1003.4 |
| LogSeries | `pmf` | 4.6 | 1.1 | 11.0 | 2.36 | 10.0 |
| LogSeries | `logpmf` | 4.7 | 1.1 | 12.0 | 2.53 | 10.5 |
| LogSeries | `cdf` | 1171.4 | 125.9 | 2818589.6 | 2406.08 | 22396.4 |
| LogSeries | `logcdf` | 1310.4 | 143.5 | 2753646.1 | 2101.31 | 19190.3 |
| LogSeries | `ccdf` | 76.6 | 9.7 | 125.1 | 1.63 | 12.9 |
| LogSeries | `logccdf` | 80.6 | 9.6 | 127.6 | 1.58 | 13.3 |
| LogSeries | `icdf` | 3876.5 | 448.0 | — | — | — |
| LogSeries | `iccdf` | 3907.2 | 448.2 | — | — | — |
| Zipf | `pmf` | 15.2 | 1.9 | 81.2 | 5.34 | 42.7 |
| Zipf | `logpmf` | 11.2 | 1.7 | 83.4 | 7.47 | 49.8 |
| Zipf | `cdf` | 2376.9 | 245.8 | — | — | — |
| Zipf | `logcdf` | 1176.3 | 120.0 | — | — | — |
| Zipf | `ccdf` | 55.1 | 5.0 | — | — | — |
| Zipf | `logccdf` | 1189.3 | 119.8 | — | — | — |
| Zipf | `icdf` | 21852.4 | 2309.7 | — | — | — |
| Zipf | `iccdf` | 21832.2 | 2325.0 | — | — | — |
