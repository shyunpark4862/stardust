Arrays of 100000 points (x spread over the distribution by its quantiles); times in ms.

| distribution | method | sd 1T | sd 10T | scipy | scipy / sd 1T | scipy / sd 10T |
|---|---|---:|---:|---:|---:|---:|
| Logistic | `pdf` | 2.95 | 0.34 | 1.65 | 0.56 | 4.9 |
| Logistic | `logpdf` | 2.15 | 0.27 | 1.42 | 0.66 | 5.3 |
| Logistic | `cdf` | 1.14 | 0.20 | 0.94 | 0.82 | 4.7 |
| Logistic | `logcdf` | 2.15 | 0.29 | 1.69 | 0.79 | 5.9 |
| Logistic | `ccdf` | 1.04 | 0.20 | 0.85 | 0.81 | 4.2 |
| Logistic | `logccdf` | 2.09 | 0.29 | 1.65 | 0.79 | 5.6 |
| Logistic | `icdf` | 1.38 | 0.22 | 0.98 | 0.71 | 4.4 |
| Logistic | `iccdf` | 1.38 | 0.22 | 1.05 | 0.76 | 4.7 |
| Gumbel | `pdf` | 1.52 | 0.24 | 0.91 | 0.60 | 3.7 |
| Gumbel | `logpdf` | 1.28 | 0.22 | 0.90 | 0.70 | 4.1 |
| Gumbel | `cdf` | 1.17 | 0.21 | 0.90 | 0.78 | 4.4 |
| Gumbel | `logcdf` | 0.98 | 0.20 | 0.80 | 0.82 | 4.0 |
| Gumbel | `ccdf` | 1.71 | 0.25 | 1.25 | 0.73 | 5.0 |
| Gumbel | `logccdf` | 2.27 | 0.30 | 2.47 | 1.09 | 8.1 |
| Gumbel | `icdf` | 1.96 | 0.28 | 0.95 | 0.48 | 3.4 |
| Gumbel | `iccdf` | 1.95 | 0.28 | 1.10 | 0.56 | 4.0 |
| GEV | `pdf` | 5.17 | 0.63 | 2.56 | 0.49 | 4.0 |
| GEV | `logpdf` | 4.77 | 0.60 | 2.53 | 0.53 | 4.2 |
| GEV | `cdf` | 4.45 | 0.55 | 1.57 | 0.35 | 2.8 |
| GEV | `logcdf` | 4.00 | 0.51 | 1.43 | 0.36 | 2.8 |
| GEV | `ccdf` | 5.38 | 0.66 | 1.89 | 0.35 | 2.9 |
| GEV | `logccdf` | 5.99 | 0.77 | 4.07 | 0.68 | 5.3 |
| GEV | `icdf` | 5.52 | 0.69 | 1.47 | 0.27 | 2.1 |
| GEV | `iccdf` | 5.49 | 0.69 | 1.65 | 0.30 | 2.4 |
| Pareto | `pdf` | 3.70 | 0.49 | 0.81 | 0.22 | 1.6 |
| Pareto | `logpdf` | 2.31 | 0.31 | 1.36 | 0.59 | 4.4 |
| Pareto | `cdf` | 3.48 | 0.45 | 0.86 | 0.25 | 1.9 |
| Pareto | `logcdf` | 3.87 | 0.49 | 3.40 | 0.88 | 6.9 |
| Pareto | `ccdf` | 1.95 | 0.28 | 0.85 | 0.44 | 3.1 |
| Pareto | `logccdf` | 1.69 | 0.25 | 3.20 | 1.89 | 13.0 |
| Pareto | `icdf` | 1.67 | 0.24 | 0.90 | 0.54 | 3.7 |
| Pareto | `iccdf` | 1.67 | 0.24 | 0.87 | 0.52 | 3.5 |
| Rayleigh | `pdf` | 1.93 | 0.27 | 0.95 | 0.49 | 3.5 |
| Rayleigh | `logpdf` | 1.55 | 0.23 | 0.94 | 0.60 | 4.0 |
| Rayleigh | `cdf` | 1.57 | 0.24 | 0.84 | 0.53 | 3.5 |
| Rayleigh | `logcdf` | 2.28 | 0.31 | 2.29 | 1.00 | 7.4 |
| Rayleigh | `ccdf` | 0.38 | 0.20 | 0.54 | 1.43 | 2.7 |
| Rayleigh | `logccdf` | 0.20 | 0.19 | 0.35 | 1.72 | 1.8 |
| Rayleigh | `icdf` | 1.34 | 0.21 | 0.85 | 0.64 | 4.0 |
| Rayleigh | `iccdf` | 1.35 | 0.21 | 0.74 | 0.55 | 3.5 |
| InverseGamma | `pdf` | 5.24 | 0.65 | 1.40 | 0.27 | 2.2 |
| InverseGamma | `logpdf` | 3.33 | 0.44 | 1.30 | 0.39 | 3.0 |
| InverseGamma | `cdf` | 8.64 | 1.04 | 5.88 | 0.68 | 5.7 |
| InverseGamma | `logcdf` | 9.03 | 1.09 | 24.82 | 2.75 | 22.8 |
| InverseGamma | `ccdf` | 8.58 | 1.04 | 6.11 | 0.71 | 5.9 |
| InverseGamma | `logccdf` | 9.07 | 1.10 | 25.15 | 2.77 | 22.9 |
| InverseGamma | `icdf` | 25.64 | 3.02 | 23.99 | 0.94 | 8.0 |
| InverseGamma | `iccdf` | 26.23 | 3.02 | 23.71 | 0.90 | 7.8 |
| NoncentralChiSquared | `pdf` | 65.70 | 7.71 | 32.48 | 0.49 | 4.2 |
| NoncentralChiSquared | `logpdf` | 65.27 | 7.68 | 24.80 | 0.38 | 3.2 |
| NoncentralChiSquared | `cdf` | 29.87 | 3.43 | 15.01 | 0.50 | 4.4 |
| NoncentralChiSquared | `logcdf` | 55.77 | 6.49 | 111.56 | 2.00 | 17.2 |
| NoncentralChiSquared | `ccdf` | 43.11 | 5.31 | 14.89 | 0.35 | 2.8 |
| NoncentralChiSquared | `logccdf` | 59.71 | 7.18 | 109.06 | 1.83 | 15.2 |
| NoncentralChiSquared | `icdf` | 285.73 | 32.23 | 120.97 | 0.42 | 3.8 |
| NoncentralChiSquared | `iccdf` | 290.92 | 32.75 | 122.06 | 0.42 | 3.7 |
| NoncentralT | `pdf` | 154.27 | 16.43 | 13.13 | 0.09 | 0.8 |
| NoncentralT | `logpdf` | 152.69 | 17.53 | 13.73 | 0.09 | 0.8 |
| NoncentralT | `cdf` | 100.18 | 11.52 | 36.40 | 0.36 | 3.2 |
| NoncentralT | `logcdf` | 164.82 | 18.48 | 359.32 | 2.18 | 19.4 |
| NoncentralT | `ccdf` | 115.91 | 13.72 | 36.23 | 0.31 | 2.6 |
| NoncentralT | `logccdf` | 169.45 | 19.50 | 359.20 | 2.12 | 18.4 |
| NoncentralT | `icdf` | 666.52 | 71.49 | 318.70 | 0.48 | 4.5 |
| NoncentralT | `iccdf` | 662.47 | 71.31 | 310.87 | 0.47 | 4.4 |
| NoncentralF | `pdf` | 71.65 | 8.24 | 9.80 | 0.14 | 1.2 |
| NoncentralF | `logpdf` | 70.37 | 8.17 | 10.67 | 0.15 | 1.3 |
| NoncentralF | `cdf` | 49.86 | 5.66 | 23.60 | 0.47 | 4.2 |
| NoncentralF | `logcdf` | 79.64 | 8.92 | 171.29 | 2.15 | 19.2 |
| NoncentralF | `ccdf` | 53.26 | 6.14 | 23.09 | 0.43 | 3.8 |
| NoncentralF | `logccdf` | 79.87 | 8.96 | 183.81 | 2.30 | 20.5 |
| NoncentralF | `icdf` | 424.55 | 46.60 | 201.36 | 0.47 | 4.3 |
| NoncentralF | `iccdf` | 424.46 | 46.97 | 206.76 | 0.49 | 4.4 |

One Python float (cdf), in microseconds:

| distribution | sd | scipy | scipy / sd |
|---|---:|---:|---:|
| Logistic | 0.19 | 11.72 | 60.1 |
| Gumbel | 0.20 | 12.19 | 60.1 |
| GEV | 0.26 | 25.61 | 98.1 |
| Pareto | 0.23 | 12.29 | 52.6 |
| Rayleigh | 0.17 | 12.75 | 77.2 |
| InverseGamma | 0.30 | 12.23 | 41.2 |
| NoncentralChiSquared | 0.48 | 23.37 | 49.1 |
| NoncentralT | 0.98 | 12.80 | 13.0 |
| NoncentralF | 0.65 | 14.01 | 21.4 |
