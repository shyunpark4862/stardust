Arrays of 100000 points (x spread over the distribution by its quantiles); times in ms.

| distribution | method | sd 1T | sd 10T | scipy | scipy / sd 1T | scipy / sd 10T |
|---|---|---:|---:|---:|---:|---:|
| Logistic | `pdf` | 0.94 | 0.18 | 1.90 | 2.01 | 10.7 |
| Logistic | `logpdf` | 1.34 | 0.19 | 1.62 | 1.21 | 8.7 |
| Logistic | `cdf` | 0.85 | 0.11 | 1.00 | 1.18 | 9.0 |
| Logistic | `logcdf` | 1.07 | 0.14 | 1.84 | 1.72 | 13.1 |
| Logistic | `ccdf` | 0.82 | 0.11 | 0.92 | 1.12 | 8.1 |
| Logistic | `logccdf` | 0.95 | 0.17 | 1.63 | 1.72 | 9.6 |
| Logistic | `icdf` | 0.26 | 0.17 | 1.15 | 4.50 | 6.6 |
| Logistic | `iccdf` | 0.27 | 0.18 | 1.06 | 3.97 | 5.8 |
| Gumbel | `pdf` | 0.62 | 0.10 | 0.95 | 1.53 | 9.2 |
| Gumbel | `logpdf` | 0.77 | 0.12 | 0.88 | 1.14 | 7.4 |
| Gumbel | `cdf` | 0.62 | 0.11 | 0.93 | 1.48 | 8.8 |
| Gumbel | `logcdf` | 0.45 | 0.08 | 0.79 | 1.77 | 10.4 |
| Gumbel | `ccdf` | 0.61 | 0.10 | 1.25 | 2.04 | 12.3 |
| Gumbel | `logccdf` | 1.71 | 0.25 | 2.51 | 1.47 | 10.1 |
| Gumbel | `icdf` | 0.47 | 0.09 | 1.04 | 2.21 | 12.2 |
| Gumbel | `iccdf` | 0.74 | 0.12 | 1.14 | 1.54 | 9.9 |
| GEV | `pdf` | 1.20 | 0.21 | 2.64 | 2.21 | 12.6 |
| GEV | `logpdf` | 1.03 | 0.17 | 2.64 | 2.57 | 15.7 |
| GEV | `cdf` | 1.16 | 0.18 | 1.50 | 1.29 | 8.5 |
| GEV | `logcdf` | 0.78 | 0.13 | 1.47 | 1.89 | 11.5 |
| GEV | `ccdf` | 1.17 | 0.19 | 2.05 | 1.76 | 11.0 |
| GEV | `logccdf` | 2.37 | 0.37 | 4.32 | 1.82 | 11.5 |
| GEV | `icdf` | 1.52 | 0.23 | 1.52 | 1.00 | 6.6 |
| GEV | `iccdf` | 1.50 | 0.24 | 1.59 | 1.06 | 6.5 |
| Pareto | `pdf` | 0.65 | 0.25 | 0.88 | 1.35 | 3.5 |
| Pareto | `logpdf` | 0.76 | 0.16 | 1.39 | 1.82 | 8.8 |
| Pareto | `cdf` | 0.96 | 0.17 | 0.94 | 0.98 | 5.5 |
| Pareto | `logcdf` | 2.09 | 0.47 | 3.73 | 1.78 | 8.0 |
| Pareto | `ccdf` | 0.64 | 0.22 | 0.91 | 1.41 | 4.1 |
| Pareto | `logccdf` | 0.97 | 0.19 | 3.68 | 3.80 | 18.9 |
| Pareto | `icdf` | 0.51 | 0.10 | 0.96 | 1.88 | 10.0 |
| Pareto | `iccdf` | 0.54 | 0.10 | 0.92 | 1.72 | 9.6 |
| Rayleigh | `pdf` | 0.44 | 0.07 | 0.98 | 2.24 | 15.0 |
| Rayleigh | `logpdf` | 0.54 | 0.08 | 0.96 | 1.77 | 12.1 |
| Rayleigh | `cdf` | 0.43 | 0.07 | 0.91 | 2.14 | 13.5 |
| Rayleigh | `logcdf` | 1.30 | 0.18 | 2.32 | 1.78 | 12.6 |
| Rayleigh | `ccdf` | 0.41 | 0.07 | 0.58 | 1.43 | 8.5 |
| Rayleigh | `logccdf` | 0.27 | 0.05 | 0.36 | 1.34 | 7.6 |
| Rayleigh | `icdf` | 0.40 | 0.06 | 0.92 | 2.29 | 14.6 |
| Rayleigh | `iccdf` | 0.21 | 0.04 | 0.77 | 3.71 | 19.7 |
| InverseGamma | `pdf` | 1.54 | 0.25 | 1.43 | 0.93 | 5.8 |
| InverseGamma | `logpdf` | 1.59 | 0.23 | 1.27 | 0.80 | 5.4 |
| InverseGamma | `cdf` | 4.49 | 0.64 | 6.37 | 1.42 | 9.9 |
| InverseGamma | `logcdf` | 5.01 | 0.67 | 25.66 | 5.12 | 38.4 |
| InverseGamma | `ccdf` | 4.59 | 0.63 | 6.17 | 1.34 | 9.8 |
| InverseGamma | `logccdf` | 5.54 | 0.75 | 25.68 | 4.64 | 34.4 |
| InverseGamma | `icdf` | 22.71 | 2.66 | 24.53 | 1.08 | 9.2 |
| InverseGamma | `iccdf` | 22.61 | 2.66 | 23.72 | 1.05 | 8.9 |
| NoncentralChiSquared | `pdf` | 8.76 | 1.10 | 34.16 | 3.90 | 31.0 |
| NoncentralChiSquared | `logpdf` | 8.59 | 1.16 | 25.49 | 2.97 | 22.0 |
| NoncentralChiSquared | `cdf` | 13.19 | 1.64 | 15.62 | 1.18 | 9.5 |
| NoncentralChiSquared | `logcdf` | 29.81 | 3.26 | 111.84 | 3.75 | 34.3 |
| NoncentralChiSquared | `ccdf` | 21.70 | 2.49 | 14.86 | 0.68 | 6.0 |
| NoncentralChiSquared | `logccdf` | 25.57 | 3.00 | 109.35 | 4.28 | 36.4 |
| NoncentralChiSquared | `icdf` | 101.94 | 11.58 | 122.01 | 1.20 | 10.5 |
| NoncentralChiSquared | `iccdf` | 101.27 | 11.60 | 119.55 | 1.18 | 10.3 |
| NoncentralT | `pdf` | 14.90 | 1.71 | 13.11 | 0.88 | 7.7 |
| NoncentralT | `logpdf` | 13.72 | 1.76 | 13.44 | 0.98 | 7.6 |
| NoncentralT | `cdf` | 54.75 | 6.17 | 36.74 | 0.67 | 5.9 |
| NoncentralT | `logcdf` | 79.68 | 8.81 | 357.77 | 4.49 | 40.6 |
| NoncentralT | `ccdf` | 40.76 | 4.51 | 35.98 | 0.88 | 8.0 |
| NoncentralT | `logccdf` | 66.47 | 7.54 | 365.00 | 5.49 | 48.4 |
| NoncentralT | `icdf` | 188.07 | 20.69 | 315.81 | 1.68 | 15.3 |
| NoncentralT | `iccdf` | 186.73 | 20.95 | 309.65 | 1.66 | 14.8 |
| NoncentralF | `pdf` | 14.48 | 1.71 | 9.79 | 0.68 | 5.7 |
| NoncentralF | `logpdf` | 20.03 | 2.41 | 10.43 | 0.52 | 4.3 |
| NoncentralF | `cdf` | 32.55 | 3.70 | 23.36 | 0.72 | 6.3 |
| NoncentralF | `logcdf` | 48.03 | 5.55 | 168.84 | 3.52 | 30.4 |
| NoncentralF | `ccdf` | 29.48 | 3.40 | 23.63 | 0.80 | 7.0 |
| NoncentralF | `logccdf` | 42.34 | 4.84 | 168.66 | 3.98 | 34.8 |
| NoncentralF | `icdf` | 185.81 | 20.96 | 199.44 | 1.07 | 9.5 |
| NoncentralF | `iccdf` | 184.26 | 21.02 | 200.96 | 1.09 | 9.6 |

One Python float (cdf), in microseconds:

| distribution | sd | scipy | scipy / sd |
|---|---:|---:|---:|
| Logistic | 0.23 | 11.84 | 51.7 |
| Gumbel | 0.25 | 13.24 | 53.0 |
| GEV | 0.38 | 27.49 | 71.8 |
| Pareto | 0.26 | 13.43 | 52.5 |
| Rayleigh | 0.23 | 13.65 | 60.5 |
| InverseGamma | 0.24 | 13.25 | 54.5 |
| NoncentralChiSquared | 0.27 | 23.46 | 85.8 |
| NoncentralT | 0.35 | 12.97 | 36.9 |
| NoncentralF | 0.39 | 13.96 | 35.8 |
