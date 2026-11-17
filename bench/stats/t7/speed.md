Arrays of 1,000,000 samples; ms. numpy: Generator(PCG64), the same parameters.

| distribution | parameters | sd 1T | sd 10T | numpy | numpy / sd 1T | numpy / sd 10T |
|---|---|---:|---:|---:|---:|---:|
| Normal |  | 2.81 | 0.27 | 3.80 | 1.35 | 14.0 |
| LogNormal |  | 9.60 | 1.06 | 10.55 | 1.10 | 10.0 |
| Exponential |  | 2.14 | 0.25 | 3.49 | 1.63 | 14.1 |
| Uniform | a=0, b=1 | 0.91 | 0.12 | 2.68 | 2.95 | 22.6 |
| Laplace |  | 5.44 | 0.62 | 11.16 | 2.05 | 17.9 |
| Cauchy |  | 7.78 | 0.89 | 7.00 | 0.90 | 7.8 |
| Logistic |  | 4.80 | 0.56 | 5.96 | 1.24 | 10.6 |
| Gumbel |  | 5.79 | 0.66 | 13.83 | 2.39 | 20.9 |
| GEV | c=0.1 | 8.91 | 0.99 | — | — | — |
| Weibull | c=2 | 7.41 | 0.81 | 7.14 | 0.96 | 8.8 |
| Pareto | b=3 | 3.79 | 0.45 | 5.39 | 1.42 | 12.0 |
| Rayleigh |  | 2.09 | 0.26 | 3.73 | 1.78 | 14.4 |
| Gamma | a=3 | 9.23 | 1.04 | 8.62 | 0.93 | 8.3 |
| Gamma | a=0.5 | 17.59 | 1.99 | 22.79 | 1.30 | 11.4 |
| ChiSquared | df=5 | 9.24 | 1.03 | 8.80 | 0.95 | 8.6 |
| InverseGamma | a=3 | 9.38 | 1.04 | — | — | — |
| Beta | a=2, b=3 | 19.60 | 2.15 | 17.61 | 0.90 | 8.2 |
| Beta | a=0.5, b=0.5 | 23.92 | 2.63 | 25.65 | 1.07 | 9.7 |
| StudentT | df=5 | 12.01 | 1.36 | 12.12 | 1.01 | 8.9 |
| F | dfn=5, dfd=7 | 19.02 | 2.14 | 17.19 | 0.90 | 8.0 |
| NoncentralChiSquared | df=5, nc=2 | 14.08 | 1.54 | 12.18 | 0.86 | 7.9 |
| NoncentralT | df=5, nc=1 | 12.12 | 1.37 | — | — | — |
| NoncentralF | dfn=5, dfd=7, nc=2 | 19.97 | 2.18 | 21.17 | 1.06 | 9.7 |
| Bernoulli | p=0.3 | 0.86 | 0.11 | 7.18 | 8.32 | 62.8 |
| DiscreteUniform | low=0, high=10 | 1.89 | 0.22 | 1.61 | 0.85 | 7.5 |
| Geometric | p=0.1 | 4.57 | 0.54 | 5.25 | 1.15 | 9.7 |
| Poisson | mu=3 | 15.27 | 1.71 | 27.67 | 1.81 | 16.2 |
| Poisson | mu=100 | 21.50 | 2.40 | 20.56 | 0.96 | 8.6 |
| Binomial | n=20, p=0.3 | 19.38 | 2.24 | 31.50 | 1.63 | 14.1 |
| Binomial | n=10000, p=0.3 | 19.61 | 2.16 | 17.47 | 0.89 | 8.1 |
| NegativeBinomial | n=5, p=0.3 | 49.58 | 5.47 | 54.51 | 1.10 | 10.0 |
| Hypergeometric | total=1000, successes=300, draws=100 | 16.46 | 1.91 | 54.49 | 3.31 | 28.6 |
| BetaBinomial | n=20, a=2, b=3 | 96.43 | 10.75 | — | — | — |
| LogSeries | p=0.9 | 23.20 | 2.54 | 24.62 | 1.06 | 9.7 |
| Zipf | a=2 | 47.12 | 5.33 | 56.08 | 1.19 | 10.5 |

One Python float per call (µs):

| distribution | parameters | sd | numpy | numpy / sd |
|---|---|---:|---:|---:|
| Normal |  | 0.26 | 0.15 | 0.58 |
| LogNormal |  | 0.27 | 0.25 | 0.93 |
| Exponential |  | 0.25 | 0.15 | 0.59 |
| Uniform | a=0, b=1 | 0.25 | 0.15 | 0.61 |
| Laplace |  | 0.26 | 0.26 | 0.98 |
| Cauchy |  | 0.26 | 0.11 | 0.41 |
| Logistic |  | 0.26 | 0.25 | 0.97 |
| Gumbel |  | 0.25 | 0.25 | 0.98 |
| GEV | c=0.1 | 0.26 | — | — |
| Weibull | c=2 | 0.26 | 0.18 | 0.71 |
| Pareto | b=3 | 0.26 | 0.19 | 0.72 |
| Rayleigh |  | 0.25 | 0.18 | 0.71 |
| Gamma | a=3 | 0.27 | 0.24 | 0.92 |
| Gamma | a=0.5 | 0.27 | 0.26 | 0.96 |
| ChiSquared | df=5 | 0.26 | 0.18 | 0.69 |
| InverseGamma | a=3 | 0.27 | — | — |
| Beta | a=2, b=3 | 0.28 | 0.24 | 0.88 |
| Beta | a=0.5, b=0.5 | 0.28 | 0.27 | 0.94 |
| StudentT | df=5 | 0.27 | 0.18 | 0.69 |
| F | dfn=5, dfd=7 | 0.28 | 0.25 | 0.89 |
| NoncentralChiSquared | df=5, nc=2 | 0.27 | 0.25 | 0.91 |
| NoncentralT | df=5, nc=1 | 0.27 | — | — |
| NoncentralF | dfn=5, dfd=7, nc=2 | 0.29 | 0.32 | 1.10 |
| Bernoulli | p=0.3 | 0.24 | 0.22 | 0.93 |
| DiscreteUniform | low=0, high=10 | 0.25 | 0.52 | 2.08 |
| Geometric | p=0.1 | 0.25 | 0.15 | 0.60 |
| Poisson | mu=3 | 0.26 | 0.18 | 0.67 |
| Poisson | mu=100 | 0.28 | 0.17 | 0.61 |
| Binomial | n=20, p=0.3 | 0.28 | 0.25 | 0.92 |
| Binomial | n=10000, p=0.3 | 0.28 | 0.25 | 0.89 |
| NegativeBinomial | n=5, p=0.3 | 0.32 | 0.34 | 1.06 |
| Hypergeometric | total=1000, successes=300, draws=100 | 0.51 | 0.51 | 0.99 |
| BetaBinomial | n=20, a=2, b=3 | 0.37 | — | — |
| LogSeries | p=0.9 | 0.28 | 0.18 | 0.64 |
| Zipf | a=2 | 0.31 | 0.21 | 0.70 |
