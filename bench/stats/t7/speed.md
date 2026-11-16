Arrays of 1,000,000 samples; ms. numpy: Generator(PCG64), the same parameters.

| distribution | parameters | sd 1T | sd 10T | numpy | numpy / sd 1T | numpy / sd 10T |
|---|---|---:|---:|---:|---:|---:|
| Normal |  | 6.52 | 0.64 | 3.57 | 0.55 | 5.6 |
| LogNormal |  | 12.43 | 1.41 | 10.33 | 0.83 | 7.3 |
| Exponential |  | 2.92 | 0.35 | 3.43 | 1.18 | 9.9 |
| Uniform | a=0, b=1 | 2.38 | 0.29 | 2.70 | 1.13 | 9.1 |
| Laplace |  | 7.63 | 0.89 | 11.06 | 1.45 | 12.5 |
| Cauchy |  | 11.36 | 1.31 | 6.84 | 0.60 | 5.2 |
| Logistic |  | 7.04 | 0.80 | 5.90 | 0.84 | 7.4 |
| Gumbel |  | 7.31 | 0.84 | 13.91 | 1.90 | 16.6 |
| GEV | c=0.1 | 11.16 | 1.26 | — | — | — |
| Weibull | c=2 | 7.95 | 0.90 | 7.12 | 0.89 | 7.9 |
| Pareto | b=3 | 4.51 | 0.53 | 5.21 | 1.15 | 9.8 |
| Rayleigh |  | 3.13 | 0.36 | 3.69 | 1.18 | 10.2 |
| Gamma | a=3 | 11.13 | 1.27 | 8.52 | 0.77 | 6.7 |
| Gamma | a=0.5 | 31.37 | 3.51 | 22.47 | 0.72 | 6.4 |
| ChiSquared | df=5 | 11.30 | 1.27 | 8.74 | 0.77 | 6.9 |
| InverseGamma | a=3 | 12.43 | 1.41 | — | — | — |
| Beta | a=2, b=3 | 21.12 | 2.32 | 17.69 | 0.84 | 7.6 |
| Beta | a=0.5, b=0.5 | 50.04 | 5.72 | 25.58 | 0.51 | 4.5 |
| StudentT | df=5 | 15.84 | 1.78 | 11.88 | 0.75 | 6.7 |
| F | dfn=5, dfd=7 | 20.44 | 2.32 | 17.30 | 0.85 | 7.4 |
| NoncentralChiSquared | df=5, nc=2 | 21.03 | 2.39 | 12.24 | 0.58 | 5.1 |
| NoncentralT | df=5, nc=1 | 15.97 | 1.81 | — | — | — |
| NoncentralF | dfn=5, dfd=7, nc=2 | 44.33 | 4.96 | 21.12 | 0.48 | 4.3 |
| Bernoulli | p=0.3 | 2.46 | 0.29 | 6.70 | 2.72 | 22.8 |
| DiscreteUniform | low=0, high=10 | 2.41 | 0.30 | 1.59 | 0.66 | 5.4 |
| Geometric | p=0.1 | 7.14 | 0.82 | 5.16 | 0.72 | 6.3 |
| Poisson | mu=3 | 14.96 | 1.73 | 27.20 | 1.82 | 15.7 |
| Poisson | mu=100 | 27.62 | 3.27 | 20.25 | 0.73 | 6.2 |
| Binomial | n=20, p=0.3 | 18.77 | 2.20 | 31.01 | 1.65 | 14.1 |
| Binomial | n=10000, p=0.3 | 40.53 | 4.77 | 17.05 | 0.42 | 3.6 |
| NegativeBinomial | n=5, p=0.3 | 76.98 | 8.95 | 51.69 | 0.67 | 5.8 |
| Hypergeometric | total=1000, successes=300, draws=100 | 23.80 | 2.74 | 53.44 | 2.25 | 19.5 |
| BetaBinomial | n=20, a=2, b=3 | 106.85 | 12.42 | — | — | — |
| LogSeries | p=0.9 | 23.44 | 2.61 | 24.43 | 1.04 | 9.4 |
| Zipf | a=2 | 47.85 | 5.50 | 54.53 | 1.14 | 9.9 |

One Python float per call (µs):

| distribution | parameters | sd | numpy | numpy / sd |
|---|---|---:|---:|---:|
| Normal |  | 0.25 | 0.15 | 0.59 |
| LogNormal |  | 0.26 | 0.23 | 0.90 |
| Exponential |  | 0.24 | 0.15 | 0.61 |
| Uniform | a=0, b=1 | 0.24 | 0.14 | 0.57 |
| Laplace |  | 0.25 | 0.25 | 1.01 |
| Cauchy |  | 0.26 | 0.11 | 0.41 |
| Logistic |  | 0.26 | 0.24 | 0.93 |
| Gumbel |  | 0.25 | 0.24 | 0.99 |
| GEV | c=0.1 | 0.26 | — | — |
| Weibull | c=2 | 0.25 | 0.18 | 0.71 |
| Pareto | b=3 | 0.25 | 0.17 | 0.70 |
| Rayleigh |  | 0.25 | 0.17 | 0.67 |
| Gamma | a=3 | 0.27 | 0.24 | 0.89 |
| Gamma | a=0.5 | 0.29 | 0.26 | 0.88 |
| ChiSquared | df=5 | 0.26 | 0.17 | 0.66 |
| InverseGamma | a=3 | 0.27 | — | — |
| Beta | a=2, b=3 | 0.28 | 0.24 | 0.87 |
| Beta | a=0.5, b=0.5 | 0.32 | 0.26 | 0.81 |
| StudentT | df=5 | 0.26 | 0.17 | 0.66 |
| F | dfn=5, dfd=7 | 0.27 | 0.24 | 0.90 |
| NoncentralChiSquared | df=5, nc=2 | 0.28 | 0.24 | 0.87 |
| NoncentralT | df=5, nc=1 | 0.26 | — | — |
| NoncentralF | dfn=5, dfd=7, nc=2 | 0.31 | 0.32 | 1.01 |
| Bernoulli | p=0.3 | 0.25 | 0.23 | 0.92 |
| DiscreteUniform | low=0, high=10 | 0.25 | 0.50 | 2.04 |
| Geometric | p=0.1 | 0.26 | 0.15 | 0.60 |
| Poisson | mu=3 | 0.26 | 0.17 | 0.67 |
| Poisson | mu=100 | 0.29 | 0.17 | 0.57 |
| Binomial | n=20, p=0.3 | 0.30 | 0.25 | 0.83 |
| Binomial | n=10000, p=0.3 | 0.36 | 0.24 | 0.65 |
| NegativeBinomial | n=5, p=0.3 | 0.33 | 0.33 | 0.99 |
| Hypergeometric | total=1000, successes=300, draws=100 | 0.50 | 0.50 | 1.01 |
| BetaBinomial | n=20, a=2, b=3 | 0.38 | — | — |
| LogSeries | p=0.9 | 0.27 | 0.16 | 0.60 |
| Zipf | a=2 | 0.30 | 0.21 | 0.71 |
