1,000,000 samples per case (seed: crc32 of the case).

| distribution | parameters | test | p-value |
|---|---|---|---:|
| Normal | mu=1, sigma=2 | KS | 0.986 |
| LogNormal | mu=0.5, sigma=1.5 | KS | 0.305 |
| Exponential | scale=2 | KS | 0.0342 |
| Uniform | a=-1, b=3 | KS | 0.565 |
| Laplace | mu=0.5, sigma=2 | KS | 0.247 |
| Cauchy | mu=0, sigma=1 | KS | 0.137 |
| Logistic | mu=0, sigma=1 | KS | 0.0675 |
| Gumbel | mu=1, sigma=2 | KS | 0.636 |
| GEV | c=0.3 | KS | 0.397 |
| GEV | c=-0.4 | KS | 0.318 |
| GEV | c=0 | KS | 0.0643 |
| Weibull | c=0.7 | KS | 0.748 |
| Pareto | b=2.5 | KS | 0.482 |
| Rayleigh | scale=1.5 | KS | 0.0722 |
| Gamma | a=3, scale=2 | KS | 0.642 |
| Gamma | a=0.3 | KS | 0.872 |
| Gamma | a=1e+06 | KS | 0.565 |
| ChiSquared | df=0.5 | KS | 0.164 |
| ChiSquared | df=7 | KS | 0.329 |
| InverseGamma | a=0.5 | KS | 0.768 |
| InverseGamma | a=4, scale=3 | KS | 0.945 |
| Beta | a=0.2, b=0.3 | KS | 0.792 |
| Beta | a=2, b=5 | KS | 0.54 |
| Beta | a=0.5, b=3 | KS | 0.605 |
| StudentT | df=0.5 | KS | 0.163 |
| StudentT | df=5 | KS | 0.0445 |
| StudentT | df=10000 | KS | 0.104 |
| F | dfn=0.7, dfd=3 | KS | 0.342 |
| F | dfn=10, dfd=20 | KS | 0.997 |
| NoncentralChiSquared | df=0.5, nc=3 | KS | 0.135 |
| NoncentralChiSquared | df=4, nc=50 | KS | 0.0796 |
| NoncentralChiSquared | df=4, nc=0 | KS | 0.156 |
| NoncentralT | df=3, nc=-2 | KS | 0.0347 |
| NoncentralT | df=30, nc=5 | KS | 0.709 |
| NoncentralF | dfn=3, dfd=8, nc=4 | KS | 0.38 |
| Bernoulli | p=0.3 | chi2 (1 dof) | 0.817 |
| DiscreteUniform | low=-3, high=7 | chi2 (9 dof) | 0.347 |
| Geometric | p=0.2 | chi2 (28 dof) | 0.162 |
| Geometric | p=1e-06 | chi2 (999 dof) | 0.139 |
| Poisson | mu=0.7 | chi2 (5 dof) | 0.254 |
| Poisson | mu=9.9 | chi2 (21 dof) | 0.0655 |
| Poisson | mu=10 | chi2 (21 dof) | 0.709 |
| Poisson | mu=1234.5 | chi2 (181 dof) | 0.955 |
| Poisson | mu=1e+12 | chi2 (999 dof) | 0.707 |
| Binomial | n=20, p=0.3 | chi2 (14 dof) | 0.567 |
| Binomial | n=20, p=0.9 | chi2 (8 dof) | 0.895 |
| Binomial | n=10000, p=0.3 | chi2 (227 dof) | 0.415 |
| Binomial | n=1e+09, p=0.7 | chi2 (1000 dof) | 0.988 |
| Binomial | n=1e+06, p=1e-05 | chi2 (21 dof) | 0.639 |
| NegativeBinomial | n=5, p=0.3 | chi2 (38 dof) | 0.719 |
| NegativeBinomial | n=0.3, p=0.01 | chi2 (182 dof) | 0.949 |
| Hypergeometric | total=1000, successes=300, draws=100 | chi2 (29 dof) | 0.121 |
| Hypergeometric | total=50, successes=45, draws=40 | chi2 (5 dof) | 0.867 |
| Hypergeometric | total=1e+07, successes=4e+06, draws=500000 | chi2 (952 dof) | 0.805 |
| BetaBinomial | n=20, a=2, b=3 | chi2 (20 dof) | 0.811 |
| BetaBinomial | n=100, a=0.3, b=0.2 | chi2 (100 dof) | 0.747 |
| LogSeries | p=0.9 | chi2 (33 dof) | 0.584 |
| LogSeries | p=0.999 | chi2 (359 dof) | 0.762 |
| LogSeries | p=0.001 | chi2 (1 dof) | 0.657 |
| Zipf | a=2 | chi2 (48 dof) | 0.773 |
| Zipf | a=1.3 | chi2 (306 dof) | 0.858 |
| Zipf | a=6 | chi2 (3 dof) | 0.351 |
