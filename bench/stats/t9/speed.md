100,000 points (vectors); ms.

| distribution | method | sd 1T | sd 10T | scipy | scipy / sd 1T | scipy / sd 10T |
|---|---|---:|---:|---:|---:|---:|
| MultivariateNormal d=3 | logpdf | 1.50 | 0.28 | 1.18 | 0.78 | 4.2 |
| MultivariateNormal d=3 | sample | 1.90 | 0.23 | 3.41 | 1.79 | 15.1 |
| MultivariateT d=3 | logpdf | 2.02 | 0.52 | 1.43 | 0.71 | 2.8 |
| MultivariateT d=3 | sample | 4.27 | 0.51 | 5.60 | 1.31 | 11.1 |
| MultivariateNormal d=10 | logpdf | 2.61 | 0.37 | 1.46 | 0.56 | 3.9 |
| MultivariateNormal d=10 | sample | 4.91 | 0.60 | 10.14 | 2.07 | 16.8 |
| MultivariateT d=10 | logpdf | 3.17 | 0.40 | 1.71 | 0.54 | 4.3 |
| MultivariateT d=10 | sample | 6.99 | 0.86 | 12.77 | 1.83 | 14.8 |
| MultivariateNormal d=50 | logpdf | 14.77 | 1.80 | 5.85 | 0.40 | 3.2 |
| MultivariateNormal d=50 | sample | 29.17 | 3.68 | 49.92 | 1.71 | 13.6 |
| MultivariateT d=50 | logpdf | 14.80 | 1.83 | 6.28 | 0.42 | 3.4 |
| MultivariateT d=50 | sample | 31.21 | 3.89 | 55.36 | 1.77 | 14.2 |
| Multinomial K=4 n=20 | logpmf | 2.50 | 0.30 | 8.17 | 3.27 | 27.0 |
| Multinomial K=4 n=20 | sample | 13.83 | 1.70 | 13.97 | 1.01 | 8.2 |
| Dirichlet K=4 | logpdf | 1.78 | 0.22 | 2.71 | 1.52 | 12.6 |
| Dirichlet K=4 | sample | 5.62 | 0.62 | 7.93 | 1.41 | 12.9 |
