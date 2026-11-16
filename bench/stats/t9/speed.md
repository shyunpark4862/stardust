100,000 points (vectors); ms.

| distribution | method | sd 1T | sd 10T | scipy | scipy / sd 1T | scipy / sd 10T |
|---|---|---:|---:|---:|---:|---:|
| MultivariateNormal d=3 | logpdf | 1.95 | 0.46 | 1.19 | 0.61 | 2.6 |
| MultivariateNormal d=3 | sample | 2.24 | 0.51 | 3.25 | 1.45 | 6.4 |
| MultivariateT d=3 | logpdf | 3.96 | 0.50 | 1.39 | 0.35 | 2.8 |
| MultivariateT d=3 | sample | 4.60 | 0.57 | 5.44 | 1.18 | 9.5 |
| MultivariateNormal d=10 | logpdf | 9.69 | 1.47 | 1.36 | 0.14 | 0.9 |
| MultivariateNormal d=10 | sample | 6.96 | 0.89 | 9.85 | 1.41 | 11.1 |
| MultivariateT d=10 | logpdf | 12.22 | 1.72 | 1.69 | 0.14 | 1.0 |
| MultivariateT d=10 | sample | 10.01 | 1.25 | 12.42 | 1.24 | 9.9 |
| MultivariateNormal d=50 | logpdf | 127.66 | 16.27 | 5.91 | 0.05 | 0.4 |
| MultivariateNormal d=50 | sample | 66.38 | 8.85 | 48.68 | 0.73 | 5.5 |
| MultivariateT d=50 | logpdf | 131.53 | 16.67 | 6.34 | 0.05 | 0.4 |
| MultivariateT d=50 | sample | 68.50 | 9.16 | 54.17 | 0.79 | 5.9 |
| Multinomial K=4 n=20 | logpmf | 2.42 | 0.30 | 7.84 | 3.24 | 25.9 |
| Multinomial K=4 n=20 | sample | 15.79 | 2.21 | 13.68 | 0.87 | 6.2 |
| Dirichlet K=4 | logpdf | 4.61 | 0.56 | 2.59 | 0.56 | 4.6 |
| Dirichlet K=4 | sample | 10.05 | 1.25 | 7.62 | 0.76 | 6.1 |
