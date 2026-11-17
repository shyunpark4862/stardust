10,000 matrices; ms.

| distribution | method | sd 1T | sd 10T | scipy | scipy / sd 1T | scipy / sd 10T |
|---|---|---:|---:|---:|---:|---:|
| Wishart p=3 | logpdf | 0.95 | 0.18 | 81.74 | 85.84 | 466.9 |
| Wishart p=3 | sample | 0.53 | 0.09 | 8.60 | 16.31 | 94.0 |
| InverseWishart p=3 | logpdf | 1.47 | 0.26 | 42.88 | 29.19 | 166.2 |
| InverseWishart p=3 | sample | 0.77 | 0.15 | 10.53 | 13.63 | 70.3 |
| Wishart p=10 | logpdf | 9.70 | 1.28 | 89.76 | 9.25 | 70.1 |
| Wishart p=10 | sample | 5.38 | 0.63 | 16.86 | 3.13 | 26.8 |
| InverseWishart p=10 | logpdf | 25.97 | 3.20 | 48.32 | 1.86 | 15.1 |
| InverseWishart p=10 | sample | 8.34 | 0.97 | 18.46 | 2.21 | 19.0 |
| MatrixNormal 3x2 | logpdf | 0.71 | 0.10 | 0.57 | 0.80 | 5.8 |
| MatrixNormal 3x2 | sample | 0.37 | 0.06 | 0.90 | 2.44 | 14.4 |
| MatrixNormal 10x5 | logpdf | 3.04 | 0.38 | 3.82 | 1.26 | 10.1 |
| MatrixNormal 10x5 | sample | 3.77 | 0.43 | 6.12 | 1.62 | 14.2 |
