10,000 matrices; ms.

| distribution | method | sd 1T | sd 10T | scipy | scipy / sd 1T | scipy / sd 10T |
|---|---|---:|---:|---:|---:|---:|
| Wishart p=3 | logpdf | 0.94 | 0.16 | 78.62 | 83.94 | 506.8 |
| Wishart p=3 | sample | 0.61 | 0.08 | 8.37 | 13.65 | 98.7 |
| InverseWishart p=3 | logpdf | 1.46 | 0.25 | 41.30 | 28.34 | 162.1 |
| InverseWishart p=3 | sample | 0.85 | 0.11 | 10.57 | 12.49 | 96.3 |
| Wishart p=10 | logpdf | 9.68 | 1.28 | 84.90 | 8.77 | 66.3 |
| Wishart p=10 | sample | 5.96 | 0.70 | 16.49 | 2.77 | 23.4 |
| InverseWishart p=10 | logpdf | 25.45 | 3.07 | 45.97 | 1.81 | 15.0 |
| InverseWishart p=10 | sample | 8.39 | 1.08 | 18.32 | 2.18 | 16.9 |
| MatrixNormal 3x2 | logpdf | 0.75 | 0.12 | 0.52 | 0.70 | 4.5 |
| MatrixNormal 3x2 | sample | 0.44 | 0.08 | 0.89 | 2.01 | 10.6 |
| MatrixNormal 10x5 | logpdf | 14.60 | 1.77 | 3.54 | 0.24 | 2.0 |
| MatrixNormal 10x5 | sample | 4.32 | 0.48 | 6.00 | 1.39 | 12.5 |
