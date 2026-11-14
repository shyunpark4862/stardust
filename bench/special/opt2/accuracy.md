| function | points | region | sd max | sd 99.9% | compared with | its max | its 99.9% |
|---|---:|---|---:|---:|---|---:|---:|
| `gammainc` | 2406 | guaranteed | 8.87 | 5.63 | `scipy.special.gammainc` | 4.5e+15 | 6.56e+09 |
| `gammaincc` | 2361 | guaranteed | 13.4 | 7.98 | `scipy.special.gammaincc` | 2.36e+10 | 6.06e+03 |
| `gammaincinv` | 2004 | guaranteed | 50.1 | 42.7 | `scipy.special.gammaincinv` | inf | 498 |
| `gammainccinv` | 1903 | guaranteed | 37.5 | 26 | `scipy.special.gammainccinv` | inf | 31.3 |
| `betainc` | 1979 | guaranteed | 16.7 | 13.7 | `scipy.special.betainc` | inf | 1.06e+04 |
| `betaincc` | 1973 | guaranteed | 18.6 | 17.5 | `scipy.special.betaincc` | inf | 1.07e+11 |
| `betaincinv` | 1440 | guaranteed | 59 | 8.67 | `scipy.special.betaincinv` | inf | nan |
| `betainccinv` | 1254 | guaranteed | 19.8 | 15.9 | `scipy.special.betainccinv` | inf | nan |
| `stdtr` | 1501 | guaranteed | 8.95 | 7.7 | `scipy.special.stdtr` | 8.78e+15 | 289 |
| `stdtrit` | 1198 | guaranteed | 6.89 | 6.21 | `scipy.special.stdtrit` | inf | nan |
| `chdtr` | 1200 | guaranteed | 5.3 | 5.02 | `scipy.special.chdtr` | 8.28e+03 | 4.19e+03 |
| `chdtrc` | 1200 | guaranteed | 9.69 | 7.88 | `scipy.special.chdtrc` | 1.57e+06 | 1.28e+04 |
| `chdtri` | 1200 | guaranteed | 7.24 | 6.5 | `scipy.special.chdtri` | 25.1 | 9.61 |
| `fdtr` | 1500 | guaranteed | 12.4 | 7.82 | `scipy.special.fdtr` | 778 | 686 |
| `fdtrc` | 1500 | guaranteed | 10.7 | 9.25 | `scipy.special.fdtrc` | 948 | 847 |
| `fdtri` | 1007 | guaranteed | 23 | 21.2 | `scipy.special.fdtri` | 4.87e+16 | 114 |
| `pdtr` | 1400 | guaranteed | 4.26 | 3.72 | `scipy.special.pdtr` | 3.13e+04 | 1.29e+04 |
| `pdtrc` | 1400 | guaranteed | 13.1 | 7.24 | `scipy.special.pdtrc` | 4.9e+07 | 2.3e+07 |
| `bdtr` | 1200 | guaranteed | 4.94 | 4.81 | `scipy.special.bdtr` | 9.01e+04 | 5.2e+04 |
| `bdtrc` | 1200 | guaranteed | 7.44 | 6.52 | `scipy.special.bdtrc` | 1.6e+05 | 9.43e+04 |
| `nbdtr` | 1199 | guaranteed | 7.21 | 6.59 | `scipy.special.nbdtr` | inf | nan |
| `nbdtrc` | 1200 | guaranteed | 8.79 | 8.41 | `scipy.special.nbdtrc` | inf | nan |
| `chndtr` | 1200 | guaranteed | 6.84 | 6.52 | `scipy.special.chndtr(x, df, nc)` | 8.8e+15 | 8.65e+15 |
| `chndtrc` | 1187 | guaranteed | 9.52 | 7.51 | `scipy.stats.ncx2.sf` | 8.73e+15 | 6.83e+15 |
| `chndtrix` | 953 | guaranteed | 4.49 | 3.51 | `scipy.special.chndtrix(p, df, nc)` | inf | nan |
| `nctdtr` | 1199 | guaranteed | 7.77 | 5.39 | `scipy.special.nctdtr` | inf | nan |
| `nctdtrc` | 1198 | guaranteed | 7.51 | 6.5 | `scipy.stats.nct.sf` | 4.11e+306 | 5.14e+236 |
| `nctdtrit` | 926 | guaranteed | 8.8 | 5.68 | `scipy.special.nctdtrit` | 8.8e+15 | 6.73e+15 |
| `ncfdtr` | 1200 | guaranteed | 9.03 | 7.75 | `scipy.special.ncfdtr` | 7.15e+15 | 5.53e+05 |
| `ncfdtrc` | 1200 | guaranteed | 15.2 | 9.89 | `scipy.stats.ncf.sf` | 7.05e+03 | 4.11e+03 |
| `ncfdtri` | 955 | guaranteed | 9.88 | 8.52 | `scipy.special.ncfdtri` | inf | nan |
| `expn` | 2705 | guaranteed | 6.28 | 4.96 | `scipy.special.expn` | 3.39e+18 | 1.15e+09 |
| `hyp2f1` | 509 | guaranteed | 11 | 10.8 | `scipy.special.hyp2f1` | 208 | 129 |
| `hyp2f1` | 991 | loose | 3.07e+03 | 605 | `scipy.special.hyp2f1` | 1.83e+12 | 3.49e+11 |
| `hyp1f1` | 1600 | guaranteed | 78.6 | 71.6 | `scipy.special.hyp1f1` | 248 | 194 |
| `hyp1f1` | 400 | loose | 2.2e+03 | 1.86e+03 | `scipy.special.hyp1f1` | 2.33e+03 | 2.22e+03 |
