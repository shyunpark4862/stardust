| function | points | region | sd max | sd 99.9% | compared with | its max | its 99.9% |
|---|---:|---|---:|---:|---|---:|---:|
| `stdtr` | 1501 | guaranteed | 8.95 | 7.7 | `scipy.special.stdtr` | 8.78e+15 | 289 |
| `stdtrit` | 1198 | guaranteed | 7.35 | 6.76 | `scipy.special.stdtrit` | inf | nan |
| `chndtr` | 1200 | guaranteed | 6.84 | 6.51 | `scipy.special.chndtr(x, df, nc)` | 8.8e+15 | 8.65e+15 |
| `chndtrc` | 1187 | guaranteed | 9.52 | 7.51 | `scipy.stats.ncx2.sf` | 8.73e+15 | 6.83e+15 |
| `chndtrix` | 953 | guaranteed | 4.61 | 3.77 | `scipy.special.chndtrix(p, df, nc)` | inf | nan |
| `nctdtr` | 1199 | guaranteed | 7.77 | 5.39 | `scipy.special.nctdtr` | inf | nan |
| `nctdtrc` | 1198 | guaranteed | 7.51 | 6.5 | `scipy.stats.nct.sf` | 4.11e+306 | 5.14e+236 |
| `nctdtrit` | 926 | guaranteed | 9.8 | 7.23 | `scipy.special.nctdtrit` | 8.8e+15 | 6.73e+15 |
| `ncfdtr` | 1200 | guaranteed | 9.03 | 7.75 | `scipy.special.ncfdtr` | 7.15e+15 | 5.53e+05 |
| `ncfdtrc` | 1200 | guaranteed | 15.2 | 9.89 | `scipy.stats.ncf.sf` | 7.05e+03 | 4.11e+03 |
| `ncfdtri` | 955 | guaranteed | 9.66 | 8.51 | `scipy.special.ncfdtri` | inf | nan |
