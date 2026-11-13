| function | points | region | sd max | sd 99.9% | compared with | its max | its 99.9% |
|---|---:|---|---:|---:|---|---:|---:|
| `gamma` | 7019 | guaranteed | 6.24 | 4.41 | `scipy.special.gamma` | 8.02e+15 | 7.05e+15 |
| `rgamma` | 7719 | guaranteed | 5.11 | 4.01 | `scipy.special.rgamma` | inf | nan |
| `gammaln` | 4937 | guaranteed | 5.53 | 2.43 | `scipy.special.gammaln` | inf | nan |
| `gammaln` | 1732 | mixed | 2.09 | 1.83 | `scipy.special.gammaln` | 1.34 | 1.31 |
| `digamma` | 4136 | guaranteed | 2.11 | 1.71 | `scipy.special.digamma` | 2.11 | 1.56 |
| `digamma` | 1695 | mixed | 14.9 | 8.63 | `scipy.special.digamma` | 2.55e+13 | 1.37e+13 |
| `polygamma` | 3918 | guaranteed | 17.3 | 8.03 | `scipy.special.polygamma` | 8.98e+15 | 8.7e+15 |
| `polygamma` | 974 | mixed | 3.13 | 2.68 | `scipy.special.polygamma` | 3.13 | 2.93 |
| `beta` | 4752 | guaranteed | 9.03 | 7.31 | `scipy.special.beta` | 3.1e+08 | 1.8e+08 |
| `beta` | 748 | mixed | 6.78 | 5.01 | `scipy.special.beta` | 839 | 572 |
| `betaln` | 4500 | mixed | 7.63 | 6.02 | `scipy.special.betaln` | inf | nan |
| `binom` | 5480 | guaranteed | 9.26 | 8.38 | `scipy.special.binom` | 2.8e+04 | 1.16e+04 |
| `binom` | 1120 | mixed | 7.98 | 6.91 | `scipy.special.binom` | inf | nan |
| `zeta` | 6112 | guaranteed | 24.2 | 20 | `scipy.special.zeta` | 5.48e+15 | 2.02e+14 |
| `zeta_hurwitz` | 3500 | guaranteed | 5.51 | 2.97 | `scipy.special.zeta` | 44.5 | 16.7 |
| `zeta_hurwitz` | 800 | mixed | 6.23 | 4 | `scipy.special.zeta` | 6.23 | 4 |
