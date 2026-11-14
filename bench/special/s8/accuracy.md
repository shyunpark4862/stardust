| function | points | region | sd max | sd 99.9% | compared with | its max | its 99.9% |
|---|---:|---|---:|---:|---|---:|---:|
| `wofz` | 7401 | guaranteed | 2.01 | 1.84 | `scipy.special.wofz` | 255 | 108 |
| `erf_complex` | 5498 | guaranteed | 3.44 | 2.39 | `scipy.special.erf` | inf | nan |
| `erfc_complex` | 5402 | guaranteed | 3.44 | 2.59 | `scipy.special.erfc` | inf | nan |
| `erfcx_complex` | 5402 | guaranteed | 1.94 | 1.71 | `scipy.special.erfcx` | inf | nan |
| `loggamma` | 7363 | guaranteed | 8.21 | 5.86 | `scipy.special.loggamma` | 224 | 111 |
| `lambertw_c0` | 5138 | guaranteed | 1.86 | 1.36 | `scipy.special.lambertw(k=0)` | inf | 6.52e+07 |
| `lambertw_cm1` | 5138 | guaranteed | 1.75 | 0.999 | `scipy.special.lambertw(k=-1)` | inf | 7e+07 |
| `lambertw_c1` | 5138 | guaranteed | 1.76 | 0.954 | `scipy.special.lambertw(k=1)` | 1.04e+08 | 4.26e+07 |
| `lambertw_c4` | 5138 | guaranteed | 0.947 | 0.703 | `scipy.special.lambertw(k=4)` | 0.696 | 0.562 |
