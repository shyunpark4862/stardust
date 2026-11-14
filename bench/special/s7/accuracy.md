| function | points | region | sd max | sd 99.9% | compared with | its max | its 99.9% |
|---|---:|---|---:|---:|---|---:|---:|
| `hyp1f1` | 1600 | guaranteed | 6.91 | 6.17 | `scipy.special.hyp1f1` | 248 | 194 |
| `hyp1f1` | 400 | loose | 2.28e+13 | 1.38e+13 | `scipy.special.hyp1f1` | 2.33e+03 | 2.22e+03 |
| `hyp2f1` | 509 | guaranteed | 11 | 10.8 | `scipy.special.hyp2f1` | 208 | 129 |
| `hyp2f1` | 991 | loose | 3.5e+03 | 877 | `scipy.special.hyp2f1` | 1.83e+12 | 3.49e+11 |
| `lambertw` | 3000 | guaranteed | 0.905 | 0.873 | `scipy.special.lambertw.real` | 4.21e+07 | 4.21e+07 |
| `lambertw_m1` | 2250 | guaranteed | 1.38 | 1.18 | `scipy.special.lambertw(k=-1).real` | 4.48e+11 | 4.27e+11 |
| `airy_ai` | 2975 | guaranteed | 57.3 | 27.4 | `scipy.special.airy[0]` | inf | nan |
| `airy_aip` | 2975 | guaranteed | 34.8 | 29.6 | `scipy.special.airy[1]` | inf | nan |
| `airy_bi` | 2975 | guaranteed | 43.5 | 25.4 | `scipy.special.airy[2]` | inf | nan |
| `airy_bip` | 2975 | guaranteed | 50.3 | 22.5 | `scipy.special.airy[3]` | inf | nan |
