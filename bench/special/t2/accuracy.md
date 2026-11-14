# T2 정확도: 비중심 분포 함수 (2026-11-15)

`.venv/bin/python bench/special/accuracy.py chndtr chndtrc chndtrix nctdtr nctdtrc nctdtrit ncfdtr ncfdtrc ncfdtri chdtr chdtrc fdtr fdtrc betainc betaincc`

- mpmath(50자리) 참값 대비 ULP다. 역함수는 ULP/max(1, κ)다(§4의 계약). 비교 대상의 인자 순서는 scipy에 맞춰 넘겼다. 위꼬리 세 개는 scipy.special에 없어서 scipy.stats의 sf와 비교했다.
- 아래 다섯 줄(chdtr부터 betaincc까지)은 T2에서 고친 T1·S4 경로의 회귀 확인이다. T1 측정과 같은 값이다.
- 참값에서 빠진 점:
  - 역함수에서 근이 double 범위 밖인 점: chndtrix 247점, ncfdtri 245점(0과 inf), nctdtrit 124점(±inf). 각 함수로 tests/special/test_noncentral.py에서 확인한다.
  - nctdtrit에서 mpmath가 30초 안에 끝내지 못한 점 150점.
  - chndtrc에서 mpmath가 30초 안에 끝내지 못한 점 13점.

| function | points | region | sd max | sd 99.9% | compared with | its max | its 99.9% |
|---|---:|---|---:|---:|---|---:|---:|
| `chndtr` | 1200 | guaranteed | 6.84 | 6.52 | `scipy.special.chndtr(x, df, nc)` | 8.8e+15 | 8.65e+15 |
| `chndtrc` | 1187 | guaranteed | 9.52 | 7.51 | `scipy.stats.ncx2.sf` | 8.73e+15 | 6.83e+15 |
| `chndtrix` | 953 | guaranteed | 4.49 | 3.72 | `scipy.special.chndtrix(p, df, nc)` | inf | nan |
| `nctdtr` | 1199 | guaranteed | 7.77 | 5.39 | `scipy.special.nctdtr` | inf | nan |
| `nctdtrc` | 1198 | guaranteed | 7.51 | 6.5 | `scipy.stats.nct.sf` | 4.11e+306 | 5.14e+236 |
| `nctdtrit` | 926 | guaranteed | 8.8 | 5.68 | `scipy.special.nctdtrit` | 8.8e+15 | 6.73e+15 |
| `ncfdtr` | 1200 | guaranteed | 12.1 | 10.9 | `scipy.special.ncfdtr` | 7.15e+15 | 5.53e+05 |
| `ncfdtrc` | 1200 | guaranteed | 22.6 | 20.7 | `scipy.stats.ncf.sf` | 7.05e+03 | 4.11e+03 |
| `ncfdtri` | 955 | guaranteed | 9.88 | 8.52 | `scipy.special.ncfdtri` | inf | nan |
| `chdtr` | 1200 | guaranteed | 6.05 | 5.44 | `scipy.special.chdtr` | 8.28e+03 | 4.19e+03 |
| `chdtrc` | 1200 | guaranteed | 9.69 | 7.88 | `scipy.special.chdtrc` | 1.57e+06 | 1.28e+04 |
| `fdtr` | 1500 | guaranteed | 16.4 | 11.8 | `scipy.special.fdtr` | 778 | 686 |
| `fdtrc` | 1500 | guaranteed | 18.1 | 11.6 | `scipy.special.fdtrc` | 948 | 847 |
| `betainc` | 1979 | guaranteed | 16.7 | 13.7 | `scipy.special.betainc` | inf | 1.06e+04 |
| `betaincc` | 1973 | guaranteed | 18.6 | 17.5 | `scipy.special.betaincc` | inf | 1.07e+11 |
