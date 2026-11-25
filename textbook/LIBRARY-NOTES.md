# 교과서를 쓰며 찾은 라이브러리 이상

교과서 세션은 라이브러리를 고치지 않는다. 찾은 것을 여기에 적어 두고, 사용자가 라이브러리 세션에 전할지 정한다.

## 1. 역변환 표본의 균등값이 1이 될 수 있다 (2026-11-27, 53장 검토)

- 위치: `csrc/stats/sample.hpp:49` `u() { return (static_cast<double>(next() >> 11) + 0.5) * 0x1p-53; }`
- `m = next() >> 11`은 53비트이다. `m ≥ 2^52`이면 `m + 0.5`를 binary64로 나타낼 수 없어 짝수 쪽으로 반올림된다. 홀수 `m`은 `m + 1`로, 짝수 `m`은 `m`으로 간다.
- `m = 2^53 − 1`이면 `m + 0.5`가 `2^53`이 되어 `u = 1.0`이다(확률 2^-53). 그러면 82행의 `D::iccdf(1.0 - u, ...)`가 `iccdf(0)`, 곧 지지집합의 위 끝을 부른다. 역변환으로 뽑는 분포(코시, 라플라스, 로지스틱 등)에서는 `+inf`가 나온다.
- 위쪽 절반에서는 주석(11, 48행)이 말하는 "(0, 1), 1/2에 대칭"도 정확하지 않다.
- 고칠 방향(제안): 위 52비트로 `(m + 0.5) * 2^-52`를 쓰면(2m + 1 < 2^53) 값이 정확하고 0과 1이 나오지 않으며 정확히 대칭이다.

## linalg: LU 작업 버퍼의 leading dimension이 2의 거듭제곱일 때 (2026-11-25, 48장 시범 장을 쓰며 측정)

- 위치: `csrc/linalg/ops/common.hpp`의 `lu_factor`는 행렬을 ld = n으로 패킹한다(`pack(..., a, n, true)`), `lu`도 ld = m.
- 관찰(Apple M4 Pro, Accelerate `dgetrf`, 한 번 호출의 최솟값): n = 1024에서 lda = 1024이면 초당 149×10⁹ 플롭, lda = 1032이면 235×10⁹ 플롭(1.58배). n = 512는 74 대 94(1.27배), n = 256은 33 대 35. 측정 코드는 `textbook/figures/48/lda2.c`.
- 원인: 열 길이가 2의 큰 거듭제곱의 배수이면 같은 행의 원소들이 캐시의 같은 묶음에 몰린다.
- 제안(고치지 않음, 사용자에게 알릴 것): n이 2의 큰 거듭제곱(예: 512의 배수)일 때 작업 버퍼의 ld를 몇 원소 늘리기. 결과 비트가 바뀔 수 있으므로 비트 관계(`inv(a) == solve(a, I)` 등)와 레이아웃 불변성 테스트를 함께 확인해야 한다.

## linalg: 우변 3개의 삼각 풀이도 trsv가 빠르다 (2026-11-25, 48장 측정)

- 위치: `csrc/linalg/lapack.hpp`의 `kTrsvMaxK = 2`(우변 2열 이하만 열마다 `trsv`, 3열부터 `trtrs`/`getrs` 안의 `trsm`).
- 관찰(Apple M4 Pro, Accelerate, 하삼각 풀이, 여러 번 잰 최솟값, `textbook/figures/48/kcols.c`): k = 3에서 trsv 세 번이 trsm보다 빠르다. n = 100: 4.3 대 7.9µs, n = 400: 36.8 대 57.7µs, n = 1600: 331 대 581µs. k = 4부터는 trsm이 빠르다(n = 1600: 440 대 305µs). trsm은 홀수 k에서 유난히 느리다(k = 5: 506µs, k = 6: 516µs).
- 제안(고치지 않음, 사용자에게 알릴 것): `kTrsvMaxK`를 3으로. 결과 비트가 바뀌므로 비트 관계 테스트를 함께 확인.

## array: 스칼라 제수의 정수 바닥 나눗셈은 마법수 곱으로 두 배 빨라진다 (2026-11-25, 2장 측정)

- 위치: `csrc/array/kernels/scalar_ops.hpp`의 `FloorDivide::f(i64, i64)`는 원소마다 `SDIV`(와 `MSUB`)를 쓴다. 제수가 스칼라(브로드캐스트된 0차원)인 경우의 따로 된 경로는 없다(`ops/ufunc.hpp`의 가중치만 있음).
- 관찰(Apple M4 Pro, clang -O2, 2^20개 int64, 50번 중 최솟값, `textbook/figures/02/divbench.cpp`): 제수 7에서 원소당 0.47ns(stardust와 같은 식) 대 0.21ns(부호 접기 `u = a ^ (a >> 63)` 뒤 63비트 마법수 `m = ceil(2^(63+l)/d)`로 `mulhi(u, m) >> (l-1)`, 다시 `^ s`). 제수 3, 10, 10^9+7에서도 같다. 2^20개 무작위 값과 극단값(INT_MIN, INT_MAX, ±d 근처)에서 결과가 비트까지 같다.
- 제안(고치지 않음, 사용자에게 알릴 것): 제수가 스칼라이고 양수이면 마법수를 한 번 계산해 루프에 쓰기(음수 제수, d = 1, 2의 거듭제곱은 따로). `remainder`도 `a - q*d`로 같은 이득. 결과는 정수라 비트가 바뀌지 않는다.

## frame: 여러 키 접기의 넘침 가드는 행 수가 2^31을 넘으면 보장되지 않는다 (2026-11-25, 2장)

- 위치: `csrc/frame/ops/hash.cpp` `factorize_rows`. `acc_n > 2^62 / next.ngroups`이면 `acc`를 다시 번호 매겨 `acc_n`을 나타난 조합 수(≤ 행 수 n)로 줄인 뒤 곱한다. 그런데 줄인 뒤에도 `acc_n * next.ngroups`가 2^62를 넘는지는 다시 보지 않는다.
- 행 수 n ≤ 2^31이면 두 값이 모두 n 이하라 곱이 2^62 이하로 안전하다. n > 2^31이고 두 키의 서로 다른 값이 모두 2^31개를 넘으면 `a[i] * m + b[i]`가 넘쳐 서로 다른 키 쌍이 같은 코드를 받을 수 있다(조용한 오답). 아주 큰 표에서만 생기는 일이다.
- 제안(고치지 않음): 다시 번호 매긴 뒤에도 곱이 넘치면 해시로 (acc, code) 쌍을 직접 번호 매기기, 또는 128비트 키.

## frame: composite_keys가 범위 밖 키에서 부호 있는 곱으로 넘칠 수 있다 (2026-11-25, 2장 깊이 검토)

- 위치: `csrc/frame/ops/join.cpp` `composite_keys`의 `acc = acc * ranges[j].span + static_cast<std::int64_t>(k)`(192–195행 근처). 왼쪽 행의 값이 오른쪽 범위 밖이면 `k`(부호 없는 오프셋)가 매우 클 수 있고, 그 값을 `int64`로 바꿔 더한 뒤 다음 열에서 `acc * span`을 하면 부호 있는 넘침(정의되지 않은 동작)이 날 수 있다. 결과는 `in`이 거짓이라 버려지지만 UB 자체는 남는다.
- 제안(고치지 않음): `acc`를 `uint64_t`로 계산하거나, 범위 밖이 확인되면 접기를 멈추기.

## random: 배열 범위 integers의 offset_of는 원소마다 축 수만큼 나눗셈을 두 번 한다 (2026-11-25, 2장 깊이 검토)

- 위치: `csrc/random/random.cpp` `offset_of`(66행 근처): 원소 i마다 `i % shape[d]`, `i /= shape[d]`를 축마다 한다. 범위가 배열(브로드캐스트 뷰)인 `integers`가 원소마다 low와 high에 대해 두 번 부른다.
- 제안(고치지 않음, 측정 전): shape은 호출 동안 고정이므로 축마다 마법수(2장 Algorithm alg:magic의 부호 없는 판)를 한 번 구해 쓰거나, 다차원 반복자로 오프셋을 누적하기. 이득은 재 보지 않았다.

## special: Hurwitz 제타의 직접 합에서 a += 1의 반올림이 s배로 커진다 (2026-11-25, 3장 측정)

- 위치: `csrc/special/ops/zeta.hpp` `euler_maclaurin`(그리고 같은 꼴의 `hurwitz_relative`, `hurwitz_scaled`)의 `a += 1.0; b = std::pow(a, -s);`. q가 정수가 아니면 q + i가 반올림되고, a의 상대오차 δ가 a^-s에서 sδ가 된다.
- 관찰(`textbook/figures/03/hurwitz_fix.py`, 정확한 유리수 참값, q ∈ (0.01, 100] 무작위 300점; 공개 저장소 복제본의 `sd.special.zeta`도 같은 최대 오차): s = 10에서 9.11ULP, s = 20에서 8.08ULP. 합의 낮은 부분을 two_sum으로 들고 다니며 b ← b (1 − s·lo/a)로 일차 보정하면 3.49, 3.38ULP. s = 2, 3에서는 변화 없음.
- 제안(고치지 않음): 위의 일차 보정, 또는 a를 double-double로 두기. 결과 비트가 바뀌므로 골든 갱신 필요.

## linalg: 노름의 척도 맞춤 경로가 a_max로 나눈다 (2026-11-25, 4장)

- 위치: `csrc/linalg/ops/vnorm.hpp` `finish_pnorm`의 `out[k] / amax`. 넘치거나 아주 작은 입력에서만 타는 경로이다.
- 관찰: a_max 대신 그에 가까운 2의 거듭제곱(예: `ldexp(1, -ilogb(amax))`)을 곱하면 나눗셈과 그 반올림이 없어지고 곱셈이 정확하다. 결과의 끝 비트가 바뀔 수 있다. 측정한 정확도 문제는 없다(`textbook/figures/04/measure.py`: 2^-1070–2^1010 척도에서 모두 올바르게 반올림).
- 제안(고치지 않음, 우선순위 낮음): 2의 거듭제곱 척도. 골든 갱신 필요.

## array: 복소 sqrt가 시스템 csqrt를 그대로 써서 |y| < 2^-511이면 작은 부분을 0으로 돌려준다 (2026-11-25, 5장 측정)

- 위치: `csrc/array/kernels/scalar_ops.hpp`의 `SD_FLOAT_FN(Sqrt, …, call(csqrt, z))`. Apple libm의 `csqrt(x + iy)`는 x > 0이고 |y| < 2^-511(≈1.49e-154)이면 허수부를 0으로, x < 0이면 실수부를 0으로 돌려준다(y²이 정규수 아래로 내려가는 곳). numpy도 같은 libm을 써서 같다. Python `cmath.sqrt`와 `special/cplx.hpp`의 Kahan 식은 정확하다.
- 관찰(`textbook/figures/05/branch.py`, `cauchy.py`; 공개 저장소 복제본): `sd.sqrt(1 + 1e-160j)`의 허수부가 0(참값 5e-161). 그 결과 복소 단계 미분 `Im f(x+ih)/h`가 h < 1.5e-154에서 0.11만큼 틀린다(시험 함수 e^x/√(sin³x+cos³x)).
- 제안(고치지 않음): numpy와 비트를 맞추는 것이 설계 목표라면 그대로 두되 문서에 적기. 정확도를 택한다면 Kahan 식(x ≥ 0: (t, y/(2t)), t = √((|x|+|z|)/2), 넘침 쪽 척도 맞춤 포함).

## array: 복소 log1p가 numpy 식이라 작은 |w|에서 실수부의 상대 정확도를 잃는다 (2026-11-25, 5장 측정)

- 위치: `csrc/array/kernels/scalar_ops.hpp` `complex_log1p`: Re = log(hypot(1 + a, b)). 1 + a를 먼저 반올림하므로 상대오차가 약 u/|w|이다.
- 관찰(`textbook/figures/05/branch.py`, w = |w|e^{2i}, 십진 참값): |w| = 1e-8에서 1.2e-8, |w| < 1e-16이면 실수부가 0. `special/cplx.hpp`의 `clog1p`(Re = log1p(fma(a, a+2, b²))/2)는 전 범위에서 1e-16 아래.
- 제안(고치지 않음): `clog1p`와 같은 식으로 바꾸기. numpy와 결과가 달라지므로 설계 결정이 필요.

## special: cplx.hpp의 csqrt는 |z|가 최댓값의 절반을 넘으면 넘친다 (2026-11-25, 5장)

- 위치: `csrc/special/cplx.hpp` `csqrt`의 `0.5 * (fabs(x) + hypot(x, y))`. |x| + |z|가 2^1024를 넘으면 inf가 된다. 지금 쓰는 곳(lambertw의 √(2(ez+1)), 가지점 근처)은 인자가 작아 문제가 없다.
- 제안(고치지 않음, 우선순위 낮음): 일반용으로 쓰게 되면 |z| > 2^1020에서 z/4의 제곱근에 2를 곱하는 척도 맞춤. 같은 파일의 `clog`는 |z| ≈ 1에서 log(hypot)라 상대 정확도를 잃는다(1 + 1e-10i에서 0, 참값 5e-21). 지금 쓰는 곳은 절대오차면 충분하다.

## interpolate: aaa.cpp의 복소 나눗셈은 fma가 없는 Smith 식이다 (2026-11-25, 5장 깊이 검토)

- 위치: `csrc/interpolate/aaa.cpp` 88–98행의 `cdiv`(주석 "numpy's complex quotient")는 `(ar + ai * rat) * scl`처럼 곱과 합을 따로 반올림한다. `csrc/array/kernels/scalar_ops.hpp`의 `cdiv`는 같은 식을 `fma`로 쓴다. 둘 다 "numpy의 것"이라 하지만 끝 비트가 다를 수 있다.
- 제안(고치지 않음): 하나로 모으기(scipy와 비트를 맞추려는 의도라면 주석에 그 이유를 적기).

## special: bessel.hpp의 "86 ULP" 주석과 단위 (2026-11-25, 6장 측정, 우선순위 낮음)

- 위치: `csrc/special/ops/bessel.hpp` 91–92행 주석 "over n steps float64 loses ~n/2 ULP (86 ULP at nu = 286, x = 100, measured)".
- 관찰(`textbook/figures/06/downrec.py`, `threeway.py`; 십진 참값에서 출발한 같은 점화): ν = 286.3, x = 100에서 J를 아래로 binary64로 점화하면 오차가 짝의 크기 max(|J_v|, |J_v+1|) 기준으로 최대 47u, double-double이면 1.03u. 400걸음에서는 보통 binary64 40u, 빠른 경로 식(계수를 double-double 1/x로 한 번 반올림, fma 한 번) 32u(J)·43u(Y). 주석의 86 ULP와 n/2는 무엇을 기준으로 잰 것인지(값 자체의 ULP인지, 정규화를 거친 최종값인지) 적혀 있지 않다.
- 제안(고치지 않음): 주석에 측정 기준과 스크립트를 적기. 동작 변경은 없음.

## htest: 동점 없는 큰 Mann–Whitney는 일이 큰 Grouped 동적 계획으로 떨어진다 (2026-11-25, 7장 측정)

- 위치: `csrc/htest/rank.cpp` `mwu_lane`. 동점이 없어도 log2 C(N, c) ≥ 124이면 정수 곱 공식(일 c(b+1)) 대신 `Grouped`(일 max(N c (s_max+1), 100·상태 수))로 간다. 곱 공식은 뺄셈 때문에 부동소수점으로 옮길 수 없어서이다.
- 관찰(`textbook/figures/07/worklim.py`): c = d이면 135까지만 정확 계산이 상한 1e9 안에 든다(0.11초). (300, 3000)의 'exact'는 9.35e11로 거절된다. 정수 곱 공식이라면 약 1.35e8이다.
- 제안(고치지 않음): 동점이 없을 때는 확률을 쌓는 양수 점화(Mann–Whitney의 f(c,d,u) 볼록결합, 꼬리 b까지만)나 정수 경로를 넓히는 방법(256비트, 또는 꼬리만 세는 모듈러 계산)을 검토할 만하다.

## htest: Mann–Whitney 정수 경로에 칸 수 상한 검사가 없다 (2026-11-25, 7장 깊이 검토)

- 위치: `csrc/htest/rank.cpp` 273행 근처. 정수 경로의 일 c(b+1)만 상한과 견주고 i128 배열 P[0..b]의 칸 수(b+1, 다른 커널은 1e7로 묶음)는 보지 않는다.
- 관찰(검토 때 복제본으로 측정): c = 3, d = 4×10^6에서 일은 상한 안이지만 RSS가 약 446MB였다.
- 제안(고치지 않음): 다른 커널처럼 kStateCap(1e7) 검사를 더하기.

## optimize: L-BFGS-B 결과에 사영 기울기 노름이 없고 기본 ftol이 gtol보다 먼저 걸린다 (2026-11-25, 8장 측정)

- 위치: `csrc/optimize/lbfgsb.cpp` 1385–1418행(멈춤 판정), `src/stardust/optimize/_minimize.py`(결과). 커널은 `sbgnrm`(사영 기울기 노름)을 계산하지만 결과에 돌려주지 않는다.
- 관찰(`textbook/figures/08/box_ftol.py`): n = 50, 고유값 [1, 100]의 상자 제약 이차 문제를 기본 옵션으로 풀면 gtol을 1e-5, 1e-8, 1e-11로 바꿔도 모두 19번째 반복에서 상대 감소(ftol = 2.2e-9) 판정으로 멈추고 사영 기울기는 1.03e-3, `converged`는 참이다.
- 제안(고치지 않음): `sbgnrm`을 `optimality` 필드로 돌려주기(`lsq_linear`·`least_squares`와 이름이 맞다). 판정은 scipy 그대로 두고 문서에 경고를 단다.

## optimize: lsq_linear의 optimality 뜻이 방법마다 다르고 'trf'의 해는 경계에 붙지 않는다 (2026-11-25, 8장 측정)

- 관찰(`textbook/figures/08/bvls_neg.py`): A = I, b = (−1, −2, 5), [0, 1]³에서 'bvls'의 optimality는 −1.0(묶인 변수의 엄격한 상보성 여유가 그대로 나옴), 'trf'의 x는 (4.1e-11, 5e-324, 1 − 2⁻⁵³)로 정확한 해 (0, 0, 1)의 경계에 붙지 않는다.
- 제안(고치지 않음): 문서화하고, 'trf'도 `active_mask`인 변수를 경계로 붙이는 마무리(bvls의 O117과 같은 꼴)를 검토.

## special: logsumexp가 |값| ≪ 1에서 상대 정확도를 잃는다 (2026-11-25, 8장 측정)

- 위치: `csrc/special/reduce.cpp`. m + log(Σ e^(x−m))에서 Σ가 1 + (작은 수)이면 작은 수가 반올림으로 사라진다.
- 관찰(`textbook/figures/08/lse.py`, 십진 참값): lse(0, −40)은 참값 4.25e-18 대신 0, lse(−1e−20, −50)은 상대오차 2퍼센트. 계약(1e-14·max(1, |v|)) 안이다.
- 제안(고치지 않음): 최댓값 자리 k를 빼고 m + log1p(Σ_{i≠k} e^(x_i−m))로 계산하면 상대 정확도를 얻는다. 같은 최댓값 패스에서 argmax를 얻으면 추가 비용이 거의 없다.

## regression: lasso의 dual_gap은 반올림 바닥 근처에서 믿을 수 없다 (2026-11-25, 8장 측정, 우선순위 낮음)

- 관찰(`textbook/figures/08/gap_gram.py`): Gram 경로(p ≤ 1024)는 R² = vᵀΩv − 2bᵀc + bᵀGb의 소거로 상대오차 1e-9의 R²을 쓰고, 바닥 근처의 간격이 정확한 값과 최대 4배 다르다(1.0e-15 대 4.1e-15). 음수(−2.8e-17)도 나온다(검토 때 측정). 수렴 판정에는 영향이 없다.
- 제안(고치지 않음): 문서에 "uP 수준 아래의 값은 반올림"을 적기.

## stats: 표집용 균등수 u() = (m + ½)·2⁻⁵³가 위쪽 절반에서 반올림되어 1.0이 나올 수 있다 (2026-11-25, 9장 수학 검토, 우선순위 높음)

- 위치: `csrc/stats/sample.hpp` 49행 `(static_cast<double>(next() >> 11) + 0.5) * 0x1p-53`. 같은 `w.u()`를 `sample_dist.hpp`의 여러 표집기(97, 98, 166, 516, 525, 532, 835행 근처)와 `by_inverse`가 쓴다.
- 관찰(순수 binary64 산술로 확인): m ≥ 2⁵²이면 m + 0.5가 53비트에 들지 않아 짝수 쪽으로 반올림된다. m = 2⁵³ − 1이면 u = 1.0이 정확히 나와(확률 2⁻⁵³) `by_inverse`가 `iccdf(0)`을 불러 지지의 위 끝(정규면 +∞)을 낸다. m = 2⁵²이면 u = 0.5. 위쪽 절반의 u는 2⁻⁵³의 짝수배뿐이어서 1 − u의 가장 작은 0 아닌 값이 2⁻⁵²이고 대칭이 깨진다(정규 역변환이라면 위 끝 8.13, 아래 끝 −8.29). 주석 "(0, 1), symmetric about 1/2"와 다르다.
- 제안(고치지 않음): 단어의 맨 위 비트로 쪽을 고르고 나머지 52비트로 작은 쪽 s = (m′ + ½)·2⁻⁵³ ∈ (0, ½)을 만들어(정확) `b ? iccdf(s) : icdf(s)`로 부르기. 닫힌 꼴 역변환을 쓰는 표집기들도 u 대신 (쪽, s) 쌍을 받게 하면 양쪽 꼬리가 모두 2⁻⁵⁴까지 닿는다.

## stats: 이산 분위수의 Cornish–Fisher 시작 문턱(kWideSd = 8)이 높다 (2026-11-25, 9장 깊이 검토, 우선순위 중간)

- 위치: `csrc/stats/discrete.hpp` 811행 `kWideSd = 8.0`(Poisson, 이항, 음이항, 초기하, 베타이항이 씀).
- 관찰(`textbook/figures/09/search_evals.py`, 탐색을 옮긴 코드에 복제본의 cdf·ccdf): 무작위 p에서 술어 계산 평균이 평균에서 시작하면 μ = 10에서 4.3, 63.9에서 6.1인데 Cornish–Fisher에서 시작하면 2.0. 검토 때 잰 원소당 시간(1스레드, 메모 끔): μ = 63.9에서 775 ns, μ = 64.1(CF 시작)에서 358 ns.
- 제안(고치지 않음): 문턱을 σ ≥ 1–2 정도로 낮추기. 답은 정확한 탐색이 정하므로 결과 비트는 바뀌지 않는다.

## stats: 표집의 by_inverse 경로에 이르는 분포가 없다 (2026-11-25, 9장 깊이 검토, 우선순위 낮음)

- 위치: `csrc/stats/sample.hpp` 8–12, 88–92행. 단변량 31개 분포가 모두 `Custom<D>` 특수화를 가져 기본 경로 `by_inverse`는 쓰이지 않는다. 주석 "without one the sample is by inversion"은 현재 쓰는 분포가 없다는 점을 적지 않는다.
- 제안(고치지 않음): 주석에 그 사실을 적거나 시험이 이 경로를 실제로 부르는지 확인. u() 문제(위 항목)를 고칠 때 이 경로도 함께 고친다.
- 덧붙임(위 u() 항목의 영향, 검토 때 복제본 헤더를 컴파일해 확인): u = 1에서 Laplace·로지스틱은 +∞, Cauchy는 1.6e16, 지수 ziggurat 꼬리는 경계 R의 점질량. 정규 ziggurat 꼬리와 기하는 무해.

## stats: MultivariateNormal.cdf의 "3σ" 오차는 약 99퍼센트 구간이다 (2026-11-25, 9장 깊이 검토, 문서)

- 관찰(`textbook/figures/09/mvn_cover.py`): 무작위 300문제에서 실제 오차는 모두 보고 오차와 atol 안(비율 중앙값 0.23, 최대 0.97). 따로 잰 d = 4, ρ = 1/2, b = 0, atol = 1e-6에서는 실제 4.43e-7 대 보고 4.13e-7(비율 1.07). 이동 12개의 표준오차 3배는 t₁₁에서 약 99퍼센트 구간이다.
- 제안(고치지 않음): 문서에 "약 99퍼센트(이동 12개의 t 분포)"라고 적기.

## array: 신호 NaN에서 maximum·minimum의 스칼라 경로와 NEON 경로가 다르다 (2026-11-25, 10장 깊이·수학 검토, 우선순위 중간)

- 위치: `csrc/array/kernels/scalar_ops.hpp` 114–125행 `fmax_nan`·`fmin_nan`(받은 NaN을 그대로 돌려줌)과 NEON `vmaxq_f64`·`vminq_f64`(FMAX·FMIN: 신호 NaN을 먼저 고르고 조용하게 바꿈). `NanMaximum`·`NanMinimum`과 FMAXNM·FMINNM도 신호 NaN에서 다르다.
- 관찰(`textbook/figures/10/review_sd.py`, 복제본): x[0] = 신호 NaN 0x7FF0000000000001에서 `sd.maximum(x, 0)`은 n ≤ 3이면 0x7FF0000000000001, n = 5의 첫 원소(벡터)는 0x7FF8000000000001, 마지막 원소(스칼라 꼬리)는 0x7FF0000000000001. f(x)[i] == f(x[i]) 약속이 깨진다. (조용한 NaN a, 신호 NaN b)에서는 fmax_nan이 a를, FMAX가 quiet(b)를 낸다. numpy에서 가져온 배열이 신호 NaN 비트를 그대로 지니므로 실제로 들어올 수 있다.
- 제안(고치지 않음): 스칼라 쪽을 ProcessNaNs(신호 NaN 우선, 조용하게 바꾸기)로 맞추기. 시험의 특수값 목록(test_ufunc.py 등)에 신호 NaN·페이로드 NaN·음의 NaN을 넣기.

## array: nextafter 주석의 "약 8배"가 지금 측정과 다르다 (2026-11-25, 10장 깊이 검토, 문서)

- 위치: `csrc/array/kernels/scalar_ops.hpp` 369–370행.
- 관찰(`textbook/figures/10/review_sd.py`): 10⁶개, 1스레드 sd 581 µs 대 numpy 1450 µs(약 2.5배), 기본 10스레드 78 µs.
- 제안(고치지 않음): 주석의 수치를 고치거나 측정 조건을 적기.

## array: sd.log(-1)의 NaN 페이로드가 numpy와 다르다 (2026-11-25, 10장 깊이 검토, 참고)

- 관찰: sd.log(-1)은 vForce가 만든 0x7FF8000000000033, numpy는 기본 NaN 0x7FF8000000000000. 비트 정책상 문제는 아니다(NaN 규칙은 "NaN이다"까지만 같음).
