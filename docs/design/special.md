# 특수함수 영역 설계 (sd.special)

> 상태: **확정** (2026-11-12). 1–3라운드(Q1–Q19)와 그 뒤 사용자 결정(Q20, 2026-11-20의 리뷰 결정; Q61, 2026-11-22), 2026-11-22 성능 작업의 추천안(Q21–Q60)을 반영했다. 모든 함수가 구현되어 있다.

`sd.special`(scipy.special의 재구현)의 확정 사항과 내부 구조를 적는다.

- 공통 규칙과 통합 결정은 [DESIGN.md](../../DESIGN.md), 층 구조는 [ARCHITECTURE.md](../../ARCHITECTURE.md)에 있다. 공개 함수와 시그니처는 [docs/api/special.md](../api/special.md)에 있다.
- 배열 영역의 내부(원소별 엔진, 버퍼 변환, 풀, NEON 규칙)는 [array.md](array.md)를 인용한다.
- 이 영역은 나중의 `sd.stats`(scipy.stats)의 바닥이다. 분포의 cdf·ppf는 대부분 여기의 함수로 계산된다.
- 수치는 Apple M4 Pro 한 대에서 잰 값이다. 비교 대상은 scipy 1.18.1이다.

---

## 1. 기반

### 1.1 결정 기록
| 번호 | 결정 (2026-11-12) |
|---|---|
| Q1 언어 | C++20 + nanobind, 같은 `_core`의 `csrc/special/`. Python 층은 거의 없다 |
| Q2 범위 | **대** (§1.2) |
| Q3 정확도 | 보장 영역에서 상대오차 1e-14 이내. 극단 영역은 잰 한계를 문서에 적는다 (§2.3) |
| Q4 근사 계수 | 공개된 계수를 출처를 밝혀 가져온다. 계수를 직접 만들지 않는다 (§4.1) |
| Q5 NEON | 분기가 적은 핫 함수만 손으로 짠다: erf/erfc/ndtr/ndtri, expit/logit, gammaln (§3.4) |
| Q6 dtype | float64 실수만. bool/int64는 float64로 바꾸고, complex는 `TypeError` (§2.1). 예외는 Q7 |
| Q7 복소수 | 실수를 모두 끝낸 뒤 마지막 마일스톤에서 erf/wofz, loggamma, lambertw만 복소수를 받는다. 복소수 Bessel(AMOS급)은 하지 않는다 |
| Q8 구간 한도 | 근사 방식을 바꾸는 갈래는 함수마다 2–3개. 정의역을 옮기는 변환(반사, 점화식, 선형 변환, 대칭)은 세지 않는다. 퇴화 경우는 일반 극한 공식 하나로 처리한다 (§4.2). 예외: 불완전 베타는 측정 뒤 사용자 결정으로 다섯 방식이다 (§5.5) |
| Q9 보장 영역 밖 | 값은 계산하고, 정확도가 느슨하다는 것을 문서에 적는다. 정해진 시간 안에 값을 낼 수 없거나(반복 상한, §1.3) 반올림 한계를 증명하지 못하는 식밖에 없으면 정의역 안에서도 NaN이다(사용자 결정 2026-11-17·18). 그런 범위는 docs/api/special.md의 함수별 절에 적는다 |
| Q10 정답지 | 엄밀한 구간 산술 Arb(python-flint)를 테스트 의존성으로 더한다(처음에는 mpmath, 2026-11-16 Arb로 바꿨다, §6). 정확도는 고정밀 값 대비 ULP로 잰다. scipy는 차이와 성능을 비교하는 데 쓴다 (§6) |
| Q11 n항 디스패처 | 배열 영역에 일반 n항 원소별 디스패처를 더하고 `kMaxOperands`를 5로 올린다. 별도 마일스톤(S0)이고, 기존 함수의 결과는 비트까지 같다 (§3.1) |
| Q12 이름 | scipy 이름을 쓴다. 입력은 모두 `/` 앞, `out=`은 `*` 뒤. sd에 이미 있는 함수는 다시 두지 않는다 |
| Q13 리덕션 옵션 | `logsumexp`, `softmax`, `log_softmax`의 옵션은 `axis`(기본 `None`, int·tuple·리스트도 받는다)와 `logsumexp`의 `keepdims`(기본 `False`)뿐이다. scipy의 `b`, `return_sign`은 없다 |
| Q14 시작 | 통합 로드맵 M7이 끝난 뒤 |
| Q15 무거운 함수 | `jv yv iv kv`와 `jve yve ive kve`(실수 차수, 실수 인자), `hyp1f1`, `hyp2f1`(실수), `lambertw`(가지 0, −1), `airy`(`AiryResult`) |
| Q16 커널 안의 `exp`/`log` | Q5의 NEON 함수는 커널 안에서 1024원소 버퍼로 vForce를 한꺼번에 부르고, 스칼라 입력도 n = 1로 같은 경로를 탄다. 나머지 함수는 libm (§3.4) |
| Q17 정수 차수 인자 | int64와 float64를 받는다. 정수가 아닌 값과 음수는 NaN. 자르지 않는다 (§2.1) |
| Q18 진동 함수의 영점 근처 | 상대오차 대신 `|오차| ≤ 1e-14 × 포락선`을 보장한다. 영점 인수분해는 하지 않는다 (§2.3) |
| Q19 복소수의 대상 | `wofz`, `loggamma`(새 함수)와 `erf erfc erfcx lambertw`의 complex 입력 (§5.13) |
| Q20 J의 0 판정 (사용자 결정 2026-11-19) | `jv`·`jve`는 0 < x < ν에서 엄밀한 상한(DLMF 10.14.5의 e^(−ν(α − tanh α)), x = ν sech α, 또는 10.14.4의 (x/2)^ν/Γ(ν + 1))이 e^−746.2(가장 작은 서브노멀의 절반 e^−745.13 아래, 여유 1)보다 작으면 0이다. 값도 증명된 0도 없는 곳만 NaN이다(1e7걸음 상한, §1.3). `yv`는 증명할 하한이 없어 옛 판정(order_saturates)만 −inf다 (§5.8) |
| Q21 J·Y의 float64 경로 (추천안, 사용자 위임 2026-11-22) | `jv yv jve yve`는 \|ν\| < 500, x ≥ 2⁻²⁰⁰에서 J와 Y를 따로 float64로 계산한다(`fast_j`, `fast_y`, §5.8). 결과가 정상 범위가 아니면 일반 경로다. 옛 규칙은 모든 차수에서 Temme(CF1을 앞뒤로 두 번)와 double-double 점화식, x ≥ 30의 Hankel이었다. 이유: 흔한 입력(ν < 20, x < 50)에서 약 3–4배. 참값 파일 `jv` 최대 460 → 208 ULP(포락선 배율 23.2 → 13, 계약 0.262 → 0.181), `yv` 393 → 207 ULP(포락선 배율 13.9 → 15, 계약 0.244 → 0.247). 경계 500은 float64 점화의 손실이 옛 최대를 넘지 않는 곳이다(올리지 않는 것은 Q61, §10) |
| Q22 iv·kv의 float64 경로 (추천안, 사용자 위임 2026-11-22) | `iv kv ive kve`는 \|ν\| < 30, 2⁻²⁰⁰ ≤ x < 700에서 Boost보다 먼저 float64 경로를 쓴다(`fast_iv_kv`, §5.8). I만 필요하고 x² < 4(ν + 1)이면 양항 멱급수, 그 밖은 K_μ, K_{μ+1}과 K의 위로 점화, I는 Wronskian으로 낸다. 값이 정상 범위가 아니면 Boost, 그다음 일반 경로다(§5.14의 S10 라우팅을 바꿈). 이유: Boost보다 빠르고 더 정확하다. 참값 파일 `iv` 39 → 9 ULP(계약 0.531 → 0.18), `ive` 45 → 10, `kv` 22, `kve` 16 그대로. Arb 흔한 입력 집합에서 `iv` 최대 13(Boost 40), `kv` 16 ULP |
| Q23 Airy의 구간 (추천안, 사용자 위임 2026-11-22) | −9.7 < x < −2는 ±1/3, ±2/3 차수 격자 둘의 Miller 재귀(Neumann 합으로 맞춤), 1 < x < 9.7은 K 한 쌍과 I의 비, x ≥ 9.7(ζ ≥ 20)은 지수 감소 점근전개다(§5.9). 옛 규칙은 일반 J·Y, I·K 경로를 차수마다 불렀다. 이유: 참값 혼합에서 약 2.8배. 바뀐 구간의 포락선 배율 최대가 같거나 줄었다(Ai의 (−9.7, −2) 17.5 → 7, Ai′ 15.7 → 10, Bi 16.9 → 11.2, Bi′ 22.3 → 14.9). 계약 최대(Ai 0.901 등)는 위상을 못 가르는 \|x\| ~ 1e12 근처라 그대로다 |
| Q24 감마의 float64 Lanczos (추천안, 사용자 위임 2026-11-22) | `gamma`, `rgamma`는 2⁻³⁰ ≤ \|x\| ≤ 170에서 Lanczos 식을 float64로 계산한다(`gamma_plain`, x < 0은 반사, §5.2). `gamma_scaled`도 z ≤ 170이면 이 값이다. 옛 규칙은 S3의 후보 B(Cephes)와 S10의 Boost `tgamma`였다. 이유: Arb 대비 (0, 170]에서 최대 10 대 11 ULP, 99백분위 3–4 대 5–6으로 Boost보다 정확하고 약 1.3배 빠르다. m·2ᵉ 산술을 뺀 같은 식이라 `gamma_scaled`보다 약 3배 빠르다. 참값 파일 `gamma` 10 → 7 ULP(계약 0.116 → 0.081), `rgamma` 5 → 6 ULP(계약 0.076 → 0.075) |
| Q25 sinpi·cospi (추천안, 사용자 위임 2026-11-22) | Apple에서는 libm의 `__sinpi`, `__cospi`(같은 정확한 인자 축소)를 쓴다. 옛 규칙은 `fmod` 축소 뒤 sin(π r)이었다. 이유: Arb 대비 1 대 2 ULP이고 약 2배 빠르다. 영점도 같다. 다만 음의 홀수 정수에서 sinpi가 −0이다(IEEE 754의 sinPi). 반사식을 쓰는 함수의 비트가 바뀌었다: `digamma` 혼합 영역 14.9 → 17.1(계약 0.33 → 0.379) |
| Q26 초기하 급수의 긴 합 (추천안, 사용자 위임 2026-11-22) | 양항 급수(`hyp2f1`의 a, b, c, z > 0, `hyp1f1`의 a, b, y > 0)의 보정 루프는 인자를 2⁻³²의 배수와 나머지로 나눠 a + k를 정확히 만든다(두 TwoSum 대신 덧셈 하나, §5.11). float64 첫 패스가 거의 늘 기준을 넘지 못하는 곳(`hyp2f1`: z > 0.4나 abz/c > 2, `hyp1f1`: y > 2나 ay/b > 2)은 첫 패스 없이 보정 루프로 간다. 이유: 참값 혼합에서 `hyp2f1` 약 1.8배, `hyp1f1` 약 1.7배. 참값 파일 `hyp2f1` 11, `hyp1f1` 16 ULP 그대로. 다시 보고 둔 것: 첫 패스의 받음 기준 4(Σ(k+1)\|t_k\| ≤ 4\|s\|). 16으로 올리면 Arb 흔한 입력 집합에서 최대 6 → 17 ULP, 기준 없이 34 ULP라(항의 오차가 걸음마다 쌓인다) 그대로 둔다 |
| Q27 hyp1f1의 Boost 구간 재검토 (추천안, 사용자 위임 2026-11-22) | 보장 영역의 x ≥ 0, b < 11, a ≤ 100은 그대로 Boost다. Q26의 우리 급수가 그 구간에서 1 ULP(Boost 16–26)지만, a가 100까지이거나 a, b가 작은 점을 섞으면 Boost보다 약 1.3배 느려 §5.14의 규칙(느리면 바꾸지 않는다)대로 둔다 |
| Q28 Halley의 멈춤 (`lambertw`, 추천안, 사용자 위임 2026-11-22) | 걸음이 반복값의 2⁻²⁶ 아래이면 그 걸음의 결과를 낸다(3차 수렴이라 남는 오차는 K·2⁻⁷⁸). 옛 규칙은 2⁻⁵³ 아래의 걸음을 기다렸다. f의 반올림 때문에 그런 걸음이 오지 않아 복소 z ∈ [−1, 3] × [−2, 2]의 11%가 이웃 double 사이를 64번 돌았다(스칼라 약 7배 느림). 참값 파일 `lambertw_c0` 2 → 3 ULP(계약 0.028 → 0.064), `lambertw_c1` 계약 0.028 → 0.032, 나머지 같음 |
| Q29 복소 loggamma의 1–2 둘레 꼴 (추천안, 사용자 위임 2026-11-22) | loggamma(1 + w) = −γw + Σ_{j≤J} μ(w/j) + 꼬리_J(w), J = min(6, ⌊4\|w\|₁⌋)다(§5.13). 떼어낸 μ들은 로그 하나로 합치고, J < 6의 꼬리 계수 (−1)ᵏζ(k, J + 1)/k는 표의 ζ(k, 7)에 양의 항만 더해 컴파일할 때 만들며, J = 6의 꼬리는 \|w\|에 맞춰 줄인다. 옛 규칙은 j마다 log1p와 26항 급수, 꼬리 49항이었다. 이유: 배열 약 1.8배, 스칼라 약 1.5배. 참값 파일 최대 10 ULP 그대로, 상대 최대 1.0e-15 → 1.47e-15(계약 0.101 → 0.147) |
| Q30 wofz의 연분수 (추천안, 사용자 위임 2026-11-22) | Laplace 연분수의 한 단계 (k/2)/t를 (k/2)·conj(t)/\|t\|²로 나눗셈 하나에 계산하고(옛: Smith 알고리즘, 사슬 위의 나눗셈 둘), 격자 Taylor 재귀의 /(n + 1)은 −2/n 표의 곱이다. 이유: 연분수 구간 약 2배, `wofz` 전체 약 1.6배. 참값 파일 계약 0.031 그대로(포락선 배율 2). `erf erfc erfcx`의 complex 입력도 같은 경로다 |
| Q31 NEON + vForce 청크 커널의 대상 (추천안, 사용자 위임 2026-11-22) | Q5·Q16의 대상에 `log_expit`(vvexp, vvlog1p), `ellipk`, `ellipe`(vvlog, 다항식은 레인)를 더했다(§3.4). 옛 규칙은 원소마다 libm이었다. 이유: 배열에서 약 3–5배. 참값 파일 `log_expit` 1 → 2 ULP(계약 0.0195 → 0.0395), `ellipk` 2 → 3, `ellipe` 2 그대로. 원소 하나도 같은 버퍼 경로라 `log_expit` 스칼라 호출은 vForce를 두 번 부르는 만큼 느려졌다(`f(x)[i] == f(x[i])`를 지키려고 받아들였다) |
| Q32 beta의 a ≫ b (추천안, 사용자 위임 2026-11-22) | a ≥ 2²⁰·max(1, b²), 0 < b < 171이면 B(a, b) = Γ(b)·a^(−b)·(1 + G₁/a + G₂/a²)(DLMF 5.11.13, G₁ = −b(b − 1)/2, G₂ = b(b + 1)(b − 1)(3b − 2)/24)다(`beta_far`). 다음 항은 (b²/a)³ ≤ 2⁻⁶⁰ 아래이고 a^(−b)가 서브노멀이면 쓰지 않는다. 옛 규칙은 m·2ᵉ의 Γ 비율이었다. 이유: 참값 혼합의 절반 넘는 점이 이 꼴이고 약 3배 빠르다. 참값 파일 `beta` 53 ULP(계약 0.822) 그대로 |
| Q33 Boost를 부르는 꼴 (추천안, 사용자 위임 2026-11-22) | `i0 i1 k0 k1`(과 e-스케일 판)은 `cyl_bessel_i/k(0.0, x)` 대신 Boost의 0·1차 유리식(`bessel_i0` 등)을 직접 부른다. 일반 함수에 넘긴 차수가 상수로 보이지 않아 분기를 거쳤다(약 1.6배). `gamma rgamma digamma zeta expi lambertw`는 자주 지나는 경로만 함수 안에 두고 드문 경로는 따로 뗀 함수(`noinline`)다(`digamma` 약 1.8배, 나머지 10–30%). 값은 같다 |
| Q34 m·2ᵉ 산술의 빠른 길 (추천안, 사용자 위임 2026-11-22) | `scaled`, `value`는 정상 범위의 수를 `frexp`/`ldexp` 호출 대신 비트로 나누고 붙인다(같은 값). `exp_scaled`는 \|x\| < 700이면 libm `exp`, `pow_scaled`는 결과가 정상 범위이면 libm `pow` 그대로다(둘 다 1 ULP 안). 이유: m·2ᵉ를 쓰는 경로(Bessel 일반 경로, 초기하 변환, `beta_scaled`, `binom`)가 약 1.2–1.5배. 참값 파일에서 계약을 넘거나 크게 나빠진 함수는 없다 |
| Q35 정수 모양의 로그 꼴 (추천안, 사용자 위임 2026-11-22) | `igam`은 a = 1이면 P = −expm1(−x), Q = e^(−x)의 닫힌 꼴이다(각각 한 번 반올림). sd.stats의 log Q(`gamma_detail::finite_log_upper`)는 정수 a ≤ 30의 유한합을 지수와 떼어 −y + log s로 낸다. a = 1은 모든 y에서 −y 그대로이고, a ≥ 2는 y ≥ a, y ≤ 690이고 log s ≤ y/2인 곳만이다(그 밖은 Q의 로그). `Poisson`의 logcdf 커널은 k = 0을 이 스칼라 꼴로 보낸다. 옛 규칙은 반올림한 Q의 로그, 또는 급수 P의 log1p(−P)였다. 이유: `Poisson(0.5).logcdf(0)`이 −0.5000000000000001이었다(P가 한 ULP 높았다). 그래서 `ilogcdf(−0.5)`가 0에서 1로 바뀌었다. 이제 logcdf(0) = −μ가 무작위 4000개 μ에서 비트까지 맞고, `ChiSquared(2).logccdf(x)` = −x/2도 4000개 x에서 맞다. log s ≤ y/2의 조건이 없으면 중앙값 근처에서 −y + log s가 상쇄해 `Poisson(19.2).logcdf(17)`이 13 ULP였다. 조건을 두면 정확도 스윕 `stats/poisson_logcdf`는 기준선(10 ULP) 그대로다 |
| Q36 a = ½와 유한합 (추천안, 사용자 위임 2026-11-22) | `igam`은 a = ½이면 P = erf(√x)(x < 0.2275), Q = erfc(√x)다. erfc는 `erfc_sqrt`로 계산하고 e^(−x)를 x 자체에서 만들어 √의 반올림이 들어가지 않는다. 2a가 정수이고 1 ≤ a ≤ 30인 연분수 구간(x > a)은 유한합 Q(n, x) = Σ_{j<n} D(j, x), Q(n + ½, x) = erfc(√x) + Σ_{j<n} D(j + ½, x)다. 합은 마지막 항부터 Horner 꼴(fma)로 더한다. 옛 규칙은 이 모양들도 급수와 연분수였다. 이유: a = ½는 x ≈ 1에서 60항 가까이 걸렸다(t의 BGRAT가 이것을 부른다). 앞으로 더한 합은 a = 30에서 3 ULP를 잃었고, Horner 꼴은 2 ULP 안이다. 참값 파일 `pdtrc` 13 → 6 ULP(계약 0.148 → 0.080), `pdtr` 4 그대로, `chdtr` 5 그대로 |
| Q37 x ≈ 1의 작은 a (추천안, 사용자 위임 2026-11-22) | x < 1.1의 작은 a 꼴(Boost `tgamma_small_upper_part`)을 x ≥ ½에서는 a ≤ 0.75x에 더해 a < 1 전부에 쓴다. 옛 규칙(Boost의 a ≤ 0.75x)은 0.75x < a < 1을 급수 P에 두었다. 이유: 그곳은 Q가 작은 쪽이거나 그에 가깝다. 급수의 1 − P는 Arb 대비 최대 46 ULP였고(a ∈ (0.2, 1), x ∈ (0.5, 1.1)의 3000점), 이 꼴은 최대 6 ULP다. 역함수의 근도 따라 좋아졌다: `gammainccinv(0.834, 0.318)`이 11 → 1 ULP다. 속도는 같다 |
| Q38 감마 급수의 두 항 걸음 (추천안, 사용자 위임 2026-11-22) | P의 급수 Σxⁿ/((a+1)…(a+n))는 한 걸음에 두 항이다. 두 번째 항은 첫 항에 두 비의 곱을 곱해서 만든다. 옛 규칙은 한 항씩이었다. 이유: 걸음 사이의 의존 사슬이 곱 하나와 합 하나로 반이 된다(흔한 입력 `gammainc` 약 1.2배). Arb 대비 값이 바뀐 점 80개에서 최대 6 → 4 ULP, 평균 1.7 → 1.2 ULP다 |
| Q39 나눗셈 없는 연분수 (추천안, 사용자 위임 2026-11-22) | 불완전 감마·베타의 연분수(`cf_fast`, `special/ops/cf_fast.hpp`)는 두 패스 구조를 그대로 둔다. 앞으로는 수렴분자·분모 A_n, B_n의 삼항 점화식(fma 둘)으로 항 수를 찾는다. 멈춤 조건은 Lentz와 같은 \|A_nB_{n−1} − A_{n−1}B_n\| ≤ tol·\|A_{n−1}B_n\|이고, 차는 fma로 한 번만 반올림한다. 뒤로는 꼬리를 몫 P/Q로 들고 끝에서 한 번 나눈다. 값은 2^±256 안에 두고, 계수가 2²⁰⁰(a_n은 2⁴⁰⁰)을 넘거나 멈춤의 곱이 2⁻⁹⁰⁰ 아래이면 `cf.hpp`의 `lentz_backward`로 간다(sd.stats의 깊은 역함수가 x = 1e200에서 부른다). BFRAC의 항은 나눗셈 하나로 만든다(1/den과 1/(den + 2)를 1/(den(den + 2))에서). 옛 규칙은 Lentz였다(사슬 위의 나눗셈 둘, 뒤로 갈 때 항마다 나눗셈 하나). 이유: 연분수 구간 약 1.2–1.3배. 참값 혼합의 99.6%가 비트까지 같고 나머지는 4 ULP 안이다. 참값 파일 `betainc` 22, `gammaincc` 13 ULP 그대로 |
| Q40 감마 역함수의 출발점 (추천안, 사용자 위임 2026-11-22) | `gammaincinv`, `gammainccinv`는 Boost `find_inverse_gamma`(DiDonato–Morris의 근사와 큰 a의 Temme)를 출발점으로만 쓴다. 그 뒤는 `igam`에 대한 우리 Halley 걸음이다(`halley_fast`, Q41). §5.14의 라우팅을 바꿨다. 옛 규칙은 Boost의 `gamma_p_inv`/`gamma_q_inv`에 우리 Halley 한 걸음(`igam_polish`, 지웠다)이었다. 이유: Boost 역함수가 반 자릿수까지 도는 평가가 빠져 약 2.5배다. Arb 대비 흔한 입력 400점에서 옛 길과 새 길 모두 평균 0.4, 최대 4 ULP다. 참값 파일 `gammainccinv` 최대 1684 → 1061 ULP(배율 5 그대로, 계약 0.0765 → 0.0632), `gammaincinv` 1303 그대로(계약 0.0725 → 0.083), `chdtri` 507 → 409(배율 5.4 → 5.7). 근은 이제 우리 순방향 함수의 근이라 정방향 오차를 그대로 따른다 |
| Q41 Halley의 이른 멈춤 (불완전 함수의 역함수, 추천안, 사용자 위임 2026-11-22) | `halley_fast`(`special/ops/halley_fast.hpp`)는 `halley`에 끝내는 길을 하나 더한다. Halley 걸음 d(로그 걸음이나 이분이 아닌 것)가 다음 세 조건을 모두 만족하면 그 결과를 마지막으로 낸다. 첫째, 남길 오차 \|C\|d³가 2⁻⁵⁶x의 1/16 아래다(Halley 상수 C = h²/12 − h′/6, h = F″/F′는 닫힌 꼴). 둘째, d ≤ 2⁻¹⁶x다. 셋째, d(\|h\| + √\|h′\|) ≤ 2⁻¹⁰이다. 옛 규칙은 잔차가 2⁻⁵⁰F 아래이거나 걸음이 2⁻⁵⁴x 아래일 때만 끝냈다. 그런 걸음은 수렴한 점에서 한 번 더 평가해야 보였다. 이유: 평가가 하나 줄어든다(`betaincinv` 흔한 입력 3.1 → 2.2회, 약 1.4배). Arb 대비 정확도는 같다(감마 흔한 입력 평균 0.41 대 0.42 ULP). 참값 파일 `betaincinv` p999 2960 → 1411 ULP(배율 9 → 6), `betainccinv` 최대 3979 → 1888(배율 10 → 8). `lambertw`의 Q28과 같은 생각이다 |
| Q42 베타 앞 인자의 직접 꼴 (추천안, 사용자 위임 2026-11-22) | 모양 2⁻¹⁹–2¹², x, y ≥ 2⁻⁹⁹⁹, \|L\| ≤ 700이면 B = x^a y^b/B(a, b) = e^L·F다. L = a log x + b log y + S(c) − S(a) − S(b)는 double-double, F = F(c)/(F(a)F(b))다. 여기서 Γ(z) = e^S·F이고, z < 13이면 S = 0, F = Γ(z)(`gamma_small`), z ≥ 13이면 Stirling의 S와 F = √(2π)Γ*(z)다. c = a + b는 S에서 double-double이고, F에서는 1차 보정 Γ(c.hi)(1 + ψ(c.hi)c.lo)이다(ψ는 1e-2면 된다). 모양에만 달린 S, F는 스레드마다 마지막 (min, max) 짝을 둔다(키의 순수 함수라 비트는 적중과 상관없다). BPSER는 y^b를 곱했다 다시 나누지 않고 x^a/B(a, b)를 직접 받는다(`direct_power`). 그 밖은 §5.4의 Stirling 정규화 꼴이다. 옛 규칙은 모든 모양에서 double-double μ항 둘과 Γ* 셋이었다. 이유: 앞 인자가 흔한 입력 `betainc`의 절반이었다(흔한 입력 약 1.15배, `betaincinv` 약 1.4배). Arb 대비 다섯 점 비교에서 새 꼴 1–2 ULP, 옛 꼴 최대 10 ULP다. 참값 파일 `fdtr` 12 → 8 ULP, `betainc` 22 그대로, `betaincc` 19 → 21 |
| Q43 BGRAT의 r (추천안, 사용자 위임 2026-11-22) | BGRAT의 r = z^b e^(−z)/Γ(b)는 Q(b, z)를 낸 `igam`의 x·P′(x)를 z.lo만큼 1차 보정해 쓴다. 옛 규칙은 앞 인자를 한 번 더 만들었다. 이유: BGRAT 시간의 약 1/5였다(`stdtr`·`fdtr` 약 1.2배). Arb 대비 값이 바뀐 `fdtr` 60점에서 최대 8 → 6 ULP, 평균 2.1 → 1.4 ULP다 |
| Q44 대칭점 I₁/₂(a, a) (추천안, 사용자 위임 2026-11-22) | `ibeta`는 a = b, x = ½(정확히)이면 ½, ½이다. 옛 규칙은 BFRAC였고 ½에서 몇 ULP를 양쪽으로 벗어났다. 이유: 이산 분위수가 이 점에서 갈린다. `NegativeBinomial(100, ½).iccdf(½)`는 ccdf(99) = I₁/₂(100, 100) = ½이라 99다. 골든 격자에서 HEAD도 틀렸던 점이 맞게 됐다: `NegativeBinomial` 격자의 icdf·median 100 → 99, `Beta(1e-10, 1e-10)`의 median 0.4999994 → 0.5, `F(1e-10, 1e-10)`의 median 0.99997 → 1 |
| Q45 betainc의 Boost 구간 재검토 (추천안, 사용자 위임 2026-11-22) | a, b ≤ 6은 그대로 Boost다(§5.14). Q39, Q42 뒤로 우리 `ibeta`는 흔한 입력의 그 구간에서 Boost와 비슷하다. 하지만 참값 혼합의 그 구간에서는 Boost가 약 1.25배 빨라, 느리면 바꾸지 않는다는 규칙대로 둔다 |
| Q46 큰 자유도의 stdtrit (추천안, 사용자 위임 2026-11-22) | df ≥ 1000·max(1, z²)(z는 p의 정규 편차)이면 t = z + g₁/df + g₂/df² + g₃/df³ + g₄/df⁴(A&S 26.7.5의 Cornish–Fisher)다. 다음 항은 t의 2⁻⁵⁶ 아래다. 옛 규칙은 Hill의 시작에 꼬리 함수의 Halley 걸음(그 모양에서는 BGRAT, 평가 1–2회)이었다. 깊은 꼬리는 betaincinv였다. 이유: 참값 혼합의 약 1/3이 이 꼴이라 약 1.8배다. Arb 대비 값이 바뀐 40점에서 최대 4 → 2 ULP다. `stdtrit(1e10, 5e-324)`는 NaN이었고, 이제 올바르게 반올림한 −38.46740704115129다(골든 `special/stdtrit grid`). 참값 파일 7.16e5 ULP(배율 7) 그대로 |
| Q47 ndtri_exp의 꼬리 (추천안, 사용자 위임 2026-11-22) | y ≥ −745이면 AS 241의 꼬리 유리식에 L = log min(p, 1 − p) = y를 바로 넣는다(exp와 log가 없다). ndtri의 유리식이 s = √(−L) = 27.3(p = 5e-324)까지 맞으므로 값은 ndtri(exp(y))와 같은 식이다. 그 아래는 점근식 log Φ(x) = −x²/2 − log(−x) − log(2π)/2 + log S(S는 8항, 9번째 항은 1e-19 아래)에 대한 Newton 두 걸음이다. 출발점은 x² = w − log(w − log w)다. 잔차의 x²/2 + y는 two_prod로 정확하다. y < −1e8은 고정점 세 걸음이다. 옛 규칙은 ndtri(exp(y))와, 그 아래 고정점 다섯 걸음 + `log_ndtr`의 Newton 여덟 걸음까지였다. 이유: 약 3배. 참값 파일 4 ULP 그대로(p999 4 → 3). Arb 대비 y = −1000에서 올바르게 반올림했다(옛 1 ULP) |
| Q48 비중심 역함수의 Newton (추천안, 사용자 위임 2026-11-22) | `chndtrix`, `ncfdtri`, `nctdtrit`는 먼저 log(꼬리/목표)에 Newton을 한다(`newton_root`). 양수 변수는 log x 눈금, t는 asinh 눈금이다. 꼬리와 밀도는 순방향과 같은 순서로 구한다: 덩어리 합(lite)의 밀도, 걷기가 걸음마다 모은 Σw_jG′_j, t의 반대쪽 적분의 ∫s φ(z) f_S ds다. 밀도는 꼬리 길이 꼬리/밀도로 넘긴다(m·2ᵉ의 몫이라 밀도가 꼬리보다 먼저 언더플로하는 무거운 꼬리도 된다). 걸음은 눈금에서 2부터 시작하는 상한을 두고, 같은 쪽으로 잘리면 두 배로 키운다. 지나온 점으로 브래킷을 좁히고, 브래킷을 벗어나는 걸음은 이분으로 바꾼다. 끝나는 조건은 두 Newton 걸음에서 잰 이차 상수로 다음 오차가 눈금의 2⁻⁵³ 아래일 때, 또는 걸음이 2⁻⁵⁰ 아래일 때다. 24걸음 안에 끝나지 않거나 밀도가 없으면 옛 브래킷과 zeroin으로 간다. df ≤ 2이고 꼬리가 1e-8 아래이며 비중심도가 그 꼬리 쪽이면 출발점은 중심 t의 분위수다(꼬리가 \|t\|^−df로 떨어져 정규형 출발이 너무 가깝다). 옛 규칙은 늘 브래킷 + 이분 + zeroin이었다(평가 5–12회, 새 길 3–4회). 이유: 약 2배. 참값 파일 `nctdtrit` 250 → 500 ULP(배율 7 → 5, 계약 0.092 → 0.073; 근은 순방향 함수의 근이고, 오차는 그 함수의 오차다), `chndtrix` 152 → 375(배율 3.5 → 4), `ncfdtri` 496 → 523(계약 0.144 → 0.102) |
| Q49 혼합 걷기의 산술 (추천안, 사용자 위임 2026-11-22) | 먼 끝에서 걷는 `mixture_walk`의 무게와 T의 비는 "double + 오차" 산술(`lite::Lite`, 걸음마다 상대 2⁻¹⁰⁰)로 한다. 나눗셈은 걸음마다 많아야 하나다(T의 비; 아래 족의 무게 비는 1/μ의 곱). 바깥으로 가는 상한은 걸음마다 나눗셈 하나와 상대 여유 2⁻⁴⁰을 둔 곱이다(b < 1의 e^u는 1/(1 − u) ≥ e^u로 바꾼다). 역함수에 쓰도록 밀도 Σw_jG′_j도 같이 모은다. 옛 규칙은 double-double과 `recip`(나눗셈 + Newton 보정)이었다. 이유: 걷기 약 1.3배. 값은 옛 걷기와 1 ULP 안이다. 걷기는 이제 μ ≤ 2e4의 float64 걷기가 받지 못한 곳(μ > 2e4, 틀이 맞지 않는 곳)의 뒷길이다 |
| Q50 덩어리 합의 첫 항 인자 (추천안, 사용자 위임 2026-11-22) | lite의 첫 항 인자 e^(−μ)/Γ(a + 1), e^(−μ)Γ(a + b)/(Γ(a + 1)Γ(b))는 모양 2⁻¹⁹–2¹²에서 Q42의 Γ = e^S·F 나눔으로 만든다(`first_factor`). 그 밖은 `dd::loggamma` 셋이다. 옛 규칙은 늘 `dd::loggamma` 셋이었다. 이유: 그것이 nc < 1의 `ncfdtr` 시간의 1/3이었다(그 입력 약 1.4배). 정확도: `gamma_small`이 하나에 1 ULP라 첫 항의 오차 예산이 몇 단위 는다. 참값 파일 `ncfdtrc` 15 그대로(계약 0.185 → 0.168), `ncfdtr` p999 6 → 7(최대 7 그대로) |
| Q51 혼합 합의 float64 걷기 (추천안, 사용자 위임 2026-11-22) | μ ≤ 2e4의 확률(wide가 아닌 합)은 `plain_walk`가 먼저 더한다(§5.16). 무게 w와 T는 j*(가장 큰 항)의 직접값에서 출발해 양쪽 바깥으로 비율을 곱하고, G의 점화가 불안정한 먼 쪽(j*에서 J까지)은 부분합으로 바꿔(Abel 합) J의 G 하나만 쓴다. 비율은 정확한 값에서 한 번만 반올림한다(몫은 나머지로, 곱은 fma로, 모양 s0 + j는 hi + lo로). 비율은 8개씩 미리 계산해 나눗셈이 사슬 밖에 있다. 틀(프레임)이 맞지 않거나 걷기가 끝나지 않으면 `mixture_walk`(Q49)다. 옛 규칙은 모든 μ에서 먼 끝에서 걷는 "double + 오차" 걷기였다. 이유: 걷기 약 2–3배(약 27√μ걸음의 "double + 오차" 연산이 약 18√μ걸음의 float64 연산이 된다). 내준 정확도: 참값 파일 `chndtr` 최대 5 → 14 ULP(p999 4 → 11), `chndtrc` 10 → 16(8 → 13), `ncfdtr` 7 → 18(6 → 17), `ncfdtrc` 15 → 26(10 → 17), `nctdtr` 5 → 15(5 → 12), `nctdtrc` 7 → 10(6 → 10), 역함수 `ncfdtri` 배율 9.9 → 21.6, `chndtrix` p999 152 → 375 ULP(배율 3.5 → 3.8). sd.stats 스윕: `noncentralchisquared_cdf` 8 → 22 ULP, 비중심 cdf·ccdf·로그 꼴 최대 ≤ 26(logccdf 33), p999 4–8 → 11–17, `noncentralf_icdf` 배율 9.3 → 12. 계약을 넘은 곳은 없다. 모든 항이 양수라 꼬리(p → 0, 1 − p → 0)의 상대 정확도는 그대로이고, μ > 2e4는 옛 걷기다(μ = 1e9에서 float64 걷기는 1000 ULP였다) |
| Q52 작은 쪽을 먼저 (추천안, 사용자 위임 2026-11-22) | 덩어리 합이 받지 않은 확률은 점이 분포 가운데서 표준편차로 얼마나 떨어졌는지(z, 대략)를 보고, 물은 쪽이 큰 쪽이면(\|z\| > 1.5) 다른 쪽을 먼저 더한다. 그것이 ¼ 이하이면 1 − 그것을 낸다(`small_side`). 물은 쪽을 직접 더한 값이 1 − 2⁻²⁰과 1 사이이면 다른 쪽도 더해 1 − 그것으로 짝을 맞춘다. 옛 규칙은 물은 쪽을 직접 더했다(1에 가까운 쪽은 무게의 종 전체를 걷는다). 이유: `nctdtrc(14861, 94.6, 1.74)` = 1이 다른 쪽 0.1배의 비용으로 끝난다. 짝 맞춤은 cdf + ccdf가 1에서 5.6e-16 벗어나던 곳(`ncfdtr(1e-100, 1e-100, 3000, 10)`)을 반올림 안으로 한다. 경계를 ¾로 두면 참값 혼합에서 25–40% 느려서 1 − 2⁻²⁰이다(그 사이에서는 두 합이 각자의 오차, 약 1e-15 안에서 맞는다). 값이 정확히 1이면 맞추지 않는다 |
| Q53 덩어리 합과 걷기의 비용 모형 (추천안, 사용자 위임 2026-11-22) | `lite_pick`은 걷기의 비용을 `walk_fixed` + 15√μ(덩어리 합 항 단위)로 보고, 그보다 긴 덩어리 합은 아래·위 합 모두 시작하지 않는다. `walk_fixed`는 감마 족 70, 베타 족 110, t 150이다. 옛 규칙은 60 + 4·(9√μ + 12)이고 위 합에는 상한이 없었다. 이유: 참값 점에서 걷기가 더 빨라지는 곳이 감마 족 μ ≈ 100, 베타 족과 t μ ≈ 20–100이다. 값은 두 길 모두 Q51의 정확도 안이다 |
| Q54 t의 반대쪽 꼬리 (추천안, 사용자 위임 2026-11-22) | 비중심도에서 먼 쪽 꼬리(P(T > t), t > 0 > δ와 그 반사)는 다른 쪽(양항 급수)이 ⅞ 이하이면 1 − 그것이다(빼기에서 7 ULP까지). 그 밖은 덩어리 합의 두 족의 차 (cp − cq)/2(cq ≤ cp/4일 때, 상대오차가 두 부분의 5/3배 이하)이고, 그것도 아니면 적분이다. 적분의 마디는 8개씩 따로 계산하고(사슬이 겹친다), sinh·cosh는 지수 하나에서, δ의 세 함수도 지수 하나에서 만든다. 옛 규칙은 늘 적분이었다. 이유: 적분의 간격은 df ~ 1 이상에 맞춰져 작은 df에서 꼬리가 거의 줄지 않는다. `nctdtr(0.001, 1, −10)`이 1700 ULP 틀렸고(Arb 0.15755778894661205), 짝의 합이 1에서 3.8e-14 벗어났다. 이제 Arb와 같다. 속도는 그 점에서 약 3배, 참값 혼합 `nctdtr`은 Q51과 함께 약 2배 |
| Q55 아주 작은 자유도 (추천안, 사용자 위임 2026-11-22) | 덩어리 합과 float64 걷기는 베타 모양 b > 0(`kMinShape`)만 받는다(0.5·5e-324는 0이고, b = 0을 `first_factor`가 감마 족의 표시로 읽었다). `ncfdtr`은 dfd/2 < 2⁻¹⁰⁰⁰이면 e^(−μ)·fdtr(dfn, dfd, x)와 그 보수 e^(−μ)·fdtrc + (1 − e^(−μ))다(j ≥ 1 항은 절대 2⁻⁹⁸⁸ 아래). 이때는 dfn = 5e-324를 1e-323으로 바꾸지 않는다(nc = 0의 fdtr와 이어진다). dfn/2 < 2⁻⁷²이면 j = 0 항을 fdtr·fdtrc에서, 나머지를 모양 1에서 시작하는 혼합(h = 1)으로 더하고 작은 쪽에서 짝을 맞춘다. w < 2⁻¹⁰⁰⁰의 위꼬리는 1 − 아래꼬리가 아니라 e^(−μ)·fdtrc + (1 − e^(−μ))다. `nctdtr`은 df < 2⁻⁷¹이고 1083·df가 작은 쪽 꼬리 min(Φ(δ), Φ(−δ))의 2⁻⁶⁰ 아래(또는 둘 다 2⁻¹⁰⁰⁰ 아래)이면 Φ(−δ), 위꼬리 Φ(δ)다(T ≤ t가 Z ≤ −δ와 1083·df 안에서 같다). 옛 길의 `BetaFamily`의 아래로 걷는 셈은 s0 + b가 걸음의 1들보다 훨씬 작아지면 정규화한다. 옛 규칙은 모든 df에서 같은 합이었다. 이유: 골든 격자의 `ncfdtr(5e-324, 5e-324, 1, 0.25)`가 e^(−1/2)이고 `ncfdtrc`가 1이라 보수가 아니었고, HEAD도 0과 1이었다(해석적 값 e^(−μ)/2와 ½ + …). 고친 뒤 df 5e-324–1e-3, nc 5e-324–3000, x 1e-300–1e300의 짝은 모두 4.5e-16 안이다. Arb 대비 `ncfdtrc(1e-300, 100.5, 1e-300, 0.001)` = 3.494045989713936e-298(Arb 3.49404598971e-298, 전에는 NaN). 참값 파일의 정확도는 그대로다. 남은 한계: t의 df 1e-200–1e-10, δ = ±30, t = ±1e300에서 작은 쪽은 NaN이고 큰 쪽은 0.99999999999999967이다(HEAD와 같다; 작은 쪽은 df·log 크기의 값인데 그것을 내는 길이 없다) |
| Q56 beta·binom의 Lanczos 비 (추천안, 사용자 위임 2026-11-22) | a, b > 0, a + b ≥ 40(Boost 구간 밖), `beta_far`(Q32)가 아니고 a ≤ 2⁴⁰이면 B(a, b) = L(a)L(b)/L(c)·e^(−(g−½))·cgh^(−½)·(agh/cgh)^(a−½)·(bgh/cgh)^(b−½)다(`beta_lanczos`). 세 지수 함수는 상수 e^(−(g−½))로 모인다. 밑 agh/cgh는 반올림한 몫에 정확한 나머지에서 얻은 1 + e를 곱하고, e는 거듭제곱에 (a − ½)e로 1차로 들어간다(2⁻²⁶ 위는 exp). 결과가 정상 범위가 아니면 옛 길이다. `binom`의 정수가 아닌 k(0 < k < n ≤ 2⁴⁰)는 1/((n + 1)·B(k + 1, n − k + 1))이고, k + 1과 n − k + 1의 낮은 부분은 al·log(agh/cgh)로 들어간다. 옛 규칙은 c ≤ 170이면 `gamma_plain` 셋(binom은 + digamma), 그 위는 m·2ᵉ의 Γ 비율이었다. 이유: 참값 혼합에서 `beta` 약 1.3배, `binom` 약 1.5배. 참값 파일 `beta` 최대 53 ULP 그대로(몇 점이 4–6 ULP 나빠짐), `binom` 9 → 11 ULP |
| Q57 리덕션의 exp (추천안, 사용자 위임 2026-11-22) | `logsumexp`, `softmax`, `log_softmax`의 exp(x − m)은 입력 둘의 청크 루프(`chunked_loop2`)다: x − m의 TwoSum을 레인에서, e^hi를 array/kernels/vmath.hpp의 NEON exp(0.52 ULP)로, (1 + lo) 보정을 레인에서 한다. 홀수 마지막 원소는 같은 연산의 스칼라다(vmath의 짝). 옛 규칙은 원소마다 libm exp였다. 이유: 그것이 시간의 대부분이었다(약 1.3–1.4배). 정확도 계약(1e-14) 안이고 scipy와의 상대 차이 최대 2.3e-15다. x − m < −708인 원소가 섞이면 vmath가 그 레인에서 libm을 불러 느려진다(실제 입력에서는 드물다) |
| Q58 스칼라 호출의 짧은 길 (추천안, 사용자 위임 2026-11-22) | 출력이 하나인 원소별 함수는 모든 입력이 정확히 Python float이고 `out=`이 없으면 run_nary 대신 Op의 float64 루프(run_nary가 그 입력에 고르는 것)를 스택의 원소 하나에 부르고 Python float를 돌려준다(`bind/nary_bind.hpp`의 `call`). int, complex, 배열, `out=`은 그대로 run_nary다. 이유: 인자 처리와 0차원 배열이 스칼라 호출의 약 15 ns였다. 값은 같은 루프라 같다. sd.stats의 `call_fast`가 같은 생각이다 |
| Q59 K_μ, K_{μ+1}의 계산 (추천안, 사용자 위임 2026-11-22) | `k_pair_scaled`는 x ≥ 20이면 K만의 대수 점근급수(가장 작은 항 ~ e^(−2x), x = 20에서 2⁻⁵⁷ 아래)이고, 2 ≤ x < 20의 Steed CF2는 d_i = D_{i−1}/D_i의 분모 점화(fma)와 곱 P_i로 Δh_i = P_i/(D_iD_{i−1})를 만들어 나눗셈을 사슬에서 뺀다. 1/a_i와 1/(D_iD_{i−1})은 나눗셈 하나, 1/(i + 1)은 표다. 옛 규칙은 x ≥ 2 전부 Lentz 꼴의 CF2(사슬 위의 나눗셈)였다. 이유: x ∈ [2, 3)에서 약 1.3배, 흔한 입력 `kv`는 몇 % 안이다. 참값 파일 `iv kv ive kve` 최대 그대로 |
| Q60 다시 보고 둔 것 (추천안, 사용자 위임 2026-11-22) | `expit`, `log_expit`의 exp는 vForce 그대로다. vmath의 NEON exp는 정규 입력에서 약 1.3배였지만 \|x\| > 708이 4분의 1인 참값 혼합에서 vvexp보다 느렸다(밖의 레인을 모아 vvexp 한 번에 넘겨도). BFRAC의 연분수를 앞 패스만으로 끝내는 것(마지막 수렴분자의 몫)은 약 5%에 정확도가 조금 나빠 두지 않았다. 불완전 베타의 앞 인자를 float64로 만드는 것은 오차가 지수 E에 비례해(꼬리에서 E·eps) 꼬리의 상대 정확도를 깨므로 하지 않는다 |
| Q61 J·Y float64 경로의 차수 경계 (사용자 결정 2026-11-22) | 지금대로 둔다: Q21의 경계 500이다. 1100으로 올리면 참값 파일의 점에서 `jv`·`yv`가 약 4배 빨라지지만 최대 포락선 ULP가 `jv` 13 → 29, `yv` 15 → 41(계약 대비 0.18 → 0.40, 0.25 → 0.56)이 되어 손실이 몇 ULP를 넘는다 (§10) |

### 1.2 범위
- **있는 것** (함수 목록과 시그니처는 docs/api/special.md)
  - 오차함수: `erf erfc erfcx erfinv erfcinv ndtr log_ndtr ndtri`
  - 감마: `gamma gammaln rgamma digamma polygamma beta betaln binom`
  - 불완전 함수: `gammainc gammaincc gammaincinv gammainccinv betainc betaincc betaincinv betainccinv`
  - 로지스틱·정보: `expit logit log_expit xlogy xlog1py entr rel_entr`
  - 리덕션: `logsumexp softmax log_softmax`
  - 제타와 지수적분: `zeta expi expn`
  - Bessel: 0·1차 `i0 i0e i1 i1e j0 j1 y0 y1 k0 k0e k1 k1e`, 임의 실수 차수 `jv yv iv kv jve yve ive kve`
  - Airy: `airy`
  - 타원적분: `ellipk ellipe`
  - 초기하와 Lambert W: `hyp1f1 hyp2f1 lambertw`
  - 복소수(마지막 마일스톤): `wofz`(새 함수), `loggamma`(새 함수), `erf erfc erfcx lambertw`의 complex 입력 (Q19)
- **없는 것**
  - sd에 이미 있는 함수: `expm1 log1p round` 등 (Q12)
  - 별칭: `jn`(= `jv`), `psi`(= `digamma`), `yn`(`yv`가 대신한다). 함수 하나에 이름 하나다 (docs/api/array.md §1.1)
  - Mathieu, 회전타원체, Struve, Kelvin, 직교다항식 `eval_*`, 구면조화, 구면 Bessel, `ellipkinc`/`ellipeinc`, `dawsn`, `erfi`, `fresnel`, `owens_t` 등
  - scipy의 오류 보고 체계(`errstate`, `seterr`, `SpecialFunctionWarning`): 경고를 내지 않는다 (API.md §2)

### 1.3 공통 원칙
- **모든 공개 함수는 원소별 함수다** (리덕션 셋과 `airy` 제외). 브로드캐스트, `out=`, Python 스칼라 결과, 버퍼 변환은 배열의 원소별 함수와 같다 (docs/api/array.md §6.2).
- **경로는 원소의 값으로만 고른다.** 급수/연분수/점근전개 중 무엇을 쓸지는 그 원소의 인자 값이 정한다. 이것은 함수의 정의의 일부다 (libm과 같다).
  - 배열의 크기, 위치, 정렬, 스레드 수로 경로를 고르지 않는다. 반복 횟수와 멈춤도 원소의 값만으로 정해진다. 그래서 원소별 함수의 결과는 스레드 수와 무관하게 비트까지 같다(골든 출력이 스레드 1·3·10에서 같다). 2026-11-22의 우선순위(사용자 결정)는 스레드 수와 무관한 비트를 더는 요구하지 않지만, 이 영역의 원소별 함수는 지금도 그렇다. 같은 기계, 같은 스레드 수의 반복 실행이 같은 비트인 것은 그대로 지킨다.
  - 배열 영역의 "데이터를 보고 알고리즘을 고르는 휴리스틱을 넣지 않는다"(array.md §1.4)는 배열 전체를 보고 고르는 것을 말한다. 원소 하나의 값으로 근사 구간을 고르는 것은 여기에 해당하지 않는다.
- **NEON 경로는 스칼라 정의와 비트가 같다** (array.md §5.2와 같은 규칙). 테스트가 세 레이아웃(연속, 스칼라 브로드캐스트, strided)을 비트로 비교한다.
- **scipy와 같은 비트는 목표가 아니다** (사용자 결정 2026-11-20, 2026-11-22에 넓힘: 속도가 먼저이고 정확도 계약은 몇 ULP 안에서 늦출 수 있다). 반올림이 아닌 동작(특수값, NaN·±0, 극점, 정의역)은 scipy를 따르고(§2.2, 다른 곳은 §2.4), 값은 Arb 참값 대비 정확도 계약(§2.3) 안이면 된다. 그래서 더 빠르거나 더 정확한 평가 순서를 쓴다. 지키는 비트 약속은 스레드 수·레이아웃과 무관함, NEON = 스칼라, 우리 골든 출력이다.
  - **축약 위치는 코드가 정한다**: 빌드는 `-ffp-contract=off`이고 FMA는 적은 곳에서만 생긴다(array.md §1.3). 다항식의 Horner는 단계마다 FMA 하나다(`common.hpp`의 `horner`와 레인의 `simd::horner`가 같은 식, §3.4). 같은 x의 sin과 cos는 `base/sincos.hpp`의 `sd::sincos`로 한 번에 구한다(`__sincos_stret`). 빌드의 `-fno-builtin-sin/-cos`는 그대로라 컴파일러가 스스로 합치지 않으므로, 어느 경로에서도 같은 함수가 불린다.
- **원소의 값 때문에 실패하지 않는다** (docs/api/array.md §1.6). 정의역 밖은 NaN, 극점은 ±inf다.
- **모든 호출은 정해진 시간 안에 끝난다** (사용자 결정 2026-11-17; docs/design/stats.md §1.3과 같다). 반복에는 상한을 상수로 두고, 상한에 닿으면 부분합이나 마지막 반복값이 아니라 NaN을 낸다. 보장 영역 안에서는 상한에 닿지 않는다. 마지막 비트에서 두 값을 오가는 반복은 그 값을 낸다.
  - 상한: Halley 100회, 불완전 감마의 급수·연분수 1e5항, BPSER·BFRAC 1e5항, `expn`·`expi` 급수, Carlson RD 200회, AGM 64회, `lambertw` 64회(걸음이 2⁻²⁶ 아래면 그 걸음에서 멈춘다, Q28), Bessel 멱급수 500항(float64 경로 200항), 실수 차수 J·Y의 점화와 CF1 1e7(CF2 1e5; 1e7걸음에 약 0.25초; float64 경로의 Miller 재귀는 600걸음 아래), Hurwitz 제타의 직접 합 82항, `hyp1f1`·`hyp2f1` 급수 1e6항(극한 꼴 1e5항). 길이가 정해진 전개(BGRAT 30, BASYM 20, Hankel·Debye)는 그대로다.
  - Boost가 평가 오류(급수 1e6항, 근 찾기 200회)를 내면 `bst::call`이 NaN으로 바꾼다.
  - `pow_scaled`·`exp_scaled`는 2ᵏ를 제곱으로 만든다(1014번 이하).
- **알고리즘 선택 규칙**: 함수마다 후보(§5)를 아래 순서로 견주어 고른다. 고른 것과 이유는 그 마일스톤의 해설에 측정과 함께 적는다.
  1. 보장 영역에서 정확도 목표(§2.3)를 지키는가
  2. 구간 한도(Q8)를 지키는가
  3. 빠른가 (1스레드, 원소 10⁶개)
  - 이 순서로 정할 수 없거나, 보장 영역을 줄여야 할 때는 사용자에게 묻는다.

### 1.4 층과 파일
```text
sd.special (Python)          src/stardust/special/__init__.py  (_core.special 재노출)
      │
bind/special_bind.cpp        함수 등록, 에러 메시지 이름 (bind/errors.hpp)
      │
csrc/special/                ─────▶ array/ (NdIter, n항 디스패처, 버퍼 변환, 리덕션 드라이버)
                             ─────▶ parallel/ (풀), accelerate/ (vForce, §3.4)
```
- `special/`은 `array/`를 쓴다. `array/`는 `special/`을 쓰지 않는다. 같은 x의 sin과 cos를 한 번에 구하는 `base/sincos.hpp`는 의존이 없어 바닥 층(`base/`)에 두고 두 영역이 함께 쓴다 (ARCHITECTURE §1).
- 파일

  | 파일 | 내용 |
  |---|---|
  | `special/common.hpp` | 상수(π, √π, 오일러 상수, ε), 다항식·유리식 평가(Horner, Estrin), 계수 표 형식 |
  | `special/cf.hpp` | 연분수(수정 Lentz), 급수 합의 공용 반복기와 수렴 판정 |
  | `special/prefix.hpp` | `x^a e^{-x} / Γ(a)`, `x^a (1-x)^b / B(a,b)` 같은 앞 인자를 Lanczos로 계산 (§5.4) |
  | `special/ops/{erf,gamma,incgamma,incbeta,logistic,zeta,expint,bessel01,bessel,airy,ellip,hyper,lambertw,wofz,loggamma,distr,noncentral,testing}.hpp` | 계열마다 Op 구조체 (스칼라 정의) |
  | `special/kernels/simd.hpp`, `special/kernels/chunked.hpp` | Q5의 NEON 커널: 레인 둘과 스칼라가 함께 쓰는 연산(`simd`), 입력 하나·둘의 1024원소 청크 루프(`chunked`, Q57) |
  | `special/gamma_core.hpp`, `special/dd.hpp`, `special/halley.hpp`, `special/cplx.hpp`, `special/boost.hpp`, `base/sincos.hpp`(array와 함께 씀) | 감마 계열의 바닥(정확한 인자, `sinpi`, 가수·지수 분리, float64 Lanczos `gamma_plain`), double-double, 역함수의 Halley, 복소수 보조 함수, Boost.Math 호출(§5.14), 같은 인자의 sin·cos(§1.3) |
  | `special/reduce.cpp` | `logsumexp`, `softmax`, `log_softmax` |
  | `special/coeffs/*.hpp` | 가져온 계수 표. 파일 머리에 출처와 라이선스 |
  | `special/cost.hpp` | 연산마다 원소당 비용과 가중치 (§3.3) |
  | `special/cost_table.hpp` | 함수마다 잰 원소당 비용. `python -m bench.speed cost`가 만든다 (§3.3) |
- 테스트 `tests/special/`, 벤치마크 `bench/special/`, 기준값 생성 `tools/special_ref.py` (§6)
- **측정용 비공개 함수** (사용자 결정 2026-11-20): 엔진 시험 함수(`_fma3`, `_sum4`, `_powers4`, `_cf_tan`)는 `stardust._core.special`에 그대로 둔다. `sd.special`(`__all__`)에는 없다. 측정(`bench/speed/cases/special.py`)과 엔진 테스트가 같은 빌드에서 부르기 때문이다. 무게 표(`_weight`)에도 들어간다.
- **바인딩의 시그니처** (사용자 결정 2026-11-20): 모든 함수에 docs/api/special.md와 같은 시그니처를 붙인다(`nb::sig`, 원소별 함수는 `Op::args`에서 만든다: `erf(x, /, *, out=None)`). 입력은 위치 전용 그대로이고 이름은 `help()`와 nanobind의 호출 에러에 보인다. 호출 모양 에러는 nanobind의 꼴 그대로다(API.md §2.1의 예외).

---

## 2. 의미론 규칙

### 2.1 입력과 dtype
- 피연산자는 `sd.ndarray`와 Python 스칼라다 (docs/api/array.md §6.2).
- bool, int64는 float64로 계산한다 (버퍼 변환, array.md §4.3). 결과는 float64다.
- complex128은 `TypeError`다. 마지막 마일스톤(S8) 뒤에는 §1.2의 복소수 함수만 complex를 받는다.
- 모두 Python 스칼라이면 결과는 Python `float`다.
- **정수 차수 인자**(`expn`의 `n`, `polygamma`의 `n`, Q17): int64와 float64를 받는다. 값이 정수가 아니거나 음수이면 그 원소는 NaN이다. scipy처럼 잘라 쓰지 않는다 (§2.2).

### 2.2 특수값
- 특수값(±0, ±inf, NaN, 극점, 정의역 경계)은 **scipy를 따른다.** 예: `gamma(0.0)`은 `inf`, `gamma(-0.0)`은 `-inf`, `gamma(-1.0)`은 `nan`, `rgamma(-1.0)`은 `0.0`, `logit(0.0)`은 `-inf`
- 다음 경우는 scipy를 따르지 않고 수학적 값을 쓴다. 해당 마일스톤을 시작하기 전에 목록을 사용자와 확정한다.
  - scipy가 실수로 표현할 수 없는 값에 `inf`를 낸다. 예: `hyp2f1(0.5, 0.5, 1.5, 2.0)`의 참값은 복소수인데 scipy는 `inf`다. 우리는 NaN이다.
  - scipy가 경고와 함께 인자를 바꾼다. 예: `expn(1.5, 1.0)`은 n을 1로 잘라 `E₁(1)`을 낸다. 우리는 NaN이다 (Q17).
- 함수별 특수값 표는 docs/api/special.md의 각 절에 둔다.

### 2.3 정확도 계약
- **보장 영역**: 함수마다 docs/api/special.md에 적는다. 그 안에서 참값(Arb, §6) 대비 상대오차 ≤ 1e-14다.
  - 1e-14는 2⁻⁵² 단위로 약 45이고, 두 값 사이의 double 수(ULP, `ulp_distance`)로는 이진 자릿수 안의 위치에 따라 45–90이다. 실제 목표는 그보다 작고, 잰 최대 ULP를 표(bench/special/accuracy.md)로 남긴다.
  - 표의 ULP는 `ulp_distance`라 정수다. 이 문서에서 소수로 적은 ULP 값은 참값과의 차를 그 자리의 ULP 간격으로 나눈 값이다.
  - **진동 함수**(`j0 j1 y0 y1 jv yv jve yve`, x < 0의 `airy`, Q18): 영점 근처에서는 상대오차를 정의할 수 없다. 그래서 `|오차| ≤ 1e-14 × A(x)`를 보장한다. `A(x)`는 포락선(진폭)이다. 예: Hankel 점근의 진폭 `√(2/(πx))`, Airy는 `π^{-1/2}|x|^{-1/4}`. 포락선의 정확한 식은 함수마다 docs/api/special.md에 적는다.
- **느슨한 영역**: 보장 영역 밖이지만 정의역 안이다. 값을 계산하고, 잰 최대 오차를 문서에 적는다. 값을 낼 수 없는 곳(반복 상한, 손실을 아는 식이 없는 곳)은 NaN이고 그 범위를 함수별 절에 적는다 (Q9).
- **오버플로·언더플로**: 참값이 float64 범위를 넘으면 ±inf, 서브노멀 아래면 0이다. 결과가 서브노멀이면 상대오차 대신 절대오차 2⁻¹⁰⁷⁴·45를 본다.
- 테스트의 허용치는 이 계약에서 나온다. scipy와의 차이는 허용치가 아니라 보고 대상이다 (§6).

### 2.4 scipy와 다르게 동작하는 곳
| scipy | sd.special | 이유 |
|---|---|---|
| 리스트, numpy 배열, numpy 스칼라를 받는다 | `sd.ndarray`와 Python 스칼라만 (numpy 스칼라도 받는다) | docs/api/array.md §6.2 |
| `out`을 위치 인자로도 받는다 | keyword-only | docs/api/array.md §1.1 |
| complex 입력을 대부분 받는다 | 실수만 (S8 뒤 일부) | Q6, Q7 |
| `logsumexp(a, axis, b, keepdims, return_sign)` | `logsumexp(x, /, *, axis=None, keepdims=False)` | Q13 |
| `erfc`, `ndtr`, `log_ndtr`의 서브노멀 꼬리가 0(−0.0) | 참값(서브노멀) | 서브노멀 규칙(§2.3), 사용자 결정 (S2, 2026-11-13) |
| `erfcinv(5e-324)`가 inf | 27.21 | z/2의 언더플로, 사용자 결정 (S2) |
| `erfinv(-0.0)`이 0.0 | −0.0 | 홀함수의 부호, 사용자 결정 (S2) |
| gamma·rgamma의 서브노멀 꼬리와 깊은 음수의 부호, `gammaln(±5e-324)`·`betaln(5e-324, 1)` = inf, `rgamma(-5e-324)` = −0.0, beta·betaln의 NaN 입력 → inf, `zeta(inf, q>1)` = NaN | 참값 | 사용자 결정 (S3, 2026-11-14). 같은 원칙으로 `beta(inf, 1e300)`, 거대한 k의 `binom` |
| `binom(n, k)`가 음의 정수 n에서 NaN | 정수 k ≥ 0이면 n(n−1)…(n−k+1)/k! | 사용자 결정 (S3) |
| `expi(x)`가 709.78 < x < 716.4에서 inf, `expn(n, x)`의 서브노멀 꼬리가 0, `expn`의 n이 int32를 넘으면 NaN | 참값 | 사용자 결정 (S5, 2026-11-14) |
| `expn(inf, x)`가 1.7e-10 | NaN (Q17: 정수가 아니다) | 사용자 결정 (S5) |
| `hyp1f1(-1, 2, inf)` = inf, `hyp1f1(0, 0, x)` = inf, `hyp2f1(a, b, c, -inf)` = NaN | 참값·극한 | 사용자 결정 (S7, 2026-11-14) |
| `lambertw`(−1/e에 가장 가까운 double) = NaN | −1 (분기점으로 본다) | 사용자 결정 (S7, Boost와 같다) |
| `erf(1e300+1e300j)` = NaN, `loggamma`의 무한대 인자 NaN, `lambertw(5e-324, k≠0)` = inf + nan·i, `lambertw(infj)` 등 NaN, `lambertw`(−1/e에 가장 가까운 double + 0j) = NaN | 참값·극한 | 사용자 결정 (S8, 2026-11-14) |
| `loggamma`의 극점이 nan + nan·i | inf + nan·i (허수부는 방향마다 다르다) | 사용자 결정 (S8) |
| `lambertw(x − 0j, k=-1)`이 −1/e < x < 0에서 실수(위쪽 값) | 아래쪽 극한(가지 자름의 ±0j는 C99식) | 사용자 결정 (S8) |
| 결과가 오버플로하는 곳의 ±inf 부호(반올림한 위상), `erf(5e-324+30j)` = inf | 정확한 위상의 부호, 참값 | 같은 원칙 (S8) |
| 실수축·허수축 위 0인 부분의 부호가 일관되지 않다(`erfc(0+0j)`은 −0, `erfc(0.5+0j)`은 +0) | 도함수의 방향 | S8 (docs/api/special.md §12) |
| Bessel·Airy의 ±inf 인자에서 NaN, `i0`·`i1`의 713–713.98 inf, `k0`·`k1`의 5e-324, 차수 극한(`yv(inf, x)` 등)의 NaN과 `yve`의 부호, 음의 차수 x = 0의 `iv(-1.5, 0)`·`yv(-½, 0)`·`yv(-1.5, 0)`, e-스케일 판의 x = 0 NaN, `airy(105)` | 참값 | 사용자 결정 (S6, 2026-11-14). `k0(5e-324)`는 같은 원칙 |
| `gammainc(1e-300, 5e-324)` > 1, `gammainc(5e-324, x)` = 0, `gammainc(1, 5e-324)` = 0, `betainc(2, 1e300, 5e-324)` = NaN, `gammaincinv(5e-324, 0.5)`·`betaincinv(2, 3, 5e-324)`·`betainccinv(2, 3, 5e-324)` = NaN, 근이 언더플로하는 `betaincinv`가 DBL_MIN | 참값 | 사용자 결정 (S4, 2026-11-14) |
| `expit(x)`가 x < −709.78에서 0 | 참값 eˣ(서브노멀), x < −745.13에서 0 | 서브노멀 규칙(§2.3), 사용자 결정 (S1, 2026-11-13) |
| `lambertw`는 언제나 complex를 돌려주고 `tol=`이 있다 | 실수 입력이면 float64. `tol=` 없음 | Q6, 결정성 (§5.11) |
| `airy`는 네 배열의 tuple | `AiryResult(ai, aip, bi, bip)` namedtuple | DESIGN §9.7 |
| 경고(`SpecialFunctionWarning`) | 내지 않는다 | API.md §2 |
| 별칭(`psi`, `jn`) | 없다 | Q12 |
| `bdtr(k<0)` NaN, `bdtr(k>n)` NaN, `pdtr(k<0)` NaN, `nbdtr(k, 0, p)` NaN | 참값 0, 1, 0, 1 | 참값·극한, 사용자 결정 (T1, 2026-11-15) |
| `bdtr`, `nbdtr`의 n을 내린다 | `bdtr`은 정수가 아니면 NaN, `nbdtr`은 실수 n | 사용자 결정 (T1) |
| `fdtr`, `fdtri`의 무한 자유도가 NaN | 극한 분포(카이제곱, 1의 점질량) | 참값·극한 (T1) |
| `stdtrit(3, 5e-324)`가 inf | −6.07e107(참값) | 참값 (T1) |
| `chndtr(x, df, nc)`, `chndtrix(p, df, nc)`: x가 맨 앞 | `chndtr(df, nc, x)`, `chndtrix(df, nc, p)`: x가 맨 뒤 | 다른 분포 함수와 일관성, 사용자 결정 (T2, 2026-11-15) |
| 비중심 분포의 위꼬리 함수가 없다 | `chndtrc`, `nctdtrc`, `ncfdtrc` | 꼬리의 상대 정확도, 사용자 결정 (T2) |
| `chndtr(0, nc, x)`, 무한 자유도·무한 nc의 `chndtr`·`nctdtr`·`ncfdtr`가 NaN | 참값·극한(0의 점질량, 극한 분포) | 참값·극한 (T2) |
| `betaincinv(inf, b, p)`, `betaincinv(a, inf, p)`(0 < p < 1)가 NaN | 1, 0 (`betainccinv`도) | `betainc(inf, b, x) = 0`과 같은 극한, 사용자 결정 (2026-11-20) |
| 비중심 분포의 꼬리가 0이나 틀린 값(최대 8.8e15 ULP), δ와 반대쪽 t 꼬리가 무의미한 값 | 참값 | 참값 (T2) |

---

## 3. 엔진

### 3.1 n항 디스패처 (배열 영역에 더한다, Q11)
- 지금 배열에는 `unary<Op>`, `binary<Op>`만 있다 (`array/ops/ufunc.hpp`). `where`는 자기 경로를 갖는다.
- 더하는 것: `nary<Op, N>(operands..., out)`. 입력 N개(1–4)의 디스패치 네 단계(피연산자, 루프, 변환, 실행)를 그대로 한다 (array.md §4.4).
  - `unary`, `binary`는 손대지 않는다. 기존 함수의 결과와 경로가 비트까지 같다 (골든 출력으로 확인).
  - `kMaxOperands` 4 → 5. `hyp2f1(a, b, c, z) → out`이 5개다. `NdIter`, `BufferedLoop`의 고정 배열이 하나씩 커진다. 이 비용(스택 몇십 바이트, 루프 하나)을 S0에서 잰다.
- 출력이 여럿인 원소별 함수(`airy`의 넷)는 출력 4 + 입력 1 = 5개다. 같은 디스패처가 출력 개수를 받는다.
- **S0에서 정한 모양**
  - Op 구조체가 `nin`, `nout`, 인자 이름 `args`를 갖는다. `f`는 `nout == 1`이면 값 하나, 아니면 `std::array`를 돌려준다.
  - 출력이 여럿이면 결과는 tuple이다(스칼라 입력이면 float의 tuple). `out=`은 없다.
  - 루프가 없는 dtype의 메시지는 입력 dtype을 모두 적는다. 셋 이상이면 `float64, complex128 and float64`처럼 쉼표로 잇는다 (둘일 때는 전과 같다).
  - `kMaxOperands` 5의 비용: 짧은 원소별 호출에서 행마다 ±5% 안팎으로 흔들리고 일관된 회귀는 보이지 않았다 (커밋 4151d2a의 bench/special/s0/operands.md).

### 3.2 Op 구조체
- 배열의 `scalar_ops.hpp`와 같은 형태다: 루프가 있는 저장 타입(`ok<T>`, 여기서는 `double`만)과 원소 하나의 함수(`f`).
- 원소 함수는 `static double f(double...)`이다. 순수 함수다(전역 상태 없음, `errno` 안 봄).
- 이 형태라서 `elementwise.hpp`의 연속 fast path(array.md §5.1)가 그대로 붙는다.
- 스칼라 호출(입력이 모두 Python float, `out=` 없음, 출력 하나)은 바인딩이 run_nary를 거치지 않고 같은 float64 루프를 원소 하나에 부른다(Q58). 결과의 비트는 run_nary와 같다.

### 3.3 병렬화: 원소당 비용
- 함수마다 원소당 비용이 100배 넘게 다르다 (xlogy 2.5ns, betaincinv 744ns, scipy 1스레드 기준).
- 그래서 배열의 기준값(`kParallelThreshold`, 원소 2¹⁵개)을 그대로 쓰지 않는다. 함수마다 원소당 비용 `c`(ns)를 `special/cost.hpp`에 두고, 일의 양을 `n·c`로 센다. linalg의 비용 모델(linalg.md §7)과 같은 생각이다.
  - 배열의 이터레이터는 이미 작업량 가중치를 받는다 (레인 처리, array.md §4.5). 같은 장치를 쓴다.
  - S4에서 더함: 가중치의 상한 `kMaxWeight` = 256. 이터레이터는 태스크를 4096/가중치 원소로 자르는데, 가중치 800–8500인 불완전 함수는 태스크 하나가 원소 하나가 되어 10스레드가 4.8–7.8배에 그쳤다. 상한 256(태스크 16원소)에서 8.1–8.9배다 (커밋 4151d2a의 bench/special/s4/threads.md).
  - `c`는 함수마다 원소 10⁶개, 1스레드로 잰다. 입력은 참값 파일의 입력(보장 영역)에서 뽑는다. 잰 값은 special 속도 기준선(`bench/speed/results/special.json`)에 있고, `python -m bench.speed cost`가 그것으로 `special/cost_table.hpp`를 만든다(bench/README.md). `cost.hpp`는 연산마다 그 표의 어느 함수 값을 쓰는지만 적는다. 표에 없는 이름은 컴파일되지 않는다.
  - 표의 처음 값은 각 마일스톤에서 잰 손 상수를 옮긴 것이라 §5의 속도 개선 뒤의 값이 아니다. special 기준선을 재고 `cost`를 다시 돌리면 바뀐다. 2026-11-22의 개선 뒤 크게 어긋난 줄: `chndtr chndtrc chndtrix ncfdtr ncfdtrc ncfdtri nctdtr nctdtrc nctdtrit`(지금 값의 약 2.5–5배; Q48·Q51–Q54), `beta`, `binom`(약 2배; Q56). 48.6 ns(가중치 256) 이상인 값은 상한에 걸려 분할이 같으므로, 무거운 함수의 값이 달라져도 영향이 없다.
  - 가중치의 단위는 연속 float64 `add`의 원소당 비용이다 (`kAddNs`, 표의 `_add` 줄, 1스레드에서 0.18–0.19ns). 가중치 `w`인 함수는 원소 `2¹⁵/w`개부터 병렬이 되고, 태스크 하나는 원소 약 `2¹²/w`개다.
  - **GIL은 상한 전의 비용으로 정한다** (API.md §3): 상한은 태스크 크기를 위한 것이라, 그것으로 GIL까지 정하면 무거운 함수의 원소 128개 미만 호출이 GIL을 쥔 채 몇 초를 돌았다(`jv`의 느린 원소 127개에 6초). 그래서 바인딩(`bind/nary_bind.hpp`, `weight_for`)이 브로드캐스트 모양의 원소 수 n을 보고, 상한 걸린 가중치로는 기준 아래인데 상한 전의 비용 `c/kAddNs·n`이 2¹⁵ 이상이면 기준에 딱 닿는 가중치 ⌈2¹⁵/n⌉을 넘긴다. 그 호출은 GIL 없이 원소 약 n/8개의 태스크로 돈다. 결과 비트는 그대로다. 비용 표의 값은 전형적인 입력의 비용이라, 한 원소만 유난히 느린 입력(예: 스칼라 하나)은 그대로 GIL을 쥔다. stats의 분포 메서드도 같은 바인딩이다.
- 원소마다 반복 횟수가 다르다(연분수, Halley 반복). 정적 분할이면 스레드마다 일이 달라진다. 풀의 청크 분할(array.md §10)이 이것을 얼마나 흡수하는지 S4에서 잰다.

### 3.4 NEON (Q5)
- 대상(청크 커널, `Op::chunk`): `erf erfc ndtr log_ndtr ndtri erfinv erfcinv`(`ops/erf.hpp`), `expit logit log_expit`(`ops/logistic.hpp`), `ellipk ellipe`(`ops/ellip.hpp`, Q31). `erfcx`와 `gammaln`은 스칼라 libm 정의다.
  - `log_expit`: NEON으로 −|x|, `vvexp`, `vvlog1p`(e ≥ +0이라 vForce log1p의 −0 문제가 없다), NEON으로 x ≥ 0이면 −l, 아니면 x − l.
  - `ellipk`, `ellipe`: NEON으로 y = 1 − m(m < 0이면 1/(1 − m)), `vvlog`, 다항식은 레인(`simd::horner`, 스칼라와 같은 반올림). 특수값(NaN, m ≥ 1, −inf)은 청크 뒤에 원소마다 고친다. 스칼라 정의(`f`)는 없다.
  - 리덕션의 exp(x − m)(`reduce.cpp`의 `ExpDiff`, 입력 둘의 `chunked_loop2`, Q57): 레인에서 TwoSum, array/kernels/vmath.hpp의 NEON exp(vForce가 아니다; 레인과 홀수 끝 원소가 같은 연산이라 위치와 무관하다), 레인에서 e(1 + lo).
- 기법
  - 다항식 평가: Horner(곱셈-덧셈이 차례로 이어짐)와 Estrin(트리, 병렬도 높음)을 비교한다. `-ffp-contract=off`이므로 FMA는 코드에 적은 곳에서만 생긴다 (array.md §1.3). Horner는 단계마다 `fma(r, x, c)` 하나다(곱셈과 덧셈을 따로 하면 반올림이 하나 더 있고 의존 사슬이 약 1.6배 길다). 레인(`vfmaq_f64`)과 스칼라(`std::fma`)가 같은 헬퍼라 비트가 같다. 스칼라 정의도 같은 식을 써야 비트가 같다.
  - 분기 없애기: 두 구간을 모두 계산한 뒤 `vbslq`로 고른다. 구간이 셋이면 셋 다 계산하는 비용과 분기 비용을 잰다.
  - S3: `gammaln`은 NEON으로 짜지 않았다. 구간이 넷(작은 x, Taylor, 점화, Stirling)이고 S2의 측정(분기 방식이 빠르거나 같음)으로 보아 레인 둘의 이득이 없다. 스칼라 libm 정의다.
  - **S2의 측정 결과** (커밋 4151d2a의 bench/special/s2/variants.md): 차수 5–8의 유리식이 셋인 `erfc`와 `ndtri`에서는 원소마다 자기 구간만 계산하는 스칼라 분기가 NEON 고르기보다 빠르거나 같았다(구간이 무작위로 섞인 입력만 예외, 4.7 대 4.2 ns). 고르기용 상수와 남의 구간 계산이 레인 두 개의 이득보다 크다. 그래서 S2 함수는 1단계(exp·log의 인자)만 NEON, 2단계는 vForce, 3단계는 스칼라 분기다. 결과의 비트는 두 방식이 같다. Estrin은 Horner와 속도가 같고 오차가 조금 컸다(9.1 대 7.8 ULP).
- **안의 `exp`/`log`** (Q16): 커널이 입력을 1024원소(8KB) 스택 버퍼 단위로 처리한다.
  1. NEON으로 `exp`/`log`의 인자(예: `-x²`)를 버퍼에 만든다.
  2. vForce(`vvexp`, `vvlog`)를 버퍼 전체에 한 번 부른다.
  3. NEON으로 다항식을 계산하고 결과를 합친다.
  - 스칼라 입력과 비연속 레이아웃도 같은 버퍼 경로를 탄다 (n = 1 포함). 그래서 `f(x)[i] == f(x[i])`가 성립한다. 배열의 vForce 규칙과 같다 (array.md §5.4).
  - vForce의 오차(libm 대비 ≤ 6 ULP)는 정확도 계약(상대 1e-14) 안이다. 특수값(±0, ±inf, NaN)이 vForce에서 부호까지 맞는지는 S1에서 그 함수들에 대해 다시 확인한다.
  - 분기가 많은 나머지 함수는 원소마다 libm `exp`/`log`를 부른다.
- 그 밖의 함수는 손으로 짜지 않는다. 연속 fast path의 자동 벡터화와 스레드만 쓴다.

### 3.5 리덕션 (`logsumexp`, `softmax`, `log_softmax`)
- 정의: `logsumexp(x) = m + log(Σ exp(x - m))`, `m = max(x)`
- 두 패스로 한다: 1) `max` 2) `exp(x - m)`의 pairwise 합 (base/pairwise.hpp). 배열의 리덕션 드라이버와 블록 분할(array.md §6.5)을 쓴다. 결과는 스레드 수와 무관하다.
  - `axis`는 배열 리덕션의 규칙 그대로다(`sd.max`, `sd.sum`에 넘긴다).
  - `logsumexp`는 `max`를 두 번 부른다: `x`와 브로드캐스트할 `keepdims=True`의 m, 결과에 더할 요청한 `keepdims`의 m.
  - `x - m`은 TwoSum으로 hi + lo를 만들고 `exp(hi)·(1 + lo)`로 계산한다. 빼기를 반올림하면 exp에서 |x − m|·2⁻⁵³(−700에서 7.8e-14)을 잃는다. 이 단계는 NEON 청크 커널이다(§3.4, Q57). 나눗셈(`softmax`)과 log_softmax의 결합은 원소별 스칼라 루프 그대로다.
- 한 패스 방식(온라인 softmax: 최댓값이 바뀔 때마다 누적값을 다시 척도 조정)은 메모리 왕복이 한 번 줄지만 반올림이 늘어난다. 두 방식을 해설에서 비교하고, 공개 함수는 두 패스를 쓴다.
- `m`이 ±inf일 때: 모두 `-inf`이면 결과는 `-inf`, `+inf`가 있으면 `+inf`, NaN은 전파한다 (scipy와 같다).
- `softmax`는 `exp(x - logsumexp(x))`가 아니라 `exp(x - m) / Σ`로 계산한다. 나눗셈 하나가 log·exp 왕복보다 정확하다.

---

## 4. 수치 기법의 선

### 4.1 근사 계수 (Q4)
- 유리 근사와 Chebyshev 계수는 공개된 것을 가져온다. 출처: Cephes(scipy가 BSD로 포함), Boost.Math(Boost Software License), 논문(AS241 등)
- `special/coeffs/*.hpp`의 머리에 출처, 판, 라이선스, 원 논문을 적는다. 계수는 원본의 십진 표기를 그대로 옮긴다(비트 확인 테스트).
- 해설은 계수가 어떻게 만들어졌는지(minimax, Remez 교대 정리)를 설명하되 직접 만들지는 않는다.

### 4.2 구간 한도 (Q8)
- **세는 것**: 근사 방식이 바뀌는 갈래. 예: 거듭제곱급수 ↔ 연분수 ↔ 점근전개, 다른 계수 표로 넘어가는 경계
- **세지 않는 것**: 정의역을 옮기는 변환. 예: 반사(`Γ(x)Γ(1-x) = π/sin πx`), 점화식(`Γ(x+1) = xΓ(x)`), 대칭(`I_x(a,b) = 1 - I_{1-x}(b,a)`), Kummer·Pfaff 변환, hyp2f1의 선형 변환
- **퇴화 경우**: 일반 공식이 0/0이 되는 매개변수(예: hyp2f1의 `c-a-b` 정수)는 극한 공식 하나로 처리한다. 정수 거리마다 따로 짜지 않는다.
- **가져오지 않는 것**: Boost·Cephes가 극단 영역마다 둔 특수 분기(예: DiDonato–Morris BRATIO의 십여 갈래), 상수 미세 튜닝, 매개변수 범위마다의 fast path

### 4.3 공통 기법 (해설의 주제)
- 보수함수를 따로 계산한다: `erfc`, `gammaincc`, `betaincc`는 `1 - f`가 아니다. 꼬리에서 상대 정확도가 유지된다.
- log 공간: `gammaln`, `betaln`, 앞 인자의 log 형태로 오버플로와 소거를 피한다.
- Lentz 연분수: 분모가 0에 가까울 때를 작은 수로 바꿔 계속한다. `b0 = 0`이면 그 작은 수(2⁻¹⁰⁰⁰)가 결과의 척도가 되므로, 꼬리를 따로 계산해 나눈다 (S0에서 `_cf_tan`의 5e-324가 9e-302로 나와 찾았다).
- 역함수: 좋은 초기값 + Halley(또는 Newton). 도함수가 앞 인자(§5.4)라서 거의 공짜다.
- 정확한 빼기: `2p - 1`(p ∈ [0.25, 1]에서 Sterbenz 보조정리로 정확), `x²`를 두 부분으로 나눠 `exp(-x²)`의 오차 줄이기

---

## 5. 함수별 계획 (후보와 구간)

구간 수는 Q8로 센 수다. 후보가 둘이면 §1.3의 규칙으로 고른다.

### 5.1 오차함수
- `erf`, `erfc`: Cephes `ndtr.c`의 세 구간. `|x| < 1`은 erf 유리식, `1 ≤ x < 8`과 `x ≥ 8`은 `exp(-x²)·P/Q`, `exp(-x²)·R/S`. 음수는 대칭. `exp(-x²)`는 `x²`를 나눠 계산한다 (Cephes `expx2`). **3구간**
- `erfcx(x) = exp(x²) erfc(x)`: 같은 유리식에서 `exp(-x²)`만 빼면 된다(`x ≥ 1`). `x < 1`은 `exp(x²)·erfc(x)` 그대로, 큰 x는 점근 `1/(x√π)`. 음수는 `2exp(x²) - erfcx(-x)` (변환). **3구간**. Faddeeva 패키지의 100구간 Chebyshev는 쓰지 않는다.
- `ndtri`: Wichura AS241 (PPND16). 중심 `|q| ≤ 0.425`의 유리식, 꼬리 `r = √(-log p)`의 유리식 둘. 정확도 약 1e-16. **3구간**
- `erfinv(y)`, `erfcinv(z)`: ndtri의 핵심을 `q`로 직접 부른다. `erfinv(y)`는 `q = y/2`(정확)로 중심식을 써서 작은 y에서 `(1+y)/2`의 소거가 없다. `erfcinv(z) = -ndtri(z/2)/√2`
- `ndtr(x) = erfc(-x/√2)/2`. `log_ndtr`: `x > 6`은 `log1p(-ndtr(-x))`, 중간은 `log(ndtr(x))`, `x < -20`은 점근급수. **3구간**
- **S2에서 바꾼 것**
  - `exp(-x²)`는 Cephes `expx2`의 지수 두 번 대신 `hi = x·x`, `lo = fma(x, x, −hi)`로 `exp(−hi)·(1 − lo)`다. `ndtr`은 `t = x/√2`의 반올림을 피해 `exp(−x²/2)`를 x에서 직접 만든다.
  - `erfcx`는 `x ≥ 8`을 Cephes R/S 대신 점근급수(17항)로 한다. R/S는 x = 26.64까지 맞춘 식이라 x ≈ 10⁴에서 21 ULP가 났다. 구간은 셋(x < 1, P/Q, 점근)이다.
  - `log_ndtr`는 두 구간이다: `x ≥ 0`은 `log1p(−ndtr(−x))`, `x < 0`은 `−x²/2 + log(erfcx(−x/√2)/2)`. 두 항의 부호가 같고 erfcx는 언더플로하지 않아서 점근급수가 필요 없다.
  - `erfinv`의 중심 구간은 `y·(비율/(2√2))`다. `q = y/2`를 먼저 만들면 서브노멀 y의 마지막 비트가 사라진다. `erfcinv`의 꼬리는 `log(min(z, 2−z)) − log 2`다.

### 5.2 감마
- `gamma`: 후보 A Lanczos(Boost `lanczos13m53`, g ≈ 6.0247, 13항), 후보 B Cephes(구간 [2, 3]의 유리식 + 점화식, x > 33은 Stirling). `x < 0.5`는 반사. 오버플로는 x > 171.62
- `gammaln`: 1과 2 근처에서 값이 0이 되어 상대 정확도가 어렵다. Cephes `lgam`(x < 13은 [2, 3]으로 옮겨 `(x-2)·P/Q`, 그 위는 Stirling급수)이 이 문제를 식의 모양으로 푼다. **2구간**. 음수는 반사(`log|Γ|`)
- `rgamma = 1/Γ`: 극점에서 0. 반사와 Lanczos
- `digamma`: `x < 0` 반사(`π cot πx`), 작은 x는 점화식으로 올린 뒤 Bernoulli 점근급수, 양의 근 x₀ ≈ 1.4616 근처는 근을 중심으로 한 급수(상대 정확도). **2구간**
- `polygamma(n, x)`: `n = 0`은 digamma, `n ≥ 1`은 `(-1)^{n+1} n! ζ(n+1, x)` (§5.6의 Hurwitz zeta)
- `beta`, `betaln`: Lanczos 비(Boost `beta`)로 세 Γ를 한 식에 모아 오버플로와 소거를 피한다. 큰 인자는 log 형태
- `binom(n, k)`: 정수 k가 작으면 곱(정확), 그 밖은 `1/((n+1)·B(n-k+1, k+1))`. 음의 n은 반사. 정수가 아닌 k(0 < k < n ≤ 2⁴⁰)의 B는 `beta_lanczos`에 k + 1, n − k + 1의 낮은 부분을 넘겨 계산한다(Q56)
- **S3에서 정하고 바꾼 것**
  - `gamma`: 공개 `gamma`와 `rgamma`는 2⁻³⁰ ≤ |x| ≤ 170에서 Lanczos 식을 float64로 계산한다(`gamma_plain`: zgh = z + g − ½를 hi + lo로 들고 L(z)·zgh^(z−½)·e^(−zgh), z ≥ 100은 거듭제곱을 반씩 둘로 나눠 넘치지 않게, Q24). x < 0은 반사 −π/(x·sinpi(x)·Γ(−x))다. 결과가 정상 범위가 아니면(깊은 음수의 서브노멀 꼬리) Boost, 그다음 scaled 경로다. 같은 식의 m·2ᵉ판(`gamma_scaled`)은 z > 170에서만 m·2ᵉ로 계산하고(서브노멀 꼬리, beta, 음수 zeta, polygamma), z ≤ 170이면 `gamma_plain`의 값이다. S3의 후보 B(Cephes)는 반사식의 m·2ᵉ 경로(`gamma_pos_scaled`), `rgamma`의 170 < x ≤ 171.6, `prefix.hpp`의 작은 감마에 남았다.
  - 양수 x가 먼저다: 함수 안에는 이 경로만 두고, 특수값·음수·넘침은 따로 뗀 함수에서 본다(Q33).
  - 정확한 인자: 반사식은 `Γ(x)Γ(−x) = −π/(x·sinpi(x))`, `Γ(x+1) = x·Γ(x)`. 꼭 만들어야 하는 합(a+b, n−k+1)은 TwoSum으로 hi + lo를 들고 `Γ(hi+lo) ≈ Γ(hi)(1 + ψ(hi)·lo)`. `sinpi`·`cospi`는 인자를 정확히 줄인다(Apple에서는 libm의 `__sinpi`, `__cospi`, Q25).
  - m·2ᵉ(`Scaled`): 정상 범위의 수는 비트로 가수와 지수를 나누고 붙인다. `exp_scaled`는 |x| < 700이면 libm `exp`, `pow_scaled`는 결과가 정상 범위이면 libm `pow`이고, 그 밖에서만 2ⁿ과 조각으로 나눈다(Q34).
  - `gammaln`(x > 0): 2 둘레의 Taylor 전개(계수는 ζ(k)−1, 28항)와 Stirling 두 근사. 1 근처는 `lgamma(2+e) − log1p(e)`. Cephes의 [2, 3] 유리식은 쓰지 않는다. **2구간**
  - `beta`: a, b > 0이고 a + b < 40이면 Boost(§5.14). 큰 쪽 a가 2²⁰·max(1, b²) 이상(0 < b < 171)이면 Γ(b)·a^(−b)·(1 + G₁/a + G₂/a²)(DLMF 5.11.13, `beta_far`, Q32). 그 밖의 양수 인자(a ≤ 2⁴⁰)는 세 Lanczos 식을 한 식에 모은 L(a)L(b)/L(c)·e^(−(g−½))·cgh^(−½)·(agh/cgh)^(a−½)·(bgh/cgh)^(b−½)를 float64로 계산한다(`beta_lanczos`, Q56; 밑은 나머지로 보정한 몫). 값이 정상 범위가 아니면(언더플로) 옛 경로다: c ≤ 170이면 `gamma_plain` 셋, 그 위는 m·2ᵉ의 Γ 비율 `beta_scaled`, 한 인자가 7000을 넘는 scaled 경로는 Γ 비율 `L(a)/L(c)·e^E·cgh^{−b}`(E가 작게 남도록 정리)다
  - `zeta`: Riemann x ≥ 0은 Euler–Maclaurin(q = 1), x < 0은 함수방정식(scaled). Hurwitz는 Cephes `zeta.c`. q^{−s}가 언더플로하면 scaled로 더한다(polygamma). 공개 Riemann `zeta`는 x ≥ −1과 음의 홀수 정수 −3, …, −59(Boost가 Bernoulli 수의 표에서 읽는다)에서 Boost를 먼저 부르고, 2–52의 짝수는 올바르게 반올림한 표, 음의 짝수는 0이다(§5.14). 정수가 아닌 −60 < x < −1은 우리 함수방정식을 double로 계산한다(`zeta_detail::reflect`: 1 − x를 반올림하지 않고 넣고, 두 지수 함수를 하나로 묶어 그 반올림을 1차 보정으로 붙인다). Boost는 그 구간에서 1 − x를 반올림한 뒤 Γ에 넣고 반올림한 2π를 거듭제곱해서 x ∈ (−32, −31)에서 65 ULP였다. x ≤ −60은 인수마다 scaled로 계산한 함수방정식이다. x ≥ 0의 Euler–Maclaurin은 Boost의 결과가 정상 범위가 아닐 때만 쓴다. 함수 안에는 Boost를 먼저 부르는 x > 1(표의 짝수 제외)과 −1 ≤ x < 1(|x| ≥ 2⁻³⁰)만 두고 나머지는 따로 뗀 함수에서 본다(Q33).

### 5.3 불완전 감마 (`gammainc`, `gammaincc`)
- P와 Q 가운데 작은 쪽을 계산하고 다른 쪽은 1에서 뺀다(`igam`). 갈래는 다음 순서로 고른다.
  - a = 1: P = −expm1(−x), Q = e^(−x)의 닫힌 꼴이다(Q35).
  - a = ½: P = erf(√x)(x < 0.2275), Q = erfc(√x)다(`erfc_sqrt`: e^(−x)를 x에서 만든다, Q36).
  - x < 1.1: x < ½이면 a ≤ −0.4/log x에서, x ≥ ½이면 a ≤ 0.75x나 a < 1에서 Boost `tgamma_small_upper_part` 꼴의 Q다(Q37). 나머지는 급수 P다.
  - a > 20, |x − a| < 0.4a: Temme 균등 점근전개다(급수와 연분수가 모두 느리고 소거가 생긴다).
  - x − 1/(3x) < a: 급수 P를 한 걸음에 두 항씩 더한다(Q38).
  - 그 밖: 2a가 정수이고 1 ≤ a ≤ 30이면 유한합 Q(Horner 꼴, Q36), 아니면 연분수 Q(`cf_fast`, Q39)다.
- 앞 인자 `x^a e^{-x}/Γ(a)`는 §5.4다. 그 a·x⁻¹배(x·P′(x))를 결과와 같이 돌려준다. 역함수의 Halley와 BGRAT의 r(Q43)이 그것을 쓴다.
- **로그 꼴** (sd.stats의 `gamma_detail::log_tail`, `dist_log_tail`): 작은 쪽은 그 값의 로그, 큰 쪽은 log1p(−작은 쪽)이다. 정수 a의 log Q는 유한합의 지수를 떼어 −y + log s로 낸다. a = 1은 −y 그대로이고, a ≥ 2는 log s ≤ y/2인 곳만이다(Q35). 그래서 `Poisson.logcdf(0)` = −μ, `ChiSquared(2).logccdf(x)` = −x/2가 비트까지 맞다.

### 5.4 앞 인자 (`special/prefix.hpp`)
- `x^a e^{-x}/Γ(a)`를 따로 계산하면 a가 크면 넘친다. 큰 부분을 상쇄 없는 지수 하나에 모은다(Stirling 정규화, §5.5의 S4. Boost는 같은 일을 Lanczos 꼴 `regularised_gamma_prefix`, `ibeta_power_terms`로 한다). 베타의 중간 모양은 직접 꼴이다(아래, Q42).
- 이 값은 불완전 함수의 도함수이기도 하다. 역함수의 Halley 반복이 이것을 다시 쓴다.

- **계산** (2026-11-15에 앞 인자가 `gammainc` 시간의 3/4여서 정밀도는 그대로 두고 바꿨다):
  - **표 기반 로그**: λ = 2^k·c·(1 + r), c = i/128(97점, `coeffs/dd_log_table.hpp`, `tools/gen_dd_log_table.py`), |r| ≤ 2⁻⁷·⁵⁸. log(1 + r)는 r², r³까지 hi + lo, 나머지는 float64다(급수판의 약 1/4 시간). Arb 대비 상대오차(6만 점)는 최대 log 2⁻⁷⁷·⁶(급수판 2⁻⁶⁴·⁶), mu 2⁻⁶⁸·⁹(급수판 2⁻⁶²·⁴)로 오히려 좋다. 급수판은 비정상 입력(서브노멀 등)의 뒷길로 남겼다.
  - **나눗셈**: double-double 몫은 나눗셈 두 번과 fma 하나다(세 번째 보정항은 지수에 쓸모가 없었다).
  - **a < 13의 감마 앞 인자**: Stirling 꼴을 거치지 않고 e^(a log x − x)/Γ(a+1)로 바로 구한다(지수는 double-double).
  - **지수 분리 생략**: 결과가 정상 범위(|E| < 700)이면 `Scaled`의 frexp·ldexp를 건너뛴다.
- **베타 앞 인자의 직접 꼴** (Q42, `ibeta_detail::direct_prefix`): 모양 2⁻¹⁹–2¹², x, y ≥ 2⁻⁹⁹⁹이면 B = e^L·F다. L = a log x + b log y + S(c) − S(a) − S(b)는 double-double이고 F = F(c)/(F(a)F(b))다. Γ(z) = e^S·F이고, z < 13이면 F = Γ(z)(`gamma_small`, S = 0), 그 위는 Stirling의 S와 F = √(2π)Γ*(z)다. 이 범위에서 L의 항은 1e5 안팎이라 표 로그의 2⁻⁷⁷이 L의 1e-18이다. c = a + b는 정확히 들어간다. 모양에만 달린 S와 F는 스레드마다 마지막 짝을 두는 메모다(키의 순수 함수: 적중 여부, 스레드 수와 상관없이 같은 비트). BPSER는 x^a/B(a, b)를 직접 받는다(`direct_power`). 범위 밖, |L| > 700, 결과가 2⁻¹⁰⁰⁰ 아래이면 아래의 Stirling 정규화 꼴이다.

### 5.5 불완전 베타 (`betainc`, `betaincc`)
- 구조는 DiDonato–Morris의 BRATIO(TOMS 708)다(아래 S4). a = b이고 x = ½이면 ½이다(Q44).
- 연분수(BFRAC, 감마의 Q)는 `cf_fast`다(Q39). BFRAC의 항은 나눗셈 하나로 만든다.
- 역함수(`gammaincinv`, `gammainccinv`, `betaincinv`, `betainccinv`): 작은 쪽 p 또는 1 − p를 자기 함수에 대해 Halley로 푼다(`halley_fast`, Q41). 감마는 Boost `find_inverse_gamma`의 출발점(Q40, p < 1e-300과 a ≥ 1e10은 아래 S4의 우리 초기값)이다. 베타는 AS 109이고, a, b가 Boost 구간이면 Boost의 역함수 그대로다(§5.14).
- **S4에서 정하고 바꾼 것**
  - 앞 인자: Lanczos 대신 Stirling 정규화 `Γ(z) = √(2π/z)(z/e)^z Γ*(z)`로 큰 부분을 지수 하나 E = a·μ(x/a − 1)(μ(t) = t − log(1+t))에 모았다. E는 정확한 입력에서 double-double로 만든다(`special/dd.hpp`). scipy(Cephes)는 a log x − x − lgamma(a)를 float64로 만들어 꼬리에서 수천 ULP를 잃는다.
  - 불완전 감마: 네 갈래다. x < 1.1의 작은 a는 Boost의 `tgamma_small_upper_part`(Γ(1+a) − 1과 x^a − 1을 expm1로), 급수 P, 연분수 Q, a > 20이고 |x − a| < 0.4a이면 Temme. Temme의 c_k(η) 계수는 DLMF 8.12의 정의식을 정확한 유리수로 전개해 만들었다(`tools/gen_temme_igamma.py`, 맞춘 값이 아니다).
  - 연분수는 앞으로 항 수만 찾고 값은 꼬리부터 거꾸로 계산한다. Lentz 그대로는 60항에서 17 ULP를 잃었다. 두 패스는 이제 나눗셈 없는 꼴이다(`cf_fast`: 삼항 점화식과 몫 P/Q, Q39). 계수가 아주 크면 `cf.hpp`의 `lentz_backward`로 간다.
  - 불완전 베타: NR `betacf` 대신 DiDonato–Morris의 BRATIO(TOMS 708) 구조를 따른다. BPSER(급수), BUP(유한합으로 a·b 옮기기), BGRAT(큰 a·작은 b, x ≈ 1), BFRAC(평균 아래 연분수), BASYM(a, b > 100의 평균 근처, Temme형 균등 점근전개)이다. BGRAT와 BASYM은 측정 뒤 사용자 결정으로 넣었다(2026-11-14). 근사 방식이 다섯이라 §1.1 Q8의 "2–3구간"을 넘는다.
  - 역함수: Halley는 F와 목표가 두 배 넘게 차이 나는 동안 log F에 대한 Newton 걸음을 쓴다(꼬리에서 선형 걸음은 한 번에 F/F′ ≈ 1씩만 움직인다). 감마의 우리 초기값(Boost의 출발점을 쓰지 않는 곳)은 하한 둘(작은 x 근사, 큰 x 점근식)과 Wilson–Hilferty, 베타는 AS 109다.

- **가지와 비용** (2026-11-15에 정하고 2026-11-22에 고친 것):
  - **1 < a, b이고 b0 < 40인 가지는 연분수(bfrac)다.** TOMS 708은 여기서 bup으로 b를 (0, 1]까지 내린 뒤 bpser나 bgrat을 쓴다. 이 길은 앞 인자를 세 번 만든다. 연분수는 한 번이면 되고 더 정확했다(약 2배, `bdtr` 10.1 → 4.9 ULP, `nbdtrc` 19.4 → 8.8, `fdtrc` 18.1 → 10.7, `ncfdtrc` 22.6 → 15.2).
  - **앞 인자는 쓰는 가지만 만든다.** 도함수(역함수만 쓴다)를 위해 늘 만들지 않는다. bgrat과 basym은 쓰지 않는다. BGRAT의 r은 Q를 낸 `igam`의 x·P′(x)를 1차 보정해 쓴다(Q43). BPSER는 x^a/B(a, b)를 직접 받는다(Q42).
  - **연분수의 뒤로 가는 계산**은 앞으로 갈 때 만든 항 64개를 다시 쓴다.
  - 남은 비용: a, b ≥ 100이고 평균 근처인 basym(계수 점화식이 O(n³))과 bgrat의 d_n 점화식(O(n²)).
  - a, b ≤ 6은 Boost다(§5.14). 우리 쪽이 빨라진 뒤 다시 쟀고, 참값 혼합의 그 구간에서 Boost가 여전히 빨라 그대로 둔다(Q45).

- **작은 모양과 아주 작은 x (2026-11-17, stats 최적화 중 Arb로 찾음)**: a, b ≤ 1에서 BRATIO는 x^a ≤ 0.9이면 멱급수로 I를 구하고 1 − I를 빼기로 만든다. I가 0.9에 이르면 급수 오차가 9배가 된다(우리 9.8e-15, 같은 규칙을 쓰는 Boost 6.0e-15). 그래서 I > 1/2이면 1 − I를 BRATIO의 다른 두 방법으로 직접 구한다. 급수의 첫 인자만으로 I > 1/2 + 2⁻⁴⁰임을 알면(b ≤ 1에서는 모든 항이 양수라 하한이다) 급수를 돌리지 않는다. x < 2⁻¹⁰⁰⁰은 log x를 받는 `ibeta_tiny`가 2⁻¹⁰⁰⁰에서의 값에서 양의 항만으로 내려간다(`fdtr`·`fdtrc`의 아주 작은 인자도 이것을 쓴다; 전에는 1 − i로 1.3e-13까지 틀렸다). x가 0으로 언더플로한 인자는 (0, 1)이다. 한 모양이 2⁹⁰⁰ 이상이면 감마 극한 I_x(s, c) = P(s, c·x)를 쓴다(`gamma_limit`; 전에는 `fdtr(2, 1e308, 1)`이 1.0, 참값 0.632). Boost의 값은 a, b ≤ 6, x ≥ 2⁻¹⁰²², 결과 ≥ 2⁻⁹⁶⁰일 때 쓴다: 서브노멀 x에서 Boost는 25%까지 틀리고, 결과가 2⁻¹⁰¹⁶ 아래이면 2.2e-13까지 틀린다. 그 안에서도 Boost가 빼기로 만드는 쪽은 부르기 전에 가려 우리 것으로 보낸다(`boost_subtracts`: a, b ≤ 1, 보수 쪽, a0 < b0, x0 < 0.02, 0.6 < x0^a0 ≤ 0.9. 우리가 쓰는 Boost는 x0 ≥ 0.02에서 연분수라 빼지 않는다. 0.6은 I ≤ x0^a0/y0에서 나온다). 6만 점에서 Boost가 남는 곳은 1.7e-15, 넘긴 곳은 Boost 6.8e-15 대 우리 1.3e-15다. 큰 지수의 e^(−E)(1 − E.lo)는 E ≥ 2⁵²에서 +0이다(전에는 −0.0, E ≈ 1e300부터 NaN: `betainc(1e300, 2, 0.5)`).
- **극단 인자 정리 2차 (2026-11-18, 사용자 결정 "전부 값 또는 NaN으로")**: 극단 인자 점검에서 남았던 틀린 값 255점을 다시 재현해 83점으로 줄였다(Airy와 복소 erf·wofz를 빼면 218 → 38). 남은 것은 아래 '남은 결함'과 `hyp1f1`·`hyp2f1` 절에 적었다.
  - `binom`: n + 1을 반올림한 뒤 k의 거듭제곱에 넣어 204 ULP를 잃던 것과, n ≥ 0에서 큰 음의 비정수 k가 0 대신 ±1·±inf를 내던 것을 고쳤다. n ≥ 0의 거대 인자 가지는 인자를 정확히 넣는 식으로 다시 썼다. 무작위 8000점에서 최대 8 ULP, 새 NaN은 없다.
  - 서브노멀 모양: a, b < 2⁻⁷⁰⁰이면 `betainc`는 닫힌 꼴 b/(a + b)다(Boost는 1.0). `fdtr`·`fdtrc`·`chdtrc`의 df = 5e-324는 0이나 NaN 대신 값이다. `stdtr(0.9, -1.8e308)`은 130 → 0.8 ULP, `betaln(1e300, 1.8e308)`은 −inf → −2.0007e301이다.
  - 비중심: t 함수는 |nc| < 2⁻⁵⁰⁰이면 중심 함수로 간다(`nctdtr(0.5, -5e-324, -1e300)`이 10% 틀렸다). F의 먼 꼬리에서 구한 쪽의 보수가 1/64 아래이면 NaN이다. 역함수는 nc ≠ 0이고 p가 서브노멀이면 NaN이다(Arb도 참값을 정하지 못했고, 정방향 검증으로 틀린 것만 알았다).
  - 보장 영역에서 함께 찾은 것: `hyp1f1(0.5, 1000, -1e5)`가 4200 ULP(log Γ의 차) → 1.2 ULP, a = 1e5·b < 11의 Boost 값 83 ULP → 우리 식, `hyp2f1`의 큰 a에서 Pfaff 변환이 NaN이나 1e-12 오차를 내던 것, 넘치는 양의 합이 inf 대신 NaN이던 것.
  - `hyp1f1` 급수의 뿌리 원인: 상쇄 추정을 보정하지 않은 합으로 해서 1e17 근처에서 포화했고, 1e13 한계가 걸리지 않았다. 느슨한 영역은 이제 우리 급수(1e13을 지킬 때), b > 0의 a 점화식(double-double), 큰 |x| 전개, Boost(|a|, |b| ≤ 100만), Tricomi 순으로 시도하고 모두 안 되면 NaN이다. 느슨한 영역의 호출은 느려졌다(표본 점에서 505 → 825 ns; 속도표의 행은 아니다).
  - `hyp1f1` 보장 영역의 Boost 구간을 0 ≤ x < 16으로 줄였다(사용자 결정 2026-11-18: 보장을 넘는 곳은 우리 급수로). b < 11, a ≤ 100에서 쓰던 Boost 값은 x ≥ 18부터 4%의 점이 45 ULP를 넘고 x ≈ 500에서 297 ULP였다(무작위 4만 점, a 1e-2–100, b 1e-3–11, x 1–690). 우리 급수는 같은 점에서 최대 48 ULP(x ≈ 650, 상대 1e-14 안)이고 그 밖은 20 ULP 아래다. 정확도표의 보장 행은 78.6 → 15.9 ULP가 됐다. 속도표 입력에서는 1.2%의 점이 이 구간이라 1스레드가 2.8% 느려졌다(A/B 다섯 번 교대, 236–239 → 243–248 ms).
  - `hyp2f1`: 급수가 지수를 따로 들고, c − b와 z/(z − 1)을 정확히 넣으며, 1 − z 변환은 스케일한 Γ와 상쇄 검사(1024배 초과면 NaN)를 쓴다.
  - 속도(C++ A/B, 정확도 입력, 1스레드): 모든 행이 ±1% 안이고 `hyp2f1` +2.4%, `chndtrix` +2.1%다. 정확도표는 106행 가운데 104행이 같고 느슨한 두 행이 좋아졌다(`hyp1f1` 2.2e3 → 1.17e3, `hyp2f1` 3.07e3 → 2.73e3).
- **역함수의 극단 모양** (2026-11-20): 아주 작은 모양이 있으면 AS 109의 시작과 Halley 걸음이 그 모양의 비트를 잃는다(`betaincinv(1, 5e-324, 5e-324)`가 1, 참값 1 − 1/e). 첫 항 말고는 모든 항에 작은 모양이 곱해진다는 것으로 극한 꼴을 먼저 쓴다(`ibeta_inverse`). 둘 다 2⁻⁷⁰⁰ 아래이면 I = m + pq/(p + q)·log(u/v), m = q/(p + q)라서 u/v = e^((t − m)(1/p + 1/q))다(순방향의 b/(a + b) 갈래와 짝). 변수 쪽 모양 p만 서브노멀이면 I = 1 − p·C(u)라 근은 0이다. 다른 쪽 q가 q* = min(2⁻²⁰⁰, 2⁻⁸⁰p) 아래이면(서브노멀이든 정규수든) 1 − I = e^(−qK(u))(1 + O(q/p))라 근은 t/q에만 달리므로, t와 q를 같은 2의 거듭제곱으로 q*의 이진 자릿수까지 키운 문제의 근이다. 키운 t가 2⁻⁶⁰ 아래일 때만 그렇게 하고(오차 O(t′·t/q)는 근의 조건수 안), 그 위는 t/q > 2²⁰/p, 2¹⁴⁰이라 근 1 − v의 v가 double 아래다. 2⁻¹⁰²² 바로 위의 정규 모양에서도 Halley가 그 모양에서 1.5e-12까지 틀렸다(`betainccinv(1.17e-301, 0.137, 1.57e-300)`, κ ≈ 13; q = 2⁻¹⁰⁰⁰으로 키워도 같았다). Arb 대비 q = 2⁻¹⁰⁷⁴–2⁻⁶⁰의 무작위 1250점에서 |Δv/v|/max(1, κ)가 7.7e-16 이하다(결과가 서브노멀인 한 점은 절대오차 규칙 안). 두 모양이 2⁶⁰ 이상이고 p가 서브노멀이면 꼬리의 로그를 구할 수 없어(BFRAC가 닿지 않는다) 정규 극한에 왜도 항을 더한 근(Cornish–Fisher, 다음 항이 근의 2⁻⁷⁴ 아래)을 낸다. 한 모양이 2.3e6–2⁸⁰이고 다른 모양이 1보다 작은 큰 모양 쪽 꼬리는 그래도 NaN이다(docs/api/special.md §4).
- **남은 결함** (고치지 않음): b ≳ 1e200에서 `ibeta(14.38, 1e200, 1.5e-199)`가 NaN이고, `betaincinv(14.38, 1e200, 0.617)`이 1 − p의 근을 낸다. 2026-11-18 뒤에 남은 것: `hyp1f1` 느슨한 영역의 Boost 최대 1.2e-12·Tricomi 9e-12, `hyp2f1` 느슨한 영역 1.7e-12(z < −9, |c| > 1e4), `nctdtr`·`nctdtrc`의 df ≤ 1e-3에서 2.4e-13(23점), `fdtri(14.38, 1e-300, 5e-324)` 59 단위. 점검 점 가운데 8370점은 Arb 참값이 정해지지 않았고 6798점은 시간 초과라 검증하지 못했다.
### 5.6 제타와 지수적분
- `zeta(x, q)`: Hurwitz, Euler–Maclaurin (Cephes `zeta.c`: 앞 N항 직접 합 + Bernoulli 꼬리). Riemann(`q` 생략)의 `x < 1`은 함수방정식(변환). Hurwitz의 `x < 1`은 NaN (scipy와 같다)
- `expn(n, x)`: `x ≤ 1` 거듭제곱급수, `x > 1` Lentz 연분수 (NR `expint`), 큰 n은 균등 점근. **3구간**
- `expi(x)`: `x < 0`은 `-E₁(-x)` (변환), `0 < x < 40`은 거듭제곱급수(양의 항만이라 소거 없음), 큰 x는 점근급수. **2구간**. Cephes `ei.c`의 여섯 구간 유리식은 쓰지 않는다.
- **S5에서 정하고 바꾼 것**
  - `expn`: 큰 n의 균등 점근은 넣지 않았다. 연분수의 b = x + n이 크면 곧바로 수렴하고, 급수는 x^k/k!로 줄어서 n = 1e5까지 6.3 ULP다. **2구간**
  - `expi`: 양의 0점 x₀ = 0.3725… 둘레 [0.3, 0.45]에 Taylor 급수를 더했다(정확한 d = x − x₀, 계수는 `tools/gen_ei_taylor.py`). 거듭제곱급수는 거기서 γ + log x와 상쇄해 scipy가 5e15 ULP를 잃는다. **3구간**
  - Hurwitz의 음수 q(정수 s): ζ(s, q) = ζ(s, d) + (−1)ˢ(ζ(s, 1 − d) − ζ(s, 1 − q)), d = q의 소수부다. 반정수 q이고 s가 홀수이면 0 옆의 두 합이 정확히 지워져 ζ(s, 1 − q)만 남는다(−1/2 < q < 0도. 판정은 2(q + N) − 1 = 0을 정확히 만든다: q + 1 자체는 q > −1/2에서 반올림된다). 큰 s에서 0 옆의 두 첫 항이 double을 넘어 직접 합이 inf − inf이면, 가까운 쪽의 첫 항 x₁⁻ˢ에 대한 상대값으로 1 ± r + R₁ ± rR₂(r = (x₁/x₂)ˢ, 홀수 s의 1 − r은 expm1)를 m·2ᵉ로 더한다(`hurwitz_negative`). `polygamma(3000, -2.25)`가 NaN이었다.
  - 공개 `expi`는 x ≥ −700에서 Boost를 먼저 부른다(§5.14). 위의 세 구간(x > 0)은 Boost의 결과가 정상 범위가 아닐 때만, −E₁(−x)는 그때와 x < −700(서브노멀 꼬리)에서 쓴다. 이 둘과 특수값은 따로 뗀 함수다(Q33).

### 5.7 Bessel 0·1차 (`i0 i1 k0 k1 j0 j1 y0 y1`와 e-스케일 판)
- `i0 i1 k0 k1`: Cephes Chebyshev 전개, 함수마다 **2구간** (x ≤ 8 / x > 8 또는 x ≤ 2 / x > 2). e-스케일 판은 같은 전개에서 `exp`를 빼면 된다. 공개 함수는 Boost의 0·1차 유리식(`bessel_i0`, `bessel_i1`, `bessel_k0`, `bessel_k1`)을 직접 부르고(Q33, §5.14), Cephes 전개는 그 값이 정상 범위가 아닐 때 쓴다. `k0`은 0 < x < 700, `k1`은 2⁻¹⁰⁰⁰ ≤ x < 700에서 값이 늘 정상 범위라 다른 검사 없이 Boost 값을 낸다.
- `j0 j1 y0 y1`: Cephes. `x ≤ 5`는 유리식, `x > 5`는 Hankel 점근(진폭과 위상의 유리식). **2구간**. Cephes는 첫 영점들을 인수로 빼서(`(x² - r₁²)…`) 영점 근처의 상대 정확도를 얻는다. 우리는 이것을 쓰지 않는다. 정확도는 포락선 기준이다 (Q18, §2.3)

### 5.8 Bessel 임의 실수 차수 (`jv yv iv kv`와 e-스케일 판)
- **J·Y의 float64 경로** (`fast_j`, `fast_y`, Q21): |ν| < 500, x ≥ 2⁻²⁰⁰에서 J와 Y를 따로 계산한다(`jv`는 Y를, `yv`는 J를 만들지 않는다. 음의 정수가 아닌 차수만 반사식에 둘 다 쓴다). ν = n + μ, n = round(ν), |μ| ≤ ½. 결과가 정상 범위가 아니면(언더플로, 오버플로, 0) 아래의 일반 경로가 다시 계산한다.
  - J: x² < 2(ν + 1)이면 멱급수(항이 한 걸음에 절반 아래로 줄어 상쇄는 많아야 약 3배; Γ(ν + 1)은 ν·Γ(ν)). x ≥ 25(`kHankelFrom`)이고 `hankel_pq`가 차수 ν를 받으면 Hankel(J만, 차수 ν + 1은 만들지 않는다). ν < x이면 J_μ, J_{μ+1}에서 위로 점화한다(차수 x 아래에서 점화는 중립이다). 그 두 값은 μ = 0이면 `j0`·`j1`의 유리식, μ = −½이면 닫힌 꼴 √(2/(πx))·(cos x, sin x), x ≥ 25이면 Hankel이다. 나머지는 Miller의 뒤로 재귀를 차수 N = top + 10·top^(1/3) + 8(top = max(ν, x))에서 시작해 Neumann 합(DLMF 10.23.15) (x/2)^μ/Γ(μ + 1) = J_μ + Σ_{k≥1} (μ + 2k)(μ + 1)_{k−1}/k! J_{μ+2k}로 맞춘다.
  - Y: x ≥ 25이고 Hankel이 차수 ν를 받으면 Hankel. 아니면 Y_μ, Y_{μ+1}을 만들고(μ = 0은 `y0`·`y1`, μ = −½은 닫힌 꼴, x ≥ 25는 Hankel, x < 3(`kTemmeBelow`)은 Temme 급수, 그 밖은 Steed CF2와 Miller의 J_μ, J′_μ를 Wronskian으로 맞춘 값) 위로 점화한다. Y가 지배적인 해라 위로 점화가 안정하다.
  - Miller의 시작 차수: 잘못된 시작값의 오차는 (J_N/Y_N)(Y_k/J_k)로 줄고, 전환점 근처에서 N − top ≈ 7.4·top^(1/3)이 필요하다. float64 실행을 긴 시작과 견주어(top 0.5–100) f₀/f₁의 비는 7·top^(1/3) + 3에서 정착해 Y용은 8·top^(1/3) + 6이다. Neumann 합은 차수 top + 2·top^(1/3)까지의 항을 쓰므로 더 필요하다: 8·top^(1/3) + 6에서 94 ULP, 9·top^(1/3) + 8부터 차이가 없어 10·top^(1/3) + 8이다.
  - 점화의 계수 2v/x는 2v·(1/x의 double-double)를 fma로 한 번에 반올림한다. 반올림한 1/x는 x 자체를 1 ULP 옮겨 결과가 x·J′/J ULP 틀린다. 나눗셈을 걸음마다 하는 것보다 빠르다.
  - Hankel 쌍(μ, μ + 1, x ≥ 25)은 항이 처음부터 줄고 2⁻⁵⁶에 닿을 때까지 다시 커지지 않으므로(커지기 시작하는 k ≈ 2x − 2 ≥ 48에서 항은 1e-21 아래) 받음 검사 없이 두 차수의 곱을 한 루프에서 만든다. 위상의 cos·sin(πμ/2)는 Apple libm의 `__sincospi_stret` 한 번이다.
  - 정확도: Arb 흔한 입력 집합(ν ∈ [−20, 50], x ≤ 60, 2만 점)에서 J 포락선 최대 15.5 ULP(99% 8), Y 16(99% 8). Temme와 CF2의 경계를 3으로 둔 것은 x ∈ [1.5, 6], ν < 3의 Arb 집합에서 경계 2가 32, 3이 21, 4가 35 ULP였기 때문이다.
- **I·K의 float64 경로** (`fast_iv_kv`, Q22): |ν| < 30, 2⁻²⁰⁰ ≤ x < 700이고 값이 정상 범위일 때. I만 필요하고 x² < 4(ν + 1)이면 양항 멱급수다. 그 밖은 K_μ, K_{μ+1}(`k_pair_scaled`, e^x 스케일: x < 2는 Temme 급수, 2 ≤ x < 20은 나눗셈을 사슬에서 뺀 Steed CF2, x ≥ 20은 K만의 점근급수, Q59)에서 K를 위로 점화하고(K가 지배적인 해라 안정하다), I는 Wronskian I_ν K_{ν+1} + I_{ν+1} K_ν = 1/x로 낸다. I_{ν+1}/I_ν는 뒤로 재귀(`i_ratio`)로 구한다. 양항이라 상쇄가 없고, 시작 차수 N − ν = √(40x) + 10이다(float64 실행으로 잰 필요치는 √(30x) + 4 이하). 음의 정수가 아닌 차수의 `iv`는 I_a + (2/π)sin(aπ)K_a다. 이 경로의 앞에는 Boost가 없다. Boost는 이 경로가 값을 내지 못할 때 쓴다(§5.14).
- **일반 경로** (차수 500 이상, float64 경로가 값을 내지 못한 J·Y; |ν| ≥ 30의 I·K는 아래 Debye): Temme 방법 (NR `bessjy`, `bessik`):
  - 차수를 `ν = n + μ`, `|μ| ≤ ½`로 나눈다.
  - `x < 2`: Temme 급수로 `Y_μ`, `Y_{μ+1}` (`K_μ`, `K_{μ+1}`)
  - `x ≥ 2`: Steed 방법(CF2)으로 같은 값(K는 x ≥ 20이면 점근급수, Q59)
  - CF1(Lentz)로 `J_{ν+1}/J_ν` 비, Y·K는 앞으로 점화, Wronskian으로 J·I를 맞춘다.
  - 큰 x(x ≥ 30이고 `hankel_pq`가 받는 곳, 큰 ν에서는 x ≳ (ν + 1)²/4; 아래 'Hankel의 경계'): Hankel 점근 (CF1의 반복이 x에 비례해 늘기 때문)
  - **3구간**. 큰 ν의 Debye 균등 점근은 넣지 않는다. 큰 ν는 맞지만 O(ν)로 느리다. 느슨한 영역에 시간도 적는다.
  - S10에서 바꿈(코드리뷰), S11에서 넓힘: `iv kv ive kve`는 ν ≥ 30에서 Debye 균등 점근전개(DLMF 10.41.3–4, U₀–U₁₀, `coeffs/debye_u.hpp`, `tools/gen_debye.py`; 다음 항은 ν = 30에서 2e-16)를 쓴다(S10에서는 ν ≥ 1e4, U₀–U₄). ν = 100–1e4에서 원소당 14–121 µs이던 것이 약 0.15 µs가 됐다. O(ν) 점화식은 `iv(1e9, 1e9)`에 26초가 걸리고 CF1의 항 수 상한(5e7)에 걸려 0을 냈다(참값은 inf). 지수 νη(스케일 판은 νη ∓ x)는 ν와 s = √(ν² + x²)가 크게 상쇄하므로 double-double로 만들고 한 번만 지수를 취한다. ν = 1e4–4e4에서 점화식과 7e-16 안으로 일치한다. `jv yv`의 큰 차수는 그대로 O(ν)다.
- 음의 ν는 반사 공식(변환). 음의 x는 정수 ν에서만 정의(scipy와 같다), 그 밖은 NaN
- **큰 차수의 0 판정** (Q20, 2026-11-19, `bessel_detail::j_underflows`): J에 균등 전개가 없어서, 1e7걸음을 넘는 점화(ν − x > 1e7)는 NaN이었다. `jv(3e7, 1.9e7)`(참값 e^(−7.6e6))도 그랬다. 옛 판정은 J = 0과 Y = −inf를 함께 내려고 (x/2)^ν/Γ(ν + 1) < e^(−760)/(2ν), x < ν/2를 요구했다. 이제 J만은 DLMF 10.14.5의 |J_ν(ν sech α)| ≤ e^(−ν(α − tanh α))로도 판정한다. α − tanh α = atanh t − t, t = √(1 − (x/ν)²)는 t < 1/2에서 양항 급수 t³/3 + t⁵/5 + …(x가 ν에 다가갈 때 상쇄가 없다. t는 ν − x에서 만들어 ν = 1e30 근처도 맞다), 그 위에서 log(ν/x) + log1p(t) − t다(atanh는 아주 작은 x에서 t가 1로 반올림돼 inf였다: 첫 시도에서 `jv(0.5, 5e-324)`가 0). 상한이 e^−746.2 아래면 0이다(반올림된 상한의 오차는 이 여유보다 훨씬 작다). 경계 근처의 점(ν = 60, 1e3, 1e4, 경계의 0.99–1.03배)은 Arb 참값과 맞는다: 경계 아래는 참값이 2⁻¹⁰⁷⁵ 아래이고, 위의 값은 비트까지 같거나 1 ULP 안이다. ν = 60에서는 이 상한이 느슨해(참값 4e-326, 상한 e^−746.2) 경계의 1.03배에서도 참값이 0으로 반올림된다. Y는 엄밀한 하한이 없어(DLMF 10.14는 J의 상한뿐이다) 넓히지 않았다. Debye 꼴 |J Y| ≈ 1/(πν tanh α)는 점근식이라 증명이 아니다.
- **x = ν의 큰 차수**: CF1의 항 수는 x = ν에서 약 6ν^(1/3)이고, ν ≳ 1e12에서는 그 뒤의 CF2와 점화가 끝나지 않아 NaN이다(docs/api/special.md §8.2 (다)). x ≥ 2⁵³이면 CF1의 걸음 2/x가 b = 2ν/x의 반 ULP 아래라 b가 상수 2 + δ로 남고, 상수 b의 연분수는 약 18/√δ항에서야 수렴한다. δ < 2e-12이면 상한 1e7항을 넘는 것이 미리 보이므로 걷지 않고 NaN이다(`jv(1e300, 1e300)`이 45 ms를 걸은 뒤 NaN이었다).
- **Hankel의 경계**: `hankel_pq`는 첫 항 (4ν² − 1)/(8x)가 2를 넘으면 받지 않고, `jy_hankel`은 차수 ν + 1도 부른다. 그래서 큰 차수에서 Hankel은 x ≥ ((2ν + 2)² − 1)/16 ≈ (ν + 1)²/4에서만 쓰인다(ν = 1.2e7–2e7에서 잰 경계 ν²/x = 4.0). 코드 주석의 'ν² > 16x'는 틀린 값이었다.

### 5.9 Airy (`airy`)
- 구간 (Q23):
  - −2 ≤ x ≤ 1: Maclaurin 급수
  - 1 < x < 9.7(ζ < 20): K, I 꼴 Ai = √(x/3)K_{1/3}(ζ)/π 등(모든 항이 양수). K_{1/3}, K_{2/3}은 차수 −1/3의 K 한 쌍(`k_pair_scaled`, K_{−1/3} = K_{1/3}) 하나에서, K_{4/3}은 점화로 얻는다. I_{1/3}, I_{−1/3}은 I의 비(`i_ratio`)와 Wronskian으로, I_{2/3}은 그 비로 낸다. e^{±ζ}는 m·2ᵉ다.
  - x ≥ 9.7(ζ ≥ 20): 지수 감소 점근 Ai = e^{−ζ}U₋/(2√π x^{1/4}), Bi = e^{ζ}U₊/(√π x^{1/4}) 등(DLMF 9.7.5–9.7.8, u_k, v_k의 합). 빠뜨린 열세한 부분은 e^{−2ζ}배다.
  - −9.7 < x < −2(ζ ∈ (1.9, 20)): J_{±1/3}, J_{±2/3}(DLMF 9.6.6–9.6.9). 차수 −1/3 + k와 −2/3 + k의 두 격자를 Miller 재귀(§5.8)로 내리고 Neumann 합으로 맞춘다(1/Γ(2/3), 1/Γ(1/3)은 상수).
  - x ≤ −9.7: 진동 점근(DLMF 9.7.9–9.7.12)
  - 일반 J·Y, I·K 경로를 차수마다 두 번 부르던 옛 구간(−9.7 < x < −2, 1 < x < 200)은 없앴다.
- 출력 넷(Ai, Ai′, Bi, Bi′)을 한 번에 계산한다. 같은 중간값을 함께 쓴다. ζ의 low 부분은 Bessel 값에 1차 보정(도함수는 J′_v = J_{v−1} − (v/ζ)J_v 등의 이웃 차수)으로 붙인다.

- **S6에서 정하고 바꾼 것**
  - 0·1차: Cephes 그대로이되 Hankel 위상을 반올림하지 않는다(libm의 sin x, cos x로 덧셈정리), e^±x는 m·2^e, log(x/2)는 log x − log 2.
  - 실수 차수(지금은 §5.8의 일반 경로): Γ₁·Γ₂는 1/Γ(1+z)의 Taylor 계수(생성), x ≥ 30이고 ν < x이면 작은 차수의 Hankel에서 위로 재귀(진동 영역의 CF1은 0.3·x ULP를 잃는다), 다섯 재귀 모두 double-double, x² < ν + 1이면 거듭제곱급수, Hankel은 항이 앞항의 2배를 넘지 않을 때만. **4구간**(급수, Temme/Steed, 위로 재귀, Hankel). 차수 500 아래는 Q21의 float64 경로가 먼저다
  - Airy: 위상 ζ를 double-double로. |x| ≲ 1e12까지 포락선 기준 안이다.

### 5.10 타원적분 (`ellipk(m)`, `ellipe(m)`; scipy와 같은 매개변수 m)
- `ellipk`: AGM. `K = π / (2·AGM(1, √(1-m)))`, 이차 수렴(약 5회). `m < 0`은 변환. m → 1의 log 특이점은 `1-m`이 정확하므로(m ≥ 0.5에서 Sterbenz) AGM이 그대로 처리한다.
- `ellipe`: AGM의 `E = K·(1 - Σ 2^{n-1} c_n²)`는 m → 1에서 `∞ × 0`의 소거가 생긴다. 그래서 Carlson 대칭형 `E = R_F(0, 1-m, 1) - (m/3)·R_D(0, 1-m, 1)`(복제 정리)를 쓴다. 두 방법의 대비가 해설의 주제다.
- **S5에서 바꾼 것** (사용자 결정 2026-11-14): 공개 함수는 Cephes `ellpk`·`ellpe`의 다항식 `P(y) − log(y)·Q(y)`, `P(y) − log(y)·y·Q(y)`(y = 1 − m)다. 위의 AGM·Carlson과 정확도가 같고(최대 2.3/2.0 대 2.9/4.2 ULP) 4–7배 빠르다. m < 0은 y = 1/(1−m)를 직접 만든다.
  - 계산은 NEON + vForce 청크 커널이다(§3.4, Q31): log y는 청크 전체에 `vvlog` 한 번, 다항식은 레인 둘. 스칼라 libm 정의는 없앴다.
  - 견준 Carlson 쪽은 위의 식이 아니라 두 항이 모두 양수인 `E = (y/3)(R_D(0, y, 1) + R_D(0, 1, y))`(DLMF 19.25.1)다. 위의 식은 m → 1에서 K − (K − E)의 상쇄가 있다(y = 1e-16에서 약 20배).

### 5.11 초기하와 Lambert W
- `hyp1f1(a, b, x)`: 거듭제곱급수, `x < 0`은 Kummer 변환 `e^x M(b-a, b, -x)`, 큰 |x|는 점근전개. **3구간**. `b`가 0 이하의 정수이면 극점. 큰 |a|에서의 소거는 느슨한 영역으로 적는다.
- `hyp2f1(a, b, c, z)`, z < 1의 실수:
  - `|z| ≤ ½` 근처는 급수. 나머지는 선형 변환(`z → z/(z-1)`, `1-z`, `1/z`, `1/(1-z)`)으로 급수가 빨리 수렴하는 곳으로 옮긴다 (변환, 구간 수에 세지 않음)
  - `c-a-b`나 `a-b`가 정수이면 변환 공식이 0/0이 된다. digamma가 들어간 극한 공식 하나로 처리한다 (AS 15.3.10–12 형태, Q8)
  - a나 b가 0 이하의 정수이면 다항식(유한합). 0 < z < 1에서 그 합이 kMaxCancel2(1e16)를 넘게 상쇄하면 Pfaff 변환으로 w = z/(z − 1) < 0의 다항식 (1 − z)⁻ᵖ ₂F₁(p, c − q; c; w)를 더한다(변환이라 Q8의 갈래에 세지 않는다). c > 0, c − q > 0이면 모든 항이 양수다. 직접 합이 되는 곳은 그대로다
  - z ≥ 1: 다항식이면 값, 아니면 참값이 복소수이거나 발산하므로 NaN(z = 1은 `c-a-b > 0`이면 Gauss 공식)
- `lambertw(x, *, k=0)`, 가지 k ∈ {0, −1}:
  - 초기값: 분기점(−1/e) 근처는 `p = √(2(ex+1))`의 급수, 그 밖은 log 점근(`L₁ - L₂ + L₂/L₁`), k = 0의 작은 x는 Padé
  - Halley 반복을 수렴할 때까지(상한 있음). scipy의 `tol=`은 두지 않는다: 결과의 정확도는 함수가 정한다.
  - k = 0은 x ≥ −1/e, k = −1은 −1/e ≤ x < 0에서 정의. 그 밖은 실수 값이 없으므로 NaN (S8 뒤 complex 입력은 complex 결과)

- **S7에서 정하고 바꾼 것**
  - `hyp1f1`: 급수 항을 double-double로(연이은 곱이 k ULP를 쌓아 x ≈ 500에서 120 ULP였다; 지금은 아래의 보정 루프), Kummer의 b − a도 double-double, x ≤ −700 근처는 Kummer 형태의 점근식(Γ(b − a)를 1차 보정). b < 0에서 두 형태가 모두 상쇄하면 양의 b′에서 b에 대한 재귀(DLMF 13.3.2). 큰 |a|(a < 0)의 상쇄는 느슨한 영역으로 남았다(Bessel형 전개를 넣지 않음).
  - `hyp2f1`: 급수 항을 double-double로(지금은 아래의 보정 루프). c − a − b가 정수에 1e-3 안으로 가깝지만 정수가 아니면 1 − z ≥ 1e-3에서 급수, 1e-8 안이면 정수 m의 극한식.
  - `lambertw`: 분기점 둘레 급수의 계수를 정의에서 생성(`tools/gen_lambertw_branch.py`), e·x + 1은 e를 double-double로.

- **급수의 두 단계**: 두 급수는 항의 곱이 k ULP씩 오차를 쌓는다(a + k의 반올림은 k가 같은 이진 자릿수에 있는 동안 같아서 걸음마다 같은 쪽으로 쌓인다).
  - **float64 첫 패스**: 그냥 더하면서 Σ(k+1)|t_k|를 함께 잰다. 이 값이 4|s| 이하이면(한 걸음에 반올림이 여섯 번이라 항 k는 많아야 약 3(k+1) ULP, 합은 많아야 약 12 ULP; 잰 최대 4.1) 그 합을 쓴다. 합은 보상 합이다. 항이 처음부터 줄어드는 입력 대부분이 여기서 끝난다. 기준을 16으로 늘리면 Arb 흔한 입력 집합에서 최대 17 ULP, 기준이 없으면 34 ULP라 4를 둔다(Q26).
  - **긴 급수는 보정 루프**: 곱과 몫의 반올림 오차를 fma로 구해 1차까지 들고 간다(항과 합이 hi + lo, 항당 나눗셈 1번). 남는 오차는 2차다(z ∈ [0.5, 0.9]에서 최대 2.7 ULP). 양항 급수(`hyp2f1`의 a, b, c, z > 0, `hyp1f1`의 a, b, y > 0)에서 인자가 [1/16, 2¹⁹) 안이면 인자를 2⁻³²의 배수 vh와 나머지 vl로 나눠 vh + k를 정확히 만들고 vl은 곱에 1차로 넣는다(`f_long_positive_split`, `m_long_positive_split`; |vl/(vh + k)| ≤ 2⁻²⁸이라 버린 2차는 2⁻⁵⁶ 아래). 두 TwoSum 대신 덧셈 하나다.
  - **첫 패스를 건너뛰는 곳**: `hyp2f1`의 z > 0.4나 첫 비 abz/c > 2, `hyp1f1`의 y > 2나 ay/b > 2에서는 첫 패스가 거의 늘 기준을 넘지 못한다(항이 커지며 평균 차수가 늘어난다). 그런 입력은 곧바로 보정 루프로 간다(Q26).
  - **해 봤지만 버린 것**: 1 − z 변환의 경계를 0.9에서 0.6으로 내리면 빠르지만, a, b가 크면 두 항이 상쇄해 값이 완전히 틀렸다(7e10 ULP). 0.5–0.9에서 1 − z 변환은 s = c − a − b가 정수에 가까운 점에서 두 항이 상쇄해 보장 영역의 계약을 지키지 못한다.
  - 남은 것은 z가 0.9에 가까운 `hyp2f1`이다. 0.9ᵏ만으로도 수백 항이 필요하다(속도표는 bench/special/speed.md). scipy는 여기서 float64 급수를 쓰고 100–200 ULP를 잃는다.
- **`lambertw`의 Halley 멈춤** (Q28): 걸음이 반복값의 2⁻²⁶ 아래이면 그 걸음의 결과를 낸다. 실수의 `halley`, `halley_log`와 복소수의 `challey`, `challey_log`가 같다. 64회 상한과 '마지막 걸음이 마지막 비트이면 그 값'은 그대로다.

### 5.12 로지스틱·정보
- `expit`: `x ≥ 0`이면 `1/(1+e^{-x})`, 아니면 `e^x/(1+e^x)`. NEON은 둘 다 계산해 고른다.
- `logit(p)`: `log(p/(1-p))`는 p ≈ ½에서 상대 정확도를 잃는다. `p ∈ [0.25, 0.75]`는 `2·atanh(2p-1)`(빼기가 정확), 그 밖은 `log(p/(1-p))`. **2구간**
  - S1에서 바꿈: 처음 계획한 바깥 식 `log(p) - log1p(-p)`는 vForce에 맞는 `log1p`가 없어서(−0의 부호를 잃음, array.md §5.4) 쓰지 않았다. 바깥에서는 `|log(p/(1-p))| ≥ log 3`이라 비의 반올림이 몇 ULP에 그친다. 잰 최대 2.8 ULP다.
- `log_expit`: e = exp(−|x|)에서 `x ≥ 0`은 `−log1p(e)`, 아니면 `x − log1p(e)`. NEON + vForce 청크 커널이다(§3.4, Q31)
- `xlogy`, `xlog1py`, `entr`: 정의 그대로, 0의 규약은 scipy를 따른다 (`xlogy(0, y)`는 y가 NaN이 아니면 0)
- `rel_entr(x, y)`: `x·log(x/y)` 그대로면 x ≈ y에서 비의 반올림 오차가 0에 가까운 log에 그대로 남는다(잰 상대오차 7e-5). 그래서 `x/y ∈ [½, 2]`이면 `x·log1p((x-y)/y)`(x − y는 Sterbenz로 정확), 비가 0·서브노멀·inf가 되면 `x·(log x − log y)`. scipy 1.18도 x ≈ y에서 정확하다.

### 5.13 복소수 (S8)
- 대상은 Q19로 정했다: `wofz`, `loggamma`, 그리고 `erf erfc erfcx lambertw`의 complex 입력. complex 입력이면 결과도 complex128이다. 실수와 복소수를 둘 다 받는 함수는 바인딩이 입력의 dtype으로 루프를 고른다(실수 Op와 복소 Op가 따로 있다).
- `wofz`(Faddeeva): 위쪽 반평면 Re z ≥ 0만 계산하고 나머지는 대칭 w(−z̄) = conj w(z)와 w(z) = 2exp(−z²) − w(−z)로 옮긴다.
  - 0 ≤ Re z ≤ 7, 0 ≤ Im z ≤ 4: 0.1 격자의 가장 가까운 점 z₀ 둘레 Taylor 급수. 표(`coeffs/wofz_grid.hpp`, `tools/gen_wofz_grid.py`)에는 w(z₀)만 double-double로 두고, 계수는 w′ = −2zw + 2i/√π의 재귀 a_{n+1} = −(2z₀aₙ + 2a_{n−1})/(n+1)로 만든다(−2/(n+1)은 표의 곱). |z − z₀| ≤ 0.0707이라 20항 안팎이다.
  - 그 밖: Laplace 연분수를 뒤에서부터. 항 수는 Poppe–Wijers 식을 Johnson이 맞춘 것에 4를 더했다(더하지 않으면 76 ULP, 4이면 2.8 ULP). 한 단계 t ← z − (k/2)/t는 (k/2)·conj(t)/|t|²로 나눗셈 하나다(Q30; t는 위쪽 반평면에 있고 |z| ≤ 3e8이라 |t|²가 넘치지 않는다). |z|가 3e8을 넘으면 첫 항 i/(√π z)만 쓴다. 앞으로 가는 Wallis 재귀(루프에 나눗셈 없음)는 2배 빨랐지만 계약 대비가 0.031에서 0.145로 커져 쓰지 않았다.
  - exp(−z²): −z² = (y−x)(y+x) − 2ixy를 double-double로 정확히 만들고, 위상의 두 조각을 libm의 sin·cos(정확한 인자 축소)에 넣는다.
  - 설계의 후보(Weideman 유리 근사, Poppe–Wijers 한 식)는 쓰지 않았다. scipy(Johnson의 Faddeeva 패키지, 연분수와 Algorithm 916)를 재 보니 위쪽 반평면에서도 178 ULP라, 정확도 기준(§1.3)으로 격자 Taylor를 골랐다.
- `erfcx(z) = w(iz)`. `erfc(z)`는 Re z ≥ 0에서 exp(−z²)·w(iz)(m·2ᵉ 꼴로 곱해 오버플로를 피한다), Re z < 0에서 2 − erfc(−z). `erf(z)`는 |z| < ½에서 Taylor 급수, 그 밖 Re z ≥ 0에서 1 − erfc(z), Re z < 0에서 −erf(−z). 실수축·허수축은 따로 다뤄 0인 부분을 정확히 0으로 낸다.
- `loggamma`: Re z > 7이나 |Im z| > 7은 Stirling 8항, Re z < 0.1은 반사식(Hare의 2πik), 0.1 ≤ Re z ≤ 4이고 |Im z| ≤ 3이면 아래의 상쇄 없는 꼴, 나머지는 위로 올리는 점화식(곱이 음의 축을 건너는 횟수를 센다).
  - 1과 2의 영점 둘레에서 점화식은 Stirling 값(크기 13 안팎)과 곱의 로그가 크게 상쇄한다(잰 최대 293 ULP, scipy 439). 그곳은 Re z ≤ 1.5까지 log1p(z−2) + loggamma(z−1)로 내린 뒤, w = z − 1로 loggamma(1+w) = −γw + Σ_{j≤J} μ(w/j) + Σ_k (−1)^k ζ(k, J+1)/k · w^k (μ(t) = t − log(1+t))로 계산한다(Q29).
    - J = min(6, ⌊4|w|₁⌋): 떼어낸 t = w/j는 모두 |t|₁ ≥ ¼이고, 꼬리의 비는 |w|/(J + 1)이다(J < 6에서 ¼ 아래, J = 6에서 0.45까지).
    - 떼어낸 μ들의 합은 Σt − log Π(1 + t)로 로그 하나다. 곱에서 1을 뺀 값을 따로 들고 가서(q ← q + t + q·t) 그것이 ½ 아래이면 log1p(q), 아니면 log(곱)이다. 모든 인수의 실수부가 양수이고 편각이 Im w와 같은 부호라, 곱이 음의 축을 건너는 횟수를 세어 2πi를 더한다.
    - 꼬리 계수: J = 6은 표 d_k = (−1)^k ζ(k, 7)/k(`coeffs/loggamma_taylor.hpp`, 49항)를 |w|²의 문턱으로 고른 길이까지만(|d_k w^k| ≈ ρ^k/k, ρ = |w|/7), J < 6은 ζ(k, J + 1) = ζ(k, 7) + Σ_{j=J+1}^{6} j^{−k}(양의 항만 더하므로 상쇄가 없다)로 컴파일할 때 만든 28항이다.
    - 각 부분이 서로 상쇄하지 않는다. 옛 꼴(j마다 log1p 또는 26항 급수, 꼬리 49항)은 없앴다.
- `lambertw`의 complex: 분기점 근처(|z + 1/e| < 0.3, k = 0과 위쪽의 k = −1, 아래쪽의 k = 1)는 S7의 분기점 급수(e·z + 1을 double-double로), 그 밖은 log1p(z)나 점근식 L₁ − L₂ + L₂/L₁에서 시작하는 Halley 반복. |z|가 1e150을 넘거나 (k ≠ 0에서) 1e-150보다 작으면 w + log w = log z + 2πik에 Halley를 쓴다(w·eʷ가 넘치지 않게). 실수축 위의 실수 가지(k = 0, 위쪽의 k = −1, 아래쪽의 k = 1)는 실수 함수의 값을 그대로 낸다.
- **S8에서 정하고 바꾼 것**
  - `wofz`: 설계 후보 대신 격자 Taylor + 연분수(위). 연분수 항 수에 여유 4.
  - `loggamma`: scipy와 같은 구조에 1–2 둘레의 상쇄 없는 꼴을 더했다(Taylor 원판 둘 대신).
  - 복소수의 오차는 크기 기준, 영점 쪽은 포락선(사용자 결정). 오버플로하는 결과의 부호는 정확한 위상을 따른다.

### 5.14 Boost.Math (S10)
- **사용자 결정 (2026-11-14, S9 뒤)**: "Boost로 갈아끼는 게 이득인 부분은 다 갈고, 손해인 부분만 우리가 작성한다." 계수만 가져오지 않고 Boost.Math를 직접 부른다. 헤더는 쓰는 것만 `third_party/boost_math/`에 둔다(125개, 2.7 MB, BSL-1.0, `tools/vendor_boost_math.py`).
- **손해의 기준** (사용자 결정): 구간마다 Boost가 계약(1e-14, 특수값은 우리 규칙)을 참값 파일의 모든 점에서 지키고, 1스레드에서 느리지 않을 때(10% 안은 같은 것으로 본다)만 바꾼다. 더 정확해도 느리면 우리 것이다(`erf` 계열, `gammaln`, `j0 j1 y0 y1`). 일부 점만 틀리면 구간을 잘게 나눠 지키는 구간만 바꾼다. 같은 기준으로 우리 쪽이 빨라진 구간은 다시 우리 것이다(2026-11-22: `gamma` Q24, `iv kv ive kve` Q22, 감마 역함수의 반복 Q40). `hyp1f1`(Q27)과 `betainc`(Q45)의 Boost 구간은 다시 재고 두었다.
- **부르는 꼴**: Boost 호출은 `bst::call`(평가 오류 → NaN)로 감싼다. 상수 인자가 Boost의 분기를 정하는 곳(0·1차 Bessel)은 Boost의 해당 함수를 직접 부르고, 자주 지나는 Boost 호출만 Op 함수 안에 두고 드문 경로는 따로 뗀 함수다(Q33).
- **바꾼 구간** (`csrc/special/boost.hpp`; 측정은 커밋 4151d2a의 `bench/special/boost/routing.md`). 특수값(NaN, inf, 0, 극점)은 Boost를 부르기 전에 우리가 정하고, Boost의 결과가 정상 범위가 아니면(서브노멀, 0, inf, NaN) 우리 코드로 다시 계산한다.

  | 함수 | Boost 구간 | 우리 구간 (이유) |
  |---|---|---|
  | `zeta` | x ≥ −1, 음의 홀수 정수 −3 … −59 (자명한 영점과 2–52의 짝수 제외) | 정수가 아닌 −60 < x < −1 (Boost 65 ULP), x ≤ −60 (Boost 1.7e3 ULP) |
  | `i0 i1 k0 k1` | 전부 (Boost의 0·1차 유리식 `bessel_i0` 등을 직접, Q33) | 결과가 정상 범위 밖(i0·i1의 713 근처 참값, k1(5e-324)) |
  | `iv`, `kv` | \|ν\| < 30(`kv`는 \|ν\| < 400), x < 1e4에서 float64 경로(Q22, §5.8)가 값을 내지 못한 곳(x ≥ 700, \|ν\| ≥ 30 포함) | float64 경로(\|ν\| < 30, x < 700: Boost보다 빠르고 더 정확하다), 그 밖 (Boost가 결과가 작은 곳에서 30배까지). ν ≥ 30의 `iv`는 Debye. `kv`의 30 ≤ ν < 400은 둘 다 계약 안이고 Boost가 5% 빨라 Boost |
  | `ive`, `kve` (S11) | float64 경로가 값을 내지 못한 곳의 Boost 값 × e^{∓x}, 0 < x < 700 | float64 경로(Q22), 그 밖 |
  | `i0e i1e k0e k1e` (S11) | Boost의 값 × e^{∓x}, 0 < x < 700 | 그 밖 |
  | `betaln` (S11) | log(Boost의 beta), a, b > 0, a + b < 40 | 그 밖. `rgamma` = 1/Boost gamma는 16% 느려져서 되돌렸다 |
  | `expi` | x ≥ −700 | 서브노멀 꼬리 |
  | `lambertw` (실수, 두 가지) | \|x + 1/e\| ≥ 0.05 | 분기점 근처 (Boost 2e7 ULP) |
  | `digamma` | x > 0 | x < 0 (Boost가 극점 근처에서 틀림) |
  | `gamma` | float64 Lanczos(`gamma_plain`)의 값이 정상 범위 밖인 x > −170과 170 < x ≤ 171.62 | 2⁻³⁰ ≤ \|x\| ≤ 170(Q24: Boost보다 정확하고 빠르다), 서브노멀 꼬리 |
  | `beta` | a, b > 0, a + b < 40 | 큰 인자 (Boost 1.8e3 ULP) |
  | `gammaincinv`, `gammainccinv` | 출발점만: `find_inverse_gamma`, p ≥ 1e-300, a < 1e10, 출발점이 1e-300 이상의 정상 수 (Q40) | 반복 전부(우리 `igam`에 대한 Halley; Boost의 역함수는 반 자릿수에서 멈춰 우리 한 걸음을 더 붙였었다), 서브노멀 근, 깊은 꼬리 |
  | `betainc`, `betaincc` | a, b ≤ 6 | 큰 a, b (Boost 18배까지) |
  | `betaincinv`, `betainccinv` | a, b ≥ 10 / a, b ≥ 1 | 작은 a, b (Boost가 언더플로를 틀림) |
  | `hyp1f1` | 느슨한 영역 전부, 보장 영역의 x ≥ 0이고 b < 11 | 보장 영역의 나머지 (Boost 208 ULP) |
- 그대로 둔 것: Airy(우리는 네 값을 한 번에 계산하고(Q23) Boost는 넷을 따로 불러 몇 배 느리며 |x| > 15에서 틀린다), `expit`, `logit`, `erf` 계열, `expn`, `polygamma`, 타원적분, `jv`·`yv`, 불완전 감마(Boost가 느리거나 틀린다). Boost에 없는 함수(복소수 전부, `erfcx`, 정규분포, e-스케일 Bessel, `hyp2f1`, Hurwitz 제타, 리덕션)도 우리 것이다.
- `hyp1f1`의 느슨한 영역에서 Boost도 값을 내지 못하면(큰 음의 a), a < −1e3이면 Tricomi의 Bessel 함수 전개(A&S 13.3.7)를 쓴다. |a| ≥ 1e3의 mpmath 참값 46점에서 Boost와 같은 수준(상대 1e-12 안팎)이었다. `hyp1f1(-1e12, 2, 0.5)`가 inf에서 −3.47e-10이 됐다(코드리뷰).
- 내부 호출: `hyp2f1`과 `hyp1f1`의 우리 식이 부르는 `Digamma::f`도 바뀐 구간에서는 Boost다. `Gamma::f`는 Q24 뒤로 2⁻³⁰ ≤ |x| ≤ 170에서 `gamma_plain`이다. `betainc`와 `betaincc`의 합은 4.5 ULP까지 1에서 벗어난다(모양 [1e-5, 6]의 34만 점; Boost가 둘을 따로 계산한다. 최대인 점은 둘 다 Boost 값이다).

---

### 5.15 분포 함수 (`special/ops/distr.hpp`, T1)
- 15개 모두 기존 함수의 조합이다. 조합에서 정확도를 잃지 않는 것이 일의 거의 전부였다.
- **인자를 반올림하지 않는다.** `stdtr`의 x = df/(df + t²), y = t²/(df + t²)와 `fdtr`의 w, 1 − w를 입력에서 double-double로 만들고, 반올림하지 않은 채 불완전 베타의 핵심(`ibeta_detail::ibeta(a, b, x, y)`, §5.4)에 넘긴다. 매개변수가 크면 I_x는 x에 민감하다(조건수 ~ a·x). 그래서 x를 double로 반올림하면 꼬리에서 100–300 ULP를 잃었다(처음 측정: fdtr 290, fdtrc 363). dd로 넘긴 뒤 16–18 ULP다. 핵심은 I와 1 − I를 한 번에 주므로 보수도 같은 호출이다.
- **작은 모양은 반올림한 인자로 `betainc`를 부른다.** `stdtr`은 df/2 ≤ 6, `fdtr`·`fdtrc`는 dfn/2, dfd/2 ≤ 6이면 x, y 중 작은 쪽을 double로 반올림해 `Betainc::f`·`Betaincc::f`(§5.14의 Boost 구간)에 넘긴다. I_x의 x에 대한 조건수가 a·x ≤ 6이라 반올림이 3 ULP 안이다(`t_tail`, `f_value`).
- **작은 인자 급수**: 인자가 2⁻¹⁰⁰⁰ 아래(언더플로 근처)이면 I_x(a, b) ≈ x^a/(a B(a, b))를 직접 쓴다. 예: `stdtr(1, −1e200)` = 3.2e-201인데 x = 1e-400은 언더플로한다.
- **역함수는 작은 쪽을 푼다.** `stdtrit`, `fdtri`는 x와 1 − x 중 작은 쪽을 betaincinv/betainccinv로 푼다. 어느 쪽이 작은지는 p를 I₁/₂(a, b)와 비교해서 정한다. 처음에는 p ≤ 1/2로 골라서, x가 아주 작은데 1 − x를 풀었다. 그 결과 `stdtrit` 4e-9, `fdtri`는 0을 냈다.
- **서브노멀 자유도**: aB(a, b) = Γ(a + 1)Γ(b)/Γ(a + b)는 a → 0에서 1로 유한한데, 곱 a·B(a, b)는 B가 넘치는 a(서브노멀)에서 inf나 NaN이었다. B가 넘치면 1 + a/b로 둔다(`shape_beta`; B = (a + b)/(ab)(1 + O(a + b))). df = 5e-324는 df/2가 double이 아니라(0) 극한을 쓴다: `stdtr`은 0.5, `stdtrit`은 df < 2⁻¹⁰⁰⁰에서 ±inf(|t| = √df·(q·aB)^(−1/df)), `fdtri`는 dfn 쪽이면 0, dfd 쪽이면 fdtr이 dfd/2에 선형이라 2p의 근, 둘 다면 질량 비교다. 마지막 x = (dfd/dfn)·(1 − v)/v(와 w/(1 − w) 꼴)는 dfd/dfn이 정규 double이면 그 곱 그대로이고, 그 밖이면 네 가수를 곱하고 지수를 더해 ldexp 한 번으로 낸다(`scaled_ratio`). 전에는 dfd/dfn이 0으로 언더플로하고 근이 double 밖이라 v = 0일 때 inf × 0 = NaN이었다(`fdtri(2, 5e-324, 0.1)`, `fdtri(10, 1e-323, 0.1)`, `fdtri(1e20, 1e-310, 0.1)`; 참값은 fdtr(가장 큰 double) < p라 inf, 2026-11-22). 몫이 정규인 입력의 비트는 그대로다. 골든은 `special/fdtri grid`의 한 점(`fdtri(1e10, 1e-300, 1e-300)`, 몫 1e-310이 서브노멀)이 바뀌어 다시 기록했다: 6.070021611074974e-300 → 6.070021611074994e-300(Arb 6.07002161107499320e-300; 옛 몫이 비트를 잃어 상대 3.3e-15 틀렸다). `chdtrc`·`chdtri`의 홀수 끝 비트 df도 Q(a, y)가 a에 선형인 것으로 반을 정확히 든다(Q(df/2, y) = Q(df, y)/2). `exp_dd`는 exp(hi)가 0이나 inf이면 그 값을 바로 낸다. 거대한 지수의 lo는 수천이라 1 + lo가 음수가 되어 부호를 뒤집었다(`stdtrit(1e-20, 0.3)`이 +inf, `fdtri(1, 1e-16, 0.99)`가 −inf).
- **역함수의 먼 꼬리**: 근이 2⁻⁶⁰ 아래이면 급수 p ≈ x^a/(aB)를 거꾸로 풀어 거듭제곱으로 얻는다. 지수 −log(paB)/a를 double로 계산하면, 반올림한 지수에 |log|(~450)이 곱해져 191 ULP를 잃었다. 그래서 dd로 계산하고 exp를 한 번만 반올림한다(`exp_dd`, `log_product`; 서브노멀 p는 2⁶⁴를 곱해 둔다). 그 뒤 `stdtrit` 6.9 ULP다.
- **`ndtri_exp`**: |p − ½| ≤ ¼(y ∈ [log ¼, log ¾])이면 q = expm1(y + log 2)/2를 바로 만들어 ndtri의 가운데 유리식 q·R(q²)에 넣는다. log 2는 hi + lo(hi = double의 log 2)로 더하고, y = −log 2 근처에서 y + hi는 정확하다(Sterbenz). 전에는 exp(y) − ½에서 q의 비트를 잃어 `ndtri_exp(-0.6931471805599453)`가 0이었다(참값 2.9e-17; |q| < 0.01에서 상대 3.6e-11까지, 2026-11-22). 구간은 잰 값으로 골랐다: Arb 대비 6500점에서 |q| < 0.3이면 이 길이, 그 위(0.3–0.425)는 아래의 옛 길이 더 정확했다(새 길 6.3e-16, 옛 길 3.4e-16). 고친 뒤 구간의 최대 상대오차는 6.7e-16이다(옛 길 3.6e-11). 정확도 스윕 `special/ndtri_exp`는 최대 ULP/배율 146 → 4, 최대 상대오차 1 → 4.8e-16이다(계약 대비는 그대로). 골든은 `special/ndtri_exp`의 여덟(grid, random, large, scalar, ints, strided, broadcast, out)이 그 구간의 점에서 1–14 ULP 바뀌어 다시 기록했다(예: scalar의 y = −0.7은 −0.00855947858248031 → −0.008559478582480285, Arb −0.008559478582480282). 그 밖은 AS 241에 L = log min(p, 1 − p)을 로그 없이 넣는다(Q47): y > log ½이면 1 − p = −expm1(y)로 −ndtri(1 − p), y ≥ −745이면 L = y 자체다(가운데 유리식은 p = e^y ≥ 0.075에서 q = e^y − ½). ndtri의 꼬리 유리식은 s = √(−L) = 27.3(p = 5e-324)까지 맞으므로, 값은 ndtri(exp(y))와 같은 식에서 exp와 log만 빠진 것이다. y < −745는 점근식 log Φ(x) = −x²/2 − log(−x) − log(2π)/2 + log S(S = 1 − 1/x² + 3/x⁴ − …, |x| ≥ 38.5에서 8항, 9번째 항은 1e-19 아래)에 대한 Newton 두 걸음이다. 출발점은 x² = w − log(w − log w), w = −2y − log 2π이고, 잔차의 x²/2 + y는 two_prod로 정확하다. 그래서 오차는 eps·log|x|, x에 대해 상대 eps·log|x|/x²다. 옛 `log_ndtr`의 Newton은 기울기 φ/Φ가 크기 x²/2인 두 수의 차여서 |x| > 1e6부터 자릿수를 잃었고, 지웠다. y < −1e8은 x² = −2y − log 2π − 2 log(−x) + 2 log S의 고정점 세 걸음이다(한 걸음에 1/x² < 1e-8씩 줄어든다). y < −2¹⁰⁰⁰에서는 −2y가 넘치므로(y ≤ −8.99e307) 같은 고정점을 x²/2에 대해 풀고 x = −√2·√(−y − log(2π)/2 − log(−x) + log S)로 쓴다(2026-11-19). `sd.stats.Normal`의 `ilogcdf`·`ilogccdf`는 log q < −2¹⁰⁰⁰에서 이 값을 그대로 쓴다(Newton의 z²/2가 넘친다).
- **`stdtrit`의 갈래**: df ≥ 1000·max(1, z²)(z는 p의 정규 편차)이면 Cornish–Fisher 전개(A&S 26.7.5, 1/df⁴까지)로 평가 없이 낸다(Q46). 다음 항이 t의 2⁻⁵⁶ 아래다. 그 밖에서 df ≥ 1이고 양쪽 꼬리 확률이 1e-12보다 크면 Hill의 근사(CACM 396, 6자리)에서 출발해 꼬리 함수에 Halley 걸음을 한다. 걸음이 4e-6|t| 아래이면 남는 오차가 1e-17이라 대부분 평가 한 번이다(betaincinv의 네댓 번 대신; 2026-11-15). 참값 최대 7.4 ULP, 무작위 4천 점(df 1–1.6e5) 최대 4 ULP다. 더 깊은 꼬리와 df < 1은 betaincinv다.
- **`stdtr`의 큰 df**: t_tail의 I_x(df/2, ½)는 x가 1에 가까워 BGRAT이고, 그 Q(½, z)는 erfc(√z)다(Q36).
- **S4의 결함을 고쳤다.** T1을 재다가 `betaincinv`, `betainccinv`가 a, b < 1의 약 8%에서 0이나 1을 돌려주는 것을 찾았다. AS109의 처음 추정이 U자 밀도에서 엉뚱한 끝을 골라 (0, 1) 밖의 값(예: 1 − x = 5400)을 냈고, 코드는 그것을 "근이 언더플로한다"로 읽었다. 이제는 범위 밖이면 다른 끝의 근사를, 그것도 안 되면 0.5에서 출발한다(`as109`). 40만 개 표본에서 scipy와 모두 일치한다. 그 영역의 참값 파일도 다시 만들었다. 옛 참값은 출발점으로 쓴 우리 함수의 0을 그대로 받아 0이 들어 있었다.
- **참값 생성** (`tools/special_ref.py`, Arb; §6): 공(ball)이 자릿수 손실을 그대로 드러내므로, 정밀도를 올려도 공이 좁아지지 않는 식을 피한다.
  - I_x(a, b)는 자기 급수와 1 − I_{1−x}(b, a) 가운데 항이 훨씬 적은 쪽으로 구한다(`_ibeta`). Arb의 `beta_lower`는 a, b ~ 1e4의 평균 근처에서 공이 좁아지지 않는다. 1 − x는 두 배로 올린 정밀도에서 정확하다.
  - 역함수 `fdtri`는 w = dfn·x/(dfn·x + dfd)가 아니라 x에 대해 푼다(작은 쪽 꼬리의 log에서, log x로 Newton). w는 x가 크면 1로 반올림되기 때문이다.

### 5.16 비중심 분포 (`special/ops/noncentral.hpp`, T2)
- **Boost는 쓰지 않는다.** scipy 1.18이 쓰는 Boost.Math의 비중심 분포를 먼저 쟀다. 중앙값은 1 ULP 남짓이었지만 깊은 꼬리에서 0을 내거나 수백–수천 ULP가 틀렸다(최대 8.8e15). Boost 정책(§5.14)에 따라 직접 계산하고, 헤더도 옮기지 않았다.
- **포아송 혼합**: 카이제곱 S = Σ_j D(j, λ/2) P(df/2 + j, x/2), F는 I_w(dfn/2 + j, dfd/2)로 같다. D(a, μ) = μ^a e^(−μ)/Γ(a+1)이다. t는 x = t²/(df + t²)에서 Φ(−δ) + ½Σ[D(j, δ²/2) I_x(j+½, df/2) + sign(δ) D(j+½, δ²/2) I_x(j+1, df/2)]다. 위꼬리는 G를 Q, 1 − I로 바꾼 같은 꼴이라 양의 항만 더한다.
- **한 번의 걷기**: G(s) = G(s+1) + T(s)(아래 족) 또는 G(s+1) = G(s) + T(s)(위 족)다. 이 점화식은 T를 더하는 방향으로만 안정하다. 그래서 반대쪽의 먼 끝 J에서 출발한다. J는 무게 비율과 G 비율의 엄밀한 상한(Γ(s, y) ≤ y^(s−1)e^(−y)·y/(y−s+1) 등)의 곱이 2⁻⁵⁶ 아래로 떨어지는 곳이다. 거기서 G·T·무게를 한 번씩 직접 구하고, 정점을 지나 항이 사라질 때까지 걷는다(`mixture_walk`). 처음에는 정점과 양 끝 세 곳에서 직접 계산했는데, 호출 한 번에 직접 계산이 6–9번 들어가 훨씬 느렸다. 이 걷기는 μ ≤ 2e4의 float64 걷기가 받지 못한 곳(μ > 2e4, 틀이 맞지 않는 곳)과 sd.stats의 넓은 꼬리(wide)의 길이다. 바깥으로 가는 상한은 걸음마다 나눗셈 하나다(무게 비와 G 비 상한의 곱, 상대 여유 2⁻⁴⁰; b < 1의 e^((1−b)/(s+b))는 1/(1 − u) ≥ e^u로 바꾼다, Q49).
- **float64 걷기** (`plain_walk`, Q51): μ ≤ 2e4의 확률은 먼저 이 걷기다.
  - 시작은 j*(가장 큰 항)다. w(j*), T(j*)를 직접 구하고, w와 T는 거기서 양쪽 바깥으로 비율을 곱한다. 곱은 어느 쪽으로도 안정하고, 걸음마다의 반올림은 항의 상대 ULP로 남는다(n걸음 밖에서 약 √n ULP, 그곳의 항은 이미 작다).
  - G의 점화는 j* 쪽으로만 안정하다. 그래서 먼 쪽(j*에서 J까지)은 부분합으로 바꿔 더한다. 아래 족(J > j*)은 Σ_{j*..J} w_jG_j = G_J W_J + Σ_{i=j*..J−1} T_i W_i(W_i = Σ_{j*..i} w_j), 위 족(J < j*)은 G_J W′_J + Σ_{i=J..j*−1} T_i W′_{i+1}(W′_i = Σ_{i..j*} w_j)다. 모든 항이 양수다. J는 `mixture_walk`와 같은 엄밀한 상한으로 같은 루프에서 정한다. 직접값은 셋(j*의 w와 T, J의 G)이고, G(j*)는 G_J에 바깥 길의 T를 더한 것이다. 가까운 쪽은 j*에서 G 자신의 안정한 점화로 항이 사라질 때까지 걷는다.
  - 비율은 정확한 값에서 한 번만 반올림한다. 몫은 정확한 나머지로 고친다. 곱(d/y, d/μ의 double-double 역수)은 fma 하나다. 모양 s0 + j, s + b는 hi + lo로 들고 낮은 부분을 1차로 넣는다. 반올림한 뒤 반 ULP 아래의 보정을 더하면 늘 같은 쪽으로 사라져 걸음마다 쌓인다(d·(1/μ).hi + d·(1/μ).lo가 걸음마다 −0.45 ULP, 이진 구간 안에서 같은 s0 + j의 반올림이 수백 걸음에 100 ULP 넘게).
  - 비율(무게와 T, 바깥 상한)은 8개씩 미리 계산한다. 각 비율은 걸음 수에서 바로 만들어 서로 독립이므로 나눗셈의 지연(약 18 사이클)이 항의 사슬 밖에 있다(사슬 위일 때 걸음이 약 3배 길었다). 무게 비와 T 비는 나눗셈 하나를 나눠 쓴다. b < 1의 베타 족은 바깥 상한이 비율로 바로 나오지 않아 따로 한 패스다.
  - 틀: w는 2^kw, T는 2^kt, G는 2^kg를 곱해 들고 j*에서 각각 약 1이다(T가 G보다 훨씬 작은 1 근처 꼬리가 있다). 먼 쪽에서 w가 2⁶⁰⁰을 넘어 오르면(j*가 꼬리에 있고 무게의 최빈값이 그 너머) 2⁶⁰⁰을 w에서 T로 옮긴다(곱은 그대로). T가 G의 2⁻⁹⁰⁰ 아래이면 G에 더하지 않는다(서브노멀 인자가 걸음마다 약 3배 느렸다; T가 제 틀에서 2⁸⁰⁰까지 오르면 옛 걷기다). 합은 m·2ᵉ로 낸다(아주 작은 확률도 지수를 지킨다).
  - 틀이 맞지 않거나(T(j*)가 2⁻⁹⁶⁰ 아래, G가 T 틀에 넘침) 걷기가 끝나지 않거나 μ > 2e4이면 `mixture_walk`다. μ의 경계는 오차가 걷기의 길이와 함께 자라서다(j* 근처의 항은 약 √μ개 비율의 독립 반올림의 합이다: μ ~ 5000에서 p999 10–20 ULP, μ = 1e9에서 1000 ULP).
  - 밀도(역함수의 Newton): Σ w_j T(s_j) s_j에 족의 인자(감마 족은 1/y)를 곱한 값이고, `mixture_walk`의 밀도와 같다.
- **드리프트 없는 점화식**: 무게와 T는 비율 μ/(j+1+h), y/(s+1), x(s+b)/(s+1)로 걷는다. 먼 끝에서 double로 걸으면 32걸음에 수십 ULP가 쌓였다(8걸음마다 직접 다시 맞추면 정확하지만 느리다; float64 걷기는 j*에서 바깥으로 걷고 비율을 한 번만 반올림해 이것을 피한다). 그래서 이 걷기는 "double + 오차" 산술(`lite::Lite`: 곱은 fma의 정확한 나머지와 교차항을 들고 정규화하지 않는다, 걸음마다 상대 2⁻¹⁰⁰)로 걷는다(Q49; 옛 double-double과 `recip`보다 약 1.3배). 나눗셈은 걸음마다 많아야 하나다. 모양 s + 1, s + b는 걸음마다 ±1을 정확히 더한다. G와 합은 보상 합이다. 역함수를 위해 밀도 Σ w_j G′_j(감마 족은 T(s − 1), 베타 족은 T(s)·s; 먼 끝의 항까지)도 같이 모은다.
- **모양 s = s0 + j는 double-double이다.** double로 반올림하면 G가 s·eps·|∂log G/∂s|만큼 움직인다. 꼬리에서는 s·eps·|log(y/s)|이고 s = 800에서 250 ULP였다. G와 T는 s.hi에서 구한 뒤 s.lo만큼 1차 보정한다. T는 log-도함수(digamma)로, G는 전방 차분으로 보정한다. t의 δ²/2도 같은 이유로 double-double이다(δ = 17에서 89 ULP).
- **프레임**: 항이 double 범위 밖이어도 비트를 잃지 않게 다음처럼 다룬다. 무게와 G의 직접값은 지수가 분리된 꼴(`Scaled`)이다. 걷는 동안 무게는 2^±500 안에 두고 남는 지수는 G로 옮긴다. G가 2^±500을 넘으면 합과 함께 프레임을 옮긴다. 아주 작은 G(< 2⁻¹⁰⁰⁰)는 앞 인자 × 급수나 연분수로 직접 구한다. 그래서 서브노멀 결과도 절대오차 규칙을 지킨다.
- **t의 반대쪽 꼬리**: t > 0이고 δ < 0인 P(T > t)에서는 두 급수가 상쇄한다(scipy는 1.4e-76에 1.6e-33, 그 아래는 0). 그래서 Q = ∫ ½erfcx(z/√2) e^(E(v)) dv로 계산한다. z = a + t e^v이고, E는 −z²/2와 S = √(V/df)의 로그 밀도의 합이다.
  - E를 정류점 v*(닫힌 꼴)에서 E*(double-double)와 상쇄 없는 나머지로 나눈다.
  - 적분은 v − v* = c·A·sinh(u/A) 위의 사다리꼴 규칙이다. c = min(σ, 0.4), A = 8, 간격 0.25다. 마디는 8개씩 따로 계산하고, sinh u·cosh u는 지수 하나(|u| ≤ ½은 expm1)에서, δ의 e^δ − 1 − δ와 e^(2δ) − 1 − 2δ도 지수 하나에서 만든다(Q54).
  - 이 적분은 비교 순서의 마지막이다(Q54). 먼저 다른 쪽(양항 급수)을 구해 그것이 ⅞ 이하이면 1 − 그것이다. 그 밖은 덩어리 합의 두 족 cp = Σ D(j, μ)(1 − I_x(j + ½, b)), cq = Σ D(j + ½, μ)(1 − I_x(j + 1, b))의 차 (cp − cq)/2이고, cq ≤ cp/4일 때만이다(y^b·erf(|δ|/√2) > ¼이면 합을 시작하지 않는다). 간격이 df ~ 1 이상에 맞춰져 작은 df에서는 꼬리가 거의 줄지 않아, `nctdtr(0.001, 1, −10)`이 1700 ULP 틀렸었다.
  - E 안의 e^(2v) 때문에 해석적 띠의 폭이 σ와 상관없이 π/4로 묶인다. 그래서 정점 근처를 선형으로 촘촘히 두어야 했다. A = 4이면 2e-13, A = 8이면 1e-17이었다.
- **역함수**: 작은 쪽 꼬리의 log(F/목표)를 근으로 푼다(Q48).
  - log F − log 목표로 하면 F ≈ 1e-238에서 각 로그가 546 eps의 절대오차를 가져 근이 190 ULP 움직였다.
  - 출발점은 Patnaik 근사(카이제곱·F)와 정규 근사(t)다. t는 df ≤ 2이고 꼬리가 1e-8 아래이며 비중심도가 그 꼬리 쪽이면 중심 t의 분위수가 더 멀 때 그것이다(꼬리가 |t|^−df로 떨어진다: df = 0.22에서 P(T ≤ t) = 1e-34는 t = −6e146).
  - 먼저 Newton이다(`newton_root`): 양수 변수는 log x, t는 asinh x 눈금의 걸음이다. 꼬리와 꼬리 길이(꼬리/밀도)는 순방향과 같은 순서로 구한다(덩어리 합의 밀도, 걷기의 밀도 합, t의 반대쪽 적분의 ∫s φ(z) f_S ds; `*_tail_len`). 그래서 근은 순방향 함수의 근이다. 길이는 m·2ᵉ의 몫이라, 밀도가 꼬리보다 먼저 언더플로하는 무거운 꼬리에서도 쓸 수 있다. 걸음은 눈금에서 2가 상한이고, 같은 쪽으로 잘리면 두 배로 키운다. 지나온 점으로 브래킷을 좁히고, 브래킷을 벗어나는 걸음은 이분으로 바꾼다. 긴 asinh 걸음은 sinh(asinh x + du)이고, 짧은 걸음은 x cosh du + √(1 + x²) sinh du다(긴 걸음을 이 꼴로 하면 상쇄로 x가 그대로 남았다). 끝나는 조건은 두 Newton 걸음에서 잰 이차 상수로 다음 오차가 max(|x|, 길이)의 2⁻⁵³ 아래일 때, 또는 걸음이 그 2⁻⁵⁰ 아래일 때다. 평가는 3–4회다.
  - 밀도가 없거나(순방향의 가장자리, 아주 작은 인자, 합이 끝나지 않음) 24걸음 안에 끝나지 않으면 옛 길이다. 로그 또는 asinh 눈금에서 브래킷을 잡아 2배 폭까지 이분하고, x 자체에서 Brent법으로 마무리한다(평가 5–12회). 근이 double 범위 밖이면 0이나 inf다. 서브노멀 간격보다 작은 걸음은 진전이 없으니 걸음을 키운다.
- **측정 범위 밖의 견고함** (2만 점 무작위 입력: 자유도·nc 1e5, x 1e±300):
  - **Chernoff로 거르는 꼬리** (`chndtr`·`chndtrc`만): 적률생성함수 (1 − 2s)^(−df/2)e^(nc·s/(1 − 2s))에서 s = −1/2, 1/4로 P(X ≤ x) ≤ e^(x/2)·2^(−df/2)·e^(−nc/4), P(X ≥ x) ≤ e^(−x/4)·2^(df/2)·e^(nc/2)다. 한계가 e⁻¹⁰⁰⁰ 아래이면 걷지 않고 0과 1이다. 무게의 최빈값이 2⁵²를 넘어 걸을 수 없던 `chndtr(1, 1e16, 1)`(NaN)이 0이 된다. sd.stats의 로그 꼴은 꼬리의 값이 필요하므로 이 검사를 쓰지 않는다(`chndtr_tail`).
  - **무시할 합**: j*의 무게와 G 상한의 대략적인 로그(lgamma)가 −2000 아래이면 0을 돌려준다. 이것이 없을 때 x ~ 1e100의 꼬리는 j* ~ 2⁵²에서 J까지 걸으며 멈추지 않았다. 지수 −1.4e5 아래에서는 Scaled의 exp가 포화해 NaN도 냈다.
  - **작은 인자의 꼬리**: t나 f가 거대해서 작은 인자(df/(df+t²), dfd/(dfn·f+dfd))가 2⁻¹⁰⁰⁰ 아래이면 I_y(b, s) ≈ y^b Γ(b+s)/(Γ(b+1)Γ(s))로 혼합 합을 바로 더한다. 이것이 없을 때 F의 위꼬리는 0이었다(dfd = 0.01, f = 1e300의 참값은 0.031). b·log y는 double-double이다(−1e3에 달해 double이면 4e-14).
  - **기타**: 확률은 [0, 1]로 자른다(무게의 합이 1을 1 ULP 넘었다). j*를 찾는 이분법의 상한이 정수가 아니어서(4μ) 끝나지 않던 결함도 고쳤다.
- **비용을 줄인 것** (2026-11-15, 2026-11-22):
  - j*는 무게의 최빈값에서 비율을 한 번 보고, 꼬리 쪽일 때만 이분한다.
  - 무시할 합의 검사(lgamma 둘)는 j*가 최빈값이 아니거나 인자가 모양에서 2²⁰배 떨어진 때만 한다.
  - 덩어리 합(lite)의 첫 항 인자는 Q42의 Γ = e^S·F 나눔으로 만든다(`first_factor`, Q50; 옛 `dd::loggamma` 셋).
  - 역함수의 Brent법(뒷길)은 걸음이 3e-9|x| 아래가 되면 마지막 두 점의 할선으로 끝낸다(브래킷을 닫는 평가 둘을 아낀다. 오차는 두 걸음의 곱이라 1e-17 아래다).
  - `mixture_walk`에 남은 비용은 구조에서 온다: 먼 끝의 G(모양 보정의 차분까지 불완전 함수 2번), 앞 인자, 약 27√μ걸음의 "double + 오차" 연산이다. 그래서 μ ≤ 2e4는 float64 걷기가 먼저다(Q51).
  - 덩어리 합과 걷기는 `lite_pick`의 비용 모형으로 고른다: 걷기는 `walk_fixed`(감마 족 70, 베타 족 110, t 150) + 15√μ 항이고, 그보다 긴 덩어리 합(아래·위 모두)은 시작하지 않는다(Q53).
  - 작은 쪽을 먼저 더한다(`small_side`, Q52): z(분포 가운데에서의 대략의 표준편차 거리: `ncx2_z`, `ncf_z`, `nct_z`)가 물은 쪽이 크다고 하면(\|z\| > 1.5) 다른 쪽을 먼저 더하고, ¼ 이하이면 1 − 그것이다. 1에 가까운 쪽의 걷기는 무게의 종 전체를 걸었다(G가 모든 항에서 거의 1이다). 작은 쪽은 제 정점의 항만이거나, e⁻²⁰⁰⁰ 아래라 걷지 않는다. 물은 쪽을 직접 더한 값이 1 − 2⁻²⁰과 1 사이이면 다른 쪽도 더해 1 − 그것으로 짝을 맞춘다(작은 쪽이 2⁻²⁰ 아래인 짝은 반올림 안에서 1이 된다). t에서 다른 쪽이 반대쪽 꼬리이면 z를 믿지 않는다(그 적분이 이쪽 급수보다 비싸다).
- **T1·S4의 결함을 고쳤다** (T2의 넓은 입력이 찾았다):
  - **`fdtr`**: dfd가 아주 작고 dfn·x가 거대하면(`fdtr(3218.14, 0.0129, 1e20)`) 1.5e-9가 틀렸다. w = 1 − 4e-26을 나눗셈으로 만든 double-double은 1에서의 거리를 절대 1e-32까지만 갖는다. 그런데 `bgrat`이 log w를 그 w에서 계산했다. 이제 x > ½이면 log x = log(1 − y)를 작은 쪽 y에서 계산한다. S4·T1 참값의 정확도는 그대로다.
  - **`chdtr`**: x가 서브노멀이면 x/2가 반올림되거나 0이 되었다(`chdtr(0.18, 5e-324)` = 0, 참값 7.9e-30). 이제 x < 2⁻¹⁰²¹에서는 (x/2)^a/Γ(a+1)을 double-double 지수로 계산한다.
- **참값** (tools/special_ref.py): 두 꼬리 모두 양의 항의 합으로 계산한다. 아래 족은 먼 끝에서 거꾸로, 위 족은 0에서 앞으로 더하고, 끝의 항과 꼬리가 합의 1e-45 아래가 될 때까지 늘린다. t의 반대쪽 꼬리는 Arb의 엄밀한 적분(`acb.integral`)으로 정의 그대로 적분하고, 적분 구간 밖의 상한을 공에 더한다. 역함수의 밀도는 카이제곱이 Bessel 식, t가 두 CDF의 차, F가 ₁F₁ 식이다.

- **아주 작은 자유도** (2026-11-19): `chndtr`·`chndtrc`는 0 < df < 2⁻⁹⁶⁰, x > 0이면 df = 0의 법칙에서 낸다(df/2가 아주 작으면 불완전 감마가 NaN이었다: `chndtrc(2**-1022, 1, 1e-300)`, 같은 날 다시 찾은 `chndtrc(1e-307, 1, 1e-300)`; df ≲ 1e-305의 작은 x). j ≥ 1 항의 모양 j와 a + j(a = df/2)의 차이는 상대 a|log y| < 2⁻⁹⁵⁰이다. j = 0 항은 df = 0에서 0의 점질량이고, 여기서는 위꼬리에 e^(−μ)Q(a, y)를 더한다. Q(a, y) = Γ(a, y)/Γ(a) = a·E₁(y)(1 + O(a))다(Γ(a) = 1/a, Γ(a, y) → E₁(y)). 이 항은 nc도 아주 작으면 꼬리의 큰 몫이다(`chndtrc(2**-1022, 1e-300, 1e-300)` = 5.00008e-301; 처음에는 nc ≥ 2⁻⁸⁹⁹에서만 df = 0을 써서 거기가 NaN이었다). 아래꼬리는 그 곁에서 1로 반올림되는 df = 0의 값이다. x = 0은 0(df > 0에는 점질량이 없다)이다. 역함수(`chndtrix`, `ncfdtri`)는 브래킷을 못 찾으면, 가장 큰 double에서도 꼬리가 목표에 못 미칠 때 inf, 가장 작은 양수에서 이미 넘을 때 0이다(`ncfdtri(5e-324, 5, 2, 0.5)`가 NaN이었다). `ncfdtr`의 w < 2⁻¹⁰⁰⁰ 꼴(j = 0 항만)은 다음 항을 무시해도 되는지를 μw(a + b + 1) ≤ 2⁻⁶⁰·min(1, a)로 보았는데, 항의 비는 μ·I_w(a + 1, b)/I_w(a, b) = μw(a + b)/(a + 1)이라 a로 나눌 까닭이 없었다. dfn이 약 2⁻⁹⁹⁷ 아래이면 늘 실패해 NaN이었다(1e-300은 w가 2⁻¹⁰⁰⁰ 위라 다른 길로 값). 이제 max(1, μ)·w(a + b)/(a + 1) ≤ 2⁻⁶⁰이다(위꼬리 1 − e^(−μ) ≈ μ에 대해서도 상대 2⁻⁶⁰). j = 0 항의 a는 2¹⁰⁰⁰을 곱한 뒤에 반으로 나눈다(df·2¹⁰⁰⁰·½): 먼저 0.5·df를 하면 서브노멀 df의 홀수 끝 비트에서 반올림돼 `chndtrc(5e-324, 5e-324, 1e-300)`이 0이었다(참값 1.71e-321). `chdtrc`도 끝 비트가 홀수인 서브노멀 v에서 반이 double이 아니라 Q(v/2, x/2) = ½Q(v, x/2)(a → 0에서 선형, 상대 O(a))로 둔다(`chdtrc(1.5e-323, 1e-300)`이 6.8e-321이었다, 참값 5.12e-321; 전에는 v = 5e-324만 그랬다). Arb(4000비트) 대비 절대 2⁻¹⁰⁷⁴ 한 단위 안이다(2026-11-19). `ncfdtr`·`ncfdtrc`·`ncfdtri`는 dfd/2가 2⁻¹⁰⁰⁰ 이상이면 dfn = 5e-324(반이 0으로 반올림)를 1e-323으로 바꿔 계산한다: 첫 베타 모양 a = 0이 혼합 합에서 NaN을 냈다(w가 아주 작지 않은 f ≈ 1e23–1e173; 그 밖은 j = 0 항의 꼴이라 값). 값은 j = 0 항의 1 − I_w(a, b) ≈ a|log w|만큼만 움직이고, 그것은 절대 2⁻¹⁰⁶³ 아래다(2026-11-19).
- **아주 작은 모양** (2026-11-22, Q55):
  - 덩어리 합과 float64 걷기는 베타 모양 b > 0만 받는다(`kMinShape`). dfd = 5e-324의 반은 0이고, b = 0을 `first_factor`가 감마 족의 표시로 읽어 `ncfdtr(5e-324, 5e-324, 1, 0.25)`가 e^(−½), `ncfdtrc`가 1이었다. b = 0은 옛 걷기다. 정확한 서브노멀 반(2⁻¹⁰²² 등)은 그대로 이 길들이 받는다(옛 걷기는 거기서 NaN이었다).
  - `ncfdtr`, dfd/2 < 2⁻¹⁰⁰⁰: j ≥ 1 항은 I_w(a + j, b) ≤ b(−log(1 − w)) ≤ 2⁻¹⁰⁰⁰·2100이라 합은 e^(−μ)·fdtr(dfn, dfd, x)이고(절대 2⁻⁹⁸⁸ 안; 값이 2⁻⁹³⁰ 위이면 상대), 위꼬리는 e^(−μ)·fdtrc + (1 − e^(−μ))다. 이때 dfn을 바꾸지 않아 nc = 0의 fdtr와 이어진다(`ncfdtr(5e-324, 5e-324, μ, f)` = e^(−μ)/2).
  - `ncfdtr`, dfn/2 < 2⁻⁷²: j ≥ 1 항의 a는 상대 a·|log w| ≤ 2100a < 2⁻⁶¹만 움직이므로 합은 j = 0 항 e^(−μ)·fdtr(·fdtrc)과 모양 1에서 시작하는 혼합 Σ_{j≥1} D(j, μ)I_w(j, b)(무게 D(j + 1, μ), h = 1)이다. w < 2⁻¹⁰⁰⁰이면 j = 0 항만이다. 두 쪽을 다 구하고 물은 쪽은 작은 쪽의 보수다. `ncfdtrc(1e-300, 100.5, 1e-300, 0.001)`이 NaN이었고 이제 Arb와 같다.
  - w < 2⁻¹⁰⁰⁰의 F 위꼬리는 1 − 아래꼬리가 아니라 e^(−μ)·fdtrc + (1 − e^(−μ))다(1/64 아래에서 자릿수를 잃어 NaN이었다: `ncfdtrc(1e-10, 1, 1e-300, 1e-300)`, 참값 3.6e-8).
  - `nctdtr`, df < 2⁻⁷¹: T = (Z + δ)/S에서 S = √(V/df)가 s보다 클 확률은 (df/2)log(2/(df s²))쯤이고, t가 의미를 갖게 하는 모든 s(> 1/1.8e308)에서 1083·df 아래다. 그래서 P(T ≤ t)는 t와 상관없이 Φ(−δ)에서 1083·df 안이고, 그것이 min(Φ(δ), Φ(−δ))의 2⁻⁶⁰ 아래이면(또는 둘 다 2⁻¹⁰⁰⁰ 아래) Φ(−δ), 위꼬리 Φ(δ)다. 그 밖(예: df = 1e-200, δ = 30, t = 10: 참값 7.2e-198, Φ(−30) = 4.9e-198)은 합이다. 역함수는 이 극한에서 ±inf나 0이다(분포가 t에 따라 변하지 않는다).
  - 옛 걷기(`BetaFamily::Steps`)의 아래로 걷는 셈 s + b는 s0 + b로 내려가며 걸음의 1들보다 훨씬 작아질 수 있어(높은 부분이 0이 된다) 나누기 전에 정규화한다(`ncfdtr(1e-100, 1e-100, 3000, 0.25)`가 NaN이었다).
  - 짝: df 5e-324–1e-3, nc 5e-324–3000, x 1e-300–1e300의 격자에서 cdf + ccdf는 모두 1에서 4.5e-16 안이다. 남은 한계는 t의 df 1e-200–1e-10, δ = ±30, t = ±1e300이다: 작은 쪽은 NaN, 큰 쪽은 0.99999999999999967이다. 작은 쪽은 df·log 크기의 값인데 그것을 내는 길이 없다.

## 6. 정답지와 측정
- **기준값**: `tools/special_ref.py`가 Arb(python-flint)로 기준값을 만들어 `tests/special/ref/*.npz`에 커밋한다. 테스트는 Arb를 매번 부르지 않는다. python-flint는 이 도구와 일부 테스트의 의존성이다 (`pyproject.toml`의 `test`).
  - **엄밀한 공**: 값은 참값을 반드시 품는 공 [중심 ± 반지름]이다. 반지름이 크기의 2⁻¹¹⁰ 아래가 될 때까지 정밀도를 두 배씩 올린다(`tools/arbref.py` `rigorous`, 최대 8192비트). 그러면 hi + lo는 참값을 반올림한 것이다. 정착하지 않으면 그 점을 뺀다. 2⁻¹⁰²² 아래 값은 절대 2⁻¹¹³²까지만 정착시킨다(서브노멀의 반올림을 정하기에 충분하다). 복소수는 작은 쪽 부분을 2⁻¹²⁰|f|까지만 정착시킨다(계약이 |오차|/|f|이므로).
  - **우리가 정하는 자르기**: Arb가 스스로 보증하지 못하는 것은 우리가 정한 급수의 자르기다. 그 꼬리의 상한(기하급수 상한, 부분적분의 나머지 등)을 공의 반지름에 더한다(`add_error`, `geometric_tail`). Arb의 `gamma_lower`, `beta_lower`, `hypgeom_1f1`이 큰 매개변수(a ≳ 1000, a, b ~ 1e4)에서 반지름을 줄이지 못하는 곳은 꼬리 상한을 붙인 급수로 직접 썼다.
  - **역함수**: 근 x는 x(1 ± 2⁻¹¹⁰)에서 부호가 바뀌는 것으로 인증한다(`certify_root`). Newton은 공의 중심에서 하고, 인증이 안 되면 정밀도를 올린다.
  - **mpmath에서 바꾼 이유** (2026-11-16): mpmath의 규칙("두 정밀도가 30자리까지 일치")은 두 정밀도가 함께 틀린 값을 받아들였다. 바꾸면서 찾은 옛 참값의 오류: `betaincinv` 41점·`betainccinv` 119점은 mpmath가 조용히 실패해 우리 함수의 출발값이 참값으로 들어가 있었다(I_x/p − 1 = 1.7e-12). `zeta_hurwitz` 14점(7.3e-28, mpmath는 300자리에서야 맞음), `lambertw` 4점(허수부 lo), 리덕션 17점(1 + e⁻²⁰⁵를 잃음), 오래된 `wofz` 입력 64점. 계수 표 `zeta_minus_one`의 j = 94–110 lo(1e-28 상대, ζ(j) − 1의 상쇄)도 고쳤다. 다른 계수 표 7개는 바이트까지 같았다. 또 mpmath가 버렸던 점 약 900개를 이제 얻는다(`betainccinv` +248, `nctdtrit` +150 등). 메모리도 일정하다: mpmath의 quad는 구간 끝점마다 노드를 캐시해 140GB까지 쌓였다.
  - **새 참값이 드러낸 라이브러리 오차**: `gammainccinv`(a = 9.5e5, q = 3.3e-33)가 1.21e-14였다. Boost의 `gamma_q_inv`는 Halley 반복을 자릿수의 절반에서 멈추므로 큰 a에서 100 ULP까지 남는다. 이제 Boost의 근에서 우리 P, Q로 Halley 한 걸음을 더 한다(`igam_polish`, 오차 e가 e³이 된다). 참값 최대가 `gammaincinv` 9.3e-15 → 7.3e-16, `gammainccinv` 1.2e-14 → 2.4e-15가 됐다. 1백만 원소 비용은 10 ms 안팎 그대로다.
  - 입력 집합: 함수마다 보장 영역의 격자 + 로그 균등 무작위 + 경계(구간 경계의 앞뒤 ULP, 극점, 0, 정의역 끝)
- **정확도 보고**: `python -m bench.accuracy`가 참값 파일의 모든 점에서 잰 기준선(`bench/accuracy/results/special.json`)으로 [bench/special/accuracy.md](../../bench/special/accuracy.md)의 표를 만든다(bench/README.md). 함수마다 보장 영역·느슨한 영역·혼합 척도 영역의 최대·99.9% ULP(`ulp_distance`: 두 값 사이에 있는 double의 수), 역함수의 ULP/max(1, κ), 혼합 척도, 계약 대비를 낸다. `run --compare`는 scipy·R·Boost.Math의 오차를 같은 점, 같은 척도로 재어 라이브러리별 열에 둔다.
- **성능**: scipy와 같은 자리에서 번갈아 잰다 (통합 때의 `sameseat` 방식). 1스레드와 10스레드, 원소 10⁶개와 스칼라 1개.
- **비트 테스트**: NEON 경로 = 스칼라 정의, 스레드 1·3·10이 같은 비트, `f(x)[i] == f(x[i])`. 특수값의 scipy 비트 비교에서 `log_expit`의 유한한 값은 2 ULP 허용이다(vForce, Q31). `jv(0, x)`의 영점 옆 double은 0이 아니라 Arb 값 ± 1e-14·포락선으로 본다(차수 0은 유리식, Q21)

## 7. 기준선 (scipy 1.18.1, 원소 10⁶개, 1스레드)
| 함수 | ms | 함수 | ms |
|---|---|---|---|
| erf | 9.8 | gammainc | 63 |
| erfc | 7.4 | gammaincc | 63 |
| erfcx | 16.3 | gammaincinv | 245 |
| erfinv | 13.3 | betainc | 201 |
| ndtr | 10.4 | betaincinv | 744 |
| log_ndtr | 16.9 | expit | 5.7 |
| ndtri | 7.0 | logit | 6.7 |
| gamma | 15.2 | xlogy | 2.6 |
| gammaln | 14.2 | logsumexp (1000×1000, axis=1) | 8.0 |
| digamma | 11.8 | i0 | 12.1 |
| beta | 47.6 | j0 | 3.2 |
| betaln | 48.5 | zeta | 84 |

- scipy.special의 원소별 함수는 1스레드다. 계산이 무거운 함수는 풀(P코어 10)만으로도 수 배 차이를 기대한다. 메모리에 묶이는 가벼운 함수(xlogy, expit)는 그렇지 않다.

---

## 8. 마일스톤
- 마일스톤 S0–S9와 진행 원칙은 [ROADMAP.md](../../ROADMAP.md)의 "특수함수 영역 (sd.special)"에 있다. 진행 상태도 그곳에 적는다.
- 나중의 `sd.stats`는 S4가 끝나면 설계를 시작할 수 있다 (분포의 cdf·ppf가 S2–S4에 달려 있다).

---

## 9. 해설의 주제
- 유리 근사와 minimax: 계수는 어디서 오고, 왜 구간을 나누는가 (S2)
- 보수함수와 log 공간: 1 − f가 왜 틀리는가 (S1, S2)
- Lanczos 근사와 앞 인자: 넘치지 않게 식을 모으기 (S3, S4)
- 급수와 연분수의 수렴 영역, Lentz 방법 (S4)
- 역함수: 초기값의 질과 Halley 반복 횟수 (S2, S4)
- Euler–Maclaurin, Bernoulli 수 (S3)
- AGM과 Carlson 대칭형: 같은 적분의 두 길 (S5)
- 두 패스와 한 패스 softmax (S5)
- Chebyshev 전개, Hankel 점근, Temme 방법과 Steed 방법 (S6)
- 선형 변환과 퇴화 극한 (S7)
- 성능: 원소당 비용 모델로 정한 병렬 기준, 분기 없는 NEON, Horner와 Estrin (S0–S3)

---

## 10. 마일스톤을 시작하기 전에 물을 것
결정 Q16–Q19는 §1.1에 있다. 아래는 측정이나 목록이 있어야 정할 수 있어서, 해당 마일스톤을 시작하기 전(또는 측정 뒤)에 사용자에게 묻는다.
- **매 마일스톤**: 특수값을 scipy와 다르게 할 곳의 목록 (§2.2), 함수마다의 보장 영역 (§2.3)
- **S0**: `kMaxOperands` 5의 비용 측정 결과 (§3.1)
- **알고리즘 후보**: §1.3의 규칙으로 정할 수 없거나 보장 영역을 줄여야 할 때 (예: §5.2의 `gamma` 후보 A/B가 정확도에서 갈리지 않고 속도만 다를 때는 규칙으로 정한다)
- **미정: 비중심 순방향의 남은 속도 차**. 참값 혼합에서 `ncfdtr`은 R의 약 0.3배, `nctdtr`은 scipy의 약 0.46배다(Q51–Q54 뒤). R의 `pnbeta`는 절대 1e-9에서 멈추는 AS 226/R84라 그만큼의 정확도를 내주면 걸음이 약 2/3로 준다. 그 밖의 길: float64 걷기의 8개 비율 블록을 NEON 두 레인으로(추정 1.3–1.5배), 덩어리 합(μ < 20)을 float64로 비율을 한 번만 반올림해(추정 1.5배). 앞의 것은 정확도 결정이라 묻는다.
- **미정: 불완전 베타의 float64 앞 인자**. `betainc`·`bdtr`·`fdtr`·`stdtr`는 R의 약 0.6–0.75배다. BFRAC 시간의 약 절반이 double-double 앞 인자(dd 로그 2–4개)다. float64 앞 인자는 오차가 지수 E에 비례해(꼬리에서 E·eps) 꼬리의 상대 정확도를 깨므로(가드레일) 하지 않았다. 몸통(E ≤ 몇)에서만 float64를 쓰는 것은 측정이 남았다.
- **미정: `hyp2f1`의 긴 합**. z → 0.9에서 양항 보정 루프는 항마다 약 4 사이클(합의 덧셈 사슬과 나눗셈 하나)이다. 짝수·홀수 항의 두 사슬과 두 항마다 나눗셈 하나로 약 2배를 기대한다. 1 − z 변환은 두 항의 상쇄 때문에 정확도를 잃을 수 있어 하지 않았다.
- **J·Y float64 경로의 차수 경계** (Q21, Q61). 사용자 결정 2026-11-22: 지금대로 둔다(500). 이유: 1100으로 올리면 참값 파일의 점에서 `jv`·`yv`가 약 4배 빨라지지만(차수 1000 안팎의 double-double 점화가 빠진다) 최대 포락선 ULP가 `jv` 13 → 29, `yv` 15 → 41이 된다(계약 대비 0.18 → 0.40, 0.25 → 0.56, 계약 안). 손실이 '몇 ULP'를 넘는다. 차수 n의 float64 점화는 약 n/30 ULP를 잃는다(700에서 15.4, 21 ULP).
