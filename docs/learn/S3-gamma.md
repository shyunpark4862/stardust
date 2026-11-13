# S3. 감마, 베타, 제타

> `gamma rgamma gammaln digamma polygamma beta betaln binom zeta` 아홉 개다. 설계는 docs/design/special.md §5.2, §5.6, API는 docs/api/special.md §3, §7이다. 핵심부는 `csrc/special/gamma_core.hpp`다.

## 1. 결과
| 항목 | 결과 |
|---|---|
| 테스트 | 3,435개 통과 (S2의 3,394 + 41). 골든 출력 세 영역 불일치 0 |
| 정확도 (보장 영역, 최대 ULP) | `gamma` 6.2, `rgamma` 5.1, `gammaln` 5.5, `digamma` 2.1, `polygamma` 17, `beta` 9.0, `binom` 9.3, Riemann `zeta` 24, Hurwitz 5.5 |
| 혼합 한계 영역 (오차/max(1,\|값\|), 2⁻⁵² 단위; 한계는 45) | `gammaln` 2.1, `digamma` 15, `polygamma` 3.1, `beta` 6.8, `betaln` 7.6, `binom` 8.0, Hurwitz 6.2 |
| scipy의 같은 표 | `beta` 3.1e8, `binom` 2.8e4, Riemann `zeta`의 음수 쪽 5e15(서브노멀·오버플로 처리), `gammaln` 1 근처 2.9e5 ULP |
| 속도 (10⁶개, scipy / sd) | 1스레드 0.18–1.24배, 10스레드 1.3–9.1배 (§6) |
| 컴파일 경고 | 없음 |

사용자 결정 (2026-11-14, S3를 시작하기 전)
- scipy가 틀린 값을 내는 다섯 갈래를 참값으로 한다(docs/api/special.md §3의 굵은 줄).
- `binom`은 k가 0 이상의 정수이면 모든 n에서 표준 정의 n(n−1)…(n−k+1)/k!다.
- 보장 영역: 상대 1e-14는 `gamma` `rgamma` Riemann `zeta`의 전체, `gammaln` `digamma` `polygamma`의 x > 0, `beta`의 a, b > 0, Hurwitz의 q > 0, `binom`의 n ≥ 0이다. 그 밖과 `betaln`은 |오차| ≤ 1e-14·max(1, |값|)다.

같은 원칙(참값)으로 처리했고 결정 목록에 없던 것 (확인 부탁)
- `rgamma(x)`의 양수 서브노멀 꼬리(171.6 < x < 177.8): scipy는 0이다. 결정 (3)의 음수 꼬리와 같은 갈래로 보았다.
- `beta(inf, 1e300)`은 0이다(scipy NaN). `binom`에서 k가 거대한 경우: `binom(-0.5, 1e300)`은 5.6e-151(scipy −5.1e-151), `binom(0, 1e300)`은 0(scipy −2.9e-301), `binom(-2.5, 1e300)`은 +inf(scipy −inf)다.

## 2. 인자를 반올림하지 않는다
- **문제**: Γ의 인자에 대한 상대 민감도는 ψ(z)·z다. 인자를 float64로 반올림하면 z log z / 2 ULP를 잃는다. z = 170이면 870 ULP다. 그래서 1 − x, a + b, n − k + 1 같은 합을 만들어 Γ에 넘기면 안 된다.
- **반사식**: `Γ(1−x)` 대신 `Γ(x)Γ(−x) = −π/(x·sinpi(x))`를 쓴다. −x는 정확하다. ψ도 `ψ(1−x)` 대신 `ψ(−x) − 1/x`를 쓴다.
- **점화식**: `Γ(v+1)`은 `v·Γ(v)`로 계산한다(binom).
- **꼭 만들어야 하는 합**은 TwoSum으로 hi + lo를 들고 1차로 보정한다: `Γ(hi+lo) ≈ Γ(hi)·(1 + ψ(hi)·lo)`. a + b(beta)와 n − k + 1(binom)이 그렇다.
- **sinpi, cospi**: `fmod(x, 2)`는 정확하다. 그 뒤 [−½, ½]로 접는 빼기도 Sterbenz로 정확하다. 그래서 극점 옆에서도 sin(πx)의 상대 정확도가 산다.
  - lo가 소수부 전체를 들고 있을 때(`binom(-0.5, 1e300)`에서 n − k + 1 = −1e300 + 0.5)는 1차 근사가 맞지 않는다. 그래서 정확한 항등식 `sinpi(hi)cospi(lo) + cospi(hi)sinpi(lo)`를 쓴다.

## 3. Γ를 가수와 지수로 나눠 들고 다닌다 (scaled)
- Γ(z)는 z = 171.6에서 넘친다. 그러나 Γ로 만드는 몫(1/Γ의 서브노멀 꼬리, beta, 음수 쪽 zeta, n!·ζ)은 보통의 수다.
- 그래서 m·2^e로 들고 다니다가 마지막에 한 번 반올림한다(`Scaled`, `value`). 큰 거듭제곱과 지수도 같은 방식으로 나눈다.
  - `pow_scaled(a, t)`: a = m·2^ea(m ∈ [1/√2, √2))이면 a^t = m^t·2^(ea·t)다. ea·t는 FMA로 정수와 소수로 정확히 나눈다. m^t는 범위 안에 들도록 같은 크기의 조각 k개로 계산한다.
  - `exp_scaled(x)`: Cody–Waite 방식이다. log 2를 hi + lo 두 부분으로 들고 2^n·e^r로 나눈다.
- **Lanczos 형(후보 A)**: `Γ(z) = L(z)·zgh^(z−½)·e^(−zgh)`. zgh = z + g − ½도 hi + lo로 들고 다닌다. 그렇지 않으면 반올림이 (z − ½)배로 커진다(S2의 x² 문제와 같다).
- **beta의 큰 인자** (a > 7000):
  - exp(log B)는 지수가 수백이라 오차를 수백 배로 키운다. 처음 짠 것은 2e-13이었다.
  - 그래서 `Γ(a)/Γ(c) = L(a)/L(c)·e^E·cgh^(−b)`로 정리했다. 여기서 `E = b(b+g)/cgh + (a−½)(log1p(u) − u)`, u = −b/cgh다.
  - B가 표현 가능한 곳에서는 E가 1보다 작다. `log1p(u) − u`는 |u|가 작으면 급수로 계산해 상쇄를 피한다.
- **polygamma의 큰 n**: n!·ζ(n+1, x)에서 ζ가 먼저 언더플로한다(n = 100, x = 19416). 그래서 Hurwitz의 각 항을 첫 항 q^(−s)에 대한 비로 더하고, q^(−s)는 scaled로 한 번만 곱한다.

## 4. gamma: 후보 A와 B (§1.3의 규칙)
| 후보 | x > 0 최대 ULP | 99.9% | 속도 (10⁶개, 1스레드) |
|---|---:|---:|---:|
| A Lanczos (Boost `lanczos13m53`, scaled) | 6.5 | 4.9 | 31.7 ns |
| **B Cephes** (점화 + [2, 3] P/Q, 33부터 Stirling) | **4.4** | **4.0** | **8.2 ns** |

- B가 정확도에서도 속도에서도 앞서서 B를 골랐다(규칙 1, 3번). 구간 수는 둘 다 한도 안이다.
- A는 버리지 않았다. 오버플로를 넘어가야 하는 경로(scaled)는 A의 꼴이 자연스럽다. 비교용으로 `_gamma_lanczos`를 남겼다.

## 5. 함수별로 고른 것
- **`gammaln`** (x > 0): 근사는 둘이다.
  - 2 둘레의 Taylor 급수: `lgamma(2+e) = (1−γ)e + Σ (−1)^k (ζ(k)−1)/k·e^k`. 계수가 2⁻ᵏ로 줄어 |e| ≤ ½에서 28항이면 된다. 계수는 맞춘 값이 아니라 급수 자체의 값이다(mpmath로 계산).
  - Stirling: x ≥ 13
  - 1 근처는 `lgamma(2+e) − log1p(e)`(e = x − 1, 정확)로 계산한다. log Γ가 0인 1과 2 근처에서 상대 정확도를 낸다. scipy는 x ≈ 1에서 2.9e5 ULP다.
  - Cephes의 [2, 3] 유리식은 쓰지 않게 됐다.
- **`digamma`**: Cephes `psi`와 Boost의 [1, 2] 유리식이다. 양의 근 x₀ = 1.4616…을 인수로 뺀 꼴이라 근 옆에서도 상대 정확도가 산다. (−1, 0)은 `ψ(x+1) − 1/x`로 계산해 0 근처의 상쇄를 피한다.
- **`zeta`**
  - Riemann x ≥ 0: Euler–Maclaurin(q = 1)
  - x < 0: 함수방정식 `(2π)^x/π·sinpi(x/2)·yΓ(y)·ζ(1+y)`, y = −x. (2π)^x는 log(2π)를 두 부분으로 들고 FMA로 지수의 오차까지 잡는다.
  - 0 < x < 1의 24 ULP: 부분합과 꼬리 적분이 약 20배 상쇄해서 생긴다. 계약 안이라 그대로 두었다.
  - mpmath의 Hurwitz zeta는 40자리에서 q가 크면 틀린다(1e-9). 그래서 기준값 도구를 바꿨다(§7).
- **`binom`**
  - 정수 k < 20: scipy와 같은 곱(num/den)이다. 0의 부호까지 맞추려는 것이다.
  - 음의 정수 n(사용자 결정): `(−1)^k binom(k − n − 1, k)`
  - 그 밖: `Γ(n+1)/Γ(k+1)·(1/Γ(n−k+1))`. n이나 k가 거대하면 반사식과 Γ 비율을 쓴다.
- **NEON**: Q5의 대상인 `gammaln`은 NEON으로 짜지 않았다. 구간이 넷이고, S2에서 분기 방식이 고르기보다 빠르거나 같다는 측정이 있어서다. 스칼라 libm 정의다(설계 §3.4에 적음).

## 6. 속도 (`bench/special/s3/speed.md`, 10⁶개, ms)
| 함수 | sd 1스레드 | sd 10스레드 | scipy | scipy / sd 1 | scipy / sd 10 |
|---|---:|---:|---:|---:|---:|
| `gamma` | 18.0 | 2.45 | 16.4 | 0.91 | 6.70 |
| `rgamma` | 20.4 | 2.74 | 17.1 | 0.84 | 6.22 |
| `gammaln` | 10.9 | 1.47 | 12.1 | 1.11 | 8.24 |
| `digamma` | 8.92 | 1.21 | 11.0 | 1.24 | 9.13 |
| `polygamma` | 117 | 16.5 | 119 | 1.02 | 7.19 |
| `beta` | 59.7 | 8.42 | 32.4 | 0.54 | 3.85 |
| `betaln` | 78.7 | 11.0 | 25.8 | 0.33 | 2.35 |
| `binom` | 99.2 | 13.8 | 44.5 | 0.45 | 3.21 |
| `zeta` (Riemann) | 89.3 | 12.4 | 16.2 | 0.18 | 1.31 |
| `zeta` (Hurwitz) | 86.2 | 12.0 | 85.9 | 1.00 | 7.19 |

- **1스레드에서 scipy보다 느린 함수가 있다.**
  - `beta` `betaln` `binom`: 정확도 장치(TwoSum 보정의 ψ, scaled 산술, Γ 비율) 값이다. scipy는 같은 입력에서 최대 3e8 ULP가 틀린다.
  - `gamma` `rgamma`: 입력의 절반이 극점 근처의 음수라 `sinpi`(libm sin)가 붙는다. 양수만이면 8.2 ns다.
  - Riemann `zeta`: scipy는 x > 1에서 Cephes의 구간별 유리식(`zetac`)을 쓴다. 우리는 항마다 `pow`를 부르는 Euler–Maclaurin이다. 5.5배 느리다.
- 흔한 경우의 지름길: `rgamma`과 음수 `gamma`는 |x| ≤ 170이면 scaled 없이 직접 나눈다. `beta`는 a + b ≤ 170이면 Cephes Γ 셋으로 계산한다(89 → 60 ns, 정확도 그대로).
- 10스레드에서는 모두 scipy보다 빠르다(1.3–9.1배).

## 7. 정답지 도구가 틀린 곳
- **mpmath는 자릿수를 조용히 잃는다.**
  - `beta(98, 4.4e288)`: 40자리에서는 a + b가 98을 잃는다. 기준값이 350으로 나왔다(참값 −64791).
  - Hurwitz zeta(22.2, 276.8): 40자리에서 1e-9가 틀린다.
- **고친 것** (`tools/special_ref.py`의 `evaluate`)
  - 시작 자릿수를 인자의 십진 지수만큼 올린다(50 + log10|인자| + 10).
  - 두 배 자릿수와 30자리까지 같아질 때까지 두 배로 늘린다.
  - 처음에는 두 배 비교만 했는데, 50자리와 100자리가 둘 다 b를 잃어 "합의"하는 일이 있었다.

## 8. 다음 (S4 불완전 함수) 전에 물을 것
- docs/design/special.md §10에 따라 S4 함수(`gammainc gammaincc betainc betaincc`와 역함수 넷)의 scipy와 다른 특수값, 보장 영역을 묻는다.
- 불완전 베타에 세 번째 구간(큰 a, b의 Temme 전개)을 넣을지는 측정 뒤 묻는다.
