# S8. 복소수

> `wofz`, `loggamma`(새 함수)와 `erf erfc erfcx lambertw`의 complex 입력이다. 설계는 docs/design/special.md §5.13, API는 docs/api/special.md §12다. 코드는 `csrc/special/ops/wofz.hpp`, `loggamma.hpp`, `lambertw.hpp`다.

## 1. 결과
| 항목 | 결과 |
|---|---|
| 테스트 | 3,597개 통과 (S7의 3,564 + 33). 골든 출력 세 영역 불일치 0 |
| 정확도 (참값 대비 최대 ULP, 크기 기준, 영점 쪽은 포락선 배율로 나눔, `bench/special/s8/accuracy.md`) | `wofz` 2.0, `erf` 3.4, `erfc` 3.4, `erfcx` 1.9, `loggamma` 8.2, `lambertw` 0.9–1.9(k = 0, ±1, 4) |
| scipy의 같은 표 | `wofz` 255, `loggamma` 224, `lambertw` k = 0, ±1에서 6.5e7–1e8(분기점 근처), `erf` 계열은 참값이 유한한 곳에서 inf나 NaN을 낸다 |
| 속도 (10⁶개, scipy / sd) | 1스레드 `wofz`·`erf` 계열 0.84–0.94배, `loggamma` 0.69배, `lambertw` 1.4–2.3배. 10스레드 5.7–20배 (§6) |
| 컴파일 경고 | 없음 |

사용자 결정 (2026-11-14, S8을 시작하기 전)
- 복소수의 상대오차는 크기 기준 |오차| / |값|이다(성분별이 아니다).
- `erf erfc erfcx wofz`의 영점 쪽은 포락선 기준이다. 계산식 f = A − B의 두 항 크기로 |오차| ≤ 1e-14·(|A| + |B|): `erfc`(Re z < 0)는 2 − erfc(−z), `erf`(|z| ≥ 2)는 1 − erfc(±z), `wofz`(Im z < 0)는 2exp(−z²) − w(−z), `erfcx`(Re z < 0)도 같은 꼴. 영점이 없는 쪽은 상대.
- `loggamma`는 S3의 `gammaln`과 같은 꼴: Re z > 0은 상대, Re z ≤ 0은 |오차| ≤ 1e-14·max(1, |f|).
- scipy가 틀리거나 NaN을 내는 특수값은 참값·극한(`lambertw(5e-324, k≠0)`, 무한대 인자의 극한, −1/e에 가장 가까운 double의 복소 `lambertw`, `erf(1e300+1e300j) = 1`). 그 밖은 scipy를 따른다(가지 자름 위 ±0j는 C99식 위·아래 극한, `lambertw(0, k≠0) = −inf+0j`). `loggamma`의 극점은 inf + nan·i.

## 2. wofz
- **scipy를 먼저 쟀다**: scipy는 S. G. Johnson의 Faddeeva 패키지(연분수 + Zaghloul–Ali Algorithm 916)다. 아래쪽 반평면의 최대 447 ULP는 exp(−z²)의 위상 탓이지만, 위쪽 반평면에서도 |z| ≈ 6이나 실수축 근처 x ≈ 8에서 88–178 ULP였다. 그대로 따라 하면 1e-14(45 ULP)를 못 지킨다.
- **격자 Taylor**: 0 ≤ Re z ≤ 7, 0 ≤ Im z ≤ 4는 0.1 격자의 가장 가까운 점 z₀ 둘레 Taylor 급수다. 계수를 표에 두지 않고 w′ = −2zw + 2i/√π의 재귀 a_{n+1} = −(2z₀aₙ + 2a_{n−1})/(n+1)로 만들므로, 표에는 w(z₀)만 double-double로 2,911점 있다(`tools/gen_wofz_grid.py`, mpmath로 잰 함수의 값이고 맞춘 값이 아니다).
  - 재귀의 반올림 오차가 커지지 않을까 걱정했다. 동차해는 exp(−z²)꼴이라 z₀ 둘레에서 그 크기 비는 |exp(−2z₀h)| ≤ e^{1.14}(|h| ≤ 0.0707)에 그친다. exp(−z₀²)가 아무리 커도 비율로는 상관없다. 잰 최대는 2.5 ULP였다.
- **연분수의 항 수**: Johnson이 맞춘 Poppe–Wijers 식 그대로면 76 ULP, 2를 더하면 5.0, 4를 더하면 2.8 ULP였다. 4를 더했다.
- **큰 |z|**: 처음에는 x + y > 1e7에서 첫 항 i/(√π z)만 썼다. 다음 항이 1/(2z²)이라 |z| ≈ 7e6에서 87 ULP였다(scipy도 같은 87.8). 3e8로 올렸다.
- **exp(−z²)의 위상**: −2xy를 two_prod로 정확히(hi + lo) 만들고 cos·sin을 두 조각으로 합쳤다. libm의 sin·cos는 큰 인자도 정확히 축소한다. |xy|가 double을 넘으면 위상은 NaN이다.
- **오버플로**: 크기를 m·2ᵉ로 두고 곱한 뒤에 한 번만 내린다. exp(−z²)가 넘쳐도 w(iz)가 작으면 erfc가 유한하다. 처음에는 (y−x)(y+x)를 double-double로 먼저 만들어서 |x| ≈ 1e300에서 inf − inf가 NaN이 됐다. 범위를 double로 먼저 가르고, 넘치는 쪽은 지수를 1e6으로 둬서 value()가 0은 0으로, 나머지는 ±inf로 내리게 했다.

## 3. erf, erfc, erfcx
- `erfcx(z) = w(iz)`, `erfc`는 Re z ≥ 0에서 exp(−z²)·w(iz), Re z < 0에서 2 − erfc(−z), `erf`는 |z| < ½에서 Taylor, 그 밖은 1 − erfc다. 정확도는 포락선으로 나눠 1.9–3.4 ULP다.
- **허수축**: erf(30i)를 1 − erfc(30i)로 계산하면 실수부가 1이 됐다(참값 0). erfc(30i)의 실수부 1은 w(−30)의 실수부 e^{−900}(언더플로)와 e^{900}의 곱이라 사라진다. 크기 기준으로는 오차가 아니지만, 축 위의 값은 따로 계산해 0인 부분을 정확히 0으로 낸다.
- **0의 부호**: scipy의 실수축 위 허수부 부호는 일관되지 않다(erfc(0+0j)는 −0, erfc(0.5+0j)는 +0). 비트까지 맞추지 않고 도함수의 방향(Im f(x + iy) ≈ y·f′(x))을 따른다.

## 4. loggamma
- scipy와 같은 구조(Stirling, 반사, 점화식, 1·2 둘레의 Taylor 원판)로 먼저 만들었다. 1–2 사이의 넓은 구역(Re z 0.3–4, |Im z| < 2.6)에서 최대 293 ULP였다(scipy 439). 점화식은 loggamma(z + n)(Stirling, 크기 13 안팎)에서 log(z(z+1)…)를 빼는데, 값이 0.1 정도면 100배 넘게 상쇄한다.
- **극을 빼낸 급수**: loggamma(1 + w) = −γw + Σ_j μ(w/j) + Σ_k (−1)^k ζ(k, J+1)/k·w^k (μ(t) = t − log(1+t)). ζ(k) = Σ_{j≤J} j^{−k} + ζ(k, J+1)로 나누고 앞부분을 닫힌 꼴로 더한 것이다. μ는 O(t²)이고 각 부분이 서로 상쇄하지 않는다. 남은 급수는 |w| < J+1에서 수렴한다. 2의 영점은 log1p(z−2) + loggamma(z−1)로 한 단계 내려서 1차까지 정확하다.
  - J = 4, |Im z| ≤ 2로 넣자 최대 20.5 ULP였고 남은 최대가 구역 바로 바깥(|Im z| ≈ 2.1)이었다. J = 6, |Im z| ≤ 3으로 넓혀 8.2 ULP가 됐다.
  - 처음에는 μ의 급수를 Horner로 쓰면서 부호를 두 번 뒤집어 1e15 ULP가 나왔다. 계수 (−1)^m/m에 t·s를 더해야 했다.
- double-double 판(Stirling과 곱의 로그)도 생각했다. 편각을 double-double로 얻으려면 double-double sin·cos가 필요해서 이 방법을 골랐다.

## 5. lambertw (complex)
- 분기점 근처는 S7의 급수(p = √(2(ez + 1)), e·z + 1을 double-double로)이고, k = −1은 위쪽, k = 1은 아래쪽에서 p의 부호를 바꾼다. 제곱근은 Kahan 식으로 0의 부호를 지킨다. 그 밖은 Halley 반복이다.
- 초기값 log1p(z)를 |z| < 1에만 썼더니 z = 1.03 − 0.11i에서 다른 가지로 수렴했다(점근식 L₁ − L₂는 |log z|가 작아 나쁘다). |z| < 3, Re z > −½로 넓혔다.
- |z| > 1e150이나 (k ≠ 0에서) |z| < 1e-150은 w·eʷ가 넘치거나 사라지므로 w + log w = log z + 2πik에 Halley를 쓴다. 그 크기에서 |W| > 300이라 로그의 가지 문제가 없다.
- **(−1/e, 0) 위의 x − 0j, k = −1**: C99식으로 아래쪽 극한(= W₁ 위쪽 값의 켤레)이다. scipy는 이곳에서 실수(위쪽 값)를 낸다. 참값 파일은 mpmath에 부호 있는 0이 없어서 conj(W_{−k}(x))로 만든다.

## 6. 측정
속도 (`bench/special/s8/speed.md`, 10⁶개, 표본은 참값 입력, ms)

| 함수 | sd 1스레드 | sd 10스레드 | scipy | scipy / sd 1 | scipy / sd 10 |
|---|---:|---:|---:|---:|---:|
| `wofz` | 56.0 | 7.6 | 50.1 | 0.89 | 6.59 |
| `erf` (complex) | 46.5 | 6.5 | 38.9 | 0.84 | 6.03 |
| `erfc` (complex) | 57.2 | 7.3 | 49.6 | 0.87 | 6.76 |
| `erfcx` (complex) | 47.3 | 6.7 | 44.4 | 0.94 | 6.65 |
| `loggamma` | 110.6 | 13.4 | 76.7 | 0.69 | 5.72 |
| `lambertw` k = 0 | 136.3 | 15.8 | 193.2 | 1.42 | 12.2 |
| `lambertw` k = −1 | 128.4 | 14.7 | 295.2 | 2.30 | 20.1 |
| `lambertw` k = 1 | 135.4 | 15.9 | 280.6 | 2.07 | 17.6 |

- `wofz` 계열이 1스레드에서 조금 느린 것은 격자 Taylor의 20항과 exp(−z²)의 double-double 위상 탓이다. `loggamma`는 1–2 둘레의 μ 여섯 개(log1p와 atan2)가 비싸다.
- 스칼라 한 번 호출은 0.16–0.40 µs(scipy 0.11–1.1 µs)다. 복소수는 입력 dtype으로 루프를 고르는 단계가 하나 더 있다.
- 참값 파일: `wofz` 7,401점, `erf`·`erfc`·`erfcx` 5,400–5,500점, `loggamma` 7,363점, `lambertw` 가지마다 5,138점. 처음 만들 때 erf의 영점을 찾는 findroot가 엉뚱한 초기값에서 끝나지 않아 10분을 썼다. 큰 |z| 몇 점으로 점당 시간을 재서 원인이 참값 계산이 아님을 확인했고, 알려진 영점 근삿값에서 시작하게 고쳤다.

## 7. 확인 부탁
- **결정 목록에 없던 scipy와의 차이** (같은 "참값" 원칙으로 처리):
  - 결과가 오버플로하는 곳(예: `erfc(1+30j)`, `erf(30+1e300j)`)의 ±inf 부호는 정확한 위상을 따른다. scipy는 반올림한 2xy의 위상이나 NaN이다.
  - `erf(5e-324+30j)`의 실수부는 참값 5.39e67이다(scipy inf).
  - 실수축·허수축 위 0인 부분의 부호는 도함수의 방향을 따른다(scipy는 일관되지 않다).
  - `lambertw(x − 0j, k=-1)`(−1/e < x < 0)은 아래쪽 극한이다(scipy는 실수). 결정의 "C99식 위·아래 극한"을 그대로 적용한 결과다.
- **설계에서 바꾼 방법**: `wofz`는 설계 후보(Weideman, Poppe–Wijers 한 식) 대신 격자 Taylor + 연분수, `loggamma`는 1–2 둘레에 극을 빼낸 급수를 더했다.
- **1스레드 속도**: `wofz`·`erf` 계열 0.84–0.94배, `loggamma` 0.69배다.

## 8. 다음 (S9 마무리) 전에 물을 것
- 정확도·성능 보고서의 형식과 범위(S0–S8 전체를 한 문서로 모을지), 안내서(`docs/guide/`)의 special 장 구성.

## 9. 승인
- (대기)
