# 15장. 특수함수의 안쪽

> 함수 하나를 45 ULP 안에 넣으려면 무엇이 필요한가? 참값은 어디서 오고, 반올림은 어디서 쌓이고, 어떻게 막았나?

## 1. 연산 하나의 모양
특수함수 하나는 C++ 구조체(Op) 하나다 (`csrc/special/common.hpp`).

```
struct Betainc : Real {
    static constexpr const char* name = "betainc";
    static constexpr int nin = 3, nout = 1;          // 입력 셋, 결과 하나
    static constexpr std::array<const char*, 3> args{"a", "b", "x"};  // 메시지의 이름
    static constexpr double cost_ns = cost::kBetainc;  // 원소 하나의 비용
    static double f(double a, double b, double x);     // 원소 하나
};
```

- **n항 디스패처**: 배열 영역의 이터레이터(3장)에 입력 n개·결과 m개를 받는 루프(`nary_loop`)를 더했다. 피연산자는 최대 5개(`kMaxOperands`)다. `hyp2f1`이 입력 넷과 결과 하나라 4에서 늘렸다 (docs/learn/S0-foundation.md).
- **비용과 병렬화**: `cost_ns`를 덧셈 한 번의 비용(0.19 ns)으로 나눈 값이 가중치다. 이터레이터는 "덧셈 4096번어치"씩 작업을 자르므로, `erf`(가중치 28)는 원소 146개, `hyp2f1`(상한 256)은 16개마다 나눌 수 있다. 상한이 없으면 수천 ns 함수가 원소 한두 개씩 잘려 작업 관리 비용이 커진다.
- **복소수**: S8의 함수는 `f(std::complex<double>)`을 가진 Op를 따로 두고, 바인딩이 입력 dtype을 보고 실수 Op와 복소 Op 중 하나를 고른다.
- **덩어리 커널**: 분기가 적은 함수(`erf`, `ndtr`, `expit` 등)는 원소별 `f` 대신 1024개씩 처리하는 `chunk`를 둔다. NEON으로 인자를 만들고, Accelerate의 vForce로 exp·log를 한꺼번에 계산한 뒤, 원소별로 마무리한다 (`csrc/special/kernels/chunked.hpp`). 스칼라 하나도 같은 커널에 n = 1로 들어가 비트가 같다.

```python
import stardust as sd

w = sd.special._core._weight
print(w("expit"), w("erf"), w("hyp2f1"))  # 7 28 256
```

## 2. 방법을 고르는 순서
docs/design/special.md §1.3의 규칙이다. **정확도가 먼저, 다음이 속도, 마지막이 구간 수**다.
- 함수마다 설계에서 후보를 두고, 구현한 뒤 참값으로 재서 고른다. 재 보니 설계와 달라진 곳이 많다. 타원적분은 설계의 AGM·Carlson 대신 Cephes 다항식이 같은 정확도로 4–8배 빨랐고(S5), `wofz`는 후보 대신 격자 Taylor가 됐다(S8).
- **구간 수는 적게.** 구간 경계마다 두 방법의 오차가 만나고, 테스트할 곳이 는다. 정의역 변환(반사, 점화식, Kummer·Pfaff 변환)은 구간으로 세지 않는다.
- **Boost.Math를 부르는 구간** (S10, docs/design/special.md §5.14): 구간마다 Boost가 계약을 참값 파일의 모든 점에서 지키고 1스레드에서 느리지 않으면 Boost를 부른다(`csrc/special/boost.hpp`). 특수값은 그 전에 우리가 정하고, Boost의 결과가 정상 범위 밖(서브노멀, 0, inf, NaN)이면 우리 코드로 다시 계산한다. `zeta`(x > −60)는 이렇게 5배 빨라졌고, `hyp1f1`의 느슨한 영역은 상대 3.9e-3에서 2.8e-13이 됐다.
- **계수는 맞추지 않는다** (Q4). 공개 계수(Cephes, AS241 등)를 출처와 함께 가져오거나, 급수의 계수처럼 정의에서 나오는 값은 생성기(`tools/gen_*.py`)가 mpmath로 만든다. Remez로 새 근사를 맞추지 않았다.

## 3. 참값과 ULP
- `tools/special_ref.py`가 함수마다 수천 점의 입력을 만들고(보장 영역의 격자, 로그 균등 무작위, 경계·극점·영점 근처), mpmath로 값을 계산해 `tests/special/ref/<이름>.npz`에 hi + lo(double-double)로 저장한다. 테스트는 mpmath를 부르지 않고 이 파일과 비교한다.
- **mpmath도 조용히 틀린다.** 그래서 자릿수를 정하는 규칙이 있다. 인자의 크기만큼 자릿수를 올려 시작하고, 두 배의 자릿수로 한 번 더 계산해 30자리가 같아질 때까지 늘린다. 이 검사가 잡은 것: `beta`의 a + b(a/b = 1e290), 큰 차수의 `besselk`·`bessely`(S6, 자릿수를 따로 더 올렸다), `betainc`의 상보 상쇄(S4).
- **점마다 시간 제한(30초)**: 시간 제한 없이 13시간을 돈 적이 있다(S4). 시간을 넘긴 점은 버리고 그 수를 출력한다. 오래 걸리는 측정은 결과를 바로바로 출력하게 한다.
- 오차는 `bench/special/accuracy.py`가 보장 영역과 느슨한 영역으로 나눠 최대와 99.9%를 내고, scipy의 같은 값을 옆에 둔다. 테스트(`tests/special/test_reference.py`)는 각 점이 계약(14장 §3) 안인지 본다.

## 4. 반올림이 쌓이는 곳: double-double
float64 연산 하나는 반 ULP를 틀린다. 문제는 그 오차가 **쌓이거나 커지는** 곳이다. 그곳은 수 하나를 두 double의 합 hi + lo(약 106비트)로 들고 다닌다 (`csrc/special/dd.hpp`). 덧셈 오차는 `two_sum`, 곱셈 오차는 `two_prod`(fma 한 번)로 정확히 얻는다.

| 어디서 | 무엇이 문제였나 | 바꾼 것 |
|---|---|---|
| 지수의 인자 x^a e^{−x} (불완전 감마·베타, S4) | a·log x − x가 1e5면 절대오차 1e-11이 결과의 상대오차가 된다 | 지수를 double-double로 |
| Bessel 점화식 (S6) | 수백 단계를 지나며 ULP가 쌓인다 | 점화식 전체를 double-double로 |
| 급수의 항 (S7 `hyp1f1`·`hyp2f1`) | 항을 연이어 곱하면 k번째 항이 k ULP를 지닌다. x ≈ 500에서 합을 이루는 항이 k ≈ 500이다 | 항과 a + k를 double-double로. 120 ULP → 0.5 ULP |
| Γ의 인자 (S3, S7) | b − a를 반올림하면 그 오차가 Γ(b−a)와 x^(−a)의 지수에 들어간다 | b − a를 double-double로 넘긴다 |
| 분기점 −1/e (S7 `lambertw`) | e·x + 1이 마지막 비트들만 남는다(scipy 2e7 ULP) | e를 hi + lo로, fma(e_hi, x, 1) + e_lo·x |
| exp(−z²)의 위상 (S8) | −2xy를 반올림하면 위상이 틀린다(scipy 447 ULP) | two_prod로 정확히 만들어 sin·cos에 두 조각으로 |

- double-double은 10배쯤 비싸다. 필요한 곳에만 쓴다. `hyp1f1`·`hyp2f1`이 scipy보다 느린 이유가 이것이다.
- **반례**: S7에서 "24항 안에 끝나는 급수는 float64로 먼저"를 넣었다가 되돌렸다. 24항이면 24 ULP라는 근거는 항이 모두 양수일 때만 맞다. 항이 상쇄하는 급수에서는 그 오차가 상쇄의 배율만큼 커졌다.

## 5. 넘침을 피하는 m·2ᵉ
- Γ(171)은 double을 넘지만 Γ(171.5)/Γ(170.5)는 평범한 수다. 중간값이 넘치는 곳은 값을 (m, e)로 들고 다닌다(`Scaled`, `csrc/special/gamma_core.hpp`). 곱셈은 m끼리 곱하고 e끼리 더한다. 마지막에 한 번만 `ldexp`로 내린다.
- `exp_scaled(x)`는 x = n·log 2 + r로 나눠 eʳ와 n을 돌려주고, `pow_scaled(a, t)`는 a^t를 조각내 libm의 정확도를 지킨다.
- 복소 `erfc(z) = exp(−z²)·w(iz)`도 이 꼴이다. exp(−z²)가 넘쳐도 w(iz)가 작으면 결과는 유한하다. 넘치는 쪽은 지수를 크게 둬서, 0인 성분은 0으로, 나머지는 ±inf로 내려간다.

## 6. 상쇄를 피하는 식
같은 함수도 식에 따라 상쇄가 다르다. 방법을 고를 때 가장 먼저 보는 것이다.
- `rel_entr(x, y)`: x ≈ y에서 x·log(x/y)는 비의 반올림이 0에 가까운 log에 남는다(7e-5). x·log1p((x−y)/y)로 바꾼다. x − y는 Sterbenz 정리로 정확하다.
- `logit(p)`: p ≈ ½에서 2·atanh(2p − 1).
- `loggamma`(S8): 1과 2의 영점 둘레에서 점화식은 Stirling 값(크기 13)과 곱의 로그가 100배 넘게 상쇄했다(293 ULP). ζ(k) = Σ_{j≤6} j^{−k} + ζ(k, 7)로 나눠 앞의 여섯 극을 μ(t) = t − log(1+t)로 닫힌 꼴로 빼내자 각 부분이 서로 상쇄하지 않았다(8 ULP).
- 영점이 있는 쪽의 반사식(복소 `erfc`의 2 − erfc(−z))처럼 상쇄를 피할 수 없는 곳은 계약 자체를 포락선으로 정했다(14장 §3.2).

## 7. 표를 줄이는 법: 계수를 그 자리에서
`wofz`(S8)의 격자 Taylor는 0.1 간격 격자점 2,911개의 w(z₀)만 저장한다. Taylor 계수는 w가 만족하는 미분방정식 w′ = −2zw + 2i/√π에서
a_{n+1} = −(2z₀aₙ + 2a_{n−1})/(n+1)
로 그 자리에서 만든다. 표가 계수 20개씩이었다면 20배 컸다. 재귀의 반올림이 커지는지는 동차해 exp(−z²)의 크기 비 |exp(−2z₀h)| ≤ e^{1.14}로 따졌고, 잰 최대는 2.5 ULP였다.

## 8. 잘못 간 길에서 배운 것
- **긴 측정은 점마다 시간 제한을 걸고 바로 출력한다.** 13시간 동안 아무 출력 없이 돈 측정(S4)과, 영점을 찾는 findroot가 끝나지 않아 10분을 쓴 일(S8)이 있었다.
- **알고리즘 실험은 C++로.** 파이썬으로 흉내 내고 mpmath로 재는 실험은 느리고, 실제 코드와 다르게 반올림한다.
- **정답 도구도 의심한다.** mpmath의 큰 차수 Bessel이 틀렸고(S6), 두 자릿수 비교가 그것을 잡았다.
- **scipy를 먼저 잰다.** scipy의 방법을 따라 하기 전에 그 방법이 기준을 지키는지 재 보면, 따라 할지 말지가 바로 정해진다(S8 `wofz`: scipy 위쪽 반평면 178 ULP).

## 더 읽을 곳
- docs/design/special.md §3(엔진), §5(함수마다의 방법), §6(정답지와 측정)
- `csrc/special/dd.hpp`, `gamma_core.hpp`, `kernels/chunked.hpp`, `tools/special_ref.py`
- docs/learn/S4-incomplete.md(double-double의 시작), S6-bessel-airy.md(점화식과 정답 도구), S8-complex.md(격자 Taylor와 loggamma)
