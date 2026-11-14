# 14장. `sd.special` 쓰기

> 65개 특수함수는 어떤 규칙을 함께 따르나? "정확하다"는 말은 정확히 무엇을 약속하고, scipy와는 어디서 다른가?

## 1. 무엇이 있나
`sd.special`은 scipy.special의 부분집합이다. 이름은 scipy와 같고, 마일스톤 S1–S8에서 묶음별로 들어왔다 (ROADMAP.md, docs/api/special.md).

| 묶음 | 함수 | API |
|---|---|---|
| 로지스틱·정보 | `expit logit log_expit xlogy xlog1py entr rel_entr` | §5 |
| 오차함수·정규분포 | `erf erfc erfcx erfinv erfcinv ndtr log_ndtr ndtri` | §2 |
| 감마·베타·제타 | `gamma rgamma gammaln digamma polygamma beta betaln binom zeta` | §3 |
| 불완전 감마·베타와 역함수 | `gammainc gammaincc gammaincinv gammainccinv betainc betaincc betaincinv betainccinv` | §4 |
| 리덕션·지수적분·타원적분 | `logsumexp softmax log_softmax expi expn ellipk ellipe` | §6, §7, §10 |
| Bessel·Airy | `j0 j1 y0 y1 i0 i0e i1 i1e k0 k0e k1 k1e jv yv iv kv jve yve ive kve airy` | §8, §9 |
| 초기하·Lambert W | `hyp1f1 hyp2f1 lambertw` | §11 |
| 복소수 | `wofz loggamma` (그리고 `erf erfc erfcx lambertw`의 complex 입력) | §12 |

```python
import stardust as sd

print(len(sd.special.__all__))  # 65
print(sd.special.erf(0.5))  # 0.5204998778130465
```

## 2. 호출 규칙
4장의 원소별 연산과 같은 길을 지나간다. 다른 점만 적는다.
- **입력은 위치로만, `out=`은 키워드로만**: `betainc(a, b, x, /, *, out=None)`. 결과가 여럿이면(`airy`) `out=`이 없다.
- **dtype**: 실수 함수는 float64로 계산한다. bool과 int64는 float64로 바꿔 받고 complex128은 TypeError다. 예외는 §12의 복소수 함수다. `erf erfc erfcx lambertw`는 입력이 complex이면 복소 루프로 가고, `wofz`는 실수를 받아도 complex를 돌려준다.
- **브로드캐스트**: 입력 여럿은 배열처럼 브로드캐스트한다. `betainc(2.0, 3.0, x)`처럼 스칼라 매개변수와 배열 하나도 된다.
- **Python 스칼라에는 Python 스칼라**: 배열을 넣으면 배열, Python float를 넣으면 float가 나온다.
- **결과가 여럿**: `airy`는 `AiryResult(ai, aip, bi, bip)` namedtuple이다(13장의 `<함수>Result` 규칙).
- **리덕션**: `logsumexp`, `softmax`, `log_softmax`는 `axis=None`과 `keepdims`만 받는다.
- 별칭(`psi`, `jn`)이 없고, 경고를 내지 않는다.

```python
import numpy as np
import stardust as sd

x = sd.asarray(np.array([0.1, 0.5, 0.9]))
print(sd.special.betainc(2.0, 3.0, x).dtype)  # float64
print(type(sd.special.erf(1)))  # <class 'float'>
print(sd.special.airy(1.0).ai)  # 0.13529241631288147
print(sd.special.erf(1 + 1j))  # (1.3161512816979477+0.19045346923783468j)
```

## 3. "정확하다"의 약속
### 3.1 상대오차 1e-14
- 보장 영역에서 결과의 **참값 대비 상대오차는 1e-14 이하**다. float64의 한 칸(ULP)이 상대 2.2e-16이니 약 45 ULP다. 측정한 최대는 대부분 0.5–10 ULP다 (docs/learn/S9-final-report.md).
- **참값**은 mpmath로 50자리 넘게 계산해 저장소에 둔 값이다(15장 §3). scipy와 비교하지 않는다. scipy도 틀리는 곳이 있다.
- **서브노멀** 결과(2⁻¹⁰²² 아래)는 상대오차가 의미를 잃으므로 절대오차 45·2⁻¹⁰⁷⁴로 잰다.

### 3.2 상대오차를 지킬 수 없는 곳
함수에 따라 1e-14 상대를 double로 낼 수 없는 곳이 있다. 그곳은 무엇을 지키는지 함수마다 정해 두었다 (docs/api/special.md 각 절).

| 함수 | 지키는 것 | 왜 |
|---|---|---|
| `gammaln`(x ≤ 0), `betaln`, `binom`, `polygamma` 등 | \|오차\| ≤ 1e-14·max(1, \|값\|) | 0 근처를 지나는 함수의 반사식 |
| 역함수(`gammaincinv` 등) | \|Δx/x\| ≤ 1e-14·max(1, κ), κ = \|F/(x·F′)\| | 입력 한 칸의 차이가 결과를 κ칸 움직인다 |
| 진동하는 Bessel·Airy (첫 영점 뒤) | \|오차\| ≤ 1e-14·M(x), M = √(J² + Y²) 등 | 영점 근처는 어떤 방법도 상대 정확도를 못 낸다 |
| 복소 `erf erfc erfcx wofz`의 영점 쪽 | \|오차\| ≤ 1e-14·(\|A\| + \|B\|), f = A − B | 같은 이유. A, B는 계산식의 두 항 |
| 복소수 일반 | \|오차\| / \|값\| (크기 기준) | 실수부·허수부를 따로 재지 않는다 |

### 3.3 느슨한 영역
- **보장 영역 밖도 계산한다.** NaN은 수학적 정의역 밖에서만 낸다. 대신 그곳의 정확도는 잰 값으로 문서에 적는다.
- 느슨한 영역이 있는 함수는 셋이다: `hyp1f1`(a < 0, x > 0의 큰 상쇄, 잰 최대 상대 3.9e-3), `hyp2f1`(z > 0.9 등, 6.6e-13), Airy(|x| > 1e12, 위상의 한계).

## 4. 특수값: 참값이 먼저
scipy와 다른 특수값은 모두 docs/design/special.md §2.4의 표에 있다. 마일스톤마다 사용자에게 물어 정했고, 원칙은 하나였다: **scipy가 틀리거나 NaN을 내는 곳은 참값이나 극한**이다.
- 서브노멀 꼬리: scipy가 0을 내는 `expit(-720)`, `erfc(27)`은 참값(서브노멀)이다.
- 0에 가까운 인자: `gammaln(5e-324)`은 744.44다(scipy inf).
- 수학적 극한: `hyp1f1(0, 0, x) = 1`(scipy inf), `hyp2f1(a, b, c, -inf)`, `loggamma(inf + 0j)`.
- 분기점과 가지 자름: `lambertw`는 −1/e에 가장 가까운 double을 분기점으로 보고 −1을 낸다(실수 입력). 복소 입력의 자름 위 x ± 0i는 허수부 0의 부호로 위·아래 극한을 고른다(C99).

```python
import stardust as sd

print(sd.special.expit(-720.0))  # 2.0322308024e-313
print(sd.special.gammaln(5e-324))  # 744.4400719213812
print(sd.special.lambertw(-0.36787944117144233))  # -1.0
print(sd.special.hyp1f1(0.0, 0.0, 2.0))  # 1.0
print(sd.special.loggamma(complex(-2.5, 0.0)).imag, sd.special.loggamma(complex(-2.5, -0.0)).imag)  # -9.42477796076938 9.42477796076938
```

## 5. 결정성
6장의 규칙을 그대로 따른다.
- 같은 입력은 스레드 수와 상관없이 같은 비트를 낸다. 원소 하나가 하나의 식으로 계산되고, 나누는 것은 원소의 묶음뿐이다.
- `f(x)[i] == f(x[i])`: 배열 경로와 스칼라 경로가 비트까지 같다. NEON으로 여러 원소를 함께 계산하는 함수(`erf`, `expit` 등)도 스칼라 하나를 같은 커널에 넣는다.
- 반복법(Halley, 연분수)의 멈춤 조건이 원소마다 다르게 끝나도, 그 끝은 입력만으로 정해진다.

```python
import numpy as np
import stardust as sd

x = sd.asarray(np.linspace(0.1, 30.0, 100000))
sd.set_num_threads(1)
a = np.asarray(sd.special.kv(2.5, x))
sd.set_num_threads(10)
b = np.asarray(sd.special.kv(2.5, x))
print(np.array_equal(a.view(np.uint64), b.view(np.uint64)), a[7] == sd.special.kv(2.5, float(np.asarray(x)[7])))  # True True
```

## 6. 얼마나 빠른가
수치는 docs/learn/S9-final-report.md §3에 있다. 요약하면
- 10⁶개 배열에서 대부분의 함수는 1스레드로도 scipy와 비슷하거나 빠르고, 10스레드에서는 몇 배에서 수십 배 빠르다. scipy.special의 ufunc는 1스레드다.
- 1스레드에서 크게 느린 함수가 있다: `hyp2f1`과 보장 영역 일부의 `hyp1f1`(급수를 double-double로 계산), 큰 차수의 `iv`·`kv`(Debye 전개를 넣지 않아 차수에 비례), `beta`·`betaln` 일부. 정확도와 맞바꾼 것이고, 후속 후보로 적어 두었다.
- S10부터 20개 함수의 일부 구간은 Boost.Math를 부른다. 그 구간에서 Boost가 계약을 지키고 느리지 않다고 잰 곳이다(15장 §2).
- 함수마다 원소 하나의 비용(`cost.hpp`)을 재서, 비싼 함수는 작은 배열에서도 스레드를 나눈다(15장 §1).

## 더 읽을 곳
- docs/api/special.md: 함수마다의 규칙, 보장 영역, 특수값 표
- docs/design/special.md §1(결정 Q1–Q19), §2.4(scipy와 다른 것), §5(함수마다의 방법)
- docs/learn/S0-foundation.md – S8-complex.md: 마일스톤마다의 과정과 측정
- 15장: 이 약속을 지키는 안쪽
