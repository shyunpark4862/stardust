# 보간 영역 설계 (sd.interpolate)

> 상태: **확정** (2026-11-23). 큰 결정(§1.1 IP1–IP3)은 사용자가 정했고, 나머지 결정(IP4–IP82, IP90–IP130)은 추천안이다(사용자 위임). IP0–IP9는 완료·승인(위임)이다(IP9는 `sd.spatial.Delaunay` 위의 흩어진 점 보간으로, 설계가 [spatial_geometry.md](spatial_geometry.md) §2.6, §3.8, §3.9에 있다). 모든 공개 이름(`docs/api/interpolate.md`, 배열 영역의 `sd.interp`는 `docs/api/array.md` §8.5)이 구현되어 있고 이 문서는 구현과 맞다.

`sd.interpolate`(점에서 주어진 값을 지나는 함수를 만들고 계산하기)의 확정 사항과 내부 구조를 적는다.

- 공통 규칙은 [DESIGN.md](../../DESIGN.md), 층 구조는 [ARCHITECTURE.md](../../ARCHITECTURE.md), 모든 영역의 API 규칙은 [API.md](../../API.md)에 있다. 공개 함수와 시그니처는 [docs/api/interpolate.md](../api/interpolate.md)에 있다. `sd.interp`(IP2)는 배열 영역의 함수라 `docs/api/array.md` §8.5에 있다.
- 비교 대상은 scipy 1.18.1의 `scipy.interpolate`, numpy 2.5.4의 `numpy.interp`, R 4.6.2의 `stats::approx`·`spline`·`splinefun`이다. R의 보간 패키지(`akima`, `pracma`, `signal`)는 설치되어 있지 않다. 설치하지 않는다(선형과 3차 스플라인만 R과 견준다, §7.4).
- 바닥은 배열(`csrc/array/`), 바닥 층의 공용 커널(`csrc/base/`), 스레드풀(`csrc/parallel/`), linalg의 LAPACK 래퍼(`csrc/linalg/lapack.hpp`: RBF의 분해와 랭크 검사, AAA의 SVD와 일반화 고유값), spatial의 KD 트리와 거리 커널(`csrc/spatial/kdtree.hpp`·`metric.hpp`: RBF의 이웃과 거리, 가장 가까운 점 보간), 배열의 NEON 수학 커널(`csrc/array/kernels/vmath.hpp`: RBF 핵의 log·exp)이다. 선형 커널 하나를 배열의 `sd.interp`와 이 영역이 함께 쓴다(§3.2).
- 이 문서의 잰 값은 Apple M4 Pro, macOS 26.6.2에서 본 것이다. 탐침은 `tools/interpolate_probe.py`(기록은 `tests/interpolate/data/`, 시간과 측정은 `build/logs/interpolate_probe/`)이고 §3.2, §3.3, §3.4, §5가 그 결과를 옮긴다.
- ε은 float64의 기계 엡실론 2⁻⁵² ≈ 2.22×10⁻¹⁶, u는 단위 반올림 2⁻⁵³(= ε/2), γ_k = ku/(1 − ku)다(spatial.md, sparse.md §4와 같은 기호).

---

## 1. 기반

### 1.1 결정 기록 (사용자, 2026-11-23)
| 번호 | 결정 |
|---|---|
| IP1 범위 | 보간만이다. **1차원**: 선형, `CubicSpline`(not-a-knot, clamped, natural, periodic), `PchipInterpolator`, `Akima1DInterpolator`(makima 포함), `make_interp_spline`(모든 차수), `BSpline`, `PPoly`. **다항식·유리식**: `BarycentricInterpolator`, `KroghInterpolator`, `AAA`(scipy의 것), scipy에 있으면 그 함수 꼴(`pchip_interpolate` 등)도. **격자**: `RegularGridInterpolator`, `interpn`. **흩어진 점**: `RBFInterpolator`(`neighbors=`는 `sd.spatial.KDTree`로). **곡선 연산**(scipy에 있는 것): 미분, 역미분, 적분, 근, 늘이기(`extend`). 빼는 것: 매끄럽게 하는 스플라인(`make_smoothing_spline`, s > 0의 `splrep`, `UnivariateSpline` …)은 나중의 `sd.smooth`로, Qhull에 기대는 것(`griddata`의 linear·cubic, `LinearNDInterpolator`, `CloughTocher2DInterpolator`; `NearestNDInterpolator`는 Qhull이 아니라 KD 트리라 §9에서 정한다), 옛 이름(`interp1d`, `interp2d`, FITPACK 감싸개; 어느 것을 뺄지는 §9에서 정한다) (사용자 결정 2026-11-23) |
| IP2 `interp` | numpy의 `interp`는 최상위 `sd.interp`다(배열 영역의 함수, 뜻이 맞는 곳은 numpy의 뜻). 계산은 이 영역의 1차원 선형 길과 같은 식, 같은 비트다 (사용자 결정 2026-11-23) |
| IP3 위치 | 최상위 `sd.interpolate`, 평평한 이름공간이다. 보간 객체는 클래스다(`KDTree`와 같은 데이터 구조의 예외) (사용자 결정 2026-11-23) |
| IP4 선형 커널의 식 | §9의 1번 (a)를 따른다 (추천안, 사용자 위임 2026-11-23) |
| IP5 `sd.interp`의 `xp` 검사 | §9의 2번 (a)를 따른다 (추천안, 사용자 위임 2026-11-23) |
| IP6 받는 입력 | §9의 3번 (a)를 따른다 (추천안, 사용자 위임 2026-11-23) |
| IP7 뺄 옛 이름 | §9의 4번 (a)를 따른다 (추천안, 사용자 위임 2026-11-23) |
| IP8 `NearestNDInterpolator` | §9의 5번 (a)를 따른다 (추천안, 사용자 위임 2026-11-23) |
| IP9 `FloaterHormannInterpolator` | §9의 6번 (a)를 따른다 (추천안, 사용자 위임 2026-11-23) |
| IP10 도우미 클래스 | §9의 7번 (a)를 따른다 (추천안, 사용자 위임 2026-11-23) |
| IP11 복소수 | §9의 8번 (a)를 따른다 (추천안, 사용자 위임 2026-11-23) |
| IP12 `construct_fast`와 고치는 메서드 | §9의 9번 (a)를 따른다 (추천안, 사용자 위임 2026-11-23) |
| IP13 바깥의 규칙 | §9의 10번 (a)를 따른다 (추천안, 사용자 위임 2026-11-23) |
| IP14 옵션의 자리 | §9의 11번 (a)를 따른다 (추천안, 사용자 위임 2026-11-23) |
| IP15 3차 계열의 계수 순서 | §9의 12번 (a)를 따른다 (추천안, 사용자 위임 2026-11-23) |
| IP16 스플라인의 연립방정식 풀이 | §9의 13번 (a)를 따른다 (추천안, 사용자 위임 2026-11-23) |
| IP17 periodic의 검사 | §9의 14번 (a)를 따른다 (추천안, 사용자 위임 2026-11-23) |
| IP18 근 찾기 | §9의 15번 (a)를 따른다 (추천안, 사용자 위임 2026-11-23) |
| IP19 세로축의 유한하지 않은 값 | §9의 16번 (a)를 따른다 (추천안, 사용자 위임 2026-11-23) |
| IP20 AAA의 멈춤 알림과 극점 | §9의 17번 (a)를 따른다 (추천안, 사용자 위임 2026-11-23) |
| IP21 `Barycentric`의 무게 순서 | §9의 18번 (a)를 따른다 (추천안, 사용자 위임 2026-11-23) |
| IP22 Krogh의 높은 차수 | §9의 19번 (a)를 따른다 (추천안, 사용자 위임 2026-11-23) |
| IP23 RGI 스플라인 방법의 계수 | §9의 20번 (a)를 따른다 (추천안, 사용자 위임 2026-11-23) |
| IP24 RGI의 NaN 평가점 | §9의 21번 (a)를 따른다 (추천안, 사용자 위임 2026-11-23) |
| IP25 RBF의 `smoothing`, 낮은 `degree`, 같은 자료점 | §9의 22번 (a)를 따른다 (추천안, 사용자 위임 2026-11-23) |
| IP26 선형 커널의 꼴 | IP4의 식을 갈래 없이 쓴다: 두 식(t·b + (1 − t)·a와, b에서 자른 a + t·(b − a))을 모두 계산하고 끝값의 부호와 t로 고른다(비교는 비트 연산으로 묶는다). 갈래 꼴과 같은 비트이고(84만 6천 점에서 차이 0), 무작위 순서의 평가점에서 갈래 꼴은 numpy 식보다 3.7배, 갈래 없는 꼴은 1.2배 느리다(§3.2) (추천안, 사용자 위임 2026-11-23) |
| IP27 선형 커널의 끝과 넘침 | t = 0이면 a, t = 1이면 b를 그대로 돌려준다(0의 부호까지 같은 비트라 노드의 구간을 어느 쪽으로 골라도 같다). 폭 x₁ − x₀이 넘치면 x, x₀, x₁에 1/2을 곱해 t를 구한다(§3.2) (추천안, 사용자 위임 2026-11-23) |
| IP28 주기 조건의 허용 | `make_interp_spline(bc_type='periodic')`도 `CubicSpline`과 같은 `allclose(rtol=1e-15, atol=1e-15)`로 검사한다(`[scipy와 다름]`: scipy의 `make_interp_spline`은 rtol 1e-5라 cancel/y1e15의 끝값이 10¹⁰ 달라도 받는다; 한 영역에 한 규칙) (추천안, 사용자 위임 2026-11-23) |
| IP29 RBF의 `degree=-1` | 다항식을 빼는 −1도 핵의 최소 차수보다 낮으면 `ValueError`다(scipy는 경고 없이 받는다; IP25와 같은 이유) (추천안, 사용자 위임 2026-11-23) |
| IP30 RGI의 `nu` | `__call__(xi, /, *, method=None, nu=None)`: scipy 1.18.1의 `nu`(축마다의 미분 차수)를 받는다. `slinear`, `cubic`, `quintic`에서만이다(scipy와 같다) (추천안, 사용자 위임 2026-11-23) |
| IP31 `NearestNDInterpolator`의 옵션 | `rescale`은 둔다(scipy의 뜻). `tree_options`와 `__call__`의 질의 옵션은 뺀다(sd의 `KDTree`는 `leafsize`만 받고 결과가 그것과 무관하다). `xi`는 (…, d) 배열이나 d개 배열의 튜플이다(RGI와 같다) (추천안, 사용자 위임 2026-11-23) |
| IP32 `AAA.clean_up`의 인자 이름 | 생성자와 같은 `clean_up_tol`이다(`[scipy와 다름]`: scipy의 메서드는 `cleanup_tol`; 한 클래스에 한 철자) (추천안, 사용자 위임 2026-11-23) |
| IP33 Akima 문턱의 범위 | 무게 합의 문턱 10⁻⁹·max의 max는 열마다의 최댓값이다(`[scipy와 다름]`: scipy는 모든 열에서 하나를 잡는다; 배치의 열이 서로 독립이라는 §2.1의 약속) (추천안, 사용자 위임 2026-11-23) |
| IP34 시험 데이터 모듈의 이름 | `tests/interpolate/samples.py`다(§7.1의 처음 이름 `data.py`는 기록 폴더 `tests/interpolate/data/`와 겹친다) (추천안, 사용자 위임 2026-11-23) |
| IP35 B-스플라인 계의 행 순서 | 도함수 조건(`natural`, `clamped`, (차수, 값))의 행은 그 끝의 보간 행 바로 뒤에 둔다. scipy의 순서(도함수 행이 먼저)는 피벗 없는 소거에서 0 피벗이 나오고, 이 순서는 시험 자료 전부에서 성장 인자가 1.97 이하였다(§3.4) (추천안, 사용자 위임 2026-11-23) |
| IP36 넘치는 3차 계수 | §9의 23번 (a)를 따른다: 만들기에서 계수가 유한하지 않게 되면 `ValueError`이고 메시지가 그 구간을 말한다(IP2 마일스톤에서 쓴다) (추천안, 사용자 위임 2026-11-23) |
| IP37 구간 찾기의 자리 | 점 아래(같은 값 포함)의 노드 수를 세는 탐색을 바닥 층의 `base/interval.hpp`에 두고 배열의 `sd.interp`와 이 영역의 `interpolate/search.hpp`가 함께 쓴다(배열은 다른 영역을 include하지 않는다; §6의 처음 안은 `interpolate/search.hpp` 하나였다). 위치만으로 자른 256점 블록이 정렬되어 있으면 앞 점의 수에서 질주하고, 아니면 8점을 나란히 갈래 없이 반씩 줄인다. 어느 길이든 같은 수라서 결과는 길과 스레드 수와 무관하다(§3.1) (추천안, 사용자 위임 2026-11-23) |
| IP38 t의 넘침과 무한한 점 | `lerp_t`는 폭뿐 아니라 x − x₀가 넘칠 때(정의역에서 아주 먼 늘이기)도 1/2을 곱해 계산하고, ±inf인 점의 t는 ±inf다. 구간 안의 성질(§3.2의 1–5)은 그대로이고 계약도 같다 (추천안, 사용자 위임 2026-11-23) |
| IP39 `sd.interp`의 NaN 평가점 | NaN 평가점은 `xp`가 한 점이어도 NaN이다(`[numpy와 다름]`: numpy는 한 점이면 NaN에도 `fp[0]`; §2.1의 규칙 하나). complex `fp`의 NaN 평가점은 numpy처럼 허수부가 0인 NaN이다. `period`에서 감은 값이 같은 노드는 원래 순서를 지킨다(안정 정렬; numpy는 정해 두지 않은 퀵정렬) (추천안, 사용자 위임 2026-11-23) |
| IP40 늘이기 계약의 고침 | 늘이기의 한계는 γ₁₀·max(1, \|t\|)·(\|a\| + \|b\|) + 3η다. IP0 마일스톤의 증명은 \|t\| > 1에서만 \|y\| ≤ 2\|t\|M을 썼고 −1 < t < 0에서는 마지막 합의 반올림 u\|y\| ≈ u\|a\|가 γ₁₀\|t\|(\|a\| + \|b\|)를 넘는다(참값 시험의 uniform/6, x = −5×10⁻³²⁴에서 찾음; §3.2). 구간 안의 한계 γ₆(\|a\| + \|b\|) + 3η는 그대로다. `tests/interpolate/refdata.linear_bound`도 고쳤고, 3η = 1.5·2⁻¹⁰⁷⁴는 double이 아니라서 그 위의 double 2⁻¹⁰⁷³로 잰다(전에는 η = 2⁻¹⁰⁷⁵가 0으로 반올림되어 한계에서 빠져 있었다) (추천안, 사용자 위임 2026-11-23) |
| IP41 이 영역의 선형 길 | `interpolate/linear.cpp`가 열 묶음의 꺾은선을 세 바깥 규칙(NaN, 늘이기, `'periodic'`)으로 계산하고, IP3 마일스톤의 k = 1 B-스플라인과 IP5 마일스톤의 1차원 격자 선형이 이것으로 평가한다. 점마다 t 하나를 모든 열이 쓴다. 공개 이름이 없어 비공개 `_core.interpolate.linear(x, xp, fp, extrapolate)`다. NaN 평가점은 complex의 두 부분 모두 NaN이다(scipy가 두 부분을 따로 계산하는 것과 같다; `sd.interp`는 numpy를 따른다) (추천안, 사용자 위임 2026-11-23) |
| IP42 Hermite 계수의 연산 순서 | IP15의 하나의 Hermite 함수는 scipy `CubicHermiteSpline`의 순서다: D = (y₁ − y₀)/h, t = ((s₀ + s₁) − 2D)/h, c₃ = t/h, c₂ = (D − s₀)/h − t, c₁ = s₀, c₀ = y₀(곱이 합에 묶이지 않는다). 같은 기울기면 scipy와 같은 비트의 계수이고, `PchipInterpolator`·`Akima1DInterpolator`는 기울기 식도 scipy의 연산 순서 그대로라 시험 곡선 전부에서 scipy와 계수가 비트까지 같았다. h²을 따로 만들지 않아 간격이 작은 곳의 언더플로가 없다 (추천안, 사용자 위임 2026-11-23) |
| IP43 `PPoly`의 중단점 | 유한하고 엄격히 증가하거나 엄격히 감소해야 한다(`ValueError`). `[scipy와 다름]`: scipy는 차이가 0 이상인지만 보아 같은 중단점(폭이 0인 조각)과 무한한 중단점을 받는다. `extend`도 같은 규칙이라 새 중단점이 끝 중단점과 같으면 에러다(scipy는 받는다). 폭이 0인 조각은 어떤 점도 고르지 않아 뜻이 없고, 감소하는 중단점의 거울 규칙(x[i+1] < x ≤ x[i], 끝은 닫힘)은 scipy와 같다 (추천안, 사용자 위임 2026-11-23) |
| IP44 삼중대각 풀이의 꼴 | 피벗 없는 Thomas 소거(IP16): 분해의 사슬은 나눗셈 l = a/u이고 피벗의 역수 w = 1/u를 사슬 밖에서 만들어 뒤 대입은 곱한다(나눗셈 둘의 사슬보다 1.2배 빠르다; 한 행에 반올림 하나가 더해지고 §3.3의 증명이 센다). 열이 8개 이하면 분해, 오른쪽, 앞 대입을 한 번의 훑기로 하고, 그보다 많으면 분해 한 번 뒤 열 묶음을 나란히 대입한다. 두 길은 열마다 같은 연산이라 한 열을 따로 만들든 배치 안에서 만들든 같은 비트다. 계산한 피벗이 양수가 아니면 `ValueError`다: not-a-knot의 끝 두 간격의 비가 약 2⁵⁰을 넘을 때만 생기고 그때는 반올림한 행렬 자체가 작업 정밀도에서 특이하다(scipy의 `gbsv`도 피벗을 바꾸지 않는 자리다) (추천안, 사용자 위임 2026-11-23) |
| IP45 근 찾기의 구현 | IP18(§9의 15 (a))의 단조 구간 나누기를 이렇게 한다: (1) 조각의 임계점은 도함수의 근이다(2차는 상쇄 없는 공식, 그 위는 같은 알고리즘을 재귀로). (2) 끝의 부호가 다른 단조 구간은 double을 순서대로 센 정수(±0은 같은 0)에서 이분한다: 걸음은 언제나 64번 이하다(처음 안의 "64 + 지수 범위의 걸음"을 대신한다). (3) 부호는 보정한 Horner(Graillat–Langlois–Louvet의 오차 없는 곱과 합)로 정한다: \|p\|가 u²Σ\|cₑsᵉ\|의 몇 배를 넘는 곳에서 부호가 맞아 2⁻³⁰ 떨어진 두 근을 마지막 비트까지 찾는다. (4) 이웃 부호가 같은 극값의 값이 평가 계약(§4) 안이면 중근으로 넣는다. (5) 늘이는 끝 조각은 Cauchy 한계의 두 배까지만 본다(그 밖에 근이 없다). `solve(y)`는 상수 계수에서 y를 한 번 빼고 같은 길이다. `extrapolate='periodic'`은 scipy처럼 True로 친다 (추천안, 사용자 위임 2026-11-23) |
| IP46 PCHIP 단조성의 둘째 성질 | §4의 PCHIP 단조성 (2) "계산한 계수 그대로의 3차식이 정확히 단조"는 반올림 때문에 일반적으로 서지 않는다: 노드 기울기가 0이면 정확한 산술에서 그 끝의 도함수가 0이고, 반올림한 c₂, c₃로는 그 값이 ±반올림 크기다(hypothesis가 x = (1, 2, 8), y = (0, 2, 3)에서 찾음). (2)는 "도함수 2차식이 계수 계약(§4의 Hermite 계수 한계)의 도함수만큼 안에서 부호를 지킨다"로 고친다. (1)과 (3)은 그대로다 (추천안, 사용자 위임 2026-11-23) |
| IP47 Akima의 평평한 구간 | §2.4의 "평평한 구간(같은 값이 셋 이상)은 정확히 평평하다"를 고친다: makima는 그렇다(무게 합이 문턱 아래인 띠를 빼고). akima는 구간 안쪽 노드의 기울기가 언제나 0이지만, 구간 끝 노드가 Akima(1970)의 정하지 않은 경우(m₁ = m₂ ≠ m₃ = m₄, 무게 합 0)이면 두 기울기의 평균이라 평평하지 않을 수 있다(scipy와 같은 비트; 시험 자료 flat/akima의 첫 구간 끝 노드가 6.5) (추천안, 사용자 위임 2026-11-23) |
| IP48 `'periodic'` 객체의 역미분 | `extrapolate=False`다(scipy 1.18.1에서 확인; 처음 안의 "True"는 scipy와 달랐다). 도함수는 `'periodic'`을 지킨다 (추천안, 사용자 위임 2026-11-23) |
| IP49 `integrate`의 끝 | 끝이 NaN이면 결과가 NaN이다(`[scipy와 다름]`: scipy는 `ValueError('Integral bounds not in order')`; 평가점의 NaN과 같은 규칙). `'periodic'`의 온전한 주기와 나머지는 감소하는 중단점에서도 [min x, max x]에서 센다. 조각의 적분은 Horner(cₑ/(e + 1))·s이고 조각들의 합은 배열의 pairwise 트리(§3.5)다 (추천안, 사용자 위임 2026-11-23) |
| IP50 메서드 인자의 세부 | `derivative(nu=-k)`는 `antiderivative(nu=k)`, `antiderivative(nu=-k)`는 `derivative(nu=k)`다(scipy와 같다). `__call__`의 `nu`는 0 이상의 int다(API.md §2의 int 규칙이라 `nu=1.0`은 `TypeError`; scipy는 받는다). `PPoly(axis=)`는 다른 sd 함수처럼 음수 축을 받는다(scipy의 `PPoly`는 음수를 에러로 한다; `CubicSpline`은 받는다). 결과의 0차원은 Python 스칼라, 배치의 `roots`는 중첩 list다 (추천안, 사용자 위임 2026-11-23) |
| IP51 비정상 수 계수의 바닥 | 3차 계열의 계수 계약(§4)과 PCHIP 단조성의 (1)·(2)에 비정상 수의 절대 바닥을 더한다: \|ĉ₃ − c₃\|에 2⁻¹⁰⁷⁴(1 + 3/h + 2/h²), \|ĉ₂ − c₂\|에 2⁻¹⁰⁷⁴(2 + 3/h), \|sᵢ\| ≤ 3(1 + 4u)\|Δ\|에 2⁻¹⁰⁷⁴. 결과가 비정상 수인 반올림(D의 나눗셈, 합, h로 나누기)은 각각 절대 2⁻¹⁰⁷⁵까지 틀리고 상대 한계로는 덮이지 않는다. 메인 세션의 빠른 묶음에서 hypothesis가 찾았다: x = (1, 7, 8), y = (0, 2⁻¹⁰²², 1 + 2⁻¹⁰²²)의 첫 구간은 끝 기울기가 둘 다 0이고 c₃ = −2.06×10⁻³¹⁰, c₂ = 1.85×10⁻³⁰⁹라 도함수가 s = h에서 정확한 0이 아니라 −1.2×10⁻³²²(약 −24·2⁻¹⁰⁷⁴)였다. 커널을 고치지 않는다: 그 크기의 계수는 double로 상대 정밀도를 가질 수 없고 `c`는 계수 그 자체라 다른 척도로 둘 수 없으며, 평가값의 단조성(PCHIP의 (3), 평가 계약의 2⁻¹⁰⁷⁴ 항)은 그대로 선다. 같은 자료에서 안쪽 노드의 w₁/Δ가 넘쳐 기울기가 0이 되는 것도 기울기 계약(40u·max\|Δ\|) 안이다(scipy도 같은 식이라 같은 값) (추천안, 사용자 위임 2026-11-23) |
| IP52 `nu`와 생성자의 `None` | `__call__`의 `nu`가 차수보다 크면 0이고, NaN 평가점과 `extrapolate=False`의 바깥은 NaN이다(`[scipy와 다름]`: scipy는 바깥에서도 0; `PPoly`와 같은 규칙). `derivative(nu > k)`는 scipy처럼 `ValueError`다(차수 −1의 B-스플라인이 없다). `BSpline(extrapolate=None)`은 클래스의 기본 `True`다(§2.3의 표; `[scipy와 다름]`: scipy는 `bool(None)`이라 `False`) (추천안, 사용자 위임 2026-11-23) |
| IP53 배치 계의 피벗 | 피벗 없는 소거는 증명이 서는 계에만 쓴다: 도함수 조건이 없을 때(완전 비음, 성장 1), `make_interp_spline`의 매듭에서 각 끝의 도함수 차수가 1, …, r일 때(끝 블록이 삼각, 성장 1), k = 3의 natural(성장 < 2, §3.4). 그 밖의 차수 집합과 주어진 매듭의 도함수 조건은 처음부터 부분 피벗(LAPACK `gbtf2`의 꼴)이다: 피벗 없는 성장이 첫 간격의 1/h²으로 자란다(k = 5의 {2, 3}에서 1.1×10⁸). 피벗 없는 분해가 0이나 유한하지 않은 피벗을 만나도 부분 피벗으로 다시 하고, 그것도 막히면 `LinAlgError`다 (추천안, 사용자 위임 2026-11-23) |
| IP54 `PPoly.from_spline`의 중단점 | 기본 구간 [t[k], t[n]]의 서로 다른 매듭이다. 그래서 세 바깥 규칙 모두에서 B-스플라인과 같은 함수다. `[scipy와 다름]`: scipy는 모든 매듭(겹친 것 포함)이라 폭이 0인 조각이 생기고(IP43이 받지 않는다) `extrapolate=False`·`'periodic'`에서 B-스플라인과 다른 함수다(주기가 t[−1] − t[0]). 계수는 조각의 k + 1개 계수를 차분한 저차 스플라인의 값이다(§3.4) (추천안, 사용자 위임 2026-11-23) |
| IP55 de Boor 재귀의 무게와 계약 | 값 단계의 무게는 두 몫 (t − x)/(t − t), (x − t)/(t − t)다(scipy는 hh/(t − t)를 먼저 나눠 간격이 비정상 수면 넘친다: subnormal/x에서 scipy의 k = 1 평가는 inf, k ≥ 2 배치 행렬은 inf). 평가 계약은 γ_{6k+1}·Σ\|cⱼ\|B̃ⱼ(x) + (6k + 2)·2⁻¹⁰⁷⁴·(1 + Σ\|cⱼ\|)(B̃: 항을 절댓값으로 한 같은 재귀), 만들기 계약은 성분마다 2γ_{8k+8}\|A⁻¹\|\|A\|\|ĉ\|다(§3.4, §4; 처음 안의 ‖·‖∞ 꼴을 대신한다) (추천안, 사용자 위임 2026-11-23) |
| IP56 periodic의 풀이 | scipy의 접힌 계(n − 1개 미지수)를 열을 ⌊k/2⌋ 돌린 띠 A_b와 구석 UVᵀ로 나눠 Woodbury로 푼다(scipy와 같은 꼴). A_b는 완전 비음이고 대각이 양수라 피벗 없이, 작은 계(k 이하)는 부분 피벗의 가우스 소거로 푼다. n ≤ k는 scipy의 (n + k − 1)² 계를 부분 피벗으로 푼다 (추천안, 사용자 위임 2026-11-23) |
| IP57 `BSpline.integrate`의 바깥 | `extrapolate=False`는 [a, b]와 기본 구간의 겹침을 적분한다(배치도 같다; `[scipy와 다름]`: scipy는 열 하나면 FITPACK `splint`로 겹침을 세고, 배치면 두 끝이 한쪽 바깥일 때 NaN이다). 끝이 NaN이면 NaN(IP49), complex 계수도 적분한다(scipy는 에러) (추천안, 사용자 위임 2026-11-23) |
| IP58 `from_power_basis`와 `design_matrix`의 세부 | `from_power_basis`는 `CubicSpline`만 받는다(scipy와 같다; 다른 형은 `TypeError`, scipy는 `NotImplementedError`), 배치도 받는다(scipy는 열 하나), 모르는 `bc_type`은 `ValueError`(scipy는 `TypeError`). `design_matrix`의 `x`는 1차원이고 비어도 된다((0, n); scipy는 빈 `x`에서 에러), 색인은 int64다(`sd.sparse`의 색인 형) (추천안, 사용자 위임 2026-11-23) |
| IP59 역미분의 누적합 | 열마다 차례 합이다(numpy `cumsum`의 순서, scipy `splantider`와 같은 연산). §3.9의 처음 안(길이로 정한 블록)을 대신한다: 10⁶ 계수의 누적합이 1 ms 아래라 블록이 필요 없고, 열 묶음으로 나눠도 열마다 같은 연산이다 (추천안, 사용자 위임 2026-11-23) |
| IP60 `make_interp_spline`의 작은 경우 | `bc_type='periodic'`이고 k = 0, 1이면 scipy처럼 끝값을 검사한 뒤 주기가 아닌 스플라인(`extrapolate=True`)이다. 끝점의 도함수 조건이 넘치면(간격이 비정상 수) 그 사실을 말하는 `ValueError`다(scipy도 `ValueError`, "infs or NaNs"). 점은 둘 이상이어야 한다(다른 1차원 보간과 같다) (추천안, 사용자 위임 2026-11-23) |
| IP61 k = 1의 길 | t[1..n]이 엄격히 증가하면 노드 t[1..n], 값 c[:n]의 선형 길(IP41)로 평가한다: `make_interp_spline(k=1)`과 `BSpline(t, c, 1)`이 세 바깥 규칙 모두에서 그 길과 같은 비트이고 기본 구간에서 `sd.interp`와 같은 비트다. 안쪽 매듭이 겹치면(뛰는 꺾은선) de Boor다 (추천안, 사용자 위임 2026-11-23) |
| IP62 `add_xi`의 무게 | 더한 뒤의 무게는 모든 노드를 입력 순서로 처음부터 다시 곱한 것이라 같은 노드로 새로 만든 객체와 비트까지 같다. 사용자가 `wi`를 주었어도 그렇다(§3.6). `[scipy와 다름]`: scipy는 있던 무게에 새 인수를 곱해 고치고(용량 척도 4/(max − min)를 곱한 인수), 주어진 `wi`에도 그 인수를 곱해 척도가 섞인다. `wi`를 주어도 노드가 서로 다른지 검사한다(scipy는 `wi`를 주면 검사하지 않는다) (추천안, 사용자 위임 2026-11-23) |
| IP63 `FloaterHormannInterpolator`의 점 | 점은 실수, 1차원, 유한하고 서로 달라야 한다(`ValueError`). 순서는 어느 것이든 받아 scipy처럼 정렬하고(무게는 정렬한 점의 식이다), 값이 유한하지 않은 표본(배치면 그 행의 어느 열이든)은 scipy처럼 버린다. `d`는 남은 표본 수 n에 대해 0 ≤ d < n이어야 한다. `[scipy와 다름]`: scipy는 같은 점을 `numpy.unique`로 말없이 하나만 남기고, `d`를 버리기 전의 점 수와 견준다(버린 뒤 d ≥ n이면 무게가 모두 0이라 결과가 NaN이다) (추천안, 사용자 위임 2026-11-23) |
| IP64 차수를 넘는 도함수 | `Barycentric`·`Krogh`의 n차 이상 도함수(n은 노드 수)는 0이고 NaN 평가점에서는 NaN이다(IP52와 같은 규칙; `[scipy와 다름]`: scipy는 NaN 점에서도 0). 그 아래 차수의 NaN 점은 계산 그대로 NaN이다 (추천안, 사용자 위임 2026-11-23) |
| IP65 유리식의 결과 모양 | `FloaterHormannInterpolator(axis=k)`의 결과는 다른 1차원 보간처럼 평가점의 모양 전체가 `axis` 자리에 들어간다(§2.1). `[scipy와 다름]`: scipy의 `_BarycentricRational`은 평가점의 첫 축만 `axis`로 옮겨, 2차원 이상의 평가점과 0이 아닌 `axis`에서 모양이 다르다 (추천안, 사용자 위임 2026-11-23) |
| IP66 `wi`의 척도와 `yi`의 모양 | `BarycentricInterpolator.wi`는 참 무게에 공통의 2의 거듭제곱을 곱해 가장 큰 크기가 [0.5, 1)에 오게 한 것이다(공식은 공통 배수와 무관하다; scipy의 무게는 용량 척도가 곱해진 다른 배수다). `yi`는 값의 원래 모양(노드가 `axis`를 따라)이다(`[scipy와 다름]`: scipy의 `yi`는 내부의 (n, r) 꼴이다). 값이 없을 때(`yi=None`) 부르면 `ValueError`다(scipy는 내부 에러) (추천안, 사용자 위임 2026-11-23) |
| IP67 AAA의 인자와 `converged` | `rtol`은 `None`이거나 유한하고 0 이상, `clean_up_tol`도 유한하고 0 이상, `clean_up`은 `True`·`False`, `max_terms`는 1 이상의 int다(`ValueError`·`TypeError`; scipy는 `rtol`과 `clean_up_tol`을 검사하지 않는다). 유한한 표본이 하나도 없으면 `ValueError`다(scipy는 빈 배열의 `argmax` 에러). 걸음은 min(`max_terms`, 표본 수)까지다(모든 표본이 지지점이면 오차 0으로 멈춘다). `converged`는 멈춤 판정 max\|f − r\| ≤ rtol·max\|f\|가 섰는지다: scipy는 마지막 걸음(m = `max_terms` − 1)에서 수렴해도 경고하므로 그 경우만 scipy의 경고와 뜻이 다르다. 열을 정규화한 SVD에서 노름이 0인 열은 그대로 둔다(scipy는 0으로 나눠 NaN) (추천안, 사용자 위임 2026-11-23) |
| IP68 무게중심 평가의 세부 | 합은 노드 순서의 차례 합이다(분자는 항마다 fma 하나). 노드와의 거리가 비정상 수라 sⱼ = wⱼ/(x − xⱼ)가 넘치면 가장 가까운 노드의 값이다(scipy는 inf/inf = NaN). 무한한 평가점은 유리식(`AAA`, `FloaterHormannInterpolator`)이면 극한 Σwf/Σw(scipy와 같다; 배치면 열마다, scipy는 모든 열을 한 수로 더한다), 다항식(`Barycentric`)이면 NaN이다(scipy와 같다). 복소 평가는 역수를 2의 거듭제곱으로 맞춘 뒤 구해 제곱이 넘치거나 언더플로하지 않는다 (추천안, 사용자 위임 2026-11-23) |
| IP69 Krogh의 평가 | 값은 Newton 꼴의 Horner(점마다 fma n − 1번), 도함수는 중심을 x로 옮기는 Horner를 거듭하는 것이다: m번째 걸음이 끝나면 m번 자리가 p⁽ᵐ⁾(x)/m!이다. 값과 각 차수가 한 걸음 줄의 같은 비트라 `__call__`, `derivative(der=m)`, `derivatives()`의 m번 행이 비트까지 같다. scipy는 값을 거듭제곱 곱의 합으로, 도함수를 따로 된 재귀로 계산해 끝자리가 다르다(계약 안). 점의 순서는 입력 그대로이고 Leja 순서로 바꾸지 않는다(§9의 19의 남은 질문): 반복 노드의 도함수 자료는 순서에 묶여 있고, 바꾸면 scipy와 같은 입력에서 반올림이 달라지며, 계약(§4)이 불안정의 크기를 그대로 보인다 (추천안, 사용자 위임 2026-11-23) |
| IP70 Higham 한계의 확인 | §4의 `Barycentric` 한계를 이 구현의 연산 순서로 증명해 O(u²) 없는 꼴로 고쳤다: N개 노드에서 \|p̂ − p\| ≤ (γ₃ₙ₊₁Σ\|ℓⱼyⱼ\| + γ₃ₙΛ\|p\|)/(1 − γ₃ₙ₋₁Λ)(N = n + 1의 Higham 꼴로 (3n + 4)u, (3n + 3)u 첫 차; 처음 안의 Higham 2004 (3n + 2)u보다 한 칸 큰 것은 마지막 나눗셈을 Λ 항에 넣어서다). 무게의 반올림(입력 순서의 곱, 지수를 따로 셈)이 들어 있다. 증명은 §3.6 (추천안, 사용자 위임 2026-11-23) |
| IP71 `clean_up`의 SVD와 큰 표본 | 다시 맞추는 SVD는 표본 행이 지지점 수 이상이면 `gesdd`의 'S'다(V^H는 'A'와 같은 행렬이다). scipy는 'A'라 U가 (표본 수)²이고 10⁵ 표본이면 80 GB다. AAA의 열은 걸음마다 더해 M × (걸음 수)만 든다(scipy는 M × `max_terms`를 먼저 잡는다) (추천안, 사용자 위임 2026-11-23) |
| IP72 다항식·유리식의 속도 사례 크기 | 평가 사례(`Barycentric`, `FloaterHormannInterpolator`, AAA)는 10⁵점이다(§5의 IP0 측정은 10⁶점). scipy는 (평가점 × 노드) 배열로 평가해 10⁶점·101노드에서 프로세스가 2.4 GB를 쥔다. `Krogh` 평가는 노드 20개라 10⁶점 그대로다 (추천안, 사용자 위임 2026-11-23) |
| IP73 격자의 축 | `points`는 1차원 배열의 튜플이나 리스트다(아니면 `TypeError`). 축마다 1점 이상, 유한, 엄격히 증가하거나 감소해야 한다(`ValueError`; `[scipy와 다름]`: scipy는 무한한 노드를 받는다). 감소하는 축은 만들 때 뒤집고 `grid`와 `values`도 뒤집은 것이다(scipy와 같다). 점이 하나인 축은 `linear`·`nearest`에서 계산에 들지 않는다(그 축을 따라 상수, scipy와 같다) (추천안, 사용자 위임 2026-11-23) |
| IP74 격자의 유한하지 않은 값 | `cubic`·`quintic`·`pchip`은 값이 유한해야 한다(`ValueError`; scipy도 에러다: 반복 풀이의 "RHS must contain only finite numbers"와 PCHIP의 검사). 방법을 바꾼 호출도 그때 검사한다. `linear`·`nearest`·`slinear`는 IEEE로 퍼뜨리고, 그 규칙은 1차원 선형 길의 것이다: NaN·inf 노드는 그 점의 무게가 0이 아닐 때만 결과에 든다. `[scipy와 다름]`: scipy의 `linear`는 꼭짓점 무게의 합이라 무게가 0인 NaN도 0·NaN으로 칸 전체에 퍼지고, `slinear`는 반복 풀이라 에러다(sd의 `slinear`는 풀이가 없는 `linear`와 같은 함수다) (추천안, 사용자 위임 2026-11-23) |
| IP75 격자 옵션의 검사 | `bounds_error`는 `True`·`False`만(`TypeError`; scipy는 참·거짓으로 칠 수 있는 무엇이든), `fill_value`는 `None`이거나 수(bool과 배열은 `TypeError`; scipy는 배열과 `True`도 받는다)이고 실수 값에 complex는 scipy처럼 `ValueError`, 속성은 float·complex로 둔다. `nu`는 d개의 0 이상 int의 시퀀스다(스칼라는 `TypeError`, float는 API.md §2의 int 규칙으로 `TypeError`; scipy는 스칼라에 `ValueError`, float를 받는다). 속성은 읽기만 한다 (추천안, 사용자 위임 2026-11-23) |
| IP76 평가점의 꼴 | `xi`는 (…, d) 배열, d의 배수 길이인 1차원 배열(scipy처럼 (−1, d)로 읽는다), d개 좌표의 튜플(배열과 스칼라가 섞여도 scipy처럼 브로드캐스트)이다. 스칼라만의 튜플은 한 점이라 결과가 Python 스칼라다(0차원 배열이 없다; scipy는 0차원 배열). 스칼라 `xi`는 `ValueError`(scipy는 내부 `IndexError`). 범위 밖의 점은 평가와 한 번에 찾고 가장 작은 축을 말한다(`one of the requested xi is out of bounds in dimension i`, scipy의 문장) (추천안, 사용자 위임 2026-11-23) |
| IP77 격자 선형의 꼴 | 축마다 구간과 t는 1차원 선형 길의 것(`base/interval.hpp`, `lerp_t`)이고 2ᵈ 꼭짓점을 마지막 축부터 `lerp`로 줄인다. `nearest`도 같은 t다. `slinear`는 `nu`가 없거나 모두 0이면 이 길이고, 아니면 k = 1 텐서곱 B-스플라인의 de Boor다. `fill_value=None`의 바깥은 같은 커널의 늘이기라 무한한 좌표는 그 직선의 값(±inf, 또는 끝값이 같으면 NaN)이다(`[scipy와 다름]`: scipy는 inf·0으로 NaN) (추천안, 사용자 위임 2026-11-23) |
| IP78 텐서곱 스플라인의 풀이와 평가 | 계수는 축 0부터 차례로, 축마다 `make_interp_spline`의 배치 계(bspline.cpp의 행·띠·기저 그대로, 피벗 없이 풀고 막히면 부분 피벗)를 그 축의 모든 줄에 푼다(줄 묶음 P개 × 열 8개 묶음, 줄과 열마다 같은 연산). 차수마다 한 번 만들어 객체에 둔다(scipy는 방법을 바꾼 호출마다 다시 만든다). 평가는 (k + 1)ᵈ 계수를 마지막 축부터 bspline.cpp의 합(첫 곱, 그 뒤 fma)으로 줄인다. 그래서 1차원은 `make_interp_spline`과 같은 비트다. 기저와 구간은 bspline.cpp와 함께 쓰는 `interpolate/deboor.hpp`다(옮긴 것뿐이라 B-스플라인의 비트는 그대로) (추천안, 사용자 위임 2026-11-23) |
| IP79 텐서곱 스플라인의 계약 | 만들기: \|ĉ − c\| ≤ Σ_a (⊗_{b>a}\|A_b⁻¹\|)·2γ_{8k+8}\|A_a⁻¹\|\|A_a\|\|R̂_a\| 성분마다(R̂_a는 축 a까지 푼 계수, 1차원 계약을 뒤 축의 정확한 풀이로 옮긴 것; `refdata.tensor_collocation_bound`). 평가: \|ŷ − Σⱼ cⱼ∏_a B_{jₐ}(xₐ)\| ≤ γ_{d(6k+1)}Σⱼ\|cⱼ\|∏_a B̃_{jₐ} + d(6k + 2)·2⁻¹⁰⁷⁴·(1 + 2ᵈΣⱼ\|cⱼ\|)(축마다의 1차원 평가 계약을 쌓은 것: γₐ(1 + γₐ) + γₐ ≤ γ₂ₐ; `refdata.tensor_bound`). 끝에서 끝은 평가 한계에 계수 한계 × ∏B̃를 더한 것이다(§4) (추천안, 사용자 위임 2026-11-23) |
| IP80 `pchip`의 창 | 축마다 구간 j의 값은 노드 w₀ … w₀ + 3(w₀ = clamp(j − 1, 0, n − 4))의 값으로만 정해진다(j, j + 1의 기울기가 읽는 노드가 모두 그 안이다). 4ᵈ개 값을 마지막 축부터 1차원 PCHIP으로 줄이고, 기울기는 cubic.cpp와 함께 쓰는 `interpolate/pchip.hpp`, 계수와 Horner는 cubic.cpp·ppoly.cpp의 연산 그대로다. 그래서 1차원은 `PchipInterpolator`와, d차원은 scipy의 알고리즘(축마다 `PchipInterpolator`를 접기)에 sd의 `PchipInterpolator`를 넣은 것과 같은 비트다 (추천안, 사용자 위임 2026-11-23) |
| IP81 격자 선형의 계약 | §4의 처음 안 "γ₆d·Σw_c\|v_c\| 꼴"을 증명한 꼴로 바꾼다: 축마다의 1차원 선형 계약을 쌓은 재귀(층마다 linear_bound(\|â\|, \|b̂\|, t)에 입력의 한계 × (1 − t), t; 바깥은 \|1 − t\|, \|t\|; `refdata.grid_linear_exact`)이고, 안쪽 점에서 닫힌 꼴로 \|ŷ − y\| ≤ d·(γ₁₃M + 3η)(M은 2ᵈ 꼭짓점 값의 최대 크기)다. 1차원 계약이 \|a\| + \|b\|의 꼴이라 무게 Σw_c\|v_c\|의 꼴은 이 커널로는 증명되지 않는다 (추천안, 사용자 위임 2026-11-23) |
| IP82 격자의 속도 사례 | IP0의 사례 여섯에 `quintic` 평가 100², 10⁵점을 더한다. `cubic`·`quintic`의 scipy 비교는 반복 풀이라 값 검사를 하지 않는다(`differs`) (추천안, 사용자 위임 2026-11-23) |
| IP90 같은 자료점의 검사 | 만들기 전에 자료점을 (좌표, 색인) 순서로 정렬해 같은 점의 묶음을 찾고, 묶음 안에 `smoothing`이 0인 점이 둘 이상이면 `LinAlgError`다(두 행이 같아 계가 정확히 특이하다). 메시지는 그런 짝 가운데 둘째 점의 색인이 가장 작은 짝을 말한다. 한쪽이라도 `smoothing`이 양수면 풀이에 맡긴다. `neighbors=`에서도 만들 때 검사한다: 두 점이 한 이웃 집합에 들지 않을 수도 있지만 한 자리에 값이 둘인 자료는 보간의 뜻이 없다 (추천안, 사용자 위임 2026-11-23) |
| IP91 `epsilon`과 거리 | r = ε·‖x − y‖이고 거리는 spatial의 `euclidean` 짝 커널(spatial.md SPT26의 순서와 조심하는 길)이라 `cdist`와 같은 비트다. `[scipy와 다름]`: scipy는 εx와 εy를 먼저 반올림한 뒤 거리를 잰다(반올림 크기만 다르다). `epsilon`은 유한하고 양수여야 한다(`ValueError`; scipy는 0과 음수를 받고, 음수는 εy끼리의 거리라 절댓값처럼 된다) (추천안, 사용자 위임 2026-11-23) |
| IP92 계의 검사와 풀이 | 단항식 행렬 P(n × q)의 랭크를 풀기 전에 numpy `matrix_rank`의 판정(`gesdd`의 특잇값, 허용 s_max·max(n, q)·ε)으로 보고, q보다 작으면 그 랭크를 말하는 `LinAlgError`다(scipy는 풀이가 실패한 뒤에만 보고, 실패하지 않으면 직선 위의 점에서 10¹¹ 크기의 값을 낸다). 풀이는 `sytrf`(아래 삼각) + `sytrs`(이 마일스톤이 더한 linalg 내부 래퍼)이고, `sytrf`의 정확한 0 피벗이나 유한하지 않은 계수는 "singular"의 `LinAlgError`다. 꼬리의 shift·scale은 scipy의 (max ± min)/2를 0.5·max ± 0.5·min으로 계산한다(넘침이 없고, 반이 비정상 수인 경우를 빼면 같은 비트) (추천안, 사용자 위임 2026-11-23) |
| IP93 평가점의 모양 | `RBFInterpolator.__call__(x)`와 `NearestNDInterpolator.__call__(xi)`는 (…, d) 배열을 받고 결과는 그 앞 모양에 값의 나머지 모양을 붙인 것이다(scipy의 RBF는 (Q, d)만). 한 점((d,))이고 값이 스칼라면 Python 스칼라다. `NearestNDInterpolator`의 `xi`는 d개 배열(이나 수)의 튜플도 받아 브로드캐스트한다(IP31). `NearestNDInterpolator`는 d = 1도 받는다(scipy는 좌표가 둘 이상이어야 한다) (추천안, 사용자 위임 2026-11-23) |
| IP94 유한하지 않은 평가점 | NaN 좌표가 있는 점은 NaN이다(§2.1). 무한한 좌표는 전역 RBF가 그 식을 그대로 계산한다(scipy와 같다; `gaussian`이면 다항식 꼬리의 값). 이웃이 있어야 하는 길(`neighbors=`의 국소 RBF, `NearestNDInterpolator`)에서는 유한하지 않은 점이 NaN이다(`[scipy와 다름]`: scipy는 KD 트리의 `ValueError`) (추천안, 사용자 위임 2026-11-23) |
| IP95 `neighbors ≥ n` | k ≥ n이면 모든 점의 이웃 집합이 전체로 같으므로 만들 때 전역 계를 한 번 푼다. 이웃 길로 푼 것과 같은 계라 같은 비트이고(`test_rbf.py`), 특이한 계의 에러가 부를 때가 아니라 만들 때 난다. scipy는 부를 때마다 n점의 계를 푼다 (추천안, 사용자 위임 2026-11-23) |
| IP96 국소 계의 묶음 | 평가점마다 `KDTree.query(k)`의 색인을 정렬해 집합으로 하고, 평가점을 (집합의 사전 순서, 점 번호)로 정렬해 같은 집합끼리 묶는다. 집합마다 그 점들만의 shift·scale로 (k + q)차 계를 한 번 풀고(scipy와 같다) 그 집합의 평가점을 계산한다. 집합끼리 병렬이고 국소 계도 LAPACK `sytrf`와 같은 대입이다(작은 계의 대입은 IP101). 한 점의 값은 그 집합만으로 정해져 다른 점, 점의 순서, 부르기의 나눔, 스레드 수와 무관하고, 그 집합의 자료점으로 만든 전역 RBF와 비트까지 같다. 실패한 집합이 여럿이면 점 번호가 가장 작은 점의 집합을 말한다 (추천안, 사용자 위임 2026-11-23) |
| IP97 RBF의 계약 | §4의 RBF 계약을 이렇게 정한다. (1) 풀이: 성분마다 \|Âĉ − d\| ≤ γ_{4N}(\|Â\| + \|L̂\|\|D̂\|\|L̂\|ᵀ)\|ĉ\|(Higham 정리 11.3의 Bunch–Kaufman 꼴, N = n + q; Â는 계산한 계, L̂·D̂는 그 `sytrf` 인수이고 `sd.linalg.ldl`이 같은 분해다)이고 증가 인자 ‖\|L̂\|\|D̂\|\|L̂\|ᵀ‖∞/‖Â‖∞를 함께 적는다. (2) 평가: 계산한 계수·shift·scale의 정확한 함수 대비 Σⱼ\|ĉⱼ\|Eⱼ(1 + γ_N) + γ_NΣⱼ\|ĉⱼ\|\|φⱼ\| + Σₖ\|b̂ₖ\|\|mₖ\|(γ_{3gₖ}(1 + γ_N) + γ_N), Eⱼ = 1.01γ_{d+5}\|rφ′(r)\| + γ₄\|φⱼ\| + 8·2⁻¹⁰⁷⁴(r의 상대 반올림 γ_{d+5}는 거리 커널과 ε의 곱, γ₄는 핵 식의 반올림; libm의 `log`·`exp`가 1 ULP라는 spatial.md의 가정 L). 끝에서 끝은 계의 조건수가 정해 계약이 없고 정확도 표에 scipy와 나란히 적는다 (추천안, 사용자 위임 2026-11-23) |
| IP98 `NearestNDInterpolator`의 세부 | 동점은 작은 색인이다(`KDTree.query(k=1)`의 (거리, 색인) 순서). `[scipy와 다름]`: scipy의 트리는 같은 거리의 점 가운데 하나를 정해 두지 않고 고른다(시험 자료의 늘인 격자 198점 가운데 1점에서 다른 점을 골랐다). `rescale=True`는 scipy의 뜻(축마다 평균을 빼고 범위로 나눈다, 범위가 0이면 1)이고 평균은 배열의 pairwise 합이다. 값은 float64·complex128이고 유한하지 않아도 된다(고르기만 한다). `rescale`은 `True`·`False`만 받는다 (추천안, 사용자 위임 2026-11-23) |
| IP99 선형 커널의 블록 꼴 | `base/lerp.hpp`에 `lerp_block`(점, 두 노드, 두 값에서 t와 값)과 `lerp_ab_block`(두 값과 t)을 더한다: 256점 블록을 한 번의 벡터화된 고리로 계산하고 연산은 `lerp_t`의 흔한 길과 `lerp`의 유한한 길(`detail::lerp_finite`, `lerp`가 함께 쓴다) 그대로라 같은 비트다. 덮지 못한 점(NaN·무한한 점, 넘치는 폭, 유한하지 않은 끝값)이 있는 블록만 그 점을 스칼라 커널로 다시 계산한다. `sd.interp`(실수 값), 이 영역의 선형 길(열 하나), RGI의 `linear`(실수 값, 꼭짓점 64개 이하: 블록의 모든 꼭짓점 값을 먼저 모아 격자 읽기를 겹친다)가 쓴다. 정렬한 평가점의 `sd.interp`(노드 10³)가 3.23 → 2.74 ms(numpy 2.70), RGI `linear` 1000²가 30.4 → 16.3 ms(1스레드), 4.46 → 2.20 ms(10스레드)다(§5) (추천안, 사용자 위임 2026-11-23) |
| IP100 RBF 핵의 `log`·`exp`와 레인 | `thin_plate_spline`의 log와 `gaussian`의 exp는 배열 영역의 NEON 커널(`array/kernels/vmath.hpp`)의 스칼라 짝 `log1`·`exp1`이다(전: libm). 평가는 평가점 넷을, 계 채우기는 한 열의 칸 둘을 NEON 레인으로 나란히 계산하고 칸마다 스칼라와 같은 연산(거리의 fma 사슬, sqrt, εr, 핵, 자료점 차례의 fma)이라 같은 비트다. 거리의 조심하는 길과 vmath가 덮지 못한 인자(r = 0, |r²| > 708)는 그 레인만 스칼라다. 비트가 바뀌는 것은 두 핵뿐이고(옛 값: libm, 새 값: vmath; 둘 다 1 ULP 안, vmath는 Arb 대비 최대 1 ULP를 잰 것) 평가 계약의 가정 L(1 ULP)이 그대로라 계약은 바뀌지 않는다. 평가 10⁴점(n = 10³)이 55.0 → 19.0 ms(scipy 57.5), 만들기 n = 10³이 7.8 → 5.8 ms다 (추천안, 사용자 위임 2026-11-23) |
| IP101 RBF 계의 대입 | `sytrf`(Accelerate) 뒤의 대입을 LAPACK `dsytrs`의 단계(교환, L D y = b, Lᵀx = y; 뒤 대입의 내적은 i mod 4의 부분합 넷을 (s₀ + s₁) + (s₂ + s₃)로 묶는다, 크기로만 정한 순서)를 그대로 쓴 `ldlt_solve`로 한다. 국소 계(N = 53)에서 라이브러리의 `dsytrs`가 3.5 µs로 분해(4.25 µs)만큼 들었다. 모든 RBF 결과의 끝자리가 바뀐다(대입의 반올림 순서). 풀이 계약(Higham 정리 11.3의 Bunch–Kaufman 꼴)은 같은 대입이라 그대로다. 분해까지 직접 쓰는 작은 계의 LDLᵀ(IP96이 IP7로 미룬 것)는 쓰지 않는다: 분해의 이득은 4 µs 아래이고 피벗이 `sd.linalg.ldl`(계약 시험이 쓰는 분해)과 같아야 한다. `neighbors=50`이 178 → 145 ms(IP100의 계 채우기 포함)이고, linalg의 내부 래퍼 `lapack::sytrs`는 이 영역에서 더 쓰지 않는다 (추천안, 사용자 위임 2026-11-23) |
| IP102 점 평가의 병렬 문턱 | `sd.interp`, 이 영역의 선형 길, `PPoly` 평가의 작업량 추정을 점마다 2·log₂n + 8(전: log₂n + 4; 노드 10³의 무작위 순서에서 잰 4 ns에 맞춘다)로, 태스크를 1024점(전: 4096점)으로 한다. 2,000–10,000점의 호출이 10스레드에서 나뉜다(노드 10³, 5000점의 `sd.interp`가 21 → 13 µs). 점마다 한 커널이라 결과는 나눔과 무관하다 (추천안, 사용자 위임 2026-11-23) |
| IP103 짧은 배치의 삼중대각 | `CubicSpline`의 열 8개 이하 길(IP44)은 행렬을 저장하지 않는다: 안쪽 행의 원소는 노드에서 그 자리에서 만들고 사슬의 피벗과 위 대각은 레지스터에 두며, 뒤 대입에 쓸 피벗의 역수만 배열로 둔다(전: 배열 다섯). periodic의 열 8개 이하 길도 분해, 결합 열과 자료의 앞 대입을 한 번에 훑고 뒤 대입 둘을 한 번에 한다. 같은 연산, 같은 순서라 같은 비트다. 10⁶점 not-a-knot이 12.05 → 8.6 ms, periodic이 17.4 → 10.1 ms(scipy 12.1, 22.2)다 (추천안, 사용자 위임 2026-11-23) |
| IP104 `PPoly.from_spline`의 기저 | 조각마다의 차수 5 이하 기저는 펼친 `basis_fixed`(같은 연산)이고, d! = 1, 2로 나누기는 역수의 곱이다(2의 거듭제곱이라 같은 몫의 같은 반올림). 같은 비트이고 10⁶ 조각이 31.9 → 21.3 ms(scipy 28.4)다 (추천안, 사용자 위임 2026-11-23) |
| IP105 쓰지 않는 길 | (1) 긴 열 하나의 삼중대각을 병렬 순환 소거나 양끝에서 만나는 분해로 하지 않는다(§3.9가 IP7로 미룬 것): IP103 뒤 10⁶점의 분해 사슬(행마다 나눗셈·곱·뺄셈, 약 16 사이클)은 8.6 ms 중 3.6 ms이고, 사슬을 반으로 줄여도 1.3배 아래인데 모든 `CubicSpline`의 비트가 바뀐다. (2) AAA의 QR 갱신(§3.6)은 나중 후보로 둔다: 만들기가 이미 scipy보다 빠르고(abs 9.5 ms 대 12.0) 같은 알고리즘의 반올림만 바뀐다. (3) `NearestNDInterpolator`와 `neighbors=`의 질의는 spatial의 `KDTree.query`가 시간의 거의 전부라(10⁶점의 408 ms 중 408 ms) 이 영역에서 고칠 것이 없다 (추천안, 사용자 위임 2026-11-23) |
| IP106 코드 리뷰의 고침 | (1) `spline_slopes`의 열 8개 이하 길과 periodic, `check_x`·`all_finite`, `BSpline.from_power_basis`의 계산이 크기와 상관없이 GIL을 쥐었다: 문턱(2¹⁵) 이상이면 놓는다(API.md §3). (2) aaa.cpp는 언제나 GIL을 놓았다(`nb::gil_scoped_release`): 다른 커널처럼 작업량 문턱의 `parallel::Unlocked`로 한다. (3) `neighbors=k`의 국소 계는 태스크마다 (k + q)²을 검사 없이 잡았다: 전역 계와 같은 물리 메모리 검사를 만들 때 한다(`MemoryError`). (4) `AAA`·`FloaterHormannInterpolator`의 메서드를 틀리게 부르면 비공개 기반 클래스 `_Rational`의 이름이 나왔다(`_Rational.roots() got an unexpected keyword argument ...`): 클래스마다 제 이름의 사본을 둔다(API.md §2.1의 내부 이름 규칙) (추천안, 사용자 위임 2026-11-23) |
| IP107 쓰지 않는 linalg 래퍼 | IP6 마일스톤이 `csrc/linalg/lapack.hpp`에 더한 내부 래퍼 `lapack::sytrs`(`dsytrs`/`zhetrs`)를 뺀다: IP101이 RBF의 대입을 이 영역의 `ldlt_solve`로 바꾼 뒤 부르는 곳이 없다. 공개 linalg와 모든 결과는 그대로다. IP4의 `ggev`는 AAA·Floater–Hormann의 극점·영점이 쓰므로 둔다 (추천안, 사용자 위임 2026-11-23) |
| IP108 메서드의 틀린 호출 모양 | 보간 객체의 메서드(`integrate`, `derivative`, `antiderivative`, `roots`, `solve`, `extend`, `insert_knot`, `poles`, `residues`, `clean_up`, `add_xi`, `set_yi`, 클래스 메서드 `from_spline`·`basis_element`·`design_matrix`·`from_power_basis` …)도 틀린 호출 모양이 그 이름으로 시작한다(`integrate(): takes 2 positional arguments but 3 were given`). 객체를 부르는 `__call__`은 객체의 클래스 이름이다(`CubicSpline(): unexpected keyword argument 'foo'`; 물려준 `PPoly`의 이름이 아니다). `_calls.methods`와 새 `_calls.call`이 감싼다. 감싸기는 호출마다 수십 ns이고, 이 객체들의 한 번 호출은 인자 검사와 배열 만들기로 µs 단위라 API.md §2.1의 작은 메서드 예외에 넣지 않는다 (추천안, 사용자 위임 2026-11-23) |
| IP109 `DataFrame` 입력 | 데이터 인자는 `sd.frame`의 `DataFrame`도 받는다: 열을 나란히 둔 float64 2차원 배열(결측은 NaN)이고 `sd.spatial`이 점 집합을 받는 것과 같다. 흩어진 점(`RBFInterpolator`의 `y`·평가점, `NearestNDInterpolator`의 `x`·`xi`, RGI의 `xi`)에 쓰고, 1차원이어야 하는 인자에 주면 차원의 `ValueError`다. 메시지의 받는 것 목록은 `a frame Series or DataFrame`이다 (추천안, 사용자 위임 2026-11-23) |
| IP110 피클 | 모든 보간 객체는 `pickle`·`copy`·`deepcopy`가 된다. `sd.ndarray`가 피클되지 않으므로(`sd.spatial`의 SPT56과 같은 까닭) `__reduce__`가 객체의 슬롯을 내보내고 배열은 바이트·모양·dtype·쓰기 금지 여부로 바꾼다. 다시 만들지 않고 슬롯을 그대로 되돌리므로(만든 계수, AAA의 고른 지지점, 캐시한 극점까지) 같은 비트를 계산한다. 다시 만들기(KDTree의 꼴)는 RBF의 전역 풀이·AAA의 반복을 다시 하고 Accelerate의 스레드 모드에 따라 끝자리가 달라질 수 있어 고르지 않았다 (추천안, 사용자 위임 2026-11-23) |
| IP111 노드 하나의 상수 | 노드(지지점)가 하나인 무게중심 꼴(`BarycentricInterpolator`, `AAA`, `FloaterHormannInterpolator`; 실수와 복소수)은 유한한 모든 점에서 그 값 그대로다(무한한 점의 극한도 그 값; 다항식의 무한한 점은 그대로 NaN). 둘째 무게중심 공식대로 계산하면 (s·y)/s가 두 번 반올림되어 끝자리가 1 ULP 틀린다. 같은 상수인 `KroghInterpolator`(Horner)와 맞는다 (추천안, 사용자 위임 2026-11-23) |
| IP112 무게의 합이 0인 유리식 | Floater–Hormann(과 AAA)의 무한한 점의 값 Σwf/Σw는 scipy와 같은 식 그대로 둔다. 정확한 산술에서 Σw = 0이면(고른 노드 11개와 d = 3처럼) 극한이 없고 계산값과 아주 큰 극점·영점은 반올림의 결과다. 반올림 안의 Σw를 0으로 보고 NaN을 내는 규칙은 문턱이 자료마다 맞지 않고 AAA의 극점·영점(LAPACK `ggev`)과 함께 정해야 해서 두지 않고, API 문서에 읽는 법을 적는다 (추천안, 사용자 위임 2026-11-23) |
| IP113 `CubicSpline`의 `bc_type` 짝 | 길이 2인 튜플은 scipy처럼 언제나 (왼쪽 끝, 오른쪽 끝)의 짝이다. `bc_type=(1, 0.0)`은 왼쪽 끝이 `1`이라 `ValueError`이고 메시지가 짝으로 읽었다는 것과 받은 원소를 말한다. 양 끝의 같은 도함수 조건은 `((1, 0.0), (1, 0.0))`이다 (추천안, 사용자 위임 2026-11-23) |
| IP114 주어진 매듭의 피벗 | 도함수 조건이 없으면 `t`를 주어도 피벗 없이 푼다(§3.4의 1: 배치 행렬이 완전 비음이라 성장 1; IP53의 글과 같다). 잰 값(기본 not-a-knot 매듭을 `t`로 준 계): 정확한 유리수 풀이와 견준 만들기 계약 단위가 10⁻⁵–10⁵배 무작위 간격 n = 40의 30벌에서 피벗 없이 k = 2, 3, 4, 5 각각 0.047, 0.036, 0.016, 0.020이고 부분 피벗은 78, 131, 2.9×10⁶, 9.8×10⁴(계약 밖: 완전 비음 행렬에서 행을 바꾸면 성분별 안정성이 깨진다)다. 성분별 역오차 max|y − Aĉ|/(|A||ĉ| + |y|)(잔차는 fma로 나눈 곱의 정확한 합)는 n = 10³, 10⁵, 10⁶, k = 2–5, 고른·Chebyshev·10⁻²–10²·10⁻⁵–10⁵배 무작위 간격 전부에서 피벗 없이 3.42u 이하, 부분 피벗은 무작위 간격에서 7.2×10¹³u까지다. 시간은 n = 10⁵ k = 3 무작위 간격 2.10 대 3.53 ms, n = 10⁶ 20.2 대 44.3 ms다. n = 10⁶, k = 5, 무작위 간격에서는 부분 피벗이 0 피벗으로 막혔다(`LinAlgError`)가 피벗 없이는 풀렸다 (추천안, 사용자 위임 2026-11-23) |
| IP115 흩어진 점 보간의 범위 | `LinearNDInterpolator`는 d = 2, 3, `CloughTocher2DInterpolator`는 d = 2, `griddata`는 `'nearest'`(모든 d), `'linear'`(d = 2, 3), `'cubic'`(d = 2)다. 1차원 `griddata`의 `'linear'`·`'cubic'`은 `ValueError`이고 메시지가 `sd.interp`와 `CubicSpline`을 짚는다(spatial_geometry.md §9의 20) (추천안, 사용자 위임 2026-11-23) |
| IP116 `fill_value` | 볼록 껍질 밖의 값이다(기본 NaN). 안팎은 `find_simplex`의 정확한 판정이다. NaN 좌표의 평가점은 언제나 NaN이고 무한한 좌표의 점은 바깥이다. `None`은 받지 않는다(spatial_geometry.md §9의 21) (추천안, 사용자 위임 2026-11-23) |
| IP117 같은 점의 값 | 같은 자료점의 값이 비트까지 같으면 받고(대표는 가장 작은 색인) 다르면 `ValueError`다. 메시지는 대표와, 값이 다른 같은 점 가운데 가장 작은 색인을 말한다(`data points 4 and 20 are equal and their values differ`; spatial_geometry.md §9의 22) (추천안, 사용자 위임 2026-11-23) |
| IP118 Clough–Tocher의 기울기 | scipy의 전역 방법(곡률을 줄이는 Gauss–Seidel 훑기)이고 `tol=1e-6`, `maxiter=400`(scipy의 기본)을 둔다. 훑기는 색인 순서 하나이고 `maxiter`에서 멈춘다. 수렴 여부는 속성 `converged`, 훑은 횟수는 `niter`다(spatial_geometry.md §9의 23; 멈춤 판정은 IP123) (추천안, 사용자 위임 2026-11-23) |
| IP119 평가점과 옵션 | `xi`는 (…, d) 배열이나 d개 배열의 튜플이다. `points` 자리에 `sd.spatial.Delaunay`를 받는다(`rescale=True`와 함께면 `ValueError`). `rescale`은 `NearestNDInterpolator`와 같은 뜻으로 둔다(평균과 범위). `simplex_tolerance`는 없다(spatial_geometry.md §9의 24) (추천안, 사용자 위임 2026-11-23) |
| IP120 속성과 피클 | 속성은 `points`, `values`, `tri`, `fill_value`(Clough–Tocher는 `grad`, `converged`, `niter`도)다. 피클은 이 영역의 규칙(IP110)대로 슬롯을 내보낸다: `tri`는 점으로 다시 만들어지는 같은 `Delaunay`이고 기울기는 다시 추정하지 않는다(spatial_geometry.md §9의 25) (추천안, 사용자 위임 2026-11-23) |
| IP121 무게중심 좌표와 값 | 평가점이 든 단체(`find_simplex`의 것)에서 λ_k = D_k/D를 spatial의 값 커널(`det2`·`det3`, 1.002u)로 구하고 Σλ_k f_k를 꼭짓점 차례로 더한다(첫 항은 곱, 나머지는 fma 하나씩). 무게가 0인 항은 건너뛴다: 자료점에서는 D_k가 D와 같은 호출이라 무게 1이고 값이 비트까지 자료이며(−0.0도), 다른 꼭짓점의 NaN·inf가 들지 않는다. D가 [2⁻⁹⁰⁰, 2⁹⁰⁰] 밖이면(비정상 수 크기나 DBL_MAX 둘레의 점) 그 점들을 2의 거듭제곱 하나로 늘이거나 줄인 사본의 행렬식으로 한다(비가 같다). Clough–Tocher의 변을 건너는 방향의 계수 g_k는 scipy의 식(이웃 무게중심의 무게중심 좌표)을 반올림한 무게중심 없이 g_k = (D + D_{k+2}(w) − D_k(w))/(3(D_k(w) − D))로 구한다(w는 이웃의 남은 꼭짓점; 분모는 음수 둘의 합이라 상대적으로 정확하다). 조각 3차식은 scipy와 같은 계수 식이고 작은 삼각형은 가장 작은 좌표의 맞은편(scipy의 첫 최소), 평가는 de Casteljau의 정한 차례다. 자료점(무게 1, 0, 0)에서는 자료 그대로다. 좌표가 2¹⁰⁰⁰을 넘는 삼각형은 변을 2⁻ᵉ배 한 점의 차로, 기울기를 2ᵉ배로 쓴다(변이 넘칠 수 있다) (추천안, 사용자 위임 2026-11-23) |
| IP122 `points`의 튜플 | `points`(과 `NearestNDInterpolator`의 `x`)는 d개 1차원 좌표 배열의 튜플이어도 된다(scipy의 `griddata((x, y), z, …)` 꼴). 길이가 다르거나 1차원이 아니면 `ValueError`다. `RBFInterpolator`의 `y`는 scipy처럼 배열만이다 (추천안, 사용자 위임 2026-11-23) |
| IP123 기울기 추정의 계산 | 방향 있는 변마다 e와 w = (e/\|e\|)/\|e\|²를 한 번, 꼭짓점마다 Q = Σ4ewᵀ의 역행렬을 한 번, 열마다 자료 항 Σ6(f_i − f_j)w를 한 번 구하고 훑기는 s = 자료 항 + Σ2(e·y_j)w와 y_i = −Q⁻¹s만 한다(scipy와 같은 반복, 반올림만 다르다). 멈춤은 한 훑기의 가장 큰 \|y_new − y_old\|가 `tol` × (그 훑기 뒤 기울기 성분의 가장 큰 크기)보다 작을 때다: 점과 값의 단위와 무관하다(`[scipy와 다름]`: scipy는 점마다 max(1, \|y\|)로 나눠 기울기가 1보다 작은 자료에서 절대 변화를 본다; 좌표를 10⁶배 한 자료에서 scipy는 일찍 멈춰 엄격한 `tol`의 값과 8×10⁻⁴ 다르고 sd는 9.5×10⁻⁹로 그대로다). 변의 길이가 2^±500 밖이면 점을 2의 거듭제곱으로 옮겨(길이의 로그 가운데로) 추정하고 기울기를 되돌린다(같은 함수; 보통의 자료는 옮기지 않는다). 열(복소수는 실수부와 허수부)은 태스크이고 열 안은 한 스레드라 스레드 수와 무관하다. 대표가 아닌 같은 점은 대표의 기울기다. `niter`는 열 가운데 가장 많은 훑기, `converged`는 모든 열이 `tol`로 멈췄는지다. 꼭짓점의 이웃은 `Delaunay.vertex_neighbor_vertices`(만들면 `Delaunay`가 쥔다)를 쓴다 (추천안, 사용자 위임 2026-11-23) |
| IP124 삼각분할 위의 보간의 계약 | 선형은 \|ŷ − y\| ≤ γ_{d+5}Σλ_k\|f_k\| + (d + 2)·2⁻¹⁰⁷⁴(spatial_geometry.md §4의 유도). Clough–Tocher는 주어진 기울기의 조각 3차식(정확한 g_k와 계수)에 대해 \|ŷ − y\| ≤ 256u·K + 32·2⁻¹⁰⁷⁴, K = (1 + max_k\|g_k\|)(max\|f\| + 삼각형의 여섯 방향 도함수 \|∂ₓf_i\|\|e_x\| + \|∂_yf_i\|\|e_y\|의 최대). 256은 연산 차례대로 센 한계(약 200u·K: 계수 오차 63u·K, de Casteljau의 반올림 36u·K, 무게 오차가 옮기는 104u·K)를 올린 것이고, 잰 값은 시험 점 집합 48개에서 3.5u·K(0.014 계약 단위)다(§4) (추천안, 사용자 위임 2026-11-23) |
| IP125 값의 유한성 | `CloughTocher2DInterpolator`는 값이 유한해야 한다(`ValueError`; 기울기가 전역 반복이라 NaN 하나가 모든 기울기로 퍼진다; IP19와 같은 규칙). `LinearNDInterpolator`는 국소 계산이라 IEEE로 퍼뜨리되 무게가 0이 아닌 점에만 든다(IP121) (추천안, 사용자 위임 2026-11-23) |
| IP126 위치 찾기와 평가 | `find_simplex`의 걷기를 spatial의 `tri::locate_each`(같은 덩이 256점, 같은 차례, 한 스레드에서 2¹²점부터는 칸 차례)로 하고 점마다 찾은 태스크에서 바로 평가한다(그 단체가 캐시에 있을 때). 작업량 추정은 점마다 128 + 16(d + 2) + 2·열 수(Clough–Tocher는 128 + 160 + 40·열 수)이고 2¹⁵ 이상이면 나누고 GIL을 놓는다 (추천안, 사용자 위임 2026-11-23) |
| IP127 `griddata` | 객체를 만들어 한 번 부른 것과 같은 비트다. `'nearest'`는 `NearestNDInterpolator`(모든 d; `fill_value`는 쓰지 않는다; 1차원 `points` 배열은 n개 점, 1차원 `xi`는 그 점들, scipy와 같다), `'linear'`·`'cubic'`은 두 클래스(기본 `tol`·`maxiter`)이고, 에러의 말은 `griddata(): `로 시작한다. 3차원 `'cubic'`은 `method 'cubic' is for 2-D points, not 3-D`다 (추천안, 사용자 위임 2026-11-23) |
| IP128 네이티브 경계 | `interpolate/scattered.cpp`는 spatial의 `_Delaunay` 객체(`DelaunayObj`, `spatial/delaunay2.hpp`로 옮겼다)를 `nb::cast`로 읽고 `tri::locate_each`, 값 커널을 쓰고, 꼭짓점의 이웃은 `Delaunay.vertex_neighbor_vertices`의 배열로 받는다(`interpolate/ → spatial/`은 이미 있는 의존이다). spatial의 공개 동작과 결과는 그대로다(`find_simplex`는 `locate_each`를 부르는 꼴이 되었다) (추천안, 사용자 위임 2026-11-23) |
| IP129 `fill_value`의 형 | 수(int, float, numpy 수)이고 bool·`None`·배열은 `TypeError`다. 값이 complex면 복소수도 되고 실수 `fill_value`는 (`fill_value`, 0)이다(scipy의 `complex(fill_value)`; 기본값의 바깥은 nan+0j). 실수 값에 복소수 `fill_value`는 `ValueError`다(RGI와 같은 말) (추천안, 사용자 위임 2026-11-23) |
| IP130 속성의 꼴 | `points`는 준 그대로(`rescale` 전; `[scipy와 다름]`: scipy의 `points`는 옮긴 좌표), `values`는 (n, …) 그대로(scipy는 (n, 곱)), `grad`는 (n, …, 2)이고 값의 dtype(scipy는 (n, 곱, 2)), 모두 쓰기 금지다. scipy의 `is_complex`, `scale`, `offset`, `values_shape`은 두지 않는다 (추천안, 사용자 위임 2026-11-23) |

### 1.2 범위
마일스톤 열은 그 이름을 만든 §8의 마일스톤이고 "나중"은 첫 판에 넣지 않은 후보다.

**이름공간**: 모든 공개 이름이 `sd.interpolate`에 바로 있다. scipy의 `from scipy.interpolate import CubicSpline`은 `from stardust.interpolate import CubicSpline`이 된다. 하위 모듈(scipy의 `fitpack`, `interpnd`, `polyint` 같은 옛 모듈 이름)은 없다.

**numpy → sd**
| numpy | sd | 마일스톤 |
|---|---|---|
| `numpy.interp(x, xp, fp, left=None, right=None, period=None)` | `sd.interp(x, xp, fp, /, *, left=None, right=None, period=None)` (배열 영역; `xp`의 검사, 키워드만, §2.2) | IP1 |

**scipy.interpolate → sd.interpolate**
| scipy | sd.interpolate | 마일스톤 |
|---|---|---|
| `PPoly` | 같은 이름. `__call__`, `derivative`, `antiderivative`, `integrate`, `roots`, `solve`, `extend`, `from_spline`, 속성 `c`, `x`, `extrapolate`, `axis` | IP2 |
| `CubicHermiteSpline` | 같은 이름(§9의 7) | IP2 |
| `CubicSpline` | 같은 이름, `bc_type`은 scipy의 넷과 (차수, 값) 튜플 | IP2 |
| `PchipInterpolator`, `pchip_interpolate` | 같은 이름 | IP2 |
| `Akima1DInterpolator` | 같은 이름, `method='akima'` 또는 `'makima'` | IP2 |
| `BSpline` | 같은 이름. `__call__`, `derivative`, `antiderivative`, `integrate`, `basis_element`, `design_matrix`(`sd.sparse.csr_array`), `insert_knot`, `from_power_basis`, 속성 `t`, `c`, `k`, `extrapolate`, `axis`, `tck` | IP3 |
| `make_interp_spline` | 같은 이름(모든 차수 k ≥ 0, `bc_type`) | IP3 |
| `BarycentricInterpolator`, `barycentric_interpolate` | 같은 이름(`rng`는 없다, §9의 18) | IP4 |
| `KroghInterpolator`, `krogh_interpolate` | 같은 이름 | IP4 |
| `AAA` | 같은 이름. `poles`, `residues`, `roots`, `clean_up`, 속성 `support_points`, `support_values`, `weights`, `errors` | IP4 |
| `FloaterHormannInterpolator` | 같은 이름(IP9) | IP4 |
| `RegularGridInterpolator`, `interpn` | 같은 이름. 방법 `linear nearest slinear cubic quintic pchip` | IP5 |
| `RBFInterpolator` | 같은 이름, 핵 여덟, `neighbors=`는 `sd.spatial.KDTree` | IP6 |
| `NearestNDInterpolator` | 같은 이름, `rescale`만(IP8, IP31) | IP6 |
| `construct_fast`, `from_bernstein_basis` | 없음: 앞의 것은 검사 없는 생성자이고(§9의 9) 뒤의 것은 `BPoly`가 있어야 한다 | — |
| `BPoly`, `NdBSpline`, `NdPPoly` | 없음, 나중(§9의 7) | 나중 |
| `make_lsq_spline`, `make_smoothing_spline`, `make_splrep`, `make_splprep`, `generate_knots`, `UnivariateSpline`과 그 무리, `SmoothBivariateSpline` 등 | 없음: 맞추기와 매끄럽게 하기는 `sd.smooth`(IP1) | 나중 |
| `LinearNDInterpolator`, `CloughTocher2DInterpolator`, `griddata` | 같은 이름, `sd.spatial.Delaunay` 위(IP115–IP130; `LinearNDInterpolator`는 d = 2, 3) | IP9 |
| `interp1d`, `interp2d`, `Rbf`, `InterpolatedUnivariateSpline`, `RectBivariateSpline`, `RectSphereBivariateSpline`, `splrep splev splint sproot spalde splder splantider insert bisplrep bisplev`, `lagrange`, `approximate_taylor_polynomial`, `pade` | 없음(§9의 4; 대응표는 API 문서에) | — |
| RGI의 `solver`, `solver_args`, 방법 `cubic_legacy`·`quintic_legacy`, `interpn`의 `splinef2d` | 없음(§3.7, §9의 20) | — |

### 1.3 공통 원칙 (다른 영역에서 이어받는 것)
- 데이터는 위치로, 옵션은 키워드로만 받는다. 별칭이 없다(§9의 11).
- **같은 기계, 같은 스레드 수, 같은 입력이면 실행마다 같은 비트다**(API.md §3). 이 영역은 스레드 수와도 무관하게 한다: 분할은 크기로만 정하고, 점 하나의 값은 어느 태스크에서 계산하든 같은 커널이다(§3.9). LAPACK을 부르는 곳(RBF의 전역 풀이, AAA의 SVD와 극점)은 Accelerate의 스레드 모드가 같을 때 같다.
- 반올림이 아닌 동작(결과의 모양과 형, 기본값, 고르는 구간, 바깥의 규칙, 경계 조건의 뜻, 에러가 나는 입력)은 scipy 1.18.1, `sd.interp`는 numpy 2.5.4와 같다(API.md §4의 규칙 2). 다르게 정한 곳은 `[scipy와 다름]`, `[numpy와 다름]`으로 이유와 함께 적는다. 반올림 크기의 결과는 정확도 계약(§4) 안이면 된다.
- 조용히 틀린 값을 돌려주지 않는다. 정렬되지 않은 가로축처럼 결과가 뜻을 잃는 입력은 에러다. numpy의 `interp`는 정렬되지 않은 `xp`에서 에러 없이 틀린 값을 내고(`interp([0.5, 1.5, 2.5], [0, 2, 1, 3], [0, 10, 20, 30])`가 `[2.5, 7.5, 27.5]`), `sd.interp`는 이것을 막는다(§2.2).
- 경고를 내지 않는다. scipy가 경고하는 곳(AAA의 수렴 실패와 Froissart 쌍, Krogh의 높은 차수, RBF의 낮은 `degree`, 복소수 평가점의 실수부만 쓰기)은 에러이거나 상태 속성이다(§2.9).
- 에러 종류와 메시지는 API.md §2·§2.1이다. 메시지는 부른 함수의 이름으로 시작한다(`interp(): ...`, 생성자는 `CubicSpline(): ...`, 메서드는 `integrate(): ...`; 객체를 부르는 `__call__`은 클래스 이름이다).
- 참값은 Arb(python-flint 0.9.0)와 정확한 유리수(`fmpq`, `fmpq_mat`)만 쓴다. mpmath는 쓰지 않는다.
- 모든 호출은 정해진 시간 안에 끝난다. 만들기와 평가는 직접 계산이고(반복 풀이 없음), 반복이 있는 곳은 입력과 무관한 상한이 있다: AAA는 `max_terms`(기본 100)와 표본 수 가운데 작은 것, 근 찾기는 조각마다 고정된 이분 횟수(§3.5), LAPACK의 고유값·특이값 분해는 LAPACK의 상한이다.

### 1.4 다른 영역과의 관계
- **배열**: `sd.interp`는 배열 영역의 함수다(IP2). 선형 커널 `base/lerp.hpp`를 배열의 `array/ops/interp.cpp`와 이 영역의 1차원 선형·k = 1 B-스플라인·격자 선형이 함께 쓴다(같은 x의 sin·cos를 함께 구하는 `base/sincos.hpp`를 배열과 special이 함께 쓰는 것과 같은 자리). 구간 찾기는 배열의 `searchsorted`와 같은 순서 규칙(NaN이 가장 크다)이다(§3.1).
- **linalg**: RBF의 전역 연립방정식은 대칭 부정치 LDLᵀ(`sytrf`; 대입은 이 영역의 `ldlt_solve`, IP101)이고 꼬리의 랭크는 `gesdd`로 본다. AAA는 `gesdd`와 일반화 고유값(`ggev`, 이 영역이 더한 내부 래퍼, §9의 17)을 `linalg/lapack.hpp`로 부른다. 스플라인의 띠 연립방정식은 이 영역의 피벗 없는 풀이다(§3.3, §9의 13). `sd.linalg.solve_banded`와는 끝자리가 다를 수 있다.
- **spatial**: `RBFInterpolator(neighbors=k)`와 가장 가까운 점 보간은 `KDTree`의 kNN을 쓴다. 동점이 (거리, 색인) 순서로 정해져 있어서(spatial.md SPT4) 이웃 집합이 실행과 스레드 수와 무관하다. scipy는 동점 순서를 약속하지 않아 같은 거리의 이웃이 갈리면 RBF의 국소 계와 가장 가까운 점의 값이 달라질 수 있다.
- **sparse**: `BSpline.design_matrix`가 `sd.sparse.csr_array`를 돌려준다(Python 층에서만 의존한다).
- **integrate**: `PPoly.integrate`는 조각 다항식의 정확한 역미분이다. `sd.integrate.quad`와 견줄 수 있지만 서로 쓰지 않는다.
- **나중의 영역**: `sd.smooth`(매끄럽게 하는 스플라인, 최소제곱 스플라인)가 `BSpline`·`PPoly`와 이 영역의 띠 풀이와 de Boor 평가를 쓴다. 그림 영역의 색띠 보간과 등고선의 내부 보간을 `sd.interp`로 옮길 수 있다. 약속이 아니라 메모다.

---

## 2. API

### 2.1 입력
- **받는 것**: 보간 객체의 생성자, 그 메서드와 호출, 함수 꼴(`pchip_interpolate`, `interpn` …)은 API.md §5의 상위 함수 입력(`sd.ndarray`, 수의 시퀀스, numpy 배열, frame의 `Series`)과 frame의 `DataFrame`(IP109)을 받는다(§9의 3). `sd.interp`는 배열 영역의 함수라 `sd.ndarray`와 Python 스칼라만 받는다.
- **가로축**(`x`, `xi`, 격자의 `points`, 매듭 `t`, 평가점): float64로 계산한다. bool·int64는 float64로 바꾼다. complex128은 `TypeError`다(`[scipy와 다름]`: scipy는 복소수 평가점의 실수부만 쓰고 `ComplexWarning`을 낸다). 예외는 `AAA`와 `FloaterHormannInterpolator`의 평가점이다: 복소평면의 점을 받는다(scipy와 같다).
- **세로축**(`y`, 계수 `c`, 격자의 `values`, RBF의 `d`): float64 또는 complex128이다. complex를 받는 곳은 scipy와 같다: `PPoly`, `CubicHermiteSpline`, `CubicSpline`, `BSpline`, `make_interp_spline`, `Barycentric`, `Krogh`, `FloaterHormannInterpolator`, `AAA`, RGI의 `pchip`을 뺀 방법, `RBFInterpolator`, `NearestNDInterpolator`, `sd.interp`. `PchipInterpolator`, `pchip_interpolate`, `Akima1DInterpolator`와 RGI의 `pchip`은 실수만이다(scipy와 같이 `ValueError`; 단조성과 기울기의 부호가 실수의 개념이다; §9의 8).
- **배치(`axis=`)**: 세로축은 여러 차원이고 `axis`(기본 0, scipy와 같다)가 가로축에 맞는 축이다. 결과의 모양은 scipy와 같다: 평가점의 모양이 `axis` 자리에 들어간다(`CubicSpline(x, y[2, 6, 3], axis=1)(x_new[4, 5])`는 (2, 4, 5, 3)). 배치의 열마다 같은 계산이고 결과가 열마다 독립이다.
- **가로축의 검사**: 1차원, 유한, 엄격히 증가해야 한다(`ValueError`; scipy와 같다). `PPoly`의 중단점과 RGI의 격자는 scipy처럼 엄격히 감소해도 된다. `Krogh`는 같은 점의 반복이 도함수 자료라서 받는다(scipy와 같다). `Barycentric`과 `FloaterHormannInterpolator`는 서로 다른 점이어야 한다(IP63). `sd.interp`의 `xp`는 줄지 않아야 한다(§2.2).
- **세로축의 유한성**: 전역 풀이로 계수를 만드는 곳(`CubicSpline`, `PchipInterpolator`, `Akima1DInterpolator`, `CubicHermiteSpline`, `make_interp_spline`, `RBFInterpolator`)은 `y`(`dydx`, `d`)가 유한해야 한다(`ValueError`; `RBFInterpolator`만 `[scipy와 다름]`이고, `check_finite=False`는 없다). 국소 계산(`sd.interp`, RGI의 값, 주어진 계수의 `PPoly`·`BSpline`)은 IEEE로 퍼뜨린다(§9의 16).
- **평가점의 NaN**: 결과는 NaN이다. 어느 보간이든 같다(§2.3).
- **결과 dtype**: float64이거나, 세로축이 complex면 complex128이다. 결과는 보통의 새 배열이다(결과 객체가 아니다). 0차원 결과는 Python 스칼라다(docs/api/array.md §1.2; scipy는 0차원 numpy 배열).

### 2.2 `sd.interp` (IP2, 배열 영역)
```
interp(x, xp, fp, /, *, left=None, right=None, period=None)
```
- numpy의 뜻이다: `xp`(1차원, float64)와 `fp`(같은 길이, float64 또는 complex128)가 주는 꺾은선을 `x`(모양 그대로, 또는 Python 스칼라)에서 계산한다. `x < xp[0]`은 `left`(기본 `fp[0]`), `x > xp[-1]`은 `right`(기본 `fp[-1]`)다. `period`가 있으면 `left`·`right`를 무시하고 `xp`와 `x`를 그 주기로 감는다(numpy처럼 감은 `xp`를 정렬하고 양 끝에 한 점씩 더한다). `xp`가 한 점이면 상수다(numpy와 같다; NaN 평가점은 NaN, IP39).
- **`xp`의 검사**: 유한하고 줄지 않아야 한다(`ValueError`, `interp(): 'xp' must be nondecreasing`; `[numpy와 다름]`: numpy는 검사하지 않는다). 같은 값의 반복은 받는다: numpy와 같이 계단(그 점에서 오른쪽 값)이다. `period`가 있으면 정렬은 검사하지 않는다(numpy처럼 감은 뒤 정렬한다). 비용은 `xp` 한 번 훑기다(10⁶점에서 0.14 ms, 같은 크기의 `interp`의 0.3%; §3.2의 측정).
- **값**: 구간 안은 §3.2의 선형 커널이다. 노드에서는 `fp`의 값 그대로이고, 단조인 `fp`에서는 결과도 단조이고 구간의 두 끝값 사이에 있다. 유한하지 않은 `fp`의 규칙은 numpy와 같다(구간 안에서 한쪽 끝이 ±inf면 그 inf, 두 끝이 반대 부호의 inf면 NaN, NaN 끝이 있으면 NaN; 노드에서는 그 값). complex `fp`는 실수부와 허수부를 따로 계산한다(numpy와 같다).
- **결과**: `x`의 모양, `fp`의 dtype이다. `x`가 Python 스칼라면 Python `float`·`complex`다(numpy는 `numpy.float64`).
- `[numpy와 다름]`: 데이터는 위치만, `left`·`right`·`period`는 키워드만이다(numpy는 `interp(x, xp, fp, 0, 1)`처럼 위치로도 받는다). `period=0`은 `ValueError`(numpy와 같다), NaN·inf `period`도 `ValueError`(numpy는 NaN이면 NaN 결과, inf면 경고와 함께 감지 않은 값을 낸다).
- **같은 비트의 약속**: `xp[0] ≤ x ≤ xp[-1]`에서 `sd.interp(x, xp, fp)`는 `make_interp_spline(xp, fp, k=1)(x)`, `BSpline(t, fp, 1)(x)`(같은 매듭), `RegularGridInterpolator((xp,), fp, method='linear' 또는 'slinear')(x)`, `interpn`의 1차원 선형과 비트까지 같다(같은 구간, 같은 t, 같은 커널, §3.2). `xp`에 반복이 없을 때다(B-스플라인과 격자는 반복을 받지 않는다). scipy에서는 이 넷이 서로 다르다: 노드 101개와 평가점 10⁴개에서 1차원 RGI `linear`와 `np.interp`는 54%, RGI `slinear`와 `linear`는 49%, k = 1 `BSpline`과 RGI `slinear`는 78%의 점에서만 비트가 같았다(`tests/interpolate/data/scipy.json`의 `bits/linear_paths`).

### 2.3 바깥의 규칙 (하나의 규칙과 그 밖)
**규칙**: 구간이 있는 1차원 보간(`PPoly`, `CubicHermiteSpline`, `CubicSpline`, `PchipInterpolator`, `Akima1DInterpolator`, `BSpline`, `make_interp_spline`의 결과)은 모두 `extrapolate=`이고 뜻이 하나다.

| 값 | 뜻 |
|---|---|
| `True` | 끝 조각의 다항식을 그대로 늘인다 |
| `False` | 정의역 `[x[0], x[-1]]`(B-스플라인은 `[t[k], t[n]]`) 밖은 NaN이다 |
| `'periodic'` | 정의역의 길이로 감아서 계산한다 |
| `None` | 생성자에서는 그 클래스의 기본, 메서드(`__call__`, `integrate`, `roots`, `solve`)에서는 객체의 값 |

- **기본**은 scipy의 것 그대로다: `True`. 예외는 둘이다: `bc_type='periodic'`인 `CubicSpline`과 `make_interp_spline`은 `'periodic'`, `Akima1DInterpolator`는 `False`다(scipy 1.18.1과 같다; §9의 10).
- 정의역의 끝점은 안이다(`x[-1]`은 마지막 구간으로 계산한다, scipy와 같다). 평가점의 NaN은 언제나 NaN이고 바깥으로 치지 않는다.

**규칙 밖** (이유는 각 함수가 따르는 라이브러리의 뜻이다)
| 대상 | 인자 | 기본 | 바깥 |
|---|---|---|---|
| `sd.interp` | `left`, `right`, `period` | 끝값으로 자른다 | numpy의 뜻(§2.2) |
| `RegularGridInterpolator`, `interpn` | `bounds_error`, `fill_value` | `ValueError` | `bounds_error=False`면 `fill_value`(기본 NaN), `fill_value=None`이면 끝 조각을 늘인다(scipy와 같다) |
| `BarycentricInterpolator`, `KroghInterpolator`, `AAA`, `RBFInterpolator`, (`FloaterHormannInterpolator`, `NearestNDInterpolator`) | 없음 | 그 식을 어디서나 계산한다 | 정의역이 없다(scipy와 같다) |

- `[scipy와 다름]`: RGI의 평가점에 NaN이 있으면 `bounds_error=True`여도 에러가 아니라 그 점이 NaN이다. scipy는 NaN을 범위 밖으로 쳐서 `ValueError`(`One of the requested xi is out of bounds`)를 내면서도 `bounds_error=False`에서는 `fill_value`가 아니라 NaN을 돌려준다. NaN은 범위 밖의 값이 아니라 정해지지 않은 값이라 다른 보간과 같이 NaN을 퍼뜨린다(§9의 21).
- 선형의 바깥(k = 1 B-스플라인, RGI의 `fill_value=None`)도 §3.2의 커널이라 1차원에서 둘은 비트까지 같다.

### 2.4 조각 다항식 (IP2)
```
PPoly(c, x, /, *, extrapolate=None, axis=0)
CubicHermiteSpline(x, y, dydx, /, *, axis=0, extrapolate=None)
CubicSpline(x, y, /, *, axis=0, bc_type='not-a-knot', extrapolate=None)
PchipInterpolator(x, y, /, *, axis=0, extrapolate=None)
Akima1DInterpolator(x, y, /, *, axis=0, method='akima', extrapolate=None)
pchip_interpolate(xi, yi, x, /, *, der=0, axis=0)
PPoly.__call__(x, /, *, nu=0, extrapolate=None)
PPoly.derivative(*, nu=1)
PPoly.antiderivative(*, nu=1)
PPoly.integrate(a, b, /, *, extrapolate=None)
PPoly.roots(*, discontinuity=True, extrapolate=None)
PPoly.solve(y=0.0, /, *, discontinuity=True, extrapolate=None)
PPoly.extend(c, x, /)
PPoly.from_spline(tck, /, *, extrapolate=None)
```
- **클래스**: 모두 Python 클래스이고 계수 배열(`c`: (k + 1, m, …), 높은 차수가 먼저; `x`: (m + 1,))을 쥔다. 뒤의 넷은 `PPoly`의 하위 클래스다(scipy와 같다). 계산은 네이티브 호출 한 번이다. 계수와 중단점은 쓰기 금지 배열이다(생성자가 복사한다; `[scipy와 다름]`: scipy는 쓸 수 있는 배열을 쥐어 고치면 객체가 바뀐다).
- **구간**: 점 x는 x[i] ≤ x < x[i+1]인 구간 i로 계산하고 x = x[-1]은 마지막 구간이다(scipy와 같다; 불연속인 조각 다항식과 도함수에서 어느 쪽 값인지가 이 규칙이다).
- **`nu`**: 0 이상의 int다. 차수보다 크면 0이고 NaN 평가점과 `extrapolate=False`의 바깥은 NaN이다(IP52와 같은 규칙). 음수는 `ValueError`(scipy와 같다; 역미분은 `antiderivative`).
- **`CubicSpline`의 `bc_type`**: `'not-a-knot'`, `'clamped'`, `'natural'`, `'periodic'`, 또는 양 끝의 `(차수, 값)` 튜플이나 그 짝(차수 1, 2; 값은 배치 모양). 끝마다 다른 조건도 받는다(`('natural', 'not-a-knot')`). `'periodic'`은 양 끝 모두여야 하고 `y[0]`과 `y[-1]`이 scipy의 허용(`allclose(rtol=1e-15, atol=1e-15)`) 안이어야 한다(§9의 14). 점이 둘이면 직선, 셋이고 not-a-knot이면 포물선이다(scipy와 같다).
- **`Akima1DInterpolator`**: `method='akima'`는 Akima(1970), `'makima'`는 수정 Akima다(scipy와 같은 식). 무게 합의 문턱 10⁻⁹·max의 max는 열마다다(IP33). makima는 평평한 구간(같은 값이 셋 이상)이 정확히 평평하다. akima는 구간 안쪽 노드의 기울기가 0이고, 구간 끝 노드가 Akima의 정하지 않은 경우(m₁ = m₂ ≠ m₃ = m₄)면 평균 기울기라 평평하지 않을 수 있다(IP47, scipy와 같다).
- **`roots`·`solve`**: 조각마다 실근을 찾아 정렬된 1차원 배열로 돌려준다. 배치면 배치 모양의 중첩 list다(`[scipy와 다름]`: scipy는 dtype `object`의 배열이고 sd의 배열에는 object dtype이 없다). `discontinuity=True`면 중단점에서 부호가 바뀌는 점도 근으로 친다. 조각이 0과 같은 다항식이면 그 구간의 왼쪽 끝과 NaN을 넣는다(scipy와 같다, `[0., nan]`). 알고리즘은 §3.5(§9의 15).
- **`integrate(a, b)`**: a > b면 부호가 바뀐다. `extrapolate=False`이고 a, b가 정의역 밖이면 NaN, `'periodic'`이면 온전한 주기의 수 × 한 주기의 적분 + 나머지다(scipy와 같다). 끝이 NaN이면 NaN이다(IP49, `[scipy와 다름]`).
- **`antiderivative`**: 상수는 x[0]에서 0이고 이어지게 고른다(scipy와 같다). `'periodic'` 객체의 역미분은 `extrapolate=False`다(주기가 아니므로; scipy 1.18.1과 같다, IP48). `derivative`와 `antiderivative`는 같은 클래스의 객체를 돌려주고, 음수 `nu`는 서로를 부른다(IP50, scipy와 같다).
- **`extend(c, x)`**: 오른쪽(또는 감소하는 중단점이면 왼쪽)에 구간을 더한다. 객체를 고치는 유일한 메서드다(scipy와 같다). 고친 뒤의 계수도 쓰기 금지 배열이다. 새 중단점은 끝 중단점을 넘어야 한다(같으면 에러, IP43).
- **중단점**: 유한하고 엄격히 단조다(IP43, `[scipy와 다름]`: scipy는 같은 중단점과 무한한 중단점을 받는다).

### 2.5 B-스플라인 (IP3)
```
BSpline(t, c, k, /, *, extrapolate=True, axis=0)
make_interp_spline(x, y, /, *, k=3, t=None, bc_type=None, axis=0)
BSpline.__call__(x, /, *, nu=0, extrapolate=None)
BSpline.derivative(*, nu=1)
BSpline.antiderivative(*, nu=1)
BSpline.integrate(a, b, /, *, extrapolate=None)
BSpline.basis_element(t, /, *, extrapolate=True)
BSpline.design_matrix(x, t, k, /, *, extrapolate=False)
BSpline.insert_knot(x, /, *, m=1)
BSpline.from_power_basis(pp, /, *, bc_type='not-a-knot')
```
- **`BSpline`**: 매듭 `t`(줄지 않음, 길이 n + k + 1), 계수 `c`(첫 축이 n), 차수 `k`(0 이상의 int). 검사는 scipy와 같다(매듭이 줄지 않음, 개수, 기본 구간 [t[k], t[n]]이 비지 않음). 평가는 de Boor다(§3.4). 정의역 안의 구간 규칙은 t[l] ≤ x < t[l+1], x = t[n]은 마지막 구간이다(scipy와 같다).
- **`make_interp_spline`**: 차수 k의 보간 스플라인이다. 매듭을 주지 않으면 scipy의 규칙(k = 0은 t = (x₀, …, xₙ₋₁, xₙ₋₁)과 c = y라 왼쪽에서 이어지는 계단이고 x ≥ xₙ₋₁에서는 y[-1], k = 1은 노드 그대로, 홀수 k는 not-a-knot, 짝수 k는 double로 계산한 이웃 노드의 중점에서 앞뒤 k/2개를 뺀 것; `bc_type`이 있으면 그에 맞는 매듭)이다. 참값 도구는 이 규칙으로 매듭을 만들고 `tests/interpolate/test_reference_data.py`가 그 매듭이 scipy의 매듭과 비트까지 같은 것을 본다. `bc_type`은 `None`, `'not-a-knot'`, `'natural'`, `'clamped'`, `'periodic'`, 양 끝의 (차수, 값) 목록이다. 짝이 맞지 않는 조건은 scipy와 같은 `ValueError`다(`k=5, bc_type='natural'`). `'periodic'`의 끝값 검사는 `CubicSpline`과 같은 `allclose(rtol=1e-15, atol=1e-15)`다(IP28).
- **k = 1**: 계수는 `y` 그대로이고(scipy와 같다), 평가는 §3.2의 선형 커널이다(IP61). 그래서 노드에서 `y`를 그대로 돌려준다(`[scipy와 다름]`: scipy의 de Boor는 시험 자료 random/50의 노드 50개 가운데 9개에서 끝자리가 달랐다).
- **`nu`**: 0 이상(scipy의 `nu=-1`은 `NotImplementedError`; sd는 `ValueError`). 차수보다 크면 0이고 NaN 평가점과 `extrapolate=False`의 바깥은 NaN이다(IP52). `derivative(nu > k)`는 `ValueError`(scipy와 같다).
- **배치 계** (IP35, IP53): 도함수 조건의 행은 그 끝의 보간 행 바로 안쪽에 둔다(왼쪽은 차수가 오르는 순서, 오른쪽은 내리는 순서). 증명이 서는 계(도함수 조건 없음, 각 끝의 차수 1, …, r, k = 3의 natural)는 피벗 없이, 그 밖은 부분 피벗으로 푼다. 정칙이 아니면 `sd.linalg.LinAlgError`다(scipy와 같은 판정).
- **`integrate`**: `extrapolate=False`는 [a, b]와 기본 구간의 겹침이다(IP57). 끝이 NaN이면 NaN.
- **`design_matrix`**: (len(x), n) `sd.sparse.csr_array`다(scipy는 `scipy.sparse.csr_array`). 행마다 k + 1개의 저장 자리이고 0인 값도 저장한다(scipy와 같다). `x`는 1차원이고 유한해야 하며 `extrapolate=False`면 기본 구간 안이어야 한다(IP58).
- `PPoly.from_spline(BSpline)`과 `BSpline.from_power_basis(CubicSpline)`가 두 표현을 잇는다. `from_spline`의 중단점은 기본 구간의 서로 다른 매듭이다(IP54). `tck`는 (t, c, k) 튜플이다.

### 2.6 다항식·유리식 (IP4)
```
BarycentricInterpolator(xi, yi=None, /, *, axis=0, wi=None)
barycentric_interpolate(xi, yi, x, /, *, axis=0, der=0)
KroghInterpolator(xi, yi, /, *, axis=0)
krogh_interpolate(xi, yi, x, /, *, der=0, axis=0)
FloaterHormannInterpolator(points, values, /, *, d=3, axis=0)
AAA(x, y, /, *, rtol=None, max_terms=100, clean_up=True, clean_up_tol=1e-13)
```
- **`BarycentricInterpolator`**: 둘째 무게중심 공식이다. 메서드 `__call__`, `derivative(x, /, *, der=1)`, `derivatives(x, /, *, der=None)`, `add_xi`, `set_yi`(scipy와 같다; 이 둘은 객체를 고친다). 무게는 정해진 순서로 지수를 따로 세는 곱이다(§3.6). `[scipy와 다름]`: `rng`가 없다(scipy는 무게의 넘침을 피하려고 점을 무작위로 섞으므로 같은 입력이 실행마다 끝자리가 다를 수 있다). 노드에서는 그 값 그대로다(scipy와 같다). `add_xi` 뒤의 무게는 새 객체의 무게와 같다(IP62), `wi`는 가장 큰 크기가 [0.5, 1)인 공통 배수이고 `yi`는 값의 원래 모양이다(IP66).
- **`KroghInterpolator`**: Newton 꼴의 Hermite 보간이다. 같은 점의 반복은 도함수 자료다(scipy와 같다). 메서드 `__call__`, `derivative`, `derivatives`. 차수가 높을 때의 경고는 없다(§9의 19). 같은 값의 노드는 붙어 있어야 한다(scipy와 같이 `ValueError`). n차 이상의 도함수는 0, NaN 점에서는 NaN이다(IP64; `Barycentric`도 같다).
- **`FloaterHormannInterpolator`**: scipy의 Floater–Hormann이다(IP9). 점은 서로 다른 실수이고 정렬해서 쓰며, 값이 유한하지 않은 표본은 버린다(IP63). 평가점은 실수나 복소수다. 메서드 `poles`, `residues`, `roots`(뒤의 둘은 1차원 값만; scipy와 같다), 속성 `weights`. 결과의 모양은 다른 1차원 보간과 같다(IP65).
- **`AAA`**: scipy의 AAA다(Nakatsukasa–Sète–Trefethen). 점과 값은 실수나 복소수이고, 유한하지 않은 표본은 버린다(scipy와 같다). `rtol` 기본은 ε^(3/4)이다(scipy와 같다). 지지점 고르기의 동점은 앞의 색인이다(scipy의 `argmax`와 같다). `[scipy와 다름]`: `max_terms`에 이르러도 경고가 없고 `converged` 속성이 `False`다. `clean_up()`이 지운 Froissart 쌍의 수를 돌려주는 것은 scipy와 같지만 경고는 없고, 인자 이름은 생성자와 같은 `clean_up_tol`이다(§9의 17, IP32). 노드(지지점)에서는 그 값 그대로다(scipy와 같다). 인자의 검사와 `converged`의 뜻은 IP67이다.

### 2.7 격자 (IP5)
```
RegularGridInterpolator(points, values, /, *, method='linear', bounds_error=True, fill_value=nan)
RegularGridInterpolator.__call__(xi, /, *, method=None, nu=None)
interpn(points, values, xi, /, *, method='linear', bounds_error=True, fill_value=nan)
```
- `points`는 축마다의 1차원 격자(엄격히 증가하거나 감소; scipy와 같다), `values`는 (n₁, …, n_d, …)이고 뒤의 축은 배치다. `xi`는 (…, d)다. 결과는 (…, 배치)다(scipy와 같다).
- **방법**: `linear`(축마다의 선형을 차례로, §3.7), `nearest`(가운데에서는 작은 색인; scipy와 같다), `slinear`(k = 1; `linear`와 같은 비트, `[scipy와 다름]`: scipy는 두 길이 끝자리가 다르다), `cubic`, `quintic`(축마다 not-a-knot 스플라인의 텐서곱), `pchip`(축마다 PCHIP). 축마다 필요한 점의 수는 scipy와 같다(`linear`·`nearest`는 1점도 받는다, `slinear` 2점, `cubic`·`pchip` 4점, `quintic` 6점).
- `cubic`·`quintic`의 계수는 직접 풀이다(§3.7). `[scipy와 다름]`: `solver`, `solver_args`가 없다. scipy는 이 둘의 계수를 반복 풀이(기본 `gcrotmk`)로 구해 결과가 그 허용오차 크기로 틀린다(2차식 자료 6 × 6의 `cubic`이 참값 25 대신 24.999999999999986이다; `tests/interpolate/data/scipy.json`의 `rgi/cubic_quadratic`).
- `fill_value`는 `values`와 맞는 형의 수이거나 `None`이다(scipy와 같이 complex를 실수 값에 주면 `ValueError`).
- `__call__`의 `nu`는 축마다의 미분 차수이고 `slinear`, `cubic`, `quintic`에서만 받는다(IP30; scipy와 같다).
- 축의 검사, 유한하지 않은 값, 옵션의 검사, 평가점의 꼴은 IP73–IP76이다. `cubic`·`quintic`의 계수는 차수마다 한 번 만들어 둔다(IP78).

### 2.8 흩어진 점 (IP6)
```
RBFInterpolator(y, d, /, *, neighbors=None, smoothing=0.0, kernel='thin_plate_spline', epsilon=None, degree=None)
RBFInterpolator.__call__(x, /)
NearestNDInterpolator(x, y, /, *, rescale=False)
NearestNDInterpolator.__call__(xi, /)
LinearNDInterpolator(points, values, /, *, fill_value=nan, rescale=False)
CloughTocher2DInterpolator(points, values, /, *, fill_value=nan, tol=1e-06, maxiter=400, rescale=False)
griddata(points, values, xi, /, *, method='linear', fill_value=nan, rescale=False)
```
- 핵은 scipy의 여덟(`linear thin_plate_spline cubic quintic multiquadric inverse_multiquadric inverse_quadratic gaussian`)이고 `epsilon`과 `degree`의 기본은 scipy와 같다(척도 불변인 넷은 `epsilon` 기본 1, 다른 넷은 `epsilon`이 있어야 한다). r = ε·‖x − y‖이고 거리는 spatial의 `euclidean` 짝 커널이다(IP91). 다항식 항은 자료의 상자로 옮기고 늘린 좌표의 단항식이다(scipy와 같은 차례).
- `neighbors=k`면 평가점마다 가장 가까운 k개 자료점의 국소 RBF다(k > n이면 n; scipy와 같다). 이웃은 `sd.spatial.KDTree`의 (거리, 색인) 순서다(§1.4). 같은 이웃 집합의 평가점은 한 번 푼 계수를 함께 쓴다(scipy와 같다; IP96). k ≥ n이면 만들 때 전역 계를 한 번 푼다(IP95).
- `smoothing`, 낮은 `degree`, 같은 자료점은 IP25·IP90이다. 다항식을 빼는 `degree=-1`도 핵의 최소보다 낮으면 `ValueError`다(IP29). 꼬리의 랭크는 풀기 전에 본다(IP92).
- 평가점은 (…, d) 배열이다(IP93). NaN 좌표의 점은 NaN, 이웃이 있어야 하는 길에서는 유한하지 않은 점이 NaN이다(IP94).
- `NearestNDInterpolator`(IP8, IP31, IP98): `KDTree.query(k=1)`의 색인으로 값을 고른다. 동점은 작은 색인이다. `x`는 좌표 배열의 튜플이어도 된다(IP122).
- **삼각분할 위의 보간** (IP9 마일스톤, IP115–IP130): `LinearNDInterpolator`(d = 2, 3), `CloughTocher2DInterpolator`(d = 2), `griddata`는 `sd.spatial.Delaunay`의 정확한 삼각분할과 `find_simplex` 위에 있다. 뜻, 바깥, 같은 점, 기울기의 설계는 [spatial_geometry.md](spatial_geometry.md) §2.6, 수치 방법은 §3.8과 아래 §3.8의 끝, 계약은 §4다. 속성과 에러의 말은 API 문서 §7이다.
- `[scipy와 다름]`: 음수 `smoothing`은 `ValueError`(scipy는 받는다), 같은 자료점이 있고 `smoothing`이 0이면 `LinAlgError`(scipy는 에러 없이 값을 낸다), 꼬리의 랭크가 모자라면 `LinAlgError`(scipy는 직선 위의 점에서 10¹¹ 크기의 값을 말없이 낸다), `degree`가 핵의 최소보다 낮으면 `ValueError`(scipy는 0 이상이면 경고, −1이면 말없이 받는다), `epsilon`은 유한한 양수(scipy는 0과 음수를 받는다), 이웃이 있어야 하는 길의 유한하지 않은 점은 NaN(scipy는 `ValueError`), `NearestNDInterpolator`의 동점은 작은 색인(scipy는 정해 두지 않음).

### 2.9 scipy·numpy와 다른 곳 (모음)
위의 `[scipy와 다름]`, `[numpy와 다름]`을 모은다. 이유는 그 자리에 있다.
- 공통: 옵션은 키워드만, 쓰기 금지 계수, `check_finite` 없음(언제나 검사), complex 평가점은 `TypeError`, 0차원 결과는 Python 스칼라, 경고 대신 에러나 상태 속성.
- `sd.interp`: `xp`의 검사, 키워드만인 `left`·`right`·`period`, 유한하지 않은 `period`는 `ValueError`, 한 점의 `xp`에서도 NaN 평가점은 NaN(IP39), 끝값이 둘 다 −0이면 −0, 노드 사이의 단조와 범위(numpy의 식은 오른쪽 노드 바로 왼쪽에서 구간의 범위를 넘을 수 있다: 폭이 위치와 비슷한 구간 2만 개 중 159개, §3.2).
- 1차원: k = 1 B-스플라인이 노드에서 정확, `nu < 0`은 `ValueError`, `nu > k`의 바깥은 NaN(IP52), `BSpline(extrapolate=None)`은 `True`(IP52), `from_spline`의 중단점(IP54), 비정상 수 간격의 B-스플라인(IP55), `integrate(extrapolate=False)`의 배치(IP57), `from_power_basis`의 배치와 에러 종류(IP58), `roots`의 배치 결과는 목록, `construct_fast`·`from_bernstein_basis` 없음, `PPoly`의 중단점은 유한하고 엄격히 단조(IP43), `integrate`의 NaN 끝은 NaN(IP49), `roots`가 중근과 붙은 근을 찾음(IP45), 넘치는 계수는 에러(IP36).
- 다항식·유리식: `Barycentric`의 `rng` 없음, AAA의 `converged`와 경고 없는 `clean_up`, Krogh의 경고 없음, `add_xi`가 무게를 처음부터 다시 셈(IP62), Floater–Hormann의 같은 점은 에러(IP63), 차수를 넘는 도함수의 NaN 점(IP64), 유리식 결과의 모양(IP65), `wi`의 척도와 `yi`의 모양(IP66), AAA 인자의 검사(IP67), 넘치는 sⱼ와 배치의 무한 점(IP68).
- 격자: `slinear`와 `linear`가 같은 비트, `cubic`·`quintic`의 직접 풀이와 `solver` 없음, 옛 방법 없음, NaN 평가점은 NaN, 무한한 노드는 에러(IP73), NaN 값은 무게가 0이 아닌 점에만 들고 `slinear`도 받는다(IP74), `bounds_error`·`fill_value`·`nu`의 검사(IP75), 스칼라 튜플의 결과는 Python 스칼라(IP76), `fill_value=None`의 무한한 좌표는 직선의 값(IP77).
- 흩어진 점: 음수 `smoothing`, 같은 자료점, 랭크가 모자란 꼬리, 낮은 `degree`, 0이나 음수의 `epsilon`이 에러(IP25, IP29, IP90–IP92), ε·거리의 r(IP91), (…, d) 평가점(IP93), 이웃 길의 유한하지 않은 점은 NaN(IP94), `NearestNDInterpolator`의 동점은 작은 색인(IP98).
- 삼각분할 위의 보간: 정확한 Delaunay(가까운 점을 버리지 않음)와 정확한 안팎(허용오차와 `simplex_tolerance` 없음), 자료점에서 자료 그대로(IP121), 값이 다른 같은 자료점은 에러(IP117), `LinearNDInterpolator`는 d = 2, 3만(IP115), 1차원 `griddata`의 `'linear'`·`'cubic'`은 에러(IP115), 경고 대신 `converged`(IP118), 단위와 무관한 멈춤 판정(IP123), `points`·`values`·`grad`의 꼴(IP130).

---

## 3. 수치 방법

### 3.1 구간 찾기
- 평가점마다 구간 번호를 이분 탐색으로 찾는다. 규칙은 오른쪽 열린 구간(x[i] ≤ x < x[i+1], 마지막은 닫힘)이고 정의역 밖은 끝 구간(늘이기)이나 NaN(`False`)이다. 평가점이 정렬되어 있으면 앞 점의 구간에서 시작하는 질주 탐색이 빠르다. 어떤 탐색을 쓰든 구간은 규칙으로 정해지므로 결과는 같다.
- **구현** (IP1 마일스톤, IP37): `base/interval.hpp`가 점 아래(같은 값 포함)의 노드 수 c를 센다. `sd.interp`는 c를 numpy의 `binary_search_with_guess`처럼 쓰고(c = 0이면 `left`, c = n이면 마지막 노드 또는 `right`, 그 밖은 [xp[c − 1], xp[c]]; 반복 노드는 폭이 0인 구간을 고르지 않아 계단이 된다), 이 영역은 c − 1을 첫 구간과 끝 구간으로 자른다(`interpolate/search.hpp`). 평가점을 위치만으로 256점 블록으로 나누고, 블록이 정렬되어 있으면(NaN 없음) 앞 점의 c에서 1, 2, 4, …칸씩 질주한 뒤 마지막 배가 구간을 반씩 줄이고, 아니면 8점을 나란히 갈래 없이 반씩 줄인다(`searchsorted`의 탐색). 두 길은 같은 c를 내므로 결과는 블록의 길, 태스크 나누기, 스레드 수와 무관하다. 정렬된 평가점에서 노드 10⁶개, 10⁶점의 `sd.interp`가 1스레드 8.5 ms(numpy 9.57), 무작위 순서는 15.2 ms(numpy 172)다(§5).
- scipy의 `BSpline`은 앞 점의 구간에서 한 칸씩 걷는다. 그래서 무작위 순서의 평가점에서 점마다 O(n)이다: IP0 마일스톤의 탐침에서 평가점 10⁵개가 매듭 10³개에서 11.0 ms, 10⁴개에서 88.5 ms였고 10⁶개 매듭에 10⁶점은 3분 넘게 끝나지 않았다. sd는 순서와 무관하게 O(log n)이다.
- 감소하는 중단점(`PPoly`, RGI)은 만들 때 뒤집은 사본으로 찾는다(값의 계산은 원래 구간의 다항식이다).

### 3.2 선형 커널 (`base/lerp.hpp`; IP1)
점 x가 구간 [x₀, x₁]에 있고 끝값이 a, b일 때(IP4, IP26, IP27):

1. t = fl(fl(x − x₀)/fl(x₁ − x₀)). 폭 fl(x₁ − x₀)이나 fl(x − x₀)가 넘치면 x, x₀, x₁에 1/2을 곱한 값으로 같은 식을 쓴다(1/2을 곱한 두 값의 차는 넘치지 않는다; 뒤의 것은 정의역에서 아주 먼 늘이기에서만 생긴다, IP38). ±inf인 점의 t는 ±inf다.
2. t = 0이면 r = a, t = 1이면 r = b다. 그 밖에서 a와 b의 부호가 다르면(0 포함) r = fl(fl(t·b) + fl(fl(1 − t)·a)), 아니면 r = fl(a + fl(t·fl(b − a)))를 b 쪽으로 넘은 값에서 b로 자른다(t ≤ 1에서 b > a면 min(r, b), b < a면 max(r, b); t > 1이면 반대). 유한하지 않은 끝값은 아래의 갈래다.

- 이 꼴은 C++20 `std::lerp`(P0811)의 구성에 양 끝의 갈래(IP27)를 더한 것이다. 코드는 두 식을 모두 계산하고 고르는 갈래 없는 꼴로 쓴다(IP26; 같은 비트).
- **블록 꼴** (IP7 마일스톤, IP99): `lerp_block`·`lerp_ab_block`이 256점 블록을 한 번의 벡터화된 고리로 계산한다(`lerp_t`의 흔한 길과 유한한 끝값의 꼴 그대로라 같은 비트). 덮지 못한 점(NaN·무한한 점, 넘치는 폭, 유한하지 않은 끝값)이 있는 블록만 그 점을 위의 스칼라 커널로 다시 계산한다.
- **성질과 증명.** 반올림이 단조라는 것(u ≤ v면 fl(u) ≤ fl(v))과, 표현 가능한 c에 대해 |v| ≤ |c|면 |fl(v)| ≤ |c|라는 것만 쓴다.
  1. *t의 단조와 끝*: x ↦ fl(x − x₀)는 단조이고 폭은 양수로 고정이라 t는 x에 대해 줄지 않는다. x = x₀이면 fl(x − x₀) = 0이라 t = 0, x = x₁이면 분자와 분모가 같은 수라 t = 1이고, x₀ ≤ x ≤ x₁이면 0 ≤ fl(x − x₀) ≤ fl(x₁ − x₀)이라 0 ≤ t ≤ 1이다. 1/2을 곱하는 갈래도 같은 논증이다.
  2. *양 끝에서 정확*: 2번의 첫 두 갈래다(0의 부호까지).
  3. *a = b면 a*: 0이 아니면 같은 부호 갈래에서 fl(b − a) = 0이라 r = fl(a + 0) = a, 둘 다 0이면 반대 부호 갈래에서 t·a + (1 − t)·a의 두 곱이 a와 같은 부호의 0이라 합도 a다(−0 둘이면 −0).
  4. *t에 대해 단조*: 같은 부호 갈래에서 D = fl(b − a)의 부호가 고정이라 t ↦ fl(a + fl(t·D))는 단조이고, b에서 자르는 것(min이나 max와 상수)도 단조를 지킨다. b > a면 0 < t < 1에서 fl(t·D) ≥ 0이라 r ≥ a이고 자른 값은 b 이하라, t = 0의 a와 t = 1의 b에 이어진다. 반대 부호 갈래(b ≥ 0 ≥ a)에서 t ↦ fl(t·b)는 줄지 않고, t ↦ fl(1 − t)는 늘지 않으며 a ≤ 0을 곱해 줄지 않는 함수가 되어 그 합의 반올림도 줄지 않는다. 이 식의 t = 0, 1에서의 값은 a, b와 0의 부호만 다를 수 있다. 갈래는 a, b로만 정해지므로 t에 대해 바뀌지 않는다. b < a도 대칭이다.
  5. *0 ≤ t ≤ 1이면 a와 b 사이*: 반대 부호 갈래에서 |t·b| ≤ |b|라 fl(t·b)는 0과 b 사이, fl(1 − t) ∈ [0, 1]이라 fl(fl(1 − t)·a)는 a와 0 사이이고, 합은 a와 b 사이라 그 반올림도 그렇다. 같은 부호 갈래는 4의 r ≥ a와 자르기다.
  - 1과 4를 합치면 r은 x에 대해 단조이고, 2로 노드에서 정확하며, 5로 구간 끝값 사이에 있다. 그래서 단조인 자료의 꺾은선은 전체가 단조이고, 노드의 구간을 어느 쪽으로 골라도 값이 같아 `sd.interp`, k = 1 B-스플라인, 격자 선형의 1차원이 구간 규칙이 달라도 같은 비트다(§2.2).
- **오차 한계.** t̂ = t(1 + θ₃), |θ₃| ≤ γ₃(t는 x의 정확한 값으로 정한 비율; 뺄셈 둘과 나눗셈 하나). y = a + t(b − a), M = max(|a|, |b|), η = 2⁻¹⁰⁷⁵(곱이 비정상 수가 될 때 반올림의 절대 오차; 비정상 수의 합과 차는 정확하다).
  - 같은 부호 갈래: r̃ = (a + t(b − a)(1 + θ₅))(1 + δ)라 |r̃ − y| ≤ γ₅(1 + u)|t||b − a| + u|y|이고 |b − a| ≤ M이다. 0 ≤ t ≤ 1이면 |y| ≤ M이라 (γ₅(1 + u) + u)M ≤ γ₆M, 늘이기(t < 0 또는 t > 1)에서는 T = max(1, |t|)로 |y| ≤ |a| + |t||b − a| ≤ (1 + |t|)M ≤ 2TM이라 (γ₅(1 + u)|t| + 2uT)M ≤ (γ₅(1 + u) + 2u)TM ≤ γ₇TM이다(두 부등식은 분모를 곱해 정리하면 각각 6u − 36u² ≤ 6u − 30u², 7u − 54u² + 35u³ ≤ 7u − 35u²). IP0 마일스톤의 판은 이 자리에 |t|를 썼는데, −1 < t < 0이면 마지막 합의 반올림 u|y| ≈ u|a|가 |t|에 비례하지 않아 틀렸다(IP40; 참값 시험의 uniform/6, x = −5×10⁻³²⁴에서 t ≈ −2.5×10⁻³²³인데 오차가 9×10⁻³²⁴였다). y는 자르는 쪽의 안에 있으므로 자르기는 오차를 늘리지 않는다.
  - 반대 부호 갈래: fl(t̂·b)는 t·b와 γ₅|t||b|(합의 반올림 포함) 안이고, fl(fl(1 − t̂)·a)(합의 반올림 포함)와 (1 − t)a의 차는 (γ₃|t| + γ₃|1 − t| + γ₃²|t|)|a| 이하다. 0 ≤ t ≤ 1이면 γ₃ + γ₃² ≤ γ₄라 합은 γ₅(|a| + |b|) 이하, 늘이기에서는 |t| ≤ T, |1 − t| ≤ 1 + |t| ≤ 2T라 (3γ₃ + γ₃²)T|a| + γ₅T|b| ≤ γ₁₀T(|a| + |b|)다.
  - 비정상 수: 곱 둘(같은 부호 갈래는 하나)의 언더플로가 각 η 이하이고 마지막 합의 (1 + u)를 더해도 3η 아래다. t̂ 자체가 비정상 수일 때의 η·|b − a| ≤ 2ηM은 위 부등식의 여유(u² 크기) 안이다.
  - 그래서 **0 ≤ t ≤ 1이면 |r − y| ≤ γ₆(|a| + |b|) + 3η, 늘이기에서는 γ₁₀·max(1, |t|)·(|a| + |b|) + 3η**다(§4). 노드에서는 0이다. 1/2을 곱하는 갈래(IP27, IP38)는 1/2을 곱한 값이 정확한 한 같은 t̂이고, x₀가 비정상 수라 반올림되어도 그 차(2⁻¹⁰⁷⁵ 이하)는 넘칠 만큼 큰 x − x₀에 묻힌다.
- **유한하지 않은 끝값**: t가 0이나 1이면 그 끝값, 0 < t < 1이면 한쪽만 ±inf일 때 그 inf, 같은 부호의 inf 둘이면 그 inf, 반대 부호면 NaN, NaN이 있으면 NaN이다. numpy의 결과와 같다(`tests/interpolate/data/numpy.json`의 `nonfinite/*`). 식 2번은 이 경우를 따로 갈라 계산한다. 늘이기(k = 1 B-스플라인, RGI의 `fill_value=None`)에서 끝값이 유한하지 않으면 a + t(b − a)를 IEEE로 계산한 값이다.
- 복소수는 실수부와 허수부에 같은 t로 따로 쓴다.
- **측정** (`tools/interpolate_probe.py kernels`, `tools/interpolate_probe_kernels.cpp`, sd의 플래그 -O3 -mcpu=native -ffp-contract=off). 무작위 구간 3000개(끝값의 크기 2⁻⁶–2⁶, 같은 값·거의 같은 값·반대 부호를 섞음)에서 끝점 둘레의 이웃 double 40개씩과 고른 점 200개, 모두 84만 6천 점:

  | 안 | 끝이 틀린 구간 | a = b가 틀린 구간 | 단조가 깨진 구간 | 범위를 넘은 구간 | 10⁶점, 노드 10³ 무작위 / 정렬 (ms) | 노드 10⁶ 무작위 / 정렬 (ms) |
  |---|---:|---:|---:|---:|---:|---:|
  | (a) 위의 꼴, 갈래 | 0 | 0 | 0 | 0 | 32.4 / 9.0 | 86.8 / 50.4 |
  | (a) 위의 꼴, 갈래 없음(IP26) | 0 | 0 | 0 | 0 | 10.7 / 9.4 | 48.5 / 30.1 |
  | (b) numpy: fma(s, x − x₀, a) | 1364 | 0 | 0 | 689 | 8.8 / 8.3 | 44.5 / 27.4 |
  | (c) fma(t, b − a, a) | 1109 | 0 | 0 | 531 | 8.8 / 8.4 | 43.9 / 27.3 |
  | (d) (1 − t)a + tb | 0 | 299 | 741 | 529 | 8.8 / 8.3 | 43.9 / 27.3 |

  - 갈래 없는 꼴은 갈래 꼴과 모든 점에서 비트가 같았다. 갈래 꼴이 무작위 순서에서 느린 것은 끝값의 부호와 t로 가르는 갈래의 예측 실패다. 갈래 없는 꼴은 numpy 식보다 무작위 순서에서 1.09–1.22배, 정렬한 평가점에서 1.10–1.13배 느리다. `xp`의 검사(IP5)는 10⁶점에서 0.14 ms다.
  - numpy 2.5.4 휠의 `interp`는 fl(fma(s, x − xⱼ, fⱼ)), s = fl((fⱼ₊₁ − fⱼ)/(xⱼ₊₁ − xⱼ))와 4만 점에서 모두 같았다(소스의 `slope*(x - xp[j]) + fp[j]`를 컴파일러가 FMA로 합친 것; 합치지 않은 식과는 28,079점만 같다). 이 식은 왼쪽 노드에서만 정확하고(numpy는 x = xⱼ를 따로 갈라 그 값을 낸다), 폭이 위치와 비슷한 구간 2만 개 가운데 159개에서 오른쪽 노드 바로 왼쪽의 값이 구간의 범위를 넘었다. 간격이 비정상 수인 자료(subnormal/x)에서는 s가 넘쳐 구간 안이 inf다. sd는 끝자리가 numpy와 다를 수 있고(계약 안, §4), 단조와 범위는 sd만 지킨다.
- **0의 부호**: 끝값이 둘 다 −0이면 구간 안도 −0이다(3). numpy는 +0을 낸다(fma(0, d, −0) = +0). `[numpy와 다름]`이다.

### 3.3 3차 계열의 기울기와 계수 (IP2)
모든 3차 계열은 노드의 기울기 sᵢ를 구한 뒤 Hermite 꼴로 구간의 계수를 만든다: Δᵢ = (yᵢ₊₁ − yᵢ)/hᵢ, c₃ = (sᵢ + sᵢ₊₁ − 2Δᵢ)/hᵢ², c₂ = (3Δᵢ − 2sᵢ − sᵢ₊₁)/hᵢ, c₁ = sᵢ, c₀ = yᵢ (scipy `CubicHermiteSpline`과 같은 꼴). 계수의 연산 순서는 scipy의 것 하나이고(IP42: t = ((s₀ + s₁) − 2D)/h, c₃ = t/h, c₂ = (D − s₀)/h − t) 모든 3차 계열이 같은 함수로 만든다(IP15). 코드는 `interpolate/cubic.cpp`(기울기, Hermite 계수)와 `interpolate/banded.hpp`(피벗 없는 삼중대각 풀이)다.
- **`CubicSpline`**: 기울기의 삼중대각 계(hᵢ sᵢ₋₁ + 2(hᵢ₋₁ + hᵢ) sᵢ + hᵢ₋₁ sᵢ₊₁ = 3(hᵢ Δᵢ₋₁ + hᵢ₋₁ Δᵢ))와 끝의 조건이다. not-a-knot은 scipy처럼 첫 행과 끝 행을 삼중대각에 맞춘 꼴로 쓴다. periodic은 순환 계이고 scipy는 띠 풀이 두 번과 한 자리 보정으로 푼다(Sherman–Morrison 꼴).
  - **풀이** (IP16, IP44, IP103): 피벗 없는 Thomas 소거다. 계의 행렬은 배치의 모든 열에 같으므로 한 번 분해하고 열마다 앞·뒤 대입을 한다(열이 8개 이하면 분해·오른쪽·앞 대입을 한 번에 훑는다; 같은 연산이라 같은 비트). 열은 행 안에서 이어져 있어 대입 한 걸음이 한 행의 열을 한꺼번에 계산한다. 분해의 사슬은 lᵢ = aᵢ/uᵢ₋₁, uᵢ = bᵢ − lᵢcᵢ₋₁이고 뒤 대입은 사슬 밖에서 만든 wᵢ = fl(1/uᵢ)을 곱한다. 열이 8개 이하면 행렬을 저장하지 않고(안쪽 행의 원소는 노드에서 그 자리에서 만든다) wᵢ만 배열로 둔다(IP103; 같은 연산이라 같은 비트).
  - **피벗 없는 풀이의 증명** (IP2 마일스톤). 행렬(scipy의 행): 안쪽 행 i는 [hᵢ, 2(hᵢ₋₁ + hᵢ), hᵢ₋₁]이고, 끝 행은 natural·(2, v)가 [2h₀, h₀], clamped·(1, v)가 [1, 0], not-a-knot이 [h₁, x₂ − x₀](끝은 [xₙ₋₁ − xₙ₋₃, hₙ₋₃]), 점 셋의 양 끝 not-a-knot은 포물선의 [1, 1, 0], [h₁, 2(h₀ + h₁), h₀], [0, 1, 1]이다. 모든 원소가 0 이상이다.
    1. *정확한 피벗은 양수다.* 가정 uᵢ₋₁ ≥ cᵢ₋₁(앞 행의 위 대각)이면 lᵢcᵢ₋₁ = hᵢcᵢ₋₁/uᵢ₋₁ ≤ hᵢ라 uᵢ ≥ 2hᵢ₋₁ + hᵢ ≥ cᵢ = hᵢ₋₁이다. natural(2h₀ ≥ h₀)·clamped(1 ≥ 0)로 시작하면 가정이 처음부터 서고, not-a-knot으로 시작하면 u₀ = h₁ < c₀ = h₀ + h₁이지만 l₁ = h₁/h₁ = 1이라 u₁ = 2(h₀ + h₁) − (h₀ + h₁) = h₀ + h₁ ≥ c₁ = h₀로 둘째 행부터 선다. 끝 행은 natural이 2hₙ₋₂ − hₙ₋₂hₙ₋₃/uₙ₋₂ ≥ hₙ₋₂, clamped가 1, not-a-knot이 hₙ₋₃(1 − (hₙ₋₃ + hₙ₋₂)/uₙ₋₂) > 0(uₙ₋₂ ≥ 2hₙ₋₃ + hₙ₋₂; 점 셋에서 natural로 시작하면 u₁ = 2h₀ + 1.5h₁)이고, 포물선은 1, 2h₀ + h₁, (h₀ + h₁)/(2h₀ + h₁)이다. 그래서 L ≥ 0, U ≥ 0이고 |L||U| = LU = A다.
    2. *계산한 피벗도 양수다.* 안쪽 행과 natural·clamped 끝에서 빼는 값 fl(lᵢcᵢ₋₁)은 대각의 반 이하라(ĉᵢ₋₁/ûᵢ₋₁ ≤ 1 + O(u)) 상쇄가 없다. not-a-knot 시작의 l₁은 같은 double h₁끼리의 나눗셈이라 정확히 1이다. not-a-knot의 끝 피벗만 상쇄가 있고 fl(d/ûₙ₋₂) ≤ 1이면 0 이상이다. 그 조건은 hₙ₋₃ ≳ 6u(hₙ₋₃ + hₙ₋₂)일 때 서고, 아니면 IP44의 에러다(그때는 반올림한 행렬이 작업 정밀도에서 특이하다).
    3. *역오차.* 연산마다 반올림 하나다(-ffp-contract=off). 분해 ûᵢ = (bᵢ − l̂ᵢcᵢ₋₁(1 + δ₁))(1 + δ₂), l̂ᵢ = aᵢ(1 + δ₃)/ûᵢ₋₁에서 L̂Û = Â + ΔA₁, |ΔA₁| ≤ γ₁|L̂||Û|; 앞 대입은 |ΔL| ≤ γ₂|L̂|; 뒤 대입 ŝᵢ = (ẑᵢ − cᵢŝᵢ₊₁(1 + δ))(1 + δ)·ŵᵢ(1 + δ), ŵᵢ = (1 + δ)/ûᵢ는 |ΔU| ≤ γ₄|Û|(IP44의 역수가 하나를 더한다)다. 그래서 (Â + ΔA)ŝ = b̂, |ΔA| ≤ (γ₁ + γ₂ + γ₄ + γ₂γ₄)|L̂||Û| ≤ γ₇|L̂||Û|이고, 2에서 L̂, Û ≥ 0이라 |L̂||Û| = |L̂Û| = |Â + ΔA₁| ≤ (1 + γ₁)|Â|, 곧 |ΔA| ≤ γ₈|Â|다(Higham, *Accuracy and Stability of Numerical Algorithms*의 LU 역오차와 같은 셈을 삼중대각에서 원소마다 한 것). 원소 자체의 반올림(h, 2(h + h), x₂ − x₀; 상대 γ₂)을 더하면 정확한 자료의 행렬 A에 대해 |δA| ≤ γ₁₀|A|다. 오른쪽은 항마다 상대 γ₁₁ 안이다(D의 γ₃, 곱과 합; not-a-knot 행이 연산이 가장 많다). |b|ₐ를 항의 절댓값으로 만든 오른쪽이라 하면 |δb| ≤ γ₁₁|b|ₐ다.
    4. *앞 오차.* A(ŝ − s) = δb − δAŝ에서 **|ŝ − s| ≤ γ₁₁|A⁻¹|(|A||ŝ| + |b|ₐ)** (모든 경계 조건, 성분마다; `tests/interpolate/refdata.spline_bound`가 γ₁₂로 잰다).
    5. *따름* (natural, clamped): 행마다 대각이 비대각 합의 두 배라 D = diag(A)⁻¹로 B = DA = I + E, ‖E‖∞ ≤ 1/2, ‖B⁻¹‖∞ ≤ 2, ‖B‖∞ ≤ 3/2이고 |A⁻¹||A| = |B⁻¹||B|다. ‖D|b|ₐ‖∞ ≤ (3/2)M(M = maxⱼ|Dⱼ|), ‖s‖∞ ≤ 3M이라 |ŝᵢ − sᵢ| ≤ γ₁₁(3·3M + 2·(3/2)M)(1 + O(u)) ≤ 150uM이다. not-a-knot에는 이 따름이 없다: 끝 행이 대각 우세가 아니어서 간격이 10¹⁰배 다른 spacing/1e10에서 |ŝ − s|/(uM)이 7×10⁹이다(4의 꼴로는 0.07 계약 단위).
    6. *periodic* (scipy의 압축 꼴): Ac(순환 계의 앞 n − 2행·열)는 natural과 같은 양의 대각 우세 삼중대각(첫 행 [2(hₙ₋₂ + h₀), hₙ₋₂])이라 두 풀이(자료의 오른쪽과 결합 열)가 1–4 그대로다. 결합 열의 해 s₂ = Ac⁻¹(−h₀e₀ − hₙ₋₄eₙ₋₃)는 ‖D b₂‖∞ < 1/2에서 ‖s₂‖∞ < 1이고, Schur 보수인 분모 2(hₙ₋₂ + hₙ₋₃) + hₙ₋₃s₂[0] + hₙ₋₂s₂[n − 3]은 hₙ₋₂ + hₙ₋₃ 이상이라(세 배를 넘는 상쇄가 없다) 마지막 기울기와 s = s₁ + sₙ₋₂s₂의 오차가 uM의 상수배다. 상수는 증명 대신 잰다(아래).
  - **측정** (IP2 마일스톤; `tests/interpolate/test_cubic.py`, `bench/accuracy/interpolate.py`의 `coefficients`): 정확한 유리수의 피벗 없는 소거에서 실수 곡선 21개(n ≥ 4) 모두 natural, clamped, not-a-knot의 max (|L||U|)ᵢⱼ/|Aᵢⱼ|이 정확히 1이다(IP0 마일스톤). 계산한 기울기는 4의 꼴로 natural 0.084, clamped 0.098, not-a-knot 0.069, periodic 0.053 계약 단위이고, uM 단위로 natural 5.4, clamped 4.8, periodic 2.3(5의 150 안)이다. scipy(LAPACK `gbsv`, 부분 피벗)의 기울기를 같은 꼴로 재면 natural 2.99, clamped 3.29, not-a-knot 6.51로 한계를 넘는다(monotone/flats30의 0인 걸음 둘레; 부분 피벗은 성분마다의 역안정성을 주지 않는다). IP0 마일스톤의 탐침(n = 80 natural)에서 scipy는 기울기마다 최대 47 ULP 떨어졌다. 그래서 계약은 성분마다의 상대가 아니라 위의 꼴이다(§4).
- **`PchipInterpolator`**: Fritsch–Butland의 가중 조화 평균(scipy와 같은 식, 같은 연산 순서 sᵢ = 1/((w₁/Δᵢ₋₁ + w₂/Δᵢ)/(w₁ + w₂))): Δᵢ₋₁Δᵢ ≤ 0이면 sᵢ = 0, w₁ = 2hᵢ + hᵢ₋₁, w₂ = hᵢ + 2hᵢ₋₁. 끝은 scipy의 세 점 한쪽 식과 그 자르기다. 기울기 하나가 이웃 두 구간으로만 정해지는 국소 계산이라 조각으로 나눠 병렬이다.
  - **계약의 증명**: 안쪽은 같은 부호의 항의 합과 곱·나눗셈뿐이라 상대 γ₁₂(D의 γ₃ 포함)이고, 갈래(부호, 0)는 계산한 D의 부호가 정확한 D의 부호와 같아(언더플로가 없으면) 정확하다. 가중 조화 평균은 |s| ≤ 3min(|Δᵢ₋₁|, |Δᵢ|)라(w₂/w₁ ≤ 2) |ŝ − s| ≤ 36uM_loc다. 끝 식은 분자의 상쇄가 γ₇((2h₀ + h₁)|Δ₀| + h₀|Δ₁|)/(h₀ + h₁) ≤ γ₁₀(2|Δ₀| + |Δ₁|)이고, 부호와 3Δ₀ 자르기의 갈래가 반올림으로 바뀌는 경계에서는 두 값이 그 한계 안에서 같다. 그래서 **|ŝᵢ − sᵢ| ≤ 40u·max(그 식이 읽는 |Δ| 둘)**이다(잰 값 0.068).
- **`Akima1DInterpolator`**: Akima(1970)와 makima의 무게(scipy와 같은 식과 연산 순서, 끝의 가상 기울기 둘씩 포함). 무게 합이 문턱 이하면 두 기울기의 평균이다(scipy와 같다; 문턱의 max는 열마다, IP33). 국소 계산이다.
  - **계약의 증명**: 노드의 창 m₀…m₃(= m[i…i + 3])에서 M을 가장 큰 |m|이라 하면 m의 오차는 E = 3γ₈M 안이다(밖으로 늘인 m 포함). f₁, f₂의 절대오차는 3E + γ₃f, 무게 w = f₂/f₁₂의 오차는 9E/f₁₂ + γ₆, t = m₁ + w(m₂ − m₁)의 오차는 **3E + 6uM + |m₂ − m₁|(9E/f₁₂ + γ₆)**이다(평균 갈래는 2E; `refdata.akima_bound`는 1.5배로 잰다). |m₂ − m₁|/f₁₂가 조건수다: f₁₂가 반올림 크기면 Akima의 기울기 자체가 자료의 반올림으로 정해지지 않는다. 문턱 갈래가 반올림으로 바뀌는 점(f₁₂가 문턱의 상대 u 안)은 계약 밖이다(시험 자료에는 없다). 잰 값 0.29.
- 점이 둘이면 모든 계열이 직선이다(scipy와 같다).
- 계수 c₂, c₃은 기울기와 Δ가 유한해도 넘칠 수 있다(값 1e300 둘레, 간격 0.01 둘레). 그때는 만들기의 `ValueError`이고 메시지가 그 구간을 말한다(IP36).

### 3.4 B-스플라인 (IP3)
코드는 `interpolate/bspline.cpp`와 `interpolate/banded.hpp`의 `BandSolver`다. 매듭 t는 N = n + k + 1개, 계수 c는 (행, 열)이고 처음 n행만 쓴다(scipy의 관례). complex 계수는 실수부와 허수부를 나란한 double 열로 본다.
- **구간**: 점 아래(같은 값 포함)의 매듭 수 c에서 l = c − 1을 [k, n − 1]로 자른다(`base/interval.hpp`의 탐색, §3.1). 정의역 안은 t[l] ≤ x < t[l+1], x ≥ t[n]은 n − 1, x < t[k]는 k이고 scipy의 걷기가 내는 l과 같다. 무작위 순서의 점도 점마다 O(log N)이다.
- **평가** (IP55): scipy의 재귀(`_deBoor_D`)로 l의 B 값 k + 1개를 차수 0부터 올린다: 차수 j의 단계는 m = 1, …, j에 대해 B_{m−1} += hh_{m−1}·(t_{l+m} − x)/(t_{l+m} − t_{l+m−j}), B_m = hh_{m−1}·(x − t_{l+m−j})/(t_{l+m} − t_{l+m−j})다(같은 매듭이면 그 B는 0). scipy는 hh/(t − t)를 먼저 나누는데 간격이 비정상 수면 그 몫이 넘친다(subnormal/x에서 scipy의 k = 1 평가가 inf이고 k ≥ 2의 배치 행렬이 inf라 만들지 못한다). 두 몫은 기본 구간 안에서 각각 [0, 1]이라 모든 단계가 넘치지 않는 볼록 결합이다. 도함수는 차수 k − ν까지 올린 뒤 scipy의 단계 w = j·hh_{m−1}/(t_{l+m} − t_{l+m−j}), B_{m−1} −= w, B_m = w를 ν번 한다. 값은 c[l − k]B_{l−k}에서 시작해 항마다 fma 하나다. k가 2–5이고 ν = 0이면 같은 연산을 펼친 꼴이다(같은 비트, n = 10³에서 평가 시간이 13.7 → 7.6 ms).
  - **계약의 증명**: 값 단계에서 몫 하나는 뺄셈 둘과 나눗셈 하나(γ₃), 곱과 같은 부호 두 항의 합이 둘을 더해 B 값마다 단계당 반올림 다섯이고 차수 k까지 γ₅ₖ다. 도함수 단계는 정수배, 간격, 나눗셈, 뺄셈의 넷이고 뺄셈은 상쇄할 수 있어, 모든 항을 절댓값으로 한 같은 재귀 B̃로 적는다(기본 구간 안의 ν = 0에서 B̃ = B ≥ 0, 합 1). 합은 곱 하나와 fma k개라 γₖ₊₁이다. 결과가 비정상 수인 반올림은 각 2⁻¹⁰⁷⁵까지 틀린다. 그래서 **|ŷ − Σⱼ cⱼBⱼ^(ν)(x)| ≤ γ_{6k+1}·Σⱼ|cⱼ|B̃ⱼ(x) + (6k + 2)·2⁻¹⁰⁷⁴·(1 + Σⱼ|cⱼ|)**다(c는 계산한 계수, 늘이는 바깥도 같은 꼴). Σ|cⱼ|B̃ⱼ가 넘치면(large/1e300의 높은 도함수: 항이 넘쳐 합의 부호도 정해지지 않는다) 유한한 한계가 없다.
- **k = 1** (IP61): t[1..n]이 엄격히 증가하면 노드 t[1..n]과 값 c[:n]의 선형 길(§3.2, `interpolate/linear.cpp`)이다. 기본 구간 [t[1], t[n]], 끝 조각의 직선, [t[1], t[n]]으로 감기가 B-스플라인의 뜻 그대로다. 안쪽 매듭이 겹치면(뛰는 꺾은선) de Boor다(오른쪽에서 이어진다).
- **매듭** (`bspline_knots`): scipy의 `_not_a_knot`(짝수 k는 double로 계산한 (x[i+1] + x[i])/2), `_augknt`, `_periodic_knots`를 같은 연산으로 옮겼다. 시험 곡선 전부에서 scipy의 매듭과 비트까지 같다.
- **배치 행렬의 연립방정식** (IP16, IP35, IP53): 행은 x₀의 보간, 왼쪽 도함수 조건(차수가 오르는 순서), x₁ … xₙ₋₂의 보간, 오른쪽 도함수 조건(차수가 내리는 순서), xₙ₋₁의 보간이다. 띠의 너비(대각 왼쪽 p, 오른쪽 q)는 행에서 잰다. 분해는 §3.3과 같은 연산(l = a/u, a − l·u, 뒤 대입은 1/u의 곱)이고 열은 8개씩 묶어 나란히 대입한다(열마다 같은 연산이라 한 열을 따로 풀어도 같은 비트). 행 채우기는 행 조각으로 나란히 한다.
  1. *도함수 조건이 없을 때* (not-a-knot, 주어진 매듭): 배치 행렬은 완전 비음이고(Karlin; de Boor, *A Practical Guide to Splines* XIII) Schoenberg–Whitney 조건 아래 정칙이라 피벗 없는 소거의 L, U가 비음이다(de Boor–Pinkus 1977, "Backward error analysis for totally positive linear systems"). |L||U| = |A|, 성장 1이다.
  2. *각 끝의 차수가 1, …, r일 때* (clamped, k = 5의 (1, 2) 등): `make_interp_spline`의 매듭은 끝점이 k + 1겹이라 x₀의 보간 행은 e₀이고 차수 ν의 행은 열 0, …, ν에만 있다(Bⱼ는 x₀ 둘레에서 (x − x₀)ʲ처럼 사라진다). 차수가 오르는 순서면 그 블록이 아래 삼각이라 U의 행은 대각 하나뿐이고 L의 원소는 Aᵢⱼ/Uⱼⱼ다. 이 행들로 지워도 다른 열이 바뀌지 않아 안쪽 행은 1의 완전 비음 계로 남는다. 오른쪽은 차수가 내리는 순서로 xₙ₋₁의 보간 행 앞에 두어 행마다 대각 왼쪽에 원소가 없다(U의 행 그대로). 성장 1이다.
  3. *k = 3의 natural* (각 끝에 2차 도함수 하나): 왼쪽 행은 (6/h₀²)·[1, −(1 + r), r], r = h₀/(h₀ + h₁)이고 e₀로 첫 열을 지우면 피벗이 −(1 + r)·6/h₀²(음수)다. 이 행으로 x₁의 보간 행 [(1 − r)², r(1 − r + s), …](s = (h₁ + h₂)/(h₀ + h₁ + h₂))의 열 1을 지우면 열 2에 δ = (1 − r)²r/(1 + r)(행의 척도를 맞춘 값)가 더해지고 그 자리의 성장은 1 + 2δ/A = 1 + 2(1 − r)²/((1 + r)(1 − r + s))다. s ≥ 1 − r이라 **≤ 1 + (1 − r)/(1 + r) < 2**다. 완전 비음 행렬의 대각에 δ > 0을 더해도 완전 비음이라(그 원소를 포함한 소행렬식이 δ·(소행렬식 ≥ 0)만큼 는다) 나머지는 1의 논증 그대로다. 오른쪽 행은 (6/h²)·[q, −(1 + q), 1](h = hₙ₋₂, q = h/(h + hₙ₋₃))이고 성장은 1 + 2qρ/(1 + q)다. ρ = U′/u는 xₙ₋₂ 행의 피벗 u와 그 오른쪽 원소 U′ = Bₙ(xₙ₋₂) = (1 − q)²의 비다. 1/u는 앞의 행들의 계를 마지막 단위 벡터로 푼 해의 마지막 계수라, x₀ … xₙ₋₃에서 0, xₙ₋₂에서 1, x₀에서 natural, xₙ₋₁에서 값과 기울기가 0인 3차 스플라인 s의 계수다. 마지막 구간에서 s = aτ² + (1 − a)τ³(τ = (xₙ₋₁ − x)/h)이고 ρ = (1 − q)²a/(3q)다. 모멘트 Mᵢ = s″(xᵢ)의 xₙ₋₃ 식(왼쪽의 0 자료가 Mₙ₋₄ = −λMₙ₋₃, 0 ≤ λ ≤ 1/2로 감쇠한다)과 xₙ₋₂의 기울기 연속을 풀면 ε = h/hₙ₋₃, θ = hₙ₋₃/β ∈ (0, 1/2](β = 2(hₙ₋₄ + hₙ₋₃) − hₙ₋₄λ)로 a = (9ε + 3ε² + 3ε²θ + 6 − 3θ)/(3ε + 4 − 2θ)이고 1.5(1 + ε)(3ε + 4 − 2θ) − (분자) = 1.5ε(1 + ε)(1 − 2θ) ≥ 0이다. 그래서 a ≤ 3/(2(1 − q)), ρ ≤ (1 − q)/(2q)이고 **성장 ≤ 1 + (1 − q)/(1 + q) < 2**다. 두 끝을 합쳐 성분마다 (|L||U|)ᵢⱼ ≤ 2|Aᵢⱼ|다(n = 3에서 두 끝이 같은 행을 만나도 각 한계가 그대로다).
  4. *그 밖* (1, …, r이 아닌 차수의 집합: k = 5의 {2, 3}, {1, 3}, k = 4의 {2}; 주어진 매듭과 도함수 조건): 피벗 없는 성장은 첫 간격 h에 대해 1/h²으로 자란다(x = (0, 10⁻⁸, 1, 2, 3, 4, 5)/5의 k = 5 {2, 3}에서 1.1×10⁸). 처음부터 부분 피벗으로 분해한다(LAPACK `gbtf2`의 꼴: 띠 안에서 행을 바꾸고 단계마다 승수를 따로 둔다; 위 너비 p + q). 피벗 없는 분해가 0이나 유한하지 않은 피벗을 만나도 행렬을 다시 채워 부분 피벗으로 다시 하고, 그것도 막히면 `LinAlgError`다(Schoenberg–Whitney가 깨지거나 같은 조건이 두 번).
  - **역오차와 앞 오차** (1–3): 원소의 반올림은 보간 행이 상대 γ₅ₖ, 끝점의 도함수 행이 γ₄ₖ다(끝점이 k + 1겹이라 차수 k − ν의 값이 정확한 1, 0이고, 도함수 단계의 뺄셈은 부호가 번갈아 같은 부호의 합이다). 띠의 LU와 대입이 γ_{3k+4}|L̂||Û|(행의 내적이 k + 1항 이하, 뒤 대입의 역수가 하나 더; §3.3의 3과 같은 셈)이고 |L̂||Û| ≤ 2|Â|(1 + O(u))라 (A + ΔA)ĉ = y, |ΔA| ≤ 2γ_{8k+8}|A|다(y와 도함수 값은 자료 그대로). 그래서 **|ĉ − c| ≤ 2γ_{8k+8}|A⁻¹||A||ĉ|** (성분마다, A는 정확한 자료의 배치 행렬; `refdata.collocation_bound`). 끝점의 도함수 행이 넘치면(간격이 비정상 수) 만들기 전에 `ValueError`다(IP60).
- **periodic** (IP56): scipy의 접힌 계다: 미지수 d는 n − 1개이고 cⱼ = d_{j mod (n−1)}, 행은 마지막을 뺀 노드의 보간이다. 열을 s = ⌊k/2⌋ 돌리면 띠 A_b와 구석 원소 몇 개(UVᵀ)로 나뉜다. A_b는 주기로 늘인 배치 행렬의 연속한 정사각 부분이라 완전 비음이고 대각 Bᵢ₊ₛ(xᵢ)가 양수라 정칙이다: 1의 피벗 없는 소거가 그대로 선다. Woodbury(scipy와 같은 꼴): A_bZ = U(구석의 열, k 이하), (I + VᵀZ)g = VᵀA_b⁻¹y는 부분 피벗의 가우스 소거, d = A_b⁻¹y − Zg다. n ≤ k는 scipy처럼 k − 1개의 도함수 연속 조건과 n개 보간 조건의 (n + k − 1)² 계를 부분 피벗으로 푼다. Woodbury의 결합과 작은 계의 상수는 증명 대신 잰다.
- **측정** (`tests/interpolate/test_make_interp_spline.py`, `bench/accuracy/interpolate.py`의 `coefficients`): 정확한 유리수 소거에서 무작위 간격(10⁻⁵–10⁵배) 60벌의 성장은 not-a-knot(k = 3, 5), clamped, k = 5의 (1, 2), k = 4의 (1, 2)·(1)이 모두 1, natural이 1.99999…(< 2, 위의 한계와 같다)이고 scipy의 순서(도함수 행이 먼저)는 0 피벗이다. 계수의 계약 단위는 k2 0.032, k3 0.024, k4 0.021, k5 0.019, k3_natural 0.021, k3_clamped 0.022, k3_periodic 0.018, k5_periodic 0.0066이다. scipy(LAPACK `gbsv`, 부분 피벗)는 같은 한계로 0.041, 0.034, 0.025, 0.024, natural 4.3×10¹³(step/10의 c₀: 행 e₀의 해가 y₀에서 1.9×10⁻⁸ 떨어진다), clamped 0.030, periodic 0.013, 0.010이고 subnormal/x는 만들지 못한다.
- **`derivative`·`antiderivative`** (IP59): scipy `splder`·`splantider`의 식과 연산 순서다: 도함수는 ((c_{i+1} − cᵢ)·k)/(t_{i+k+1} − t_{i+1})에 0 k행, 역미분은 cᵢ·(t_{i+k+1} − tᵢ)의 누적합을 k + 1로 나누고 앞에 0, 뒤에 마지막 값 k + 2행이다(누적합은 열마다 차례 합). 0인 간격(안쪽 매듭이 k + 1겹)과 `nu > k`는 scipy처럼 `ValueError`다. `'periodic'`의 역미분은 `extrapolate=False`(scipy와 같다).
- **`integrate`** (IP57): 역미분 F의 F(b) − F(a)다. `extrapolate=False`는 a, b를 기본 구간으로 자르고(겹침만 센다), `'periodic'`은 scipy와 같은 온전한 주기와 나머지다.
- **`insert_knot`**: scipy `_insert`(FITPACK `insert`, Boehm 1980)의 식과 연산 순서라 같은 (t, c)에서 scipy와 같은 비트다. 주기 스플라인의 경계 매듭과 계수의 복사도 같다.
- **`PPoly.from_spline`** (IP54): 중단점은 기본 구간의 서로 다른 매듭이고, 조각 j의 d차 계수는 그 조각의 k + 1개 계수를 d번 차분한(scipy `splder`의 식; 그 간격은 언제나 양수) 차수 k − d 스플라인의 왼쪽 중단점 값을 d!로 나눈 것이다. k = 0의 xₙ₋₁과 그 오른쪽은 B-스플라인이 빈 마지막 구간의 c[−1]이고 `PPoly`는 폭이 0인 조각이 없어(IP43) 마지막 조각을 잇는다.
- **`from_power_basis`** (IP58): scipy의 식(Marsden 항등식)을 blossom 꼴 cⱼ = Σ_d a_d·e_d(t_{j+1} − xᵢ, …, t_{j+k} − xᵢ)/C(k, d)로 계산한다(조각 i = min(j, n − 2), a_d는 그 조각의 d차 계수, e_d는 기본 대칭 다항식). scipy의 `_diff_dual_poly`는 d! 대신 d를 곱해 k ≤ 3에서만 같은 수학인데, 받는 것이 `CubicSpline`뿐이라 차이가 없다.
- **`design_matrix`**: 점마다 그 구간의 B 값 k + 1개와 열 l − k, …, l이다(0인 값도 저장; scipy와 같다). `extrapolate='periodic'`은 점을 감고, `True`는 끝 구간의 다항식이다.

### 3.5 조각 다항식의 연산 (IP2)
코드는 `interpolate/ppoly.cpp`다. 계수 (k + 1, m, 열)에서 complex는 실수부와 허수부를 나란한 double 열로 본다(근을 뺀 모든 연산이 실수 인자의 선형이다).
- **평가**: s = fl(x − xᵢ)에서 계수의 Horner이고 단계마다 `fma` 하나다(코드가 정한 곳의 FMA). 도함수는 정수배한 계수의 Horner다(계수를 미리 바꾸지 않고 평가 중에 곱한다). 3차이고 `nu = 0`이면 펼친 꼴이다. 감소하는 중단점은 −x의 사본으로 같은 구간 찾기를 쓴다(그 오른쪽 열린 규칙이 x에서 scipy의 거울 규칙 x[i+1] < x ≤ x[i]다).
  - **계약의 증명**: ŝ = s(1 + ε)이고 rⱼ = fl(rⱼ₋₁ŝ + âⱼ)는 반올림 하나다. 차수 e인 항의 계수는 e + 1번(최고 차수는 e번)의 (1 + δ)를 거치고, ŝᵉ이 e번의 (1 + ε)를, 도함수의 정수배 곱이 하나를 더한다. 언더플로는 반올림마다 절대 η = 2⁻¹⁰⁷⁵이고 |s|ᵉ로 퍼진다. 그래서 **|p̂ − p(x)| ≤ Σₑ γ₂ₑ₊₂·fₑ|cₑ||s|ᵉ + 2⁻¹⁰⁷⁴Σₑ|s|ᵉ** (e는 미분한 뒤의 차수, fₑ는 그 정수배, s는 정확한 x − xᵢ)다. 노드(s = 0)에서는 상수 계수 그대로다.
- **`derivative`**: 계수에 d!/(d − ν)!를 곱한다(Python 층의 배열 곱, 반올림 한 번). **`antiderivative`**: 계수를 (d + 1)…(d + ν)로 나누고(반올림 한 번), 차수 q < ν인 상수를 q = ν − 1부터 정한다: c_q[i] = c_q[i − 1] + Σ_{d>q} c_d[i − 1]·C(d, q)·h_{i−1}^{d−q}(앞 조각의 끝에서 그 도함수가 이어지게). Σ는 h의 Horner이고, 누적합은 길이로만 정한 4096조각 블록마다 차례 합을 하고 블록의 오프셋을 차례로 더한다(블록 하나면 그냥 차례 합; 스레드 수와 무관). 상수는 x[0]에서 0이다.
- **`integrate`**: 조각마다 국소 역미분 F(s) = s·Horner(cₑ/(e + 1))의 차이를 구하고(안쪽 조각은 F(h)), 조각들을 배열의 pairwise 트리로 더한다(`base/pairwise.hpp`의 plan·combine: 2¹⁴조각을 넘으면 잎을 나란히 계산해도 같은 비트). 감소하는 중단점에서도 적분의 방향은 a에서 b다(IP49).
- **`roots`·`solve`** (§9의 15, IP45): 조각마다 (1) 도함수의 근으로 단조 구간을 나눈다(재귀; 2차는 상쇄 없는 근의 공식), (2) 단조 구간의 양 끝에서 부호가 다르면 double을 순서대로 센 정수에서 이분해 이웃한 두 double까지 좁힌다(언제나 64걸음 이하; 걸음마다 평가 한 번), (3) 이웃 부호가 같은 극값의 값이 평가 계약 안이면 접하는 근으로 넣는다. 부호는 보정한 Horner(오차 없는 곱 fma(r, s, −rs)와 합의 오차를 둘째 Horner로 더한다)로 정해 \|p\|가 u²Σ\|cₑsᵉ\|의 몇 배를 넘는 곳에서 맞다. 늘이는 끝 조각은 Cauchy 한계 1 + max\|cⱼ/c_k\|의 두 배까지만 본다. 조각마다 독립이라 조각으로 나눠 병렬이고, 결과는 구간 순서로 모아 scipy처럼 앞 근과 같은 값을 거르고 중단점의 부호 변화(`discontinuity`)와 0인 조각의 [x[i], nan]을 넣는다. scipy는 3차 이하를 닫힌 식, 그 위를 동반 행렬의 고유값으로 구한다: 중근을 놓치고 2⁻³⁰ 떨어진 두 근을 7×10⁻⁹ 틀린다. IP2 마일스톤에서 n = 10⁶ 구간이 sd 1스레드 0.41초, 10스레드 0.086초, scipy 1.28초였다.
- **`extend`**: 계수와 중단점을 이어 붙인 새 배열을 만든다(쓰기 금지 배열을 고치지 않는다; 중단점은 IP43).

### 3.6 다항식·유리식 (IP4)
파일은 `interpolate/polyint.cpp`(무게, 실수 평가 커널, 노드의 도함수 값, Krogh, Floater–Hormann의 무게, 복소 평가 커널)와 `interpolate/aaa.cpp`(AAA의 반복, 극점·영점·유수, `clean_up`)다. 값은 (n, K) 행 우선이고 복소 값은 실수부와 허수부를 나란히 둔 (n, 2K) double이다: 노드와 무게가 실수면 모든 식이 값에 대해 실수 계수의 선형이라 복소 열은 실수 열 둘이다. 한 열은 혼자든 배치 안이든, 어느 스레드에서든 같은 연산이다.

- **`Barycentric`의 무게** (IP21): wⱼ = 1/∏ₖ≠ⱼ(xⱼ − xₖ)를 입력 순서로 곱한다. 차이 d마다 비트로 d = f·2ᵉ(\|f\| ∈ [0.5, 1))로 나눠 f끼리 곱하고 e를 따로 더하며, 256번 곱할 때마다 곱을 다시 [0.5, 1)로 맞춘다(그사이 곱은 2⁻²⁵⁷ 위라 언제나 정상 수다). 2의 거듭제곱 척도는 반올림을 바꾸지 않으므로 이것은 "지수가 끝없는 53비트 곱"과 같은 비트이고 맞추는 빈도와 무관하다. 마지막에 모든 무게에 같은 2^(−T)를 곱해 가장 큰 크기를 [0.5, 1)에 둔다(공식은 공통 배수와 무관하다, IP66). 그보다 2¹¹⁰⁰배 넘게 작은 무게는 0이다(그런 노드는 거리가 10³⁰⁰배 넘게 벌어진 자료에서만 생긴다). 같은 노드는 차이가 정확히 0이라 바로 찾는다. scipy의 무작위 섞기(`rng`)는 넘침을 피하려는 것이라 필요 없다. 노드 n개에 n²번 곱이고 노드 조각으로 나눈다(10⁴노드 79.5 ms / 9.5 ms, scipy 159 ms).
- **평가 커널** (`Barycentric`, `FloaterHormannInterpolator`와 실수 AAA가 함께 쓴다): sⱼ = wⱼ/(x − xⱼ), 분자 Σsⱼyⱼ(항마다 fma 하나), 분모 Σsⱼ를 노드 순서로 차례로 더하고 나눈다. 열이 하나면 점 여덟을 한 번에(점마다 같은 연산이라 NEON의 `fdiv.2d`가 된다), 열이 여럿이면 점마다 sⱼ를 한 번 만들어 열마다 쓴다. x가 노드와 같거나 sⱼ가 넘치면(분모가 유한하지 않으면) 가장 가까운 노드의 값이다(IP68). 무한한 점은 유리식이면 극한 Σwf/Σw, 다항식이면 NaN이다.
  - **증명** (IP70, §4의 한계). u = 2⁻⁵³, |θₖ| ≤ γₖ는 반올림 k번의 곱 ∏(1 + δ)^(±1)이다(Higham의 보조정리). 언더플로가 없다고 둔다(무게, sⱼ, 곱). (1) 무게: 차 N − 1번, 곱 N − 2번(첫 곱 1·f는 정확), 역수 한 번이라 ŵⱼ = c·wⱼ(1 + θ₂ₙ₋₂)(c = 2^(−T), N은 노드 수). (2) sⱼ: 차와 나눗셈이 더해져 ŝⱼ = c·λⱼ(1 + θ₂ₙ), λⱼ = wⱼ/(x − xⱼ). (3) 분자의 fma 줄은 항 하나를 많아야 N번 반올림하므로 N̂ = c·Σλⱼyⱼ(1 + θ₃ₙ⁽ʲ⁾), 분모는 N − 1번이라 D̂ = c·Σλⱼ(1 + θ₃ₙ₋₁⁽ʲ⁾). (4) ℓ(x) = ∏(x − xₖ)를 곱하면 ℓλⱼ = ℓⱼ(x)이고 Σℓⱼyⱼ = p, Σℓⱼ = 1이라 N̂ℓ/c = p + e_N, \|e_N\| ≤ γ₃ₙΣ\|ℓⱼyⱼ\|, D̂ℓ/c = 1 + e_D, \|e_D\| ≤ γ₃ₙ₋₁Λ(Λ = Σ\|ℓⱼ\|). (5) p̂ = (N̂/D̂)(1 + δ)에서 p̂ − p = (e_N(1 + δ) + pδ − p·e_D)/(1 + e_D)이고 u\|p\| ≤ uΛ\|p\|라 \|p̂ − p\| ≤ (γ₃ₙ₊₁Σ\|ℓⱼyⱼ\| + γ₃ₙΛ\|p\|)/(1 − γ₃ₙ₋₁Λ)다. p는 float 자료의 정확한 보간 다항식이라 이것은 끝에서 끝의 계약이다. 여덟 점 묶음과 열 여럿의 길도 점마다 같은 연산 순서라 같다.
- **도함수**: scipy와 같은 재귀다. 노드에서의 미분 행렬 Dᵢⱼ = (wⱼ/wᵢ)/(xᵢ − xⱼ)를 scipy의 연산 순서 wⱼ/((xᵢ − xⱼ)wᵢ)로, Dᵢᵢ = −Σⱼ≠ᵢDᵢⱼ로 만들어(행렬을 저장하지 않고 행마다) v ↦ Dv를 하고, k차 도함수는 Dᵏy를 노드 값으로 한 같은 평가다. Dᵏy는 차수마다 객체에 쌓아 둔다. 한 번의 Dv는 무게의 비(θ₄ₙ₋₄)와 차·곱·나눗셈으로 D̂ᵢⱼ = Dᵢⱼ(1 + θ₄ₙ₋₁), 대각은 N − 2번 더해 θ₅ₙ₋₃, 곱의 합 N번으로 \|(D̂v)ᵢ − (Dv)ᵢ\| ≤ γ₆ₙΣⱼ≠ᵢ\|Dᵢⱼ\|(\|vⱼ\| + \|vᵢ\|)이다. 앞 차수의 오차 e는 \|D\|\|e\|로 옮겨 가고, 평가는 계산한 노드 값의 보간 다항식 q̂에 대한 위의 한계에 Σ\|ℓⱼ(x)\|\|eⱼ\|를 더한다(`refdata.rational_nodal_bound`, 테스트가 정확한 유리수 D로 잰다).
- **`Krogh`**: scipy의 Hermite 나눗셈 차분표를 열마다 그대로 한다(같은 점이 s번째 반복이면 그 값을 s!로 나눠 다음 단계를 시작하고, 앞 열의 단계 값을 쓴다; 붙어 있지 않은 같은 점은 `ValueError`). 평가는 Newton 꼴의 Horner, 도함수는 IP69의 중심 옮기기다: m번째 걸음은 k = n − 2, …, m에서 bₖ ← bₖ + (x − x₍ₖ₋ₘ₎)bₖ₊₁(fma)이고 끝나면 bₘ = p⁽ᵐ⁾(x)/m!이다. 증명(§4): (1) 차분표: 값마다 정확한 자료의 선형 결합으로 펼치면 항마다 단계당 반올림 셋(차 둘, 나눗셈 하나)과 s!의 나눗셈 하나라 \|ĉₖ − cₖ\| ≤ γ₃ₖ₊₂C̃ₖ이다. C̃는 같은 점화식을 (\|a\| + \|b\|)/\|xᵢ − xₖ\|와 \|y\|/s!로 한 절댓값 표, 곧 펼친 항의 절댓값 합이다. 결과가 비정상 수인 나눗셈은 절대 2⁻¹⁰⁷⁵씩 틀리므로 같은 점화식으로 옮긴 바닥 Ẽ를 더한다. (2) 평가: cⱼ에서 bₘ까지의 길은 m + 1번의 걸음에서 j − m번 자리를 옮기고 옮길 때마다 (x − xₖ) 하나를 곱한다. 항이 지나는 fma는 걸음마다 많아야 max(1, 그 걸음에서 옮긴 수)라 모두 m + 1 + (j − m) 이하, 차의 반올림은 j − m 이하라 항마다 2n − 1 − m 이하다. 그래서 \|p̂⁽ᵐ⁾ − p⁽ᵐ⁾\| ≤ m!(γ₂ₙ₊₁Ã_m + 바닥)이고 Ã_m은 같은 걸음을 \|ĉ\|와 \|x − xₖ\|로 한 값, 바닥은 2⁻¹⁰⁷⁴((m + 1)Ã_m(1) + 1)이다(넣은 오차가 뒤의 길로만 옮겨 가고, 처음 걸음에서 시작한 길의 합이 그 위다). 끝에서 끝은 (1)의 한계를 같은 걸음으로 옮겨 더한다(`refdata.krogh_bound`).
- **`AAA`** (§9의 17, IP20, IP67): scipy 1.18.1의 단계 그대로이고 실수 자료는 실수 연산(`dgesdd`, `dggev`), 복소 자료는 복소 연산(`zgesdd`, `zggev`)이다. (1) 값이 유한하지 않은 표본을 버리고 점을 정렬해(복소수는 실수부, 허수부 순) 같은 점은 처음 표본만 남긴다(numpy의 `unique`). (2) R은 f의 평균(numpy의 pairwise 합을 그대로 옮김)에서 시작한다. 걸음 m은 남은 표본 가운데 \|f − R\|가 가장 큰 것(같으면 앞, NaN이 있으면 첫 NaN: numpy의 `argmax`)을 지지점으로 하고 Cauchy 열 1/(z − zⱼ)과 Loewner 열 (f − fⱼ)C를 더한다. (3) 무게는 남은 표본 행의 Loewner 행렬을 `gesdd`('S')로 분해한 가장 작은 특이값의 오른쪽 특이벡터다(같은 최솟값이 여럿이면 그 벡터들의 합/√개수). σ₁/σₘᵢₙ > 1/(3ε)가 한 번 되면 그 뒤로는 열을 노름으로 나눈 행렬을 분해하고 무게를 다시 노름으로 나눈다. 행이 열보다 적으면 `gesdd`('A')의 영공간(랭크 문턱 σ_max·ε·max(행, 열)) 기저의 합/√크기다. (4) R = N/D(N = C(wf), D = Cw, 0인 무게의 열은 뺀다)이고 D가 유한하지 않은 행(지지점)은 f다. (5) max\|f − R\| ≤ rtol·max\|f\|이면 수렴, 아니면 min(`max_terms`, 표본 수) 걸음에서 멈춘다. 끝에 0인 무게의 지지점을 뺀다. 몫은 numpy의 것(복소수는 Smith 알고리즘), 합은 식이 주는 순서라 scipy와의 차이는 scipy가 BLAS로 하는 행렬 곱(N, D)의 반올림뿐이다. 시험 자료 넷 모두에서 지지점이 같은 순서로 같다(`tests/interpolate/test_aaa.py`). 오차가 반올림 크기(기본 rtol ε^(3/4)보다 아래)로 내려간 뒤의 걸음은 그 반올림이 고르므로 scipy와 갈릴 수 있다(scipy 문서의 rtol = 0 예에서 33번째 걸음, 오차 2×10⁻¹⁴).
  - **극점·영점**: 화살촉 행렬 쌍 E = [0 wᵀ; 1 diag(zⱼ)], B = diag(0, 1, …, 1)의 일반화 고유값이다(영점은 E의 첫 행이 (wf)ᵀ). `linalg/lapack.hpp`에 더한 내부 래퍼 `ggev`(벡터 없이, `dggev`/`zggev`)로 구하고 α/β를 numpy처럼 나눠 유한한 것만 남긴다(scipy와 같은 행렬, 같은 Accelerate). 공개 linalg API는 그대로다. 유수는 scipy의 식 N(p)/D′(p)다. 크기 m + 1의 QZ라 지지점 2000개면 15초쯤 든다(scipy도 같은 계산이다).
  - **`clean_up`**: \|유수\|/(극점과 표본의 거리)가 `clean_up_tol`×(\|f\|의 기하평균)보다 작은 극점마다 가장 가까운 지지점을 빼고, 남은 표본 행의 Loewner 행렬의 마지막 오른쪽 특이벡터로 무게를 다시 맞춘다(scipy와 같다; SVD는 IP71). 찾은 쌍의 수를 돌려준다.
  - 반복마다의 SVD가 시간의 거의 전부다(M × m, m ≤ `max_terms`). 열을 더하는 QR 갱신으로 SVD를 줄이는 길은 나중 후보로 두었다(IP105; 결과는 같은 알고리즘의 반올림 차이).
  - **평가와 그 계약**: 실수 지지점·무게·평가점이면 위의 평가 커널이고, 무엇이든 복소수면 복소 커널이다: λⱼ = wⱼ·(1/(z − zⱼ))이고 역수는 (r, m)을 2의 거듭제곱으로 맞춘 (r′, m′)의 conj/(r′² + m′²)(fma)라 제곱이 넘치지 않는다. 계산한 (지지점, 값, 무게) 그대로의 정확한 유리식 r에 대해 \|r̂ − r\| ≤ (γₐΣ\|λⱼfⱼ\| + γ_b\|r\|Σ\|λⱼ\|)/(\|Σλⱼ\| − γ_cΣ\|λⱼ\|)이다. 실수 커널은 (a, b, c) = (N + 3, N + 2, N + 1): ŝⱼ = λⱼ(1 + θ₂)에서 위 (3)–(5)와 같다. 복소 커널은 (3N + 16, 2N + 16, 2N + 8): 차의 반올림이 \|ẑ − z\| ≤ u\|z\|라 역수가 γ₅, 곱이 √2γ₂로 \|λ̂ⱼ − λⱼ\| ≤ γ₈\|λⱼ\|, 성분마다 항당 fma 둘이라 분자의 합이 √2γ₂ₙ ≤ γ₃ₙ, 분모가 √2γₙ₋₁ ≤ γ₂ₙ, 마지막 몫이 γ₇이다(\|r\| ≤ \|r\|Σ\|λ\|/\|Σλ\|로 b에 넣는다).
- **`FloaterHormannInterpolator`** (IP9, IP63): 무게는 scipy의 닫힌 식 wₖ = (−1)^(k−d)Σᵢ∈Jₖ 1/∏ⱼ₌ᵢ..ᵢ₊ᵈ,ⱼ≠ₖ\|xₖ − xⱼ\|를 scipy와 같은 순서(곱을 차례로, 합을 차례로)로 해 시험 자료에서 scipy와 비트까지 같다. 한 무게의 항은 모두 같은 부호라 차 d번, 곱 d − 1번, 역수 한 번, 합 d번으로 \|ŵₖ − wₖ\| ≤ γ₃d\|wₖ\|(상대)다. 평가는 위의 커널이고(실수 점은 `Barycentric`과 같은 함수), 끝에서 끝은 무게의 상대 오차가 λⱼ에 곱해지므로 (a, b, c)에 3d를 더한 같은 꼴이다.

### 3.7 격자 (IP5)
- **`linear`**: 축마다 구간과 t를 §3.1·§3.2로 구하고, 2ᵈ 꼭짓점의 값에서 마지막 축부터 첫 축까지 차례로 선형 커널을 쓴다(축 하나마다 꼭짓점이 반으로 준다). scipy는 꼭짓점 무게의 곱(∏ tᵢ 또는 1 − tᵢ)의 합이다. 차례로 쓰는 꼴은 노드에서 정확하고, 결과가 꼭짓점 값의 최솟값과 최댓값 사이이며, 1차원에서 `sd.interp`와 같은 비트다.
- **`nearest`**: 축마다 t ≤ 0.5면 왼쪽, 아니면 오른쪽 노드다(scipy와 같다; t는 같은 1번 식).
- **`slinear`**: k = 1의 텐서곱이고 계산은 `linear`와 같은 함수다.
- **`cubic`·`quintic`**: 텐서곱 스플라인의 계수는 축마다의 1차원 보간 풀이를 차례로 쓴 것과 정확히 같다(배치 행렬이 크로네커 곱이다). 그래서 축마다 §3.4의 띠 풀이를 그 축의 모든 줄에 한 번씩 쓴다(줄마다 독립이라 병렬). 반복 풀이가 없다. 평가는 축마다 de Boor를 차례로(국소 (k + 1)ᵈ 계수). scipy의 `gcrotmk` 길은 100² 만들기 4.27 ms, 10⁵점 평가 10.0 ms다(§5).
- **`pchip`**: 1차원 PCHIP의 값은 이웃 네 노드로만 정해진다(기울기가 국소). 그래서 평가점마다 4ᵈ개의 값으로 축마다 PCHIP을 차례로 쓰면 scipy의 결과와 같은 함수다. scipy는 평가점마다 첫 축의 줄 전체로 1차원 보간을 새로 만들어 100² 격자의 10⁴점이 564 ms다(§5).
- **구현**(IP5 마일스톤, `interpolate/grid.cpp`): 평가점을 256점 블록으로 나누고 축마다 `count_le_block`으로 구간을 찾는다. 꼭짓점·계수·창의 위치는 첫 축이 가장 높은 자리인 번호이고, 줄이기는 언제나 마지막 축부터다(IP77, IP78, IP80). 범위 밖 검사는 평가와 한 번에 한다(IP76). 텐서곱 계수의 풀이는 축마다 줄 묶음과 8열 묶음으로 나눈다. 1차원에서 선형·`slinear`는 `sd.interp`, `cubic`·`quintic`은 `make_interp_spline`, `pchip`은 `PchipInterpolator`와 비트까지 같다(`tests/interpolate/test_grid.py`).

### 3.8 RBF (IP6)
`csrc/interpolate/rbf.cpp`(핵, 계, 풀이, 평가, 국소 계)와 `src/stardust/interpolate/_rbf.py`(인자, 이웃 질의, `NearestNDInterpolator`)다.
- **핵**: scipy의 여덟을 같은 식으로 계산한다: `linear` −r, `thin_plate_spline` (r·r)·log r(r = 0이면 0), `cubic` (r·r)·r, `quintic` −((r²·r²)·r), `multiquadric` −√(r·r + 1), `inverse_multiquadric` 1/√(r·r + 1), `inverse_quadratic` 1/(r·r + 1), `gaussian` exp(−(r·r)). log와 exp는 배열 영역의 NEON 커널(`array/kernels/vmath.hpp`)의 스칼라 짝이다(IP100: 레인과 같은 비트, Arb 대비 1 ULP 안). r = ε·‖x − y‖이고 거리는 spatial의 `euclidean` 짝 커널(`spatial/metric.hpp`의 `pair_scalar`와 `euclid_finish`: 차의 제곱을 fma로 차례로 더하고, 합이 아주 작거나 넘치면 2의 거듭제곱으로 늘려 다시 잰다)이라 한 짝의 거리가 `cdist`와 같은 비트다(IP91).
- **꼬리**: 차수 `degree` 이하의 단항식(scipy의 `combinations_with_replacement` 차례, Python 층이 (q, d)의 지수 배열로 넘긴다)을 x̂ = (x − shift)/scale에서 좌표 차례로 인수를 하나씩 곱해 계산한다. shift = 0.5·max + 0.5·min, scale = 0.5·max − 0.5·min(0이면 1)이고 계의 점들에서 잰다(전역은 모든 자료점, 국소는 그 집합; IP92).
- **계**: [[K + diag(λ), P], [Pᵀ, 0]] [a; b] = [d; 0](N = n + q)를 열 우선의 아래 삼각에 채운다(대칭). 열 32개씩 병렬이고 칸마다 자기 커널이라 나눔과 무관하게 같은 비트다. 대각은 φ(0) + λᵢ다.
- **검사**: 같은 자료점(IP90)은 Python 층이 만들기 전에 `rbf_duplicate`(정렬과 같은 행의 묶음)로 본다. P의 랭크는 풀기 전에 numpy `matrix_rank`의 판정으로 본다(`gesdd` 'N', 허용 s_max·max(n, q)·ε; IP92).
- **풀이**: `sytrf`(Bunch–Kaufman, 아래 삼각) 뒤 LAPACK `dsytrs`의 단계를 그대로 쓴 대입(`ldlt_solve`, IP101)으로 N × s 오른쪽을 열마다 푼다(scipy의 `dsysv`와 같은 분해). complex 값은 실수 열 둘이다(행렬은 실수). 정확한 0 피벗이나 유한하지 않은 계수는 특이하다고 알린다. 전역 계는 (n + q)² × 8바이트를 한 번 잡고 물리 메모리보다 크면 할당 전에 `MemoryError`다.
- **평가**: 점마다 Σⱼ φ(rⱼ)cⱼ를 자료점 차례로 항마다 fma 하나로 더한 뒤 단항식 항을 같은 꼴로 더한다. 합의 순서는 크기로만 정해져 평가점 조각(병렬)과 무관하다. 평가점 넷을 NEON 레인으로 나란히 계산하고(값 열이 8개 이하일 때), 계 채우기는 한 열의 칸 둘씩이다(IP100; 칸마다 같은 연산). 작업 메모리는 조각마다 열 수만큼이다.
- **`neighbors=k`**(IP96): Python 층이 유한한 평가점의 `KDTree.query(k)` 색인을 넘기고, `rbf_local`이 행마다 정렬해 집합으로 하고, 평가점을 (집합의 사전 순서, 점 번호)로 정렬해 묶는다. 집합마다 그 점들을 모은 사본에서 위의 계 만들기, 검사, 풀이, 평가를 그대로 쓴다(그래서 그 집합의 전역 RBF와 같은 비트다). 집합끼리 병렬이고 태스크마다 LAPACK 작업 공간(`Scratch`)을 따로 쥔다. 실패는 점 번호가 가장 작은 점의 집합을 상태 코드로 돌려주고 Python 층이 메시지를 만든다. 국소 계는 작아서(k ≈ 50) LAPACK 호출의 고정비가 크다. IP7 마일스톤에서 쟀다(N = 53): `sytrf` 4.25 µs, Accelerate의 `dsytrs` 3.5 µs, 랭크의 `gesdd` 1.8 µs. 대입만 직접 쓰고(IP101) 분해는 Accelerate에 둔다.
- **`NearestNDInterpolator`**(§9의 5, IP98): 유한한 평가점의 `KDTree.query(k=1)` 색인으로 값을 고른다. 동점은 작은 색인이다. `rescale`은 scipy의 뜻이고 `tree_options`와 질의 옵션은 없다(IP31).
- **삼각분할 위의 보간**(IP9 마일스톤; `csrc/interpolate/scattered.cpp`와 `src/stardust/interpolate/_scattered.py`): Python 층이 인자를 정리하고(`points`의 튜플, `rescale`, `fill_value`), `sd.spatial.Delaunay`를 만들거나 받은 것을 쓰고, 같은 점의 값을 본다(`scattered_duplicate`: 대표의 행과 바이트가 다른 가장 작은 점). 네이티브 함수 셋이 계산한다: `linear_nd`(IP121, IP126), `ct_gradients`(IP123), `ct_eval`(IP121). 모두 `_Delaunay`의 삼각분할을 직접 읽는다(IP128). complex 값은 실수부와 허수부를 나란한 실수 열로 계산한다(같은 연산이라 따로 계산한 것과 같은 비트). 방법의 자세한 것은 [spatial_geometry.md](spatial_geometry.md) §3.8, §3.9다.

### 3.9 병렬화와 결정성
**같은 입력이면 스레드 수와 상관없이 같은 결과다.** 점 하나의 값은 어느 태스크에서든 같은 커널이고, 합이 있는 곳은 크기로만 정한 순서다. 테스트는 1, 2, 3, 7, 10스레드에서 같은 바이트를 확인한다(§7).

| 병렬 경로 | 나누는 방법 | 결과가 같은 이유 |
|---|---|---|
| 평가(`__call__`, `sd.interp`, RGI, RBF) | 평가점 조각(크기로만) | 점 하나는 한 커널이다. 구간 찾기는 결과를 바꾸지 않는다 |
| 3차 계열·B-스플라인의 계수 | 배치의 열 묶음 | 열마다 같은 분해와 대입이다 |
| PCHIP·Akima의 기울기, 계수 만들기 | 노드 조각 | 국소 계산이다 |
| 역미분의 상수 | 길이로만 정한 조각의 부분합을 차례로 | 순서가 크기로 정해진다 |
| B-스플라인의 역미분 누적합(IP59), 배치 행렬의 풀이 | 열 묶음(8열) | 열마다 같은 차례 합, 같은 분해와 대입이다 |
| B-스플라인 배치 행렬의 채우기, `from_spline` | 행 조각, 조각 묶음 | 행·조각마다 독립이다 |
| `integrate` | pairwise 트리 | 배열의 pairwise 합과 같다 |
| `roots` | 구간 조각 | 구간마다 독립, 구간 순서로 모은다 |
| 격자 스플라인의 계수 | 축마다 줄 묶음 | 줄마다 같은 풀이다 |
| RBF의 계, 평가, 국소 계 | 열 조각(32열), 평가점 조각, 이웃 집합 | 칸마다 자기 커널이다. 집합은 KD 트리의 전순서로 정해지고 집합마다 같은 계와 풀이다 |
| 다항식·유리식의 평가, `Barycentric`의 무게와 노드 도함수, Floater–Hormann의 무게 | 평가점 조각, 노드 조각(크기로만) | 점·노드 하나는 한 커널이다(여덟 점 묶음도 점마다 같은 연산) |
| Krogh의 차분표 | 배치의 열 묶음 | 열마다 같은 점화식이다 |
| 삼각분할 위의 보간의 평가 | `find_simplex`의 덩이(256점, IP126) | 점 하나는 한 커널이고 단체는 정의로 정해진다(가장 작은 번호) |
| Clough–Tocher의 기울기 | 값의 열(열 안은 한 스레드, IP123) | 훑기 순서가 하나다 |
| RBF 전역 풀이, AAA의 SVD·고유값 | Accelerate | Accelerate의 스레드 모드가 같으면 같다(linalg.md §1.2) |

- 하나의 긴 열의 삼중대각·띠 풀이는 차례 계산이다(분해와 대입이 앞뒤로 이어진다). 10⁶점 하나는 1스레드 시간이다. 병렬 순환 소거는 반올림이 달라지므로(크기로만 나누면 스레드 수와는 여전히 무관) IP7 마일스톤에서 재고 쓰지 않기로 했다(IP105).
- 병렬 기준값은 배열의 규칙이다(array.md §10.4): 작업량이 float64 연산 2¹⁵개 분량 이상이면 나누고, 그 이상이면 스레드 수와 상관없이 GIL을 놓는다(API.md §3). 점 평가(`sd.interp`, 선형 길, `PPoly`)의 작업량은 점마다 2·log₂n + 8(열과 계수가 여럿이면 8 대신 그 수에 비례)이고 태스크는 1024점이다(IP102; 선형 길은 1024 / 열 수, `PPoly`는 4096 / (계수 수 × 열 수)와 256 가운데 큰 것).

### 3.10 메모리
- 사용자가 모양으로 요청한 출력(평가점 × 배치, RBF의 (n + q)² 계)은 할당 전에 물리 메모리와 견주어 넘으면 `MemoryError`다(API.md §2).
- RBF 평가와 RGI 스플라인의 작업 메모리는 조각 크기로 정한 상한이 있다. 전역 RBF는 (n + q)² × 8바이트를 한 번 잡는다(n = 10⁴에서 800 MB). `neighbors=k`는 태스크마다 (k + q)² × 8바이트이고 만들 때 같은 물리 메모리 검사를 한다(IP106).
- 객체는 계수 배열 하나와 중단점(또는 매듭)을 쥔다. 감소하는 중단점의 뒤집은 사본(§3.1)만 더 든다.

---

## 4. 정확도 계약
참값은 Arb와 정확한 유리수다. 입력이 float64라 자료와 계수는 정확한 유리수이고, 계약은 두 겹으로 적는다. **만들기**: 계산한 계수(기울기, B-스플라인 계수, 무게)를 같은 자료의 정확한 해(유리수 풀이를 올바르게 반올림한 것)와 견준다. **평가**: 계산한 계수가 나타내는 함수를 그 계수 그대로(정확한 유리수로) 평가한 값과 견준다. 두 겹을 합친 끝에서 끝의 오차(정확한 자료의 정확한 보간 대비)는 정확도 표에 보이지만 계약은 두 겹이다. 계약 단위(오차 ÷ 한계)로 적고 경계는 1이다(spatial.md §4와 같은 꼴). 상수의 증명은 §3에 있고, 잰 값은 정확도 표(`bench/interpolate/accuracy.md`)와 테스트의 값이다.

- **선형** (`sd.interp`, k = 1 B-스플라인, 격자 선형의 1차원)
  | 계약 | 메모 |
  |---|---|
  | \|ŷ − y\| ≤ γ₆(\|a\| + \|b\|) + 3η(0 ≤ t ≤ 1), γ₁₀·max(1, \|t\|)·(\|a\| + \|b\|) + 3η(늘이기, IP40), y = (1 − t)a + tb의 정확한 값(t는 x의 정확한 값으로), η = 2⁻¹⁰⁷⁵ | 증명은 §3.2(IP0 마일스톤, 늘이기는 IP1 마일스톤에서 고침). 갈래마다 같은 부호 γ₆·max(\|a\|, \|b\|)와 γ₇·max(1, \|t\|)·max, 반대 부호 γ₅(\|a\| + \|b\|)와 γ₁₀·max(1, \|t\|)·(\|a\| + \|b\|). `tests/interpolate/refdata.linear_bound`. 잰 값(IP1 마일스톤, `bench/accuracy/interpolate.py`의 `linear`): `sd.interp`가 곡선 안 1,736점에서 최대 0.235, 커널 구간 안 1,072점에서 0.500(비정상 수 끝값 tiny-values: 반올림 반 칸이 한계 2⁻¹⁰⁷³의 반), 선형 길의 늘이기 점에서 0.272 계약 단위. 같은 점의 numpy·scipy(k = 1)·R(`approx`)은 곡선 0.259·0.229·0.234, 구간 0.719·0.500·0.719이고 유한하지 않은 값을 낸 점(numpy 60, scipy 64, R 26)은 빼고 잰 것이다 |
  | 노드에서 0 오차, 단조인 자료에서 단조, 구간 끝값 사이 | 정확한 성질(§3.2). hypothesis와 끝점 둘레의 이웃 double로 시험 |
- **격자** (IP5 마일스톤에서 증명; IP79, IP81). 잰 값은 `tests/interpolate/test_grid.py`와 정확도 표(`grid`)다.
  | 대상 | 계약 | 잰 값과 메모 |
  |---|---|---|
  | 선형·`slinear`(d차원) | 축마다 1차원 선형 계약을 쌓은 재귀(`refdata.grid_linear_exact`), 안쪽에서 \|ŷ − y\| ≤ d·(γ₁₃M + 3η). 노드에서 0 오차, 꼭짓점 값의 범위 안(정확한 성질). 1차원은 선형 계약 그대로 | grid.npz 400점과 바깥 점에서 0.34(scipy 0.17, `slinear` 0.54). `nearest`는 고른 노드의 값 그대로(0 오차) |
  | `cubic`·`quintic`의 계수 | 성분마다 Σ_a (⊗_{b>a}\|A_b⁻¹\|)·2γ_{8k+8}\|A_a⁻¹\|\|A_a\|\|R̂_a\|(`refdata.tensor_collocation_bound`) | 시험 격자(grid.npz의 `cubic` 다섯, `quintic` 넷)에서 0.016(`cubic`), 0.0044(`quintic`) |
  | `cubic`·`quintic`의 평가 | γ_{d(6k+1)}Σⱼ\|cⱼ\|∏B̃ + d(6k + 2)·2⁻¹⁰⁷⁴·(1 + 2ᵈΣ\|cⱼ\|)(`refdata.tensor_bound`; ν > 0, 바깥 포함) | 평가 0.11(`cubic`), 0.10(`quintic`; ν = 0, 1, 2, 안팎). 끝에서 끝(계수 한계를 옮긴 것) 0.012(`cubic`), 0.0045(`quintic`). scipy의 반복 풀이는 같은 단위로 3.0×10⁸, 1.1×10⁸이고 상대오차 1.4×10⁻⁵, 2.6×10⁻⁵(sd 6.6×10⁻¹⁶, 8.3×10⁻¹⁵) |
  | `pchip` | 계약 없음(1차원 `PchipInterpolator`를 접은 것과 같은 비트, 계수는 scipy와 같은 비트) | 끝에서 끝 상대오차 3.1×10⁻¹⁶(scipy 5.3×10⁻¹⁶), scipy와의 차이는 값 크기의 2.9 ULP 이하(평가점의 38%가 같은 비트) |
- **조각 다항식의 평가** (Horner, IP2 마일스톤에서 증명, §3.5): \|p̂ − p(x)\| ≤ Σₑ γ₂ₑ₊₂·fₑ\|cₑ\|\|s\|ᵉ + 2⁻¹⁰⁷⁴Σₑ\|s\|ᵉ (s = x − xᵢ 정확한 값, e는 미분한 뒤의 차수, fₑ = d!/(d − ν)!), p는 계산한 계수 그대로의 정확한 값(`refdata.ppoly_bound`). 잰 값: 실수 곡선 21개 × 7방법(`CubicSpline` 넷, PCHIP, akima, makima) × ν = 0–3의 평가점(끝 조각을 늘인 바깥 포함)에서 최대 0.49 계약 단위(akima; 다른 방법 0.46–0.48). scipy는 자기 계수를 거듭제곱 합으로 평가해 같은 한계로 1.04다. 끝에서 끝(정확한 자료의 정확한 보간) 상대오차는 spacing/1e10을 빼고 1.4×10⁻¹³ 이하(spacing/1e10은 조건 때문에 2.5×10⁻⁶; scipy 4.1×10⁻⁶).
- **3차 계열의 만들기** (IP2 마일스톤에서 증명, §3.3)
  | 대상 | 계약 | 잰 값과 메모 |
  |---|---|---|
  | `CubicSpline`의 기울기(모든 `bc_type`) | \|ŝ − s\| ≤ γ₁₁\|A⁻¹\|(\|A\|\|ŝ\| + \|b\|ₐ) 성분마다(A, b는 정확한 자료의 계; `refdata.spline_bound`는 γ₁₂) | natural 0.084, clamped 0.098, not-a-knot 0.069, periodic 0.053. scipy는 같은 꼴로 2.99, 3.29, 6.51, 0.053 |
  | natural, clamped의 따름 | \|ŝᵢ − sᵢ\| ≤ 150·u·maxⱼ\|Dⱼ\| | 대각 우세(행렬 ‖(DA)⁻¹‖∞ ≤ 2). 잰 값 5.4, 4.8(uM 단위). periodic도 같은 한계로 2.3(압축 꼴의 상수는 잼). not-a-knot에는 없다(spacing/1e10에서 7×10⁹) |
  | `PchipInterpolator` | \|ŝᵢ − sᵢ\| ≤ 40·u·max(그 식이 읽는 \|D\| 둘) | 0.068(scipy와 같은 비트) |
  | `Akima1DInterpolator` | 3E + 6uM + \|m₂ − m₁\|(9E/f₁₂ + γ₆), E = 3γ₈M(평균 갈래 2E; `refdata.akima_bound`는 1.5배) | 0.29(scipy와 같은 비트). \|m₂ − m₁\|/f₁₂가 조건수다 |
  | 계수 c₂, c₃ | \|ĉ₃ − c₃\| ≤ γ₈(\|s₀\| + \|s₁\| + 2\|D\|)/h² + 2⁻¹⁰⁷⁴(1 + 3/h + 2/h²), \|ĉ₂ − c₂\| ≤ γ₇(3\|D\| + 2\|s₀\| + \|s₁\|)/h + 2⁻¹⁰⁷⁴(2 + 3/h) (계산한 기울기의 정확한 Hermite 계수 대비; 바닥은 결과가 비정상 수인 반올림 여섯 번의 2⁻¹⁰⁷⁵, IP51; c₁ = s₀, c₀ = y₀는 정확; `refdata.hermite_bound`) | 0.22, 0.27 |
  | 넘치는 계수 | 유한하지 않은 계수는 `ValueError`이고 구간을 말한다(IP36) | large/1e300의 clamped, subnormal/x의 모든 방법 |
- **B-스플라인** (IP3 마일스톤에서 증명, §3.4; IP55)
  | 대상 | 계약 | 잰 값과 메모 |
  |---|---|---|
  | 평가(모든 k, ν = 0, …, k + 1, 늘이는 바깥 포함) | \|ŷ − Σⱼ cⱼBⱼ^(ν)(x)\| ≤ γ_{6k+1}·Σⱼ\|cⱼ\|B̃ⱼ(x) + (6k + 2)·2⁻¹⁰⁷⁴·(1 + Σⱼ\|cⱼ\|) (c는 계산한 계수, B̃는 항을 절댓값으로 한 재귀; 기본 구간의 ν = 0에서 B̃ = B; `refdata.bspline_bound`). k = 1, ν = 0은 선형 계약(IP61) | 실수 곡선 23개 × 방법 10의 평가점에서 최대 0.22 계약 단위(`test_bspline.py`). 정확도 표(ν = 0): k2 0.20, k3 0.20, k4 0.20, k5 0.17, natural 0.18, clamped 0.22, periodic 0.17, 0.10, k1 0.27(선형). scipy의 계수를 scipy가 평가하면 같은 한계로 0.22, 0.19, 0.20, 0.20, 0.17, 0.19, 0.13, 0.11이고 k1은 subnormal/x에서 inf |
  | 만들기(`make_interp_spline`의 모든 `bc_type`, 증명이 서는 피벗 없는 계) | \|ĉ − c\| ≤ 2γ_{8k+8}\|A⁻¹\|\|A\|\|ĉ\| 성분마다(A는 정확한 자료의 배치 행렬, periodic은 접힌 계; `refdata.collocation_bound`). k = 0, 1은 c = y 그대로 | k2 0.032, k3 0.024, k4 0.021, k5 0.019, natural 0.021, clamped 0.022, periodic 0.018(k3), 0.0066(k5). scipy는 0.041, 0.034, 0.025, 0.024, natural 4.3×10¹³, clamped 0.030, periodic 0.013, 0.010. 부분 피벗의 길(IP53의 그 밖)은 scipy와 견준다 |
  | 피벗 없는 소거의 성장 | 성분마다 (\|L\|\|U\|)ᵢⱼ ≤ g\|Aᵢⱼ\|, g = 1(도함수 조건 없음, 각 끝의 차수 1, …, r), g < 2(k = 3의 natural) | 정확한 소거에서 잰 값과 같다(§3.4의 측정) |
- **다항식·유리식** (IP4 마일스톤에서 증명, §3.6; IP70). N은 노드(지지점) 수, 언더플로가 없다고 둔다. 잰 값은 `tests/interpolate/test_polyint.py`·`test_aaa.py`와 정확도 스윕(`polynomial`, `aaa`)이다.
  | 대상 | 계약 | 잰 값과 메모 |
  |---|---|---|
  | `Barycentric`(끝에서 끝, 무게의 반올림 포함) | \|p̂ − p\| ≤ (γ₃ₙ₊₁Σⱼ\|ℓⱼ(x)yⱼ\| + γ₃ₙΛ(x)\|p(x)\|)/(1 − γ₃ₙ₋₁Λ(x)), p는 float 자료의 정확한 보간 다항식, Λ는 Lebesgue 함수(`refdata.barycentric_bound`; polynomial.npz의 `lebesgue`, `abs_terms`). 처음 안의 Higham 2004 첫 차 꼴(O(u²))을 대신한다(IP70) | 0.064(scipy 0.042, 같은 단위). 노드에서 0 오차 |
  | `Barycentric`의 도함수 | 노드 값 Dᵏy의 한계(차수마다 γ₆ₙΣⱼ≠ᵢ\|Dᵢⱼ\|(\|vⱼ\| + \|vᵢ\|) + \|D\|·앞 차수의 한계) 위에 계산한 노드 값의 평가 한계와 Σ\|ℓⱼ\|·노드 한계(`refdata.rational_nodal_bound`) | 1차 0.022, 2차 0.020(scipy 0.023, 0.018) |
  | `Krogh`의 차분표 | \|ĉₖ − cₖ\| ≤ γ₃ₖ₊₂C̃ₖ + Ẽₖ(C̃: 절댓값 표, Ẽ: 비정상 수의 바닥; `refdata.krogh_table_bound`) | 0.023 |
  | `Krogh`의 평가(계산한 계수 대비) | \|p̂⁽ᵐ⁾ − p⁽ᵐ⁾\| ≤ m!(γ₂ₙ₊₁Ã_m(\|ĉ\|) + 2⁻¹⁰⁷⁴((m + 1)Ã_m(1) + 1)), Ã_m은 m + 1번의 Horner 걸음을 절댓값으로 한 것(`refdata.krogh_bound`); 끝에서 끝은 차분표의 한계를 같은 걸음으로 더한다 | 0.10, 끝에서 끝 0.0039(1차 4×10⁻⁵, 2차 1×10⁻⁵; scipy 0.012). 고른 점 30개의 Runge 자료에서 Ã가 커지는 것이 계약에 보인다(경고 없음, IP22) |
  | 유리식의 평가(계산한 지지점·값·무게 대비) | \|r̂ − r\| ≤ (γₐΣ\|λⱼfⱼ\| + γ_b\|r\|Σ\|λⱼ\|)/(\|Σλⱼ\| − γ_cΣ\|λⱼ\|), λⱼ = wⱼ/(z − zⱼ); 실수 (N + 3, N + 2, N + 1), 복소 (3N + 16, 2N + 16, 2N + 8)(`refdata.rational_bound`) | Floater–Hormann 0.10; AAA tan 0.31, abs 0.096, circle 0.056(복소, Arb), nonfinite 0.24(scipy의 유리식을 scipy가 평가하면 0.31, 0.042, 0.042, 0.27) |
  | Floater–Hormann의 무게와 끝에서 끝 | \|ŵₖ − wₖ\| ≤ γ₃d\|wₖ\|; 끝에서 끝은 (a, b, c)에 3d를 더한다(polynomial.npz의 `fh3`) | 무게 0.24(scipy와 같은 비트), 끝에서 끝 0.059(scipy 0.068) |
  | AAA의 성질 | (1) 지지점에서 0 오차(정확한 성질), (2) 수렴한 객체는 표본에서 max\|r(zᵢ) − fᵢ\| ≤ rtol·max\|f\|(sd의 평가로) | (2)의 잰 값(rtol·max\|f\| 단위) tan 7×10⁻⁵, abs 0.38, circle 0.99, nonfinite 1.3×10⁻⁴. 무게 자체는 SVD의 결과라 계약이 없다(알고리즘의 경로) |
- **RBF** (IP6 마일스톤, IP97; `tests/interpolate/refdata.py`의 RBF 부분, `test_rbf.py`, 정확도 표 `rbf`)
  | 대상 | 계약 | 잰 값과 메모 |
  |---|---|---|
  | 풀이 | 성분마다 \|Âĉ − d\| ≤ γ_{4N}(\|Â\| + \|L̂\|\|D̂\|\|L̂\|ᵀ)\|ĉ\|(Higham 정리 11.3, N = n + q; Â는 계산한 계, L̂·D̂는 그 `sytrf` 인수, 잔차는 오차 없는 곱과 `fsum`으로 정확히; `refdata.rbf_backward_bound`) | 핵 여덟과 `smoothing` 0.1, 시험 자료 다섯 벌에서 최대 0.0099(gaussian; scipy의 계와 그 `scipy.linalg.ldl` 인수로 0.010). 증가 인자 ‖\|L̂\|\|D̂\|\|L̂\|ᵀ‖∞/‖Â‖∞는 1.25–5.96(`thin_plate_spline`이 가장 크다) |
  | 평가 | 계산한 계수·shift·scale의 정확한 함수(Arb) 대비 Σⱼ\|ĉⱼ\|Eⱼ(1 + γ_N) + γ_NΣⱼ\|ĉⱼ\|\|φⱼ\| + Σₖ\|b̂ₖ\|\|mₖ\|(γ_{3gₖ}(1 + γ_N) + γ_N), Eⱼ = 1.01γ_{d+5}\|rφ′(r)\| + γ₄\|φⱼ\| + 8·2⁻¹⁰⁷⁴(`refdata.rbf_eval_bound`; libm의 `log`·`exp`가 1 ULP라는 가정 L) | 최대 0.056(gaussian; scipy의 계수를 scipy가 평가하면 같은 한계로 0.11) |
  | 자료점의 잔차, 끝에서 끝 | 계약 없음(계의 조건수) | 잔차 max\|f(yᵢ) + λᵢaᵢ − dᵢ\|/max\|d\|는 linear 4.9×10⁻¹⁶에서 gaussian 2.9×10⁻¹¹(scipy 7.0×10⁻¹⁶, 1.8×10⁻¹¹). 끝에서 끝 상대오차(정확한 자료의 정확한 보간 대비)는 linear 5.1×10⁻¹⁶, thin_plate_spline 7.5×10⁻¹⁶, gaussian(ε = 3) 2.2×10⁻⁷(scipy 6.2×10⁻¹⁶, 8.9×10⁻¹⁶, 2.1×10⁻⁷), `neighbors=8` 2.6×10⁻¹⁶ |
  | `neighbors=k` | 점마다 그 이웃 집합의 자료점으로 만든 전역 RBF와 같은 비트(그래서 위의 계약을 그대로 진다) | 동점이 많은 격자에서 k = 1, 2, 3, 5, 9; 점의 순서, 반복, 한 점씩 부르기, 1·3·10스레드에서 같은 비트 |
  | `NearestNDInterpolator` | 고른 자료점의 값 그대로(0 오차), (거리, 색인)이 가장 작은 점 | 무차별 대입(`cdist`)과 같은 점 |
- **삼각분할 위의 보간** (IP9 마일스톤, IP124; `refdata.linear_nd_bound`·`ct_bound`, `test_scattered.py`, 정확도 표 `scattered`). y는 sd의 삼각분할 위의 정확한 값이다(무게중심 좌표와 g_k를 유리수로; 참값 `scattered.npz`).
  | 대상 | 계약 | 잰 값과 메모 |
  |---|---|---|
  | `LinearNDInterpolator` | \|ŷ − y\| ≤ γ_{d+5}Σλ_k\|f_k\| + (d + 2)·2⁻¹⁰⁷⁴. 유도: λ_k = D_k/D(값 커널 둘 1.002u와 나누기: 3.01u), Σλ̂_kf_k를 꼭짓점 차례로(곱 하나와 fma d개: (1 + u)^(d+1)), (1 + 3.01u)(1 + u)^(d+1) ≤ (1 + u)^(d+5). 자료점에서 0 오차(정확한 성질) | `sd.spatial`의 시험 점 집합 48개(무작위, 퇴화, 거의 퇴화, 극단 범위; 1000행 이하)의 2,342점에서 최대 0.40(far3/1e8grid). 자료점은 모두 자료 그대로. scipy를 제 삼각분할(Qhull)의 정확한 값과 같은 단위로 재면 49(점 셋의 3차원), 자료점 67개가 끝자리가 다르고, 극단 범위의 7벌은 `QhullError`, 거의 퇴화한 벌에서 버린 점 때문에 끝에서 끝의 상대오차가 0.57까지다 |
  | `CloughTocher2DInterpolator`(주어진 기울기) | \|ŷ − y\| ≤ 256u·K + 32·2⁻¹⁰⁷⁴, K = (1 + max_k\|g_k\|)(max\|f\| + max의 \|∂ₓf_i\|\|e_x\| + \|∂_yf_i\|\|e_y\|). 유도(대강의 셈, 1차 항): 방향 도함수 3.01u, 꼭짓점 쪽 계수 3u, g_k의 오차 10u(1 + \|g_k\|)(분자는 값 커널 셋의 합, 분모는 같은 부호의 합), 안쪽 계수 c₀₁₁₁ 등 80u·K, 가운데 계수 63u·K; de Casteljau의 반올림 9u·Σ\|c\| ≤ 36u·K; 무게 b̂의 오차 Σ\|δb\| ≤ 8.7u가 기울기 3·max\|c\| ≤ 12K로 옮기는 104u·K; 합 약 200u·K를 256으로 올린다. 자료점에서 0 오차 | 기울기가 유한한 2차원 23벌의 1,311점에서 최대 3.5u·K(0.014 계약 단위; normal2/1000). scipy를 제 삼각분할과 기울기의 정확한 3차식과 같은 단위로 재면 7.5×10⁴(far2/1e6uniform: 10⁶ + 10⁻³ × 균등에서 무게중심 변환의 오차가 g_k로 간다), 자료점 51개가 끝자리가 다르다. 기울기가 float64를 넘는 subnormal2와 변의 비가 2¹⁰⁰⁰ 가까운 mixed2/1e300은 계약 밖이다(기울기가 ±inf·NaN, scipy는 두 벌 다 `QhullError`) |
  | Clough–Tocher의 기울기 | 계약 없음(반복의 `tol`만큼 정해진다; 같은 입력이면 같은 비트) | 1차식 2x + 3y + 1의 무작위 2,000점에서 값의 오차 3.1×10⁻⁹(scipy 같은 크기), `tol=1e-15`에서 10⁻¹³ 아래. 무작위 벌에서 scipy의 기울기와 상대 2×10⁻⁷ 안(멈춤 판정만 다르다) |
- **`roots`** (IP2 마일스톤): 돌려준 근 r̂마다 (1) sd가 평가하는 함수 P(fl(t − xᵢ))(P는 계산한 계수 그대로)의 정확한 값이 r̂와 그 이웃 double 사이에서 부호가 바뀌거나 r̂에서 0이고(정확한 유리수로 확인), 그렇지 않으면 \|P(r̂)\|가 보정한 Horner의 오차 2(u\|p\| + γ₂ₖ²Σ\|cₑ\|\|s\|ᵉ) 안이며, (2) 접하는 근이면 \|P(r̂)\|가 평가 계약 안이다. 빠뜨린 근이 없는지는 roots.npz(Arb로 분리한 실근과 중복도)로 본다: 단순근은 4 ULP, 중근은 √(16uΣ\|cₑ\|\|s\|ᵉ) 안에 돌려준 근이 있다. 잰 값: 참값 근 36개를 모두 0 ULP로 찾았다(중근 0.25, 1.5와 2⁻³⁰ 떨어진 두 근 포함). scipy는 중근 둘을 놓치고(missed 2) 붙은 두 근을 7×10⁻⁹ 틀린다.
- **`integrate`·`antiderivative`·`derivative`** (IP2 마일스톤): `derivative`는 계수마다 곱셈 한 번(정확한 정수배의 상대 u), `antiderivative`의 이동한 계수는 나눗셈 한 번, 상수 c_q[i]는 앞 조각들의 Horner 합 i개의 누적합이라 γ_{i+2k+4}·Σ_{j<i}Σₑ\|cₑ⁽ʲ⁾\|\|h_j\|^{e+1}/(e + 1) 안(블록이 둘 이상이면 블록 수만큼 더한다), 적분은 조각마다 평가 계약 꼴의 Horner와 pairwise 합 γ_{⌈log₂ m⌉+1}이다. 테스트(`tests/interpolate/test_ppoly.py`)는 정확한 유리수의 역미분과 적분 대비 그 한계로 본다.
- **PCHIP의 단조성** (성질 시험, `tests/interpolate/test_pchip.py`): 단조인 자료(시험 자료의 단조 곡선, 고정 씨앗의 무작위 단조 자료 300개, 걸음이 10⁻³²²–10⁻³⁰⁰인 비정상 수 둘레의 자료 200개, hypothesis 150개(깊은 묶음 1500개); 0인 걸음 포함)에서 (1) 계산한 기울기가 0이거나 Δ와 같은 부호이고 \|sᵢ\| ≤ 3(1 + 4u)\|Δ\| + 2⁻¹⁰⁷⁴(바닥은 IP51), (2) 계산한 계수의 도함수 2차식이 [0, h]에서 계수 계약(위 표의 c₂, c₃ 한계, 비정상 수의 바닥 포함)의 도함수만큼 안에서 자료의 부호를 지킨다(정확한 유리수; IP46: 노드 기울기가 0이면 그 끝의 도함수가 반올림 크기라 정확한 단조는 서지 않는다; IP51: 계수가 비정상 수면 그 반올림이 절대 2⁻¹⁰⁷⁵씩이다), (3) 정렬한 평가점의 값이 평가 계약 안에서만 거꾸로 가고 구간 끝값을 평가 계약 안에서만 넘는다. 셋 다 통과한다. IP0 마일스톤의 탐침에서 scipy는 무작위 단조 자료 2000개에서 구간 끝값을 최대 3 ULP 넘었다(평가의 반올림; 단조가 깨진 곳은 없었다). makima는 평평한 구간이 정확히 평평하고, akima는 구간 안쪽 노드의 기울기가 0인 것을 시험한다(IP47).
- scipy·numpy·R과의 차이는 계약이 아니라 정확도 표의 비교 패키지 열로 보인다.

---

## 5. 성능
`tools/interpolate_probe.py time`이 잰 scipy 1.18.1, numpy 2.5.4, R 4.6.2의 시간이다(ms, 세 번 중 가장 좋은 값(1회로 잰 것은 표시), 기본 설정, 부하 평균 1.3–1.4). scipy와 numpy의 보간 커널은 1스레드다(RBF의 `dsysv`와 AAA의 SVD는 Accelerate). 자료는 [0, 1)의 균등 노드와 표준정규 값, 평가점은 균등 무작위(정렬한 것은 표시). 속도 사례(`bench/speed/cases/interpolate.py`, 48개)는 같은 크기를 잰다.

| 일 | scipy | numpy | R | 메모 |
|---|---:|---:|---:|---|
| 선형 n = 10³, 평가 10⁶ | 10.2 (k = 1 `BSpline`, 10⁵점) | 37.2 | 10 (`approx`) | 정렬한 평가점은 numpy 2.68 |
| 선형 n = 10⁶, 평가 10⁶ | — | 172 | — | 정렬한 평가점은 numpy 9.57 |
| `CubicSpline` 만들기 n = 10⁶ | 12.1 | — | 23 (`splinefun` natural) | natural 12.1, periodic 22.2 |
| `CubicSpline` 만들기 n = 10³, 열 1000 | 12.4 | — | — | 배치 |
| `CubicSpline` 평가 10⁶ (n = 10³ / 10⁶) | 42.0 / 137 | — | 21 (n = 10³) | |
| `make_interp_spline` k = 3 만들기 n = 10⁶ | 44.3 | — | — | k = 5, n = 10⁵는 6.61 |
| `BSpline` k = 3 평가, 무작위 10⁵점 (n = 10³ / 10⁴) | 11.1 / 88.7 (1회) | — | — | §3.1의 O(n) 걷기. 정렬한 10⁶점은 n = 10⁶에서 20.4 |
| `PchipInterpolator` 만들기 n = 10⁶ / 평가 10⁶(n = 10⁶) | 16.3 / 141 | — | — | |
| `Akima1DInterpolator` 만들기 n = 10⁶ | 8.81 | — | — | makima 9.69 |
| `derivative` + `antiderivative` / `integrate` (n = 10⁶) | 5.75 / 4.07 | — | — | |
| `roots` (n = 10³ / 10⁶) | 1.2 / 1265 (1회) | — | — | |
| `BarycentricInterpolator` 평가 10⁶ (n = 101) | 270 | — | — | 만들기 0.154 |
| `AAA` M = 1000 (`tan` / \|x\|) | 0.46 / 12.5 (1회) | — | — | 프로세스의 첫 호출은 0.13초가 더 든다. 평가 10⁶점 10.8 |
| RGI linear 2-D 1000², 10⁶점 / 3-D 100³, 10⁶점 | 89.6 / 130 | — | — | `nearest` 2-D 92.1 |
| RGI cubic 2-D 100² 만들기 / 10⁵점 | 4.27 / 10.0 (1회) | — | — | 반복 풀이(§2.7) |
| RGI pchip 2-D 100², 10⁴점 | 564 (1회) | — | — | 점마다 1차원 보간을 새로 만든다(§3.7) |
| RBF tps 만들기 n = 10³ / 4000 | 35.2 / 1770 (1회) | — | — | 평가 10⁴점(n = 10³) 57.6 |
| RBF `neighbors=50`, n = 10⁵, 10⁴점 | 217 (1회) | — | — | |
| `NearestNDInterpolator` n = 10⁵, 10⁶점 | 462 (1회) | — | — | |

- **IP2 마일스톤의 잰 값** (ms, 따뜻한 상태로 일곱 번 중 가장 좋은 값, sd 1스레드 / 10스레드 대 scipy; 속도 사례의 smoke는 한 번 부른 차가운 값이라 따로다): `CubicSpline` 만들기 n = 10⁶ not-a-knot 11.7 / 10.6 (12.1), natural 11.8 / 10.7 (12.1), periodic 17.1 / 16.2 (22.1), n = 10³에 열 1000개 2.2 / 1.9 (12.3); `PchipInterpolator` 7.3 / 1.5 (16.2); akima 4.5 / 1.9 (8.6); 평가 10⁶점 n = 10⁶ 20.7 / 2.4 (136), n = 10³ 3.6 / 0.43 (42.0); `derivative` + `antiderivative` 3.8 / 1.1 (5.7); `integrate` 1.9 / 0.28 (4.0); `roots` n = 10⁶ 405 / 86 (1280), n = 10³ 0.48 / 0.14 (1.2). 긴 열 하나의 기울기는 차례 계산이라(§3.9) 10스레드가 거의 같고, 사슬의 지연(분해의 나눗셈)이 시간의 대부분이다(IP44).
- **IP3 마일스톤의 잰 값** (ms, 따뜻한 상태로 일곱 번 중 가장 좋은 값, sd 1스레드 / 10스레드 대 scipy): `make_interp_spline` k = 3 n = 10⁶ 21.7 / 17.5 (42.6), natural 21.0 / 17.0 (42.3), periodic 39.7 / 39.6 (48.7), k = 5 n = 10⁵ 3.6 / 2.6 (6.3); 평가 무작위 10⁵점 n = 10³ 0.75 / 0.10 (10.9), n = 10⁴ 0.89 / 0.11 (86.7), 정렬한 10⁶점 n = 10⁶ 10.8 / 1.4 (17.6), 무작위 10⁶점 n = 10⁶ 16.8 / 2.3 (scipy는 3분 넘게 끝나지 않아 재지 않는다); `derivative` + `antiderivative` 2.2 / 2.2 (3.5); `integrate` 1.1 / 1.1 (2.9); `PPoly.from_spline` 31.9 / 6.9 (28.4). 긴 열 하나의 배치 계는 분해와 대입이 차례 계산이라 10스레드에서 행 채우기만 빨라진다.
- **IP4 마일스톤의 잰 값** (ms, 따뜻한 상태로 일곱 번 중 가장 좋은 값, sd 1스레드 / 10스레드 대 scipy): `BarycentricInterpolator` 만들기 10⁴노드 79.5 / 9.5 (159), 평가 10⁵점(101노드) 4.7 / 0.58 (26.7), 2차 도함수 10⁵점 4.7 / 0.58 (26.9); `KroghInterpolator` 평가 10⁶점(20노드) 1.5 / 0.29 (10.7), 모든 도함수 10⁵점 9.0 / 1.1 (18.6); `FloaterHormannInterpolator` 평가 10⁵점(101노드) 4.8 / 0.58 (6.0); `AAA` tan 0.11 / 0.10 (0.40), abs 9.5 / 9.5 (12.0), 나선(복소, rtol 10⁻¹³) 2.2 / 2.2 (2.8), 평가 10⁵점 1.9 / 0.23 (2.6). AAA의 만들기는 LAPACK의 SVD가 거의 전부라 스레드로 빨라지지 않는다. 평가 사례는 10⁵점이다(IP72).
- **IP5 마일스톤의 잰 값** (ms, 따뜻한 상태로 일곱 번 중 가장 좋은 값, sd 1스레드 / 10스레드 대 scipy): `linear` 1000², 10⁶점 29.9 / 8.9 (92.1), 100³, 10⁶점 49.6 / 8.9 (142), `nearest` 1000², 10⁶점 12.9 / 2.0 (98.6); `cubic` 만들기 100² 0.12 / 0.04 (3.6), `quintic` 0.18 / 0.05 (12.7), `cubic` 1000² 8.0 / 1.5; 평가 100², 10⁵점 `cubic` 2.8 / 0.38 (10.6), `quintic` 5.5 / 0.78 (16.9); `pchip` 100², 10⁴점 0.88 / 0.13 (scipy 600, 1회). 1000² `linear`의 10스레드는 격자(8 MB)를 무작위로 읽는 지연이라 3.4배에 그친다.
- **IP6 마일스톤의 잰 값** (ms, 따뜻한 상태로 일곱 번 중 가장 좋은 값(n = 4000과 이웃·`NearestND`는 세 번), sd 1스레드 / 10스레드 대 scipy): `RBFInterpolator` 만들기 n = 10³ 7.8 / 5.8 (34.8), n = 4000 243 / 204 (1780; 계 채우기는 병렬, `sytrf`는 Accelerate), 평가 10⁴점(n = 10³) 53.3 / 5.9 (57.5), `neighbors=50` n = 10⁵의 10⁴점 177 / 24.3 (243), `NearestNDInterpolator` n = 10⁵의 10⁶점 437 / 50.9 (528). 속도 사례 smoke(한 번 부른 차가운 값)도 같은 크기다.
- **IP7 마일스톤의 잰 값** (ms, 따뜻한 상태로 잰 가장 좋은 값, 고치기 전과 뒤의 빌드를 번갈아 세 바퀴(기계를 함께 써서 부하 평균 2–2.5), sd 1스레드 / 10스레드, 전 → 뒤; 괄호는 IP0–IP6의 scipy·numpy): `sd.interp` 노드 10³ 정렬한 10⁶점 3.12 / 0.35 → 2.81 / 0.31(numpy 2.68), 무작위 4.90 / 0.55 → 4.66 / 0.51(numpy 37.2, R 10), 노드 10⁶ 정렬 8.82 → 8.48(numpy 9.57)(IP99); `CubicSpline` 만들기 10⁶점 not-a-knot 12.60 / 11.09 → 9.25 / 7.83, natural 12.30 → 9.12(scipy 12.1), periodic 18.17 / 16.23 → 10.96 / 9.50(scipy 22.2)(IP103); `PPoly.from_spline` 33.4 / 5.03 → 21.9 / 3.21(scipy 28.4)(IP104); RGI `linear` 1000² 29.2 / 3.77 → 16.0 / 2.26, 100³ 44.4 / 5.48 → 26.8 / 3.37(scipy 92.1, 142)(IP99); RBF 평가 10⁴점(n = 10³) 55.0 / 5.84 → 19.6 / 2.09(scipy 57.5), 만들기 n = 10³ 7.45 / 5.35 → 5.38 / 4.62(scipy 34.8), `neighbors=50` 181 / 23.2 → 149 / 19.3(scipy 243)(IP100, IP101); 노드 10³의 5000점 `sd.interp` 10스레드 21 → 13 µs(IP102). 다른 사례(3차 계열의 평가·기울기, B-스플라인, 다항식·유리식, AAA, RGI 스플라인·`pchip`·`nearest`, `NearestNDInterpolator`)는 전과 3% 안에서 같고(`integrate` 2.09 → 2.35는 부하의 흔들림; 코드는 그대로), 작은 호출(10점)은 0.8–2.6 µs 그대로다(RBF 만들기 50점 19 → 15 µs, 평가 10점 2.8 → 2.0 µs). 1000² `linear`의 10스레드는 블록에서 꼭짓점 값을 먼저 모아 격자 읽기가 겹치면서 7.1배로 올랐다(IP5 마일스톤의 3.4배는 점마다 읽기를 기다렸다). `NearestNDInterpolator`는 spatial의 `KDTree.query`가 시간의 거의 전부다(IP105).
- **목표의 확인** (IP7 마일스톤): 평가는 1스레드에서 비교 대상 가운데 가장 빠른 것 이상이다. 하나의 예외는 노드가 적고 평가점이 정렬된 `sd.interp`로 numpy와 같은 크기(2.81 대 2.68; numpy는 기울기를 미리 구한 fma 하나라 IP4의 식보다 연산이 적다)다. 10스레드는 `sd.interp`·`Barycentric`·RBF 평가가 9배 안팎, 3차 계열의 평가 9배, RGI `linear` 7배다. 무작위 순서의 `BSpline` 평가는 n = 10⁴에서 scipy의 89배다. 만들기는 1스레드에서 모두 scipy 이상이다(`from_spline`이 IP104로 뒤집혔다). 긴 열 하나는 사슬 지연이라 스레드로 빨라지지 않는다(IP105). RGI `pchip`은 scipy의 650배, `cubic`·`quintic`은 직접 풀이로 3.6배·2.9배다. RBF는 n = 4000에서 7배(IP6 마일스톤의 값; 이 마일스톤은 n = 10³만 다시 쟀다), `neighbors`와 `NearestND`는 10스레드에서 scipy의 12.6배·11.5배다. `roots`는 n = 10⁶에서 10스레드 4.7배다(IP2 마일스톤의 값; 구간마다의 일이 고르지 않다).
- **IP9 마일스톤의 잰 값** (ms, `bench.speed`의 사례 열, 세 바퀴의 중앙값 가운데 가장 좋은 값, sd 1스레드 / 10스레드 대 scipy와 그 배수; 부하 평균 2.8–3.6; 삼각분할은 준비에서 만들고 평가는 같은 수의 점): `LinearNDInterpolator` 평가 2차원 10⁴ 0.807 / 0.106 (13.8; 17 / 130배), 10⁵ 9.58 / 1.25 (532; 56 / 425배), 3차원 10⁴ 1.80 / 0.225 (11.0; 6.1 / 49배), 10⁵ 24.7 / 3.95 (384; 16 / 97배); `CloughTocher2DInterpolator` 만들기(주어진 삼각분할에서 기울기, 그리고 한 점) 10⁴ 0.884 / 0.863 (2.32; 2.6 / 2.7배), 10⁵ 12.1 / 10.9 (26.0; 2.1 / 2.4배), 평가 10⁴ 1.25 / 0.167 (14.3; 11 / 85배), 10⁵ 13.7 / 1.89 (545; 40 / 289배); `griddata` 10⁴점 `'linear'` 4.50 / 1.75 (34.1; 7.6 / 20배), `'cubic'` 6.28 / 3.05 (38.7; 6.2 / 13배). 10⁶점(따로 잰 값, 1스레드 / 10스레드): 선형 평가 242 / 43(scipy 42,617, 한 번), Clough–Tocher 평가 304 / 63(41,475, 한 번). 선형 평가는 같은 점의 `find_simplex`(10⁵ 6.9, 10⁶ 151)보다 10⁵에서 1.4배, 10⁶에서 1.6배다: 행렬식 넷과 꼭짓점 값 읽기(무작위 차례라 캐시 밖)가 더 든다. 기울기의 훑기는 한 열이 한 스레드라 10스레드에서도 그대로다(목표대로). 이웃 배열을 `Delaunay.vertex_neighbor_vertices`에서 받기 전에는(매번 다시 만들었다) 10⁵의 만들기가 15.8 ms였다.
- **목표**(IP0 마일스톤에서 정한 것; 잰 결과는 위의 목표의 확인)
  - 평가(`sd.interp`, 조각 다항식, B-스플라인, RGI, RBF, `Barycentric`): 1스레드에서 scipy·numpy·R 가운데 가장 빠른 것 이상, 10스레드에서 거의 스레드 수만큼. 무작위 순서의 `BSpline` 평가는 구간 찾기로 수십 배(n = 10⁴에서 scipy의 O(n) 걷기 대비).
  - 만들기: 3차 계열과 B-스플라인은 1스레드에서 scipy 이상(피벗 없는 풀이, 배치 열의 NEON). 긴 열 하나는 차례 계산이라 스레드로 빨라지지 않는다(§3.9).
  - RGI의 `pchip`은 국소 평가로 수백 배, `cubic`·`quintic`은 직접 풀이로 scipy 이상이면서 반복 풀이의 오차가 없다.
  - RBF: 전역 만들기는 K를 병렬로 만들고 분해는 Accelerate라 n = 4000에서 scipy 이상, `neighbors`와 `NearestND`는 KD 트리 질의와 국소 계의 병렬로 10스레드에서 scipy의 5배 이상.
  - `roots`: 구간마다 독립이라 10스레드에서 거의 스레드 수만큼.
- **나중 후보**: AAA의 QR 갱신(§3.6, IP105). 긴 열의 병렬 순환 소거(IP105)와 국소 RBF 계의 분해를 직접 쓰는 것(IP101)은 재 보고 쓰지 않기로 했다.

---

## 6. 층과 파일
- **층**: `interpolate/`는 `array/`, `base/`(`lerp.hpp`, 이 영역과 배열이 함께 쓴다), `parallel/`, `linalg/`(`lapack.hpp`: `sytrf`, `gesdd`, `ggev`), `spatial/`(`kdtree.hpp`: RBF의 이웃, `NearestND`; `metric.hpp`: RBF의 거리; `delaunay2.hpp`·`triangulation.hpp`·`predicates.hpp`: 삼각분할 위의 보간의 `_Delaunay` 객체, 위치 찾기, 값 커널), 배열의 `array/kernels/vmath.hpp`(RBF 핵의 log·exp) 위에 있다. 다른 영역은 `interpolate/`를 모른다. `sd.interp`는 배열 영역의 `array/ops/interp.cpp`이고 `base/lerp.hpp`만 함께 쓴다(배열은 다른 영역을 include하지 않는다). Python 층의 `stardust.interpolate`는 `stardust.spatial`(`KDTree`, `Delaunay`)과 `stardust.sparse`(`design_matrix`)를 쓴다.
- `bench/areas.py`에서 `interpolate`는 `spatial`·`graph` 뒤, `fft`·`plot` 앞이다. ARCHITECTURE §1의 그림과 §2의 목록, API.md의 색인 표, README.md의 표, tests/README.md·bench/README.md의 영역 목록에 한 줄씩 있다.
- `base/lerp.hpp`(IP1, IP99; 아래의 파일은 모두 `csrc/` 아래다): §3.2의 t와 선형 커널(`lerp_t`, `lerp`, 유한한 끝값의 `detail::lerp_finite`), 유한하지 않은 끝값의 갈래, 복소수판, 256점 블록 꼴 `lerp_block`·`lerp_ab_block`.
- `base/interval.hpp`(IP1, IP37): 점 아래의 노드 수(갈래 없는 8점 나란한 이분, 질주, 정렬된 블록을 고르는 `count_le_block`). 배열의 `array/ops/interp.cpp`와 `interpolate/search.hpp`가 함께 쓴다.
- `interpolate/`
  - `search.hpp`(IP1): 구간 번호(노드 수에서), 바깥의 규칙(`extrapolate`의 세 값, 주기의 감기). 감소 중단점은 IP2
  - `linear.cpp`(IP1): k = 1 평가와 격자 선형의 1차원 공용 길
  - `banded.hpp`(IP2, IP3): 피벗 없는 삼중대각 분해와 열 묶음 대입(IP44), 일반 띠의 `BandSolver`(피벗 없는 분해, 부분 피벗의 길, IP53)와 작은 밀집 계의 부분 피벗 풀이(periodic B-스플라인). 순환 계의 보정은 `cubic.cpp`에 있다
  - `cubic.cpp`(IP2): 3차 계열의 기울기(`CubicSpline`의 경계 조건, PCHIP, Akima·makima), Hermite 계수
  - `ppoly.cpp`(IP2): 평가, `antiderivative`의 상수, `integrate`, `roots`·`solve`, 중단점과 유한성의 검사(`derivative`와 `extend`는 Python 층의 배열 연산)
  - `bspline.cpp`(IP3): 구간과 de Boor(펼친 k = 2–5), 매듭 규칙, 배치 행렬과 `make_interp_spline`의 계(periodic의 Woodbury 포함), 도함수·역미분, 매듭 넣기, 두 표현 사이의 변환, `design_matrix`의 CSR 배열(적분과 k = 1은 Python 층에서 역미분과 선형 길로)
  - `polyint.cpp`(IP4): `Barycentric`의 무게와 실수 평가 커널·노드 도함수, `Krogh`의 차분표와 평가, Floater–Hormann의 무게, 복소 무게중심 평가 커널
  - `aaa.cpp`(IP4): AAA의 반복, 극점·유수·영점, `clean_up`
  - `grid.cpp`(IP5): RGI의 `linear`·`nearest`·`slinear`, 텐서곱 스플라인의 축마다 풀이와 평가, `pchip`의 국소 평가
  - `deboor.hpp`(IP5): B-스플라인의 구간과 de Boor 기저(bspline.cpp에서 옮김, grid.cpp와 함께 쓴다), `pchip.hpp`(IP5): PCHIP 노드 기울기(cubic.cpp에서 옮김, grid.cpp와 함께 쓴다)
  - `rbf.cpp`(IP6, IP7): 핵(레인 꼴 포함), 같은 자료점과 랭크의 검사, 계 만들기, `sytrf` 뒤의 대입 `ldlt_solve`, 평가(`eval_lanes`), 이웃 집합의 묶음과 국소 계
  - `scattered.cpp`(IP9): 삼각분할 위의 보간: 같은 점의 값 검사, 무게중심 평가, Clough–Tocher의 계수와 평가, 기울기 추정(IP121–IP128)
  - `interpolate.hpp`: 파일마다의 `bind_*`
- 바인딩 `bind/interpolate_bind.cpp`(IP1): `_core.interpolate` 하위 모듈. `sd.interp`의 바인딩은 배열의 바인딩에 있다.
- `src/stardust/interpolate/` (하위 패키지 없음): `__init__.py`(공개 이름 스물하나의 `__all__`), `_common.py`(입력 정리, 가로축 검사, `extrapolate`·`axis`의 검사, 메시지), `_ppoly.py`(`PPoly`와 3차 계열, `pchip_interpolate`; `PPoly.from_spline` 포함), `_bspline.py`(`BSpline`, `make_interp_spline`), `_polyint.py`(`Barycentric`, `Krogh`과 함수 꼴), `_aaa.py`(`AAA`, `FloaterHormannInterpolator`와 그 비공개 기반 `_Rational`), `_grid.py`(`RegularGridInterpolator`, `interpn`), `_rbf.py`(`RBFInterpolator`, `NearestNDInterpolator`, 흩어진 점의 `points`·`xi`·`rescale` 정리), `_scattered.py`(`LinearNDInterpolator`, `CloughTocher2DInterpolator`, `griddata`).

---

## 7. 참값과 검증
다른 영역과 같은 체계다(ARCHITECTURE §6, tests/README.md, bench/README.md). 모든 도구는 `bench/cli.py`의 공통 옵션(영역, `--match`, `--jobs`)을 쓴다. 긴 참값 생성은 프로세스 10개까지, 메모리를 지켜보며(RSS가 평평한지) 돈다.

### 7.1 시험 데이터 (`tests/interpolate/samples.py`)
모두 식과 고정한 씨앗으로 만든다(IP34: 모듈 이름은 기록 폴더 `data/`와 겹치지 않게 `samples`). 곡선 25개(실수 한 열 23개, 복소수, 3열 배치), 선형 커널의 구간 62개(점 1,147개), 격자 10개, 흩어진 점 6벌, 삼각분할의 점 집합 48벌, 조각 다항식 7개, 다항식 보간 6벌, AAA 표본 4벌이다. 모든 기록과 참값이 자료의 `digest`를 가진다.
- 1차원: 고른 노드, 무작위 노드, 간격이 10¹⁰배 다른 노드, 체비쇼프 점, 노드 2·3·4·5개(경계 조건의 특수한 경우), 단조 자료(0인 걸음 포함), 평평한 구간이 있는 자료, 계단, 진동(Runge), 큰 값(1e300 둘레)과 아주 작은 값, 상쇄(x나 y가 1e15 + 작은 변화), 비정상 수 간격, periodic용 자료(y[0] = y[-1]), 복소수 값, 배치 열. 평가점은 노드, 노드 양옆의 double, 중점, 무작위 점, 양 끝 밖의 점이다.
- 선형 커널: 끝점 둘레의 이웃 double, a와 b의 부호가 다른 짝, 같은 부호의 거의 같은 값, a = b, ±0, inf와 NaN 끝값, x₁ − x₀가 비정상 수인 구간, 폭이 넘치는 구간, DBL_MAX 둘레와 비정상 수의 끝값, 구간 밖의 점.
- 격자: 1–4차원, 축마다 1점·2점·감소 격자, NaN 값.
- 흩어진 점: 2·3차원 균등, 군집, 같은 점, 직선 위의 점(단항식 랭크 부족), 거리 동점이 많은 격자 점(이웃의 동점).
- 삼각분할의 점 집합(`tri_ids()`, `tri_set(id)`; IP9): `sd.spatial`의 계산 기하 시험 점 집합(`tests/spatial/geompoints.py`) 가운데 유효하고 1000행 이하인 48벌(무작위, 퇴화: 격자·한 원 위·같은 점·경계점, 거의 퇴화, 극단 범위)에 좌표를 상자로 옮긴 매끄러운 값, 평가점은 상자 안 균등 64점과 처음 8개의 자료점이다.
- 근: 접하는 근, 붙은 두 근, 0과 같은 조각, 중단점의 부호 변화.

### 7.2 참값 (`tools/interpolate_ref.py` → `tests/interpolate/ref/`, `tests/reffile.py`의 npz)
- 스플라인의 정확한 계수: 자료가 float64라 `fmpq_mat`의 정확한 풀이로 기울기와 계수를 구해 올바르게 반올림한다(n = 80 natural이 8 ms). B-스플라인은 같은 유리수 풀이와 ‖A⁻¹‖∞.
- 평가의 참값: 계산한 계수(sd가 만든 것과 참값 계수 둘 다)를 정확한 유리수로 평가한다. 핵 함수가 드는 RBF와 AAA는 Arb의 구간으로 반올림이 정해질 때까지 정밀도를 늘린다.
- 계약의 오른쪽(Λₙ, Σ\|ℓⱼyⱼ\|, ‖A⁻¹‖∞)과 근의 분리도 Arb다. 파일은 작게 둔다(노드 수백, 평가점 수천).
- 묶음은 아홉이다(모두 9,267점, 0.9 MiB): `linear`(정확한 a + t(b − a); 2,954), `slopes`(`CubicSpline` 네 조건, PCHIP, Akima·makima의 정확한 기울기; 577), `bspline`(k = 0–5, k = 3의 natural·clamped, k = 3·5의 periodic 계수와 곡선마다의 ‖A⁻¹‖∞; 627), `curves`(그 보간들의 정확한 값, 끝에서 끝; 1,870), `polynomial`(보간 다항식과 두 도함수, Krogh의 Hermite 자료, Lebesgue 함수, Σ\|ℓⱼyⱼ\|, Floater–Hormann d = 3; 378), `grid`(RGI의 linear·cubic·quintic·pchip; 400), `rbf`(핵 여덟과 `smoothing`·`neighbors`, Arb; 83), `roots`(Arb로 분리한 실근과 중복도; 36), `scattered`(IP9: sd의 삼각분할 위의 선형 값과 Σλ_k\|f_k\|, 2차원은 sd가 추정한 기울기(묶음 입력 `grad`)의 Clough–Tocher 값과 K; 단체는 `find_simplex`의 것을 정확한 판정으로 확인하고 무게중심 좌표와 g_k는 유리수; 2,342). 매듭은 scipy의 규칙으로 만들고 scipy의 매듭과 비트까지 같다. 같은 스플라인의 두 구성(`CubicSpline`의 기울기 계와 B-스플라인의 배치 행렬)이 참값에서 같은 것도 테스트가 본다. AAA는 sd가 계산한 무게가 있어야 해서 참값 파일이 없고, 테스트가 그 자리에서 Arb로 평가한다. scipy의 주기 B-스플라인은 바깥 매듭을 double로 x₀ − Δ처럼 계산해 주기가 반올림 크기로만 맞으므로 `k3_periodic`은 `periodic`과 10⁻¹³ 안에서만 같다.

### 7.3 테스트 (`tests/interpolate/`)
`test_api_surface.py`(API 문서의 시그니처), `test_interp.py`(`sd.interp`의 numpy 뜻: `left`·`right`·`period`, 반복 노드의 계단, 유한하지 않은 값, `xp`의 검사; 배열 영역의 테스트에도 둔다), `test_lerp.py`(§3.2의 성질: 끝점 정확, 단조, 범위, a = b, 유한하지 않은 끝값; hypothesis), `test_linear_bits.py`(`sd.interp`, k = 1, RGI `linear`·`slinear`, `interpn`의 같은 비트; 늘이기의 k = 1과 `fill_value=None`), `test_ppoly.py`(평가 계약, 미분·역미분·적분, `extend`, 감소하는 중단점, scipy·R 기록), `test_cubic.py`(기울기 계약, 경계 조건마다, n = 2·3, 배치, periodic의 검사, Hermite 계수, IP36), `test_pchip.py`(§4의 단조성 세 겹과 Akima의 평평한 구간), `test_bspline.py`, `test_make_interp_spline.py`, `test_roots.py`, `test_polyint.py`, `test_aaa.py`, `test_grid.py`, `test_rbf.py`, `test_scattered.py`(IP9: 두 계약을 `scattered.npz`로(Clough–Tocher는 그 기울기를 주어), 자료점에서 자료 그대로, 1차식, 바깥과 NaN·inf 점, 같은 점, complex와 배치, `xi`의 꼴, `points`의 꼴(배열, 튜플, `Delaunay`), `rescale`, 기울기의 멈춤, `griddata`, 에러, scipy와 견주기), `test_reference.py`(§4의 계약, Arb 참값), `test_reference_data.py`(§7.4의 기록과 참값), `test_errors.py`(§2의 모든 에러와 메시지), `test_threads.py`(1, 2, 3, 7, 10스레드에서 같은 바이트).
- scipy·numpy와는 그 자리에서 견준다: 반올림이 아닌 동작(모양, 구간, 바깥, 에러)은 같고, 값은 계약 안이다.

### 7.4 scipy·numpy·R의 기록 (`tools/interpolate_probe.py` → `tests/interpolate/data/`)
- 탐침 도구 하나가 numpy `interp`의 경계와 식, scipy 객체의 경계와 기본값과 시험 자료에서의 값, R의 값, 커널의 측정, 시간을 낸다. 기록: `scipy.json`(713), `numpy.json`(53), `r.json`(94). 모든 기록은 그 자료의 `digest`를 가진다. `tests/interpolate/test_reference_data.py`가 기록이 참값과 맞는 것과 설계가 다르게 정한 동작(§2.9)이 기록에 보이는 것을 본다.
- R은 `approx`(`rule=2` ↔ `sd.interp`, `rule=1`의 NA ↔ `extrapolate=False`의 NaN), `spline`·`splinefun`의 `natural`·`periodic` ↔ `CubicSpline`만 비교 패키지다. R의 `fmm`(기본)은 끝 조건이 not-a-knot과 다르다(같은 6점의 x = 0.3이 0.30271 대 0.30143). `monoH.FC`는 Fritsch–Carlson이라 PCHIP(Fritsch–Butland)과 다르다. R의 `approx`는 `x`를 정렬하고 같은 `x`를 평균으로 합치고 NA를 버리므로, 그런 자료는 R 열에서 뺀다.

### 7.5 정확도·속도 체계와 골든
- **정확도** (`bench/accuracy/interpolate.py`): 스윕 `linear`(sd·numpy·scipy·R의 선형), `coefficients`(3차 계열과 B-스플라인의 만들기 계약; 비교 패키지 scipy), `evaluation`(조각 다항식·B-스플라인의 평가 계약), `polynomial`(`Barycentric`, `Krogh`), `aaa`, `grid`(RGI의 방법마다), `rbf`, `roots`, `scattered`(IP9: `linear`·`clough_tocher` 영역, 제 삼각분할과 기울기의 정확한 값 대비 계약 단위, 참값 대비 끝에서 끝, 자료점에서 다른 값의 수, Qhull이 거부한 벌의 수). 끝에서 끝의 오차도 열로 보인다.
- **속도** (`bench/speed/cases/interpolate.py`, 58개): 사례마다 그것을 만든 마일스톤과 필요한 이름을 적는다(spatial의 사례 파일과 같은 틀). §5의 크기와 정렬·무작위 평가점, 배치 열. 결과는 값을 견준다(반복 풀이인 scipy RGI `cubic`·`quintic`은 `differs`, IP82).
- **골든** (`tests/golden/cases_interpolate.py`의 418개(IP9의 30개 포함), 기록 `tests/golden/data/interpolate.json`): 모든 공개 이름 × 경계 자료, 바깥의 세 값, 틀린 호출, 병렬 길에 드는 큰 사례. 1·3·10스레드의 결과가 모두 같아야 한다. `sd.interp`는 배열의 골든(`cases_array.py`)에도 둔다.

---

## 8. 마일스톤
| 마일스톤 | 내용 | 끝났다는 뜻 |
|---|---|---|
| IP0 준비 — 완료·승인(위임, 2026-11-23) | API 문서 초안(`docs/api/interpolate.md`: 모든 공개 이름의 시그니처, scipy·numpy → sd 대응표와 뺀 이름의 바꿔 쓰기, §9의 답), 영역 등록(`bench/areas.py`, ARCHITECTURE, API.md의 색인, README), 시험 데이터(§7.1), 참값 도구(§7.2), 탐침 도구와 기록(§7.4), §3.2 선형 커널의 증명과 numpy 식·안 넷의 측정, 피벗 없는 띠 풀이의 증명 계획, 속도 사례의 틀 | 문서와 도구가 커밋되고 탐침 기록이 있다. §9의 답이 §1.1의 결정으로 옮겨졌다 |
| IP1 선형과 `sd.interp` — 완료·승인(위임, 2026-11-23) | `base/lerp.hpp`, `sd.interp`(배열 영역의 문서·테스트·골든 포함), 구간 찾기 | 선형 계약과 성질, numpy와 반올림이 아닌 동작이 같음, 속도 사례(numpy, R) |
| IP2 3차 계열 — 완료·승인(위임, 2026-11-23) | `PPoly`(평가, 미분, 역미분, 적분, 근, `solve`, `extend`), `CubicHermiteSpline`, `CubicSpline`, `PchipInterpolator`, `pchip_interpolate`, `Akima1DInterpolator` | 만들기·평가 계약, PCHIP 단조성, R의 natural·periodic과 비교, 골든, scipy와의 속도 사례 |
| IP3 B-스플라인 — 완료·승인(위임, 2026-11-23) | `BSpline`(모든 메서드), `make_interp_spline`(모든 k와 `bc_type`), `from_spline`·`from_power_basis`, k = 1의 같은 비트 | 계약, `sd.interp`와 같은 비트, 무작위 순서 평가의 속도 |
| IP4 다항식·유리식 — 완료·승인(위임, 2026-11-23) | `BarycentricInterpolator`, `barycentric_interpolate`, `KroghInterpolator`, `krogh_interpolate`, `AAA`, `FloaterHormannInterpolator`, linalg 래퍼(`ggev`) | 계약, AAA의 성질과 scipy와 같은 지지점 순서(같은 자료에서), 속도 |
| IP5 격자 — 완료·승인(위임, 2026-11-23) | `RegularGridInterpolator`(방법 여섯), `interpn` | 선형 같은 비트, 스플라인의 직접 풀이 계약, `pchip`의 국소 평가가 scipy와 같은 함수, 속도 |
| IP6 흩어진 점 — 완료·승인(위임, 2026-11-23) | `RBFInterpolator`(전역, `neighbors`), `NearestNDInterpolator`, linalg 래퍼(`sytrs`; IP7이 대입을 직접 쓰고(IP101) IP8이 뺐다(IP107)) | 역오차·평가 계약, 이웃의 동점이 결과를 바꾸지 않음, 속도 |
| IP7 성능과 일관성 점검 — 완료·승인(위임, 2026-11-23) | 모든 공개 함수를 크기마다 scipy·numpy·R과 견주어 고치기, 문턱(병렬, 탐색, 조각 크기), 메시지·시그니처의 영역 간 일관성, 코드 리뷰 | 속도 표와 정확도 표가 기준선이 되고 §5의 목표를 적은 대로 확인한다 |
| IP8 문서 — 완료·승인(위임, 2026-11-23) | `docs/api/interpolate.md`의 확정, 이 문서를 구현대로 고치기(잰 값, 정한 상수, 파일), 쓰지 않게 된 linalg 래퍼 `sytrs`의 제거(IP107) | 문서가 코드와 맞고 `test_api_surface`가 통과한다 |
| IP9 흩어진 점 보간 — 완료·승인(위임, 2026-11-23) | `sd.spatial.Delaunay` 위의 `LinearNDInterpolator`(d = 2, 3), `CloughTocher2DInterpolator`, `griddata`(IP115–IP130; 설계는 spatial_geometry.md §2.6, §3.8, §3.9), 참값 `scattered`, 정확도 스윕 `scattered`, 속도 사례 10개, 골든 30개, API 문서 §7 | 두 계약(§4), 자료점에서 자료 그대로, scipy와 견주기(값, 바깥, 퇴화 입력), 1·3·10스레드에서 같은 바이트, 속도 사례 |

진행 원칙은 다른 영역과 같았다: 마일스톤마다 시작 전에 §9의 질문을 묻고, 구현하고, 정확도·속도를 재고, 커밋하고, 결과를 전달하고, 승인받으면 다음으로 간다. IP1이 선형 커널과 구간 찾기를 만들므로 맨 앞이다. IP2와 IP4는 서로 기대지 않아 나란히 할 수 있다. IP3은 IP2의 띠 풀이를, IP5는 IP1과 IP3을, IP6은 spatial의 `KDTree`를 쓴다. 기존 영역의 결과는 비트까지 같아야 한다(배열에는 새 함수 하나와 바닥의 커널 둘(`base/lerp.hpp`, `base/interval.hpp`)을 더했고, linalg에는 내부 LAPACK 래퍼 `ggev`만 더했다).

**나중 후보** (첫 판 밖): `BPoly`·`NdBSpline`·`NdPPoly`(§9의 7), 매끄럽게 하는 스플라인과 최소제곱 스플라인(`sd.smooth`), 매개변수 곡선(`make_splprep`), AAA의 QR 갱신(IP105).

---

## 9. 마일스톤을 시작하기 전에 물을 것
1–22는 모두 (a)로 정했다(§1.1 IP4–IP25, 추천안, 사용자 위임 2026-11-23). 23은 IP0 마일스톤의 측정에서 나온 질문이고 IP1 마일스톤에서 (a)로 정했다(IP36).

추천을 (a)에 두었고 답은 §1.1(IP4–IP25, IP36)에 옮겼다.

**IP0 전**
1. **선형 커널의 식**. (a) §3.2의 꼴(`std::lerp`의 성질: 양 끝 정확, a = b면 a, 단조, 범위 안; t는 올바른 나눗셈 한 번). 추천: 노드에서 정확하니 구간을 어느 쪽으로 골라도 값이 같아 `sd.interp`·k = 1·격자 선형이 같은 비트가 되고, 단조인 자료의 결과가 단조다. 비용은 점마다 나눗셈 하나와 비교 둘이다(IP0에서 잰다). 끝값이 둘 다 −0인 구간은 −0이다(`[numpy와 다름]`: numpy는 +0). (b) numpy의 fma(s, x − xⱼ, fⱼ): 이 휠의 numpy와 거의 같은 비트지만 오른쪽 노드에서 정확하지 않고 범위를 넘는다(§3.2). 구간 규칙이 다른 길끼리 같은 비트가 깨진다. (c) a + t(b − a)(fma): 왼쪽 끝만 정확하다. (d) (1 − t)a + tb: 양 끝은 정확하지만 a = b에서 a가 아닐 수 있고 단조가 아니다.
2. **`sd.interp`의 `xp` 검사**. (a) 유한하고 줄지 않아야 한다(`ValueError`); 반복은 numpy처럼 계단이다. 추천: 정렬되지 않은 `xp`의 결과는 뜻이 없는 값이고, 반복은 numpy에서 뜻이 정해진 계단이다. 비용은 `xp` 한 번 훑기(10⁶점에서 1 ms 아래로 어림; IP1에서 잰다). (b) numpy처럼 검사하지 않는다: 조용히 틀린 값이 남는다. (c) 엄격히 증가: 계단을 쓰던 numpy 코드가 깨진다.
3. **받는 입력**. (a) 보간 객체와 함수 꼴은 API.md §5의 상위 함수 입력, `sd.interp`는 `sd.ndarray`와 Python 스칼라. 추천: 보간은 밖에서 가져온 자료를 처음 넣는 자리이고 spatial·stats와 같다. `sd.interp`는 배열 영역의 부품이라 배열 함수의 규칙이다. (b) 모두 `sd.ndarray`만: 객체마다 `sd.asarray`가 든다.
4. **뺄 옛 이름**. (a) §1.2 표의 옛 이름을 모두 뺀다(`interp1d`, `interp2d`, `Rbf`, FITPACK 감싸개와 그 함수 꼴, `lagrange`, `approximate_taylor_polynomial`, `pade`). API 문서에 바꿔 쓰기 표를 둔다(`interp1d(kind='linear')` → `sd.interp`나 `make_interp_spline(k=1)`, `'cubic'` → `make_interp_spline(k=3)`, `'nearest'·'previous'·'next'` → `make_interp_spline(k=0)`과 `searchsorted`, `InterpolatedUnivariateSpline` → `make_interp_spline`, `RectBivariateSpline(s=0)` → RGI `cubic` 등). 추천: scipy 문서가 옛 것으로 적은 이름이고, 같은 일에 이름이 하나다. (b) `interp1d`만 얇은 감싸개로 둔다: 가장 흔히 쓰이지만 같은 일에 두 이름이고 `kind`의 뜻이 B-스플라인 차수와 섞여 있다.
5. **`NearestNDInterpolator`**. (a) 넣는다(IP6, `KDTree.query(k=1)`; 동점은 작은 색인). 추천: Qhull이 아니고 이미 있는 트리 질의 하나라 일이 작으며, 동점이 정해진 판이 된다. `griddata`는 `method='nearest'`만 있으면 뜻이 반쪽이라 넣지 않는다. (b) Qhull 보간과 함께 나중에.
6. **`FloaterHormannInterpolator`**. (a) 넣는다(IP4). 추천: `Barycentric`의 평가 커널을 그대로 쓰고 무게가 닫힌 식이라 일이 작다. 고른 점에서 다항식 보간보다 안정한 유리식이다. (b) 나중에: IP1이 나열하지 않았다.
7. **도우미 클래스**. (a) `CubicHermiteSpline`은 넣고(3차 계열의 바탕이고 공개 이름이다), `BPoly`·`NdBSpline`·`NdPPoly`는 나중이다. 추천: 앞의 것은 일이 없고, 뒤의 셋은 표현과 차원을 늘리는 일이라 첫 판의 쓰임(1차원 보간, RGI)에 필요 없다. (b) 넷 다 넣는다. (c) 넷 다 뺀다: `CubicHermiteSpline`이 없으면 기울기를 아는 사용자가 쓸 클래스가 없다.
8. **복소수**. (a) scipy가 받는 곳만 받는다(§2.1: `Pchip`·`Akima`·RGI `pchip`은 실수만, AAA만 복소수 점). 추천: 단조성과 기울기의 부호가 실수의 개념이다. (b) `Pchip`·`Akima`도 실수부·허수부를 따로: scipy와 달라지고 뜻이 없다.
9. **`construct_fast`와 고치는 메서드**. (a) `construct_fast`는 없고, 객체를 고치는 scipy의 메서드(`extend`, `add_xi`, `set_yi`)는 둔다. 계수는 쓰기 금지 배열이다. 추천: 검사 없는 생성자는 틀린 객체를 만들 수 있고, 고치는 메서드는 scipy의 뜻이라 사용자 코드가 기댄다. (b) 고치는 메서드도 빼고 새 객체를 돌려준다: scipy와 다르다.
10. **바깥의 규칙**. (a) §2.3: 1차원 구간 보간은 `extrapolate=`의 한 뜻(True / False → NaN / `'periodic'`), 기본은 scipy 그대로(`Akima`의 False 포함), `sd.interp`(numpy)와 RGI(`bounds_error`, `fill_value`)는 그 라이브러리의 뜻. 추천: 인자의 이름과 값과 뜻이 하나이고, 기본값은 옮겨 오는 코드가 조용히 다른 값을 내지 않게 scipy를 따른다. (b) `Akima`의 기본도 True로: 한 규칙에 가깝지만 scipy와 다르다. (c) RGI에도 `extrapolate=`: 에러와 채움 값의 뜻을 담지 못한다.
11. **옵션의 자리**. (a) 데이터는 위치만, 옵션(`axis`, `bc_type`, `extrapolate`, `k`, `method`, `kernel` …)은 키워드만. 추천: 라이브러리의 공통 꼴이고 spatial·sparse와 같다. scipy의 `CubicSpline(x, y, 0, 'natural')`, `make_interp_spline(x, y, 3)`, `np.interp(x, xp, fp, 0, 1)`은 `TypeError`다. (b) scipy처럼 위치로도: 다른 영역에 없는 꼴이다.

**IP2 전**
12. **3차 계열의 계수 순서**. (a) 모든 3차 계열이 하나의 Hermite 함수(§3.3)로 계수를 만든다. 추천: 같은 기울기면 어느 클래스든 같은 비트이고, `CubicHermiteSpline(x, y, s)`와 `CubicSpline`이 계수 단계에서 갈리지 않는다. (b) 클래스마다 scipy의 순서를 따른다.
13. **스플라인의 연립방정식 풀이**. (a) 피벗 없는 삼중대각·띠 풀이를 이 영역에 둔다(대각 우세·완전 양으로 안정성을 증명; 분해 한 번, 열 묶음 대입, NEON). not-a-knot의 끝 행은 증명이 서지 않으면 끝의 작은 블록만 피벗한다. 추천: 배치 열에서 LAPACK을 열마다 부르지 않고, 결과가 Accelerate의 모드와 무관하며, 증명이 짧다. (b) linalg의 `solve_tridiagonal`(`gtsv`)·`solve_banded`(`gbsv`): 부분 피벗이라 증명이 필요 없지만 열마다 호출이고 모드에 기댄다.
14. **periodic의 검사**. (a) scipy와 같다(`allclose(rtol=1e-15, atol=1e-15)`, 계산은 `y[0]`을 양 끝 값으로). 추천: 반올림 크기로만 다른 끝값을 받는 scipy 코드가 그대로 돈다. 메시지는 축 번호를 말한다. (b) 정확히 같아야 한다: 끝값을 계산으로 만든 자료(sin(2π))가 막힌다.
15. **근 찾기**. (a) §3.5의 단조 구간 나누기와 고정 상한의 이분법(접하는 근은 평가 계약으로). 추천: 동반 행렬의 고유값 없이 차수와 무관하게 같은 길이고, 근마다 부호가 바뀌는 자리를 double 하나까지 좁혀 계약이 짧다. (b) scipy처럼 3차 이하 닫힌 식, 그 위 동반 행렬: 닫힌 3차 식은 상쇄로 자릿수를 잃는다.
16. **세로축의 유한하지 않은 값**. (a) §2.1: 전역 풀이(3차 계열, `make_interp_spline`, RBF)는 `ValueError`, 국소 계산(`sd.interp`, RGI, 주어진 계수의 `PPoly`·`BSpline`)은 퍼뜨린다. `check_finite`는 없다. 추천: 전역 풀이에서 NaN 하나는 모든 계수를 NaN으로 만들어 어디서 왔는지 보이지 않고, 국소 계산의 NaN은 그 자리에만 보인다. 앞 넷은 scipy와 같고 RBF만 `[scipy와 다름]`이다(scipy는 NaN `d`를 받아 NaN을 낸다). (b) 모두 퍼뜨린다: scipy의 3차 계열과 다르다.
23. **유한한 함수의 넘치는 계수**. 값이 1e300 둘레이고 간격이 작으면 3차 계수 c₃ = (sᵢ + sᵢ₊₁ − 2Δᵢ)/hᵢ²가 넘쳐, 함수는 유한한데 `PPoly`의 평가가 노드에서도 NaN이다(scipy는 시험 자료 large/1e300의 `CubicSpline`에서 그렇다; `tests/interpolate/test_reference_data.py`). (a) 만들기에서 계수가 유한하지 않게 되면 `ValueError`(어느 구간인지 말한다). 추천: 조용히 NaN을 내지 않고, `PPoly`의 계수는 공개 속성이라 다른 표현으로 바꿀 수 없다. (b) scipy처럼 NaN을 퍼뜨린다. (c) 평가에서 넘침을 피하는 꼴(Hermite 꼴로 평가): 계수 속성 `c`와 평가가 다른 표현이 된다.

**IP4 전**
17. **AAA의 멈춤 알림과 극점**. (a) 경고 대신 `converged` 속성(bool), `clean_up()`은 수만 돌려준다. 극점·영점은 `linalg/lapack.hpp`에 `ggev` 래퍼를 더해 scipy와 같은 화살촉 행렬 쌍으로 구한다(공개 linalg API는 늘리지 않는다). 추천: 상태를 볼 자리가 있고, 행렬 쌍이 극점의 정의 그대로다. (b) 극점을 표준 고유값 문제로 바꿔(`geev`) 래퍼 없이: 바꾸는 과정에 0인 가짜 고유값이 생겨 지우는 판정이 필요하다. (c) 수렴하지 않으면 에러: 마지막 근사가 쓸모 있는 경우가 많다.
18. **`Barycentric`의 무게 순서**. (a) `rng` 없이 §3.6의 지수를 세는 곱(입력 순서). 추천: 결과가 실행마다 같고 넘침이 없다. (b) scipy처럼 `rng`로 섞는다: 같은 입력에 실행마다 다른 끝자리다.
19. **Krogh의 높은 차수**. (a) 경고도 에러도 없다. 계약(§4)이 오차의 크기를 보이고 API 문서가 30점 넘는 고른 점의 불안정을 적는다. Leja 순서로 차분표를 안정하게 하는 길은 IP4에서 재 보고 정한다. 추천: 같은 점 수라도 자료와 점 배치에 따라 정확하기도 해서 문턱 하나로 막을 근거가 없다. (b) 30점 넘으면 `ValueError`: 안정한 경우까지 막는다.

**IP5 전**
20. **RGI 스플라인 방법의 계수**. (a) 축마다 직접 풀이의 텐서곱(§3.7), `solver`·`solver_args`·옛 방법 없음. 추천: 정확한 해에 계약이 서고 반복 풀이의 허용오차가 결과에 들어가지 않는다. (b) scipy처럼 풀이를 고르게: `sd.sparse`의 반복법을 부르게 되고 결과가 그 허용오차다.
21. **RGI의 NaN 평가점**. (a) 언제나 NaN(§2.3). 추천: 다른 모든 보간과 같고, NaN은 범위 밖이 아니다. (b) scipy처럼 `bounds_error=True`면 `ValueError`.

**IP6 전**
22. **RBF의 `smoothing`, 낮은 `degree`, 같은 자료점**. (a) `smoothing`은 scipy의 뜻(수나 점마다의 배열, 유한하고 0 이상)으로 두고, `degree`가 핵의 최소보다 낮으면 `ValueError`, `smoothing`이 0인데 같은 자료점이 있으면 `LinAlgError`다. 추천: `smoothing`은 같은 계의 대각에 더하는 것이라 빼도 일이 줄지 않고, 같은 자료점을 풀 수 있게 하는 길이기도 하다. `sd.smooth`의 매끄럽게 하는 스플라인과는 다른 물건(핵 보간의 정규화)이다. (b) `smoothing`을 빼고 0만: 보간만이라는 범위에 더 맞지만 scipy 코드의 `smoothing=`이 `TypeError`가 되고, 나중의 `sd.smooth`에 같은 계를 다시 둔다. (c) 낮은 `degree`도 받는다: 풀 수 없을 수 있는 계를 말없이 푼다.
