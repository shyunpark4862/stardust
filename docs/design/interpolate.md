# 보간 영역 설계 (sd.interpolate)

> 상태: **설계 초안** (2026-11-23). 큰 결정(§1.1 IP1–IP3)은 사용자가 정했다. §9의 질문 스물둘은 모두 추천안으로 정했다(IP4–IP25, 사용자 위임 2026-11-23). IP0 마일스톤은 완료·승인(위임, 2026-11-23)이고 그 결정이 IP26–IP35다. IP1 마일스톤도 완료·승인(위임, 2026-11-23)이고 배열 영역의 `sd.interp`를 공개했다. 그 결정이 IP36–IP41이다. `sd.interpolate`에 공개한 이름은 아직 없다.

`sd.interpolate`(점에서 주어진 값을 지나는 함수를 만들고 계산하기)의 확정 사항과 내부 구조의 안을 적는다.

- 공통 규칙은 [DESIGN.md](../../DESIGN.md), 층 구조는 [ARCHITECTURE.md](../../ARCHITECTURE.md), 모든 영역의 API 규칙은 [API.md](../../API.md)에 있다. 공개 함수와 시그니처의 안은 [docs/api/interpolate.md](../api/interpolate.md)다. `sd.interp`(IP2)는 배열 영역의 함수라 IP1 마일스톤에서 `docs/api/array.md` §8.5에 들어갔다.
- 비교 대상은 scipy 1.18.1의 `scipy.interpolate`, numpy 2.5.4의 `numpy.interp`, R 4.6.2의 `stats::approx`·`spline`·`splinefun`이다. R의 보간 패키지(`akima`, `pracma`, `signal`)는 설치되어 있지 않다. 설치하지 않는다(선형과 3차 스플라인만 R과 견준다, §7.4).
- 바닥은 배열(`csrc/array/`), 바닥 층의 공용 커널(`csrc/base/`), 스레드풀(`csrc/parallel/`), linalg의 LAPACK 래퍼(`csrc/linalg/lapack.hpp`: RBF의 풀이, AAA의 SVD), spatial의 KD 트리(`csrc/spatial/kdtree.hpp`: RBF의 이웃, 가장 가까운 점 보간)다. 선형 커널 하나를 배열의 `sd.interp`와 이 영역이 함께 쓴다(§3.2).
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

### 1.2 범위 (안)
마일스톤은 §8이다. "IPn"은 그 마일스톤에서 만든다는 뜻이고 "나중"은 첫 판에 넣지 않는 후보다. "?"는 §9의 답에 달렸다.

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
| `griddata`, `LinearNDInterpolator`, `CloughTocher2DInterpolator` | 없음: Qhull(IP1) | 나중 |
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
- **linalg**: RBF의 전역 연립방정식은 대칭 부정치 LDLᵀ(`sytrf`, 래퍼에 `sytrs`를 더한다), AAA는 `gesdd`와 일반화 고유값(`ggev`, §9의 17)을 `linalg/lapack.hpp`로 부른다. 스플라인의 띠 연립방정식은 이 영역의 피벗 없는 풀이다(§3.3, §9의 13). `sd.linalg.solve_banded`와는 끝자리가 다를 수 있다.
- **spatial**: `RBFInterpolator(neighbors=k)`와 가장 가까운 점 보간은 `KDTree`의 kNN을 쓴다. 동점이 (거리, 색인) 순서로 정해져 있어서(spatial.md SPT4) 이웃 집합이 실행과 스레드 수와 무관하다. scipy는 동점 순서를 약속하지 않아 같은 거리의 이웃이 갈리면 RBF의 국소 계와 가장 가까운 점의 값이 달라질 수 있다.
- **sparse**: `BSpline.design_matrix`가 `sd.sparse.csr_array`를 돌려준다(Python 층에서만 의존한다).
- **integrate**: `PPoly.integrate`는 조각 다항식의 정확한 역미분이다. `sd.integrate.quad`와 견줄 수 있지만 서로 쓰지 않는다.
- **나중의 영역**: `sd.smooth`(매끄럽게 하는 스플라인, 최소제곱 스플라인)가 `BSpline`·`PPoly`와 이 영역의 띠 풀이와 de Boor 평가를 쓴다. 그림 영역의 색띠 보간과 등고선의 내부 보간을 `sd.interp`로 옮길 수 있다. 약속이 아니라 메모다.

---

## 2. API (안)

### 2.1 입력
- **받는 것**: 보간 객체의 생성자, 그 메서드와 호출, 함수 꼴(`pchip_interpolate`, `interpn` …)은 API.md §5의 상위 함수 입력(`sd.ndarray`, 수의 시퀀스, numpy 배열, frame의 `Series`)을 받는다(§9의 3). `sd.interp`는 배열 영역의 함수라 `sd.ndarray`와 Python 스칼라만 받는다.
- **가로축**(`x`, `xi`, 격자의 `points`, 매듭 `t`, 평가점): float64로 계산한다. bool·int64는 float64로 바꾼다. complex128은 `TypeError`다(`[scipy와 다름]`: scipy는 복소수 평가점의 실수부만 쓰고 `ComplexWarning`을 낸다). 예외는 `AAA`다: 복소평면의 점을 받는다(scipy와 같다).
- **세로축**(`y`, 계수 `c`, 격자의 `values`, RBF의 `d`): float64 또는 complex128이다. complex를 받는 곳은 scipy와 같다: `PPoly`, `CubicHermiteSpline`, `CubicSpline`, `BSpline`, `make_interp_spline`, `Barycentric`, `Krogh`, `AAA`, RGI의 `pchip`을 뺀 방법, `RBFInterpolator`, `sd.interp`. `PchipInterpolator`와 `Akima1DInterpolator`는 실수만이다(scipy와 같이 `ValueError`; 단조성과 기울기의 부호가 실수의 개념이다; §9의 8).
- **배치(`axis=`)**: 세로축은 여러 차원이고 `axis`(기본 0, scipy와 같다)가 가로축에 맞는 축이다. 결과의 모양은 scipy와 같다: 평가점의 모양이 `axis` 자리에 들어간다(`CubicSpline(x, y[2, 6, 3], axis=1)(x_new[4, 5])`는 (2, 4, 5, 3)). 배치의 열마다 같은 계산이고 결과가 열마다 독립이다.
- **가로축의 검사**: 1차원, 유한, 엄격히 증가해야 한다(`ValueError`; scipy와 같다). `PPoly`의 중단점과 RGI의 격자는 scipy처럼 엄격히 감소해도 된다. `Krogh`는 같은 점의 반복이 도함수 자료라서 받는다(scipy와 같다). `Barycentric`은 서로 다른 점이어야 한다(scipy와 같다). `sd.interp`의 `xp`는 줄지 않아야 한다(§2.2).
- **세로축의 유한성**: 전역 풀이로 계수를 만드는 곳(`CubicSpline`, `PchipInterpolator`, `Akima1DInterpolator`, `make_interp_spline`, `RBFInterpolator`)은 `y`가 유한해야 한다(`ValueError`; scipy의 앞 넷과 같고, `check_finite=False`는 없다). 국소 계산(`sd.interp`, RGI의 값, 주어진 계수의 `PPoly`·`BSpline`)은 IEEE로 퍼뜨린다(§9의 16).
- **평가점의 NaN**: 결과는 NaN이다. 어느 보간이든 같다(§2.3).
- **결과 dtype**: float64이거나, 세로축이 complex면 complex128이다. 결과는 보통의 새 배열이다(결과 객체가 아니다). 0차원 결과는 Python 스칼라다(array.md §2.2; scipy는 0차원 numpy 배열).

### 2.2 `sd.interp` (IP2, 배열 영역)
```
interp(x, xp, fp, /, *, left=None, right=None, period=None)
```
- numpy의 뜻이다: `xp`(1차원, float64)와 `fp`(같은 길이, float64 또는 complex128)가 주는 꺾은선을 `x`(모양 그대로, 또는 Python 스칼라)에서 계산한다. `x < xp[0]`은 `left`(기본 `fp[0]`), `x > xp[-1]`은 `right`(기본 `fp[-1]`)다. `period`가 있으면 `left`·`right`를 무시하고 `xp`와 `x`를 그 주기로 감는다(numpy처럼 감은 `xp`를 정렬하고 양 끝에 한 점씩 더한다). `xp`가 한 점이면 상수다(numpy와 같다; NaN 평가점은 NaN, IP39).
- **`xp`의 검사**: 유한하고 줄지 않아야 한다(`ValueError`, `interp(): 'xp' must be nondecreasing`; `[numpy와 다름]`: numpy는 검사하지 않는다). 같은 값의 반복은 받는다: numpy와 같이 계단(그 점에서 오른쪽 값)이다. `period`가 있으면 정렬은 검사하지 않는다(numpy처럼 감은 뒤 정렬한다). 비용은 `xp` 한 번 훑기다(10⁶점에서 0.14 ms, 같은 크기의 `interp`의 0.3%; §3.2의 측정).
- **값**: 구간 안은 §3.2의 선형 커널이다. 노드에서는 `fp`의 값 그대로이고, 단조인 `fp`에서는 결과도 단조이고 구간의 두 끝값 사이에 있다. 유한하지 않은 `fp`의 규칙은 numpy와 같다(구간 안에서 한쪽 끝이 ±inf면 그 inf, 두 끝이 반대 부호의 inf면 NaN, NaN 끝이 있으면 NaN; 노드에서는 그 값). complex `fp`는 실수부와 허수부를 따로 계산한다(numpy와 같다).
- **결과**: `x`의 모양, `fp`의 dtype이다. `x`가 Python 스칼라면 Python `float`·`complex`다(numpy는 `numpy.float64`).
- `[numpy와 다름]`: 데이터는 위치만, `left`·`right`·`period`는 키워드만이다(numpy는 `interp(x, xp, fp, 0, 1)`처럼 위치로도 받는다). `period=0`은 `ValueError`(numpy와 같다), NaN·inf `period`도 `ValueError`(numpy는 NaN이면 NaN 결과, inf면 감지 않은 값을 이 초안에서 확인함).
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
- **`nu`**: 0 이상의 int다. 차수보다 크면 0이다(scipy와 같다). 음수는 `ValueError`(scipy와 같다; 역미분은 `antiderivative`).
- **`CubicSpline`의 `bc_type`**: `'not-a-knot'`, `'clamped'`, `'natural'`, `'periodic'`, 또는 양 끝의 `(차수, 값)` 튜플이나 그 짝(차수 1, 2; 값은 배치 모양). 끝마다 다른 조건도 받는다(`('natural', 'not-a-knot')`). `'periodic'`은 양 끝 모두여야 하고 `y[0]`과 `y[-1]`이 scipy의 허용(`allclose(rtol=1e-15, atol=1e-15)`) 안이어야 한다(§9의 14). 점이 둘이면 직선, 셋이고 not-a-knot이면 포물선이다(scipy와 같다).
- **`Akima1DInterpolator`**: `method='akima'`는 Akima(1970), `'makima'`는 수정 Akima다(scipy와 같은 식). 무게 합의 문턱 10⁻⁹·max의 max는 열마다다(IP33). 평평한 구간(같은 값이 셋 이상)은 정확히 평평하다(§4의 성질).
- **`roots`·`solve`**: 조각마다 실근을 찾아 정렬된 1차원 배열로 돌려준다. 배치면 배치 모양의 중첩 list다(`[scipy와 다름]`: scipy는 dtype `object`의 배열이고 sd의 배열에는 object dtype이 없다). `discontinuity=True`면 중단점에서 부호가 바뀌는 점도 근으로 친다. 조각이 0과 같은 다항식이면 그 구간의 왼쪽 끝과 NaN을 넣는다(scipy와 같다, `[0., nan]`). 알고리즘은 §3.5(§9의 15).
- **`integrate(a, b)`**: a > b면 부호가 바뀐다. `extrapolate=False`이고 a, b가 정의역 밖이면 NaN, `'periodic'`이면 온전한 주기의 수 × 한 주기의 적분 + 나머지다(scipy와 같다).
- **`antiderivative`**: 상수는 x[0]에서 0이고 이어지게 고른다(scipy와 같다). `'periodic'` 객체의 역미분은 `extrapolate=True`다(주기가 아니므로; scipy와 같다).
- **`extend(c, x)`**: 오른쪽(또는 감소하는 중단점이면 왼쪽)에 구간을 더한다. 객체를 고치는 유일한 메서드다(scipy와 같다). 고친 뒤의 계수도 쓰기 금지 배열이다.

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
- **k = 1**: 계수는 `y` 그대로이고(scipy와 같다), 평가는 §3.2의 선형 커널이다. 그래서 노드에서 `y`를 그대로 돌려준다(`[scipy와 다름]`: scipy의 de Boor는 시험 자료 random/50의 노드 50개 가운데 9개에서 끝자리가 달랐다).
- **`nu`**: 0 이상(scipy의 `nu=-1`은 `NotImplementedError`; sd는 `ValueError`). 차수보다 크면 0이다.
- **`design_matrix`**: (len(x), n) `sd.sparse.csr_array`다(scipy는 `scipy.sparse.csr_array`). 행마다 k + 1개의 저장 자리이고 0인 값도 저장한다(scipy와 같다).
- `PPoly.from_spline(BSpline)`과 `BSpline.from_power_basis(PPoly)`가 두 표현을 잇는다. `tck`는 (t, c, k) 튜플이다.

### 2.6 다항식·유리식 (IP4)
```
BarycentricInterpolator(xi, yi=None, /, *, axis=0, wi=None)
barycentric_interpolate(xi, yi, x, /, *, axis=0, der=0)
KroghInterpolator(xi, yi, /, *, axis=0)
krogh_interpolate(xi, yi, x, /, *, der=0, axis=0)
AAA(x, y, /, *, rtol=None, max_terms=100, clean_up=True, clean_up_tol=1e-13)
```
- **`BarycentricInterpolator`**: 둘째 무게중심 공식이다. 메서드 `__call__`, `derivative(x, /, *, der=1)`, `derivatives(x, /, *, der=None)`, `add_xi`, `set_yi`(scipy와 같다; 이 둘은 객체를 고친다). 무게는 정해진 순서로 지수를 따로 세는 곱이다(§3.6). `[scipy와 다름]`: `rng`가 없다(scipy는 무게의 넘침을 피하려고 점을 무작위로 섞으므로 같은 입력이 실행마다 끝자리가 다를 수 있다). 노드에서는 그 값 그대로다(scipy와 같다).
- **`KroghInterpolator`**: Newton 꼴의 Hermite 보간이다. 같은 점의 반복은 도함수 자료다(scipy와 같다). 메서드 `__call__`, `derivative`, `derivatives`. 차수가 높을 때의 경고는 없다(§9의 19).
- **`AAA`**: scipy의 AAA다(Nakatsukasa–Sète–Trefethen). 점과 값은 실수나 복소수이고, 유한하지 않은 표본은 버린다(scipy와 같다). `rtol` 기본은 ε^(3/4)이다(scipy와 같다). 지지점 고르기의 동점은 앞의 색인이다(scipy의 `argmax`와 같다). `[scipy와 다름]`: `max_terms`에 이르러도 경고가 없고 `converged` 속성이 `False`다. `clean_up()`이 지운 Froissart 쌍의 수를 돌려주는 것은 scipy와 같지만 경고는 없고, 인자 이름은 생성자와 같은 `clean_up_tol`이다(§9의 17, IP32). 노드(지지점)에서는 그 값 그대로다(scipy와 같다).

### 2.7 격자 (IP5)
```
RegularGridInterpolator(points, values, /, *, method='linear', bounds_error=True, fill_value=nan)
RegularGridInterpolator.__call__(xi, /, *, method=None, nu=None)
interpn(points, values, xi, /, *, method='linear', bounds_error=True, fill_value=nan)
```
- `points`는 축마다의 1차원 격자(엄격히 증가하거나 감소; scipy와 같다), `values`는 (n₁, …, n_d, …)이고 뒤의 축은 배치다. `xi`는 (…, d)다. 결과는 (…, 배치)다(scipy와 같다).
- **방법**: `linear`(축마다의 선형을 차례로, §3.7), `nearest`(가운데에서는 작은 색인; scipy와 같다), `slinear`(k = 1; `linear`와 같은 비트, `[scipy와 다름]`: scipy는 두 길이 끝자리가 다르다), `cubic`, `quintic`(축마다 not-a-knot 스플라인의 텐서곱), `pchip`(축마다 PCHIP). 축마다 필요한 점의 수는 scipy와 같다(`linear`·`nearest`는 1점도 받는다, `cubic` 4점, `quintic` 6점).
- `cubic`·`quintic`의 계수는 직접 풀이다(§3.7). `[scipy와 다름]`: `solver`, `solver_args`가 없다. scipy는 이 둘의 계수를 반복 풀이(기본 `gcrotmk`)로 구해 결과가 그 허용오차 크기로 틀린다(이 초안에서 2차식 자료 6 × 6의 `cubic`이 참값 25에서 3.6×10⁻¹⁴ 떨어졌다).
- `fill_value`는 `values`와 맞는 형의 수이거나 `None`이다(scipy와 같이 complex를 실수 값에 주면 `ValueError`).
- `__call__`의 `nu`는 축마다의 미분 차수이고 `slinear`, `cubic`, `quintic`에서만 받는다(IP30; scipy와 같다).

### 2.8 흩어진 점 (IP6)
```
RBFInterpolator(y, d, /, *, neighbors=None, smoothing=0.0, kernel='thin_plate_spline', epsilon=None, degree=None)
RBFInterpolator.__call__(x, /)
```
- 핵은 scipy의 여덟(`linear thin_plate_spline cubic quintic multiquadric inverse_multiquadric inverse_quadratic gaussian`)이고 `epsilon`과 `degree`의 기본과 검사는 scipy와 같다(척도 불변인 넷은 `epsilon` 기본 1, 다른 넷은 `epsilon`이 있어야 한다). 다항식 항은 자료의 상자로 옮기고 늘린 좌표의 단항식이다(scipy와 같다).
- `neighbors=k`면 평가점마다 가장 가까운 k개 자료점의 국소 RBF다(k > n이면 n; scipy와 같다). 이웃은 `sd.spatial.KDTree`의 (거리, 색인) 순서다(§1.4). 같은 이웃 집합의 평가점은 한 번 푼 계수를 함께 쓴다(scipy와 같다).
- `smoothing`, 낮은 `degree`, 같은 자료점은 IP25다. 다항식을 빼는 `degree=-1`도 핵의 최소보다 낮으면 `ValueError`다(IP29).
- `[scipy와 다름]`: 음수 `smoothing`은 `ValueError`(scipy는 받는다), 같은 자료점이 있고 `smoothing`이 0이면 `LinAlgError`(scipy는 이 초안에서 에러 없이 값을 냈다), `degree`가 핵의 최소보다 낮으면 `ValueError`(scipy는 경고).

### 2.9 scipy·numpy와 다른 곳 (모음)
위의 `[scipy와 다름]`, `[numpy와 다름]`을 모은다. 이유는 그 자리에 있다.
- 공통: 옵션은 키워드만, 쓰기 금지 계수, `check_finite` 없음(언제나 검사), complex 평가점은 `TypeError`, 0차원 결과는 Python 스칼라, 경고 대신 에러나 상태 속성.
- `sd.interp`: `xp`의 검사, 키워드만인 `left`·`right`·`period`, 유한하지 않은 `period`는 `ValueError`, 한 점의 `xp`에서도 NaN 평가점은 NaN(IP39), 끝값이 둘 다 −0이면 −0, 노드 사이의 단조와 범위(numpy의 식은 오른쪽 노드 바로 왼쪽에서 구간의 범위를 넘을 수 있다: 폭이 위치와 비슷한 구간 2만 개 중 159개, §3.2).
- 1차원: k = 1 B-스플라인이 노드에서 정확, `nu < 0`은 `ValueError`, `roots`의 배치 결과는 목록, `construct_fast`·`from_bernstein_basis` 없음.
- 다항식·유리식: `Barycentric`의 `rng` 없음, AAA의 `converged`와 경고 없는 `clean_up`, Krogh의 경고 없음.
- 격자: `slinear`와 `linear`가 같은 비트, `cubic`·`quintic`의 직접 풀이와 `solver` 없음, 옛 방법 없음, NaN 평가점은 NaN.
- RBF: 음수 `smoothing`, 같은 자료점, 낮은 `degree`가 에러.

---

## 3. 수치 방법 (안)

### 3.1 구간 찾기
- 평가점마다 구간 번호를 이분 탐색으로 찾는다. 규칙은 오른쪽 열린 구간(x[i] ≤ x < x[i+1], 마지막은 닫힘)이고 정의역 밖은 끝 구간(늘이기)이나 NaN(`False`)이다. 평가점이 정렬되어 있으면 앞 점의 구간에서 시작하는 질주 탐색이 빠르다. 어떤 탐색을 쓰든 구간은 규칙으로 정해지므로 결과는 같다.
- **구현** (IP1 마일스톤, IP37): `base/interval.hpp`가 점 아래(같은 값 포함)의 노드 수 c를 센다. `sd.interp`는 c를 numpy의 `binary_search_with_guess`처럼 쓰고(c = 0이면 `left`, c = n이면 마지막 노드 또는 `right`, 그 밖은 [xp[c − 1], xp[c]]; 반복 노드는 폭이 0인 구간을 고르지 않아 계단이 된다), 이 영역은 c − 1을 첫 구간과 끝 구간으로 자른다(`interpolate/search.hpp`). 평가점을 위치만으로 256점 블록으로 나누고, 블록이 정렬되어 있으면(NaN 없음) 앞 점의 c에서 1, 2, 4, …칸씩 질주한 뒤 마지막 배가 구간을 반씩 줄이고, 아니면 8점을 나란히 갈래 없이 반씩 줄인다(`searchsorted`의 탐색). 두 길은 같은 c를 내므로 결과는 블록의 길, 태스크 나누기, 스레드 수와 무관하다. 정렬된 평가점에서 노드 10⁶개, 10⁶점이 1스레드 8.1 ms(numpy 9.3), 무작위 순서는 15.2 ms(numpy 171)다(§5).
- scipy의 `BSpline`은 앞 점의 구간에서 한 칸씩 걷는다. 그래서 무작위 순서의 평가점에서 점마다 O(n)이다: 이 초안에서 평가점 10⁵개가 매듭 10³개에서 11.0 ms, 10⁴개에서 88.5 ms였고 10⁶개 매듭에 10⁶점은 3분 넘게 끝나지 않았다. sd는 순서와 무관하게 O(log n)이다.
- 감소하는 중단점(`PPoly`, RGI)은 만들 때 뒤집은 사본으로 찾는다(값의 계산은 원래 구간의 다항식이다).

### 3.2 선형 커널 (`base/lerp.hpp`; IP1)
점 x가 구간 [x₀, x₁]에 있고 끝값이 a, b일 때(IP4, IP26, IP27):

1. t = fl(fl(x − x₀)/fl(x₁ − x₀)). 폭 fl(x₁ − x₀)이나 fl(x − x₀)가 넘치면 x, x₀, x₁에 1/2을 곱한 값으로 같은 식을 쓴다(1/2을 곱한 두 값의 차는 넘치지 않는다; 뒤의 것은 정의역에서 아주 먼 늘이기에서만 생긴다, IP38). ±inf인 점의 t는 ±inf다.
2. t = 0이면 r = a, t = 1이면 r = b다. 그 밖에서 a와 b의 부호가 다르면(0 포함) r = fl(fl(t·b) + fl(fl(1 − t)·a)), 아니면 r = fl(a + fl(t·fl(b − a)))를 b 쪽으로 넘은 값에서 b로 자른다(t ≤ 1에서 b > a면 min(r, b), b < a면 max(r, b); t > 1이면 반대). 유한하지 않은 끝값은 아래의 갈래다.

- 이 꼴은 C++20 `std::lerp`(P0811)의 구성에 양 끝의 갈래(IP27)를 더한 것이다. 코드는 두 식을 모두 계산하고 고르는 갈래 없는 꼴로 쓴다(IP26; 같은 비트).
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
모든 3차 계열은 노드의 기울기 sᵢ를 구한 뒤 Hermite 꼴로 구간의 계수를 만든다: Δᵢ = (yᵢ₊₁ − yᵢ)/hᵢ, c₃ = (sᵢ + sᵢ₊₁ − 2Δᵢ)/hᵢ², c₂ = (3Δᵢ − 2sᵢ − sᵢ₊₁)/hᵢ, c₁ = sᵢ, c₀ = yᵢ (scipy `CubicHermiteSpline`과 같은 꼴). 계수의 연산 순서는 IP2에서 하나로 정해 모든 3차 계열이 같은 함수로 만든다(§9의 12).
- **`CubicSpline`**: 기울기의 삼중대각 계(hᵢ sᵢ₋₁ + 2(hᵢ₋₁ + hᵢ) sᵢ + hᵢ₋₁ sᵢ₊₁ = 3(hᵢ Δᵢ₋₁ + hᵢ₋₁ Δᵢ))와 끝의 조건이다. not-a-knot은 scipy처럼 첫 행과 끝 행을 삼중대각에 맞춘 꼴로 쓴다. periodic은 순환 계이고 scipy는 띠 풀이 두 번과 한 자리 보정으로 푼다(Sherman–Morrison 꼴).
  - **풀이**: 안쪽 행은 행마다 대각이 비대각 합의 두 배인 대각 우세라 피벗 없는 소거(Thomas)가 안정하다. 계의 행렬은 배치의 모든 열에 같으므로 한 번 분해하고 열마다 앞·뒤 대입을 한다. 열 여럿은 NEON 레인에 나란히 둔다. natural·clamped·(차수, 값) 조건은 대각 우세가 그대로다. not-a-knot의 끝 행은 대각 우세가 아니지만 아래의 양수 논증이 그대로 선다(§9의 13).
  - **피벗 없는 풀이의 증명 계획** (IP2에서 쓴다): natural·clamped·(차수, 값)·not-a-knot의 행렬(scipy의 행; not-a-knot의 첫 행은 [h₁, x₂ − x₀], 끝 행은 [xₙ₋₁ − xₙ₋₃, hₙ₋₃])은 모든 원소가 0 이상이다. 피벗 uᵢ는 귀납으로 양수이고 다음 행의 위 대각보다 크다: 안쪽 행에서 uᵢ₋₁ ≥ hᵢ₋₂(앞 행의 위 대각)면 lᵢ·hᵢ₋₂ ≤ hᵢ라 uᵢ ≥ 2hᵢ₋₁ + hᵢ > hᵢ₋₁이고, not-a-knot의 첫 두 피벗은 h₁과 h₀ + h₁, 끝 피벗은 hₙ₋₃(1 − (xₙ₋₁ − xₙ₋₃)/uₙ₋₂) > 0이다(uₙ₋₂ ≥ 2hₙ₋₃ + hₙ₋₂). 그래서 L ≥ 0, U ≥ 0이고 |L||U| = LU = A가 원소마다 성립해, 피벗 없는 LU의 성분별 역오차 |ΔA| ≤ γ₃|L̂||Û|(Higham, *Accuracy and Stability of Numerical Algorithms*의 LU 역오차 정리)가 |ΔA| ≲ γ₃|A|가 된다. 계산한 피벗도 정확한 피벗과 상대 γ 크기만 달라 양수로 남는 것과, 그 역오차에서 §4의 K를 얻는 것(행을 대각으로 나눈 계의 Skeel 조건수)은 IP2에서 적는다. periodic은 같은 양수 삼중대각에 Sherman–Morrison의 한 자리 보정이다.
  - **측정** (`tools/interpolate_probe.py banded`, 정확한 유리수의 피벗 없는 소거): 시험 자료의 실수 곡선 21개(n ≥ 4) 모두에서 natural, clamped, not-a-knot의 max (|L||U|)ᵢⱼ/|Aᵢⱼ|이 정확히 1이다(간격이 10¹⁰배 다른 spacing/1e10과 비정상 수 간격 포함).
  - scipy는 같은 계를 LAPACK `gbsv`(부분 피벗)로 푼다. 이 초안에서 n = 80의 natural 기울기를 정확한 유리수 풀이(`fmpq_mat`, 8 ms)와 견주니 scipy는 기울기마다 최대 47 ULP(작은 기울기의 상대) 떨어졌다. 그래서 계약은 성분마다의 상대가 아니라 기울기 크기의 척도로 적는다(§4).
- **`PchipInterpolator`**: Fritsch–Butland의 가중 조화 평균(scipy와 같은 식): Δᵢ₋₁Δᵢ ≤ 0이면 sᵢ = 0, 아니면 (w₁ + w₂)/sᵢ = w₁/Δᵢ₋₁ + w₂/Δᵢ, w₁ = 2hᵢ + hᵢ₋₁, w₂ = hᵢ + 2hᵢ₋₁. 끝은 scipy의 세 점 한쪽 식과 그 자르기다. 기울기 하나가 이웃 두 구간으로만 정해지는 국소 계산이라 조각으로 나눠 병렬이다.
- **`Akima1DInterpolator`**: Akima(1970)와 makima의 무게(scipy와 같은 식, 끝의 가상 기울기 둘씩 포함). 무게가 0/0이면 두 기울기의 평균이다(scipy와 같다). 국소 계산이다.
- 점이 둘이면 모든 계열이 직선이다(scipy와 같다).
- 계수 c₂, c₃은 기울기와 Δ가 유한해도 넘칠 수 있다(값 1e300 둘레, 간격 0.01 둘레). 그 처리는 §9의 23이다.

### 3.4 B-스플라인 (IP3)
- **평가**: de Boor 재귀다. 기본 구간 안에서 재귀의 무게 (x − t_l)/(t_{l+k+1−r} − t_l)는 [0, 1]이라 모든 단계가 볼록 결합이다. k = 1은 §3.2의 커널이다(재귀의 한 단계와 같은 수학이고 노드에서 정확하다). 도함수 `nu`는 차분한 계수로 차수를 낮춰 같은 재귀다.
- **보간의 연립방정식**: 배치점에서의 배치 행렬은 띠 행렬(너비 k)이고 완전 양(totally positive)이라 피벗 없는 가우스 소거가 안정하다(de Boor–Pinkus 1977; 출처의 확인과 상수는 IP3). 완전 비음 행렬의 피벗 없는 LU는 L ≥ 0, U ≥ 0이라 3.3의 논증처럼 |L||U| = |A|다. 그래서 `CubicSpline`과 같이 피벗 없는 띠 풀이를 한 번 분해, 열마다 대입으로 한다(§9의 13). 도함수 조건(`bc_type`)의 행은 그 끝의 보간 행 바로 뒤에 둔다(IP35): 도함수 행에는 음수가 있어 완전 비음이 깨지고, scipy의 순서(도함수 행이 맨 앞)로는 정확한 소거에서 모든 시험 곡선이 0 피벗을 냈다. IP35의 순서로는 k = 3 natural의 성장 인자가 1.97 이하(작은 곡선일수록 큼), clamped가 1이었다. k = 3, 5의 not-a-knot 배치 행렬은 21개 곡선 모두 성장 인자 1이다(`tools/interpolate_probe.py banded`). 끝의 2 × 2 블록의 상수는 IP3에서 증명한다. periodic은 순환 띠 계이고 scipy는 Woodbury 꼴로 푼다. 같은 꼴을 쓰되 작은 보정 계의 풀이는 IP3에서 고른다.
- **`derivative`·`antiderivative`·`integrate`**: 매듭과 계수의 차분·누적합(scipy와 같은 식). 누적합은 계수 순서의 차례 합이다(길이로만 정한 조각의 부분합을 그 순서로 더한다).
- **`insert_knot`**: Boehm의 매듭 넣기(scipy의 FITPACK `insert`와 같은 뜻).

### 3.5 조각 다항식의 연산 (IP2)
- **평가**: s = fl(x − xᵢ)에서 계수의 Horner이고 단계마다 `fma` 하나다(코드가 정한 곳의 FMA). 도함수는 정수배한 계수의 Horner다(계수를 미리 바꾸지 않고 평가 중에 곱한다; 계약 §4).
- **`derivative`·`antiderivative`**: 계수에 (m + 1)…를 곱하거나 나눈다(반올림 한 번씩). 역미분의 이어 붙이는 상수는 구간 순서의 누적합이다(길이로만 정한 조각, 조각 합을 차례로).
- **`integrate`**: 역미분의 차이를 구간마다 구해 pairwise 합으로 더한다(배열의 pairwise 합과 같은 트리).
- **`roots`·`solve`** (§9의 15): 조각마다 (1) 도함수의 근으로 단조 구간을 나눈다(재귀; 2차는 상쇄 없는 근의 공식), (2) 단조 구간의 양 끝에서 부호가 다르면 이분법으로 이웃한 두 double의 부호가 바뀌는 자리까지 좁힌다(고정 상한: 64 + 지수 범위의 걸음; 각 걸음에서 다항식 평가 한 번), (3) 극값에서의 값이 평가 계약 안에서 0이면 접하는 근으로 넣는다. 조각마다 독립이라 조각으로 나눠 병렬이고 결과는 구간 순서로 모은다. scipy는 3차 이하를 닫힌 식, 그 위를 동반 행렬의 고유값으로 구한다. 이 초안에서 scipy의 `roots`는 n = 10⁶ 구간에서 1.28초였다.
- **`extend`**: 계수와 중단점을 이어 붙인 새 배열을 만든다(쓰기 금지 배열을 고치지 않는다).

### 3.6 다항식·유리식 (IP4)
- **`Barycentric`의 무게**: wⱼ = 1/∏ₖ≠ⱼ(xⱼ − xₖ)를 입력 순서로 곱하되 곱마다 `frexp`로 지수를 따로 세어 넘침과 언더플로가 없다. 마지막에 모든 무게를 같은 2ᵉ로 나눠 가장 큰 무게의 지수를 0 둘레에 둔다(공식은 무게의 공통 배수에 무관하다). scipy의 무작위 섞기(`rng`)는 넘침을 피하려는 것이라 필요 없다. 평가는 둘째 공식이고 x가 노드와 같으면 그 값이다(scipy와 같다). 도함수는 scipy와 같은 재귀(Schneider–Werner)다.
- **`Krogh`**: Hermite 나눗셈 차분표와 Newton 꼴의 Horner다(scipy와 같은 식). 점의 순서는 입력 그대로다(바꾸면 반올림이 달라진다; Leja 순서는 §9의 19에서 다시 본다).
- **`AAA`** (§9의 17): scipy의 단계 그대로다: 오차가 가장 큰 표본을 지지점으로 더하고, Loewner 행렬의 가장 작은 특이값의 오른쪽 특이벡터(`gesdd`)를 무게로, 조건이 나빠지면(σ₁/σₘᵢₙ > 1/(3ε)) 열을 정규화한 SVD로, 행이 열보다 적으면 영공간으로 한다. 멈춤은 max|f − r| ≤ rtol·‖f‖∞이거나 `max_terms`다. 극점·영점은 화살촉 행렬 쌍의 일반화 고유값(`ggev`), 유수는 scipy의 식이다. `clean_up`은 scipy의 Froissart 판정과 다시 맞추기다.
  - 반복마다의 SVD가 시간의 거의 전부다(M × m, m ≤ `max_terms`). 열을 더하는 QR 갱신으로 SVD를 줄이는 길은 IP7에서 재 본다(결과는 같은 알고리즘의 반올림 차이).
- **`FloaterHormannInterpolator`**(§9의 6): 무게는 scipy의 닫힌 식, 평가는 무게중심 꼴이라 `Barycentric`의 평가 커널을 함께 쓴다.

### 3.7 격자 (IP5)
- **`linear`**: 축마다 구간과 t를 §3.1·§3.2로 구하고, 2ᵈ 꼭짓점의 값에서 마지막 축부터 첫 축까지 차례로 선형 커널을 쓴다(축 하나마다 꼭짓점이 반으로 준다). scipy는 꼭짓점 무게의 곱(∏ tᵢ 또는 1 − tᵢ)의 합이다. 차례로 쓰는 꼴은 노드에서 정확하고, 결과가 꼭짓점 값의 최솟값과 최댓값 사이이며, 1차원에서 `sd.interp`와 같은 비트다.
- **`nearest`**: 축마다 t ≤ 0.5면 왼쪽, 아니면 오른쪽 노드다(scipy와 같다; t는 같은 1번 식).
- **`slinear`**: k = 1의 텐서곱이고 계산은 `linear`와 같은 함수다.
- **`cubic`·`quintic`**: 텐서곱 스플라인의 계수는 축마다의 1차원 보간 풀이를 차례로 쓴 것과 정확히 같다(배치 행렬이 크로네커 곱이다). 그래서 축마다 §3.4의 띠 풀이를 그 축의 모든 줄에 한 번씩 쓴다(줄마다 독립이라 병렬). 반복 풀이가 없다. 평가는 축마다 de Boor를 차례로(국소 (k + 1)ᵈ 계수). scipy의 `gcrotmk` 길은 이 초안에서 100² 만들기 4.2 ms, 10⁵점 평가 10.4 ms였다.
- **`pchip`**: 1차원 PCHIP의 값은 이웃 네 노드로만 정해진다(기울기가 국소). 그래서 평가점마다 4ᵈ개의 값으로 축마다 PCHIP을 차례로 쓰면 scipy의 결과와 같은 함수다. scipy는 평가점마다 첫 축의 줄 전체로 1차원 보간을 새로 만들어 이 초안에서 100² 격자의 10⁴점이 573 ms였다.

### 3.8 RBF (IP6)
- **계**: [[K + diag(λ), P], [Pᵀ, 0]] [a; b] = [d; 0], K는 핵 φ(ε‖yᵢ − yⱼ‖), P는 옮기고 늘린 좌표의 단항식(scipy와 같다). 대칭 부정치라 LDLᵀ(`sytrf` + `sytrs`, Bunch–Kaufman)로 푼다(scipy의 `dsysv`와 같은 분해). 풀이가 특이하면 `LinAlgError`이고 단항식 행렬의 랭크가 모자라면 그 사실을 말한다(scipy와 같은 판정). 같은 자료점은 만들기 전에 찾는다(§9의 22).
- **K 만들기**: 짝의 거리는 spatial의 `euclidean` 짝 커널(spatial.md SPT26)과 같은 순서로 구하고, 대칭이라 반쪽만 계산해 복사한다. 행 조각으로 병렬이다.
- **평가**: 평가점 조각마다 핵 벡터와 계수의 곱이다. 메모리는 조각 크기로 정한 상한이다(scipy의 메모리 예산과 같은 생각). 합의 순서는 자료점 순서의 고정된 꼴이다(조각과 무관).
- **`neighbors=k`**: 평가점마다 `KDTree.query(k)`의 이웃(정렬된 색인 집합)을 구하고, 같은 집합끼리 묶어(색인 집합의 정렬과 고유화) 집합마다 (k + q)차의 국소 계를 한 번 푼다. 묶음은 집합의 사전 순서라 스레드 수와 무관하다. 국소 계는 작아서(k ≈ 50) LAPACK 호출의 고정비가 크다. 작은 계의 직접 LDLᵀ를 IP7에서 잰다(linalg.md §7.5의 AMX 직렬 문제와 같은 고민).
- **`NearestNDInterpolator`**(§9의 5): `KDTree.query(k=1)`의 색인으로 값을 고른다. 동점은 작은 색인이다. scipy의 `rescale`은 두고 `tree_options`와 질의 옵션은 뺀다(IP31).

### 3.9 병렬화와 결정성
**같은 입력이면 스레드 수와 상관없이 같은 결과다.** 점 하나의 값은 어느 태스크에서든 같은 커널이고, 합이 있는 곳은 크기로만 정한 순서다. 테스트는 1, 2, 3, 7, 10스레드에서 같은 바이트를 확인한다(§7).

| 병렬 경로 | 나누는 방법 | 결과가 같은 이유 |
|---|---|---|
| 평가(`__call__`, `sd.interp`, RGI, RBF) | 평가점 조각(크기로만) | 점 하나는 한 커널이다. 구간 찾기는 결과를 바꾸지 않는다 |
| 3차 계열·B-스플라인의 계수 | 배치의 열 묶음 | 열마다 같은 분해와 대입이다 |
| PCHIP·Akima의 기울기, 계수 만들기 | 노드 조각 | 국소 계산이다 |
| 역미분의 상수, B-스플라인의 누적합 | 길이로만 정한 조각의 부분합을 차례로 | 순서가 크기로 정해진다 |
| `integrate` | pairwise 트리 | 배열의 pairwise 합과 같다 |
| `roots` | 구간 조각 | 구간마다 독립, 구간 순서로 모은다 |
| 격자 스플라인의 계수 | 축마다 줄 묶음 | 줄마다 같은 풀이다 |
| RBF의 K, 평가, 국소 계 | 행 조각, 평가점 조각, 이웃 집합 | 집합은 KD 트리의 전순서로 정해진다 |
| RBF 전역 풀이, AAA의 SVD·고유값 | Accelerate | Accelerate의 스레드 모드가 같으면 같다(linalg.md §1.2) |

- 하나의 긴 열의 삼중대각·띠 풀이는 차례 계산이다(분해와 대입이 앞뒤로 이어진다). 10⁶점 하나는 1스레드 시간이다. 병렬 순환 소거는 반올림이 달라지므로(크기로만 나누면 스레드 수와는 여전히 무관) IP7에서 재 보고 정한다.
- 병렬 기준값은 배열의 규칙이다(array.md §10.4): 작업량이 float64 연산 2¹⁵개 분량 이상이면 나누고, 그 이상이면 스레드 수와 상관없이 GIL을 놓는다(API.md §3).

### 3.10 메모리
- 사용자가 모양으로 요청한 출력(평가점 × 배치, RBF의 (n + q)² 계)은 할당 전에 물리 메모리와 견주어 넘으면 `MemoryError`다(API.md §2).
- RBF 평가와 RGI 스플라인의 작업 메모리는 조각 크기로 정한 상한이 있다. 전역 RBF는 (n + q)² × 8바이트를 한 번 잡는다(n = 10⁴에서 800 MB).
- 객체는 계수 배열 하나와 중단점(또는 매듭)을 쥔다. 감소하는 중단점의 뒤집은 사본(§3.1)만 더 든다.

---

## 4. 정확도 계약 (안)
참값은 Arb와 정확한 유리수다. 입력이 float64라 자료와 계수는 정확한 유리수이고, 계약은 두 겹으로 적는다. **만들기**: 계산한 계수(기울기, B-스플라인 계수, 무게)를 같은 자료의 정확한 해(유리수 풀이를 올바르게 반올림한 것)와 견준다. **평가**: 계산한 계수가 나타내는 함수를 그 계수 그대로(정확한 유리수로) 평가한 값과 견준다. 두 겹을 합친 끝에서 끝의 오차(정확한 자료의 정확한 보간 대비)는 정확도 표에 보이지만 계약은 두 겹이다. 계약 단위(오차 ÷ 한계)로 적고 경계는 1이다(spatial.md §4와 같은 꼴). 상수는 각 마일스톤에서 증명하고 잰 값은 그때 채운다.

- **선형** (`sd.interp`, k = 1 B-스플라인, 격자 선형의 1차원)
  | 계약 | 메모 |
  |---|---|
  | \|ŷ − y\| ≤ γ₆(\|a\| + \|b\|) + 3η(0 ≤ t ≤ 1), γ₁₀·max(1, \|t\|)·(\|a\| + \|b\|) + 3η(늘이기, IP40), y = (1 − t)a + tb의 정확한 값(t는 x의 정확한 값으로), η = 2⁻¹⁰⁷⁵ | 증명은 §3.2(IP0 마일스톤, 늘이기는 IP1 마일스톤에서 고침). 갈래마다 같은 부호 γ₆·max(\|a\|, \|b\|)와 γ₇·max(1, \|t\|)·max, 반대 부호 γ₅(\|a\| + \|b\|)와 γ₁₀·max(1, \|t\|)·(\|a\| + \|b\|). `tests/interpolate/refdata.linear_bound`. 잰 값(IP1 마일스톤, `bench/accuracy/interpolate.py`의 `linear`): `sd.interp`가 곡선 안 1,736점에서 최대 0.235, 커널 구간 안 1,072점에서 0.500(비정상 수 끝값 tiny-values: 반올림 반 칸이 한계 2⁻¹⁰⁷³의 반), 선형 길의 늘이기 점에서 0.272 계약 단위. 같은 점의 numpy·scipy(k = 1)·R(`approx`)은 곡선 0.259·0.229·0.234, 구간 0.719·0.500·0.719이고 유한하지 않은 값을 낸 점(numpy 60, scipy 64, R 26)은 빼고 잰 것이다 |
  | 노드에서 0 오차, 단조인 자료에서 단조, 구간 끝값 사이 | 정확한 성질(§3.2). hypothesis와 끝점 둘레의 이웃 double로 시험 |
- **격자 선형** (d차원): \|ŷ − y\| ≤ γ₆d·Σ_c w_c\|v_c\|(w_c는 정확한 꼭짓점 무게, 합 1) 꼴, 노드에서 0 오차, 꼭짓점 값의 범위 안(정확한 성질). `nearest`는 고른 노드의 값 그대로(0 오차)와 동점 규칙.
- **조각 다항식의 평가** (Horner): \|p̂ − p(x)\| ≤ γ_{k+1}·Σₘ (m + 1)\|cₘ\|\|s\|ᵐ 꼴(s = x − xᵢ, k는 차수; s의 반올림과 FMA Horner의 k번의 반올림). 도함수는 같은 꼴에 정수배. 후보, IP2에서 증명.
- **3차 계열의 만들기**
  | 대상 | 계약(후보) | 메모 |
  |---|---|---|
  | `CubicSpline` natural·clamped·(차수, 값) | \|ŝᵢ − sᵢ\| ≤ K·u·maxⱼ\|Δⱼ\| | 행을 대각으로 나눈 계의 ∞-노름 조건수가 3 이하(대각 우세)라 상수가 작다. K는 IP2 |
  | not-a-knot, periodic | 같은 꼴, K는 IP2에서 증명하거나 잰 값의 여유 | §3.3의 끝 행 |
  | `PchipInterpolator`, `Akima1DInterpolator` | \|ŝᵢ − sᵢ\| ≤ K·u·max(\|Δᵢ₋₁\|, \|Δᵢ\|) (Akima는 이웃 넷) | 국소 식의 반올림 셈 |
  | 계수 c₂, c₃ | 기울기와 Δ의 오차를 Hermite 꼴로 옮긴 한계 | 같은 함수로 만든다(§3.3) |
- **B-스플라인**: 평가는 기본 구간 안에서 \|ŷ − Σ cⱼBⱼ(x)\| ≤ γ_{c·k}·Σⱼ\|cⱼ\|Bⱼ(x)(볼록 결합, Bⱼ ≥ 0, 합 1; c는 IP3). 만들기는 ‖ĉ − c‖∞ ≤ K·u·‖A⁻¹‖∞·‖y‖∞ 꼴이고 ‖A⁻¹‖∞는 참값 도구가 Arb로 구해 함께 적는다(매듭 배치에 따라 커질 수 있다).
- **`Barycentric`**: 둘째 공식의 한계 \|p̂ − p\| ≤ (3n + 4)u·Σⱼ\|ℓⱼ(x)yⱼ\| + (3n + 2)u·Λₙ(x)\|p(x)\| + O(u²)(Higham 2004; 무게의 반올림 포함, 확인은 IP4). 오른쪽은 참값 도구가 Arb로 구한다.
- **`Krogh`**: 나눗셈 차분표의 앞 오차와 Horner의 한계(차분표의 절댓값으로 적는 꼴, IP4). 점이 많고 고르면 커진다: 계약이 그 크기를 보이고 정확도 표가 잰다.
- **`AAA`**: (1) 지지점에서 0 오차(정확한 성질), (2) 계산한 (지지점, 값, 무게)의 무게중심 유리식을 Arb로 평가한 값 대비 첫째 공식 꼴의 한계, (3) 수렴한 객체는 표본에서 max\|r(zᵢ) − fᵢ\| ≤ rtol·‖f‖∞(sd의 평가로; 성질). 무게 자체는 SVD의 결과라 계약이 없다(알고리즘의 경로).
- **RBF**: (1) 풀이의 역오차 ‖Ac − d‖ ≤ K·u·‖A‖·‖ĉ‖(LDLᵀ의 증가 인자를 함께 적는다), (2) 계산한 계수로의 평가가 Arb 평가 대비 γ·Σⱼ\|ĉⱼ\|\|φ(rⱼ)\| + 핵 함수의 반올림(libm의 `log`, `sqrt`, `exp`가 1 ULP라는 가정; spatial.md의 가정 L과 같은 꼴), (3) 자료점에서의 잔차를 표에 적는다.
- **`roots`**: 돌려준 근 r̂마다 (1) 이웃 double에서 sd의 평가 부호가 바뀌거나, (2) 접하는 근이면 \|p̂(r̂)\|가 평가 계약 안이다. Arb로는 계수 그대로의 다항식이 [r̂ − δ, r̂ + δ]에 근을 갖는지(δ = 평가 한계 ÷ \|p′\|의 하한) 본다. 빠뜨린 근이 없는지는 Arb의 근 분리로 본다(계약 안에서 두 근이 붙은 경우는 하나로 셀 수 있다).
- **`integrate`·`antiderivative`·`derivative`**: 계수 연산의 반올림 셈(나눗셈·곱셈 한 번씩, 상수의 누적합 γ, 적분의 pairwise 합 γ_{⌈log₂ m⌉+k}).
- **PCHIP의 단조성** (성질 시험): 단조인 자료에서 (1) 계산한 기울기가 0이거나 Δ와 같은 부호이고 sᵢ/Δ가 3(1 + 4u) 이하, (2) 계산한 계수 그대로의 3차식이 구간마다 정확히 단조(도함수 2차식의 부호를 Arb로 확인), (3) 정렬한 평가점의 값이 평가 계약 안에서만 거꾸로 가고 구간 끝값을 평가 계약 안에서만 넘는다. scipy는 이 초안의 무작위 단조 자료 2000개에서 구간 끝값을 최대 3 ULP 넘었다(평가의 반올림; 단조가 깨진 곳은 없었다). Akima·makima는 평평한 구간이 정확히 평평한 것을 시험한다.
- scipy·numpy·R과의 차이는 계약이 아니라 정확도 표의 비교 패키지 열로 보인다.

---

## 5. 성능 (안)
`tools/interpolate_probe.py time`이 잰 scipy 1.18.1, numpy 2.5.4, R 4.6.2의 시간이다(ms, 세 번 중 가장 좋은 값(1회로 잰 것은 표시), 기본 설정, 부하 평균 1.3–1.4). scipy와 numpy의 보간 커널은 1스레드다(RBF의 `dsysv`와 AAA의 SVD는 Accelerate). 자료는 [0, 1)의 균등 노드와 표준정규 값, 평가점은 균등 무작위(정렬한 것은 표시). 사례(`bench/speed/cases/interpolate.py`, 36개)는 마일스톤마다 공개되는 이름과 함께 같은 크기를 잰다.

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

- **목표**(IP7에서 확인한다)
  - 평가(`sd.interp`, 조각 다항식, B-스플라인, RGI, RBF, `Barycentric`): 1스레드에서 scipy·numpy·R 가운데 가장 빠른 것 이상, 10스레드에서 거의 스레드 수만큼. 무작위 순서의 `BSpline` 평가는 구간 찾기로 수십 배(n = 10⁴에서 scipy의 O(n) 걷기 대비).
  - 만들기: 3차 계열과 B-스플라인은 1스레드에서 scipy 이상(피벗 없는 풀이, 배치 열의 NEON). 긴 열 하나는 차례 계산이라 스레드로 빨라지지 않는다(§3.9).
  - RGI의 `pchip`은 국소 평가로 수백 배, `cubic`·`quintic`은 직접 풀이로 scipy 이상이면서 반복 풀이의 오차가 없다.
  - RBF: 전역 만들기는 K를 병렬로 만들고 분해는 Accelerate라 n = 4000에서 scipy 이상, `neighbors`와 `NearestND`는 KD 트리 질의와 국소 계의 병렬로 10스레드에서 scipy의 5배 이상.
  - `roots`: 구간마다 독립이라 10스레드에서 거의 스레드 수만큼.
- **나중에 볼 것**: 긴 열 하나의 삼중대각을 병렬 순환 소거로(§3.9), AAA의 QR 갱신(§3.6), 정렬한 평가점의 질주 탐색과 SIMD 평가.

---

## 6. 층과 파일 (안)
- **층**: `interpolate/`는 `array/`, `base/`(`lerp.hpp`, 이 영역과 배열이 함께 쓴다), `parallel/`, `linalg/`(`lapack.hpp`: `sytrf`·`sytrs`, `gesdd`, `ggev`), `spatial/`(`kdtree.hpp`: RBF의 이웃, `NearestND`) 위에 있다. 다른 영역은 `interpolate/`를 모른다. `sd.interp`는 배열 영역의 `array/ops/interp.cpp`이고 `base/lerp.hpp`만 함께 쓴다(배열은 다른 영역을 include하지 않는다). Python 층의 `stardust.interpolate`는 `stardust.spatial`(`KDTree`)과 `stardust.sparse`(`design_matrix`)를 쓴다.
- `bench/areas.py`에서 `interpolate`는 `spatial`·`graph` 뒤, `plot` 앞이다. ARCHITECTURE §1의 그림과 §2의 목록, API.md의 색인 표, README.md의 표, tests/README.md·bench/README.md의 영역 목록에 한 줄씩 있다. `src/stardust/interpolate/__init__.py`는 공개 이름이 없는 뼈대다.
- `base/lerp.hpp`(IP1; 아래의 파일은 모두 `csrc/` 아래다): §3.2의 t와 선형 커널, 유한하지 않은 끝값의 갈래, 복소수판.
- `base/interval.hpp`(IP1, IP37): 점 아래의 노드 수(갈래 없는 8점 나란한 이분, 질주, 정렬된 블록을 고르는 `count_le_block`). 배열의 `array/ops/interp.cpp`와 `interpolate/search.hpp`가 함께 쓴다.
- `interpolate/`
  - `search.hpp`(IP1): 구간 번호(노드 수에서), 바깥의 규칙(`extrapolate`의 세 값, 주기의 감기). 감소 중단점은 IP2
  - `linear.cpp`(IP1): k = 1 평가와 격자 선형의 1차원 공용 길
  - `banded.hpp`(IP2): 피벗 없는 삼중대각·띠 분해와 열 묶음 대입, 순환 계의 보정
  - `cubic.cpp`(IP2): 3차 계열의 기울기(`CubicSpline`의 경계 조건, PCHIP, Akima·makima), Hermite 계수
  - `ppoly.cpp`(IP2): 평가, `derivative`, `antiderivative`, `integrate`, `roots`·`solve`, `extend`
  - `bspline.cpp`(IP3): de Boor, 매듭 규칙, 배치 행렬, `make_interp_spline`의 계, 도함수·역미분·적분, 매듭 넣기, 두 표현 사이의 변환
  - `polyint.cpp`(IP4): `Barycentric`의 무게와 평가·도함수, `Krogh`의 차분표와 평가, (Floater–Hormann)
  - `aaa.cpp`(IP4): AAA의 반복, 극점·유수·영점, `clean_up`
  - `grid.cpp`(IP5): RGI의 `linear`·`nearest`·`slinear`, 텐서곱 스플라인의 축마다 풀이와 평가, `pchip`의 국소 평가
  - `rbf.cpp`(IP6): 핵, 계 만들기와 풀이, 평가, 이웃 집합의 묶음과 국소 계
  - `interpolate.hpp`: 파일마다의 `bind_*`
- 바인딩 `bind/interpolate_bind.cpp`(IP1): `_core.interpolate` 하위 모듈. `sd.interp`의 바인딩은 배열의 바인딩에 있다.
- `src/stardust/interpolate/` (하위 패키지 없음): `__init__.py`(공개 이름), `_common.py`(입력 정리, 가로축 검사, `extrapolate`·`axis`의 검사, 메시지), `_ppoly.py`(`PPoly`와 3차 계열, `pchip_interpolate`), `_bspline.py`(`BSpline`, `make_interp_spline`), `_polyint.py`(`Barycentric`, `Krogh`과 함수 꼴, (Floater–Hormann)), `_aaa.py`, `_grid.py`(`RegularGridInterpolator`, `interpn`), `_rbf.py`(`RBFInterpolator`, (`NearestNDInterpolator`)).

---

## 7. 참값과 검증 (안)
다른 영역과 같은 체계다(ARCHITECTURE §6, tests/README.md, bench/README.md). 모든 도구는 `bench/cli.py`의 공통 옵션(영역, `--match`, `--jobs`)을 쓴다. 긴 참값 생성은 프로세스 10개까지, 메모리를 지켜보며(RSS가 평평한지) 돈다.

### 7.1 시험 데이터 (`tests/interpolate/samples.py`)
모두 식과 고정한 씨앗으로 만든다(IP34: 모듈 이름은 기록 폴더 `data/`와 겹치지 않게 `samples`). 곡선 25개(실수 한 열 23개, 복소수, 3열 배치), 선형 커널의 구간 62개(점 1,147개), 격자 10개, 흩어진 점 6벌, 조각 다항식 7개, 다항식 보간 6벌, AAA 표본 4벌이다. 모든 기록과 참값이 자료의 `digest`를 가진다.
- 1차원: 고른 노드, 무작위 노드, 간격이 10¹⁰배 다른 노드, 체비쇼프 점, 노드 2·3·4·5개(경계 조건의 특수한 경우), 단조 자료(0인 걸음 포함), 평평한 구간이 있는 자료, 계단, 진동(Runge), 큰 값(1e300 둘레)과 아주 작은 값, 상쇄(x나 y가 1e15 + 작은 변화), 비정상 수 간격, periodic용 자료(y[0] = y[-1]), 복소수 값, 배치 열. 평가점은 노드, 노드 양옆의 double, 중점, 무작위 점, 양 끝 밖의 점이다.
- 선형 커널: 끝점 둘레의 이웃 double, a와 b의 부호가 다른 짝, 같은 부호의 거의 같은 값, a = b, ±0, inf와 NaN 끝값, x₁ − x₀가 비정상 수인 구간, 폭이 넘치는 구간, DBL_MAX 둘레와 비정상 수의 끝값, 구간 밖의 점.
- 격자: 1–4차원, 축마다 1점·2점·감소 격자, NaN 값.
- 흩어진 점: 2·3차원 균등, 군집, 같은 점, 직선 위의 점(단항식 랭크 부족), 거리 동점이 많은 격자 점(이웃의 동점).
- 근: 접하는 근, 붙은 두 근, 0과 같은 조각, 중단점의 부호 변화.

### 7.2 참값 (`tools/interpolate_ref.py` → `tests/interpolate/ref/`, `tests/reffile.py`의 npz)
- 스플라인의 정확한 계수: 자료가 float64라 `fmpq_mat`의 정확한 풀이로 기울기와 계수를 구해 올바르게 반올림한다(이 초안의 n = 80 natural이 8 ms). B-스플라인은 같은 유리수 풀이와 ‖A⁻¹‖∞.
- 평가의 참값: 계산한 계수(sd가 만든 것과 참값 계수 둘 다)를 정확한 유리수로 평가한다. 핵 함수가 드는 RBF와 AAA는 Arb의 구간으로 반올림이 정해질 때까지 정밀도를 늘린다.
- 계약의 오른쪽(Λₙ, Σ\|ℓⱼyⱼ\|, ‖A⁻¹‖∞)과 근의 분리도 Arb다. 파일은 작게 둔다(노드 수백, 평가점 수천).
- 묶음은 여덟이다(모두 6,925점, 0.7 MiB): `linear`(정확한 a + t(b − a); 2,954), `slopes`(`CubicSpline` 네 조건, PCHIP, Akima·makima의 정확한 기울기; 577), `bspline`(k = 0–5, k = 3의 natural·clamped, k = 3·5의 periodic 계수와 곡선마다의 ‖A⁻¹‖∞; 627), `curves`(그 보간들의 정확한 값, 끝에서 끝; 1,870), `polynomial`(보간 다항식과 두 도함수, Krogh의 Hermite 자료, Lebesgue 함수, Σ\|ℓⱼyⱼ\|, Floater–Hormann d = 3; 378), `grid`(RGI의 linear·cubic·quintic·pchip; 400), `rbf`(핵 여덟과 `smoothing`·`neighbors`, Arb; 83), `roots`(Arb로 분리한 실근과 중복도; 36). 매듭은 scipy의 규칙으로 만들고 scipy의 매듭과 비트까지 같다. 같은 스플라인의 두 구성(`CubicSpline`의 기울기 계와 B-스플라인의 배치 행렬)이 참값에서 같은 것도 테스트가 본다. AAA는 sd가 계산한 무게가 있어야 해서 참값 파일이 없고, 테스트가 그 자리에서 Arb로 평가한다. scipy의 주기 B-스플라인은 바깥 매듭을 double로 x₀ − Δ처럼 계산해 주기가 반올림 크기로만 맞으므로 `k3_periodic`은 `periodic`과 10⁻¹³ 안에서만 같다.

### 7.3 테스트 (`tests/interpolate/`)
`test_api_surface.py`(API 문서의 시그니처), `test_interp.py`(`sd.interp`의 numpy 뜻: `left`·`right`·`period`, 반복 노드의 계단, 유한하지 않은 값, `xp`의 검사; 배열 영역의 테스트에도 둔다), `test_lerp.py`(§3.2의 성질: 끝점 정확, 단조, 범위, a = b, 유한하지 않은 끝값; hypothesis), `test_linear_bits.py`(`sd.interp`, k = 1, RGI `linear`·`slinear`, `interpn`의 같은 비트; 늘이기의 k = 1과 `fill_value=None`), `test_ppoly.py`, `test_cubic.py`(경계 조건마다, n = 2·3, 배치, periodic의 검사), `test_pchip.py`(§4의 단조성 세 겹), `test_akima.py`(평평한 구간), `test_bspline.py`, `test_make_interp_spline.py`, `test_roots.py`, `test_polyint.py`, `test_aaa.py`, `test_grid.py`, `test_rbf.py`, `test_reference.py`(§4의 계약, Arb 참값), `test_errors.py`(§2의 모든 에러와 메시지), `test_threads.py`(1, 2, 3, 7, 10스레드에서 같은 바이트).
- scipy·numpy와는 그 자리에서 견준다: 반올림이 아닌 동작(모양, 구간, 바깥, 에러)은 같고, 값은 계약 안이다.

### 7.4 scipy·numpy·R의 기록 (`tools/interpolate_probe.py` → `tests/interpolate/data/`)
- 탐침 도구 하나가 numpy `interp`의 경계와 식, scipy 객체의 경계와 기본값과 시험 자료에서의 값, R의 값, 커널의 측정, 시간을 낸다. 기록: `scipy.json`(713), `numpy.json`(53), `r.json`(94). 모든 기록은 그 자료의 `digest`를 가진다. `tests/interpolate/test_reference_data.py`가 기록이 참값과 맞는 것과 설계가 다르게 정한 동작(§2.9)이 기록에 보이는 것을 본다.
- R은 `approx`(`rule=2` ↔ `sd.interp`, `rule=1`의 NA ↔ `extrapolate=False`의 NaN), `spline`·`splinefun`의 `natural`·`periodic` ↔ `CubicSpline`만 비교 패키지다. R의 `fmm`(기본)은 끝 조건이 not-a-knot과 다르다(이 초안에서 같은 6점의 x = 0.3이 0.30271 대 0.30143). `monoH.FC`는 Fritsch–Carlson이라 PCHIP(Fritsch–Butland)과 다르다. R의 `approx`는 `x`를 정렬하고 같은 `x`를 평균으로 합치고 NA를 버리므로, 그런 자료는 R 열에서 뺀다.

### 7.5 정확도·속도 체계와 골든
- **정확도** (`bench/accuracy/interpolate.py`): 스윕 `linear`(sd·numpy·scipy·R의 선형), `coefficients`(3차 계열과 B-스플라인의 만들기 계약; 비교 패키지 scipy), `evaluation`(조각 다항식·B-스플라인의 평가 계약), `polynomial`(`Barycentric`, `Krogh`), `aaa`, `grid`(RGI의 방법마다), `rbf`, `roots`. 끝에서 끝의 오차도 열로 보인다.
- **속도** (`bench/speed/cases/interpolate.py`): 사례마다 마일스톤과 필요한 이름을 적어 두고 그 마일스톤이 끝나면 사례가 된다(spatial의 사례 파일과 같은 틀). §5의 크기와 정렬·무작위 평가점, 배치 열. 결과는 값을 견준다.
- **골든** (`tests/golden/cases_interpolate.py`, 기록 `tests/golden/data/interpolate.json`): 모든 공개 이름 × 경계 자료, 바깥의 세 값, 틀린 호출. 1·3·10스레드의 결과가 모두 같아야 한다. `sd.interp`는 배열의 골든(`cases_array.py`)에도 둔다.

---

## 8. 마일스톤 (안)
| 마일스톤 | 내용 | 끝났다는 뜻 |
|---|---|---|
| IP0 준비 — 완료·승인(위임, 2026-11-23) | API 문서 초안(`docs/api/interpolate.md`: 모든 공개 이름의 시그니처, scipy·numpy → sd 대응표와 뺀 이름의 바꿔 쓰기, §9의 답), 영역 등록(`bench/areas.py`, ARCHITECTURE, API.md의 색인, README), 시험 데이터(§7.1), 참값 도구(§7.2), 탐침 도구와 기록(§7.4), §3.2 선형 커널의 증명과 numpy 식·안 넷의 측정, 피벗 없는 띠 풀이의 증명 계획, 속도 사례의 틀 | 문서와 도구가 커밋되고 탐침 기록이 있다. §9의 답이 §1.1의 결정으로 옮겨졌다 |
| IP1 선형과 `sd.interp` — 완료·승인(위임, 2026-11-23) | `base/lerp.hpp`, `sd.interp`(배열 영역의 문서·테스트·골든 포함), 구간 찾기 | 선형 계약과 성질, numpy와 반올림이 아닌 동작이 같음, 속도 사례(numpy, R) |
| IP2 3차 계열 | `PPoly`(평가, 미분, 역미분, 적분, 근, `solve`, `extend`), `CubicHermiteSpline`, `CubicSpline`, `PchipInterpolator`, `pchip_interpolate`, `Akima1DInterpolator` | 만들기·평가 계약, PCHIP 단조성, R의 natural·periodic과 비교, 골든, scipy와의 속도 사례 |
| IP3 B-스플라인 | `BSpline`(모든 메서드), `make_interp_spline`(모든 k와 `bc_type`), `from_spline`·`from_power_basis`, k = 1의 같은 비트 | 계약, `sd.interp`와 같은 비트, 무작위 순서 평가의 속도 |
| IP4 다항식·유리식 | `BarycentricInterpolator`, `barycentric_interpolate`, `KroghInterpolator`, `krogh_interpolate`, `AAA`, (Floater–Hormann), linalg 래퍼(`ggev`) | 계약, AAA의 성질과 scipy와 같은 지지점 순서(같은 자료에서), 속도 |
| IP5 격자 | `RegularGridInterpolator`(방법 여섯), `interpn` | 선형 같은 비트, 스플라인의 직접 풀이 계약, `pchip`의 국소 평가가 scipy와 같은 함수, 속도 |
| IP6 흩어진 점 | `RBFInterpolator`(전역, `neighbors`), (`NearestNDInterpolator`), linalg 래퍼(`sytrs`) | 역오차·평가 계약, 이웃의 동점이 결과를 바꾸지 않음, 속도 |
| IP7 성능과 일관성 점검 | 모든 공개 함수를 크기마다 scipy·numpy·R과 견주어 고치기, 문턱(병렬, 탐색, 조각 크기), 메시지·시그니처의 영역 간 일관성, 코드 리뷰 | 속도 표와 정확도 표가 기준선이 되고 §5의 목표를 적은 대로 확인한다 |
| IP8 문서 | `docs/api/interpolate.md`의 확정, 이 문서를 구현대로 고치기(잰 값, 증명한 상수) | 문서가 코드와 맞고 `test_api_surface`가 통과한다 |

진행 원칙은 다른 영역과 같다: 마일스톤마다 시작 전에 §9의 질문을 묻고, 구현하고, 정확도·속도를 재고, 커밋하고, 결과를 전달하고, 승인받으면 다음으로 간다. IP1이 선형 커널과 구간 찾기를 만들므로 맨 앞이다. IP2와 IP4는 서로 기대지 않아 나란히 할 수 있다. IP3은 IP2의 띠 풀이를, IP5는 IP1과 IP3을, IP6은 spatial의 `KDTree`를 쓴다. 기존 영역의 결과는 비트까지 같아야 한다(배열에는 새 함수 하나와 바닥의 커널 하나를 더하고, linalg에는 LAPACK 래퍼만 더한다).

**나중 후보** (첫 판 밖): `BPoly`·`NdBSpline`·`NdPPoly`(§9의 7), 매끄럽게 하는 스플라인과 최소제곱 스플라인(`sd.smooth`), Qhull 보간(`griddata`, `LinearNDInterpolator`, `CloughTocher2DInterpolator`), 매개변수 곡선(`make_splprep`), 긴 열의 병렬 삼중대각(§3.9).

---

## 9. 마일스톤을 시작하기 전에 물을 것
1–22는 모두 (a)로 정했다(§1.1 IP4–IP25, 추천안, 사용자 위임 2026-11-23). 23은 IP0 마일스톤의 측정에서 나온 질문이고 IP1 마일스톤에서 (a)로 정했다(IP36).

추천을 (a)에 둔다. 답이 정해지면 IP4부터 §1.1에 옮긴다.

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
