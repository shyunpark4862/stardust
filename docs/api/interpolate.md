# sd.interpolate API

> 상태: **확정** (2026-11-23; 승인(위임)). §3–§7의 공개 이름 열여덟과 그 메서드·속성이 모두 구현되어 있다: 조각 다항식과 3차 계열(§3: `PPoly CubicHermiteSpline CubicSpline PchipInterpolator Akima1DInterpolator pchip_interpolate`), B-스플라인(§4: `BSpline make_interp_spline`, 그리고 `PPoly.from_spline`), 다항식·유리식(§5: `BarycentricInterpolator barycentric_interpolate KroghInterpolator krogh_interpolate FloaterHormannInterpolator AAA`), 격자(§6: `RegularGridInterpolator interpn`), 흩어진 점(§7: `RBFInterpolator NearestNDInterpolator`). 배열 영역의 `sd.interp`(§2)도 공개되어 있다. 시그니처는 scipy 1.18.1의 `scipy.interpolate`의 이름과 인자 이름에 stardust의 규칙과 IP1–IP107의 결정을 입힌 것이다.

- 설계와 결정의 근거는 [docs/design/interpolate.md](../design/interpolate.md)(§1.1 IP1–IP107), 공통 규칙은 [API.md](../../API.md)다.
- 언어 표시가 없는 코드 블록은 `tests/interpolate/test_api_surface.py`가 읽는다. 블록의 줄은 모듈의 이름(`name(...)`, 클래스는 `name(...)`의 생성자), 클래스의 메서드(`Class.method(...)`)와 속성(`Class.attr`)이다. 블록의 모듈 수준 이름이 `sd.interpolate.__all__` 전부이고, 시그니처가 블록과 같아야 한다.
- `sd.interp`는 배열 영역의 함수다(IP2). 시그니처와 규칙은 [docs/api/array.md §8.5](array.md)에 있다(`tests/array/test_api_surface.py`가 읽는다). 이 문서의 §2는 이 영역과 함께 쓰는 것만 적는다.

## §1. 공통 규칙

### §1.1 이름과 호출 모양 (IP3, IP7, IP11, IP14)
- **이름공간은 평평하다**: 모든 공개 이름이 `sd.interpolate`에 바로 있다. scipy의 `from scipy.interpolate import CubicSpline`은 `from stardust.interpolate import CubicSpline`이다. scipy의 옛 하위 모듈(`fitpack`, `fitpack2`, `interpnd`, `polyint`, `rbf`, `ndgriddata`, `interpolate`)은 없다.
- **보간 객체는 클래스다**: 만든 뒤 여러 번 부른다(`CubicSpline(x, y)(x_new)`). 함수 꼴(`pchip_interpolate`, `barycentric_interpolate`, `krogh_interpolate`, `interpn`)은 객체를 만들어 한 번 부른 것과 같은 비트다.
- **데이터는 위치로만, 옵션은 키워드로만** 받는다. `axis`, `bc_type`, `extrapolate`, `k`(`make_interp_spline`), `method`, `kernel`, `nu`, `der`도 키워드만이다(`[scipy와 다름]`: scipy의 `CubicSpline(x, y, 0, 'natural')`, `make_interp_spline(x, y, 3)`, `PPoly(c, x, True)`는 `TypeError`다). 스플라인을 정하는 수(`BSpline(t, c, k)`의 `k`, `PPoly.solve`의 `y`)는 위치다.
- **별칭이 없다**: `kernel`, `method`, `bc_type`의 문자열은 소문자 정식 이름만 받는다(`'Gaussian'`은 `ValueError`; `[scipy와 다름]`: scipy의 RBF는 대문자를 받는다). `check_finite`(언제나 검사한다, IP19), `rng`·`random_state`(IP21), `solver`·`solver_args`(IP23), `construct_fast`(IP12), `tree_options`(IP31)는 없다. 주면 모르는 키워드의 `TypeError`다.
- 틀린 호출 모양은 부른 이름으로 시작한다(API.md §2.1): `interp(): ...`, `CubicSpline(): ...`, `integrate(): ...`(메서드와 클래스 메서드도 그 이름; IP108); 객체를 부르는 `__call__`은 객체의 클래스 이름이다(`CubicSpline(): unexpected keyword argument 'foo'`, `CubicSpline(): 'x' must be ...`).

### §1.2 입력 (IP6, IP11, IP19)
- **받는 것**: 생성자, 메서드, 함수 꼴은 API.md §5의 상위 함수 입력(`sd.ndarray`, 수의 시퀀스, numpy 배열, `sd.frame`의 `Series`)과 `sd.frame`의 `DataFrame`을 받는다(IP109). `DataFrame`은 열을 나란히 둔 float64 2차원 배열이라(결측은 NaN) `sd.spatial`처럼 흩어진 점(`RBFInterpolator`의 `y`, `NearestNDInterpolator`의 `x`, 평가점)으로 쓴다. 1차원이어야 하는 인자에 주면 차원의 `ValueError`다. `sd.interp`는 배열 영역의 함수라 `sd.ndarray`와 Python 스칼라만 받는다.
- **가로축**(`x`, `xi`, 격자의 `points`, 매듭 `t`, 평가점): float64로 계산한다. bool·int64는 float64로 바꾼다. complex128은 `TypeError`다(`[scipy와 다름]`: scipy는 복소수 평가점의 실수부만 쓰고 `ComplexWarning`을 낸다). 예외는 `AAA`와 `FloaterHormannInterpolator`의 평가점 `z`다: 복소평면의 점을 받는다(scipy와 같다).
- **세로축**(`y`, 계수 `c`, 격자의 `values`, RBF의 `d`): float64 또는 complex128이다. `PchipInterpolator`, `pchip_interpolate`, `Akima1DInterpolator`, RGI의 `method='pchip'`은 실수만 받는다(complex는 `ValueError`; scipy와 같다). 다른 모든 것은 complex를 받고 실수부와 허수부를 같은 계산으로 따로 낸다.
- **가로축의 검사**: 1차원, 유한, 엄격히 증가해야 한다(`ValueError`; scipy와 같다). `PPoly`의 중단점과 RGI의 격자는 엄격히 감소해도 된다. `KroghInterpolator`는 같은 점의 반복을 도함수 자료로 받는다. `BarycentricInterpolator`와 `FloaterHormannInterpolator`는 서로 다른 점이어야 한다.
- **세로축의 유한성** (IP19): 전역 풀이로 계수를 만드는 것(`CubicSpline`, `PchipInterpolator`, `Akima1DInterpolator`, `CubicHermiteSpline`, `make_interp_spline`, `RBFInterpolator`)은 `y`(`dydx`, `d`)가 유한해야 한다(`ValueError`). `RBFInterpolator`만 `[scipy와 다름]`이다(scipy는 NaN `d`에서 NaN을 낸다). 국소 계산(`sd.interp`, 주어진 계수의 `PPoly`·`BSpline`, RGI, `Barycentric`, `Krogh`)은 IEEE 규칙으로 퍼뜨린다.
- **배치(`axis=`)**: 세로축은 여러 차원이고 `axis`(기본 0)가 가로축에 맞는 축이다. 결과의 모양은 scipy와 같다: 평가점의 모양이 `axis` 자리에 들어간다(`CubicSpline(x, y, axis=1)`에서 `y`가 (2, 6, 3), 평가점이 (4, 5)면 결과는 (2, 4, 5, 3)). 배치의 열마다 같은 계산이고 결과가 열마다 독립이다(IP33).
- **평가점의 NaN**: 결과는 NaN이다. 어느 보간이든, 바깥의 규칙과 무관하게 같다(IP24).
- **결과**: float64이거나, 세로축이 complex면 complex128이다. 결과는 보통의 새 배열이고 0차원 결과는 Python `float`·`complex`다(array.md §1.2; scipy는 0차원 numpy 배열).

### §1.3 바깥의 규칙 (IP13)
구간이 있는 1차원 보간(`PPoly`, `CubicHermiteSpline`, `CubicSpline`, `PchipInterpolator`, `Akima1DInterpolator`, `BSpline`, `make_interp_spline`의 결과)은 모두 `extrapolate=`이고 뜻이 하나다.

| 값 | 뜻 |
|---|---|
| `True` | 끝 조각의 다항식을 그대로 늘인다 |
| `False` | 정의역 `[x[0], x[-1]]`(B-스플라인은 `[t[k], t[n]]`) 밖은 NaN이다 |
| `'periodic'` | 정의역의 길이로 감아서 계산한다 |
| `None` | 생성자에서는 그 클래스의 기본, 메서드(`__call__`, `integrate`, `roots`, `solve`)에서는 객체의 값 |

- 기본은 scipy 그대로 `True`다. 예외는 둘이다: `bc_type='periodic'`인 `CubicSpline`과 `make_interp_spline`은 `'periodic'`, `Akima1DInterpolator`는 `False`다.
- 정의역의 끝점은 안이다(`x[-1]`은 마지막 구간으로 계산한다).
- 그 밖의 바깥: `sd.interp`는 numpy의 `left`·`right`·`period`(§2), RGI와 `interpn`은 `bounds_error`·`fill_value`(§6)이고, `BarycentricInterpolator`, `KroghInterpolator`, `FloaterHormannInterpolator`, `AAA`, `RBFInterpolator`, `NearestNDInterpolator`는 정의역이 없어 그 식을 어디서나 계산한다(scipy와 같다).

### §1.4 결정성과 같은 비트
- **같은 입력이면 스레드 수와 상관없이 같은 결과다**(API.md §3). LAPACK을 부르는 곳(RBF의 전역 풀이, AAA의 SVD, `AAA`와 `FloaterHormannInterpolator`의 극점·영점·유수)은 Accelerate의 스레드 모드가 같을 때 같다.
- **선형은 어느 길이든 같은 비트다**: `xp[0] ≤ x ≤ xp[-1]`에서 `sd.interp(x, xp, fp)`, `make_interp_spline(xp, fp, k=1)(x)`, `BSpline`의 k = 1(같은 매듭), `RegularGridInterpolator((xp,), fp, method='linear')`와 `'slinear'`, `interpn`의 1차원 선형이 비트까지 같다(`xp`에 반복이 없을 때). 늘이기에서도 k = 1과 RGI의 `fill_value=None`이 같다. 선형 커널 하나가 모든 길을 계산한다(docs/design/interpolate.md §3.2). `[scipy와 다름]`: scipy에서는 이 길들의 끝자리가 다르다(1차원 RGI `linear`와 `np.interp`는 점의 54%, `slinear`와 `linear`는 49%에서만 같았다).
- **3차 계열은 같은 기울기면 같은 계수다**(IP15): `CubicHermiteSpline(x, y, s)`와 같은 기울기의 `CubicSpline`, `PchipInterpolator`, `Akima1DInterpolator`는 계수가 비트까지 같다.
- 노드에서는 자료 그대로다: 선형, k = 0·1 B-스플라인, `Barycentric`, `AAA`의 지지점, RGI의 `linear`·`nearest`.
- 노드(지지점)가 하나인 `Barycentric`, `AAA`, `FloaterHormannInterpolator`, `KroghInterpolator`는 상수이고 유한한 모든 점에서 그 값 그대로다(IP111). 무한한 점에서는 `AAA`·`FloaterHormannInterpolator`의 극한과 `KroghInterpolator`도 그 값이고, `Barycentric`은 다항식의 규칙대로 NaN이다.
- **피클과 복사**(IP110): 모든 보간 객체는 `pickle`, `copy.copy`, `copy.deepcopy`가 된다. 되돌린 객체는 같은 배열(계수, 지지점, 캐시한 극점)을 비트까지 갖고 같은 비트를 계산한다. 고치는 메서드(`add_xi`, `set_yi`, `extend`, `clean_up`)는 복사본만 고친다.

### §1.5 에러와 경고
- API.md §2의 종류와 꼴이다. 경고는 내지 않는다. scipy가 경고하는 곳은 이렇다.

  | scipy의 경고 | sd |
  |---|---|
  | 복소수 평가점(`ComplexWarning`) | `TypeError` |
  | RBF의 낮은 `degree`(`UserWarning`) | `ValueError` (IP25, IP29) |
  | Krogh의 30차 넘는 차수(`UserWarning`) | 없음. 결과는 계산하고 정확도 계약이 오차의 크기를 보인다(IP22) |
  | AAA의 수렴 실패(`RuntimeWarning`) | `converged` 속성이 `False`(IP20) |
  | AAA `clean_up`의 Froissart 쌍 | `clean_up()`이 지운 수를 돌려준다(경고 없음) |

- 요청한 출력(평가점 × 배치, RBF의 (n + q)² 계, `neighbors=k`의 (k + q)² 국소 계)이 물리 메모리보다 크면 할당 전에 `MemoryError`다(API.md §2).

### §1.6 scipy·numpy와 다른 점 (모음)
- 공통: 옵션은 키워드만, 별칭 없음, 쓰기 금지 계수(생성자가 복사한다), `check_finite` 없음, complex 평가점은 `TypeError`, 0차원 결과는 Python 스칼라, 경고 대신 에러나 상태 속성, 선형의 모든 길이 같은 비트, 3차 계열의 계수가 같은 비트.
- `sd.interp`: `xp`의 검사, 키워드만인 `left`·`right`·`period`, 유한하지 않은 `period`는 `ValueError`, 노드 사이의 단조와 범위, 끝값이 둘 다 −0이면 −0(§2).
- 1차원: k = 1 B-스플라인이 노드에서 정확, `nu < 0`은 `ValueError`(`__call__`의 `nu`는 int만, IP50), B-스플라인의 `nu > k`는 바깥(`extrapolate=False`)에서 NaN(IP52), `BSpline(extrapolate=None)`은 `True`(IP52), `PPoly.from_spline`의 중단점은 기본 구간의 서로 다른 매듭(IP54), 비정상 수 간격의 B-스플라인을 만들고 평가한다(IP55), `BSpline.integrate(extrapolate=False)`의 배치는 겹침의 적분(IP57), `from_power_basis`의 배치와 에러 종류(IP58), `roots`의 배치 결과는 중첩 list, `roots`가 중근과 붙은 근을 찾음(IP45), 주기 검사의 허용이 하나(IP28), Akima의 문턱은 열마다(IP33), `PPoly`의 중단점은 유한하고 엄격히 단조(IP43), `integrate`의 NaN 끝은 NaN(IP49), 넘치는 3차 계수는 `ValueError`(IP36), `PPoly(axis=)`가 음수 축을 받음(IP50).
- 다항식·유리식: `Barycentric`의 `rng` 없음, AAA의 `converged`와 경고 없는 `clean_up`, `clean_up`의 인자 이름(IP32), Krogh의 경고 없음, `add_xi`가 무게를 처음부터 다시 셈(IP62), `FloaterHormannInterpolator`의 같은 점은 에러(IP63), 차수를 넘는 도함수의 NaN 점은 NaN(IP64), 유리식 결과의 모양(IP65), `wi`의 척도와 `yi`의 모양(IP66), AAA 인자의 검사(IP67), 노드에 비정상 수만큼 가까운 점은 그 노드의 값(IP68).
- 격자: `slinear`와 `linear`가 같은 비트, `cubic`·`quintic`의 직접 풀이와 `solver` 없음, 옛 방법(`cubic_legacy`, `quintic_legacy`, `slinear_legacy`, `interpn`의 `splinef2d`) 없음, NaN 평가점은 NaN, 무한한 노드는 에러(IP73), NaN 값은 무게가 0이 아닌 점에만 들고 `slinear`도 받는다(IP74), `bounds_error`·`fill_value`·`nu`의 검사(IP75), 스칼라 튜플의 결과는 Python 스칼라(IP76), `fill_value=None`의 무한한 좌표는 직선의 값(IP77).
- 흩어진 점: 음수 `smoothing`, 같은 자료점, 랭크가 모자란 다항식 꼬리, 핵의 최소보다 낮은 `degree`(−1 포함), 0이나 음수의 `epsilon`이 에러(IP25, IP29, IP90–IP92), `kernel`은 소문자만, r은 ε·거리(IP91), 평가점은 (…, d) 배열(IP93), 이웃이 있어야 하는 길의 유한하지 않은 점은 NaN(IP94), `NearestNDInterpolator`의 동점은 작은 색인이고 좌표가 하나여도 된다(IP93, IP98).

## §2. `sd.interp` (배열 영역)
- 시그니처와 규칙은 docs/api/array.md §8.5다: numpy의 뜻(IP2), `xp`의 검사(IP5), 키워드만인 `left`·`right`·`period`, 노드에서 정확하고 단조이며 구간 끝값 사이의 값, 끝값이 둘 다 −0이면 −0, NaN 평가점은 NaN(한 점의 `xp`도, IP39).
- 이 영역과 함께 쓰는 것: 선형 커널(`base/lerp.hpp`)과 구간 찾기(`base/interval.hpp`, IP37)가 하나라서, `xp[0] ≤ x ≤ xp[-1]`에서 `sd.interp(x, xp, fp)`는 이 영역의 선형 길(`make_interp_spline(xp, fp, k=1)`, k = 1 `BSpline`, 1차원 RGI `linear`·`slinear`, `interpn`)과 비트까지 같다(`xp`에 반복이 없을 때; §1.4). 정확도 계약은 docs/design/interpolate.md §4의 선형 행이다.

## §3. 조각 다항식
```
PPoly(c, x, /, *, extrapolate=None, axis=0)
PPoly.__call__(x, /, *, nu=0, extrapolate=None)
PPoly.derivative(*, nu=1)
PPoly.antiderivative(*, nu=1)
PPoly.integrate(a, b, /, *, extrapolate=None)
PPoly.roots(*, discontinuity=True, extrapolate=None)
PPoly.solve(y=0.0, /, *, discontinuity=True, extrapolate=None)
PPoly.extend(c, x, /)
PPoly.from_spline(tck, /, *, extrapolate=None)
PPoly.c
PPoly.x
PPoly.extrapolate
PPoly.axis
CubicHermiteSpline(x, y, dydx, /, *, axis=0, extrapolate=None)
CubicSpline(x, y, /, *, axis=0, bc_type='not-a-knot', extrapolate=None)
PchipInterpolator(x, y, /, *, axis=0, extrapolate=None)
Akima1DInterpolator(x, y, /, *, axis=0, method='akima', extrapolate=None)
pchip_interpolate(xi, yi, x, /, *, der=0, axis=0)
```
- **`PPoly`**: 계수 `c`는 (k + 1, m, …)이고 높은 차수가 먼저다. 구간 i의 다항식은 지역 변수 s = x − x[i]의 다항식이다. `axis`가 0이 아니면 받은 `c`는 (…, k + 1, m, …)이고 k + 1이 `axis` 자리다(scipy와 같다; 음수 축도 받는다, IP50). 중단점 `x`는 (m + 1,)이고 유한하며 엄격히 증가하거나 엄격히 감소한다(IP43; `[scipy와 다름]`: scipy는 같은 중단점과 무한한 중단점을 받는다). 계수와 중단점은 쓰기 금지 배열이다(생성자가 복사한다; `[scipy와 다름]`: scipy는 쥔 배열을 고치면 객체가 바뀐다). 주어진 계수의 NaN과 inf는 IEEE로 퍼진다(IP19). 뒤의 넷(`CubicHermiteSpline`, `CubicSpline`, `PchipInterpolator`, `Akima1DInterpolator`)은 `PPoly`의 하위 클래스이고, 계수가 유한하지 않게 되면 만들 때 `ValueError`이고 메시지가 그 구간을 말한다(IP36; scipy는 1e300 둘레의 자료에서 노드에서도 NaN인 객체를 만든다).
- **구간**: x[i] ≤ x < x[i+1]인 구간 i로 계산하고 x = x[-1]은 마지막 구간이다(감소하는 중단점은 거울 규칙 x[i+1] < x ≤ x[i]; scipy와 같다; 불연속인 조각 다항식과 도함수에서 어느 쪽 값인지가 이 규칙이다).
- **`nu`**: 0 이상의 int다(`nu=1.0`은 `TypeError`; scipy는 받는다). 차수보다 크면 0이다(NaN 평가점과 `extrapolate=False`의 바깥은 NaN). 음수는 `ValueError`다(역미분은 `antiderivative`). `derivative(nu=-k)`는 `antiderivative(nu=k)`이고 반대도 같다(scipy와 같다). 둘 다 같은 클래스의 객체를 돌려준다(scipy와 같다).
- **`CubicSpline`의 `bc_type`**: 양 끝에 같은 조건인 이름 `'not-a-knot'`, `'clamped'`, `'natural'`, `'periodic'`이거나, (왼쪽 끝, 오른쪽 끝)의 짝이다. 짝의 원소는 이름이나 `(차수, 값)`(차수 1, 2; 값은 배치 모양)이라 끝마다 다른 조건도 받는다(`('natural', 'not-a-knot')`, `((1, 0.0), (2, 0.0))`). 길이 2인 튜플은 언제나 짝으로 읽는다: 양 끝의 같은 도함수 조건은 `((1, 0.0), (1, 0.0))`이고, `bc_type=(1, 0.0)`은 왼쪽 끝이 `1`인 짝이라 `ValueError`다(scipy와 같다). `'periodic'`은 양 끝 모두여야 하고 `y[0]`과 `y[-1]`이 `allclose(rtol=1e-15, atol=1e-15)` 안이어야 한다(IP17; 메시지는 축 번호를 말한다). 점이 둘이면 직선(clamped는 기울기 0의 3차식), 셋이고 양 끝이 not-a-knot이면 포물선이다(scipy와 같다). 점이 하나면 `ValueError`다.
- **`PchipInterpolator`**: Fritsch–Butland의 가중 조화 평균과 scipy의 끝 식(세 점 한쪽 식과 그 자르기)이다. 단조인 자료에서 결과가 단조다.
- **`Akima1DInterpolator`**: `method='akima'`는 Akima(1970), `'makima'`는 수정 Akima다(scipy와 같은 식, 끝의 가상 기울기 둘씩 포함). 무게의 합이 그 열의 최댓값의 10⁻⁹배 이하면 두 기울기의 평균이다(scipy와 같은 문턱; `[scipy와 다름]`: scipy는 최댓값을 모든 열에서 하나로 잡아 한 열의 크기가 다른 열의 기울기를 바꾼다, IP33). makima는 평평한 구간(같은 값이 셋 이상)이 정확히 평평하다. akima는 그 구간의 안쪽 노드에서 기울기가 0이지만 구간 끝 노드가 Akima의 정하지 않은 경우(양쪽 두 기울기가 각각 같은 경우)면 평균 기울기라 평평하지 않을 수 있다(IP47; scipy와 같다). PCHIP과 Akima의 계수는 시험 자료 전부에서 scipy와 비트까지 같다(IP42).
- **`roots`·`solve`**: 조각마다 실근을 찾아 정렬된 1차원 배열로 돌려준다. 배치면 배치 모양의 중첩 list다(`[scipy와 다름]`: scipy는 dtype `object`의 배열이다). `discontinuity=True`면 중단점에서 부호가 바뀌는 점도 근이다. 조각이 0과 같으면 그 구간의 왼쪽 끝과 NaN을 넣는다(scipy와 같다). `extrapolate=True`면 첫 조각과 끝 조각의 늘인 다항식의 근도 넣는다(scipy와 같다). `extrapolate='periodic'`은 True로 친다(scipy와 같다). 접하는 근(중근)도 찾고 2⁻³⁰ 떨어진 두 근도 마지막 비트까지 가른다(IP45; `[scipy와 다름]`: scipy는 (x − 0.25)²(x + 1) 꼴의 중근을 놓치고, 2⁻³⁰ 떨어진 두 근을 7×10⁻⁹ 틀린다; docs/design/interpolate.md §3.5). `solve(y)`의 `y`는 유한한 수다. complex 계수는 `ValueError`다(scipy와 같다).
- **`integrate(a, b)`**: a > b면 부호가 바뀐다. `extrapolate=False`이고 a, b가 정의역 밖이면 NaN, `'periodic'`이면 온전한 주기의 수 × 한 주기의 적분 + 나머지다(scipy와 같다). a나 b가 NaN이면 NaN이다(IP49; `[scipy와 다름]`: scipy는 `ValueError`). 배치면 열마다의 배열, 아니면 Python 스칼라다.
- **`antiderivative`**: 상수는 x[0]에서 0이고 이어지게 고른다. `'periodic'` 객체의 역미분은 `extrapolate=False`다(IP48; scipy와 같다).
- **`extend(c, x)`**: 오른쪽(감소하는 중단점이면 왼쪽)에 구간을 더한다. `x`는 새 조각마다 한 점(오른쪽에 더하면 새 조각의 오른쪽 끝)이다. 차수가 다르면 낮은 쪽의 위를 0으로 채운다. 객체를 고치는 메서드다(IP12; scipy와 같다). 고친 뒤의 계수도 쓰기 금지 배열이다. 새 중단점이 끝 중단점과 같으면 `ValueError`다(IP43; scipy는 폭이 0인 조각을 만든다).
- **`from_spline(tck)`**: `BSpline`이나 `(t, c, k)` 튜플을 같은 함수의 `PPoly`로 바꾼다(중단점과 바깥은 §4).
- **`pchip_interpolate(xi, yi, x, der=0)`**: `PchipInterpolator(xi, yi, axis=axis)(x, nu=der)`와 같다. `der`가 시퀀스면 각 차수의 결과 list다(scipy와 같다).

## §4. B-스플라인
```
BSpline(t, c, k, /, *, extrapolate=True, axis=0)
BSpline.__call__(x, /, *, nu=0, extrapolate=None)
BSpline.derivative(*, nu=1)
BSpline.antiderivative(*, nu=1)
BSpline.integrate(a, b, /, *, extrapolate=None)
BSpline.basis_element(t, /, *, extrapolate=True)
BSpline.design_matrix(x, t, k, /, *, extrapolate=False)
BSpline.insert_knot(x, /, *, m=1)
BSpline.from_power_basis(pp, /, *, bc_type='not-a-knot')
BSpline.t
BSpline.c
BSpline.k
BSpline.extrapolate
BSpline.axis
BSpline.tck
make_interp_spline(x, y, /, *, k=3, t=None, bc_type=None, axis=0)
```
- **`BSpline`**: 매듭 `t`(줄지 않음, 길이 n + k + 1), 계수 `c`(첫 축이 n), 차수 `k`(0 이상의 int). 검사는 scipy와 같다. 정의역 안의 구간 규칙은 t[l] ≤ x < t[l+1]이고 x ≥ t[n]은 l = n − 1이다(scipy와 같다; k = 0에서는 그 구간이 비어 있어 마지막 노드의 값이 c[-1]이다). 평가는 de Boor이고 무작위 순서의 평가점도 점마다 O(log n)이다(scipy는 앞 점의 구간에서 걸어 무작위 순서에서 점마다 O(n)이다: 매듭 10⁴개에 무작위 10⁵점이 sd 0.89 ms, scipy 86.7 ms). 계수는 첫 축이 보간 축인 쓰기 금지 사본이고(`axis`는 그 축을 앞으로 옮긴다; scipy와 같다) n개보다 많아도 되며 처음 n개만 쓴다. `extrapolate=None`은 클래스의 기본 `True`다(IP52; `[scipy와 다름]`: scipy는 `False`). 주어진 계수의 NaN과 inf는 IEEE로 퍼진다(IP19).
- **`make_interp_spline`**: 차수 k의 보간 스플라인이다. 매듭을 주지 않으면 scipy의 규칙이다: k = 0은 t = (x₀, …, xₙ₋₁, xₙ₋₁)과 c = y(왼쪽에서 이어지는 계단, 마지막 노드에서는 y[-1]), k = 1은 t = (x₀, x₀, …, xₙ₋₁, xₙ₋₁)과 c = y, 홀수 k는 not-a-knot(안쪽 매듭 x[(k+1)/2 : −(k+1)/2]), 짝수 k는 이웃한 노드의 중점(double로 계산한 (x[i] + x[i+1])/2)에서 앞뒤 k/2개를 뺀 것, `bc_type`이 (차수, 값) 목록이거나 `'natural'`·`'clamped'`면 노드에 양 끝 k겹을 더한 것, `'periodic'`이면 scipy의 주기 매듭이다. `bc_type`은 `None`, `'not-a-knot'`, `'natural'`, `'clamped'`, `'periodic'`, 양 끝의 (차수, 값) 목록이다. 짝이 맞지 않는 조건은 scipy와 같은 `ValueError`다(`k=5, bc_type='natural'`). `'periodic'`의 검사는 `CubicSpline`과 같은 `allclose(rtol=1e-15, atol=1e-15)`다(IP28; `[scipy와 다름]`: scipy의 `make_interp_spline`은 rtol 1e-5라 1e15 둘레의 자료는 끝값이 10¹⁰ 달라도 받는다).
- **k = 1**: 계수는 `y` 그대로이고 평가는 선형 커널이라 `sd.interp`와 같은 비트다(§1.4). 노드에서 `y`를 그대로 돌려준다(`[scipy와 다름]`: scipy는 무작위 노드 50개 중 9개에서 끝자리가 달랐다).
- **`nu`**: 0 이상(scipy의 `nu=-1`은 `NotImplementedError`; sd는 `ValueError`). 차수보다 크면 0이고 NaN 평가점과 `extrapolate=False`의 바깥은 NaN이다(IP52, `PPoly`와 같다; `[scipy와 다름]`: scipy는 바깥에서도 0).
- **`design_matrix`**: (len(x), n) `sd.sparse.csr_array`다(scipy는 `scipy.sparse.csr_array`). 행마다 k + 1개의 저장 자리이고 0인 값도 저장한다(scipy와 같다). `x`는 유한한 1차원이고(비어도 된다), `extrapolate=False`면 기본 구간 [t[k], t[n]] 안이어야 한다(`ValueError`). `extrapolate`는 `True`, `False`, `'periodic'`이다(IP58).
- **배치 계**: 도함수 조건의 행은 그 끝의 보간 행 바로 안쪽에 둔다(IP35). 도함수 조건이 없을 때(`t`를 주어도), 각 끝의 차수가 1, …, r일 때, k = 3의 natural은 피벗 없이(성장 1, 1, 2 미만; docs/design/interpolate.md §3.4의 증명), 그 밖의 차수 집합과 `t`를 준 도함수 조건은 부분 피벗으로 푼다(IP53). 그래서 `t`에 기본 매듭을 주면 `t` 없이 만든 것과 비트까지 같다. 정칙이 아니면(Schoenberg–Whitney가 깨지거나 같은 조건이 두 번) `sd.linalg.LinAlgError`다(scipy와 같다). 간격이 비정상 수인 끝의 도함수 조건은 넘쳐서 `ValueError`다(IP60). `bc_type='periodic'`은 k = 0, 1에서 끝값만 검사하고 주기가 아닌 스플라인이다(scipy와 같다).
- **`derivative`·`antiderivative`**: scipy `splder`·`splantider`의 식이다(도함수의 계수는 뒤에 0 k개, 역미분은 t[0]에서 0). `derivative(nu > k)`와 안쪽 매듭이 k + 1겹인 도함수는 `ValueError`다(scipy와 같다). `'periodic'`의 역미분은 `extrapolate=False`다.
- **`integrate(a, b)`**: a > b면 부호가 바뀐다. `extrapolate=False`는 [a, b]와 기본 구간 [t[k], t[n]]의 겹침을 적분하고(`[scipy와 다름]`: scipy의 배치는 두 끝이 한쪽 바깥이면 NaN), `'periodic'`은 온전한 주기와 나머지다(scipy와 같다). 끝이 NaN이면 NaN이다(IP49와 같다). complex 계수도 적분한다(IP57).
- **`basis_element(t)`**: k = len(t) − 2의 B-스플라인 하나다: 앞뒤에 t[0] − 1, t[−1] + 1을 k개씩 붙인 매듭과 k 자리의 계수 1(scipy와 같다).
- **`insert_knot(x, m=1)`**: 같은 함수에 매듭 x를 m번 넣은 새 `BSpline`이다(Boehm; scipy와 같은 식이라 같은 (t, c)에서 같은 비트). x는 기본 구간 안이어야 한다.
- **`from_power_basis(pp, bc_type='not-a-knot')`**: `CubicSpline` `pp`를 `bc_type`이 정하는 매듭의 B-스플라인으로 바꾼다(scipy와 같은 매듭과 식; 다른 형은 `TypeError`, 배치도 받는다, IP58).
- **`PPoly.from_spline(tck)`**: `BSpline`이나 (t, c, k) 튜플을 `PPoly`로 바꾼다. 중단점은 기본 구간 [t[k], t[n]]의 서로 다른 매듭이라 `extrapolate`의 세 값 모두에서 B-스플라인과 같은 함수다(IP54; `[scipy와 다름]`: scipy는 모든 매듭이라 폭이 0인 조각이 생기고 바깥에서 다른 함수다). 다만 k = 0의 t[n]과 그 오른쪽은 B-스플라인이 c[−1](빈 마지막 구간)이고 `PPoly`는 마지막 조각을 잇는다. `extrapolate=None`이면 `BSpline`의 값, 튜플이면 `True`다.
- `tck`는 (t, c, k) 튜플이다.

## §5. 다항식·유리식
```
BarycentricInterpolator(xi, yi=None, /, *, axis=0, wi=None)
BarycentricInterpolator.__call__(x, /)
BarycentricInterpolator.derivative(x, /, *, der=1)
BarycentricInterpolator.derivatives(x, /, *, der=None)
BarycentricInterpolator.add_xi(xi, yi=None, /)
BarycentricInterpolator.set_yi(yi, /, *, axis=None)
BarycentricInterpolator.xi
BarycentricInterpolator.yi
BarycentricInterpolator.wi
barycentric_interpolate(xi, yi, x, /, *, axis=0, der=0)
KroghInterpolator(xi, yi, /, *, axis=0)
KroghInterpolator.__call__(x, /)
KroghInterpolator.derivative(x, /, *, der=1)
KroghInterpolator.derivatives(x, /, *, der=None)
krogh_interpolate(xi, yi, x, /, *, der=0, axis=0)
FloaterHormannInterpolator(points, values, /, *, d=3, axis=0)
FloaterHormannInterpolator.__call__(z, /)
FloaterHormannInterpolator.poles()
FloaterHormannInterpolator.residues()
FloaterHormannInterpolator.roots()
FloaterHormannInterpolator.weights
AAA(x, y, /, *, rtol=None, max_terms=100, clean_up=True, clean_up_tol=1e-13)
AAA.__call__(z, /)
AAA.poles()
AAA.residues()
AAA.roots()
AAA.clean_up(*, clean_up_tol=1e-13)
AAA.support_points
AAA.support_values
AAA.weights
AAA.errors
AAA.converged
```
- **`BarycentricInterpolator`**: 둘째 무게중심 공식이다. 노드 `xi`는 1차원, 유한, 서로 다른 실수이고 순서는 상관없다(1개도 된다). 무게는 입력 순서로 곱하되 곱마다 지수를 따로 세어 넘침이 없다(IP21; `[scipy와 다름]`: `rng`가 없다. scipy는 점을 무작위로 섞으므로 같은 입력이 `rng`마다 끝자리가 다르다). `wi`는 참 무게의 공통 배수로 가장 큰 크기가 [0.5, 1)이다(scipy의 무게와는 공통 배수가 다르다; 공식은 그 배수와 무관하다, IP66). `wi=`를 주면 그것을 쓴다. 노드에서는 그 값 그대로다. `derivative(x, der=k)`는 k차 도함수(k = 0은 값), `derivatives(x, der=m)`은 0차부터 m − 1차까지를 첫 축에 쌓은 배열이다(`der=None`이면 노드 수만큼). 노드 수 이상의 차수는 0이고 NaN 점에서는 NaN이다(IP64). `add_xi`, `set_yi`는 객체를 고친다(scipy와 같다): `add_xi` 뒤의 무게는 모든 노드로 새로 만든 객체와 비트까지 같다(IP62; `[scipy와 다름]`: scipy는 있던 무게를 고친다). `yi`는 값의 원래 모양이다(IP66; scipy는 내부의 (n, r) 꼴). 값 없이 만든 객체(`yi=None`)를 부르면 `ValueError`다. 값은 IEEE대로 퍼뜨린다(NaN 값은 그것이 들어가는 곳을 NaN으로).
- **`KroghInterpolator`**: Newton 꼴의 Hermite 보간이다. 같은 점의 반복은 도함수 자료다(`xi=[0, 0, 1]`, `yi=[f(0), f′(0), f(1)]`; scipy와 같다). 같은 값의 노드는 붙어 있어야 한다(`ValueError`). 값은 Horner, 도함수는 중심을 x로 옮기는 Horner를 거듭해 계산하고(IP69), `__call__`, `derivative(der=m)`, `derivatives()`의 m번 행이 비트까지 같다. 점이 많고 고르면 불안정하다(30점 넘는 고른 점에서 오차가 커진다). 경고는 없다(IP22): 정확도 계약(§9)이 그 크기를 보인다.
- **`FloaterHormannInterpolator`** (IP9): 혼합 차수 `d`(기본 3)의 Floater–Hormann 유리식이다. `points`는 서로 다른 유한한 실수이고 어느 순서든 scipy처럼 정렬해서 쓴다. 값이 유한하지 않은 표본은 scipy처럼 버리고, `d`는 남은 표본 수 n에 대해 0 ≤ d < n이어야 한다(IP63; `[scipy와 다름]`: scipy는 같은 점을 말없이 하나만 남긴다). 무게는 scipy의 닫힌 식이고 같은 연산 순서라 scipy와 같은 비트다. 평가는 `Barycentric`과 같은 무게중심 꼴이고, `z`는 실수나 복소수다. 무한한 `z`는 극한 Σwf/Σw다(scipy와 같다; 배치면 열마다). 정확한 산술에서 Σw가 0인 자료(고른 노드 11개와 d = 3처럼 분모의 최고차 계수가 0)에서는 이 극한이 없고, 계산한 Σw는 반올림이 남긴 작은 수다. 그때 무한한 `z`의 값과 `poles()`·`roots()`의 크기가 아주 큰 것(Σw를 0으로 볼 때 무한대로 가는 고윳값)은 반올림의 결과라 뜻이 없다(scipy도 같은 식이다). |Σw|가 Σ\|w\|의 몇 ε 안이면 그렇게 읽는다. 결과의 모양은 다른 1차원 보간처럼 `z`의 모양이 `axis` 자리에 들어간다(IP65; `[scipy와 다름]`: scipy는 `z`의 첫 축만 옮긴다). `poles()`, `roots()`, `residues()`는 AAA와 같은 계산이고 뒤의 둘은 1차원 값만 받는다(배치는 `ValueError`; scipy는 `NotImplementedError`).
- **`AAA`**: scipy의 AAA(Nakatsukasa–Sète–Trefethen)를 단계마다 그대로 한다. 점과 값은 1차원의 실수나 복소수이고, 값이 유한하지 않은 표본은 버리고 점을 정렬한다(같은 점은 처음 표본; scipy와 같다). 점은 유한해야 하고 유한한 표본이 하나는 있어야 한다(`ValueError`). `rtol` 기본은 ε^(3/4)이고 `rtol`과 `clean_up_tol`은 유한하고 0 이상이어야 한다(IP67; scipy는 검사하지 않는다). 지지점 고르기의 동점은 앞의 색인이다(scipy의 `argmax`와 같다). 같은 자료에서 scipy와 같은 지지점을 같은 순서로 고른다(시험 자료 넷 모두; 오차가 반올림 크기로 내려간 뒤의 걸음만 반올림에 따라 갈릴 수 있다). max\|f − r\| ≤ rtol·max\|f\|이면 멈추고 `converged`가 `True`, min(`max_terms`, 표본 수) 걸음에 이르면 `False`다(IP20; scipy는 경고). `clean_up()`은 Froissart 쌍을 지우고 다시 맞춘 뒤 찾은 수를 돌려준다(경고 없음). 인자 이름은 생성자와 같은 `clean_up_tol`이다(IP32; `[scipy와 다름]`: scipy의 메서드는 `cleanup_tol`). 지지점에서는 그 값 그대로다. `support_points`는 고른 순서다. `poles()`, `roots()`, `residues()`는 complex128이다(scipy와 같다). 실수 자료의 `support_points`, `support_values`, `weights`는 float64다.
- **함수 꼴**: `barycentric_interpolate(xi, yi, x, der=)`와 `krogh_interpolate(xi, yi, x, der=)`는 객체를 만들어 `der`가 0이면 `__call__`, 그 밖이면 `derivative`(시퀀스면 각 차수의 list)를 부른 것과 같은 비트다.

## §6. 격자
```
RegularGridInterpolator(points, values, /, *, method='linear', bounds_error=True, fill_value=nan)
RegularGridInterpolator.__call__(xi, /, *, method=None, nu=None)
RegularGridInterpolator.grid
RegularGridInterpolator.values
RegularGridInterpolator.method
RegularGridInterpolator.bounds_error
RegularGridInterpolator.fill_value
interpn(points, values, xi, /, *, method='linear', bounds_error=True, fill_value=nan)
```
- `points`는 축마다의 1차원 격자(엄격히 증가하거나 감소), `values`는 (n₁, …, n_d, …)이고 뒤의 축은 배치다. `xi`는 (…, d) 배열이거나 d개 배열의 튜플이다(scipy와 같다). 결과는 (…, 배치)다.
- **방법**: `linear`(축마다 선형 커널을 마지막 축부터 차례로), `nearest`(t ≤ 0.5면 왼쪽 노드; 가운데에서는 작은 색인, scipy와 같다), `slinear`(k = 1의 텐서곱; `linear`와 같은 비트), `cubic`, `quintic`(축마다 not-a-knot 스플라인의 텐서곱), `pchip`(축마다 PCHIP을 마지막 축부터; scipy와 같은 함수). 축마다 필요한 점의 수는 scipy와 같다(`linear`·`nearest`는 1점, `slinear` 2점, `cubic`·`pchip` 4점, `quintic` 6점).
- **`cubic`·`quintic`의 계수**: 축마다의 직접 풀이다(IP23). `[scipy와 다름]`: `solver`, `solver_args`가 없다. scipy는 계수를 반복 풀이(기본 `gcrotmk`)로 구해 결과가 그 허용오차만큼 틀린다(2차식 자료 6 × 6의 참값 25가 24.999999999999986, 표준정규 값의 6 × 7 격자에서 참값과 3×10⁻⁵).
- **`nu`** (IP30): 축마다의 미분 차수(길이 d의 int 시퀀스)다. `slinear`, `cubic`, `quintic`에서만 받는다(그 밖은 `ValueError`; scipy와 같다).
- **바깥**: `bounds_error=True`면 범위 밖의 점이 `ValueError`, `False`면 `fill_value`(기본 NaN)다. `fill_value=None`이면 끝 조각을 늘인다(선형의 늘이기는 k = 1 B-스플라인과 같은 비트). `fill_value`는 `values`와 맞는 형의 수이거나 `None`이다(실수 값에 complex는 `ValueError`; scipy와 같다).
- **NaN 평가점** (IP24): `bounds_error`와 무관하게 그 점이 NaN이다(`[scipy와 다름]`: scipy는 `bounds_error=True`면 `ValueError`(범위 밖으로 친다), `False`면 `fill_value`가 아니라 NaN이다).
- **축** (IP73): `points`는 1차원 배열의 튜플이나 리스트다. 축마다 1점 이상, 유한해야 한다(`[scipy와 다름]`: scipy는 무한한 노드를 받는다). 감소하는 축은 만들 때 뒤집고, 속성 `grid`·`values`는 뒤집은(오름차순) 것이다(scipy와 같다). 점이 하나인 축은 `linear`·`nearest`에서 그 축을 따라 상수다.
- **값** (IP74): float64 또는 complex128이고 뒤의 축은 배치다. `cubic`·`quintic`·`pchip`은 값이 유한해야 한다(`ValueError`; scipy도 에러). `linear`·`nearest`·`slinear`는 IEEE로 퍼뜨리되 1차원 선형 길의 규칙이라 NaN·inf 노드는 그 점의 무게가 0이 아닐 때만 결과에 든다(`[scipy와 다름]`: scipy의 `linear`는 무게가 0인 NaN도 칸 전체에 퍼뜨리고 `slinear`는 에러다).
- **옵션** (IP75): `bounds_error`는 `True`·`False`, `fill_value`는 `None`이거나 수(bool과 배열은 `TypeError`), `nu`는 d개의 0 이상 int의 시퀀스(스칼라와 float는 `TypeError`)다. 속성은 읽기만 한다. `__call__(method=...)`로 다른 방법을 부르면 그 방법의 검사를 그때 하고, `cubic`·`quintic`의 계수는 차수마다 한 번 만들어 둔다(IP78).
- **평가점** (IP76): (…, d) 배열, d의 배수 길이인 1차원 배열((−1, d)로 읽는다), d개 좌표의 튜플(스칼라와 배열을 브로드캐스트)이다. 스칼라만의 튜플은 한 점이라 결과가 Python 스칼라다. 범위 밖의 점은 `ValueError`(`one of the requested xi is out of bounds in dimension i`, 가장 작은 축)다. `fill_value=None`에서 무한한 좌표는 끝 조각의 직선 값이다(`linear`; `[scipy와 다름]`: scipy는 NaN).
- **같은 비트**: 1차원에서 `linear`·`slinear`는 `sd.interp`(안)와 k = 1 `make_interp_spline`(바깥), `cubic`·`quintic`은 `make_interp_spline(k=3, 5)`, `pchip`은 `PchipInterpolator`와 비트까지 같다. d차원 `pchip`은 scipy의 알고리즘(축마다 1차원 PCHIP을 마지막 축부터 접기)과 같은 함수이고, 평가점마다 이웃 4ᵈ개 값만 읽는다(IP80).
- `interpn(points, values, xi, ...)`은 `RegularGridInterpolator(points, values, ...)(xi)`와 같다. `method='splinef2d'`는 없다.

## §7. 흩어진 점
```
RBFInterpolator(y, d, /, *, neighbors=None, smoothing=0.0, kernel='thin_plate_spline', epsilon=None, degree=None)
RBFInterpolator.__call__(x, /)
NearestNDInterpolator(x, y, /, *, rescale=False)
NearestNDInterpolator.__call__(xi, /)
```
- **`RBFInterpolator`**: `y`는 (n, d) 자료점(유한, 1점 이상), `d`는 (n, …) 값(float64나 complex128, 유한)이다. 핵은 scipy의 여덟이다: `linear`(−r), `thin_plate_spline`(r² log r), `cubic`(r³), `quintic`(−r⁵), `multiquadric`(−√(1 + r²)), `inverse_multiquadric`(1/√(1 + r²)), `inverse_quadratic`(1/(1 + r²)), `gaussian`(exp(−r²)), r = `epsilon`·거리(거리는 `sd.spatial`의 `euclidean`과 같은 비트; IP91). 척도 불변인 앞의 넷은 `epsilon` 기본 1, 뒤의 넷은 `epsilon`이 있어야 한다(`ValueError`; scipy와 같다). `epsilon`은 유한한 양수다(`[scipy와 다름]`: scipy는 0과 음수를 받는다).
- **`degree`**: 더하는 다항식의 차수다. 기본은 핵의 최소 차수(`multiquadric`·`linear` 0, `thin_plate_spline`·`cubic` 1, `quintic` 2)와 0 가운데 큰 것이다(scipy와 같다). 핵의 최소보다 낮으면 −1(다항식 없음)을 포함해 `ValueError`다(IP25, IP29; `[scipy와 다름]`: scipy는 0 이상의 낮은 값에 경고하고 −1은 말없이 받는다). 단항식이 자료점(이웃)보다 많으면 `ValueError`다(scipy와 같다).
- **`smoothing`**: 수나 (n,) 배열, 유한하고 0 이상이다(IP25; `[scipy와 다름]`: scipy는 음수를 받는다). 계의 대각에 더한다.
- **풀리지 않는 계**: 같은 자료점이 둘 이상 있고 그 `smoothing`이 0이면 만들 때 `LinAlgError`다(두 점의 색인을 말한다; IP90; `[scipy와 다름]`: scipy는 에러 없이 값을 낸다). 단항식 행렬의 랭크가 모자라면(직선 위의 점에 `degree=1`) 풀기 전에 `LinAlgError`이고 그 랭크를 말한다(IP92; scipy는 이 경우 10¹¹ 크기의 값을 말없이 낸다). 그 밖에 특이한 계도 `LinAlgError`다. 풀이는 Bunch–Kaufman LDLᵀ(scipy의 `dsysv`와 같은 분해)다.
- **평가점** (IP93, IP94): `x`는 (…, d) 배열이고 결과는 (…) + `d`의 나머지 모양이다(`[scipy와 다름]`: scipy는 (Q, d)만 받는다). 한 점에 스칼라 값이면 Python 스칼라다. NaN 좌표의 점은 NaN이고, 무한한 좌표는 전역 RBF가 그 식을 계산한다(scipy와 같다).
- **`neighbors=k`**: 평가점마다 가장 가까운 k개 자료점의 국소 RBF다(k > n이면 n). 이웃은 `sd.spatial.KDTree`의 (거리, 색인) 순서라 같은 거리의 이웃은 작은 색인이 이기고, 한 점의 값은 그 이웃 집합의 자료점으로 만든 `RBFInterpolator`와 비트까지 같다: 다른 평가점, 점의 순서, 스레드 수와 무관하다(IP96). 같은 이웃 집합의 평가점은 한 번 푼 계수를 함께 쓴다(scipy와 같다). k ≥ n이면 만들 때 전역 계를 풀고 `neighbors=None`과 같은 비트다(IP95). 유한하지 않은 평가점은 NaN이다(`[scipy와 다름]`: scipy는 `ValueError`). 국소 계가 풀리지 않으면 부를 때 `LinAlgError`이고 점 번호를 말한다.
- **`NearestNDInterpolator`** (IP8, IP31, IP98): 평가점마다 가장 가까운 자료점의 값이다(`sd.spatial.KDTree.query(k=1)`; 동점은 작은 색인, `[scipy와 다름]`: scipy의 트리는 동점 가운데 하나를 정해 두지 않고 고른다). `x`는 (n, d) 자료점(d = 1도 된다; scipy는 둘 이상), `y`는 (n, …) 값(float64나 complex128, 유한하지 않아도 된다)이다. `rescale=True`면 축마다 자료점의 평균을 빼고 범위(0이면 1)로 나눈 좌표에서 찾는다(scipy와 같다). `xi`는 (…, d) 배열이거나 d개 배열의 튜플(브로드캐스트한다)이다. 유한하지 않은 평가점은 NaN이다(scipy는 `ValueError`). `tree_options`와 `__call__`의 질의 옵션은 없다(`[scipy와 다름]`).

## §8. scipy·numpy → sd.interpolate

**numpy**
| numpy | sd | 절 |
|---|---|---|
| `numpy.interp(x, xp, fp, left=None, right=None, period=None)` | `sd.interp(x, xp, fp, /, *, left=None, right=None, period=None)`(배열 영역) | §2, array.md §8.5 |

**scipy.interpolate**
| scipy | sd.interpolate | 절 |
|---|---|---|
| `PPoly`, `CubicHermiteSpline`, `CubicSpline`, `PchipInterpolator`, `pchip_interpolate`, `Akima1DInterpolator` | 같은 이름 | §3 |
| `BSpline`, `make_interp_spline` | 같은 이름 | §4 |
| `BarycentricInterpolator`, `barycentric_interpolate`, `KroghInterpolator`, `krogh_interpolate`, `FloaterHormannInterpolator`, `AAA` | 같은 이름 | §5 |
| `RegularGridInterpolator`, `interpn` | 같은 이름(방법 `linear nearest slinear cubic quintic pchip`) | §6 |
| `RBFInterpolator`, `NearestNDInterpolator` | 같은 이름 | §7 |
| `BPoly`, `NdBSpline`, `NdPPoly` | 없음(나중 후보, IP10) | — |
| `make_lsq_spline`, `make_smoothing_spline`, `make_splrep`, `make_splprep`, `generate_knots`, `UnivariateSpline`, `LSQUnivariateSpline`, `SmoothBivariateSpline`, `LSQBivariateSpline`, `SmoothSphereBivariateSpline`, `LSQSphereBivariateSpline`, `BivariateSpline`, `splprep` | 없음: 맞추기와 매끄럽게 하기는 나중의 `sd.smooth`(IP1) | — |
| `griddata`, `LinearNDInterpolator`, `CloughTocher2DInterpolator` | 없음: Qhull(IP1). 가장 가까운 점은 `NearestNDInterpolator` | — |
| `construct_fast`, `PPoly.from_bernstein_basis` | 없음(IP12; 뒤의 것은 `BPoly`가 있어야 한다) | — |

**뺀 옛 이름의 바꿔 쓰기** (IP7)
| scipy | sd.interpolate |
|---|---|
| `interp1d(x, y)`(`kind='linear'`) | `sd.interp(x_new, x, y)`(끝값으로 자름) 또는 `make_interp_spline(x, y, k=1)`(늘임). `bounds_error`·`fill_value`는 `extrapolate=False`의 NaN이나 `sd.interp`의 `left`·`right` |
| `interp1d(kind='nearest' / 'previous' / 'next')` | `make_interp_spline(x, y, k=0)`이 `'previous'`(마지막 노드는 y[-1])다. `'next'`와 `'nearest'`는 `sd.searchsorted(x, x_new, side=...)`로 고른 색인의 `y` |
| `interp1d(kind='zero' / 'slinear' / 'quadratic' / 'cubic')`, `kind=k`(정수) | `make_interp_spline(x, y, k=0 / 1 / 2 / 3)`, `k=k` |
| `interp2d` | 격자 자료는 `RegularGridInterpolator((x, y), z, method=...)`, 흩어진 자료는 `RBFInterpolator` |
| `Rbf` | `RBFInterpolator(y, d, kernel=..., epsilon=...)`(핵의 이름과 `epsilon`의 뜻이 다르다: `Rbf`의 `epsilon`은 나누는 수) |
| `InterpolatedUnivariateSpline(x, y, k=3)` | `make_interp_spline(x, y, k=3)`(매듭은 not-a-knot) |
| `RectBivariateSpline(x, y, z)`(s = 0) | `RegularGridInterpolator((x, y), z, method='cubic')` |
| `RectSphereBivariateSpline` | 없음(구 위의 격자는 나중 후보) |
| `splrep(x, y, s=0)`, `splev(x, tck)` | `make_interp_spline(x, y)`, `BSpline(*tck)(x)` 또는 `make_interp_spline(...)(x)` |
| `splint(a, b, tck)`, `sproot(tck)` | `BSpline.integrate(a, b)`, `PPoly.from_spline(tck).roots()` |
| `spalde(x, tck)`, `splder(tck, n)`, `splantider(tck, n)` | 차수마다 `BSpline(*tck)(x, nu=k)`, `BSpline.derivative(nu=n)`, `BSpline.antiderivative(nu=n)` |
| `insert(x, tck, m)` | `BSpline.insert_knot(x, m=m)` |
| `bisplrep`, `bisplev` | 격자 자료는 `RegularGridInterpolator(method='cubic')`, 흩어진 자료는 `RBFInterpolator` |
| `lagrange(x, w)` | `BarycentricInterpolator(x, w)`(같은 다항식을 안정하게 계산한다) |
| `approximate_taylor_polynomial(f, x, degree, scale)` | `sd.differentiate.derivative`로 도함수를 구해 다항식을 만든다 |
| `pade(an, m)` | 없음(유리 근사는 `AAA`) |
| RGI의 `method='slinear_legacy' / 'cubic_legacy' / 'quintic_legacy'`, `interpn(method='splinef2d')` | `method='slinear' / 'cubic' / 'quintic'`(직접 풀이라 옛 방법과 같은 함수를 반복 풀이의 오차 없이 낸다) |

## §9. 정확도
- 참값은 정확한 유리수와 Arb다(tests/interpolate/ref, tools/interpolate_ref.py). 계약은 두 겹이다: **만들기**는 계산한 계수(기울기, B-스플라인 계수, 무게)를 같은 자료의 정확한 해와, **평가**는 계산한 계수가 나타내는 함수를 그 계수 그대로 정확히 평가한 값과 견준다(docs/design/interpolate.md §4).
- **선형**(`sd.interp`, k = 1, 격자 선형의 1차원)의 계약(증명은 docs/design/interpolate.md §3.2): 정확한 t와 y = a + t(b − a)에 대해 0 ≤ t ≤ 1이면 |ŷ − y| ≤ γ₆(|a| + |b|) + 3η, 늘이기에서는 γ₁₀|t|(|a| + |b|) + 3η다(γ_k = ku/(1 − ku), u = 2⁻⁵³, η = 2⁻¹⁰⁷⁵). 노드에서 0 오차, 단조인 자료에서 단조, 구간 끝값 사이는 정확한 성질이다. numpy와 R의 기록도 시험 자료에서 이 한계 안이다(numpy는 간격이 비정상 수인 자료에서 기울기가 넘쳐 inf를 낸다).
- **B-스플라인** (증명은 docs/design/interpolate.md §3.4): 평가는 |ŷ − Σⱼ cⱼBⱼ^(ν)(x)| ≤ γ_{6k+1}·Σⱼ|cⱼ|B̃ⱼ(x) + (6k + 2)·2⁻¹⁰⁷⁴·(1 + Σⱼ|cⱼ|)(c는 계산한 계수, B̃는 항을 절댓값으로 한 재귀이고 기본 구간의 ν = 0에서 B 자체; k = 1의 ν = 0은 선형 계약), `make_interp_spline`의 계수는 성분마다 |ĉ − c| ≤ 2γ_{8k+8}|A⁻¹||A||ĉ|(A는 정확한 자료의 배치 행렬)다. 시험 곡선에서 평가는 최대 0.22, 계수는 최대 0.032 계약 단위다. scipy의 계수는 같은 한계로 natural에서 4.3×10¹³이다(끝점의 행 e₀를 부분 피벗이 섞는다).
- **다항식·유리식** (증명은 docs/design/interpolate.md §3.6, N은 노드 수): `Barycentric`은 무게의 반올림까지 넣은 끝에서 끝으로 |p̂ − p| ≤ (γ₃ₙ₊₁Σ|ℓⱼ(x)yⱼ| + γ₃ₙΛ(x)|p(x)|)/(1 − γ₃ₙ₋₁Λ(x))(p는 float 자료의 정확한 보간 다항식, Λ는 Lebesgue 함수), `Krogh`는 차분표 |ĉₖ − cₖ| ≤ γ₃ₖ₊₂C̃ₖ(절댓값 표)와 평가 m!γ₂ₙ₊₁Ã_m(Horner 걸음의 절댓값), 유리식(Floater–Hormann, AAA)은 계산한 지지점·값·무게의 정확한 유리식 대비 (γₐΣ|λⱼfⱼ| + γ_b|r|Σ|λⱼ|)/(|Σλⱼ| − γ_cΣ|λⱼ|)(λⱼ = wⱼ/(z − zⱼ); 실수 (N + 3, N + 2, N + 1), 복소 (3N + 16, 2N + 16, 2N + 8))이고 Floater–Hormann의 무게는 상대 γ₃d 안이다. 시험 자료에서 `Barycentric` 0.064, `Krogh` 끝에서 끝 0.0039, Floater–Hormann 끝에서 끝 0.059, AAA 0.31 이하 계약 단위다. 수렴한 AAA는 표본에서 max|r(zᵢ) − fᵢ| ≤ rtol·max|f|다.
- **격자** (IP79, IP81; 근거는 docs/design/interpolate.md §4): `linear`·`slinear`는 축마다의 1차원 선형 계약을 마지막 축부터 쌓은 것이고 안쪽 점에서 |ŷ − y| ≤ d·(γ₁₃M + 3η)(M은 2ᵈ 꼭짓점 값의 최대 크기)다. `cubic`·`quintic`의 계수는 성분마다 Σ_a(⊗_{b>a}|A_b⁻¹|)·2γ_{8k+8}|A_a⁻¹||A_a||R̂_a|(R̂_a는 축 a까지 푼 계수), 평가는 γ_{d(6k+1)}Σⱼ|cⱼ|∏B̃ + d(6k + 2)·2⁻¹⁰⁷⁴·(1 + 2ᵈΣⱼ|cⱼ|)다. 시험 격자에서 선형 0.34, 계수 0.016, 평가 0.11, 끝에서 끝 0.012(`cubic`)·0.0045(`quintic`) 계약 단위이고, scipy의 반복 풀이는 같은 끝에서 끝 단위로 10⁸ 크기다. `pchip`의 끝에서 끝 상대오차는 3.1×10⁻¹⁶이다.
- **RBF** (IP97; 근거는 docs/design/interpolate.md §4): 풀이는 성분마다 |Âĉ − d| ≤ γ_{4N}(|Â| + |L̂||D̂||L̂|ᵀ)|ĉ|(Â는 계산한 계, L̂·D̂는 그 LDLᵀ 인수, N = n + q), 평가는 계산한 계수의 정확한 함수 대비 핵 식과 합의 반올림(거리와 ε의 곱이 상대 γ_{d+5}, libm의 `log`·`exp`가 1 ULP라는 가정)이다. 시험 자료에서 풀이 0.0099, 평가 0.056 계약 단위 이하(scipy 0.010, 0.11)다. 끝에서 끝의 오차는 계의 조건수가 정한다(linear 5×10⁻¹⁶에서 ε = 3의 gaussian 2×10⁻⁷, scipy와 같은 크기). `neighbors=`의 값은 이웃 집합의 전역 RBF와 같은 비트다.
- 반올림 크기의 결과는 scipy·numpy·R과 끝자리가 다를 수 있다(API.md §4의 규칙 3).
