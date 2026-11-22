# sd.interpolate API

> 상태: **초안** (IP2, 2026-11-23). IP2 마일스톤이 조각 다항식과 3차 계열(§3: `PPoly CubicHermiteSpline CubicSpline PchipInterpolator Akima1DInterpolator pchip_interpolate`)을 공개했다(`PPoly.from_spline`은 `BSpline`과 함께 IP3 마일스톤; IP1 마일스톤은 배열 영역의 `sd.interp`를 공개했다). 아래 시그니처는 scipy 1.18.1의 `scipy.interpolate`의 이름과 인자 이름에 stardust의 규칙과 IP1–IP35의 결정을 입힌 안이고, 마일스톤 IP1–IP6에서 하나씩 공개된다(docs/design/interpolate.md §8: `sd.interp`는 IP1 마일스톤, 조각 다항식과 3차 계열은 IP2 마일스톤, B-스플라인은 IP3 마일스톤, 다항식·유리식은 IP4 마일스톤, 격자는 IP5 마일스톤, 흩어진 점은 IP6 마일스톤).

- 설계와 결정의 근거는 [docs/design/interpolate.md](../design/interpolate.md)(§1.1 IP1–IP51), 공통 규칙은 [API.md](../../API.md)다.
- 언어 표시가 없는 코드 블록은 `tests/interpolate/test_api_surface.py`가 읽는다. 블록의 줄은 모듈의 이름(`name(...)`, 클래스는 `name(...)`의 생성자), 클래스의 메서드(`Class.method(...)`)와 속성(`Class.attr`)이다. 모듈 수준의 이름이 계획한 공개 이름 전부이고, 공개된 것은 시그니처가 블록과 같아야 한다.
- `sd.interp`는 배열 영역의 함수다(IP2). IP1 마일스톤에서 공개되었고 시그니처와 규칙은 [docs/api/array.md §8.5](array.md)에 있다(`tests/array/test_api_surface.py`가 읽는다). 이 문서의 §2는 이 영역과 함께 쓰는 것만 적는다.

## §1. 공통 규칙

### §1.1 이름과 호출 모양 (IP3, IP7, IP11, IP14)
- **이름공간은 평평하다**: 모든 공개 이름이 `sd.interpolate`에 바로 있다. scipy의 `from scipy.interpolate import CubicSpline`은 `from stardust.interpolate import CubicSpline`이다. scipy의 옛 하위 모듈(`fitpack`, `fitpack2`, `interpnd`, `polyint`, `rbf`, `ndgriddata`, `interpolate`)은 없다.
- **보간 객체는 클래스다**: 만든 뒤 여러 번 부른다(`CubicSpline(x, y)(x_new)`). 함수 꼴(`pchip_interpolate`, `barycentric_interpolate`, `krogh_interpolate`, `interpn`)은 객체를 만들어 한 번 부른 것과 같은 비트다.
- **데이터는 위치로만, 옵션은 키워드로만** 받는다. `axis`, `bc_type`, `extrapolate`, `k`(`make_interp_spline`), `method`, `kernel`, `nu`, `der`도 키워드만이다(`[scipy와 다름]`: scipy의 `CubicSpline(x, y, 0, 'natural')`, `make_interp_spline(x, y, 3)`, `PPoly(c, x, True)`는 `TypeError`다). 스플라인을 정하는 수(`BSpline(t, c, k)`의 `k`, `PPoly.solve`의 `y`)는 위치다.
- **별칭이 없다**: `kernel`, `method`, `bc_type`의 문자열은 소문자 정식 이름만 받는다(`'Gaussian'`은 `ValueError`; `[scipy와 다름]`: scipy의 RBF는 대문자를 받는다). `check_finite`(언제나 검사한다, IP19), `rng`·`random_state`(IP21), `solver`·`solver_args`(IP23), `construct_fast`(IP12), `tree_options`(IP31)는 없다. 주면 모르는 키워드의 `TypeError`다.
- 틀린 호출 모양은 부른 이름으로 시작한다(API.md §2.1): `interp(): ...`, `CubicSpline(): ...`, `integrate(): ...`; 객체를 부르는 `__call__`은 클래스 이름이다(`CubicSpline(): 'x' must be ...`).

### §1.2 입력 (IP6, IP11, IP19)
- **받는 것**: 생성자, 메서드, 함수 꼴은 API.md §5의 상위 함수 입력(`sd.ndarray`, 수의 시퀀스, numpy 배열, `sd.frame`의 `Series`)을 받는다. `sd.interp`는 배열 영역의 함수라 `sd.ndarray`와 Python 스칼라만 받는다.
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
- **같은 입력이면 스레드 수와 상관없이 같은 결과다**(API.md §3). LAPACK을 부르는 곳(RBF의 전역 풀이, AAA의 SVD와 극점)은 Accelerate의 스레드 모드가 같을 때 같다.
- **선형은 어느 길이든 같은 비트다**: `xp[0] ≤ x ≤ xp[-1]`에서 `sd.interp(x, xp, fp)`, `make_interp_spline(xp, fp, k=1)(x)`, `BSpline`의 k = 1(같은 매듭), `RegularGridInterpolator((xp,), fp, method='linear')`와 `'slinear'`, `interpn`의 1차원 선형이 비트까지 같다(`xp`에 반복이 없을 때). 늘이기에서도 k = 1과 RGI의 `fill_value=None`이 같다. 선형 커널 하나가 모든 길을 계산한다(docs/design/interpolate.md §3.2). `[scipy와 다름]`: scipy에서는 이 길들의 끝자리가 다르다(1차원 RGI `linear`와 `np.interp`는 점의 54%, `slinear`와 `linear`는 49%에서만 같았다).
- **3차 계열은 같은 기울기면 같은 계수다**(IP15): `CubicHermiteSpline(x, y, s)`와 같은 기울기의 `CubicSpline`, `PchipInterpolator`, `Akima1DInterpolator`는 계수가 비트까지 같다.
- 노드에서는 자료 그대로다: 선형, k = 0·1 B-스플라인, `Barycentric`, `AAA`의 지지점, RGI의 `linear`·`nearest`.

### §1.5 에러와 경고
- API.md §2의 종류와 꼴이다. 경고는 내지 않는다. scipy가 경고하는 곳은 이렇다.

  | scipy의 경고 | sd |
  |---|---|
  | 복소수 평가점(`ComplexWarning`) | `TypeError` |
  | RBF의 낮은 `degree`(`UserWarning`) | `ValueError` (IP25, IP29) |
  | Krogh의 30차 넘는 차수(`UserWarning`) | 없음. 결과는 계산하고 정확도 계약이 오차의 크기를 보인다(IP22) |
  | AAA의 수렴 실패(`RuntimeWarning`) | `converged` 속성이 `False`(IP20) |
  | AAA `clean_up`의 Froissart 쌍 | `clean_up()`이 지운 수를 돌려준다(경고 없음) |

- 요청한 출력(평가점 × 배치, RBF의 (n + q)² 계)이 물리 메모리보다 크면 할당 전에 `MemoryError`다(API.md §2).

### §1.6 scipy·numpy와 다른 점 (모음)
- 공통: 옵션은 키워드만, 별칭 없음, 쓰기 금지 계수(생성자가 복사한다), `check_finite` 없음, complex 평가점은 `TypeError`, 0차원 결과는 Python 스칼라, 경고 대신 에러나 상태 속성, 선형의 모든 길이 같은 비트, 3차 계열의 계수가 같은 비트.
- `sd.interp`: `xp`의 검사, 키워드만인 `left`·`right`·`period`, 유한하지 않은 `period`는 `ValueError`, 노드 사이의 단조와 범위, 끝값이 둘 다 −0이면 −0(§2).
- 1차원: k = 1 B-스플라인이 노드에서 정확, `nu < 0`은 `ValueError`(`__call__`의 `nu`는 int만, IP50), `roots`의 배치 결과는 중첩 list, `roots`가 중근과 붙은 근을 찾음(IP45), 주기 검사의 허용이 하나(IP28), Akima의 문턱은 열마다(IP33), `PPoly`의 중단점은 유한하고 엄격히 단조(IP43), `integrate`의 NaN 끝은 NaN(IP49), 넘치는 3차 계수는 `ValueError`(IP36), `PPoly(axis=)`가 음수 축을 받음(IP50).
- 다항식·유리식: `Barycentric`의 `rng` 없음, AAA의 `converged`와 경고 없는 `clean_up`, `clean_up`의 인자 이름(IP32), Krogh의 경고 없음.
- 격자: `slinear`와 `linear`가 같은 비트, `cubic`·`quintic`의 직접 풀이와 `solver` 없음, 옛 방법(`cubic_legacy`, `quintic_legacy`, `slinear_legacy`, `interpn`의 `splinef2d`) 없음, NaN 평가점은 NaN.
- RBF: 음수 `smoothing`, 같은 자료점, 핵의 최소보다 낮은 `degree`(−1 포함)가 에러, `kernel`은 소문자만.

## §2. `sd.interp` (배열 영역; IP1 마일스톤에서 공개)
- 시그니처와 규칙은 docs/api/array.md §8.5다: numpy의 뜻(IP2), `xp`의 검사(IP5), 키워드만인 `left`·`right`·`period`, 노드에서 정확하고 단조이며 구간 끝값 사이의 값, 끝값이 둘 다 −0이면 −0, NaN 평가점은 NaN(한 점의 `xp`도, IP39).
- 이 영역과 함께 쓰는 것: 선형 커널(`base/lerp.hpp`)과 구간 찾기(`base/interval.hpp`, IP37)가 하나라서, `xp[0] ≤ x ≤ xp[-1]`에서 `sd.interp(x, xp, fp)`는 이 영역의 선형 길(`make_interp_spline(xp, fp, k=1)`, k = 1 `BSpline`, 1차원 RGI `linear`·`slinear`, `interpn`)과 비트까지 같다(`xp`에 반복이 없을 때; §1.4). 정확도 계약은 docs/design/interpolate.md §4의 선형 행이다.

## §3. 조각 다항식 (IP2 마일스톤)
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
- **`CubicSpline`의 `bc_type`**: `'not-a-knot'`, `'clamped'`, `'natural'`, `'periodic'`, 또는 양 끝의 `(차수, 값)`(차수 1, 2; 값은 배치 모양)이나 그 짝이다. 끝마다 다른 조건도 받는다(`('natural', 'not-a-knot')`). `'periodic'`은 양 끝 모두여야 하고 `y[0]`과 `y[-1]`이 `allclose(rtol=1e-15, atol=1e-15)` 안이어야 한다(IP17; 메시지는 축 번호를 말한다). 점이 둘이면 직선(clamped는 기울기 0의 3차식), 셋이고 양 끝이 not-a-knot이면 포물선이다(scipy와 같다). 점이 하나면 `ValueError`다.
- **`PchipInterpolator`**: Fritsch–Butland의 가중 조화 평균과 scipy의 끝 식(세 점 한쪽 식과 그 자르기)이다. 단조인 자료에서 결과가 단조다.
- **`Akima1DInterpolator`**: `method='akima'`는 Akima(1970), `'makima'`는 수정 Akima다(scipy와 같은 식, 끝의 가상 기울기 둘씩 포함). 무게의 합이 그 열의 최댓값의 10⁻⁹배 이하면 두 기울기의 평균이다(scipy와 같은 문턱; `[scipy와 다름]`: scipy는 최댓값을 모든 열에서 하나로 잡아 한 열의 크기가 다른 열의 기울기를 바꾼다, IP33). makima는 평평한 구간(같은 값이 셋 이상)이 정확히 평평하다. akima는 그 구간의 안쪽 노드에서 기울기가 0이지만 구간 끝 노드가 Akima의 정하지 않은 경우(양쪽 두 기울기가 각각 같은 경우)면 평균 기울기라 평평하지 않을 수 있다(IP47; scipy와 같다). PCHIP과 Akima의 계수는 시험 자료 전부에서 scipy와 비트까지 같다(IP42).
- **`roots`·`solve`**: 조각마다 실근을 찾아 정렬된 1차원 배열로 돌려준다. 배치면 배치 모양의 중첩 list다(`[scipy와 다름]`: scipy는 dtype `object`의 배열이다). `discontinuity=True`면 중단점에서 부호가 바뀌는 점도 근이다. 조각이 0과 같으면 그 구간의 왼쪽 끝과 NaN을 넣는다(scipy와 같다). `extrapolate=True`면 첫 조각과 끝 조각의 늘인 다항식의 근도 넣는다(scipy와 같다). `extrapolate='periodic'`은 True로 친다(scipy와 같다). 접하는 근(중근)도 찾고 2⁻³⁰ 떨어진 두 근도 마지막 비트까지 가른다(IP45; `[scipy와 다름]`: scipy는 (x − 0.25)²(x + 1) 꼴의 중근을 놓치고, 2⁻³⁰ 떨어진 두 근을 7×10⁻⁹ 틀린다; docs/design/interpolate.md §3.5). `solve(y)`의 `y`는 유한한 수다. complex 계수는 `ValueError`다(scipy와 같다).
- **`integrate(a, b)`**: a > b면 부호가 바뀐다. `extrapolate=False`이고 a, b가 정의역 밖이면 NaN, `'periodic'`이면 온전한 주기의 수 × 한 주기의 적분 + 나머지다(scipy와 같다). a나 b가 NaN이면 NaN이다(IP49; `[scipy와 다름]`: scipy는 `ValueError`). 배치면 열마다의 배열, 아니면 Python 스칼라다.
- **`antiderivative`**: 상수는 x[0]에서 0이고 이어지게 고른다. `'periodic'` 객체의 역미분은 `extrapolate=False`다(IP48; scipy와 같다).
- **`extend(c, x)`**: 오른쪽(감소하는 중단점이면 왼쪽)에 구간을 더한다. `x`는 새 조각마다 한 점(오른쪽에 더하면 새 조각의 오른쪽 끝)이다. 차수가 다르면 낮은 쪽의 위를 0으로 채운다. 객체를 고치는 메서드다(IP12; scipy와 같다). 고친 뒤의 계수도 쓰기 금지 배열이다. 새 중단점이 끝 중단점과 같으면 `ValueError`다(IP43; scipy는 폭이 0인 조각을 만든다).
- **`from_spline(tck)`**: `BSpline`이나 `(t, c, k)` 튜플을 같은 함수의 `PPoly`로 바꾼다(IP3 마일스톤에서 `BSpline`과 함께 공개한다).
- **`pchip_interpolate(xi, yi, x, der=0)`**: `PchipInterpolator(xi, yi, axis=axis)(x, nu=der)`와 같다. `der`가 시퀀스면 각 차수의 결과 list다(scipy와 같다).

## §4. B-스플라인 (IP3 마일스톤)
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
- **`BSpline`**: 매듭 `t`(줄지 않음, 길이 n + k + 1), 계수 `c`(첫 축이 n), 차수 `k`(0 이상의 int). 검사는 scipy와 같다. 정의역 안의 구간 규칙은 t[l] ≤ x < t[l+1]이고 x ≥ t[n]은 l = n − 1이다(scipy와 같다; k = 0에서는 그 구간이 비어 있어 마지막 노드의 값이 c[-1]이다). 평가는 de Boor이고 무작위 순서의 평가점도 점마다 O(log n)이다(scipy는 앞 점의 구간에서 걸어 무작위 순서에서 점마다 O(n)이다).
- **`make_interp_spline`**: 차수 k의 보간 스플라인이다. 매듭을 주지 않으면 scipy의 규칙이다: k = 0은 t = (x₀, …, xₙ₋₁, xₙ₋₁)과 c = y(왼쪽에서 이어지는 계단, 마지막 노드에서는 y[-1]), k = 1은 t = (x₀, x₀, …, xₙ₋₁, xₙ₋₁)과 c = y, 홀수 k는 not-a-knot(안쪽 매듭 x[(k+1)/2 : −(k+1)/2]), 짝수 k는 이웃한 노드의 중점(double로 계산한 (x[i] + x[i+1])/2)에서 앞뒤 k/2개를 뺀 것, `bc_type`이 (차수, 값) 목록이거나 `'natural'`·`'clamped'`면 노드에 양 끝 k겹을 더한 것, `'periodic'`이면 scipy의 주기 매듭이다. `bc_type`은 `None`, `'not-a-knot'`, `'natural'`, `'clamped'`, `'periodic'`, 양 끝의 (차수, 값) 목록이다. 짝이 맞지 않는 조건은 scipy와 같은 `ValueError`다(`k=5, bc_type='natural'`). `'periodic'`의 검사는 `CubicSpline`과 같은 `allclose(rtol=1e-15, atol=1e-15)`다(IP28; `[scipy와 다름]`: scipy의 `make_interp_spline`은 rtol 1e-5라 1e15 둘레의 자료는 끝값이 10¹⁰ 달라도 받는다).
- **k = 1**: 계수는 `y` 그대로이고 평가는 선형 커널이라 `sd.interp`와 같은 비트다(§1.4). 노드에서 `y`를 그대로 돌려준다(`[scipy와 다름]`: scipy는 무작위 노드 50개 중 9개에서 끝자리가 달랐다).
- **`nu`**: 0 이상(scipy의 `nu=-1`은 `NotImplementedError`; sd는 `ValueError`). 차수보다 크면 0이다.
- **`design_matrix`**: (len(x), n) `sd.sparse.csr_array`다(scipy는 `scipy.sparse.csr_array`). 행마다 k + 1개의 저장 자리이고 0인 값도 저장한다(scipy와 같다).
- `PPoly.from_spline(BSpline)`과 `BSpline.from_power_basis(PPoly)`가 두 표현을 잇는다. `tck`는 (t, c, k) 튜플이다.

## §5. 다항식·유리식 (IP4 마일스톤)
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
- **`BarycentricInterpolator`**: 둘째 무게중심 공식이다. 무게는 입력 순서로 곱하되 곱마다 지수를 따로 세어 넘침이 없다(IP21; `[scipy와 다름]`: `rng`가 없다. scipy는 점을 무작위로 섞으므로 같은 입력이 `rng`마다 끝자리가 다르다). 노드에서는 그 값 그대로다. `add_xi`, `set_yi`는 객체를 고친다(scipy와 같다).
- **`KroghInterpolator`**: Newton 꼴의 Hermite 보간이다. 같은 점의 반복은 도함수 자료다(`xi=[0, 0, 1]`, `yi=[f(0), f′(0), f(1)]`; scipy와 같다). 점이 많고 고르면 불안정하다(30점 넘는 고른 점에서 오차가 커진다). 경고는 없다(IP22).
- **`FloaterHormannInterpolator`** (IP9): 혼합 차수 `d`(기본 3, 0 ≤ d < 점의 수)의 Floater–Hormann 유리식이다. 무게는 scipy의 닫힌 식, 평가는 `Barycentric`과 같은 무게중심 꼴이다. `z`는 실수나 복소수다.
- **`AAA`**: scipy의 AAA(Nakatsukasa–Sète–Trefethen)다. 점과 값은 실수나 복소수이고, 유한하지 않은 표본은 버린다. `rtol` 기본은 ε^(3/4)이다. 지지점 고르기의 동점은 앞의 색인이다(scipy의 `argmax`와 같다). `max_terms`에 이르면 멈추고 `converged`가 `False`다(IP20; scipy는 경고). `clean_up()`은 Froissart 쌍을 지우고 다시 맞춘 뒤 지운 수를 돌려준다. 인자 이름은 생성자와 같은 `clean_up_tol`이다(IP32; `[scipy와 다름]`: scipy의 메서드는 `cleanup_tol`). 지지점에서는 그 값 그대로다.
- **함수 꼴**: `barycentric_interpolate(xi, yi, x, der=)`와 `krogh_interpolate(xi, yi, x, der=)`는 객체를 만들어 `der`가 0이면 `__call__`, 그 밖이면 `derivative`(시퀀스면 각 차수의 list)를 부른 것과 같다.

## §6. 격자 (IP5 마일스톤)
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
- `interpn(points, values, xi, ...)`은 `RegularGridInterpolator(points, values, ...)(xi)`와 같다. `method='splinef2d'`는 없다.

## §7. 흩어진 점 (IP6 마일스톤)
```
RBFInterpolator(y, d, /, *, neighbors=None, smoothing=0.0, kernel='thin_plate_spline', epsilon=None, degree=None)
RBFInterpolator.__call__(x, /)
NearestNDInterpolator(x, y, /, *, rescale=False)
NearestNDInterpolator.__call__(xi, /)
```
- **`RBFInterpolator`**: `y`는 (n, d) 자료점, `d`는 (n, …) 값이다. 핵은 scipy의 여덟이다: `linear`(−r), `thin_plate_spline`(r² log r), `cubic`(r³), `quintic`(−r⁵), `multiquadric`(−√(1 + r²)), `inverse_multiquadric`(1/√(1 + r²)), `inverse_quadratic`(1/(1 + r²)), `gaussian`(exp(−r²)), r = `epsilon`·거리. 척도 불변인 앞의 넷은 `epsilon` 기본 1, 뒤의 넷은 `epsilon`이 있어야 한다(`ValueError`; scipy와 같다).
- **`degree`**: 더하는 다항식의 차수다. 기본은 핵의 최소 차수(`multiquadric`·`linear` 0, `thin_plate_spline`·`cubic` 1, `quintic` 2)와 0 가운데 큰 것이다(scipy와 같다). 핵의 최소보다 낮으면 −1(다항식 없음)을 포함해 `ValueError`다(IP25, IP29; `[scipy와 다름]`: scipy는 0 이상의 낮은 값에 경고하고 −1은 말없이 받는다).
- **`smoothing`**: 수나 (n,) 배열, 유한하고 0 이상이다(IP25; `[scipy와 다름]`: scipy는 음수를 받는다). 계의 대각에 더한다.
- **같은 자료점**: `smoothing`이 0인데 같은 자료점이 있으면 `LinAlgError`다(`[scipy와 다름]`: scipy는 에러 없이 값을 낸다). 단항식 행렬의 랭크가 모자라면(직선 위의 점에 `degree=1`) `LinAlgError`이고 그 사실을 말한다(scipy는 이 경우 10¹¹ 크기의 값을 말없이 낸다).
- **`neighbors=k`**: 평가점마다 가장 가까운 k개 자료점의 국소 RBF다(k > n이면 n). 이웃은 `sd.spatial.KDTree`의 (거리, 색인) 순서라 같은 거리의 이웃이 실행과 스레드 수와 무관하다. 같은 이웃 집합의 평가점은 한 번 푼 계수를 함께 쓴다(scipy와 같다).
- **`NearestNDInterpolator`** (IP8, IP31): 평가점마다 가장 가까운 자료점의 값이다(`sd.spatial.KDTree.query(k=1)`; 동점은 작은 색인). `rescale=True`면 자료점의 상자를 단위 상자로 늘린 좌표에서 찾는다(scipy와 같다). `xi`는 (…, d) 배열이거나 d개 배열의 튜플이다. `tree_options`와 `__call__`의 질의 옵션은 없다(`[scipy와 다름]`).

## §8. scipy·numpy → sd.interpolate

**numpy**
| numpy | sd | 마일스톤 |
|---|---|---|
| `numpy.interp(x, xp, fp, left=None, right=None, period=None)` | `sd.interp(x, xp, fp, /, *, left=None, right=None, period=None)`(배열 영역) | IP1 |

**scipy.interpolate**
| scipy | sd.interpolate | 마일스톤 |
|---|---|---|
| `PPoly`, `CubicHermiteSpline`, `CubicSpline`, `PchipInterpolator`, `pchip_interpolate`, `Akima1DInterpolator` | 같은 이름 | IP2 |
| `BSpline`, `make_interp_spline` | 같은 이름 | IP3 |
| `BarycentricInterpolator`, `barycentric_interpolate`, `KroghInterpolator`, `krogh_interpolate`, `FloaterHormannInterpolator`, `AAA` | 같은 이름 | IP4 |
| `RegularGridInterpolator`, `interpn` | 같은 이름(방법 `linear nearest slinear cubic quintic pchip`) | IP5 |
| `RBFInterpolator`, `NearestNDInterpolator` | 같은 이름 | IP6 |
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
- 다른 계약(조각 다항식의 Horner, 3차 계열과 B-스플라인의 만들기, `Barycentric`, `Krogh`, AAA, RBF, 근)은 그 마일스톤에서 증명해 이 절에 적는다. 반올림 크기의 결과는 scipy·numpy·R과 끝자리가 다를 수 있다(API.md §4의 규칙 3).
