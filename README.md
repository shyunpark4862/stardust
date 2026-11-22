# stardust

배열, 선형대수, 데이터프레임, 특수함수, 통계, 회귀, 난수, 최적화를 처음부터 다시 만드는 라이브러리다 ("rebuilding the wheel").
- 계산 코어는 C++20, Python 바인딩은 nanobind로 만든다. 확장 모듈은 `stardust._core` 하나다.
- 대상 기기는 Apple M4 Pro 한 대다. NEON, Accelerate(BLAS, LAPACK, vForce)를 쓰고, 스레드풀은 직접 만들었다.
- 영역 사이의 일관성을 가장 앞에 둔다: 같은 개념은 어느 영역에서든 같은 이름, 같은 시그니처, 같은 동작이다.
- 결과는 sd의 스레드 수와 무관하게 비트까지 같다(BLAS·LAPACK을 부르는 계산은 Accelerate의 스레드 모드가 같을 때).
- numpy, scipy, pandas가 정답지다. 반올림이 아닌 동작(dtype, 정수 연산, 정렬 순서, NaN 규칙, 기본값, 멈춤 규칙)은 정답지와 같고, 반올림 크기의 결과는 Arb 참값과 견준 정확도 계약 안에 있다(마지막 자리는 정답지와 다를 수 있다). 일부러 다르게 한 곳은 문서에 이유와 함께 적었다 (API.md §4).
- 결과 객체(회귀, 최적화, 선형대수의 namedtuple)의 배열은 읽기 전용이다. 고쳐 쓰려면 `.copy()`를 쓴다 (API.md §2.2).
- `std`/`var`의 기본은 `correction=0`이다(numpy와 같다). `sd.stats`의 `describe`와 `sem`만 scipy.stats처럼 `correction=1`이다. 표본 분산은 `correction=1`로 적는다 (API.md §6).

```python
import stardust as sd

x = sd.arange(12.0).reshape((3, 4))
y = sd.sin(x.T) @ x            # vForce, BLAS
x[x > 5] = 0.0                  # boolean mask scatter
sd.sum(x, axis=0), x.argmax(axis=1), sd.sort(x, descending=True)

a = sd.asarray([[2.0, 1.0], [1.0, 3.0]])
print(sd.linalg.solve(a, sd.asarray([1.0, 2.0])).tolist())  # [0.19999999999999996, 0.6000000000000001]
```

데이터프레임:
```python
import stardust as sd

df = sd.frame.DataFrame({"k": ["b", "a", "b", "a"], "v": [1, 2, 3, None]})
g = df.groupby("k").sum()
print(g["k"].tolist(), g["v"].tolist())  # ['a', 'b'] [2, 4]
print(df["v"].dtype is sd.int64, df["v"].isna().tolist())  # True [False, False, False, True]
```

영역을 넘어서: Series를 배열로 꺼내 `sd.linalg`에 넘긴다. NA 없는 숫자 컬럼은 복사 없이 읽기 전용 뷰가 되고, `DataFrame.to_array()`는 linalg가 바로 읽는 F-order 행렬이다.
```python
import stardust as sd

df = sd.frame.DataFrame({"x1": [1.0, 2.0, 3.0, 4.0], "x2": [0.0, 1.0, 0.0, 1.0], "y": [1.0, 3.0, 3.0, 5.0]})
y = df["y"].to_array()                       # sd.ndarray, 컬럼과 메모리를 함께 쓴다
print(sd.linalg.vector_norm(y))  # 6.6332495807108
beta = sd.linalg.lstsq(df[["x1", "x2"]].to_array(), y).solution
print([round(b, 12) for b in beta.tolist()])  # [1.0, 1.0]
```

## 영역
| 영역 | 이름공간 | 내용 | API |
|---|---|---|---|
| 배열 | `sd` | N차원 `ndarray`(dtype 넷, 임의의 strides, 0차원 없음), ufunc, 리덕션, 인덱싱, 정렬과 unique, `matmul` | [docs/api/array.md](docs/api/array.md) |
| 선형대수 | `sd.linalg` | 함수 37개: 분해, 선형 시스템, 노름, 특수 구조, `expm`. 행렬 묶음(배치)을 받고 F-order 배열을 돌려준다 | [docs/api/linalg.md](docs/api/linalg.md) |
| 데이터프레임 | `sd.frame` | `Series`, `DataFrame`, `Index`, NA, Copy-on-Write, groupby, merge, sort, `read_csv`. `to_array()`로 배열과 잇는다 | [docs/api/frame.md](docs/api/frame.md) |
| 특수함수 | `sd.special` | scipy.special의 부분집합: 로지스틱·정보, 오차함수, 감마·베타·제타, 불완전 감마·베타, 리덕션·지수적분·타원, Bessel·Airy, 초기하·Lambert W, 복소수 판(erf·erfc·erfcx·lambertw·wofz·loggamma) | [docs/api/special.md](docs/api/special.md) |
| 통계 | `sd.stats` | scipy.stats의 부분집합: 단변량 분포 31(신형 API `Normal(mu=0, sigma=1).cdf(x)`), 다변량 7, 행렬 분포 3, 기술통계 11. 검정과 상관은 아직 없다 | [docs/api/stats.md](docs/api/stats.md) |
| 회귀 | `sd.regression` | 선형(OLS·WLS·GLS)·GLM·정규화(ridge·lasso·elastic net, 교차검증)·강건·분위·비선형 최소제곱, 모형 비교·검정·진단, 변수 선택. 함수 + 결과 객체, 추론과 예측을 함께 | [docs/api/regression.md](docs/api/regression.md) |
| 난수 | `sd.random` | Philox4×64-10 계수기형 생성기(스레드 수와 무관한 결정성), Generator와 `random`·`integers`·`choice`·`permutation`·`spawn`. 분포별 표본은 `sd.stats` | [docs/api/random.md](docs/api/random.md) |
| 최적화 | `sd.optimize` | scipy.optimize의 부분집합: 수치 미분, 비선형·선형 최소제곱(`least_squares`, `lsq_linear`, `nnls`), 스칼라 근과 최솟값, 배치 스칼라 풀이, `minimize`(Nelder–Mead, BFGS, L-BFGS-B, newton-cg, trust 방법), `root` | [docs/api/optimize.md](docs/api/optimize.md) |
| 가설검정 | `sd.htest` | scipy.stats의 가설검정(마일스톤마다 공개): 평균과 분산(t 검정, 분산분석, 분산의 동질성), 적합도(카이제곱·Cressie–Read, D'Agostino, Jarque–Bera, Shapiro–Wilk, Anderson–Darling, Kolmogorov–Smirnov, Cramér–von Mises), 순위 검정(Mann–Whitney, Wilcoxon, Kruskal–Wallis, Friedman, Brunner–Munzel, Ansari–Bradley, Mood, 중앙값, Page), 표본끼리의 적합도(두 표본 Kolmogorov–Smirnov·Cramér–von Mises, k 표본 Anderson–Darling, Epps–Singleton), 상관(Pearson, Spearman, 점 이연, Kendall, 가중 τ, Somers' D, Chatterjee ξ), 분할표(카이제곱 독립성, Fisher 정확 검정(r × c 포함), Barnard, Boschloo, McNemar, Cochran Q, 오즈비, 상대위험), 비율과 이항(이항 검정, 분위수 검정, 포아송 평균의 E-test, statsmodels의 비율 z·카이제곱 검정, 비율의 신뢰구간, 두 독립 비율의 검정과 구간), 다중 비교와 보정(Tukey HSD·Games–Howell, Dunnett, statsmodels의 `multipletests`, `false_discovery_control`, p값의 결합). 읽기 전용 결과 객체, 배치, `nan_policy` | [docs/api/htest.md](docs/api/htest.md) |
| 희소 | `sd.sparse` | 만드는 중: scipy.sparse의 희소 배열(`coo_array csr_array csc_array dok_array`; 지금은 형식과 변환, 연산·곱·리덕션, 인덱싱과 만들기 함수, 직접 풀이, 반복법, 고유값과 특이값)과 scipy.sparse.linalg의 희소 선형대수(직접 풀이, 반복법, 고유값·특이값, 행렬 함수, `LinearOperator`)를 하위 모듈 없이 `sd.sparse`에 | [docs/api/sparse.md](docs/api/sparse.md) |
| 적분 | `sd.integrate` | 만드는 중: scipy.integrate의 적분(QUADPACK의 `quad`와 겹친 적분, 벡터 값·다차원 적응 구적, 배치 tanh-sinh와 급수, 고정 규칙, 표본의 구적 규칙). 지금은 결과 클래스만 | [docs/api/integrate.md](docs/api/integrate.md) |
| 미분 | `sd.differentiate` | scipy.differentiate의 배치 수치 미분: `derivative`, `jacobian`, `hessian`(C2, 승인 대기) | [docs/api/differentiate.md](docs/api/differentiate.md) |
| 공간 | `sd.spatial` | 준비 중(공개 이름은 SPT1부터): scipy.spatial의 거리(`pdist`, `cdist`, `squareform`, 거리 함수 열넷)와 `KDTree`, 전체 거리 행렬 없이 한 번에 끝나는 이웃 찾기(`kneighbors`, `radius_neighbors`). 같은 짝은 어디서든 같은 비트, 동점은 (거리, 색인) 순서, 결과는 스레드 수·계산 길과 무관 | [docs/api/spatial.md](docs/api/spatial.md) |
| 그래프 | `sd.graph` | 준비 중(공개 이름은 GR1 마일스톤부터): 인접 행렬(`sd.sparse` 배열이나 밀집 배열)로 주는 그래프의 알고리즘. scipy.sparse.csgraph 전부(순회, 연결 요소, 최단 경로, 최소 신장 트리, 최대 흐름, 매칭, 역 Cuthill–McKee, 라플라시안)와 `linear_sum_assignment`, 위상 정렬·단절점·다리·이중 연결 요소·코어 수·PageRank·매개 중심성. 결과는 정의로 정해져 스레드 수와 무관 | [docs/api/graph.md](docs/api/graph.md) |
| 보간 | `sd.interpolate` | 진행 중(IP2 마일스톤까지: 조각 다항식과 3차 스플라인·PCHIP·Akima): scipy.interpolate의 보간(조각 다항식과 3차 스플라인·PCHIP·Akima, 모든 차수의 B-스플라인, 무게중심·Krogh·Floater–Hormann·AAA, 정칙 격자, RBF와 가장 가까운 점)과 numpy의 `interp`(배열 영역의 `sd.interp`). 선형은 어느 길이든 같은 비트이고 노드에서 정확하며 단조인 자료에서 단조, 스플라인 계수는 직접 풀이, 결과는 스레드 수와 무관 | [docs/api/interpolate.md](docs/api/interpolate.md) |
| 푸리에 변환 | `sd.fft` | FT2 마일스톤까지: 1차원 복소 변환 `fft`·`ifft`(자체 혼합 기수와 Bluestein), 실수·에르미트 변환 `rfft`·`irfft`·`hfft`·`ihfft`(실수 입력의 `fft`도 실수 길)와 도우미 6개. 계획: scipy.fft의 변환 전부(복소, 실수·에르미트, 다차원, DCT·DST 형 1–4, FFTLog)와 도우미(`fftfreq`, `fftshift`, `next_fast_len`). 모든 길이를 자체 혼합 기수와 Bluestein으로 계산하고, 한 줄의 결과는 배치·레이아웃·스레드 수와 무관하다. 정확도는 Arb와 견준 노름 상대오차의 계약 | [docs/api/fft.md](docs/api/fft.md) |
| 그림 | `sd.plot` | 수학 함수의 그림(`function parametric polar implicit region contour density phase surface`: 표본·범위·극점을 알아서 정한다)과 논문에 싣는 그림(`line scatter bar errorbar histogram boxplot strip qq heatmap`, 검정 결과의 괄호 `compare`, 회귀선 `fit`, 진단 `diagnostics`). 자체 렌더러(PDF, SVG, PNG; 3D는 깊이 버퍼), 모든 글자는 LaTeX가 조판한다 |

- 모든 영역이 함께 쓰는 규칙(dtype 객체, 에러 종류와 메시지, 결과의 배열, 스레드와 결정성, 정답지와 같아야 하는 것, 받는 입력, 같은 이름의 다른 뜻)은 [API.md](API.md)에 있다.

## 문서
| 문서 | 내용 |
|---|---|
| [API.md](API.md) | 공통 규칙과 영역별 API 문서의 색인 |
| [DESIGN.md](DESIGN.md) | 통합의 결정과 일관성 규칙 |
| [ARCHITECTURE.md](ARCHITECTURE.md) | 층 구조, 디렉터리, 공통 부품 |
| `docs/design/` | 영역의 설계: [배열](docs/design/array.md), [선형대수](docs/design/linalg.md), [데이터프레임](docs/design/frame.md), [특수함수](docs/design/special.md), [통계](docs/design/stats.md), [회귀](docs/design/regression.md), [난수](docs/design/random.md), [최적화](docs/design/optimize.md), [가설검정](docs/design/htest.md), [희소](docs/design/sparse.md), [수치 미적분](docs/design/integrate.md), [공간](docs/design/spatial.md), [그래프](docs/design/graph.md), [보간](docs/design/interpolate.md), [푸리에 변환](docs/design/fft.md) |
| [ROADMAP.md](ROADMAP.md) | 진행 순서 |
| [tests/README.md](tests/README.md) | 테스트, 골든, 커버리지의 실행 |
| [bench/README.md](bench/README.md) | 속도와 정확도 측정 |

## 빌드와 테스트
macOS (Apple Silicon), Python 3.15에서 확인했다.
```bash
python3.15 -m venv .venv && .venv/bin/pip install nanobind scikit-build-core
```
```bash
.venv/bin/pip install --no-build-isolation -e ".[test]"
```
```bash
.venv/bin/python tools/test.py
```
- `test` 묶음(pyproject.toml)은 테스트가 쓰는 정답지와 도구다: pytest, pytest-xdist, pytest-cov, hypothesis, numpy, scipy, pandas, polars, python-flint(Arb 참값), statsmodels, scikit-learn.
- 속도·정확도 측정은 `bench` 묶음(`".[bench]"`: `test`와 tqdm, pyarrow, threadpoolctl)을 더 쓴다 (bench/README.md).

문서와 통합의 검사:
- `.venv/bin/python tools/check_refs.py`: 코드, 테스트, 문서의 `docs/design/array.md §6.2` 같은 참조가 모두 있는 절을 가리키는지 본다.
- `.venv/bin/python tools/golden.py check` (영역 이름을 주면 그 영역만): 공개 API의 결과를 저장한 골든 출력과 스레드 수 1, 3, 10에서 비트까지 비교한다. 케이스 수는 `tools/golden.py list`가 센다(2026-11-20: 8영역 24,422개; tests/README.md §2).
