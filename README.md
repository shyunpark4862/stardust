# stardust

배열, 선형대수, 데이터프레임을 처음부터 다시 만드는 교육용 라이브러리다 ("rebuilding the wheel").
- 계산 코어는 C++20, Python 바인딩은 nanobind로 만든다. 확장 모듈은 `stardust._core` 하나다.
- 대상 기기는 Apple M4 Pro 한 대다. NEON, Accelerate(BLAS, LAPACK, vForce)를 쓰고, 스레드풀은 직접 만들었다.
- 영역 사이의 일관성을 가장 앞에 둔다: 같은 개념은 어느 영역에서든 같은 이름, 같은 시그니처, 같은 동작이다.
- 결과는 스레드 수와 무관하게 비트까지 같다. numpy, scipy, pandas가 정답지이고, 다른 곳은 문서에 이유와 함께 적었다.

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
| 특수함수 | `sd.special` | **S0–S11 완료 (로지스틱·정보 7, 오차함수 8, 감마·베타·제타 9, 불완전 감마·베타 8, 리덕션·지수적분·타원 7, Bessel·Airy 21, 초기하·Lambert W 3, 복소수 2와 erf·erfc·erfcx·lambertw의 complex)** ([ROADMAP.md](ROADMAP.md) S0–S11). scipy.special의 부분집합: 오차함수, 감마, 불완전 감마·베타, Bessel, 초기하 등 | [docs/api/special.md](docs/api/special.md) |
| 통계 | `sd.stats`, `sd.stats.tests` | **설계 초안 (2026-11-15), 구현 전.** scipy.stats의 부분집합: 단변량 분포 31(신형 API `Normal(mu, sigma).cdf(x)`), 다변량 7, 기술통계 11. 검정과 상관은 `sd.stats.tests` | [docs/api/stats.md](docs/api/stats.md) |
| 난수 | `sd.random` | **설계 초안 (2026-11-15), 구현 전.** 계수기형 생성기(스레드 수와 무관한 결정성), Generator 객체와 기본 도구. 분포별 표본은 `sd.stats` | [docs/api/random.md](docs/api/random.md) |

- 세 영역이 함께 쓰는 규칙(dtype 객체, 에러 종류와 메시지, 스레드)은 [API.md](API.md)에 있다.
- 세 교육용 프로젝트(stardust-numpy, stardust-linalg, stardust-pandas)를 합쳐 만들었다. 옛 저장소는 동결했다.

## 문서
| 문서 | 내용 |
|---|---|
| [API.md](API.md) | 공통 규칙과 영역별 API 문서의 색인 |
| [DESIGN.md](DESIGN.md) | 통합의 결정과 일관성 규칙 |
| [ARCHITECTURE.md](ARCHITECTURE.md) | 층 구조, 디렉터리, 공통 부품 |
| `docs/design/` | 영역의 설계: [배열](docs/design/array.md), [선형대수](docs/design/linalg.md), [데이터프레임](docs/design/frame.md), [특수함수](docs/design/special.md) |
| [ROADMAP.md](ROADMAP.md) | 진행 순서와 마일스톤. 안내서와 마일스톤 해설은 2026-11-15에 지웠다(나중에 한 번에 다시 쓴다) |

## 빌드와 테스트
macOS (Apple Silicon), Python 3.15에서 확인했다.
```bash
python3.15 -m venv .venv && .venv/bin/pip install nanobind scikit-build-core pytest hypothesis numpy scipy pandas polars
```
```bash
.venv/bin/pip install --no-build-isolation -e .
```
```bash
.venv/bin/python -m pytest -q
```

문서와 통합의 검사:
- `.venv/bin/python tools/check_refs.py`: 코드, 테스트, 문서의 `docs/design/array.md §6.2` 같은 참조가 모두 있는 절을 가리키는지 본다.
- `.venv/bin/python bench/merge/golden/golden.py check array` (`linalg`, `frame`도): 옛 패키지로 기록한 골든 출력 12,424개와 비트까지 비교한다. 일부러 바꾼 결과는 `bench/merge/golden/expected.py`에 이유와 함께 적혀 있다.
