# stardust

배열, 선형대수, 데이터프레임을 처음부터 다시 만드는 교육용 라이브러리다 ("rebuilding the wheel").
- 계산 코어는 C++20, Python 바인딩은 nanobind로 만든다.
- 대상 기기는 Apple M4 Pro 한 대다. NEON, Accelerate(BLAS, LAPACK, vForce)를 쓰고, 스레드풀은 직접 만들었다.
- 영역 사이의 일관성을 가장 앞에 둔다: 같은 개념은 어느 영역에서든 같은 이름, 같은 시그니처, 같은 동작이다.

```python
import stardust as sd

x = sd.arange(12.0).reshape((3, 4))
y = sd.sin(x.T) @ x            # vForce, BLAS
x[x > 5] = 0.0                  # boolean mask scatter
sd.sum(x, axis=0), x.argmax(axis=1), sd.sort(x, descending=True)
```

## 영역
| 영역 | 이름공간 | 상태 | API |
|---|---|---|---|
| 배열 | `sd` | 옮김 (M1) | [docs/api/array.md](docs/api/array.md) |
| 선형대수 | `sd.linalg` | `legacy/linalg`에서 옮길 예정 (M2–M3) | |
| 데이터프레임 | `sd.frame` | `legacy/pandas`에서 옮길 예정 (M5–M6) | |

세 교육용 프로젝트(stardust-numpy, stardust-linalg, stardust-pandas)를 하나로 합치는 중이다. 진행 순서는 [ROADMAP.md](ROADMAP.md), 결정은 [DESIGN.md](DESIGN.md), 구조는 [ARCHITECTURE.md](ARCHITECTURE.md)에 있다.

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

통합 전후의 결과가 비트까지 같은지는 골든 출력으로 확인한다 (`bench/merge/golden/golden.py`, [docs/learn/M0-repository-history-baseline.md](docs/learn/M0-repository-history-baseline.md)).
