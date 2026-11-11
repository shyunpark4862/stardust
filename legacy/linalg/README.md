# stardust-linalg

[stardust-numpy](../stardust-numpy) 위에 올린 교육용 선형대수 라이브러리다.
- 계산은 Apple Accelerate의 LAPACK에 맡기고, 감싸는 층(배치, 메모리 레이아웃, 에러 처리, 병렬화)을 직접 만든다.
- 함수의 시그니처와 동작은 모두 한 가지 규칙을 따른다. 자세한 내용은 [API.md](API.md)에 있다.
- 대상 기기는 Apple M4 Pro 한 대, Python은 3.15다.

```python
import stardust_numpy as sdnp
import stardust_linalg as sdla
```

## 상태
M11까지 마쳤다. 기본 단계와 확장 단계의 함수가 모두 들어갔고(randomized SVD는 난수 생성을 정한 뒤로 미룸), 호출 하나의 비용을 줄였고, 배치와 긴 합을 여러 코어로 나눈다.

numpy 2.5.4 / scipy 1.18.1과 같은 줄 248개를 비교하면(10스레드): 195줄이 10% 이상 빠르고, 43줄이 같고, 10줄이 느리다(M11 이후의 최적화 포함).
- 작은 행렬은 약 2.8배, 배치는 약 7배 빠르다(numpy와 scipy는 배치를 코어 하나로 돈다).
- 큰 행렬 하나의 분해는 같은 Accelerate LAPACK이라 거의 같다.
- 느린 곳은 큰 C-order 삼각 풀기(레이아웃과 무관한 비트를 지키는 전치 비용), 큰 `matrix_power`, 아주 짧은 합이다.

마일스톤별로 들어간 것:
- M1: `solve`, `inv`, `det`, `slogdet`, `LinAlgError`
- M2: `vecdot`, `outer`, `trace`, `diagonal`, `matrix_power`, `vector_norm`, `matrix_norm` (fro, ±1, ±inf)
- M3: `solve_triangular`, `cholesky`, `cholesky_solve`, `lu`, `lu_solve`
- M4: `qr`, `qr_pivoted`, `lstsq`
- M5: `eigh`, `eigvalsh`, `svd`, `svdvals`, `matrix_rank`, `pinv`, `cond`, `matrix_norm` (2, −2, nuc)
- M6: `eig`, `eigvals`, `schur`, `hessenberg`, `ldl`
- M7: `solve_tridiagonal`, `solve_banded`, `solve_sylvester`, 기준선 벤치마크 (`bench/baseline.py`)
- M8: 최적화 (workspace 재사용, 패킹, pairwise 합, 16바이트 정렬)
- M9: 배치와 긴 합의 병렬화 (스레드 수는 `sdnp.set_num_threads`를 따름, 결과는 스레드 수와 무관)
- M10: `kron`, `cross`, `expm` (`expm`은 LAPACK에 없어서 직접 구현: Al-Mohy–Higham 2009)
- M11: 마일스톤 끝 커밋을 다시 빌드해 같은 자리에서 비교한 최종 보고서, API.md와 모듈을 맞추는 테스트

```python
a = sdnp.asarray([[2.0, 1.0], [1.0, 3.0]])
sdla.solve(a, sdnp.asarray([1.0, 2.0]))   # ≈ [0.2, 0.6]
sdla.slogdet(a)                            # SlogdetResult(sign=1.0, logabsdet=1.609...)
```

## 문서
- **처음 본다면**: [안내서](docs/guide/README.md). 구조, 알고리즘, 설계 이유를 주제별로 설명한 13장짜리 교과서다.
- 설계: [DESIGN.md](DESIGN.md), [API.md](API.md), [ARCHITECTURE.md](ARCHITECTURE.md), [ROADMAP.md](ROADMAP.md)
- 마일스톤별 해설: [docs/learn/](docs/learn/README.md)
- 벤치마크: `bench/`, 측정 기록은 `bench/results/`
- 설계 결정을 위한 측정: `experiments/`

## 빌드와 테스트
stardust-numpy가 옆 폴더(`../stardust-numpy`)에 있어야 한다.
```bash
python3.15 -m venv .venv && .venv/bin/pip install nanobind scikit-build-core pytest hypothesis numpy scipy
```
```bash
.venv/bin/pip install --no-build-isolation --config-settings=build-dir="$PWD/build/stardust-numpy" ../stardust-numpy
```
```bash
.venv/bin/pip install --no-build-isolation -e .
```
```bash
.venv/bin/python -m pytest -q
```
