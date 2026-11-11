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
M6까지 진행했다.
- M1: `solve`, `inv`, `det`, `slogdet`, `LinAlgError`
- M2: `vecdot`, `outer`, `trace`, `diagonal`, `matrix_power`, `vector_norm`, `matrix_norm` (fro, ±1, ±inf)
- M3: `solve_triangular`, `cholesky`, `cholesky_solve`, `lu`, `lu_solve`
- M4: `qr`, `qr_pivoted`, `lstsq`
- M5: `eigh`, `eigvalsh`, `svd`, `svdvals`, `matrix_rank`, `pinv`, `cond`, `matrix_norm` (2, −2, nuc)
- M6: `eig`, `eigvals`, `schur`, `hessenberg`, `ldl`

```python
a = sdnp.asarray([[2.0, 1.0], [1.0, 3.0]])
sdla.solve(a, sdnp.asarray([1.0, 2.0]))   # ≈ [0.2, 0.6]
sdla.slogdet(a)                            # SlogdetResult(sign=1.0, logabsdet=1.609...)
```

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
