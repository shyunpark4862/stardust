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
M0(환경과 설계 검증)까지 진행했다. 공개 함수는 M1부터 들어온다.

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
