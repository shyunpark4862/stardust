# sd.random API (초안)

> 상태: **초안** (2026-11-15). 결정과 내부 구조는 [docs/design/random.md](../design/random.md). 아직 구현되지 않았다.

## §1. Generator
```
default_rng(seed=None, /)

rng.random(shape=(), /)
rng.integers(low, high=None, /, *, shape=(), endpoint=False)
rng.choice(a, /, *, shape=(), replace=True, p=None)
rng.permutation(x, /)
rng.shuffle(x, /)
```
- 전역 함수와 전역 상태가 없다. 난수는 언제나 Generator에서 받는다.
- 생성기는 계수기형이다(Philox 계열, docs/design/random.md §2). 같은 시드와 같은 호출 순서면 같은 값이고, 스레드 수와 상관없다. numpy의 `default_rng`와 값은 다르다.
- `random`은 [0, 1)의 float64, `integers`는 int64다.
- 분포의 표본은 여기가 아니라 `sd.stats`의 각 분포에 있다: `sd.stats.Normal(mu=0, sigma=1).sample((1000,), rng=rng)`.
- 시드의 타입, `spawn`, 상태 읽기·쓰기, `shuffle`이 제자리인지는 시작 전에 정한다(docs/design/random.md §6).
