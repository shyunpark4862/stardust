# sd.random API

> 상태: **T3 완료·승인** (2026-11-15). 결정과 내부 구조는 [docs/design/random.md](../design/random.md) (R1–R11).

## §1. Generator
```
default_rng(seed=None, /)

rng.random(shape=None, /)
rng.integers(low, high=None, /, *, shape=None, endpoint=False)
rng.choice(a, /, *, shape=None, replace=True, p=None)
rng.permutation(x, /)
rng.spawn(n, /)
rng.state
```
- 물리 메모리보다 큰 결과를 요청하면 스트림을 움직이기 전에 `MemoryError`다: `permutation(): cannot allocate 8796093022208 bytes: more than this machine's physical memory`(API.md §2). 생성기의 `position`은 그대로다.
- 전역 함수와 전역 상태가 없다. 난수는 언제나 Generator에서 받는다. 분포의 표본은 `sd.stats`의 각 분포에 있다(`X.sample(shape, rng=rng)`).
- 생성기는 **Philox4×64-10**이다(R5). 같은 시드와 같은 호출 순서면 같은 값이고, **스레드 수와 상관없다**. 한 호출을 둘로 나누어 불러도 앞의 원소들은 같다. numpy의 `default_rng`와 값은 다르다.
- **`shape`**: `None`이면 스칼라(Python `float`/`int`), 또는 int, int의 tuple이다. `integers`에서 `low`·`high`가 배열이면 `None`은 그 둘을 브로드캐스트한 모양이다(numpy의 `size=None`과 같다). 0인 축도 된다.

### §1.1 `default_rng(seed=None, /)`
- `seed`: 0 이상의 int(크기 제한 없음), 그런 int의 시퀀스, `None`(OS 엔트로피). 시퀀스는 list·tuple이고, 길이가 있고 차례로 읽을 수 있는 객체(배열, set)도 읽은 순서대로 받는다. Generator를 주면 그대로 돌려준다.
- 키는 numpy의 `SeedSequence`와 같은 해시로 만든다(R6). 그래서 비슷한 시드(0, 1, 2)도 서로 무관한 수열이다.

### §1.2 `random(shape=None, /)`
- `shape`(모든 메서드 공통): `None`이면 스칼라, int나 int의 tuple이면 배열이다. `()`는 `ValueError`다(0차원 배열이 없다. 메시지는 `메서드(): 'shape' must not be ()`로 같다).
- [0, 1)의 float64. 64비트의 위 53비트 × 2⁻⁵³ (numpy와 같은 변환).

### §1.3 `integers(low, high=None, /, *, shape=None, endpoint=False)`
- [low, high)의 int64(`endpoint=True`면 [low, high]). `integers(n)`은 [0, n)이다.
- `low`·`high`는 int(int64 범위) 또는 int64 배열이고, 서로와 `shape`에 브로드캐스트한다(R10).
- 모든 원소에서 `low < high`(`endpoint=True`면 `low <= high`)여야 한다. 아니면 `ValueError`(`integers(n)`으로 불렀으면 메시지는 `'high' must be positive`다). int64 범위 밖의 int는 `OverflowError`(`choice`의 `a`, `permutation`의 `x`도 같다).
- 치우침이 없다(Lemire의 방법, 기각은 원소 안에서).

### §1.4 `choice(a, /, *, shape=None, replace=True, p=None)`
- `a`: 1차원 배열, 또는 int n(`arange(n)`에서 고른다).
- `p`: 각 원소의 확률(1차원 float 배열, 길이 n). 음수·NaN이 없고 합이 1에서 sqrt(eps) 안이어야 한다(numpy와 같은 허용치, R10). 합으로 나눠 쓴다. complex 배열은 `TypeError`다. 비복원 추출에서 0보다 큰 확률은 아무리 작아도(약 2e-307 아래 포함) 확률 0인 원소보다 먼저 뽑힌다.
- `replace=False`: 같은 원소를 두 번 고르지 않는다. 표본 수가 모집단보다 크면 `ValueError`, `p`가 있으면 0이 아닌 확률의 수보다 커도 `ValueError`다. 가중치가 있으면 Efraimidis–Spirakis의 경주로 고른다(numpy와 방법이 달라 같은 시드의 분포는 같지만 값은 다르다).

### §1.5 `permutation(x, /)`
- int n이면 `arange(n)`을 섞은 int64 배열, 배열이면 첫 축을 따라 섞은 **새 배열**이다. 입력은 바꾸지 않는다.
- `shuffle`은 없다(R9): stardust에는 인자를 제자리에서 바꾸는 함수가 없다. 제자리가 필요하면 `x[...] = rng.permutation(x)`.

### §1.6 `spawn(n, /)`, `state`, 복사와 피클
- `spawn(n)`: 키가 다른 독립 생성기 n개의 리스트(numpy `SeedSequence.spawn`과 같은 키). 다시 부르면 번호가 이어진다.
- `state`: 시드 수열과 위치를 담은 dict(`bit_generator`, `entropy`, `spawn_key`, `n_children_spawned`, `position`). 대입하면 그 상태로 돌아간다. 잘못된 값의 에러는 모두 `.state: '키' …`로 시작한다(dict가 아니거나 키가 빠진 경우 포함).
- `copy.copy`, `copy.deepcopy`, `pickle`은 같은 상태의 독립 생성기를 만든다.

### §1.7 여러 Python 스레드
- 한 Generator를 여러 스레드가 함께 부르면 호출마다 자기 구간을 원자적으로 예약한다(R8). 같은 난수를 두 번 쓰지 않지만, 어느 스레드가 어느 구간을 받는지는 실행 순서에 따른다. 재현이 필요하면 스레드마다 `spawn`한 생성기를 쓴다.

### §1.8 속도 (bench/random/t3_speed.md, 원소 10⁶)
- 1스레드에서 numpy(PCG64) 대비 `random` 2.4배, `integers` 1.1–3.7배, `choice` 1.0배(가중치 2.9배), `permutation` 1.3–1.6배. 10스레드에서는 8–26배이고, 섞기만 순차라 1.5–1.6배다.
- 큰 모집단에서 작은 표본을 비복원으로 뽑는 `choice(10**6, shape=1000, replace=False)`는 0.03 ms(numpy 0.01 ms)다.
