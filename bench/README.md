# 벤치마크

속도는 `bench/speed/` 하나로 잰다. 정확도 측정(`accuracy.md`와 그 스크립트)은 영역 폴더에 따로 있고 이 체계에 들지 않는다.

`bench/<영역>/` 아래의 마일스톤 폴더(`t4/`, `g5/`, `m3_views.py` 등)는 그때의 기록이다. 그 표의 숫자는 그 커밋의 것이라 지금과 다를 수 있다. 지금 상태의 속도는 영역마다 생성되는 `bench/<영역>/speed.md` 하나다.

## 구성

| 파일 | 하는 일 |
|---|---|
| `bench/speed/harness.py` | 시간 재기(예열, 배치 크기, 최솟값·중앙값·잡음), 스레드 조절, 기계·라이브러리 정보 |
| `bench/speed/registry.py` | 사례 모델(`Case`, `Ref`, `RRef`)과 고르기 |
| `bench/speed/cases/<영역>.py` | 영역의 사례 목록 (`cases()`) |
| `bench/speed/rsession.py` | 실행 하나에 R 프로세스 하나 |
| `bench/speed/run.py`, `worker.py`, `proc.py` | 감독 프로세스와 측정 프로세스 (긴 실행, 아래) |
| `bench/speed/results.py` | 실행 파일과 기준선 파일 |
| `bench/speed/tables.py` | `bench/<영역>/speed.md` 생성 |
| `bench/speed/compare.py` | 기준선과 비교해 느려진 것 찾기 |
| `bench/speed/cost.py` | C++ 비용표 생성 (`csrc/stats/cost_table.hpp`, `csrc/special/cost_table.hpp`) |
| `bench/speed/results/<영역>.json` | 영역의 기준선 (저장소에 둔다) |
| `build/speed/` | 실행 파일과 로그 (git이 무시한다) |

명령은 저장소 맨 위에서 `.venv/bin/python -m bench.speed …`로 돌린다(`uv run --no-sync python -m bench.speed …`도 같다).

## 빠른 시작

```
.venv/bin/python -m bench.speed list --area stats --quick       # 사례 보기 (Q: 빠른 묶음, C: 비용표)
.venv/bin/python -m bench.speed run --quick                       # 모든 영역의 빠른 묶음, 1라운드
.venv/bin/python -m bench.speed run --area special --match erf    # 이름에 erf가 든 사례, 3라운드
.venv/bin/python -m bench.speed run --area stats --smoke          # 사례가 모두 도는지만 (한 번씩 부른다)
.venv/bin/python -m bench.speed compare build/speed/<실행>.json   # 기준선보다 느려진 것
.venv/bin/python -m bench.speed promote build/speed/<실행>.json   # 기준선에 넣고 speed.md 다시 만들기
.venv/bin/python -m bench.speed cost                              # C++ 비용표 다시 만들기
```

고르기 옵션은 `list`, `estimate`, `run`이 같이 쓴다. `--area A`(여러 번), `--match P`(정규식, `*`나 `?`가 있으면 glob; 여러 번 주면 하나만 맞아도 된다), `--group G`, `--quick`. 사례 id는 `영역/묶음/이름 [크기]`이고 `--match`는 id 전체에서 찾는다. 정규식이라 크기의 `[`, `^`는 `\[`, `\^`로 쓴다.

## 사례

사례 하나는 한 크기에서의 호출 하나다. 데이터를 만드는 `setup`과, 그 데이터로 잴 함수를 돌려주는 공장(factory)들로 이루어진다. 변환(numpy → sd, 상수 열 붙이기)은 `setup`이나 공장에서 해서 시간에 들지 않게 한다.

```python
Case(area="stats", group="연속 분포", name="Normal.cdf", size="10^5", n=100_000,
     setup=lambda: {...},                       # 사례마다, 라운드마다 한 번
     sd=lambda d: lambda: X.cdf(d["sx"]),
     refs=(Ref("scipy", lambda d: lambda: norm.cdf(d["x"])),),
     quick=True, cost=("stats", "normal.cdf"))
```

- `threads`: sd의 스레드 수들. 기본 `(1, 10)`(10은 이 기계의 성능 코어 수이자 sd의 기본값).
- `Ref(label, make, threads=...)`: Python 비교 대상. `"fixed"`(기본)는 라운드마다 한 번, 그 라이브러리의 기본 스레드로 잰다(numpy·scipy의 원소별 계산은 1스레드, BLAS는 기본 스레드). `"follow"`는 sd의 스레드 수마다 BLAS 스레드를 같게 묶어(threadpoolctl) 잰다. 회귀처럼 BLAS가 일의 대부분인 비교에 쓴다.
- `RRef(label, expr, data, setup, packages)`: R 비교 대상. 실행 전체가 R 프로세스 하나를 쓰고, 꾸러미는 한 번만 불러온다. `data`의 배열은 float64 원시 파일로 넘어가고 시간은 R 안에서 같은 규칙으로 잰다. R은 1스레드다.
- `Ref.n`: 비교 대상이 너무 느려 입력의 일부만 재는 경우 그 원소 수. 표는 사례의 크기로 환산하고 `*`를 붙인다.
- `quick`: 빠른 묶음. 영역마다 1–3분 안에 도는 대표 사례들이다.
- `cost`: 이 사례의 1스레드 시간이 들어가는 C++ 비용표와 그 키 (아래).
- 묶음(`group`) 이름은 생성되는 표의 절 제목이라 짧은 한글 명사구로 쓴다. `note`는 표의 비고 열이다.

## 시간 재는 규칙

- 처음 한 번은 예열로 부르고 버린다.
- 배치 하나가 20 ms 이상이 되도록 배치 크기를 1부터 두 배씩 늘린다. 그다음 배치를 7번(한 측정에 2초를 넘으면 줄이되 3번 이상) 재고 배치마다 한 번 부른 시간을 남긴다. 한 번에 5초가 넘는 호출은 라운드마다 한 번만 잰다.
- 라운드마다 모든 사례를 돌고, 사례 안에서 sd와 비교 대상을 잇달아 잰다. 기계가 느려진 때가 비율의 양쪽에 같이 걸린다.
- 보고하는 시간은 모든 라운드의 모든 배치 중 최솟값이다. OS가 가장 덜 방해한 값이다. 잡음 폭은 (중앙값 ÷ 최솟값 − 1)이다.
- 배치 안에서는 Python의 쓰레기 수집을 끈다.

## 결과

- **실행 파일** (`build/speed/<날짜-시각>[-이름].json`): 실행 하나의 결과. 커밋과 작업 중 변경 여부, 기계(칩, 코어, macOS, 메모리), 라이브러리 판(numpy, scipy, pandas, statsmodels, scikit-learn, R과 그 꾸러미), BLAS와 그 스레드 수, sd의 기본 스레드 수, 시작할 때의 부하, 사례마다 측정(대상, 스레드, 최솟값, 중앙값, 잡음, 배치 수, 배치당 호출 수, 배치 시간 전부), 사례의 라운드당 시간, 부하, 최대 메모리가 들어간다.
- **기준선** (`bench/speed/results/<영역>.json`): 영역마다 하나, 저장소에 둔다. `promote`(또는 `run --promote`)는 실행이 잰 사례만 바꾸고 나머지는 둔다. 그래서 함수 하나만 다시 재서 넣을 수 있다. 사례마다 어느 실행에서 왔는지 적고, 등록부에서 사라진 사례는 뺀다. 끝나지 않은 실행과 `--smoke` 실행은 넣지 않는다.
- **표** (`bench/<영역>/speed.md`): `promote`가 기준선으로 다시 만든다(`table` 명령으로 따로 만들 수도 있다). 손으로 고치지 않는다. 설계·API 문서는 숫자를 옮겨 적지 않고 이 표를 가리킨다.

## 비교

`compare RUN.json`은 실행의 영역들의 기준선(또는 `--base FILE`)과 비교한다. sd 측정이 기준선보다 10% 넘게 느리면(ROADMAP의 기준선 규칙) 적고, 그 차이가 두 측정의 잡음 폭 중 큰 것보다 크면 "느려짐", 아니면 "잡음 안"이다. 비교 대상 라이브러리 시간의 변화(중앙값)도 보인다. numpy·scipy도 같이 느려졌다면 코드가 아니라 기계 상태가 달라진 것이다. 잡음 폭 밖으로 느려진 것이 있으면 종료 코드가 1이다. `--faster`는 빨라진 것도 보인다.

## C++ 비용표

병렬 분할은 원소마다 잰 1스레드 비용으로 일의 양을 센다. 두 표가 기준선에서 생성된다.

- `csrc/stats/cost_table.hpp`: 키 `분포.메서드`(예: `normal.cdf`). stats의 연속·이산·극값 분포 사례가 채운다. 같은 키의 사례가 여럿이면(이산 분포의 좁은·넓은 지지) 기하평균이다.
- `csrc/special/cost_table.hpp`: 키는 함수 이름(예: `erf`, `erf_complex`, `_add`). special의 10⁶ 원소 사례가 채운다. `csrc/special/cost.hpp`는 연산마다 이 표의 어느 이름을 쓰는지만 적는다. 표에 없는 이름은 컴파일되지 않는다.

`cost`는 기준선에 있는 키만 바꾸고 나머지 줄은 지금 파일의 값을 둔다. `cost --check`는 쓰지 않고 바뀔지만 알린다(바뀌면 종료 코드 1). 헤더가 바뀌면 다음 `import stardust`에서 C++가 다시 빌드된다.

## 긴 실행

전체 재측정은 몇 시간이 걸릴 수 있다. 그래서 실행기는 이렇게 동작한다.

- **측정 프로세스**: 측정은 감독 프로세스가 띄운 작업 프로세스(`worker.py`, 자기 프로세스 그룹)에서 한다. 감독은 사례마다 걸린 시간과, 작업 프로세스와 그 자식(R)의 상주 메모리 합을 2초마다 본다(macOS에서는 libproc로 읽어 측정을 방해할 프로세스를 띄우지 않는다). 사례 하나(한 라운드의 모든 스레드와 비교 대상)가 15분을 넘거나 메모리가 16 GB를 넘으면 그룹을 죽이고 그 사례를 "제한 시간 초과"나 "메모리 초과"로 적은 뒤 새 작업 프로세스로 다음 사례를 잰다. `--timeout 분`, `--max-rss GB`로 바꾼다. 작업 프로세스는 40사례마다 새로 띄워 쌓인 메모리를 버린다.
- **이어 달리기**: 사례 하나가 끝날 때마다 실행 파일을 다시 쓴다(배치 시간까지). 중단(Ctrl-C, 전원)되면 많아야 사례 하나를 잃는다. `run --resume build/speed/<실행>.json`이 남은 곳부터 같은 실행으로 잇는다.
- **예상 시간**: `estimate`는 기준선의 사례별 라운드당 시간으로 고른 사례의 시간을 어림한다. 실행 중에는 줄마다 경과와 남은 예상 시간이 나온다. 남은 시간은 지금까지 기준선보다 얼마나 빨랐는지로 고친다.
- **잠자기 막기**: 감독이 살아 있는 동안 `caffeinate -i`가 기계를 깨워 둔다.
- **로그**: 실행 파일 옆의 `.log`에 진행 줄과 작업 프로세스의 stderr가 쌓인다.
- **조용한 기계**: 재기 전에 다른 프로세스가 쓰는 CPU를 3초 동안 직접 잰다(커널의 CPU 시간 계수기와 libproc로 읽어 프로세스를 띄우지 않는다). 1코어(`--quiet-cores`)를 넘게 쓰면 가장 바쁜 프로세스들을 보여 주고, 조용해질 때까지 최대 10분 기다린다. 그래도 바쁘면 경고를 남기고 시작한다. `--no-wait`는 기다리지 않고, `--strict`는 바쁘면 멈춘다(종료 코드 3). `--smoke`는 검사하지 않는다.
- **바쁠 때 잰 사례**: 실행 중에도 사례마다 다른 프로세스가 평균 몇 코어를 썼는지(`other_cpu`) 적는다. 실행 자신(감독, 작업 프로세스, R)의 CPU는 뺀다. 1코어를 넘은 사례는 `busy`로 적고 진행 줄에 알린다. 표에서는 이름 옆에 †가 붙고, `compare`는 그 사례를 "바쁠 때 잼"으로 두어 느려짐으로 세지 않는다. `run --rerun-busy build/speed/<실행>.json`은 그 사례들만 다시 잰다. 그 실행을 `promote`하면 기준선의 그 사례들만 바뀐다.
- **부하**: 1분 부하를 사례마다 적는다(CPU 계수기를 읽을 수 없는 기계에서는 시작할 때 3을 넘으면 경고한다).
- **알림**: 1분 넘게 걸린 실행이 끝나면 macOS 알림을 띄운다(`--no-notify`로 끈다).

## 전체를 다시 잴 때

```
.venv/bin/python -m bench.speed estimate                      # 걸릴 시간
.venv/bin/python -m bench.speed run --area stats --smoke      # 먼저 모든 사례가 도는지
.venv/bin/python -m bench.speed run --area stats --label stats
.venv/bin/python -m bench.speed compare build/speed/<실행>.json
.venv/bin/python -m bench.speed promote build/speed/<실행>.json
.venv/bin/python -m bench.speed cost                          # stats·special을 다시 쟀으면
.venv/bin/python -m pytest -q                                 # 비용표가 바뀌었으면 다시 빌드되고 테스트
```

영역마다 따로 재서 넣어도 된다. 기준선이 아직 없는 영역은 `compare`가 비교할 것이 없고, `estimate`는 그 사례들을 "시간 모름"으로 센다.
