# 벤치마크

벤치마크는 두 체계로 나뉜다. 속도는 `bench/speed/`(`python -m bench.speed`)가 재고, 정확도는 `bench/accuracy/`(`python -m bench.accuracy`)가 잰다. 둘 다 결과에서 영역마다 표를 만든다. 속도는 `bench/<영역>/speed.md`, 정확도는 `bench/<영역>/accuracy.md`다. 이 문서는 속도 체계를 다룬다.

마일스톤마다 따로 두었던 옛 벤치 스크립트와 결과(`t4/`, `g5/`, `m3_views.py`, `bench/merge/` 등)는 지웠다. 커밋 4151d2a에 남아 있다. 지금 상태의 속도는 영역마다 생성되는 `bench/<영역>/speed.md` 하나다.

## 구성

| 파일 | 하는 일 |
|---|---|
| `bench/console.py` | 콘솔 출력(제목 블록, 표, 진행 막대, 메모; 정확도 측정·테스트도 같이 쓴다) |
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
.venv/bin/python -m bench.speed run --area stats --smoke          # 사례가 모두 도는지만 (한 번씩, 10개 프로세스로 나눠)
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
- `Ref(label, make, threads=...)`: Python 비교 대상. `"fixed"`(기본)는 라운드마다 한 번, 그 라이브러리의 기본 스레드로 잰다(numpy·scipy의 원소별 계산은 1스레드, BLAS는 기본 스레드). `"follow"`는 sd의 스레드 수마다 BLAS 스레드를 같게 묶어 잰다. 회귀처럼 BLAS가 일의 대부분인 비교에 쓴다.
- `RRef(label, expr, data, setup, packages)`: R 비교 대상. 실행 전체가 R 프로세스 하나를 쓰고, 꾸러미는 한 번만 불러온다. `data`의 배열은 float64 원시 파일로 넘어가고 시간은 R 안에서 같은 규칙으로 잰다. R은 1스레드다.
- `Ref.n`: 비교 대상이 너무 느려 입력의 일부만 재는 경우 그 원소 수. 표는 사례의 크기로 환산하고 `*`를 붙인다.
- **비교 대상은 정확도 측정과 같다.** 영역마다 같은 일을 하는 비교 대상이 정확도 측정(`bench/accuracy`)에 있으면 속도 사례에도 둔다. array는 numpy와 R, linalg는 numpy·scipy.linalg·R, special은 scipy·R·Boost.Math, stats는 scipy·R(mvtnorm)·Boost.Math, regression은 statsmodels·scikit-learn·R, optimize는 scipy·R이다. special과 stats의 R 식과 Boost 이름은 정확도 측정의 표(`bench/accuracy/compare.py`의 `R_SPECIAL`, `BOOST_SPECIAL`, `R_DIST`, `R_METHOD`, `BOOST_*`)를 그대로 가져다 쓴다. 그래서 한쪽에 함수를 더하면 다른 쪽에도 따라온다. 한 줄씩 Python으로 부르는 스칼라 사례에는 R과 Boost를 두지 않는다. 그 비용은 배열 사례에서 본다.
- **Boost.Math**(`bench/speed/boost.py`, `boost_speed.cpp`): 정확도 측정이 받아 둔 Boost 소스(`build/comparators/boost-math`)를 sd와 같은 최적화 옵션으로 빌드한다. 배열 하나에 ctypes 호출 한 번이고, 원소마다 Boost 함수를 직접 부르는 C++ 반복문이다. 이름은 호출마다 한 번 풀고, 매개변수가 스칼라인 분포는 호출마다 한 번 만든다. 소스가 없으면 Boost 열을 빼고 돈다.
- **optimize의 R**: 목적 함수, 잔차, 기울기를 R로 다시 쓴다(sd는 sd 배열, scipy는 numpy 배열로 쓴 것과 같은 원칙이다). 정확도 측정은 reticulate로 같은 Python 함수를 부르지만, 속도에서는 그 다리의 비용이 시간을 차지하기 때문이다. R의 반복 상한과 허용오차(optim의 maxit·pgtol·lmm, nls.lm의 ftol·ptol·gtol·maxfev)는 sd·scipy 호출과 같은 멈춤 일이 되도록 Python 쪽 값에 맞추고, 정확도 측정과 같은 정의(`bench/accuracy/optimize.py`의 `r_optim_control`, `r_nlslm_control`)를 가져다 쓴다. Boost의 빌드 옵션도 정확도 측정의 것(`bench.accuracy.boost.FLAGS`)을 쓴다. R의 알고리즘이 다른 곳(optim의 BFGS·Nelder–Mead, 차분 기울기)은 사례의 비고에 적는다.
- `quick`: 빠른 묶음. 영역마다 1–3분 안에 도는 대표 사례들이다.
- `cost`: 이 사례의 1스레드 시간이 들어가는 C++ 비용표와 그 키 (아래).
- 사례 id(`영역/묶음/이름 [크기]`)는 영어로만 쓴다(테스트가 확인한다). 묶음(`group`)은 짧은 영어 키(`continuous`, `binary-f64` 등)이고, 생성되는 표의 절 제목은 사례 파일의 `TITLES`(키 → 한글 제목)에서 온다. 비교 대상 이름(`Ref.label`)도 영어다. `note`는 표의 비고 열이라 한글로 쓴다.

## 시간 재는 규칙

- 처음 한 번은 예열로 부르고 버린다.
- 배치 하나가 20 ms 이상이 되도록 배치 크기를 1부터 두 배씩 늘린다. 그다음 배치를 7번(한 측정에 2초를 넘으면 줄이되 3번 이상) 재고 배치마다 한 번 부른 시간을 남긴다. 한 번에 5초가 넘는 호출은 라운드마다 한 번만 잰다.
- 라운드마다 모든 사례를 돌고, 사례 안에서 sd와 비교 대상을 잇달아 잰다. 기계가 느려진 때가 비율의 양쪽에 같이 걸린다.
- 보고하는 시간은 모든 라운드의 모든 배치 중 최솟값이다. OS가 가장 덜 방해한 값이다. 잡음 폭은 (중앙값 ÷ 최솟값 − 1)이다.
- 배치 안에서는 Python의 쓰레기 수집을 끈다. 측정마다 그 전에 한 번 모은다(statsmodels의 결과는 순환 참조로 GB 단위를 남긴다).
- **같은 입력을 되풀이하지 않는다**: 가지가 많은 계산(정렬, argsort, 고유값, searchsorted, 마스크로 걸러내기, groupby, merge 등)은 같은 모양의 입력 여럿을 미리 만들어 차례로 부른다(`harness.rotate`, 입력 수는 `fresh_count(n)`: 원소 10⁶이면 4개, 작으면 최대 64개). 한 입력만 되풀이하면 분기 예측기와 캐시가 그 입력을 배워 실제보다 빨라진다(numpy argsort 10³: 7.6 µs, 새 입력이면 29 µs). 원소별 계산은 한 입력을 쓴다.
- **스레드**: Accelerate(numpy·scipy와 sd의 BLAS·LAPACK)는 스레드 수를 프로세스가 뜰 때 `VECLIB_MAXIMUM_THREADS`로만 읽고 threadpoolctl로는 바뀌지 않는다. 그래서 감독은 BLAS 스레드 설정마다 작업 프로세스를 따로 띄운다. sd의 1스레드 측정과 `"follow"` 비교 대상의 1스레드 측정은 `VECLIB_MAXIMUM_THREADS=1`인 프로세스에서, sd의 기본 스레드(성능 코어 수, 10) 측정과 `"fixed"` 비교 대상과 R은 그 변수 없이 띄운 프로세스(Accelerate 기본)에서 잰다. scikit-learn의 OpenMP는 threadpoolctl로 같이 묶는다. 사례의 데이터는 프로세스마다 만든다.

## 결과

- **실행 파일** (`build/speed/<날짜-시각>[-이름].json`): 실행 하나의 결과. 커밋과 작업 중 변경 여부, 기계(칩, 코어, macOS, 메모리), 라이브러리 판(numpy, scipy, pandas, statsmodels, scikit-learn, R과 그 꾸러미), BLAS와 그 스레드 수, sd의 기본 스레드 수, 시작할 때의 부하, 사례마다 측정(대상, 스레드, 최솟값, 중앙값, 잡음, 배치 수, 배치당 호출 수, 배치 시간 전부), 사례의 라운드당 시간, 부하, 최대 메모리가 들어간다.
- **기준선** (`bench/speed/results/<영역>.json`): 영역마다 하나, 저장소에 둔다. `promote`(또는 `run --promote`)는 실행이 잰 사례만 바꾸고 나머지는 둔다. 그래서 함수 하나만 다시 재서 넣을 수 있다. 사례마다 어느 실행에서 왔는지 적고, 등록부에서 사라진 사례는 뺀다. 끝나지 않은 실행과 `--smoke` 실행은 넣지 않는다.
- **표** (`bench/<영역>/speed.md`): `promote`가 기준선으로 다시 만든다(`table` 명령으로 따로 만들 수도 있다). 손으로 고치지 않는다. 설계·API 문서는 숫자를 옮겨 적지 않고 이 표를 가리킨다.

## 비교

`compare RUN.json`은 실행의 영역들의 기준선(또는 `--base FILE`)과 비교한다. sd 측정이 기준선보다 10% 넘게 느리면(ROADMAP의 기준선 규칙) 적고, 그 차이가 두 측정의 잡음 폭 중 큰 것보다 크면 `slower`, 아니면 `within noise`다(보고서는 영어다). 비교 대상 라이브러리 시간의 변화(중앙값)도 보인다. numpy·scipy도 같이 느려졌다면 코드가 아니라 기계 상태가 달라진 것이다. 잡음 폭 밖으로 느려진 것이 있으면 종료 코드가 1이다. `--faster`는 빨라진 것도 보인다.

## C++ 비용표

병렬 분할은 원소마다 잰 1스레드 비용으로 일의 양을 센다. 두 표가 기준선에서 생성된다.

- `csrc/stats/cost_table.hpp`: 키 `분포.메서드`(예: `normal.cdf`). stats의 연속·이산·극값 분포 사례가 채운다. 같은 키의 사례가 여럿이면(매개변수 묶음이 여럿) 기하평균이다. 이산 분포의 좁은 지지 사례는 키 `분포.메서드@memo`로 따로 잰다. 스칼라 매개변수에서 정수점 기억 표가 답하는 호출(원소당 1–2 ns)의 비용이고, 표에서 `memo = true`인 줄이 된다. 넓은 지지의 계산(수백 ns–수십 µs)과 섞으면 좁은 지지 호출이 무거운 가중치를 받아 10스레드가 1스레드보다 느려진다.
- `csrc/special/cost_table.hpp`: 키는 함수 이름(예: `erf`, `erf_complex`, `_add`). special의 10⁶ 원소 사례가 채운다. `csrc/special/cost.hpp`는 연산마다 이 표의 어느 이름을 쓰는지만 적는다. 표에 없는 이름은 컴파일되지 않는다.

`cost`는 기준선에 있는 키만 바꾸고 나머지 줄은 지금 파일의 값을 둔다. `cost --check`는 쓰지 않고 바뀔지만 알린다(바뀌면 종료 코드 1). 헤더가 바뀌면 다음 `import stardust`에서 C++가 다시 빌드된다.

## 긴 실행

전체 재측정은 몇 시간이 걸릴 수 있다. 그래서 실행기는 이렇게 동작한다.

- **측정 프로세스**: 측정은 감독 프로세스가 띄운 작업 프로세스(`worker.py`, 자기 프로세스 그룹)에서 한다. 감독은 사례마다 걸린 시간과, 작업 프로세스와 그 자식(R)의 상주 메모리 합을 2초마다 본다(macOS에서는 libproc로 읽어 측정을 방해할 프로세스를 띄우지 않는다). 사례 하나(한 라운드의 모든 스레드와 비교 대상)가 15분을 넘거나 메모리가 16 GB를 넘으면 그룹을 죽이고 그 사례를 `timeout`이나 `memory`로 적은 뒤 새 작업 프로세스로 다음 사례를 잰다. `--timeout 분`, `--max-rss GB`로 바꾼다. 작업 프로세스는 40사례마다 새로 띄워 쌓인 메모리를 버린다.
- **이어 달리기**: 사례 하나가 끝날 때마다 실행 파일을 다시 쓴다(배치 시간까지). 중단(Ctrl-C, 전원)되면 많아야 사례 하나를 잃는다. `run --resume build/speed/<실행>.json`이 남은 곳부터 같은 실행으로 잇는다.
- **예상 시간**: `estimate`는 기준선의 사례별 라운드당 시간으로 고른 사례의 시간을 어림한다. 실행 중에는 진행 막대에 경과와 남은 예상 시간이 나온다. 남은 시간은 지금까지 기준선보다 얼마나 빨랐는지로 고친다.
- **잠자기 막기**: 감독이 살아 있는 동안 `caffeinate -i`가 기계를 깨워 둔다.
- **로그**: 실행 파일 옆의 `.log`에 콘솔에 나온 줄(시각과 함께), 사례마다 한 줄의 진행, 작업 프로세스의 stderr가 쌓인다.
- **조용한 기계**: 재기 전에 다른 프로세스가 쓰는 CPU를 3초 동안 직접 잰다(커널의 CPU 시간 계수기와 libproc로 읽어 프로세스를 띄우지 않는다). 1코어(`--quiet-cores`)를 넘게 쓰면 가장 바쁜 프로세스들을 보여 주고, 조용해질 때까지 최대 10분 기다린다. 그래도 바쁘면 경고를 남기고 시작한다. `--no-wait`는 기다리지 않고, `--strict`는 바쁘면 멈춘다(종료 코드 3). `--smoke`는 검사하지 않는다.
- **바쁠 때 잰 사례**: 실행 중에도 사례마다 다른 프로세스가 평균 몇 코어를 썼는지(`other_cpu`) 적는다. 실행 자신(감독, 작업 프로세스, R)의 CPU는 뺀다. 1코어를 넘은 사례는 `busy`로 적고 진행 막대 위에 알린다. 표에서는 이름 옆에 †가 붙고, `compare`는 그 사례를 `busy`로 두어 느려짐으로 세지 않는다. `run --rerun-busy build/speed/<실행>.json`은 그 사례들만 다시 잰다. 그 실행을 `promote`하면 기준선의 그 사례들만 바뀐다.
- **부하**: 1분 부하를 사례마다 적는다(CPU 계수기를 읽을 수 없는 기계에서는 시작할 때 3을 넘으면 경고한다).
- **알림**: 1분 넘게 걸린 실행이 끝나면 macOS 알림을 띄운다(`--no-notify`로 끈다).
- **콘솔**: 콘솔 출력은 모두 영어다. 시작할 때 실행 요약(실행 id, 커밋, 사례·라운드 수, 스레드, 예상 시간, 다른 프로세스의 CPU, 결과·로그 경로)을 회귀 결과 요약과 같은 78칸 틀로 보인다. 실행 중에는 tqdm 진행 막대(라운드, 비율, 경과 < 남은 예상, 지금 사례)를 보이고, 사례가 끝날 때마다 그 위에 결과를 몇 줄로 적는다. sd의 스레드별 시간(10스레드는 1스레드보다 몇 배 빠른지), 비교 대상마다 그 시간과 'sd is 2.51x faster/slower'(같은 스레드 수의 sd와, 자기 기본 스레드로 도는 비교 대상은 sd의 가장 적은 스레드와 견준다), 바빴으면 그 사실이다. 느린 쪽은 빨간색이다. 줄은 터미널 폭(60–100칸)을 넘지 않게 나눈다. 실패와 중단도 그 위에 한 줄씩 적는다. 끝나면 요약(측정·실패·바쁨 수, 최대 메모리), 영역별 표(사례, 실패, 바쁨, 라운드당 시간), 영역 × 비교 대상 표(사례 수, 비교 대상 ÷ sd의 중앙값, sd가 느린 사례 수), 실패 목록, 번호 붙은 메모(다시 재기·이어 달리기 명령)를 보인다. `list`, `estimate`, `compare`, `promote`, `table`, `cost`도 같은 틀(제목 블록, 표, 메모)로 보인다. `compare --out`으로 저장하는 보고서는 문서에 붙이기 좋게 Markdown 표다. 터미널이 아니면(파일로 돌리면) 막대 대신 사례마다 한 줄을 쓴다. 꾸미는 도구(`bench/console.py`)는 정확도 측정과 `tools/test.py`도 같이 쓴다. 진행 막대에는 tqdm이 필요하다. 벤치에 필요한 패키지는 선택 의존성 묶음 `bench`(`uv pip install -e ".[bench]"`: test 묶음과 tqdm, pyarrow, threadpoolctl)에 있다. tqdm이 없으면 막대 대신 줄로 쓴다.
- **점검 실행의 병렬**: `--smoke`는 아무것도 재지 않으므로 사례를 여러 갈래로 나눠 동시에 돌린다. 프로세스는 R 세션까지 합쳐 `--jobs`개(기본 10) 이하다. 갈래마다 작업 프로세스 하나를 띄우고, R을 쓰면 갈래 수를 절반으로 줄인다. 메모리 제한은 모든 갈래의 작업 프로세스를 합쳐 본다. 합이 넘으면 메모리를 가장 많이 쥔 작업 프로세스의 사례만 멈추고, 그 사례는 마지막에 혼자 다시 돌린다. 다른 갈래와 함께 돌 때만 넘는 사례가 있어서다. 측정 실행은 늘 사례를 하나씩 잰다.

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
