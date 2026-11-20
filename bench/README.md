# 측정

측정은 두 체계다. 속도는 `bench/speed/`(`python -m bench.speed`, §1)가 재고, 정확도는 `bench/accuracy/`(`python -m bench.accuracy`, §2)가 잰다. 두 체계는 같은 절 구성(명령, 사례·스윕, 결과 파일, 비교, 종료 코드)이고, 비교 대상 라이브러리와 R·Boost.Math를 같이 쓴다. 둘 다 기준선에서 영역마다 표를 만든다: 속도는 `bench/<영역>/speed.md`, 정확도는 `bench/<영역>/accuracy.md`다. 설계·API 문서는 숫자를 옮겨 적지 않고 이 표를 가리킨다. 테스트, 골든, 커버리지는 [tests/README.md](../tests/README.md)에 있다.

명령은 저장소 맨 위에서 `.venv/bin/python -m bench.speed …`, `.venv/bin/python -m bench.accuracy …`로 돌린다. 필요한 패키지는 선택 의존성 묶음 `bench`(`.venv/bin/pip install -e ".[bench]"`: test 묶음과 tqdm, pyarrow, threadpoolctl)에 있다.

## 도구 공통 관례

모든 측정·시험 도구가 명령행을 `bench/cli.py`로 만든다(tests/README.md의 같은 표).

| 관례 | 뜻 |
|---|---|
| `AREA …` | 위치 인자. 영역 이름 `array linalg frame special stats regression random optimize`(`bench/areas.py`, README.md의 표 순서) 가운데 그 도구가 다루는 것. 주지 않으면 모두다. 정확도는 참값이 있는 여섯(array, linalg, special, stats, regression, optimize)을 다룬다. |
| `--match PATTERN` | id가 맞는 것만 남긴다. `*`, `?`가 있으면 id 전체와 맞춰 보는 glob이고, 없으면 부분 문자열이다. `[`, `]`는 보통 글자라 표에서 복사한 id(`erf(x) [10^6]`)가 그대로 맞는다. 여러 번 주면 하나만 맞아도 된다. |
| `--jobs N` | 작업 프로세스 수. 기본 8, 최대 10. |
| `--out FILE` | 결과를 쓸 파일(폴더가 아니다). |
| 종료 코드 | 0 모두 됐다. 1 실패, 계약을 넘은 값, 기준선보다 나빠진 측정, 또는 견줄 기준선이 없다. 2 명령행 오류나 도구 자신의 오류. 체계마다의 뜻은 §1.5, §2.5. |
| 로그 | `build/logs/<도구>/<시각>.log`(`speed`, `accuracy`), 도구마다 최근 20개를 둔다. |

- 콘솔 출력은 모두 영어이고 `bench/console.py`의 같은 틀이다(제목 블록, 표, 진행 막대, 번호 붙은 메모). `tools/test.py`, `tools/golden.py`, `tools/coverage.py`도 같이 쓴다. 진행 막대에는 tqdm이 필요하고, 없으면 막대 대신 줄로 쓴다.

## 1. 속도 (`bench/speed`)

속도의 기준선(`bench/speed/results/<영역>.json`)과 표(`bench/<영역>/speed.md`)는 아직 없다. 전체 측정을 한 번 하면 생긴다(§1.9). 그 전까지 C++ 비용표는 이전 파일의 값을 둔다(§1.6).

### 1.1 명령

| 파일 | 하는 일 |
|---|---|
| `bench/speed/harness.py` | 시간 재기(예열, 배치 크기, 최솟값·중앙값·잡음), 스레드 조절, 기계·라이브러리 정보 |
| `bench/speed/registry.py` | 사례 모델(`Case`, `Ref`, `RRef`, `Check`), 고르기, 사례 id 규칙 |
| `bench/speed/check.py` | 점검 실행의 결과 확인(sd와 비교 대상의 값 견주기, NaN 비율) |
| `bench/speed/cases/<영역>.py` | 영역의 사례 목록 (`cases()`) |
| `bench/speed/rsession.py` | 실행 하나에 R 프로세스 하나 |
| `bench/speed/run.py`, `worker.py`, `proc.py` | 감독 프로세스와 측정 프로세스 (§1.7) |
| `bench/speed/results.py` | 실행 파일과 기준선 파일 |
| `bench/speed/tables.py` | `bench/<영역>/speed.md` 생성 |
| `bench/speed/compare.py` | 기준선과 비교해 느려진 것 찾기 |
| `bench/speed/cost.py` | C++ 비용표 생성 (`csrc/stats/cost_table.hpp`, `csrc/special/cost_table.hpp`) |

```
.venv/bin/python -m bench.speed list stats --quick                 # 사례 보기 (Q: 빠른 묶음, C: 비용표)
.venv/bin/python -m bench.speed run --quick                        # 모든 영역의 빠른 묶음, 1라운드
.venv/bin/python -m bench.speed run special --match erf            # id에 erf가 든 사례, 3라운드
.venv/bin/python -m bench.speed run stats --smoke                  # 사례가 모두 돌고 값이 맞는지만 (§1.4)
.venv/bin/python -m bench.speed compare build/speed/<실행>.jsonl   # 기준선보다 느려진 것
.venv/bin/python -m bench.speed promote build/speed/<실행>.jsonl   # 기준선에 넣고 speed.md 다시 만들기
.venv/bin/python -m bench.speed cost                               # C++ 비용표 다시 만들기
```

고르기는 `list`, `estimate`, `run`이 같이 쓴다: 영역 위치 인자, `--match`(공통 관례), `--quick`(빠른 묶음). 사례 id는 `영역/묶음/이름 [크기]`이고 `--match`는 id 전체에서 찾는다. `--match 'stats/discrete/*'`처럼 묶음 하나를 고른다.

### 1.2 사례

사례 하나는 한 크기에서의 호출 하나다. 데이터를 만드는 `setup`과, 그 데이터로 잴 함수를 돌려주는 공장(factory)들로 이루어진다. 변환(numpy → sd, 상수 열 붙이기)은 `setup`이나 공장에서 해서 시간에 들지 않게 한다.

```python
Case(area="stats", group="continuous", name="Normal.cdf", size="10^5", n=100_000,
     setup=lambda: {...},                       # 사례마다, 라운드마다 한 번
     sd=lambda d: lambda: X.cdf(d["sx"]),
     refs=(Ref("scipy", lambda d: lambda: norm.cdf(d["x"])),),
     quick=True, cost=("stats", "normal.cdf"))
```

- `threads`: sd의 스레드 수들. 기본 `(1, 10)`(10은 이 기계의 성능 코어 수이자 sd의 기본값).
- `Ref(label, make, threads=...)`: Python 비교 대상. `"fixed"`(기본)는 라운드마다 한 번, 그 라이브러리의 기본 스레드로 잰다(numpy·scipy의 원소별 계산은 1스레드). `"follow"`는 sd의 스레드 수마다 BLAS 스레드를 같게 묶어 잰다. BLAS·LAPACK이 일의 대부분인 비교는 모두 `"follow"`다: array의 matmul·cov·corrcoef·tensordot, linalg 전체, regression, optimize.
- `RRef(label, expr, data, setup, packages, sets)`: R 비교 대상. 실행 전체가 R 프로세스 하나를 쓰고, 꾸러미는 한 번만 불러온다. `data`의 배열은 float64 원시 파일로 넘어가고 시간은 R 안에서 같은 규칙으로 잰다. R은 1스레드다(`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`로 띄운다; 이 기계의 R은 OpenMP판 OpenBLAS를 쓰고 그 변수만 듣는다). `sets`는 입력을 돌려 쓰는 사례의 입력 묶음 목록(사례 데이터의 키)이고, R도 다른 쪽처럼 호출마다 다음 묶음을 쓴다.
- `differs`(`Ref`, `RRef`): 비교 대상이 이 입력에서 원래 다른 계산을 할 때 그 까닭(R의 `choose`는 정수가 아닌 k를 반올림한다 등). 결과 확인에서 빼고, 표의 비고에 붙는다.
- `check`(`Check`): 결과 확인 방법(§1.4). 기본은 sd와 첫 Python 비교 대상의 값을 그대로 견준다. 결과의 꼴이 다르면 견줄 값을 고르는 함수(`sd`, `ref`)를, 값을 견줄 수 없으면 까닭을 `skip`에 둔다(두 난수 흐름, 대입문 등).
- `Ref.n`: 비교 대상이 너무 느려 입력의 일부만 재는 경우 그 원소 수. 표는 사례의 크기로 환산하고 `*`를 붙인다.
- **비교 대상은 정확도 측정과 같다.** 영역마다 같은 일을 하는 비교 대상이 정확도 측정(`bench/accuracy`)에 있으면 속도 사례에도 둔다. array는 numpy와 R, linalg는 numpy·scipy.linalg·R, special은 scipy·R·Boost.Math, stats는 scipy·R(mvtnorm)·Boost.Math, regression은 statsmodels·scikit-learn·R, optimize는 scipy·R이다. special과 stats의 R 식과 Boost 이름은 정확도 측정의 표(`bench/accuracy/compare.py`의 `R_SPECIAL`, `BOOST_SPECIAL`, `R_DIST`, `R_METHOD`, `BOOST_*`)를 그대로 가져다 쓴다. 그래서 한쪽에 함수를 더하면 다른 쪽에도 따라온다. 한 줄씩 Python으로 부르는 스칼라 사례에는 R과 Boost를 두지 않는다. 그 비용은 배열 사례에서 본다.
- **Boost.Math**(`bench/speed/boost.py`, `boost_speed.cpp`): 정확도 측정이 쓰는 Boost 소스(`build/comparators/boost-math`)를 정확도 측정과 같은 함수(`bench.accuracy.boost.build`)와 같은 옵션(`bench.accuracy.boost.FLAGS`: sd의 최적화 옵션 `-O3 -mcpu=native -ffp-contract=off`와 `-std=c++17`)으로 빌드한다. 빌드한 라이브러리의 이름에 소스·옵션·Boost 체크아웃의 커밋이 들어가므로 체크아웃을 올리면 다시 빌드한다. 비교 대상의 Boost 판과 sd.special이 담은 Boost 판(`third_party/boost_math/README.md`)은 다를 수 있어 결과 파일의 환경에 둘 다 적는다. 사례 목록은 정적 이름 표(`bench.accuracy.compare`의 `BOOST_SPECIAL`, `BOOST_DISTS`)만 보므로 `list`는 아무것도 빌드하지 않고, 빌드는 비교 대상을 처음 부를 때 한다. 배열 하나에 ctypes 호출 한 번이고, 원소마다 Boost 함수를 직접 부르는 C++ 반복문이다. 이름은 호출마다 한 번 풀고, 매개변수가 스칼라인 분포는 호출마다 한 번 만든다. 인자는 sd의 순서로 받는다(`chndtr(df, nc, x)` 등). 소스가 없으면 Boost 열을 빼고 돈다.
- **optimize의 R**: 목적 함수, 잔차, 기울기를 R로 다시 쓴다(sd는 sd 배열, scipy는 numpy 배열로 쓴 것과 같은 원칙이다). 정확도 측정은 reticulate로 같은 Python 함수를 부르지만, 속도에서는 그 다리의 비용이 시간을 차지하기 때문이다. R의 반복 상한과 허용오차(optim의 maxit·pgtol·lmm, nls.lm의 ftol·ptol·gtol·maxfev)는 sd·scipy 호출과 같은 멈춤 일이 되도록 Python 쪽 값에 맞추고, 정확도 측정과 같은 정의(`bench/accuracy/optimize.py`의 `r_optim_control`, `r_nlslm_control`)를 가져다 쓴다. Boost의 빌드 옵션도 정확도 측정의 것(`bench.accuracy.boost.FLAGS`)을 쓴다. R의 알고리즘이 다른 곳(optim의 BFGS·Nelder–Mead, 차분 기울기)은 사례의 비고에 적는다.
- `quick`: 빠른 묶음. 영역마다 1–3분 안에 도는 대표 사례들이다.
- frame의 `read_csv` 사례는 `build/speed/frame-data/`에 한 번 써 둔 파일을 읽는다. 파일 이름에 그 파일을 만드는 코드의 해시와 pandas 판이 들어가므로, 둘 중 하나가 바뀌면 옛 파일을 읽지 않고 새로 쓴다.
- `cost`: 이 사례의 1스레드 시간이 들어가는 C++ 비용표와 그 키 (아래).
- 사례 id(`영역/묶음/이름 [크기]`)는 ASCII로만 쓴다(테스트가 확인한다). 묶음(`group`)은 짧은 영어 키(`continuous`, `binary-f64` 등)이고, 생성되는 표의 절 제목은 사례 파일의 `TITLES`(키 → 한글 제목)에서 온다. 비교 대상 이름(`Ref.label`)도 ASCII다. `note`는 표의 비고 열이라 한글로 쓴다.


#### 시간 재는 규칙

- 처음 한 번은 예열로 부르고 버린다.
- 배치 하나가 20 ms 이상이 되도록 배치 크기를 1부터 두 배씩 늘린다. 그다음 배치를 7번(한 측정에 2초를 넘으면 줄이되 3번 이상) 재고 배치마다 한 번 부른 시간을 남긴다. 한 번에 5초가 넘는 호출은 라운드마다 한 번만 잰다.
- 라운드마다 모든 사례를 돌고, 사례 안에서 sd와 비교 대상을 잇달아 잰다. 기계가 느려진 때가 비율의 양쪽에 같이 걸린다.
- 보고하는 시간은 모든 라운드의 모든 배치 중 최솟값이다. OS가 가장 덜 방해한 값이다. 잡음 폭은 (중앙값 ÷ 최솟값 − 1)이다.
- 배치 안에서는 Python의 쓰레기 수집을 끈다. 측정마다 그 전에 한 번 모은다(statsmodels의 결과는 순환 참조로 GB 단위를 남긴다). R은 배치 안에서 쓰레기 수집을 끌 수 없어 배치 전에 `gc()`를 한 번 부른다. 그래서 할당이 많은 R 호출(lm, rq 등)은 배치 안의 수집 시간까지 들어간다. R의 시계는 `Sys.time()`이다(마이크로초 단위; `proc.time()`의 경과 시간은 이 기계에서 밀리초 단위라 20 ms 배치에 거칠다).
- **같은 입력을 되풀이하지 않는다**: 가지가 많은 계산(정렬, argsort, 고유값, searchsorted, 마스크로 걸러내기, groupby, merge 등)은 같은 모양의 입력 여럿을 미리 만들어 차례로 부른다(`harness.rotate`, 입력 수는 `fresh_count(n)`: 원소 10⁶이면 4개, 작으면 최대 64개). 한 입력만 되풀이하면 분기 예측기와 캐시가 그 입력을 배워 실제보다 빨라진다(numpy argsort 10³: 7.6 µs, 새 입력이면 29 µs). 원소별 계산은 한 입력을 쓴다.
- **스레드**: Accelerate(numpy·scipy와 sd의 BLAS·LAPACK)는 스레드 수를 프로세스가 뜰 때 `VECLIB_MAXIMUM_THREADS`로만 읽고 threadpoolctl로는 바뀌지 않는다. 그래서 감독은 BLAS 스레드 설정마다 작업 프로세스를 따로 띄운다. sd의 1스레드 측정과 `"follow"` 비교 대상의 1스레드 측정은 `VECLIB_MAXIMUM_THREADS=1`인 프로세스에서, sd의 기본 스레드(성능 코어 수, 10) 측정과 `"fixed"` 비교 대상은 그 변수 없이 띄운 프로세스(Accelerate 기본)에서 잰다. R은 늘 1스레드다. scikit-learn의 OpenMP는 threadpoolctl로 같이 묶는다. 사례의 데이터는 프로세스마다 만든다. 라운드가 여럿이면 라운드마다 작업 프로세스를 새로 띄운다. 짧은 병렬 호출은 프로세스마다 빠른 쪽이나 느린 쪽에 자리 잡을 수 있어서(10스레드 `a + b` 2^16에서 4.5 µs와 10.9 µs), 최솟값이 여러 프로세스에서 나오게 한다.

### 1.3 결과 파일

- **실행 파일** (`build/speed/<날짜-시각>[-이름].jsonl`, git이 무시한다): 실행 하나의 결과를 JSON Lines로 한 줄씩 덧붙인다(첫 줄은 실행의 머리, 그다음은 사례가 끝날 때마다 그 사례의 기록, 진행 상황, 바뀐 환경; `bench/speed/results.py`). 커밋과 작업 중 변경 여부, 기계(칩, 코어, macOS, 메모리), 라이브러리 판(numpy, scipy, pandas, statsmodels, scikit-learn, R과 그 꾸러미, 비교 대상의 Boost와 sd.special이 담은 Boost), BLAS(numpy·scipy가 빌드된 BLAS·LAPACK, 스레드 변수, R이 쓰는 BLAS·LAPACK과 그 스레드 변수), sd의 기본 스레드 수, 시작할 때의 부하, 사례마다 측정(대상, 스레드, 최솟값, 중앙값, 잡음, 배치 수, 배치당 호출 수, 배치 시간 전부, 결과의 NaN 비율, 점검 실행이면 결과 확인), 사례의 라운드당 시간, 부하, 최대 메모리가 들어간다.
- **기준선** (`bench/speed/results/<영역>.json`): 영역마다 하나, 저장소에 둔다. `promote`(또는 `run --promote`)는 측정 하나(대상, 스레드) 단위로 합친다. 실행이 잰 측정만 바꾸고, 실행에 없거나 실패한 측정(1스레드만 잰 실행의 10스레드, 시간 초과한 R 등)은 기준선 값을 둔다. 그래서 함수 하나, 스레드 수 하나만 다시 재서 넣을 수 있다. 측정마다 어느 실행에서 몇 라운드로 왔는지 적고, 배치 시간 전부는 넣지 않는다(실행 파일에만 있다). 기준선보다 적은 라운드로 잰 측정은 `--force` 없이는 넣지 않는다. 등록부에서 사라진 사례는 뺀다. 끝나지 않은 실행과 `--smoke` 실행은 넣지 않는다.
- **표** (`bench/<영역>/speed.md`): `promote`가 기준선으로 다시 만든다(`table` 명령으로 따로 만들 수도 있다). 손으로 고치지 않는다.
- **로그** (`build/logs/speed/<날짜-시각>.log`, 최근 20개): 콘솔에 나온 줄(시각과 함께), 사례마다 한 줄의 진행, 작업 프로세스의 stderr. 로그 경로는 실행 파일의 `options.log`에도 적는다.

### 1.4 비교

**결과 확인.** `run --smoke`는 사례마다 sd와 비교 대상을 한 번씩 부르고 결과를 견준다(`bench/speed/check.py`). 견주는 상대는 첫 Python 비교 대상과 모든 R·Boost 비교 대상이고, `differs`가 있는 비교 대상은 뺀다. R의 값은 같은 R 세션에서 받아 온다(목록이면 구성 요소별).

- 두 값은 둘 다 NaN이거나, 같거나, 큰 쪽 크기의 1e-6(사례의 `Check.rtol`) 안이면 맞는다. 0 근처에서 다르게 언더플로하는 값(1e-300 안)도 맞는다.
- 한쪽만 NaN인 값은 틀린 것으로 세지 않고 NaN 비율로 따로 적는다. 비교 대상이 sd보다 NaN을 1%p 넘게 더 내면 표의 비고에 그 비율이 붙는다(측정 실행도 예열 호출의 결과로 NaN 비율을 적는다).
- 둘 다 값을 낸 점 중 절반(`Check.agree`) 이상이 맞아야 통과한다. special·stats의 입력은 함수의 어려운 구석에 몰려 있어 비교 대상의 정확도가 떨어지는 점이 섞이지만, 다른 함수를 계산하는 비교 대상(인자 순서가 틀리거나 인자를 반올림하는 것)은 거의 모든 점에서 틀린다.
- 결과 확인이 실패한 사례는 실패로 센다.
- **점검 실행의 병렬**: `--smoke`는 아무것도 재지 않으므로 사례를 여러 갈래로 나눠 동시에 돌린다. 프로세스는 R 세션까지 합쳐 `--jobs`개(기본 8, 최대 10) 이하다. 점검 실행은 사례마다 작업 프로세스 하나에서 sd와 모든 비교 대상을 불러 결과를 견준다. 갈래마다 작업 프로세스 하나를 띄우고, R을 쓰면 갈래 수를 절반으로 줄인다. 메모리 제한은 모든 갈래의 작업 프로세스를 합쳐 본다. 합이 넘으면 메모리를 가장 많이 쥔 작업 프로세스의 사례만 멈추고, 그 사례는 마지막에 혼자 다시 돌린다. 다른 갈래와 함께 돌 때만 넘는 사례가 있어서다. 측정 실행은 늘 사례를 하나씩 잰다.

**기준선과의 비교.** `compare RUN`은 실행의 영역들의 기준선(또는 `--base FILE`)과 비교한다. sd 측정이 기준선보다 10% 넘게 느리면(ROADMAP의 기준선 규칙) 적고, 그 차이가 두 측정의 잡음 폭 중 큰 것보다 크면 `slower`, 아니면 `within noise`다(보고서는 영어다). 비교 대상 라이브러리 시간의 변화(중앙값)도 보인다. numpy·scipy도 같이 느려졌다면 코드가 아니라 기계 상태가 달라진 것이다. `busy`로 적힌 사례(§1.7)는 느려짐으로 세지 않는다. `--faster`는 빨라진 것도 보이고, `--out FILE`은 보고서를 Markdown 표로 쓴다. 실행의 사례가 기준선에 하나도 없으면 비교할 것이 없다고 알린다.

### 1.5 종료 코드

| 코드 | 뜻 |
|---:|---|
| 0 | 모두 됐다 |
| 1 | 실패한 사례가 있다(예외, 시간 초과, 메모리 초과, 점검 실행의 결과 확인 실패; 결과 파일은 남는다), `compare`가 잡음 밖의 느려짐을 찾았거나 비교할 기준선이 없다, `cost --check`가 바뀔 헤더를 찾았다, `--strict`로 바빠서 멈췄다 |
| 2 | 명령행 오류, 넣을 수 없는 실행을 `promote`하려 했다(점검 실행, 끝나지 않은 실행, 적은 라운드) |
| 130 | Ctrl-C로 멈췄다(`--resume`으로 잇는다) |

### 1.6 C++ 비용표

병렬 분할은 원소마다 잰 1스레드 비용으로 일의 양을 센다. 두 표가 기준선에서 생성된다.

- `csrc/stats/cost_table.hpp`: 키 `분포.메서드`(예: `normal.cdf`). stats의 연속·이산·극값 분포 사례가 채운다. 같은 키의 사례가 여럿이면(매개변수 묶음이 여럿) 기하평균이다. 이산 분포의 좁은 지지 사례는 키 `분포.메서드@memo`로 따로 잰다. 스칼라 매개변수에서 정수점 기억 표가 답하는 호출(원소당 1–2 ns)의 비용이고, 표에서 `memo = true`인 줄이 된다. 넓은 지지의 계산(수백 ns–수십 µs)과 섞으면 좁은 지지 호출이 무거운 가중치를 받아 10스레드가 1스레드보다 느려진다. 아직 잰 적 없는 `@memo` 키의 줄은 `bench/speed/cost.py`의 값으로 채워, 첫 측정 전에도 좁은 지지 가중치가 켜진다: 역함수(icdf, iccdf, ilogcdf, ilogccdf)의 24줄은 점검 실행에서 잰 1스레드 시간(`MEMO_PRESET_NS`, 원소당 9–1080 ns), 나머지는 `MEMO_DEFAULT_NS`(1.5 ns)다. 측정하면 잰 값으로 바뀐다.
- `csrc/special/cost_table.hpp`: 키는 함수 이름(예: `erf`, `erf_complex`, `_add`). special의 10⁶ 원소 사례가 채운다. `csrc/special/cost.hpp`는 연산마다 이 표의 어느 이름을 쓰는지만 적는다. 표에 없는 이름은 컴파일되지 않는다.

`cost`는 기준선에 있는 키만 바꾸고 나머지 줄은 지금 파일의 값을 둔다. `cost --check`는 쓰지 않고 바뀔지만 알린다(바뀌면 종료 코드 1). 아직 기준선이 없는 영역, 먹이는 사례가 없어 영영 바뀌지 않는 줄(Bernoulli의 넓은 지지 줄 등), 줄이 없는 사례 키도 함께 알린다. 헤더가 바뀌면 다음 `import stardust`에서 C++가 다시 빌드된다.

### 1.7 긴 실행

전체 재측정은 몇 시간이 걸릴 수 있다. 그래서 실행기는 이렇게 동작한다.

- **측정 프로세스**: 측정은 감독 프로세스가 띄운 작업 프로세스(`worker.py`, 자기 프로세스 그룹)에서 한다. 감독은 사례마다 걸린 시간과, 작업 프로세스와 그 자식(R)의 상주 메모리 합을 2초마다 본다(macOS에서는 libproc로 읽어 측정을 방해할 프로세스를 띄우지 않는다). 사례 하나(한 라운드의 모든 스레드와 비교 대상)가 15분을 넘거나 메모리가 16 GB를 넘으면 그룹을 죽이고 그 사례를 `timeout`이나 `memory`로 적은 뒤 새 작업 프로세스로 다음 사례를 잰다. `--timeout 분`, `--max-rss GB`로 바꾼다. 작업 프로세스는 40사례마다, 그리고 라운드가 바뀔 때마다 새로 띄운다.
- **이어 달리기**: 사례 하나가 끝날 때마다 실행 파일에 그 사례의 줄을 덧붙인다(배치 시간까지). 중단(Ctrl-C, 전원)되면 많아야 사례 하나를 잃는다(끊긴 마지막 줄은 읽을 때 버린다). `run --resume build/speed/<실행>.jsonl`이 남은 곳부터 같은 실행으로 잇는다. `run --rerun-busy`는 원래 실행의 라운드 수, 스레드 수, R 사용 여부를 따른다(명령행에 주면 그 값).
- **예상 시간**: `estimate`는 기준선의 사례별 라운드당 시간으로 고른 사례의 시간을 어림한다. 실행 중에는 진행 막대에 경과와 남은 예상 시간이 나온다. 남은 시간은 지금까지 기준선보다 얼마나 빨랐는지로 고친다.
- **잠자기 막기**: 감독이 살아 있는 동안 `caffeinate -i`가 기계를 깨워 둔다.
- **조용한 기계**: 재기 전에 다른 프로세스가 쓰는 CPU를 3초 동안 직접 잰다(커널의 CPU 시간 계수기와 libproc로 읽어 프로세스를 띄우지 않는다). 1코어(`--quiet-cores`)를 넘게 쓰면 가장 바쁜 프로세스들을 보여 주고, 조용해질 때까지 최대 10분 기다린다. 그래도 바쁘면 경고를 남기고 시작한다. `--no-wait`는 기다리지 않고, `--strict`는 바쁘면 멈춘다(종료 코드 1). `--smoke`는 검사하지 않는다.
- **바쁠 때 잰 사례**: 실행 중에도 사례마다 다른 프로세스가 평균 몇 코어를 썼는지(`other_cpu`) 적는다. 실행 자신(감독, 작업 프로세스, R)의 CPU는 뺀다. 1코어를 넘은 사례는 `busy`로 적고 진행 막대 위에 알린다. 표에서는 이름 옆에 †가 붙고, `compare`는 그 사례를 `busy`로 두어 느려짐으로 세지 않는다. `run --rerun-busy build/speed/<실행>.jsonl`은 그 사례들만 다시 잰다. 그 실행을 `promote`하면 기준선의 그 사례들만 바뀐다.
- **부하**: 1분 부하를 사례마다 적는다(CPU 계수기를 읽을 수 없는 기계에서는 시작할 때 3을 넘으면 경고한다).
- **알림**: 1분 넘게 걸린 실행이 끝나면 macOS 알림을 띄운다(`--no-notify`로 끈다).

### 1.8 콘솔

시작할 때 실행 요약(실행 id, 커밋, 사례·라운드 수, 스레드, 예상 시간, 다른 프로세스의 CPU, 결과·로그 경로)을 78칸 틀로 보인다. 실행 중에는 tqdm 진행 막대(라운드, 비율, 경과 < 남은 예상, 지금 사례)를 보이고, 사례가 끝날 때마다 그 위에 결과를 몇 줄로 적는다. sd의 스레드별 시간(10스레드는 1스레드보다 몇 배 빠른지), 비교 대상마다 그 시간과 'sd is 2.51x faster/slower'(같은 스레드 수의 sd와, 자기 기본 스레드로 도는 비교 대상은 sd의 가장 적은 스레드와 견준다), 바빴으면 그 사실이다. 느린 쪽은 빨간색이다. 줄은 터미널 폭(60–100칸)을 넘지 않게 나눈다. 실패와 중단도 그 위에 한 줄씩 적는다. 끝나면 요약(측정·실패·바쁨 수, 최대 메모리), 영역별 표(사례, 실패, 바쁨, 라운드당 시간), 영역 × 비교 대상 표(사례 수, 비교 대상 ÷ sd의 중앙값, sd가 느린 사례 수), 실패 목록, 번호 붙은 메모(다시 재기·이어 달리기 명령)를 보인다. `list`, `estimate`, `compare`, `promote`, `table`, `cost`도 같은 틀이다. 터미널이 아니면(파일로 돌리면) 막대 대신 사례마다 한 줄을 쓴다.

### 1.9 전체를 다시 잴 때

```
.venv/bin/python -m bench.speed estimate                       # 걸릴 시간
.venv/bin/python -m bench.speed run stats --smoke              # 먼저 모든 사례가 돌고 값이 맞는지
.venv/bin/python -m bench.speed run stats --label stats
.venv/bin/python -m bench.speed compare build/speed/<실행>.jsonl
.venv/bin/python -m bench.speed promote build/speed/<실행>.jsonl
.venv/bin/python -m bench.speed cost                           # stats·special을 다시 쟀으면
.venv/bin/python tools/test.py                                 # 비용표가 바뀌었으면 다시 빌드되고 테스트
```

영역마다 따로 재서 넣어도 된다. 기준선이 아직 없는 영역은 `compare`가 비교할 것이 없고(종료 코드 1), `estimate`는 그 사례들을 "시간 모름"으로 센다.

## 2. 정확도 (`bench/accuracy`)

참값 대비 오차를 함수마다 재서 표로 만들고, 기준선보다 나빠진 것을 찾는다. 참값은 모두 미리 계산해 커밋한 파일이고, 측정은 파일만 읽는다. 측정 때 계산하는 것은 sd(와 비교 대상)의 값과, 결과마다 다른 잔차(기울기, ‖F‖, KKT)뿐이다.

### 2.1 명령

| 파일 | 하는 일 |
|---|---|
| `bench/accuracy/sources.py` | 영역마다의 스윕과 스윕 하나를 재는 법 |
| `bench/accuracy/common.py` | ULP 거리 `ulp_distance`, 영역 하나의 요약, 일반 스윕 |
| `bench/accuracy/array.py`, `linalg.py`, `multivariate.py`, `optimize.py` | 그 영역(stats는 다변량 묶음)의 스윕과 지표 |
| `bench/accuracy/parallel.py` | 작업 프로세스로 스윕을 나눠 재기 |
| `bench/accuracy/compare.py`, `regcompare.py`, `linalg_ops.py` | 비교 대상(scipy·numpy·statsmodels·scikit-learn·R·Boost.Math)의 표 |
| `bench/accuracy/rsession.py` | 작업 프로세스마다 R 프로세스 하나 |
| `bench/accuracy/boost.py`, `boost_comparator.cpp` | Boost.Math 비교 대상의 빌드 (속도 측정과 같은 함수, 같은 옵션) |
| `bench/accuracy/results.py`, `tables.py`, `report.py` | 실행 파일과 기준선, 표 생성, 콘솔 출력 |

```
.venv/bin/python -m bench.accuracy list special --match erf          # 스윕 보기
.venv/bin/python -m bench.accuracy run                               # sd만, 모든 영역
.venv/bin/python -m bench.accuracy run --compare                     # 비교 대상 라이브러리도
.venv/bin/python -m bench.accuracy run special --match erf --jobs 1
.venv/bin/python -m bench.accuracy compare build/accuracy/<실행>.json   # 기준선보다 나빠진 것
.venv/bin/python -m bench.accuracy promote build/accuracy/<실행>.json   # 기준선에 넣고 accuracy.md 다시 만들기
.venv/bin/python -m bench.accuracy table                             # 기준선에서 표만 다시
```

- 영역은 참값이 있는 여섯이다: array, linalg, special, stats(다변량·행렬 분포는 그 안의 묶음, 스윕 id `stats/mv/…`), regression, optimize. 스윕 id는 `영역/이름`이고 `--match`는 그 id에 맞춘다.
- `--jobs`개(기본 8, 최대 10) 작업 프로세스가 스윕을 나눠 잰다. 워커는 sd 1스레드이고 몇 스윕마다 새로 띄운다. 결과는 프로세스 수와 무관하다.
- `run --promote`는 재고 바로 기준선에 넣는다. `-q`는 터미널에 아무것도 쓰지 않는다(로그만).
- 콘솔은 머리 블록, 스윕마다 sd와 비교 대상의 값을 말로 적은 몇 줄(터미널에서는 그 아래 진행 막대), 끝에 요약 블록·영역별 표·번호 붙은 Notes다. 같은 줄이 시각과 함께 `build/logs/accuracy/<시각>.log`에 남는다.

### 2.2 스윕과 참값

스윕 하나는 참값 파일 하나의 메서드 하나다(입력, 참값 hi + lo, 영역, 계약의 배율). 영역마다 `tests/<영역>/refdata.py`가 파일을 읽고 계약 대비 오차를 내며, 테스트와 측정이 같은 함수를 쓴다.

| 영역 | 참값 | 생성기 |
|---|---|---|
| special | `tests/special/ref` | `tools/special_ref.py` |
| stats(단변량) | `tests/stats/ref` | `tools/stats_ref.py` |
| stats(다변량·행렬 분포, 다변량 cdf) | `tests/stats/ref/mv` | `tools/stats_ref.py` (`tools/mvref.py`, `tools/mvcdfref.py`를 부른다) |
| regression | `tests/regression/ref/<종류>.json` | `tools/regression_ref.py` (식은 `tools/regref.py`) |
| array(vForce 15개·pow·복소 abs, 리덕션) | `tests/array/ref` | `tools/array_ref.py` |
| linalg | `tests/linalg/ref`(행렬, 참값, 조건수) | `tools/linalg_ref.py` |
| optimize | `tests/optimize/ref`(`scalar_roots.json`, `scalar_minima.json`, `derivatives.npz`), NIST StRD 인증값 | `tools/optimize_ref.py` |

- 폴더의 뜻: `ref/`는 Arb나 정확한 산술로 만든 참값, `data/`는 바깥에서 가져온 입력과 다른 라이브러리의 기록(R의 회귀 결과 `tests/regression/data/*_r.json`, scipy·Fortran의 optimize 실행 등)이다. 참값 도구는 Arb(python-flint)나 정확한 산술만 쓴다(mpmath는 쓰지 않는다).
- npz 참값 파일은 모두 한 형식이다(`tests/reffile.py`): 점의 입력, 묶음 번호와 묶음의 입력, 메서드마다의 참값(hi + lo)과 그 점, 보장 영역·배율 같은 필드, 출처(`arb`, `target_bits`, `tool`). JSON 참값도 `_provenance`에 같은 출처를 적는다.
- 생성기는 참값이 정해진 정밀도(8192비트)와 반복 상한 안에서 정해지지 않는 점을 버리고, 요청한 점 수(`n_requested`)와 버린 점(`dropped`)을 파일에 적는다. 형식을 바꿔 옮긴 파일은 `n_requested = -1`이다(기록 없음).
- 참값 폴더의 크기 한도는 영역마다 60MiB이고, 영역의 `tests/<영역>/ref`와 `ref_*` 폴더를 합한 크기다. `tools/check.py`가 잰다. 2026-11-20: stats 35.5, array 18.8, special 11.3, linalg 4.3, regression 0.8, optimize 0.07MiB.
- 생성기는 위치 인자로 만들 함수·분포·묶음을 고르고(`--help`), `--jobs`(기본 8, 최대 10; regression 기본 6; optimize는 한 프로세스)를 받는다.

**지표와 계약.**

- 지표는 (스윕, 영역)마다: 최대·99.9% ULP, 배율이 1보다 큰 점이 있는 함수의 ULP/배율(역함수의 max(1, κ), Bessel·Airy·복소 오차함수·wofz의 영점 근처 포락선 배율), 0을 지나는 값의 mixed(|오차|/max(1, |값|), 2⁻⁵² 단위), 최대 상대오차, 계약 대비(1 이하가 계약 안). 영역에 따라 상대오차/(u·조건수), 잔차/(n·eps·척도), 기울기·‖F‖·KKT 잔차, 수렴하지 못한 수를 더한다(표의 범례). 회귀는 참값이 유한한데 sd가 NaN·inf를 낸 점의 수도 센다.
- **ULP**는 어디서나 `ulp_distance`(`bench/accuracy/common.py`)다: 두 값 사이에 있는 double의 수(0이면 같은 값, 1이면 이웃; ±0은 0, NaN끼리 0). 참값과 견줄 때는 참값을 double로 반올림한 값(hi)과의 거리다.
- 계약
  - special·stats(다변량 포함): `tests/<영역>/refdata.py`의 척도(보장 영역은 상대 1e-14, 아주 작은 값은 절대, 역함수는 max(1, κ)를 곱한 한계, 0을 지나는 영역은 mixed). 다변량 t cdf는 d = 2에서 상대 5e-15, d = 3에서 1e-14다.
  - array: 원소별 함수마다 Arb 참값에서의 ULP 한계(`tests/array/refdata.py`의 `ULP_BOUND`, 3–5; 복소 abs 3). numpy와의 거리는 기록만 한다. 리덕션은 계약 없이 기록만 한다.
  - linalg: 잔차/(n·eps·척도) ≤ 10.
  - regression: 선형 모형(ols·wls)의 계수 벡터가 설계 행렬의 κ₂ ≤ 1e11에서 노름 상대오차 1e-14 안이다. 그 밖의 필드는 수치 계약이 없어 기준선과의 비교만 한다.
  - optimize: 결과로 확인되는 멈춤 판정(구간법의 xtol + rtol·|x*|, 기울기 판정의 gtol). 그 밖의 측정(xtol = 0의 ULP, 최적값까지의 거리 등)은 기준선과의 비교만 한다.

**비교 대상**(`--compare`): scipy, numpy(없는 것은 scipy.linalg), statsmodels, scikit-learn, R(base와 MASS·robustbase·glmnet·Matrix·mvtnorm), Boost.Math 단독(`bench/accuracy/boost_comparator.cpp`).

- 같은 참값, 같은 점에서 같은 척도로 잰 그 라이브러리의 오차를 표의 라이브러리별 열에 둔다. 같은 양을 계산하는 공개 함수만 재고(없으면 —), R·Boost가 없으면 not run이다. 참값은 유한한데 그 라이브러리가 NaN·inf를 낸 점은 빼고 센다(제외 열).
- Boost 헤더는 저장소 밖 `build/comparators/boost-math`(또는 `SD_BOOST_MATH`)에 `git clone --depth 1 https://github.com/boostorg/math.git`으로 둔다. 첫 실행에 clang++로 빌드하고, 소스·옵션·체크아웃의 커밋이 바뀌면 다시 빌드한다. 컴파일 옵션은 속도 측정과 같고 부동소수 축약은 끈다. 표에는 비교 대상의 Boost 판과 sd.special이 담은 Boost 판(`third_party/boost_math/README.md`)을 둘 다 적는다.
- optimize는 R의 uniroot·optimize·optim·nls.lm이 reticulate로 같은 Python 시험 함수를 부르고, 반복 상한·허용오차는 R에 같은 설정이 있으면 Python 호출의 기본값에 맞춘다. R의 `qr(LAPACK = TRUE)`는 열을 바꾸는 QR이라 linalg의 `qr_pivoted`와 견준다. R 회귀에서 R이 겹친다고 본 열의 계수는 제외로 세고, glm이 기본 설정으로 시작값을 찾지 못하면 표에 시작값 필요로 적는다.
- 비교 대상의 호출마다 시간·메모리(2GB) 한도가 있고, 넘으면 멈춤이다. 시간 한도는 비교 대상마다 다르고 표의 범례에 적힌다.

### 2.3 결과 파일

- **실행 파일** (`build/accuracy/<시각>-<커밋>[+].json`, git이 무시한다; `+`는 커밋과 다른 작업 트리): 스윕과 영역마다의 지표와 가장 나쁜 점, 비교 대상의 결과, 환경(커밋, 바뀐 파일, 기계, Python, 영역마다 참값 폴더의 SHA-256, 비교 대상의 판). 형식은 `bench/accuracy/results.py`에 있다.
- **기준선** (`bench/accuracy/results/<영역>.json`, 커밋한다): 영역마다 하나. stats의 다변량 묶음도 `stats.json`에 든다. 스윕마다 어느 실행에서 왔는지와 그 실행의 환경을 둔다. `promote`는 실행이 잰 스윕만 바꾸고 나머지는 둔다.
- **표** (`bench/<영역>/accuracy.md`, stats의 다변량 묶음은 `bench/stats/accuracy_multivariate.md`): `promote`나 `table`이 기준선에서 만든다. 손으로 고치지 않는다.

### 2.4 비교

`compare RUN`은 실행을 그 영역들의 기준선과 견준다. 계약을 넘는 것, 계약 안이라도 sd의 어느 지표든 기준선보다 커진 것(값이 없어지거나 NaN이 된 것 포함), 없어진 영역이나 스윕, 기준선보다 점이 적은 영역을 모두 `WORSE`로 적는다. 비교 대상의 변화는 `CHANGED`로 알리기만 하고 종료 코드에 넣지 않는다. 기준선이 없으면 견줄 것이 없다고 알린다. deep 등급의 `tests/bench/test_accuracy_baseline.py`가 sd만으로 `run`과 `compare`를 돌린다.

### 2.5 종료 코드

| 코드 | 뜻 |
|---:|---|
| 0 | 모두 됐다 |
| 1 | 계약을 넘는 스윕이 있다(`run`, `compare`), sd의 지표가 기준선보다 나빠졌거나 영역·스윕이 없어졌다, 견줄 기준선이 없다(`compare`) |
| 2 | 명령행 오류나 도구 자신의 오류 |
