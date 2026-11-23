# differentiate 속도

이 파일은 `build/speed/20261122-051631-c8-after.jsonl`에서 만든 것이다. 손으로 고치지 않는다. 다시 만들기: `.venv/bin/python -m bench.speed table differentiate` (재는 법은 bench/README.md).

- 시간: 한 번 부를 때의 시간. 한 배치가 20 ms 이상이 되도록 여러 번 부르고, 모든 라운드의 배치 중 가장 빠른 값이다.
- 배율(÷ sd): 비교 대상 시간 ÷ sd 시간. 1보다 크면 sd가 빠르다. **굵은 배율**은 sd가 느린 줄이다.
- 잡음: 그 줄의 측정마다 (배치 시간의 중앙값 ÷ 최솟값 − 1)을 구한 것 중 가장 큰 값.
- 스레드: sd의 스레드 수. 한 줄에 sd 스레드별 열이 있는 표에서 비교 대상은 그 라이브러리의 기본 스레드로 한 번 잰다(numpy·scipy의 원소별 계산은 1스레드). BLAS·LAPACK이 일의 대부분인 비교 대상은 BLAS 스레드를 sd와 같게 묶어 '스레드' 열이 있는 표로 보인다. R은 늘 1스레드다.
- 비고: 비교 대상이 sd보다 NaN을 1%p 넘게 더 내면 그 비율을 적는다(그만큼 일찍 끝나는 호출이 섞인다).

## 측정 환경

| 실행 | 커밋 | 기계 | 사례 수 |
|---|---|---|---:|
| 20261122-051631-895d55c+ | 895d55c + 작업 중 변경 | Apple M4 Pro, 14코어, macOS 26.6.2 | 10 |

라이브러리: stardust 0.0.1, numpy 2.5.4, scipy 1.18.1, pandas 3.1.1, pyarrow 26.0.1, statsmodels 0.15.0, sklearn 1.9.1, threadpoolctl 3.8.0, Boost.Math boostorg/math 75cb4c4 2026-11-18.

BLAS: numpy accelerate, scipy Accelerate.

## 비교 대상을 잰 방법

- numpy, scipy, pandas, statsmodels, scikit-learn: 같은 프로세스에서 Python으로 부르고 sd와 같은 규칙으로 잰다. 입력은 그 라이브러리의 배열(numpy)로 미리 만들어 둔다.

## 배치 도함수

- scipy ÷ sd: 1스레드 중앙값 8.50배(기하평균 8.96배, sd가 느린 줄 0/4), 10스레드 중앙값 12.1배(기하평균 12.4배, sd가 느린 줄 0/4)

| 사례 | 크기 | sd 1스레드 | sd 10스레드 | scipy | ÷ sd 1 | ÷ sd 10 | 잡음 |
|---|---|---:|---:|---:|---:|---:|---:|
| `derivative exp` | 10 | 10.9 µs | 10.7 µs | 223 µs | 20.5 | 20.8 | 2% |
| `derivative exp` | 10^3 | 50.4 µs | 51.3 µs | 429 µs | 8.50 | 8.36 | 2% |
| `derivative exp` | 10^5 | 3.94 ms | 1.10 ms | 17.1 ms | 4.35 | 15.5 | 2% |
| `derivative exp one-sided` | 10^3 | 47.2 µs | 46.4 µs | 401 µs | 8.50 | 8.64 | 2% |

## 야코비안과 헤시안

- scipy ÷ sd: 1스레드 중앙값 16.1배(기하평균 16.9배, sd가 느린 줄 0/6), 10스레드 중앙값 16.0배(기하평균 16.8배, sd가 느린 줄 0/6)

| 사례 | 크기 | sd 1스레드 | sd 10스레드 | scipy | ÷ sd 1 | ÷ sd 10 | 잡음 | 비고 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `jacobian Rosenbrock` | n=3 | 15.6 µs | 15.5 µs | 270 µs | 17.3 | 17.4 | 1% |  |
| `jacobian Rosenbrock` | n=10 | 17.7 µs | 17.6 µs | 276 µs | 15.6 | 15.7 | 1% |  |
| `jacobian Rosenbrock` † | n=30 | 31.7 µs | 32.1 µs | 295 µs | 9.30 | 9.19 | 2% |  |
| `hessian Rosenbrock` | n=3 | 65.6 µs | 66.0 µs | 4.06 ms | 61.9 | 61.4 | 2% | 33 calls |
| `hessian Rosenbrock` | n=10 | 572 µs | 578 µs | 5.11 ms | 8.93 | 8.84 | 1% |  |
| `jacobian vector` | n=10 | 17.5 µs | 17.9 µs | 292 µs | 16.7 | 16.3 | 1% | f: R^10 → R^10 |

† 잴 때 다른 프로세스가 CPU를 1코어 넘게 썼다(사례의 `other_cpu`). 다시 재는 것이 좋다.
