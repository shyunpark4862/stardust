# 옛 문서 참조 → linalg.md (임시)

> 코드와 테스트의 옛 문서 참조를 `docs/design/linalg.md`로 바꾸는 데 쓰는 표다. 바꾸기가 끝나면 지운다 (ROADMAP M7).
>
> - 옛 문서: `stardust-linalg/DESIGN.md`, `stardust-linalg/ARCHITECTURE.md` (동결된 옛 저장소)
> - "→ §N"은 `linalg.md §N`이다. 여러 개면 첫 번째가 주된 자리다. 루트 문서는 "루트 DESIGN §N"처럼 적는다.
> - 옛 `API.md §N` 참조는 이 표의 대상이 아니다. `docs/api/linalg.md`의 해당 절(공통 규칙이면 루트 `API.md`)로 바꾼다.
> - stardust-numpy 문서를 인용한 linalg 주석은 5절에 있다.

## 1. 옛 DESIGN.md

| 옛 참조 | 내용 | 새 자리 |
|---|---|---|
| DESIGN 머리말 "원칙" | 일관된 시그니처와 동작, 가급적 효율적으로 | §1.3 (라이브러리 전체는 루트 DESIGN §1.4) |
| DESIGN §1.1 (1라운드 Q1) | stardust-numpy에 의존하는 별도 패키지 | 대응 없음: 폐기 (루트 DESIGN §2). `matmul`을 다시 만들지 않는다는 부분만 §1.1 |
| DESIGN §1.2 (Q2) | Accelerate LAPACK을 감싼다, expm은 직접, 확인한 루틴 목록 | §1.2 |
| DESIGN §1.3 | 이름 `stardust-linalg`, `sdla` | 대응 없음: 루트 DESIGN §1.1, §2 |
| DESIGN §1.4 | 공통 규칙: 배치, dtype, 경고 없음, Array API 이름 | §2.1 (경고 §2.1, 이름 §2.6) |
| DESIGN §1.5 (2라운드 R1) | 빌드 환경, ILP64 | §1.2 (인터페이스). 빌드는 루트 DESIGN §1.5, ARCHITECTURE §4.4, §5 |
| DESIGN §1.6 (R7) | git 정책 | 대응 없음: 루트 DESIGN §5.2 |
| DESIGN §2.1 | 범위 (기본, 확장, 제외) | §1.1 |
| DESIGN §2.2 (R2) | 양방향 buffer protocol | §3 (M3의 모습), 지금의 경로는 §3.6. 폐기는 루트 DESIGN §2, §3.1 |
| DESIGN §2.3 (R3) | 메모리 레이아웃: 입력 F-order 복사, 출력 F-order 그대로 | §4 (입력 패킹은 §5, 기각한 트릭은 §4, §5.7) |
| DESIGN §2.4 (R4) | 실패할 때의 동작 | §2.2 (에러), §2.3 (유한성), §2.5 (빈 행렬), §6.5 (배치 인덱스) |
| DESIGN §2.5 (R5) | 배치 병렬화, 풀 복사, 스레드 수, 결정성 | §7, §8 (스레드 수 §3.5, 결정성 §8.5). 풀 복사는 폐기: 루트 ARCHITECTURE §4.1, §8.1 |
| DESIGN §2.6 (R6) | 비교 기준 (numpy, scipy), 잔차 검사 | §12.1 (테스트는 §13) |
| DESIGN §2.7 (R8) | 함수마다 드라이버 하나, `driver=` 없음 | §1.3 (드라이버 표는 §10.1) |
| DESIGN §2.8 (3라운드) | 시그니처 결정과 이유, 기준 라이브러리의 불일치 | §2.6 (시그니처 자체는 `docs/api/linalg.md`) |
| S1 | 대각 오프셋 `k` | §2.6 |
| S2 | `svd(mode=)`, 기본 `'reduced'` | §2.6 |
| S3 | 삼각 규칙: 아래 삼각만 읽기, `upper=False`, `solve_triangular`의 `upper` 필수 | §2.6 (읽는 원소의 검사는 §2.3) |
| S4 | `eig`의 고유값 정렬 (실수부, 허수부) | §2.6 (구현은 §10.6) |
| S5 | `cholesky_solve`, `lu_solve`에 분해 결과를 그대로 | §2.6 (구현은 §10.2, §10.3) |
| S6 | `lstsq`는 `(solution, rank, singular_values)` | §2.6 (드라이버 근거는 §10.4) |
| S7 | 순열은 int64 `perm`, tuple의 마지막 | §2.6 |
| S8 | 결과 형태가 옵션으로 바뀌지 않는다 | §2.6 |
| S9 | `eigh(a, b=None, /)` | §2.6 (구현은 §10.5) |
| S10 | 배열 인자는 `sd.ndarray`만, `outer` 배치, `diagonal` 복사본 | §2.6 (입력은 §3.2, `diagonal`은 §10.8) |
| S11 | 빈 행렬의 관례, 배치 크기 0, `cond`만 ValueError | §2.5 |
| DESIGN §2.9 | M0 설계 검증 (전체) | 아래 행들 |
| (§2.9 환경) | numpy 2.5.4, scipy 1.18.1, 둘 다 Accelerate | §12.1 |
| (§2.9 실험 1, 버퍼 왕복) | 복사 없음, stride 유지, 수명 | §3.6 (M2 현재). F-order 결과의 이후 연산 속도는 §4 |
| (§2.9 실험 2, 고정 비용) | asarray 113ns, 블록 82ns, namedtuple 77ns, `get_num_threads` 7ns, `GetBuffer` 5ns, numpy 호출 시간 | §3.6 (namedtuple은 §3.3) |
| (§2.9 실험 3, 동시 호출) | 10스레드 동시 호출에서 비트가 같다 | §1.2 |
| (§2.9 실험 4, 결정성) | 스레드와 `BLASSetThreading`이 비트를 바꾸지 않는다 | §1.2, §8.5 |
| (§2.9 실험 5, 내부 스레드) | `getrf`의 n ≥ 768만 내부 스레드 | §1.2 (배치 분할에 주는 영향은 §7.3) |
| DESIGN §2.10 | M1에서 정한 것 | 아래 행들 |
| (§2.10 `solve`) | `getrf` + `getrs` | §10.2 |
| (§2.10 `inv`) | 단위행렬 우변 `getrs` | §10.2 |
| (§2.10 c = 10) | 잔차 허용오차 | §13 |
| (§2.10 배치 실패의 보고) | 가장 작은 인덱스 | §6.5 |
| (§2.10 에러를 고르는 순서) | | §2.4 |
| DESIGN §2.11 | M2에서 정한 것 | 아래 행들 |
| (§2.11 `diagonal`의 dtype) | 계산하지 않는 함수는 dtype 유지 | §2.1, §10.8 |
| (§2.11 `matrix_power`의 NaN 검사) | 호출 단위 규칙 | §2.3 |
| (§2.11 `trace(dtype=)`) | `sd.sum`과 같은 명시적 변환 | §2.1 |
| (§2.11 complex 원소 곱) | numpy와 같은 FMA 식 | §10.8 |
| (§2.11 p-노름의 재조정) | 2⁻⁹⁶⁹, 두 번째 패스 | §10.8 |
| (§2.11 노름의 특수값) | | §10.8 (빈 집합은 §2.5) |
| (§2.11 `vecdot`의 축) | | §10.8 |
| DESIGN §2.12 | M3에서 정한 것 | 아래 행들 |
| (§2.12 삼각만 읽고 검사) | | §2.3, §5.4 |
| (§2.12 `cholesky(upper=True)`) | Lᴴ | §10.3 |
| (§2.12 `lu`는 특이에서도 에러 없음) | | §2.2, §10.2 |
| (§2.12 삼각 인수의 대각 0) | | §2.2 (구현 §10.3) |
| (§2.12 에러 순서) | | §2.4 |
| (§2.12 `lu_solve`) | | §10.2 |
| (§2.12 complex 피벗 경계 √2) | | §10.2 |
| (§2.12 레이아웃 불변성) | | §5.7 |
| (§2.12 기준선, C-order 패킹의 캐시 문제) | | §5.3 (해결은 §5.6) |
| DESIGN §2.13 | M4에서 정한 것 | 아래 행들 |
| (§2.13 `gelsd`) | | §10.4 |
| (§2.13 `rtol ≥ 1`, k = 0, `rtol` 해석) | | §10.4 (정의는 §2.6) |
| (§2.13 `qr`, `qr_pivoted`) | | §10.4 |
| (§2.13 기준선, workspace 질의) | | §6.3 |
| DESIGN §2.14 | M5에서 정한 것 | 아래 행들 |
| (§2.14 드라이버 표) | | §10.1 |
| (§2.14 값 전용 모드) | | §10.5 |
| (§2.14 `cond` 특이 → inf) | | §10.5 (규칙은 §2.2) |
| (§2.14 `matrix_norm` 2/−2/nuc 검사) | | §2.3 |
| (§2.14 `matrix_rank`, `pinv` 허용오차) | | §10.5 |
| (§2.14 `eigh(a, b)`) | | §10.5 |
| (§2.14 수렴 실패 메시지) | | §2.2 |
| (§2.14 numpy와의 비트 비교, `gesdd`의 `lwork`) | | §10.5, §12.2 (workspace 재사용의 조건은 §6.3) |
| DESIGN §2.15 | M6에서 정한 것 | 아래 행들 |
| (§2.15 드라이버, `sytrf` 대 rook) | | §10.1, §10.6 |
| (§2.15 `eig`의 결과, 켤레쌍 조립) | | §10.6 |
| (§2.15 `ldl`) | | §10.6 |
| (§2.15 `hessenberg` n ≤ 2) | | §10.6 |
| (§2.15 "벤치마크는 두 번 이상") | | §12.3 |
| DESIGN §2.16 | M7에서 정한 것 | 아래 행들 |
| (§2.16 드라이버, `trsyl3`) | | §10.7 |
| (§2.16 (1, 1)은 `gtsv`) | 드라이버 규칙의 예외 | §10.7 (원칙은 §1.3) |
| (§2.16 공통 고유값 → `LinAlgError`) | | §10.7, §2.2 |
| (§2.16 `scale`) | | §10.7 |
| (§2.16 읽는 원소만, 띠 저장) | | §2.3, §10.7 |
| (§2.16 검사 순서) | | §2.4 |
| (§2.16 모양, 결과 dtype, scipy의 실수 Schur 버그) | | §10.7 |
| (§2.16 기준선 스크립트) | 고정된 줄 묶음 | §12.3 |
| DESIGN §2.17 | M8에서 정한 것 | 아래 행들 |
| (§2.17 pairwise 합) | | §8.2 |
| (§2.17 16바이트 정렬) | | §6.4 |
| (§2.17 workspace를 한 번만) | | §6.3 |
| (§2.17 scratch를 0으로 채우지 않음) | | §6.3 |
| (§2.17 패킹의 빠른 경로) | | §5.3 |
| (§2.17 읽기만 하는 삼각은 그 자리에서) | | §5.5 |
| (§2.17 그 밖의 작은 고정비) | | §9 |
| (§2.17 하지 않은 것: 결과 배열의 고정비) | | §3.6 (M3에서 없앤다) |
| (§2.17 검증, `m8_bits`) | | §13 (비트 동일성 테스트) |
| DESIGN §2.18 | M9에서 정한 것 | 아래 행들 |
| (§2.18 스레드풀 복사, 스레드 수 동기화) | | 대응 없음: 풀은 하나다 (§8.1, 루트 ARCHITECTURE §4.1). 스레드 수는 §3.5 |
| (§2.18 `_set_tuning`) | | §7.7 |
| (§2.18 계열별 고정 상수) | | §7.1 |
| (§2.18 두 상수) | | §7.2 |
| (§2.18 작은 띠의 `gbsv`, 비용 0) | | §7.4 |
| (§2.18 짧은 합에 할당 없음) | | §8.3 |
| (§2.18 큰 행렬도 나눈다) | | §7.3 |
| (§2.18 결정성, 실패) | | §8.5, §6.5 |
| (§2.18 출력 하나의 긴 합) | | §8.3 |
| (§2.18 확장성 관찰, AMX) | | §7.6 |
| (§2.18 설명되지 않은 테스트 실패) | | §13 ("남은 위험") |
| DESIGN §2.19 | M10에서 정한 것 | 아래 행들 |
| (§2.19 `kron`, `cross`) | | §10.8 (numpy와 다른 점은 §2.7) |
| (§2.19 `expm` 알고리즘, θ₁₃, `I + 2(V−U)⁻¹U`, 2¹⁰⁰ 가드) | | §11.1, §11.4 |
| (§2.19 대각·삼각 입력) | | §11.3 |
| (§2.19 노름: 정확히, 상한 먼저, `lacn2`) | | §11.2 |
| (§2.19 복소수 절댓값, 누산기 8개) | | §11.2 |
| (§2.19 `kExpm`) | | §7.1 |
| (§2.19 정확도) | | §11.5 |
| (§2.19 실패와 특수값) | | §11.4 |
| (§2.19 결과, 측정 주의) | | §11.5, §12.3 |
| DESIGN §2.20 | M11 | 아래 행들 |
| (§2.20 단계 비교, 번갈아 재기) | | §12.3 |
| (§2.20 결과) | | §12.4 (지금 위치만) |
| (§2.20 API 표면 테스트) | | §13 |
| DESIGN §2.21 | M11 이후 최적화 세 가지 | 아래 행들 |
| (§2.21 Schur 분해 동시에, `kSchurPairMin`) | | §8.4 |
| (§2.21 LU 계열의 태스크 수 상한) | | §7.5 |
| (§2.21 큰 C-order 전치를 풀에서, 2¹⁷) | | §5.6 |
| (§2.21 editable 설치) | | §12.3 |
| (§2.21 결정이 필요해서 하지 않은 것) | | §12.5 |
| DESIGN §2.22 | 코드리뷰에서 찾은 버그 | 아래 행들 |
| (§2.22 `expm` 비정규 노름, inf 노름) | | §11.4 |
| (§2.22 큰 정수 `ord`) | | §10.5 |
| (§2.22 `d8` 상한) | | §11.2 |
| (§2.22 고치지 않은 6건) | | §10.9 |
| DESIGN §3 | 미확정 | 대응 없음 (아래 3절). randomized SVD는 §1.1 |

## 2. 옛 ARCHITECTURE.md

| 옛 참조 | 내용 | 새 자리 |
|---|---|---|
| ARCHITECTURE §1 (A1) | Python과 C++의 경계, 래퍼 비용, Python 객체 캐시 | §3.1 (캐시는 §3.4, 지금의 캐시는 §3.6) |
| ARCHITECTURE §2 | 디렉터리 구조 | §1.4 (경로는 루트 ARCHITECTURE §2, §2.1) |
| ARCHITECTURE §3 | 데이터 흐름 | §6 (단계 목록), §3, §5 |
| ARCHITECTURE §3.1 | InputView (`Input`) | §3.2 (M3의 모습), 지금은 §3.6 |
| ARCHITECTURE §3.2 | 패킹 (한 패스, 출력 자리에 직접, 그 자리에서 읽기) | §5 (§5.1, §5.2, §5.5) |
| ARCHITECTURE §3.3 | OutBlock, 0차원 결과는 스칼라 | §3.3 (M3의 모습), 지금은 §3.6 |
| ARCHITECTURE §3.4 | 배치 드라이버 | §6 |
| (§3.4 workspace) | | §6.3 |
| (§3.4 정렬) | | §6.4 |
| (§3.4 실패 보고, A2) | | §6.5 |
| A2 | 가장 작은 실패 인덱스 | §6.5 |
| ARCHITECTURE §3.5 | LAPACK 래퍼, cblas 직접 호출 | §1.2 |
| ARCHITECTURE §4 | 스레드풀 | §8.1 (풀 자체는 array.md §10) |
| (§4 스레드 수) | | §3.5 |
| (§4 배치 분할, 비용 모델) | | §7 |
| (§4 출력 하나의 긴 합) | | §8.3 |
| (§4 M11 이후의 세 가지) | | §7.5, §5.6, §8.4 |
| (§4 두 개의 풀) | | 대응 없음 (아래 3절) |
| (§4 M0 결과) | | §1.2 |
| ARCHITECTURE §5 | 함수별 메모 | §10 (드라이버 표 §10.1) |
| (§5 `expm`) | | §11 |
| (§5 LAPACK을 쓰지 않는 함수) | | §10.8 |
| ARCHITECTURE §6 | 테스트와 벤치마크, 잔차 `c` | §13 (벤치마크는 §12) |

## 3. 대응이 없는 옛 항목

| 옛 항목 | 이유 |
|---|---|
| DESIGN §1.1, §1.3 (별도 패키지, `sdla`) | 폐기되었다. 루트 DESIGN §1.1, §2가 대신한다 |
| DESIGN §1.6 (git 정책: 문서는 로컬 전용) | 루트 DESIGN §5.2가 대신한다 (모두 git에 넣는다) |
| DESIGN §2.5, §2.18, ARCHITECTURE §4의 "스레드풀을 복사한다", "두 개의 풀" | 풀은 하나다 (루트 ARCHITECTURE §4.1). linalg는 `parallel/pool.*`를 쓴다 |
| DESIGN §2.5, ARCHITECTURE §4의 "기준 크기 이상이면 Accelerate 내부 스레드에 맡긴다" | 측정으로 뒤집혔다. 크기 기준이 없다 (§7.3) |
| DESIGN §2.9 실험 5의 "`solve`를 `getrf + getrs`로 만들면 빠를 수 있다, M1에서 정한다" | 결정되었다 (§10.2) |
| DESIGN §2.10의 "M1의 배치 루프는 직렬이므로 첫 실패에서 멈춘다" | 지금은 나눈 배치도 같은 규칙이다 (§6.5) |
| DESIGN §2.11의 "`matrix_norm`의 2, −2, nuc는 M5 전까지 `NotImplementedError`" | 이력이다. 지금은 구현되어 있다 (§10.5) |
| DESIGN §2.11–§2.19의 "M_n 기준선 측정" 수치 (예: M2의 `vecdot` 0.06배, M7의 109/80/40, M8의 124/77/28, M9의 169/50/10) | 이력이다. 지금 위치만 §12.4에 있다. 각 수치가 근거였던 결정은 해당 절에 남겼다 |
| DESIGN §2.16의 "M8 목록" | 이력이다. 그 항목들의 결과는 §5.6, §6.3, §8.2, §8.3, §9에 있다 |
| DESIGN §2.12의 "`l`과 `u`를 `getrf` 압축 형식으로 다시 붙여 `getrs`" | 지금 코드와 다르다. `trtrs` 두 번이 지금의 모습이다 (§10.2) |
| DESIGN §2.20의 "회귀만 고치고 나머지는 기록한다" | 옛 프로젝트의 마일스톤 운영 결정이다. 통합의 회귀 기준은 ROADMAP 진행 원칙 |
| DESIGN §2.22의 "코드와 어긋난 주석을 고쳤다", "문서도 맞췄다" | 이력이다 |
| DESIGN §3 "미확정"과 "설계 문서는 모두 확정되었다" | 이력 기록이다. 남은 미확정(randomized SVD)은 §1.1 |
| 각 절의 라운드 번호(1라운드, 2라운드 R1–R8, 3라운드, Q1, Q2), "사용자 결정 2026-11-11", "추천안 수락" | 확정 절차의 이력이다 (DESIGN §5.4) |
| DESIGN의 실험 경로(`experiments/m*_*/`, `out/*.txt`)와 `bench/results/*` | 옛 저장소에 남는다. 수치와 근거만 옮겼다 (4절의 표) |
| ARCHITECTURE 머리말 "A1, A2는 추천안을 수락했다" | 이력이다 |
| ARCHITECTURE §3.2의 "M1–M7에서는 똑같이 패킹한다, M8에서 측정한다" | 이력이다. 지금은 §5.5 |
| ARCHITECTURE §3.3의 "M0에서 확인할 것" | 확인이 끝났다 (§3.6) |
| ARCHITECTURE §3.4의 "M1–M7: 호출할 때마다 크기를 묻는다" | 이력이다. 지금은 §6.3 |
| ARCHITECTURE §4의 "호출마다 `sdnp.get_num_threads()`를 부른다" | 지금 코드와 다르다. 나누기로 할 때만 부르고(`begin_parallel`), M3에서는 `parallel::num_threads()`를 직접 읽는다 (§3.5, §3.6) |

## 4. 코드에만 있는 표시

### 4.1 ID와 결정 표시
| 코드의 표시 | 뜻 | 새 자리 |
|---|---|---|
| "A2" (`batch.hpp`) | 가장 작은 실패 인덱스 | §6.5 |
| "S3" (`test_m3_factor.py`) | `upper` 필수 | §2.6 |
| "S4" (`nonsym.cpp`) | `eig`의 complex 순서 | §2.6, §10.6 (배열의 순서는 array.md §2.6) |
| "S4" (`types.hpp`, "complex -> real ... (S4)") | stardust-numpy의 S4: complex → 실수는 TypeError | array.md §2.4 |
| "S6" (`qr.cpp`) | `lstsq`의 결과 | §2.6, §10.4 |
| "S10" (`basic.cpp`) | `outer` 배치, `diagonal` 복사본 | §2.6 |
| "S11" (`norms.cpp`, `vnorm.hpp`) | 빈 집합의 노름 0 | §2.5 |
| "Z1" (`outblock.hpp`) | 0차원 결과는 Python 스칼라 | §3.3 (규칙은 array.md §2.2) |
| "user decision" (`pairwise.hpp`, `cost.hpp`, `special.cpp`, `expm.cpp`, `basic.cpp`, `test_m8_opt.py`) | | 각각 §8.2·§8.3, §7.1, §10.7, §11.2·§11.3, §10.8(`kron`) |

### 4.2 마일스톤 표시 → 측정 근거의 자리
DESIGN.md §5.4에 따라 주석의 마일스톤 번호는 지우고, 측정 근거가 필요하면 아래 절을 가리킨다.

| 표시 | 새 자리 |
|---|---|
| M1 | §10.2 (`solve`, `inv`), §13 (c = 10) |
| M2 | §10.8 |
| M3 | §5.4, §10.2, §10.3 |
| M4 | §10.4 |
| M5 | §10.1, §10.5 |
| M6 | §10.6 |
| M7 | §10.7 |
| M8 | pairwise §8.2, 정렬 §6.4, workspace와 0 채우기 §6.3, 패킹 빠른 경로 §5.3, 그 자리에서 읽기 §5.5, 고정비 §9 |
| M9 | 비용 모델 §7.1, 두 상수 §7.2, 띠 §7.4, 긴 합 §8.3, 실패 §6.5 |
| M10 | `expm` §11, `kron`/`cross` §10.8, `kExpm` §7.1 |
| M11 | API 표면 테스트 §13, 측정 방법 §12.3 |
| "after M11" | 태스크 수 상한 §7.5, 큰 C-order 패킹 §5.6, Schur 쌍 §8.4 |
| `ROADMAP.md M1`–`M10` (ops 파일 머리말) | 지운다. 함수 계열은 §10의 절이 대신한다 |

### 4.3 옛 저장소의 실험·문서 경로 → 근거의 자리
| 코드의 경로 | 새 자리 |
|---|---|
| `experiments/m1_solve_driver` | §10.2 |
| `experiments/m4_lstsq_driver` | §10.4 |
| `experiments/m5_drivers` | §10.1, §10.5 |
| `experiments/m6_ldl_driver` | §10.6 |
| `experiments/m7_drivers` | §10.7 |
| `experiments/m8_bits` | §13 |
| `experiments/m9_threshold` (`out/percost.txt`, `split.txt`, `tune.txt`) | §7.1, §7.2 |
| `experiments/m9_threshold/out/band.txt` | §7.4 |
| `experiments/m9_threshold/out/large.txt`, `scaling.txt` | §7.3, §7.6 |
| `experiments/m10_expm` (`out/accuracy.txt`, `timing.txt`, `estimate.txt`, `percost.txt`) | §11.5, §11.2, §11.2, §7.1 |
| `experiments/opt_impl/out/schur_pair.txt` | §8.4 |
| `experiments/opt_impl/out/lu_scaling.txt`, `lu_scaling_k.txt`, `cap_check.txt`, `experiments/opt_analysis/out/lapack_scaling.txt` | §7.5 |
| `experiments/opt_impl/out/pack.txt` | §5.6 |
| `docs/learn/next-optimizations.md §1` | §8.4 |
| `docs/learn/next-optimizations.md §8` | §5.6 (하지 않은 것은 §12.5) |
| `docs/learn/M11-final-report.md` | §12.4 |
| 안내서 "11장 §7.5" (`expm`의 극단값 측정) | §11.4 |

## 5. stardust-numpy 문서 참조 → array.md
linalg 코드와 옛 linalg 문서가 인용한 stardust-numpy 문서다. `docs/design/array-refmap.md`, `docs/api/array-refmap.md`를 따른다.

| 옛 참조 (인용한 곳) | 내용 | 새 자리 |
|---|---|---|
| stardust-numpy ARCHITECTURE §6.5 (`input.cpp`, `outblock.cpp`, 옛 DESIGN §2.2, ARCHITECTURE §3.1, §3.3) | 버퍼 프로토콜: 내보내는 format, 복사 없이 가져오기 | array.md §12.3 |
| stardust-numpy ARCHITECTURE §6.4 (옛 DESIGN §1.5, §2.3, §2.5) | matmul: ILP64, F-order 입력의 전치 플래그, 배치 | array.md §9 (ILP64 §1.3, 레이아웃 §9.2, 배치 §9.4) |
| stardust-numpy ARCHITECTURE:125-126 (옛 DESIGN §2.3) | 이터레이터의 축 재정렬과 합치기 | array.md §4.2 |
| stardust-numpy ARCHITECTURE §0 (옛 ARCHITECTURE 머리말) | 하드웨어 사실 | array.md §1.1 |
| stardust-numpy DESIGN S3 (`types.hpp`, 옛 DESIGN §2.11) | float → int 포화, NaN → 0 | array.md §2.4 |
| stardust-numpy DESIGN S4 (`types.hpp`) | complex → 실수 TypeError | array.md §2.4 |
| stardust-numpy API §6 (`basic.cpp`, 옛 DESIGN §2.11) | 리덕션의 `dtype=` | `docs/api/array.md` §7 |
| stardust-numpy `scalar_ops.hpp` cmul (`types.hpp`, 옛 DESIGN §2.11) | complex 곱의 FMA 식 | array.md §5.3 (`array/kernels/scalar_ops.hpp`) |
| stardust-numpy의 `reduce.hpp`, M8 (`pairwise.hpp`) | pairwise 합과 그 병렬화 | array.md §6.2 |
| stardust-numpy의 `kThreshold`, `kTaskWork` (`pairwise.hpp`) | 2¹⁵, 2¹² | array.md §10.4 |
| "stardust-numpy's 32 axes" (`vnorm.hpp`) | 최대 32차원 | array.md §3.3 |
| "as stardust-numpy's own buffers", 128B (`outblock.cpp`) | 할당 정렬 | array.md §3.2 |
| "bool is rejected (as in stardust-numpy)" (`args.hpp`) | Python bool은 int 인자가 아니다 | array.md §2.8 (`bind/pyutil.hpp`의 인자 해석) |
| `sdnp.get_num_threads`, `sdnp.set_num_threads` (`batch.cpp`, `pyrefs.*`, 옛 DESIGN §2.5) | 스레드 수 API | array.md §10.3, 루트 API.md §3 |
| stardust-numpy M7 (NEON pairwise 잎, 옛 M11 보고서) | | array.md §5.2, §6.2 (linalg 쪽 후보는 §12.5) |
| stardust-numpy의 풀(`csrc/parallel/pool.*`, 옛 DESIGN §2.18, ARCHITECTURE §4) | 상주 워커, spin, 중첩 금지, GIL | array.md §10.1, §10.2 |
