# 테스트 등급·마커·hypothesis 정리 계획

상태: 적용(2026-11-20). §6의 1·2는 사용자 결정대로(quick `derandomize=True`, deep 10배·기본 1000) 적용했고, 3은 도구로만 둔다(`external` 마커는 등록만 하고 비어 있다), 4는 quick에 둔다(serial 쪽에서 돈다; 사용자 결정 2026-11-20). 적용하면서 바뀐 것: `serial`은 파일에 마커를 다는 대신 `tests/conftest.py`의 `SERIAL` 모듈 목록으로 정했고(다른 세션의 파일을 건드리지 않으려고), 스레드 수 테스트는 시간을 재지 않아 serial에서 뺐다(§5 표의 xdist 열보다 좁다). 지금의 사용법은 [README.md](README.md) §1이다. 아래는 계획 당시의 기록이다. 수치는 같은 날 포매터 적용 뒤(89320e0) 평소 빌드에서 잰 것이다: 전체 7,728개 통과, 벽시계 153초, 테스트 시간의 합 144초, 최대 RSS 3.7GB.

## 1. 등급

| 등급 | 무엇 | 언제 | 목표 시간 |
|---|---|---|---|
| `quick` | 마커가 없는 모든 테스트. hypothesis는 지금 예제 수 | 기본(`pytest`만 치면) | 2분 미만 |
| `deep` | `@pytest.mark.deep` 테스트 + hypothesis 예제 10배 + (다음 단계) 정확도 측정 스윕 | `SD_TIER=deep` | 제한 없음 |
| `external` | R이나 Fortran 컴파일러를 실제로 부르는 테스트 | `SD_TIER=all` | 제한 없음 |

- `deep`은 `quick`을 포함하고 `all`은 셋 다다.
- 지금은 R·Fortran을 실행 중에 부르는 테스트가 하나도 없다(R·gfortran의 결과는 `tests/*/data`, `tests/*/ref`에 기록되어 있고 테스트는 그 기록만 읽는다). 그래서 `external`은 처음엔 비어 있다. 첫 후보는 기록을 다시 만들어 저장된 것과 같은지 보는 검사(`tools/gen_regression_r.py`, `tools/gen_quantreg_r.py`, `tools/gen_robust_r.py`, `tools/gen_nls_r.py`, `tools/gen_optimize_fortran.py`)다. 이것들을 테스트로 감쌀지는 따로 정한다(§6 질문 3).
- 골든 비교(`tools/golden.py check`, 약 20초)는 `deep`에 넣는다. pytest 안에서 부르는 얇은 테스트 하나(`tests/golden/test_golden.py`, `@pytest.mark.deep`)다.

## 2. 하나의 장치

- 새 파일 `tests/conftest.py`(루트)가 모든 것을 정한다.
  - 마커 등록: `deep`, `external`, `serial`(xdist 단계에서 쓴다). `pyproject.toml`의 `[tool.pytest.ini_options]`에 `addopts = "--strict-markers"`를 더해 오타 마커를 막는다.
  - 환경 변수 **`SD_TIER`** 하나(`quick` 기본, `deep`, `all`)가 고르는 테스트와 hypothesis 프로필을 함께 정한다. 명령줄 `--tier=`도 같은 것을 받는다(`--tier`가 이긴다). 고르지 않은 등급의 테스트는 건너뛰기가 아니라 deselect한다(요약에 `N deselected`로 보인다).
  - hypothesis 프로필 `quick`, `deep`을 등록하고 `SD_TIER`에 맞춰 불러온다. 둘 다 `deadline=None`(지금 모든 파일이 그렇게 쓴다).
- 새 파일 `tests/_tier.py`: `examples(n)`은 `quick`에서 n, `deep`에서 10n을 돌려준다. 파일마다 다른 기본값(150, 200, 300, 60)은 그대로 둔다. 파일마다 예제 수가 다른 이유(케이스 하나의 비용)는 남기고, 바꾸는 손잡이만 하나로 모은다.
- 옛 환경 변수 `SDLA_EXAMPLES`(linalg 9개 파일), `SDPD_EXAMPLES`(frame 9개 파일)는 없앤다. 바꾸는 곳:
  - `settings(max_examples=int(os.environ.get("SDLA_EXAMPLES", 150)), deadline=None)` → `settings(max_examples=examples(150), deadline=None)` (linalg: `test_nonsymmetric`, `test_optimizations`, `test_basic_norms`, `test_factorizations`, `test_solve_inv_det`, `test_qr_lstsq`, `test_kron_cross_expm`, `test_structured_solvers`, `test_spectral`)
  - `EXAMPLES = int(os.environ.get("SDPD_EXAMPLES", N))` → `EXAMPLES = examples(N)` (frame: `test_elementwise`, `test_reduce`, `test_sort`, `test_groupby`, `test_indexing`, `test_repr`, `test_unique`, `test_merge`, `test_csv`)
  - 고정 값 `max_examples=200`(`frame/test_construction`), `max_examples=60`(`frame/test_reduce`의 하나)도 `examples(200)`, `examples(60)`으로.
  - `settings`가 없는 `@given`(array 9개 파일)은 프로필의 기본값을 따른다: `quick` 100, `deep` 1000.
- 재현성: `quick` 프로필은 `derandomize=True`(매번 같은 예제)로, `deep`은 무작위(새 예제를 찾는다)로 하자고 제안한다(§6 질문 1). 실패한 예제는 hypothesis가 출력하는 `@reproduce_failure`로 다시 돈다.

## 3. `deep`으로 옮길 테스트

quick의 2분을 넘기는 것은 몇 개의 무거운 테스트다. 이것들만 `@pytest.mark.deep`을 단다(모듈의 다른 테스트는 quick에 남는다).

| 테스트 | 시간(초) | 이유 |
|---|---:|---|
| `regression/test_glm.py::test_noncanonical_against_arb` | 26.8 | Arb 참값 계산 |
| `regression/test_glm.py::test_inner_normal_equations_against_arb` | 5.5 | Arb 참값 계산 |
| `stats/test_review_1119b.py::test_noncentral_f_at_nc_zero` | 10.6 | 넓은 격자 |
| `regression/test_robust.py::test_theil_passes` | 3.4 | 큰 입력 반복 |
| `stats/test_review_1119c.py::test_inverse_gamma_subnormal_shape` | 3.1 | 넓은 격자 |
| `stats/test_review_1119.py::test_noncentral_chi_square_large_nc` | 3.1 | 넓은 격자 |
| `bench/test_speed_harness.py::test_supervisor_limits_and_failures`, `test_supervisor_resume` | 20.5 | 하위 프로세스를 띄우고 시간을 잰다(bench 세션의 파일이라 그 세션과 정한다) |

이것들을 빼면 quick의 테스트 시간 합은 약 71초다(지금 144초). 벽시계로는 약 80초로 예상한다.

## 4. 순서와 확인

1. `tests/conftest.py`, `tests/_tier.py`, `pyproject.toml`의 `addopts`·마커 등록.
2. 18개 파일의 환경 변수를 `examples()`로 바꾼다(값은 그대로라 quick의 결과는 지금과 같다).
3. §3의 테스트에 `@pytest.mark.deep`.
4. 확인: `SD_TIER=quick`이 모으는 테스트 = 지금 전체 − §3, 그 시간 < 2분, `SD_TIER=deep`이 모두 통과, `--strict-markers`에 걸리는 것 없음, `tools/check.py` 깨끗.
5. `tests/README.md` §1을 새 등급으로 고친다.

## 5. 모듈별 등급

표기: 테스트 수와 시간은 2026-11-20 측정. 등급의 `quick + deep: 이름`은 그 이름의 테스트만 deep이라는 뜻이다. xdist 열은 다음 단계(병렬 실행)에서 따로 한 프로세스로 돌려야 할 이유다(`serial` 마커 후보).

### (루트)

| 모듈 | 테스트 | 시간(초) | 등급 | hypothesis | xdist |
|---|---:|---:|---|---|---|
| `test_build.py` | 9 | 0.0 | quick |  |  |
| `test_error_messages.py` | 115 | 0.1 | quick |  |  |
| `test_gil_release.py` | 3 | 6.0 | quick |  | GIL·시간 |
| `test_out_of_range.py` | 19 | 0.0 | quick |  |  |

### array

| 모듈 | 테스트 | 시간(초) | 등급 | hypothesis | xdist |
|---|---:|---:|---|---|---|
| `array/test_api_surface.py` | 3 | 0.0 | quick |  |  |
| `array/test_array_api_additions.py` | 95 | 0.2 | quick |  |  |
| `array/test_asarray.py` | 47 | 0.1 | quick | 1개, 기본(100) |  |
| `array/test_astype_assignment.py` | 31 | 0.1 | quick | 1개, 기본(100) |  |
| `array/test_buffer_protocol.py` | 46 | 0.1 | quick | 1개, 기본(100) |  |
| `array/test_creation.py` | 96 | 0.0 | quick |  |  |
| `array/test_indexing_advanced.py` | 120 | 0.1 | quick | 1개, 기본(100) |  |
| `array/test_indexing_basic.py` | 49 | 0.1 | quick | 1개, 기본(100) |  |
| `array/test_iter.py` | 14 | 0.0 | quick |  |  |
| `array/test_join.py` | 37 | 0.0 | quick |  |  |
| `array/test_matmul.py` | 82 | 0.0 | quick |  |  |
| `array/test_ndarray.py` | 37 | 0.1 | quick |  |  |
| `array/test_reduction.py` | 822 | 0.2 | quick | 2개, 기본(100) |  |
| `array/test_sort_unique.py` | 121 | 0.7 | quick | 3개, 기본(100) |  |
| `array/test_statistics.py` | 136 | 0.1 | quick |  |  |
| `array/test_threads.py` | 7 | 1.1 | quick |  | 스레드 수 |
| `array/test_ufunc.py` | 585 | 0.2 | quick | 1개, 기본(100) |  |
| `array/test_vforce.py` | 40 | 0.1 | quick |  |  |
| `array/test_views.py` | 60 | 0.1 | quick | 1개, 기본(100) |  |

### bench

| 모듈 | 테스트 | 시간(초) | 등급 | hypothesis | xdist |
|---|---:|---:|---|---|---|
| `bench/test_speed_harness.py` | 21 | 21.7 | quick |  | 하위 프로세스·시간 |

### frame

| 모듈 | 테스트 | 시간(초) | 등급 | hypothesis | xdist |
|---|---:|---:|---|---|---|
| `frame/test_api_surface.py` | 3 | 0.0 | quick |  |  |
| `frame/test_construction.py` | 154 | 0.3 | quick | 3개, 고정 200 |  |
| `frame/test_correction_range.py` | 2 | 0.0 | quick |  |  |
| `frame/test_csv.py` | 53 | 0.3 | quick | 1개, SDPD_EXAMPLES(200) |  |
| `frame/test_elementwise.py` | 101 | 3.7 | quick | 12개, SDPD_EXAMPLES(300) |  |
| `frame/test_groupby.py` | 17 | 1.0 | quick | 1개, SDPD_EXAMPLES(200) |  |
| `frame/test_indexing.py` | 70 | 0.8 | quick | 2개, SDPD_EXAMPLES(300) |  |
| `frame/test_merge.py` | 8 | 0.7 | quick | 1개, SDPD_EXAMPLES(300) |  |
| `frame/test_options.py` | 79 | 0.0 | quick |  |  |
| `frame/test_reduce.py` | 17 | 1.0 | quick | 3개, SDPD_EXAMPLES(300) |  |
| `frame/test_repr.py` | 10 | 3.7 | quick | 2개, SDPD_EXAMPLES(300) |  |
| `frame/test_sort.py` | 15 | 0.8 | quick | 1개, SDPD_EXAMPLES(300) |  |
| `frame/test_threads.py` | 78 | 11.5 | quick |  | 스레드 수 |
| `frame/test_unique.py` | 11 | 1.2 | quick | 2개, SDPD_EXAMPLES(300) |  |

### interop

| 모듈 | 테스트 | 시간(초) | 등급 | hypothesis | xdist |
|---|---:|---:|---|---|---|
| `interop/test_frame_array.py` | 39 | 0.0 | quick |  |  |

### linalg

| 모듈 | 테스트 | 시간(초) | 등급 | hypothesis | xdist |
|---|---:|---:|---|---|---|
| `linalg/test_api_surface.py` | 2 | 0.0 | quick |  |  |
| `linalg/test_basic_norms.py` | 64 | 0.5 | quick | 6개, SDLA_EXAMPLES(150) |  |
| `linalg/test_build.py` | 1 | 0.0 | quick |  |  |
| `linalg/test_factorizations.py` | 27 | 0.4 | quick | 6개, SDLA_EXAMPLES(150) |  |
| `linalg/test_kron_cross_expm.py` | 10 | 0.4 | quick | 3개, SDLA_EXAMPLES(150) |  |
| `linalg/test_lstsq_numpy_bits.py` | 56 | 0.0 | quick |  |  |
| `linalg/test_lstsq_small_rtol.py` | 7 | 0.0 | quick |  |  |
| `linalg/test_nonsymmetric.py` | 21 | 0.4 | quick | 4개, SDLA_EXAMPLES(150) |  |
| `linalg/test_optimizations.py` | 50 | 0.2 | quick | 3개, SDLA_EXAMPLES(150) |  |
| `linalg/test_optimizations_small.py` | 15 | 0.4 | quick |  |  |
| `linalg/test_options.py` | 23 | 0.0 | quick |  |  |
| `linalg/test_parallel.py` | 79 | 0.1 | quick |  | 스레드 수 |
| `linalg/test_qr_lstsq.py` | 15 | 0.2 | quick | 3개, SDLA_EXAMPLES(150) |  |
| `linalg/test_review_fixes.py` | 16 | 1.0 | quick |  |  |
| `linalg/test_solve_inv_det.py` | 24 | 0.2 | quick | 3개, SDLA_EXAMPLES(150) |  |
| `linalg/test_spectral.py` | 32 | 0.5 | quick | 6개, SDLA_EXAMPLES(150) |  |
| `linalg/test_structured_solvers.py` | 6 | 0.3 | quick | 3개, SDLA_EXAMPLES(150) |  |
| `linalg/test_tiny_singular_values.py` | 9 | 0.0 | quick |  |  |

### optimize

모듈 12개, 2,859개다(2026-11-20에 다시 센 값; 시간은 다른 작업과 기계를 나눈 채 한 프로세스로 잰 것이라 조용한 기계보다 길 수 있다).

| 모듈 | 테스트 | 시간(초) | 등급 | hypothesis | xdist |
|---|---:|---:|---|---|---|
| `optimize/test_api_surface.py` | 6 | 0.0 | quick |  |  |
| `optimize/test_diff.py` | 100 | 0.1 | quick |  |  |
| `optimize/test_elementwise.py` | 278 | 1.5 | quick |  |  |
| `optimize/test_least_squares.py` | 31 | 5.5 | quick |  |  |
| `optimize/test_lsq_linear.py` | 23 | 0.6 | quick |  |  |
| `optimize/test_minimize.py` | 197 | 13.9 | quick |  |  |
| `optimize/test_minimize_newton.py` | 554 | 51.5 | quick |  |  |
| `optimize/test_problems.py` | 170 | 0.3 | quick |  |  |
| `optimize/test_reference_data.py` | 10 | 0.1 | quick |  |  |
| `optimize/test_result.py` | 10 | 0.0 | quick |  |  |
| `optimize/test_root.py` | 37 | 3.5 | quick |  |  |
| `optimize/test_scalar.py` | 1443 | 0.5 | quick |  |  |

### random

| 모듈 | 테스트 | 시간(초) | 등급 | hypothesis | xdist |
|---|---:|---:|---|---|---|
| `random/test_alloc_errors.py` | 8 | 0.0 | quick |  | 메모리 한계 |
| `random/test_random.py` | 52 | 0.2 | quick |  |  |

### regression

| 모듈 | 테스트 | 시간(초) | 등급 | hypothesis | xdist |
|---|---:|---:|---|---|---|
| `regression/test_against_r.py` | 103 | 0.0 | quick |  |  |
| `regression/test_api_surface.py` | 2 | 0.0 | quick |  |  |
| `regression/test_glm.py` | 118 | 34.5 | quick + deep: `test_noncanonical_against_arb`, `test_inner_normal_equations_against_arb` |  |  |
| `regression/test_kernel_paths.py` | 29 | 3.1 | quick |  |  |
| `regression/test_linear.py` | 56 | 0.1 | quick |  |  |
| `regression/test_nls.py` | 139 | 0.9 | quick |  |  |
| `regression/test_penalized.py` | 45 | 0.5 | quick |  |  |
| `regression/test_quantreg.py` | 111 | 1.4 | quick |  |  |
| `regression/test_reference_tool.py` | 13 | 0.4 | quick |  |  |
| `regression/test_review_decisions.py` | 95 | 0.5 | quick |  |  |
| `regression/test_robust.py` | 103 | 9.5 | quick + deep: `test_theil_passes` |  |  |
| `regression/test_robust_diagnostics.py` | 51 | 0.4 | quick |  |  |

### special

| 모듈 | 테스트 | 시간(초) | 등급 | hypothesis | xdist |
|---|---:|---:|---|---|---|
| `special/test_api_surface.py` | 3 | 0.0 | quick |  |  |
| `special/test_bessel_airy.py` | 25 | 0.3 | quick |  |  |
| `special/test_complex.py` | 15 | 0.0 | quick |  |  |
| `special/test_distributions.py` | 57 | 0.0 | quick |  |  |
| `special/test_engine.py` | 23 | 0.0 | quick |  |  |
| `special/test_erf.py` | 78 | 0.0 | quick |  |  |
| `special/test_extreme.py` | 103 | 0.1 | quick |  |  |
| `special/test_gamma.py` | 25 | 0.2 | quick |  |  |
| `special/test_hyper_lambertw.py` | 8 | 0.0 | quick |  |  |
| `special/test_incomplete.py` | 26 | 0.0 | quick |  |  |
| `special/test_jv_zero_bound.py` | 20 | 0.5 | quick |  |  |
| `special/test_logistic.py` | 32 | 0.0 | quick |  |  |
| `special/test_noncentral.py` | 66 | 0.0 | quick |  |  |
| `special/test_reduce_expint_ellip.py` | 12 | 0.0 | quick |  |  |
| `special/test_reference.py` | 194 | 0.0 | quick |  |  |
| `special/test_review_1119.py` | 6 | 0.1 | quick |  |  |

### stats

| 모듈 | 테스트 | 시간(초) | 등급 | hypothesis | xdist |
|---|---:|---:|---|---|---|
| `stats/test_descriptive.py` | 61 | 1.0 | quick |  |  |
| `stats/test_discrete.py` | 93 | 1.5 | quick |  |  |
| `stats/test_distributions.py` | 148 | 1.9 | quick |  |  |
| `stats/test_matrix.py` | 35 | 0.2 | quick |  |  |
| `stats/test_multivariate.py` | 59 | 0.5 | quick |  |  |
| `stats/test_mvcdf.py` | 32 | 3.2 | quick |  |  |
| `stats/test_reference.py` | 1110 | 1.1 | quick |  |  |
| `stats/test_review_1119.py` | 49 | 3.1 | quick + deep: `test_noncentral_chi_square_large_nc` |  |  |
| `stats/test_review_1119b.py` | 96 | 10.7 | quick + deep: `test_noncentral_f_at_nc_zero` |  |  |
| `stats/test_review_1119c.py` | 114 | 3.2 | quick + deep: `test_inverse_gamma_subnormal_shape` |  |  |
| `stats/test_sample.py` | 263 | 0.4 | quick |  |  |
| `stats/test_small_shapes.py` | 17 | 0.0 | quick |  |  |


## 6. 정할 것 (적용 전에 묻는다)

1. `quick` 프로필의 hypothesis를 `derandomize=True`(같은 예제, 재현 가능, 병렬 실행에서도 결정적)로 할지, 지금처럼 무작위로 둘지. 추천: `derandomize=True`.
2. `deep`의 배수(10배)와 프로필 기본값(1000).
3. R·Fortran 생성기를 `external` 테스트로 감쌀지, 도구로만 둘지.
4. bench 감독자 테스트 둘을 `deep`으로 옮길지(bench 세션의 파일).
