# stardust 내부 구조

라이브러리의 층, 디렉터리, 공통 부품, 그리고 측정·시험 체계의 자리를 적는다. 결정의 근거는 [DESIGN.md](DESIGN.md)에, 영역 안의 구조(이터레이터, 배치 드라이버, Copy-on-Write, 분포 엔진 등)는 영역 문서 `docs/design/<영역>.md`에 있다.

## 1. 층 구조

영역은 열아홉이다: array, linalg, frame, special, stats, regression, random, optimize, htest, sparse, integrate, differentiate, spatial, graph, interpolate, fft, smooth, cluster, plot(`bench/areas.py`의 순서). Python 하위 모듈 하나와 `csrc/`의 디렉터리 하나가 영역 하나다. plot(그림)은 맨 위 층이다: 모든 영역의 결과를 받고 어느 영역도 plot을 모른다(docs/design/plot.md §1.4).

```
 Python:  sd   sd.linalg   sd.frame   sd.special   sd.stats   sd.regression   sd.random   sd.optimize
          sd.htest   sd.sparse   sd.integrate   sd.differentiate   sd.spatial   sd.graph   sd.interpolate   sd.fft   sd.smooth   sd.cluster   sd.plot
            │       │          │           │            │            │               │            │
 bind/ ─────┴───────┴──────────┴───────────┴────────────┴────────────┴───────────────┴────────────┴──  _core 확장 모듈 하나
                                                        │
                     htest/      ──▶ stats/, special/, random/, array/, parallel/
                     spatial/    ──▶ array/, base/, parallel/, accelerate/(이웃 찾기의 GEMM 고르기)
                     graph/      ──▶ sparse/(표준 꼴 CSR 읽기와 전치), array/, base/, parallel/
                     interpolate/ ──▶ array/, base/(lerp·interval: 배열의 interp와 함께 쓰는 선형 커널과 구간 찾기), parallel/, linalg/(LAPACK 래퍼: RBF의 분해와 랭크 검사, AAA의 SVD와 일반화 고유값), spatial/(KD 트리와 거리 커널: RBF의 이웃과 거리)
                     smooth/     ──▶ array/, base/(buffer·interval), parallel/, interpolate/(de Boor의 B-스플라인 값), spatial/(KD 트리와 거리 커널), fft/(계획과 DCT)
                     cluster/    ──▶ array/, base/, parallel/, spatial/(거리 커널과 KD 트리), graph/(합집합–찾기 `unionfind.hpp`), stats/(혼합 모형 `sample`의 정규 난수 `sample_dist.hpp`), accelerate/(k-평균 거르기의 dgemm)
                     fft/        ──▶ array/, base/(sincos: 회전 인자), parallel/
                     sparse/     ──▶ linalg/, array/, base/, parallel/, accelerate/(Accelerate의 희소 LU와 BLAS)
                     regression/ ──▶ linalg/, special/, random/, htest/(Jarque–Bera의 중심 합), parallel/
                     optimize/   ──▶ linalg/, differentiate/(차분의 간격), elementwise/(원소별 반복 고리)
                     differentiate/ ──▶ elementwise/(derivative의 원소별 고리), linalg/(accurate의 fastsum 합), parallel/, array/  (steps는 표준 라이브러리만 쓴다)
                     integrate/  ──▶ array/(pairwise 합과 cumulative_sum의 스캔), parallel/, elementwise/(tanhsinh·nsum의 원소별 고리), linalg/(Accelerate의 BLAS: quad_vec의 2-노름)
                     elementwise/ ──▶ array/, parallel/
                     stats/      ──▶ special/, random/
                     special/, linalg/, random/, frame/(interop만) ──▶ array/
                                                        │
                                  array/ ──▶ base/, parallel/, accelerate/
```

- **아래로만 의존한다.** 위 층은 아래 층을 쓰고, 아래 층은 위 층을 모른다.
  - `array/`는 다른 영역을 include하지 않는다. 같은 x의 sin과 cos를 한 번에 구하는 `base/sincos.hpp`(docs/design/array.md §1.3)는 의존이 없어 바닥 층에 두고 array와 special이 함께 쓴다.
  - `linalg/`는 `array/`의 `NDArray`를 직접 받는다. buffer protocol을 거치지 않는다 (DESIGN §3.1). LAPACK 래퍼(`linalg/lapack.hpp`, `linalg/ops/common.hpp`)는 regression과 optimize도 쓴다.
  - `frame/`은 `array/`를 모른다. 둘을 잇는 코드는 `frame/interop.{hpp,cpp}` 한 곳에만 둔다 (DESIGN §4.3). `array/` 헤더를 include하는 frame 파일은 `interop.cpp` 하나다. 다른 frame 파일은 `interop.hpp`(Column과 nanobind만 쓰는 선언)를 통해 부른다 (docs/design/frame.md §5.4).
  - `stats/`는 special의 커널(불완전 감마·베타, 오차함수 등)과 random의 생성기를 쓴다. `regression/`은 linalg의 LAPACK 래퍼, special의 double-double·오차함수·감마·로지스틱 함수, random의 생성기, htest의 중심 합 커널(`htest/normality.hpp`: `jarque_bera()`), parallel의 스레드풀(`parallel/pool.hpp`)을 쓴다. Python 쪽에서 `regression.nls`는 optimize의 최소제곱(`csrc/optimize/lsq.cpp`)을 쓴다.
  - `spatial/`은 짝 거리의 합을 자기 순서(타일 순서)로 하므로 linalg의 합에 기대지 않는다(docs/design/spatial.md §3.1). Python 층의 `stardust.spatial`은 `stardust.sparse`(반경 결과의 `tocsr()`)와 `stardust.linalg`(`mahalanobis`의 기본 `VI`)를 쓴다. 계산 기하(`Delaunay`, `ConvexHull`, `Voronoi`)는 직접 쓴 정확한 술어와 `array/`·`parallel/`만 쓰고 BLAS·LAPACK과 밖의 기하 라이브러리를 부르지 않는다(docs/design/spatial_geometry.md §6).
  - `graph/`는 sparse의 표준 꼴 CSR(`sparse/format.hpp`)을 읽기만 하고 sparse는 graph를 모른다(docs/design/graph.md §1.4, §6). Python 층의 `stardust.graph`는 `stardust.sparse`(입력, `csr_array` 결과, `LinearOperator`)를 쓴다.
  - `interpolate/`의 선형 커널(`base/lerp.hpp`)과 구간 찾기(`base/interval.hpp`)는 바닥 층에 두고 배열의 `sd.interp`(`array/ops/interp.cpp`)와 함께 쓴다(`base/sincos.hpp`와 같은 자리; docs/design/interpolate.md §1.4, §3.2). 배열은 interpolate를 모른다. 삼각분할 위의 보간(`interpolate/scattered.cpp`)은 spatial의 `_Delaunay` 객체(`spatial/delaunay2.hpp`의 `DelaunayObj`), 위치 찾기(`triangulation.hpp`의 `locate_each`), 값 커널(`predicates.hpp`)을 쓴다(docs/design/interpolate.md IP128). Python 층의 `stardust.interpolate`는 `stardust.spatial`(`KDTree`, `Delaunay`)과 `stardust.sparse`(`design_matrix`)를 쓴다.
  - `fft/`는 다른 영역을 쓰지 않는다. 모든 길이를 자체 C++로 계산하고 Accelerate의 vDSP를 부르지 않는다(docs/design/fft.md FT30, FT66; vDSP의 래퍼 `fft/vdsp.hpp`는 라이브러리에 빌드되지 않고 탐침 `tools/fft_probe_kernels.cpp`만 쓴다). Python 층의 `stardust.fft`도 다른 영역을 쓰지 않는다(FFTLog의 계수도 `fft/fftlog.hpp`의 double-double log Γ로 만든다: docs/design/fft.md FT58).
  - `smooth/`는 보간의 B-스플라인 값(`interpolate/deboor.hpp`), spatial의 KD 트리와 거리 커널(`spatial/kdtree.hpp`), fft의 계획과 r2r(`fft/plan.hpp`, `fft/r2r.hpp`)를 부르기만 하고 그 영역들은 smooth를 모른다. 한 변수 찾기(fminbound, brentq)와 Nelder–Mead는 optimize와 같은 알고리즘을 C++ 함수 인자로 받는 판을 이 영역에 둔다(`smooth/search.hpp`, `kreg.cpp`, `bandwidth.cpp`; docs/design/smooth.md §1.4, §6). Python 층의 `stardust.smooth`는 `stardust.interpolate`(결과의 `BSpline`, 보간 극한의 `make_interp_spline`), `stardust.stats`(구간의 분위수, 표본의 잡음), `stardust.random`, `stardust.linalg`, `stardust.frame`, `stardust._summary`를 쓴다.
  - `cluster/`는 spatial의 짝 거리 커널(`spatial/metric.hpp`: k-평균과 계층 군집의 거리가 `sd.spatial.cdist`와 같은 비트)과 KD 트리를 부르기만 하고 spatial은 cluster를 모른다(docs/design/cluster.md §1.4, §6). 계층 군집(`cluster/hierarchy.cpp`, `cluster/mst.cpp`; CL1 마일스톤)은 spatial의 짝 커널과 KD 트리(`kd_build`, 증명된 가지치기 `KdCut`)를 부른다. 분할(`cluster/kmeans.cpp`, `kmedoids.cpp`, `fuzzy.cpp`; CL2 마일스톤)의 k-평균 거르기는 Accelerate의 `dgemm`을 부른다(결과의 비트는 바꾸지 않는다: docs/design/cluster.md CL60). 밀도 군집(`cluster/density.cpp`; CL3 마일스톤)은 spatial의 KD 트리 질의(`KdSearch`, 반경 찾기의 증명된 가지치기)와 graph의 합집합–찾기(`graph/unionfind.hpp`: graph의 약한 연결 요소와 같은 머리 파일)를 부르고 graph는 cluster를 모른다. 가우스 혼합(`cluster/mixture.cpp`; CL4 마일스톤)은 자체 고정 순서 커널(작은 Cholesky와 그 역, NEON의 이차형식과 2차 모멘트)이고 Accelerate를 쓰지 않는다. 결과의 `sample`은 stats의 정규 난수(`stats/sample_dist.hpp`의 ziggurat: `sd.stats.Normal`의 `sample`과 같은 꼴)를 머리 파일로 부르고 stats는 cluster를 모른다. Python 층의 `stardust.cluster`는 `stardust.random`(`rng`의 두 단어), `stardust.spatial`(`squareform`, 거리 이름), `stardust.frame`, `stardust._summary`, `stardust._calls`를 쓴다.
  - Python 층에서 `stardust.integrate`와 `stardust.differentiate`는 인자 검사 도우미(`_callable`, `_check_int`, `_real` 등)를 `stardust.optimize._common`에서, 원소별 무리의 입력(`_inputs`, `_args`, `_shaped`)을 `stardust.optimize._elementwise`에서 가져온다. C++의 반복 고리는 바닥 폴더 `csrc/elementwise/`에 있어 optimize를 거치지 않는다(docs/design/integrate.md C16, §6.3).
  - Python 경계의 공용 도우미 `bind/pyutil.hpp`(배열 받기와 만들기)와 `bind/errors.hpp`(에러 메시지에 공개 함수 이름 붙이기)는 모든 영역의 `ops/`가 쓴다. 영역마다의 바인딩 파일(`bind/<영역>_bind.cpp`)은 맨 위 층이다.
- **새 영역은 디렉터리 하나와 Python 하위 모듈 하나로 붙는다.** 아래 층만 쓰고, 기존 영역을 고치지 않는다.
  - 새 영역은 `docs/design/<영역>.md`, `docs/api/<영역>.md`, `tests/<영역>/`, 골든 케이스 `tests/golden/cases_<영역>.py`, 속도 사례 `bench/speed/cases/<영역>.py`를 함께 갖고, `bench/areas.py`의 목록에 들어간다. 참값이 있으면 `tests/<영역>/ref/`, `tests/<영역>/refdata.py`, `tools/<영역>_ref.py`와 정확도 스윕을 더한다.

## 2. 디렉터리 구조

```
stardust/
  pyproject.toml  CMakeLists.txt
  README.md  API.md  DESIGN.md  ARCHITECTURE.md  ROADMAP.md

  src/stardust/
    __init__.py                 배열 API, 스레드 수, dtype 객체 다섯 (sd.string 포함)
    _format.py                  배열 repr
    _results.py                 모든 영역의 결과 namedtuple (<함수>Result, DESIGN §9.7)
    _calls.py                   호출 꼴 에러(인자 수, 모르는 키워드)를 라이브러리의 꼴로 (API.md §2.1)
    linalg/                     __init__.py(_core.linalg 재노출), _errors.py(LinAlgError)
    frame/                      __init__.py와 밑줄 이름의 모듈들(_series, _frame, _index, _column, _groupby, _merge, _io, …)
    special/  random/           __init__.py: _core의 하위 모듈을 공개 이름으로
    stats/                      _distribution(공통 기반), _continuous, _discrete, _multivariate, _matrix, _descriptive
    regression/                 _linear, _glm, _penalized, _robust, _quantile, _nls
    optimize/                   _scalar, _elementwise, _minimize, _lsq, _linear, _root, _diff, _result, _common
    htest/                      _common, _parametric, _rank, _correlation, _contingency, _proportion, _gof, _multicomp, _resampling, _result
    sparse/                     _base, _compressed, _coo, _dok, _construct, _index, _ops, _reduce, _direct, _iterative, _eigen, _matfuncs,
                                _interface, _interop, _result, _common
    integrate/                  _quadpack, _quad_vec, _cubature, _tanhsinh, _samples, _rules, _gauss, _result
    differentiate/              _derivative, _weights, _differences, _result
    spatial/                    _common, _distance, _neighbors, _radius(RadiusResult), _kdtree (docs/design/spatial.md §6),
                                _delaunay, _hull, _voronoi (계산 기하; docs/design/spatial_geometry.md §6)
    graph/                      _common, _traversal, _convert, _paths, _tree, _flow(MaximumFlowResult), _matching,
                                _ordering, _structure, _centrality (docs/design/graph.md §6)
    interpolate/                __init__.py, _common, _ppoly (IP2 마일스톤), _bspline (IP3 마일스톤), _polyint _aaa (IP4 마일스톤), _grid (IP5 마일스톤), _rbf (IP6 마일스톤; docs/design/interpolate.md §6)
    fft/                        _common, _basic, _realtransforms, _helper, _fftlog (docs/design/fft.md §6)
    smooth/                     __init__.py, _results(회귀형 결과의 공통 기반; SM0 마일스톤), _common, _local (SM1 마일스톤), _loess (SM2 마일스톤), _spline _fitpack (SM3 마일스톤), _kreg (SM4 마일스톤), _kde _bandwidth (SM5 마일스톤), _savgol (SM6 마일스톤; docs/design/smooth.md §6)
    cluster/                    __init__.py, _results(결과의 공통 기반 `ClusterResult`; CL0 마일스톤), _common (CL0 마일스톤), _hierarchy(계층과 scipy의 도구; CL1 마일스톤), _kmeans _kmedoids _fuzzy (CL2 마일스톤), _density (CL3 마일스톤), _mixture (CL4 마일스톤) (docs/design/cluster.md §6)
    plot/                       _figure(칸·배치·내보내기), _layers, _basic, _sample·_function(함수 그림), _field(두 변수), _stat, _surface(3D), _adapt(곡면의 적응형 표본),
                                _scene(도형 목록), _text(LaTeX 조판과 글리프 윤곽선의 캐시), _dvi, _ticks, _color, _svg·_pdf·_png
                                (docs/design/plot.md §7)

  csrc/
    base/         buffer(참조 카운트 메모리 블록), dtype(DType enum), pairwise(pairwise 합의 정의),
                  saturate(float64 → int64 명시적 변환), int_sum(int64의 정확한 합), sincos(같은 x의 sin·cos),
                  lerp(선형 커널), interval(구간 찾기; 배열의 interp와 interpolate가 함께 쓴다)
    parallel/     pool(스레드풀 하나, GIL 해제, FirstRow, Unlocked)
    accelerate/   blas(dot, gemv, gemm), vforce(연산과 무관한 vForce 래퍼)
    array/        ndarray  iter  convert  parallel_limits
                  kernels/  elementwise buffered cast scalar_ops reduce sort simd_sort neon vforce vmath
                  ops/      asarray copy creation indexing manipulation shape matmul reduction
                            covariance quantile histogram searching interp sort ufunc lanes pieces dlpack dtype_info
    linalg/       args batch pack cost errors input outblock pyrefs types pairwise fastsum lapack
                  ops/      lu basic norms vnorm factor qr spectral svals nonsym special expm common
    frame/        column bitmap convert parallel interop
                  ops/      cast elementwise groupby hash join reduce selection sort radix parse pairwise pow5_table
                  io/       csv
    special/      common dd cf halley prefix reduce gamma_core cplx boost cost cost_table
                  ops/      logistic erf gamma loggamma zeta incgamma incbeta distr noncentral expint ellip
                            bessel bessel01 airy hyper lambertw wofz cf_fast halley_fast testing
                  kernels/  simd chunked
                  coeffs/   생성한 계수 표 (tools/gen_*.py)
    stats/        engine chunk chunk_dd numeric discrete dd_linalg mvcdf sample sample_dist cost_table
                  dist/     분포 계열마다 한 파일 (normal, gamma, beta, t_f, noncentral, discrete_basic, …)
                  coeffs/   생성한 계수 표
    regression/   linear lsq slopes glm penalized robust quantile select dot2 dd2 unitdev
    random/       philox random
    optimize/     scalar batch lsq minpack minimize objective wolfe hessian nelder_mead bfgs lbfgsb
                  newton_cg trust root hybrd linear
    differentiate/ steps(고정 간격 차분의 간격: optimize가 씀), derivative(derivative·jacobian·hessian; docs/design/integrate.md §3.8),
                  accurate(정확한 차분 `Differences`의 C++ 고리: optimize의 approx_fprime·check_grad와 regression의 nls가 씀; docs/design/integrate.md C124)
    integrate/    samples(표본 규칙과 fixed_quad의 합), gauss(Gauss–Legendre의 점과 가중치), gauss_table(생성한 표; docs/design/integrate.md §3.7), quadpack(QUADPACK의 QAGS·QAGI·QAGP 템플릿과 quad·nquad의 갈래·겹침; §3.1, §3.2), tanhsinh(tanhsinh·nsum: scipy의 표·합·오차 추정과 nsum의 갈래; §3.5, §3.6), adaptive(quad_vec·cubature: scipy의 Python 알고리즘, 힙, 규칙, 치환; §3.3, §3.4), numpy_ops(numpy의 pairwise 합과 linalg.norm)
    htest/        parametric rank correlation contingency proportion gof gof2 multicomp resampling와 헤더(정확한 분포, 점근 분포, MLE, 레인)
    sparse/       format canonical convert construct index binop arith spmv spgemm reduce dok triangular direct iterative
                  krylov arpack svds parallel
    spatial/      metric(타일 순서의 짝 커널과 거리마다의 연산), distance(pdist·cdist·배치 거리 함수·squareform; docs/design/spatial.md §3.1),
                  neighbors(kneighbors·radius_neighbors의 전수와 GEMM 고르기; §3.2), kdtree(KD 트리 만들기와 kNN; §3.3, §3.4)
                  계산 기하(docs/design/spatial_geometry.md §6): expansion(전개식), wideint(여러 낱말 정수),
                  predicates(정확한 술어와 값 커널; §3.1, §3.2; .cpp는 테스트용 비공개 바인딩), sfc(사전 순서, 힐베르트, BRIO),
                  hull(볼록 껍질: 정적 필터의 미리 거르기, 단조 사슬, 퀵헐, 평평한 면 합치기; §3.6),
                  triangulation(삼각분할의 공용: 병렬 사전 순서 정렬, 표준 꼴, 위치 격자와 걷기, 지연 속성; §3.5),
                  delaunay2(2차원 분할 정복과 `_Delaunay`; §3.3), delaunay3(3차원 BRIO 순서의 하나씩 넣기와 잠금 낱말의 줄 넣기; §3.4),
                  voronoi(쌍대: 외심이 같은 단체의 묶음, 꼭짓점의 분자 커널, 능선과 영역; §3.7)
    graph/        input(밀집 → CSR, 전치, 대칭 확인), traversal(순회, 연결 요소, 위상 정렬), unionfind.hpp(약한 연결 요소의 합집합–찾기; cluster의 DBSCAN도 쓴다),
                  paths(최단 경로의 정의와 블록 FW; docs/design/graph.md §3.2), yen, tree(최소 신장 트리), flow,
                  matching(Hopcroft–Karp와 배정의 LAPJV), ordering(역 Cuthill–McKee, 라플라시안), structure,
                  centrality(PageRank, 매개 중심성; docs/design/graph.md §6)
    interpolate/  search linear (IP1 마일스톤), banded cubic ppoly (IP2 마일스톤), bspline (IP3 마일스톤), polyint aaa (IP4 마일스톤), grid deboor pchip (IP5 마일스톤), rbf (IP6 마일스톤; docs/design/interpolate.md §6)
    fft/          twiddle(회전 인자), cfft(혼합 기수 Stockham 복소 FFT, 4단계, 두 줄 커널), bluestein·ddmath, rfft(실수 FFT), plan(계획과 캐시), driver(축 구동기),
                  basic·helper·args(변환과 도우미), r2r(DCT·DST), fftlog(FFTLog), vdsp(vDSP DFT의 래퍼, 탐침만; docs/design/fft.md §6)
    smooth/       local (SM1 마일스톤), loess (SM2 마일스톤), bandqr search spline fitpack (SM3 마일스톤), gauss_sum kreg (SM4 마일스톤), gauss_tree kde bandwidth (SM5 마일스톤), savgol (SM6 마일스톤; docs/design/smooth.md §6)
    cluster/      common.hpp(열쇠, 압축 꼴 색인, 2의 거듭제곱 척도, 덩이 짝 합, 난수의 원소 번호), mst(Prim, KD 트리 Borůvka; 상호 도달 거리), hierarchy(NN-chain, 일반 알고리즘, 행렬 없는 길, 트리 위의 높이 재귀, DIANA, cophenet, 최적 잎 순서; CL1 마일스톤), partition.hpp kmeans kmedoids fuzzy (CL2 마일스톤), density(DBSCAN, HDBSCAN, OPTICS; CL3 마일스톤), mixture(가우스 혼합의 EM; CL4 마일스톤) (docs/design/cluster.md §6)
    plot/         raster(2D 래스터라이저: 면적 누적, 획의 윤곽, 점 찍기, 그림 붙이기), zbuffer(3D 깊이 버퍼), levels(marching squares; docs/design/plot.md §3.5, §5, §6)
    elementwise/  loop(원소별 반복 고리: optimize의 배치 넷, 미분·적분의 원소별 무리; docs/design/integrate.md §6.2)
    bind/         module.cpp(_core와 하위 모듈), <영역>_bind.cpp, stats_{desc,matrix,mv,mvcdf,sample}.cpp,
                  nary_bind.hpp(원소별 함수의 공통 바인딩), pyutil, errors

  third_party/boost_math/       special이 쓰는 Boost.Math 헤더 (README.md에 커밋)

  tests/
    conftest.py  _tier.py       등급(quick, deep, all), 마커, hypothesis 프로필, --match, restore_threads
    _bits.py                    값이 같은지 보는 도우미 (bits_equal, bits_equal_any_nan, value_equal, max_rel_error)
    reffile.py                  npz 참값 파일의 한 형식
    _api_doc.py                 API 문서의 시그니처 블록 파서 (영역마다의 test_api_surface가 쓴다)
    test_*.py                   영역을 넘는 것: 빌드와 공용 풀, 에러 메시지 형식, GIL 해제, 범위 밖 입력, 도구의 명령행
    <영역>/                     영역의 테스트(test_<주제>.py), conftest.py, refdata.py, ref/(참값), data/(바깥 입력, 다른 라이브러리의 기록)
    interop/                    frame ↔ 배열의 공유, 복사, 읽기 전용 뷰, Copy-on-Write
    golden/                     골든 케이스 cases_<영역>.py, 기록 data/<영역>.json, test_golden.py
    bench/                      측정 체계 자신의 테스트, 정확도 기준선 비교(test_accuracy_baseline.py)

  bench/
    areas.py  cli.py            영역 목록 하나, 모든 측정·시험 도구의 명령행 관례
    console.py                  콘솔 출력의 공통 틀
    speed/                      속도 측정 (cases/<영역>.py, results/<영역>.json 기준선)
    accuracy/                   정확도 측정 (results/<영역>.json 기준선)
    <영역>/                     기준선에서 생성한 표: speed.md, accuracy.md

  tools/
    test.py  golden.py  coverage.py    테스트, 골든, 커버리지 (tests/README.md)
    <영역>_ref.py                      Arb 참값 생성기 (special, stats, array, linalg, regression, optimize, htest, sparse, integrate, differentiate, spatial, graph, interpolate, fft, smooth, cluster)
    arbref.py  regref.py  mvref.py  mvcdfref.py   생성기가 함께 쓰는 Arb 도우미와 식
    gen_*.py                           C++ 계수 표(csrc/*/coeffs, frame의 pow5_table)와 다른 라이브러리의 기록(R, scipy, Fortran)
    check.py  format.py  check_refs.py  procrun.py  vendor_boost_math.py

  docs/
    design/<영역>.md            영역의 결정과 내부 구조, 측정 근거
    api/<영역>.md               영역별 공개 API (test_api_surface가 읽는다)
```

## 3. Python과 C++의 경계

- **"얇은 Python, 두꺼운 C++"**: 계산은 C++에 있고, Python은 인자 정리, 결과 객체, 메시지를 맡는다.
  - 예외: `Series`, `DataFrame`, 분포 객체(`sd.stats`), 회귀·최적화의 결과 객체는 Python 클래스다.
- **확장 모듈은 `_core` 하나다** (DESIGN §3.2).
  - `nb::module_::def_submodule`로 영역마다 `_core.linalg`, `_core.frame`, `_core.special`, `_core.stats`, `_core.regression`, `_core.random`, `_core.optimize`, `_core.differentiate`, `_core.integrate`, `_core.htest`, `_core.sparse`를 만든다(`bind/module.cpp`가 차례로 부른다). spatial의 `_core.spatial`은 공개 함수가 생기는 SPT1부터다. graph의 `_core.graph`는 GR1 마일스톤부터, interpolate의 `_core.interpolate`는 IP1 마일스톤부터, fft의 `_core.fft`는 FT1 마일스톤부터, smooth의 `_core.smooth`는 SM1 마일스톤부터, cluster의 `_core.cluster`는 CL0 마일스톤부터다(CL0은 시제품 `_proto_*`만; CL1 마일스톤부터 계층의 커널).
  - nanobind 타입 테이블이 하나라서, C++ 어디서든 `is_ndarray`/`as_ndarray`(`bind/pyutil.hpp`)로 배열을 직접 받는다.
- **linalg는 배열을 C++에서 직접 주고받는다.**
  - 입력은 `const NDArray&`이고, 결과는 linalg가 정한 strides로 `new_array_strided(dtype, shape, strides, nbytes)`가 `NDArray`를 직접 만든다. 레이아웃(행렬마다 열 우선, 배치 축은 바깥에 C-order, 길이 0인 축도 그대로 곱함)은 linalg의 정책이라 linalg에 두고, 메모리와 플래그만 배열이 맡는다.
  - `pyrefs`에는 결과 namedtuple과 `LinAlgError`만 있다. 스레드 수는 `parallel::num_threads()`로 읽는다.
  - 배열을 직접 주고받아, buffer protocol과 `sd.asarray`를 거치던 때보다 결과 배열 하나가 싸다(docs/design/linalg.md §3.6). 남는 비용은 `NDArray`와 그 Python 객체를 만드는 것이고, 그 Python 객체는 인스턴스 풀에서 다시 쓴다(docs/design/array.md §11, A1).

## 4. 공통 부품

### 4.1 스레드풀
- 하나만 둔다(`parallel/pool.hpp`): 상주 워커, spin 후 대기, atomic 카운터, 호출 스레드 참여, 중첩 금지. 잠든 워커는 일이 깨우기를 갚을 만큼(약 20 µs, `kWakeWorkNs`) 길 때만 깨우고, 호출자는 `parallel_for(ntasks, fn, task_ns)`로 태스크 하나의 시간을 알릴 수 있다(docs/design/array.md §10.1, A15). 풀이 다른 Python 스레드의 일로 바쁘면 직렬 경로로 간다(`available()`). 병렬로 도는 동안 GIL을 놓는다. `FirstRow`(병렬 검사에서 실패한 가장 앞의 위치: 에러가 스레드 수와 상관없이 같은 위치를 말한다)와 `Unlocked`(풀을 쓰지 않는 긴 직렬 구간에서 GIL 놓기)가 함께 있다.
- **병렬화 기준값은 영역마다 따로 둔다.** 측정해서 정한 값이 다르기 때문이다. 이름은 같고 이름공간이 다르다.
  - array: `sd::kParallelThreshold` = 2¹⁵, 전체 float 리덕션의 `sd::kReduceThreshold` = 2¹⁶, `sd::kParallelTask`(`array/parallel_limits.hpp`, docs/design/array.md §10.4)
  - frame: `sd::frame::kParallelThreshold` = 2¹⁷, `for_chunks`, `Unlocked`(`frame/parallel.hpp`)
  - linalg: 비용 모델(`linalg/cost.hpp`)
  - special·stats: 원소마다 잰 비용표(`csrc/special/cost_table.hpp`, `csrc/stats/cost_table.hpp`; bench/README.md의 C++ 비용표)
- 스레드 수 API는 `sd.set_num_threads`, `sd.get_num_threads` 하나다. 하위 모듈에는 두지 않는다.

### 4.2 Buffer
- `base/buffer.hpp`의 `sd::Buffer` 하나를 배열과 Column이 함께 쓴다. 정렬은 128B이고, 256B 이하의 데이터는 Buffer 객체 안에 둔다. 외부 메모리를 `adopt`하고, `size()`로 바이트 수를 준다.
- 배열 뷰와 Column이 같은 Buffer를 가리킬 수 있다(`Series.to_array`). Copy-on-Write 규칙은 그대로다: 읽기 전용 배열 뷰가 `shared_ptr<Buffer>`를 하나 쥐고 있으면 `use_count()`가 늘어 있으므로 frame 쪽 쓰기가 먼저 복사한다 (DESIGN §4.3, docs/design/frame.md §5.4).

### 4.3 dtype
- C++ enum은 `base/dtype.hpp`에 하나 둔다: `Bool, Int64, Float64, Complex128, String`.
  - 컨테이너마다 받는 부분집합을 명시한다: 배열은 앞의 넷(`kNumArrayDTypes`, `is_array_dtype`), Column은 `Complex128`을 뺀 넷.
- `visit_dtype`은 배열 dtype만 받는다. 모든 커널이 모든 dtype을 인스턴스화하지 않게 하기 위해서다. frame 코드는 dtype마다 `switch`로 고르고, 그 `switch`들이 `Complex128`을 "frame dtype이 아님"으로 따로 적는다. `interop.cpp`는 고정폭 dtype의 바이트를 복사할 뿐이라 `itemsize`로 충분하다.

### 4.4 Accelerate 래퍼
- 영역을 넘어 쓰는 BLAS(`accelerate/blas.hpp`: 배열의 `matmul`·공분산)와 vForce(`accelerate/vforce.hpp`: 배열·special·stats의 원소별 초월함수)는 `accelerate/`에 둔다. 배열의 `exp`·`expm1`·`log1p`·`cos`는 vForce가 아니라 자체 NEON 커널(`array/kernels/vmath.hpp`, docs/design/array.md §5.4)이고, stats의 chunk 커널·mvcdf와 special의 `softmax`·`logsumexp`도 같은 exp를 부른다(docs/design/stats.md T55). LAPACK 래퍼는 `linalg/lapack.hpp`에 있다(§1).
- 컴파일 정의 `ACCELERATE_NEW_LAPACK`, `ACCELERATE_LAPACK_ILP64`는 모듈 전체에 건다.

## 5. 빌드
- `nanobind_add_module(_core NB_STATIC NOMINSIZE ...)` 하나.
- 컴파일 플래그는 모듈 전체에 하나다: `-mcpu=native -ffp-contract=off -fno-builtin-sin -fno-builtin-cos -Wall -Wextra -Wpedantic`
  - `-ffp-contract=off`: 컴파일러가 곱셈과 덧셈을 FMA로 스스로 합치지 않는다. FMA는 재서 이득이 확인된 곳에만 명시적으로(`std::fma`, NEON `vfmaq`) 쓴다 (DESIGN.md §1.6).
  - `-fno-builtin-sin`, `-fno-builtin-cos`: 같은 x의 `sin`과 `cos`를 컴파일러가 `sincos` 하나로 바꾸지 않는다. 함께 구하는 곳은 코드가 `base/sincos.hpp`로 명시한다. 같은 이유다.
- editable 설치(`pip install --no-build-isolation -e ".[test]"`)는 `import stardust`에서 바뀐 파일만 다시 컴파일한다. 빌드 폴더는 `build/<휠 태그>`다.

## 6. 측정·시험 체계

네 체계가 서로 다른 질문에 답한다. 모두 `bench/areas.py`의 영역 목록과 `bench/cli.py`의 명령행 관례를 같이 쓴다.

| 체계 | 묻는 것 | 자리 | 문서 |
|---|---|---|---|
| 테스트와 등급 | 동작이 맞는가 (quick, deep, all) | `tests/`, `tools/test.py` | tests/README.md §1 |
| 골든 | 결과가 비트까지 그대로인가 (스레드 수 1, 3, 10) | `tests/golden/`, `tools/golden.py` | tests/README.md §2 |
| 정확도 | Arb 참값에서 얼마나 떨어졌고, 계약 안이며, 기준선보다 나빠지지 않았는가 | `bench/accuracy/`, 참값 `tests/<영역>/ref/`, 생성기 `tools/<영역>_ref.py` | bench/README.md §2 |
| 속도 | 비교 대상보다 얼마나 빠르고, 기준선보다 느려지지 않았는가 | `bench/speed/` | bench/README.md §1 |

- 커버리지(`tools/coverage.py`, tests/README.md §3)는 테스트와 골든을 계측 빌드로 돌린다.
- 정확도와 테스트는 참값 파일을 읽고 계약 대비 오차를 내는 함수(`tests/<영역>/refdata.py`)를 같이 쓴다. 속도와 정확도는 비교 대상 라이브러리, R 세션 규칙, Boost.Math 빌드를 같이 쓴다.
- 기준선(`bench/speed/results/`, `bench/accuracy/results/`)과 골든 기록(`tests/golden/data/`)은 커밋한다. 실행 파일과 로그(`build/`)는 커밋하지 않는다.
