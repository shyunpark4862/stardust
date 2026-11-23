"""bench/<area>/accuracy.md, generated from the area's baseline (or a run file); the stats area's
multivariate group (sweeps mv/...) has its own table, bench/stats/accuracy_multivariate.md."""

from __future__ import annotations

import math

from . import results

HEAD = {
    "special": "sd.special",
    "stats": "sd.stats",
    "regression": "sd.regression",
    "array": "sd 배열",
    "linalg": "sd.linalg",
    "optimize": "sd.optimize",
    "htest": "sd.htest",
    "sparse": "sd.sparse",
    "integrate": "sd.integrate",
    "differentiate": "sd.differentiate",
    "spatial": "sd.spatial",
    "graph": "sd.graph",
    "interpolate": "sd.interpolate",
    "fft": "sd.fft",
    "smooth": "sd.smooth",
}
MV = "mv/"  # the stats area's multivariate group: its sweeps' name prefix
MV_HEAD = "sd.stats 다변량·행렬 분포"
MV_PATH = "bench/stats/accuracy_multivariate.md"

# the paragraph under the title (default: the Arb references of tests/<area>/ref)
INTRO = {
    "mv": (
        "참값은 Arb로 계산한 값이다(tests/stats/ref/mv, tools/stats_ref.py): 밀도·엔트로피는 행렬식과 풀이를 주어진 "
        "행렬 그대로 Arb의 LU로, cdf는 Plackett 항등식과 척도 혼합으로 2⁻⁶⁴까지 구했다. factor_t의 d > 3만 2단 적응 "
        "구적(scipy, 약 1e-11)이고 그 영역은 QMC를 atol 1e-5로 본다. 영역과 한계는 tests/stats/refdata.py의 계약 "
        "함수가 정하고 tests/stats/test_mvcdf.py·test_reference.py도 같은 함수로 판정한다. 밀도·엔트로피의 점은 모두 "
        "보장 영역(κ ≤ 1e9, d·p ≤ 100, Wishart df ≤ 1e30) 안이고 계약은 |오차| ≤ 1e-14·max(1, |값|), pdf는 그 오차가 "
        "exp를 지난 상대 1e-14·max(1, |logpdf|)를 점마다 본다(|logpdf| < 700인 점). cdf는 guaranteed가 상대 1e-14, "
        "loose가 그 밖의 정규수 범위(bvn 10·eps·max(1, −ln P), tvn 8u·조건수), tiny가 P < 1e-300(절대 1e-300), "
        "정확한 t(bvt, tvt, factor_t의 d ≤ 3)가 상대 5e-15(d = 2)·1e-14(d = 3), QMC(atol=…)가 오차 추정 ≤ atol, "
        "실제 오차 ≤ 1.5·atol, 실제 오차 ≤ 2·추정 + 여유다. 상대오차/(u·조건수)와 tvn의 영역을 정하는 조건수는 sd 자신의 "
        "값의 중앙 차분으로 잰다."
    ),
    "array": (
        "원소별 함수(vForce를 쓰는 float64 함수 15개와 pow, 복소수의 절댓값)의 참값은 tests/array/ref의 Arb 참값이다"
        "(tools/array_ref.py; test_vforce.py의 범위와 함수마다 오차가 몰리는 곳: 영점·극 근처, 정의역의 끝, "
        "서브노멀과 아주 큰 인자). 계약은 함수마다 참값을 double로 반올림한 값에서 정해진 ULP 안이다"
        "(docs/api/array.md, tests/array/refdata.py의 ULP_BOUND: exp·acos·atan·atan2·pow 3, log·log2·log10·asin·"
        "sinh·cosh·tanh·asinh·acosh 4, sin·atanh 5, 복소수의 절댓값 3). 계약 대비는 그 점의 ULP를 한계로 나눈 값이고, "
        "numpy와의 ULP 거리는 기록만 한다. 리덕션의 참값과 조건수는 tests/array/ref/reductions.npz(같은 도구)에 있다: "
        "정확한 계산으로 한 번 만들었다(합은 Shewchuk의 fsum으로 올바르게 반올림한 값과 그 반올림 오차, 내적과 행렬 곱은 "
        "곱을 정확히 나눈 뒤의 fsum, 분산·표준편차는 정확한 유리수, 곱은 Arb). 입력은 함께 저장하고, 20만 원소를 넘는 "
        "것은 시드와 모양으로 다시 만들어 저장한 SHA-256과 견준다. 문서에 수치 계약이 없어 계약 대비는 없고, 기준선보다 "
        "나빠지면 compare가 짚는다. 영역 열은 입력의 종류와 모양이다."
    ),
    "linalg": (
        "행렬은 tests/linalg/ref(tools/linalg_ref.py)의 것이다: 정사각·대칭(양의 정부호 포함)·직사각 행렬을 조건수 "
        "κ₂ = 1–1e12에서 U diag(s) Vᵀ로 만들고, expm은 노름 1e-2–50의 정규·비정규 행렬이다. 계약은 잔차다"
        "(docs/design/linalg.md §13, tests/linalg/refdata.py의 residual()과 conftest.py의 RESIDUAL_C; 테스트 "
        "tests/linalg/test_reference.py가 같은 함수로 판정한다): 잔차를 n·eps·척도로 나눈 값이 c = 10 이하다"
        "(n = max(m, n), 무한 노름: 벡터는 원소 절댓값의 최대, 행렬은 행 합의 최대). solve·lu_solve는 "
        "‖Ax − b‖/(‖A‖‖x‖ + ‖b‖), inv는 ‖AX − I‖/(‖A‖‖X‖ + 1), lstsq는 ‖Aᵀ(Ax − b)‖/(‖Aᵀ‖(‖A‖‖x‖ + ‖b‖)), "
        "lu·qr·cholesky·svd는 ‖A − 인수의 곱‖/‖A‖, eigh는 ‖AV − V diag(w)‖/‖A‖, eig는 ‖AV − V diag(w)‖/(‖A‖‖V‖), "
        "qr_pivoted는 ‖A[:, perm] − QR‖/‖A‖이고, 직교 인수는 ‖QᴴQ − I‖도 본다. R의 qr(LAPACK = TRUE)는 열을 바꾸는 "
        "QR(geqp3)이라 qr_pivoted와 견준다. 답이 하나로 정해지는 값(solve_x·lstsq_x는 해, det, eigvalsh, svdvals, "
        "eigvals, expm)은 Arb 참값(주어진 행렬 그대로의 LU, acb_mat.eig, AᵀA의 고유값의 제곱근, arb_mat.exp) 대비 "
        "오차를 기록만 한다(수치 계약이 없고, compare가 기준선과 견준다). 그 조건수는 solve_x가 κ∞(A), lstsq_x가 "
        "κ₂(A), det가 Σ|a_ij (A⁻¹)_ji|, eigvalsh·svdvals가 ‖A‖₂/|λ|, eigvals가 ‖A‖₂/(|λ|·|y_lᴴ x_r|), expm이 "
        "max(1, ‖A‖)다. 영역 열은 κ₂다."
    ),
    "regression": (
        "참값은 tests/regression/ref의 Arb 참값이다(tools/regression_ref.py). 문제마다 결과의 필드 하나가 한 줄이다"
        "(영역 열). 문서에 적힌 수치 계약은 선형 모형(ols·wls)의 계수 벡터다(docs/design/regression.md §3.1): "
        "설계 행렬의 κ₂ ≤ 1e11이면 노름 상대오차 ≤ 1e-14이고, params 줄의 계약 대비가 그것이다(κ₂는 상수 열과 "
        "가중치의 제곱근을 곱한 설계 행렬의 특잇값으로 잰다). 다른 필드는 수치 계약이 없어 계약 대비가 없고, "
        "기준선보다 나빠지면 compare가 짚는다. 비교 패키지는 statsmodels, scikit-learn(ElasticNet), R(lm·glm, MASS의 "
        "rlm, robustbase의 lmrob, glmnet)이다. 적합한 모형의 검정(model_tests)은 statsmodels의 같은 이름 함수와 R의 "
        "car(linearHypothesis·Anova·vif)·lmtest(lrtest·gqtest·resettest, White 식의 bptest), anova·drop1, "
        "영향 측도(hatvalues·cooks.distance·dffits·dfbetas)이고, 변수 선택(selection)은 참값의 stepwise 경로와 크기별 "
        "부분집합을 statsmodels로 다시 적합한 AIC·BIC와 leaps의 regsubsets가 낸 Cp·수정 R²다(R step의 AIC는 상수가 "
        "달라 넣지 않는다). 가중치가 0인 관측은 선택 비교에서 뺀다. 변수 선택의 best_cp 줄에서 sd의 최대 상대오차와 "
        "계약 대비는 참값이 아니라 n에 대한 상대오차(|sd − 참값|/n, 한계 1e-13)다: Cp는 크기가 n인 항들의 차라 0 근처를 "
        "지난다(docs/api/regression.md §9.3). 그 줄의 비교 패키지 열은 참값에 대한 상대오차 그대로다."
    ),
    "optimize": (
        "반복 방법의 결과는 멈춤 판정을 만족한 점이라, 계약은 결과만으로 확인되는 판정이다(docs/design/optimize.md §4). "
        "구간법(brentq·toms748·bisect)의 기본 허용오차에서는 근이 Arb 근(tests/optimize/ref/scalar_roots.json, "
        "tools/optimize_ref.py; 근 문제 204개)에서 xtol + rtol·|x*| 안이거나 그 점의 f가 double로 정확히 0이다"
        "(계약 대비는 거리/(xtol + rtol·|x|)). 기울기로 멈추는 minimize(bfgs, trust-ncg, trust-exact, 그리고 "
        "l-bfgs-b는 투영 기울기로 멈추는 문제 목록)는 수렴했다고 하면 정확한 기울기의 노름이 gtol 이하다(bfgs·"
        "l-bfgs-b는 ∞-노름 1e-5, trust는 2-노름 1e-4; MGH 문제 tests/optimize/mgh.py, 헤시안이 필요한 방법은 "
        "n ≤ 10). 그 밖은 기록만 한다(기준선과의 비교): xtol = 0과 find_root의 근의 ULP(Arb 근은 이웃한 두 double의 "
        "괄호라 0은 괄호 안, 곧 그 둘 중 하나다; 근이 0이면 절대 거리), minimize_scalar(brent·golden·bounded)와 "
        "find_minimum이 멈춘 점에서 가장 가까운 Arb 극소점까지의 거리(절대·상대·허용오차 대비; 문제 57개, "
        "tests/optimize/ref/scalar_minima.json), 수렴한 minimize의 ‖∇f‖∞와 MGH의 알려진 최솟값 f* 대비 초과, "
        "least_squares가 수렴한 적합의 NIST StRD 인증값(27문제 × 시작점 2) 대비 모수 상대오차(인증값에 6자리까지 "
        "닿는 실행의 목록과 다른 점에서 멈춘 적합), root(MINPACK 방정식계 55경우)가 수렴한 점의 ‖F‖∞, lsq_linear·nnls의 "
        "KKT 잔차, approx_fprime(기본 차분)과 해석적 도함수의 노름 상대 차이(MGH 47경우의 x0; 도함수는 "
        "tests/optimize/ref/derivatives.npz, tools/optimize_ref.py로 Arb에서 계산했다). 비교 패키지는 "
        "scipy(같은 이름의 방법)와 R이다. R은 reticulate로 같은 Python 시험 함수를 불러(함수 값이 같은 비트) "
        "uniroot(brentq와 같은 xtol), optimize(minimize_scalar 'bounded'와 같은 Brent fmin, tol = xatol = 1e-5), "
        "optim(BFGS·L-BFGS-B·Nelder-Mead), minpack.lm의 nls.lm('lm')을 잰다. 반복 상한과 허용오차는 R에 같은 설정이 "
        "있으면 Python 호출의 기본값에 맞췄다: optim은 maxit = 200n(Nelder-Mead는 f 호출 수, BFGS는 반복 수), "
        "L-BFGS-B는 maxit = 15000·lmm = 10·pgtol = 1e-5, nls.lm은 ftol = ptol = gtol = 1e-8·maxfev = 100n·maxiter = 1024. "
        "멈춤 판정 자체는 R의 것이다(optim의 BFGS·Nelder-Mead는 f의 상대 감소 √eps로 멈춘다). 속도 측정"
        "(bench/speed)도 같은 호출을 잰다."
    ),
    "spatial": (
        "참값은 tests/spatial/ref의 Arb와 정확한 유리수 참값이다(tools/spatial_ref.py; 시험 점은 tests/spatial/points.py). "
        "묶음(참값 파일)마다 한 줄 묶음이고 거리 방법(가중 판 w_…)이 영역이며 짝 하나가 점 하나다. sd의 값은 거리 함수"
        "(sd.spatial.euclidean(u, v) 등)이고 pdist·cdist는 같은 짝에 같은 비트다(한 짝 커널). 계약은 "
        "tests/spatial/refdata.py의 한계다(docs/design/spatial.md §4; 테스트가 같은 함수로 판정한다): 합의 깊이 "
        "m = min(n, 32) + ⌈n/32⌉ − 1의 γ로 적은 거리마다의 상대 한계, chebyshev와 가중치 없는 hamming·jaccard는 올바른 "
        "반올림, cosine·correlation은 절대와 상대를 섞은 한계, mahalanobis는 |z|ᵀ|VI||z|에 대한 한계, haversine은 20u다. "
        "비교 패키지는 scipy 1.18.1(같은 이름의 거리 함수), scikit-learn 1.9.1(euclidean_distances·manhattan_distances·"
        "cosine_distances·haversine_distances: 전개식과 1 − cos의 꼴), R 4.6.2의 stats::dist(euclidean·manhattan·maximum·"
        "minkowski, 좌표가 음이 아닌 정상 수이고 0/0 항이 없는 짝의 canberra(R의 항은 |x − y|/|x + y|이고 분모가 DBL_MIN 아래인 항을 빼고 비율을 다시 맞춘다), counting의 jaccard에 binary)이고 같은 짝에서 그 자리에서 잰다. neighbors 묶음(SPT2)은 neighbors.npz의 질의마다 "
        "k = 5 이웃(sd.spatial.kneighbors; euclidean·cityblock·chebyshev)이 점 하나이고 거리와 점 집합이 영역이다: 격자 집합은 "
        "Arb의 (RN(d), 색인) 순서와 거리가 같으면 0, 아니면 inf이고, 그 밖은 계약 η 안의 참 kNN인지(돌려준 점의 참 거리 ≤ "
        "d₍ₖ₎(1 + η), 빠진 점 ≥ d₍ₖ₎(1 − η))를 ηd₍ₖ₎ 단위로 잰다. 비교 패키지는 scipy의 KDTree.query와 scikit-learn의 "
        "NearestNeighbors(algorithm='brute')다. kdtree 묶음(SPT3)은 같은 점과 판정을 sd.spatial.KDTree(X).query(Q, k=5, "
        "p=1·2·inf)로 재고, 비교 패키지는 scipy와 scikit-learn의 KDTree.query다."
    ),
    "interpolate": (
        "참값은 tests/interpolate/ref의 정확한 유리수 참값이다(tools/interpolate_ref.py; 시험 데이터는 "
        "tests/interpolate/samples.py). linear 묶음(IP1)은 linear.npz의 정확한 a + t(b − a)를 한 번 반올림한 값과 "
        "sd.interp를 견준다: curves 영역은 실수 곡선마다 [x[0], x[-1]] 안의 평가점(곡선 전체로 부른다), intervals 영역은 "
        "선형 커널의 구간(끝점 둘레의 이웃 double, 반대 부호, 거의 같은 값, ±0, 비정상 수와 넘치는 폭)의 안쪽 점(두 노드로 "
        "부른다)이다. 바깥은 sd.interp가 numpy처럼 끝값으로 자르므로 재지 않는다. 계약은 tests/interpolate/refdata.py의 "
        "linear_bound다(docs/design/interpolate.md §4: γ₆(|a| + |b|) + 3η). 비교 패키지는 numpy 2.5.4의 interp, scipy "
        "1.18.1의 make_interp_spline(k=1), R 4.6.2의 approx(rule = 2)이고 같은 점에서 그 자리에서 잰다. 비교 패키지가 "
        "유한하지 않은 값을 낸 점(numpy의 기울기가 비정상 수 간격에서 넘치는 곳 등)은 excluded로 센다. "
        "IP2의 세 묶음: coefficients는 3차 계열의 노드 기울기(마지막 노드 빼고, 계수의 c₁)를 slopes.npz와 견주고 계약은 "
        "refdata의 spline_bound·pchip_bound·akima_bound다. evaluation은 그 객체를 곡선의 평가점(끝 조각은 늘임)에서 "
        "계산해 자기 계수를 정확한 유리수로 평가한 값과 견주고(계약 ppoly_bound), e2e_rel은 curves.npz(정확한 자료의 정확한 "
        "보간)와의 끝에서 끝 상대오차다. roots는 roots.npz의 참값 근마다 가장 가까운 근까지의 ulp와 빠뜨린 근(missed)·"
        "참값에 없는 근(extra)의 수다. 비교 패키지는 scipy(자기 계수와 자기 평가)와 R의 splinefun(natural, periodic; "
        "끝에서 끝만, 정의역 안)이다."
    ),
    "htest": (
        "참값은 tests/htest/ref의 Arb와 정확한 유리수 참값이다(tools/htest_ref.py; 문제는 tests/htest/problems.py). "
        "묶음(검정의 족)마다 한 줄이고, 결과의 필드(통계량, p값, 자유도, 신뢰구간)가 영역이며 문제 하나가 점 하나다. "
        "계약은 tests/htest/refdata.py의 척도다(docs/design/htest.md §4; 테스트가 같은 함수로 판정한다): 통계량은 "
        "상대 1e-14·max(1, κ_stat)(κ_stat은 D'Agostino 변환이 받는 적률, Anderson–Darling의 상쇄하는 합에 대한 "
        "조건수, 그 밖은 1), p값은 1e-14·max(1, κ_p)·max(1, κ_stat)(κ_p는 통계량에 대한 p값의 조건수), "
        "Cramér–von Mises의 p값(한 표본은 Csörgő–Faraway 근사식의 1 − cdf, 두 표본의 점근은 극한 분포의 1 − cdf)은 "
        "절대 1e-15도 받고, 자유도는 1e-14, t 검정의 "
        "신뢰구간 끝은 추정값과 반폭의 크기에 대한 1e-14다. 정의한 근사(D'Agostino, AS R94, 보간 표, Pelz–Good, "
        "Csörgő–Faraway)는 그 근사식의 값이 참값이다. 비교 패키지는 scipy 1.18.1·statsmodels 0.15.0·R 4.6.2의 기록이다"
        "(tests/htest/data; tools/gen_htest_scipy.py, tools/gen_htest_r.py): 같은 문제에서 같은 척도로 잰 통계량과 "
        "p값이고, R의 통계량은 정의가 scipy와 같은 검정만이다."
    ),
}


def num(v, digits=3):
    if v is None:
        return "-"
    if isinstance(v, str):
        return v
    if math.isinf(v):
        return "inf"
    if v == 0:
        return "0"
    a = abs(v)
    if 1e-3 <= a < 1e5:
        return f"{v:.{digits}g}"
    return f"{v:.{digits - 1}e}".replace("e-0", "e-").replace("e+0", "e+")


def contract_cell(r):
    c = r.get("contract")
    if c is None:
        return "-"
    return f"**{num(c)} 초과**" if c > 1 else num(c, 2)


# (key, header, cell): a column appears in an area's table when some sweep there has its key.
COLUMNS = [
    ("ulp_max", "최대 ULP", lambda r: num(r["ulp_max"])),
    ("ulp_p999", "99.9% ULP", lambda r: num(r["ulp_p999"])),
    ("ulp_scale_max", "ULP/배율", lambda r: num(r["ulp_scale_max"])),
    ("mixed_max", "mixed (최대 / 99.9%)", lambda r: f"{num(r['mixed_max'])} / {num(r['mixed_p999'])}"),
    ("rel_max", "최대 상대오차", lambda r: num(r["rel_max"])),
    ("norm_rel", "노름 상대오차", lambda r: num(r["norm_rel"])),
    ("norm_inf_rel_max", "노름 상대오차(∞)", lambda r: num(r["norm_inf_rel_max"])),
    ("numpy_dist_max", "numpy와의 ULP", lambda r: num(r["numpy_dist_max"])),
    ("nonfinite", "NaN·inf 점", lambda r: str(r["nonfinite"])),
    ("abs_max", "최대 절대오차", lambda r: num(r["abs_max"])),
    ("rel_cond_max", "상대오차/(u·조건수)", lambda r: num(r["rel_cond_max"])),
    ("backward_max", "잔차/(n·eps·척도)", lambda r: num(r["backward_max"])),
    ("orth_max", "직교성/(n·eps)", lambda r: num(r["orth_max"])),
    ("grad_max", "최대 ‖∇f‖", lambda r: num(r["grad_max"])),
    ("excess_max", "최대 (f − f*)/max(1, f*)", lambda r: num(r["excess_max"])),
    ("tol_ratio_max", "거리/허용오차", lambda r: num(r["tol_ratio_max"])),
    ("fnorm_max", "최대 ‖F‖∞", lambda r: num(r["fnorm_max"])),
    ("kkt_max", "KKT 잔차/eps", lambda r: num(r["kkt_max"])),
    ("unconverged", "수렴 못 함", lambda r: str(r["unconverged"])),
    ("off_target", "판정 밖", lambda r: str(r["off_target"])),
    ("est_ratio_max", "오차/추정", lambda r: num(r["est_ratio_max"])),
    ("contract", "계약 대비", lambda r: contract_cell(r)),
]

# (column key, line): a line appears when its column is in the table ("region": the areas whose
# region column is the guaranteed region, not a result field).
LEGEND = [
    ("ulp_max", "- ULP: sd의 값과 참값을 double로 반올림한 값 사이에 있는 double의 수(ulp_distance; 0이면 올바르게 "
     "반올림한 값, 1이면 그 이웃). 99.9%는 그 분위수다."),
    ("ulp_scale_max", "- ULP/배율: ULP를 그 점의 계약 배율로 나눈 값. 배율은 상대 한계 1e-14에 곱하는 수로, 역함수는 "
     "max(1, κ)(κ는 입력에 대한 상대 조건수), Bessel·Airy와 복소 오차함수·wofz는 영점 근처에서 max(1, M/|f|)"
     "(M은 크기의 포락선)다. 배율이 1보다 큰 점이 있는 함수에만 있다. 계약 대비처럼 배율을 나눈 값이라 둘이 함께 "
     "움직인다(상대 계약이면 ULP/배율 약 45–90이 계약 대비 1이다). 최대 ULP 열은 배율을 나누지 않은 값이라, 배율이 큰 "
     "점(역함수가 0을 지나는 곳, 진동 함수의 영점 근처)에서는 계약 안이어도 크다."),
    ("mixed_max", "- mixed: |sd − 참값|/max(1, |참값|)를 2⁻⁵² 단위로 잰 값(45가 1e-14다). 0을 지나는 값(log 밀도, "
     "엔트로피, gammaln 등)의 계약 척도이고, 그 함수에만 있다. 최대와 99.9%다."),
    ("rel_max", "- 상대오차: |sd − 참값|/|참값|."),
    ("norm_rel", "- 노름 상대오차: 벡터 필드의 ‖sd − 참값‖₂/‖참값‖₂."),
    ("norm_inf_rel_max", "- 노름 상대오차(∞): max|sd − 참값|/max|참값|, 벡터·행렬의 원소 가운데 가장 큰 오차를 가장 큰 "
     "참값으로 나눈 값."),
    ("numpy_dist_max", "- numpy와의 ULP: sd의 값과 numpy의 값 사이에 있는 double의 수의 최대. 기록만 하고 계약에 넣지 "
     "않는다(마지막 자리는 numpy와 달라도 된다)."),
    ("nonfinite", "- NaN·inf 점: 참값은 유한한데 sd가 NaN이나 inf를 낸 점의 수. 그런 점의 오차는 inf로 센다. mixed로 "
     "재는 영역에서는 0을 지나는 값의 상대오차가 뜻이 없어 상대오차 대신 이 수를 둔다."),
    ("abs_max", "- 최대 절대오차: |sd − 참값|의 최대."),
    ("rel_cond_max", "- 상대오차/(u·조건수): 상대오차를 입력의 반올림(u = 2⁻⁵³)이 조건수만큼 커진 크기로 나눈 값. "
     "조건수는 입력마다 |d log f/d log x|의 합이다(합이면 Σ|x|/|Σx|)."),
    ("backward_max", "- 잔차/(n·eps·척도): 함수마다의 잔차(위 설명)를 n·eps와 그 척도로 나눈 값의 최대. 계약은 10 이하다."),
    ("orth_max", "- 직교성/(n·eps): 직교 인수의 ‖QᴴQ − I‖∞/(n·eps)의 최대."),
    ("grad_max", "- 최대 ‖∇f‖: 결과에서의 정확한 기울기의 노름(gradient stop 영역은 그 방법의 판정 노름)."),
    ("excess_max", "- 최대 (f − f*)/max(1, f*): MGH의 알려진 최솟값보다 높은 정도. 다른 극소점에서 멈추면 크다."),
    ("tol_ratio_max", "- 거리/허용오차: 가장 가까운 Arb 극소점까지의 거리를 그 점에서 방법의 멈춤 허용오차(brent "
     "1.48e-8·|x*| + 1e-11, golden 1.49e-8·|x*|, bounded √eps·|x*| + xatol/3, find_minimum √eps·|x*|)로 나눈 값. "
     "극소점에서 f가 평평할수록(f″ = 0) 크다: f를 double로 구분할 수 있는 거리가 허용오차보다 넓다."),
    ("fnorm_max", "- 최대 ‖F‖∞: 수렴한 점에서의 방정식계의 잔차."),
    ("kkt_max", "- KKT 잔차/eps: 자유 변수의 기울기와 경계에서 잘못된 부호의 기울기 가운데 가장 큰 것을 "
     "eps·‖A‖₁(‖A‖∞‖x‖∞ + ‖b‖∞)로 나눈 값."),
    ("unconverged", "- 수렴 못 함: 수렴하지 못했거나 에러를 낸 실행의 수(그 실행은 다른 열에 들지 않는다)."),
    ("off_target", "- 판정 밖: 영역의 문제 목록(l-bfgs-b가 투영 기울기로 멈추는 문제, least_squares가 인증값에 6자리까지 "
     "닿는 실행)에 있는데 그 판정에 닿지 못한 실행의 수. 목록은 sd의 결과와 상관없이 고정이라 늘면 compare가 짚는다."),
    ("est_ratio_max", "- 오차/추정: 실제 오차를 sd가 돌려준 오차 추정(표준오차의 3배)으로 나눈 값의 최대. "
     "추정은 통계적이라 1을 넘는 점이 있다."),
    ("contract", "- 계약 대비: 그 영역의 계약 척도(보장 영역은 상대 1e-14, 아주 작은 값은 절대, mixed는 "
     "|오차|/max(1, |값|))로 잰 가장 큰 오차를 한계로 나눈 값. 1 이하가 계약 안이다."),
    ("region", "- 영역: guaranteed는 보장 영역, loose와 mixed는 그 밖(mixed는 그 척도로 잰다)."),
]  # fmt: skip


def _legend(area: str, cols, group=None) -> list[str]:
    keys = {c[0] for c in cols} | ({"region"} if area in ("special", "stats") and group is None else set())
    return [line for key, line in LEGEND if key in keys]


def parts(area: str) -> list[tuple[str, str, str | None]]:
    """The tables of an area: (path, title, group); group "mv" holds the sweeps named mv/...,
    None the others."""
    if area == "stats":
        return [("bench/stats/accuracy.md", HEAD["stats"], None), (MV_PATH, MV_HEAD, "mv")]
    return [(f"bench/{area}/accuracy.md", HEAD.get(area, area), None)]


def _in_group(sw, group) -> bool:
    return sw["name"].startswith(MV) if group == "mv" else not sw["name"].startswith(MV)


def render(area: str, data: dict, group: str | None = None) -> str:
    """The table of an area's sweeps in data (group "mv": the stats area's multivariate group)."""
    sweeps = sorted((s for s in data["sweeps"] if s["area"] == area and _in_group(s, group)), key=lambda s: s["name"])
    runs = data.get("runs") or {data["run"]: data["env"]}
    used = {s.get("run") for s in sweeps} | {s.get("comparators_run") for s in sweeps}
    if None not in used and set(runs) & used:
        runs = {k: v for k, v in runs.items() if k in used}
    present = {k for sw in sweeps for r in sw["regions"].values() for k, v in r.items() if v is not None}
    cols = [c for c in COLUMNS if c[0] in present]
    source = f"{area}.json" + (" (mv/로 시작하는 sweep)" if group == "mv" else "")
    title = MV_HEAD if group == "mv" else HEAD.get(area, area)
    lines = [
        f"# {title} 정확도",
        "",
        f"`.venv/bin/python -m bench.accuracy promote`가 bench/accuracy/results/{source}에서 만든다. 손으로 고치지 않는다.",
        "",
        INTRO.get(group or area) or f"참값은 tests/{area}/ref의 Arb 참값이다. 오차는 그 파일의 모든 점에서 잰다.",
        "",
        *_legend(area, cols, group),
        "",
        "측정: " + ", ".join(_run_words(k, v, area) for k, v in sorted(runs.items())) + ".",
        "",
    ]
    ccols = _comparator_columns(sweeps)
    if ccols:
        at = next(i for i, line in enumerate(lines) if line.startswith("측정: "))
        lines[at - 1 : at - 1] = [_comparator_legend(data, ccols, sweeps)]
    heads = [c[1] for c in cols] + [f"{lib} {h}" for lib, _, h in ccols]
    lines.append("| 함수 | 영역 | 점 | " + " | ".join(heads) + " |")
    lines.append("|---|---|---:|" + "---:|" * len(heads))
    for sw in sweeps:
        name = sw["name"][len(MV) :] if group == "mv" else sw["name"]
        for label, r in sw["regions"].items():
            cells = [c[2](r) if c[0] in r and r[c[0]] is not None else "" for c in cols]
            cells += [_comparator_cell(sw, lib, key, label) for lib, key, _ in ccols]
            lines.append(f"| `{name}` | {label} | {r['points']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _run_words(run, env, area) -> str:
    """A run on the "측정:" line: its date, commit (dirty: the working tree differed) and the
    SHA-256 of the area's references it measured."""
    words = f"{env.get('date', '?')}, {env.get('commit', '?')}{' dirty' if env.get('dirty') else ''}"
    sha = (env.get("references") or {}).get(area)
    if sha:
        words += f", 참값 {sha[:12]}"
    return f"{run} ({words})"


# the comparator measures shown, in this order, where some library has them
COMPARATOR_KEYS = [
    ("ulp_max", "최대 ULP"), ("ulp_p999", "99.9% ULP"), ("mixed_max", "mixed 최대"), ("rel_max", "최대 상대오차"),
    ("norm_rel", "노름 상대오차"), ("rel_cond_max", "상대오차/(u·조건수)"), ("backward_max", "잔차/(n·eps·척도)"),
    ("orth_max", "직교성/(n·eps)"), ("grad_max", "최대 ‖∇f‖"), ("fnorm_max", "최대 ‖F‖∞"),
    ("tol_ratio_max", "거리/허용오차"),
    ("kkt_max", "KKT 잔차/eps"), ("unconverged", "수렴 못 함"), ("contract", "계약 대비"), ("excluded", "제외"),
]  # fmt: skip


def _comparator_columns(sweeps):
    libs = []
    for sw in sweeps:
        for lib in sw.get("comparators") or {}:
            if lib not in libs:
                libs.append(lib)
    out = []
    for lib in libs:
        seen = set()
        for sw in sweeps:
            regs = (sw.get("comparators") or {}).get(lib)
            if isinstance(regs, dict) and "error" not in regs:
                for r in regs.values():
                    seen |= {k for k, v in r.items() if v not in (None, 0) or k == "contract" and v is not None}
        out += [(lib, k, h) for k, h in COMPARATOR_KEYS if k in seen]
        if not any(c[0] == lib for c in out):  # "not run" or errors only: one column to say so
            out.append((lib, "ulp_max", "최대 ULP"))
    return out


def _comparator_cell(sw, lib, key, label):
    regs = (sw.get("comparators") or {}).get(lib)
    if regs is None:
        return "—"
    if isinstance(regs, str):
        return regs
    if "error" in regs:
        e = regs["error"]
        if "Guarded" in e:
            return "멈춤"
        return "시작값 필요" if "starting values" in e else "에러"
    r = regs.get(label)
    if r is None or r.get(key) is None:
        return "—" if r is None else ""
    if key == "contract":
        return contract_cell(r)
    if key in ("excluded", "unconverged"):
        return str(r[key])
    return num(r[key])


def _comparator_legend(data, ccols, sweeps):
    libs = []
    for lib, _, _ in ccols:
        if lib not in libs:
            libs.append(lib)
    vers = {}
    for env in (data.get("runs") or {data.get("run"): data.get("env", {})}).values():
        vers.update({k: v for k, v in (env or {}).get("comparators", {}).items() if v})
    names = ", ".join(f"{lib} {vers[lib]}" if lib in vers else lib for lib in libs)
    secs = sorted({sw["comparator_seconds"] for sw in sweeps if sw.get("comparator_seconds")})
    limit = "·".join(f"{s:g}" for s in secs) + "초" if secs else "정해진 시간"
    boost = ""
    if "Boost" in libs:
        own = vers.get("Boost in sd.special")
        boost = (
            " Boost 열은 비교용 체크아웃(" + (vers.get("Boost") or "boostorg/math") + ")의 것이고, sd.special에 담긴 "
            "Boost.Math는 따로 고정한 사본이다("
            + (f"커밋 {own}, " if own else "")
            + "third_party/boost_math/README.md)."
        )
    return (
        f"- 비교 패키지({names}): 같은 참값, 같은 점에서 같은 척도로 잰 그 라이브러리의 오차다. 같은 양을 계산하는 "
        "공개 함수가 없으면 —, R이 없는 기계에서는 not run, 함수가 입력 전체를 거부하면 에러, 시간("
        + limit
        + ")·메모리(2GB) 한도를 넘으면 멈춤, R의 glm이 기본 설정으로 시작값을 찾지 못하면 시작값 필요다. 제외: 참값은 "
        "유한한데 그 라이브러리가 NaN·inf를 낸 점의 수(정의가 다르거나 실패한 점; 다른 열에서 빠진다). compare는 이 "
        "열들의 변화를 알리기만 하고 종료 코드에 넣지 않는다." + boost
    )


def write(areas, data=None) -> list:
    out = []
    for area in areas:
        d = data or results.load_baseline(area)
        if d is None:
            continue
        for rel, _, group in parts(area):
            if not any(sw["area"] == area and _in_group(sw, group) for sw in d["sweeps"]):
                continue
            path = results.ROOT / rel
            path.parent.mkdir(parents=True, exist_ok=True)  # bench/<area>/ may hold nothing else
            path.write_text(render(area, d, group))
            out.append(path)
    return out
