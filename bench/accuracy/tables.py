"""bench/<area>/accuracy.md, generated from the area's baseline (or a run file)."""

from __future__ import annotations

import math

from . import results

HEAD = {
    "special": "sd.special",
    "stats": "sd.stats",
    "regression": "sd.regression",
    "multivariate": "sd.stats 다변량·행렬 분포",
    "array": "sd 배열",
    "linalg": "sd.linalg",
    "optimize": "sd.optimize",
}

# where an area's table goes (default bench/<area>/accuracy.md)
PATH = {"multivariate": "bench/stats/accuracy_multivariate.md"}

# the paragraph under the title (default: the Arb references of tests/<area>/ref)
INTRO = {
    "multivariate": (
        "참값은 Arb로 계산한 값이다: 밀도·엔트로피는 tests/stats/ref_mv(tools/mv_ref.py; 행렬식과 풀이는 주어진 "
        "행렬 그대로 Arb의 LU), cdf는 tests/stats/ref_mvcdf(tools/mvcdf_ref.py; factor_t는 2단 적응 구적의 약 1e-11 "
        "참값)다. 밀도·엔트로피의 점은 모두 보장 영역(κ ≤ 1e9, d·p ≤ 100, Wishart df ≤ 1e30) 안이고 계약은 "
        "|오차| ≤ 1e-14·max(1, |값|), pdf는 그 오차가 exp를 지난 상대 1e-14·max(1, |logpdf|)다(|logpdf| < 700인 점). "
        "cdf의 영역과 한계는 tests/stats/test_mvcdf.py의 것이다: guaranteed는 상대 1e-14, loose는 그 밖의 정규수 "
        "범위(bvn 10·eps·max(1, −ln P), tvn 8u·조건수), tiny는 P < 1e-300(절대 1e-300), bvt·tvt는 상대 5e-15·1e-14, "
        "QMC(atol=…)는 오차 추정 ≤ atol, 실제 오차 ≤ 1.5·atol, 실제 오차 ≤ 2·추정 + 여유이고, factor_t의 d ≤ 3은 "
        "정확한 혼합이라 상대 1e-13이다. 상대오차/(u·조건수)의 조건수는 sd 자신의 값의 중앙 차분으로 잰다."
    ),
    "array": (
        "원소별 함수(vForce를 쓰는 float64 함수 15개와 pow)의 참값은 tests/array/ref의 Arb 참값이다"
        "(tools/array_ref.py; test_vforce.py의 범위와 함수마다 오차가 몰리는 곳: 영점·극 근처, 정의역의 끝, "
        "서브노멀과 아주 큰 인자). 계약은 numpy와 8 ULP 안(docs/api/array.md)이라 계약 대비는 numpy까지의 ULP 거리를 "
        "8로 나눈 값이고, 같은 점에서 numpy 자신의 Arb 대비 오차를 함께 적는다. 리덕션의 참값과 조건수는 "
        "tests/array/ref/reductions.npz(같은 도구)에 있다: 정확한 계산으로 한 번 만들었다(합은 Shewchuk의 fsum으로 "
        "올바르게 반올림한 값과 그 반올림 오차, 내적과 행렬 곱은 곱을 정확히 나눈 뒤의 fsum, 분산·표준편차는 정확한 "
        "유리수, 곱은 Arb). 입력은 함께 저장하고, 20만 원소를 넘는 것은 시드와 모양으로 다시 만들어 저장한 SHA-256과 "
        "견준다. 문서에 수치 계약이 없어 계약 대비는 없고, 기준선보다 나빠지면 compare가 짚는다. 영역 열은 입력의 "
        "종류와 모양이다."
    ),
    "linalg": (
        "행렬은 tests/linalg/ref(tools/linalg_ref.py)의 것이다: 정사각·대칭(양의 정부호 포함)·직사각 행렬을 조건수 "
        "κ₂ = 1–1e12에서 U diag(s) Vᵀ로 만들고, expm은 노름 1e-2–50의 정규·비정규 행렬이다. 계약은 잔차다"
        "(docs/design/linalg.md §13, tests/linalg/conftest.py의 RESIDUAL_C): 잔차를 n·eps·척도로 나눈 값이 c = 10 "
        "이하다(n = max(m, n), 무한 노름). solve·lu_solve는 ‖Ax − b‖/(‖A‖‖x‖ + ‖b‖), inv는 ‖AX − I‖/(‖A‖‖X‖ + 1), "
        "lstsq는 ‖Aᵀ(Ax − b)‖/(‖A‖(‖A‖‖x‖ + ‖b‖)), lu·qr·cholesky·svd는 ‖A − 인수의 곱‖/‖A‖, eigh는 "
        "‖AV − V diag(w)‖/‖A‖, eig는 ‖AV − V diag(w)‖/(‖A‖‖V‖)이고, 직교 인수는 ‖QᴴQ − I‖도 본다. "
        "답이 하나로 정해지는 값(solve_x·lstsq_x는 해, det, eigvalsh, svdvals, eigvals, expm)은 Arb 참값(주어진 "
        "행렬 그대로의 LU, acb_mat.eig, AᵀA의 고유값의 제곱근, arb_mat.exp) 대비 오차를 기록만 한다(수치 계약이 "
        "없고, compare가 기준선과 견준다). 그 조건수는 solve_x가 κ∞(A), lstsq_x가 κ₂(A), det가 Σ|a_ij (A⁻¹)_ji|, "
        "eigvalsh·svdvals가 ‖A‖₂/|λ|, eigvals가 ‖A‖₂/(|λ|·|y_lᴴ x_r|), expm이 max(1, ‖A‖)다. 영역 열은 κ₂다."
    ),
    "optimize": (
        "반복 방법의 결과는 멈춤 판정을 만족한 점이라, 계약은 결과만으로 확인되는 판정이다(docs/design/optimize.md §4). "
        "구간법(brentq·toms748·bisect)의 기본 허용오차에서는 근이 Arb 근(tests/optimize/data/scalar_roots.json, "
        "tools/optimize_ref.py; 근 문제 204개)에서 xtol + rtol·|x*| 안이거나 그 점의 f가 double로 정확히 0이다"
        "(계약 대비는 거리/(xtol + rtol·|x|)). 기울기로 멈추는 minimize(bfgs, trust-ncg, trust-exact, 투영 기울기로 "
        "멈춘 l-bfgs-b)는 수렴했다고 하면 정확한 기울기의 노름이 gtol 이하다(bfgs·l-bfgs-b는 ∞-노름 1e-5, trust는 "
        "2-노름 1e-4; MGH 문제 tests/optimize/mgh.py, 헤시안이 필요한 방법은 n ≤ 10). 그 밖은 기록만 한다(기준선과의 "
        "비교): xtol = 0과 find_root의 근의 ULP(근이 0이면 절대 거리), 수렴한 minimize의 ‖∇f‖∞와 MGH의 알려진 "
        "최솟값 f* 대비 초과, least_squares가 NIST StRD 인증값(27문제 × 시작점 2)에 6자리까지 닿은 적합의 모수 "
        "상대오차와 다른 점에서 멈춘 적합, root(MINPACK 방정식계 55경우)가 수렴한 점의 ‖F‖∞, lsq_linear·nnls의 "
        "KKT 잔차, approx_fprime(기본 차분)과 해석적 도함수의 노름 상대 차이(MGH 47경우의 x0; 도함수는 "
        "tests/optimize/ref/derivatives.npz, tools/optimize_accuracy_ref.py로 Arb에서 계산했다)."
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
    ("ulp_cond_max", "ULP/배율", lambda r: num(r["ulp_cond_max"])),  # the key keeps its old name: baselines hold it
    ("mixed_max", "mixed (최대 / 99.9%)", lambda r: f"{num(r['mixed_max'])} / {num(r['mixed_p999'])}"),
    ("rel_max", "최대 상대오차", lambda r: num(r["rel_max"])),
    ("norm_rel", "노름 상대오차", lambda r: num(r["norm_rel"])),
    ("nonfinite", "NaN·inf 점", lambda r: str(r["nonfinite"])),
    ("abs_max", "최대 절대오차", lambda r: num(r["abs_max"])),
    ("rel_cond_max", "상대오차/(u·조건수)", lambda r: num(r["rel_cond_max"])),
    ("backward_max", "잔차/(n·eps·척도)", lambda r: num(r["backward_max"])),
    ("orth_max", "직교성/(n·eps)", lambda r: num(r["orth_max"])),
    ("grad_max", "최대 ‖∇f‖", lambda r: num(r["grad_max"])),
    ("excess_max", "최대 (f − f*)/max(1, f*)", lambda r: num(r["excess_max"])),
    ("fnorm_max", "최대 ‖F‖∞", lambda r: num(r["fnorm_max"])),
    ("kkt_max", "KKT 잔차/eps", lambda r: num(r["kkt_max"])),
    ("unconverged", "수렴 못 함", lambda r: str(r["unconverged"])),
    ("est_ratio_max", "오차/추정", lambda r: num(r["est_ratio_max"])),
    ("contract", "계약 대비", lambda r: contract_cell(r)),
]

# (column key, line): a line appears when its column is in the table ("region": the areas whose
# region column is the guaranteed region, not a result field).
LEGEND = [
    ("ulp_max", "- ULP: |sd − 참값|을 참값에서의 float64 간격으로 나눈 값. 99.9%는 그 분위수다."),
    ("ulp_cond_max", "- ULP/배율: ULP를 그 점의 계약 배율로 나눈 값. 배율은 상대 한계 1e-14에 곱하는 수로, 역함수는 "
     "max(1, κ)(κ는 입력에 대한 상대 조건수), Bessel·Airy와 복소 오차함수·wofz는 영점 근처에서 max(1, M/|f|)"
     "(M은 크기의 포락선)다. 배율이 1보다 큰 점이 있는 함수에만 있다. 계약 대비처럼 배율을 나눈 값이라 둘이 함께 "
     "움직인다(상대 계약이면 ULP/배율 약 45–90이 계약 대비 1이다). 최대 ULP 열은 배율을 나누지 않은 값이라, 배율이 큰 "
     "점(역함수가 0을 지나는 곳, 진동 함수의 영점 근처)에서는 계약 안이어도 크다."),
    ("mixed_max", "- mixed: |sd − 참값|/max(1, |참값|)를 2⁻⁵² 단위로 잰 값(45가 1e-14다). 0을 지나는 값(log 밀도, "
     "엔트로피, gammaln 등)의 계약 척도이고, 그 함수에만 있다. 최대와 99.9%다."),
    ("rel_max", "- 상대오차: |sd − 참값|/|참값|."),
    ("norm_rel", "- 노름 상대오차: 벡터 필드의 ‖sd − 참값‖₂/‖참값‖₂."),
    ("nonfinite", "- NaN·inf 점: 참값은 유한한데 sd가 NaN이나 inf를 낸 점의 수. 그런 점의 오차는 inf로 센다."),
    ("abs_max", "- 최대 절대오차: |sd − 참값|의 최대."),
    ("rel_cond_max", "- 상대오차/(u·조건수): 상대오차를 입력의 반올림(u = 2⁻⁵³)이 조건수만큼 커진 크기로 나눈 값. "
     "조건수는 입력마다 |d log f/d log x|의 합이다(합이면 Σ|x|/|Σx|)."),
    ("backward_max", "- 잔차/(n·eps·척도): 함수마다의 잔차(위 설명)를 n·eps와 그 척도로 나눈 값의 최대. 계약은 10 이하다."),
    ("orth_max", "- 직교성/(n·eps): 직교 인수의 ‖QᴴQ − I‖∞/(n·eps)의 최대."),
    ("grad_max", "- 최대 ‖∇f‖: 결과에서의 정확한 기울기의 노름(gradient stop 영역은 그 방법의 판정 노름)."),
    ("excess_max", "- 최대 (f − f*)/max(1, f*): MGH의 알려진 최솟값보다 높은 정도. 다른 극소점에서 멈추면 크다."),
    ("fnorm_max", "- 최대 ‖F‖∞: 수렴한 점에서의 방정식계의 잔차."),
    ("kkt_max", "- KKT 잔차/eps: 자유 변수의 기울기와 경계에서 잘못된 부호의 기울기 가운데 가장 큰 것을 "
     "eps·‖A‖₁(‖A‖∞‖x‖∞ + ‖b‖∞)로 나눈 값."),
    ("unconverged", "- 수렴 못 함: 수렴하지 못했거나 에러를 낸 실행의 수(그 실행은 다른 열에 들지 않는다)."),
    ("est_ratio_max", "- 오차/추정: 실제 오차를 sd가 돌려준 오차 추정(표준오차의 3배)으로 나눈 값의 최대. "
     "추정은 통계적이라 1을 넘는 점이 있다."),
    ("contract", "- 계약 대비: 그 영역의 계약 척도(보장 영역은 상대 1e-14, 아주 작은 값은 절대, mixed는 "
     "|오차|/max(1, |값|))로 잰 가장 큰 오차를 한계로 나눈 값. 1 이하가 계약 안이다."),
    ("region", "- 영역: guaranteed는 보장 영역, loose와 mixed는 그 밖(mixed는 그 척도로 잰다)."),
]  # fmt: skip


def _legend(area: str, cols) -> list[str]:
    keys = {c[0] for c in cols} | ({"region"} if area in ("special", "stats") else set())
    return [line for key, line in LEGEND if key in keys]


def render(area: str, data: dict) -> str:
    sweeps = sorted((s for s in data["sweeps"] if s["area"] == area), key=lambda s: s["name"])
    runs = data.get("runs") or {data["run"]: data["env"]}
    present = {k for sw in sweeps for r in sw["regions"].values() for k, v in r.items() if v is not None}
    cols = [c for c in COLUMNS if c[0] in present]
    lines = [
        f"# {HEAD.get(area, area)} 정확도",
        "",
        "`.venv/bin/python -m bench.accuracy promote`가 bench/accuracy/results/"
        f"{area}.json에서 만든다. 손으로 고치지 않는다.",
        "",
        INTRO.get(area)
        or "참값은 tests/"
        + area
        + "/ref의 Arb 참값이다. 오차는 그 파일의 모든 점에서 잰다."
        + (
            " 문제마다 결과의 필드 하나가 한 줄이고(영역 열), 문서에 적힌 수치 계약이 없어 계약 대비는 없다."
            " 기준선보다 나빠지면 compare가 짚는다."
            if area == "regression"
            else ""
        ),
        "",
        *_legend(area, cols),
        "",
        "측정: "
        + ", ".join(
            f"{k} ({v.get('date', '?')}, {v.get('commit', '?')}{' dirty' if v.get('dirty') else ''})"
            for k, v in sorted(runs.items())
        )
        + ".",  # fmt: skip
        "",
    ]
    ccols = _comparator_columns(sweeps)
    if ccols:
        at = next(i for i, line in enumerate(lines) if line.startswith("측정: "))
        lines[at - 1 : at - 1] = [_comparator_legend(data, ccols)]
    heads = [c[1] for c in cols] + [f"{lib} {h}" for lib, _, h in ccols]
    lines.append("| 함수 | 영역 | 점 | " + " | ".join(heads) + " |")
    lines.append("|---|---|---:|" + "---:|" * len(heads))
    for sw in sweeps:
        for label, r in sw["regions"].items():
            cells = [c[2](r) if c[0] in r and r[c[0]] is not None else "" for c in cols]
            cells += [_comparator_cell(sw, lib, key, label) for lib, key, _ in ccols]
            lines.append(f"| `{sw['name']}` | {label} | {r['points']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


# the comparator measures shown, in this order, where some library has them
COMPARATOR_KEYS = [
    ("ulp_max", "최대 ULP"), ("ulp_p999", "99.9% ULP"), ("mixed_max", "mixed 최대"), ("rel_max", "최대 상대오차"),
    ("norm_rel", "노름 상대오차"), ("rel_cond_max", "상대오차/(u·조건수)"), ("backward_max", "잔차/(n·eps·척도)"),
    ("orth_max", "직교성/(n·eps)"), ("grad_max", "최대 ‖∇f‖"), ("fnorm_max", "최대 ‖F‖∞"),
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
        return "멈춤" if "Guarded" in e else "에러"
    r = regs.get(label)
    if r is None or r.get(key) is None:
        return "—" if r is None else ""
    if key == "contract":
        return contract_cell(r)
    if key in ("excluded", "unconverged"):
        return str(r[key])
    return num(r[key])


def _comparator_legend(data, ccols):
    libs = []
    for lib, _, _ in ccols:
        if lib not in libs:
            libs.append(lib)
    vers = {}
    for env in (data.get("runs") or {data.get("run"): data.get("env", {})}).values():
        vers.update({k: v for k, v in (env or {}).get("comparators", {}).items() if v})
    names = ", ".join(f"{lib} {vers[lib]}" if lib in vers else lib for lib in libs)
    return (
        f"- 비교 패키지({names}): 같은 참값, 같은 점에서 같은 척도로 잰 그 라이브러리의 오차다. 같은 양을 계산하는 "
        "공개 함수가 없으면 —, R이 없는 기계에서는 not run, 함수가 입력 전체를 거부하면 에러, 시간(30초)·메모리(2GB) "
        "한도를 넘으면 멈춤이다. 제외: 참값은 유한한데 그 라이브러리가 NaN·inf를 낸 점의 수(정의가 다르거나 실패한 "
        "점; 다른 열에서 빠진다). compare는 이 열들의 변화를 알리기만 하고 종료 코드에 넣지 않는다."
    )


def write(areas, data=None) -> list:
    out = []
    for area in areas:
        d = data or results.load_baseline(area)
        if d is None:
            continue
        path = results.ROOT / PATH.get(area, f"bench/{area}/accuracy.md")
        path.parent.mkdir(parents=True, exist_ok=True)  # bench/<area>/ may hold nothing else
        path.write_text(render(area, d))
        out.append(path)
    return out
