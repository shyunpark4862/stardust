"""bench/<area>/accuracy.md, generated from the area's baseline (or a run file)."""

from __future__ import annotations

import math

from . import results

HEAD = {
    "special": "sd.special",
    "stats": "sd.stats",
    "regression": "sd.regression",
    "multivariate": "sd.stats 다변량·행렬 분포",
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
        "정확한 혼합이라 상대 1e-13이다."
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
    ("ulp_cond_max", "ULP/max(1, κ)", lambda r: num(r["ulp_cond_max"])),
    ("mixed_max", "mixed (최대 / 99.9%)", lambda r: f"{num(r['mixed_max'])} / {num(r['mixed_p999'])}"),
    ("rel_max", "최대 상대오차", lambda r: num(r["rel_max"])),
    ("norm_rel", "노름 상대오차", lambda r: num(r["norm_rel"])),
    ("abs_max", "최대 절대오차", lambda r: num(r["abs_max"])),
    ("rel_cond_max", "상대오차/(u·조건수)", lambda r: num(r["rel_cond_max"])),
    ("est_ratio_max", "오차/추정", lambda r: num(r["est_ratio_max"])),
    ("contract", "계약 대비", lambda r: contract_cell(r)),
]


# (column key, line) for the columns above that the fixed legend in render() does not explain
LEGEND_EXTRA = [
    ("abs_max", "- 최대 절대오차: |sd − 참값|의 최대."),
    ("rel_cond_max", "- 상대오차/(u·조건수): 입력의 반올림(u = 2⁻⁵³)이 조건수만큼 커진 크기에 대한 상대오차. 조건수는 "
     "입력마다 |d log P/d log x|의 합이다(sd 자신의 값의 중앙 차분)."),
    ("est_ratio_max", "- 오차/추정: 실제 오차를 sd가 돌려준 오차 추정(표준오차의 3배)으로 나눈 값의 최대. "
     "추정은 통계적이라 1을 넘는 점이 있다."),
]  # fmt: skip


def render(area: str, data: dict) -> str:
    sweeps = sorted((s for s in data["sweeps"] if s["area"] == area), key=lambda s: s["name"])
    runs = data.get("runs") or {data["run"]: data["env"]}
    present = {k for sw in sweeps for r in sw["regions"].values() for k, v in r.items() if v is not None}
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
        "- ULP: |sd − 참값|을 참값에서의 float64 간격으로 나눈 값. 99.9%는 그 분위수다.",
        "- ULP/max(1, κ): 역함수의 계약 단위(κ는 입력에 대한 상대 조건수). 역함수만 있다.",
        "- mixed: |sd − 참값|/max(1, |참값|)를 2⁻⁵² 단위로 잰 값(45가 1e-14다). 0을 지나는 값(log 밀도, "
        "엔트로피, gammaln 등)의 계약 척도이고, 그 함수에만 있다. 최대와 99.9%다.",
        "- 상대오차: |sd − 참값|/|참값|. 노름 상대오차: 벡터 필드의 ‖sd − 참값‖₂/‖참값‖₂.",
        "- 계약 대비: 그 영역의 계약 척도(보장 영역은 상대 1e-14, 아주 작은 값은 절대, mixed는 "
        "|오차|/max(1, |값|))로 잰 가장 큰 오차를 한계로 나눈 값. 1 이하가 계약 안이다.",
        "- 영역: guaranteed는 보장 영역, loose와 mixed는 그 밖(mixed는 그 척도로 잰다).",
        *[line for key, line in LEGEND_EXTRA if key in present],
        "",
        "측정: "
        + ", ".join(
            f"{k} ({v.get('date', '?')}, {v.get('commit', '?')}{' dirty' if v.get('dirty') else ''})"
            for k, v in sorted(runs.items())
        )
        + ".",  # fmt: skip
        "",
    ]
    cols = [c for c in COLUMNS if c[0] in present or c[0] in ("ulp_max", "rel_max")]
    lines.append("| 함수 | 영역 | 점 | " + " | ".join(c[1] for c in cols) + " |")
    lines.append("|---|---|---:|" + "---:|" * len(cols))
    for sw in sweeps:
        for label, r in sw["regions"].items():
            cells = [c[2](r) if c[0] in r and r[c[0]] is not None else "" for c in cols]
            lines.append(f"| `{sw['name']}` | {label} | {r['points']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write(areas, data=None) -> list:
    out = []
    for area in areas:
        d = data or results.load_baseline(area)
        if d is None:
            continue
        path = results.ROOT / PATH.get(area, f"bench/{area}/accuracy.md")
        path.write_text(render(area, d))
        out.append(path)
    return out
