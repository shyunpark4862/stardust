"""bench/<area>/accuracy.md, generated from the area's baseline (or a run file)."""

from __future__ import annotations

import math

from . import results

HEAD = {
    "special": "sd.special",
    "stats": "sd.stats",
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


def render(area: str, data: dict) -> str:
    sweeps = sorted((s for s in data["sweeps"] if s["area"] == area), key=lambda s: s["name"])
    runs = data.get("runs") or {data["run"]: data["env"]}
    lines = [
        f"# {HEAD.get(area, area)} 정확도",
        "",
        "`.venv/bin/python -m bench.accuracy promote`가 bench/accuracy/results/"
        f"{area}.json에서 만든다. 손으로 고치지 않는다.",
        "",
        "참값은 tests/" + area + "/ref의 Arb 참값이다. 오차는 그 파일의 모든 점에서 잰다.",
        "",
        "- ULP: |sd − 참값|을 참값에서의 float64 간격으로 나눈 값. 99.9%는 그 분위수다.",
        "- ULP/max(1, κ): 역함수의 계약 단위(κ는 입력에 대한 상대 조건수). 역함수만 있다.",
        "- mixed: |sd − 참값|/max(1, |참값|)를 2⁻⁵² 단위로 잰 값(45가 1e-14다). 0을 지나는 값(log 밀도, "
        "엔트로피, gammaln 등)의 계약 척도이고, 그 함수에만 있다. 최대와 99.9%다.",
        "- 상대오차: |sd − 참값|/|참값|.",
        "- 계약 대비: 그 영역의 계약 척도(보장 영역은 상대 1e-14, 아주 작은 값은 절대, mixed는 "
        "|오차|/max(1, |값|))로 잰 가장 큰 오차를 한계로 나눈 값. 1 이하가 계약 안이다.",
        "- 영역: guaranteed는 보장 영역, loose와 mixed는 그 밖(mixed는 그 척도로 잰다).",
        "",
        "측정: "
        + ", ".join(
            f"{k} ({v.get('date', '?')}, {v.get('commit', '?')}{' dirty' if v.get('dirty') else ''})"
            for k, v in sorted(runs.items())
        )
        + ".",  # fmt: skip
        "",
        "| 함수 | 영역 | 점 | 최대 ULP | 99.9% ULP | ULP/max(1, κ) | mixed | 최대 상대오차 | 계약 대비 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in sweeps:
        for label, r in s["regions"].items():
            cond = num(r.get("ulp_cond_max")) if "ulp_cond_max" in r else ""
            mixed = f"{num(r['mixed_max'])} / {num(r['mixed_p999'])}" if "mixed_max" in r else ""
            lines.append(
                f"| `{s['name']}` | {label} | {r['points']} | {num(r['ulp_max'])} | {num(r['ulp_p999'])} | {cond} | "
                f"{mixed} | {num(r['rel_max'])} | {contract_cell(r)} |"
            )
    return "\n".join(lines) + "\n"


def write(areas, data=None) -> list:
    out = []
    for area in areas:
        d = data or results.load_baseline(area)
        if d is None:
            continue
        path = results.ROOT / "bench" / area / "accuracy.md"
        path.write_text(render(area, d))
        out.append(path)
    return out
