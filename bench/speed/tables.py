"""bench/<area>/speed.md from an area's baseline (or any results file).

One section per group, in the registry's order. A group whose comparators all run at
their libraries' default threads gets one row per case (sd at each thread count in its
own column); a group with a comparator whose BLAS follows sd's thread count gets one
row per case and thread count. A case's note gets the comparators' `differs` reasons
(from the run) and a line for each comparator that gives NaN more often than sd
(check.nan_note).
"""

from __future__ import annotations

import math
import pathlib
import statistics

from . import results
from .check import nan_note
from .harness import fmt_ratio, fmt_time
from .registry import Case

THREADS_WORD = "스레드"


def _ratio_cell(r):
    if r is None:
        return "-"
    s = fmt_ratio(r)
    return f"**{s}**" if r < 1 else s


def _scaled(t: dict | None, n_case) -> tuple[float | None, bool]:
    """A comparator's time at the case's element count (it may time a sample)."""
    if t is None or t.get("min") is None:
        return None, False
    if t.get("n") and n_case and t["n"] != n_case:
        return t["min"] * n_case / t["n"], True
    return t["min"], False


def _cell(t, n_case, foot):
    v, scaled = _scaled(t, n_case)
    if v is None:
        return "실패" if t is not None and t.get("error") and t["error"] != "not measured" else "-"
    if scaled:
        foot.add("scaled")
        return fmt_time(v) + "\\*"
    return fmt_time(v)


def _name_cell(c, foot) -> str:
    if c.get("busy"):
        foot.add("busy")
        return f"`{c['name']}` †"
    return f"`{c['name']}`"


def case_note(c: dict) -> str:
    """The note column: the case's note and the NaN notes of its comparators."""
    sd_nan = [t.get("nan") for t in c["timings"] if t["label"] == "sd" and t.get("nan") is not None]
    sd = max(sd_nan) if sd_nan else None
    parts = [c.get("note") or ""]
    seen = set()
    for t in c["timings"]:
        if t["label"] == "sd" or t["label"] in seen:
            continue
        note = nan_note(t["label"], t.get("nan"), sd)
        if note:
            seen.add(t["label"])
            parts.append(note)
    return "; ".join(p for p in parts if p)


def _noise(ts) -> str:
    vals = [t["noise"] for t in ts if t is not None and t.get("noise") is not None]
    return f"{max(vals) * 100:.0f}%" if vals else "-"


def _ref_labels(cases: list[dict]):
    """(fixed labels, follow labels, sd thread counts), in order of appearance."""
    fixed, follow, ths = [], [], []
    for c in cases:
        for t in c["timings"]:
            if t["label"] == "sd":
                if t["threads"] not in ths:
                    ths.append(t["threads"])
            elif t["threads"] is None:
                if t["label"] not in fixed:
                    fixed.append(t["label"])
            elif t["label"] not in follow:
                follow.append(t["label"])
    return fixed, follow, sorted(ths)


def _summary(cases, fixed, follow, ths) -> list[str]:
    lines = []
    for label in follow + fixed:
        parts = []
        for th in ths:
            rs = []
            for c in cases:
                sd = results.timing(c, "sd", th)
                ref = results.timing(c, label, th if label in follow else None)
                v, _ = _scaled(ref, c.get("n"))
                if sd and sd.get("min") and v:
                    rs.append(v / sd["min"])
            if rs:
                slow = sum(r < 1 for r in rs)
                gm = math.exp(sum(map(math.log, rs)) / len(rs))
                parts.append(
                    f"{th}{THREADS_WORD} 중앙값 {fmt_ratio(statistics.median(rs))}배"
                    f"(기하평균 {fmt_ratio(gm)}배, sd가 느린 줄 {slow}/{len(rs)})"
                )
        if parts:
            lines.append(f"- {label} ÷ sd: " + ", ".join(parts))
    return lines


def group_table(cases: list[dict]) -> list[str]:
    fixed, follow, ths = _ref_labels(cases)
    note_of = {c["id"]: case_note(c) for c in cases}
    notes = any(note_of.values())
    sizes = any(c.get("size") for c in cases)
    foot: set[str] = set()
    out = _summary(cases, fixed, follow, ths)
    if out:
        out.append("")
    if not follow:
        head = ["사례"] + (["크기"] if sizes else []) + [f"sd {t}{THREADS_WORD}" for t in ths]
        for label in fixed:
            head += [label] + [f"÷ sd {t}" for t in ths]
        head += ["잡음"] + (["비고"] if notes else [])
        out += [
            "| " + " | ".join(head) + " |",
            "|" + "|".join(["---"] * (1 + sizes) + ["---:"] * (len(head) - 1 - sizes - notes) + ["---"] * notes) + "|",
        ]
        for c in cases:
            sds = [results.timing(c, "sd", t) for t in ths]
            row = (
                [_name_cell(c, foot)]
                + ([c.get("size", "")] if sizes else [])
                + [_cell(t, c.get("n"), foot) for t in sds]
            )
            used = list(sds)
            for label in fixed:
                ref = results.timing(c, label, None)
                used.append(ref)
                row.append(_cell(ref, c.get("n"), foot))
                v, _ = _scaled(ref, c.get("n"))
                for sd in sds:
                    row.append(_ratio_cell(v / sd["min"] if v and sd and sd.get("min") else None))
            row.append(_noise(used))
            if notes:
                row.append(note_of[c["id"]])
            out.append("| " + " | ".join(row) + " |")
    else:
        head = ["사례"] + (["크기"] if sizes else []) + [THREADS_WORD, "sd"]
        for label in follow + fixed:
            head += [label, "÷ sd"]
        head += ["잡음"] + (["비고"] if notes else [])
        out += [
            "| " + " | ".join(head) + " |",
            "|" + "|".join(["---"] * (1 + sizes) + ["---:"] * (len(head) - 1 - sizes - notes) + ["---"] * notes) + "|",
        ]
        for c in cases:
            for th in ths:
                sd = results.timing(c, "sd", th)
                if sd is None:
                    continue
                row = (
                    [_name_cell(c, foot)]
                    + ([c.get("size", "")] if sizes else [])
                    + [str(th), _cell(sd, c.get("n"), foot)]
                )
                used = [sd]
                for label in follow + fixed:
                    ref = results.timing(c, label, th if label in follow else None)
                    used.append(ref)
                    v, _ = _scaled(ref, c.get("n"))
                    row += [_cell(ref, c.get("n"), foot), _ratio_cell(v / sd["min"] if v and sd.get("min") else None)]
                row.append(_noise(used))
                if notes:
                    row.append(note_of[c["id"]])
                out.append("| " + " | ".join(row) + " |")
        if fixed:
            out += ["", f"{', '.join(fixed)}은(는) 스레드 수와 상관없이 한 번 잰 값이다(그 라이브러리의 기본 스레드)."]
    if "busy" in foot:
        out += ["", "† 잴 때 다른 프로세스가 CPU를 1코어 넘게 썼다(사례의 `other_cpu`). 다시 재는 것이 좋다."]
    if "scaled" in foot:
        out += ["", "\\* 비교 대상이 느려 입력의 일부로 재고, 원소 수에 비례해 사례의 크기로 환산한 값이다."]
    return out


def _runs_of(c: dict, default) -> set:
    return {t.get("run") or c.get("run") or default for t in c["timings"]}


def _env_lines(data: dict, cases: list[dict]) -> list[str]:
    runs = data.get("runs") or {data["run"]: data["env"]}
    default = data.get("run")
    used = set().union(*(_runs_of(c, default) for c in cases)) if cases else set()
    lines = ["| 실행 | 커밋 | 기계 | 사례 수 |", "|---|---|---|---:|"]
    for run_id, env in runs.items():
        if run_id not in used:
            continue
        git = env.get("git", {})
        commit = (git.get("commit") or "?") + (" + 작업 중 변경" if git.get("dirty") else "")
        m = env.get("machine", {})
        machine = f"{m.get('chip', '?')}, {m.get('cpus', '?')}코어, macOS {m.get('macos') or '?'}"
        n = sum(1 for c in cases if run_id in _runs_of(c, default))
        lines.append(f"| {run_id} | {commit} | {machine} | {n} |")
    versions = {}
    for run_id, env in runs.items():
        if run_id in used:
            for k, v in (env.get("versions") or {}).items():
                versions.setdefault(k, set()).add(str(v))
            for k, v in (env.get("r_packages") or {}).items():
                versions.setdefault(f"R {k}", set()).add(str(v))
    if versions:
        lines += ["", "라이브러리: " + ", ".join(f"{k} {'/'.join(sorted(v))}" for k, v in versions.items()) + "."]
    blas = set()
    for run_id, env in runs.items():
        if run_id in used:
            for lib in ("numpy", "scipy"):
                name = ((env.get("blas") or {}).get(lib) or {}).get("blas", {}).get("name")
                if name:
                    blas.add(f"{lib} {name}")
            r = (env.get("r_blas") or {}).get("blas")
            if r:
                blas.add(f"R {pathlib.Path(r).name}")
    if blas:
        lines += ["", "BLAS: " + ", ".join(sorted(blas)) + "."]
    return lines


# How each comparator library is timed, for the tables' "비교 대상을 잰 방법" section
HOW = {
    "R": (
        "R: 실행 하나에 R 세션(`R --vanilla`) 하나를 띄우고, 사례의 입력을 float64 파일로 넘긴 뒤 R 안에서 "
        "같은 규칙(예열, 배치 크기를 두 배씩 늘려 20 ms 이상, 배치 중 가장 빠른 값)으로 잰다. R의 벡터 함수를 "
        "한 번 부르는 식이다. R은 1스레드로 돈다(`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`; R이 쓰는 BLAS는 "
        "위 'BLAS' 줄). R의 쓰레기 수집은 배치 안에서 끌 수 없어 배치 전에 한 번 돌린다. 사례의 비고에 R의 호출이 "
        "다른 점을 적는다."
    ),
    "Boost": (
        "Boost.Math: C++ 반복문(`bench/speed/boost_speed.cpp`)이 배열의 원소마다 Boost 함수를 직접 부른다. "
        "함수 이름은 호출마다 한 번 풀고, 매개변수가 스칼라인 분포는 호출마다 한 번 만든다. sd의 최적화 "
        "옵션(-O3 -mcpu=native -ffp-contract=off)과 -std=c++17로 빌드하고, Python에서는 배열 하나에 ctypes 호출 "
        "한 번이다. "
        "오류 정책은 정확도 측정의 비교 대상과 같다(오류 대신 NaN·inf)."
    ),
    "python": (
        "numpy, scipy, pandas, statsmodels, scikit-learn: 같은 프로세스에서 Python으로 부르고 sd와 같은 규칙으로 "
        "잰다. 입력은 그 라이브러리의 배열(numpy)로 미리 만들어 둔다."
    ),
}


def _how_lines(cases: list[dict]) -> list[str]:
    libs = {t.get("lib") for c in cases for t in c["timings"] if t["label"] != "sd"}
    lines = ["- " + HOW["python"]] if libs - {"R", "Boost"} else []
    lines += ["- " + HOW[k] for k in ("R", "Boost") if k in libs]
    return ["", "## 비교 대상을 잰 방법", "", *lines] if lines else []


def render(area: str, data: dict, registry_cases: list[Case], source: str) -> str:
    by_id = {c["id"]: c for c in data["cases"] if c["area"] == area}
    order = [c for c in registry_cases if c.area == area]
    groups: dict[str, list[dict]] = {}  # section title -> records
    missing = 0
    for c in order:
        rec = by_id.get(c.id)
        if rec is None or not any(t.get("min") is not None for t in rec["timings"]):
            missing += 1
            continue
        groups.setdefault(c.title or c.group, []).append(rec)
    shown = [c for g in groups.values() for c in g]
    lines = [
        f"# {area} 속도",
        "",
        f"이 파일은 `{source}`에서 만든 것이다. 손으로 고치지 않는다. 다시 만들기: "
        f"`.venv/bin/python -m bench.speed table {area}` (재는 법은 bench/README.md).",
        "",
        "- 시간: 한 번 부를 때의 시간. 한 배치가 20 ms 이상이 되도록 여러 번 부르고, 모든 라운드의 배치 중 가장 빠른 값이다.",
        "- 배율(÷ sd): 비교 대상 시간 ÷ sd 시간. 1보다 크면 sd가 빠르다. **굵은 배율**은 sd가 느린 줄이다.",
        "- 잡음: 그 줄의 측정마다 (배치 시간의 중앙값 ÷ 최솟값 − 1)을 구한 것 중 가장 큰 값.",
        "- 스레드: sd의 스레드 수. 한 줄에 sd 스레드별 열이 있는 표에서 비교 대상은 그 라이브러리의 기본 스레드로 한 번 잰다"
        "(numpy·scipy의 원소별 계산은 1스레드). BLAS·LAPACK이 일의 대부분인 비교 대상은 BLAS 스레드를 sd와 같게 묶어 "
        "'스레드' 열이 있는 표로 보인다. R은 늘 1스레드다.",
        "- 비고: 비교 대상이 sd보다 NaN을 1%p 넘게 더 내면 그 비율을 적는다(그만큼 일찍 끝나는 호출이 섞인다).",
        "",
        "## 측정 환경",
        "",
        *_env_lines(data, shown),
        *_how_lines(shown),
    ]
    for group, cases in groups.items():
        lines += ["", f"## {group}", "", *group_table(cases)]
    if missing:
        lines += ["", f"등록부의 사례 중 {missing}개는 아직 결과가 없다."]
    fails = [
        (c["id"], t["label"], t.get("threads"), t.get("error") or c.get("failure"))
        for c in shown
        for t in c["timings"]
        if t.get("min") is None and t.get("error") != "not measured"
    ]
    if fails:
        lines += ["", "## 측정 실패", "", "| 사례 | 대상 | 스레드 | 오류 |", "|---|---|---:|---|"]
        for cid, label, th, err in fails:
            err = (err or "").replace("|", "\\|").replace("\n", " ")
            lines.append(f"| `{cid}` | {label} | {th if th is not None else '기본'} | {err} |")
    return "\n".join(lines) + "\n"


def table_path(area: str) -> pathlib.Path:
    return results.ROOT / "bench" / area / "speed.md"


def write(area: str, data: dict, registry_cases: list[Case], source: str, path=None) -> pathlib.Path:
    path = pathlib.Path(path) if path else table_path(area)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(area, data, registry_cases, source))
    return path
