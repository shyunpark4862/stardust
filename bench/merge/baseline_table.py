"""The benchmark baseline as one table (bench/merge/baseline.md), and later the
comparison of stardust against it (ROADMAP.md, safety net 3).

    .venv-legacy/bin/python bench/merge/baseline_table.py                 # baseline/ -> baseline.md
    .venv/bin/python bench/merge/baseline_table.py --compare DIR OUT.md   # DIR: same file names, measured on stardust

The three old scripts write different JSON; this reads them all as
{area: {column: {row: seconds}}}. Each row keeps its best over the rounds.
"""

import argparse
import json
import math
import pathlib

HERE = pathlib.Path(__file__).parent
BAND = 1.1  # within 10%: the noise band between two runs of the same build (the old projects' rule)


def load(d):
    d = pathlib.Path(d)
    out = {"array": {}, "linalg": {}, "frame": {}}
    for p in sorted(d.glob("*.json")):
        j = json.loads(p.read_text())
        area, _, col = p.stem.partition("-")
        if area == "array":  # final_report.py: {"times": {row: seconds}}, one file per round
            col = col.rsplit("-r", 1)[0]
            rows = {k: v for k, v in j["times"].items()}
        elif area == "linalg":  # stage_compare.py: {"rows": {row: us | {"us", "lib"} | None}}
            rows = {}
            for k, v in j["rows"].items():
                if v is not None:
                    rows[k] = (v["us"] if isinstance(v, dict) else v) * 1e-6
        else:  # final_compare.py: {"rows": {row: ns}}
            rows = {k: v * 1e-9 for k, v in j["rows"].items()}
        cur = out[area].setdefault(col, {})
        for k, v in rows.items():
            cur[k] = v if k not in cur else min(cur[k], v)
    return out


def fmt(s):
    if s >= 1e-3:
        return f"{s * 1e3:.2f}ms"
    if s >= 1e-6:
        return f"{s * 1e6:.1f}µs"
    return f"{s * 1e9:.0f}ns"


def label(k):
    return k.replace(" | ", " · ")  # the scripts' row keys use " | ", which splits a markdown cell


def gmean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


REFS = {"array": ["numpy"], "linalg": ["ref"], "frame": ["pandas", "polars-1", "polars"]}


def baseline_md(data, env):
    lines = ["# 벤치마크 기준선 (M0)", "", "옛 세 패키지를 마지막 커밋에서 빌드해 한자리에서 잰 값이다 (M7부터 동결한 옛 저장소 `../stardust-{numpy,linalg,pandas}`에 있다). 행마다 여러 라운드 중 가장 빠른 값.", "",
             "```", env.strip(), "```", ""]
    for area, cols in data.items():
        if not cols:
            continue
        names = [c for c in ["legacy", "legacy-1"] if c in cols] + [c for c in REFS[area] if c in cols]
        rows = list(cols["legacy"])
        lines += [f"## {area} ({len(rows)}줄)", ""]
        for ref in REFS[area]:
            if ref in cols:
                ks = [k for k in rows if k in cols[ref]]
                r = [cols[ref][k] / cols["legacy"][k] for k in ks]
                faster = sum(x > BAND for x in r)
                slower = sum(x < 1 / BAND for x in r)
                lines.append(f"- {ref} / legacy: 기하평균 {gmean(r):.2f}배, {len(ks)}줄 중 legacy가 빠름 {faster}, 비슷 {len(ks) - faster - slower}, 느림 {slower}")
        lines += ["", "| 행 | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
        for k in rows:
            lines.append(f"| {label(k)} | " + " | ".join(fmt(cols[c][k]) if k in cols[c] else "-" for c in names) + " |")
        lines.append("")
    return "\n".join(lines)


def compare_md(old, new):
    lines = ["# 통합판 대 기준선", "", f"new/legacy 시간 비율. 1보다 크면 통합판이 느리다. {BAND}배를 넘는 줄에 ✱.", ""]
    for area in old:
        for col in ["legacy", "legacy-1"]:
            if col not in old[area] or col not in new.get(area, {}):
                continue
            ks = [k for k in old[area][col] if k in new[area][col]]
            r = {k: new[area][col][k] / old[area][col][k] for k in ks}
            slow = [k for k in ks if r[k] > BAND]
            lines += [f"## {area}, {col}: 기하평균 {gmean(list(r.values())):.3f}, {BAND}배 넘게 느려진 줄 {len(slow)}/{len(ks)}", ""]
            lines += ["| 행 | legacy | new | 비율 |", "|---|---:|---:|---:|"]
            for k in ks:
                lines.append(f"| {label(k)} | {fmt(old[area][col][k])} | {fmt(new[area][col][k])} | {r[k]:.2f}{' ✱' if r[k] > BAND else ''} |")
            lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", nargs=2, metavar=("DIR", "OUT"))
    args = ap.parse_args()
    base = HERE / "baseline"
    if args.compare:
        pathlib.Path(args.compare[1]).write_text(compare_md(load(base), load(args.compare[0])) + "\n")
        return
    env = (base / "environment.txt").read_text()
    (HERE / "baseline.md").write_text(baseline_md(load(base), env) + "\n")
    print(HERE / "baseline.md")


if __name__ == "__main__":
    main()
