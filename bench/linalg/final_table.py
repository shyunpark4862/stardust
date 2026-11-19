"""M11: tables from bench/results/final/*.json (bench/stage_compare.sh).

Writes bench/results/final/table.md:
  1. per stage: rows faster / equal (+-10%) / slower than numpy or scipy
  2. per stage and kind of input: geometric mean of ref/sdla
  3. representative rows, every stage side by side
  4. every row
Usage: .venv/bin/python bench/final_table.py
"""

import json
import math
import pathlib

DIR = pathlib.Path(__file__).parent / "results" / "final"
STAGES = ["M7", "M8", "M9", "M10", "M10-1thread"]
KINDS = {
    "single small": ("(3,3)", "(32,32)"),
    "single large": ("(300,300)", "(1000,1000)"),
    "batch": ("(1000,3,3)", "(100,32,32)"),
    "long": ("(1000000,)", "(1000000,3)", "n=100000"),
}
PICK = [
    "solve | (3,3)",
    "solve | (1000,1000)",
    "inv | (32,32)",
    "vecdot | (1000000,)",
    "vector_norm | (1000000,)",
    "diagonal | (1000,1000)",
    "matrix_norm 1 | (1000,3,3)",
    "matrix_power 3 | (100,32,32)",
    "solve_triangular | (1000,1000)",
    "lu_solve | (1000,1000)",
    "qr | (1000,3,3)",
    "eigh | (1000,3,3)",
    "svd | (100,32,32)",
    "eig | (100,32,32)",
    "solve_sylvester | (1000,1000)",
    "solve_banded 2,2 | (1000,3,3)",
    "expm | (1000,3,3)",
    "expm | (300,300)",
]


def fmt(us):
    if us is None:
        return "-"
    if us >= 1000:
        return f"{us / 1000:.2f}ms"
    return f"{us:.1f}µs" if us >= 10 else f"{us:.2f}µs"


def kind(key):
    shape = key.split(" | ")[1]
    return next((k for k, v in KINDS.items() if shape in v), "other")


def main():
    data = {s: json.loads((DIR / f"{s}.json").read_text()) for s in STAGES}
    ref = json.loads((DIR / "ref.json").read_text())["rows"]
    keys = list(data["M10"]["rows"])
    out = [
        f"stages measured in {data['M10']['rounds']} alternating rounds, best per row; "
        f"threads: " + ", ".join(f"{s} {data[s]['threads']}" for s in STAGES),
        "",
    ]

    out += [
        "## 1. Against numpy / scipy (ref/sdla: > 1.1 faster, < 0.9 slower)",
        "",
        "| stage | rows | faster | equal | slower |",
        "|---|---:|---:|---:|---:|",
    ]
    for s in STAGES:
        c = [0, 0, 0]
        n = 0
        for k in keys:
            t, r = data[s]["rows"].get(k), ref.get(k)
            if t is None or r is None:
                continue
            q = r["us"] / t
            c[0 if q > 1.1 else 2 if q < 0.9 else 1] += 1
            n += 1
        out.append(f"| {s} | {n} | {c[0]} | {c[1]} | {c[2]} |")

    out += [
        "",
        "## 2. Geometric mean of ref/sdla by kind of input (rows every stage has)",
        "",
        "| kind | rows | " + " | ".join(STAGES) + " |",
        "|---|---:|" + "---:|" * len(STAGES),
    ]
    common = [k for k in keys if all(data[s]["rows"].get(k) is not None for s in STAGES) and ref.get(k)]
    for kd in list(KINDS) + ["all"]:
        ks = [k for k in common if kd == "all" or kind(k) == kd]
        if not ks:
            continue
        g = [math.exp(sum(math.log(ref[k]["us"] / data[s]["rows"][k]) for k in ks) / len(ks)) for s in STAGES]
        out.append(f"| {kd} | {len(ks)} | " + " | ".join(f"{v:.2f}" for v in g) + " |")

    out += [
        "",
        "## 3. Representative rows",
        "",
        "| row | " + " | ".join(STAGES) + " | ref |",
        "|---|" + "---:|" * (len(STAGES) + 1),
    ]
    for k in PICK:
        r = ref.get(k)
        out.append(
            f"| {k.replace(' | ', ' ')} | "
            + " | ".join(fmt(data[s]["rows"].get(k)) for s in STAGES)
            + f" | {fmt(r['us']) if r else '-'} {r['lib'] if r else ''} |"
        )

    out += [
        "",
        "## 4. Every row",
        "",
        "| row | " + " | ".join(STAGES) + " | ref | ref/M10 |",
        "|---|" + "---:|" * (len(STAGES) + 2),
    ]
    for k in keys:
        r = ref.get(k)
        t = data["M10"]["rows"].get(k)
        q = f"{r['us'] / t:.2f}" if r and t else "-"
        out.append(
            f"| {k.replace(' | ', ' ')} | "
            + " | ".join(fmt(data[s]["rows"].get(k)) for s in STAGES)
            + f" | {fmt(r['us']) if r else '-'} | {q} |"
        )
    (DIR / "table.md").write_text("\n".join(out) + "\n")
    print("\n".join(out[: out.index("## 3. Representative rows") + 3 + len(PICK)]))


if __name__ == "__main__":
    main()
