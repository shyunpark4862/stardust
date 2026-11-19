"""Old packages and stardust measured side by side in one sitting (the phase
reports): legacy and new runs alternate, so the machine's drift hits both.

    .venv/bin/python bench/merge/sameseat_table.py DIR > DIR/table.md

DIR holds array-{legacy,new,legacy1,new1}-r*.json (bench/array/final_report.py),
linalg-{legacy,new,legacy1,new1,ref}.json (bench/linalg/stage_compare.py) and
frame-{legacy,new,legacy1,new1,pandas}.json (bench/frame/final_compare.py), as
bench/merge/sameseat.sh writes them; areas whose files are missing are skipped.
Each row keeps its best over the rounds.
"""

import glob
import json
import math
import pathlib
import sys

D = pathlib.Path(sys.argv[1])
BAND = 1.1


def best(pattern, key):
    out = {}
    for p in sorted(glob.glob(str(D / pattern))):
        for k, v in json.loads(pathlib.Path(p).read_text())[key].items():
            if v is None:
                continue
            v = v["us"] if isinstance(v, dict) else v
            out[k] = min(out.get(k, math.inf), v)
    return out


def gmean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def fmt_us(us):
    return f"{us / 1000:.2f}ms" if us >= 1000 else f"{us:.1f}µs" if us >= 1 else f"{us * 1000:.0f}ns"


def section(title, old, new, unit_s, scale=1.0):
    if not old or not new:
        return
    keys = [k for k in old if k in new]
    r = {k: new[k] / old[k] for k in keys}
    slow = [k for k in keys if r[k] > BAND]
    fast = [k for k in keys if r[k] < 1 / BAND]
    print(f"## {title}\n")
    print(f"- {len(keys)}줄, new/legacy 기하평균 **{gmean(list(r.values())):.3f}** (1보다 작으면 통합판이 빠르다)")
    print(f"- 10% 넘게 느려진 줄 {len(slow)}, 10% 넘게 빨라진 줄 {len(fast)}\n")
    if slow or fast:
        print("| 행 | legacy | new | 비율 |\n|---|---:|---:|---:|")
        for k in slow + fast:
            o, n = (old[k] * 1e6, new[k] * 1e6) if unit_s else (old[k] * scale, new[k] * scale)
            print(f"| {k.replace(' | ', ' · ')} | {fmt_us(o)} | {fmt_us(n)} | {r[k]:.2f} |")
        print()


def ref_section(title, ref, new):
    keys = [k for k in ref if k in new]
    if not keys:
        return
    r = [ref[k] / new[k] for k in keys]
    print(f"## {title}\n")
    print(
        f"- {len(keys)}줄: 통합판이 10% 넘게 빠름 {sum(x > BAND for x in r)}, 비슷 {sum(1 / BAND <= x <= BAND for x in r)}, "
        f"느림 {sum(x < 1 / BAND for x in r)}. ref/new 기하평균 {gmean(r):.2f}\n"
    )


section("배열, 기본 스레드", best("array-legacy-r*.json", "times"), best("array-new-r*.json", "times"), True)
section("배열, 1스레드", best("array-legacy1-r*.json", "times"), best("array-new1-r*.json", "times"), True)
section("linalg, 기본 스레드", best("linalg-legacy.json", "rows"), best("linalg-new.json", "rows"), False)
section("linalg, 1스레드", best("linalg-legacy1.json", "rows"), best("linalg-new1.json", "rows"), False)
ref_section("linalg 대 numpy/scipy (기본 스레드)", best("linalg-ref.json", "rows"), best("linalg-new.json", "rows"))
section("frame, 기본 스레드", best("frame-legacy.json", "rows"), best("frame-new.json", "rows"), False, 1e-3)
section("frame, 1스레드", best("frame-legacy1.json", "rows"), best("frame-new1.json", "rows"), False, 1e-3)
ref_section("frame 대 pandas (기본 스레드)", best("frame-pandas.json", "rows"), best("frame-new.json", "rows"))
