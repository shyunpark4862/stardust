"""M3: the per-call cost of the array/linalg boundary, before and after.

    .venv/bin/python bench/merge/m3_boundary.py OUT.json [--label L]
    .venv/bin/python bench/merge/m3_boundary.py --table BEFORE.json AFTER.json

Small inputs, where a call is mostly fixed cost: the boundary (buffer-protocol
input, a result block wrapped by sd.asarray, the thread count read from
Python) was measured at about 200 ns per result before the merge
(docs/design/linalg.md §3.6). Each row: best of 7 rounds of a timed loop.
"""

import json
import pathlib
import sys
import timeit

import numpy as np

import stardust as sd

L = sd.linalg


def rows():
    rng = np.random.default_rng(0)

    def arr(shape, spd=False):
        a = rng.standard_normal(shape)
        if len(shape) >= 2 and shape[-1] == shape[-2]:
            n = shape[-1]
            a = a @ np.swapaxes(a, -1, -2) + n * np.eye(n) if spd else a + n * np.eye(n)
        return sd.asarray(a)

    a2, a3, s3, v3 = arr((2, 2)), arr((3, 3)), arr((3, 3), spd=True), arr((3,))
    b3 = arr((3,))
    batch = arr((1000, 3, 3))
    bb = arr((1000, 3, 1))
    return [
        ("det 2x2", lambda: L.det(a2)),
        ("slogdet 2x2", lambda: L.slogdet(a2)),
        ("inv 2x2", lambda: L.inv(a2)),
        ("solve 3x3", lambda: L.solve(a3, b3)),
        ("cholesky 3x3", lambda: L.cholesky(s3)),
        ("qr 3x3", lambda: L.qr(a3)),
        ("eigh 3x3", lambda: L.eigh(s3)),
        ("svd 3x3", lambda: L.svd(a3)),
        ("vector_norm 3", lambda: L.vector_norm(v3)),
        ("matrix_norm 3x3", lambda: L.matrix_norm(a3)),
        ("trace 3x3", lambda: L.trace(a3)),
        ("solve (1000,3,3)", lambda: L.solve(batch, bb)),
        ("inv (1000,3,3)", lambda: L.inv(batch)),
    ]


def best_ns(fn):
    n = 1
    while timeit.timeit(fn, number=n) < 0.02:
        n *= 2
    return min(timeit.repeat(fn, number=n, repeat=7)) / n * 1e9


def main():
    if sys.argv[1] == "--table":
        a, b = (json.loads(pathlib.Path(p).read_text()) for p in sys.argv[2:4])
        print(f"| 행 | {a['label']} | {b['label']} | 차이 | 비율 |\n|---|---:|---:|---:|---:|")
        for k in a["rows"]:
            x, y = a["rows"][k], b["rows"][k]
            print(f"| {k} | {x:.0f}ns | {y:.0f}ns | {y - x:+.0f}ns | {y / x:.2f} |")
        return
    out = pathlib.Path(sys.argv[1])
    label = sys.argv[sys.argv.index("--label") + 1] if "--label" in sys.argv else out.stem
    res = {name: best_ns(fn) for name, fn in rows()}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"label": label, "threads": sd.get_num_threads(), "rows": res}, indent=1) + "\n")
    for k, v in res.items():
        print(f"{k:20} {v:10.0f} ns")


if __name__ == "__main__":
    main()
