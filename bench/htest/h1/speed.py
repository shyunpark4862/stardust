"""H1: the speed of sd.htest's tests of means and variances against scipy 1.18.1 (docs/design/htest.md
§5). Results: bench/htest/h1/speed.md (the raw output in speed.json). A standalone script until the
area joins bench/speed (the cases to move there are listed in docs/design/htest.md §7).

Sizes: each sample (one, two or three) of n = 10, 10^3 and 10^6 normal values (the second shifted
by 0.1, the third by -0.05 with spread 1.2), and a batch of 10^4 lanes of 30 (axis=-1). scipy and
sd get the same numpy arrays (sd converts them: the conversion is part of its time).

Time is the best of up to 5 runs (fewer when the runs pass 2 s), in ms. scipy with numpy's and
Accelerate's threads at 1 (its loops are one thread); sd on 1 and on 10 threads
(sd.set_num_threads). Warnings are silenced (the comparison is the time).

Run: .venv/bin/python bench/htest/h1/speed.py
"""

import json
import os
import pathlib
import platform
import sys
import time
import warnings

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

HERE = pathlib.Path(__file__).parent
SIZES = (10, 1000, 10**6)
BATCH = (10**4, 30)


def best(fn, reps=5, budget=2.0):
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
        if sum(out) > budget:
            break
    return min(out) * 1e3


def cases():
    """(test, k samples, sd call, scipy call, options)"""
    import scipy.stats as st

    import stardust as sd

    h = sd.htest
    return [
        (
            "ttest_1samp",
            1,
            lambda *a, **k: h.ttest_1samp(a[0], 0.0, **k),
            lambda *a, **k: st.ttest_1samp(a[0], 0.0, **k),
            {},
        ),
        ("ttest_rel", 2, h.ttest_rel, st.ttest_rel, {}),
        ("ttest_ind", 2, h.ttest_ind, st.ttest_ind, {}),
        ("ttest_ind (Welch)", 2, h.ttest_ind, st.ttest_ind, {"equal_var": False}),
        ("ttest_ind (trim=0.2)", 2, h.ttest_ind, st.ttest_ind, {"trim": 0.2}),
        ("f_oneway (3)", 3, h.f_oneway, st.f_oneway, {}),
        ("f_oneway (3, Welch)", 3, h.f_oneway, st.f_oneway, {"equal_var": False}),
        ("alexandergovern (3)", 3, h.alexandergovern, st.alexandergovern, {}),
        ("bartlett (3)", 3, h.bartlett, st.bartlett, {}),
        ("levene (3)", 3, h.levene, st.levene, {}),
        ("fligner (3)", 3, h.fligner, st.fligner, {}),
    ]


def main():
    import numpy as np
    import scipy

    import stardust as sd

    warnings.simplefilter("ignore")
    g = np.random.Generator(np.random.PCG64(20261121))
    locs = ((0.0, 1.0), (0.1, 1.0), (-0.05, 1.2))
    rows = []
    for name, k, sdf, spf, kw in cases():
        row = {"test": name}
        for n in SIZES + ("batch",):
            shape = BATCH if n == "batch" else n
            data = [g.normal(m, s, size=shape) for m, s in locs[:k]]
            extra = {"axis": -1} if n == "batch" else {}
            key = "batch" if n == "batch" else str(n)
            row[f"scipy {key}"] = best(lambda: spf(*data, **kw, **extra))
            for th in (1, 10):
                sd.set_num_threads(th)
                row[f"sd{th} {key}"] = best(lambda: sdf(*data, **kw, **extra))
        sd.set_num_threads(10)
        rows.append(row)
        print(name, {k: round(v, 4) for k, v in row.items() if k != "test"}, flush=True)
    meta = {
        "scipy": scipy.__version__,
        "numpy": np.__version__,
        "python": sys.version.split()[0],
        "machine": platform.machine(),
        "date": time.strftime("%Y-%m-%d"),
    }
    (HERE / "speed.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=1) + "\n")
    keys = [str(n) for n in SIZES] + ["batch"]
    head = ["test"] + [f"{lib} {k}" for k in keys for lib in ("scipy", "sd1", "sd10")]
    lines = [
        "# sd.htest H1: speed against scipy",
        "",
        f"scipy {meta['scipy']}, numpy {meta['numpy']}, Python {meta['python']}, {meta['machine']}, "
        f"{meta['date']}. ms, the best of up to 5 runs. scipy on one thread; sd on 1 (sd1) and 10 (sd10) threads. "
        "n: each sample's size (two or three samples where the test takes them); batch: 10^4 lanes of 30 "
        "(axis=-1). Made by bench/htest/h1/speed.py.",
        "",
        "| " + " | ".join(head) + " |",
        "|" + "---|" * len(head),
    ]
    for r in rows:
        lines.append("| " + " | ".join([r["test"]] + [f"{r[h]:.3g}" for h in head[1:]]) + " |")
    lines += ["", "scipy / sd (10 threads), the speedup:", "", "| test | " + " | ".join(keys) + " |"]
    lines.append("|" + "---|" * (len(keys) + 1))
    for r in rows:
        lines.append(
            "| " + r["test"] + " | " + " | ".join(f"{r[f'scipy {k}'] / r[f'sd10 {k}']:.1f}" for k in keys) + " |"
        )
    (HERE / "speed.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
