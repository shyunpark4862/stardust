"""O2: sd.optimize.root_scalar, minimize_scalar and the batched find_root, bracket_root,
find_minimum, bracket_minimum against scipy 1.18.1 (docs/design/optimize.md §5): time, the time
outside the user's function per call of it (the method's own cost), nfev; batch throughput at
10^3-10^6 elements against scipy.optimize.elementwise and against a loop of brentq per element.

Problems:
  - scalar roots: sin x - x/2 on (pi/2, pi) (APS 1, x0 = 2) and x^3 - 2x - 5 on (2, 3)
    (Chandrupatla 1, x0 = 2.5), written with math (the same Python function for both libraries);
  - scalar minima: e^x - 5x (Chandrupatla's minimization 3), bracket (0, 1, 3), bounds (0, 3);
  - batches: O0's problems, x^3 - 2x - c = 0 on (0, 3) (bracket_root from 0) and the minimum
    of (x - c)^2 + 0.01 c x^4 from (-1, 0.5, 11) (bracket_minimum from 0), c = linspace(1, 10, n)
    passed per element by args (O39); each library's function on its own arrays (sd for sd,
    numpy for scipy).

Time is the best of up to 5 runs (fewer when one run takes over 2 s), in ms. "own" is the time
outside f (summed with perf_counter around each call) divided by the calls of f (scalar) or the
elements' function values (batch, per element). Each thread count runs in its own process
(sd.set_num_threads; the kernels are serial, the sd arithmetic inside f is not).

Run: .venv/bin/python bench/optimize/o2/speed.py
"""

import json
import math
import os
import pathlib
import subprocess
import sys
import time
import warnings

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parents[2]


class Timed:
    """f wrapped with the time spent inside it and the number of calls (of the last run)."""

    def __init__(self, f):
        self.f, self.t, self.n = f, 0.0, 0

    def __call__(self, x, *args):
        t0 = time.perf_counter()
        v = self.f(x, *args)
        self.t += time.perf_counter() - t0
        self.n += 1
        return v


def best(solve, f, reps=5, budget=2.0):
    """(ms, own ms, calls, result) of the fastest run; own: outside f."""
    out, total = None, 0.0
    for _ in range(reps):
        tf = Timed(f)
        t0 = time.perf_counter()
        r = solve(tf)
        dt = time.perf_counter() - t0
        total += dt
        if out is None or dt * 1e3 < out[0]:
            out = (dt * 1e3, (dt - tf.t) * 1e3, tf.n, r)
        if total > budget:
            break
    return out


def scalar_cases():
    import scipy.optimize as so

    import stardust as sd

    aps1 = lambda x: math.sin(x) - x / 2
    aps1d = lambda x: math.cos(x) - 0.5
    aps1dd = lambda x: -math.sin(x)
    ch1 = lambda x: x**3 - 2 * x - 5
    ch1d = lambda x: 3 * x * x - 2
    ch1dd = lambda x: 6 * x
    mn = lambda x: math.exp(x) - 5 * x
    rows = []
    for name, f, d, dd, br, x0 in (
        ("APS 1", aps1, aps1d, aps1dd, (math.pi / 2, math.pi), 2.0),
        ("Chandrupatla 1", ch1, ch1d, ch1dd, (2.0, 3.0), 2.5),
    ):
        for m in ("brentq", "toms748", "bisect", "newton", "halley", "secant"):
            kw = {"method": m}
            if m in ("brentq", "toms748", "bisect"):
                kw["bracket"] = br
            else:
                kw["x0"] = x0
            if m in ("newton", "halley"):
                kw["fprime"] = d
            if m == "halley":
                kw["fprime2"] = dd
            a = best(lambda tf: sd.optimize.root_scalar(tf, **kw), f)
            b = best(lambda tf: so.root_scalar(tf, **kw), f)
            rows.append(
                {
                    "group": "root",
                    "problem": name,
                    "method": m,
                    "sd_ms": a[0],
                    "sp_ms": b[0],
                    "sd_own_us": a[1] * 1e3 / a[2],
                    "sp_own_us": b[1] * 1e3 / b[2],
                    "sd_calls": a[2],
                    "sp_calls": b[2],
                    "sd_iter": a[3].n_iter,
                    "sp_iter": b[3].iterations,
                    "same_x": bool(a[3].x == b[3].root),
                }
            )
    for m, kw in (
        ("brent", {"bracket": (0.0, 1.0, 3.0)}),
        ("golden", {"bracket": (0.0, 1.0, 3.0)}),
        ("bounded", {"bounds": (0.0, 3.0)}),
    ):
        a = best(lambda tf: sd.optimize.minimize_scalar(tf, method=m, **kw), mn)
        b = best(lambda tf: so.minimize_scalar(tf, method=m, **kw), mn)
        rows.append(
            {
                "group": "minimum",
                "problem": "e^x - 5x",
                "method": m,
                "sd_ms": a[0],
                "sp_ms": b[0],
                "sd_own_us": a[1] * 1e3 / a[2],
                "sp_own_us": b[1] * 1e3 / b[2],
                "sd_calls": a[2],
                "sp_calls": b[2],
                "sd_iter": a[3].n_iter,
                "sp_iter": b[3].nit,
                "same_x": bool(a[3].x == b[3].x),
            }
        )
    return rows


def batch_cases(sizes):
    import numpy as np
    from scipy.optimize import brentq
    from scipy.optimize import elementwise as ew

    import stardust as sd

    rows = []
    f = lambda x, c: x**3 - 2 * x - c
    g = lambda x, c: (x - c) ** 2 + 0.01 * c * x**4
    for n in sizes:
        c = np.linspace(1.0, 10.0, n)
        sc = sd.asarray(c)
        for name, sd_solve, sp_solve, fn_ in (
            (
                "find_root",
                lambda tf: sd.optimize.find_root(tf, (0.0, 3.0), args=(sc,)),
                lambda tf: ew.find_root(tf, (0.0, 3.0), args=(c,)),
                f,
            ),
            (
                "bracket_root",
                lambda tf: sd.optimize.bracket_root(tf, 0.0, args=(sc,)),
                lambda tf: ew.bracket_root(tf, 0.0, args=(c,)),
                f,
            ),
            (
                "find_minimum",
                lambda tf: sd.optimize.find_minimum(tf, (-1.0, 0.5, 11.0), args=(sc,)),
                lambda tf: ew.find_minimum(tf, (-1.0, 0.5, 11.0), args=(c,)),
                g,
            ),
            (
                "bracket_minimum",
                lambda tf: sd.optimize.bracket_minimum(tf, 0.0, args=(sc,)),
                lambda tf: ew.bracket_minimum(tf, 0.0, args=(c,)),
                g,
            ),
        ):
            a = best(sd_solve, fn_)
            b = best(sp_solve, fn_)
            nf_sd = int(sd.sum(a[3].nfev))
            nf_sp = int(np.sum(b[3].nfev))
            same = bool(np.array_equal(np.asarray(a[3].nfev.tolist()), np.asarray(b[3].nfev)))
            rows.append(
                {
                    "group": "batch",
                    "n": n,
                    "function": name,
                    "sd_ms": a[0],
                    "sp_ms": b[0],
                    "sd_own_ns": a[1] * 1e6 / nf_sd,
                    "sp_own_ns": b[1] * 1e6 / nf_sp,
                    "sd_calls": a[2],
                    "sp_calls": b[2],
                    "values_per_element": nf_sd / n,
                    "same_nfev": same,
                }
            )
        if n <= 10**4:  # a loop of root_scalar / brentq, one element at a time (total time only)
            cs = c.tolist()
            fr = lambda x, cc: x**3 - 2 * x - cc
            a = best(
                lambda tf: [sd.optimize.root_scalar(lambda x, cc=cc: tf(x, cc), bracket=(0.0, 3.0)).x for cc in cs],
                fr,
            )
            b = best(lambda tf: [brentq(lambda x, cc=cc: tf(x, cc), 0.0, 3.0) for cc in cs], fr)
            rows.append({"group": "loop", "n": n, "sd_ms": a[0], "sp_ms": b[0], "calls": a[2]})
    return rows


def worker(threads):
    sys.path.insert(0, str(ROOT))
    warnings.simplefilter("ignore")
    import stardust as sd

    sd.set_num_threads(threads)
    rows = (scalar_cases() if threads == 1 else []) + batch_cases([10**3, 10**4, 10**5, 10**6])
    print(json.dumps(rows))


def main():
    import numpy
    import scipy

    results = {}
    for t in (1, 10):
        env = dict(os.environ, VECLIB_MAXIMUM_THREADS=str(t), OMP_NUM_THREADS=str(t), OPENBLAS_NUM_THREADS=str(t))
        out = subprocess.run(
            [sys.executable, __file__, "--worker", str(t)], env=env, capture_output=True, text=True, check=True
        )
        results[t] = json.loads(out.stdout.strip().splitlines()[-1])
    (HERE / "speed.json").write_text(
        json.dumps({"scipy": scipy.__version__, "numpy": numpy.__version__, "rows": results}, indent=1) + "\n"
    )
    lines = [
        f"scipy {scipy.__version__}, numpy {numpy.__version__}",
        "",
        "| problem | method | sd ms | scipy ms | sd own us/call | scipy own us/call | calls sd/scipy | iter sd/scipy | same x |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for a in results[1]:
        if a["group"] in ("root", "minimum"):
            lines.append(
                f"| {a['problem']} | {a['method']} | {a['sd_ms']:.3g} | {a['sp_ms']:.3g} | {a['sd_own_us']:.3g} | "
                f"{a['sp_own_us']:.3g} | {a['sd_calls']}/{a['sp_calls']} | {a['sd_iter']}/{a['sp_iter']} | {a['same_x']} |"
            )
    lines += [
        "",
        "| n | function | sd ms (1/10 thr) | scipy ms (1/10 thr) | sd own ns/value | scipy own ns/value | "
        "calls sd/scipy | values per element | same nfev |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    b10 = [r for r in results[10] if r["group"] == "batch"]
    for a, b in zip([r for r in results[1] if r["group"] == "batch"], b10):
        lines.append(
            f"| {a['n']} | {a['function']} | {a['sd_ms']:.3g} / {b['sd_ms']:.3g} | {a['sp_ms']:.3g} / {b['sp_ms']:.3g} | "
            f"{a['sd_own_ns']:.3g} | {a['sp_own_ns']:.3g} | {a['sd_calls']}/{a['sp_calls']} | "
            f"{a['values_per_element']:.3g} | {a['same_nfev']} |"
        )
    lines += ["", "| n | loop of sd root_scalar ms | loop of scipy brentq ms | calls |", "|---:|---:|---:|---:|"]
    for a in results[1]:
        if a["group"] == "loop":
            lines.append(f"| {a['n']} | {a['sd_ms']:.3g} | {a['sp_ms']:.3g} | {a['calls']} |")
    text = "\n".join(lines) + "\n"
    (HERE / "speed.log").write_text(text)
    print(text)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        worker(int(sys.argv[2]))
    else:
        main()
