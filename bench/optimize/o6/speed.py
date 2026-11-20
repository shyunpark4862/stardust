"""O6: the sweep of sd.optimize against scipy 1.18.1 (docs/design/optimize.md §5, O97): every public
function and method at the sizes of the O1-O5 benches, run by their own case code (bench/optimize/o1-o5
/speed.py: the same problems, the same timing rules), plus what they leave out: check_grad, the '3-point'
Jacobian and bounds of least_squares, bounds of L-BFGS-B and Nelder-Mead.

Each row: the time (best of up to 5 runs, fewer when one run takes over 2 s) of sd and scipy, in ms, and
the time outside the user's callbacks per unit (the method's own cost): per iteration (ms) for the
iterative methods, per call of f (us) for the scalar ones, per function value (ns) for the batches.
Each thread count runs in its own process (VECLIB_MAXIMUM_THREADS for Accelerate, sd.set_num_threads for
sd).

Run: .venv/bin/python bench/optimize/o6/speed.py LABEL     # writes speed_LABEL.json
     .venv/bin/python bench/optimize/o6/speed.py --table BEFORE AFTER   # speed.md from two runs
"""

import importlib.util
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


def load(name):
    spec = importlib.util.spec_from_file_location(f"bench_{name}", HERE.parent / name / "speed.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Timed:
    def __init__(self, f):
        self.f, self.t = f, 0.0

    def __call__(self, *a):
        t0 = time.perf_counter()
        v = self.f(*a)
        self.t += time.perf_counter() - t0
        return v


def best(solve, funcs, iters, reps=5, budget=2.0):
    """(ms, own per iteration in ms, result) of the fastest run."""
    out, total = None, 0.0
    for _ in range(reps):
        tf = [Timed(f) if f is not None else None for f in funcs]
        t0 = time.perf_counter()
        r = solve(*tf)
        dt = time.perf_counter() - t0
        total += dt
        inside = sum(t.t for t in tf if t is not None)
        if out is None or dt * 1e3 < out[0]:
            out = (dt * 1e3, (dt - inside) * 1e3 / max(iters(r), 1), r)
        if total > budget:
            break
    return out


def row(func, method, variant, size, sd_ms, sp_ms, sd_own=None, sp_own=None, unit="ms/iter", note=""):
    return {
        "func": func,
        "method": method,
        "variant": variant,
        "size": str(size),
        "sd_ms": sd_ms,
        "sp_ms": sp_ms,
        "sd_own": sd_own,
        "sp_own": sp_own,
        "unit": unit,
        "note": note,
    }


def extra_rows():
    """The cases the O1-O5 benches leave out."""
    import numpy as np
    from scipy import optimize as so

    import stardust as sd

    sys.path.insert(0, str(ROOT))
    from tests.optimize import mgh

    rows = []
    # check_grad on scipy's rosen
    for n in (10, 100, 1000):
        x = np.tile([-1.2, 1.0], n // 2)
        xs = sd.asarray(x)
        f = lambda t: float(so.rosen(np.asarray(t)))  # noqa: E731
        g = lambda t: so.rosen_der(np.asarray(t))  # noqa: E731
        a = best(lambda tf, tg: sd.optimize.check_grad(tf, tg, xs, epsilon=2.0**-26), (f, g), lambda r: 1)
        b = best(lambda tf, tg: so.check_grad(tf, tg, x), (f, g), lambda r: 1)
        rows.append(row("check_grad", "forward", "epsilon=2^-26", f"n={n}", a[0], b[0], a[1], b[1], "ms/call"))
        a = best(lambda tf, tg: sd.optimize.check_grad(tf, tg, xs), (f, g), lambda r: 1)
        rows.append(row("check_grad", "accurate", "default", f"n={n}", a[0], b[0], a[1], b[1], "ms/call", "sd 5-point"))
    # least_squares: '3-point' and bounds
    c = mgh.BY_ID["mgh19-n11-m65-x1"]
    Fs = lambda x: sd.asarray(c.fun(x.tolist()))  # noqa: E731
    Fn = lambda x: np.array(c.fun(x.tolist()))  # noqa: E731
    for meth in ("trf", "lm"):
        a = best(
            lambda tf: sd.optimize.least_squares(tf, c.x0(), jac="3-point", method=meth), (Fs,), lambda r: r.n_iter
        )
        b = best(
            lambda tf: so.least_squares(tf, np.array(c.x0()), jac="3-point", method=meth),
            (Fn,),
            lambda r: r.njev or r.nfev,
        )
        rows.append(row("least_squares", meth, "3-point", "Osborne 2", a[0], b[0], a[1], b[1]))
    rng = np.random.default_rng(0)
    for m_, n in ((1000, 10), (10000, 100)):
        X = rng.standard_normal((m_, n)) / np.sqrt(n)
        y = np.exp(X @ np.full(n, 0.3)) + 0.05 * rng.standard_normal(m_)
        Xs, ys = sd.asarray(X), sd.asarray(y)
        F1 = lambda t, X=Xs, y=ys: y - sd.exp(sd.matmul(X, t))  # noqa: E731
        J1 = lambda t, X=Xs: -sd.reshape(sd.exp(sd.matmul(X, t)), (X.shape[0], 1)) * X  # noqa: E731
        F2 = lambda t, X=X, y=y: y - np.exp(X @ t)  # noqa: E731
        J2 = lambda t, X=X: -np.exp(X @ t)[:, None] * X  # noqa: E731
        bounds = ([0.0] * n, [0.25] * n)
        a = best(
            lambda tf, tj: sd.optimize.least_squares(tf, [0.1] * n, jac=tj, bounds=bounds), (F1, J1), lambda r: r.n_iter
        )
        b = best(
            lambda tf, tj: so.least_squares(tf, np.full(n, 0.1), jac=tj, bounds=bounds), (F2, J2), lambda r: r.njev
        )
        rows.append(row("least_squares", "trf", "analytic, bounds", f"({m_}, {n})", a[0], b[0], a[1], b[1]))
    # minimize with bounds: L-BFGS-B and Nelder-Mead on rosen
    fs, gs, _, _ = load("o4").sd_rosen()
    for n, cap in ((10, None), (100, None), (1000, 300)):
        x0 = np.tile([-1.2, 0.8], n // 2)  # inside the bounds (O52)
        lb, ub = [-2.0] * n, [0.8] * n
        opts = {} if cap is None else {"max_iter": cap}
        sopts = {} if cap is None else {"maxiter": cap}
        a = best(
            lambda tf, tg: sd.optimize.minimize(
                tf, x0.tolist(), jac=tg, method="l-bfgs-b", bounds=(lb, ub), options=opts
            ),
            (fs, gs),
            lambda r: r.n_iter,
        )
        b = best(
            lambda tf, tg: so.minimize(tf, x0, jac=tg, method="L-BFGS-B", bounds=list(zip(lb, ub)), options=sopts),
            (so.rosen, so.rosen_der),
            lambda r: r.nit,
        )
        rows.append(row("minimize", "l-bfgs-b", "analytic, bounds", f"n={n}", a[0], b[0], a[1], b[1]))
    for n in (2, 10):
        x0 = np.tile([-1.2, 0.8], n // 2)  # inside the bounds (O52)
        lb, ub = [-2.0] * n, [0.8] * n
        a = best(
            lambda tf: sd.optimize.minimize(tf, x0.tolist(), method="nelder-mead", bounds=(lb, ub)),
            (fs,),
            lambda r: r.n_iter,
        )
        b = best(
            lambda tf: so.minimize(tf, x0, method="Nelder-Mead", bounds=list(zip(lb, ub))), (so.rosen,), lambda r: r.nit
        )
        rows.append(row("minimize", "nelder-mead", "bounds", f"n={n}", a[0], b[0], a[1], b[1]))
    return rows


def normalize(o1, o2, o3, o4, o5):
    rows = []
    for r in o1:
        if r["group"] == "approx_fprime":
            rows.append(
                row("approx_fprime", "forward", "epsilon=2^-26", r["size"], r["sd_fwd_ms"], r["sp_ms"], unit="ms")
            )
            rows.append(
                row(
                    "approx_fprime",
                    "accurate",
                    "default",
                    r["size"],
                    r["sd_ms"],
                    r["sp_ms"],
                    unit="ms",
                    note="sd 5-point",
                )
            )
            continue
        meth, jac = r["method"].split("/")
        rows.append(row("least_squares", meth, jac, r["size"], r["sd_ms"], r["sp_ms"], r["sd_own"], r["sp_own"]))
    for r in o2:
        g = r["group"]
        if g == "root":
            rows.append(
                row(
                    "root_scalar",
                    r["method"],
                    "",
                    r["problem"],
                    r["sd_ms"],
                    r["sp_ms"],
                    r["sd_own_us"],
                    r["sp_own_us"],
                    "us/call",
                )
            )
        elif g == "minimum":
            rows.append(
                row(
                    "minimize_scalar",
                    r["method"],
                    "",
                    r["problem"],
                    r["sd_ms"],
                    r["sp_ms"],
                    r["sd_own_us"],
                    r["sp_own_us"],
                    "us/call",
                )
            )
        elif g == "batch":
            rows.append(
                row(
                    r["function"],
                    "",
                    "",
                    f"10^{round(math.log10(r['n']))}",
                    r["sd_ms"],
                    r["sp_ms"],
                    r["sd_own_ns"],
                    r["sp_own_ns"],
                    "ns/value",
                )
            )
        elif g == "loop":
            rows.append(
                row(
                    "root_scalar loop",
                    "brentq",
                    "per element",
                    f"10^{round(math.log10(r['n']))}",
                    r["sd_ms"],
                    r["sp_ms"],
                    unit="ms",
                    note="scipy brentq",
                )
            )
    for r in o3:
        size = f"n={r['n']}" + (f" ({r['cap']} it)" if r["cap"] else "")
        rows.append(row("minimize", r["method"], r["grad"], size, r["sd_ms"], r["sp_ms"], r["sd_own"], r["sp_own"]))
    for r in o4:
        size = f"n={r['n']}" + (f" ({r['cap']} it)" if r["cap"] else "")
        rows.append(row("minimize", r["method"], r["hess"], size, r["sd_ms"], r["sp_ms"], r["sd_own"], r["sp_own"]))
    for r in o5["root"]:
        rows.append(
            row(
                "root",
                r["method"],
                "jac" if r["jac"] else "diff",
                f"n={r['n']}",
                r["sd_ms"],
                r["sp_ms"],
                r.get("sd_own"),
                r.get("sp_own"),
            )
        )
    for r in o5["linear"]:
        func = "nnls" if r["what"] == "nnls" else "lsq_linear"
        rows.append(
            row(
                func,
                "" if func == "nnls" else r["what"],
                "",
                f"({r['m']}, {r['n']})",
                r["sd_ms"],
                r["sp_ms"],
                unit="ms",
            )
        )
    return rows


def worker(threads):
    sys.path.insert(0, str(ROOT))
    warnings.simplefilter("ignore")
    import numpy as np

    np.seterr(all="ignore")
    import stardust as sd

    sd.set_num_threads(threads)
    m1, m2, m3, m4, m5 = (load(f"o{k}") for k in range(1, 6))
    o2rows = (m2.scalar_cases() if threads == 1 else []) + m2.batch_cases([10**3, 10**4, 10**6])
    rows = normalize(m1.cases(), o2rows, m3.cases(), m4.cases(), {"root": m5.root_rows(), "linear": m5.linear_rows()})
    rows += extra_rows()
    print(json.dumps(rows))


def run(label):
    import numpy
    import scipy

    out = {"scipy": scipy.__version__, "numpy": numpy.__version__, "rows": {}}
    for t in (1, 10):
        env = dict(os.environ, VECLIB_MAXIMUM_THREADS=str(t), OMP_NUM_THREADS=str(t), OPENBLAS_NUM_THREADS=str(t))
        p = subprocess.run([sys.executable, __file__, "--worker", str(t)], env=env, capture_output=True, text=True)
        if p.returncode:
            sys.stderr.write(p.stderr)
            raise SystemExit(p.returncode)
        out["rows"][str(t)] = json.loads(p.stdout.strip().splitlines()[-1])
    (HERE / f"speed_{label}.json").write_text(json.dumps(out, indent=1) + "\n")


def g(v):
    return "" if v is None else f"{v:.3g}"


def table(before, after):
    B = json.loads((HERE / f"speed_{before}.json").read_text())["rows"]
    A = json.loads((HERE / f"speed_{after}.json").read_text())["rows"]

    def key(r):
        return (r["func"], r["method"], r["variant"], r["size"])

    lines = [
        "| function | method | variant | size | sd ms before (1/10) | sd ms after (1/10) | scipy ms (1/10) | "
        "scipy / sd after (1) | own before | own after | scipy own | unit |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    b1 = {key(r): r for r in B["1"]}
    b10 = {key(r): r for r in B["10"]}
    a10 = {key(r): r for r in A["10"]}
    for r in A["1"]:
        k = key(r)
        rb, rb10, ra10 = b1.get(k), b10.get(k), a10.get(k)
        before = f"{g(rb['sd_ms'])} / {g(rb10['sd_ms']) if rb10 else ''}" if rb else ""
        after = f"{g(r['sd_ms'])} / {g(ra10['sd_ms']) if ra10 else ''}"
        sp = f"{g(r['sp_ms'])} / {g(ra10['sp_ms']) if ra10 else ''}"
        ratio = r["sp_ms"] / r["sd_ms"] if r["sd_ms"] else math.nan
        lines.append(
            f"| {r['func']} | {r['method']} | {r['variant']} | {r['size']} | {before} | {after} | {sp} | "
            f"{ratio:.2f} | {g(rb['sd_own']) if rb else ''} | {g(r['sd_own'])} | {g(r['sp_own'])} | {r['unit']} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        worker(int(sys.argv[2]))
    elif len(sys.argv) > 3 and sys.argv[1] == "--table":
        sys.stdout.write(table(sys.argv[2], sys.argv[3]))
    else:
        run(sys.argv[1] if len(sys.argv) > 1 else "run")
