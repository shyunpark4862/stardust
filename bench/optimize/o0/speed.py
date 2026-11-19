"""O0: the baseline of sd.optimize, scipy 1.18.1's time and function counts on representative
problems of every method (docs/design/optimize.md §5). Results: bench/optimize/o0/speed.md
(the raw output in speed.log and speed.json).

Problems, written with numpy so that scipy's overhead and not Python loops is measured:
  - scalar roots: sin x - x/2 on [pi/2, pi] (APS 1) and x^3 - 2x - 5 on [2, 3] (Chandrupatla 1),
    x0 = 3 and 2.5 for the derivative methods;
  - scalar minimum: exp x - 5x (Chandrupatla's f3) with the bracket (0, 1, 3), bounds (0, 3);
  - batched (elementwise): x^3 - 2x - c = 0 and (x - c)^2 + c x^4 minima for c of 10^3-10^6
    elements, against a loop of brentq / minimize_scalar for 10^3 and 10^4;
  - minimize: the extended Rosenbrock function (scipy.optimize.rosen and its derivatives) at
    n = 10, 100, 1000 (L-BFGS-B also 10^4) from x0 = (-1.2, 1, -1.2, 1, ...);
  - least_squares: y_i - exp(x_i . theta) with x_i ~ N(0, 1/n), theta* = 0.3, start 0, at
    (m, n) = (10^3, 10), (10^5, 10), (10^4, 100), analytic and '2-point' Jacobians;
  - root: Broyden tridiagonal at n = 10, 100, 1000, with and without the Jacobian;
  - lsq_linear (trf, bvls) with bounds [-0.5, 0.5] and nnls, b = A linspace(-1, 1, n) + N(0, 1)
    (so that about half the bounds are active) on N(0, 1) matrices of the
    least squares sizes;
  - approx_fprime of rosen at n = 10, 100, 1000.

Time is the best of up to 5 runs (fewer when one run takes over 2 s), in ms. Each thread count
runs in its own process with VECLIB_MAXIMUM_THREADS set (numpy and scipy use Accelerate); the
callbacks are Python, so threads only enter the BLAS/LAPACK calls (L-BFGS-B's, trust-exact's
and the large least squares' factorizations).

Run: uv run --no-sync python bench/optimize/o0/speed.py
"""

import json
import os
import pathlib
import subprocess
import sys
import time
import warnings

HERE = pathlib.Path(__file__).parent


def best(fn, reps=5, budget=2.0):
    out, v = [], None
    for _ in range(reps):
        t0 = time.perf_counter()
        v = fn()
        out.append(time.perf_counter() - t0)
        if sum(out) > budget:
            break
    return min(out) * 1e3, v


def cases():
    import numpy as np
    from scipy import optimize as so
    from scipy.optimize import elementwise as ew

    rows = []

    def add(group, method, size, fn, counts):
        ms, r = best(fn)
        rows.append({"group": group, "method": method, "size": size, "ms": ms, **counts(r)})

    def nf(r):
        return {k: int(getattr(r, k)) for k in ("nfev", "njev", "nhev", "nit") if getattr(r, k, None) is not None}

    # scalar roots
    f1 = lambda x: np.sin(x) - x / 2  # noqa: E731
    f1d = lambda x: np.cos(x) - 0.5  # noqa: E731
    f1dd = lambda x: -np.sin(x)  # noqa: E731
    f2 = lambda x: x**3 - 2 * x - 5  # noqa: E731
    f2d = lambda x: 3 * x**2 - 2  # noqa: E731
    f2dd = lambda x: 6 * x  # noqa: E731
    for name, f, d, dd, br, x0 in (
        ("aps.01", f1, f1d, f1dd, (np.pi / 2, np.pi), 3.0),
        ("chandrupatla.1", f2, f2d, f2dd, (2.0, 3.0), 2.5),
    ):
        for m in ("brentq", "toms748", "bisect"):
            add(
                "root_scalar",
                m,
                name,
                lambda m=m, f=f, br=br: so.root_scalar(f, bracket=br, method=m),
                lambda r: {"nfev": r.function_calls, "nit": r.iterations},
            )
        add(
            "root_scalar",
            "newton",
            name,
            lambda f=f, d=d, x0=x0: so.root_scalar(f, x0=x0, fprime=d, method="newton"),
            lambda r: {"nfev": r.function_calls, "nit": r.iterations},
        )
        add(
            "root_scalar",
            "halley",
            name,
            lambda f=f, d=d, dd=dd, x0=x0: so.root_scalar(f, x0=x0, fprime=d, fprime2=dd, method="halley"),
            lambda r: {"nfev": r.function_calls, "nit": r.iterations},
        )
        add(
            "root_scalar",
            "secant",
            name,
            lambda f=f, x0=x0: so.root_scalar(f, x0=x0, method="secant"),
            lambda r: {"nfev": r.function_calls, "nit": r.iterations},
        )
    g3 = lambda x: np.exp(x) - 5 * x  # noqa: E731
    for m in ("brent", "golden"):
        add(
            "minimize_scalar",
            m,
            "chandrupatla_min.3",
            lambda m=m: so.minimize_scalar(g3, bracket=(0, 1, 3), method=m),
            nf,
        )
    add(
        "minimize_scalar",
        "bounded",
        "chandrupatla_min.3",
        lambda: so.minimize_scalar(g3, bounds=(0, 3), method="bounded"),
        nf,
    )

    # batched scalars
    for k in (3, 4, 5, 6):
        c = np.linspace(1.0, 10.0, 10**k)
        add(
            "elementwise",
            "find_root",
            f"1e{k}",
            lambda c=c: ew.find_root(lambda x, c: x**3 - 2 * x - c, (0.0, 3.0), args=(c,)),
            lambda r: {"nfev": int(np.max(r.nfev)), "nit": int(np.max(r.nit))},
        )
        add(
            "elementwise",
            "bracket_root",
            f"1e{k}",
            lambda c=c: ew.bracket_root(lambda x, c: x**3 - 2 * x - c, 0.0, args=(c,)),
            lambda r: {"nfev": int(np.max(r.nfev)), "nit": int(np.max(r.nit))},
        )
        add(
            "elementwise",
            "find_minimum",
            f"1e{k}",
            lambda c=c: ew.find_minimum(lambda x, c: (x - c) ** 2 + 0.01 * c * x**4, (-1.0, 0.5, 11.0), args=(c,)),
            lambda r: {"nfev": int(np.max(r.nfev)), "nit": int(np.max(r.nit))},
        )
        add(
            "elementwise",
            "bracket_minimum",
            f"1e{k}",
            lambda c=c: ew.bracket_minimum(lambda x, c: (x - c) ** 2 + 0.01 * c * x**4, 0.0, args=(c,)),
            lambda r: {"nfev": int(np.max(r.nfev)), "nit": int(np.max(r.nit))},
        )
        if k <= 4:
            add(
                "elementwise",
                "loop of brentq",
                f"1e{k}",
                lambda c=c: [so.brentq(lambda x, cc=cc: x**3 - 2 * x - cc, 0.0, 3.0) for cc in c],
                lambda r: {},
            )
            add(
                "elementwise",
                "loop of minimize_scalar(brent)",
                f"1e{k}",
                lambda c=c: [
                    so.minimize_scalar(lambda x, cc=cc: (x - cc) ** 2 + 0.01 * cc * x**4, bracket=(-1.0, 0.5, 11.0))
                    for cc in c
                ],
                lambda r: {},
            )

    # minimize on the extended Rosenbrock function
    for n in (10, 100, 1000, 10000):
        x0 = np.tile([-1.2, 1.0], n // 2)
        meths = (
            ["L-BFGS-B", "L-BFGS-B/fd"]
            if n == 10000
            else ["Nelder-Mead", "BFGS", "BFGS/fd", "L-BFGS-B", "L-BFGS-B/fd", "Newton-CG", "trust-ncg", "trust-exact"]
        )
        if n == 1000:
            meths = [m for m in meths if m not in ("Nelder-Mead", "BFGS/fd")]
        if n == 100:
            meths = [m for m in meths if m != "Nelder-Mead"] + ["Nelder-Mead"]
        for m in meths:
            meth, _, fd = m.partition("/")
            kw = {}
            if not fd and meth != "Nelder-Mead":
                kw["jac"] = so.rosen_der
            if meth in ("Newton-CG", "trust-ncg"):
                kw["hessp"] = so.rosen_hess_prod
            if meth == "trust-exact":
                kw["hess"] = so.rosen_hess
            add(
                "minimize",
                m,
                f"n={n}",
                lambda meth=meth, kw=kw, x0=x0: so.minimize(so.rosen, x0, method=meth, **kw),
                nf,
            )

    # least squares: y - exp(X theta)
    rng = np.random.default_rng(0)
    for m_, n in ((1000, 10), (100000, 10), (10000, 100)):
        X = rng.standard_normal((m_, n)) / np.sqrt(n)
        y = np.exp(X @ np.full(n, 0.3)) + 0.05 * rng.standard_normal(m_)
        F = lambda t, X=X, y=y: y - np.exp(X @ t)  # noqa: E731
        J = lambda t, X=X: -np.exp(X @ t)[:, None] * X  # noqa: E731
        for meth in ("trf", "lm"):
            for jac in ("analytic", "2-point"):
                add(
                    "least_squares",
                    f"{meth}/{jac}",
                    f"({m_}, {n})",
                    lambda F=F, J=J, meth=meth, jac=jac, n=n: so.least_squares(
                        F, np.zeros(n), jac=J if jac == "analytic" else "2-point", method=meth
                    ),
                    nf,
                )
        A = rng.standard_normal((m_, n))
        b = A @ np.linspace(-1.0, 1.0, n) + rng.standard_normal(m_)
        for meth in ("trf", "bvls"):
            add(
                "lsq_linear",
                meth,
                f"({m_}, {n})",
                lambda A=A, b=b, meth=meth: so.lsq_linear(A, b, bounds=(-0.5, 0.5), method=meth),
                nf,
            )
        add("nnls", "nnls", f"({m_}, {n})", lambda A=A, b=b: so.nnls(A, b), lambda r: {})

    # root on the Broyden tridiagonal system
    def broyden(x):
        f = (3.0 - 2.0 * x) * x + 1.0
        f[1:] -= x[:-1]
        f[:-1] -= 2.0 * x[1:]
        return f

    def broyden_jac(x):
        n = x.size
        j = np.diag(3.0 - 4.0 * x)
        j[np.arange(1, n), np.arange(n - 1)] = -1.0
        j[np.arange(n - 1), np.arange(1, n)] = -2.0
        return j

    for n in (10, 100, 1000):
        for meth in ("hybr", "lm"):
            for jac in ("analytic", "none"):
                add(
                    "root",
                    f"{meth}/{jac}",
                    f"n={n}",
                    lambda n=n, meth=meth, jac=jac: so.root(
                        broyden, -np.ones(n), method=meth, jac=broyden_jac if jac == "analytic" else None
                    ),
                    nf,
                )
    for n in (10, 100, 1000):
        x = np.tile([-1.2, 1.0], n // 2)
        add("approx_fprime", "approx_fprime", f"n={n}", lambda x=x: so.approx_fprime(x, so.rosen), lambda r: {})
    return rows


def worker():
    warnings.simplefilter("ignore")
    import numpy as np

    np.seterr(all="ignore")
    print(json.dumps(cases()))


def main():
    import numpy
    import scipy

    results = {}
    for t in (1, 10):
        env = dict(os.environ, VECLIB_MAXIMUM_THREADS=str(t), OMP_NUM_THREADS=str(t), OPENBLAS_NUM_THREADS=str(t))
        out = subprocess.run(
            [sys.executable, __file__, "--worker"], env=env, capture_output=True, text=True, check=True
        )
        results[t] = json.loads(out.stdout.strip().splitlines()[-1])
    (HERE / "speed.json").write_text(
        json.dumps({"scipy": scipy.__version__, "numpy": numpy.__version__, "rows": results}, indent=1) + "\n"
    )
    lines = [
        f"scipy {scipy.__version__}, numpy {numpy.__version__}",
        "",
        "| group | method | size | 1 thread (ms) | 10 threads (ms) | nfev | njev | nhev | nit |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for a, b in zip(results[1], results[10]):
        cell = lambda k: str(a.get(k, "")) if a.get(k, "") == b.get(k, "") else f"{a.get(k, '')}/{b.get(k, '')}"  # noqa: E731
        lines.append(
            f"| {a['group']} | {a['method']} | {a['size']} | {a['ms']:.3g} | {b['ms']:.3g} | {cell('nfev')} | "
            f"{cell('njev')} | {cell('nhev')} | {cell('nit')} |"
        )
    text = "\n".join(lines) + "\n"
    (HERE / "speed.log").write_text(text)
    print(text)


if __name__ == "__main__":
    if "--worker" in sys.argv:
        worker()
    else:
        main()
