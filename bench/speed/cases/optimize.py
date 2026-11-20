"""sd.optimize against scipy.optimize. The problems, sizes and starting points are those of
bench/optimize/o1-o6/speed.py (the same generators and seeds):

  - least_squares ('trf', 'lm', analytic Jacobian): the residuals y_i - exp(x_i . theta)
    (x_i ~ N(0, 1/n), theta* = 0.3, from 0) at (m, n) = (10^3, 10) and (10^4, 100), and the
    Moré-Garbow-Hillstrom Osborne 2 problem (n = 11, m = 65, tests/optimize/mgh.py) from x0;
  - minimize on scipy's generalized Rosenbrock function from (-1.2, 1, -1.2, 1, ...): BFGS at
    n = 1000 (40 iterations), L-BFGS-B at n = 10^4 (300 iterations), Newton-CG with the
    Hessian at n = 1000 (200 iterations), trust-exact at n = 100 (to convergence);
  - root ('hybr', 'lm', with the Jacobian) on Broyden's tridiagonal system at n = 100 from -10;
  - find_root and find_minimum over 10^6 elements (x^3 - 2x - c and (x - c)^2 + 0.01 c x^4,
    c from 1 to 10);
  - lsq_linear ('trf', 'bvls', x >= 0) at (10^4, 100) and nnls at (100, 10), on a random A.

The user functions are written with each library's arrays (sd for sd, numpy for scipy), as
the old scripts did, so a time includes them; the iteration caps are the same for both and
the steps are the same (the methods follow scipy's), so both do the same iterations. A call
solves from the same start each time and keeps nothing, so nothing is rotated. scipy follows
sd's thread count (its BLAS pool). The small problems (Osborne 2, (10^3, 10), trust-exact at
n = 100, root at n = 100, nnls) have nothing to split and run at one thread only.
"""

from __future__ import annotations

import warnings

from ..registry import Case, Ref

AREA = "optimize"


def _quiet(f):
    """f with the comparators' overflow and convergence warnings ignored."""

    def g():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return f()

    return g


def _size(m, n):
    return f"({m:,}, {n})"


# ------------------------------------------------------------------------------ least_squares


def _exp_setup(m, n):
    """The (m, n) data of bench/optimize/o1: one generator draws (10^3, 10), (10^5, 10) and
    (10^4, 100) in turn, so the later sizes draw the earlier ones first."""

    def setup():
        import numpy as np

        import stardust as sd

        rng = np.random.default_rng(0)
        for mm, nn in ((1000, 10), (100_000, 10), (10_000, 100)):
            X = rng.standard_normal((mm, nn)) / np.sqrt(nn)
            y = np.exp(X @ np.full(nn, 0.3)) + 0.05 * rng.standard_normal(mm)
            if (mm, nn) == (m, n):
                break
        Xs, ys = sd.asarray(X), sd.asarray(y)
        return {
            "x0": [0.0] * n,
            "sF": lambda t: ys - sd.exp(sd.matmul(Xs, t)),
            "sJ": lambda t: -sd.reshape(sd.exp(sd.matmul(Xs, t)), (m, 1)) * Xs,
            "nF": lambda t: y - np.exp(X @ t),
            "nJ": lambda t: -np.exp(X @ t)[:, None] * X,
        }

    return setup


def _mgh_setup(cid):
    def setup():
        import numpy as np

        import stardust as sd
        from tests.optimize import mgh

        c = mgh.BY_ID[cid]
        return {
            "x0": c.x0(),
            "sF": lambda x: sd.asarray(c.fun(x.tolist())),
            "sJ": lambda x: sd.asarray(c.jac(x.tolist())),
            "nF": lambda x: np.array(c.fun(x.tolist())),
            "nJ": lambda x: np.array(c.jac(x.tolist())),
        }

    return setup


def _lsq_cases():
    out = []
    problems = (
        ("Osborne 2", "mgh19-n11-m65-x1", None, (1,), "MGH 19번 문제(n = 11, m = 65), x0에서 시작"),
        (_size(1000, 10), None, (1000, 10), (1,), ""),
        (_size(10_000, 100), None, (10_000, 100), (1, 10), ""),
    )
    for size, cid, shape, threads, note in problems:
        setup = _mgh_setup(cid) if cid else _exp_setup(*shape)
        for method in ("trf", "lm"):

            def sdf(d, method=method):
                import stardust as sd

                return lambda: sd.optimize.least_squares(d["sF"], d["x0"], jac=d["sJ"], method=method)

            def spf(d, method=method):
                import numpy as np
                from scipy import optimize as so

                x0 = np.asarray(d["x0"], dtype=float)
                return _quiet(lambda: so.least_squares(d["nF"], x0, jac=d["nJ"], method=method))

            out.append(
                Case(
                    area=AREA,
                    group="비선형 최소제곱",
                    name=f"least_squares {method}",
                    size=size,
                    setup=setup,
                    sd=sdf,
                    refs=(Ref("scipy", spf, threads="follow"),),
                    threads=threads,
                    quick=method == "trf" and shape == (1000, 10),
                    note=note,
                )
            )
    return out


# ------------------------------------------------------------------------------ minimize


def _sd_rosen():
    """scipy's rosen, rosen_der, rosen_hess written with sd's arrays (bench/optimize/o4)."""
    import stardust as sd

    def f(x):
        a = x[1:] - x[:-1] * x[:-1]
        b = 1.0 - x[:-1]
        return float(sd.sum(100.0 * (a * a) + b * b))

    def g(x):
        xm, xm_m1, xm_p1 = x[1:-1], x[:-2], x[2:]
        out = sd.zeros(x.shape[0])
        out[1:-1] = 200 * (xm - xm_m1**2) - 400 * (xm_p1 - xm**2) * xm - 2 * (1 - xm)
        out[0] = -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0])
        out[-1] = 200 * (x[-1] - x[-2] ** 2)
        return out

    def h(x):
        n = x.shape[0]
        H = sd.zeros((n, n))
        flat = sd.reshape(H, (-1,))
        off = -400 * x[:-1]
        flat[1 :: n + 1] = off
        flat[n :: n + 1] = off
        d = sd.zeros(n)
        d[0] = 1200 * x[0] ** 2 - 400 * x[1] + 2
        d[-1] = 200
        d[1:-1] = 202 + 1200 * x[1:-1] ** 2 - 400 * x[2:]
        flat[:: n + 1] = d
        return H

    return f, g, h


# (method, scipy's name, n, max_iter or None, with the Hessian, threads, quick)
MINIMIZE = (
    ("bfgs", "BFGS", 1000, 40, False, (1, 10), False),
    ("l-bfgs-b", "L-BFGS-B", 10_000, 300, False, (1, 10), True),
    ("newton-cg", "Newton-CG", 1000, 200, True, (1, 10), False),
    ("trust-exact", "trust-exact", 100, None, True, (1,), False),
)


def _minimize_cases():
    out = []
    for method, sp_name, n, cap, hess, threads, quick in MINIMIZE:
        x0 = [-1.2 if i % 2 == 0 else 1.0 for i in range(n)]

        def sdf(d, method=method, x0=x0, cap=cap, hess=hess):
            import stardust as sd

            f, g, h = _sd_rosen()
            kw = {"jac": g, "options": {} if cap is None else {"max_iter": cap}}
            if hess:
                kw["hess"] = h
            return lambda: sd.optimize.minimize(f, x0, method=method, **kw)

        def spf(d, sp_name=sp_name, x0=x0, cap=cap, hess=hess):
            import numpy as np
            from scipy import optimize as so

            kw = {"jac": so.rosen_der, "options": {} if cap is None else {"maxiter": cap}}
            if hess:
                kw["hess"] = so.rosen_hess
            x = np.array(x0)
            return _quiet(lambda: so.minimize(so.rosen, x, method=sp_name, **kw))

        variant = "기울기·헤세 행렬" if hess else "기울기"
        steps = f"{cap}회 반복" if cap else "수렴까지"
        out.append(
            Case(
                area=AREA,
                group="다변수 최소화",
                name=f"minimize {method}",
                size=f"n={n:,}",
                sd=sdf,
                refs=(Ref("scipy", spf, threads="follow"),),
                threads=threads,
                quick=quick,
                note=f"Rosenbrock, {variant}, {steps}",
            )
        )
    return out


# ------------------------------------------------------------------------------ root


def _root_setup():
    import numpy as np

    import stardust as sd

    def sf(x):
        out = (3.0 - 2.0 * x) * x + 1.0
        out[1:] = out[1:] - x[:-1]
        out[:-1] = out[:-1] - 2.0 * x[1:]
        return out

    def sj(x):
        n = x.shape[0]
        J = sd.zeros((n, n))
        flat = sd.reshape(J, (-1,))
        flat[:: n + 1] = 3.0 - 4.0 * x
        flat[n :: n + 1] = -1.0
        flat[1 :: n + 1] = -2.0
        return J

    def nf(x):
        out = (3.0 - 2.0 * x) * x + 1.0
        out[1:] -= x[:-1]
        out[:-1] -= 2.0 * x[1:]
        return out

    def nj(x):
        n = x.shape[0]
        J = np.zeros((n, n))
        flat = J.reshape(-1)
        flat[:: n + 1] = 3.0 - 4.0 * x
        flat[n :: n + 1] = -1.0
        flat[1 :: n + 1] = -2.0
        return J

    return {"x0": [-10.0] * 100, "sF": sf, "sJ": sj, "nF": nf, "nJ": nj}


def _root_cases():
    out = []
    for method in ("hybr", "lm"):

        def sdf(d, method=method):
            import stardust as sd

            return lambda: sd.optimize.root(d["sF"], d["x0"], method=method, jac=d["sJ"])

        def spf(d, method=method):
            import numpy as np
            from scipy import optimize as so

            x0 = np.array(d["x0"])
            return _quiet(lambda: so.root(d["nF"], x0, method=method, jac=d["nJ"]))

        out.append(
            Case(
                area=AREA,
                group="연립방정식",
                name=f"root {method}",
                size="n=100",
                setup=_root_setup,
                sd=sdf,
                refs=(Ref("scipy", spf, threads="follow"),),
                threads=(1,),
                quick=method == "hybr",
                note="Broyden 삼중대각 연립방정식, 야코비 행렬을 줌",
            )
        )
    return out


# ------------------------------------------------------------------------------ elementwise


def _batch_setup(n):
    def setup():
        import numpy as np

        import stardust as sd

        c = np.linspace(1.0, 10.0, n)
        return {"c": c, "sc": sd.asarray(c)}

    return setup


def _cubic(x, c):
    return x**3 - 2 * x - c


def _quartic(x, c):
    return (x - c) ** 2 + 0.01 * c * x**4


def _elementwise_cases():
    n = 1_000_000
    out = []
    for name, f, bracket, quick, note in (
        ("find_root", _cubic, (0.0, 3.0), True, "x³ − 2x − c = 0, 구간 (0, 3)"),
        ("find_minimum", _quartic, (-1.0, 0.5, 11.0), False, "(x − c)² + 0.01·c·x⁴, 세 점 (−1, 0.5, 11)"),
    ):

        def sdf(d, name=name, f=f, bracket=bracket):
            import stardust as sd

            solve = getattr(sd.optimize, name)
            return lambda: solve(f, bracket, args=(d["sc"],))

        def spf(d, name=name, f=f, bracket=bracket):
            from scipy.optimize import elementwise as ew

            solve = getattr(ew, name)
            return _quiet(lambda: solve(f, bracket, args=(d["c"],)))

        out.append(
            Case(
                area=AREA,
                group="원소별 풀이",
                name=name,
                size="10^6",
                n=n,
                setup=_batch_setup(n),
                sd=sdf,
                refs=(Ref("scipy.optimize.elementwise", spf, threads="follow", lib="scipy"),),
                quick=quick,
                note=note + ", c는 1부터 10까지",
            )
        )
    return out


# ------------------------------------------------------------------------------ linear


def _linear_setup(m, n):
    """The data of bench/optimize/o5: one generator (seed 9) draws (100, 10), (1000, 100) and
    (10^4, 100) in turn."""

    def setup():
        import numpy as np

        import stardust as sd

        rng = np.random.default_rng(9)
        for mm, nn in ((100, 10), (1000, 100), (10_000, 100)):
            A = rng.standard_normal((mm, nn))
            b = A @ rng.standard_normal(nn) + rng.standard_normal(mm)
            if (mm, nn) == (m, n):
                break
        return {"A": A, "b": b, "sA": sd.asarray(A), "sb": sd.asarray(b), "n": n}

    return setup


def _linear_cases():
    out = []
    for method in ("trf", "bvls"):

        def sdf(d, method=method):
            import stardust as sd

            return lambda: sd.optimize.lsq_linear(d["sA"], d["sb"], bounds=(0.0, float("inf")), method=method)

        def spf(d, method=method):
            import numpy as np
            from scipy import optimize as so

            bounds = (np.zeros(d["n"]), np.full(d["n"], np.inf))
            return _quiet(lambda: so.lsq_linear(d["A"], d["b"], bounds=bounds, method=method))

        out.append(
            Case(
                area=AREA,
                group="선형 최소제곱",
                name=f"lsq_linear {method}",
                size=_size(10_000, 100),
                setup=_linear_setup(10_000, 100),
                sd=sdf,
                refs=(Ref("scipy", spf, threads="follow"),),
                quick=method == "trf",
                note="x ≥ 0",
            )
        )

    def sd_nnls(d):
        import stardust as sd

        return lambda: sd.optimize.nnls(d["sA"], d["sb"])

    def sp_nnls(d):
        from scipy import optimize as so

        return lambda: so.nnls(d["A"], d["b"])

    out.append(
        Case(
            area=AREA,
            group="선형 최소제곱",
            name="nnls",
            size=_size(100, 10),
            setup=_linear_setup(100, 10),
            sd=sd_nnls,
            refs=(Ref("scipy", sp_nnls, threads="follow"),),
            threads=(1,),
        )
    )
    return out


def cases() -> list[Case]:
    return _lsq_cases() + _minimize_cases() + _root_cases() + _elementwise_cases() + _linear_cases()
