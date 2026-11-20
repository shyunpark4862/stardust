"""sd.optimize against scipy.optimize. The problems, sizes, starting points and seeds are those
of bench/optimize/o1-o6/speed.py:

  - least_squares ('trf', 'lm'): the Moré-Garbow-Hillstrom problems Rosenbrock (n = 2), Osborne 2
    (n = 11, m = 65) and Brown almost-linear (n = 40) from x0 (tests/optimize/mgh.py), and the
    residuals y_i - exp(x_i . theta) (x_i ~ N(0, 1/n), theta* = 0.3, from 0) at (m, n) = (10^3,
    10), (10^5, 10), (10^4, 100); with the Jacobian, with '2-point' and '3-point' differences,
    and with bounds (0 <= theta <= 0.25 from 0.1, 'trf');
  - minimize on scipy's generalized Rosenbrock function from (-1.2, 1, -1.2, 1, ...): every
    method, with the gradient and with differences, the Hessian, its product, its differences
    or none (Newton-CG), and bounds (-2 <= x <= 0.8 from (-1.2, 0.8, ...), L-BFGS-B and
    Nelder-Mead), from n = 2 (the fixed cost of a call) to 10^4; large n are capped at the same
    iteration count for both;
  - root ('hybr', 'lm', with the Jacobian and with differences) on Broyden's tridiagonal system
    from -10 at n = 10 and 100;
  - root_scalar (every method) on sin x - x/2 (APS 1) and x^3 - 2x - 5 (Chandrupatla 1), and
    minimize_scalar (every method) on e^x - 5x: the same Python function for both libraries;
  - find_root, bracket_root, find_minimum, bracket_minimum over 10^3 to 10^6 elements
    (x^3 - 2x - c and (x - c)^2 + 0.01 c x^4, c from 1 to 10): the split into threads starts at
    2^15 elements;
  - approx_fprime and check_grad of scipy's rosen at n = 10, 100, 1000: sd's default (5-point
    differences) and forward differences with scipy's step (epsilon = 2^-26);
  - lsq_linear ('trf', 'bvls', x >= 0) and nnls on a random A at (100, 10), (1000, 100) and
    (10^4, 100).

The user functions of the array problems are written with each library's arrays (sd for sd,
numpy for scipy), as the old scripts did, so a time includes them; the steps follow scipy's, so
both libraries do the same iterations. A call solves from the same start each time and keeps
nothing, so nothing is rotated. scipy follows sd's thread count (its BLAS pool). A case runs at
10 threads too only where something can split: least_squares from m n >= 4 10^5 (the QR), the
element-wise solvers from 10^5 elements, minimize from n = 1000, lsq_linear and nnls at
(10^4, 100).

Left out: the old scripts' Nelder-Mead at n = 1000, BFGS and L-BFGS-B with differences at
n = 1000, trust-exact at n = 1000 (0.5 s: the Hessian's factorization, as at n = 100), and the
loop of root_scalar against brentq per element.
"""

from __future__ import annotations

import math
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

LSQ_GROUP = "비선형 최소제곱"
MGH = {"Rosenbrock": "mgh01-n2-m2-x1", "Osborne 2": "mgh19-n11-m65-x1", "Brown 거의 선형": "mgh27-n40-m40-x1"}
MGH_NOTE = {
    "Rosenbrock": "MGH 1번 문제(n = 2, m = 2)",
    "Osborne 2": "MGH 19번 문제(n = 11, m = 65)",
    "Brown 거의 선형": "MGH 27번 문제(n = 40, m = 40)",
}


def _exp_setup(m, n, x0=0.0):
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
            "x0": [x0] * n,
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


def _lsq_case(size, setup, method, jac, threads, note, bounds=None, quick=False):
    """jac: "analytic" (the user's Jacobian), "2-point" or "3-point"."""

    def sdf(d):
        import stardust as sd

        J = d["sJ"] if jac == "analytic" else jac
        kw = {} if bounds is None else {"bounds": bounds(d)}
        return lambda: sd.optimize.least_squares(d["sF"], d["x0"], jac=J, method=method, **kw)

    def spf(d):
        import numpy as np
        from scipy import optimize as so

        J = d["nJ"] if jac == "analytic" else jac
        kw = {} if bounds is None else {"bounds": bounds(d)}
        x0 = np.asarray(d["x0"], dtype=float)
        return _quiet(lambda: so.least_squares(d["nF"], x0, jac=J, method=method, **kw))

    name = f"least_squares {method}"
    if jac != "analytic":
        name += f", jac='{jac}'"
    if bounds is not None:
        name += ", bounds"
    return Case(
        area=AREA,
        group=LSQ_GROUP,
        name=name,
        size=size,
        setup=setup,
        sd=sdf,
        refs=(Ref("scipy", spf, threads="follow"),),
        threads=threads,
        quick=quick,
        note=note,
    )


def _lsq_cases():
    out = []
    problems = [(name, _mgh_setup(cid), None, MGH_NOTE[name]) for name, cid in MGH.items()]
    problems += [(_size(m, n), _exp_setup(m, n), (m, n), "") for m, n in ((1000, 10), (100_000, 10), (10_000, 100))]
    differenced = ("Osborne 2", "Brown 거의 선형", _size(1000, 10), _size(10_000, 100))
    for size, setup, shape, note in problems:
        threads = (1, 10) if shape and shape[0] * shape[1] >= 400_000 else (1,)
        for method in ("trf", "lm"):
            out.append(
                _lsq_case(size, setup, method, "analytic", threads, note, quick=method == "trf" and shape == (1000, 10))
            )
            if size in differenced:
                out.append(_lsq_case(size, setup, method, "2-point", threads, note))
            if size == "Osborne 2":
                out.append(_lsq_case(size, setup, method, "3-point", threads, note))
    for m, n in ((1000, 10), (10_000, 100)):
        out.append(
            _lsq_case(
                _size(m, n),
                _exp_setup(m, n, x0=0.1),
                "trf",
                "analytic",
                (1, 10) if m * n >= 400_000 else (1,),
                "0 ≤ θ ≤ 0.25, 0.1에서 시작",
                bounds=lambda d: ([0.0] * len(d["x0"]), [0.25] * len(d["x0"])),
            )
        )
    return out


# ------------------------------------------------------------------------------ minimize

MIN_GROUP = "다변수 최소화"
SCIPY_METHOD = {
    "nelder-mead": "Nelder-Mead",
    "bfgs": "BFGS",
    "l-bfgs-b": "L-BFGS-B",
    "newton-cg": "Newton-CG",
    "trust-ncg": "trust-ncg",
    "trust-exact": "trust-exact",
}


def _sd_rosen():
    """scipy's rosen, rosen_der, rosen_hess, rosen_hess_prod written with sd's arrays
    (bench/optimize/o4)."""
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

    def hp(x, p):
        out = sd.zeros(p.shape[0])
        out[0] = (1200 * x[0] ** 2 - 400 * x[1] + 2) * p[0] - 400 * x[0] * p[1]
        out[1:-1] = -400 * x[:-2] * p[:-2] + (202 + 1200 * x[1:-1] ** 2 - 400 * x[2:]) * p[1:-1] - 400 * x[1:-1] * p[2:]
        out[-1] = -400 * x[-2] * p[-2] + 200 * p[-1]
        return out

    return {"f": f, "g": g, "h": h, "hp": hp}


def _np_rosen():
    from scipy import optimize as so

    return {"f": so.rosen, "g": so.rosen_der, "h": so.rosen_hess, "hp": so.rosen_hess_prod}


# (method, derivatives, n, max_iter or None, bounds, quick). derivatives: "-" (none: Nelder-Mead),
# "jac" (the gradient), "diff" (the gradient by differences), "hess", "hessp", "hess=2-point",
# "hess=None" (the gradient only).
MINIMIZE = (
    ("nelder-mead", "-", 2, None, False, False),
    ("nelder-mead", "-", 10, None, False, False),
    ("nelder-mead", "-", 100, 2000, False, False),
    ("nelder-mead", "-", 10, None, True, False),
    ("bfgs", "jac", 2, None, False, False),
    ("bfgs", "jac", 10, None, False, False),
    ("bfgs", "jac", 100, None, False, False),
    ("bfgs", "jac", 1000, 40, False, False),
    ("bfgs", "diff", 10, None, False, False),
    ("bfgs", "diff", 100, 100, False, False),
    ("l-bfgs-b", "jac", 2, None, False, False),
    ("l-bfgs-b", "jac", 100, None, False, False),
    ("l-bfgs-b", "jac", 1000, 300, False, False),
    ("l-bfgs-b", "jac", 10_000, 300, False, True),
    ("l-bfgs-b", "diff", 100, 100, False, False),
    ("l-bfgs-b", "jac", 100, None, True, False),
    ("l-bfgs-b", "jac", 1000, 300, True, False),
    ("newton-cg", "hess", 2, None, False, False),
    ("newton-cg", "hess", 100, None, False, False),
    ("newton-cg", "hess", 1000, 200, False, False),
    ("newton-cg", "hessp", 100, None, False, False),
    ("newton-cg", "hessp", 1000, 200, False, False),
    ("newton-cg", "hess=2-point", 100, None, False, False),
    ("newton-cg", "hess=2-point", 1000, 200, False, False),
    ("newton-cg", "hess=None", 100, None, False, False),
    ("newton-cg", "hess=None", 1000, 200, False, False),
    ("trust-ncg", "hess", 100, None, False, False),
    ("trust-ncg", "hess", 1000, 300, False, False),
    ("trust-ncg", "hessp", 100, None, False, False),
    ("trust-ncg", "hessp", 1000, 300, False, False),
    ("trust-exact", "hess", 2, None, False, False),
    ("trust-exact", "hess", 10, None, False, False),
    ("trust-exact", "hess", 100, None, False, False),
)
_DERIV_NAME = {
    "-": "",
    "jac": "",
    "diff": ", jac=None",
    "hess": ", hess",
    "hessp": ", hessp",
    "hess=2-point": ", hess='2-point'",
    "hess=None": ", hess=None",
}
_DERIV_NOTE = {
    "-": "",
    "jac": "기울기",
    "diff": "차분 기울기",
    "hess": "기울기·헤세 행렬",
    "hessp": "기울기·헤세 행렬 곱",
    "hess=2-point": "기울기·차분 헤세 행렬",
    "hess=None": "기울기만(헤세 행렬 곱은 기울기의 차분; 기울기 호출이 scipy의 절반가량)",
}


def _minimize_kwargs(fs, deriv, cap, bounds, n, scipy):
    kw = {}
    if deriv != "-" and deriv != "diff":
        kw["jac"] = fs["g"]
    if deriv == "hess":
        kw["hess"] = fs["h"]
    elif deriv == "hessp":
        kw["hessp"] = fs["hp"]
    elif deriv == "hess=2-point":
        kw["hess"] = "2-point"
    if cap is not None:
        kw["options"] = {"maxiter": cap} if scipy else {"max_iter": cap}
    if bounds:
        lb, ub = [-2.0] * n, [0.8] * n
        kw["bounds"] = list(zip(lb, ub)) if scipy else (lb, ub)
    return kw


def _minimize_cases():
    out = []
    for method, deriv, n, cap, bounds, quick in MINIMIZE:
        x0 = [(-1.2 if i % 2 == 0 else (0.8 if bounds else 1.0)) for i in range(n)]

        def sdf(d, method=method, deriv=deriv, n=n, cap=cap, bounds=bounds, x0=x0):
            import stardust as sd

            fs = _sd_rosen()
            kw = _minimize_kwargs(fs, deriv, cap, bounds, n, scipy=False)
            return lambda: sd.optimize.minimize(fs["f"], x0, method=method, **kw)

        def spf(d, method=method, deriv=deriv, n=n, cap=cap, bounds=bounds, x0=x0):
            import numpy as np
            from scipy import optimize as so

            fs = _np_rosen()
            kw = _minimize_kwargs(fs, deriv, cap, bounds, n, scipy=True)
            x = np.array(x0)
            return _quiet(lambda: so.minimize(fs["f"], x, method=SCIPY_METHOD[method], **kw))

        parts = ["Rosenbrock"]
        if _DERIV_NOTE[deriv]:
            parts.append(_DERIV_NOTE[deriv])
        if bounds:
            parts.append("−2 ≤ x ≤ 0.8")
        parts.append(f"{cap}회 반복" if cap else "수렴까지")
        out.append(
            Case(
                area=AREA,
                group=MIN_GROUP,
                name=f"minimize {method}{_DERIV_NAME[deriv]}" + (", bounds" if bounds else ""),
                size=f"n={n:,}",
                sd=sdf,
                refs=(Ref("scipy", spf, threads="follow"),),
                threads=(1, 10) if n >= 1000 else (1,),
                quick=quick,
                note=", ".join(parts),
            )
        )
    return out


# ------------------------------------------------------------------------------ root


def _root_setup(n):
    def setup():
        import numpy as np

        import stardust as sd

        def sf(x):
            out = (3.0 - 2.0 * x) * x + 1.0
            out[1:] = out[1:] - x[:-1]
            out[:-1] = out[:-1] - 2.0 * x[1:]
            return out

        def sj(x):
            k = x.shape[0]
            J = sd.zeros((k, k))
            flat = sd.reshape(J, (-1,))
            flat[:: k + 1] = 3.0 - 4.0 * x
            flat[k :: k + 1] = -1.0
            flat[1 :: k + 1] = -2.0
            return J

        def nf(x):
            out = (3.0 - 2.0 * x) * x + 1.0
            out[1:] -= x[:-1]
            out[:-1] -= 2.0 * x[1:]
            return out

        def nj(x):
            k = x.shape[0]
            J = np.zeros((k, k))
            flat = J.reshape(-1)
            flat[:: k + 1] = 3.0 - 4.0 * x
            flat[k :: k + 1] = -1.0
            flat[1 :: k + 1] = -2.0
            return J

        return {"x0": [-10.0] * n, "sF": sf, "sJ": sj, "nF": nf, "nJ": nj}

    return setup


def _root_cases():
    out = []
    for n in (10, 100):
        for method in ("hybr", "lm"):
            for jac in (True, False):

                def sdf(d, method=method, jac=jac):
                    import stardust as sd

                    J = d["sJ"] if jac else None
                    return lambda: sd.optimize.root(d["sF"], d["x0"], method=method, jac=J)

                def spf(d, method=method, jac=jac):
                    import numpy as np
                    from scipy import optimize as so

                    x0 = np.array(d["x0"])
                    J = d["nJ"] if jac else None
                    return _quiet(lambda: so.root(d["nF"], x0, method=method, jac=J))

                out.append(
                    Case(
                        area=AREA,
                        group="연립방정식",
                        name=f"root {method}" + ("" if jac else ", jac=None"),
                        size=f"n={n}",
                        setup=_root_setup(n),
                        sd=sdf,
                        refs=(Ref("scipy", spf, threads="follow"),),
                        threads=(1,),
                        quick=method == "hybr" and jac and n == 100,
                        note="Broyden 삼중대각 연립방정식, " + ("야코비 행렬을 줌" if jac else "차분 야코비 행렬"),
                    )
                )
    return out


# ------------------------------------------------------------------------------ scalar


def _aps1(x):
    return math.sin(x) - x / 2


def _ch1(x):
    return x**3 - 2 * x - 5


# problem -> (f, f', f'', bracket, x0)
SCALAR_ROOTS = {
    "APS 1": (_aps1, lambda x: math.cos(x) - 0.5, lambda x: -math.sin(x), (math.pi / 2, math.pi), 2.0),
    "Chandrupatla 1": (_ch1, lambda x: 3 * x * x - 2, lambda x: 6 * x, (2.0, 3.0), 2.5),
}
SCALAR_ROOT_NOTE = {"APS 1": "sin x − x/2, 구간 (π/2, π)", "Chandrupatla 1": "x³ − 2x − 5, 구간 (2, 3)"}


def _scalar_cases():
    out = []
    for problem, (f, d1, d2, bracket, x0) in SCALAR_ROOTS.items():
        for method in ("brentq", "toms748", "bisect", "newton", "halley", "secant"):
            kw = {"method": method}
            if method in ("brentq", "toms748", "bisect"):
                kw["bracket"] = bracket
                start = ""
            else:
                kw["x0"] = x0
                start = f", x0 = {x0}"
            if method in ("newton", "halley"):
                kw["fprime"] = d1
            if method == "halley":
                kw["fprime2"] = d2

            def sdf(d, f=f, kw=kw):
                import stardust as sd

                return lambda: sd.optimize.root_scalar(f, **kw)

            def spf(d, f=f, kw=kw):
                from scipy import optimize as so

                return _quiet(lambda: so.root_scalar(f, **kw))

            out.append(
                Case(
                    area=AREA,
                    group="스칼라 근·최솟값",
                    name=f"root_scalar {method}",
                    size=problem,
                    sd=sdf,
                    refs=(Ref("scipy", spf),),
                    threads=(1,),
                    quick=method == "brentq" and problem == "APS 1",
                    note=SCALAR_ROOT_NOTE[problem] + start,
                )
            )

    def mn(x):
        return math.exp(x) - 5 * x

    for method, kw, note in (
        ("brent", {"bracket": (0.0, 1.0, 3.0)}, "세 점 (0, 1, 3)"),
        ("golden", {"bracket": (0.0, 1.0, 3.0)}, "세 점 (0, 1, 3)"),
        ("bounded", {"bounds": (0.0, 3.0)}, "구간 (0, 3)"),
    ):

        def sdf(d, method=method, kw=kw):
            import stardust as sd

            return lambda: sd.optimize.minimize_scalar(mn, method=method, **kw)

        def spf(d, method=method, kw=kw):
            from scipy import optimize as so

            return _quiet(lambda: so.minimize_scalar(mn, method=method, **kw))

        out.append(
            Case(
                area=AREA,
                group="스칼라 근·최솟값",
                name=f"minimize_scalar {method}",
                size="eˣ − 5x",
                sd=sdf,
                refs=(Ref("scipy", spf),),
                threads=(1,),
                note=note,
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


# (function, f, start, sizes, quick size, note)
ELEMENTWISE = (
    ("find_root", _cubic, ((0.0, 3.0),), (10**3, 10**4, 10**5, 10**6), 10**6, "x³ − 2x − c = 0, 구간 (0, 3)"),
    ("bracket_root", _cubic, (0.0,), (10**4, 10**6), None, "x³ − 2x − c, 0에서 시작"),
    (
        "find_minimum",
        _quartic,
        ((-1.0, 0.5, 11.0),),
        (10**3, 10**4, 10**5, 10**6),
        None,
        "(x − c)² + 0.01·c·x⁴, 세 점 (−1, 0.5, 11)",
    ),
    ("bracket_minimum", _quartic, (0.0,), (10**4, 10**6), None, "(x − c)² + 0.01·c·x⁴, 0에서 시작"),
)


def _elementwise_cases():
    out = []
    for name, f, start, sizes, quick_n, note in ELEMENTWISE:
        for n in sizes:

            def sdf(d, name=name, f=f, start=start):
                import stardust as sd

                solve = getattr(sd.optimize, name)
                return lambda: solve(f, *start, args=(d["sc"],))

            def spf(d, name=name, f=f, start=start):
                from scipy.optimize import elementwise as ew

                solve = getattr(ew, name)
                return _quiet(lambda: solve(f, *start, args=(d["c"],)))

            out.append(
                Case(
                    area=AREA,
                    group="원소별 풀이",
                    name=name,
                    size=f"10^{round(math.log10(n))}",
                    n=n,
                    setup=_batch_setup(n),
                    sd=sdf,
                    refs=(Ref("scipy.optimize.elementwise", spf, threads="follow", lib="scipy"),),
                    threads=(1, 10) if n >= 10**5 else (1,),
                    quick=n == quick_n,
                    note=note + ", c는 1부터 10까지",
                )
            )
    return out


# ------------------------------------------------------------------------------ derivatives


def _derivative_setup(n):
    def setup():
        import numpy as np
        from scipy import optimize as so

        import stardust as sd

        x = np.tile([-1.2, 1.0], n // 2)
        return {
            "x": x,
            "sx": sd.asarray(x),
            "f": lambda t: float(so.rosen(np.asarray(t))),
            "g": lambda t: so.rosen_der(np.asarray(t)),
        }

    return setup


def _derivative_cases():
    out = []
    for n in (10, 100, 1000):
        for func in ("approx_fprime", "check_grad"):
            for forward in (False, True):

                def sdf(d, func=func, forward=forward):
                    import stardust as sd

                    kw = {"epsilon": 2.0**-26} if forward else {}
                    if func == "approx_fprime":
                        return lambda: sd.optimize.approx_fprime(d["sx"], d["f"], **kw)
                    return lambda: sd.optimize.check_grad(d["f"], d["g"], d["sx"], **kw)

                def spf(d, func=func):
                    from scipy import optimize as so

                    if func == "approx_fprime":
                        return lambda: so.approx_fprime(d["x"], d["f"])
                    return lambda: so.check_grad(d["f"], d["g"], d["x"])

                note = (
                    "Rosenbrock(scipy의 rosen), 둘 다 전진 차분"
                    if forward
                    else "Rosenbrock(scipy의 rosen), sd 기본은 5점 차분(f를 약 10n번), scipy는 전진 차분(n + 1번)"
                )
                out.append(
                    Case(
                        area=AREA,
                        group="미분 근사",
                        name=func + (", epsilon=2^-26" if forward else ""),
                        size=f"n={n:,}",
                        setup=_derivative_setup(n),
                        sd=sdf,
                        refs=(Ref("scipy", spf),),
                        threads=(1,),
                        quick=func == "approx_fprime" and forward and n == 100,
                        note=note,
                    )
                )
    return out


# ------------------------------------------------------------------------------ linear

LINEAR_SHAPES = ((100, 10), (1000, 100), (10_000, 100))


def _linear_setup(m, n):
    """The data of bench/optimize/o5: one generator (seed 9) draws the shapes of LINEAR_SHAPES
    in turn."""

    def setup():
        import numpy as np

        import stardust as sd

        rng = np.random.default_rng(9)
        for mm, nn in LINEAR_SHAPES:
            A = rng.standard_normal((mm, nn))
            b = A @ rng.standard_normal(nn) + rng.standard_normal(mm)
            if (mm, nn) == (m, n):
                break
        return {"A": A, "b": b, "sA": sd.asarray(A), "sb": sd.asarray(b), "n": n}

    return setup


def _linear_cases():
    out = []
    for m, n in LINEAR_SHAPES:
        threads = (1, 10) if m * n >= 10**6 else (1,)
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
                    size=_size(m, n),
                    setup=_linear_setup(m, n),
                    sd=sdf,
                    refs=(Ref("scipy", spf, threads="follow"),),
                    threads=threads,
                    quick=method == "trf" and m == 10_000,
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
                size=_size(m, n),
                setup=_linear_setup(m, n),
                sd=sd_nnls,
                refs=(Ref("scipy", sp_nnls, threads="follow"),),
                threads=threads,
            )
        )
    return out


def cases() -> list[Case]:
    return (
        _lsq_cases()
        + _minimize_cases()
        + _root_cases()
        + _scalar_cases()
        + _elementwise_cases()
        + _derivative_cases()
        + _linear_cases()
    )
