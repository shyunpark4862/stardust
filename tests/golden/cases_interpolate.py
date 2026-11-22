"""Golden cases for sd.interpolate (tools/golden.py has the contract; docs/design/interpolate.md §7.5).

sd.interp's cases are the array area's (cases_array.py, milestone IP1). Milestone IP2 brought the piecewise
polynomials and the cubic family: PPoly (evaluation with derivatives, the three outside rules, derivative,
antiderivative, integrate, roots, solve, extend; decreasing breakpoints, an axis, complex coefficients),
CubicHermiteSpline, CubicSpline (every bc_type, the 2- and 3-node special cases, periodic, batches, complex),
PchipInterpolator, pchip_interpolate and Akima1DInterpolator (akima, makima), on edge data of
tests/interpolate/samples.py's kinds rebuilt from fixed literals and L.rng(seed); objects large enough for the
parallel paths at 3 and 10 threads (evaluation, the coefficients, the antiderivative's blocked prefix sums, the
pairwise integral, the roots); and the wrong calls (their errors are results). A case returns the coefficients,
the breakpoints and the values it computed (not the object, whose methods the encoder would call recursively).
Milestone IP3 brought the B-splines: make_interp_spline (k = 0 .. 5, every bc_type, periodic with n <= k and
n > k, knots given, the orders that pivot, batches, axes, complex values), BSpline (evaluation with derivatives
and the three outside rules, derivative, antiderivative, integrate, insert_knot, basis_element, design_matrix,
from_power_basis, k = 1 through the linear path) and PPoly.from_spline, with large cases for the parallel paths.
"""

import math

NAN, INF = math.nan, math.inf


def _parts(p):
    return (p.c, p.x, p.extrapolate, p.axis)


def _bparts(b):
    return (b.t, b.c, b.k, b.extrapolate, b.axis)


def _rules(c, q):
    return (c(q), c(q, nu=2), c(q, extrapolate="periodic"), c.integrate(-0.5, 2.3), c.integrate(0.9, 0.1))


def _bspline_cases(L, curves, q):
    np = L.np
    IP = L.sd.interpolate
    for name in ("random", "uniform", "spacing", "large", "tiny", "cancel", "two", "three"):
        x, y = curves[name]
        qq = x[0] + (x[-1] - x[0]) * q
        for k in range(6):
            yield (
                f"make_interp_spline {name} k={k}",
                lambda x=x, y=y, k=k, qq=qq: (lambda b: _bparts(b) + (b(qq), b(qq, nu=1), b(qq, nu=k + 1),
                                                                       b(qq, extrapolate=False)))(
                    IP.make_interp_spline(x, y, k=k)),
            )  # fmt: skip
        for bc in ("natural", "clamped"):
            yield (
                f"make_interp_spline {name} {bc}",
                lambda x=x, y=y, bc=bc, qq=qq: (lambda b: _bparts(b) + (b(qq), b(qq, nu=2)))(
                    IP.make_interp_spline(x, y, bc_type=bc)
                ),
            )
    x, y = curves["random"]
    yp = y.copy()
    yp[-1] = yp[0]
    for k in (2, 3, 4, 5):
        yield (
            f"make_interp_spline periodic k={k}",
            lambda k=k: (lambda b: _bparts(b) + (b(q * 3 - 1),))(IP.make_interp_spline(x, yp, k=k, bc_type="periodic")),
        )
    yield (
        "make_interp_spline periodic small",
        lambda: [_bparts(IP.make_interp_spline([0.0, 1.0, 2.5, 3.0], [0.0, 1.0, 2.0, 0.0], k=k, bc_type="periodic"))
                 for k in (3, 4, 5, 6)],
    )  # fmt: skip
    yield "make_interp_spline periodic k=1", lambda: _bparts(IP.make_interp_spline(x, yp, k=1, bc_type="periodic"))
    yield (
        "make_interp_spline orders",
        lambda: [_bparts(IP.make_interp_spline(x, y, k=k, bc_type=bc)) for k, bc in (
            (3, ([(1, 0.5)], [(2, -1.0)])), (5, ([(1, 0.0), (2, 1.0)], [(2, 0.0), (1, 3.0)])), (2, ([(1, 1.0)], None)),
            (5, ([(2, 0.5), (3, -1.0)], [(3, 0.2), (2, 0.0)])), (4, ([(2, 1.0)], [(1, 0.0), (2, 0.3)])))],
    )  # fmt: skip
    yield (
        "make_interp_spline knots given",
        lambda: (
            _bparts(IP.make_interp_spline(x, y, k=3, t=np.r_[(x[0],) * 4, x[2:-2], (x[-1],) * 4])),
            _bparts(
                IP.make_interp_spline(
                    x, y, k=3, t=np.r_[(x[0],) * 4, x[2:-1], (x[-1],) * 4], bc_type=([(1, 0.0)], None)
                )
            ),
        ),  # fmt: skip
    )
    yb = L.rng(3107).standard_normal((2, 12, 3))
    yield (
        "make_interp_spline batch axis",
        lambda: (lambda b: _bparts(b) + (b(q.reshape(-1, 1)[:6]), b.integrate(0.2, 0.7)))(
            IP.make_interp_spline(x, yb, axis=1, bc_type=([(1, np.ones((2, 3)))], "natural"))
        ),
    )
    yield (
        "make_interp_spline complex",
        lambda: (lambda b: _bparts(b) + (b(q),))(IP.make_interp_spline(x, y + 1j * y[::-1], bc_type="clamped")),
    )
    yield "make_interp_spline int data", lambda: _bparts(IP.make_interp_spline([0, 1, 2, 4, 5], [1, 0, 3, 3, 2]))
    b = IP.make_interp_spline(x, y)
    for ext in (True, False, "periodic"):
        yield (
            f"BSpline extrapolate {ext}",
            lambda ext=ext: _rules(IP.BSpline(b.t, b.c, 3, extrapolate=ext), q * 3 - 1),
        )
    yield (
        "BSpline calculus",
        lambda: tuple(
            _bparts(s) for s in (b.derivative(), b.derivative(nu=3), b.antiderivative(nu=2), b.derivative(nu=-1))
        ),
    )
    periodic = IP.make_interp_spline(x, yp, bc_type="periodic")
    yield (
        "BSpline insert_knot",
        lambda: tuple(
            _bparts(s) for s in (b.insert_knot(0.5), b.insert_knot(float(x[3]), m=3), periodic.insert_knot(0.4))
        ),
    )
    elements = ([0, 1, 2, 3, 4], [0, 1, 1, 2], [0.0, 2.0])
    yield (
        "BSpline basis_element",
        lambda: [(_bparts(e), e(q * 6 - 1)) for e in (IP.BSpline.basis_element(t) for t in elements)],
    )

    def design():
        dm = IP.BSpline.design_matrix
        mats = (
            dm(x, b.t, 3),
            dm(q[:41] * 3 - 1, b.t, 3, extrapolate=True),
            dm(q[:41] * 3 - 1, b.t, 3, extrapolate="periodic"),
        )
        return [(m.data, m.indices, m.indptr, m.shape) for m in mats]

    yield "BSpline design_matrix", design
    yield (
        "BSpline from_power_basis",
        lambda: [
            _bparts(IP.BSpline.from_power_basis(IP.CubicSpline(x, yy, bc_type=bc), bc_type=bc))
            for bc, yy in (("not-a-knot", y), ("natural", y), ("clamped", y), ("periodic", yp))
        ],
    )
    uniform = IP.BSpline(np.arange(8.0), [1.0, 2.0, 3.0, 4.0], 3, extrapolate="periodic")
    splines = (b, IP.make_interp_spline(x, y, k=0), uniform, (b.t, b.c, 3))
    yield "PPoly from_spline", lambda: [_parts(IP.PPoly.from_spline(s)) for s in splines]
    yield "BSpline k=1", lambda: _rules(IP.make_interp_spline(x, y, k=1), q)
    # large: the parallel paths (evaluation, the collocation fill, the column solves)
    n = 40000
    bigx = np.sort(L.rng(3108).random(n))
    bigy = L.rng(3109).standard_normal(n)
    bigq = L.rng(3110).random(150000) * 1.2 - 0.1

    def large():
        c = IP.make_interp_spline(bigx, bigy)
        return c.c, c(bigq), c(bigq, nu=2), c.antiderivative().c, c.integrate(0.0, 1.0)

    yield "large make_interp_spline", large
    yield (
        "large make_interp_spline k=5 natural",
        lambda: (IP.make_interp_spline(bigx, bigy, k=5).c, IP.make_interp_spline(bigx, bigy, bc_type="natural").c),
    )
    yb2 = L.rng(3111).standard_normal((300, 400))
    yield "large make_interp_spline batch", lambda: IP.make_interp_spline(np.arange(300.0), yb2).c
    # wrong calls
    yield "error bspline k", lambda: IP.make_interp_spline([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], k=3)
    yield "error bspline bc", lambda: IP.make_interp_spline([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 1.0], bc_type="foo")
    yield (
        "error bspline singular",
        lambda: IP.make_interp_spline([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [0.0] * 6, t=[0.0] * 4 + [0.1, 0.2] + [5.0] * 4),
    )
    yield "error bspline knots", lambda: IP.BSpline([0.0, 1.0, 0.5, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0] * 4, 3)
    yield (
        "error bspline derivative",
        lambda: IP.make_interp_spline([0.0, 1.0, 2.0, 3.0], [1.0, 0.0, 2.0, 1.0]).derivative(nu=4),
    )


def _curves(L):
    np = L.np
    r = L.rng(3101)
    x = np.sort(r.random(12))
    out = {
        "random": (x, r.standard_normal(12)),
        "uniform": (np.arange(9) / 8, np.sin(np.arange(9.0))),
        "spacing": (
            np.concatenate([[0.0], np.cumsum(np.where(np.arange(9) % 2, 1e-10, 1.0))]),
            np.cos(np.arange(10.0)),
        ),
        "monotone": (x, np.concatenate([[0.0], np.cumsum(np.where(r.random(11) < 0.3, 0.0, r.exponential(1.0, 11)))])),
        "flat": (np.arange(14) / 13, np.array([1.0, 1, 1, 2, 3, 3, 3, 3, 2.5, 0, 0, 0, 1, 1])),
        "large": (x, 1e300 * r.standard_normal(12)),
        "tiny": (x, 1e-300 * r.standard_normal(12)),
        "cancel": (np.arange(16) / 15, 1e15 + r.standard_normal(16)),
        "two": (np.array([0.0, 1.0]), np.array([2.0, 7.0])),
        "three": (np.array([0.0, 1.0, 3.0]), np.array([0.0, 1.0, 0.0])),
    }
    return out


def cases(L):
    np = L.np
    IP = L.sd.interpolate
    r = L.rng(3102)
    q = np.concatenate([np.linspace(-0.3, 1.3, 41), [NAN, INF, -INF, 0.0, 1.0, -0.0]])
    curves = _curves(L)
    for name, (x, y) in curves.items():
        qq = x[0] + (x[-1] - x[0]) * q
        for bc in ("not-a-knot", "natural", "clamped"):
            yield (
                f"CubicSpline {name} {bc}",
                lambda x=x, y=y, bc=bc, qq=qq: (lambda p: _parts(p) + (p(qq), p(qq, nu=1), p(qq, nu=3), p(qq, nu=4)))(
                    IP.CubicSpline(x, y, bc_type=bc)
                ),
            )
        yield (
            f"PchipInterpolator {name}",
            lambda x=x, y=y, qq=qq: (lambda p: _parts(p) + (p(qq), p(qq, nu=2)))(IP.PchipInterpolator(x, y)),
        )
        for meth in ("akima", "makima"):
            yield (
                f"Akima1DInterpolator {name} {meth}",
                lambda x=x, y=y, qq=qq, m=meth: (lambda p: _parts(p) + (p(qq), p(qq, extrapolate=True)))(
                    IP.Akima1DInterpolator(x, y, method=m)
                ),
            )
    # boundary conditions: tuples, mixed ends, periodic, complex values
    x, y = curves["random"]
    yp = y.copy()
    yp[-1] = yp[0]
    yield (
        "CubicSpline periodic",
        lambda: (lambda p: _parts(p) + (p(q * 3 - 1), p(q, extrapolate=True)))(
            IP.CubicSpline(x, yp, bc_type="periodic")
        ),
    )
    yield (
        "CubicSpline periodic three",
        lambda: _parts(IP.CubicSpline([0.0, 1.0, 3.0], [0.0, 1.0, 0.0], bc_type="periodic")),
    )
    yield (
        "CubicSpline periodic tolerance",
        lambda: IP.CubicSpline(x, np.append(yp[:-1], yp[0] + 1e-14), bc_type="periodic"),
    )
    yield "CubicSpline tuple bc", lambda: _parts(IP.CubicSpline(x, y, bc_type=((1, 0.5), (2, -3.0))))
    yield "CubicSpline mixed bc", lambda: _parts(IP.CubicSpline(x, y, bc_type=("natural", "not-a-knot")))
    yield "CubicSpline complex", lambda: (lambda p: _parts(p) + (p(q),))(IP.CubicSpline(x, y + 1j * y[::-1]))
    yield "CubicSpline complex bc", lambda: _parts(IP.CubicSpline(x, y, bc_type=((1, 2j), "clamped")))
    yb = r.standard_normal((2, 12, 3))
    yield (
        "CubicSpline batch axis",
        lambda: (lambda p: _parts(p) + (p(q.reshape(-1, 1)[:6]), p.integrate(0.2, 0.7)))(
            IP.CubicSpline(x, yb, axis=1, bc_type=((1, np.ones((2, 3))), "natural"))
        ),
    )
    yield "CubicSpline int data", lambda: _parts(IP.CubicSpline([0, 1, 2, 4], [1, 0, 3, 3]))
    yield (
        "CubicSpline overflow",
        lambda: IP.CubicSpline(
            [0.0, 1.0, 2.0, 2.00001, 3.0, 4.0], [0.0, 1e300, 0.0, 1e300, 0.0, 1e300], bc_type="natural"
        ),
    )
    yield "CubicSpline subnormal spacing", lambda: IP.CubicSpline(np.arange(8) * 5e-324 * 7, r.standard_normal(8))
    yield "CubicHermiteSpline", lambda: (lambda p: _parts(p) + (p(q),))(IP.CubicHermiteSpline(x, y, y[::-1]))
    yield "CubicHermiteSpline complex", lambda: _parts(IP.CubicHermiteSpline(x, y, 1j * y))
    yield "pchip_interpolate", lambda: IP.pchip_interpolate(x, y, q)
    yield "pchip_interpolate der list", lambda: IP.pchip_interpolate(x, y, q, der=[0, 1, 3])
    yield "pchip_interpolate scalar", lambda: IP.pchip_interpolate(x, y, 0.5, der=1)
    # PPoly: rules, decreasing breakpoints, calculus, roots
    c = r.standard_normal((4, 6, 2))
    bx = np.cumsum(r.uniform(0.5, 1.5, 7))
    qq = np.concatenate([np.linspace(bx[0] - 1, bx[-1] + 1, 31), bx, [NAN]])
    for ext in (True, False, "periodic"):
        yield f"PPoly extrapolate {ext}", lambda ext=ext: (
            lambda p: (p(qq), p(qq, nu=2), p.integrate(-0.5, 12.3), p.integrate(4.0, 1.0)))(
            IP.PPoly(c, bx, extrapolate=ext))  # fmt: skip
    yield (
        "PPoly decreasing",
        lambda: (lambda p: (p(qq), p.integrate(1.0, 5.0), p.antiderivative().c))(IP.PPoly(c, bx[::-1].copy())),
    )
    yield (
        "PPoly axis",
        lambda: (lambda p: _parts(p) + (p(qq[:5].reshape(5, 1)),))(
            IP.PPoly(r.standard_normal((2, 4, 6, 3)), bx, axis=1)
        ),
    )
    yield (
        "PPoly complex",
        lambda: (lambda p: (p(qq), p.antiderivative(nu=2).c, p.integrate(0.0, 9.0)))(IP.PPoly(c + 1j * c[::-1], bx)),
    )
    yield "PPoly derivatives", lambda: (
        lambda p: (p.derivative().c, p.derivative(nu=3).c, p.derivative(nu=9).c, p.derivative(nu=-1).c,
                   p.antiderivative(nu=3).c))(IP.PPoly(c, bx))  # fmt: skip
    yield "PPoly roots", lambda: (lambda p: (p.roots(), p.roots(discontinuity=False), p.roots(extrapolate=False),
                                            p.solve(0.5)))(IP.PPoly(c[:, :, 0], bx))  # fmt: skip
    yield "PPoly roots batch", lambda: IP.PPoly(c, bx).roots()
    tangent = np.column_stack([np.convolve(np.convolve([1, -0.25], [1, -0.25]), [1, 1.0]),
                               -np.convolve(np.convolve([1, -0.5], [1, -0.5]), [1, -2.0])])  # fmt: skip
    yield "PPoly double roots", lambda: IP.PPoly(tangent, [0.0, 1.0, 2.0]).roots()
    close = np.convolve(np.convolve([1, -0.5], [1, -(0.5 + 2.0**-30)]), [1, -3.0]).reshape(4, 1)
    yield "PPoly close roots", lambda: IP.PPoly(close, [0.0, 1.0]).roots(extrapolate=False)
    yield "PPoly zero piece", lambda: IP.PPoly(np.column_stack([[1.0, -0.5], [0.0, 0.0], [1.0, -2.25]]),
                                               [0.0, 1.0, 2.0, 3.0]).roots()  # fmt: skip

    def extend():
        p = IP.PPoly([[1.0, 2.0, 3.0]], [0.0, 1.0, 2.0, 3.0])
        p.extend([[5.0, 6.0]], [4.0, 5.0])
        p.extend([[7.0], [-1.0]], [-1.0])
        return _parts(p)

    yield "PPoly extend", extend
    yield from _bspline_cases(L, curves, q)
    # large: the parallel paths (evaluation, coefficients, prefix sums over several blocks, pairwise sums, roots)
    n = 40000
    bigx = np.sort(L.rng(3103).random(n))
    bigy = L.rng(3104).standard_normal(n)
    bigq = L.rng(3105).random(150000) * 1.2 - 0.1
    yield "large CubicSpline", lambda: (lambda p: (p.c, p(bigq), p.antiderivative(nu=2).c, p.integrate(0.0, 1.0),
                                                  p.roots()))(IP.CubicSpline(bigx, bigy))  # fmt: skip
    yield "large PchipInterpolator", lambda: (lambda p: (p.c, p(bigq, nu=1)))(IP.PchipInterpolator(bigx, bigy))
    yield "large Akima1DInterpolator", lambda: IP.Akima1DInterpolator(bigx, bigy, method="makima").c
    yb2 = L.rng(3106).standard_normal((300, 400))
    yield "large CubicSpline batch", lambda: IP.CubicSpline(bigx[:300] * 0 + np.arange(300.0), yb2).c
    # wrong calls
    yield "error unsorted", lambda: IP.CubicSpline([0.0, 2.0, 1.0], [0.0, 1.0, 2.0])
    yield "error nan y", lambda: IP.PchipInterpolator([0.0, 1.0, 2.0], [0.0, NAN, 2.0])
    yield "error complex x", lambda: IP.CubicSpline([0.0, 1j], [0.0, 1.0])
    yield "error complex pchip", lambda: IP.PchipInterpolator([0.0, 1.0], [0.0, 1j])
    yield "error bc_type", lambda: IP.CubicSpline([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], bc_type="Natural")
    yield "error periodic", lambda: IP.CubicSpline([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], bc_type="periodic")
    yield "error method", lambda: IP.Akima1DInterpolator([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], method="Akima")
    yield "error extrapolate", lambda: IP.PPoly([[1.0]], [0.0, 1.0], extrapolate=1)
    yield "error nu", lambda: IP.PPoly([[1.0]], [0.0, 1.0])(0.5, nu=-1)
    yield "error positional option", lambda: IP.CubicSpline([0.0, 1.0], [0.0, 1.0], 0)
    yield "error roots complex", lambda: IP.PPoly([[1j]], [0.0, 1.0]).roots()
    yield "error extend", lambda: IP.PPoly([[1.0]], [0.0, 1.0]).extend([[1.0]], [0.5])
