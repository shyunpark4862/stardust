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
"""

import math

NAN, INF = math.nan, math.inf


def _parts(p):
    return (p.c, p.x, p.extrapolate, p.axis)


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
