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
Milestone IP4 brought the polynomial and rational interpolants: BarycentricInterpolator (values, derivatives, the
weights, wi given, add_xi, set_yi, complex values, batches and axes), KroghInterpolator (Hermite data, derivatives),
barycentric_interpolate, krogh_interpolate, FloaterHormannInterpolator (d = 0 .. 3, complex points, poles, zeros,
residues, samples dropped) and AAA (the sample sets of tests/interpolate/samples.py, with and without clean_up, a run
that does not converge, the Froissart doublets of scipy's example), with large cases for the parallel paths. A case
returns arrays and attributes, not the objects (the encoder would call clean_up()).
Milestone IP5 brought the regular grids: RegularGridInterpolator (the six methods on 1-, 2- and 3-D grids with
random spacing, a decreasing axis, an axis of one node, a batch, complex values; the outside rules, NaN points, nu,
a method other than the object's, the shapes of xi) and interpn, with large cases for the parallel paths (evaluation,
the coefficients' solves).
Milestone IP6 brought the scattered-data interpolants: RBFInterpolator (every kernel on uniform, clustered and lattice
points, smoothing, neighbors with ties, degree -1 and above the minimum, complex values with a batch, one point, a NaN
point) and NearestNDInterpolator (rescale, ties, a tuple of coordinates), with large cases for the parallel paths.
Milestone IP9 brought the interpolants on sd.spatial's Delaunay triangulation: LinearNDInterpolator (2-D and 3-D;
uniform, grid, cocircular and duplicated points, points on edges and vertices, outside, NaN and inf points, fill_value,
complex values with a batch, rescale, a Delaunay as the points, a tuple of coordinate arrays, values that are not
finite), CloughTocher2DInterpolator (the same with the gradients, converged and niter, tol and maxiter, points far from
the origin and near the largest doubles) and griddata (its three methods, 1-D and 5-D nearest), with large cases for
the parallel paths, and the wrong calls.
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


def _rparts(r):
    return (r.support_points, r.support_values, r.weights, r.errors, r.converged)


def _polyint_cases(L, curves, q):
    np = L.np
    IP = L.sd.interpolate
    for name in ("random", "uniform", "spacing", "large", "tiny", "cancel", "two", "three"):
        x, y = curves[name]
        qq = x[0] + (x[-1] - x[0]) * q
        yield (
            f"BarycentricInterpolator {name}",
            lambda x=x, y=y, qq=qq: (lambda b: (b.wi, b(qq), b.derivative(qq), b.derivative(qq, der=2),
                                                b.derivatives(qq[:7], der=4)))(IP.BarycentricInterpolator(x, y)),
        )  # fmt: skip
        yield (
            f"KroghInterpolator {name}",
            lambda x=x, y=y, qq=qq: (lambda k: (k(qq), k.derivative(qq, der=2), k.derivatives(qq[:7])))(
                IP.KroghInterpolator(x, y)),
        )  # fmt: skip
        for d in (0, 1, 3):
            if d < x.size:
                yield (
                    f"FloaterHormannInterpolator {name} d={d}",
                    lambda x=x, y=y, qq=qq, d=d: (lambda f: (f.weights, f(qq), f(qq[:9] + 0.25j)))(
                        IP.FloaterHormannInterpolator(x, y, d=d)),
                )  # fmt: skip
    x, y = curves["random"]
    yb = L.rng(3110).standard_normal((2, 12, 3))
    yield "BarycentricInterpolator complex", lambda: IP.BarycentricInterpolator(x, y + 1j * y[::-1])(q)
    yield (
        "BarycentricInterpolator batch axis",
        lambda: (lambda b: (b(q.reshape(-1, 1)[:6]), b.derivatives(0.4, der=3), b.yi))(
            IP.BarycentricInterpolator(x, yb, axis=1)),
    )  # fmt: skip
    yield "BarycentricInterpolator wi", lambda: IP.BarycentricInterpolator(x, y, wi=np.ones(12))(q)
    yield "BarycentricInterpolator one node", lambda: (lambda b: (b.wi, b(q), b.derivative(q)))(
        IP.BarycentricInterpolator([2.0], [5.0]))  # fmt: skip

    def changed():
        b = IP.BarycentricInterpolator(x[:6], y[:6])
        b.add_xi(x[6:], y[6:])
        first = (b.xi, b.wi, b(q))
        b.set_yi(2 * y)
        e = IP.BarycentricInterpolator(x)
        e.add_xi([7.5])
        return first + (b(q), b.derivative(q), e.xi, e.wi, e.yi)

    yield "BarycentricInterpolator add_xi set_yi", changed
    xh = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 2.0])
    yh = np.array([1.0, 2.0, 0.0, -1.0, 3.0, 1.0])
    yield "KroghInterpolator hermite", lambda: (lambda k: (k(q * 3 - 0.5), k.derivatives(q[:5] * 2, der=8)))(
        IP.KroghInterpolator(xh, yh))  # fmt: skip
    yield "KroghInterpolator complex batch", lambda: IP.KroghInterpolator(xh, np.stack([yh, 1j * yh], axis=1),
                                                                          axis=0)(q)  # fmt: skip
    yield "KroghInterpolator axis", lambda: IP.KroghInterpolator(x, yb, axis=1).derivative(q[:4], der=1)
    yield "barycentric_interpolate", lambda: (IP.barycentric_interpolate(x, y, q),
                                              IP.barycentric_interpolate(x, y, 0.5, der=[0, 1, 2]))  # fmt: skip
    yield "krogh_interpolate", lambda: (IP.krogh_interpolate(xh, yh, q), IP.krogh_interpolate(xh, yh, 0.5, der=3))
    yield "FloaterHormannInterpolator rational parts", lambda: (lambda f: (f.poles(), f.roots(), f.residues()))(
        IP.FloaterHormannInterpolator(np.linspace(-5, 5, 15), 1 / (1 + np.linspace(-5, 5, 15) ** 2)))  # fmt: skip
    yield "FloaterHormannInterpolator batch", lambda: IP.FloaterHormannInterpolator(x, yb, axis=1, d=2)(q[:5])
    yield "FloaterHormannInterpolator dropped", lambda: IP.FloaterHormannInterpolator(
        x, np.where(np.arange(12) == 4, NAN, y))(q)  # fmt: skip
    # AAA on samples rebuilt from samples.py's formulas
    zl = np.linspace(-1, 1, 1000)
    zc = np.exp(2j * np.pi * np.arange(256) / 256)
    zn = np.linspace(0, 1, 200)
    fn = 1 / (zn - 0.5 + 1e-3)
    fn[[10, 50]] = [NAN, INF]
    sets = {"tan": (zl, np.tan(np.pi * zl / 2)), "abs": (zl, np.abs(zl)), "circle": (zc, np.exp(zc) / (zc - 1.5)),
            "nonfinite": (zn, fn)}  # fmt: skip
    for name, (z, f) in sets.items():
        for clean in (True, False):
            yield (
                f"AAA {name} clean_up={clean}",
                lambda z=z, f=f, clean=clean: (lambda r: _rparts(r) + (r(q * 2 - 1), r(q[:9] + 0.5j), r.poles(),
                                                                       r.roots()))(IP.AAA(z, f, clean_up=clean)),
            )  # fmt: skip
    yield "AAA residues", lambda: IP.AAA(zc, np.exp(zc) / (zc - 1.5)).residues()
    yield "AAA max_terms", lambda: _rparts(IP.AAA(zl, np.abs(zl), max_terms=6))
    yield "AAA rtol", lambda: _rparts(IP.AAA(zl, np.abs(zl), rtol=1e-6))

    def froissart():
        z = np.exp(1j * 2 * np.pi * np.linspace(0, 1, num=1000))
        r = IP.AAA(z, np.log(2 + z**4) / (1 - 16 * z**4), rtol=0, max_terms=30, clean_up=False)
        return (r.clean_up(), r.clean_up(clean_up_tol=1e-8)) + _rparts(r)

    yield "AAA clean_up", froissart
    yield "AAA tiny", lambda: _rparts(IP.AAA([0.5], [2.0])) + (IP.AAA([0.5], [2.0])(3.0),)
    # large: the parallel paths
    xl = np.cos(np.pi * np.arange(3001) / 3000)
    ql = L.rng(3111).uniform(-1.1, 1.1, 150000)
    yield "large BarycentricInterpolator", lambda: (lambda b: (b.wi, b(ql), b.derivative(ql[:5000])))(
        IP.BarycentricInterpolator(xl, np.exp(xl)))  # fmt: skip
    yield "large KroghInterpolator", lambda: IP.KroghInterpolator(xl[::150], np.exp(xl[::150])).derivatives(ql, der=3)
    yield "large FloaterHormannInterpolator", lambda: (lambda f: (f.weights, f(ql)))(
        IP.FloaterHormannInterpolator(xl, np.sin(5 * xl)))  # fmt: skip
    zz = np.exp(2j * np.pi * np.arange(2000) / 2000)
    yield "large AAA", lambda: (lambda r: _rparts(r) + (r(ql * (0.5 + 0.3j)),))(IP.AAA(zz, np.exp(zz) / (zz - 1.2)))
    # wrong calls
    yield "error barycentric repeated", lambda: IP.BarycentricInterpolator([0.0, 1.0, 0.0], [1.0, 2.0, 3.0])
    yield "error krogh repeated", lambda: IP.KroghInterpolator([0.0, 1.0, 0.0], [1.0, 2.0, 3.0])
    yield "error krogh complex x", lambda: IP.KroghInterpolator([0.0, 1.0], [1.0, 2.0])(0.5j)
    yield "error aaa no finite", lambda: IP.AAA([0.0, 1.0], [NAN, INF])
    yield "error aaa rtol", lambda: IP.AAA([0.0, 1.0], [1.0, 2.0], rtol=-1.0)
    yield "error floater d", lambda: IP.FloaterHormannInterpolator([0.0, 1.0], [1.0, 2.0], d=2)
    yield "error floater residues batch", lambda: IP.FloaterHormannInterpolator([0.0, 1.0, 2.0], [[1.0, 2.0]] * 3,
                                                                                d=1).residues()  # fmt: skip


def _rbf_cases(L):
    np = L.np
    IP = L.sd.interpolate
    r = L.rng(3120)
    a = np.arange(5) / 4
    pts = {
        "uniform2d": r.random((40, 2)),
        "uniform3d": r.random((30, 3)),
        "cluster2d": np.array([[0.2, 0.2], [0.8, 0.3], [0.5, 0.9]])[r.integers(0, 3, 40)] + r.normal(0, 0.03, (40, 2)),
        "lattice2d": np.stack(np.meshgrid(a, a, indexing="ij"), axis=-1).reshape(-1, 2),
    }
    kernels = ("linear", "thin_plate_spline", "cubic", "quintic", "multiquadric", "inverse_multiquadric",
               "inverse_quadratic", "gaussian")  # fmt: skip
    for name, y in pts.items():
        d = np.sin(3 * y[:, 0]) + np.cos(2 * y[:, -1])
        x = np.vstack([y.min(0) + (y.max(0) - y.min(0)) * r.random((12, y.shape[1])), y[:2],
                       np.full((1, y.shape[1]), NAN)])  # fmt: skip
        for kern in kernels:
            eps = None if kern in kernels[:4] else 3.0
            yield f"RBFInterpolator {name} {kern}", lambda y=y, d=d, x=x, k=kern, e=eps: IP.RBFInterpolator(
                y, d, kernel=k, epsilon=e)(x)  # fmt: skip
        yield f"RBFInterpolator {name} smoothing", lambda y=y, d=d, x=x: IP.RBFInterpolator(
            y, d, smoothing=np.linspace(0, 0.1, len(y)))(x)  # fmt: skip
        yield f"RBFInterpolator {name} neighbors", lambda y=y, d=d, x=x: (
            IP.RBFInterpolator(y, d, neighbors=7)(x),
            IP.RBFInterpolator(y, d, neighbors=3, kernel="linear")(x))  # fmt: skip
        yield f"NearestNDInterpolator {name}", lambda y=y, d=d, x=x: (
            IP.NearestNDInterpolator(y, d)(x), IP.NearestNDInterpolator(y, d, rescale=True)(x))  # fmt: skip
    y, x = pts["uniform2d"], pts["uniform2d"][:5] + 0.01
    dc = np.stack([y[:, 0], y[:, 1] * 1j], axis=1).reshape(40, 2, 1)
    yield "RBFInterpolator complex batch", lambda: IP.RBFInterpolator(y, dc, kernel="cubic")(x.reshape(5, 1, 2))
    yield "RBFInterpolator degree", lambda: (IP.RBFInterpolator(y, y[:, 0], kernel="gaussian", epsilon=2.0,
                                                                degree=-1)(x),
                                             IP.RBFInterpolator(y, y[:, 0], degree=3)(x))  # fmt: skip
    yield "RBFInterpolator one point", lambda: IP.RBFInterpolator([[0.5, 0.5]], [2.0], kernel="linear")([0.0, 1.0])
    yield "RBFInterpolator neighbors ties", lambda: IP.RBFInterpolator(pts["lattice2d"], pts["lattice2d"][:, 1],
                                                                       neighbors=2, kernel="linear")(
        [[0.125, 0.125], [0.375, 0.5], [0.5, 0.625]])  # fmt: skip
    yield "NearestNDInterpolator ties tuple", lambda: IP.NearestNDInterpolator(pts["lattice2d"], np.arange(25.0))(
        (np.array([0.125, 0.375, 0.5, 1.0]), np.array([0.125, 0.625, 0.375, 0.875])))  # fmt: skip
    # large: the parallel paths (the system, the evaluation, the local systems, the nearest points)
    yl = L.rng(3121).random((600, 2))
    xl = L.rng(3122).random((20000, 2))
    yield "large RBFInterpolator", lambda: IP.RBFInterpolator(yl, np.sin(5 * yl[:, 0]) * yl[:, 1])(xl)
    yb = L.rng(3123).random((3000, 3))
    yield "large RBFInterpolator neighbors", lambda: IP.RBFInterpolator(yb, np.cos(yb.sum(1)), neighbors=20)(
        L.rng(3124).random((4000, 3)))  # fmt: skip
    yield "large NearestNDInterpolator", lambda: IP.NearestNDInterpolator(yb, np.cos(yb.sum(1)))(
        L.rng(3125).random((100000, 3)))  # fmt: skip
    # wrong calls
    yd = pts["uniform2d"][:10].copy()
    yd[6] = yd[3]
    t = np.linspace(0, 1, 10)
    yield "error rbf duplicate", lambda: IP.RBFInterpolator(yd, yd[:, 0])
    yield "error rbf collinear", lambda: IP.RBFInterpolator(np.column_stack([t, 0.5 * t + 0.25]), t)
    yield "error rbf smoothing", lambda: IP.RBFInterpolator(yd[:5], t[:5], smoothing=-1.0)
    yield "error rbf degree", lambda: IP.RBFInterpolator(yd[:5], t[:5], degree=-1)
    yield "error rbf epsilon", lambda: IP.RBFInterpolator(yd[:5], t[:5], kernel="gaussian")
    yield "error rbf kernel", lambda: IP.RBFInterpolator(yd[:5], t[:5], kernel="Gaussian", epsilon=1.0)
    yield "error rbf points", lambda: IP.RBFInterpolator(yd[:5], t[:5])([0.5, 0.5, 0.5])
    yield "error nearest shape", lambda: IP.NearestNDInterpolator(yd[:5], t[:4])


def _scattered_cases(L):
    np = L.np
    IP = L.sd.interpolate
    r = L.rng(3130)
    a = np.arange(6) / 5
    pts = {
        "uniform2d": r.random((50, 2)),
        "uniform3d": r.random((40, 3)),
        "grid2d": np.stack(np.meshgrid(a, a, indexing="ij"), axis=-1).reshape(-1, 2),
        "circle2d": np.array([[2.0, 1.0], [1.0, 2.0], [-1.0, 2.0], [-2.0, 1.0], [-2.0, -1.0], [-1.0, -2.0], [1.0, -2.0],
                              [2.0, -1.0], [0.0, 0.0]]),
        "cube3d": np.array([[i, j, k] for i in (0.0, 1.0) for j in (0.0, 1.0) for k in (0.0, 1.0)] + [[0.5, 0.5, 0.5]]),
    }  # fmt: skip
    pts["duplicate2d"] = np.vstack([pts["uniform2d"][:20], pts["uniform2d"][[3, 7, 3]]])

    def field(y):
        return np.sin(3 * y[:, 0]) + np.cos(2 * y[:, -1]) + 0.1 * y.sum(axis=1)

    for name, y in pts.items():
        d = y.shape[1]
        f = field(y)
        lo, hi = y.min(0), y.max(0)
        x = np.vstack([lo + (hi - lo) * r.random((15, d)), y[:3], (y[0] + y[1]) / 2, hi + 0.5,
                       np.full((1, d), NAN), np.where(np.arange(d) == 0, INF, 0.5)[None, :]])  # fmt: skip
        yield f"LinearNDInterpolator {name}", lambda y=y, f=f, x=x: (
            IP.LinearNDInterpolator(y, f)(x), IP.LinearNDInterpolator(y, f, fill_value=-2.5)(x))  # fmt: skip
        if d == 2:
            yield f"CloughTocher2DInterpolator {name}", lambda y=y, f=f, x=x: (
                lambda c: (c.grad, c.niter, c.converged, c(x), IP.CloughTocher2DInterpolator(y, f, fill_value=0.0)(x)))(
                IP.CloughTocher2DInterpolator(y, f))  # fmt: skip
    y, x = pts["uniform2d"], r.random((6, 2))
    f = field(y)
    fb = np.stack([f, f * 1j, f**2], axis=1).reshape(50, 3, 1)
    yield "LinearNDInterpolator complex batch", lambda: IP.LinearNDInterpolator(y, fb, fill_value=1 + 2j)(
        x.reshape(3, 2, 2))  # fmt: skip
    yield "CloughTocher2DInterpolator complex batch", lambda: (lambda c: (c.grad, c(x.reshape(3, 2, 2))))(
        IP.CloughTocher2DInterpolator(y, fb))  # fmt: skip
    yield "LinearNDInterpolator rescale tuple", lambda: (
        IP.LinearNDInterpolator(y * [1e6, 1e-3], f, rescale=True)(x * [1e6, 1e-3]),
        IP.LinearNDInterpolator((y[:, 0], y[:, 1]), f)((x[:, 0], x[:, 1])),
        IP.LinearNDInterpolator(L.sd.spatial.Delaunay(y), f)([0.5, 0.5]))  # fmt: skip
    fn = np.where(np.arange(50) % 7 == 0, NAN, np.where(np.arange(50) % 11 == 0, INF, f))
    yield "LinearNDInterpolator nonfinite values", lambda: IP.LinearNDInterpolator(y, fn)(np.vstack([y, x]))
    CT = IP.CloughTocher2DInterpolator
    yield (
        "CloughTocher2DInterpolator options",
        lambda: [
            (c.grad, c.niter, c.converged, c(x))
            for c in (
                CT(y, f, maxiter=1),
                CT(y, f, tol=0.0, maxiter=7),
                CT(y, f, tol=1e-12, rescale=True),
                CT(y, 1e6 * f),
            )
        ],
    )
    yield "CloughTocher2DInterpolator far and huge", lambda: (
        IP.CloughTocher2DInterpolator(1e8 + y, f)(1e8 + x), IP.CloughTocher2DInterpolator(1e300 * (y - 0.5), f)(
            1e300 * (x - 0.5)), IP.CloughTocher2DInterpolator(1e-200 * y, f).grad)  # fmt: skip
    t = np.linspace(0, 1, 9)
    yield "griddata", lambda: (
        IP.griddata(y, f, (x[:, 0], x[:, 1])), IP.griddata(y, f, x, method="cubic", fill_value=0.0),
        IP.griddata(y, f, x, method="nearest", rescale=True), IP.griddata(t, t**2, [0.1, 0.55], method="nearest"),
        IP.griddata(r.random((20, 5)), t[:4].repeat(5), r.random((3, 5)), method="nearest"),
        IP.griddata(pts["uniform3d"], field(pts["uniform3d"]), pts["uniform3d"][:4] * 0.9))  # fmt: skip
    # large: the parallel paths (the located points by blocks, the gradients' columns)
    yl = L.rng(3131).random((30000, 2))
    xl = L.rng(3132).random((100000, 2))
    fl = np.sin(5 * yl[:, 0]) * yl[:, 1]
    yield "large LinearNDInterpolator", lambda: IP.LinearNDInterpolator(yl, fl)(xl)
    y3 = L.rng(3133).random((10000, 3))
    yield "large LinearNDInterpolator 3d", lambda: IP.LinearNDInterpolator(y3, np.cos(y3.sum(1)))(
        L.rng(3134).random((50000, 3)))  # fmt: skip
    yield "large CloughTocher2DInterpolator", lambda: (lambda c: (c.grad, c.niter, c(xl)))(
        IP.CloughTocher2DInterpolator(yl, np.stack([fl, fl**2, -fl], axis=1)))  # fmt: skip
    # wrong calls
    yd = pts["duplicate2d"]
    yield "error scattered duplicate", lambda: IP.LinearNDInterpolator(yd, np.arange(23.0))
    yield "error scattered collinear", lambda: IP.CloughTocher2DInterpolator([[0, 0], [1, 1], [2, 2]], [1, 2, 3])
    yield "error scattered columns", lambda: IP.CloughTocher2DInterpolator(pts["uniform3d"], pts["uniform3d"][:, 0])
    yield "error scattered fill_value", lambda: IP.LinearNDInterpolator(y, f, fill_value=1j)
    yield "error scattered tol", lambda: IP.CloughTocher2DInterpolator(y, f, tol=-1.0)
    yield "error scattered maxiter", lambda: IP.CloughTocher2DInterpolator(y, f, maxiter=0)
    yield "error scattered values", lambda: IP.CloughTocher2DInterpolator(y, np.where(t[:1] > 0, f, NAN))
    yield "error scattered xi", lambda: IP.LinearNDInterpolator(y, f)([0.5, 0.5, 0.5])
    yield "error griddata 1-D", lambda: IP.griddata(t, t, [0.5])
    yield "error griddata cubic 3-D", lambda: IP.griddata(pts["uniform3d"], pts["uniform3d"][:, 0], [[0.5] * 3],
                                                          method="cubic")  # fmt: skip


def _grid_cases(L):
    np = L.np
    IP = L.sd.interpolate
    RGI = IP.RegularGridInterpolator
    r = L.rng(3140)

    def axis(n, desc=False):
        p = np.sort(np.concatenate([[0.0, 1.0], r.random(n - 2)]))
        return p[::-1].copy() if desc else p

    a1 = (axis(9),)
    a2 = (axis(8), axis(7, desc=True))
    a3 = (axis(6), axis(7), axis(6))
    v1, v2, v3 = r.standard_normal(9), r.standard_normal((8, 7)), r.standard_normal((6, 7, 6))
    q1 = np.concatenate([np.linspace(-0.3, 1.3, 33), [NAN, 0.0, 1.0]]).reshape(-1, 1)
    q2 = np.vstack([r.random((40, 2)) * 1.4 - 0.2, [[NAN, 0.5], [0.0, 1.0], [1.0, 0.0], [0.5, INF]]])
    q3 = np.vstack([r.random((40, 3)) * 1.4 - 0.2, [[0.5, NAN, 0.5]]])
    for meth in ("linear", "nearest", "slinear", "cubic", "quintic", "pchip"):
        for name, pts, v, q in (("1-D", a1, v1, q1), ("2-D", a2, v2, q2), ("3-D", a3, v3, q3)):
            yield (
                f"RegularGridInterpolator {meth} {name}",
                lambda pts=pts, v=v, q=q, m=meth: (
                    RGI(pts, v, method=m, bounds_error=False, fill_value=None)(q),
                    RGI(pts, v, method=m, bounds_error=False)(q),
                    RGI(pts, v, method=m, bounds_error=False, fill_value=-2.5)(q),
                    RGI(pts, v, method=m)(q[np.all((q >= 0) & (q <= 1) | np.isnan(q), axis=-1)]),
                ),
            )
        yield f"error RegularGridInterpolator {meth} bounds", lambda m=meth: RGI(a2, v2, method=m)(q2)
        yield f"interpn {meth}", lambda m=meth: IP.interpn(a2, v2, q2[:20] * 0.5 + 0.25, method=m)
    for meth in ("slinear", "cubic", "quintic"):
        yield (
            f"RegularGridInterpolator {meth} nu",
            lambda m=meth: tuple(RGI(a2, v2, method=m, bounds_error=False, fill_value=None)(q2, nu=nu)
                                 for nu in ((1, 0), (0, 2), (1, 1), (6, 0))),
        )  # fmt: skip
    yb = r.standard_normal((8, 7, 3))
    yield "RegularGridInterpolator batch", lambda: tuple(RGI(a2, yb, method=m)(q2[:30] * 0.5 + 0.25)
                                                         for m in ("linear", "nearest", "cubic", "pchip"))  # fmt: skip
    yield (
        "RegularGridInterpolator complex",
        lambda: tuple(
            RGI(a2, v2 + 1j * v2[::-1], method=m, bounds_error=False, fill_value=1j)(q2) for m in ("linear", "cubic")
        ),
    )
    one = r.standard_normal((1, 7))
    yield "RegularGridInterpolator one node", lambda: (
        RGI((np.array([0.5]), a2[1]), one)([[0.5, 0.3]]),
        RGI((np.array([0.5]), a2[1]), one, method="nearest", bounds_error=False, fill_value=None)([[2.0, 0.3]]),
    )  # fmt: skip
    yield "RegularGridInterpolator attributes", lambda: (
        lambda o: (o.grid, o.values, o.method, o.bounds_error, o.fill_value))(RGI(a2, v2, fill_value=3))  # fmt: skip
    yield "RegularGridInterpolator method change", lambda: RGI(a2, v2)(q2[:10] * 0.5 + 0.25, method="quintic")
    yield (
        "RegularGridInterpolator tuple xi",
        lambda: (
            RGI(a2, v2)((np.array([0.1, 0.2]), 0.5)),
            RGI(a2, v2)((0.25, 0.75)),
            RGI(a2, v2)([0.1, 0.2, 0.3, 0.4]),
        ),
    )
    yield "RegularGridInterpolator nan values", lambda: (lambda w: RGI(a2, w)(q2[:40] * 0.5 + 0.25))(
        np.where(np.arange(56).reshape(8, 7) == 24, NAN, v2))  # fmt: skip
    # large: the parallel paths
    big = (np.sort(L.rng(3141).random(300)), np.sort(L.rng(3142).random(250)))
    vb = L.rng(3143).standard_normal((300, 250))
    qb = L.rng(3144).random((150000, 2)) * 1.2 - 0.1
    for meth in ("linear", "nearest", "cubic", "pchip"):
        yield (
            f"large RegularGridInterpolator {meth}",
            lambda m=meth: RGI(big, vb, method=m, bounds_error=False, fill_value=None)(
                qb[:40000] if m == "pchip" else qb
            ),
        )
    # wrong calls
    yield "error grid method", lambda: RGI(a2, v2, method="Linear")
    yield "error grid points", lambda: RGI((a2[0], a2[1][[0, 2, 1, 3, 4, 5, 6]]), v2)
    yield "error grid values shape", lambda: RGI(a2, v2[:7])
    yield "error grid few points", lambda: RGI((a2[0][:3], a2[1]), v2[:3], method="cubic")
    yield "error grid bounds", lambda: RGI(a2, v2)([[1.5, 0.5]])
    yield "error grid fill", lambda: RGI(a2, v2, fill_value=1j)
    yield "error grid nu", lambda: RGI(a2, v2)([[0.5, 0.5]], nu=(1, 0))
    yield "error grid xi", lambda: RGI(a2, v2)([[0.5, 0.5, 0.5]])
    yield "error grid nan cubic", lambda: RGI(a2, np.where(v2 > 1, NAN, v2), method="cubic")
    yield "error grid complex pchip", lambda: RGI(a2, v2 + 1j, method="pchip")
    yield "error interpn", lambda: IP.interpn(a2, v2, [[0.5, 0.5]], method="splinef2d")


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
    yield from _polyint_cases(L, curves, q)
    yield from _grid_cases(L)
    yield from _rbf_cases(L)
    yield from _scattered_cases(L)
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
