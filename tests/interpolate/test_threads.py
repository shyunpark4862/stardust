"""Results of sd.interp, the interpolate area's linear path and the cubic family do not depend on the thread count
(docs/design/interpolate.md §1.3, §3.9): at 1, 2, 3, 7 and 10 threads the same bytes, on inputs large enough to split
into many tasks (random and sorted points, a period, complex values, a batch of columns; the slopes and coefficients
of every member, evaluation and its derivatives, the antiderivative's prefix sums over several blocks, the pairwise
integral and the roots; the B-splines' collocation solves of every kind, their evaluation and calculus)."""

import numpy as np
import pytest

import stardust as sd

from .._bits import bits_equal

pytestmark = pytest.mark.usefixtures("restore_threads")

A = sd.asarray
LINEAR = sd._core.interpolate.linear


def _results():
    g = np.random.default_rng(21)
    out = []
    for n in (2, 1000, 10**5):
        xp = np.sort(g.standard_normal(n))
        fp = g.standard_normal(n)
        x = g.standard_normal(150_000) * 1.2
        x[::997] = np.nan
        xs = np.sort(x)
        out.append(sd.interp(A(x), A(xp), A(fp)))
        out.append(sd.interp(A(xs), A(xp), A(fp)))
        out.append(sd.interp(A(x), A(xp), A(fp + 1j * fp[::-1])))
        out.append(sd.interp(A(x), A(xp), A(fp), period=0.7))
        out.append(sd.interp(A(x), A(xp), A(fp), left=-3.0, right=4.0))
        cols = np.stack([fp, fp[::-1], -fp], axis=1)
        for rule in (0, 1, 2):
            out.append(LINEAR(A(x), A(xp), A(cols), rule))
            out.append(LINEAR(A(xs), A(xp), A(cols.astype(complex)), rule))
    return out


def _cubic_results():
    IP = sd.interpolate
    g = np.random.default_rng(22)
    out = []
    n = 30_000
    x = np.sort(g.random(n))
    y = g.standard_normal(n)
    yp = y.copy()
    yp[-1] = yp[0]
    yb = g.standard_normal((300, 400))
    xb = np.sort(g.random(300))
    q = g.random(120_000) * 1.2 - 0.1
    q[::991] = np.nan
    objs = [IP.CubicSpline(x, y), IP.CubicSpline(x, yp, bc_type="periodic"), IP.CubicSpline(x, y, bc_type="natural"),
            IP.PchipInterpolator(x, y), IP.Akima1DInterpolator(x, y, method="makima"),
            IP.CubicSpline(xb, yb), IP.CubicSpline(xb, yb + 1j * yb[::-1], bc_type="clamped"),
            IP.PchipInterpolator(xb, yb), IP.Akima1DInterpolator(xb, yb),
            IP.CubicHermiteSpline(x, y, y[::-1])]  # fmt: skip
    for p in objs:
        qq = q if p.c.ndim == 2 else q[:2000]  # a batch of 400 columns: fewer points (120,000 x 400 doubles were
        out.append(p.c)  # 384 MB per call and 12 GB at two thread counts)
        out.append(p(qq))
        out.append(p(np.sort(qq), nu=2))
        out.append(p(qq, extrapolate="periodic"))
    p = objs[0]
    a = p.antiderivative(nu=2)
    out += [a.c, p.integrate(-0.05, 1.07), objs[5].integrate(0.1, 0.8), objs[6].antiderivative().c]
    out += [r for r in p.roots()] if isinstance(p.roots(), list) else [p.roots()]
    out += list(objs[7].roots())
    return out


def _bspline_results():
    IP = sd.interpolate
    g = np.random.default_rng(23)
    out = []
    n = 60_000
    x = np.sort(g.random(n))
    y = g.standard_normal(n)
    yp = y.copy()
    yp[-1] = yp[0]
    xb = np.sort(g.random(400))
    yb = g.standard_normal((400, 300))
    q = g.random(150_000) * 1.2 - 0.1
    q[::983] = np.nan
    zero, one = np.zeros(300), np.ones(300)
    objs = [
        IP.make_interp_spline(x, y),
        IP.make_interp_spline(x, y, k=5),
        IP.make_interp_spline(x, y, k=2),
        IP.make_interp_spline(x, y, bc_type="natural"),
        IP.make_interp_spline(x, yp, bc_type="periodic"),
        IP.make_interp_spline(x, y, k=1),
        IP.make_interp_spline(xb, yb),
        IP.make_interp_spline(xb, yb + 1j * yb[::-1], bc_type="clamped"),
        IP.make_interp_spline(xb, yb, k=4, bc_type=([(2, zero)], [(1, one), (2, zero)])),  # pivots (IP53)
    ]
    for b in objs:
        qq = q if b.c.ndim == 1 else q[:2000]  # a batch of 300 columns: fewer points (the values' memory)
        out.append(b.c)
        out.append(b(qq))
        out.append(b(np.sort(qq), nu=min(2, b.k)))
        out.append(b(qq, extrapolate="periodic"))
    b = objs[0]
    out += [b.antiderivative(nu=2).c, b.derivative(nu=2).c, b.integrate(-0.05, 1.07), objs[6].integrate(0.1, 0.8),
            b.insert_knot(0.5, m=2).c, IP.PPoly.from_spline(b).c, IP.PPoly.from_spline(objs[6]).c,
            IP.BSpline.design_matrix(np.sort(q[np.isfinite(q)]), b.t, 3, extrapolate=True).data,
            IP.BSpline.from_power_basis(IP.CubicSpline(xb, yb)).c]  # fmt: skip
    return out


def test_thread_counts():
    sd.set_num_threads(1)
    base = _results() + _cubic_results() + _bspline_results()
    for t in (2, 3, 7, 10):  # one thread count at a time against the first (the results of all five were 2 GB)
        sd.set_num_threads(t)
        got = _results() + _cubic_results() + _bspline_results()
        for k, (x, y) in enumerate(zip(base, got)):
            assert bits_equal(x, y), (t, k)
        del got
