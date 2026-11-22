"""Results of sd.interp, the interpolate area's linear path and the cubic family do not depend on the thread count
(docs/design/interpolate.md §1.3, §3.9): at 1, 2, 3, 7 and 10 threads the same bytes, on inputs large enough to split
into many tasks (random and sorted points, a period, complex values, a batch of columns; the slopes and coefficients
of every member, evaluation and its derivatives, the antiderivative's prefix sums over several blocks, the pairwise
integral and the roots)."""

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
        out.append(p.c)
        out.append(p(q))
        out.append(p(np.sort(q), nu=2))
        out.append(p(q, extrapolate="periodic"))
    p = objs[0]
    a = p.antiderivative(nu=2)
    out += [a.c, p.integrate(-0.05, 1.07), objs[5].integrate(0.1, 0.8), objs[6].antiderivative().c]
    out += [r for r in p.roots()] if isinstance(p.roots(), list) else [p.roots()]
    out += list(objs[7].roots())
    return out


def test_thread_counts():
    got = {}
    for t in (1, 2, 3, 7, 10):
        sd.set_num_threads(t)
        got[t] = _results() + _cubic_results()
    for t in (2, 3, 7, 10):
        for k, (x, y) in enumerate(zip(got[1], got[t])):
            assert bits_equal(x, y), (t, k)
