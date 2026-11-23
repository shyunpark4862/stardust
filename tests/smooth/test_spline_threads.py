"""The splines of sd.smooth do not depend on the thread count (decision SM3; docs/design/smooth.md §3.8): at 1, 2,
3, 7 and 10 threads the same bytes for the lam search's parallel grid (GCV and CV, a batch), the hat norms of
predict_interval over many points, the P-spline and FITPACK's smoothing."""

import numpy as np
import pytest

import stardust as sd

from .._bits import bits_equal

pytestmark = pytest.mark.usefixtures("restore_threads")

A = sd.asarray


def _results():
    g = np.random.default_rng(31)
    x = np.sort(10.0 * g.random(3000))
    y = np.sin(x) + 0.3 * g.standard_normal(x.size)
    Y = np.stack([y, np.cos(x) + 0.1 * g.standard_normal(x.size)], axis=1)
    out = []
    r = sd.smooth.smoothing_spline(A(x), A(y))
    out += [r.fittedvalues, r.leverage, r.spline.c, r.lam, r.gcv, r.cv]
    out.append(r.predict_interval(A(np.linspace(-1.0, 11.0, 5000))))
    c = sd.smooth.smoothing_spline(A(x), A(Y), criterion="cv")
    out += [c.fittedvalues, c.lam]
    p = sd.smooth.pspline(A(x), A(y), n_segments=200)
    out += [p.fittedvalues, p.lam, p.predict_interval(A(np.linspace(0.0, 10.0, 5000)))]
    out.append(sd.smooth.make_splrep(A(x), A(y), s=0.09 * x.size).c)
    return out


def test_thread_counts():
    sd.set_num_threads(1)
    base = _results()
    for t in (2, 3, 7, 10):
        sd.set_num_threads(t)
        for k, (a, b) in enumerate(zip(base, _results())):
            assert bits_equal(a, b), (t, k)
