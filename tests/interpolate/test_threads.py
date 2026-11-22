"""Results of sd.interp and of the interpolate area's linear path do not depend on the thread count
(docs/design/interpolate.md §1.3, §3.9): at 1, 2, 3, 7 and 10 threads the same bytes, on inputs large enough to split
into many tasks (random and sorted points, a period, complex values, a batch of columns)."""

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


def test_thread_counts():
    got = {}
    for t in (1, 2, 3, 7, 10):
        sd.set_num_threads(t)
        got[t] = _results()
    for t in (2, 3, 7, 10):
        for k, (x, y) in enumerate(zip(got[1], got[t])):
            assert bits_equal(x, y), (t, k)
