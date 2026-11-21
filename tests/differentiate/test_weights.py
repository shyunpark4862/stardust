"""derivative's difference weights (decision C28; stardust/differentiate/_weights.py): every weight
is the nearest double to the exact weight of scipy's nodes, checked against Arb's solution of the
Vandermonde system, and scipy's LAPACK-solved weights are not (docs/design/integrate.md §3.8)."""

import numpy as np
import pytest

from stardust.differentiate._weights import weights

CASES = [(1, 2.0), (2, 2.0), (4, 2.0), (6, 2.0), (4, 1.5), (4, 4.0), (4, 0.5), (3, 3.0), (5, 1.1), (4, 2.25), (8, 2.0)]


def _arb_weights(n, c):
    from flint import arb, arb_mat, ctx

    old = ctx.prec
    ctx.prec = 400
    try:
        c, d = arb(c), arb(c).sqrt()
        central = (
            [-(1 / c ** (k - 1)) for k in range(n, 0, -1)] + [arb(0)] + [1 / c ** (k - 1) for k in range(1, n + 1)]
        )
        side = [arb(0)] + [1 / d**k for k in range(2 * n)]
        out = []
        for nodes in (central, side):
            m = len(nodes)
            A = arb_mat([[nodes[i] ** j for i in range(m)] for j in range(m)])
            b = arb_mat([[1 if j == 1 else 0] for j in range(m)])
            x = A.solve(b)
            out.append([x[i, 0] for i in range(m)])
        return out
    finally:
        ctx.prec = old


def _nearest(v, ball):
    """v is the double nearest to every point of the Arb ball (the ball is far inside the
    rounding interval of v, or v and the ball are 0)."""
    from flint import arb

    if v == 0.0:
        return ball.contains(0) and float(ball.rad()) < 1e-100
    lo, hi = np.nextafter(v, -np.inf), np.nextafter(v, np.inf)
    a = arb(float(v))
    return bool(abs(ball - a) < abs(ball - arb(float(lo)))) and bool(abs(ball - a) < abs(ball - arb(float(hi))))


def _scipy_weights(n, fac):
    i = np.arange(-n, n + 1)
    h = np.sign(i) / fac ** (np.abs(i) - 1.0)
    b = np.zeros(2 * n + 1)
    b[1] = 1
    wc = np.linalg.solve(np.vander(h, increasing=True).T, b)
    wc[n] = 0
    for k in range(n):
        wc[-k - 1] = -wc[k]
    i = np.arange(2 * n + 1)
    h = np.sign(i) / np.sqrt(fac) ** (i - 1.0)
    return wc, np.linalg.solve(np.vander(h, increasing=True).T, b)


@pytest.mark.parametrize("n,c", CASES)
def test_correctly_rounded(n, c):
    wc, wr = weights(n, c)
    ac, ar = _arb_weights(n, c)
    assert len(wc) == len(wr) == 2 * n + 1
    for got, want in ((wc, ac), (wr, ar)):
        for v, ball in zip(got, want):
            assert _nearest(v, ball), (n, c, v, ball.str(25))
    # the central ones: antisymmetric and the centre exactly 0, as scipy imposes
    assert wc[n] == 0.0 and all(wc[i] == -wc[-1 - i] for i in range(n))


def test_scipy_weights_differ():
    # docs/design/integrate.md §7.4: at order 8 and step factor 2, 8 of the 9 central weights
    wc, wr = weights(4, 2.0)
    sc, sr = _scipy_weights(4, 2.0)
    assert sum(a != b for a, b in zip(wc, sc)) == 8 and sum(a != b for a, b in zip(wr, sr)) == 9
    rel = max(abs(a - b) / abs(a) for a, b in zip(wc + wr, list(sc) + list(sr)) if a)
    assert 1e-11 < rel < 1e-9


def test_cached_and_rational_root():
    assert weights(4, 2.0) is weights(4, 2.0)
    # c = 4: d = 2 exactly, the one-sided nodes are rationals
    _, wr = weights(2, 4.0)
    assert np.isclose(sum(wr), 0.0, atol=1e-12)
