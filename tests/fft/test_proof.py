"""The proof of the contract's constant (docs/design/fft.md §4.2, decision FT31) holds for the code as
it is: the butterflies' constants are the nearest doubles, every stage's bound per log2 of its radix is
below kappa = 8 (tests/fft/proof.py propagates the roundings of csrc/fft/cfft.hpp's butterflies), the
twiddles of the plans are within mu = 1.5u of the truth (Arb), and Bluestein's rho is what the
contract's judge computes independently."""

import math

import numpy as np
import pytest

import stardust as sd

from . import proof, refdata

C = sd._core.fft


def test_constants_are_nearest_doubles():
    for p, (c, s) in proof.constants().items():
        assert len(c) == len(s) == p
        for which, vals in (("c", c), ("s", s)):
            _, nearest = proof.constant_errors(p, vals, which)
            assert all(nearest), (p, which)
        for m in range(1, p):  # the symmetries the butterfly's index arithmetic relies on
            assert c[m] == c[p - m] and s[m] == -s[p - m]


def test_every_stage_within_kappa():
    t = proof.table()
    for r, (_eb, _es, per) in t.items():
        assert per <= proof.KAPPA, (r, per)
    # the values the code comment and the design quote (cfft.hpp, docs/design/fft.md §4.2)
    want = {2: 1.414, 4: 4.000, 3: 5.768, 5: 9.912, 7: 13.669, 11: 22.875}
    for r, v in want.items():
        assert abs(t[r][0] - v) < 5e-4, (r, t[r][0])


def _twiddle_error(n, ks):
    from flint import acb, arb, ctx

    w = np.asarray(C._twiddles(n))
    worst = 0.0
    old = ctx.prec
    ctx.prec = 160
    try:
        for k in ks:
            exact = acb(arb(-2 * int(k)) / n).exp_pi_i()
            d = abs(acb(float(w[k, 0]), float(w[k, 1])) - exact)
            worst = max(worst, float((d / arb(2.0**-53)).upper()))
    finally:
        ctx.prec = old
    return worst


@pytest.mark.parametrize("n", [12, 100, 1000, 1024, 2310, 3072, 4096, 4620])
def test_twiddles_within_mu(n):
    assert _twiddle_error(n, range(n)) <= proof.MU


def test_twiddles_of_long_plans_within_mu():
    g = np.random.default_rng(5)
    for n in (1 << 16, 15 << 10, 1 << 20, 131220):
        ks = g.integers(0, n, 3000)
        assert _twiddle_error(n, ks) <= proof.MU, n


def test_bluestein_plan_constants():
    for n in (13, 127, 1009, 2018, 10007, 30021, 65537):
        info = C._plan_info(n)
        assert info["path"] == "bluestein"
        assert info["m"] == refdata.bluestein_m(n)
        assert info["m"] >= 2 * n - 1 and refdata.is_smooth11(info["m"])
        assert set(info["radices"]) <= {2, 3, 4, 5}
        assert info["rho"] >= 1.0
        assert math.isclose(info["rho"], refdata.bluestein_rho(n), rel_tol=1e-9), n
    for n in (1, 2, 11, 12, 1000, 1024, 2310, 9240, 121):
        assert C._plan_info(n)["path"] == "mixed"
