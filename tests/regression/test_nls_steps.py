"""regression.nls's difference steps (sd.optimize's kernel and accurate differences): a parameter
far below the model's scale (D-6) and bounds narrower than the steps (LSQ-4)."""

import numpy as np
import pytest

import stardust as sd


def _decay():
    rng = np.random.default_rng(1)
    xn = np.linspace(0, 5, 40)
    yn = 3.0 * np.exp(-0.7 * xn) + 0.2 - 0.05 * xn + 0.03 * rng.standard_normal(40)

    def f(x, a, b, c):
        return a * sd.exp(-b * x) + c

    def jac(x, a, b, c):
        e = sd.exp(-b * x)
        return sd.stack([e, -a * x * e, x * 0.0 + 1.0], axis=1)

    return sd.asarray(xn.tolist()), sd.asarray(yn.tolist()), f, jac


def test_a_parameter_near_zero_has_accurate_differences():
    # c ends at its bound 0 (from 2.7e-12): the start 2^-8 |c| was at rounding level, so the
    # Jacobian's c column (and the standard errors) were off by 0.3 %; now as the analytic one's
    x, y, f, jac = _decay()
    a = sd.regression.nls(f, x, y, p0=[1.0, 0.0, 1.0], bounds=(0, 10))
    b = sd.regression.nls(f, x, y, p0=[1.0, 0.0, 1.0], bounds=(0, 10), jac=jac)
    for u, v in zip(a.bse.tolist(), b.bse.tolist()):
        assert abs(u - v) <= 1e-9 * v
    for u, v in zip(a.params.tolist(), b.params.tolist()):
        assert abs(u - v) <= 1e-9 * max(1.0, abs(v))


def test_bounds_narrower_than_the_steps():
    # the model is never called outside the bounds (fdjac2's step is scipy's adjust_one_sided),
    # and the accurate Jacobian's 5-point stencils that cannot fit say so
    t = np.linspace(0, 1, 20)
    y = 1.0 + 0.5 * t
    lb, ub = [-10, 0.5], [10, 0.5 + 1e-9]
    seen = []

    def f(x, a, b):
        seen.append(b)
        return a + b * x

    with pytest.raises(ValueError, match=r"nls\(\): the bounds are too narrow for the differences of parameter 1"):
        sd.regression.nls(f, sd.asarray(t.tolist()), sd.asarray(y.tolist()), p0=[0.0, 0.5 + 5e-10], bounds=(lb, ub))
    assert seen and all(lb[1] <= b <= ub[1] for b in seen)
    # wider than fdjac2's step but not than the stencils' smallest: fits within the bounds
    seen.clear()
    ub = [10, 0.5 + 1e-7]
    r = sd.regression.nls(f, sd.asarray(t.tolist()), sd.asarray(y.tolist()), p0=[0.0, 0.5 + 5e-8], bounds=(lb, ub))
    assert all(lb[1] <= b <= ub[1] for b in seen)
    assert abs(r.params[0] - 1.0) < 1e-6


def test_lm_needs_every_tolerance_above_u():
    # least_squares('lm')'s rule (scipy's): a tolerance below u would let lmder stop on "the
    # tolerance is too small" (its info 6-8); 'trf' needs only one of them
    x, y, f, _ = _decay()
    msg = r"nls\(\): 'ftol', 'xtol' and 'gtol' must each be >= 2.2e-16 \(the machine epsilon\) when method='lm'"
    for key in ("ftol", "xtol", "gtol"):
        with pytest.raises(ValueError, match=msg):
            sd.regression.nls(f, x, y, p0=[1.0, 1.0, 1.0], **{key: 0.0})
    r = sd.regression.nls(f, x, y, p0=[1.0, 1.0, 1.0], method="trf", gtol=0.0)
    assert r.converged
