"""S7: hyp1f1 hyp2f1 lambertw (docs/api/special.md §11).

Accuracy is tests/special/test_reference.py's (hyp1f1, hyp2f1 guaranteed on the
positive-term region per point, loose elsewhere). Here: the special values against scipy
except where docs/design/special.md §2.4 says otherwise (S7 user decisions), polynomials,
poles, and lambertw's branch argument."""

import numpy as np
import pytest
import scipy.special as sp

import stardust as sd

S = sd.special


def same_bits(a, b):
    a, b = np.float64(a), np.float64(b)
    return bool(np.isnan(a) and np.isnan(b)) or a.view(np.uint64) == b.view(np.uint64)


def test_hyp1f1_values():
    assert S.hyp1f1(1.0, 2.0, 0.0) == 1.0 and S.hyp1f1(-2.0, -2.0, 1.0) == 2.5
    assert abs(S.hyp1f1(-2.0, -3.0, 1.0) - 11 / 6) < 1e-15
    assert S.hyp1f1(1.0, -2.0, 1.0) == np.inf and S.hyp1f1(1.0, 0.0, 1.0) == np.inf  # poles (scipy)
    assert S.hyp1f1(1.0, 2.0, -np.inf) == 0.0 and S.hyp1f1(1.0, 2.0, np.inf) == np.inf
    for x in [0.3, 2.0, 50.0]:  # M(1, 2, x) = (e^x - 1) / x
        assert abs(S.hyp1f1(1.0, 2.0, x) / np.expm1(x) * x - 1) < 1e-15
    assert np.isnan(S.hyp1f1(np.nan, 1.0, 1.0))


def test_hyp1f1_true_values():
    # S7 user decision (scipy inf)
    assert S.hyp1f1(-1.0, 2.0, np.inf) == -np.inf  # 1 - x/2
    assert S.hyp1f1(0.0, 0.0, 1.0) == 1.0 and S.hyp1f1(0.0, -3.0, 5.0) == 1.0


def test_hyp2f1_values():
    assert abs(S.hyp2f1(1.0, 1.0, 2.0, 0.5) - 2 * np.log(2)) < 4e-16  # -log(1-z)/z
    assert S.hyp2f1(1.0, 1.0, 3.0, 1.0) == 2.0  # Gauss
    assert S.hyp2f1(-2.0, 1.0, 1.0, 5.0) == 16.0  # a polynomial: (1 - z)^2
    assert S.hyp2f1(-1.0, 1.0, -1.0, 0.5) == 1.5 and S.hyp2f1(-1.0, 1.0, -2.0, 0.5) == 1.25
    assert S.hyp2f1(1.0, 1.0, 0.0, 0.5) == np.inf  # pole (scipy)
    assert np.isnan(S.hyp2f1(1.0, 1.0, 2.0, 2.0))  # complex (scipy inf: docs/api/special.md §11)
    assert np.isnan(S.hyp2f1(1.0, 1.0, 2.0, 1.0))  # diverges
    for z in [-0.9, -5.0, -1e6]:
        assert abs(S.hyp2f1(1.0, 1.0, 2.0, z) / (np.log1p(-z) / -z) - 1) < 1e-14


def test_hyp2f1_true_values():
    assert S.hyp2f1(1.0, 1.0, 2.0, -np.inf) == 0.0  # the limit (S7 user decision; scipy NaN)


def test_lambertw_values():
    W = S.lambertw
    assert W(0.0) == 0.0 and same_bits(W(-0.0), -0.0) and W(np.inf) == np.inf
    assert W(0.0, k=-1) == -np.inf and np.isnan(W(1.0, k=-1)) and np.isnan(W(-0.5))
    assert abs(W(1.0) - 0.5671432904097838) < 1e-16  # the omega constant
    # S7 user decision: the float64 nearest -1/e is the branch point (scipy NaN; Boost -1)
    assert W(-0.36787944117144233) == -1.0 and W(-0.36787944117144233, k=-1) == -1.0
    assert np.isnan(W(np.nextafter(-0.36787944117144233, -1)))


def test_lambertw_branch_argument():
    with pytest.raises(ValueError, match=r"^lambertw\(\)"):
        S.lambertw(1.0, k=1)
    with pytest.raises(TypeError, match=r"^lambertw\(\)"):
        S.lambertw(1.0, k=0.0)
    with pytest.raises(TypeError, match=r"^lambertw\(\)"):
        S.lambertw(1.0, k=True)
    x = sd.asarray(np.array([-0.2, -0.1]))
    got = np.asarray(S.lambertw(x, k=-1))
    assert np.allclose(got, np.real(sp.lambertw(np.array([-0.2, -0.1]), -1)), rtol=1e-15)
