"""Every function with a reference file (tests/special/ref, made by tools/special_ref.py)
meets its accuracy contract (docs/api/special.md §1.4): relative error at most 1e-14
in the guaranteed region (times max(1, kappa) for the S4 inverses); outside it, at most
the documented loose bound."""

import numpy as np
import pytest

import stardust as sd

from . import refdata

GUARANTEED = 1e-14

# Outside the guaranteed region: the measured bound each function documents
# (docs/api/special.md, the function's section). The engine's test function:
LOOSE = {
    # Near pi/2 the denominator 1 - x^2/(3 - ...) cancels to almost nothing, and the rounding of
    # x^2 is all that is left: 0.83 measured at a few ULP from pi/2 (np.tan, with its argument
    # reduction, stays at 2 ULP). The guaranteed region |x| <= 1 measures 6.7 ULP.
    "_cf_tan": 1.0,
    # S7 (user decision: guaranteed only where a positive-term form exists). hyp1f1's loose
    # region goes to Boost.Math (S10, special/boost.hpp; ours cancelled up to 1e29 there,
    # 3.9e-3 measured): 2.8e-13 measured. hyp2f1's: z > 0.9, z < -1, negative parameters
    # (measured 5.8e-13).
    "hyp1f1": 1e-12,
    "hyp2f1": 1e-12,
}


MIXED = refdata.MIXED
ALIAS = refdata.ALIAS
mixed_error = refdata.mixed_error


def function(name):
    if name in refdata.KW:
        base, kw = refdata.KW[name]
        f = getattr(sd.special, base)
        return lambda *a: f(*a, **kw)
    if name in refdata.PART:
        base, i = refdata.PART[name]
        f = getattr(sd.special, base)
        return lambda *a: f(*a)[i]
    name = ALIAS.get(name, name)
    return getattr(sd.special, name, None) or getattr(sd.special._core, name)


# A subnormal result is held to an absolute error instead (docs/design/special.md §2.3):
# 45 units of the smallest subnormal, about 1e-14 relative at the bottom of the normal range.
SUBNORMAL_ABS = 45 * 2.0**-1074


def contract_error(got, hi, lo):
    """The relative error, or for a subnormal reference the absolute error scaled so that
    the subnormal bound maps onto the relative one (1 means "the bound")."""
    rel = refdata.rel_error(got, hi, lo)
    sub = np.isfinite(hi) & (np.abs(hi) < 2.0**-1022)
    with np.errstate(invalid="ignore", over="ignore"):
        absolute = refdata._diff(got, hi, lo) / SUBNORMAL_ABS * GUARANTEED
    return np.where(sub, absolute, rel)


@pytest.mark.parametrize("name", refdata.names())
def test_accuracy(name):
    xs, hi, lo, guaranteed = refdata.load(name)
    got = np.asarray(function(name)(*(sd.asarray(x) for x in xs)))
    err = contract_error(got, hi, lo) / refdata.scale(name)  # the inverses: max(1, kappa) (S4)
    if guaranteed.any():
        assert err[guaranteed].max() <= GUARANTEED, (
            name,
            [x[guaranteed][np.argmax(err[guaranteed])] for x in xs],
            err[guaranteed].max(),
        )
    if (~guaranteed).any():
        if name in MIXED:
            m = mixed_error(got, hi, lo)[~guaranteed]
            assert m.max() <= GUARANTEED, (name, [x[~guaranteed][np.argmax(m)] for x in xs], m.max())
        else:
            assert err[~guaranteed].max() <= LOOSE[name], (name, xs[0][~guaranteed][np.argmax(err[~guaranteed])])


@pytest.mark.parametrize("name", refdata.names())
def test_scalar_path_gives_the_same_bits(name):
    # f(x)[i] == f(x[i]) on the reference inputs (docs/api/special.md §1.4)
    xs, *_ = refdata.load(name)
    f = function(name)
    got = np.asarray(f(*(sd.asarray(x) for x in xs)))
    idx = np.linspace(0, got.size - 1, 200).astype(int)
    scalar = complex if np.iscomplexobj(xs[0]) else float
    for i in idx:
        v = f(*(scalar(x[i]) for x in xs))
        assert np.array([v], dtype=got.dtype).view(np.uint64).tolist() == got[i : i + 1].view(np.uint64).tolist(), (
            name,
            i,
        )
