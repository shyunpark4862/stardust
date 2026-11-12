"""Every function with a reference file (tests/special/ref, made by tools/special_ref.py)
meets its accuracy contract (docs/api/special.md §1.4): relative error at most 1e-14
in the guaranteed region; outside it, at most the documented loose bound."""

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
}


def function(name):
    return getattr(sd.special, name, None) or getattr(sd.special._core, name)


@pytest.mark.parametrize("name", refdata.names())
def test_accuracy(name):
    xs, hi, lo, guaranteed = refdata.load(name)
    got = np.asarray(function(name)(*(sd.asarray(x) for x in xs)))
    err = refdata.rel_error(got, hi, lo)
    assert err[guaranteed].max() <= GUARANTEED, (name, xs[0][guaranteed][np.argmax(err[guaranteed])])
    if (~guaranteed).any():
        assert err[~guaranteed].max() <= LOOSE[name], (name, xs[0][~guaranteed][np.argmax(err[~guaranteed])])


@pytest.mark.parametrize("name", refdata.names())
def test_scalar_path_gives_the_same_bits(name):
    # f(x)[i] == f(x[i]) on the reference inputs (docs/api/special.md §1.4)
    xs, *_ = refdata.load(name)
    f = function(name)
    got = np.asarray(f(*(sd.asarray(x) for x in xs)))
    idx = np.linspace(0, got.size - 1, 200).astype(int)
    for i in idx:
        v = f(*(float(x[i]) for x in xs))
        assert np.float64(v).view(np.uint64) == got[i].view(np.uint64), (name, i)
