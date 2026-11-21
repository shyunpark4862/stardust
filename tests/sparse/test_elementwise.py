"""The element-wise functions of SP2 (docs/api/sparse.md §3.2; decision SP36): methods only, the
functions with f(0) = 0 that scipy has as methods (scipy.sparse's _ufuncs_with_fixed_point_at_zero).

A.f() keeps A's pattern (the stored zeros too) and its values are bit for bit the array's function
on the stored values: A.sin().data is sd.sin(A.data). The array's names differ for four of them
(decision SP46: the array's names asin, atan, asinh, atanh and round, which are scipy's arcsin,
arctan, arcsinh, arctanh and rint; no aliases); deg2rad and rad2deg are numpy's x * (pi / 180) and
x * (180 / pi). A COO array's repeated positions are summed first (a nonlinear f, scipy does it in
place; here A does not change), a dok_array gives CSR (scipy's result for abs and power). The array
area's own functions differ from numpy's in the last bits for a few (sd.sin is its own), so scipy is
compared in value and pattern only."""

import math

import numpy as np
import pytest

import stardust as sd

from .._bits import bits_equal
from . import matrices as M
from .conftest import LIVE, coo, npy

S = sd.sparse
FORMATS = ("coo", "csr", "csc", "dok")
NAMES = ("sin", "tan", "asin", "atan", "sinh", "tanh", "asinh", "atanh", "sqrt", "expm1", "log1p",
         "deg2rad", "rad2deg", "floor", "ceil", "trunc", "round", "sign")  # fmt: skip
SCIPY = {"asin": "arcsin", "atan": "arctan", "asinh": "arcsinh", "atanh": "arctanh", "round": "rint"}  # scipy's names
FLOAT_ONLY = ("floor", "ceil", "trunc", "deg2rad", "rad2deg")  # no complex loop (round has one, as numpy's rint)


def array_function(name, v):
    if name == "deg2rad":
        return sd.multiply(sd.astype(v, sd.float64), math.pi / 180.0)
    if name == "rad2deg":
        return sd.multiply(sd.astype(v, sd.float64), 180.0 / math.pi)
    return getattr(sd, name)(v)


def test_the_list_is_scipys_under_the_arrays_names():
    from scipy.sparse._base import _ufuncs_with_fixed_point_at_zero

    assert {f.__name__ for f in _ufuncs_with_fixed_point_at_zero} == {SCIPY.get(n, n) for n in NAMES}
    for name in NAMES:
        assert callable(getattr(S.csr_array, name))
        assert name in ("deg2rad", "rad2deg") or callable(getattr(sd, name))  # the array has it, same name
    for old in SCIPY.values():  # no aliases (SP24)
        assert not hasattr(S.csr_array, old)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("mid", ["random_unsym/50x5", "coo/dups", "coo/int", "coo/bool", "coo/nan_inf", "coo/complex"])
def test_values_are_the_array_function_on_the_stored_values(name, mid):
    m = M.get(mid)
    if m.data.dtype.kind == "c" and name in FLOAT_ONLY:
        with pytest.raises(TypeError, match=rf"^{name}\(\): not supported for complex128"):
            getattr(coo(m), name)()
        return
    for fmt in FORMATS:
        A = coo(m).asformat(fmt)
        R = getattr(A, name)()
        if fmt == "dok":
            assert R.format == "csr"
            base = A.tocsr()
        elif fmt == "coo":
            assert R.format == "coo" and R.has_canonical_format
            base = A.tocsr()
        else:
            assert R.format == fmt
            base = A
        got = npy(R.tocsr().data) if fmt == "coo" else npy(R.data)
        assert R.nnz == base.nnz  # the pattern, stored zeros included
        assert bits_equal(got, npy(array_function(name, base.data))), fmt


def test_deg2rad_is_numpys_bits():
    x = np.random.default_rng(2).standard_normal(1000) * 1e3
    A = S.csr_array(sd.asarray(x))
    assert bits_equal(npy(A.deg2rad().data), np.deg2rad(x))
    assert bits_equal(npy(A.rad2deg().data), np.rad2deg(x))
    assert bits_equal(npy(S.csr_array(sd.asarray([1.5, -2.5, 0.5])).round().data), np.rint([1.5, -2.5, 0.5]))


@pytest.mark.parametrize("name", NAMES)
@LIVE
def test_pattern_and_values_match_scipy(name):
    m = M.get("random_unsym/50x5")
    A = coo(m).asformat("csr")
    s = m.scipy("csr")
    R = getattr(A, name)()
    with np.errstate(invalid="ignore", divide="ignore"):
        r = getattr(s, SCIPY.get(name, name))()
    assert R.format == r.format and R.nnz == r.nnz
    with np.errstate(invalid="ignore"):
        np.testing.assert_allclose(npy(R.toarray()), r.toarray(), rtol=4e-16, atol=0)


def test_the_input_does_not_change():
    A = coo(M.get("coo/dups"))
    before = A.nnz
    A.sin()
    abs(A)
    A.power(2)
    assert A.nnz == before and not A.has_canonical_format  # scipy sums A's repeats in place


def test_dtypes_are_the_arrays():
    I = S.csr_array(sd.asarray([[4, 0], [0, -9]]))
    assert I.sqrt().dtype == sd.float64 and I.sign().dtype == sd.int64 and I.floor().dtype == sd.int64
    B = S.csr_array(sd.asarray([[True, False]]))
    assert B.sin().dtype == sd.float64 and B.deg2rad().dtype == sd.float64
    assert np.isnan(npy(I.sqrt().data)[1])  # sqrt(-9): NaN, as the array's
