"""The reductions of SP2 (docs/api/sparse.md §3.3, docs/design/sparse.md §2.4, §2.6, §3.3): sum,
mean, max, min, nanmax, nanmin, argmax, argmin.

The sums are defined by their order (§2.6): an axis sum adds each line in index order from zero
(scipy's sum across its lines -- ones @ A, A @ ones -- is the same sequential sum; along its own
lines it uses np.add.reduceat, numpy's pairwise sum, which differs in the last bits from 8 terms;
with axis=None scipy's data.sum() can differ too), the whole sum is sd.sum of the values in C
order; the Arb references of tests/sparse/ref/sums.npz hold them to the contract. A COO array's
repeats are summed first, so the four formats give the same bits. max/min/argmax/argmin are the array's functions on
A.toarray() (the implicit zeros take part unless explicit=True). The results' types follow SP37
(an axis gives a dense sd.ndarray; scipy's max(axis=) is a COO array) and SP40 (keepdims, as the
array's; explicit=, as scipy's)."""

import math

import numpy as np
import pytest
import scipy.sparse as sp

import stardust as sd

from .._bits import bits_equal, value_equal
from . import matrices as M
from . import refdata as R
from .conftest import LIVE, coo, npy

S = sd.sparse
FORMATS = ("coo", "csr", "csc", "dok")
SUMS = R.points("sums")
SMALL = [i for i in M.ids(small=True) if M.get(i).shape[0] * M.get(i).shape[1] <= 40000]
EXTREMES = ("max", "min", "nanmax", "nanmin")


def as_format(m, fmt):
    A = coo(m)
    if fmt == "csc" and A.ndim != 2:
        return None
    return A.asformat(fmt)


def model_line_sums(m, axis):
    """The sums along axis in the order of §2.6 (pieces of L joined as a sequential sum of fewer
    than 16 pieces), from the canonical entries of m; int64 wraps, bool counts."""
    r, c, v = m.canonical()
    lines, pos = (r, c) if axis == 1 else (c, r)
    n = m.shape[0] if axis == 1 else m.shape[1]
    kind = v.dtype.kind
    zero = 0j if kind == "c" else (0 if kind in "ib" else 0.0)
    out = [[] for _ in range(n)]
    for k in np.lexsort((pos, lines)):
        out[lines[k]].append(v[k])
    res = []
    for terms in out:
        parts = [terms[i : i + R.L] for i in range(0, max(len(terms), 1), R.L)] if len(terms) > R.L else [terms]
        sums = []
        for p in parts:
            acc = zero
            for t in p:
                acc = acc + (int(t) if kind in "ib" else t)
            sums.append(acc)
        acc = sums[0] if len(sums) == 1 else sum(sums, zero)
        res.append(acc)
    dtype = np.int64 if kind in "ib" else v.dtype
    if kind in "ib":
        res = [((x + 2**63) % 2**64) - 2**63 for x in res]
    return np.array(res, dtype=dtype)


# ---- sums ----


@pytest.mark.parametrize("mid", SMALL)
@pytest.mark.parametrize("axis", [0, 1])
def test_axis_sums_are_the_ordered_sums_in_every_format(mid, axis):
    m = M.get(mid)
    want = model_line_sums(m, axis)
    for fmt in FORMATS:
        A = as_format(m, fmt)
        got = A.sum(axis=axis)
        assert isinstance(got, sd.ndarray) and bits_equal(npy(got), want), fmt


@pytest.mark.parametrize("mid", SMALL)
def test_the_whole_sum_is_the_pairwise_sum_in_c_order(mid):
    m = M.get(mid)
    r, c, v = m.canonical()
    want = sd.sum(sd.asarray(v))
    for fmt in FORMATS:
        got = as_format(m, fmt).sum()
        assert type(got) is type(want) and bits_equal(got, want), fmt


@pytest.mark.parametrize("p", SUMS, ids=lambda p: p["matrix"])
def test_sums_meet_the_contract(p):
    m = M.get(p["matrix"])
    if m.data.dtype.kind in "ib":  # exact
        return
    A = as_format(m, "csr")
    av = np.abs(m.data)
    rows = np.bincount(m.row, weights=av, minlength=m.shape[0]) if m.nnz else np.zeros(m.shape[0])
    cols = np.bincount(m.col, weights=av, minlength=m.shape[1]) if m.nnz else np.zeros(m.shape[1])
    rl = np.bincount(m.row, minlength=m.shape[0])
    cl = np.bincount(m.col, minlength=m.shape[1])
    assert R.spmv_error(npy(A.sum(axis=1)), p["row_sums_hi"], p["row_sums_lo"], rows, rl).max(initial=0) <= 1
    assert R.spmv_error(npy(A.sum(axis=0)), p["col_sums_hi"], p["col_sums_lo"], cols, cl).max(initial=0) <= 1
    total = np.asarray([A.sum()])
    assert R.spmv_error(total, p["total_hi"], p["total_lo"], [av.sum()], [max(m.nnz, 1)]).max() <= 1


@pytest.mark.parametrize("mid", ["random_unsym/1000x8", "lap2d/30", "rect/1200x300", "random_complex/800"])
@LIVE
def test_axis_sums_are_scipys_bits(mid):
    m = M.get(mid)
    s = m.scipy("csr")
    for fmt in ("csr", "csc"):
        A = as_format(m, fmt)
        for axis in (0, 1):
            got, ref = npy(A.sum(axis=axis)), s.asformat(fmt).sum(axis=axis)
            if (axis == 1) != (fmt == "csr"):  # scipy: ones @ A, A @ ones -- the sequential sums
                assert bits_equal(got, ref), (fmt, axis)
            else:  # scipy: np.add.reduceat along its own lines, numpy's pairwise from 8 terms
                np.testing.assert_allclose(got, ref, rtol=1e-13, atol=1e-13)


def test_sum_dtypes_and_keepdims():
    B = S.csr_array(sd.asarray([[True, False, True], [False, False, True]]))
    assert B.sum() == 3 and type(B.sum()) is int
    assert B.sum(axis=0).dtype == sd.int64 and B.sum(axis=0).tolist() == [1, 0, 2]
    assert B.sum(dtype=sd.float64) == 3.0 and B.sum(axis=1, dtype="float64").dtype == sd.float64
    assert B.sum(keepdims=True).shape == (1, 1) and B.sum(axis=0, keepdims=True).shape == (1, 3)
    assert B.sum(axis=1, keepdims=True).tolist() == [[2], [1]]
    big = S.csr_array(sd.asarray([[2**62, 2**62]]))
    assert big.sum() == -(2**63) and big.sum(axis=1).tolist() == [-(2**63)]  # wraps, as arrays and scipy
    C = S.csr_array(sd.asarray([[1j, 0.0]]))
    with pytest.raises(TypeError, match=r"^sum\(\): cannot convert complex128 to float64"):
        C.sum(dtype=sd.float64)
    v = S.coo_array(sd.asarray([1.0, 0.0, 2.0]))
    assert v.sum(axis=0) == 3.0 and v.sum(axis=-1, keepdims=True).tolist() == [3.0]
    assert S.csr_array((0, 0)).sum() == 0.0 and S.csr_array((3, 0)).sum(axis=1).tolist() == [0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match=r"^sum\(\): axis 2 is out of bounds for an array of dimension 2$"):
        B.sum(axis=2)
    with pytest.raises(TypeError, match=r"^sum\(\): 'axis' must be an int or None, not tuple$"):
        B.sum(axis=(0, 1))
    with pytest.raises(TypeError, match=r"^sum\(\): 'keepdims' must be True or False, not int$"):
        B.sum(keepdims=1)


def test_mean_of_integers_is_exact_and_divided_once():
    vals = [2**60 + 1, 2**60 + 3, 7]
    A = S.coo_array((sd.asarray(vals), (sd.asarray([0, 0, 1]), sd.asarray([0, 1, 2]))), shape=(2, 3))
    assert A.mean() == float(sum(vals)) / 6
    assert A.mean(axis=1).tolist() == [float(vals[0] + vals[1]) / 3, 7 / 3]
    assert A.mean(axis=0).tolist() == [float(vals[0]) / 2, float(vals[1]) / 2, 7 / 2]
    f = S.csr_array(sd.asarray([[1.0, 0.0], [0.0, 3.0]]))
    assert f.mean() == 1.0 and f.mean(axis=0, keepdims=True).tolist() == [[0.5, 1.5]]
    c = S.csr_array(sd.asarray([[1j, 0.0]]))
    assert c.mean() == 0.5j and c.mean(axis=1).dtype == sd.complex128
    assert math.isnan(S.csr_array((0, 3)).mean())  # 0 / 0, as the array's mean (scipy: ZeroDivisionError)
    empty_rows = npy(S.csr_array((2, 0)).mean(axis=1))
    assert empty_rows.shape == (2,) and np.isnan(empty_rows).all()
    B = S.csr_array(sd.asarray([[True, False], [True, True]]))
    assert B.mean() == 0.75 and B.mean(axis=0).dtype == sd.float64


@pytest.mark.parametrize("fmt", FORMATS)
def test_mean_matches_the_sum_divided(fmt):
    m = M.get("random_unsym/50x5")
    A = as_format(m, fmt)
    assert A.mean() == A.sum() / A.size
    assert bits_equal(npy(A.mean(axis=0)), npy(A.sum(axis=0)) / 50)


# ---- max, min, argmax, argmin ----


@pytest.mark.parametrize("mid", [i for i in SMALL if M.get(i).data.dtype != np.complex128])
@pytest.mark.parametrize("name", EXTREMES + ("argmax", "argmin"))
def test_extremes_are_the_array_functions_on_the_dense_array(mid, name):
    m = M.get(mid)
    for fmt in FORMATS:
        A = as_format(m, fmt)
        if A.size == 0:
            with pytest.raises(ValueError, match=rf"^{name}\(\): undefined for an empty array"):
                getattr(A, name)()
            continue
        D = A.toarray()
        f = getattr(sd, name)
        assert value_equal(getattr(A, name)(), f(D)), fmt
        for axis in range(A.ndim):
            if A.shape[axis] == 0:
                with pytest.raises(ValueError):
                    getattr(A, name)(axis=axis)
                continue
            got = getattr(A, name)(axis=axis)
            want = f(D, axis=axis)
            if A.ndim == 1:
                assert value_equal(got, want)
            else:
                assert isinstance(got, sd.ndarray) and got.dtype == want.dtype and value_equal(npy(got), npy(want))


@pytest.mark.parametrize("name", EXTREMES)
def test_explicit_looks_at_the_stored_values_only(name):
    A = S.csr_array(sd.asarray([[-1.0, 0.0, -3.0], [0.0, 0.0, 0.0], [2.0, 5.0, 0.0]]))
    want = {"max": [-1.0, 0.0, 5.0], "min": [-3.0, 0.0, 2.0], "nanmax": [-1.0, 0.0, 5.0], "nanmin": [-3.0, 0.0, 2.0]}
    assert getattr(A, name)(axis=1, explicit=True).tolist() == want[name]  # an empty row: 0 (scipy)
    assert getattr(A, name)(explicit=True) == (5.0 if "max" in name else -3.0)
    assert A.argmax(axis=1, explicit=True).tolist() == [0, 0, 1]
    assert A.argmin(explicit=True) == 2
    with pytest.raises(ValueError, match=r"^argmax\(\): the array stores no values \(explicit=True\)$"):
        S.csr_array((2, 2)).argmax(explicit=True)
    assert S.csr_array((2, 2)).max(explicit=True) == 0.0


def test_nan_propagates_or_is_skipped():
    A = S.csr_array(sd.asarray([[0.0, float("nan"), 1.0], [float("nan"), 0.0, -1.0]]))
    assert math.isnan(A.max()) and A.nanmax() == 1.0 and A.nanmin() == -1.0
    assert A.argmax() == 1 and A.argmax(axis=1).tolist() == [1, 0]  # the first NaN (the array's rule)
    s = sp.csr_array(npy(A.toarray()))
    assert s.argmax() == 0  # [scipy와 다름]: scipy returns the first implicit zero
    allnan = S.csr_array(sd.asarray([[float("nan"), 0.0]]))
    assert allnan.nanmax() == 0.0 and math.isnan(allnan.nanmax(explicit=True))


@pytest.mark.parametrize("name", EXTREMES)
@LIVE
def test_extremes_match_scipy_values(name):
    m = M.get("random_unsym/50x5")
    A, s = as_format(m, "csr"), m.scipy("csr")
    for axis in (0, 1):
        got = getattr(A, name)(axis=axis)
        ref = getattr(s, name)(axis=axis)
        assert ref.format == "coo"  # [scipy와 다름] (SP37): here a dense sd.ndarray
        assert bits_equal(npy(got), ref.toarray())
    assert getattr(A, name)() == getattr(s, name)()


def test_argmax_ties_go_to_the_first_position():
    A = S.csr_array(sd.asarray([[0.0, 2.0, 0.0], [2.0, 0.0, 1.0]]))
    assert A.argmax() == 1 and A.argmax(axis=0).tolist() == [1, 0, 1] and A.argmin(axis=1).tolist() == [0, 1]
    N = S.csr_array(sd.asarray([[-1.0, -2.0], [-3.0, -4.0]]))
    assert N.argmax() == 0 and N.max() == -1.0  # no implicit zero
    Z = S.csr_array((sd.asarray([-1.0, 0.0]), sd.asarray([0, 1]), sd.asarray([0, 2, 2])), shape=(2, 2))
    assert Z.argmax() == 1 and Z.argmax(axis=1).tolist() == [1, 0]  # a stored zero ties an implicit one


def test_extreme_dtypes():
    B = S.csr_array(sd.asarray([[True, False], [False, False]]))
    assert B.max() is True and B.min() is False and B.max(axis=1).dtype == sd.bool
    I = S.csr_array(sd.asarray([[3, 0], [0, -2]]))
    assert type(I.max()) is int and I.nanmin() == -2
    C = S.csr_array(sd.asarray([[1j, 0.0]]))
    for name in EXTREMES + ("argmax", "argmin"):
        with pytest.raises(TypeError, match=rf"^{name}\(\): not supported for complex128"):
            getattr(C, name)()
    assert I.argmax(keepdims=True).tolist() == [[0]] and I.max(axis=0, keepdims=True).shape == (1, 2)
